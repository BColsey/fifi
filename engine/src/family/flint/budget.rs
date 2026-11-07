// The plan surfaces and the loader are the only consumers, and both are
// native- or test-gated; a default-feature lib build compiles the module
// for the diagnostic types without reaching every rule.
#![cfg_attr(not(any(test, feature = "native-cuda")), allow(dead_code))]

//! Pure budget and policy layer for the Flint EXL3 serving profile.
//!
//! Everything here is arithmetic and parsing over numbers the caller
//! supplies: no device, no probe, no process state beyond the environment
//! values handed in. The serving path (the CUDA executor and the native
//! loader) re-exports these items from their historical homes, so it keeps
//! calling the same functions; the point of the module is that the
//! diagnostic surfaces (`check`, `config effective`) compose the same
//! decisions in a host build, the way `kernels::single_row` already does
//! for the kernel name.
//!
//! One deliberate gap: the work reserve's device scratch term
//! (`State::scratch_bytes`: hot scratch, grouped scratch, health rings) is
//! sized from CUDA launch planning and is not lifted here. Estimates
//! composed from this module charge the closed-form work reserve only and
//! say so; load-time admission remains the authority.

use crate::EngError;

/// Routed experts per layer in the published Flint packs. One definition
/// for the CPU serving path, the native cache and this budget layer.
pub(crate) const EXPERTS: usize = 256;

/// The model's own context limit, which no profile exceeds (the KV plan
/// enforces the same figure).
pub(crate) const MODEL_CONTEXT_LIMIT: u32 = 262_144;

/// Device bytes every load keeps free beside its weights and named
/// charges, whatever the expert placement.
pub(crate) const FLINT_HEADROOM_BYTES: u64 = 512 << 20;

/// The offload count that names every routed layer, as in the TabbyAPI key.
pub(crate) const ALL_LAYERS: usize = 999;

/// Worker threads when the options leave `cpu_moe_threads` at 0: the
/// expert kernels are DRAM-bound, so more threads than this do not help.
pub(crate) const DEFAULT_THREADS: usize = 6;

/// Operator override of the resident experts per layer; unset means "as
/// many as fit".
pub(crate) const SLOTS_VAR: &str = "FIFI_FLINT_CUDA_EXPERT_SLOTS";

/// The switch that selects the bank.
pub(crate) const CACHE_VAR: &str = "FIFI_FLINT_CUDA_EXPERT_CACHE";

/// The switch for the CUDA allocation cache whose idle reserve the work
/// charge includes.
pub(crate) const ALLOC_CACHE_VAR: &str = "FIFI_FLINT_CUDA_ALLOC_CACHE";

/// The allocation cache's idle reserve when armed. Must equal
/// `kernels::cuda::context`'s alloc-cache `MAX_IDLE_BYTES`; the native
/// build parity-tests the pair.
pub(crate) const ALLOC_CACHE_MAX_IDLE_BYTES: u64 = 16 * 1_048_576;

/// Which routed experts the CPU serves.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct CpuMoeConfig {
    offload_layers: usize,
    split_experts: usize,
    threads: usize,
}

impl CpuMoeConfig {
    /// Validates the three option values. An offload count of
    /// [`ALL_LAYERS`] or more means every routed layer, whatever the
    /// model's layer count.
    pub(crate) fn new(
        offload_layers: u32,
        split_experts: u32,
        threads: u32,
    ) -> Result<Self, EngError> {
        let bad = |what: &str| Err(EngError::BadRequest(format!("Flint CPU MoE: {what}")));
        if offload_layers != 0 && split_experts != 0 {
            return bad("cpu_moe_offload_layers and cpu_moe_split_experts are exclusive");
        }
        // The device keeps the experts of one prefill piece; the bank sizing enforces the floor.
        if split_experts as usize >= EXPERTS {
            return bad("cpu_moe_split_experts must be below 256");
        }
        if threads > 256 {
            return bad("cpu_moe_threads must be at most 256");
        }
        Ok(Self {
            offload_layers: offload_layers as usize,
            split_experts: split_experts as usize,
            threads: if threads == 0 {
                DEFAULT_THREADS
            } else {
                threads as usize
            },
        })
    }

    pub(crate) fn enabled(&self) -> bool {
        self.offload_layers != 0 || self.split_experts != 0
    }

    pub(crate) fn offload_layers(&self) -> usize {
        self.offload_layers
    }

    pub(crate) fn split_experts(&self) -> usize {
        self.split_experts
    }

    pub(crate) fn threads(&self) -> usize {
        self.threads
    }

    /// Whether every routed expert of the `ordinal`-th routed layer runs on the CPU.
    pub(crate) fn offloads(&self, ordinal: usize) -> bool {
        ordinal < self.offload_layers
    }
}

/// What `FIFI_FLINT_CUDA_EXPERT_CACHE` asks for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ExpertCacheMode {
    /// Unset: every expert stays on the GPU when that fits, otherwise the bank takes over.
    Auto,
    /// `1`, `true`, `on`: always use the bank.
    On,
    /// `0`, `false`, `off`: every expert on the GPU, or a refusal.
    Off,
}

impl ExpertCacheMode {
    /// Exactly the listed words; a mistyped value is refused rather than guessed, because the
    /// unset default is no longer "off".
    pub(crate) fn from_env(value: Result<String, std::env::VarError>) -> Result<Self, EngError> {
        let text = match value {
            Ok(text) => text,
            Err(std::env::VarError::NotPresent) => return Ok(Self::Auto),
            Err(std::env::VarError::NotUnicode(_)) => String::from("(not UTF-8)"),
        };
        Self::from_value(Some(text.as_str()))
    }

    /// The same parse for a value a diagnostic process already captured;
    /// `None` is the unset default, and every set value resolves through
    /// the same strict word list (so an empty string is a refusal, as at
    /// executor open).
    pub(crate) fn from_value(value: Option<&str>) -> Result<Self, EngError> {
        let text = value.unwrap_or_default();
        match text.trim().to_ascii_lowercase().as_str() {
            "" if value.is_none() => Ok(Self::Auto),
            "1" | "true" | "on" => Ok(Self::On),
            "0" | "false" | "off" => Ok(Self::Off),
            _ => Err(cache_error(&format!(
                "{CACHE_VAR} must be unset, 1, true or on (always use the host buffer), or 0, false \
                 or off (every expert on the GPU); got {text:?}"
            ))),
        }
    }

    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::On => "on",
            Self::Off => "off",
        }
    }
}

/// Why the bank is in use, for the load line.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BankReason {
    /// The context does not fit with every expert on the GPU.
    Auto,
    /// The operator asked for the bank.
    Forced,
    /// The operator asked for a resident count.
    Override,
}

impl BankReason {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Forced => "forced",
            Self::Override => "override",
        }
    }
}

/// How a load keeps its experts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BankPlan {
    /// Every expert on the GPU, the load that needs no host buffer.
    Resident,
    Bank(BankReason),
}

/// The load for `mode` and `request`. `resident_fits` is whether the plain resident load would be
/// admitted; only the unset default looks at it, so a context that fits keeps today's exact path.
pub(crate) fn plan_bank(
    mode: ExpertCacheMode,
    request: BankRequest,
    resident_fits: bool,
) -> Result<BankPlan, EngError> {
    match (mode, request) {
        (ExpertCacheMode::Off, BankRequest::Fit) => Ok(BankPlan::Resident),
        (ExpertCacheMode::Off, BankRequest::Exactly(_)) => Err(cache_error(&format!(
            "{SLOTS_VAR} asks for a bank, but {CACHE_VAR}=0 keeps every expert on the GPU"
        ))),
        (_, BankRequest::Exactly(_)) => Ok(BankPlan::Bank(BankReason::Override)),
        (ExpertCacheMode::On, BankRequest::Fit) => Ok(BankPlan::Bank(BankReason::Forced)),
        (ExpertCacheMode::Auto, BankRequest::Fit) if resident_fits => Ok(BankPlan::Resident),
        (ExpertCacheMode::Auto, BankRequest::Fit) => Ok(BankPlan::Bank(BankReason::Auto)),
    }
}

/// CPU-served experts need the pinned host store: the offload mode forces the bank on, and the
/// split mode also sizes it to the experts the device keeps, so it refuses an explicit slot count
/// that would contradict that.
pub(crate) fn cpu_moe_bank(
    config: Option<CpuMoeConfig>,
    mode: ExpertCacheMode,
    request: BankRequest,
) -> Result<(ExpertCacheMode, BankRequest), EngError> {
    let Some(config) = config else {
        return Ok((mode, request));
    };
    if mode == ExpertCacheMode::Off {
        return Err(EngError::BadRequest(
            "Flint CPU experts need the host expert store: unset or enable the expert cache switch"
                .into(),
        ));
    }
    if config.split_experts() != 0 {
        if request != BankRequest::Fit {
            return Err(EngError::BadRequest(format!(
                "cpu_moe_split_experts sets the resident experts per layer: unset {SLOTS_VAR}"
            )));
        }
        let device = EXPERTS - config.split_experts();
        return Ok((ExpertCacheMode::On, BankRequest::Exactly(device)));
    }
    Ok((ExpertCacheMode::On, request))
}

/// How many experts per layer a load asks to keep on the device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BankRequest {
    /// The largest bank the device budget leaves room for, capped at all 256.
    Fit,
    /// An explicit count; a count that does not fit refuses the load rather than shrinking.
    Exactly(usize),
}

impl BankRequest {
    pub(crate) fn from_env(value: Result<String, std::env::VarError>) -> Result<Self, EngError> {
        let value = match value {
            Ok(value) => Some(value),
            Err(std::env::VarError::NotPresent) => None,
            Err(std::env::VarError::NotUnicode(_)) => {
                return Err(cache_error(&format!("{SLOTS_VAR} is not valid UTF-8")));
            }
        };
        Self::from_value(value.as_deref())
    }

    /// The same parse for a value a diagnostic process already captured:
    /// unset and an empty string both mean `fit`, exactly as the
    /// environment read resolves them.
    pub(crate) fn from_value(value: Option<&str>) -> Result<Self, EngError> {
        // Plain decimal digits only: no sign, no leading zero, no spaces.
        let value = value.unwrap_or("");
        if value.is_empty() {
            return Ok(Self::Fit);
        }
        let plain = value.bytes().all(|byte| byte.is_ascii_digit()) && !value.starts_with('0');
        plain
            .then(|| value.parse::<usize>().ok())
            .flatten()
            .filter(|count| (1..=EXPERTS).contains(count))
            .map(Self::Exactly)
            .ok_or_else(|| cache_error(&format!("{SLOTS_VAR} must be a whole number in 1..=256")))
    }
}

/// Device bytes the bank leaves free beyond the declared reserve, for every count it considers,
/// all 256 included. A bank that filled the card to its last expert let the whole card, desktop
/// included, peak 61 to 71 MiB past 22,900 MiB. One more expert in every routed layer costs 72 MiB
/// (47 layers of 1,603,584 B), so 160 MiB is a little over two experts per layer and brings the
/// peak under that bound.
pub(crate) const BANK_MARGIN_BYTES: u64 = 160 << 20;

/// What a device has for the expert bank: `free` bytes, of which `fixed` (non-routed weights,
/// KV cache, work, headroom and the bank margin) are spoken for, and `per_slot` bytes for one
/// more resident expert in every routed layer.
#[derive(Clone, Copy, Debug)]
pub(crate) struct BankBudget {
    pub(crate) free: u64,
    pub(crate) fixed: u64,
    pub(crate) per_slot: u64,
}

impl BankBudget {
    pub(crate) fn room(&self) -> u64 {
        self.free.saturating_sub(self.fixed)
    }

    /// Experts per layer that fit, capped at all 256.
    pub(crate) fn fitting(&self) -> usize {
        let count = (self.room() / self.per_slot.max(1)).min(EXPERTS as u64);
        usize::try_from(count).unwrap_or(EXPERTS)
    }

    pub(crate) fn refusal(&self, what: String, floor: usize, advice: &str) -> EngError {
        EngError::MemoryPlan(format!(
            "Flint expert bank: {what}. After {} bytes of non-routed weights, KV cache, work, \
             headroom and the {} MiB bank margin, {} of {} free device bytes remain, room for \
             {} resident experts per layer ({} bytes each across the routed layers); a prefill \
             piece needs {floor}. {advice}.",
            self.fixed,
            BANK_MARGIN_BYTES >> 20,
            self.room(),
            self.free,
            self.fitting(),
            self.per_slot
        ))
    }
}

/// The device budget of the bank: `arena_base` bytes of other weights, the named reserve terms
/// and the margin come off `free`; each resident expert in every routed layer costs `per_slot`.
/// The plain-u64 signature keeps this host-buildable; the native loader passes its `Reserve`
/// fields.
pub(crate) fn bank_budget(
    free: u64,
    arena_base: u64,
    per_slot: u64,
    kv_cache: u64,
    work: u64,
    headroom: u64,
) -> Result<BankBudget, EngError> {
    let fixed = [kv_cache, work, headroom, BANK_MARGIN_BYTES]
        .into_iter()
        .try_fold(arena_base, u64::checked_add)
        .ok_or_else(|| {
            EngError::MemoryPlan("Flint expert bank fixed charge overflows u64".into())
        })?;
    Ok(BankBudget {
        free,
        fixed,
        per_slot,
    })
}

/// Resident experts per layer for `request`. `floor` is what one prefill piece can route in a
/// single layer: a smaller bank could not hold the piece's experts at once.
pub(crate) fn choose_slots(
    request: BankRequest,
    budget: BankBudget,
    floor: usize,
) -> Result<usize, EngError> {
    let slots = match request {
        BankRequest::Fit => budget.fitting(),
        BankRequest::Exactly(count) if count > budget.fitting() => {
            let what = format!("{SLOTS_VAR}={count} does not fit");
            return Err(budget.refusal(what, floor, "Lower the count or context_limit"));
        }
        BankRequest::Exactly(count) => count,
    };
    if slots < floor {
        let what = format!("{slots} resident experts per layer is too few");
        let advice = match request {
            BankRequest::Fit => "Lower context_limit or prefill_chunk_size",
            BankRequest::Exactly(_) => "Raise the count or lower prefill_chunk_size",
        };
        return Err(budget.refusal(what, floor, advice));
    }
    Ok(slots)
}

/// The banked slot count the loader composes from [`choose_slots`], plus
/// the CPU-expert rule it applies beside it: CPU-served experts read the
/// pinned host store, which a bank of all `EXPERTS` experts never builds,
/// so a full bank is lowered by one. Returns the count and whether that
/// rule fired.
pub(crate) fn banked_slots(
    request: BankRequest,
    budget: BankBudget,
    floor: usize,
    cpu_active: bool,
) -> Result<(usize, bool), EngError> {
    let slots = choose_slots(request, budget, floor)?;
    let lowered = cpu_active && slots == EXPERTS;
    Ok((if lowered { EXPERTS - 1 } else { slots }, lowered))
}

/// The same admission rule the CUDA planner applies at load (`plan::admit`
/// in `kernels/cuda/plan.rs`): the sum of the named demands must fit the
/// free bytes. Mirrored here so a host estimate runs the identical
/// compare; the native build parity-tests the pair.
pub(crate) fn admits(
    tensors: u64,
    kv_cache: u64,
    work: u64,
    headroom: u64,
    free: u64,
) -> Result<u64, EngError> {
    let required = [kv_cache, work, headroom]
        .into_iter()
        .try_fold(tensors, u64::checked_add)
        .ok_or_else(|| EngError::MemoryPlan("cuda admission: demand overflows u64".to_owned()))?;
    if required > free {
        return Err(EngError::MemoryPlan(format!(
            "cuda admission: need {required} bytes (tensors {tensors}, KV cache {kv_cache}, work \
             {work}, headroom {headroom}), device has {free} free"
        )));
    }
    Ok(required)
}

/// Routed experts one prefill row can name in a single layer; must equal
/// the FFN plan's `TOP_K`, which the native and test builds
/// parity-test against this constant.
pub(crate) const FLOOR_EXPERTS_PER_ROW: usize = 6;

/// The per-piece expert floor for the bank: what one prefill piece of
/// `rows` rows can route in a single layer. A bank smaller than this
/// could not hold the piece's experts at once.
pub(crate) fn cache_prefill_floor(rows: usize) -> Result<usize, EngError> {
    rows.checked_mul(FLOOR_EXPERTS_PER_ROW)
        .filter(|&pairs| pairs > 0)
        .ok_or_else(|| {
            EngError::BadRequest(
                "Flint CUDA FFN: EXL3 expert cache requires a nonzero row count".into(),
            )
        })
}

/// The load-line note for a banked load.
pub(crate) fn bank_note(context: u32, slots: usize, reason: &str) -> String {
    let why = match reason {
        "auto" => format!("context {context} does not fit with every expert on the GPU"),
        "override" => "FIFI_FLINT_CUDA_EXPERT_SLOTS asks for a smaller bank".to_owned(),
        _ => "FIFI_FLINT_CUDA_EXPERT_CACHE=1 asks for the host buffer".to_owned(),
    };
    format!(
        "Flint expert bank: {why}. {slots} of 256 experts per layer stay on the GPU and all 256 sit in \
         pinned host memory. FIFI_FLINT_CUDA_EXPERT_CACHE=0 keeps every expert on the GPU instead."
    )
}

/// Which layer-major prefill pass an engine runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LayerMajor {
    /// The token-major walk of eight-row chunks.
    Off,
    /// Layer-major with the routed experts as tall tiles (`FIFI_EXL3_LAYER_MAJOR=1`).
    Tiles,
    /// Layer-major with tall projections and job-table experts (`FIFI_EXL3_LAYER_MAJOR=jobs`).
    Jobs,
}

/// FIFI_EXL3_LAYER_MAJOR, strict-parse switch for the layer-major prefill
/// mode. Unset is OFF (byte-identical walk to today); `1` selects the tile
/// pass and `jobs` the job-table pass; every other value — including a
/// non-UTF-8 one — is an error at executor open, never a silent fallback,
/// mirroring `prefill_window_from_env`'s strictness rule for opt-in knobs.
pub(crate) fn layer_major_from_env(
    value: Option<&std::ffi::OsStr>,
) -> Result<LayerMajor, EngError> {
    let Some(value) = value else {
        return Ok(LayerMajor::Off);
    };
    match value.to_str().map(str::trim) {
        Some("1") => Ok(LayerMajor::Tiles),
        Some("jobs") => Ok(LayerMajor::Jobs),
        _ => Err(EngError::BadRequest(
            "FIFI_EXL3_LAYER_MAJOR must be unset, \"1\" or \"jobs\"".into(),
        )),
    }
}

/// The allocation cache's idle reserve, the figure the work charge
/// includes when the cache is armed.
pub(crate) fn alloc_cache_reserve(enabled: bool) -> u64 {
    if enabled {
        ALLOC_CACHE_MAX_IDLE_BYTES
    } else {
        0
    }
}

/// The closed-form work reserve every load charges beside the KV cache:
/// one row costs 512 MiB and each further row of the piece 16 MiB, plus
/// the allocation-cache idle reserve. The device scratch term is NOT part
/// of this figure (see the module doc): only the load's own admission
/// charges it.
pub(crate) fn work_reserve_closed_form(chunk_rows: u32, cache_bytes: u64) -> Result<u64, EngError> {
    // Six routed + one shared expert retain <16 MiB per extra row at width <=32768.
    ((512 + 16 * (u64::from(chunk_rows).saturating_sub(1))) << 20)
        .checked_add(cache_bytes)
        .ok_or_else(|| EngError::MemoryPlan("Flint workspace reserve overflows".into()))
}

/// The environment values a Flint plan reads, captured once by a
/// diagnostic process so tests can inject them. Values are the raw
/// variable text (`None` when unset); parsing is the budget layer's own.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct FlintEnv {
    pub(crate) expert_cache: Option<String>,
    pub(crate) expert_slots: Option<String>,
    pub(crate) layer_major: Option<std::ffi::OsString>,
    pub(crate) alloc_cache: Option<String>,
}

impl FlintEnv {
    /// Read the four variables a Flint plan inventory reads. The serving
    /// path does not use this: its switches are read once at executor
    /// open, frozen for the process. A set-but-not-UTF-8 value becomes
    /// marker text the strict parsers refuse (the text switches) or stays
    /// the raw bytes (the layer-major switch), matching the executor's
    /// own refusal of it.
    pub(crate) fn capture() -> Self {
        // A set-but-not-UTF-8 value stays as marker text the strict parse
        // refuses, as the executor does, and as the raw bytes for the
        // layer-major switch.
        let read = |name: &str| {
            std::env::var_os(name).map(|value| {
                value
                    .to_str()
                    .map(String::from)
                    .unwrap_or_else(|| "<not valid UTF-8>".to_string())
            })
        };
        Self {
            expert_cache: read(CACHE_VAR),
            expert_slots: read(SLOTS_VAR),
            layer_major: std::env::var_os("FIFI_EXL3_LAYER_MAJOR"),
            alloc_cache: read(ALLOC_CACHE_VAR),
        }
    }

    /// The resolved expert-cache switch, default `auto`.
    pub(crate) fn expert_cache_mode(&self) -> Result<ExpertCacheMode, EngError> {
        ExpertCacheMode::from_value(self.expert_cache.as_deref())
    }

    /// The resolved resident-count request, default `fit`.
    pub(crate) fn bank_request(&self) -> Result<BankRequest, EngError> {
        BankRequest::from_value(self.expert_slots.as_deref())
    }

    /// The resolved layer-major pass, default `off`.
    pub(crate) fn layer_major(&self) -> Result<LayerMajor, EngError> {
        layer_major_from_env(self.layer_major.as_deref())
    }

    /// Whether the allocation cache is armed (unset keeps it on; only
    /// the listed words arm a set value), exactly as the executor's
    /// default-on switch rule resolves the same text.
    pub(crate) fn alloc_cache_enabled(&self) -> bool {
        crate::tokenizer::env_allows_value(self.alloc_cache.as_deref())
    }
}

fn cache_error(message: &str) -> EngError {
    EngError::BadRequest(format!("Flint expert cache: {message}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_bank_plan_table_is_unchanged() {
        use ExpertCacheMode as Mode;
        let bank = |reason| Ok(BankPlan::Bank(reason));
        assert_eq!(
            plan_bank(Mode::Off, BankRequest::Fit, false),
            Ok(BankPlan::Resident)
        );
        assert!(plan_bank(Mode::Off, BankRequest::Exactly(8), true).is_err());
        assert_eq!(
            plan_bank(Mode::Auto, BankRequest::Exactly(8), true),
            bank(BankReason::Override)
        );
        assert_eq!(
            plan_bank(Mode::On, BankRequest::Fit, true),
            bank(BankReason::Forced)
        );
        assert_eq!(
            plan_bank(Mode::Auto, BankRequest::Fit, true),
            Ok(BankPlan::Resident)
        );
        assert_eq!(
            plan_bank(Mode::Auto, BankRequest::Fit, false),
            bank(BankReason::Auto)
        );
    }

    #[test]
    fn mode_and_request_parse_the_operators_words() {
        assert_eq!(
            ExpertCacheMode::from_value(None).unwrap(),
            ExpertCacheMode::Auto
        );
        assert_eq!(
            ExpertCacheMode::from_value(Some(" TrUe ")).unwrap(),
            ExpertCacheMode::On
        );
        assert_eq!(
            ExpertCacheMode::from_value(Some("off")).unwrap(),
            ExpertCacheMode::Off
        );
        assert!(ExpertCacheMode::from_value(Some("yes")).is_err());
        assert_eq!(BankRequest::from_value(None).unwrap(), BankRequest::Fit);
        assert_eq!(
            BankRequest::from_value(Some("224")).unwrap(),
            BankRequest::Exactly(224)
        );
        assert!(BankRequest::from_value(Some("0")).is_err());
        assert!(BankRequest::from_value(Some("257")).is_err());
        assert!(BankRequest::from_value(Some(" 8")).is_err());
    }

    #[test]
    fn cpu_moe_bank_forces_and_sizes_the_bank() {
        let split = CpuMoeConfig::new(0, 32, 0).unwrap();
        assert_eq!(
            cpu_moe_bank(Some(split), ExpertCacheMode::Auto, BankRequest::Fit).unwrap(),
            (ExpertCacheMode::On, BankRequest::Exactly(224))
        );
        let offload = CpuMoeConfig::new(4, 0, 0).unwrap();
        let refusal =
            cpu_moe_bank(Some(offload), ExpertCacheMode::Off, BankRequest::Fit).unwrap_err();
        assert!(
            format!("{refusal:?}").contains("need the host expert store"),
            "{refusal:?}"
        );
        assert!(
            cpu_moe_bank(
                Some(split),
                ExpertCacheMode::Auto,
                BankRequest::Exactly(200)
            )
            .is_err()
        );
    }

    #[test]
    fn banked_slots_applies_the_cpu_expert_cap() {
        let fixed = 800u64 + 100 + 100 + (512 << 20) + BANK_MARGIN_BYTES;
        let budget = |free| BankBudget {
            free,
            fixed,
            per_slot: 100,
        };
        // A fitting full bank beside CPU experts is lowered, and says so.
        let (slots, lowered) =
            banked_slots(BankRequest::Fit, budget(fixed + 40_000), 6, true).unwrap();
        assert_eq!((slots, lowered), (255, true));
        // Without CPU experts the full bank stands.
        let (slots, lowered) =
            banked_slots(BankRequest::Fit, budget(fixed + 40_000), 6, false).unwrap();
        assert_eq!((slots, lowered), (256, false));
        // Below the cap the rule stays out of the way.
        let (slots, lowered) =
            banked_slots(BankRequest::Exactly(48), budget(fixed + 5_000), 6, true).unwrap();
        assert_eq!((slots, lowered), (48, false));
    }

    #[test]
    fn choose_slots_fits_caps_and_refuses_below_the_floor() {
        let fixed = 800u64 + 100 + 100 + (512 << 20) + BANK_MARGIN_BYTES;
        let budget = BankBudget {
            free: fixed + 9_000,
            fixed,
            per_slot: 100,
        };
        assert_eq!(choose_slots(BankRequest::Fit, budget, 6).unwrap(), 90);
        assert_eq!(
            choose_slots(BankRequest::Exactly(90), budget, 6).unwrap(),
            90
        );
        let refusal = choose_slots(BankRequest::Exactly(95), budget, 6).unwrap_err();
        assert!(
            format!("{refusal:?}").contains("does not fit"),
            "{refusal:?}"
        );
        let floor_refusal = choose_slots(
            BankRequest::Fit,
            BankBudget {
                free: fixed + 4_000,
                fixed,
                per_slot: 100,
            },
            48,
        )
        .unwrap_err();
        assert!(
            format!("{floor_refusal:?}").contains("too few"),
            "{floor_refusal:?}"
        );
        // The refusal names every charged term with its figure.
        assert!(format!("{floor_refusal:?}").contains("160 MiB bank margin"));
    }

    #[test]
    fn the_admission_mirror_sum_is_plain() {
        assert_eq!(admits(10, 20, 30, 40, 100).unwrap(), 100);
        let refusal = admits(10, 20, 30, 41, 100).unwrap_err();
        assert!(
            format!("{refusal:?}").contains("need 101 bytes"),
            "{refusal:?}"
        );
    }

    #[test]
    fn bank_budget_sums_the_named_fixed_terms() {
        let fixed = 10 + 20 + 30 + FLINT_HEADROOM_BYTES + BANK_MARGIN_BYTES;
        let budget = bank_budget(fixed + 2_000, 10, 100, 20, 30, FLINT_HEADROOM_BYTES).unwrap();
        assert_eq!(budget.fixed, fixed);
        assert_eq!(budget.room(), 2_000);
    }

    #[test]
    fn the_closed_form_work_reserve_matches_the_load_charge() {
        // One row 512 MiB; each further row 16 MiB; plus the alloc-cache
        // idle reserve when armed.
        assert_eq!(work_reserve_closed_form(1, 0).unwrap(), (512u64 << 20));
        assert_eq!(
            work_reserve_closed_form(8, alloc_cache_reserve(true)).unwrap(),
            ((512 + 16 * 7) << 20) as u64 + ALLOC_CACHE_MAX_IDLE_BYTES
        );
        assert_eq!(alloc_cache_reserve(false), 0);
    }

    #[test]
    fn the_prefill_floor_is_experts_per_row() {
        for (rows, floor) in [(1, 6), (8, 48), (37, 222), (38, 228), (64, 384)] {
            assert_eq!(cache_prefill_floor(rows).unwrap(), floor, "rows={rows}");
        }
        for rows in [0, usize::MAX] {
            assert!(cache_prefill_floor(rows).is_err(), "rows={rows}");
        }
    }

    #[test]
    fn layer_major_parses_strictly() {
        use std::ffi::OsStr;

        assert_eq!(layer_major_from_env(None).unwrap(), LayerMajor::Off);
        assert_eq!(
            layer_major_from_env(Some(OsStr::new("1"))).unwrap(),
            LayerMajor::Tiles
        );
        assert_eq!(
            layer_major_from_env(Some(OsStr::new("jobs"))).unwrap(),
            LayerMajor::Jobs
        );
        assert!(layer_major_from_env(Some(OsStr::new("2"))).is_err());
        assert!(layer_major_from_env(Some(OsStr::new(" true "))).is_err());
        // A set-but-empty or non-UTF-8 value is a refusal, exactly as at
        // executor open.
        assert!(layer_major_from_env(Some(OsStr::new(""))).is_err());
        #[cfg(unix)]
        {
            use std::os::unix::ffi::OsStrExt;
            assert!(layer_major_from_env(Some(OsStr::from_bytes(b"1\xff"))).is_err());
        }
    }

    #[test]
    fn flint_env_resolves_the_default_plan() {
        let env = FlintEnv::default();
        assert_eq!(env.expert_cache_mode().unwrap(), ExpertCacheMode::Auto);
        assert_eq!(env.bank_request().unwrap(), BankRequest::Fit);
        assert_eq!(env.layer_major().unwrap(), LayerMajor::Off);
        assert!(env.alloc_cache_enabled());
    }

    #[test]
    fn bank_note_names_the_reason_and_the_rollback() {
        let note = bank_note(131_072, 226, "auto");
        assert!(note.contains("context 131072"), "{note}");
        assert!(note.contains("226 of 256"), "{note}");
        assert!(note.contains("FIFI_FLINT_CUDA_EXPERT_CACHE=0"), "{note}");
    }

    /// The mirrors in this module must stay identical to the device
    /// authorities they shadow. Only a native build can see both sides.
    #[cfg(all(test, feature = "native-cuda", target_os = "linux"))]
    #[test]
    fn the_mirrors_match_the_native_authorities() {
        use crate::kernels::cuda::plan;
        let demand = plan::Demand {
            tensors: 10,
            kv_cache: 20,
            work: 30,
            headroom: 40,
        };
        assert_eq!(
            admits(10, 20, 30, 40, 100).unwrap(),
            plan::admit(&demand, 100).unwrap()
        );
        assert!(plan::admit(&demand, 99).is_err());
        assert_eq!(
            crate::kernels::cuda::CudaContext::alloc_cache_reserve(true),
            ALLOC_CACHE_MAX_IDLE_BYTES
        );
        assert_eq!(
            crate::kernels::cuda::CudaContext::alloc_cache_reserve(false),
            0
        );
    }
}
