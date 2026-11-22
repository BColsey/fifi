//! Starts the Q36 CUDA engine of one `enginesrv` config entry: the tokenizer bundle and the
//! rules on the calling thread, then the engine thread builds everything CUDA inside `make` (no
//! CUDA value crosses a thread boundary).
//!
//! Lifetime: `spawn_engine` needs an executor without a lifetime and `QwenResidentStep` borrows
//! its owners, so the projector, the weights and the step resources are `Box::leak`ed once per
//! engine, giving `QwenResidentStep<'static>` (an owning `Arc` variant would rewrite the
//! qualified step, >= 1,000 lines). ONE CUDA engine is built per process: `BuildGuard` pins the
//! flag before `spawn_engine`, commits it right after the leaks, and every retry in the same
//! process is refused.
use crate::family::exl3::layer_table::LayerTable;

use super::super::admission_sketch::{CudaFitAsk, choose_cuda_fit};
use super::super::card_short::{self, CardFigures, CardShort, CardVerdict};
use super::super::krow::{SINGLE_GATHER_NAME, check_krow_env};
use super::super::rollback::{Cover, startup_line as rollback_startup_line};
use super::super::settings::{
    DEFAULT_CUDA_PREFILL_CHUNK, KvDtype, NO_BUDGET_PIECE, Q36Settings, RollbackMode, VerifyMode,
    WeightFormat, tokenizer_pin, weight_format_name,
};
use super::{
    card_load::card_loads,
    card_session::{CardSession, Q36CudaStep},
    card_step::CardStack,
    card_weights::{load_card_weights, load_card_weights_mapped},
    dims::{KvMode, StepDims, StepOptions, TieredPlan},
    exl3_load::{BankedModel, DecidedLoad, load_exl3_dir_decided},
    exl3_moe::Exl3MoeWeights,
    pinned::{ROTARY_DIM, head_rows_of, min_free_after_load},
    placement::{PlacementPolicy, PlacementRequest, plan_placement, profiles_from_facts},
    prefill_driver::{FIXED_ADMISSION_MARGIN, PrefillBudget},
    prefill_plan::Extra,
    resources::{PagedResourcesPlan, STEP_ROWS, StepResources, check_paged_modes, moe_dims},
    rules::QwenRules,
    step::{CaptureConfig, ChunkVerdict, QwenResidentStep},
    step_plan::{KernelScratch, StepPlan},
    tiered_load::{
        attach_tiered_parts, bank_slot_bytes, expert_quant, names_without_experts,
        planning_geometry, tiered_chunk_arm,
    },
};
use crate::family::decode_step::shard::ShardTable;
use crate::family::decode_step::state_layout::{KvPool, StateLayout};
use crate::family::engine_announce::{Announcement, announce_failure, await_announced};
use crate::family::exl3_merge_plan::{
    SHARED_IN_ROUTED_VAR, resolve_switch, shared_in_routed_explicit,
};
use crate::{
    EngError,
    family::{
        QWEN3_6_35B_A3B,
        exl3::bank::{Exl3SlotResolver, ExpertStore},
        gguf_model::{GgufModel, Inventory, Role, open_qwen3_6_35b_a3b},
        mem::Facts,
        project::cuda::{
            BorrowedLoader, CudaProjector, CudaWeights, LoadOptions, Reserve, WeightLayout,
            native_load_bytes, served_admission,
        },
        qwen::qwen36::host::HostShape,
        session::DecodeSession,
        speculate::config::DraftMode,
    },
    kernels::cuda::peer::{check_peer_facts, enable_peer_access, probe_device_set},
    kernels::cuda::resident::exl3::{SERVED_SINGLE_ROW, ServedKernels},
    sched::worker::{Engine, ReadyNote, spawn_engine_noted},
    tokenizer::TokenizerBundle,
};
use sha2::{Digest, Sha256};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

#[derive(Clone, Copy, PartialEq, Eq)]
enum PrefixMemory {
    Disabled,
    Pinned,
    DeviceTip,
}

impl PrefixMemory {
    fn from_flags(
        reuse: bool,
        pinned: Option<&std::ffi::OsStr>,
        device: Option<&std::ffi::OsStr>,
        default_on: bool,
    ) -> Self {
        let device = device.map_or(default_on, |value| value == "1");
        // Explicit device requests retain their pinned parent unless the pinned kill switch wins.
        let pinned = pinned.map_or(default_on || device, |value| value == "1");
        match (reuse && pinned, device) {
            (false, _) => Self::Disabled,
            (true, true) => Self::DeviceTip,
            (true, false) => Self::Pinned,
        }
    }
}

struct RuntimeModes {
    chunked: bool,
    /// `Q36_CUDA_CHUNK_PREFILL=0` was given: the rollback of the chunk path at every KV storage.
    chunk_off: bool,
    chunk_rows: u32,
    context_staging: bool,
    fixed_admission: bool,
    wide_stage: bool,
    prefix: PrefixMemory,
}

struct StartupPlan {
    reserve: u64,
    paged: Option<PagedResourcesPlan>,
    device_first_native: Option<usize>,
    verify: bool,
    allow_unpinned: bool,
    modes: RuntimeModes,
}

impl StartupPlan {
    /// Reserve only fixed storage before weights; exact native fit wins after the load.
    fn new_device_first(
        settings: &Q36Settings,
        verify: bool,
        allow_unpinned: bool,
        modes: RuntimeModes,
    ) -> Result<Self, EngError> {
        let Some(limit) = settings.kv_offload_host_bytes else {
            return Self::new(settings, verify, allow_unpinned, modes);
        };
        let kv_mode = serve_kv_mode(settings)?;
        if kv_mode != KvMode::Int8G32 || settings.tiered.is_some() {
            return Err(EngError::BadRequest(
                "device-first KV requires Native Int8G32 without tiered experts".into(),
            ));
        }
        if limit == 0 || usize::try_from(limit).is_err() {
            return Err(EngError::BadRequest(
                "kv_offload_host_bytes must be positive and fit usize".into(),
            ));
        }
        settings.check_prefix_reuse_cap(settings.prefix_reuse)?;
        check_paged_modes(true, settings.capture, modes.chunked)?;
        let inventory = Inventory::for_spec(&QWEN3_6_35B_A3B)?;
        let shape = HostShape::from_spec(&QWEN3_6_35B_A3B, ROTARY_DIM)?;
        let dims = StepDims::new(&shape, &inventory)?;
        let options = step_options(settings.context_limit, WeightLayout::Native, kv_mode)?;
        let state = StateLayout::new(&dims, &options)?;
        let scratch = KernelScratch::for_dims(&dims, STEP_ROWS, kv_mode)?;
        let step = StepPlan::new(&dims, STEP_ROWS, &options, &scratch)?;
        let reserve = state
            .bytes()?
            .checked_add(step.arena_bytes())
            .ok_or_else(|| {
                EngError::MemoryPlan("device-first resident reserve size overflows".into())
            })?;
        let native = reserve;
        let fixed = PagedResourcesPlan::fixed_device_bytes(&dims, &options)?;
        let reserve = u64::try_from(fixed.min(native))
            .map_err(|_| EngError::MemoryPlan("device-first fixed reserve exceeds u64".into()))?;
        Ok(Self {
            reserve,
            paged: None,
            device_first_native: Some(native),
            verify,
            allow_unpinned,
            modes,
        })
    }

    fn after_load(mut self, settings: &Q36Settings, free: usize) -> Result<Self, EngError> {
        let Some(native) = self.device_first_native else {
            return Ok(self);
        };
        let available = free.checked_sub(min_free_after_load()?).ok_or_else(|| {
            EngError::MemoryPlan("device-first load is below the free-memory floor".into())
        })?;
        if native <= available {
            self.reserve = u64::try_from(native)
                .map_err(|_| EngError::MemoryPlan("native reserve exceeds u64".into()))?;
            return Ok(self);
        }
        let fixed = usize::try_from(self.reserve)
            .map_err(|_| EngError::MemoryPlan("fixed cache reserve exceeds usize".into()))?;
        let budget = available.checked_sub(fixed).ok_or_else(|| {
            EngError::MemoryPlan("device-first fixed resources exceed free VRAM".into())
        })?;
        let inventory = Inventory::for_spec(&QWEN3_6_35B_A3B)?;
        let shape = HostShape::from_spec(&QWEN3_6_35B_A3B, ROTARY_DIM)?;
        let dims = StepDims::new(&shape, &inventory)?;
        let options = step_options(
            settings.context_limit,
            WeightLayout::Native,
            KvMode::Int8G32,
        )?;
        let host_limit = settings
            .kv_offload_host_bytes
            .and_then(|bytes| usize::try_from(bytes).ok())
            .ok_or_else(|| {
                EngError::BadRequest("device-first host quota is absent or invalid".into())
            })?;
        let cache = super::paging_store::device_first::DeviceCachePlan::new(
            &dims,
            options.context_tokens,
            budget,
            host_limit,
        )?;
        let paged = PagedResourcesPlan::new_cached(dims, &inventory, &options, cache)?;
        let device_bytes = paged.device_bytes()?;
        if device_bytes > available {
            return Err(EngError::MemoryPlan(
                "device cache resources exceed post-load VRAM".into(),
            ));
        }
        self.reserve = u64::try_from(device_bytes)
            .map_err(|_| EngError::MemoryPlan("device cache reserve exceeds u64".into()))?;
        self.paged = Some(paged);
        Ok(self)
    }

    fn new(
        settings: &Q36Settings,
        verify: bool,
        allow_unpinned: bool,
        modes: RuntimeModes,
    ) -> Result<Self, EngError> {
        let kv_mode = serve_kv_mode(settings)?;
        let capacity = context_tokens(settings.context_limit, kv_mode)?;
        let paged = match settings.kv_offload_host_bytes {
            None => None,
            Some(limit) => {
                settings.check_prefix_reuse_cap(settings.prefix_reuse)?;
                check_paged_modes(true, settings.capture, modes.chunked)?;
                let limit = usize::try_from(limit).map_err(|_| {
                    EngError::BadRequest("kv_offload_host_bytes exceeds usize".into())
                })?;
                let inventory = Inventory::for_spec(&QWEN3_6_35B_A3B)?;
                let shape = HostShape::from_spec(&QWEN3_6_35B_A3B, ROTARY_DIM)?;
                let dims = StepDims::new(&shape, &inventory)?;
                let mut options =
                    step_options(settings.context_limit, WeightLayout::Native, kv_mode)?;
                options.tiered = tiered_plan_of(settings);
                Some(PagedResourcesPlan::new(dims, &inventory, &options, limit)?)
            }
        };
        let reserve = match &paged {
            Some(plan) => u64::try_from(plan.device_bytes()?)
                .map_err(|_| EngError::MemoryPlan("paged resource reserve exceeds u64".into()))?,
            None => kv_cache_bytes(capacity, kv_mode)?,
        };
        Ok(Self {
            reserve,
            paged,
            device_first_native: None,
            verify,
            allow_unpinned,
            modes,
        })
    }
}

/// The pinned checkpoint of the CUDA arm: the file NAME inside `model_dir` and its exact SIZE,
/// both matching the pinned identity the qualification tool checks before every gated run. The
/// full-file SHA-256 beside them is checked only when `Q36_VERIFY_GGUF_SHA256` is `1`: hashing
/// 20 GB costs about 15-40 s per start (ESTIMATE) and the qualification script checks it.
pub(crate) const GGUF_NAME: &str = "Qwen3.6-35B-A3B-Q4_K_M.gguf";
pub(super) const GGUF_BYTES: u64 = 20_419_565_568;
const GGUF_SHA256: &str = "671e47e0ec53c665d048b98c3ecbfd5236b5ca9c3e02ed19fc8f81f7b85140c7";
/// The full-file SHA-256 is checked only when this variable is EXACTLY `1`: unset or `0`
/// leaves the check off, and every OTHER spelling — `true`, `yes`, an empty value — is a
/// typed refusal (raised in `make`, after the CUDA context exists), so a misspelled promise
/// cannot silently skip the check. A value Rust cannot read as UTF-8 reads as unset (std's
/// `env::var`), so it leaves the check off.
const VERIFY_SHA256_VAR: &str = "Q36_VERIFY_GGUF_SHA256";
/// The admission of a checkpoint other than the pinned file: unset or `1` serves the directory's
/// `.gguf` (the highest-precision one when there are several) when the pinned file is absent,
/// and the strict model open — architecture metadata, full tensor set, types and dims — stays
/// the gate, so only a Qwen3.6-35B-A3B conversion loads. `0` keeps the pinned-only behavior;
/// every other spelling is a typed refusal raised at startup, exactly like
/// `Q36_VERIFY_GGUF_SHA256`.
pub(crate) const ALLOW_UNPINNED_VAR: &str = "Q36_ALLOW_UNPINNED_GGUF";
const CHUNK_PREFILL_VAR: &str = "Q36_CUDA_CHUNK_PREFILL";
const CHUNK_ROWS_VAR: &str = "Q36_CUDA_CHUNK_ROWS";
const CONTEXT_STAGING_VAR: &str = "Q36_CUDA_KV8_STAGE_CONTEXT";
const FIXED_ADMISSION_VAR: &str = "Q36_CUDA_KV8_FIXED_ADMISSION";
const DEVICE_FIRST_VAR: &str = "Q36_CUDA_PAGED_DEVICE_FIRST";
const BANK_OVERLAP_VAR: &str = "Q36_CUDA_BANK_OVERLAP";
pub(super) const WIDE_STAGE_VAR: &str = "Q36_CUDA_CPF_STAGE_128K";

fn device_first_wanted(value: Option<&std::ffi::OsStr>) -> Result<bool, EngError> {
    match value {
        None => Ok(false),
        Some(value) if value == "0" => Ok(false),
        Some(value) if value == "1" => Ok(true),
        Some(_) => Err(EngError::BadRequest(format!(
            "{DEVICE_FIRST_VAR} must be 0 or 1"
        ))),
    }
}

fn require_device_first_room(free: usize, reserve: u64) -> Result<(), EngError> {
    let floor = min_free_after_load()?;
    let needed = usize::try_from(reserve)
        .ok()
        .and_then(|bytes| bytes.checked_add(floor))
        .ok_or_else(|| {
            EngError::MemoryPlan("device-first resident reserve plus headroom exceeds usize".into())
        })?;
    if free < needed {
        return Err(EngError::MemoryPlan(format!(
            "device-first fully-resident KV needs {needed} free bytes, found {free}"
        )));
    }
    Ok(())
}

fn chunk_prefill_rows(value: Option<&std::ffi::OsStr>) -> Result<u32, EngError> {
    match value {
        None => Ok(1024),
        Some(value) if value == "16" => Ok(16),
        Some(value) if value == "128" => Ok(128),
        Some(value) if value == "512" => Ok(512),
        Some(value) if value == "1024" => Ok(1024),
        Some(value) if value == "2048" => Ok(2048),
        Some(_) => Err(EngError::BadRequest(format!(
            "{CHUNK_ROWS_VAR} must be 16, 128, 512, 1024 or 2048"
        ))),
    }
}

fn chunk_prefill_wanted(kv: KvDtype, value: Option<&std::ffi::OsStr>) -> Result<bool, EngError> {
    prefill_flag(CHUNK_PREFILL_VAR, matches!(kv, KvDtype::Q8), value)
}

/// Whether `Q36_CUDA_CHUNK_PREFILL=0` was given. The flag's default depends on the KV storage, but an
/// explicit 0 is the rollback of the chunk path at every storage (the value was validated by
/// `chunk_prefill_wanted`).
fn chunk_rollback_requested(value: Option<&std::ffi::OsStr>) -> bool {
    value.is_some_and(|value| value == "0")
}

/// Whether the prefill arena is admitted. The FP16 storage chunks by default through the arena, so
/// the rollback must keep it from being admitted; Int8G32 consults the flag itself.
fn arena_wanted(kv: KvMode, chunk_off: bool) -> bool {
    !(kv == KvMode::Fp16 && chunk_off)
}

fn context_staging_wanted(value: Option<&std::ffi::OsStr>) -> Result<bool, EngError> {
    prefill_flag(CONTEXT_STAGING_VAR, true, value)
}

/// The wide 131072-row staged-split candidate is DEFAULT ON (against the 32768-row stage:
/// -7.4% TTFT at 128K, token-identical at 149731 and 127788 tokens, same binary, cold boots).
/// `Q36_CUDA_CPF_STAGE_128K=0` rolls it back to the 32768-row default stage.
fn wide_stage_wanted(value: Option<&std::ffi::OsStr>) -> Result<bool, EngError> {
    prefill_flag(WIDE_STAGE_VAR, true, value)
}

/// Whether `Q36_CUDA_BANK_OVERLAP=1` was given: a banked decode's miss refills move to a copy
/// stream and the MoE blocks launch their resident-expert work before the wait, hiding the
/// refill DMA under compute. Default off: the overlap is a measured, not assumed, win on the
/// small-VRAM banks it targets.
fn bank_overlap_wanted(value: Option<&std::ffi::OsStr>) -> Result<bool, EngError> {
    prefill_flag(BANK_OVERLAP_VAR, false, value)
}

fn fixed_admission_wanted(
    kv: KvDtype,
    chunked: bool,
    context_staging: bool,
    value: Option<&std::ffi::OsStr>,
) -> Result<bool, EngError> {
    Ok(prefill_flag(FIXED_ADMISSION_VAR, true, value)?
        && matches!(kv, KvDtype::Q8)
        && chunked
        && context_staging)
}

/// Why the wide CPF stage flag is inert, when it is: the wide candidate only
/// exists inside fixed admission, so name the mode that dropped fixed admission
/// (or that fixed admission itself is off) instead of failing silently.
fn wide_stage_inert_reason(
    fixed_admission: bool,
    kv: KvDtype,
    chunked: bool,
    context_staging: bool,
    tiered: bool,
    host_offload: bool,
) -> Option<&'static str> {
    if fixed_admission {
        return None;
    }
    Some(if host_offload {
        "host KV offload"
    } else if tiered {
        "tiered mode"
    } else if !matches!(kv, KvDtype::Q8) {
        "non-Q8 KV dtype"
    } else if !chunked {
        "chunked prefill off (Q36_CUDA_CHUNK_PREFILL=0)"
    } else if !context_staging {
        "context staging off (Q36_CUDA_KV8_STAGE_CONTEXT=0)"
    } else {
        "fixed admission off (Q36_CUDA_KV8_FIXED_ADMISSION=0)"
    })
}

pub(super) fn prefill_headroom(fixed_admission: bool) -> usize {
    if fixed_admission {
        FIXED_ADMISSION_MARGIN
    } else {
        HEADROOM as usize
    }
}

fn prefill_flag(
    name: &str,
    default: bool,
    value: Option<&std::ffi::OsStr>,
) -> Result<bool, EngError> {
    match value {
        None => Ok(default),
        Some(value) if value == "0" => Ok(false),
        Some(value) if value == "1" => Ok(true),
        Some(_) => Err(EngError::BadRequest(format!("{name} must be 0 or 1"))),
    }
}
/// Legacy admission headroom. The global free-after-load floor stays at 1 GiB even when
/// fixed admission uses its measured margin; `work: 0` is explained by `load_options`.
const HEADROOM: u64 = crate::family::mem::DEVICE.load_headroom;

pub(super) static CUDA_ENGINE_BUILT: AtomicBool = AtomicBool::new(false);

pub(super) fn already_built() -> EngError {
    EngError::BadRequest("engine config: the CUDA Q36 backend builds one engine per process: its weights are leaked for the process lifetime and an in-process restart would leak 20 GB again; configure a single qwen3_6_35b_a3b entry and restart the process to reload".to_string())
}

/// The one-engine guard: `acquire` takes the flag before `spawn_engine`, `commit` pins it
/// right after the first leak, and the `Drop` releases it only for a failure BEFORE the first
/// leak, when nothing is on the device and a retry may build.
pub(super) struct BuildGuard {
    committed: bool,
}

impl BuildGuard {
    pub(super) fn acquire() -> Result<Self, EngError> {
        CUDA_ENGINE_BUILT
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| already_built())?;
        Ok(Self { committed: false })
    }

    /// Pins the flag: a failure from here on leaves the ~20 GB on the device, so a retry in
    /// the same process must be refused, not run.
    pub(super) fn commit(&mut self) {
        self.committed = true;
    }
}

impl Drop for BuildGuard {
    /// Releases the flag only uncommitted: a failure before the first leak owns nothing.
    fn drop(&mut self) {
        if !self.committed {
            CUDA_ENGINE_BUILT.store(false, Ordering::Release);
        }
    }
}

/// The start line of one MLX-only key the CUDA arm ignores (the settings record the names in the
/// order of `Q36Options`). `prefill_chunk` keeps its line until chunked CUDA prefill accepts the
/// key as the cap of `M_sess`.
pub(crate) fn ignored_mlx_line(key: &str) -> &'static str {
    match key {
        "cache_limit_bytes" => "q36 cuda: cache_limit_bytes is an MLX allocator key and is ignored",
        "session_cap_bytes" => "q36 cuda: session_cap_bytes is an MLX session key and is ignored",
        "prefill_chunk" => "q36 cuda: prefill_chunk is an MLX chunked-prefill key and is ignored",
        "prefix_store_entries" => {
            "q36 cuda: prefix_store_entries is an MLX prefix-store key and is ignored"
        }
        _ => "q36 cuda: an unknown key of the object is ignored",
    }
}

/// The start line of the session's prefill path: one line per session build, printed after the
/// arena admission — the accept names the ADMITTED `M_sess`, every refusal names its reason, and
/// the match is exhaustive (no `_` arm), so a new verdict cannot go unlabelled. `make` prints it
/// once `admit_prefill` returns; the text is pinned by the host rows of `prefill_driver_tests`.
/// `pub(super)` like the enum it takes: a `pub(crate)` fn cannot expose the `pub(super)` verdict
/// type, and every caller — `make`, the host row, the device row — is in `cuda`.
pub(super) fn prefill_start_line(verdict: &ChunkVerdict) -> String {
    match verdict {
        ChunkVerdict::Ready { m_sess } => format!("q36 cuda: prefill chunk M_sess {m_sess}"),
        ChunkVerdict::TiledLayout => "q36 cuda: prefill per token: tiled layout".to_owned(),
        ChunkVerdict::Kv8 => "q36 cuda: prefill per token: 8-bit cache".to_owned(),
        ChunkVerdict::Tiered => "q36 cuda: prefill per token: tiered mode".to_owned(),
        ChunkVerdict::NoArena => "q36 cuda: prefill per token: no arena".to_owned(),
    }
}

fn kv8_prefill_start_line(verdict: &ChunkVerdict) -> String {
    let mode = if matches!(verdict, ChunkVerdict::Ready { .. }) {
        "chunked"
    } else {
        "per-token"
    };
    format!(
        "{} | KV8 mode={mode} | {CHUNK_PREFILL_VAR}=0 disables chunking",
        prefill_start_line(verdict)
    )
}

/// The pinned GGUF inside the artifact directory: the NAME is part of the identity, so the
/// join itself is the check.
pub(super) fn pinned_gguf(artifact: &Path) -> PathBuf {
    artifact.join(GGUF_NAME)
}

/// The GGUF to serve and whether it is the pinned file. The pinned file wins whenever it is
/// present; otherwise, with the unpinned admission on, the directory's `.gguf` serves, and an
/// empty directory is a typed refusal. A directory with several files serves the first the
/// ranking gives (`gguf_select`: the largest, so the highest precision); a path to one `.gguf`
/// file serves that file.
pub(crate) fn select_gguf(
    artifact: &Path,
    allow_unpinned: bool,
) -> Result<(PathBuf, bool), EngError> {
    select_gguf_noted(artifact, allow_unpinned).map(|(path, pinned, _)| (path, pinned))
}

/// [`select_gguf`] and the log line that names the other candidates when there was a choice.
pub(crate) fn select_gguf_noted(
    artifact: &Path,
    allow_unpinned: bool,
) -> Result<(PathBuf, bool, Option<String>), EngError> {
    let pinned = pinned_gguf(artifact);
    if pinned.is_file() {
        return Ok((pinned, true, None));
    }
    if artifact.is_file()
        && artifact
            .file_name()
            .is_some_and(|name| name == std::ffi::OsStr::new(GGUF_NAME))
    {
        return Ok((artifact.to_path_buf(), true, None));
    }
    if !allow_unpinned {
        return Err(EngError::Checkpoint(format!(
            "cuda checkpoint: {} is missing (pinned: {GGUF_NAME}, {GGUF_BYTES} bytes); set \
             {ALLOW_UNPINNED_VAR}=1 to serve the directory's single .gguf of the family",
            pinned.display()
        )));
    }
    let (_, files) = crate::family::gguf_select::candidates(artifact)
        .map_err(|error| EngError::Checkpoint(format!("cuda checkpoint: {}", error.describe())))?;
    // The weights stream through the tier ladder, so the first file in rank order serves.
    let choice = crate::family::gguf_select::choose(&files, &|_| Ok(()))?;
    Ok((choice.chosen.path, false, choice.line))
}

pub(super) fn sha256_wanted(raw: Option<&str>) -> Result<bool, EngError> {
    match raw {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(other) => Err(EngError::BadRequest(format!(
            "{VERIFY_SHA256_VAR} must be 0 or 1, found {other:?}"
        ))),
    }
}

/// The `Q36_ALLOW_UNPINNED_GGUF` resolution, the same promise shape as `sha256_wanted`: unset and
/// `1` admit an unpinned checkpoint, `0` keeps the pinned-only behavior, and every other
/// spelling is a typed refusal.
pub(crate) fn allow_unpinned_wanted(raw: Option<&str>) -> Result<bool, EngError> {
    match raw {
        Some("0") => Ok(false),
        None | Some("1") => Ok(true),
        Some(other) => Err(EngError::BadRequest(format!(
            "{ALLOW_UNPINNED_VAR} must be 0 or 1, found {other:?}"
        ))),
    }
}

pub(super) fn file_sha256(path: &Path) -> Result<String, EngError> {
    let read = |error: std::io::Error| {
        EngError::Checkpoint(format!("cuda checkpoint: {}: {error}", path.display()))
    };
    let mut file = std::fs::File::open(path).map_err(read)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 4 << 20];
    loop {
        let count = file.read(&mut buffer).map_err(read)?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect())
}

/// The full-file digest check, with the digest-and-elapsed log line printed only AFTER the
/// compare: a mismatch logs nothing here — its refusal names both digests.
pub(super) fn check_sha256(path: &Path, expected: &str) -> Result<(), EngError> {
    let began = Instant::now();
    let digest = file_sha256(path)?;
    if digest != expected {
        return Err(EngError::Checkpoint(format!(
            "cuda checkpoint: {} sha256 {digest} does not match the pinned file ({expected})",
            path.display()
        )));
    }
    eprintln!(
        "q36 cuda: gguf sha256 {digest} verified in {} s",
        began.elapsed().as_secs()
    );
    Ok(())
}

/// Checks the pinned identity of the GGUF: the NAME by construction (`select_gguf`), the exact
/// SIZE always (the metadata alone), the full-file SHA-256 only when `verify` asks for it. A
/// checkpoint admitted by `Q36_ALLOW_UNPINNED_GGUF` has no pinned size or digest (the EXL3
/// precedent); its identity is the strict model open, so this check passes it through. The
/// toggle value itself is resolved by the caller (`start()`), so a bad spelling is refused at
/// startup, before any device work; unset stays OFF with no error.
pub(crate) fn check_gguf_identity(path: &Path, verify: bool, pinned: bool) -> Result<(), EngError> {
    if !pinned {
        if verify {
            eprintln!(
                "q36 cuda: {VERIFY_SHA256_VAR}=1 has no pinned digest for {}; skipping",
                path.display()
            );
        }
        return Ok(());
    }
    let metadata = std::fs::metadata(path).map_err(|_| {
        EngError::Checkpoint(format!(
            "cuda checkpoint: {} is missing (pinned: {GGUF_NAME}, {GGUF_BYTES} bytes)",
            path.display()
        ))
    })?;
    let size = metadata.len();
    if size != GGUF_BYTES {
        // Another conversion under the same name: the strict model open decides whether it loads.
        eprintln!(
            "q36 cuda: {} has {size} bytes; the pinned {GGUF_NAME} has {GGUF_BYTES}; serving it",
            path.display()
        );
    }
    if verify {
        check_sha256(path, GGUF_SHA256)?;
    }
    Ok(())
}

pub(super) fn tiered_plan_of(settings: &Q36Settings) -> Option<TieredPlan> {
    settings.tiered.map(|tiered| TieredPlan {
        timeout_ms: tiered.timeout_ms,
        preload: tiered.preload,
        swaps_milli: tiered.swaps_milli,
        ..TieredPlan::from_slots(tiered.slots, tiered.threads)
    })
}

/// The refusal of a tiered start with graph capture, before any work (the settings refuse it too;
/// this is the start's own check): the tiered step waits for the host inside every layer.
pub(super) fn check_tiered_start(settings: &Q36Settings) -> Result<(), EngError> {
    if settings.tiered.is_some() && settings.capture {
        return Err(EngError::BadRequest(
            "engine config: tiered and capture exclude each other: the tiered step is eager only"
                .to_owned(),
        ));
    }
    Ok(())
}

/// The checked KV mode of the parsed `kv_dtype` member and the layout the step runs: absent or
/// `f32` is `Fp16` — the shared "f32" names the CONTROL cache, and on the CUDA arm the control
/// cache is the fp16 cache — and `q8` is `Int8G32`, which exists as Native op lists only, so
/// the Tiled layout is a typed refusal here (the plan refuses it again, `check_layout_rows`).
pub(super) fn kv_mode_of(kv: KvDtype, layout: WeightLayout) -> Result<KvMode, EngError> {
    match (kv, layout) {
        (KvDtype::F32, _) => Ok(KvMode::Fp16),
        (KvDtype::Q8, WeightLayout::Native | WeightLayout::Exl3) => Ok(KvMode::Int8G32),
        (KvDtype::Q8, WeightLayout::Tiled) => Err(EngError::BadRequest(
            "engine config: kv_dtype q8 builds the Int8G32 KV mode, which takes the native \
             layout; Tiled is the fp16 control path"
                .to_owned(),
        )),
        // The BF16 KV form exists on the MLX arm only.
        (KvDtype::Bf16, _) => Err(EngError::BadRequest(
            "engine config: kv_dtype bf16 is an MLX-arm KV form; the CUDA arm takes f32 (the \
             fp16 control cache) or q8"
                .to_owned(),
        )),
    }
}

/// The KV mode of the config for the serve admission: resolved against the layout the load
/// PINS (`load_options` writes `WeightLayout::Native` outright), so the admission needs no
/// device work. `make` derives the mode again from the loaded weights' OWN layout, so the
/// step's mode always agrees with the admission's.
pub(super) fn serve_kv_mode(settings: &Q36Settings) -> Result<KvMode, EngError> {
    kv_mode_of(settings.kv, WeightLayout::Native)
}

/// The session options: one committed DeltaNet slot, no forked experts, the fixed KV
/// capacity of `context_limit` positions (the step's context limit IS this number), the KV
/// mode of the config, and the step layout. `make` calls it AFTER the load with the loaded
/// weights' own layout (`device.layout()?`), so the step always runs the layout the device
/// holds.
pub(super) fn step_options(
    context_limit: u32,
    layout: WeightLayout,
    kv_mode: KvMode,
) -> Result<StepOptions, EngError> {
    Ok(StepOptions {
        kv_mode,
        context_tokens: context_tokens(context_limit, kv_mode)?,
        delta_slots: 1,
        forked_experts: false,
        tiered: None,
        layout,
    })
}

/// `start` validates the limit with this before any device work.
pub(super) fn context_tokens(context_limit: u32, kv_mode: KvMode) -> Result<usize, EngError> {
    if context_limit < 2 {
        return Err(EngError::BadRequest(format!(
            "engine config: context_limit must be at least 2, found {context_limit}"
        )));
    }
    let context_tokens = usize::try_from(context_limit)
        .map_err(|_| EngError::BadRequest("engine config: context_limit exceeds usize".into()))?;
    kv_cache_bytes(context_tokens, kv_mode)?;
    Ok(context_tokens)
}

/// The KV bytes of ONE position under `kv_mode`, straight from the pool's own layout
/// arithmetic (`KvPool::bytes_per_token`, 20,480 under Fp16 and 10,880 under Int8G32, pinned
/// by the start tests), never a constant: the admission must follow the KV mode the config
/// picked.
pub(super) fn kv_bytes_per_position(kv_mode: KvMode) -> Result<u64, EngError> {
    let inventory = Inventory::for_spec(&QWEN3_6_35B_A3B)?;
    let shape = HostShape::from_spec(&QWEN3_6_35B_A3B, ROTARY_DIM)?;
    let dims = StepDims::new(&shape, &inventory)?;
    u64::try_from(KvPool::bytes_per_token(&dims, kv_mode)?).map_err(|_| {
        EngError::MemoryPlan("cuda admission: the KV bytes per position exceed u64".into())
    })
}

/// The KV cache bytes of `context_tokens` positions of `kv_mode`, in checked arithmetic: the
/// number is the loader's admission term, so an impossible product is refused at planning
/// time. A u32 limit cannot overflow u64; the largest limits are refused by the admission
/// itself.
pub(super) fn kv_cache_bytes(context_tokens: usize, kv_mode: KvMode) -> Result<u64, EngError> {
    let positions = u64::try_from(context_tokens)
        .map_err(|_| EngError::MemoryPlan("cuda admission: KV positions exceed u64".into()))?;
    kv_bytes_per_position(kv_mode)?
        .checked_mul(positions)
        .ok_or_else(|| EngError::MemoryPlan("cuda admission: the KV bytes overflow u64".into()))
}

/// The load options of the engine: THE ONE PLACE the reserve terms and the load layout live
/// (the layout is written outright, never through a derived default). The layout is `Native`,
/// qualified on the 3090; the step follows the loaded weights, so this one field is the whole
/// switch. `work: 0` is right: `CudaProjector::new` allocates the matmul workspace BEFORE the
/// admission reads the free figure, so counting it in `work` too would double-count it.
pub(super) fn load_options(kv_cache: u64, fixed_admission: bool) -> LoadOptions<'static> {
    LoadOptions {
        reserve: Reserve {
            kv_cache,
            work: 0,
            headroom: prefill_headroom(fixed_admission) as u64,
        },
        drop_page_cache: None,
        layout: WeightLayout::Native,
    }
}

pub(super) fn open_gguf(
    artifact: &Path,
    verify: bool,
    allow_unpinned: bool,
    expected_sha256: Option<&str>,
) -> Result<(GgufModel, PathBuf), EngError> {
    let (gguf, pinned, note) = select_gguf_noted(artifact, allow_unpinned)?;
    if let Some(line) = note {
        eprintln!("q36 cuda: {line}");
    }
    if !pinned {
        eprintln!(
            "q36 cuda: serving checkpoint {}, which is not the pinned file (the model open \
             validates it; {ALLOW_UNPINNED_VAR}=0 refuses it)",
            gguf.display()
        );
    }
    // An explicit content pin replaces the built-in digest, but not filename/size admission.
    check_gguf_identity(&gguf, verify && expected_sha256.is_none(), pinned)?;
    if let Some(expected) = expected_sha256 {
        check_sha256(&gguf, expected)?;
    }
    let model = open_qwen3_6_35b_a3b(&gguf, None)?;
    Ok((model, gguf))
}

/// The runtime modes one settings record selects, from the environment: the prefill path, the
/// admission style and the prefix memory. The start asks it once, and asks again for the tiered
/// variant when the card cannot carry the load.
fn runtime_modes(settings: &Q36Settings) -> Result<RuntimeModes, EngError> {
    let chunked =
        chunk_prefill_wanted(settings.kv, std::env::var_os(CHUNK_PREFILL_VAR).as_deref())?;
    let chunk_off = chunk_rollback_requested(std::env::var_os(CHUNK_PREFILL_VAR).as_deref());
    let chunk_rows = if chunked {
        chunk_prefill_rows(std::env::var_os(CHUNK_ROWS_VAR).as_deref())?
    } else {
        512
    };
    check_tiered_start(settings)?;
    let context_staging = context_staging_wanted(std::env::var_os(CONTEXT_STAGING_VAR).as_deref())?;
    let wide_stage = wide_stage_wanted(std::env::var_os(WIDE_STAGE_VAR).as_deref())?;
    let fixed_admission = fixed_admission_wanted(
        settings.kv,
        chunked,
        context_staging,
        std::env::var_os(FIXED_ADMISSION_VAR).as_deref(),
    )? && settings.tiered.is_none()
        && settings.kv_offload_host_bytes.is_none();
    if let Some(reason) = wide_stage
        .then(|| {
            wide_stage_inert_reason(
                fixed_admission,
                settings.kv,
                chunked,
                context_staging,
                settings.tiered.is_some(),
                settings.kv_offload_host_bytes.is_some(),
            )
        })
        .flatten()
    {
        // Default ON: the inert note is operator-facing only when the flag
        // was explicitly set (unset+inert is the normal 131K/FP16/tiered
        // boot, not worth a line).
        if std::env::var_os(WIDE_STAGE_VAR).is_some() {
            eprintln!("q36 cuda: wide CPF stage flag has no effect: {reason}");
        }
    }
    let modes = RuntimeModes {
        chunked,
        chunk_off,
        chunk_rows,
        context_staging,
        fixed_admission,
        wide_stage,
        prefix: PrefixMemory::from_flags(
            settings.prefix_reuse,
            std::env::var_os("Q36_CUDA_PINNED_PREFIX").as_deref(),
            std::env::var_os("Q36_CUDA_STATE_ONLY_PREFIX").as_deref(),
            settings.tiered.is_none(),
        ),
    };
    Ok(modes)
}

/// Loads and pins the tokenizer, builds the rules from the SAME loaded bundle BEFORE any device
/// work (a missing tokenizer file fails in seconds, not after a 20 GB upload), takes the
/// one-engine guard, and starts the engine thread. The state is `Loading` until `make` returns;
/// a load failure leaves `Error` with the message and releases the guard.
pub(crate) fn start(
    settings: &Q36Settings,
    ready_note: ReadyNote,
) -> Result<(Engine, TokenizerBundle, Q36Settings), EngError> {
    // The toggle is resolved HERE, before any work: a bad `Q36_VERIFY_GGUF_SHA256` spelling is
    // refused at startup, not on the engine thread after the projector exists; unset stays OFF
    // with no error (`sha256_wanted(None)` is `Ok(false)`, pinned in start_tests).
    let verify = sha256_wanted(std::env::var(VERIFY_SHA256_VAR).ok().as_deref())?;
    // Likewise the unpinned admission: a bad `Q36_ALLOW_UNPINNED_GGUF` spelling stops the start
    // before any weight is read; unset stays off.
    let allow_unpinned = allow_unpinned_wanted(std::env::var(ALLOW_UNPINNED_VAR).ok().as_deref())?;
    // Likewise `FIFI_EXL3_SHARED_IN_ROUTED`: a bad value, or `1` on a GGUF checkpoint, stops the
    // start before any weight is read. Unset on a GGUF checkpoint is off and is not refused.
    resolve_switch(
        shared_in_routed_explicit(),
        matches!(settings.weights, WeightFormat::Exl3),
    )?;
    // `draft_verify = krow`: the one qualified configuration is checked before any weight is read.
    // The step's own checks of its fields run later, once the slots are armed.
    if settings.draft_verify == VerifyMode::Krow {
        check_krow_env(
            std::env::var_os(SINGLE_GATHER_NAME).as_deref(),
            std::env::var_os(SHARED_IN_ROUTED_VAR).as_deref(),
        )?;
    }
    let device_first = device_first_wanted(std::env::var_os(DEVICE_FIRST_VAR).as_deref())?;
    let plan = startup_plan_of(settings, verify, allow_unpinned)?;
    for key in &settings.ignored_mlx_keys {
        eprintln!("{}", ignored_mlx_line(key));
    }
    let bundle = TokenizerBundle::load(&settings.tokenizer_dir, &tokenizer_pin()?)?;
    let eos = bundle.eos_token_ids().to_vec();
    let sampler_vocab = usize::try_from(bundle.vocab_size())
        .map_err(|_| EngError::Checkpoint("tokenizer vocabulary exceeds usize".into()))?;
    let head_rows = head_rows_of()?;
    let rules = QwenRules::new(eos, sampler_vocab, head_rows)?;
    // The CUDA arm refuses the prefill budget, so its worker piece is the no-budget piece.
    let piece = usize::try_from(NO_BUDGET_PIECE).map_err(|_| {
        EngError::BadRequest("engine config: the worker piece exceeds usize".into())
    })?;
    let guard = BuildGuard::acquire()?;
    let owned = settings.clone();
    // A start that leaves the context or the cache form to the card gets them from the engine
    // thread once it has read the free memory; the server's request bound needs them before the
    // weights finish loading.
    let (announce, resolved) = if settings.context_auto || settings.kv_auto {
        let (sender, receiver) = std::sync::mpsc::sync_channel(1);
        (Some(sender), Some(receiver))
    } else {
        (None, None)
    };
    let sessions = session_plan(&owned, device_first);
    // A named tiered member sized its banks itself: nothing is funded for sessions beside them,
    // so a count above one is refused here, before any weight is read.
    if let (Some(count), Some(_), true) = (sessions, owned.tiered, owned.max_seqs_set) {
        return Err(EngError::BadRequest(format!(
            "engine config: max_seqs {count} with a named tiered member: the member sizes the \
             banks and funds no sessions beside them; drop the member and the planner funds \
             them, or drop max_seqs"
        )));
    }
    // A start that named a session count asked for sessions: a step that cannot host them refuses.
    let allow_single = !owned.max_seqs_set;
    let engine = if let Some(seqs) = sessions {
        spawn_engine_noted(piece, ready_note, move |stages| {
            let (step, core) = make_announced(stages, owned, rules, plan, guard, announce)?;
            crate::family::pooled_sessions::PooledSessions::new(step, core, seqs, allow_single)
        })?
    } else if owned.cuda_devices == [0] {
        spawn_engine_noted(piece, ready_note, move |stages| {
            make_announced(stages, owned, rules, plan, guard, announce)
        })?
    } else {
        spawn_engine_noted(piece, ready_note, move |stages| {
            make_multi(stages, owned, rules, plan, guard)
        })?
    };
    let served = match resolved {
        Some(receiver) => await_announced(&receiver, "q36 cuda", || engine.handle().failure())?,
        None => settings.clone(),
    };
    Ok((engine, bundle, served))
}

/// How many sessions a start serves and how it sizes them, or `None` for the one-session step.
///
/// A start that names no `max_seqs` runs the sessions executor whenever its form can hold more
/// than one session: the plain resident GGUF step on one card. The count then comes from the
/// cache pool, and `max_seqs` is only a cap (a named 1 keeps the one-session step).
fn session_plan(settings: &Q36Settings, device_first: bool) -> Option<usize> {
    if settings.cuda_devices != [0] {
        return None;
    }
    if settings.max_seqs_set {
        return (settings.max_seqs > 1).then_some(settings.max_seqs);
    }
    let plain = settings.weights == WeightFormat::Gguf
        && !settings.speculation.enabled()
        && settings.tiered.is_none()
        && settings.expert_bank.is_none()
        && settings.kv_offload_host_bytes.is_none()
        && settings.draft_rollback == RollbackMode::Host
        && settings.draft_verify == VerifyMode::Eager
        && !device_first
        && !super::sparse_exec::armed();
    plain.then_some(DEFAULT_SESSION_CAP)
}

/// The most sessions a default start serves; the pool decides how many are live.
const DEFAULT_SESSION_CAP: usize = 8;

fn p2p_host_expert_bytes(name: &str, value: Option<&std::ffi::OsStr>) -> Result<u64, EngError> {
    match value {
        None => Ok(0),
        Some(value) => value
            .to_str()
            .and_then(|value| value.parse::<u64>().ok())
            .ok_or_else(|| {
                EngError::BadRequest(format!("{name} must be a nonnegative byte count"))
            }),
    }
}

/// The engine-thread half: everything CUDA is built here, in the gate's own order — projector,
/// memory figure, GGUF identity, open, load with admission, free-after-load floor, the step
/// options from the loaded layout, step resources, then the leaks, the step and the capture
/// setting. The SHA-256 toggle arrives already resolved (`start()` validated it), so this
/// thread never reads the environment.
/// The multi-device half of `make`: the eager whole-token driver behind the
/// same `DecodeSession` seam. The order follows the single-device gate: the
/// device-set probe and peer enablement (this used to be `check_device_set`,
/// the probe-then-refuse gate), the GGUF identity, the shard table and
/// per-card load plans, one context and one resident state per card, the
/// per-card loads through the session's own contexts, the free-memory floor
/// on every card, then the leaks, the `BuildGuard` commit and the session.
///
/// v1 refuses what the card path has no plan for: EXL3 packs, tiered expert
/// banks, host-paged KV, graph capture and speculative decoding. The card
/// kernels are native-layout only, so `kv_mode_of` sees `Native` exactly as
/// the resident step's load does.
fn make_multi(
    stages: &crate::sched::worker::Stages<'_>,
    settings: Q36Settings,
    rules: QwenRules,
    plan: StartupPlan,
    mut guard: BuildGuard,
) -> Result<(Q36CudaStep, DecodeSession), EngError> {
    if let WeightFormat::Exl3 = settings.weights {
        return Err(EngError::BadRequest(
            "cuda_devices: multi-device serving takes the GGUF checkpoint; EXL3 packs are single-device"
                .into(),
        ));
    }
    if settings.tiered.is_some() {
        return Err(EngError::BadRequest(
            "cuda_devices: multi-device serving refuses the tiered expert banks".into(),
        ));
    }
    if settings.kv_offload_host_bytes.is_some() {
        return Err(EngError::BadRequest(
            "cuda_devices: multi-device serving refuses host-paged KV (kv_offload_host_bytes)"
                .into(),
        ));
    }
    if settings.capture {
        return Err(EngError::BadRequest(
            "cuda_devices: multi-device serving refuses graph capture".into(),
        ));
    }
    if settings.speculation.mode != DraftMode::Disabled {
        return Err(EngError::BadRequest(
            "cuda_devices: multi-device serving refuses speculative decoding".into(),
        ));
    }
    check_fused_inproj(
        settings.cuda_devices.len(),
        crate::family::project::cuda::gdn_fuse_inproj(),
    )?;
    stages.begin("probing devices");
    let facts = probe_device_set(&settings.cuda_devices)?;
    check_peer_facts(&facts).map_err(|refusal| {
        EngError::BadRequest(format!(
            "cuda_devices {:?}: {refusal}",
            settings.cuda_devices
        ))
    })?;
    enable_peer_access(&settings.cuda_devices)?;
    let first = &facts[0];
    eprintln!(
        "q36 cuda: devices {:?} probed: compute capability {}.{} on every device, peer access full",
        settings.cuda_devices, first.cc_major, first.cc_minor
    );
    let (model, gguf) = open_gguf(
        &settings.artifact,
        plan.verify,
        plan.allow_unpinned,
        settings.gguf_sha256.as_deref(),
    )?;
    let inventory = Inventory::for_spec(&QWEN3_6_35B_A3B)?;
    let shape = HostShape::from_spec(&QWEN3_6_35B_A3B, ROTARY_DIM)?;
    let dims = StepDims::new(&shape, &inventory)?;
    let kv_mode = kv_mode_of(settings.kv, WeightLayout::Native)?;
    let options = step_options(settings.context_limit, WeightLayout::Native, kv_mode)?;
    stages.begin("planning shards");
    // The shard plans size each card's upload from the file's expert quant; layer 0's
    // expert gate names the tensor that carries it.
    let layers = LayerTable::new(&dims, &inventory)?;
    let gate = layers.name(0, Role::FfnGateExps)?;
    let quant = model
        .weights
        .header()
        .tensor(gate)
        .ok_or_else(|| {
            EngError::Checkpoint(format!("gguf model: tensor {gate} is not in the file"))
        })?
        .ggml_type();
    // Experimental and default off: the bandwidth-waterfill placement
    // planner redistributes the sliceable roles instead of the even deal.
    // The planner budgets every sliced weight as card-resident, so each
    // card's host-expert budget is handed to it as free-byte relief: that
    // much sliced mass leaves the card for pinned host RAM at load. The
    // receipt's reason line says which candidate won and why.
    let waterfill = std::env::var_os("Q36_P2P_WATERFILL").is_some_and(|v| v == "1");
    let mut planned = None;
    let even_table;
    let table: &ShardTable = if waterfill {
        let mut profiles = profiles_from_facts(&facts);
        let host_expert_bytes = [
            p2p_host_expert_bytes(
                "Q36_P2P_HOST_EXPERT_BYTES_0",
                std::env::var_os("Q36_P2P_HOST_EXPERT_BYTES_0").as_deref(),
            )?,
            p2p_host_expert_bytes(
                "Q36_P2P_HOST_EXPERT_BYTES",
                std::env::var_os("Q36_P2P_HOST_EXPERT_BYTES").as_deref(),
            )?,
        ];
        for (profile, relief) in profiles.iter_mut().zip(host_expert_bytes) {
            profile.free_bytes += relief;
        }
        let policy = PlacementPolicy::default();
        let request = PlacementRequest {
            dims: &dims,
            kv_mode,
            inventory: &inventory,
            quant,
            kv_capacity_rows: settings.context_limit as usize,
            delta_slots: options.delta_slots,
            profiles: &profiles,
            policy: &policy,
        };
        let plan = plan_placement(&request)?;
        eprintln!(
            "q36 cuda: placement {} - {}",
            plan.receipt.chosen, plan.receipt.reason
        );
        planned = Some(plan);
        &planned.as_ref().expect("just set").table
    } else {
        even_table = ShardTable::new(&shape, settings.cuda_devices.len(), 256)?;
        &even_table
    };
    let loads = card_loads(table, &inventory, quant)?;
    let fixed_admission = plan.modes.fixed_admission;
    // The load reserve keeps the single-device figure except its state
    // term: the card caches are SLICED (each card stores only its own K/V
    // head groups), so the largest card's sliced state replaces the full
    // replica's. Over-reserving only refuses a tight fit earlier, never
    // corrupts it, and the post-load floor check still guards every card.
    let reserve = {
        let head_dim = dims.shape.attn.head_dim;
        let full_state = u64::try_from(StateLayout::new(&dims, &options)?.bytes()?)
            .map_err(|_| EngError::MemoryPlan("card state layout overflows u64".into()))?;
        let mut max_card_state: u64 = 0;
        for card in 0..settings.cuda_devices.len() {
            let heads = table.kv_rows(card)?.len() / head_dim;
            let bytes =
                u64::try_from(StateLayout::new_with_kv_heads(&dims, &options, heads)?.bytes()?)
                    .map_err(|_| EngError::MemoryPlan("card state layout overflows u64".into()))?;
            max_card_state = max_card_state.max(bytes);
        }
        plan.reserve
            .saturating_sub(full_state)
            .saturating_add(max_card_state)
    };
    let load = load_options(reserve, fixed_admission);
    // Experimental and default off. The original variable controls card 1;
    // the new card-0 variable leaves that existing setting compatible.
    let first_var = "Q36_P2P_HOST_EXPERT_BYTES_0";
    let second_var = "Q36_P2P_HOST_EXPERT_BYTES";
    let host_expert_bytes = [
        p2p_host_expert_bytes(first_var, std::env::var_os(first_var).as_deref())?,
        p2p_host_expert_bytes(second_var, std::env::var_os(second_var).as_deref())?,
    ];
    // The chunked session's layer sums carry m*hidden values, so the
    // mailbox payload grows past the eager cap when Q36_P2P_CHUNK is set
    // (capped at the transport's MAX_VALUES by max_chunk_rows).
    let chunk_rows = super::card_prefill::configured_chunk_rows(
        std::env::var_os("Q36_P2P_CHUNK").as_deref(),
        dims.shape.hidden,
    )?;
    let chunk_payload = chunk_rows
        .map(|rows| {
            super::card_prefill::chunk_mailbox_values(rows, dims.shape.hidden, dims.shape.moe.top_k)
        })
        .transpose()?;
    // The chunked session runs typed attention on both cards. Only the
    // adaptive arm (FIFI_CUDA_ATTN_PREFILL_ADAPT=1) re-arms per device;
    // the legacy arm is armed once per process and its launch fails on
    // the second card, so the unsupported combination is refused here.
    // The whole path is experimental and its two-card arms are unqualified.
    if chunk_payload.is_some() {
        let adapt = std::env::var_os("FIFI_CUDA_ATTN_PREFILL_ADAPT");
        if !adapt.as_deref().is_some_and(|v| v == "1") {
            return Err(EngError::BadRequest(
                "Q36_P2P_CHUNK requires FIFI_CUDA_ATTN_PREFILL_ADAPT=1: the two-card \
                 launch arms its attention per device only on the adaptive path"
                    .into(),
            ));
        }
        eprintln!(
            "q36 cuda: Q36_P2P_CHUNK chunked prefill is experimental; its two-card arms are unqualified"
        );
    }
    let (stack, states) = match (planned.as_ref(), chunk_payload) {
        (Some(plan), None) => CardStack::with_table(
            &dims,
            &options,
            &settings.cuda_devices,
            &served_admission(),
            &plan.table,
        )?,
        (None, None) => {
            CardStack::new(&dims, &options, &settings.cuda_devices, &served_admission())?
        }
        (Some(plan), Some(payload)) => CardStack::with_table_payload(
            &dims,
            &options,
            &settings.cuda_devices,
            &served_admission(),
            &plan.table,
            payload,
        )?,
        (None, Some(payload)) => CardStack::with_payload(
            &dims,
            &options,
            &settings.cuda_devices,
            &served_admission(),
            &ShardTable::new(&dims.shape, settings.cuda_devices.len(), 32)?,
            Some(payload),
        )?,
    };
    stages.begin("loading weights");
    let mut loaded = Vec::with_capacity(loads.len());
    for (card, set) in loads.iter().enumerate() {
        let loader = BorrowedLoader::new(&stack.contexts()[card])?;
        let budget = host_expert_bytes.get(card).copied().unwrap_or(0);
        let weights = if budget != 0 {
            load_card_weights_mapped(&loader, &model.weights, set, card, load, budget)?
        } else {
            load_card_weights(&loader, &model.weights, set, card, load)?
        };
        loaded.push(weights);
    }
    stages.begin("checking memory floors");
    for (card, context) in stack.contexts().iter().enumerate() {
        let info = context.memory_info()?;
        let floor = min_free_after_load()?;
        if info.free < floor {
            return Err(EngError::MemoryPlan(format!(
                "cuda admission: card {card} has {} bytes free after the load, below the {}-byte floor",
                info.free, floor
            )));
        }
        eprintln!(
            "q36 cuda: card {card} (device {}) weights loaded ({} B), free after load {} B",
            settings.cuda_devices[card],
            loads[card].upload_bytes(),
            info.free
        );
    }
    // LEAK: the same lifetime pattern as the single-device `make` - the stack,
    // the layer table and the weight sets are leaked for the process
    // lifetime, giving the session its 'static views; `guard.commit()` below
    // pins the one-engine flag, so any retry after this point is refused with
    // the documented text.
    let stack: &'static CardStack = Box::leak(Box::new(stack));
    let layers: &'static LayerTable = Box::leak(Box::new(layers));
    let loaded: &'static [CudaWeights] = Box::leak(loaded.into_boxed_slice());
    let weights: &'static [&'static CudaWeights] =
        Box::leak(loaded.iter().collect::<Vec<_>>().into_boxed_slice());
    guard.commit();
    let session = CardSession::new(
        stack,
        states,
        weights,
        layers,
        &inventory,
        &dims,
        &options,
        rules.eos,
        rules.sampler_vocab,
        // The identity half of the prefix path (see `gguf_prefix_identity`).
        gguf_prefix_identity(&model.weights, &gguf, settings.gguf_sha256.as_deref())?,
        // Experimental and default off: the card path's prefix save/restore
        // is wired but NOT yet qualified (the restore passed filler-agnostic
        // bench answers and failed a deeper turn-chain check; see the branch
        // doc). Q36_P2P_PREFIX=1 opts in.
        if std::env::var_os("Q36_P2P_PREFIX").is_some_and(|v| v == "1") {
            crate::family::qwen::qwen36::settings::prefix_store_limits_of(
                settings.prefix_reuse,
                settings.prefix_store_entries,
            )
        } else {
            crate::family::session::StoreLimits {
                roots: 0,
                entries: 0,
            }
        },
        chunk_rows,
    )?;
    // Experimental and default off: chunked prefill through both cards
    // (card_prefill.rs). Q36_P2P_CHUNK=<rows> sets the chunk size, capped
    // at MAX_VALUES/hidden; the views are freshly resolved from the same
    // leaked weights the session holds, so the chunk models live beside
    // the session in the enum. A refusal logs and leaves the per-token
    // path in place.
    Ok((
        Q36CudaStep::Cards(Box::new(session)),
        DecodeSession::default(),
    ))
}

/// Multi-device serving refuses the GDN input-projection fusion: the fused
/// `attn_qkv` matrix is a row concatenation whose multi-card slice plan
/// (channel segments plus the z/beta/alpha siblings) has neither a loader
/// slice nor alignment tests. Refusing here keeps the sharded path on the
/// four separate projections the shard table already deals.
fn check_fused_inproj(cards: usize, fused: bool) -> Result<(), EngError> {
    if cards > 1 && fused {
        return Err(EngError::BadRequest(
            "cuda_devices: multi-device serving refuses INFERENCE_CUDA_GDN_FUSE_INPROJ=1; the \
             fused input projection has no multi-card slice plan"
                .into(),
        ));
    }
    Ok(())
}

/// The card-short decision of one GGUF start, asked before any weight is uploaded. The card's
/// figures are the ones the load's admission compares (the native tensor bytes, the KV reserve,
/// the headroom, the free bytes), so a start that fits answers `None` and goes on unchanged. A
/// card-short start that the ladder says card plus host can carry answers the tiered variant of
/// the settings with its startup plan and the decision; one it cannot carry prints why and goes
/// on to the load's own refusal. A config that names tiered experts, host-paged KV or a device
/// list is never asked.
fn lower_rung(
    projector: &CudaProjector,
    settings: &Q36Settings,
    plan: &StartupPlan,
    model: &GgufModel,
) -> Result<Option<(Q36Settings, StartupPlan, CardShort)>, EngError> {
    if plan.paged.is_some() || !card_short::automatic(settings, plan.device_first_native.is_some())
    {
        return Ok(None);
    }
    let inventory = Inventory::for_spec(&QWEN3_6_35B_A3B)?;
    let shape = HostShape::from_spec(&QWEN3_6_35B_A3B, ROTARY_DIM)?;
    let dims = StepDims::new(&shape, &inventory)?;
    let names = model.plan().device_names();
    let resident_names = names_without_experts(&names);
    let tensors = native_load_bytes(&model.weights, &names)?;
    let resident = native_load_bytes(&model.weights, &resident_names)?;
    let geometry = planning_geometry(&dims, &expert_quant(model.weights.header(), &dims)?)?;
    // The card and the host of this decision, read together.
    let facts = Facts::read_card(("q36 cuda tiered", "Qwen36"), projector.card_bytes())?;
    let free = facts.card_free();
    let figures = CardFigures {
        free,
        tensors,
        expert_bytes: tensors.saturating_sub(resident),
        kv_cache: plan.reserve,
        work: load_options(plan.reserve, plan.modes.fixed_admission)
            .reserve
            .work,
        headroom: prefill_headroom(plan.modes.fixed_admission) as u64,
        // The tiered variant runs with the fixed admission off: its banks admit with the legacy
        // headroom, so the ladder promises slots against that one.
        tiered_headroom: prefill_headroom(false) as u64,
        slot_bytes: bank_slot_bytes(geometry)?,
        after_load_floor: after_load_floor()?,
        session_part_bytes: session_part_bytes(&dims, settings)?,
        sessions: planned_sessions(settings),
        sessions_named: settings.max_seqs_set,
        layers: dims.shape.kinds.len(),
        experts: dims.shape.moe.experts,
        top_k: dims.shape.moe.top_k,
    };
    match card_short::decide(&figures, facts.host.budget()) {
        CardVerdict::Fits => Ok(None),
        CardVerdict::Refused { line } => {
            eprintln!("{line}");
            Ok(None)
        }
        CardVerdict::Short(short) => {
            let variant = card_short::tiered_variant(settings);
            let modes = runtime_modes(&variant)?;
            let variant_plan = StartupPlan::new(&variant, plan.verify, plan.allow_unpinned, modes)?;
            Ok(Some((variant, variant_plan, short)))
        }
    }
}

/// The load identity of a GGUF checkpoint for the prefix paths: a digest of the header, a sample of
/// every tensor's bytes, the file length and modification time, and the operator's pinned
/// full-file digest when there is one (`gguf_identity_digest` says what a sample can miss), plus
/// the pinned tokenizer digest and chat-template digest. Constant for one loaded checkpoint, so
/// folding it into a path leaves live save/restore behaviour unchanged.
fn gguf_prefix_identity(
    weights: &crate::weights::GgufWeights,
    gguf: &Path,
    pinned_sha256: Option<&str>,
) -> Result<crate::family::session::PrefixIdentity, EngError> {
    let digest = crate::weights::gguf_identity_digest(weights.header(), gguf, pinned_sha256)
        .map_err(|error| EngError::Checkpoint(format!("prefix identity: {error}")))?;
    let pin = tokenizer_pin()?;
    Ok(crate::family::session::PrefixIdentity {
        weights: crate::weights::sha256_hex(&digest).into(),
        tokenizer: crate::weights::sha256_hex(&pin.tokenizer_json).into(),
        template: crate::weights::sha256_hex(&pin.chat_template_jinja).into(),
    })
}

/// The device bytes the load's own admission asks for beside the weights at a cache form and
/// context: the cache, the work and the headroom of the plan that start would run.
fn load_need_of<'a>(
    settings: &'a Q36Settings,
    plan: &'a StartupPlan,
) -> impl Fn(KvDtype, u32) -> Option<u64> + 'a {
    move |kv, context| {
        let variant = Q36Settings {
            kv,
            context_limit: context,
            ..settings.clone()
        };
        let modes = runtime_modes(&variant).ok()?;
        let probe = StartupPlan::new(&variant, plan.verify, plan.allow_unpinned, modes).ok()?;
        let reserve = load_options(probe.reserve, probe.modes.fixed_admission).reserve;
        probe
            .reserve
            .checked_add(reserve.work)?
            .checked_add(reserve.headroom)
    }
}

/// Whether the step can run the int8 cache: its chunk prefill needs the prefill arena, which the
/// chunked prefill (the int8 cache's own default, switched off by `Q36_CUDA_CHUNK_PREFILL=0`)
/// provides. The flag is read for the int8 form, not for the plan the config builds, whose form is
/// still undecided. The answer is the prefill admission's own predicate (`needs_arena`), so the card
/// choice never plans a form the step then refuses ("fixed prefill admission: this step cannot use
/// a chunk arena").
fn int8_cache_usable() -> bool {
    let chunked = std::env::var_os(CHUNK_PREFILL_VAR)
        .and_then(|value| chunk_prefill_wanted(KvDtype::Q8, Some(&value)).ok())
        .unwrap_or(true);
    super::prefill_driver::needs_arena(WeightLayout::Native, KvMode::Int8G32, false, chunked)
}

/// Whether the prefill admission keeps a chunk arena for a cache form at a context, from the free
/// bytes before the weights are uploaded: the admission's own rule for the form (the unfixed
/// ladder for fp16, `fixed_kv8_passes` for int8) over the state at that context, with what the
/// weight load takes beyond its planned tensor bytes and the measurement band held back, the
/// headroom, the decode arena and the device prefix tip charged as the admission charges them.
/// A question it cannot answer is an error, never a yes.
fn prefill_admits_of<'a>(
    nsm: usize,
    settings: &'a Q36Settings,
    plan: &'a StartupPlan,
    (free, weights): (u64, u64),
) -> Result<impl Fn(KvDtype, u32) -> Result<bool, EngError> + 'a, EngError> {
    let dims = StepDims::new(
        &HostShape::from_spec(&QWEN3_6_35B_A3B, ROTARY_DIM)?,
        &Inventory::for_spec(&QWEN3_6_35B_A3B)?,
    )?;
    let tip = match plan.modes.prefix {
        PrefixMemory::DeviceTip => {
            crate::family::qwen::qwen36::settings::delta_state_bytes(&settings.spec)?
        }
        _ => 0,
    };
    let cap = chunk_prefill_rows(std::env::var_os(CHUNK_ROWS_VAR).as_deref())? as usize;
    let after_load = free
        .saturating_sub(weights)
        .saturating_sub(crate::family::mem::DEVICE.load_unplanned)
        .saturating_sub(crate::family::admission_sketch::ADMISSION_UNCERTAINTY_BAND_BYTES);
    let bytes = |value: u64| usize::try_from(value).unwrap_or(usize::MAX);
    Ok(move |kv: KvDtype, context: u32| {
        let mode = kv_mode_of(kv, WeightLayout::Native)?;
        let options = step_options(context, WeightLayout::Native, mode)?;
        let state = StateLayout::new(&dims, &options)?.bytes()?;
        let scratch = KernelScratch::for_dims(&dims, STEP_ROWS, mode)?;
        let decode = StepPlan::new(&dims, STEP_ROWS, &options, &scratch)?.arena_bytes();
        match mode {
            KvMode::Fp16 => super::prefill_plan::smallest_chunk_passes(
                &dims,
                nsm,
                bytes(after_load.saturating_sub(prefill_headroom(false) as u64)),
                (decode, bytes(tip)),
                state,
            ),
            KvMode::Int8G32 => {
                let resident = (state as u64) + (decode as u64) + tip;
                super::prefill_driver::fixed_kv8_passes(
                    &dims,
                    nsm,
                    (
                        bytes(after_load.saturating_sub(prefill_headroom(true) as u64)),
                        bytes(after_load.saturating_sub(resident)),
                    ),
                    (decode, bytes(tip)),
                    (state, context as usize, cap),
                )
            }
        }
    })
}

/// Whether `free` bytes hold `weights` and still leave `floor` free after the load.
fn weights_leave_the_floor(free: u64, weights: u64, floor: u64) -> bool {
    free >= weights.saturating_add(floor)
}

/// The sessions a start asks the lower rung to fund beside its banks: the named count, else the
/// tiered target. The planner takes what the card funds up to it.
fn planned_sessions(settings: &Q36Settings) -> usize {
    if settings.max_seqs_set {
        settings.max_seqs
    } else {
        card_short::TIERED_SESSION_TARGET
    }
}

/// Device bytes of one session's part at its initial cells: the planes, the DeltaNet and the
/// convolution state a session after the first keeps beside the banks.
fn session_part_bytes(dims: &StepDims, settings: &Q36Settings) -> Result<u64, EngError> {
    let kv_mode = kv_mode_of(settings.kv, WeightLayout::Native)?;
    let options = step_options(
        u32::try_from(crate::family::cell_pool::INITIAL_CELLS).unwrap_or(u32::MAX),
        WeightLayout::Native,
        kv_mode,
    )?;
    Ok(u64::try_from(StateLayout::new(dims, &options)?.bytes()?).unwrap_or(u64::MAX))
}

/// The bytes the weights must leave free once they are loaded: what the load takes beyond the
/// planned tensors, plus the free-after-load floor the start refuses below. The cache, the work
/// and the headroom come after that check, so no context changes it.
fn after_load_floor() -> Result<u64, EngError> {
    let floor = u64::try_from(min_free_after_load()?).unwrap_or(u64::MAX);
    Ok(crate::family::mem::DEVICE
        .load_unplanned
        .saturating_add(floor))
}

/// Asks the card fit again with no prefill gate: whether a fit exists at all. A fit that exists
/// without the gate and not with it means the card holds the weights and a cache but leaves no
/// room for a prefill arena at any context, which the start refuses by name.
fn require_prefill_room(
    settings: &Q36Settings,
    ask: CudaFitAsk,
    load_need: &dyn Fn(KvDtype, u32) -> Option<u64>,
) -> Result<(), EngError> {
    match choose_cuda_fit(&settings.spec, ask, load_need, &|_, _| Ok(true))? {
        Some(_) => Err(EngError::MemoryPlan(
            "q36 cuda: the card holds the weights and a KV cache but leaves no room for a prefill \
             arena at any context; name a smaller context_limit or a cache form that leaves \
             room"
                .to_owned(),
        )),
        None => Ok(()),
    }
}

/// The multiprocessor count the prefill admission sizes its chunk arena for.
fn device_nsm(projector: &CudaProjector) -> Result<usize, EngError> {
    Ok(projector.context().mmq_device_facts()?.nsm)
}

/// What the card choice reads from the device: the free bytes before the weights are uploaded and
/// the multiprocessor count.
#[derive(Clone, Copy)]
pub(crate) struct DeviceReading {
    pub(crate) free: u64,
    pub(crate) nsm: usize,
}

/// A card choice: the settings and startup plan of the cache form and context the card carries,
/// with the figures it was made from.
struct CardChoice {
    variant: Q36Settings,
    plan: StartupPlan,
    fit: super::super::admission_sketch::CudaFit,
    free: u64,
}

/// The cache form and context a start that left them to the card serves, from the device reading
/// and the weight bytes the load pins. `None` when the config fixes both (or fits no form at all,
/// which the lower rung or the load's own refusal then handles). The start and `check` both
/// decide through this function.
fn card_choice(
    reading: DeviceReading,
    settings: &Q36Settings,
    plan: &StartupPlan,
    weights: u64,
) -> Result<Option<CardChoice>, EngError> {
    if !(settings.kv_auto || settings.context_auto)
        || plan.paged.is_some()
        || plan.device_first_native.is_some()
        || !card_short::automatic(settings, false)
    {
        return Ok(None);
    }
    let free = reading.free;
    // A card that leaves the weights below the free-after-load floor holds no context: the lower
    // rung decides, before anything is uploaded (the same figure it is asked with).
    if !weights_leave_the_floor(free, weights, after_load_floor()?) {
        return Ok(None);
    }
    let ask = CudaFitAsk {
        free,
        weights,
        int8_usable: int8_cache_usable(),
        kv: (!settings.kv_auto).then_some(settings.kv),
        context: (!settings.context_auto).then_some(settings.context_limit),
    };
    let load_need = load_need_of(settings, plan);
    let Some(fit) = choose_cuda_fit(
        &settings.spec,
        ask,
        &load_need,
        &prefill_admits_of(reading.nsm, settings, plan, (free, weights))?,
    )?
    else {
        require_prefill_room(settings, ask, &load_need)?;
        return Ok(None);
    };
    let variant = Q36Settings {
        kv: fit.form,
        context_limit: fit.context,
        ..settings.clone()
    };
    let modes = runtime_modes(&variant)?;
    let variant_plan = StartupPlan::new(&variant, plan.verify, plan.allow_unpinned, modes)?;
    Ok(Some(CardChoice {
        variant,
        plan: variant_plan,
        fit,
        free,
    }))
}

/// The cache form and context of a start that left them to the card, asked before any weight is
/// uploaded. The free bytes are the ones the load's admission compares and the weight bytes are
/// its native tensors, so the choice charges what the load will. A start whose config fixes both
/// (or fits no form at all, which the lower rung or the load's own refusal then handles) answers
/// `None` and goes on unchanged. The variant carries the chosen figures with its own runtime
/// modes and startup plan, since the cache form picks the prefill path.
fn card_fit(
    projector: &CudaProjector,
    settings: &Q36Settings,
    plan: &StartupPlan,
    model: &GgufModel,
) -> Result<Option<(Q36Settings, StartupPlan)>, EngError> {
    let weights = native_load_bytes(&model.weights, &model.plan().device_names())?;
    let reading = DeviceReading {
        free: u64::try_from(projector.memory_info()?.free).unwrap_or(u64::MAX),
        nsm: device_nsm(projector)?,
    };
    let Some(choice) = card_choice(reading, settings, plan, weights)? else {
        return Ok(None);
    };
    let (free, fit) = (choice.free, choice.fit);
    let form = match fit.form {
        KvDtype::Q8 => "int8",
        KvDtype::F32 | KvDtype::Bf16 => "fp16",
    };
    if !settings.context_auto && fit.context < settings.context_limit {
        eprintln!(
            "q36 cuda: the named context_limit {} does not fit this card; starting at {}, the most the {form} KV cache holds beside the weights ({free} bytes free, {weights} bytes of weights)",
            settings.context_limit, fit.context
        );
    } else {
        eprintln!(
            "q36 cuda: context {} with the {form} KV cache, chosen from the card ({free} bytes free, {weights} bytes of weights); set context_limit and kv_dtype to override",
            fit.context
        );
    }
    Ok(Some((choice.variant, choice.plan)))
}

/// The startup plan of settings: the same one `start` builds before it reads the card.
fn startup_plan_of(
    settings: &Q36Settings,
    verify: bool,
    allow_unpinned: bool,
) -> Result<StartupPlan, EngError> {
    let modes = runtime_modes(settings)?;
    let device_first = device_first_wanted(std::env::var_os(DEVICE_FIRST_VAR).as_deref())?;
    if device_first {
        StartupPlan::new_device_first(settings, verify, allow_unpinned, modes)
    } else {
        StartupPlan::new(settings, verify, allow_unpinned, modes)
    }
}

/// The cache form and context a start with these settings would serve on this card, for
/// `check`: the same choice `start` makes (`card_choice`), from the card read now and the weights
/// of the checkpoint the start selects. `None` when the settings fix both, the card fits no form,
/// or the checkpoint cannot be opened (the load reports that itself).
pub(crate) fn planned_fit(settings: &Q36Settings) -> Result<Option<(KvDtype, u32)>, EngError> {
    if settings.weights != WeightFormat::Gguf {
        return Ok(None);
    }
    let allow_unpinned = allow_unpinned_wanted(std::env::var(ALLOW_UNPINNED_VAR).ok().as_deref())?;
    let plan = startup_plan_of(settings, false, allow_unpinned)?;
    let (gguf, _, _) = select_gguf_noted(&settings.artifact, allow_unpinned)?;
    let model = open_qwen3_6_35b_a3b(&gguf, None)?;
    let weights = native_load_bytes(&model.weights, &model.plan().device_names())?;
    let projector = CudaProjector::new()?;
    let reading = DeviceReading {
        free: u64::try_from(projector.memory_info()?.free).unwrap_or(u64::MAX),
        nsm: device_nsm(&projector)?,
    };
    planned_fit_from(reading, settings, &plan, weights)
}

/// `planned_fit` from figures already read.
fn planned_fit_from(
    reading: DeviceReading,
    settings: &Q36Settings,
    plan: &StartupPlan,
    weights: u64,
) -> Result<Option<(KvDtype, u32)>, EngError> {
    Ok(card_choice(reading, settings, plan, weights)?
        .map(|choice| (choice.variant.kv, choice.variant.context_limit)))
}

/// Positions the context is lowered by when the load's own admission refuses a context the card
/// planned.
const REPLAN_STEP: u32 = 4096;

/// The plan of a start whose load refused the context the card planned: the same cache form at
/// the largest context below the refused one that the card now holds, from a fresh read of the
/// free bytes. `None` when the start named both settings (nothing was planned) or nothing
/// smaller fits.
fn replan_smaller(
    projector: &CudaProjector,
    settings: &Q36Settings,
    plan: &StartupPlan,
    model: &GgufModel,
) -> Result<Option<(Q36Settings, StartupPlan)>, EngError> {
    if !settings.context_auto || plan.paged.is_some() || plan.device_first_native.is_some() {
        return Ok(None);
    }
    let weights = native_load_bytes(&model.weights, &model.plan().device_names())?;
    let free = u64::try_from(projector.memory_info()?.free).unwrap_or(u64::MAX);
    if !weights_leave_the_floor(free, weights, after_load_floor()?) {
        return Ok(None);
    }
    let Some(fit) = choose_cuda_fit(
        &settings.spec,
        CudaFitAsk {
            free,
            weights,
            int8_usable: true,
            kv: Some(settings.kv),
            context: Some(settings.context_limit.saturating_sub(REPLAN_STEP)),
        },
        &load_need_of(settings, plan),
        &prefill_admits_of(device_nsm(projector)?, settings, plan, (free, weights))?,
    )?
    else {
        return Ok(None);
    };
    let variant = Q36Settings {
        context_limit: fit.context,
        ..settings.clone()
    };
    let modes = runtime_modes(&variant)?;
    let variant_plan = StartupPlan::new(&variant, plan.verify, plan.allow_unpinned, modes)?;
    eprintln!(
        "q36 cuda: the load's admission refused {} positions ({free} bytes free now); starting at {}",
        settings.context_limit, fit.context
    );
    Ok(Some((variant, variant_plan)))
}

fn make_announced(
    stages: &crate::sched::worker::Stages<'_>,
    settings: Q36Settings,
    rules: QwenRules,
    plan: StartupPlan,
    guard: BuildGuard,
    announce: Option<std::sync::mpsc::SyncSender<Announcement<Q36Settings>>>,
) -> Result<(QwenResidentStep<'static>, DecodeSession), EngError> {
    let built = make(stages, settings, rules, plan, guard, announce.as_ref());
    announce_failure(&built, announce.as_ref());
    built
}

/// The engine-thread build: reads the card, settles the cache form and context the start left
/// open, tells `start` what it serves ([`make_announced`] reports a failure before that), then
/// loads.
fn make(
    stages: &crate::sched::worker::Stages<'_>,
    settings: Q36Settings,
    rules: QwenRules,
    plan: StartupPlan,
    mut guard: BuildGuard,
    announce: Option<&std::sync::mpsc::SyncSender<Announcement<Q36Settings>>>,
) -> Result<(QwenResidentStep<'static>, DecodeSession), EngError> {
    let verify = plan.verify;
    let allow_unpinned = plan.allow_unpinned;
    stages.begin("initializing CUDA");
    let projector = CudaProjector::new()?;
    let before = projector.memory_info()?;
    stages.begin("loading weights");
    let gguf = match settings.weights {
        WeightFormat::Gguf => Some(open_gguf(
            &settings.artifact,
            verify,
            allow_unpinned,
            settings.gguf_sha256.as_deref(),
        )?),
        WeightFormat::Exl3 => None,
    };
    // The cache form and context the config left open come from the free card.
    let (settings, plan) = match gguf.as_ref() {
        Some((model, _)) => match card_fit(&projector, &settings, &plan, model)? {
            Some((variant, variant_plan)) => (variant, variant_plan),
            None => (settings, plan),
        },
        None => (settings, plan),
    };
    // The lower rung: a GGUF start the card cannot carry whole serves the tiered path when card
    // plus host can, instead of refusing at the load's admission.
    let (mut settings, mut plan, card_short) = match gguf.as_ref() {
        Some((model, _)) => match lower_rung(&projector, &settings, &plan, model)? {
            Some((variant, variant_plan, short)) => (variant, variant_plan, Some(short)),
            None => (settings, plan, None),
        },
        None => (settings, plan, None),
    };
    if let Some(announce) = announce {
        // A full slot means `start` already has an answer; a closed channel means it gave up.
        let _ = announce.try_send(Ok(settings.clone()));
    }
    let mut kv_cache = plan.reserve;
    let mut fixed_admission = plan.modes.fixed_admission;
    let tiered = tiered_plan_of(&settings);
    // The banked EXL3 load's leftovers (None on every resident path): the host store and the
    // figures the resolver build below needs. Declared here because the load match's arms are
    // separate blocks.
    let mut banked_load: Option<(ExpertStore, usize, usize)> = None;
    // A tiered load leaves the routed expert stacks in the file: the banks hold the hot ones.
    let device = match (&gguf, tiered) {
        (None, _) => {
            let reserve = load_options(kv_cache, fixed_admission).reserve;
            let began = Instant::now();
            // A served EXL3 pack runs its projections on the wide kernel unless the operator names
            // another in FIFI_EXL3_SINGLE_GATHER (`off` is the shuffle kernel); bit-identical output.
            // The shared applier runs the launch-switch check the step build also runs, so a bad
            // switch combination refuses the open before the weight load instead of at step build.
            projector.context().register_served_kernels(ServedKernels {
                single_row: SERVED_SINGLE_ROW,
                indexed: None,
                odd_rates: false,
            })?;
            let mut banked_inner: Option<(ExpertStore, usize, usize)> = None;
            let device = match load_exl3_dir_decided(
                &projector,
                &settings.artifact,
                settings.expert_bank.as_ref(),
                reserve,
            )? {
                DecidedLoad::Resident(device) => device,
                DecidedLoad::Banked(banked) => {
                    let sticky = settings.expert_bank.as_ref().map_or(0, |bank| bank.sticky);
                    eprintln!(
                        "q36 cuda: EXL3 expert bank: {} of {} experts per layer resident, \
                         sticky {} (device {:.2} GiB incl. dense {:.2} GiB), host store {:.2} GiB, \
                         loaded from {} in {:.1} s",
                        banked.slots,
                        banked.experts,
                        sticky.min(banked.slots.saturating_sub(1)),
                        banked.weights.device_bytes() as f64 / (1 << 30) as f64,
                        banked.dense_bytes as f64 / (1 << 30) as f64,
                        banked.store.host_bytes() as f64 / (1 << 30) as f64,
                        settings.artifact.display(),
                        began.elapsed().as_secs_f64(),
                    );
                    let BankedModel {
                        weights,
                        store,
                        slots,
                        experts,
                        dense_bytes: _,
                    } = *banked;
                    banked_inner = Some((store, slots, experts));
                    weights
                }
            };
            banked_load = banked_inner;
            eprintln!(
                "q36 cuda: EXL3 weights loaded from {} in {:.1} s",
                settings.artifact.display(),
                began.elapsed().as_secs_f64()
            );
            device
        }
        (Some((model, _)), None) => {
            match projector.load_model(model, load_options(kv_cache, fixed_admission)) {
                Ok(loaded) => loaded.device,
                // The card shrank between the plan and the load (a desktop process grew): plan
                // once more, from a fresh read, one step below the refused context.
                Err(refusal @ EngError::MemoryPlan(_)) => {
                    let Some((variant, variant_plan)) =
                        replan_smaller(&projector, &settings, &plan, model)?
                    else {
                        return Err(refusal);
                    };
                    settings = variant;
                    plan = variant_plan;
                    kv_cache = plan.reserve;
                    fixed_admission = plan.modes.fixed_admission;
                    projector
                        .load_model(model, load_options(kv_cache, fixed_admission))?
                        .device
                }
                Err(other) => return Err(other),
            }
        }
        (Some((model, _)), Some(_)) => {
            let names = model.plan().device_names();
            let names = names_without_experts(&names);
            projector.load(
                &model.weights,
                &names,
                load_options(kv_cache, fixed_admission),
            )?
        }
    };
    let after = projector.memory_info()?;
    stages.begin("admitting memory");
    let floor = min_free_after_load()?;
    if after.free < floor {
        return Err(EngError::MemoryPlan(format!(
            "cuda admission: {} bytes free after the load is below the {}-byte floor",
            after.free, floor
        )));
    }
    let plan = plan.after_load(&settings, after.free)?;
    let StartupPlan {
        reserve: kv_cache,
        paged,
        device_first_native,
        verify: _,
        allow_unpinned: _,
        modes,
    } = plan;
    let device_first = device_first_native.is_some();
    if device_first && paged.is_none() {
        require_device_first_room(after.free, kv_cache)?;
    }
    let inventory = Inventory::for_spec(&QWEN3_6_35B_A3B)?;
    let shape = HostShape::from_spec(&QWEN3_6_35B_A3B, ROTARY_DIM)?;
    let dims = StepDims::new(&shape, &inventory)?;
    // The step layout is the loaded weights' own, never a constant or the load options: the
    // step's matvecs read the blocks the device holds (a mixed load is refused by layout()).
    // The KV mode is derived from the SAME parsed form against that layout (`kv_mode_of`),
    // so the step's mode always agrees with the admission's.
    let layout = device.layout()?;
    let kv_mode = kv_mode_of(settings.kv, layout)?;
    stages.begin("building the step");
    let mut options = step_options(settings.context_limit, layout, kv_mode)?;
    options.tiered = tiered;
    let res = match paged {
        Some(plan) => StepResources::allocate_paged(projector.context(), plan)?,
        None => {
            let res = StepResources::new(projector.context(), dims, &inventory, &options)?;
            if settings.draft_verify == VerifyMode::Krow {
                res.with_krow(projector.context(), &options)?
            } else {
                res
            }
        }
    };
    // LEAK: the three owners are leaked for the process lifetime, giving
    // `QwenResidentStep<'static>`; an owning `Arc` variant would rewrite the qualified step
    // (>= 1,000 lines). `guard.commit()` right below pins the one-engine flag, so a retry after
    // ANY failure from here on — a `QwenResidentStep::new` or `set_capture_config` error leaves
    // the ~20 GB on the device — is refused with the documented text. A process whose engine
    // ended in Error or NativeFailed is restarted, not rebuilt.
    let projector: &'static CudaProjector = Box::leak(Box::new(projector));
    let weights: &'static CudaWeights = Box::leak(Box::new(device));
    let res: &'static StepResources = Box::leak(Box::new(res));
    // The banked resolver, from the leaked weights: per-layer slot tables over the load's slot
    // banks and the host store. Leaked like its owners, for the step's life.
    let mut banked_slots = 0;
    let bank = match banked_load {
        Some((store, slots, experts)) => {
            banked_slots = slots;
            let dims = moe_dims(&res.dims);
            let blocks = (0..res.table.len())
                .map(|layer| {
                    Exl3MoeWeights::resolve_banked(weights, &res.table, layer, &dims, slots)
                })
                .collect::<Result<Vec<_>, _>>()?;
            let resolver = Exl3SlotResolver::new(
                store,
                &blocks,
                experts,
                settings.expert_bank.as_ref().map_or(0, |bank| bank.sticky),
            )?;
            // A bad value stops the start, not the first token (the other flags' rule);
            // the arming happens before the leak so no `RefCell` borrow crosses it.
            let overlap = bank_overlap_wanted(std::env::var_os(BANK_OVERLAP_VAR).as_deref())?;
            let mut resolver = resolver;
            if overlap {
                resolver.enable_overlap(projector.context())?;
            }
            Some(Box::leak(Box::new(std::cell::RefCell::new(resolver))))
        }
        None => None,
    };
    guard.commit();
    let mut step = match bank {
        Some(bank) => QwenResidentStep::new_banked(
            projector.context(),
            res,
            weights,
            &options,
            rules,
            banked_slots,
            bank,
        )?,
        None => QwenResidentStep::new(projector.context(), res, weights, &options, rules)?,
    };
    if let Some((model, path)) = gguf.as_ref() {
        step.set_prefix_identity(gguf_prefix_identity(
            &model.weights,
            path,
            settings.gguf_sha256.as_deref(),
        )?);
    }
    step.set_capture_config(CaptureConfig {
        capture: settings.capture,
        // The section-timing marks stay off on the serve path: the timer is a measurement harness,
        // not a serving feature.
        section_marks: false,
    })?;
    step.set_speculation(&settings.speculation)?;
    // Q8 chunking defaults on at 1024 rows; overrides use the supported row ladder.
    // Other sessions keep DEFAULT_CUDA_PREFILL_CHUNK; MLX `prefill_chunk` is ignored.
    let kv8_chunked = modes.chunked && kv_mode == KvMode::Int8G32;
    step.set_kv8_chunked_prefill(kv8_chunked)?;
    step.set_m_sess_cap(if kv8_chunked {
        modes.chunk_rows
    } else {
        DEFAULT_CUDA_PREFILL_CHUNK
    });

    // The prefill arena: ONE admission after the load and the step build, from the post-load
    // snapshot less the selected headroom. Fixed mode also checks live free VRAM and refuses an
    // undersized plan; legacy mode may keep the per-token arm.
    if let Some(host_limit) = settings.kv_offload_host_bytes {
        if settings.draft_rollback == RollbackMode::Device {
            return Err(EngError::BadRequest(
                "engine config: draft_rollback device needs the resident KV plane; kv_offload_host_bytes \
                 is set"
                    .into(),
            ));
        }
        if device_first {
            if let Some(rows) = res
                .paged
                .as_ref()
                .and_then(|paged| paged.state.cache_rows())
            {
                eprintln!(
                    "q36 cuda: device-first KV, cached rows {rows}, host overflow limit {host_limit} B, physical state/arena/attention reserve {kv_cache} B"
                );
            } else {
                eprintln!(
                    "q36 cuda: device-first KV, fully resident; native state/arena reserve {kv_cache} B, host payloads absent"
                );
            }
        } else {
            eprintln!(
                "q36 cuda: host-paged KV, host limit {host_limit} B, physical state/arena/attention reserve {kv_cache} B; prefix reuse, capture and chunked prefill disabled"
            );
        }
    } else {
        let mut extras = Vec::new();
        if modes.prefix != PrefixMemory::Disabled {
            step.enable_pinned_prefix()?;
            enable_kv_disk(&mut step, &settings)?;
        }
        let mut tip_bytes = 0;
        if modes.prefix == PrefixMemory::DeviceTip && step.pinned_prefix_enabled() {
            tip_bytes = step.enable_device_tip()?;
            extras.push(Extra {
                name: "device-prefix-delta",
                fixed: tip_bytes,
                bytes_per_token: 0,
            });
        }
        // The rollback slots: charged as an extra when the prefill arena is admitted; otherwise
        // (FP16 KV with chunk prefill off) nothing admits them and the load reserve that no arena
        // consumes covers them, which `enable_device_rollback` enforces before it allocates. A
        // refusal here refuses the start: there is no fallback to the host rollback.
        let arena = arena_wanted(kv_mode, modes.chunk_off);
        let mut rollback_cover = None;
        if settings.draft_rollback == RollbackMode::Device {
            let cover = if arena {
                Cover::Arena
            } else {
                Cover::Reserve {
                    headroom: prefill_headroom(modes.fixed_admission),
                }
            };
            step.request_device_rollback();
            let bytes =
                step.enable_device_rollback(settings.speculation.max_rows, cover, tip_bytes)?;
            if arena {
                extras.push(Extra {
                    name: "device-rollback-slots",
                    fixed: bytes,
                    bytes_per_token: 0,
                });
            }
            rollback_cover = Some(cover);
        }
        // `draft_verify = krow` needs the armed slots above: ask for it, check the step's own
        // fields against the one qualified configuration, then arm the K-row path.
        if settings.draft_verify == VerifyMode::Krow {
            step.request_krow_verify();
            step.check_krow_arming(settings.speculation.max_rows)?;
            step.enable_krow()?;
        }
        if let (Some(_), Some((model, _))) = (&tiered, gguf.as_ref()) {
            step.set_tiered_chunk(tiered_chunk_arm(model.weights.header(), &res.dims)?);
        }
        let mut arena_admitted = None;
        if arena {
            let device_free = after
                .free
                .saturating_sub(prefill_headroom(modes.fixed_admission));
            let verdict = step.admit_prefill(
                PrefillBudget {
                    device_free,
                    extras,
                },
                modes.context_staging,
                modes.fixed_admission,
                modes.wide_stage,
            )?;
            arena_admitted = Some(matches!(verdict, ChunkVerdict::Ready { .. }));
            let line = if kv_mode == KvMode::Int8G32 {
                kv8_prefill_start_line(&verdict)
            } else {
                prefill_start_line(&verdict)
            };
            eprintln!("{line}");
        } else {
            eprintln!("q36 cuda: prefill chunk off (Q36_CUDA_CHUNK_PREFILL=0): one token per step");
        }
        if let (Some(cover), Some(plan)) = (rollback_cover, step.device_rollback_plan()) {
            eprintln!("{}", rollback_startup_line(&plan, cover, arena_admitted)?);
        }
    }
    if let Some(plan) = tiered {
        // The banks come last: they take what the weights, the KV cache and the step state left.
        let (model, gguf) = gguf.as_ref().ok_or_else(|| {
            EngError::BadRequest("engine config: tiered experts read the GGUF file".into())
        })?;
        let slots = attach_tiered_parts(
            &mut step,
            projector.context(),
            ((model.weights.header(), gguf), &res.dims),
            &plan,
            // The sessions the planner funded beside the banks; a config that named its tiered
            // member sized its banks itself and keeps them whole.
            card_short.as_ref().map_or(1, |short| short.sessions),
        )?;
        if settings.prefix_reuse {
            step.configure_tiered_prefix_store()?;
        }
        eprintln!(
            "q36 cuda: tiered, {} slots per layer, {} cpu threads, timeout {} ms",
            slots, plan.threads, plan.timeout_ms
        );
        if let Some(short) = &card_short {
            step.set_sessions_funded(short.sessions);
            eprintln!("{}", short.line(slots, options.context_tokens));
            if settings.prefix_reuse {
                eprintln!(
                    "q36 cuda: prefix reuse follows the tiered engine's prefix store, which is off \
                     unless Q36_TIERED_PREFIX=1"
                );
            }
        }
    }
    eprintln!(
        "q36 cuda: {} weights resident ({} B), free after load {} B, context_limit {}, kv {}, \
         capture {}{}",
        weight_format_name(settings.weights),
        before.free.saturating_sub(after.free),
        after.free,
        options.context_tokens,
        match options.kv_mode {
            KvMode::Fp16 => "fp16",
            KvMode::Int8G32 => "int8g32",
        },
        if settings.capture { "on" } else { "off" },
        projector.context().single_gather_suffix() + &step.shared_in_routed_suffix()
    );
    Ok((step, DecodeSession::default()))
}

/// Why this start cannot run the on-disk prefix tier whatever the config says, when it cannot. A
/// start that keeps only some experts on the card (the explicit `tiered` option, or the lower rung
/// a card too small for the checkpoint takes) saves no prefixes by default, so a cache directory
/// would only hold entries nothing reads.
fn kv_disk_blocker(settings: &Q36Settings) -> Option<&'static str> {
    settings
        .tiered
        .is_some()
        .then_some("a tiered-experts start keeps no prefix entries to spill")
}

/// Starts the on-disk prefix tier when the config, the environment or the build default asks for
/// it, and says in one line whether it runs. A malformed override stops the start; a directory
/// that cannot be used leaves the tier off.
fn enable_kv_disk(step: &mut QwenResidentStep<'_>, settings: &Q36Settings) -> Result<(), EngError> {
    use crate::family::kv_disk as kv_cache;
    if let Some(reason) = kv_disk_blocker(settings) {
        eprintln!("q36 cuda: kv disk tier off: {reason}");
        return Ok(());
    }
    if let Some((resolved, origin)) =
        kv_cache::plan_tier(settings.kv_cache.as_ref(), "qwen36", "q36 cuda")?
    {
        step.enable_disk_tier(&resolved, origin);
    }
    Ok(())
}

#[cfg(test)]
mod after_load_floor_tests {
    use super::*;

    /// The measured start: Q4_K_M weights on a card with 21.6 GB free planned a 9,216-cell int8
    /// cache and the load then refused with 909,967,360 bytes free against the 1 GiB floor. The
    /// fit now sees that the weights alone leave less than the floor, at that level and one step
    /// below, and answers the lower rung before any upload.
    #[test]
    fn the_weights_alone_below_the_floor_leave_no_context() {
        let weights = 20_408_576_512u64;
        let floor = crate::family::mem::DEVICE.load_unplanned + (1 << 30);
        assert!(!weights_leave_the_floor(21_600_468_992, weights, floor));
        assert!(!weights_leave_the_floor(21_300_000_000, weights, floor));
        assert!(weights_leave_the_floor(weights + floor, weights, floor));
        assert!(!weights_leave_the_floor(
            weights + floor - 1,
            weights,
            floor
        ));
    }
}

#[cfg(test)]
mod chunk_prefill_flag_tests {
    use super::{
        ChunkVerdict, KvDtype, PrefixMemory, chunk_prefill_rows, chunk_prefill_wanted,
        context_staging_wanted, kv8_prefill_start_line,
    };
    use std::ffi::OsStr;

    #[test]
    fn p2p_kv_charge_matches_the_resident_state_at_128k() {
        use crate::family::decode_step::state_layout::StateLayout;

        let inventory = super::Inventory::for_spec(&super::QWEN3_6_35B_A3B).unwrap();
        let shape =
            super::HostShape::from_spec(&super::QWEN3_6_35B_A3B, super::ROTARY_DIM).unwrap();
        let dims = super::StepDims::new(&shape, &inventory).unwrap();
        for mode in [super::KvMode::Fp16, super::KvMode::Int8G32] {
            let options = super::step_options(131072, super::WeightLayout::Native, mode).unwrap();
            let state = StateLayout::new(&dims, &options).unwrap();
            assert_eq!(
                state.kv.total as u64,
                super::kv_cache_bytes(131072, mode).unwrap()
            );
        }
    }

    #[test]
    fn p2p_expert_budgets_default_off_and_validate_each_card() {
        let parse = super::p2p_host_expert_bytes;
        assert_eq!(parse("card0", None).unwrap(), 0);
        assert_eq!(parse("card1", None).unwrap(), 0);
        assert_eq!(
            parse("card0", Some(OsStr::new("2147483648"))).unwrap(),
            2 << 30
        );
        assert_eq!(
            parse("card1", Some(OsStr::new("4294967296"))).unwrap(),
            4 << 30
        );
        for invalid in ["", "-1", "two", "18446744073709551616"] {
            let error = parse("card0", Some(OsStr::new(invalid))).unwrap_err();
            assert!(format!("{error:?}").contains("card0"));
        }
        #[cfg(unix)]
        {
            use std::os::unix::ffi::OsStrExt;
            assert!(parse("card0", Some(OsStr::from_bytes(&[0xff]))).is_err());
        }
    }

    #[test]
    fn wide_stage_defaults_on_and_zero_rolls_it_back() {
        use super::wide_stage_wanted;
        // The flip: unset means ON; =0 is the documented rollback; the parse
        // rules match every prefill flag (only 0 or 1, else a typed refusal).
        assert!(wide_stage_wanted(None).unwrap());
        assert!(!wide_stage_wanted(Some(OsStr::new("0"))).unwrap());
        assert!(wide_stage_wanted(Some(OsStr::new("1"))).unwrap());
        assert!(wide_stage_wanted(Some(OsStr::new("x"))).is_err());
    }

    #[test]
    fn wide_stage_inert_reason_names_every_silent_mode() {
        use super::wide_stage_inert_reason as reason;
        // Fixed admission holds: the flag is live, no reason.
        assert_eq!(reason(true, KvDtype::Q8, true, true, false, false), None);
        // Each mode that drops fixed admission names itself, most specific first.
        assert_eq!(
            reason(false, KvDtype::Q8, true, true, false, true),
            Some("host KV offload")
        );
        assert_eq!(
            reason(false, KvDtype::Q8, true, true, true, false),
            Some("tiered mode")
        );
        assert_eq!(
            reason(false, KvDtype::F32, true, true, false, false),
            Some("non-Q8 KV dtype")
        );
        assert_eq!(
            reason(false, KvDtype::Q8, false, true, false, false),
            Some("chunked prefill off (Q36_CUDA_CHUNK_PREFILL=0)")
        );
        assert_eq!(
            reason(false, KvDtype::Q8, true, false, false, false),
            Some("context staging off (Q36_CUDA_KV8_STAGE_CONTEXT=0)")
        );
        assert_eq!(
            reason(false, KvDtype::Q8, true, true, false, false),
            Some("fixed admission off (Q36_CUDA_KV8_FIXED_ADMISSION=0)")
        );
    }

    #[test]
    fn prefix_memory_never_enables_prefix_reuse() {
        for pinned in [None, Some(OsStr::new("0")), Some(OsStr::new("1"))] {
            for device in [None, Some(OsStr::new("0")), Some(OsStr::new("1"))] {
                assert!(
                    PrefixMemory::from_flags(false, pinned, device, true) == PrefixMemory::Disabled
                );
            }
        }
    }

    #[test]
    fn prefix_memory_defaults_to_device_tip_when_reuse_is_on() {
        assert!(PrefixMemory::from_flags(true, None, None, true) == PrefixMemory::DeviceTip);
        assert!(
            PrefixMemory::from_flags(true, Some(OsStr::new("1")), None, true)
                == PrefixMemory::DeviceTip
        );
    }

    #[test]
    fn pinned_zero_restores_pageable_even_when_device_is_requested() {
        for device in [None, Some(OsStr::new("0")), Some(OsStr::new("1"))] {
            assert!(
                PrefixMemory::from_flags(true, Some(OsStr::new("0")), device, true)
                    == PrefixMemory::Disabled
            );
        }
    }

    #[test]
    fn state_only_zero_keeps_the_pinned_host_path() {
        assert!(
            PrefixMemory::from_flags(true, None, Some(OsStr::new("0")), true)
                == PrefixMemory::Pinned
        );
        assert!(
            PrefixMemory::from_flags(true, Some(OsStr::new("1")), Some(OsStr::new("0")), true)
                == PrefixMemory::Pinned
        );
    }

    #[test]
    fn invalid_prefix_flag_values_do_not_enable_the_memory_mode() {
        for value in ["", "true", "yes", "2", " 1"] {
            assert!(
                PrefixMemory::from_flags(true, Some(OsStr::new(value)), None, true)
                    == PrefixMemory::Disabled
            );
            assert!(
                PrefixMemory::from_flags(
                    true,
                    Some(OsStr::new("1")),
                    Some(OsStr::new(value)),
                    true
                ) == PrefixMemory::Pinned
            );
        }
    }

    #[test]
    fn unsupported_tiered_backend_ignores_implicit_prefix_memory() {
        assert!(PrefixMemory::from_flags(true, None, None, false) == PrefixMemory::Disabled);
        assert!(
            PrefixMemory::from_flags(true, None, Some(OsStr::new("0")), false)
                == PrefixMemory::Disabled
        );
    }

    #[test]
    fn explicit_unsupported_requests_still_reach_mode_validation() {
        assert!(
            PrefixMemory::from_flags(true, Some(OsStr::new("1")), None, false)
                == PrefixMemory::Pinned
        );
        assert!(
            PrefixMemory::from_flags(true, None, Some(OsStr::new("1")), false)
                == PrefixMemory::DeviceTip
        );
        assert!(
            PrefixMemory::from_flags(true, Some(OsStr::new("0")), Some(OsStr::new("1")), false)
                == PrefixMemory::Disabled
        );
    }

    #[cfg(unix)]
    #[test]
    fn invalid_unicode_cannot_enable_prefix_memory() {
        use std::os::unix::ffi::OsStrExt;
        let invalid = OsStr::from_bytes(&[0xff]);
        assert!(
            PrefixMemory::from_flags(true, Some(invalid), None, true) == PrefixMemory::Disabled
        );
        assert!(PrefixMemory::from_flags(true, None, Some(invalid), true) == PrefixMemory::Pinned);
    }

    #[test]
    fn context_staging_defaults_on_and_zero_restores_legacy() {
        assert!(context_staging_wanted(None).unwrap());
        assert!(!context_staging_wanted(Some(OsStr::new("0"))).unwrap());
        assert!(context_staging_wanted(Some(OsStr::new("1"))).unwrap());
    }

    #[test]
    fn fixed_admission_defaults_on_and_all_kill_switches_win() {
        assert!(
            !super::fixed_admission_wanted(KvDtype::Q8, true, true, Some(OsStr::new("0"))).unwrap()
        );
        for (kv, chunked, context, wanted) in [
            (KvDtype::Q8, true, true, true),
            (KvDtype::Q8, false, true, false),
            (KvDtype::Q8, true, false, false),
            (KvDtype::F32, true, true, false),
        ] {
            for flag in [None, Some(OsStr::new("1"))] {
                assert_eq!(
                    super::fixed_admission_wanted(kv, chunked, context, flag).unwrap(),
                    wanted
                );
            }
        }
        assert!(
            super::fixed_admission_wanted(KvDtype::Q8, false, false, Some(OsStr::new("invalid")),)
                .is_err()
        );
        assert_eq!(super::prefill_headroom(false), 1usize << 30);
        assert_eq!(super::prefill_headroom(true), 256usize << 20);
        assert_eq!(
            super::super::pinned::min_free_after_load_default(),
            1usize << 30
        );
    }

    #[test]
    fn q8_chunking_defaults_on_and_accepts_only_zero_or_one() {
        assert!(chunk_prefill_wanted(KvDtype::Q8, None).unwrap());
        assert!(!chunk_prefill_wanted(KvDtype::Q8, Some(OsStr::new("0"))).unwrap());
        assert!(chunk_prefill_wanted(KvDtype::Q8, Some(OsStr::new("1"))).unwrap());
        for kv in [KvDtype::F32, KvDtype::Q8] {
            for invalid in ["", "true", "yes", "2", " 1"] {
                assert!(chunk_prefill_wanted(kv, Some(OsStr::new(invalid))).is_err());
            }
        }
    }

    #[test]
    fn the_rollback_flag_reaches_the_arena_decision_at_fp16() {
        use super::{arena_wanted, chunk_rollback_requested};
        use crate::family::qwen::qwen36::cuda::dims::KvMode;
        assert!(!chunk_rollback_requested(None));
        assert!(!chunk_rollback_requested(Some(OsStr::new("1"))));
        assert!(chunk_rollback_requested(Some(OsStr::new("0"))));
        // FP16 chunks through the arena unless the flag is 0; Int8G32 consults the flag itself.
        assert!(arena_wanted(KvMode::Fp16, chunk_rollback_requested(None)));
        assert!(arena_wanted(
            KvMode::Fp16,
            chunk_rollback_requested(Some(OsStr::new("1")))
        ));
        assert!(!arena_wanted(
            KvMode::Fp16,
            chunk_rollback_requested(Some(OsStr::new("0")))
        ));
        assert!(arena_wanted(
            KvMode::Int8G32,
            chunk_rollback_requested(Some(OsStr::new("0")))
        ));
    }

    #[test]
    fn fp16_default_is_unchanged() {
        assert!(!chunk_prefill_wanted(KvDtype::F32, None).unwrap());
        assert!(!chunk_prefill_wanted(KvDtype::F32, Some(OsStr::new("0"))).unwrap());
        assert!(chunk_prefill_wanted(KvDtype::F32, Some(OsStr::new("1"))).unwrap());
    }

    #[test]
    fn kv8_start_line_names_actual_mode_and_kill_switch() {
        let on = kv8_prefill_start_line(&ChunkVerdict::Ready { m_sess: 1024 });
        assert!(on.contains("prefill chunk M_sess 1024 | KV8 mode=chunked"));
        assert!(on.contains("Q36_CUDA_CHUNK_PREFILL=0 disables chunking"));
        for verdict in [
            ChunkVerdict::Kv8,
            ChunkVerdict::NoArena,
            ChunkVerdict::Tiered,
        ] {
            let off = kv8_prefill_start_line(&verdict);
            assert!(off.contains("KV8 mode=per-token"));
            assert!(off.contains("Q36_CUDA_CHUNK_PREFILL=0 disables chunking"));
        }
    }

    #[test]
    fn requested_bulk_sizes_are_explicit_and_default_to_1024() {
        assert_eq!(chunk_prefill_rows(None).unwrap(), 1024);
        for rows in ["16", "128", "512", "1024", "2048"] {
            assert_eq!(
                chunk_prefill_rows(Some(OsStr::new(rows))).unwrap(),
                rows.parse::<u32>().unwrap()
            );
        }
        for bad in ["", "0", "1", "64", "1025", "4096", " 512"] {
            assert!(chunk_prefill_rows(Some(OsStr::new(bad))).is_err());
        }
    }

    #[cfg(unix)]
    #[test]
    fn invalid_unicode_cannot_silently_enable_chunking() {
        use std::os::unix::ffi::OsStrExt;
        for kv in [KvDtype::F32, KvDtype::Q8] {
            assert!(chunk_prefill_wanted(kv, Some(OsStr::from_bytes(&[0xff]))).is_err());
        }
    }

    include!("prefill_ttft_tests.rs");
}

#[cfg(test)]
mod paged_start_tests {
    use super::*;
    use crate::family::qwen::qwen36::settings::{Backend, Q36Env, Q36Options};

    /// The floor the admission reads in this process (the default, or `Q36_CUDA_MIN_FREE_MIB`),
    /// so the arithmetic tests hold under either.
    fn floor() -> usize {
        super::super::pinned::min_free_after_load().unwrap()
    }

    fn settings(body: &str) -> Q36Settings {
        let options: Q36Options = serde_json::from_str(body).unwrap();
        Q36Settings::new(
            Path::new("/not-opened"),
            Path::new("/not-opened"),
            Some(&options),
            &Q36Env::default(),
            Backend::Cuda,
        )
        .unwrap()
    }

    fn modes(chunked: bool) -> RuntimeModes {
        RuntimeModes {
            chunked,
            chunk_off: false,
            chunk_rows: 512,
            context_staging: false,
            fixed_admission: false,
            wide_stage: false,
            prefix: PrefixMemory::Disabled,
        }
    }

    #[test]
    fn a_tiered_start_leaves_the_disk_tier_off() {
        let tiered = settings(r#"{"capture":false,"tiered":{"slots":40}}"#);
        assert!(kv_disk_blocker(&tiered).is_some());
        assert!(kv_disk_blocker(&settings("{}")).is_none());
        let tiered_with_cache =
            settings(r#"{"capture":false,"tiered":{"slots":"auto"},"kv_cache":{"enabled":true}}"#);
        assert!(kv_disk_blocker(&tiered_with_cache).is_some());
    }

    #[test]
    fn absent_offload_preserves_the_resident_loader_reserve() {
        let settings = settings("{}");
        let plan = StartupPlan::new(&settings, false, false, modes(false)).unwrap();
        assert!(plan.paged.is_none());
        assert_eq!(
            plan.reserve,
            kv_cache_bytes(
                settings.context_limit as usize,
                serve_kv_mode(&settings).unwrap()
            )
            .unwrap(),
        );
    }

    #[test]
    fn paged_startup_retains_the_exact_physical_resource_plan() {
        let settings = settings(
            r#"{"kv_dtype":"q8","context_limit":131072,"kv_offload_host_bytes":2000000000,"capture":false}"#,
        );
        let plan = StartupPlan::new(&settings, false, false, modes(false)).unwrap();
        assert_eq!(
            plan.reserve,
            u64::try_from(plan.paged.as_ref().unwrap().device_bytes().unwrap()).unwrap()
        );
        assert!(plan.reserve < kv_cache_bytes(131072, KvMode::Int8G32).unwrap());
    }

    #[test]
    fn device_first_start_uses_the_complete_resident_reserve_without_host_payloads() {
        let configured = settings(
            r#"{"kv_dtype":"q8","context_limit":33280,"kv_offload_host_bytes":1,"capture":false}"#,
        );
        let plan = StartupPlan::new_device_first(&configured, false, false, modes(false)).unwrap();
        assert!(plan.paged.is_none());
        let inventory = Inventory::for_spec(&QWEN3_6_35B_A3B).unwrap();
        let shape = HostShape::from_spec(&QWEN3_6_35B_A3B, ROTARY_DIM).unwrap();
        let dims = StepDims::new(&shape, &inventory).unwrap();
        let options = step_options(33280, WeightLayout::Native, KvMode::Int8G32).unwrap();
        let state =
            crate::family::decode_step::state_layout::StateLayout::new(&dims, &options).unwrap();
        let scratch =
            super::super::step_plan::KernelScratch::for_dims(&dims, 1, KvMode::Int8G32).unwrap();
        let step = super::super::step_plan::StepPlan::new(&dims, 1, &options, &scratch).unwrap();
        let native = state.bytes().unwrap() + step.arena_bytes();
        assert!(
            plan.reserve < native as u64,
            "pre-load reserve must exclude the full KV pool"
        );
        let plan = plan.after_load(&configured, floor() + native).unwrap();
        assert!(plan.paged.is_none());
        assert_eq!(plan.reserve, native as u64);

        let absent = settings("{}");
        let plan = StartupPlan::new_device_first(&absent, false, false, modes(false)).unwrap();
        assert_eq!(
            plan.reserve,
            StartupPlan::new(&absent, false, false, modes(false))
                .unwrap()
                .reserve
        );
    }

    #[test]
    fn tiny_native_fit_is_not_rejected_by_rounded_page_reservation() {
        let configured = settings(
            r#"{"kv_dtype":"q8","context_limit":2,"kv_offload_host_bytes":1,"capture":false}"#,
        );
        let plan = StartupPlan::new_device_first(&configured, false, false, modes(false)).unwrap();
        let native = plan.device_first_native.unwrap();
        assert!(plan.reserve as usize <= native);
        let plan = plan.after_load(&configured, floor() + native).unwrap();
        assert!(plan.paged.is_none());
        assert_eq!(plan.reserve as usize, native);
    }

    #[test]
    fn device_first_selects_the_measured_cache_budget_after_loading_weights() {
        let configured = settings(
            r#"{"kv_dtype":"q8","context_limit":1280,"kv_offload_host_bytes":33554432,"capture":false}"#,
        );
        let dims = super::super::common_tests::dims();
        let paging = super::super::paging_plan::PagingPlan::new(&dims, 1280).unwrap();
        let group = paging.group.total * paging.full_layers;
        let plan = StartupPlan::new_device_first(&configured, false, false, modes(false)).unwrap();
        let fixed = plan.reserve as usize;
        let plan = plan
            .after_load(&configured, floor() + fixed + 2 * group)
            .unwrap();
        assert_eq!(
            plan.paged.as_ref().unwrap().cache_rows(),
            Some(2 * paging.group_cells)
        );
        assert_eq!(plan.reserve as usize, fixed + 2 * group);

        let plan = StartupPlan::new_device_first(&configured, false, false, modes(false)).unwrap();
        assert!(
            plan.after_load(&configured, floor() + fixed + group - 1)
                .is_err()
        );
        let mut insufficient_host = configured;
        insufficient_host.kv_offload_host_bytes = Some(1);
        let plan =
            StartupPlan::new_device_first(&insufficient_host, false, false, modes(false)).unwrap();
        assert!(
            plan.after_load(&insufficient_host, floor() + fixed + 2 * group)
                .is_err()
        );
    }

    #[test]
    fn device_first_flag_is_default_off_and_strict() {
        use std::ffi::OsStr;
        assert!(!device_first_wanted(None).unwrap());
        assert!(!device_first_wanted(Some(OsStr::new("0"))).unwrap());
        assert!(device_first_wanted(Some(OsStr::new("1"))).unwrap());
        for value in ["", "true", "yes", "2", " 1"] {
            assert!(device_first_wanted(Some(OsStr::new(value))).is_err());
        }
    }

    #[test]
    fn device_first_resident_admission_retains_the_complete_headroom() {
        assert!(require_device_first_room(floor() + 256, 256).is_ok());
        assert!(require_device_first_room(floor() + 255, 256).is_err());
        assert!(require_device_first_room(usize::MAX, u64::MAX).is_err());
    }

    #[test]
    fn device_first_retains_the_capacity_mode_refusals() {
        let base = settings(
            r#"{"kv_dtype":"q8","context_limit":320,"kv_offload_host_bytes":1,"capture":false}"#,
        );
        assert!(StartupPlan::new_device_first(&base, false, false, modes(true)).is_err());
        let mut bad = base.clone();
        bad.capture = true;
        assert!(StartupPlan::new_device_first(&bad, false, false, modes(false)).is_err());
        let mut bad = base.clone();
        bad.prefix_reuse = true;
        assert!(StartupPlan::new_device_first(&bad, false, false, modes(false)).is_err());
        let mut bad = base.clone();
        bad.kv_offload_host_bytes = Some(0);
        assert!(StartupPlan::new_device_first(&bad, false, false, modes(false)).is_err());
        let mut bad = base;
        bad.kv = KvDtype::F32;
        assert!(StartupPlan::new_device_first(&bad, false, false, modes(false)).is_err());
    }

    #[test]
    fn unsupported_paged_starts_fail_without_a_device_or_checkpoint() {
        let settings = settings(
            r#"{"kv_dtype":"q8","context_limit":320,"kv_offload_host_bytes":8388608,"capture":false}"#,
        );
        assert!(StartupPlan::new(&settings, false, false, modes(true)).is_err());
        let mut bad = settings.clone();
        bad.capture = true;
        assert!(StartupPlan::new(&bad, false, false, modes(false)).is_err());
        let mut bad = settings.clone();
        bad.prefix_reuse = true;
        assert!(StartupPlan::new(&bad, false, false, modes(false)).is_err());
        let mut bad = settings;
        bad.kv_offload_host_bytes = Some(1);
        assert!(StartupPlan::new(&bad, false, false, modes(false)).is_err());
    }
}

#[cfg(test)]
mod fused_inproj_tests {
    use super::check_fused_inproj;

    #[test]
    fn multi_device_serving_refuses_the_fused_input_projection() {
        check_fused_inproj(1, true).expect("one card keeps the fusion");
        check_fused_inproj(1, false).expect("one card without the fusion");
        check_fused_inproj(2, false).expect("multi-card without the fusion");
        let error = check_fused_inproj(2, true).expect_err("multi-card plus the fusion");
        assert!(
            format!("{error:?}").contains("GDN_FUSE_INPROJ"),
            "{error:?}"
        );
    }
}

#[cfg(test)]
mod check_choice_tests {
    use super::*;
    use crate::family::qwen::qwen36::settings::{Backend, Q36Env, Q36Options};

    /// The figures a 24 GiB card with the Q4_K_M checkpoint reads at start.
    const FREE: u64 = 24_188_420_096;
    const WEIGHTS: u64 = 20_408_576_512;
    const NSM: usize = 82;

    #[test]
    fn check_chooses_the_cache_form_and_context_the_start_chooses() {
        let settings = Q36Settings::new(
            Path::new("/m"),
            Path::new("/t"),
            Some(&Q36Options::default()),
            &Q36Env::default(),
            Backend::Cuda,
        )
        .expect("the CUDA settings");
        assert!(
            settings.kv_auto && settings.context_auto,
            "a config that names neither leaves both to the card"
        );
        let plan = startup_plan_of(&settings, false, false).expect("startup plan");
        let reading = DeviceReading {
            free: FREE,
            nsm: NSM,
        };
        let served = card_choice(reading, &settings, &plan, WEIGHTS)
            .expect("the card choice")
            .expect("a fit");
        let checked = planned_fit_from(reading, &settings, &plan, WEIGHTS)
            .expect("the check's choice")
            .expect("a fit");
        assert_eq!(checked, (served.variant.kv, served.variant.context_limit));
        assert_eq!(checked, (KvDtype::Q8, 242_688));
    }
}
