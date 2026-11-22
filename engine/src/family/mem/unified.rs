//! Unified-memory planning shared by the MLX families (the `Facts` of a Mac): the share of unified memory a start may
//! use, the bytes of a checkpoint's routed experts read from its shard headers, and whether a
//! model, a session reserve and a saved prefix fit a budget. Pure functions, compiled on every
//! host so the settings of every family and the tests read the same numbers.

/// The bytes of a checkpoint's routed expert tensors, from the shard headers: all layers
/// together, and the largest layer, which is what one decode or prefill stage reads when the
/// experts stream from disk.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ExpertBytes {
    pub(crate) total_bytes: u64,
    pub(crate) stage_bytes: u64,
}

/// The resident bytes of a model whose routed experts stream: everything but the expert
/// tensors, plus one stage of experts.
pub(crate) fn staged_model_bytes(model_bytes: u64, experts: ExpertBytes) -> u64 {
    model_bytes
        .saturating_sub(experts.total_bytes)
        .saturating_add(experts.stage_bytes)
}

/// The share of unified memory a model and its sessions may use: the memory minus an OS floor
/// of one sixth, at least 4 GiB, rounded up to whole MiB.
pub(crate) fn host_budget(memory_bytes: u64) -> u64 {
    const MIB: u64 = 1 << 20;
    let floor = (memory_bytes / 6)
        .div_ceil(MIB)
        .saturating_mul(MIB)
        .max(4 << 30);
    memory_bytes.saturating_sub(floor)
}

/// What an MLX start reads about the machine, once: the unified memory, what Metal recommends as
/// a working set (`None` where there is no Metal), and the owner's ceiling
/// (`FIFI_MLX_WORKING_SET_MIB`). Every Apple Silicon family plans and loads from this one value,
/// so a plan that accepts a start is the budget the load enforces. The type holds numbers only;
/// reading them is `mac_memory`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct MacMemory {
    pub(crate) physical_bytes: u64,
    pub(crate) metal_bytes: Option<u64>,
    pub(crate) ceiling_bytes: Option<u64>,
}

impl MacMemory {
    /// The bytes a model and its sessions may take: the host budget (the memory less the OS
    /// floor), bounded by Metal's recommended working set and by the ceiling. A ceiling above the
    /// others never raises it.
    pub(crate) fn budget(self) -> u64 {
        host_budget(self.physical_bytes)
            .min(self.metal_bytes.unwrap_or(u64::MAX))
            .min(self.ceiling_bytes.unwrap_or(u64::MAX))
    }

    /// The budget a start that sets no device budget of its own sizes itself from: `budget()`
    /// when a resident model of `model_bytes` does not fit it, and `None` when it fits, so a start
    /// that fits is not sized and behaves as it always did.
    #[cfg(any(
        test,
        all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64")
    ))]
    pub(crate) fn implied_budget(self, model_bytes: u64) -> Option<u64> {
        let budget = self.budget();
        (model_bytes > budget).then_some(budget)
    }
}

/// Why a Qwen start with no device budget of its own takes the memory budget anyway, or `None`
/// when it runs unsized as it always did. The decision reads facts only: the host plan already
/// chose to stream (`staged_cap`), the owner forces streaming, or the resident model of
/// `model_bytes` does not fit the budget.
#[cfg(any(
    test,
    all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64")
))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SizedBecause {
    HostPlanStreams,
    Forced,
    ModelDoesNotFit,
}

#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
impl SizedBecause {
    /// The start line that says why the start is sized from `budget`.
    pub(crate) fn note(self, budget: u64) -> String {
        match self {
            Self::HostPlanStreams => format!(
                "q36: the resident model does not fit the budget of {budget} bytes; routed experts stream from disk (set FIFI_Q36_MLX_DISK_EXPERTS=0 to refuse instead)"
            ),
            Self::Forced => format!(
                "q36: FIFI_Q36_MLX_DISK_EXPERTS=1 streams the routed experts from disk under the budget of {budget} bytes"
            ),
            Self::ModelDoesNotFit => format!(
                "mlx: the resident model does not fit the budget of {budget} bytes; sizing from the budget"
            ),
        }
    }
}

#[cfg(any(
    test,
    all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64")
))]
pub(crate) fn sized_because(
    memory: MacMemory,
    staged_cap: Option<u64>,
    force_stream: bool,
    model_bytes: Option<u64>,
) -> Option<SizedBecause> {
    if staged_cap.is_some() {
        Some(SizedBecause::HostPlanStreams)
    } else if force_stream {
        Some(SizedBecause::Forced)
    } else {
        model_bytes
            .and_then(|model| memory.implied_budget(model))
            .map(|_| SizedBecause::ModelDoesNotFit)
    }
}

/// What a model, a session reserve and one saved prefix need against a budget.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct EntryFit {
    pub(crate) needed: u64,
    pub(crate) budget: u64,
}

impl EntryFit {
    pub(crate) fn of(model_bytes: u64, reserve: u64, entry: u64, budget: u64) -> Self {
        Self {
            needed: model_bytes.saturating_add(reserve).saturating_add(entry),
            budget,
        }
    }

    pub(crate) fn fits(self) -> bool {
        self.needed <= self.budget
    }
}

/// Reads the machine and the checkpoint under `artifact` and says whether a start with no budget
/// of its own is sized from the memory budget, and why (`sized_because`).
#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
pub(crate) fn sizing_for(
    artifact: &Path,
    staged_cap: Option<u64>,
    force_stream: bool,
) -> Option<(SizedBecause, MacMemory)> {
    let memory = mac_memory()?;
    sized_because(memory, staged_cap, force_stream, shard_bytes(artifact))
        .map(|because| (because, memory))
}

/// The ceiling variable, in whole MiB.
pub(crate) const CEILING_MIB: &str = "FIFI_MLX_WORKING_SET_MIB";

/// The ceiling variable as bytes: unset gives `None`, a malformed or zero value is an error that
/// names the variable.
pub(crate) fn ceiling_from_env() -> Result<Option<u64>, crate::EngError> {
    std::env::var_os(CEILING_MIB)
        .as_deref()
        .map(parse_ceiling)
        .transpose()
}

pub(crate) fn parse_ceiling(value: &std::ffi::OsStr) -> Result<u64, crate::EngError> {
    value
        .to_str()
        .and_then(|value| value.parse::<u64>().ok())
        .and_then(|mib| mib.checked_mul(1 << 20))
        .filter(|&bytes| bytes != 0)
        .ok_or_else(|| {
            crate::EngError::BadRequest(format!(
                "{CEILING_MIB} must be a positive whole number of MiB that fits in u64 bytes"
            ))
        })
}

/// Reads the machine: the physical memory, Metal's recommendation where this build has Metal, and
/// the ceiling. `None` when the memory is unknown. A malformed ceiling is left out here and
/// refused by the start that loads (`DeviceBudget::detect`).
pub(crate) fn mac_memory() -> Option<MacMemory> {
    Some(MacMemory {
        physical_bytes: physical_memory_bytes()?,
        metal_bytes: metal_working_set_bytes(),
        ceiling_bytes: ceiling_from_env().ok().flatten(),
    })
}

#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
fn metal_working_set_bytes() -> Option<u64> {
    Some(crate::kernels::MlxKernels::recommended_working_set_bytes()).filter(|&bytes| bytes > 0)
}

#[cfg(not(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64")))]
fn metal_working_set_bytes() -> Option<u64> {
    None
}

#[cfg(target_os = "macos")]
pub(crate) fn physical_memory_bytes() -> Option<u64> {
    // The absolute path: a PATH lookup fails under a restricted PATH or sandbox.
    let out = std::process::Command::new("/usr/sbin/sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .ok()
        .filter(|out| out.status.success())?;
    String::from_utf8_lossy(&out.stdout).trim().parse().ok()
}

/// The bytes of this Mac that the OS can hand out without paging: the free, inactive and
/// speculative pages of `vm_stat`, read now. `None` when the tool or its output cannot be read.
#[cfg(target_os = "macos")]
pub(crate) fn available_memory_bytes() -> Option<u64> {
    use std::sync::Mutex;
    use std::time::Instant;
    // A save, a spill check and a landing each ask, often within one request: one read serves them.
    static LAST: Mutex<Option<(Instant, u64)>> = Mutex::new(None);
    let mut last = LAST.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    cached_read(&mut last, Instant::now(), || {
        std::process::Command::new("/usr/bin/vm_stat")
            .output()
            .ok()
            .filter(|out| out.status.success())
            .and_then(|out| parse_vm_stat(&String::from_utf8_lossy(&out.stdout)))
    })
}

/// A read kept for one second: a value read less than a second before `now` is returned without
/// reading again. A failed read is not kept, so the next call tries again.
#[cfg(any(test, target_os = "macos"))]
fn cached_read(
    last: &mut Option<(std::time::Instant, u64)>,
    now: std::time::Instant,
    read: impl FnOnce() -> Option<u64>,
) -> Option<u64> {
    if let Some((at, value)) = *last
        && now.saturating_duration_since(at) < std::time::Duration::from_secs(1)
    {
        return Some(value);
    }
    let value = read()?;
    *last = Some((now, value));
    Some(value)
}

/// The available bytes in the text of `vm_stat`: its page size times the free, inactive and
/// speculative page counts.
#[cfg(any(test, target_os = "macos"))]
fn parse_vm_stat(text: &str) -> Option<u64> {
    let page_size: u64 = text
        .lines()
        .next()?
        .split("page size of ")
        .nth(1)?
        .split_whitespace()
        .next()?
        .parse()
        .ok()?;
    let pages = |name: &str| -> Option<u64> {
        text.lines()
            .find_map(|line| line.strip_prefix(name)?.strip_prefix(':'))?
            .trim()
            .trim_end_matches('.')
            .parse()
            .ok()
    };
    let available = pages("Pages free")?
        .checked_add(pages("Pages inactive")?)?
        .checked_add(pages("Pages speculative")?)?;
    available.checked_mul(page_size)
}

#[cfg(all(not(target_os = "macos"), any(test, target_os = "linux")))]
pub(crate) fn physical_memory_bytes() -> Option<u64> {
    super::HostFacts::physical_total()
}

#[cfg(not(any(target_os = "macos", target_os = "linux", test)))]
pub(crate) fn physical_memory_bytes() -> Option<u64> {
    None
}

/// The layer index of a routed expert tensor name (`...layers.N...switch_mlp...`), which Qwen3.6
/// and Flint checkpoints share.
fn expert_layer(name: &str) -> Option<u32> {
    let (before, _) = name.split_once(".switch_mlp.")?;
    before
        .rsplit_once(".layers.")?
        .1
        .split('.')
        .next()?
        .parse()
        .ok()
}

/// Reads the routed expert bytes of every top-level shard under `model_dir` (`None` when no
/// shard header is readable or the checkpoint has no expert stacks).
pub(crate) fn expert_bytes_in(model_dir: &std::path::Path) -> Option<ExpertBytes> {
    let mut per_layer: std::collections::BTreeMap<u32, u64> = std::collections::BTreeMap::new();
    let mut shards = 0u32;
    for entry in std::fs::read_dir(model_dir).ok()? {
        let path = entry.ok()?.path();
        if path.extension().is_some_and(|ext| ext == "safetensors") {
            expert_bytes_of_shard(&path, &mut per_layer)?;
            shards += 1;
        }
    }
    let total_bytes = per_layer.values().sum::<u64>();
    (shards > 0 && total_bytes > 0).then(|| ExpertBytes {
        total_bytes,
        stage_bytes: per_layer.values().copied().max().unwrap_or(0),
    })
}

fn expert_bytes_of_shard(
    path: &std::path::Path,
    per_layer: &mut std::collections::BTreeMap<u32, u64>,
) -> Option<()> {
    use std::io::Read;
    let mut file = std::fs::File::open(path).ok()?;
    let mut length = [0u8; 8];
    file.read_exact(&mut length).ok()?;
    let length = usize::try_from(u64::from_le_bytes(length)).ok()?;
    if length > 64 << 20 {
        return None;
    }
    let mut header = vec![0u8; length];
    file.read_exact(&mut header).ok()?;
    let header: serde_json::Value = serde_json::from_slice(&header).ok()?;
    for (name, tensor) in header.as_object()? {
        let Some(layer) = expert_layer(name) else {
            continue;
        };
        let offsets = tensor.get("data_offsets")?.as_array()?;
        let (start, end) = (offsets.first()?.as_u64()?, offsets.get(1)?.as_u64()?);
        *per_layer.entry(layer).or_default() += end.checked_sub(start)?;
    }
    Some(())
}

/// The summed length of the top-level `*.safetensors` files, as the loader opens them.
pub(crate) fn shard_bytes(model_dir: &std::path::Path) -> Option<u64> {
    let mut total = 0u64;
    let mut shards = 0u32;
    for entry in std::fs::read_dir(model_dir).ok()? {
        let path = entry.ok()?.path();
        if path.extension().is_some_and(|ext| ext == "safetensors") {
            total = total.checked_add(std::fs::metadata(&path).ok()?.len())?;
            shards += 1;
        }
    }
    (shards > 0).then_some(total)
}

// The derived defaults of an MLX start, read by every family's settings.

use std::path::Path;

const GIB: u64 = 1 << 30;

/// A byte count in GiB, to one decimal.
pub(crate) fn gib(bytes: u64) -> String {
    format!("{:.1} GiB", bytes as f64 / GIB as f64)
}

/// The allocator cache limit bounds.
const MIN_CACHE_LIMIT_BYTES: u64 = GIB;
const MAX_CACHE_LIMIT_BYTES: u64 = 8 * GIB;

/// The least context a derived choice may land on: below it the start is refused.
pub(crate) const MIN_DERIVED_CONTEXT: u32 = 4096;
/// The step of a derived context. A multiple of the cache growth quantum (256 positions) and of
/// the 8-bit cache group (64), and the step the context search of the capability report uses.
const CONTEXT_STEP: u32 = 1024;

/// What the host and the model directory say, read once at start.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct MacHostFacts {
    /// Physical (unified) memory in bytes.
    pub(crate) memory_bytes: u64,
    /// What Metal recommends as a working set, where this build has Metal.
    pub(crate) metal_bytes: Option<u64>,
    /// The `FIFI_MLX_WORKING_SET_MIB` ceiling, in bytes.
    pub(crate) ceiling_bytes: Option<u64>,
    /// The bytes of the model's `*.safetensors` shards: the resident weights charge.
    pub(crate) model_bytes: u64,
}

/// The values derived from the host facts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct HostDefaults {
    pub(crate) os_floor_bytes: u64,
    pub(crate) prefill_budget_bytes: u64,
    pub(crate) cache_limit_bytes: u64,
}

impl MacHostFacts {
    /// Reads the physical memory of this host and the shard bytes under `model_dir`. `None` when
    /// either is unknown (no memory probe, no shard): the caller then keeps the settings as they
    /// were before derivation existed, with their original refusals.
    pub(crate) fn probe(model_dir: &Path) -> Option<Self> {
        let memory = mac_memory()?;
        let model_bytes = shard_bytes(model_dir)?;
        Some(Self {
            memory_bytes: memory.physical_bytes,
            metal_bytes: memory.metal_bytes,
            ceiling_bytes: memory.ceiling_bytes,
            model_bytes,
        })
    }

    /// The machine facts of the start, the one input of its memory budget.
    pub(crate) fn memory(self) -> MacMemory {
        MacMemory {
            physical_bytes: self.memory_bytes,
            metal_bytes: self.metal_bytes,
            ceiling_bytes: self.ceiling_bytes,
        }
    }

    pub(crate) fn defaults(self) -> HostDefaults {
        let prefill_budget_bytes = self.memory().budget();
        HostDefaults {
            os_floor_bytes: self
                .memory_bytes
                .saturating_sub(host_budget(self.memory_bytes)),
            prefill_budget_bytes,
            cache_limit_bytes: (self.memory_bytes / 9 / GIB * GIB)
                .clamp(MIN_CACHE_LIMIT_BYTES, MAX_CACHE_LIMIT_BYTES),
        }
    }
}

/// The largest context whose session reserve still fits next to the model in `budget`, when the
/// reserve is `per_position * context + fixed_bytes` (the shape of the derived reserve): a
/// multiple of 1024 positions, at least 4,096. `None` when the model and the fixed part leave no
/// room for that floor, or `per_position` is 0.
pub(crate) fn largest_context(
    facts: MacHostFacts,
    budget: u64,
    per_position: u64,
    fixed_bytes: u64,
) -> Option<u32> {
    let room = budget
        .checked_sub(facts.model_bytes)?
        .checked_sub(fixed_bytes)?;
    let positions = room.checked_div(per_position)?;
    let step = u64::from(CONTEXT_STEP);
    let rounded =
        u32::try_from(positions / step * step).unwrap_or(u32::MAX / CONTEXT_STEP * CONTEXT_STEP);
    (rounded >= MIN_DERIVED_CONTEXT).then_some(rounded)
}

/// The refusal of a host whose derived budget cannot hold the model and the session reserve.
pub(crate) fn shortfall_text(
    facts: MacHostFacts,
    budget: u64,
    reserve: u64,
    context_limit: u32,
) -> Option<String> {
    let needed = facts.model_bytes.checked_add(reserve)?;
    if needed <= budget {
        return None;
    }
    let defaults = facts.defaults();
    Some(format!(
        "this host has {} of unified memory, {} after the {} OS floor; the model needs {} and the \
         session reserve at context {context_limit} needs {} more, {} in all: short by {} ({} \
         bytes). Use a smaller model, set a smaller context_limit and cache_limit_bytes, or set \
         prefill_budget_bytes yourself",
        gib(facts.memory_bytes),
        gib(budget),
        gib(defaults.os_floor_bytes),
        gib(facts.model_bytes),
        gib(reserve),
        gib(needed),
        gib(needed - budget),
        needed - budget,
    ))
}

pub(crate) use self::ExpertBytes as ExpertFacts;

impl MacHostFacts {
    /// The facts of a start that streams its routed experts: the model keeps everything except
    /// the expert tensors resident, plus one stage of experts.
    pub(crate) fn staged(self, experts: ExpertFacts) -> Self {
        Self {
            model_bytes: staged_model_bytes(self.model_bytes, experts),
            ..self
        }
    }
}

/// Reads the routed expert bytes of the checkpoint under `model_dir`.
pub(crate) fn expert_facts(model_dir: &Path) -> Option<ExpertFacts> {
    expert_bytes_in(model_dir)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expert_bytes_come_from_the_shard_headers_per_layer() {
        let dir = std::env::temp_dir().join(format!("mlx-budget-experts-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let header = serde_json::json!({
            "language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight": {"dtype": "U32", "shape": [1], "data_offsets": [0, 100]},
            "language_model.model.layers.0.mlp.switch_mlp.down_proj.weight": {"dtype": "U32", "shape": [1], "data_offsets": [100, 150]},
            "language_model.model.layers.1.mlp.switch_mlp.gate_proj.weight": {"dtype": "U32", "shape": [1], "data_offsets": [150, 400]},
            "language_model.model.layers.1.self_attn.q_proj.weight": {"dtype": "U32", "shape": [1], "data_offsets": [400, 900]},
        })
        .to_string();
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        std::fs::write(dir.join("a.safetensors"), bytes).unwrap();
        let experts = expert_bytes_in(&dir).unwrap();
        assert_eq!((experts.total_bytes, experts.stage_bytes), (400, 250));
        assert_eq!(staged_model_bytes(1000, experts), 1000 - 400 + 250);
        std::fs::write(dir.join("a.safetensors"), [0u8; 4]).unwrap();
        assert_eq!(expert_bytes_in(&dir), None);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn the_host_budget_leaves_a_sixth_and_at_least_four_gib() {
        const GIB: u64 = 1 << 30;
        assert_eq!(host_budget(36 * GIB), 30 * GIB);
        assert_eq!(host_budget(16 * GIB), 12 * GIB);
        assert_eq!(host_budget(24 * GIB), 20 * GIB);
    }

    fn mac(memory_gib: u64, metal_gib: Option<u64>, ceiling_mib: Option<u64>) -> MacMemory {
        const GIB: u64 = 1 << 30;
        MacMemory {
            physical_bytes: memory_gib * GIB,
            metal_bytes: metal_gib.map(|gib| gib * GIB),
            ceiling_bytes: ceiling_mib.map(|mib| mib << 20),
        }
    }

    #[test]
    fn the_budget_is_the_lowest_of_the_host_budget_metal_and_the_ceiling() {
        const GIB: u64 = 1 << 30;
        // No Metal and no ceiling: the host budget of a 16, 24, 36 and 64 GiB host.
        for (memory, want) in [(16, 12 * GIB), (24, 20 * GIB), (36, 30 * GIB)] {
            assert_eq!(mac(memory, None, None).budget(), want, "{memory} GiB");
        }
        // Metal below the host budget lowers it, so a plan never admits what the load refuses.
        assert_eq!(mac(16, Some(11), None).budget(), 11 * GIB);
        assert_eq!(mac(24, Some(18), None).budget(), 18 * GIB);
        // Metal above the host budget does not raise it.
        assert_eq!(mac(36, Some(48), None).budget(), 30 * GIB);
        // The ceiling lowers and never raises.
        assert_eq!(mac(36, Some(28), Some(12288)).budget(), 12 * GIB);
        assert_eq!(mac(36, Some(28), Some(65536)).budget(), 28 * GIB);
    }

    #[test]
    fn a_start_with_no_budget_of_its_own_is_sized_only_for_a_reason() {
        const GIB: u64 = 1 << 30;
        let model = Some(20_401_094_656u64);
        // A 36 GiB host holds the model: no reason, the start runs as it always did.
        assert_eq!(sized_because(mac(36, None, None), None, false, model), None);
        // 16 GiB does not hold it, and a 24 GiB host whose Metal working set is 17 GiB does not.
        assert_eq!(
            sized_because(mac(16, None, None), None, false, model),
            Some(SizedBecause::ModelDoesNotFit)
        );
        assert_eq!(
            sized_because(mac(24, Some(17), None), None, false, model),
            Some(SizedBecause::ModelDoesNotFit)
        );
        // A set working-set ceiling below the model sizes a 36 GiB host as well.
        assert_eq!(
            sized_because(mac(36, None, Some(12 * 1024)), None, false, model),
            Some(SizedBecause::ModelDoesNotFit)
        );
        // The host plan's streaming and the owner's `1` size it whatever the shards say.
        assert_eq!(
            sized_because(mac(36, None, None), Some(30 * GIB), false, None),
            Some(SizedBecause::HostPlanStreams)
        );
        assert_eq!(
            sized_because(mac(36, None, None), None, true, None),
            Some(SizedBecause::Forced)
        );
    }

    #[test]
    fn a_start_sizes_itself_from_the_host_only_when_the_resident_model_does_not_fit() {
        const GIB: u64 = 1 << 30;
        // The shard bytes of the Qwen3.6 MLX 4-bit checkpoint (19.0 GiB) and of the Flint INT4
        // checkpoint (21.05 GiB) against the host budgets of 16, 24, 36 and 64 GiB hosts
        // (12, 20, 30 and 53.3 GiB). Qwen3.6 on a 24 GiB host fits by its shards alone; its host
        // plan adds the session reserve and streams the experts instead.
        for (name, model, sized) in [
            ("qwen36", 20_401_094_656u64, [true, false, false, false]),
            ("flint", 22_600_000_000, [true, true, false, false]),
            ("large", 80 * GIB, [true, true, true, true]),
        ] {
            for (memory, want) in [16, 24, 36, 64].into_iter().zip(sized) {
                assert_eq!(
                    mac(memory, None, None).implied_budget(model).is_some(),
                    want,
                    "{name} on {memory} GiB"
                );
            }
        }
        assert_eq!(mac(24, None, None).implied_budget(22 * GIB), Some(20 * GIB));
    }
}

#[cfg(test)]
mod vm_stat_tests {
    use super::parse_vm_stat;

    const SAMPLE: &str = "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n\
Pages free:                                  1920791.\n\
Pages active:                                 109956.\n\
Pages inactive:                               106344.\n\
Pages speculative:                              7385.\n\
Pages throttled:                                   0.\n\
Pages wired down:                             145145.\n\
Pages purgeable:                                1244.\n\
\"Translation faults\":                    27980608462.\n";

    #[test]
    fn the_available_bytes_are_the_free_inactive_and_speculative_pages() {
        assert_eq!(
            parse_vm_stat(SAMPLE),
            Some((1_920_791 + 106_344 + 7_385) * 16_384)
        );
    }

    #[test]
    fn a_text_without_the_counts_gives_none() {
        assert_eq!(parse_vm_stat(""), None);
        assert_eq!(parse_vm_stat("Pages free: 1.\n"), None);
        assert_eq!(
            parse_vm_stat("Mach (page size of 4096 bytes)\nPages free: 1.\n"),
            None
        );
    }

    #[test]
    fn a_read_is_kept_for_a_second_and_a_failure_is_not_kept() {
        use super::cached_read;
        use std::time::{Duration, Instant};
        let start = Instant::now();
        let mut last = None;
        assert_eq!(cached_read(&mut last, start, || Some(7)), Some(7));
        assert_eq!(
            cached_read(&mut last, start + Duration::from_millis(500), || Some(9)),
            Some(7),
            "within the second the first value serves"
        );
        assert_eq!(
            cached_read(&mut last, start + Duration::from_millis(1_500), || Some(9)),
            Some(9),
            "after it the read runs again"
        );
        let mut empty = None;
        assert_eq!(cached_read(&mut empty, start, || None), None);
        assert_eq!(
            cached_read(&mut empty, start, || Some(5)),
            Some(5),
            "a failed read leaves nothing kept"
        );
    }
}
