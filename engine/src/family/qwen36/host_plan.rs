//! Host-derived defaults of the MLX Qwen3.6 settings.
//!
//! An MLX engine that sets no cache limit and no prefill budget derives both from the unified
//! memory of the host, and turns on the compiled decode blocks. The derivation is pure: the
//! caller reads the host (`HostFacts::probe`) and hands the facts to `Q36Settings::new`, so the
//! same inputs give the same settings in every test and every renderer. Every explicit setting
//! (a config member or `Q36_CACHE_LIMIT`) stays an exact override and is never replaced.
//!
//! The rules are fractions of the host memory, anchored on the 36 GB Mac Studio profile
//! (`engine/config/qwen36-mac-studio-36gb.json`), which this host class runs:
//!
//! - OS floor: one sixth of the memory, at least 4 GiB, rounded up to whole MiB. The prefill
//!   budget is the memory minus the floor: 30 GiB of 36 GiB.
//! - Allocator cache limit: one ninth of the memory, rounded down to whole GiB and held between
//!   1 and 8 GiB: 4 GiB of 36 GiB. Nothing measured the upper bound; it only stops a large host
//!   from retaining allocator buffers without limit.
//! - Context: the default context stays when the model and the session reserve at that context fit
//!   the budget. Otherwise a start that sets no `context_limit` takes the largest multiple of 1024
//!   positions that fits, down to 4,096, and below that it is refused with the shortfall.
//! - Compile: on for the MoE family when no speculation runs, as the shipped profile sets it.

use std::path::Path;

const GIB: u64 = 1 << 30;
#[cfg(test)]
const MIB: u64 = 1 << 20;

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
pub(crate) struct HostFacts {
    /// Physical (unified) memory in bytes.
    pub(crate) memory_bytes: u64,
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

impl HostFacts {
    /// Reads the physical memory of this host and the shard bytes under `model_dir`. `None` when
    /// either is unknown (no memory probe, no shard): the caller then keeps the settings as they
    /// were before derivation existed, with their original refusals.
    pub(crate) fn probe(model_dir: &Path) -> Option<Self> {
        let memory_bytes = physical_memory_bytes()?;
        let model_bytes = weights_bytes(model_dir)?;
        Some(Self {
            memory_bytes,
            model_bytes,
        })
    }

    pub(crate) fn defaults(self) -> HostDefaults {
        let prefill_budget_bytes = crate::family::host_memory::host_budget(self.memory_bytes);
        HostDefaults {
            os_floor_bytes: self.memory_bytes.saturating_sub(prefill_budget_bytes),
            prefill_budget_bytes,
            cache_limit_bytes: (self.memory_bytes / 9 / GIB * GIB)
                .clamp(MIN_CACHE_LIMIT_BYTES, MAX_CACHE_LIMIT_BYTES),
        }
    }
}

/// One value of the startup line: its name, its number, and where the number came from.
pub(crate) struct NoteValue {
    pub(crate) key: &'static str,
    pub(crate) value: String,
    pub(crate) source: String,
}

fn gib(bytes: u64) -> String {
    format!("{:.1} GiB", bytes as f64 / GIB as f64)
}

/// The one startup line: the host facts, then each value with its source.
pub(crate) fn startup_line(facts: HostFacts, values: &[NoteValue]) -> String {
    let defaults = facts.defaults();
    let list = values
        .iter()
        .map(|item| format!("{}={} ({})", item.key, item.value, item.source))
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "q36: host defaults: unified memory {}, model {}, OS floor {}; {list}",
        gib(facts.memory_bytes),
        gib(facts.model_bytes),
        gib(defaults.os_floor_bytes),
    )
}

/// The largest context whose session reserve still fits next to the model in `budget`, when the
/// reserve is `per_position * context + fixed_bytes` (the shape of the derived reserve): a
/// multiple of 1024 positions, at least 4,096. `None` when the model and the fixed part leave no
/// room for that floor, or `per_position` is 0.
pub(crate) fn largest_context(
    facts: HostFacts,
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
    facts: HostFacts,
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

pub(crate) use crate::family::host_memory::ExpertBytes as ExpertFacts;

impl HostFacts {
    /// The facts of a start that streams its routed experts: the model keeps everything except
    /// the expert tensors resident, plus one stage of experts.
    pub(crate) fn staged(self, experts: ExpertFacts) -> Self {
        Self {
            model_bytes: crate::family::host_memory::staged_model_bytes(self.model_bytes, experts),
            ..self
        }
    }
}

/// Reads the routed expert bytes of the checkpoint under `model_dir`.
pub(crate) fn expert_facts(model_dir: &Path) -> Option<ExpertFacts> {
    crate::family::host_memory::expert_bytes_in(model_dir)
}

fn physical_memory_bytes() -> Option<u64> {
    crate::family::host_memory::physical_memory_bytes()
}

/// The summed length of the top-level `*.safetensors` files (symlinks followed), the way the
/// loader opens them. `None` when the directory cannot be read or holds no shard.
fn weights_bytes(model_dir: &Path) -> Option<u64> {
    crate::family::host_memory::shard_bytes(model_dir)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn facts(memory_gib: u64) -> HostFacts {
        HostFacts {
            memory_bytes: memory_gib * GIB,
            model_bytes: 20_402_000_000,
        }
    }

    #[test]
    fn the_36_gib_host_reproduces_the_shipped_profile() {
        let defaults = facts(36).defaults();
        assert_eq!(defaults.cache_limit_bytes, 4_294_967_296);
        assert_eq!(defaults.prefill_budget_bytes, 32_212_254_720);
        assert_eq!(defaults.os_floor_bytes, 6 * GIB);
    }

    #[test]
    fn other_hosts_scale_with_memory() {
        let sixty_four = facts(64).defaults();
        assert_eq!(sixty_four.cache_limit_bytes, 7 * GIB);
        assert_eq!(
            sixty_four.os_floor_bytes,
            (64 * GIB / 6).div_ceil(MIB) * MIB
        );
        assert_eq!(
            sixty_four.prefill_budget_bytes,
            64 * GIB - sixty_four.os_floor_bytes
        );
        let big = facts(128).defaults();
        assert_eq!(big.cache_limit_bytes, 8 * GIB, "held at the upper bound");
        assert_eq!(big.prefill_budget_bytes, 128 * GIB - big.os_floor_bytes);
        let small = facts(16).defaults();
        assert_eq!(small.cache_limit_bytes, GIB);
        assert_eq!(small.os_floor_bytes, 4 * GIB, "the floor is at least 4 GiB");
        assert_eq!(small.prefill_budget_bytes, 12 * GIB);
    }

    #[test]
    fn shortfall_names_the_numbers_and_only_fires_when_short() {
        let host = facts(16);
        let budget = host.defaults().prefill_budget_bytes;
        let text = shortfall_text(host, budget, 5 * GIB, 262_144).expect("16 GiB is short");
        assert!(text.contains("16.0 GiB of unified memory"), "{text}");
        assert!(
            text.contains("12.0 GiB after the 4.0 GiB OS floor"),
            "{text}"
        );
        assert!(text.contains("the model needs 19.0 GiB"), "{text}");
        assert!(text.contains("short by 12.0 GiB"), "{text}");
        assert!(text.contains("context 262144"), "{text}");
        let roomy = facts(36);
        assert!(
            shortfall_text(
                roomy,
                roomy.defaults().prefill_budget_bytes,
                8_286_699_520,
                262_144
            )
            .is_none()
        );
    }

    #[test]
    fn the_largest_context_is_a_whole_step_that_fits() {
        let host = facts(36);
        let budget = host.defaults().prefill_budget_bytes;
        let fixed = 5 * GIB;
        let per_position = 10_880;
        let context = largest_context(host, budget, per_position, fixed).expect("fits");
        assert_eq!(context % 1024, 0);
        let reserve = |positions: u32| per_position * u64::from(positions) + fixed;
        assert!(host.model_bytes + reserve(context) <= budget);
        assert!(host.model_bytes + reserve(context + 1024) > budget);
        // No room for the fixed part, a floor that does not fit, a zero charge per position.
        assert_eq!(largest_context(host, budget, per_position, budget), None);
        let tight = budget - host.model_bytes - fixed;
        assert_eq!(
            largest_context(host, budget, tight / 4095, fixed),
            None,
            "4,095 positions are below the floor"
        );
        assert_eq!(
            largest_context(host, budget, tight / 4096, fixed),
            Some(4096)
        );
        assert_eq!(largest_context(host, budget, 0, fixed), None);
    }

    #[test]
    fn the_line_names_each_value_and_its_source() {
        let line = startup_line(
            facts(36),
            &[
                NoteValue {
                    key: "cache_limit_bytes",
                    value: "4294967296".into(),
                    source: "derived: memory / 9".into(),
                },
                NoteValue {
                    key: "compile",
                    value: "true".into(),
                    source: "set".into(),
                },
            ],
        );
        assert_eq!(
            line,
            "q36: host defaults: unified memory 36.0 GiB, model 19.0 GiB, OS floor 6.0 GiB; \
             cache_limit_bytes=4294967296 (derived: memory / 9), compile=true (set)"
        );
    }

    #[test]
    fn a_directory_without_shards_gives_no_facts() {
        let dir = std::env::temp_dir().join(format!("host-plan-empty-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        assert_eq!(weights_bytes(&dir), None);
        std::fs::write(dir.join("a.safetensors"), [0u8; 10]).unwrap();
        std::fs::write(dir.join("b.safetensors"), [0u8; 5]).unwrap();
        std::fs::write(dir.join("c.json"), [0u8; 100]).unwrap();
        assert_eq!(weights_bytes(&dir), Some(15));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_staged_start_keeps_everything_but_the_experts_and_one_stage() {
        let host = facts(16);
        let experts = ExpertFacts {
            total_bytes: 18_119_393_280,
            stage_bytes: 452_984_832,
        };
        let staged = host.staged(experts);
        assert_eq!(staged.memory_bytes, host.memory_bytes);
        assert_eq!(
            staged.model_bytes,
            20_402_000_000 - 18_119_393_280 + 452_984_832
        );
    }
}
