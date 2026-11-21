//! Host-side planning for MLX resident prefill. Each family supplies the workspace cost
//! for its own tensors. The chunk ladder and DeltaNet retire-memory rule are shared.

use crate::EngError;

pub(crate) const MIN_PREFILL_CHUNK_ROWS: u32 = 4;
pub(crate) const PREFILL_CANDIDATES: [u32; 11] = [
    2048,
    1024,
    512,
    256,
    128,
    64,
    32,
    16,
    8,
    MIN_PREFILL_CHUNK_ROWS,
    1,
];

/// A chunk temporarily sets the MLX retirement cap to 512 MiB. Leave 64 MiB
/// available for other temporaries when sizing the DeltaNet recurrence.
pub(crate) const PREFILL_RETIRE_CAP: u64 = 512 << 20;
pub(crate) const RETIRE_CAP_BASE: u64 = 64 << 20;

/// The staging reads of one engine: whether it groups adjacent expert reads, overlaps the shared
/// expert with them, and reads shard ranges at explicit offsets. They are part of the engine's
/// placement, not of the process: two engines in one process, one streaming its experts and one
/// holding them resident, each carry their own value.
#[cfg(any(
    test,
    all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64")
))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct StagingReads {
    pub(crate) read_runs: bool,
    pub(crate) overlap: bool,
    pub(crate) positioned: bool,
}

/// What the three variables say, when they are set: `FIFI_MLX_EXPERT_READ_RUNS` (`prefill` on, any
/// other value off), `FIFI_MLX_OVERLAP_EXPERT_READS` and `FIFI_MLX_POSITIONED_READS` (`1` on, any
/// other value off).
#[cfg(any(
    test,
    all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64")
))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct StagingOverrides {
    pub(crate) read_runs: Option<bool>,
    pub(crate) overlap: Option<bool>,
    pub(crate) positioned: Option<bool>,
}

#[cfg(any(
    test,
    all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64")
))]
impl StagingOverrides {
    /// The variables of this process, read once.
    #[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
    pub(crate) fn from_env() -> Self {
        static CELL: std::sync::OnceLock<StagingOverrides> = std::sync::OnceLock::new();
        *CELL.get_or_init(|| {
            let flag = |name: &str, on: &str| std::env::var_os(name).map(|value| value == on);
            Self {
                read_runs: flag("FIFI_MLX_EXPERT_READ_RUNS", "prefill"),
                overlap: flag("FIFI_MLX_OVERLAP_EXPERT_READS", "1"),
                positioned: flag("FIFI_MLX_POSITIONED_READS", "1"),
            }
        })
    }
}

#[cfg(any(
    test,
    all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64")
))]
impl StagingReads {
    /// The reads of an engine whose experts `streams` from disk, under a device budget when
    /// `budgeted`: grouped reads and the overlap follow the streaming (the A/B measured them as
    /// the faster path whenever experts come from disk), positioned reads follow the budget (the
    /// experts may stream and the shard files open before the placement is known). A variable that
    /// is set decides its read.
    pub(crate) fn resolve(streams: bool, budgeted: bool, overrides: StagingOverrides) -> Self {
        Self {
            read_runs: overrides.read_runs.unwrap_or(streams),
            overlap: overrides.overlap.unwrap_or(streams),
            positioned: overrides.positioned.unwrap_or(budgeted),
        }
    }

    /// The reads of the variables of this process for an engine with this placement.
    #[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
    pub(crate) fn of_placement(streams: bool, budgeted: bool) -> Self {
        Self::resolve(streams, budgeted, StagingOverrides::from_env())
    }
}

/// Queue one independent GPU result while the worker reads the exact expert IDs.
/// Its owner passes the returned tensor into the dependent output graph.
#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
pub(crate) fn submit_shared_expert(
    kernels: &mut crate::kernels::MlxKernels,
    output: crate::tensor::ExecutionTensor,
) -> Result<crate::tensor::ExecutionTensor, EngError> {
    kernels.evaluate_async(&[&output])?;
    Ok(output)
}

/// Sort and deduplicate checkpoint expert IDs for shared page staging.
#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
pub(crate) fn sorted_expert_ids(selected: &[u32]) -> Vec<u32> {
    let mut experts = selected.to_vec();
    experts.sort_unstable();
    experts.dedup();
    experts
}

/// Index the sorted compact stack while keeping the original pair order.
#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
pub(crate) fn local_expert_ids(selected: &[u32], experts: &[u32]) -> Result<Vec<u32>, EngError> {
    selected
        .iter()
        .map(|id| {
            experts
                .binary_search(id)
                .ok()
                .and_then(|slot| u32::try_from(slot).ok())
                .ok_or_else(|| {
                    EngError::BadRequest(format!("expert {id} is outside the staged stack"))
                })
        })
        .collect()
}

/// F32 bytes for the replaced state, decayed state and outer product per token.
pub(crate) fn delta_retire_per_token(heads: u32, key_dim: u32, value_dim: u32) -> Option<u64> {
    u64::from(heads)
        .checked_mul(u64::from(key_dim))?
        .checked_mul(u64::from(value_dim))?
        .checked_mul(3 * 4)
        .filter(|&bytes| bytes > 0)
}

pub(crate) fn delta_subchunk(rows: u32, cap: u64, per_token: u64) -> Option<u32> {
    if rows == 0 || per_token == 0 {
        return None;
    }
    let fit = cap.saturating_sub(RETIRE_CAP_BASE) / per_token;
    Some(u32::try_from(fit).unwrap_or(u32::MAX).clamp(1, rows))
}

/// Find the largest fitting candidate. Both workspace models are monotone.
/// On the descending serving ladder, only price rows through the first fit.
pub(in crate::family) fn largest_fitting_chunk(
    candidates: impl IntoIterator<Item = u32>,
    remaining: u64,
    workspace: impl Fn(u32) -> Result<u64, EngError>,
) -> Result<Option<u32>, EngError> {
    let mut best = None;
    for rows in candidates {
        if rows == 0 {
            return Err(EngError::BadRequest(
                "prefill chunk size must exceed zero".into(),
            ));
        }
        if best.is_none_or(|previous| rows > previous) && workspace(rows)? <= remaining {
            best = Some(rows);
        }
    }
    Ok(best)
}

/// A one-row fallback needs no chunk allocation. Limit the search to sizes
/// the worker can actually submit before asking the family for a cost.
pub(crate) fn admitted_chunk(
    max_rows: u32,
    remaining: u64,
    workspace: impl Fn(u32) -> Result<u64, EngError>,
) -> Result<Option<u32>, EngError> {
    let sizes = PREFILL_CANDIDATES
        .into_iter()
        .filter(|&rows| rows > 1 && rows <= max_rows);
    largest_fitting_chunk(sizes, remaining, workspace)
}

#[cfg(test)]
mod staging_tests {
    use super::*;

    #[test]
    fn the_reads_follow_each_engines_own_placement() {
        let none = StagingOverrides::default();
        let streaming = StagingReads::resolve(true, true, none);
        let resident = StagingReads::resolve(false, false, none);
        assert_eq!(
            streaming,
            StagingReads {
                read_runs: true,
                overlap: true,
                positioned: true
            }
        );
        assert_eq!(
            resident,
            StagingReads {
                read_runs: false,
                overlap: false,
                positioned: false
            }
        );
        // A budgeted engine whose experts all fit still opens its shards for positioned reads.
        let fits = StagingReads::resolve(false, true, none);
        assert!(fits.positioned && !fits.overlap && !fits.read_runs);
    }

    #[test]
    fn a_set_variable_decides_its_read_for_every_placement() {
        let off = StagingOverrides {
            read_runs: Some(false),
            overlap: Some(false),
            positioned: Some(false),
        };
        assert_eq!(
            StagingReads::resolve(true, true, off),
            StagingReads {
                read_runs: false,
                overlap: false,
                positioned: false
            }
        );
        let on = StagingOverrides {
            overlap: Some(true),
            ..StagingOverrides::default()
        };
        assert!(StagingReads::resolve(false, false, on).overlap);
        assert!(!StagingReads::resolve(false, false, on).read_runs);
    }
}
