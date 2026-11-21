//! The size rules of an EXL3 routed job table: pure arithmetic, shared by the host-side scratch
//! planners (which size the sort lists and the job table) and the device-side job-table
//! machinery in [`crate::family::project::cuda::exl3_jobs`].

#[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
use crate::EngError;

/// The sorted plan of a chunk's `m * top_k` pairs: the counting sort by member and the two row
/// lists the launches read.
#[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
#[derive(Debug, PartialEq, Eq)]
pub(in crate::family) struct PairPlan {
    /// Pairs of each member.
    pub(in crate::family) counts: Vec<usize>,
    /// For each sorted position, the row of the chunk it reads.
    pub(in crate::family) order: Vec<i32>,
    /// For each sorted position, its pair `row * top_k + rank`: the place its output lands in.
    pub(in crate::family) scatter: Vec<i32>,
}

#[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
impl PairPlan {
    /// Cuts the sorted plan between two member buffers. The second source's member ids start
    /// at zero, but both row lists keep the original chunk's row/rank coordinates.
    pub(in crate::family) fn split_members(
        mut self,
        boundary: usize,
    ) -> Result<[Self; 2], EngError> {
        if boundary > self.counts.len() {
            return Err(EngError::BadRequest(
                "EXL3 moe chunk: invalid member boundary".into(),
            ));
        }
        let at = self.counts[..boundary].iter().sum();
        let second = Self {
            counts: self.counts.split_off(boundary),
            order: self.order.split_off(at),
            scatter: self.scatter.split_off(at),
        };
        Ok([self, second])
    }
}

/// Counting sort of the pairs by member: within a member the rows keep their chunk order. An id
/// outside `0..n_member` is a fault mark (-1) or a corrupt word, and refuses the chunk.
#[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
pub(in crate::family) fn plan_pairs(
    picks: &[i32],
    n_member: usize,
    top_k: usize,
) -> Result<PairPlan, EngError> {
    let mut counts = vec![0usize; n_member];
    for &pick in picks {
        let member = usize::try_from(pick).ok().filter(|&e| e < n_member);
        let Some(member) = member else {
            return Err(EngError::ModelOutput(
                "EXL3 moe chunk: the router picked no valid expert (non-finite logits)".into(),
            ));
        };
        counts[member] += 1;
    }
    let mut next = Vec::with_capacity(n_member);
    let mut at = 0;
    for &count in &counts {
        next.push(at);
        at += count;
    }
    let (mut order, mut scatter) = (vec![0i32; picks.len()], vec![0i32; picks.len()]);
    let too_many = || EngError::BadRequest("EXL3 moe chunk: the pair count exceeds i32".into());
    for (pair, &pick) in picks.iter().enumerate() {
        let slot = &mut next[pick as usize];
        order[*slot] = i32::try_from(pair / top_k).map_err(|_| too_many())?;
        scatter[*slot] = i32::try_from(pair).map_err(|_| too_many())?;
        *slot += 1;
    }
    Ok(PairPlan {
        counts,
        order,
        scatter,
    })
}

/// Jobs over a subset of the routed pairs. Every result retains its original
/// row and rank destination in the full chunk.
#[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
pub(in crate::family) fn plan_subset_pairs(
    slots: &[usize],
    original_pairs: &[usize],
    members: usize,
    top_k: usize,
) -> Result<PairPlan, EngError> {
    if slots.len() != original_pairs.len() || top_k == 0 {
        return Err(EngError::BadRequest(
            "EXL3 expert wave needs one slot for each original pair".into(),
        ));
    }
    let slots: Vec<i32> = slots
        .iter()
        .copied()
        .map(i32::try_from)
        .collect::<Result<_, _>>()
        .map_err(|_| EngError::BadRequest("EXL3 expert wave has an invalid slot".into()))?;
    let mut plan = plan_pairs(&slots, members, 1)?;
    for (row, pair) in plan.order.iter_mut().zip(&mut plan.scatter) {
        let original = original_pairs[*pair as usize];
        *pair = i32::try_from(original).map_err(|_| {
            EngError::BadRequest("EXL3 expert wave has an out-of-range destination".into())
        })?;
        *row = (original / top_k) as i32;
    }
    Ok(plan)
}

/// The most rows of one job of a routed expert: the narrowest CTA (64 rows). An expert holds a few
/// dozen rows of a chunk, and a wider CTA would decode no fewer weights and run mostly idle warps.
pub(crate) const EXL3_EXPERT_TILE_ROWS: usize = 64;

/// The widest rows window the shared EXL3 expert machinery plans for: `shared_rows`
/// (`family::exl3::moe`) and the Flash-Next chunk plan (`MAX_STEP_ROWS`) refuse wider
/// windows. A kernel cap it is not — the dense job row cap is `jobs_row_cap` (256, exl3.cu)
/// and the shared down groups split at 8 — this is the planning bound both families agreed
/// on. Widened from 64 to 256 once the shared prefill oracle passed at 1, 3, 64, 65 and 256
/// rows; the legacy path stays capped at 64. The const is the DEFAULT width; the refusal
/// bound of the Flash-Next prefill paths is the runtime `exl3_rows_ceiling()` (2,048 by
/// default, raisable by INFERENCE_EXL3_ROWS_MAX), which sits above it.
pub(crate) const MAX_EXL3_ROWS: usize = 256;

/// The rows window the shared prefill paths accept as a *supported* setting, with no
/// measurement variable: the ceiling `prefill_rows`, the chunk plan and the MoE block
/// plan check before an operator's configuration reaches a device. Raised from the 256
/// planning bound above once the shared prefill oracle passed at 1,024 and 2,048 rows
/// and two served 32K prompts ran both widths on the pack; the default stays 256.
/// `INFERENCE_EXL3_ROWS_MAX` still raises the window above this for measurements
/// (2,049 to 4,096).
pub(crate) const SUPPORTED_EXL3_ROWS: usize = 2048;

/// The widest rows window the shared EXL3 expert machinery may plan for this session:
/// `SUPPORTED_EXL3_ROWS` unless `INFERENCE_EXL3_ROWS_MAX` raises it further for a
/// measurement. Values above the supported ceiling are measurement widths, not
/// qualified served settings: the tiled dense-job writer covers them, but the oracle and
/// parity suites only pin up to `SUPPORTED_EXL3_ROWS`.
/// Only the Flash-Next prefill paths honour the raise (settings, the chunk plan, the
/// shared driver and its MoE block); the K-row shared expert of `family::exl3::moe`
/// still caps at `MAX_EXL3_ROWS`. Strict parse: digits only, 1 to 4096; anything else
/// refuses at start. Read once per process and cached.
pub(crate) fn exl3_rows_ceiling() -> Result<usize, String> {
    static CEILING: std::sync::OnceLock<Result<usize, String>> = std::sync::OnceLock::new();
    CEILING
        .get_or_init(|| {
            exl3_supported_ceiling_from(std::env::var("INFERENCE_EXL3_ROWS_MAX").ok().as_deref())
        })
        .clone()
}

/// [`exl3_rows_ceiling`] over an explicit raw value with the supported floor applied,
/// for tests: the parsed measurement width, never below `SUPPORTED_EXL3_ROWS`.
pub(crate) fn exl3_supported_ceiling_from(spec: Option<&str>) -> Result<usize, String> {
    Ok(exl3_rows_ceiling_from(spec)?.max(SUPPORTED_EXL3_ROWS))
}

/// [`exl3_rows_ceiling`] over an explicit raw value, for tests.
pub(crate) fn exl3_rows_ceiling_from(spec: Option<&str>) -> Result<usize, String> {
    const HARD_CEILING: usize = 4096;
    let raw = match spec {
        Some(raw) => raw,
        None => return Ok(MAX_EXL3_ROWS),
    };
    if raw.is_empty() || !raw.bytes().all(|b| b.is_ascii_digit()) {
        return Err("INFERENCE_EXL3_ROWS_MAX must be digits only".to_string());
    }
    let value: usize = raw
        .parse()
        .map_err(|_| "INFERENCE_EXL3_ROWS_MAX does not parse".to_string())?;
    if !(1..=HARD_CEILING).contains(&value) {
        return Err("INFERENCE_EXL3_ROWS_MAX must be 1 to 4096".to_string());
    }
    Ok(value.max(MAX_EXL3_ROWS))
}

/// i32 words of one job of the device table.
pub(crate) const EXL3_JOB_WORDS: usize = 4;

/// The most jobs of one routed pass over `routed` pairs and `experts` experts: one per expert
/// with rows plus one per full tile beyond the first of an expert.
pub(crate) fn exl3_max_jobs(experts: usize, routed: usize) -> usize {
    experts + routed.div_ceil(EXL3_EXPERT_TILE_ROWS)
}

/// Bytes of the dense table and the routed row lists plus table, in that order.
#[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
pub(crate) fn exl3_job_bytes(
    rows: usize,
    tile_rows: usize,
    experts: usize,
    top_k: usize,
) -> Option<(usize, usize)> {
    if tile_rows == 0 {
        return None;
    }
    let routed = rows.checked_mul(top_k)?;
    // The byte count is stricter than this bound; first make the shared job count safe.
    experts.checked_add(routed)?;
    let table_words = exl3_max_jobs(experts, routed).checked_mul(EXL3_JOB_WORDS)?;
    let perm_words = routed.checked_mul(2)?.checked_add(table_words)?;
    let dense_words = rows.div_ceil(tile_rows).checked_mul(EXL3_JOB_WORDS)?;
    Some((
        dense_words.checked_mul(size_of::<i32>())?,
        perm_words.checked_mul(size_of::<i32>())?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_rows_ceiling_defaults_stays_strict_and_never_lowers_the_default() {
        use super::{MAX_EXL3_ROWS, exl3_rows_ceiling_from};
        assert_eq!(exl3_rows_ceiling_from(None).unwrap(), MAX_EXL3_ROWS);
        assert_eq!(exl3_rows_ceiling_from(Some("2048")).unwrap(), 2048);
        // A measurement width below the qualified default never lowers it.
        assert_eq!(exl3_rows_ceiling_from(Some("100")).unwrap(), MAX_EXL3_ROWS);
        for bad in ["", "1 GiB", "-1", "0", "4097", "99999999999"] {
            assert!(exl3_rows_ceiling_from(Some(bad)).is_err(), "{bad} took");
        }
        assert_eq!(
            exl3_rows_ceiling_from(Some("x")).unwrap_err(),
            "INFERENCE_EXL3_ROWS_MAX must be digits only"
        );
    }

    #[test]
    fn the_supported_ceiling_admits_2048_without_the_variable_and_only_higher_with_it() {
        use super::{SUPPORTED_EXL3_ROWS, exl3_supported_ceiling_from};
        // An operator reaches 2,048 rows with no measurement variable at all.
        assert_eq!(
            exl3_supported_ceiling_from(None).unwrap(),
            SUPPORTED_EXL3_ROWS
        );
        assert_eq!(
            exl3_supported_ceiling_from(Some("512")).unwrap(),
            SUPPORTED_EXL3_ROWS
        );
        // Above the supported ceiling the variable still widens, for measurements.
        assert_eq!(exl3_supported_ceiling_from(Some("4096")).unwrap(), 4096);
        // The strict parse is unchanged.
        assert!(exl3_supported_ceiling_from(Some("4097")).is_err());
    }

    #[test]
    fn split_sources_keep_original_rows_and_scatter_with_local_member_ids() {
        // Physical capacity exceeds the real expert count. Empty slots must not become jobs.
        let plan = plan_pairs(&[7, 1, 8, 7, 1, 8], 12, 2).expect("plan");
        let [bank, ring] = plan.split_members(7).expect("split");
        assert_eq!(bank.counts, [0, 2, 0, 0, 0, 0, 0]);
        assert_eq!(bank.order, [0, 2]);
        assert_eq!(bank.scatter, [1, 4]);
        assert_eq!(ring.counts, [2, 2, 0, 0, 0]);
        assert_eq!(ring.order, [0, 1, 1, 2]);
        assert_eq!(ring.scatter, [0, 3, 2, 5]);
        assert!(
            bank.scatter
                .iter()
                .any(|&at| at as usize >= bank.order.len())
        );
        assert_eq!(bank.order.len() + ring.order.len(), 6);
    }

    #[test]
    fn split_sources_handle_empty_sources_and_preserve_full_tiles_and_tail() {
        for (member, boundary, ring_only) in [(2, 4, false), (6, 4, true)] {
            let plan = plan_pairs(&[member; 65], 9, 1).expect("plan");
            let [bank, ring] = plan.split_members(boundary).expect("split");
            let (full, empty) = if ring_only {
                (ring, bank)
            } else {
                (bank, ring)
            };
            assert_eq!(full.counts.iter().sum::<usize>(), 65);
            assert_eq!(full.counts[2], 65);
            assert_eq!(full.scatter, (0..65).collect::<Vec<_>>());
            assert_eq!(full.order, full.scatter);
            assert!(empty.order.is_empty());
            assert!(empty.scatter.is_empty());
        }
        assert!(plan_pairs(&[0], 1, 1).unwrap().split_members(2).is_err());
    }

    #[test]
    fn shared_prefill_job_bytes_cover_dense_tiles_and_routed_lists() {
        assert_eq!(exl3_job_bytes(64, 256, 512, 10), Some((16, 13_472)));
        assert_eq!(exl3_job_bytes(2048, 256, 512, 10), Some((128, 177_152)));
        assert_eq!(exl3_job_bytes(257, 256, 1, 1), Some((32, 2152)));
    }

    #[test]
    fn shared_prefill_job_bytes_refuse_zero_tiles_and_overflow() {
        assert_eq!(exl3_job_bytes(1, 0, 512, 10), None);
        assert_eq!(exl3_job_bytes(usize::MAX, 256, 512, 10), None);
        assert_eq!(exl3_job_bytes(1, 256, usize::MAX, 1), None);
        assert_eq!(exl3_job_bytes(usize::MAX, 1, 0, 0), None);
    }
}
