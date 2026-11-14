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

/// The most rows of one job of a routed expert: the narrowest CTA (64 rows). An expert holds a few
/// dozen rows of a chunk, and a wider CTA would decode no fewer weights and run mostly idle warps.
pub(crate) const EXL3_EXPERT_TILE_ROWS: usize = 64;

/// The widest rows window the shared EXL3 expert machinery plans for: `shared_rows`
/// (`family::exl3::moe`) and the Flash-Next chunk plan (`MAX_STEP_ROWS`) refuse wider
/// windows, and the chunk scratch is sized at this width. A kernel cap it is not — the dense
/// job row cap is `jobs_row_cap` (256, exl3.cu) and the shared down groups split at 8 — this
/// is the planning bound both families agreed on.
pub(crate) const MAX_EXL3_ROWS: usize = 64;

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
