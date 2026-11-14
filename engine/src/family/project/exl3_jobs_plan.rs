//! The size rules of an EXL3 routed job table: pure arithmetic, shared by the host-side scratch
//! planners (which size the sort lists and the job table) and the device-side job-table
//! machinery in [`crate::family::project::cuda::exl3_jobs`].

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
