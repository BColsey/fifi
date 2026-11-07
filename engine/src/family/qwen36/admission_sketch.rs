//! The paper version of the fixed CUDA admission ledger.
//!
//! `enginesrv check` cannot run the real CUDA admission: that path measures
//! free VRAM in the serving process *after* the weights are resident and
//! charges its budget from the device. This module is the pure arithmetic of
//! the same ledger, shared by both sides so they cannot drift apart:
//!
//! * the constants and the staging-row rule here are canonical —
//!   `cuda::prefill_driver` and `cuda::prefill_plan` alias them, and
//!   `prefill_plan` const-asserts the part charge equals the kernel cap;
//! * the terms a paper estimate cannot know exactly (the prefill chunk arena,
//!   the decode arena, eager module workspace, and the serving process's own
//!   CUDA context beyond a fresh probe) collapse into
//!   [`RUNTIME_OVERHEAD_FLOOR_BYTES`], a measured floor with provenance, and
//!   the boundary is reported through [`ADMISSION_UNCERTAINTY_BAND_BYTES`]
//!   instead of a precise claim;
//! * load-time admission stays the authority: within the band an estimate is
//!   borderline, not a verdict either way.

use crate::EngError;

/// The headroom the fixed CUDA admission keeps below measured free VRAM
/// before any charge (`budget = free - margin`). Canonical value: the runtime
/// path in `cuda::prefill_driver` aliases this constant.
pub(crate) const FIXED_ADMISSION_MARGIN_BYTES: usize = 256 << 20;

/// The minimum spare the fixed CUDA admission leaves after every charge
/// (`SPARE_MIN` in `cuda::prefill_plan`). Canonical value.
pub(crate) const PREFILL_SPARE_MIN_BYTES: usize = 64 << 20;

/// The flat per-part charge of chunked attention, equal to the kernel cap
/// `CPF_ATTN_PART_CAP` (64 MiB). `cuda::prefill_plan` defines `PART_CHARGE`
/// from the kernel constant and const-asserts the two agree, so a retuned
/// kernel cap cannot leave this copy behind.
pub(crate) const PREFILL_PART_CHARGE_BYTES: usize = 64 * 1024 * 1024;

/// The full-stage tier bound: the maximum rows one default stage carries and
/// the boundary above which [`fixed_staging_rows`] shrinks to the 32K tail —
/// a staging tier bound, not a context cap. Canonical value.
pub(crate) const FULL_STAGE_TIER_ROWS: usize = 131_072;

/// The step of the context search, and the least context it tries.
pub(crate) const CONTEXT_SEARCH_STEP: u32 = 1024;

/// The largest context, a multiple of [`CONTEXT_SEARCH_STEP`] and below `requested`, for which
/// `fits` holds; `None` when not even one step fits (or `requested` is at most one step). `fits`
/// must be monotone (a longer context never fits where a shorter one does not) and `requested` is
/// assumed not to fit: it is never asked. Shared by the `enginesrv check` proposal and the Flash-Next
/// expert budget, so both bisect the same way.
pub(crate) fn largest_fitting_context(requested: u32, fits: &dyn Fn(u32) -> bool) -> Option<u32> {
    const STEP: u32 = CONTEXT_SEARCH_STEP;
    if requested <= STEP || !fits(STEP) {
        return None;
    }
    let mut low = STEP;
    let mut high = requested;
    while high - low > STEP {
        let mid = low + (high - low) / 2 / STEP * STEP;
        if mid <= low {
            break;
        }
        if fits(mid) {
            low = mid;
        } else {
            high = mid;
        }
    }
    Some(low)
}

/// The staged KV8 rows a context of `context_limit` positions pins at boot:
/// the whole context below the tier, the 32K tail above it. Matches
/// `cuda::prefill_driver::fixed_staging_rows` by construction (it delegates
/// here). Pinned by the driver tests: 262144 -> 32768, 131072 -> 131072.
pub(crate) fn fixed_staging_rows(context_limit: usize) -> usize {
    if context_limit <= FULL_STAGE_TIER_ROWS {
        context_limit
    } else {
        32_768
    }
}

/// The bytes of the time-major FP16 staging cache for `capacity` rows. Same
/// checked arithmetic the kernel wrapper runs; canonical here so an estimate
/// and an admission compute the same number.
pub(crate) fn kv8_staging_bytes(
    capacity: usize,
    n_kv_head: usize,
    head_dim: usize,
) -> Result<usize, EngError> {
    capacity
        .checked_mul(n_kv_head)
        .and_then(|n| n.checked_mul(head_dim))
        .and_then(|n| n.checked_mul(4))
        .filter(|&bytes| bytes != 0)
        .ok_or_else(|| EngError::BadRequest("kv8 staging size is zero or overflows".into()))
}

/// What an estimate adds on top of the exact terms because it runs outside the
/// serving process: the prefill chunk arena (~58-61 MiB at the 512/1024-row
/// tiers), the decode arena (~1.1 MiB), eager module workspace, and the
/// serving process's own CUDA context beyond a fresh probe. Measured on the
/// 24 GiB RTX 3090 host of the 2026-11-03 qualification: a server holding the
/// 19.0 GiB Q4_K_M checkpoint showed ~275 MiB less free-after-weights than a
/// fresh probe-context process, plus ~555 MiB against `nvidia-smi` free
/// (n=1). Charged as one floor, never as a precise term.
pub(crate) const RUNTIME_OVERHEAD_FLOOR_BYTES: u64 = 384 << 20;

/// The measurement gap an estimate cannot close: probe free VRAM versus the
/// serving process's own post-load reading varies by a few hundred MiB with
/// driver state and baseline residency. Slack within this band of the budget
/// boundary is reported as borderline; load-time admission decides.
pub(crate) const ADMISSION_UNCERTAINTY_BAND_BYTES: u64 = 128 << 20;

/// The fixed-admission demand an estimate charges against free VRAM after the
/// weights: cache plus fixed state plus the prefix device tip (exact), the
/// staged KV8 rows (exact), the part charge, spare minimum and admission
/// margin (canonical constants), and the runtime overhead floor (measured).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct AdmissionSketch {
    pub state_bytes: u64,
    pub staging_bytes: u64,
    pub part_charge_bytes: u64,
    pub spare_min_bytes: u64,
    pub margin_bytes: u64,
    pub overhead_floor_bytes: u64,
}

impl AdmissionSketch {
    /// The total charged after the weights are resident; checked because the
    /// caller composes untrusted directory sizes into `state_bytes`.
    pub(crate) fn total_after_weights(&self) -> Result<u64, EngError> {
        let mut sum = self.state_bytes;
        for term in [
            self.staging_bytes,
            self.part_charge_bytes,
            self.spare_min_bytes,
            self.margin_bytes,
            self.overhead_floor_bytes,
        ] {
            sum = sum
                .checked_add(term)
                .ok_or_else(|| EngError::MemoryPlan("admission sketch overflows u64".into()))?;
        }
        Ok(sum)
    }
}

/// Compose the sketch from the exact state terms. `kv_bytes` is the whole
/// cache at the requested context; `fixed_state_bytes` the model's fixed
/// DeltaNet state; `prefix_tip_bytes` the device-tip prefix copy the CUDA arm
/// charges on top (the same `delta_state_bytes` figure when that arm is the
/// default); `staging_bytes` zero unless the KV cache is Q8.
pub(crate) fn admission_demand(
    kv_bytes: u64,
    fixed_state_bytes: u64,
    prefix_tip_bytes: u64,
    staging_bytes: u64,
) -> AdmissionSketch {
    AdmissionSketch {
        state_bytes: kv_bytes
            .checked_add(fixed_state_bytes)
            .and_then(|sum| sum.checked_add(prefix_tip_bytes))
            .unwrap_or(u64::MAX),
        staging_bytes,
        part_charge_bytes: PREFILL_PART_CHARGE_BYTES as u64,
        spare_min_bytes: PREFILL_SPARE_MIN_BYTES as u64,
        margin_bytes: FIXED_ADMISSION_MARGIN_BYTES as u64,
        overhead_floor_bytes: RUNTIME_OVERHEAD_FLOOR_BYTES,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn staging_rows_follow_the_tier_rule() {
        assert_eq!(fixed_staging_rows(32_768), 32_768);
        assert_eq!(fixed_staging_rows(131_072), 131_072);
        assert_eq!(fixed_staging_rows(131_073), 32_768);
        assert_eq!(fixed_staging_rows(262_144), 32_768);
    }

    #[test]
    fn staging_bytes_match_the_serve_error_figure() {
        // The 2026-11-03 qualification refusal: "67108864 staging bytes".
        assert_eq!(kv8_staging_bytes(32_768, 4, 128).unwrap(), 67_108_864);
        assert!(kv8_staging_bytes(0, 4, 128).is_err());
    }

    #[test]
    fn the_context_search_finds_the_exact_boundary_of_a_monotone_fit() {
        let step = CONTEXT_SEARCH_STEP;
        // The boundary between multiples: the largest multiple of the step that fits.
        let found = largest_fitting_context(262_144, &|c| c <= 90_000).unwrap();
        assert_eq!(found, 87 * step);
        // The boundary on a multiple, and a request that is not a multiple.
        assert_eq!(
            largest_fitting_context(262_144, &|c| c <= 64 * step),
            Some(64 * step)
        );
        assert_eq!(
            largest_fitting_context(100_000, &|c| c <= 50_000),
            Some(48 * step)
        );
        // For a set of boundaries the result fits and the next step does not.
        for limit in [1_024, 1_500, 2_048, 5_000, 131_072, 200_000, 262_143] {
            let found = largest_fitting_context(262_144, &|c| c <= limit).unwrap();
            assert_eq!(found % step, 0, "limit {limit}");
            assert!(
                found <= limit && limit < found + step,
                "limit {limit}: {found}"
            );
        }
    }

    #[test]
    fn the_context_search_reports_nothing_fits_and_never_asks_the_request() {
        let step = CONTEXT_SEARCH_STEP;
        assert_eq!(largest_fitting_context(262_144, &|_| false), None);
        // Even the least context does not fit.
        assert_eq!(largest_fitting_context(262_144, &|c| c < step), None);
        // Only the least context fits.
        assert_eq!(
            largest_fitting_context(262_144, &|c| c < 2 * step),
            Some(step)
        );
        // A request of one step or less has no room to search, whatever fits.
        assert_eq!(largest_fitting_context(step, &|_| true), None);
        assert_eq!(largest_fitting_context(100, &|_| true), None);
        // Everything fits: the answer is one step under the request, because the request is assumed
        // not to fit and is never asked.
        let asked = std::cell::RefCell::new(Vec::new());
        let found = largest_fitting_context(262_144, &|c| {
            asked.borrow_mut().push(c);
            true
        });
        assert_eq!(found, Some(262_144 - step));
        assert!(!asked.borrow().contains(&262_144));
    }

    #[test]
    fn the_sketch_sums_its_terms_checked() {
        let sketch = admission_demand(1_000, 65_863_680, 65_863_680, 67_108_864);
        let total = sketch.total_after_weights().unwrap();
        assert_eq!(
            total,
            1_000
                + 65_863_680
                + 65_863_680
                + 67_108_864
                + PREFILL_PART_CHARGE_BYTES as u64
                + PREFILL_SPARE_MIN_BYTES as u64
                + FIXED_ADMISSION_MARGIN_BYTES as u64
                + RUNTIME_OVERHEAD_FLOOR_BYTES
        );
        assert!(
            admission_demand(u64::MAX, 1, 0, 0)
                .total_after_weights()
                .is_err()
        );
    }
}
