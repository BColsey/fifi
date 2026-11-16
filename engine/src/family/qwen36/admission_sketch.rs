//! The paper version of the fixed CUDA admission ledger.
//!
//! `enginesrv check` cannot run the real CUDA admission: that path measures
//! free VRAM in the serving process *after* the weights are resident and
//! charges its budget from the device. This module is the pure arithmetic of
//! the same ledger, shared by both sides so they cannot drift apart:
//!
//! * the policy constants and the staging-row rule here are canonical —
//!   `cuda::prefill_driver` and `cuda::prefill_plan` alias them; the neutral
//!   half (search step and wrapper, staging arithmetic, sketch type, the
//!   kernel-tied part charge) lives in `family::admission_sketch`, and
//!   `prefill_plan` const-asserts the part charge equals the kernel cap;
//! * the terms a paper estimate cannot know exactly (the prefill chunk arena,
//!   the decode arena, eager module workspace, and the serving process's own
//!   CUDA context beyond a fresh probe) collapse into
//!   [`RUNTIME_OVERHEAD_FLOOR_BYTES`], a measured floor with provenance, and
//!   the boundary is reported through [`ADMISSION_UNCERTAINTY_BAND_BYTES`]
//!   instead of a precise claim;
//! * load-time admission stays the authority: within the band an estimate is
//!   borderline, not a verdict either way.

// The neutral search step, sketch type, and kernel-tied part charge live in the
// family home; private use only, so the export surface stays alias-free.
use crate::family::admission_sketch::{AdmissionSketch, PREFILL_PART_CHARGE_BYTES};

/// The headroom the fixed CUDA admission keeps below measured free VRAM
/// before any charge (`budget = free - margin`). Canonical value: the runtime
/// path in `cuda::prefill_driver` aliases this constant.
pub(crate) const FIXED_ADMISSION_MARGIN_BYTES: usize = 256 << 20;

/// The minimum spare the fixed CUDA admission leaves after every charge
/// (`SPARE_MIN` in `cuda::prefill_plan`). Canonical value.
pub(crate) const PREFILL_SPARE_MIN_BYTES: usize = 64 << 20;

/// The full-stage tier bound: the maximum rows one default stage carries and
/// the boundary above which [`fixed_staging_rows`] shrinks to the 32K tail —
/// a staging tier bound, not a context cap. Canonical value.
pub(crate) const FULL_STAGE_TIER_ROWS: usize = 131_072;

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
    /// c0 goldens (P4): the charge terms of `admission_demand` and its checked
    /// total as literal cells. `state_bytes` folds kv + fixed + prefix tip.
    #[test]
    fn admission_demand_golden_cells() {
        let sketch = admission_demand(1 << 20, 2 << 20, 3 << 20, 4 << 20);
        assert_eq!(sketch.state_bytes, 6 << 20);
        assert_eq!(sketch.staging_bytes, 4 << 20);
        assert_eq!(sketch.part_charge_bytes, PREFILL_PART_CHARGE_BYTES as u64);
        assert_eq!(sketch.spare_min_bytes, PREFILL_SPARE_MIN_BYTES as u64);
        assert_eq!(sketch.margin_bytes, FIXED_ADMISSION_MARGIN_BYTES as u64);
        assert_eq!(sketch.overhead_floor_bytes, RUNTIME_OVERHEAD_FLOOR_BYTES);
        let total = sketch.total_after_weights().expect("fits u64");
        assert_eq!(
            total,
            (6u64 << 20)
                + (4u64 << 20)
                + PREFILL_PART_CHARGE_BYTES as u64
                + PREFILL_SPARE_MIN_BYTES as u64
                + FIXED_ADMISSION_MARGIN_BYTES as u64
                + RUNTIME_OVERHEAD_FLOOR_BYTES
        );
        // The composition saturates its state instead of panicking, and the
        // checked total then REPORTS the overflow rather than wrapping.
        let huge = admission_demand(u64::MAX, u64::MAX, u64::MAX, 0);
        assert_eq!(huge.state_bytes, u64::MAX);
        assert!(huge.total_after_weights().is_err());
    }

    use super::*;

    #[test]
    fn staging_rows_follow_the_tier_rule() {
        assert_eq!(fixed_staging_rows(32_768), 32_768);
        assert_eq!(fixed_staging_rows(131_072), 131_072);
        assert_eq!(fixed_staging_rows(131_073), 32_768);
        assert_eq!(fixed_staging_rows(262_144), 32_768);
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
