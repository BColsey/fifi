//! The neutral half of the fixed CUDA admission ledger.
//!
//! Pure arithmetic and kernel-owned constants every estimate and every
//! family's admission share: the context-search step and its wrapper over
//! the device-budget search, the KV8 staging arithmetic (a checked mirror
//! of the kernel wrapper's), the sketch type that sums a demand, and the
//! chunked-attention part charge, which equals the kernel cap. The family
//! policy and measured constants (margin, spare minimum, staging tiers,
//! runtime overhead floor, uncertainty band) stay with the families that
//! measured them.

use crate::EngError;

/// The flat per-part charge of chunked attention, equal to the kernel cap
/// `CPF_ATTN_PART_CAP` (64 MiB). A kernel-side plan defines its part charge
/// from the kernel constant and const-asserts the two agree, so a retuned
/// kernel cap cannot leave this copy behind.
pub(crate) const PREFILL_PART_CHARGE_BYTES: usize = 64 * 1024 * 1024;

/// The step of the context search, and the least context it tries.
pub(crate) const CONTEXT_SEARCH_STEP: u32 = 1024;

/// The largest context, a multiple of [`CONTEXT_SEARCH_STEP`] and below `requested`, for which
/// `fits` holds; `None` when not even one step fits (or `requested` is at most one step). `fits`
/// must be monotone (a longer context never fits where a shorter one does not) and `requested` is
/// assumed not to fit: it is never asked. Shared by the `enginesrv check` proposal and the
/// expert-budget search, so both bisect the same way.
pub(crate) fn largest_fitting_context(requested: u32, fits: &dyn Fn(u32) -> bool) -> Option<u32> {
    crate::family::device_budget::largest_fitting(requested, CONTEXT_SEARCH_STEP, fits)
}

/// A demand's verdict against a measured budget, with the band around the
/// boundary as its own case. Capability's report layer maps this onto its
/// serialized `ResourceFit` (`Borderline` becomes `Unknown` plus the
/// borderline flag); the families' entries and the estimate arms share this
/// one rule so the two never drift apart.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BandVerdict {
    Fits,
    Borderline,
    DoesNotFit,
}

/// Slack at or above the band fits; inside the band (either side) the
/// verdict is borderline - the caller's authority decides; below the
/// negative band the demand does not fit.
pub(crate) fn band_verdict(slack: i128, band: i128) -> BandVerdict {
    if slack >= band {
        BandVerdict::Fits
    } else if slack >= -band {
        BandVerdict::Borderline
    } else {
        BandVerdict::DoesNotFit
    }
}

/// Whether a proposal (a largest fitting context) is worth searching for:
/// only when the per-position charge exists and the configured context is
/// not a confident fit. Both the capability estimate and the family's
/// preflight rows call this; the gate exists once.
pub(crate) fn proposal_needed(per_position: u64, verdict: BandVerdict) -> bool {
    per_position > 0 && verdict != BandVerdict::Fits
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

/// The fixed-admission demand an estimate charges against free VRAM after the
/// weights: cache plus fixed state plus the prefix device tip (exact), the
/// staged KV8 rows (exact), the part charge (kernel-owned), and the family
/// policy terms - spare minimum, admission margin, runtime overhead floor -
/// filled by the composer.
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

#[cfg(test)]
mod tests {
    use super::{BandVerdict, band_verdict, proposal_needed};

    /// The band boundaries, one byte either side of each: slack at the
    /// band fits, slack inside the band (either side) is borderline, slack
    /// below the negative band does not fit.
    #[test]
    fn band_verdict_boundaries() {
        const BAND: i128 = 128 << 20;
        assert_eq!(band_verdict(BAND + 1, BAND), BandVerdict::Fits);
        assert_eq!(band_verdict(BAND, BAND), BandVerdict::Fits);
        assert_eq!(band_verdict(BAND - 1, BAND), BandVerdict::Borderline);
        assert_eq!(band_verdict(0, BAND), BandVerdict::Borderline);
        assert_eq!(band_verdict(-BAND, BAND), BandVerdict::Borderline);
        assert_eq!(band_verdict(-BAND - 1, BAND), BandVerdict::DoesNotFit);
    }

    /// A proposal is searched only when positions charge and the
    /// configured context is not a confident fit.
    #[test]
    fn proposal_needed_boundaries() {
        for verdict in [BandVerdict::Borderline, BandVerdict::DoesNotFit] {
            assert!(proposal_needed(1, verdict));
        }
        assert!(!proposal_needed(1, BandVerdict::Fits));
        for verdict in [BandVerdict::Borderline, BandVerdict::DoesNotFit] {
            assert!(!proposal_needed(0, verdict));
        }
    }

    use super::*;

    /// The probe SEQUENCE of `largest_fitting_context` is qualified output - a
    /// later consumer writes its result into refusal text - so the answers below
    /// are pinned per predicate, derived by hand from the loop. Do NOT "improve"
    /// a row: the lattice-floor in the midpoint makes some answers land BELOW the
    /// largest fitting step when the request is not lattice-aligned, and that is
    /// shipped behavior.
    #[test]
    fn largest_fitting_context_golden_probes() {
        // Degenerate arms: a request at or below the step, or a first step
        // that does not fit, refuse.
        assert_eq!(largest_fitting_context(1024, &|_| true), None);
        assert_eq!(largest_fitting_context(2048, &|_| false), None);
        // The request itself is never asked: even a predicate that accepts
        // everything returns the step below it.
        assert_eq!(largest_fitting_context(2048, &|_| true), Some(1024));
        // A monotone cut at 200_704 (= 196 steps exactly, on the lattice) over
        // a 262_144 request: the FIRST probe lands exactly on 131_072 (the
        // staging tier bound), and the search converges onto the cut.
        let cut = |context: u32| context <= 200_704;
        assert_eq!(largest_fitting_context(262_144, &cut), Some(200_704));
        // The Q8 staging cliff as a predicate: 131_072 alone refuses, every
        // other step up to 200_704 admits. Once the first probe (131_072)
        // refuses, no row above it is ever probed: the answer is 130_048 even
        // though 200_704 would admit. Non-monotone by design; keep it.
        let cliff = |context: u32| context != 131_072 && context <= 200_704;
        assert_eq!(largest_fitting_context(262_144, &cliff), Some(130_048));
        // A cut OFF the lattice (below 100_000, itself off-lattice): the
        // lattice-floored midpoint strands the low pointer at 98_304, three
        // steps under the largest admitting step 99_328. Shipped behavior.
        let odd_cut = |context: u32| context < 100_000;
        assert_eq!(largest_fitting_context(100_000, &odd_cut), Some(98_304));
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
}
