//! The device-budget arithmetic shared by every family preflight.
//!
//! The families each know WHICH charges a load pins and HOW MUCH the search
//! step is; the ledger rules — how a set of labelled charges folds into a
//! checked total, and how a slack compares against an admission band — are
//! the same everywhere, so they live here once. Nothing in this module
//! touches a device, a feature gate, or an error string: the callers own
//! their refusal text and their gate.
//!
//! * [`largest_fitting`] is the bisecting context search (a monotone
//!   `fits`): the paper admission and its `enginesrv check` proposal
//!   search both delegate to it.
//! * [`largest_fitting_linear`] is the stop-at-first-false scan for
//!   predicates that need not be monotone (an admitting later step must not
//!   hide a refusing earlier one): the resident-context receipt uses
//!   it.
//! * [`DeviceReserve`] is the ordered, labelled charge list plus the
//!   admission band, with a checked total.

/// The KV side of a family preflight: how many positions the caller configured and
/// what one position costs. A plain pair of the caller's measurements (8-bit KV on
/// every backend); the tiered plans and other family loaders fill it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvPlan {
    /// Context positions the caller configured (positions, not bytes).
    pub positions: u64,
    /// KV bytes one position costs (8-bit KV on every backend; the number is
    /// the caller's measurement of the model's attention layers).
    pub bytes_per_position: u64,
}

/// The largest context, a multiple of `step` and below `requested`, for which
/// `fits` holds; `None` when not even one step fits (or `requested` is at
/// most one step). `fits` must be monotone (a longer context never fits
/// where a shorter one does not) and `requested` is assumed not to fit: it is
/// never asked.
pub(crate) fn largest_fitting(
    requested: u32,
    step: u32,
    fits: &dyn Fn(u32) -> bool,
) -> Option<u32> {
    if requested <= step || !fits(step) {
        return None;
    }
    let mut low = step;
    let mut high = requested;
    while high - low > step {
        let mid = low + (high - low) / 2 / step * step;
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

/// The largest context, a multiple of `step` and at most `limit`, for which
/// `admits` holds on EVERY step up to it: the scan stops at the first
/// refusal, so a non-monotone predicate cannot report an admitting step
/// beyond a refusing one. `None` when the first step does not admit.
#[allow(
    dead_code,
    reason = "the resident-context scan in capability::preflight reaches it under native-cuda; a plain host build compiles that caller away"
)]
pub(crate) fn largest_fitting_linear(
    limit: u32,
    step: u32,
    admits: &dyn Fn(u32) -> bool,
) -> Option<u32> {
    let mut largest = None;
    let mut context = step;
    while context <= limit {
        if admits(context) {
            largest = Some(context);
        } else {
            break;
        }
        context += step;
    }
    largest
}

/// An ordered, labelled charge list with the admission band. The labels are
/// the receipt's own words, in push order; the total folds with
/// [`u64::checked_add`] so an untrusted composition reports instead of
/// wrapping. The band is explicit, carried rather than global, so a future
/// ledger can widen it without touching the others.
#[derive(Debug, Clone)]
pub(crate) struct DeviceReserve {
    #[cfg_attr(
        not(test),
        allow(dead_code, reason = "no start reads the band yet; tests do")
    )]
    band_bytes: u64,
    charges: Vec<(&'static str, u64)>,
}

impl DeviceReserve {
    /// An empty reserve admitting against `band_bytes`.
    pub(crate) fn new(band_bytes: u64) -> Self {
        Self {
            band_bytes,
            charges: Vec::new(),
        }
    }

    /// Appends one labelled charge, in receipt order.
    pub(crate) fn push(mut self, label: &'static str, bytes: u64) -> Self {
        self.charges.push((label, bytes));
        self
    }

    /// The band the reserve's slack is judged against.
    #[cfg_attr(
        not(test),
        allow(dead_code, reason = "no start reads the band yet; tests do")
    )]
    pub(crate) fn band_bytes(&self) -> u64 {
        self.band_bytes
    }

    /// The charge pairs, in push order.
    pub(crate) fn charges(&self) -> &[(&'static str, u64)] {
        &self.charges
    }

    /// The folded charge total; `None` on u64 overflow.
    #[cfg_attr(
        not(test),
        allow(dead_code, reason = "read by the Apple Silicon budget only")
    )]
    pub(crate) fn checked_total(&self) -> Option<u64> {
        self.charges
            .iter()
            .try_fold(0u64, |acc, &(_, bytes)| acc.checked_add(bytes))
    }
}

#[cfg(test)]
mod tests {
    use super::{DeviceReserve, largest_fitting, largest_fitting_linear};

    /// The two searches agree on a monotone predicate, and the linear scan
    /// keeps its stop-at-first-false rule on a gap where the bisect does not
    /// (the pinned resident-receipt case).
    #[test]
    fn linear_scan_stops_at_first_refusal() {
        let cut = |context: u32| context <= 5120;
        assert_eq!(largest_fitting(8192, 1024, &cut), Some(5120));
        assert_eq!(largest_fitting_linear(8192, 1024, &cut), Some(5120));
        // A refusal at 3072 hiding an admitting range above it: the linear
        // scan stops at the refusal (2048) while the bisect jumps over it
        // (4096) — the documented reason the linear scan exists.
        let gap = |context: u32| context <= 2048 || context >= 4096;
        assert_eq!(largest_fitting_linear(8192, 1024, &gap), Some(2048));
        assert_eq!(largest_fitting(8192, 1024, &gap), Some(7168));
    }

    /// A caller adopted `largest_fitting` at step 1 in place of a private bisection.
    /// This oracle runs both against the same predicates — pinned cut shapes plus
    /// off-lattice cuts — comparing both the results and the exact list of contexts
    /// each loop asks, so a different midpoint rule cannot pass.
    #[test]
    fn largest_fitting_matches_the_step_one_bisection_oracle() {
        let legacy =
            |requested: u32, admits: &dyn Fn(u32) -> bool, asked: &mut Vec<u32>| -> Option<u32> {
                let probe = |context: u32, asked: &mut Vec<u32>| {
                    asked.push(context);
                    admits(context)
                };
                if !probe(1, asked) {
                    return None;
                }
                let (mut fits, mut over) = (1, requested);
                while over - fits > 1 {
                    let middle = fits + (over - fits) / 2;
                    if probe(middle, asked) {
                        fits = middle;
                    } else {
                        over = middle;
                    }
                }
                Some(fits)
            };
        let shared = |admits: &dyn Fn(u32) -> bool, asked: &mut Vec<u32>| -> Option<u32> {
            let recorder = std::cell::RefCell::new(Vec::<u32>::new());
            let result = largest_fitting(262_145, 1, &|context| {
                recorder.borrow_mut().push(context);
                admits(context)
            });
            asked.extend(recorder.into_inner());
            result
        };
        let compare = |admits: &dyn Fn(u32) -> bool| {
            let mut old_asked = Vec::new();
            let mut new_asked = Vec::new();
            let old = legacy(262_145, admits, &mut old_asked);
            let new = shared(admits, &mut new_asked);
            assert_eq!(new, old);
            assert_eq!(new_asked, old_asked);
        };
        // Pinned on-lattice cut, three off-lattice cuts, and the near-top arm.
        for cut in [30_720u32, 50_000, 86_013, 100_001, 262_143] {
            let admits = |context: u32| context <= cut;
            compare(&admits);
        }
        // Cuts at the top and above it: the predicate stays true everywhere
        // probed, and both loops return the last in-range context.
        for cut in [262_144u32, 262_145] {
            let admits = |context: u32| context <= cut;
            assert_eq!(largest_fitting(262_145, 1, &admits), Some(262_144));
            compare(&admits);
        }
        // The gap and refusing arms.
        let gap = |context: u32| !(20_480..=30_720).contains(&context);
        compare(&gap);
        let refuses_all = |_: u32| false;
        compare(&refuses_all);
    }

    /// The reserve folds in push order, saturates to `None` on overflow, and
    /// reports the exact labels back.
    #[test]
    fn device_reserve_folds_and_reports() {
        let reserve = DeviceReserve::new(128 << 20)
            .push("kv cache", 1 << 20)
            .push("work reserve", 2 << 20);
        assert_eq!(reserve.checked_total(), Some(3 << 20));
        assert_eq!(
            reserve.charges(),
            &[("kv cache", 1 << 20), ("work reserve", 2 << 20)]
        );
        assert_eq!(reserve.band_bytes(), 128 << 20);
        assert_eq!(
            DeviceReserve::new(0)
                .push("saturate", u64::MAX)
                .push("one more", 1)
                .checked_total(),
            None
        );
    }
}
