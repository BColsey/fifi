//! The store-tier ladder: which pinned-store tier one host budget can
//! carry, chosen family-neutrally so every bank family (Qwen3.6, Flint,
//! Bolt, Flash-Next) downshifts through the same rungs.
//!
//! The ladder, in preference order:
//! 1. [`TierChoice::Inclusive`] — every expert of every layer pinned.
//! 2. [`TierChoice::Exclusive`] at the largest spare-place count that
//!    fits, down to `min_spare_places`.
//! 3. [`TierChoice::Disk`] — no pinned store; records read through the
//!    staging ring on every miss.
//!
//! Below the disk tier's minimum (the ring staging plus the fixed
//! overhead) the choice refuses, named — the caller surfaces the two
//! numbers that disagreed.

// Ungated host code, but until a second family's start calls the ladder only tests do; the
// flash-next start reaches it through `HostPinPlan::choose_store_tier` on the CUDA path.
#![cfg_attr(not(test), allow(dead_code))]

use crate::EngError;

/// The store tier a host budget carried, or the refusal naming the
/// minimum it could not.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TierChoice {
    /// Every expert pinned: eviction has no write-back.
    Inclusive,
    /// Only the cold experts plus `spare_places` spare places per layer.
    Exclusive { spare_places: usize },
    /// No pinned store: every miss reads through the staging ring.
    Disk,
}

impl TierChoice {
    /// The log line's name of the tier and its expected speed class.
    pub(crate) fn describe(&self) -> (&'static str, &'static str) {
        match self {
            Self::Inclusive => ("inclusive store", "no write-back, fastest"),
            Self::Exclusive { .. } => ("exclusive store", "write-back on eviction"),
            Self::Disk => ("disk tier", "staging-ring reads on every miss, slowest"),
        }
    }
}

/// The family-neutral inputs of the ladder. One record's bytes, the bank
/// geometry and the ONE host budget (the MemAvailable figure clamped by
/// the enclosing cgroup — the landed one-budget snapshot); the family
/// supplies its own fixed overhead (stream scratch, staging, heap,
/// margin, OS floor) and page-cache floor, so the ladder's arithmetic is
/// exactly the family's plan arithmetic.
pub(crate) struct TierBudget {
    /// Bytes of one expert record.
    pub(crate) record_bytes: u64,
    /// Layer count.
    pub(crate) layers: usize,
    /// Experts per layer.
    pub(crate) experts: usize,
    /// Device slots per layer (the VRAM bank; cold experts per layer are
    /// `experts - slots`).
    pub(crate) slots: usize,
    /// The largest spare places per layer the exclusive tier may pin.
    pub(crate) max_spare_places: usize,
    /// The smallest viable spare places per layer (the policy's minimum).
    pub(crate) min_spare_places: usize,
    /// Host bytes every tier adds beside the store (scratch, staging,
    /// heap, margin and the OS floor, summed).
    pub(crate) overhead_bytes: u64,
    /// The page-cache floor the leftover must keep.
    pub(crate) cache_floor: u64,
    /// The disk tier's own minimum beside the overhead: the staging
    /// ring's backing bytes.
    pub(crate) ring_bytes: u64,
    /// The one host budget, bytes.
    pub(crate) available: u64,
}

impl TierBudget {
    /// The pinned bytes of the exclusive tier at `spare_places` spare
    /// places per layer: the cold experts plus the spares.
    fn exclusive_store(&self, spare_places: usize) -> Option<u64> {
        (self.layers as u64)
            .checked_mul((self.experts - self.slots + spare_places) as u64)
            .and_then(|places| places.checked_mul(self.record_bytes))
    }

    /// Whether a store of `store` bytes fits: the store, the fixed
    /// overhead and the page-cache floor together must stay inside the
    /// budget.
    fn fits(&self, store: u64) -> bool {
        store
            .checked_add(self.overhead_bytes)
            .and_then(|need| need.checked_add(self.cache_floor))
            .is_some_and(|need| need <= self.available)
    }
}

/// Chooses the tier: inclusive while it fits, then the exclusive tier at
/// the largest fitting spare-place count, then the disk tier, and a named
/// refusal below the minimum.
pub(crate) fn choose_tier(budget: &TierBudget) -> Result<TierChoice, EngError> {
    let overflow = || tier_error("the tier byte counts overflow".into());
    let inclusive_store = (budget.layers as u64)
        .checked_mul(budget.experts as u64)
        .and_then(|places| places.checked_mul(budget.record_bytes))
        .ok_or_else(overflow)?;
    if budget.fits(inclusive_store) {
        return Ok(TierChoice::Inclusive);
    }
    // The largest fitting spare count: the ladder never pins more than
    // the budget carries, and never below the policy's minimum.
    if budget.max_spare_places >= budget.min_spare_places {
        for spare in (budget.min_spare_places..=budget.max_spare_places).rev() {
            let store = budget.exclusive_store(spare).ok_or_else(overflow)?;
            if budget.fits(store) {
                return Ok(TierChoice::Exclusive {
                    spare_places: spare,
                });
            }
        }
    }
    // The disk tier: the ring's own bytes plus the fixed overhead, and
    // the page-cache floor still applies to what is left.
    if budget
        .ring_bytes
        .checked_add(budget.overhead_bytes)
        .and_then(|need| need.checked_add(budget.cache_floor))
        .is_some_and(|need| need <= budget.available)
    {
        return Ok(TierChoice::Disk);
    }
    Err(tier_error(format!(
        "the host budget of {} bytes cannot carry the minimum tier: the staging ring needs {} \
         bytes plus {} of fixed overhead and a {} byte page-cache floor",
        budget.available, budget.ring_bytes, budget.overhead_bytes, budget.cache_floor
    )))
}

fn tier_error(detail: String) -> EngError {
    EngError::MemoryPlan(format!("bank tier choice: {detail}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn budget(available: u64) -> TierBudget {
        // One layer of 10 experts, 4 device slots, 100-byte records:
        // inclusive 1000, exclusive at one spare 700, at three 900.
        TierBudget {
            record_bytes: 100,
            layers: 1,
            experts: 10,
            slots: 4,
            max_spare_places: 3,
            min_spare_places: 1,
            overhead_bytes: 50,
            cache_floor: 10,
            ring_bytes: 250,
            available,
        }
    }

    fn choice(available: u64) -> Result<TierChoice, EngError> {
        choose_tier(&budget(available))
    }

    #[test]
    fn the_ladder_picks_each_tier_at_its_boundary() {
        // Inclusive needs 1000 + 50 + 10 = 1060.
        assert_eq!(choice(1_060), Ok(TierChoice::Inclusive));
        assert_eq!(
            choice(1_059),
            Ok(TierChoice::Exclusive { spare_places: 3 }),
            "just under inclusive takes the largest fitting spare count (900 + 60 = 960)"
        );
        assert_eq!(
            choice(959),
            Ok(TierChoice::Exclusive { spare_places: 2 }),
            "800 + 60 = 860 fits, the next spare count does not"
        );
        assert_eq!(
            choice(760),
            Ok(TierChoice::Exclusive { spare_places: 1 }),
            "700 + 60 = 760 is the minimum spare tier's exact fit"
        );
        assert_eq!(
            choice(759),
            Ok(TierChoice::Disk),
            "below one spare place the ring carries it: 250 + 50 + 10 = 310"
        );
    }

    #[test]
    fn below_the_minimum_the_choice_refuses_by_name() {
        // 250 + 50 + 10 = 310 is the disk tier's minimum.
        let error = choice(309).unwrap_err();
        let text = error.describe();
        assert!(text.contains("staging ring"), "{text}");
        assert!(text.contains("309"), "{text}");
        assert!(text.contains("310") || text.contains("250"), "{text}");
    }

    #[test]
    fn a_shrunk_range_never_pins_below_the_policy_minimum() {
        // max == min: the ladder has no spare counts to shrink through,
        // so the choice falls straight to the ring when exclusive at the
        // only spare count does not fit.
        let mut tight = budget(759);
        tight.max_spare_places = 1;
        assert_eq!(
            choose_tier(&tight),
            Ok(TierChoice::Disk),
            "one spare place (760) does not fit 759"
        );
        let mut impossible = budget(759);
        impossible.min_spare_places = 2;
        assert_eq!(
            choose_tier(&impossible),
            Ok(TierChoice::Disk),
            "a minimum above max skips the exclusive rung entirely"
        );
    }

    #[test]
    fn the_overflow_refuses_instead_of_wrapping() {
        let mut huge = budget(u64::MAX);
        huge.record_bytes = u64::MAX;
        let error = choose_tier(&huge).unwrap_err();
        assert!(error.describe().contains("overflow"), "{error:?}");
    }
}
