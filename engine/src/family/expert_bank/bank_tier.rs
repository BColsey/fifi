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
//!
//! A caller that also knows the card passes a [`CardBudget`]: when the
//! whole load fits the free device bytes the ladder answers exactly as it
//! does without one; when it does not, the ladder answers
//! [`TierChoice::CardTiered`] (the routed experts stay on the host, the
//! banks take what the card has left); when the host cannot hold the
//! experts either, [`TierChoice::CardDisk`] (the experts are read from the
//! file on a miss); it refuses, naming the side that cannot carry its
//! share, only when the card cannot hold the minimum banks or the host
//! cannot carry its fixed overhead.

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
    /// The card cannot carry the whole load: the routed experts stay on the
    /// host (`host_bytes` of them) and each layer keeps at most `slots`
    /// experts on the card.
    CardTiered { slots: usize, host_bytes: u64 },
    /// The card cannot carry the whole load and the host cannot hold the experts either: the
    /// experts stay in the file, each layer keeps at most `slots` experts on the card, and a
    /// miss reads the expert through the page cache from disk. Slower than [`Self::CardTiered`],
    /// never a refusal while the host carries its fixed overhead.
    CardDisk { slots: usize },
}

/// The pooled read depth of the staging ring: how many expert records are in
/// flight at once. Every admission that charges the ring a placeholder reads
/// this one number — the context budget's first pass and the start's
/// two-phase choice — so the two can never charge different bytes for one
/// pack (the two-frames defect this ladder exists to prevent, in small).
pub(crate) const RUNG_RING_SLOTS: usize = 4;

/// The ring bytes every admission charges before the real backing is read:
/// the pooled depth's records in flight. The real figure (about four times
/// the span plus a sector) is read only once the tier already points at the
/// disk rung.
pub(crate) fn rung_ring_placeholder_bytes(record_bytes: usize) -> u64 {
    RUNG_RING_SLOTS as u64 * record_bytes as u64
}

impl TierChoice {
    /// The log line's name of the tier and its expected speed class.
    pub(crate) fn describe(&self) -> (&'static str, &'static str) {
        match self {
            Self::Inclusive => ("inclusive store", "no write-back, fastest"),
            Self::Exclusive { .. } => ("exclusive store", "write-back on eviction"),
            Self::Disk => (
                "disk tier",
                "no pinned store: every miss read through the pooled ring, about 0.12 s per \
                 decode step fully cold (span bench, pooled depth 4, DONTNEED-evicted cells \
                 of the real pack); the rung's read mode — O_DIRECT or buffered — is decided \
                 once at start and stated on the tier line; prefill rides the union-admit path",
            ),
            Self::CardTiered { .. } => (
                "card-tiered experts",
                "experts on the host, the hottest in per-layer slots on the card",
            ),
            Self::CardDisk { .. } => (
                "card-tiered experts streamed from disk",
                "experts read from the file through the page cache on a miss, the hottest in \
                 per-layer slots on the card",
            ),
        }
    }
}

/// The card's side of one load: what the whole load needs on the device and
/// how much of it the routed experts are.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CardBudget {
    /// Device bytes free now.
    pub(crate) free: u64,
    /// Bytes the whole resident load needs on the card (tensors, KV cache,
    /// work and headroom): the figure the device admission compares to
    /// `free`.
    pub(crate) demand: u64,
    /// The part of `demand` that is routed-expert tensors: the host serves
    /// these when the card cannot carry the whole load.
    pub(crate) expert_bytes: u64,
    /// Device bytes of one expert slot in one layer.
    pub(crate) slot_bytes: u64,
    /// The fewest slots per layer a token's picks need.
    pub(crate) min_slots: usize,
    /// Device bytes the carried banks must leave free beyond the `demand`'s own non-expert part:
    /// what the card-tiered load's admission keeps free in excess of the resident load's
    /// headroom the `demand` was summed with. Zero when both keep the same.
    pub(crate) carry_reserve: u64,
    /// Bytes of the `demand` the family does not keep once the start ends: the cache it builds
    /// at its ceiling and gives back before it serves (a session holds only the cells of its
    /// tokens and grows on demand). The banks may take them. Zero for a family that keeps its
    /// whole demand.
    pub(crate) kv_credit: u64,
}

impl CardBudget {
    /// Bytes the load is short of the card, or zero when it fits.
    pub(crate) fn short_by(&self) -> u64 {
        self.demand.saturating_sub(self.free)
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
    /// The card's side, for a caller that can serve its experts from the
    /// host. `None` keeps the host-only ladder.
    pub(crate) card: Option<CardBudget>,
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

impl TierBudget {
    /// The card-short answer: the experts stay on the host (which must hold
    /// them beside its fixed overhead and page-cache floor) and the banks
    /// take the card's bytes left after everything else in the load. The
    /// slot count is the most the card carries before the step's own state
    /// is allocated, so it is an upper bound the caller's bank sizing may
    /// lower. Refuses, naming both figures, when the card cannot hold the
    /// non-expert load plus the minimum banks or the host cannot hold the
    /// experts.
    fn card_tiered(&self, card: &CardBudget) -> Result<TierChoice, EngError> {
        let overflow = || tier_error("the card byte counts overflow".into());
        let resident = card
            .demand
            .checked_sub(card.expert_bytes)
            .ok_or_else(|| {
                tier_error(format!(
                    "the {} expert bytes exceed the {}-byte load demand",
                    card.expert_bytes, card.demand
                ))
            })?
            .saturating_sub(card.kv_credit)
            .checked_add(card.carry_reserve)
            .ok_or_else(overflow)?;
        let per_slot_of_layers = card
            .slot_bytes
            .checked_mul(self.layers as u64)
            .filter(|bytes| *bytes > 0)
            .ok_or_else(overflow)?;
        let room = card.free.saturating_sub(resident);
        let carried = usize::try_from(room / per_slot_of_layers)
            .unwrap_or(usize::MAX)
            .min(self.experts);
        if carried < card.min_slots {
            let minimum = per_slot_of_layers
                .checked_mul(card.min_slots as u64)
                .and_then(|banks| banks.checked_add(resident))
                .ok_or_else(overflow)?;
            return Err(tier_error(format!(
                "the card has {} bytes free; without its experts the load still needs {resident} \
                 bytes and the {} minimum slots per layer need {minimum} in all",
                card.free, card.min_slots
            )));
        }
        let host_need = card
            .expert_bytes
            .checked_add(self.overhead_bytes)
            .and_then(|need| need.checked_add(self.cache_floor))
            .ok_or_else(overflow)?;
        if host_need > self.available {
            // The host cannot hold the experts: they are read from the file on a miss, which
            // needs only the host's fixed overhead and page-cache floor.
            let streaming_need = self
                .overhead_bytes
                .checked_add(self.cache_floor)
                .ok_or_else(overflow)?;
            if streaming_need <= self.available {
                return Ok(TierChoice::CardDisk { slots: carried });
            }
            return Err(tier_error(format!(
                "the card is short by {} bytes and the host budget of {} bytes cannot hold \
                 the {} expert bytes it would serve, nor stream them from disk beside {} of \
                 fixed overhead and a {} byte page-cache floor",
                card.short_by(),
                self.available,
                card.expert_bytes,
                self.overhead_bytes,
                self.cache_floor
            )));
        }
        Ok(TierChoice::CardTiered {
            slots: carried,
            host_bytes: card.expert_bytes,
        })
    }
}

/// Chooses the tier: inclusive while it fits, then the exclusive tier at
/// the largest fitting spare-place count, then the disk tier, and a named
/// refusal below the minimum.
pub(crate) fn choose_tier(budget: &TierBudget) -> Result<TierChoice, EngError> {
    let overflow = || tier_error("the tier byte counts overflow".into());
    if let Some(card) = &budget.card
        && card.short_by() > 0
    {
        return budget.card_tiered(card);
    }
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
            card: None,
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

    /// Two layers of 10 experts and 5-byte slots: the load needs 1,000 bytes
    /// on the card, 700 of them routed experts, so the card carries the
    /// 300-byte rest plus `5 * 2` bytes per slot-of-every-layer.
    fn card_budget(free: u64) -> TierBudget {
        TierBudget {
            record_bytes: 35,
            layers: 2,
            experts: 10,
            slots: 0,
            max_spare_places: 0,
            min_spare_places: 0,
            overhead_bytes: 50,
            cache_floor: 10,
            ring_bytes: 0,
            available: 1_000,
            card: Some(CardBudget {
                free,
                demand: 1_000,
                expert_bytes: 700,
                slot_bytes: 5,
                min_slots: 2,
                carry_reserve: 0,
                kv_credit: 0,
            }),
        }
    }

    #[test]
    fn a_load_that_fits_the_card_answers_as_the_host_only_ladder_does() {
        for free in [1_000, 1_001, 50_000] {
            let with_card = card_budget(free);
            let mut without = card_budget(free);
            without.card = None;
            assert_eq!(
                choose_tier(&with_card),
                choose_tier(&without),
                "free {free}: the card arm must not change a fitting start"
            );
        }
    }

    #[test]
    fn a_card_short_load_moves_its_experts_to_the_host_and_banks_the_rest() {
        // 999 free: short by 1; 300 resident leaves 699, ten bytes per slot of
        // every layer: 69 slots, capped at the 10 experts of a layer.
        assert_eq!(
            choose_tier(&card_budget(999)),
            Ok(TierChoice::CardTiered {
                slots: 10,
                host_bytes: 700
            })
        );
        // 340 free leaves 40: four slots per layer.
        assert_eq!(
            choose_tier(&card_budget(340)),
            Ok(TierChoice::CardTiered {
                slots: 4,
                host_bytes: 700
            })
        );
        // 320 leaves 20: exactly the two-slot minimum.
        assert_eq!(
            choose_tier(&card_budget(320)),
            Ok(TierChoice::CardTiered {
                slots: 2,
                host_bytes: 700
            })
        );
    }

    #[test]
    fn a_card_that_cannot_carry_the_minimum_banks_refuses_by_name() {
        let error = choose_tier(&card_budget(319)).unwrap_err();
        let text = error.describe();
        assert!(text.contains("319 bytes free"), "{text}");
        assert!(text.contains("300"), "{text}");
        assert!(text.contains("320"), "{text}");
        assert!(text.contains("2 minimum slots"), "{text}");
    }

    #[test]
    fn a_host_that_cannot_hold_the_experts_streams_them_from_disk() {
        // 700 + 50 + 10 = 760 is the host's need to hold the experts: 759 streams them from
        // the file with the 50 of overhead and 10 of page-cache floor, 760 holds them.
        let mut short = card_budget(500);
        short.available = 759;
        assert_eq!(choose_tier(&short), Ok(TierChoice::CardDisk { slots: 10 }));
        let mut exact = card_budget(500);
        exact.available = 760;
        assert!(matches!(
            choose_tier(&exact),
            Ok(TierChoice::CardTiered { .. })
        ));
        // The minimum is the fixed overhead and the floor: 60 streams, 59 refuses by name.
        let mut least = card_budget(500);
        least.available = 60;
        assert_eq!(choose_tier(&least), Ok(TierChoice::CardDisk { slots: 10 }));
        least.available = 59;
        let text = choose_tier(&least).unwrap_err().describe();
        assert!(text.contains("short by 500 bytes"), "{text}");
        assert!(text.contains("59"), "{text}");
        assert!(text.contains("700 expert bytes"), "{text}");
        assert!(text.contains("stream them from disk"), "{text}");
    }

    #[test]
    fn the_card_arm_refuses_instead_of_wrapping() {
        let mut huge = card_budget(500);
        huge.card.as_mut().unwrap().slot_bytes = u64::MAX;
        assert!(
            choose_tier(&huge)
                .unwrap_err()
                .describe()
                .contains("overflow")
        );
        let mut inverted = card_budget(500);
        inverted.card.as_mut().unwrap().expert_bytes = 2_000;
        assert!(
            choose_tier(&inverted)
                .unwrap_err()
                .describe()
                .contains("exceed")
        );
    }
}
