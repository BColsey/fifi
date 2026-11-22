//! The card-fit planner: the largest context and the cache form a CUDA card carries.
//!
//! The planner is a pure function of byte counts, so every family feeds it the numbers it already
//! computes for its own start-up admission and the host tests cover the arithmetic without a
//! device. A family describes each cache form it can build (fp16, int8, ...) by one function: the
//! device bytes the load charges beside the weights at a candidate context (cache, fixed state,
//! staging, workspace and margins). The planner owns the rule that is the same everywhere: the
//! context ceiling, the search step, the slack band against the measured free bytes, and the
//! choice between forms.
//!
//! The form rule: forms are listed in preference order (the exact and faster form first). The
//! first form that reaches the lesser of the ceiling and the sufficient context is kept; when
//! none does, the form that reaches the longest context wins, and an earlier form wins a tie. A
//! form the config names is the only one weighed. `None` means no form fits even the smallest
//! context, and the load then reports its own refusal or takes a lower rung.

// The planner's callers are the CUDA starts; a build without them uses only the context search.
#![cfg_attr(
    not(any(test, all(feature = "native-cuda", target_os = "linux"))),
    allow(dead_code, reason = "only the CUDA starts plan a form")
)]

use crate::family::admission_sketch::largest_fitting_context;

/// One cache form a family can build.
pub(crate) struct CacheForm<'a, K> {
    /// The family's name for the form, returned in the choice.
    pub(crate) form: K,
    /// Device bytes the load charges beside the weights at `context`, or `None` when the
    /// arithmetic overflows (the context then does not fit).
    pub(crate) charge: &'a dyn Fn(u32) -> Option<u64>,
}

/// The device facts of one start.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CardBytes {
    /// Free device bytes before the weights are uploaded.
    pub(crate) free: u64,
    /// Bytes the weight load pins.
    pub(crate) weights: u64,
    /// Slack that must remain after every charge: the measurement gap between the probe and the
    /// serving process.
    pub(crate) band: u64,
}

/// The context limits of one start.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ContextLimits {
    /// The most positions the model or the config allows.
    pub(crate) ceiling: u32,
    /// The context up to which the first form is kept over a longer one.
    pub(crate) sufficient: u32,
}

/// The planned form and context.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CardFit<K> {
    pub(crate) form: K,
    pub(crate) context: u32,
}

fn fits(card: CardBytes, charge: &dyn Fn(u32) -> Option<u64>, context: u32) -> bool {
    charge(context).is_some_and(|charge| {
        i128::from(card.free) - i128::from(card.weights) - i128::from(charge)
            >= i128::from(card.band)
    })
}

/// The longest context at most `ceiling`, a multiple of the search step (or `ceiling` itself), for
/// which `fits` holds; `None` when not even one step fits. `fits` must be monotone (a longer
/// context never fits where a shorter one does not). This is the one context search: a plan whose
/// limit is not a byte count (the expert slots a context leaves, a refusal of the load's own
/// admission) passes its own predicate; a plan of byte charges goes through [`longest_context`].
pub(crate) fn longest_where(ceiling: u32, fits: &dyn Fn(u32) -> bool) -> Option<u32> {
    if fits(ceiling) {
        return Some(ceiling);
    }
    largest_fitting_context(ceiling, fits)
}

/// The longest context at most `ceiling`, a multiple of the search step (or `ceiling` itself),
/// that `charge` fits on the card.
pub(crate) fn longest_context(
    card: CardBytes,
    charge: &dyn Fn(u32) -> Option<u64>,
    ceiling: u32,
) -> Option<u32> {
    longest_where(ceiling, &|context| fits(card, charge, context))
}

/// Plans the form and context; see the module rule.
pub(crate) fn plan_card_fit<K: Copy>(
    card: CardBytes,
    forms: &[CacheForm<'_, K>],
    limits: ContextLimits,
) -> Option<CardFit<K>> {
    let reach: Vec<Option<u32>> = forms
        .iter()
        .map(|form| longest_context(card, form.charge, limits.ceiling))
        .collect();
    let enough = limits.ceiling.min(limits.sufficient);
    let kept = |index: usize| {
        reach[index].map(|context| CardFit {
            form: forms[index].form,
            context,
        })
    };
    if let Some(index) = (0..forms.len()).find(|&i| reach[i].is_some_and(|c| c >= enough)) {
        return kept(index);
    }
    let longest = reach.iter().flatten().max().copied()?;
    let index = reach.iter().position(|&c| c == Some(longest))?;
    kept(index)
}

/// Device bytes kept free beside the demand when several sessions are planned: allocations of the
/// first session that a load's demand does not list (measured on the 3090 for Bolt: the sessions
/// planner found 196 MiB less free after the first session than the demand predicts), so that a
/// planned session count survives them.
pub(crate) const SESSION_PLAN_SLACK_BYTES: u64 = 256 << 20;

/// The byte figures of a family whose cache is `per_position` bytes per cell and whose sessions
/// each hold one cache of that many cells.
#[derive(Clone, Copy, Debug)]
pub(crate) struct CellFigures {
    /// Cache bytes per cell, all layers.
    pub(crate) per_position: u64,
    /// The scratch shared by every session.
    pub(crate) scratch: u64,
    /// The headroom reserve beside everything else.
    pub(crate) headroom: u64,
    /// Fixed bytes of each session beyond its cache.
    pub(crate) seq_fixed: u64,
    /// The most cells the model carries.
    pub(crate) ceiling: usize,
    /// The count a refusal is worded for when the card carries none, and the floor a card that
    /// cannot fund every planned session serves.
    pub(crate) fallback: usize,
}

impl CellFigures {
    /// The figures of a family, `None` when any of its byte counts overflowed. `fixed` is the
    /// headroom and the per-session fixed bytes; `cells` the ceiling and the fallback.
    pub(crate) fn checked(
        per_position: Option<u64>,
        scratch: Option<u64>,
        fixed: (u64, u64),
        cells: (usize, usize),
    ) -> Option<Self> {
        Some(Self {
            per_position: per_position?,
            scratch: scratch?,
            headroom: fixed.0,
            seq_fixed: fixed.1,
            ceiling: cells.0,
            fallback: cells.1,
        })
    }

    /// The cells per layer a card carries beside `weights` while `sessions` sessions each hold
    /// that many: the largest count up to the ceiling whose first session's cache, the shared
    /// scratch, the headroom and the cache and fixed bytes of every further session fit `free`.
    /// `None` when not even the smallest count fits.
    pub(crate) fn plan(&self, free: u64, weights: u64, sessions: usize) -> Option<usize> {
        let extra = u64::try_from(sessions.max(1) - 1).unwrap_or(u64::MAX);
        let unlisted = if extra > 0 {
            SESSION_PLAN_SLACK_BYTES
        } else {
            0
        };
        let charge = |cells: u32| {
            let cache = u64::from(cells).checked_mul(self.per_position)?;
            let further = cache.checked_add(self.seq_fixed)?.checked_mul(extra)?;
            cache
                .checked_add(self.scratch)?
                .checked_add(self.headroom)?
                .checked_add(unlisted)?
                .checked_add(further)
        };
        let ceiling = u32::try_from(self.ceiling).ok()?;
        let band = 0;
        let card = CardBytes {
            free,
            weights,
            band,
        };
        let limits = ContextLimits {
            ceiling,
            sufficient: ceiling,
        };
        let form = CacheForm {
            form: (),
            charge: &charge,
        };
        plan_card_fit(card, &[form], limits).map(|fit| fit.context as usize)
    }

    /// The rows of one prefill chunk a card takes: the largest rung of `ladder` (largest first)
    /// whose scratch, `scratch_for(rows)`, the shared prefill admission leaves room for beside the
    /// weights, the fallback cache and the headroom. A rung that does not fit is skipped; the
    /// smallest rung is the floor. The variable `cap_var` may name one rung to cap the ladder.
    #[cfg_attr(not(feature = "native-cuda"), allow(dead_code))]
    pub(crate) fn prefill_rows(
        &self,
        free: u64,
        weights: u64,
        ladder: &'static [usize],
        scratch_for: &dyn Fn(usize) -> Option<u64>,
        cap_var: &str,
    ) -> usize {
        use crate::family::prefill::admission::{Admission, ChunkCost};
        let cap = std::env::var(cap_var)
            .ok()
            .and_then(|text| text.trim().parse::<usize>().ok())
            .filter(|rows| ladder.contains(rows));
        let cache = self.per_position.saturating_mul(self.fallback as u64);
        let room = free
            .saturating_sub(weights)
            .saturating_sub(cache)
            .saturating_sub(self.headroom);
        let costs: Vec<ChunkCost> = ladder
            .iter()
            .filter_map(|&m| {
                let arena = usize::try_from(scratch_for(m)?).ok()?;
                Some(ChunkCost { m, arena, part: 0 })
            })
            .collect();
        let budget = usize::try_from(room).unwrap_or(usize::MAX);
        Admission::new(budget, 0, 0, ladder)
            .admit(0, &costs, cap, cap_var)
            .ok()
            .flatten()
            .unwrap_or(ladder[ladder.len() - 1])
    }

    /// The cells a load serves: concurrency first, then context. The cache is widened beyond the
    /// fallback only as far as the planned `sessions` all still fit; a card that cannot fund them
    /// even at the fallback serves that many cells (or what one session carries, when less), so
    /// the session planner finds at least the count the fixed size gave.
    pub(crate) fn serve(&self, free: u64, weights: u64, sessions: usize) -> usize {
        let Some(one) = self.plan(free, weights, 1) else {
            return self.fallback;
        };
        let all = self.plan(free, weights, sessions).unwrap_or(0);
        all.max(one.min(self.fallback))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cell_figures_widen_beyond_the_fallback_only_as_far_as_the_sessions_fit() {
        let figures = CellFigures::checked(Some(1000), Some(0), (0, 0), (10_000, 4_000))
            .expect("the figures");
        assert_eq!(figures.plan(1 << 40, 0, 1), Some(10_000));
        assert_eq!(figures.plan(10, 0, 1), None);
        assert_eq!(figures.serve(10, 0, 1), 4_000);
        // Eight sessions each hold their own cache beside the slack: fewer cells than one.
        let free = 10_000_000 + SESSION_PLAN_SLACK_BYTES;
        let one = figures.plan(free, 0, 1).expect("one session");
        let eight = figures.plan(free * 8, 0, 8).expect("eight sessions");
        assert_eq!(one, 10_000);
        assert!(eight < one || eight == 10_000, "{eight}");
        assert!(figures.plan(free, 0, 8).is_none_or(|cells| cells < 10_000));
        assert!(figures.serve(free, 0, 8) >= figures.fallback.min(one));
        assert!(CellFigures::checked(None, Some(0), (0, 0), (1, 1)).is_none());
    }

    const MIB: u64 = 1 << 20;

    /// Two forms over a linear charge: `wide` costs 2 MiB per 1024 positions, `thin` 1 MiB.
    fn wide(context: u32) -> Option<u64> {
        Some(u64::from(context) / 1024 * 2 * MIB + 64 * MIB)
    }

    fn thin(context: u32) -> Option<u64> {
        Some(u64::from(context) / 1024 * MIB + 64 * MIB)
    }

    fn fits_card(card: CardBytes, charge: &dyn Fn(u32) -> Option<u64>, context: u32) -> bool {
        fits(card, charge, context)
    }

    fn card(free_steps: u64) -> CardBytes {
        CardBytes {
            free: 1000 * MIB + 128 * MIB + free_steps * MIB,
            weights: 1000 * MIB,
            band: 128 * MIB,
        }
    }

    const LIMITS: ContextLimits = ContextLimits {
        ceiling: 262_144,
        sufficient: 131_072,
    };

    fn forms<'a>() -> [CacheForm<'a, &'static str>; 2] {
        [
            CacheForm {
                form: "wide",
                charge: &wide,
            },
            CacheForm {
                form: "thin",
                charge: &thin,
            },
        ]
    }

    #[test]
    fn the_first_form_is_kept_where_it_reaches_the_sufficient_context() {
        // 64 + 2 x 128 = 320 MiB reaches 131,072 with the wide form.
        let fit = plan_card_fit(card(320 + 1), &forms(), LIMITS).expect("fits");
        assert_eq!((fit.form, fit.context), ("wide", 131_072));
    }

    #[test]
    fn a_longer_second_form_wins_when_the_first_falls_short() {
        let fit = plan_card_fit(card(64 + 2 * 100), &forms(), LIMITS).expect("fits");
        assert_eq!(fit.form, "thin");
        assert!(fit.context > 100 * 1024, "{}", fit.context);
    }

    #[test]
    fn the_longest_context_is_the_largest_step_that_fits() {
        let card = card(64 + 2 * 100);
        let fit = plan_card_fit(card, &forms(), LIMITS).expect("fits");
        assert!(fits(card, &thin, fit.context));
        assert!(!fits(card, &thin, fit.context + 1024));
    }

    #[test]
    fn a_roomy_card_reaches_the_ceiling_with_the_first_form() {
        let fit = plan_card_fit(card(100_000), &forms(), LIMITS).expect("fits");
        assert_eq!((fit.form, fit.context), ("wide", 262_144));
    }

    #[test]
    fn a_ceiling_below_the_sufficient_context_is_the_target() {
        let limits = ContextLimits {
            ceiling: 65_536,
            sufficient: 131_072,
        };
        // The wide form reaches the 64K ceiling: it is kept though thin would reach it too.
        let fit = plan_card_fit(card(64 + 2 * 64), &forms(), limits).expect("fits");
        assert_eq!((fit.form, fit.context), ("wide", 65_536));
    }

    #[test]
    fn a_tie_goes_to_the_earlier_form() {
        let same = [
            CacheForm {
                form: 'a',
                charge: &thin,
            },
            CacheForm {
                form: 'b',
                charge: &thin,
            },
        ];
        let fit = plan_card_fit(card(64 + 100), &same, LIMITS).expect("fits");
        assert_eq!(fit.form, 'a');
    }

    #[test]
    fn a_plan_with_its_own_limit_uses_the_same_search() {
        // A limit that is not a byte count: the plan fits up to 90,000 positions.
        let fits = |context: u32| context <= 90_000;
        assert_eq!(longest_where(262_144, &fits), Some(89_088));
        // The ceiling itself is tried first, so a ceiling that fits is returned as given.
        assert_eq!(longest_where(50_001, &fits), Some(50_001));
        // Nothing fits, or not even one step does.
        assert_eq!(longest_where(262_144, &|_| false), None);
        assert_eq!(longest_where(262_144, &|context| context < 1024), None);
        // A byte charge goes through the same search.
        let card = card(64 + 2 * 100);
        assert_eq!(
            longest_context(card, &thin, 262_144),
            longest_where(262_144, &|context| fits_card(card, &thin, context))
        );
    }

    #[test]
    fn a_card_that_carries_nothing_answers_none() {
        assert_eq!(plan_card_fit(card(0), &forms(), LIMITS), None);
        let overflow = |_: u32| None;
        let only = [CacheForm {
            form: 1,
            charge: &overflow,
        }];
        assert_eq!(plan_card_fit(card(100_000), &only, LIMITS), None);
    }
}
