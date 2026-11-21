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

/// The longest context at most `ceiling`, a multiple of the search step (or `ceiling` itself),
/// that `charge` fits on the card.
pub(crate) fn longest_context(
    card: CardBytes,
    charge: &dyn Fn(u32) -> Option<u64>,
    ceiling: u32,
) -> Option<u32> {
    if fits(card, charge, ceiling) {
        return Some(ceiling);
    }
    largest_fitting_context(ceiling, &|context| fits(card, charge, context))
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

#[cfg(test)]
mod tests {
    use super::*;

    const MIB: u64 = 1 << 20;

    /// Two forms over a linear charge: `wide` costs 2 MiB per 1024 positions, `thin` 1 MiB.
    fn wide(context: u32) -> Option<u64> {
        Some(u64::from(context) / 1024 * 2 * MIB + 64 * MIB)
    }

    fn thin(context: u32) -> Option<u64> {
        Some(u64::from(context) / 1024 * MIB + 64 * MIB)
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
