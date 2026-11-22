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
    /// The device bytes a prefill needs beside the weights and the cache whatever the context: the
    /// arena (and the attention work buffer) of the smallest chunk the family admits. A context
    /// whose cache leaves less than this is no context: the start would have no prefill.
    pub(crate) arena: u64,
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
        i128::from(card.free)
            - i128::from(card.weights)
            - i128::from(charge)
            - i128::from(card.arena)
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

/// The device bytes a plan keeps free beside its load for `sessions` sessions: every session
/// after the first takes `part_bytes` (its state beyond the cache both consumers size
/// separately), `idle_parts` more parts are built beside the sessions' own (the executor's idle
/// part, which the step runs while no session does), and a plan that serves more than one keeps
/// [`SESSION_PLAN_SLACK_BYTES`] for what the first session allocates beyond its listed demand.
/// `None` when the byte count overflows. The one rule the Bolt cell pool (no idle part) and the
/// Qwen3.6 tiered banks (one idle part) both plan against.
pub(crate) fn session_reserve_bytes(
    part_bytes: u64,
    sessions: usize,
    idle_parts: usize,
) -> Option<u64> {
    let extra = u64::try_from(sessions.max(1) - 1).ok()?;
    let parts = extra.checked_add(u64::try_from(idle_parts).ok()?)?;
    let slack = if extra == 0 {
        0
    } else {
        SESSION_PLAN_SLACK_BYTES
    };
    part_bytes.checked_mul(parts)?.checked_add(slack)
}

/// The share of the one-session slots the sessions may take from the banks, in percent: a
/// session is worth its reserve only while the banks keep three quarters of what one session
/// would have had.
pub(crate) const SESSION_SLOT_GIVE_UP_PERCENT: usize = 25;

/// How many sessions a card funds beside its banks, at most `target`: the largest count whose
/// reserve leaves the banks `slots_at(count)` slots, at least their share of the `slots_at(1)`
/// the one session keeps. `slots_at` answers `None` for a count the card cannot carry at all.
/// With `exact` the count is the target or nothing (a named `max_seqs`): the card funds it with
/// whatever slots remain, or the answer is `None` and the caller refuses by name.
pub(crate) fn funded_sessions(
    target: usize,
    exact: bool,
    slots_at: &dyn Fn(usize) -> Option<usize>,
) -> Option<usize> {
    let one = slots_at(1)?;
    if exact {
        return slots_at(target.max(1)).map(|_| target.max(1));
    }
    let keep = one - one * SESSION_SLOT_GIVE_UP_PERCENT / 100;
    let funded = (2..=target)
        .rev()
        .find(|&sessions| slots_at(sessions).is_some_and(|slots| slots >= keep));
    Some(funded.unwrap_or(1))
}

/// The most cells a shared pool is planned for: far above any card's reach.
const POOL_CELLS_LIMIT: u32 = 1 << 28;

/// The byte figures of a family whose sessions lease their cache cells from one pool.
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
}

impl CellFigures {
    /// The figures of a family, `None` when any of its byte counts overflowed. `fixed` is the
    /// headroom and the per-session fixed bytes.
    pub(crate) fn checked(
        per_position: Option<u64>,
        scratch: Option<u64>,
        fixed: (u64, u64),
    ) -> Option<Self> {
        Some(Self {
            per_position: per_position?,
            scratch: scratch?,
            headroom: fixed.0,
            seq_fixed: fixed.1,
        })
    }

    /// The cells one session may hold and the cells every session shares, for a file of `weights`
    /// bytes: the pool is what the card carries ([`CellFigures::pool`]), one session holds up to
    /// the model's context (`ceiling`) of it. A card that carries none gets `fallback`, which the
    /// admission then refuses with its own figures.
    pub(crate) fn planned(
        &self,
        free: u64,
        weights: u64,
        sessions: usize,
        (ceiling, fallback): (usize, usize),
    ) -> (usize, usize) {
        let pool = self.pool(free, weights, sessions).unwrap_or(fallback);
        (pool.min(ceiling), pool)
    }

    /// The planned `pool` cut to what a card holding `held` free bytes carries now, after the load
    /// allocated what its demand did not list.
    pub(crate) fn held_pool(&self, pool: usize, held: u64, sessions: usize) -> usize {
        pool.min(self.pool(held, 0, sessions).unwrap_or(0))
    }

    /// The cells every session shares: what the card carries beside `weights` once the shared
    /// scratch, the headroom and the fixed bytes of `sessions` sessions are charged. A single
    /// session is bounded by the model's context, not by this count. `None` when not even one
    /// cell fits.
    pub(crate) fn pool(&self, free: u64, weights: u64, sessions: usize) -> Option<usize> {
        let reserve = session_reserve_bytes(self.seq_fixed, sessions, 0);
        let charge = |cells: u32| {
            u64::from(cells)
                .checked_mul(self.per_position)?
                .checked_add(self.headroom)?
                .checked_add(reserve?)
        };
        let limits = ContextLimits {
            ceiling: POOL_CELLS_LIMIT,
            sufficient: POOL_CELLS_LIMIT,
        };
        let form = CacheForm {
            form: (),
            charge: &charge,
        };
        // The scratch every session shares is the prefill working set: the one arena charge.
        let card = CardBytes {
            free,
            weights,
            band: 0,
            arena: self.scratch,
        };
        plan_card_fit(card, &[form], limits).map(|fit| fit.context as usize)
    }

    /// The rows of one prefill chunk a card takes: the largest rung of `ladder` (largest first)
    /// whose scratch, `scratch_for(rows)`, the shared prefill admission leaves room for beside the
    /// weights, a cache of `cache_cells` and the headroom. A rung that does not fit is skipped; the
    /// smallest rung is the floor. The variable `cap_var` may name one rung to cap the ladder.
    #[cfg_attr(not(feature = "native-cuda"), allow(dead_code))]
    pub(crate) fn prefill_rows(
        &self,
        free: u64,
        weights: u64,
        (ladder, cache_cells): (&'static [usize], usize),
        scratch_for: &dyn Fn(usize) -> Option<u64>,
        cap_var: &str,
    ) -> usize {
        use crate::family::prefill::admission::{Admission, ChunkCost};
        let cap = std::env::var(cap_var)
            .ok()
            .and_then(|text| text.trim().parse::<usize>().ok())
            .filter(|rows| ladder.contains(rows));
        let cache = self.per_position.saturating_mul(cache_cells as u64);
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_pool_holds_every_cell_the_card_carries_beside_the_weights() {
        let figures = CellFigures::checked(Some(1000), Some(7), (3, 100)).expect("the figures");
        // One session: every step of 1,024 cells the card carries.
        assert_eq!(figures.pool(1_024_000 + 10, 0, 1), Some(1_024));
        assert_eq!(figures.pool(1_024_000 + 9, 0, 1), None);
        assert_eq!(figures.pool(40_960_000 + 10, 0, 1), Some(40_960));
        assert_eq!(
            figures.pool(40_960_000 + 10 + 1_023_999, 0, 1),
            Some(40_960)
        );
        // Further sessions cost their fixed bytes and the slack, not another cache.
        let free = 40_960_000 + 10 + SESSION_PLAN_SLACK_BYTES + 700;
        assert_eq!(figures.pool(free, 0, 8), Some(40_960));
        assert_eq!(figures.pool(9, 0, 1), None);
        // After the load, the pool is cut to what the card holds now, never raised.
        assert_eq!(figures.held_pool(40_960, 20_480_000 + 10, 1), 20_480);
        assert_eq!(figures.held_pool(10_240, 40_960_000 + 10, 1), 10_240);
        assert_eq!(figures.held_pool(10_240, 9, 1), 0);
        // One session holds up to the ceiling of the pool; a card that carries none is the fallback.
        assert_eq!(
            figures.planned(40_960_000 + 10, 0, 1, (10_000, 4_000)),
            (10_000, 40_960)
        );
        assert_eq!(
            figures.planned(1_024_000 + 10, 0, 1, (10_000, 4_000)),
            (1_024, 1_024)
        );
        assert_eq!(figures.planned(9, 0, 1, (10_000, 4_000)), (4_000, 4_000));
        assert!(CellFigures::checked(None, Some(0), (0, 0)).is_none());
    }

    #[test]
    fn the_session_reserve_is_the_extra_parts_the_idle_parts_and_one_slack() {
        // No idle part (Bolt): nothing for one session, extra parts and one slack beyond.
        assert_eq!(session_reserve_bytes(1000, 0, 0), Some(0));
        assert_eq!(session_reserve_bytes(1000, 1, 0), Some(0));
        assert_eq!(
            session_reserve_bytes(1000, 2, 0),
            Some(1000 + SESSION_PLAN_SLACK_BYTES)
        );
        assert_eq!(
            session_reserve_bytes(1000, 8, 0),
            Some(7000 + SESSION_PLAN_SLACK_BYTES)
        );
        // One idle part (the Qwen executor): a lone session still builds it, so it is charged.
        assert_eq!(session_reserve_bytes(1000, 1, 1), Some(1000));
        assert_eq!(
            session_reserve_bytes(1000, 4, 1),
            Some(4000 + SESSION_PLAN_SLACK_BYTES)
        );
        assert_eq!(session_reserve_bytes(u64::MAX, 3, 0), None);
    }

    #[test]
    fn sessions_are_funded_until_the_banks_would_lose_a_quarter() {
        // 200 slots with one session, 2 slots lost to each further one.
        let slots = |sessions: usize| Some(200 - 2 * (sessions - 1));
        assert_eq!(funded_sessions(4, false, &slots), Some(4));
        assert_eq!(funded_sessions(1, false, &slots), Some(1));
        // 30 slots a session: 200, 170, 140, 110: the share is 150, so two are funded.
        let heavy = |sessions: usize| Some(200 - 30 * (sessions - 1));
        assert_eq!(funded_sessions(4, false, &heavy), Some(2));
        // A card that carries one session only funds one; a card that carries none, none.
        let one = |sessions: usize| (sessions == 1).then_some(8);
        assert_eq!(funded_sessions(4, false, &one), Some(1));
        assert_eq!(funded_sessions(4, false, &|_| None), None);
        // A named count is funded whole or not at all, whatever the banks lose.
        assert_eq!(funded_sessions(4, true, &heavy), Some(4));
        assert_eq!(funded_sessions(4, true, &one), None);
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
            arena: 0,
        }
    }

    #[test]
    fn the_arena_is_funded_before_the_context_is_sized() {
        let bare = card(400);
        let with = CardBytes {
            arena: 40 * MIB,
            ..bare
        };
        let ceiling = 262_144;
        let at = |card| longest_context(card, &wide, ceiling).expect("a context");
        // The arena takes its bytes from the cache: 40 MiB is 20,480 positions of `wide`.
        assert_eq!(at(bare) - at(with), 20_480);
        // A card that holds the cache and not the arena has no context at all.
        let tight = CardBytes {
            arena: 100_000 * MIB,
            ..bare
        };
        assert_eq!(longest_context(tight, &wide, ceiling), None);
        // The form rule sees the arena too: a thin form can reach what the wide one cannot.
        let forms = forms();
        let fit = plan_card_fit(with, &forms, LIMITS).expect("a fit");
        assert!(fit.context >= at(with));
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
