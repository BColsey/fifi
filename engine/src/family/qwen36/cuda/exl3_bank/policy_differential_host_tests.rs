//! The differential of the two EXL3 slot rows: Qwen3.6's [`LayerSlots`] (sticky 0, the
//! plain-LRU arm) against the shared [`LayerLru`] that Flint and Flash-Next run, driven
//! through the same pick streams from equivalent states. The two rows differ in
//! three decision-affecting ways — per-call tick stamps with a lowest-slot tie against a
//! per-touch recency list, prefix-of-call protection against whole-pass protection, and
//! empty-slot-first fill — and this test turns each into a pinned fact, then quantifies
//! how often a realistic stream diverges. The map is the input to any later decision to
//! align the rows: until then it proves the two are NOT one rule written twice.
//!
//! Native-cuda gated with its module (the q36 EXL3 bank), so it runs in every container
//! test battery; the q36 pick golden remains the row's own pinned replay.

use crate::family::exl3::bank::LayerSlots;
use crate::family::expert_bank::policy::LayerLru;

const EXPERTS: usize = 512;
const SLOTS: usize = 48;
const CALLS: usize = 200;
const PICKS: usize = 8;
/// Measured on this file's first run — see the last test.
const LRU_MISSES: usize = 1469;
const SLOTS_ROW_MISSES: usize = 1478;

/// The xorshift stream the pick goldens use: `CALLS` calls of `PICKS` distinct experts.
fn call_picks(seed: u64, call: usize) -> Vec<usize> {
    let mut state = seed ^ (call as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    let mut picks = Vec::with_capacity(PICKS);
    while picks.len() < PICKS {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let expert = (state >> 33) as usize % EXPERTS;
        if !picks.contains(&expert) {
            picks.push(expert);
        }
    }
    picks
}

/// One step of both rows from equivalent states: the misses (incoming expert, victim
/// slot) the call caused. `LayerLru` is driven whole-pass (Flint and Flash protect every
/// demanded expert up front); `LayerSlots` is driven per call through `assign`.
struct Step {
    misses: Vec<(usize, usize)>,
}

fn step_lru(row: &mut LayerLru, picks: &[usize]) -> Step {
    let mut demanded = vec![false; EXPERTS];
    for &expert in picks {
        demanded[expert] = true;
    }
    let order: Vec<u16> = picks.iter().map(|&expert| expert as u16).collect();
    let plan = row
        .apply(&order, &demanded)
        .expect("in-range distinct picks");
    Step {
        misses: plan
            .moves
            .iter()
            .map(|mv| (usize::from(mv.incoming), usize::from(mv.slot)))
            .collect(),
    }
}

fn step_slots(table: &mut LayerSlots, picks: &[usize]) -> Step {
    let (_, misses) = table.assign(picks).expect("in-range distinct picks");
    Step {
        misses: misses
            .into_iter()
            .map(|slot| (table.expert_of(slot), slot))
            .collect(),
    }
}

/// The resident set of each row.
fn resident_of_lru(row: &LayerLru) -> Vec<usize> {
    (0..EXPERTS)
        .filter(|&expert| row.is_resident(expert))
        .collect()
}

fn resident_of_slots(table: &LayerSlots) -> Vec<usize> {
    (0..SLOTS)
        .filter_map(|slot| table.expert_in_slot(slot).map(usize::from))
        .collect()
}

/// A pair of rows in equivalent states: `LayerLru` at its identity fill (experts
/// `0..SLOTS` in slot order, recency in slot order) and `LayerSlots` filled by one call
/// with the same picks (the same residency; every slot shares one tick, which is exactly
/// the per-call stamp the differential exposes).
fn equivalent_pair() -> (LayerLru, LayerSlots) {
    let row = LayerLru::from_identity(EXPERTS, SLOTS);
    let mut table = LayerSlots::new(EXPERTS, SLOTS, 0);
    let fill: Vec<usize> = (0..SLOTS).collect();
    table.assign(&fill).expect("the identity fill");
    (row, table)
}

#[test]
fn the_tie_rule_diverges_when_a_call_touches_slots_out_of_order() {
    // Six slots, both rows holding experts 0..6. One call hits slots {5, 3} in that
    // order, another touches every other slot, so {5, 3} are the oldest pair. A miss
    // then picks its victim: LayerSlots ties both at the same tick and takes the lowest
    // slot (3); the recency list keeps the call's touch order and takes slot 5.
    let (mut row, mut table) = equivalent_pair();
    let _ = row.apply(&[5, 3], &flags(&[5, 3])).expect("hits");
    let _ = table.assign(&[5, 3]).expect("hits");
    let others: Vec<usize> = (0..SLOTS).filter(|&s| s != 5 && s != 3).collect();
    let _ = row
        .apply(
            &others.iter().map(|&e| e as u16).collect::<Vec<_>>(),
            &flags(&others),
        )
        .expect("hits");
    let _ = table.assign(&others).expect("hits");

    let lru = step_lru(&mut row, &[100]);
    let slots_row = step_slots(&mut table, &[100]);
    assert_eq!(
        lru.misses,
        vec![(100, 5)],
        "the list evicts slot 5, touched first"
    );
    assert_eq!(
        slots_row.misses,
        vec![(100, 3)],
        "the tick ties and takes slot 3"
    );
}

#[test]
fn the_protection_scope_diverges_when_a_miss_precedes_a_resident_pick() {
    // A call [miss, resident pick] on the coldest slot: LayerLru protects the whole
    // pass up front, so the miss evicts slot 1 and the resident pick stays a hit — one
    // miss. LayerSlots protects only the picks assigned so far, so the miss takes the
    // coldest slot (0, the resident pick's own slot) and the pick then misses again —
    // two misses, the resident expert moved.
    let (mut row, mut table) = equivalent_pair();

    let lru = step_lru(&mut row, &[100, 0]);
    assert_eq!(
        lru.misses,
        vec![(100, 1)],
        "the pass protects expert 0 up front"
    );
    assert_eq!(row.slot_of(0), Some(0), "expert 0 keeps its slot");

    let slots_row = step_slots(&mut table, &[100, 0]);
    assert_eq!(
        slots_row.misses.len(),
        2,
        "the miss evicted expert 0's slot and the pick re-missed: {:?}",
        slots_row.misses
    );
    assert_ne!(table.slot_row()[0], 0, "expert 0 left its own slot");
}

#[test]
fn empty_slots_fill_first_and_holes_are_layer_slots_own_rule() {
    // A fresh LayerSlots has no residency: its first misses fill the empty slots in
    // index order. The recency list has no empty state (identity is its floor), so this
    // arm pins the structural difference: the same first call evicts experts 0 and 1
    // from the list's identity fill while the empty table simply fills.
    let mut row = LayerLru::from_identity(EXPERTS, SLOTS);
    let mut table = LayerSlots::new(EXPERTS, SLOTS, 0);

    let lru = step_lru(&mut row, &[300, 301]);
    let slots_row = step_slots(&mut table, &[300, 301]);
    assert_eq!(
        lru.misses,
        vec![(300, 0), (301, 1)],
        "identity experts evicted"
    );
    assert_eq!(
        slots_row.misses,
        vec![(300, 0), (301, 1)],
        "the empty slots fill in index order — same landing, no eviction"
    );
    assert_eq!(resident_of_lru(&row).len(), SLOTS);
    assert_eq!(resident_of_slots(&table), vec![300, 301]);

    // The torn-refill hole is LayerSlots' alone: an invalidated slot reads absent in
    // both directions and the next miss refills it first, whatever the recency. The
    // list row has no counterpart, so this half pins the mechanism, not a comparison.
    table.invalidate(&[0]);
    assert!(table.slot_row()[300] < 0, "the hole reads absent");
    let step = step_slots(&mut table, &[302]);
    assert_eq!(step.misses, vec![(302, 0)], "the hole fills first");
}

#[test]
fn a_realistic_stream_diverges_and_the_divergence_is_pinned() {
    // The measured headline: from equivalent states, the golden
    // generator's own stream (200 calls, 8 distinct picks) diverges on the first call
    // and never re-converges — per-call stamps cannot reproduce per-touch order. The
    // pinned numbers are the divergence map for any later alignment decision; if a
    // change moves them, that change is a decision change.
    let (mut row, mut table) = equivalent_pair();
    let seed = 0xfeed_0001;
    let mut first_divergence: Option<usize> = None;
    let mut divergent_calls = 0usize;
    let mut total_misses_lru = 0usize;
    let mut total_misses_slots = 0usize;
    for call in 0..CALLS {
        let picks = call_picks(seed, call);
        let lru = step_lru(&mut row, &picks);
        let slots_row = step_slots(&mut table, &picks);
        total_misses_lru += lru.misses.len();
        total_misses_slots += slots_row.misses.len();
        let diverged =
            lru.misses != slots_row.misses || resident_of_lru(&row) != resident_of_slots(&table);
        if diverged {
            divergent_calls += 1;
            first_divergence.get_or_insert(call);
        }
    }
    assert_eq!(first_divergence, Some(0), "the first call already diverges");
    assert_eq!(
        divergent_calls, CALLS,
        "once apart, the rows never re-converge on this stream"
    );
    assert_eq!(total_misses_lru, LRU_MISSES, "the whole-pass row's misses");
    assert_eq!(
        total_misses_slots, SLOTS_ROW_MISSES,
        "the per-call row's misses"
    );
}

/// The demanded bitmap of `picks` (whole-pass protection, the LayerLru input).
fn flags(picks: &[usize]) -> Vec<bool> {
    let mut demanded = vec![false; EXPERTS];
    for &expert in picks {
        demanded[expert] = true;
    }
    demanded
}
