//! The admission-policy replay over recorded pick traces: three rows over the same
//! per-layer streams — the served row at 60 slots, sticky 36 (whole-call protection:
//! an earlier miss in a call may not evict an expert the same call picks later), the
//! prefix rule the bank served before that protection (only the picks already
//! assigned are protected), and the shared [`LayerLru`] run per call with the call's
//! own picks as the protected set. Per row and phase the test reports misses, uploaded
//! bytes (misses times the record bytes env) and the worst single call, so any
//! recorded trace keeps measuring the landed change against the rule it replaced.
//!
//! A child of [`policy`] on purpose: the old-rule row mirrors the pre-change
//! `assign_steps`/`victim` down to the private fields, and the second test validates
//! the mirror against the served row on every call where the two rules coincide.
//!
//! Input: `FIFI_EXL3_TRACE_DIR`, a directory of `<run>.prefill` / `<run>.decode` files as
//! the ignored banked probe dumps them (one `layer pick ...` line per resolve call). The
//! tests skip when the env is unset, so plain host gates stay green without fixtures.
//! `FIFI_EXL3_REPLAY_SLOTS` / `FIFI_EXL3_REPLAY_STICKY` override the served 60/36 row and
//! `FIFI_EXL3_RECORD_BYTES` sets the per-expert record bytes for the uploaded column.

use super::{DECAY_CALLS, EMPTY, LayerLru, LayerSlots, NO_SLOT};

/// The row's expert count, taken from the trace itself: the largest picked id plus one.
/// The shared module names no family and fixes no expert count of its own.
fn experts_of(calls: &[Vec<Call>]) -> usize {
    calls
        .iter()
        .flatten()
        .flat_map(|call| call.picks.iter().copied())
        .max()
        .unwrap_or_else(|| panic!("the trace holds no picks to size the row"))
        + 1
}

/// One resolve call from a trace file: the layer and its picks.
#[derive(Clone)]
struct Call {
    layer: usize,
    picks: Vec<usize>,
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Parses one trace file into per-layer call lists (distinct in-range picks kept).
fn parse(path: &std::path::Path) -> Vec<Vec<Call>> {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("{} does not read: {e}", path.display()));
    let mut calls: Vec<Vec<Call>> = Vec::new();
    for line in text.lines().filter(|l| !l.is_empty()) {
        let mut ids = line.split_whitespace().map(|n| {
            n.parse::<usize>()
                .unwrap_or_else(|_| panic!("{} has a non-integer id {n:?}", path.display()))
        });
        let (Some(layer), Some(first)) = (ids.next(), ids.next()) else {
            panic!("{} has a short line {line:?}", path.display());
        };
        let mut picks = vec![first];
        picks.extend(ids);
        picks.sort_unstable();
        picks.dedup();
        if calls.len() <= layer {
            calls.resize(layer + 1, Vec::new());
        }
        calls[layer].push(Call { layer, picks });
    }
    calls
}

#[derive(Default)]
struct RowTotals {
    calls: usize,
    misses: usize,
    worst_call: usize,
}

impl RowTotals {
    fn one_call(&mut self, misses: usize) {
        self.calls += 1;
        self.misses += misses;
        self.worst_call = self.worst_call.max(misses);
    }

    fn report(&self, label: &str, phase: &str, record_bytes: Option<u64>) -> String {
        let bytes = record_bytes
            .map(|b| {
                format!(
                    " uploaded {:.3} GiB",
                    self.misses as f64 * b as f64 / 2f64.powi(30)
                )
            })
            .unwrap_or_else(|| " uploaded n/a (FIFI_EXL3_RECORD_BYTES unset)".to_string());
        format!(
            "{label} {phase}: {} calls, {} misses (worst call {}),{bytes}",
            self.calls, self.misses, self.worst_call
        )
    }
}

/// Row 1, the served policy as-is: each call through the real `assign`.
fn replay_served(calls: &[Vec<Call>], experts: usize, slots: usize, sticky: usize) -> RowTotals {
    let mut row = RowTotals::default();
    for layer in calls {
        let mut table = LayerSlots::new(experts, slots, sticky);
        for call in layer {
            let (assigned, misses) = table
                .assign(&call.picks)
                .unwrap_or_else(|e| panic!("layer {} assign: {e:?}", call.layer));
            assert_eq!(assigned.len(), call.picks.len());
            row.one_call(misses.len());
        }
    }
    row
}

/// Row 3, the shared LRU per call with the call's picks as the protected set: for this
/// row the whole pass of one decode token IS the call.
fn replay_lru(calls: &[Vec<Call>], experts: usize, slots: usize) -> RowTotals {
    let mut row = RowTotals::default();
    for layer in calls {
        let mut lru = LayerLru::from_identity(experts, slots);
        for call in layer {
            let order: Vec<u16> = call.picks.iter().map(|&e| e as u16).collect();
            let mut demanded = vec![false; experts];
            for &pick in &call.picks {
                demanded[pick] = true;
            }
            let plan = lru
                .apply(&order, &demanded)
                .unwrap_or_else(|e| panic!("layer {} apply: {e}", call.layer));
            row.one_call(plan.moves.len());
        }
    }
    row
}

/// Row 2, the prefix rule the bank served before the whole-call protection: the
/// victim choice protects only the picks already assigned (the slots this call took),
/// so a miss may evict an expert the same call picks later. The body mirrors the
/// pre-change `assign_steps`/`victim` line for line (tick stamp, decay, hot set,
/// empty-first fill, the lowest `last_used` tie, pinning that yields to progress), so
/// the second test's coincidence validation is a real proof the mirror is faithful.
impl LayerSlots {
    /// One call through the mirrored old rule; returns the misses. Whether a decision
    /// actually diverged from the served row is measured outside, by lockstep state
    /// comparison — a filtered victim search can end at the same slot it would have
    /// picked anyway, so eviction counting underreports.
    fn assign_prefix(&mut self, picks: &[usize]) -> usize {
        let mut scratch = self.clone();
        let outcome = scratch.assign_steps_prefix(picks);
        *self = scratch;
        outcome
    }

    fn assign_steps_prefix(&mut self, picks: &[usize]) -> usize {
        self.tick += 1;
        if self.tick.is_multiple_of(DECAY_CALLS) {
            self.decay();
        }
        let mut taken: Vec<usize> = Vec::new();
        let mut misses = 0;
        for &expert in picks {
            if self.sticky > 0 {
                self.counts[expert] = self.counts[expert].saturating_add(1);
            }
            let slot = match self.slot_of[expert] {
                s if (0..self.slots as i32).contains(&s) => s as usize,
                _ => self.victim_prefix(&taken).expect("the picks fit the slots"),
            };
            if self.slot_of[expert] == NO_SLOT {
                let evicted = self.expert_of[slot];
                if evicted != EMPTY {
                    self.slot_of[usize::from(evicted)] = NO_SLOT;
                }
                self.expert_of[slot] = u16::try_from(expert).expect("expert fits u16");
                self.slot_of[expert] = slot as i32;
                misses += 1;
            }
            taken.push(slot);
        }
        for &slot in &taken {
            self.last_used[slot] = self.tick;
        }
        misses
    }

    /// The pre-change `victim`: an empty slot first, else the least-recently-used
    /// untouched slot, cold before hot — no whole-call protection anywhere.
    fn victim_prefix(&self, taken: &[usize]) -> Option<usize> {
        if let Some(slot) = (0..self.slots).find(|&slot| self.expert_of[slot] == EMPTY) {
            return Some(slot);
        }
        let all = |hot: bool| {
            (0..self.slots)
                .filter(|&slot| {
                    !taken.contains(&slot) && self.hot[usize::from(self.expert_of[slot])] == hot
                })
                .min_by_key(|&slot| self.last_used[slot])
        };
        all(false).or_else(|| all(true))
    }
}

/// Whole-table equality of two rows: every decision-affecting field compared, so a
/// lockstep replay can name the exact call where the new rule's protection first
/// changed a decision.
fn state_eq(a: &LayerSlots, b: &LayerSlots) -> bool {
    a.slot_of == b.slot_of
        && a.expert_of == b.expert_of
        && a.last_used == b.last_used
        && a.counts == b.counts
        && a.hot == b.hot
        && a.tick == b.tick
}

/// The old rule replayed in lockstep with the served row: `(row totals, calls whose
/// end state diverged)`. Divergence after a call means the up-front protection changed
/// at least one decision in it — the two rules differing, not the mirror drifting (the
/// second test proves that half).
fn replay_prefix(
    calls: &[Vec<Call>],
    experts: usize,
    slots: usize,
    sticky: usize,
) -> (RowTotals, usize) {
    let mut row = RowTotals::default();
    let mut divergent = 0;
    for layer in calls {
        let mut served = LayerSlots::new(experts, slots, sticky);
        let mut old = LayerSlots::new(experts, slots, sticky);
        for call in layer {
            let (_, served_misses) = served
                .assign(&call.picks)
                .unwrap_or_else(|e| panic!("assign: {e:?}"));
            let old_misses = old.assign_prefix(&call.picks);
            if !state_eq(&served, &old) {
                divergent += 1;
            }
            let _ = served_misses; // the served row's own totals come from replay_served
            row.one_call(old_misses);
        }
    }
    (row, divergent)
}

fn phase_files() -> Option<Vec<std::path::PathBuf>> {
    let dir = std::path::PathBuf::from(std::env::var_os("FIFI_EXL3_TRACE_DIR")?);
    let mut phases: Vec<std::path::PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("FIFI_EXL3_TRACE_DIR {dir:?} does not read: {e}"))
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension()
                .is_some_and(|ext| ext == "prefill" || ext == "decode")
        })
        .collect();
    phases.sort();
    Some(phases)
}

/// Replays every `<run>.<phase>` trace in the dir through the three rows and prints the
/// per-row per-phase table. Skipped without `FIFI_EXL3_TRACE_DIR`.
#[test]
fn the_three_slot_rows_replay_the_recorded_traces() {
    let Some(phases) = phase_files() else {
        eprintln!("FIFI_EXL3_TRACE_DIR unset: skipping the trace replay");
        return;
    };
    assert!(!phases.is_empty(), "the trace dir holds no phase files");
    let slots = env_usize("FIFI_EXL3_REPLAY_SLOTS", 60);
    let sticky = env_usize("FIFI_EXL3_REPLAY_STICKY", 36);
    let record_bytes = std::env::var("FIFI_EXL3_RECORD_BYTES")
        .ok()
        .and_then(|v| v.parse::<u64>().ok());
    for phase in &phases {
        let name = phase
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default()
            .to_string();
        let calls = parse(phase);
        assert!(!calls.is_empty(), "{} holds no calls", phase.display());
        let experts = experts_of(&calls);
        // A row cannot be stickier than its expert count; the clamp only bites on
        // degenerate synthetic traces (any real trace has experts far above slots).
        let sticky = sticky.min(experts.saturating_sub(1));
        let served = replay_served(&calls, experts, slots, sticky);
        let (prefix, divergent) = replay_prefix(&calls, experts, slots, sticky);
        let lru = replay_lru(&calls, experts, slots);
        println!(
            "trace {name}: {} layers, {} calls",
            calls.len(),
            served.calls
        );
        println!(
            "{}",
            served.report("served slots/sticky", &name, record_bytes)
        );
        println!(
            "{}",
            prefix.report("prefix protected (old rule)", &name, record_bytes)
        );
        println!("{name}: {divergent} calls diverged (a decision changed)");
        println!("{}", lru.report("LayerLru per call", &name, record_bytes));
        assert!(served.misses > 0, "a real trace at these slots misses");
        assert_eq!(served.calls, prefix.calls, "the rows see the same calls");
        assert_eq!(served.calls, lru.calls, "the rows see the same calls");
    }
}

/// The old-rule mirror equals the served row whenever the new protection does not fire:
/// on every call with no protected eviction the two rows must take the same miss count
/// from the same starting state, or the mirror has drifted from the pre-change
/// `assign_steps`.
#[test]
fn the_mirrored_row_equals_the_served_row_on_protection_free_calls() {
    let Some(phases) = phase_files() else {
        eprintln!("FIFI_EXL3_TRACE_DIR unset: skipping the mirror validation");
        return;
    };
    assert!(!phases.is_empty());
    let slots = env_usize("FIFI_EXL3_REPLAY_SLOTS", 60);
    let sticky = env_usize("FIFI_EXL3_REPLAY_STICKY", 36);
    let (mut checked, mut layers_diverged) = (0, 0);
    for phase in &phases {
        let layers = parse(phase);
        let experts = experts_of(&layers);
        let sticky = sticky.min(experts.saturating_sub(1));
        for layer in &layers {
            let mut served = LayerSlots::new(experts, slots, sticky);
            let mut mirrored = LayerSlots::new(experts, slots, sticky);
            for call in layer {
                // Until the first real divergence the states are equal going in, so the
                // two rules must take the same miss count; after a divergence the states
                // differ and the layer's remaining calls say nothing about the mirror.
                let (_, served_misses) = served
                    .assign(&call.picks)
                    .unwrap_or_else(|e| panic!("assign: {e:?}"));
                let mirror_misses = mirrored.assign_prefix(&call.picks);
                if state_eq(&served, &mirrored) {
                    assert_eq!(
                        served_misses.len(),
                        mirror_misses,
                        "layer {} diverges with equal end states",
                        call.layer
                    );
                    checked += 1;
                } else {
                    // The first divergence is the protection working: the old rule
                    // evicted an expert this same call picks and re-missed it, so the
                    // mirror's count is the higher one here — that difference is the
                    // measurement, not drift. Equal counts with different end states
                    // (same victim count, different victims) also stand.
                    assert!(
                        mirror_misses >= served_misses.len(),
                        "layer {} first divergence LOWERED the call's misses",
                        call.layer
                    );
                    layers_diverged += 1;
                    break;
                }
            }
        }
    }
    assert!(
        checked > 0,
        "no protection-free call to validate the mirror on"
    );
    eprintln!(
        "mirror validated on {checked} protection-free calls; {layers_diverged} layers diverged as designed"
    );
}
