//! Host side of the offline MTP acceptance study (test-only, no device).
//!
//! The study runs the target greedily and records, at every pending position `m`, the chain of
//! drafts the head proposes. This module scores such a trace the way a serving loop would: a
//! window for pending token `s_m` feeds `1 + k` rows, draws `s_{m+1}..`, keeps drafts while they
//! equal the draw (`accept_run`), and the next window starts at the last committed token. Every
//! window costs its full `k + 1` rows, whatever the recorded stream does after it, so no recorded
//! future length shortens a draft or a window; a stream that stopped early ends the window at the
//! stop token exactly as `accept_run` does. A window with no chain, or a chain shorter than `k`,
//! is an error: it is never read as a window with fewer drafts.
//!
//! The decision rule is fixed in the protocol: `speedup = E[accepted + 1] / (W(k + 1) + H(k))`
//! against a gate of 1.15, with `H(k)` the cost of drafting the k-token chain. A window and head
//! cost whose ceiling `(k + 1) / (W + H)` is below the gate is marked infeasible instead of being
//! given an acceptance above 100 percent. The Python collector in `qualification/mtp-acceptance/`
//! scores the same traces; its trace tests use the same hand-made vector as `golden()` below.

use super::accept::accept_run;
use crate::EngError;

/// End-to-end gate set for the study.
pub(crate) const GATE: f64 = 1.15;

/// The target's greedy stream and the head's chains over it.
pub(crate) struct Trace<'a> {
    /// Tokens `s_0 ..` the target drew. Holds at least `scored + max_k + 1` tokens unless it ends
    /// with a stop token.
    pub(crate) stream: &'a [u32],
    /// `chains[m][j - 1]` is the head's draft for `s_{m + j}`, with `s_m` pending. Every scored
    /// position needs a chain of at least `k` drafts.
    pub(crate) chains: &'a [Vec<u32>],
    /// Pending positions `0..scored` may start a window.
    pub(crate) scored: usize,
}

/// One window of the renewal run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct WindowOutcome {
    /// Index of the pending token.
    pub(crate) pending: usize,
    /// Tokens the window committed (accepted drafts plus the draw that ended it).
    pub(crate) committed: usize,
}

/// The first `k` drafts of the chain at `pending`, or an error when there is no such chain.
fn chain_of<'a>(trace: &Trace<'a>, pending: usize, k: usize) -> Result<&'a [u32], EngError> {
    match trace.chains.get(pending) {
        Some(chain) if chain.len() >= k => Ok(&chain[..k]),
        Some(chain) => Err(EngError::BadRequest(format!(
            "draft chain at position {pending} holds {} drafts, the window needs {k}",
            chain.len()
        ))),
        None => Err(EngError::BadRequest(format!(
            "no draft chain at position {pending}"
        ))),
    }
}

/// Renewing windows with `k` drafts over the trace, in order.
pub(crate) fn renewing_windows(
    trace: &Trace<'_>,
    k: usize,
    stops: &dyn Fn(u32) -> bool,
) -> Result<Vec<WindowOutcome>, EngError> {
    let mut out = Vec::new();
    let mut pending = 0;
    while pending < trace.scored {
        let drafts = chain_of(trace, pending, k)?;
        let run = accept_run(
            drafts,
            usize::MAX,
            |row| {
                trace.stream.get(pending + 1 + row).copied().ok_or_else(|| {
                    EngError::BadRequest("trace ends before the window's draw".into())
                })
            },
            stops,
        )?;
        let committed = run.tokens.len();
        out.push(WindowOutcome { pending, committed });
        pending += committed;
        if run.stopped {
            break;
        }
    }
    Ok(out)
}

/// Mean tokens committed per window: `E[accepted + 1]`.
pub(crate) fn mean_tokens(windows: &[WindowOutcome]) -> f64 {
    let tokens: usize = windows.iter().map(|w| w.committed).sum();
    tokens as f64 / windows.len().max(1) as f64
}

/// `E[accepted + 1] / (W(k + 1) + H(k))`: all costs in solo decode steps. `H(k)` is the cost of
/// drafting the whole k-token chain (`k * c_head` when every chained step costs the same).
pub(crate) fn speedup(mean_tokens: f64, window_cost: f64, draft_cost: f64) -> f64 {
    mean_tokens / (window_cost + draft_cost)
}

/// The largest `W(k + 1)` that still reaches the gate, or why no window can.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Budget {
    Largest(f64),
    /// The chain alone costs at least `E / gate`: no window cost pays.
    Infeasible,
}

pub(crate) fn max_window_cost(mean_tokens: f64, draft_cost: f64) -> Budget {
    max_window_cost_at(GATE, mean_tokens, draft_cost)
}

fn max_window_cost_at(gate: f64, mean_tokens: f64, draft_cost: f64) -> Budget {
    let budget = mean_tokens / gate - draft_cost;
    if budget <= 0.0 {
        Budget::Infeasible
    } else {
        Budget::Largest(budget)
    }
}

/// What a measured window cost asks of acceptance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Requirement {
    /// `E[accepted + 1]` must reach `required_tokens`, which is at most `k + 1`.
    Reachable { required_tokens: f64 },
    /// Even at full acceptance (`k + 1` tokens per window) the speedup stays below the gate.
    /// There is no acceptance to ask for, so none is reported.
    Infeasible { ceiling_speedup: f64 },
}

pub(crate) fn requirement(k: usize, window_cost: f64, draft_cost: f64) -> Requirement {
    requirement_at(GATE, k, window_cost, draft_cost)
}

fn requirement_at(gate: f64, k: usize, window_cost: f64, draft_cost: f64) -> Requirement {
    let ceiling_speedup = (k + 1) as f64 / (window_cost + draft_cost);
    if ceiling_speedup < gate {
        return Requirement::Infeasible { ceiling_speedup };
    }
    Requirement::Reachable {
        required_tokens: gate * (window_cost + draft_cost),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const STOP: u32 = 248_046;

    fn not_a_stop(_: u32) -> bool {
        false
    }

    /// The hand-made vector the Python trace tests also use (`GOLD_STREAM` and `GOLD_CHAINS`).
    fn golden() -> (Vec<u32>, Vec<Vec<u32>>) {
        (
            vec![10, 11, 12, 13, 14, 15, 16, 17, 18],
            vec![
                vec![11, 12],
                vec![12, 13],
                vec![13, 99],
                vec![14, 99],
                vec![99, 99],
                vec![99, 17],
            ],
        )
    }

    fn windows(stream: &[u32], chains: &[Vec<u32>], k: usize) -> Vec<(usize, usize)> {
        let trace = Trace {
            stream,
            chains,
            scored: chains.len(),
        };
        renewing_windows(&trace, k, &|t| t == STOP)
            .expect("windows")
            .iter()
            .map(|w| (w.pending, w.committed))
            .collect()
    }

    #[test]
    fn two_drafts_renew_at_the_last_committed_token() {
        let (stream, chains) = golden();
        let got = windows(&stream, &chains, 2);
        assert_eq!(got, [(0, 3), (3, 2), (5, 1)]);
        let outcomes: Vec<_> = got
            .iter()
            .map(|&(pending, committed)| WindowOutcome { pending, committed })
            .collect();
        assert_eq!(mean_tokens(&outcomes), 2.0);
    }

    #[test]
    fn one_draft_uses_only_the_first_chain_step() {
        let (stream, chains) = golden();
        let got = windows(&stream, &chains, 1);
        assert_eq!(got, [(0, 2), (2, 2), (4, 1), (5, 1)]);
    }

    #[test]
    fn a_stop_token_draw_ends_the_window_and_the_run() {
        let stream = [10, 11, STOP, 13, 14];
        let chains = vec![vec![11, STOP, 13], vec![STOP, 13, 14], vec![13, 14, 15]];
        let got = windows(&stream, &chains, 3);
        assert_eq!(
            got,
            [(0, 2)],
            "the stop is committed, nothing after it is drawn"
        );
    }

    #[test]
    fn a_truncated_trace_is_an_error_not_a_shorter_window() {
        let stream = [10, 11];
        let chains = vec![vec![11, 12, 13]];
        let trace = Trace {
            stream: &stream,
            chains: &chains,
            scored: 1,
        };
        assert!(renewing_windows(&trace, 3, &not_a_stop).is_err());
    }

    #[test]
    fn a_missing_or_short_chain_is_an_error_not_a_window_with_fewer_drafts() {
        let stream = [10, 11, 12, 13, 14, 15];
        let missing = vec![vec![11, 12]];
        let trace = Trace {
            stream: &stream,
            chains: &missing,
            scored: 4,
        };
        // position 0 has a chain; the window commits 3 tokens and position 3 has none
        let err = renewing_windows(&trace, 2, &not_a_stop).expect_err("no chain at position 3");
        assert!(format!("{err:?}").contains("no draft chain at position 3"));
        let short = vec![vec![11]];
        let trace = Trace {
            stream: &stream,
            chains: &short,
            scored: 1,
        };
        let err = renewing_windows(&trace, 2, &not_a_stop).expect_err("chain too short");
        assert!(format!("{err:?}").contains("holds 1 drafts"));
    }

    #[test]
    fn the_registered_example_needs_a_window_of_1_52_solo_steps() {
        // k = 1, per-position acceptance 0.75, free head: E = 1.75, so W(2) <= 1.75 / 1.15.
        assert_eq!(
            max_window_cost(1.75, 0.0),
            Budget::Largest(1.521_739_130_434_782_7)
        );
        // and a window of 1.5 steps then runs at 1.75 / 1.5
        assert!((speedup(1.75, 1.5, 0.0) - 1.166_666_666_666_666_7).abs() < 1e-12);
    }

    #[test]
    fn speedup_divides_by_the_window_and_the_draft_cost_together() {
        // E 2.0 over a window of 1.5 and a chain of 0.5 is exactly one solo step per solo step
        assert!((speedup(2.0, 1.5, 0.5) - 1.0).abs() < 1e-12);
        // 1.9 / (1.59 + 0.10) = 1.124260355...
        assert!((speedup(1.9, 1.59, 0.10) - 1.124_260_355_029_586).abs() < 1e-12);
    }

    #[test]
    fn drafting_cost_comes_off_the_window_budget_one_for_one() {
        // 1.9 / 1.15 = 1.652173913...; a draft cost of 0.10 leaves 1.552173913...
        let Budget::Largest(paid) = max_window_cost(1.9, 0.10) else {
            panic!("a draft cost of 0.10 leaves a budget");
        };
        assert!((paid - 1.552_173_913_043_478).abs() < 1e-12);
        // a three-step chain of 0.1 each: 2.5 / 1.15 = 2.173913...; minus 0.3 leaves 1.873913...
        let Budget::Largest(chain) = max_window_cost(2.5, 0.3) else {
            panic!("a three-step chain leaves a budget");
        };
        assert!((chain - 1.873_913_043_478_261).abs() < 1e-12);
    }

    #[test]
    fn a_chain_that_eats_the_whole_budget_is_infeasible() {
        assert_eq!(max_window_cost(1.2, 1.1), Budget::Infeasible);
    }

    #[test]
    fn a_budget_of_exactly_zero_is_infeasible_and_a_hair_above_it_is_not() {
        // gate 1: 0.5 tokens at a draft cost of 0.5 leave exactly zero
        assert_eq!(max_window_cost_at(1.0, 0.5, 0.5), Budget::Infeasible);
        match max_window_cost_at(1.0, 0.5000001, 0.5) {
            Budget::Largest(v) => assert!((v - 1e-7).abs() < 1e-12),
            other => panic!("expected a tiny budget, got {other:?}"),
        }
    }

    #[test]
    fn a_window_that_cannot_reach_the_gate_even_at_full_acceptance_is_marked_infeasible() {
        // W(2) = 1.65 with H(1) = 0.10: at most 2 / 1.75 = 1.142857 even at p = 1.
        match requirement(1, 1.65, 0.10) {
            Requirement::Infeasible { ceiling_speedup } => {
                assert!((ceiling_speedup - 1.142_857_142_857_143).abs() < 1e-12)
            }
            other => panic!("expected an infeasible combination, got {other:?}"),
        }
        // The Qwen3.6 EXL3 chunk entry cannot pay at any head cost.
        for (k, window) in [(1, 3.16), (2, 3.35), (3, 3.69)] {
            assert!(matches!(
                requirement(k, window, 0.0),
                Requirement::Infeasible { .. }
            ));
        }
    }

    #[test]
    fn the_ceiling_equal_to_the_gate_is_reachable_only_at_full_acceptance() {
        // k = 1, W + H = 1.0, gate 2.0: the ceiling is exactly the gate, so it is not infeasible
        assert_eq!(
            requirement_at(2.0, 1, 0.5, 0.5),
            Requirement::Reachable {
                required_tokens: 2.0
            }
        );
        assert!(matches!(
            requirement_at(2.0, 1, 0.5001, 0.5),
            Requirement::Infeasible { .. }
        ));
    }

    #[test]
    fn requirements_match_hand_computed_values() {
        // (k, W, H, required E[accepted + 1]) with the gate at 1.15
        let table = [
            (1, 1.59, 0.10, 1.9435), // 1.15 * 1.69
            (1, 1.40, 0.00, 1.61),   // 1.15 * 1.40
            (2, 1.82, 0.20, 2.323),  // 1.15 * 2.02
            (3, 2.20, 0.30, 2.875),  // 1.15 * 2.50
            (3, 2.70, 0.00, 3.105),  // 1.15 * 2.70
        ];
        for (k, w, h, want) in table {
            match requirement(k, w, h) {
                Requirement::Reachable { required_tokens } => {
                    assert!((required_tokens - want).abs() < 1e-9, "k {k} W {w} H {h}")
                }
                other => panic!("k {k} W {w} H {h}: expected reachable, got {other:?}"),
            }
        }
        // (k, W, H) that cannot reach 1.15: ceilings 2/1.75, 3/2.65, 4/3.57
        for (k, w, h) in [(1, 1.65, 0.10), (2, 2.40, 0.25), (3, 3.27, 0.30)] {
            assert!(matches!(
                requirement(k, w, h),
                Requirement::Infeasible { .. }
            ));
        }
    }
}
