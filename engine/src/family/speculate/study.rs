//! Host side of the offline MTP acceptance study (test-only, no device).
//!
//! The study runs the target greedily and records, at every pending position `m`, the chain of
//! drafts the head proposes. This module scores such a trace the way a serving loop would: a
//! window for pending token `s_m` feeds `1 + k` rows, draws `s_{m+1}..`, keeps drafts while they
//! equal the draw (`accept_run`), and the next window starts at the last committed token. Every
//! window costs its full `k + 1` rows, whatever the recorded stream does after it, so no recorded
//! future length shortens a draft or a window; a stream that stopped early ends the window at the
//! stop token exactly as `accept_run` does.
//!
//! The decision rule is fixed in the protocol: `speedup = E[accepted + 1] / (W(k + 1) + H(k))`
//! against a gate of 1.15, with `H(k)` the cost of drafting the k-token chain. A window and head
//! cost whose ceiling `(k + 1) / (W + H)` is below the gate is marked infeasible instead of being
//! given an acceptance above 100 percent. The Python collector in `qualification/mtp-acceptance/`
//! scores the same traces and uses the same golden vectors.

use super::accept::accept_run;
use crate::EngError;

/// End-to-end gate Dan set on 2026-11-05.
pub(crate) const GATE: f64 = 1.15;

/// The target's greedy stream and the head's chains over it.
pub(crate) struct Trace<'a> {
    /// Tokens `s_0 ..` the target drew. Holds at least `scored + max_k + 1` tokens unless it ends
    /// with a stop token.
    pub(crate) stream: &'a [u32],
    /// `chains[m][j - 1]` is the head's draft for `s_{m + j}`, with `s_m` pending.
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

/// Renewing windows with `k` drafts over the trace, in order.
pub(crate) fn renewing_windows(
    trace: &Trace<'_>,
    k: usize,
    stops: &dyn Fn(u32) -> bool,
) -> Result<Vec<WindowOutcome>, EngError> {
    let mut out = Vec::new();
    let mut pending = 0;
    while pending < trace.scored {
        let drafts = trace.chains.get(pending).map(Vec::as_slice).unwrap_or(&[]);
        let drafts = &drafts[..k.min(drafts.len())];
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
    /// The chain alone costs at least `E / GATE`: no window cost pays.
    Infeasible,
}

pub(crate) fn max_window_cost(mean_tokens: f64, draft_cost: f64) -> Budget {
    let budget = mean_tokens / GATE - draft_cost;
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
    let ceiling_speedup = (k + 1) as f64 / (window_cost + draft_cost);
    if ceiling_speedup < GATE {
        return Requirement::Infeasible { ceiling_speedup };
    }
    Requirement::Reachable {
        required_tokens: GATE * (window_cost + draft_cost),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const STOP: u32 = 248_046;

    fn not_a_stop(_: u32) -> bool {
        false
    }

    /// Same vector as `GOLDEN_*` in `analyze.py`.
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
    fn the_registered_example_needs_a_window_of_1_52_solo_steps() {
        // k = 1, per-position acceptance 0.75, free head: E = 1.75, so W(2) <= 1.75 / 1.15.
        let Budget::Largest(limit) = max_window_cost(1.75, 0.0) else {
            panic!("a free head leaves a budget");
        };
        assert!((limit - 1.521_739).abs() < 1e-6);
        assert!((speedup(1.75, limit, 0.0) - GATE).abs() < 1e-12);
    }

    #[test]
    fn drafting_cost_comes_off_the_window_budget_one_for_one() {
        let (Budget::Largest(free), Budget::Largest(paid)) =
            (max_window_cost(1.9, 0.0), max_window_cost(1.9, 0.10))
        else {
            panic!("both leave a budget");
        };
        assert!((free - paid - 0.10).abs() < 1e-12);
        let Budget::Largest(chain) = max_window_cost(2.5, 3.0 * 0.1) else {
            panic!("a three-step chain leaves a budget");
        };
        assert!((chain - (2.5 / GATE - 0.3)).abs() < 1e-12);
    }

    #[test]
    fn a_chain_that_eats_the_whole_budget_is_infeasible() {
        assert_eq!(max_window_cost(1.2, 1.1), Budget::Infeasible);
    }

    #[test]
    fn a_window_that_cannot_reach_the_gate_even_at_full_acceptance_is_marked_infeasible() {
        // W(2) = 1.65 with H(1) = 0.10: at most 2 / 1.75 = 1.143x even at p = 1.
        match requirement(1, 1.65, 0.10) {
            Requirement::Infeasible { ceiling_speedup } => {
                assert!((ceiling_speedup - 2.0 / 1.75).abs() < 1e-12)
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
    fn a_reachable_window_never_asks_for_more_than_full_acceptance() {
        for k in 1..=3usize {
            for window in [1.2, 1.4, 1.59, 1.82, 2.2, 2.7] {
                for draft in [0.0, 0.05, 0.10, 0.15] {
                    let h = draft * k as f64;
                    match requirement(k, window, h) {
                        Requirement::Reachable { required_tokens } => {
                            assert!(
                                required_tokens <= (k + 1) as f64 + 1e-12,
                                "k {k} W {window} H {h}"
                            )
                        }
                        Requirement::Infeasible { ceiling_speedup } => {
                            assert!(ceiling_speedup < GATE)
                        }
                    }
                }
            }
        }
        let Requirement::Reachable { required_tokens } = requirement(1, 1.59, 0.10) else {
            panic!("W(2) = 1.59 with H = 0.10 is reachable");
        };
        assert!((required_tokens - 1.15 * 1.69).abs() < 1e-12);
    }
}
