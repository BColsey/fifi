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
//! The decision rule is fixed in the protocol: `speedup = E[accepted + 1] / (W(k + 1) + k *
//! c_head)` against a gate of 1.15. The Python reference with the same golden vectors is
//! `analyze.py` in the study's scratch directory.

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

/// `E[accepted + 1] / (W(k + 1) + k * c_head)`, all costs in solo decode steps.
pub(crate) fn speedup(mean_tokens: f64, window_cost: f64, k: usize, c_head: f64) -> f64 {
    mean_tokens / (window_cost + k as f64 * c_head)
}

/// The largest `W(k + 1)` that still reaches `GATE`.
pub(crate) fn max_window_cost(mean_tokens: f64, k: usize, c_head: f64) -> f64 {
    mean_tokens / GATE - k as f64 * c_head
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
        let limit = max_window_cost(1.75, 1, 0.0);
        assert!((limit - 1.521_739).abs() < 1e-6);
        assert!((speedup(1.75, limit, 1, 0.0) - GATE).abs() < 1e-12);
    }

    #[test]
    fn a_head_step_costs_window_budget_one_for_one() {
        let free = max_window_cost(1.9, 1, 0.0);
        let paid = max_window_cost(1.9, 1, 0.10);
        assert!((free - paid - 0.10).abs() < 1e-12);
        // k chained steps pay k times.
        assert!((max_window_cost(2.5, 3, 0.1) - (2.5 / GATE - 0.3)).abs() < 1e-12);
    }
}
