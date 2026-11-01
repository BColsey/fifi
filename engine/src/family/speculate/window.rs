//! The bookkeeping of one verify window: which rows it feeds and what a verdict keeps.
//!
//! A window starts at `base_rows` committed rows and feeds `1 + k` rows: the pending token, then
//! `k` drafts. Row `j` of the window holds the model's state after `j + 1` fed tokens. When the
//! verdict keeps `n` tokens, rows `0..n` stay committed and the rest are discarded:
//!
//! * the attention cache keeps `base_rows + n` rows; later rows are overwritten by the next
//!   window, because the cache is positional;
//! * the recurrent layers (gated delta net state and convolution tail) continue from the state
//!   that the window's scan left after row `n - 1`, which is the state slot `n - 1`.

use super::accept::Accepted;
use crate::EngError;

/// One window.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Window {
    base_rows: usize,
    drafts: Vec<u32>,
    /// The drafter's confidence score of each draft; empty when the drafter has none.
    scores: Vec<f32>,
}

/// What a verdict does to the model's state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Commit {
    /// Committed rows after the window.
    pub(crate) committed_rows: usize,
    /// Rows the window fed that stay committed.
    pub(crate) kept_rows: usize,
    /// Rows the window fed that are discarded.
    pub(crate) discarded_rows: usize,
    /// The recurrent state slot that becomes the session state.
    pub(crate) state_slot: usize,
}

impl Window {
    pub(crate) fn new(base_rows: usize, drafts: Vec<u32>, scores: Vec<f32>) -> Self {
        Self {
            base_rows,
            drafts,
            scores,
        }
    }

    pub(crate) fn drafts(&self) -> &[u32] {
        &self.drafts
    }

    pub(crate) fn scores(&self) -> &[f32] {
        &self.scores
    }

    /// Rows the window feeds.
    pub(crate) fn rows(&self) -> usize {
        1 + self.drafts.len()
    }

    /// The tokens the window feeds, in row order.
    pub(crate) fn input(&self, pending: u32) -> Vec<u32> {
        let mut rows = Vec::with_capacity(self.rows());
        rows.push(pending);
        rows.extend_from_slice(&self.drafts);
        rows
    }

    /// The state change that the verdict `accepted` asks for. A verdict that keeps no row, or
    /// more rows than the window fed, did not come from this window and is refused.
    pub(crate) fn commit(&self, accepted: &Accepted) -> Result<Commit, EngError> {
        let kept_rows = accepted.rows_kept();
        if kept_rows == 0 || kept_rows > self.rows() {
            return Err(EngError::BadRequest(format!(
                "verify window: a verdict keeps {kept_rows} of {} rows",
                self.rows()
            )));
        }
        Ok(Commit {
            committed_rows: self.base_rows + kept_rows,
            kept_rows,
            discarded_rows: self.rows() - kept_rows,
            state_slot: kept_rows - 1,
        })
    }
}

/// How many drafts a window may carry: the setting's ceiling, held so that the job can still emit
/// every token the window might commit and the cache can still hold every row it feeds.
/// `remaining_tokens` is the job's output room (`None` for none); `context_rows` is the cache's
/// capacity in rows.
pub(crate) fn drafts_allowed(
    ceiling: usize,
    committed_rows: usize,
    context_rows: usize,
    remaining_tokens: Option<u32>,
) -> usize {
    let cache_room = context_rows.saturating_sub(committed_rows + 1);
    let output_room = remaining_tokens.map_or(usize::MAX, |left| (left as usize).saturating_sub(1));
    ceiling.min(cache_room).min(output_room)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn accepted(tokens: &[u32]) -> Accepted {
        Accepted {
            tokens: tokens.to_vec(),
            stopped: false,
        }
    }

    #[test]
    fn the_window_feeds_the_pending_token_then_the_drafts() {
        let window = Window::new(100, vec![7, 8], vec![]);
        assert_eq!(window.rows(), 3);
        assert_eq!(window.input(5), [5, 7, 8]);
    }

    #[test]
    fn a_full_accept_keeps_every_row_and_the_last_state_slot() {
        let window = Window::new(100, vec![7, 8], vec![]);
        let commit = window.commit(&accepted(&[7, 8, 9])).expect("commit");
        assert_eq!(
            commit,
            Commit {
                committed_rows: 103,
                kept_rows: 3,
                discarded_rows: 0,
                state_slot: 2
            }
        );
    }

    #[test]
    fn a_first_row_mismatch_keeps_one_row_and_state_slot_zero() {
        let window = Window::new(100, vec![7, 8], vec![]);
        let commit = window.commit(&accepted(&[4])).expect("commit");
        assert_eq!(
            commit,
            Commit {
                committed_rows: 101,
                kept_rows: 1,
                discarded_rows: 2,
                state_slot: 0
            }
        );
    }

    #[test]
    fn a_partial_accept_discards_the_tail() {
        let window = Window::new(10, vec![1, 2, 3], vec![]);
        let commit = window.commit(&accepted(&[1, 2, 6])).expect("commit");
        assert_eq!(
            (commit.kept_rows, commit.discarded_rows, commit.state_slot),
            (3, 1, 2)
        );
    }

    #[test]
    fn a_verdict_that_does_not_fit_the_window_is_refused() {
        let window = Window::new(10, vec![1, 2], vec![]);
        assert!(window.commit(&accepted(&[])).is_err(), "no row kept");
        assert!(
            window.commit(&accepted(&[1, 2, 3, 4])).is_err(),
            "more rows than fed"
        );
        assert!(window.commit(&accepted(&[1, 2, 3])).is_ok());
    }

    #[test]
    fn the_drafts_are_held_to_the_cache_and_the_output_room() {
        assert_eq!(drafts_allowed(4, 10, 100, None), 4);
        assert_eq!(
            drafts_allowed(4, 10, 100, Some(3)),
            2,
            "three tokens left: two drafts"
        );
        assert_eq!(drafts_allowed(4, 10, 100, Some(1)), 0);
        assert_eq!(drafts_allowed(4, 10, 100, Some(0)), 0);
        assert_eq!(drafts_allowed(4, 97, 100, None), 2, "rows 97..=99 are free");
        assert_eq!(drafts_allowed(4, 99, 100, None), 0);
        assert_eq!(drafts_allowed(4, 120, 100, None), 0);
    }
}
