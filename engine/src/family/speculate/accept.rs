//! The acceptance rule of a verify window.
//!
//! The window fed the pending token and `k` drafts, so it holds `k + 1` logits rows: row `i`
//! is the model's view after `i` drafts. The job's sampler draws one token per row, in row
//! order. Draft `i + 1` is accepted exactly when it equals the draw of row `i`; the first row
//! whose draw differs from the next draft ends the run, and that draw is the bonus token. The
//! committed tokens are the draws themselves, so they are the tokens a one-row run would have
//! drawn with the same sampler state: the rule holds for greedy and for any sampler, and needs
//! no rejection sampling.
//!
//! Rows are drawn lazily and in order. A row after the run's end is never drawn, so a seeded
//! sampler advances exactly as far as a one-row run would have advanced it.

use crate::EngError;

/// The tokens a window commits.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Accepted {
    /// The draws of rows `0..tokens.len()`: the accepted drafts, then one token the model chose.
    pub(crate) tokens: Vec<u32>,
    /// True when the last token stops the job (the caller's stop test said so).
    pub(crate) stopped: bool,
}

impl Accepted {
    /// The drafts that were accepted.
    pub(crate) fn drafts_accepted(&self) -> usize {
        self.tokens.len().saturating_sub(1)
    }

    /// The rows of the window that stay committed. Each committed token came from a row whose
    /// input was valid, so the count equals the number of tokens.
    pub(crate) fn rows_kept(&self) -> usize {
        self.tokens.len()
    }
}

/// Runs the rule. `drafts` are the guesses after the pending token; `draw(row)` draws the token
/// of a row and may fail; `stops(token)` says whether a token ends the job; `limit` is the
/// number of tokens the job may still emit (at least 1). The run never emits more than `limit`
/// tokens and never draws past a stop token.
pub(crate) fn accept_run(
    drafts: &[u32],
    limit: usize,
    mut draw: impl FnMut(usize) -> Result<u32, EngError>,
    stops: impl Fn(u32) -> bool,
) -> Result<Accepted, EngError> {
    let mut tokens = Vec::with_capacity(drafts.len() + 1);
    let mut stopped = false;
    for row in 0..=drafts.len() {
        if tokens.len() >= limit {
            break;
        }
        let token = draw(row)?;
        tokens.push(token);
        if stops(token) {
            stopped = true;
            break;
        }
        if drafts.get(row) != Some(&token) {
            break;
        }
    }
    Ok(Accepted { tokens, stopped })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{cell::Cell, rc::Rc};

    /// A sampler stand-in that returns scripted draws and counts how many it served.
    fn script(
        draws: &[u32],
    ) -> (
        impl FnMut(usize) -> Result<u32, EngError> + '_,
        Rc<Cell<usize>>,
    ) {
        let served = Rc::new(Cell::new(0));
        let counter = Rc::clone(&served);
        (
            move |row: usize| {
                counter.set(counter.get() + 1);
                Ok(draws[row])
            },
            served,
        )
    }

    fn never(_: u32) -> bool {
        false
    }

    #[test]
    fn every_draft_accepted_commits_the_draft_and_one_bonus() {
        let (draw, served) = script(&[5, 6, 7, 8]);
        let run = accept_run(&[5, 6, 7], 100, draw, never).expect("run");
        assert_eq!(run.tokens, [5, 6, 7, 8]);
        assert_eq!(run.drafts_accepted(), 3);
        assert_eq!(run.rows_kept(), 4);
        assert_eq!(served.get(), 4);
    }

    #[test]
    fn the_first_mismatch_ends_the_run_with_the_models_own_token() {
        let (draw, served) = script(&[5, 9, 7, 8]);
        let run = accept_run(&[5, 6, 7], 100, draw, never).expect("run");
        assert_eq!(run.tokens, [5, 9]);
        assert_eq!(run.drafts_accepted(), 1);
        assert_eq!(served.get(), 2, "rows after the mismatch are never drawn");
    }

    #[test]
    fn a_first_row_mismatch_is_a_plain_step() {
        let (draw, _) = script(&[1, 2]);
        let run = accept_run(&[9], 100, draw, never).expect("run");
        assert_eq!(run.tokens, [1]);
        assert_eq!(run.drafts_accepted(), 0);
    }

    #[test]
    fn a_window_without_drafts_is_a_plain_step() {
        let (draw, _) = script(&[4]);
        let run = accept_run(&[], 100, draw, never).expect("run");
        assert_eq!(run.tokens, [4]);
    }

    #[test]
    fn a_stop_token_ends_the_run_even_when_the_draft_matches_it() {
        let (draw, served) = script(&[5, 2, 7]);
        let run = accept_run(&[5, 2], 100, draw, |t| t == 2).expect("run");
        assert_eq!(run.tokens, [5, 2]);
        assert!(run.stopped);
        assert_eq!(served.get(), 2);
    }

    #[test]
    fn the_limit_cuts_the_run() {
        let (draw, served) = script(&[5, 6, 7, 8]);
        let run = accept_run(&[5, 6, 7], 2, draw, never).expect("run");
        assert_eq!(run.tokens, [5, 6]);
        assert!(!run.stopped);
        assert_eq!(served.get(), 2);
    }

    #[test]
    fn a_failed_draw_fails_the_run() {
        let result = accept_run(
            &[5, 6],
            100,
            |row| {
                if row == 1 {
                    Err(EngError::BadRequest("draw".into()))
                } else {
                    Ok(5)
                }
            },
            never,
        );
        assert!(result.is_err());
    }

    /// The committed tokens equal a one-row run for any draw sequence: a draw table where each
    /// row's draw depends only on the committed prefix is a stand-in for any sampler.
    #[test]
    fn the_committed_tokens_are_the_draws_a_one_row_run_makes() {
        // The model: the draw after a prefix is (sum of the prefix) % 7.
        let model = |prefix: &[u32]| prefix.iter().sum::<u32>() % 7;
        let mut committed = vec![3u32];
        for _ in 0..40 {
            let mut guess = Vec::new();
            let mut probe = committed.clone();
            for step in 0..3 {
                // A drafter that is right on even steps only.
                let token = model(&probe) + u32::from(step % 2 == 1);
                guess.push(token);
                probe.push(token);
            }
            let base = committed.clone();
            let run = accept_run(
                &guess,
                100,
                |row| {
                    let mut prefix = base.clone();
                    prefix.extend_from_slice(&guess[..row]);
                    Ok(model(&prefix))
                },
                never,
            )
            .expect("run");
            committed.extend_from_slice(&run.tokens);
        }
        let mut reference = vec![3u32];
        while reference.len() < committed.len() {
            let token = model(&reference);
            reference.push(token);
        }
        assert_eq!(committed, reference);
    }
}
