//! The decode window driver shared by the executors that step several sessions at once.
//!
//! A window is one call of `ExecutorOps::decode_window`: one session per row, one pending token
//! per session. The policy around the model call is the same for every family, so it lives here
//! once and a family supplies only what differs: how to split a session into its token side and
//! its model side, which rows the model refuses, what one more position costs, and one batched
//! step over rows.
//!
//! Order of a window:
//! 1. The shape is checked: equal nonempty slices within `max_seqs`.
//! 2. A window of one row calls the solo `decode`, so a lone request runs the code it always ran.
//! 3. Every row is validated before any state changes: the token side must hold a pending token
//!    and the family must accept the row (`check_row`). A refusal here is an outer error and no
//!    session has moved.
//! 4. Each row takes its policy decision through `DecodeSession::prepare_decode`: cancellation,
//!    the output limit and decoder tails finish a row without a model step.
//! 5. The rows that still need a step reserve their growth. A memory denial refuses that row only,
//!    before its feed; any other error is an outer error.
//! 6. One row left runs the solo `decode`; two or more run `step_rows` once.
//! 7. Every consumed input is recorded before any result is checked, then each row is sampled
//!    with its own sampler and finished with the policy of a solo token. An error after the step
//!    ran is an outer error: the worker discards the whole window and never publishes a partial
//!    one.

use crate::EngError;
use crate::family::executor::{ExecutorOps, GenerationState};
use crate::family::session::{
    DecodePreparation, DecodeSession, Limits, TokenStepper, sample_logits,
};
use crate::tokenizer::DecodeBatch;
use std::sync::atomic::Ordering;

type Result<T> = std::result::Result<T, EngError>;

/// What a family supplies to run its windows through [`decode_window`].
pub(crate) trait WindowRows: ExecutorOps {
    /// The model side of one session, built per call from the executor and the session.
    type Stepper<'a>: TokenStepper
    where
        Self: 'a;
    /// A growth reservation. It is held while the rows of one window are admitted and dropped
    /// before the step, because the step recharges what it uses.
    type Permit;
    /// The `operation` of the `Backend` errors this family's rows raise.
    const OPERATION: &'static str;

    /// Splits one session into its token side, its model side and the executor's limits.
    fn split<'a>(
        &'a mut self,
        session: &'a mut Self::Session,
    ) -> (&'a mut DecodeSession, Self::Stepper<'a>, Limits<'a>);

    /// The family's validation of one row that will run a model step: the position against the
    /// context limit and the token against the embedding table. It changes nothing.
    fn check_row(&self, session: &Self::Session, token: u32) -> Result<()>;

    /// The bytes one more position of this row costs; zero where the cache is preallocated.
    fn row_growth(&self, session: &Self::Session) -> Result<u64>;

    /// Reserves `bytes` of growth. A `MemoryPlan` error refuses the row only.
    fn reserve_growth(&self, bytes: u64) -> Result<Self::Permit>;

    /// Steps every session once with its pending token and returns one logits row per session,
    /// in order. Called with two or more rows. The rows advance together, so an error leaves
    /// every session in an unknown state and the caller discards them.
    fn step_rows(
        &mut self,
        sessions: &mut [&mut Self::Session],
        tokens: &[u32],
    ) -> Result<Vec<Vec<f32>>>;
}

fn window_error(message: &str) -> EngError {
    EngError::Backend {
        operation: "decode window",
        message: message.to_string(),
    }
}

fn fed_of<F: WindowRows>(family: &mut F, session: &mut F::Session) -> usize {
    let (_, step, _) = family.split(session);
    step.fed()
}

fn pending_of<F: WindowRows>(family: &mut F, session: &mut F::Session) -> Result<u32> {
    let (core, _, _) = family.split(session);
    core.checked_pending()
}

/// Advances a window of sessions by one token each. See the module documentation for the order
/// of the steps and the error rule.
pub(crate) fn decode_window<F: WindowRows>(
    family: &mut F,
    sessions: &mut [&mut F::Session],
    generations: &mut [&mut GenerationState],
) -> Result<Vec<Result<DecodeBatch>>> {
    let rows = sessions.len();
    if rows == 0 || rows != generations.len() || rows > family.max_seqs() {
        return Err(EngError::BadRequest(
            "a decode window needs equal nonempty slices within max_seqs".into(),
        ));
    }
    if rows == 1 {
        return family
            .decode(&mut *sessions[0], &mut *generations[0])
            .map(|batch| vec![Ok(batch)]);
    }
    // Validate every input before cancellation or a decoder tail can change any row.
    for slot in 0..rows {
        let token = pending_of(family, &mut *sessions[slot])?;
        let generation = &generations[slot];
        if generation.cancel.load(Ordering::SeqCst) || generation.remaining_tokens() == Some(0) {
            continue;
        }
        family.check_row(&*sessions[slot], token)?;
    }
    let mut results: Vec<Option<Result<DecodeBatch>>> = (0..rows).map(|_| None).collect();
    let mut stepping = Vec::with_capacity(rows);
    for slot in 0..rows {
        let prepared = {
            let (core, mut step, _) = family.split(&mut *sessions[slot]);
            core.prepare_decode(&mut step, &mut *generations[slot])?
        };
        match prepared {
            DecodePreparation::Finished(batch) => results[slot] = Some(Ok(batch)),
            DecodePreparation::Token(_) => {
                stepping.push((slot, family.row_growth(&*sessions[slot])?));
            }
        }
    }
    // Rows are admitted against cumulative charges; the step recharges what it uses.
    let mut permits = Vec::with_capacity(stepping.len());
    let mut admitted = Vec::with_capacity(stepping.len());
    for (slot, growth) in stepping {
        match family.reserve_growth(growth) {
            Ok(permit) => {
                permits.push(permit);
                admitted.push(slot);
            }
            Err(error @ EngError::MemoryPlan(_)) => results[slot] = Some(Err(error)),
            Err(error) => return Err(error),
        }
    }
    drop(permits);
    match admitted.len() {
        0 => {}
        1 => {
            let slot = admitted[0];
            let batch = family.decode(&mut *sessions[slot], &mut *generations[slot])?;
            results[slot] = Some(Ok(batch));
        }
        _ => step_window(family, sessions, generations, &admitted, &mut results)?,
    }
    results
        .into_iter()
        .map(|row| row.ok_or_else(|| window_error("the window omitted a result row")))
        .collect()
}

/// Runs the batched step for the admitted rows and publishes one token for each.
fn step_window<F: WindowRows>(
    family: &mut F,
    sessions: &mut [&mut F::Session],
    generations: &mut [&mut GenerationState],
    admitted: &[usize],
    results: &mut [Option<Result<DecodeBatch>>],
) -> Result<()> {
    let tokens = admitted
        .iter()
        .map(|&slot| pending_of(family, &mut *sessions[slot]))
        .collect::<Result<Vec<_>>>()?;
    let fed_before: Vec<usize> = admitted
        .iter()
        .map(|&slot| fed_of(family, &mut *sessions[slot]))
        .collect();
    let stepped = {
        let mut rows: Vec<&mut F::Session> = sessions
            .iter_mut()
            .enumerate()
            .filter_map(|(slot, session)| admitted.contains(&slot).then_some(&mut **session))
            .collect();
        family.step_rows(&mut rows, &tokens)
    };
    // Record every consumed input before any result check or sampler can fail.
    for (&slot, &before) in admitted.iter().zip(&fed_before) {
        let after = fed_of(family, &mut *sessions[slot]);
        let (core, _, _) = family.split(&mut *sessions[slot]);
        core.record_feed(before, after);
    }
    let stepped = stepped?;
    if stepped.len() != admitted.len() {
        return Err(window_error("the step returned the wrong row count"));
    }
    let vocab = family.split(&mut *sessions[admitted[0]]).2.sampler_vocab;
    let mut sampled = Vec::with_capacity(admitted.len());
    for (&slot, logits) in admitted.iter().zip(&stepped) {
        let token = sample_logits(F::OPERATION, logits, &mut generations[slot].sampler, vocab)?;
        if usize::try_from(token).map_or(true, |token| token >= vocab) {
            return Err(window_error(
                "a row returned a token outside the sampler vocabulary",
            ));
        }
        sampled.push(token);
    }
    for (&slot, token) in admitted.iter().zip(sampled) {
        let (core, mut step, limits) = family.split(&mut *sessions[slot]);
        let batch = core.finish_decode_token(&mut step, &mut *generations[slot], &limits, token)?;
        results[slot] = Some(Ok(batch));
    }
    Ok(())
}

#[cfg(test)]
#[path = "window_tests.rs"]
mod tests;
