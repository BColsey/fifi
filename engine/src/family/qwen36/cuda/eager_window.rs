//! The eager reference window of the EXL3 step: the model side of a verify window, built from
//! the one-row step so that every row has the arithmetic of a one-row run.
//!
//! `run_window` saves the committed state and feeds every row of the window through `run_token`,
//! keeping the logits of each row. Rows are drawn from those logits. A verdict that keeps every
//! row leaves the state as it is; a verdict that keeps fewer restores the saved state and feeds
//! the kept rows again, which leaves exactly the state a one-row run of those rows would have. The
//! cost is `rows` steps, one export and (after a partial accept) one import: this path proves the
//! accept and rollback rules on the real model, and a multi-row kernel path replaces its body.

use super::{state::export::ExportedState, step::QwenResidentStep};
#[cfg(test)]
use crate::family::speculate::d1_oracle::Fault;
use crate::{
    EngError,
    family::{
        sampler::Sampler,
        session::{TokenStepper, sample_logits},
        speculate::{driver::WindowModel, window::Commit},
    },
};

/// A window that has run and awaits its verdict.
pub(super) struct EagerWindow {
    /// The tokens the window fed, in row order.
    inputs: Vec<u32>,
    /// The logits row after each fed token.
    rows: Vec<Vec<f32>>,
    /// The token the device argmax chose after each fed token: what a one-row greedy run draws.
    argmax: Vec<u32>,
    /// The committed state before the window.
    saved: ExportedState,
}

fn no_window() -> EngError {
    EngError::BadRequest("verify window: no window has run".to_owned())
}

/// Test hooks: the rows a window kept and the fault injection of the D1 oracle.
#[cfg(test)]
impl QwenResidentStep<'_> {
    /// The complete logits row of every row of the window that `run_window` kept.
    pub(super) fn window_logits_rows(&self) -> Result<Vec<Vec<f32>>, EngError> {
        let window = self.window.as_ref().ok_or_else(no_window)?;
        Ok(window.rows.clone())
    }

    /// Fires the armed fault when it is `at`: poisons the state the way a failed step does (the
    /// state then refuses every step and snapshot until a reset or an import) and fails.
    fn fault_at(&mut self, at: Fault) -> Result<(), EngError> {
        if self.window_fault != Some(at) {
            return Ok(());
        }
        self.window_fault = None;
        self.poison_state();
        Err(EngError::BadRequest(format!(
            "verify window: injected fault {at:?}"
        )))
    }

    fn feed_fault(&mut self, row: usize) -> Result<(), EngError> {
        self.fault_at(Fault::Feed(row))
    }

    fn replay_fault(&mut self, row: usize) -> Result<(), EngError> {
        self.fault_at(Fault::Replay(row))
    }
}

/// Outside test builds the fault points are empty inlined calls: no field, no branch.
#[cfg(not(test))]
impl QwenResidentStep<'_> {
    #[inline(always)]
    fn feed_fault(&mut self, _row: usize) -> Result<(), EngError> {
        Ok(())
    }

    #[inline(always)]
    fn replay_fault(&mut self, _row: usize) -> Result<(), EngError> {
        Ok(())
    }
}

impl WindowModel for QwenResidentStep<'_> {
    fn run_window(&mut self, inputs: &[u32]) -> Result<(), EngError> {
        // A window that fails half way leaves rows fed and no verdict to apply: no window is
        // pending, and the caller resets the session.
        self.window = None;
        let saved = self.export_state()?;
        let mut rows = Vec::with_capacity(inputs.len());
        let mut argmax = Vec::with_capacity(inputs.len());
        for (at, &token) in inputs.iter().enumerate() {
            self.feed_fault(at)?;
            argmax.push(self.run_token(token, true)?);
            rows.push(self.logits_values());
        }
        self.window = Some(EagerWindow {
            inputs: inputs.to_vec(),
            rows,
            argmax,
            saved,
        });
        Ok(())
    }

    fn draw(
        &mut self,
        row: usize,
        sampler: &mut Sampler,
        sampler_vocab: usize,
    ) -> Result<u32, EngError> {
        let window = self.window.as_ref().ok_or_else(no_window)?;
        // A greedy request takes the device argmax, as the one-row step does.
        if sampler.is_greedy() {
            return window.argmax.get(row).copied().ok_or_else(no_window);
        }
        let logits = window.rows.get(row).ok_or_else(no_window)?;
        sample_logits(Self::OPERATION, logits, sampler, sampler_vocab)
    }

    fn commit_window(&mut self, commit: &Commit) -> Result<(), EngError> {
        let window = self.window.take().ok_or_else(no_window)?;
        if commit.discarded_rows == 0 {
            return Ok(());
        }
        self.import_state(&window.saved)?;
        for (at, &token) in window.inputs[..commit.kept_rows].iter().enumerate() {
            self.replay_fault(at)?;
            self.run_token(token, false)?;
        }
        Ok(())
    }
}
