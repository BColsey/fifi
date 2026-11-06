//! The eager reference window of the EXL3 step: the model side of a verify window, built from
//! the one-row step so that every row has the arithmetic of a one-row run.
//!
//! `run_window` saves the committed state and feeds every row of the window through `run_token`,
//! keeping the logits of each row. Rows are drawn from those logits. A verdict that keeps every
//! row leaves the state as it is; a verdict that keeps fewer restores the saved state and feeds
//! the kept rows again, which leaves exactly the state a one-row run of those rows would have. The
//! cost is `rows` steps, one export and (after a partial accept) one import: this path proves the
//! accept and rollback rules on the real model, and a multi-row kernel path replaces its body.
//!
//! Failure. A step that fails inside the window poisons the state, so the window ends with an
//! error, no pending window and a state that refuses to go on until a reset or an import. A
//! `commit_window` that fails earlier, in `complete_prefix_save` (`step.rs:1326`), `check_import`
//! or `resident_kv` (`state/export.rs:196-197`), poisons nothing: the window has already been
//! taken, and the state is the valid state at the end of the window. That predates this module and
//! is unchanged; the worker's reset after any executor error (`sched/worker.rs:1049`, the policy
//! at `executor.rs:17`) is what makes it safe. The D1 oracle's check F does not model it.

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

/// Outside test builds the fault points and the trace marks are empty inlined calls: no field, no
/// branch, no clock read.
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

    #[inline(always)]
    fn trace_mark(&mut self, _mark: Mark) {}
}

/// The points of a window at which the trace of a test build reads the clock.
#[derive(Clone, Copy)]
enum Mark {
    /// `run_window` begins.
    WindowStart,
    /// The pre-window state has been copied to the host.
    Exported,
    /// A window row has been fed and its logits row read back.
    RowDone,
    /// `commit_window` begins a rollback.
    CommitStart,
    /// The saved state has been written back.
    Imported,
    /// A kept row has been fed again.
    ReplayDone,
}

/// What one window cost on the host clock, by part, and what the device timeline says of each row.
/// Filled by the marks of a test build for the cost harness (`window_bench_device_tests`); a
/// serving build has no such field. The host times are wall times between two marks, so a row's
/// time includes its launch, the wait for the device and the logits read-back. The GPU time of a
/// row is the in-graph event pair of the step (`section_marks`), absent when the marks are off.
#[cfg(test)]
#[derive(Clone, Debug, Default)]
pub(super) struct WindowTrace {
    clock: Option<std::time::Instant>,
    pub(super) export_ms: f64,
    pub(super) row_ms: Vec<f64>,
    pub(super) row_gpu_ms: Vec<Option<f32>>,
    pub(super) import_ms: f64,
    pub(super) replay_ms: Vec<f64>,
    pub(super) replay_gpu_ms: Vec<Option<f32>>,
}

#[cfg(test)]
impl QwenResidentStep<'_> {
    fn trace_mark(&mut self, mark: Mark) {
        let now = std::time::Instant::now();
        let gpu = match mark {
            Mark::RowDone | Mark::ReplayDone => {
                self.section_times().ok().flatten().map(|t| t.whole_ms())
            }
            _ => None,
        };
        let trace = &mut self.window_trace;
        let since = trace
            .clock
            .map_or(0.0, |clock| (now - clock).as_secs_f64() * 1e3);
        match mark {
            Mark::WindowStart => *trace = WindowTrace::default(),
            Mark::Exported => trace.export_ms = since,
            Mark::RowDone => {
                trace.row_ms.push(since);
                trace.row_gpu_ms.push(gpu);
            }
            Mark::CommitStart => {
                trace.import_ms = 0.0;
                trace.replay_ms.clear();
                trace.replay_gpu_ms.clear();
            }
            Mark::Imported => trace.import_ms = since,
            Mark::ReplayDone => {
                trace.replay_ms.push(since);
                trace.replay_gpu_ms.push(gpu);
            }
        }
        trace.clock = Some(std::time::Instant::now());
    }
}

impl WindowModel for QwenResidentStep<'_> {
    fn run_window(&mut self, inputs: &[u32]) -> Result<(), EngError> {
        // A window that fails half way leaves rows fed and no verdict to apply: no window is
        // pending, and the caller resets the session.
        self.window = None;
        self.trace_mark(Mark::WindowStart);
        let saved = self.export_state()?;
        self.trace_mark(Mark::Exported);
        let mut rows = Vec::with_capacity(inputs.len());
        let mut argmax = Vec::with_capacity(inputs.len());
        for (at, &token) in inputs.iter().enumerate() {
            self.feed_fault(at)?;
            argmax.push(self.run_token(token, true)?);
            rows.push(self.logits_values());
            self.trace_mark(Mark::RowDone);
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
        self.trace_mark(Mark::CommitStart);
        self.import_state(&window.saved)?;
        self.trace_mark(Mark::Imported);
        for (at, &token) in window.inputs[..commit.kept_rows].iter().enumerate() {
            self.replay_fault(at)?;
            self.run_token(token, false)?;
            self.trace_mark(Mark::ReplayDone);
        }
        Ok(())
    }
}
