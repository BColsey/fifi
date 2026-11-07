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
//! Device rollback (`draft_rollback = device`). The same rows run, and after each row but the last
//! the DeltaNet state is copied into a slot of a dedicated device buffer (`state/rollback.rs`,
//! bookkeeping in `family/qwen36/rollback.rs`). A verdict that keeps `kept` rows copies slot
//! `kept - 1` back and truncates the KV frontier to `base + kept`; the KV cells above it are stale
//! and unreachable. There is no export, no import and no replay. It is opt-in; the default is the
//! host path above.
//!
//! The lifecycle (which effect runs, in which order, what a failure does) is
//! `family/qwen36/window_logic.rs`, generic over `WindowHost`; this file implements the trait with
//! one-line forwards and keeps the test hooks. Failure contract, both paths:
//!
//! * A failure before any row ran (a refusal at the top of `run_window`, the capacity check, a
//!   failed export, a row refused before the first launch) leaves the state valid and unchanged.
//! * A failure after a row ran, any failure of `commit_window` after its window was taken, and a
//!   verdict that is not the window's poison the state, set the recovery gate (`needs_reset`),
//!   drop a pending prefix save, invalidate the pinned pool and close the window.
//! * A window that has run all its rows and has no verdict yet awaits it. Until then every
//!   consumer of the live state refuses through `state_usable` (a decode step through it also
//!   fails the state closed), and a new window over it poisons the state.
//! * `reset_state` is the only recovery from a failed commit or an unresolved window. It is
//!   exempt from `state_usable`. `import_state` and the pinned-prefix restore are refused while
//!   the gate is set or a window is unresolved (rows ran, no confirmed commit): they never close
//!   such a window silently (DESIGN 17.10).

use super::{state::export::ExportedState, step::QwenResidentStep};
#[cfg(test)]
use crate::family::qwen36::window_logic::{CommitPoint, Mark};
#[cfg(test)]
use crate::family::speculate::d1_oracle::Fault;
use crate::{
    EngError,
    family::{
        qwen36::window_logic::{
            self, RowsFailure, RowsOut, Window, WindowHost, eager_rows, no_window,
        },
        sampler::Sampler,
        session::{TokenStepper, sample_logits},
        speculate::{driver::WindowModel, window::Commit},
    },
};

/// The window of the step: the host copy is an [`ExportedState`], a row is a logits vector.
pub(super) type EagerWindow = Window<ExportedState, Vec<f32>>;

/// Test hooks: the rows a window kept and the fault injection of the D1 oracle.
#[cfg(test)]
impl QwenResidentStep<'_> {
    /// The complete logits row of every row of the window that `run_window` kept.
    pub(super) fn window_logits_rows(&self) -> Result<Vec<Vec<f32>>, EngError> {
        let window = self.window.as_ref().ok_or_else(no_window)?;
        Ok(window.rows.clone())
    }

    /// Fires the armed fault of the capture after row `row` (device rollback).
    fn capture_fault(&mut self, row: usize) -> Result<(), EngError> {
        self.fault_at(Fault::Capture(row))
    }

    fn restore_fault(&mut self) -> Result<(), EngError> {
        self.fault_at(Fault::Restore)
    }

    /// Fires the armed fault when it is `at`: poisons the state the way a failed step does (the
    /// state then refuses every step and snapshot until a reset or an import) and fails.
    pub(super) fn fault_at(&mut self, at: Fault) -> Result<(), EngError> {
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
        self.fault_at(Fault::Replay(row))?;
        self.refusal_at(Fault::ReplayRefused(row))
    }

    /// Fires the armed fault when it is `at`, as a refusal before any device work: it fails and
    /// leaves the state as it is (valid), the way `check_import`, `resident_kv`, a pending save
    /// that cannot finish, or a row refused before launch (`cancel_enqueue`) do.
    pub(super) fn refusal_at(&mut self, at: Fault) -> Result<(), EngError> {
        if self.window_fault != Some(at) {
            return Ok(());
        }
        self.window_fault = None;
        Err(EngError::BadRequest(format!(
            "verify window: injected refusal {at:?}"
        )))
    }

    fn commit_point_fault(&mut self, at: CommitPoint) -> Result<(), EngError> {
        match at {
            CommitPoint::PrefixSave => self.refusal_at(Fault::PrefixSave),
            CommitPoint::CheckImport => self.refusal_at(Fault::CheckImport),
            CommitPoint::ResidentKv => self.refusal_at(Fault::ResidentKv),
            CommitPoint::ImportDevice => self.fault_at(Fault::ImportDevice),
        }
    }
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
    /// The copy after each row but the last (device path): the host time from the row's mark to the
    /// copy's enqueue, or to its completion when `trace_sync_capture` drained the stream.
    pub(super) capture_ms: Vec<f64>,
    /// The drop of the saved host state at the end of `commit_window` (host path).
    pub(super) drop_ms: f64,
    /// Host time spent inside `trace_mark` itself (the clock reads, the event queries, the fault
    /// counter reads): the marks' own cost, so that it is not mistaken for unattributed time.
    pub(super) mark_overhead_ms: f64,
    /// Minor page faults of the process (`/proc/self/stat`) at the marks that sample them.
    pub(super) minflt: Vec<(&'static str, u64)>,
}

/// The process's minor faults now, or `None` where `/proc/self/stat` cannot be read or parsed.
#[cfg(test)]
pub(super) fn minor_faults() -> Option<u64> {
    let stat = std::fs::read_to_string("/proc/self/stat").ok()?;
    crate::family::speculate::window_cost::parse_stat_faults(&stat).map(|(minor, _)| minor)
}

#[cfg(test)]
impl QwenResidentStep<'_> {
    fn trace_mark(&mut self, mark: Mark) {
        let now = std::time::Instant::now();
        let fault_label = match mark {
            Mark::WindowStart => Some("window_start"),
            Mark::Exported => Some("exported"),
            Mark::CommitStart => Some("commit_start"),
            Mark::Dropped => Some("dropped"),
            _ => None,
        };
        let faults = fault_label.and_then(|label| Some((label, minor_faults()?)));
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
            Mark::Captured => trace.capture_ms.push(since),
            Mark::Dropped => trace.drop_ms = since,
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
        trace.minflt.extend(faults);
        let end = std::time::Instant::now();
        // The clock restarts after the mark's own work, which is accounted apart.
        trace.mark_overhead_ms += (end - now).as_secs_f64() * 1e3;
        trace.clock = Some(end);
    }
}

impl WindowModel for QwenResidentStep<'_> {
    fn run_window(&mut self, inputs: &[u32]) -> Result<(), EngError> {
        window_logic::run_window(self, inputs)
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
        window_logic::commit_window(self, commit)
    }
}

/// The effects of a window on the step: one forward each (`window_logic` holds the decisions).
impl WindowHost for QwenResidentStep<'_> {
    type Saved = ExportedState;
    type Row = Vec<f32>;

    fn device_wanted(&self) -> bool {
        self.device_rollback_wanted()
    }

    fn tokens(&self) -> usize {
        QwenResidentStep::tokens(self)
    }

    fn store_window(&mut self, window: EagerWindow) {
        self.window = Some(window);
    }

    fn take_window(&mut self) -> Option<EagerWindow> {
        self.window.take()
    }

    fn export(&mut self) -> Result<ExportedState, EngError> {
        self.export_state()
    }

    fn run_row(&mut self, token: u32) -> Result<(u32, Vec<f32>), EngError> {
        let drawn = self.run_token(token, true)?;
        Ok((drawn, self.logits_values()))
    }

    fn replay_row(&mut self, token: u32) -> Result<(), EngError> {
        self.run_token(token, false).map(|_| ())
    }

    fn overwrite(&mut self, saved: &ExportedState) -> Result<(), EngError> {
        self.overwrite_state(saved)
    }

    fn window_awaits(&self) -> bool {
        self.window_is_open()
    }

    fn capacity(&self) -> usize {
        self.kv_capacity_rows()
    }

    fn is_valid(&self) -> bool {
        self.state_is_valid()
    }

    fn needs_reset(&self) -> bool {
        self.recovery_gate_set()
    }

    fn set_needs_reset(&mut self) {
        self.set_recovery_gate();
    }

    fn clear_needs_reset(&mut self) {
        self.clear_recovery_gate();
    }

    fn poison(&mut self) {
        self.poison_window_state();
    }

    fn drop_pending_prefix(&mut self) {
        QwenResidentStep::drop_pending_prefix(self);
    }

    fn invalidate_pool(&mut self) {
        self.invalidate_prefix_pool();
    }

    fn close_window(&mut self) {
        self.close_stored_window();
    }

    fn check_import(&mut self, saved: &ExportedState) -> Result<(), EngError> {
        self.check_saved_state(saved)
    }

    fn resident_kv(&mut self) -> Result<(), EngError> {
        self.check_resident_kv()
    }

    fn require_armed(&mut self) -> Result<(), EngError> {
        self.require_armed_for_window()
    }

    fn complete_prefix_save(&mut self) -> Result<(), EngError> {
        QwenResidentStep::complete_prefix_save(self)
    }

    fn rollback_begin(&mut self, rows: usize) -> Result<(), EngError> {
        QwenResidentStep::rollback_begin(self, rows)
    }

    fn capture(&mut self, row: usize) -> Result<(), EngError> {
        self.rollback_capture(row)?;
        #[cfg(test)]
        if self.trace_sync_capture {
            self.drain_stream()?;
        }
        Ok(())
    }

    /// A K-row window (`draft_verify = krow`, armed) runs the K-row body; every other window runs
    /// the eager loop, the default of the trait.
    fn run_rows(&mut self, inputs: &[u32]) -> Result<RowsOut<Vec<f32>>, RowsFailure> {
        if self.krow_runs(inputs.len()) {
            return self.run_krow_window(inputs);
        }
        eager_rows(self, inputs).map_err(RowsFailure::StateDecides)
    }

    fn rollback_discard(&mut self) {
        QwenResidentStep::rollback_discard(self);
    }

    fn rollback_window_complete(&mut self) -> Result<(), EngError> {
        QwenResidentStep::rollback_window_complete(self)
    }

    fn rollback_finish(&mut self, kept: usize, committed_rows: usize) -> Result<bool, EngError> {
        QwenResidentStep::rollback_finish(self, kept, committed_rows)
    }

    #[cfg(test)]
    fn trace_mark(&mut self, mark: Mark) {
        QwenResidentStep::trace_mark(self, mark);
    }

    #[cfg(test)]
    fn feed_fault(&mut self, row: usize) -> Result<(), EngError> {
        QwenResidentStep::feed_fault(self, row)
    }

    #[cfg(test)]
    fn replay_fault(&mut self, row: usize) -> Result<(), EngError> {
        QwenResidentStep::replay_fault(self, row)
    }

    #[cfg(test)]
    fn commit_fault(&mut self, at: CommitPoint) -> Result<(), EngError> {
        self.commit_point_fault(at)
    }

    #[cfg(test)]
    fn capture_fault(&mut self, row: usize) -> Result<(), EngError> {
        QwenResidentStep::capture_fault(self, row)
    }

    #[cfg(test)]
    fn restore_fault(&mut self) -> Result<(), EngError> {
        QwenResidentStep::restore_fault(self)
    }
}
