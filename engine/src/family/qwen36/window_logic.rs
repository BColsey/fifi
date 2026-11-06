//! The window lifecycle of the Qwen3.6 verify window, as host-compiled generic code.
//!
//! The step (`cuda/eager_window.rs`) forwards every primitive effect of a window (feed a row,
//! export the state, import it, copy a slot, poison, close the window, ...) to [`WindowHost`]. The
//! decisions live here: which effect runs, in which order, and what a failure does. A host test
//! implements [`WindowHost`] with a recording fake whose effects fail on demand, so the real
//! decision code is tested on the host and a deleted hook turns a test red.
//!
//! The trace marks and the fault points the cost harness and the D1 oracle use are trait methods
//! with empty default bodies: only the step's test build overrides them, so a serving build adds
//! no call.

use crate::{EngError, family::speculate::window::Commit};

/// The points of a window at which the trace of a test build reads the clock.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Mark {
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
    /// The device path has copied the state after a row into its slot.
    Captured,
    /// The saved host state has been dropped (the free or unmap of its buffers).
    Dropped,
}

/// A window that has run and awaits its verdict. `S` is the host copy of the committed state
/// (`ExportedState` on the step) and `R` one logits row.
pub(crate) struct Window<S, R> {
    /// The tokens the window fed, in row order.
    pub(crate) inputs: Vec<u32>,
    /// The logits row after each fed token.
    pub(crate) rows: Vec<R>,
    /// The token the device argmax chose after each fed token: what a one-row greedy run draws.
    pub(crate) argmax: Vec<u32>,
    /// How the committed state before the window is recovered on a partial accept.
    pub(crate) saved: WindowSave<S>,
}

/// What a window keeps to roll back with.
pub(crate) enum WindowSave<S> {
    /// The default: the committed state copied to the host; a partial accept imports it and feeds
    /// the kept rows again.
    Host(S),
    /// `draft_rollback = device`: per-row slots in device memory; a partial accept copies one back
    /// and truncates the KV frontier, with no replay.
    Device,
}

/// The primitive effects of a window. Every method is one forward on the step.
pub(crate) trait WindowHost {
    /// The host copy of the committed state.
    type Saved;
    /// One logits row.
    type Row;

    /// Whether the settings asked for the device rollback.
    fn device_wanted(&self) -> bool;
    /// Committed rows now.
    fn tokens(&self) -> usize;
    /// Forgets the stored window.
    fn clear_window(&mut self);
    /// Stores the window that awaits its verdict.
    fn store_window(&mut self, window: Window<Self::Saved, Self::Row>);
    /// Takes the stored window, if any.
    fn take_window(&mut self) -> Option<Window<Self::Saved, Self::Row>>;
    /// Copies the committed state to the host.
    fn export(&mut self) -> Result<Self::Saved, EngError>;
    /// Feeds one row and returns the device argmax and the logits row.
    fn run_row(&mut self, token: u32) -> Result<(u32, Self::Row), EngError>;
    /// Feeds a kept row again, without logits.
    fn replay_row(&mut self, token: u32) -> Result<(), EngError>;
    /// Replaces the committed state with `saved`.
    fn import(&mut self, saved: &Self::Saved) -> Result<(), EngError>;
    /// Refuses a device window on a step that holds no slots.
    fn require_armed(&mut self) -> Result<(), EngError>;
    /// Finishes a pending pinned-prefix save.
    fn complete_prefix_save(&mut self) -> Result<(), EngError>;
    /// Opens a device window over `rows` rows.
    fn rollback_begin(&mut self, rows: usize) -> Result<(), EngError>;
    /// Copies the state after row `row` into its slot.
    fn capture(&mut self, row: usize) -> Result<(), EngError>;
    /// Drops an open device window and its slot marks.
    fn rollback_discard(&mut self);
    /// Poisons the state.
    fn rollback_poison(&mut self);
    /// Marks that every row ran.
    fn rollback_window_complete(&mut self) -> Result<(), EngError>;
    /// Applies the verdict through the slots; returns whether a slot was copied back.
    fn rollback_finish(&mut self, kept: usize, committed_rows: usize) -> Result<bool, EngError>;

    /// A trace mark. Empty outside the step's test build.
    fn trace_mark(&mut self, _mark: Mark) {}
    /// The fault point of feeding row `row`. Empty outside the step's test build.
    fn feed_fault(&mut self, _row: usize) -> Result<(), EngError> {
        Ok(())
    }
    /// The fault point of replaying kept row `row`.
    fn replay_fault(&mut self, _row: usize) -> Result<(), EngError> {
        Ok(())
    }
    /// The fault point of the capture after row `row`.
    fn capture_fault(&mut self, _row: usize) -> Result<(), EngError> {
        Ok(())
    }
    /// The fault point of the device restore.
    fn restore_fault(&mut self) -> Result<(), EngError> {
        Ok(())
    }
}

pub(crate) fn no_window() -> EngError {
    EngError::BadRequest("verify window: no window has run".to_owned())
}

/// Runs the rows of a window (the model side of `WindowModel::run_window`).
pub(crate) fn run_window<H: WindowHost>(host: &mut H, inputs: &[u32]) -> Result<(), EngError> {
    // A window that fails half way leaves rows fed and no verdict to apply: no window is
    // pending, and the caller resets the session.
    host.clear_window();
    host.trace_mark(Mark::WindowStart);
    if host.device_wanted() {
        return run_window_device(host, inputs);
    }
    let saved = host.export()?;
    host.trace_mark(Mark::Exported);
    let mut rows = Vec::with_capacity(inputs.len());
    let mut argmax = Vec::with_capacity(inputs.len());
    for (at, &token) in inputs.iter().enumerate() {
        host.feed_fault(at)?;
        let (drawn, row) = host.run_row(token)?;
        argmax.push(drawn);
        rows.push(row);
        host.trace_mark(Mark::RowDone);
    }
    host.store_window(Window {
        inputs: inputs.to_vec(),
        rows,
        argmax,
        saved: WindowSave::Host(saved),
    });
    Ok(())
}

/// Applies the verdict (the model side of `WindowModel::commit_window`).
pub(crate) fn commit_window<H: WindowHost>(host: &mut H, commit: &Commit) -> Result<(), EngError> {
    let window = host.take_window().ok_or_else(no_window)?;
    let saved = match window.saved {
        WindowSave::Host(saved) => saved,
        WindowSave::Device => return commit_window_device(host, commit),
    };
    if commit.discarded_rows == 0 {
        #[cfg(test)]
        {
            drop(saved);
            host.trace_mark(Mark::Dropped);
        }
        return Ok(());
    }
    host.trace_mark(Mark::CommitStart);
    host.import(&saved)?;
    host.trace_mark(Mark::Imported);
    for (at, &token) in window.inputs[..commit.kept_rows].iter().enumerate() {
        host.replay_fault(at)?;
        host.replay_row(token)?;
        host.trace_mark(Mark::ReplayDone);
    }
    #[cfg(test)]
    {
        drop(saved);
        host.trace_mark(Mark::Dropped);
    }
    Ok(())
}

/// Runs the rows and copies the state after each row but the last into its slot. Refused before
/// any state change when the slots are not armed, a window is still open or the state is not
/// valid and idle. After the first row ran, any failure poisons the state and drops the window:
/// the state is ahead of the lineage and no verdict can come.
fn run_window_device<H: WindowHost>(host: &mut H, inputs: &[u32]) -> Result<(), EngError> {
    host.require_armed()?;
    // Device branch only: a pending prefix save finishes before the window mutates anything,
    // and a failure here precedes every mutation.
    host.complete_prefix_save()?;
    let base = host.tokens();
    host.rollback_begin(inputs.len())?;
    let mut rows = Vec::with_capacity(inputs.len());
    let mut argmax = Vec::with_capacity(inputs.len());
    let ran = (|| -> Result<(), EngError> {
        for (at, &token) in inputs.iter().enumerate() {
            host.feed_fault(at)?;
            let (drawn, row) = host.run_row(token)?;
            argmax.push(drawn);
            rows.push(row);
            host.trace_mark(Mark::RowDone);
            if at + 1 < inputs.len() {
                host.capture_fault(at)?;
                host.capture(at)?;
                host.trace_mark(Mark::Captured);
            }
        }
        Ok(())
    })();
    if let Err(error) = ran {
        host.rollback_discard();
        if host.tokens() != base {
            host.rollback_poison();
        }
        return Err(error);
    }
    // Every row ran: until the verdict a step or an export through the state is refused.
    host.rollback_window_complete()?;
    host.store_window(Window {
        inputs: inputs.to_vec(),
        rows,
        argmax,
        saved: WindowSave::Device,
    });
    Ok(())
}

/// Applies the verdict through the slots. Any failure leaves the state refusing (poisoned, or
/// already poisoned and refused by the guard) and the window closed; no failure leaves a valid
/// state that is ahead of the verdict.
fn commit_window_device<H: WindowHost>(host: &mut H, commit: &Commit) -> Result<(), EngError> {
    host.trace_mark(Mark::CommitStart);
    if commit.discarded_rows > 0
        && let Err(error) = host.restore_fault()
    {
        host.rollback_discard();
        return Err(error);
    }
    if host.rollback_finish(commit.kept_rows, commit.committed_rows)? {
        host.trace_mark(Mark::Imported);
    }
    Ok(())
}

#[cfg(test)]
#[path = "window_logic_tests.rs"]
mod tests;
