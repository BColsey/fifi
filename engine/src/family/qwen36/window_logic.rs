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

use super::rollback::check_window_fits;
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

/// A point of the host commit at which the step's test build can inject a failure. A fault at
/// one of the first three is a refusal before `begin_import` (the state stays valid); a fault at
/// `ImportDevice` is a device failure after it (the state is poisoned).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CommitPoint {
    /// `complete_prefix_save` of the commit.
    PrefixSave,
    /// `check_import` of the saved state.
    CheckImport,
    /// The resident-KV check of the import.
    ResidentKv,
    /// A device write of the import, after `begin_import`.
    ImportDevice,
}

/// A window that has run and awaits its verdict. `S` is the host copy of the committed state
/// (`ExportedState` on the step) and `R` one logits row.
pub(crate) struct Window<S, R> {
    /// The committed rows when the window began.
    pub(crate) base: usize,
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
    /// The device writes of an import: the state replaced by `saved`, the sparse sketch rewound,
    /// the pinned pool invalidated. Everything that can be refused ran before it.
    fn overwrite(&mut self, saved: &Self::Saved) -> Result<(), EngError>;
    /// Refuses a device window on a step that holds no slots.
    fn require_armed(&mut self) -> Result<(), EngError>;
    /// Finishes a pending pinned-prefix save.
    fn complete_prefix_save(&mut self) -> Result<(), EngError>;
    /// Whether a window awaits its verdict: a stored window, host or device.
    fn window_awaits(&self) -> bool;
    /// Rows the KV cache can hold.
    fn capacity(&self) -> usize;
    /// Whether the state is `Valid`.
    fn is_valid(&self) -> bool;
    /// The recovery gate: a failed or indeterminate commit set it; only a reset clears it.
    fn needs_reset(&self) -> bool;
    /// Sets the recovery gate.
    fn set_needs_reset(&mut self);
    /// Clears the recovery gate. Only a completed reset does.
    fn clear_needs_reset(&mut self);
    /// Poisons the state (the state then refuses every step and snapshot).
    fn poison(&mut self);
    /// Drops a pending pinned-prefix save without finishing it.
    fn drop_pending_prefix(&mut self);
    /// Invalidates the pinned prefix pool: no later save recycles a slot of the failed lineage.
    fn invalidate_pool(&mut self);
    /// Closes the stored window, host and device.
    fn close_window(&mut self);
    /// The import's layout and identity check of `saved`, before any device write.
    fn check_import(&mut self, saved: &Self::Saved) -> Result<(), EngError>;
    /// The import's resident-KV check, before any device write.
    fn resident_kv(&mut self) -> Result<(), EngError>;
    /// Opens a device window over `rows` rows.
    fn rollback_begin(&mut self, rows: usize) -> Result<(), EngError>;
    /// Copies the state after row `row` into its slot.
    fn capture(&mut self, row: usize) -> Result<(), EngError>;
    /// Drops an open device window and its slot marks.
    fn rollback_discard(&mut self);
    /// Marks that every row ran.
    fn rollback_window_complete(&mut self) -> Result<(), EngError>;
    /// Applies the verdict through the slots; returns whether a slot was copied back.
    fn rollback_finish(&mut self, kept: usize, committed_rows: usize) -> Result<bool, EngError>;

    /// A failure injected at a point of the commit. Empty outside the step's test build.
    fn commit_fault(&mut self, _at: CommitPoint) -> Result<(), EngError> {
        Ok(())
    }
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

fn reset_first(what: &str) -> EngError {
    EngError::BadRequest(format!(
        "verify window: {what}; the state is poisoned until reset_state"
    ))
}

/// The poison route of a failed or indeterminate window: the state is poisoned, the recovery gate
/// is set (only `reset_state` clears it), a pending prefix save is dropped and never published,
/// the pinned pool is invalidated and the window is closed.
pub(crate) fn fail_window<H: WindowHost>(host: &mut H) {
    host.poison();
    host.set_needs_reset();
    host.drop_pending_prefix();
    host.invalidate_pool();
    host.close_window();
}

/// The one check of whether the live state may be used: refused while the recovery gate is set
/// (a failed commit poisoned the state; only `reset_state` clears it) or while a window awaits its
/// verdict (the state is ahead of the lineage).
pub(crate) fn state_usable<H: WindowHost>(host: &H) -> Result<(), EngError> {
    if host.needs_reset() {
        return Err(reset_first("the recovery gate is set"));
    }
    if host.window_awaits() {
        return Err(EngError::BadRequest(
            "verify window: a window waits for its verdict; the state is ahead of it (reset first)"
                .to_owned(),
        ));
    }
    Ok(())
}

/// `import_state` of the step: replaces the committed state with `saved`. Every restore goes
/// through it, the window commit included. It refuses while the recovery gate is set, closes any
/// open window (the import replaces the state it describes; a stale snapshot of a stale window is
/// never imported), finishes a pending pinned save, runs the two refusals that precede every
/// device write, then overwrites.
pub(crate) fn import_state<H: WindowHost>(host: &mut H, saved: &H::Saved) -> Result<(), EngError> {
    restore_gate(host)?;
    host.commit_fault(CommitPoint::PrefixSave)
        .and_then(|()| host.complete_prefix_save())?;
    host.commit_fault(CommitPoint::CheckImport)
        .and_then(|()| host.check_import(saved))?;
    host.commit_fault(CommitPoint::ResidentKv)
        .and_then(|()| host.resident_kv())?;
    host.commit_fault(CommitPoint::ImportDevice)
        .and_then(|()| host.overwrite(saved))
}

/// The gate of a decode step (`run_token`): refused while the recovery gate is set, and refused
/// through a window that awaits its verdict, which would continue from its speculative rows. A step
/// through such a window also fails the state closed: the lineage is then lost.
pub(crate) fn step_gate<H: WindowHost>(host: &mut H) -> Result<(), EngError> {
    if let Err(error) = state_usable(host) {
        if !host.needs_reset() {
            fail_window(host);
        }
        return Err(error);
    }
    Ok(())
}

/// The gate of an export: the same refusal as a step, with no change to the state.
pub(crate) fn export_gate<H: WindowHost>(host: &H) -> Result<(), EngError> {
    state_usable(host)
}

/// The entry of every restoring wrapper (`import_state`, the pinned-prefix restore, and the
/// `PrefixState` restore through `import_state`): refused with "reset first" while the recovery
/// gate is set; otherwise any open window is closed, before the first fallible call, because the
/// restore replaces the state the window describes.
pub(crate) fn restore_gate<H: WindowHost>(host: &mut H) -> Result<(), EngError> {
    if host.needs_reset() {
        return Err(reset_first(
            "a restore through a failed state is refused (reset first)",
        ));
    }
    host.close_window();
    Ok(())
}

/// The first half of `reset_state`, which is EXEMPT from `state_usable`: when the gate is set or a
/// window awaits its verdict, the pending prefix save is dropped unfinished (and the pool
/// invalidated) instead of finished, so the one recovery cannot be wedged by its own check. The
/// window is closed either way.
pub(crate) fn reset_prologue<H: WindowHost>(host: &mut H) -> Result<(), EngError> {
    if state_usable(host).is_ok() {
        host.complete_prefix_save()?;
    } else {
        host.drop_pending_prefix();
        host.invalidate_pool();
    }
    host.close_window();
    Ok(())
}

/// The second half of `reset_state`, after the state was zeroed: the gate opens.
pub(crate) fn reset_epilogue<H: WindowHost>(host: &mut H) {
    host.clear_needs_reset();
}

/// What a failure of the rows of a window leaves. A failure before the first row ran (a refusal
/// at the top) leaves the state valid and unchanged; once a row ran, or the state is no longer
/// valid, the state is ahead of the lineage or half written: it fails closed.
fn row_failure<H: WindowHost>(host: &mut H, base: usize, error: EngError) -> EngError {
    if host.tokens() != base || !host.is_valid() {
        fail_window(host);
    }
    error
}

/// `fail_window`, then the error to return.
fn failed<H: WindowHost>(host: &mut H, error: EngError) -> EngError {
    fail_window(host);
    error
}

/// What a verdict must satisfy before any state changes: the state is `Valid` and not gated, the
/// verdict keeps 1 to `rows` rows, names `base + kept` committed rows, and the state is at the end
/// of the window (`base + rows`).
fn check_verdict<H: WindowHost>(
    host: &H,
    base: usize,
    rows: usize,
    commit: &Commit,
) -> Result<(), EngError> {
    if !host.is_valid() {
        return Err(reset_first("commit_window found a poisoned state"));
    }
    if host.needs_reset() {
        return Err(reset_first("commit_window found the recovery gate set"));
    }
    let kept = commit.kept_rows;
    if kept == 0 || kept > rows || commit.discarded_rows != rows - kept {
        return Err(EngError::BadRequest(format!(
            "verify window: a verdict keeps {kept} and discards {} of {rows} rows",
            commit.discarded_rows
        )));
    }
    if commit.committed_rows != base + kept {
        return Err(EngError::BadRequest(format!(
            "verify window: the verdict names {} committed rows, the window began at {} and keeps {kept}",
            commit.committed_rows, base
        )));
    }
    if host.tokens() != base + rows {
        return Err(EngError::BadRequest(format!(
            "verify window: the state holds {} rows, the window ends at {}",
            host.tokens(),
            base + rows
        )));
    }
    Ok(())
}

/// Runs the rows of a window (the model side of `WindowModel::run_window`).
pub(crate) fn run_window<H: WindowHost>(host: &mut H, inputs: &[u32]) -> Result<(), EngError> {
    host.trace_mark(Mark::WindowStart);
    // A window that is still stored has no verdict: the state is ahead of the lineage and the
    // window is indeterminate. It poisons and refuses instead of being replaced, and this check
    // precedes everything else, `complete_prefix_save` included, so that a refused call never
    // finishes a save from a state that is ahead.
    if host.window_awaits() {
        return Err(failed(
            host,
            EngError::BadRequest(
                "verify window: the previous window has no verdict; the state is poisoned until \
                 reset_state"
                    .to_owned(),
            ),
        ));
    }
    state_usable(host)?;
    if host.device_wanted() {
        return run_window_device(host, inputs);
    }
    // Refused before the export and before any row: the state stays valid and unchanged.
    check_window_fits(host.tokens(), inputs.len(), host.capacity())?;
    let base = host.tokens();
    let saved = host.export()?;
    host.trace_mark(Mark::Exported);
    let mut rows = Vec::with_capacity(inputs.len());
    let mut argmax = Vec::with_capacity(inputs.len());
    let ran = (|| -> Result<(), EngError> {
        for (at, &token) in inputs.iter().enumerate() {
            host.feed_fault(at)?;
            let (drawn, row) = host.run_row(token)?;
            argmax.push(drawn);
            rows.push(row);
            host.trace_mark(Mark::RowDone);
        }
        Ok(())
    })();
    if let Err(error) = ran {
        return Err(row_failure(host, base, error));
    }
    host.store_window(Window {
        base,
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
    // The window is taken. From here every `Err` is a failed or indeterminate commit: it poisons
    // the state and sets the recovery gate (`failed`), whatever step it came from.
    if let Err(error) = check_verdict(host, window.base, window.inputs.len(), commit) {
        return Err(failed(host, error));
    }
    if commit.discarded_rows == 0 {
        #[cfg(test)]
        {
            drop(saved);
            host.trace_mark(Mark::Dropped);
        }
        return Ok(());
    }
    host.trace_mark(Mark::CommitStart);
    // The restore, run through the same entry point as every other import: the gate, the
    // pending save, the two refusals before `begin_import`, then the overwrite.
    if let Err(error) = import_state(host, &saved) {
        return Err(failed(host, error));
    }
    host.trace_mark(Mark::Imported);
    for (at, &token) in window.inputs[..commit.kept_rows].iter().enumerate() {
        if let Err(error) = host.replay_fault(at).and_then(|()| host.replay_row(token)) {
            return Err(failed(host, error));
        }
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
        return Err(row_failure(host, base, error));
    }
    // Every row ran: until the verdict a step or an export through the state is refused.
    // `rollback_window_complete` can fail only if the window closed under it: all rows ran and
    // the state would stay valid and ahead, so the error fails the window closed like every other
    // error after the first row.
    if let Err(error) = host.rollback_window_complete() {
        return Err(failed(host, error));
    }
    host.store_window(Window {
        base,
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
    // The device restore is a consumer of the live state like every other: a set recovery gate
    // refuses it (the book still awaits its verdict here, so only the gate is asked).
    if host.needs_reset() {
        return Err(failed(
            host,
            reset_first("commit_window found the recovery gate set"),
        ));
    }
    if commit.discarded_rows > 0
        && let Err(error) = host.restore_fault()
    {
        return Err(failed(host, error));
    }
    match host.rollback_finish(commit.kept_rows, commit.committed_rows) {
        Ok(true) => host.trace_mark(Mark::Imported),
        Ok(false) => {}
        Err(error) => return Err(failed(host, error)),
    }
    Ok(())
}

#[cfg(test)]
#[path = "window_logic_tests.rs"]
mod tests;
