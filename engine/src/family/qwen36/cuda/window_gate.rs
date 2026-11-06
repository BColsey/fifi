//! The recovery gate of the verify window on the step: the effects `window_logic` needs to fail a
//! window closed. This is a child of the step, so the device owners stay private.
//!
//! A failed or indeterminate window commit poisons the state and sets `needs_reset`. Only
//! `reset_state` clears it.
use super::super::{book::Health, state::export::ExportedState};
use super::QwenResidentStep;
use crate::EngError;

impl QwenResidentStep<'_> {
    /// Whether a window awaits its verdict: the stored window (host or device), or a device
    /// window that has run all its rows.
    pub(crate) fn window_is_open(&self) -> bool {
        self.window.is_some()
            || self
                .rollback
                .as_ref()
                .is_some_and(super::super::state::rollback::RollbackSlots::awaiting_verdict)
    }

    /// Rows the KV cache can hold.
    pub(crate) fn kv_capacity_rows(&self) -> usize {
        self.state.layout().kv.capacity
    }

    /// Whether the state is `Valid`.
    pub(crate) fn state_is_valid(&self) -> bool {
        self.state.health() == Health::Valid
    }

    pub(crate) fn recovery_gate_set(&self) -> bool {
        self.needs_reset
    }

    pub(crate) fn set_recovery_gate(&mut self) {
        self.needs_reset = true;
    }

    /// Poisons the state: it then refuses every step and snapshot.
    pub(crate) fn poison_window_state(&mut self) {
        self.state.poison();
    }

    /// Drops a pending pinned-prefix save without finishing it: it is never published from a
    /// poisoned state.
    pub(crate) fn drop_pending_prefix(&mut self) {
        self.pending_prefix = None;
    }

    /// Invalidates the pinned pool so that no later save recycles a slot whose KV came from the
    /// failed lineage. The pool's only failure is an exhausted generation counter; on this
    /// failure path it cannot change the outcome (the state is poisoned either way).
    pub(crate) fn invalidate_prefix_pool(&mut self) {
        if let Some(pool) = self.prefix_pins.as_mut() {
            let _exhausted = pool.invalidate();
        }
    }

    /// Closes the stored window, host and device.
    pub(crate) fn close_stored_window(&mut self) {
        self.window = None;
        self.rollback_discard();
    }

    pub(crate) fn check_saved_state(&self, saved: &ExportedState) -> Result<(), EngError> {
        self.state.check_exported(saved)
    }

    pub(crate) fn check_resident_kv(&self) -> Result<(), EngError> {
        self.state.check_resident_kv()
    }
}
