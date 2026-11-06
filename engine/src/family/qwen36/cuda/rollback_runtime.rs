//! The step's device rollback of a verify window: arming before the first token, and the three
//! calls the window makes. This is a child of the step, so the device owners stay private.
//!
//! Armed means `self.rollback.is_some()`. A step that was asked for the device rollback
//! (`rollback_wanted`) and holds no slots refuses every window before any state change; it never
//! falls back to the host path. Start refuses when the slots cannot be allocated, so that state is
//! reachable only through a bug, and the refusal is the fail-closed answer to it.
use super::super::state::rollback::{RollbackSlots, slot_bytes_of};
use super::QwenResidentStep;
use crate::{
    EngError,
    family::qwen36::rollback::{Cover, RollbackPlan, check_uncharged_slots},
};

impl QwenResidentStep<'_> {
    /// Records that the settings asked for `draft_rollback = device`. Call it before
    /// `enable_device_rollback`; a step that wants it and is never armed refuses windows.
    pub(crate) fn request_device_rollback(&mut self) {
        self.rollback_wanted = true;
    }

    pub(crate) fn device_rollback_wanted(&self) -> bool {
        self.rollback_wanted
    }

    pub(crate) fn device_rollback_armed(&self) -> bool {
        self.rollback.is_some()
    }

    /// The plan of the armed slots, for the startup line.
    pub(crate) fn device_rollback_plan(&self) -> Option<RollbackPlan> {
        self.rollback.as_ref().map(|slots| *slots.plan())
    }

    /// Allocates the slots for windows of up to `rows` rows and returns their bytes. Refused, with
    /// the step untouched, when: the step has run, it is tiered, a sparse sketch is armed, the KV
    /// is not the resident plane, `rows` is outside 2 to 8, no admission
    /// charges the bytes and they do not fit the load reserve (`cover`), or the card cannot hold
    /// them (`MemoryPlan`). `tip_bytes` is the device tip that shares the reserve.
    pub(crate) fn enable_device_rollback(
        &mut self,
        rows: usize,
        cover: Cover,
        tip_bytes: usize,
    ) -> Result<usize, EngError> {
        let refuse = |why: &str| {
            EngError::BadRequest(format!(
                "draft_rollback device: {why}; set draft_rollback to host"
            ))
        };
        if self.started {
            return Err(refuse("the step has already run"));
        }
        if self.rollback.is_some() {
            return Err(refuse("the slots are already armed"));
        }
        if self.is_tiered() {
            return Err(refuse(
                "tiered experts keep a bank state the slots do not hold",
            ));
        }
        if self.res.sparse.is_some() || self.res.paged.is_some() {
            return Err(refuse(
                "sparse attention or a paged KV cache is armed and the frontier is not proven for it",
            ));
        }
        let plan = RollbackPlan::new(rows, slot_bytes_of(&self.state)?)?;
        if let Cover::Reserve { headroom } = cover {
            check_uncharged_slots(plan.total_bytes()?, tip_bytes, headroom)?;
        }
        let slots = RollbackSlots::new(self.context, &self.stream, &self.state, rows)?;
        let bytes = slots.bytes();
        self.rollback = Some(slots);
        Ok(bytes)
    }

    /// The slots, or the refusal of a window on a step that wants them and has none.
    fn armed_slots(&mut self) -> Result<&mut RollbackSlots, EngError> {
        self.rollback.as_mut().ok_or_else(|| {
            EngError::BadRequest(
                "verify window: draft_rollback device is set and no rollback slots are armed"
                    .into(),
            )
        })
    }

    /// Opens a device window over the committed rows.
    pub(crate) fn rollback_begin(&mut self, rows: usize) -> Result<(), EngError> {
        let Self {
            rollback,
            stream,
            state,
            ..
        } = self;
        let slots = rollback.as_mut().ok_or_else(|| {
            EngError::BadRequest(
                "verify window: draft_rollback device is set and no rollback slots are armed"
                    .into(),
            )
        })?;
        slots.begin_window(stream, state, rows)
    }

    /// Copies the state after row `row` into its slot.
    pub(crate) fn rollback_capture(&mut self, row: usize) -> Result<(), EngError> {
        let Self {
            rollback,
            stream,
            state,
            context,
            ..
        } = self;
        let slots = rollback.as_mut().ok_or_else(|| {
            EngError::BadRequest("verify window: no rollback slots are armed".into())
        })?;
        slots.capture(context, stream, state, row)
    }

    /// Applies the verdict. Returns whether a slot was copied back.
    pub(crate) fn rollback_finish(
        &mut self,
        kept: usize,
        committed_rows: usize,
    ) -> Result<bool, EngError> {
        let Self {
            rollback,
            stream,
            state,
            context,
            ..
        } = self;
        let slots = rollback.as_mut().ok_or_else(|| {
            EngError::BadRequest("verify window: no rollback slots are armed".into())
        })?;
        slots.finish(context, stream, state, kept, committed_rows)
    }

    /// Marks the state untrustworthy: the window failed after a row ran, so the state is ahead of
    /// the lineage and no verdict can come.
    pub(crate) fn rollback_poison(&mut self) {
        self.state.poison();
    }

    /// Drops an open window and every slot mark: a reset, an import, a poison, a failed window.
    pub(crate) fn rollback_discard(&mut self) {
        if let Some(slots) = self.rollback.as_mut() {
            slots.discard();
        }
    }

    /// Refuses a window on a step that wants the device rollback and holds no slots.
    pub(crate) fn require_armed_for_window(&mut self) -> Result<(), EngError> {
        self.armed_slots().map(|_| ())
    }
}

/// Test builds: what the single-row kernel dispatch launched on this step's context, the switch
/// between the two rollbacks that the cost harness rotates, and the copy microbenchmark.
#[cfg(test)]
impl QwenResidentStep<'_> {
    /// Drains the owning stream (the capture drain of the cost trace).
    pub(crate) fn drain_stream(&self) -> Result<(), EngError> {
        self.stream.synchronize()
    }

    /// Switches the window between the device rollback (`on`, needs armed slots) and the host
    /// rollback. The cost harness rotates both arms in one process so that drift hits both.
    pub(crate) fn use_device_rollback(&mut self, on: bool) -> Result<(), EngError> {
        if on && self.rollback.is_none() {
            return Err(EngError::BadRequest(
                "device rollback: no slots are armed".into(),
            ));
        }
        self.rollback_wanted = on;
        Ok(())
    }

    /// Times `copies` copies of one slot (slot 0 of the Delta pool into slot 0 of the buffer), each
    /// followed by a stream drain, in ms: the calibration of the device copy time.
    pub(crate) fn device_copy_microbench(&mut self, copies: usize) -> Result<Vec<f64>, EngError> {
        let Self {
            rollback,
            stream,
            state,
            context,
            ..
        } = self;
        let slots = rollback
            .as_mut()
            .ok_or_else(|| EngError::BadRequest("device rollback: no slots are armed".into()))?;
        slots.bench_copy(context, stream, state, copies)
    }

    pub(crate) fn kernel_launches(&self) -> crate::kernels::single_row::SingleRowLaunches {
        self.context.exl3_launch_counts()
    }
}
