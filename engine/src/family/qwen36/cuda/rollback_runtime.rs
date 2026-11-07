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
    family::{
        project::cuda::WeightLayout,
        qwen36::{
            exl3_merge_plan::{Intent, SHARED_IN_ROUTED_VAR, Source},
            krow::{KrowFacts, SINGLE_GATHER_NAME, check_krow_arming},
            rollback::{ArmingFacts, Cover, RollbackPlan, check_arming, check_uncharged_slots},
        },
    },
};

/// The refusal of a window on a step that wants the device rollback and holds no slots.
fn no_slots() -> EngError {
    EngError::BadRequest(
        "verify window: draft_rollback device is set and no rollback slots are armed".into(),
    )
}

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
        check_arming(ArmingFacts {
            started: self.started,
            armed: self.rollback.is_some(),
            tiered: self.is_tiered(),
            sparse: self.res.sparse.is_some(),
            paged: self.res.paged.is_some(),
            forked_experts: self.forked_experts,
        })?;
        let plan = RollbackPlan::new(rows, slot_bytes_of(&self.state)?)?;
        if let Cover::Reserve { headroom } = cover {
            check_uncharged_slots(plan.total_bytes()?, tip_bytes, headroom)?;
        }
        let slots = RollbackSlots::new(self.context, &self.stream, &self.state, rows)?;
        let bytes = slots.bytes();
        self.rollback = Some(slots);
        Ok(bytes)
    }

    /// Records that the settings asked for `draft_verify = krow`: from now on the merged MoE block
    /// is refused, including through the test hook that builds it (`set_shared_in_routed`). A
    /// merged block the DEFAULT built (the variable unset) yields to krow: it is dropped here, before
    /// the first token and any capture, and the reference block runs, with the reason logged; the
    /// K-row path takes one-row MoE launches only. A block an explicit `=1` asked for stays, and
    /// `check_krow_arming` then refuses it.
    pub(crate) fn request_krow_verify(&mut self) {
        self.krow_wanted = true;
        if self.merge_intent == Intent::Default && self.exl3_merge.is_some() {
            eprintln!(
                "q36 cuda: {SHARED_IN_ROUTED_VAR} unset and draft_verify is krow, using the reference block (krow takes one-row MoE launches only)"
            );
            self.exl3_merge = None;
            self.merge_source = Source::Krow;
        }
    }

    /// The arming refusals of the K-row path (`family/qwen36/krow.rs`) over this step's own fields:
    /// the slots are armed, the layout is EXL3, the window cap, the single-row kernel switch, and
    /// the merged MoE block as the step holds it (`exl3_merge`), not only as the environment says.
    pub(crate) fn check_krow_arming(&self, rows: usize) -> Result<(), EngError> {
        let single_gather = std::env::var_os(SINGLE_GATHER_NAME);
        check_krow_arming(KrowFacts {
            rollback_armed: self.rollback.is_some(),
            exl3: self.layout == WeightLayout::Exl3,
            rows,
            single_gather: single_gather.as_deref(),
            shared_in_routed: self.exl3_merge.is_some(),
        })
    }

    /// The slots, or the refusal of a window on a step that wants them and has none.
    fn armed_slots(&mut self) -> Result<&mut RollbackSlots, EngError> {
        self.rollback.as_mut().ok_or_else(no_slots)
    }

    /// Opens a device window over the committed rows. A window that is still open means the state
    /// is ahead of the lineage (rows ran, no verdict came): the state is poisoned and the window
    /// dropped, then the call is refused.
    pub(crate) fn rollback_begin(&mut self, rows: usize) -> Result<(), EngError> {
        let Self {
            rollback,
            stream,
            state,
            ..
        } = self;
        let slots = rollback.as_mut().ok_or_else(no_slots)?;
        if slots.is_open() {
            slots.discard();
            state.poison();
            return Err(EngError::BadRequest(
                "verify window: the previous window has no verdict; the state is poisoned until a \
                 reset or an import"
                    .into(),
            ));
        }
        slots.begin_window(stream, state, rows)
    }

    /// Marks that every row ran: from here until the verdict the state is ahead of the lineage.
    pub(crate) fn rollback_window_complete(&mut self) -> Result<(), EngError> {
        let slots = self.rollback.as_mut().ok_or_else(no_slots)?;
        slots.mark_window_complete()
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
        let slots = rollback.as_mut().ok_or_else(no_slots)?;
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
        let slots = rollback.as_mut().ok_or_else(no_slots)?;
        slots.finish(context, stream, state, kept, committed_rows)
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
    /// Whether the sparse attention chain is admitted on this step (it refuses the device rollback).
    pub(crate) fn sparse_attention_armed(&self) -> bool {
        self.res.sparse.is_some()
    }

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
