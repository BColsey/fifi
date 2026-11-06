//! The device-resident slots of the verify window's rollback.
//!
//! A dedicated buffer holds `rows - 1` copies of the DeltaNet state (the whole slot 0 of the Delta
//! pool, recurrent and convolution, one contiguous range: `layout.delta.total / layout.delta.slots`
//! bytes). The window copies slot 0 into slot `i` after row `i`; a verdict that keeps `kept` rows
//! copies slot `kept - 1` back and truncates the KV frontier (`finish_overwrite(base + kept)`). KV
//! data needs no copy: a row writes only the cell at its own position (`attn_prep_k` and
//! `kv8_append_k` index by `pos_dev + t`), so the rows above the frontier are stale and unreachable
//! (the attention reads keys below `kv_len = pos + 1`).
//!
//! The buffer is separate from the Delta pool because `copy_device` refuses source and destination
//! in one allocation, and because `reset` and `import_state` zero every pool slot above 0 through
//! pageable host writes. It is outside every captured graph, which bakes only slot 0 and the
//! words block, and `copy_device` refuses inside a capture. Every copy is stream ordered on the
//! owning stream, between two synchronous `run_token` calls: the stream is idle when a capture is
//! enqueued, and the next launch queues behind it.
//!
//! All the bookkeeping is `family::qwen36::rollback`, tested on the host. What only a device shows:
//! the byte range of each copy, the slot offsets and the stream order.

use super::{
    super::{book::Health, state_layout::StateLayout},
    ResidentState, StateIdentity,
    export::IMPORT_REWRITES_WORDS,
};
use crate::{
    EngError,
    family::qwen36::rollback::{Restore, RollbackBook, RollbackPlan, StateFacts},
    kernels::cuda::{CudaContext, DeviceBuffer, DeviceStream},
};

pub(crate) struct RollbackSlots {
    buffer: DeviceBuffer,
    book: RollbackBook,
    owner: u64,
    layout: StateLayout,
    identity: StateIdentity,
}

/// Bytes of one slot of `state`: its whole Delta slot 0.
pub(crate) fn slot_bytes_of(state: &ResidentState) -> Result<usize, EngError> {
    let pool = state.layout.delta;
    pool.total
        .checked_div(pool.slots)
        .filter(|&bytes| bytes > 0)
        .ok_or_else(|| EngError::BadRequest("device rollback: zero Delta slots".into()))
}

impl RollbackSlots {
    /// The slots for windows of up to `rows` rows over `state`. Refuses a state that is not valid,
    /// idle and resident, and answers a failed allocation with `MemoryPlan`, naming the bytes; the
    /// state is untouched either way.
    pub(crate) fn new(
        ctx: &CudaContext,
        stream: &DeviceStream,
        state: &ResidentState,
        rows: usize,
    ) -> Result<Self, EngError> {
        let plan = RollbackPlan::new(rows, slot_bytes_of(state)?)?;
        Self::allocate(ctx, stream, state, plan)
    }

    fn allocate(
        ctx: &CudaContext,
        stream: &DeviceStream,
        state: &ResidentState,
        plan: RollbackPlan,
    ) -> Result<Self, EngError> {
        state.resident_kv()?;
        state.book.begin_export(stream.id())?;
        let bytes = plan.total_bytes()?;
        let buffer = ctx.allocate(bytes).map_err(|error| match error {
            EngError::MemoryPlan(message) => EngError::MemoryPlan(format!(
                "device rollback: {bytes} B for {} slots: {message}",
                plan.slots()
            )),
            other => other,
        })?;
        Ok(Self {
            buffer,
            book: RollbackBook::new(plan),
            owner: stream.id(),
            layout: state.layout,
            identity: state.identity.clone(),
        })
    }

    /// Test builds only: a plan with another slot size, to exercise the refusal of an allocation
    /// the card cannot hold.
    #[cfg(test)]
    pub(crate) fn new_sized(
        ctx: &CudaContext,
        stream: &DeviceStream,
        state: &ResidentState,
        plan: RollbackPlan,
    ) -> Result<Self, EngError> {
        Self::allocate(ctx, stream, state, plan)
    }

    pub(crate) fn plan(&self) -> &RollbackPlan {
        self.book.plan()
    }

    /// Device bytes held.
    pub(crate) fn bytes(&self) -> usize {
        self.buffer.bytes()
    }

    pub(crate) fn is_open(&self) -> bool {
        self.book.is_open()
    }

    /// What the guard needs to know. `state` must be the state these slots were built for.
    fn facts(&self, stream: &DeviceStream, state: &ResidentState) -> StateFacts {
        let same = self.layout == state.layout && self.identity == state.identity;
        StateFacts {
            owner: same && stream.id() == self.owner && state.book.begin_join(stream.id()).is_ok(),
            valid: state.health() == Health::Valid,
            idle: state.book.pending_rows() == 0,
            resident_kv: state.resident_kv().is_ok(),
        }
    }

    /// Opens a window over the committed rows of `state`. Nothing is copied: the slots are written
    /// by the captures that follow each row but the last.
    pub(crate) fn begin_window(
        &mut self,
        stream: &DeviceStream,
        state: &ResidentState,
        rows: usize,
    ) -> Result<(), EngError> {
        self.facts(stream, state).require_copyable("run_window")?;
        self.book.begin_window(state.tokens(), rows)
    }

    /// Copies slot 0 of the Delta pool into the slot of row `row`, queued on `stream` behind the
    /// row's work (the row has finished: `run_token` waits). Refused for a state that is not valid,
    /// idle and owned by `stream`, so a state ahead of `tokens` is never copied.
    pub(crate) fn capture(
        &mut self,
        ctx: &CudaContext,
        stream: &DeviceStream,
        state: &ResidentState,
        row: usize,
    ) -> Result<(), EngError> {
        self.facts(stream, state).require_copyable("capture")?;
        let slot = self.book.begin_capture(row)?;
        let bytes = self.book.plan().slot_bytes;
        ctx.copy_device(
            stream,
            state.delta.range(0, bytes)?,
            self.buffer.range(self.book.plan().offset(slot)?, bytes)?,
        )?;
        self.book.note_captured(row)
    }

    /// Applies the verdict: `kept` rows stay, and the state was `committed_rows` after them in the
    /// verdict. Returns whether a slot was copied back. Order:
    /// 1. the guard (valid, idle, owner): a `Poisoned` state is refused and stays `Poisoned`
    ///    (`begin_import` below would accept it and `finish_overwrite` would heal it);
    /// 2. the book checks the verdict against the window (a mismatch poisons the state, which is
    ///    ahead of the lineage);
    /// 3. `begin_import` (the state is `Poisoned` until the end), the words block rewritten, the
    ///    slot copied into slot 0, the stream drained, then `finish_overwrite(base + kept)`.
    ///
    /// Any failure after step 1 leaves the state `Poisoned` and the window closed.
    pub(crate) fn finish(
        &mut self,
        ctx: &CudaContext,
        stream: &DeviceStream,
        state: &mut ResidentState,
        kept: usize,
        committed_rows: usize,
    ) -> Result<bool, EngError> {
        if let Err(error) = self.facts(stream, state).require_copyable("commit_window") {
            self.book.discard();
            return Err(error);
        }
        let restore = match self
            .book
            .finish_window(kept, committed_rows, state.tokens())
        {
            Ok(restore) => restore,
            Err(error) => {
                state.poison();
                return Err(error);
            }
        };
        let Some(Restore { slot, frontier }) = restore else {
            return Ok(false);
        };
        let bytes = self.book.plan().slot_bytes;
        let copy = (|| {
            let from = self.buffer.range(self.book.plan().offset(slot)?, bytes)?;
            state
                .book
                .begin_import(stream.id(), IMPORT_REWRITES_WORDS)?;
            state.clear_words(stream)?;
            ctx.copy_device(stream, from, state.delta.range(0, bytes)?)?;
            stream.synchronize()
        })();
        if let Err(error) = copy {
            state.poison();
            return Err(error);
        }
        // The rows above the frontier are stale. A tip that leases a prefix the truncation reaches
        // into (it cannot: the window started above it) is dropped all the same.
        if state
            .resident_tip
            .as_ref()
            .is_some_and(|tip| tip.positions() > frontier)
        {
            state.resident_tip = None;
        }
        state.book.finish_overwrite(frontier);
        Ok(true)
    }

    /// Test builds: `copies` timed copies of slot 0 into the first slot, each drained, in ms. The
    /// first slot is overwritten; call it outside a window.
    #[cfg(test)]
    pub(crate) fn bench_copy(
        &mut self,
        ctx: &CudaContext,
        stream: &DeviceStream,
        state: &ResidentState,
        copies: usize,
    ) -> Result<Vec<f64>, EngError> {
        if self.book.is_open() {
            return Err(EngError::BadRequest(
                "device rollback: a window is open".into(),
            ));
        }
        self.facts(stream, state).require_copyable("bench")?;
        let bytes = self.book.plan().slot_bytes;
        let mut times = Vec::with_capacity(copies);
        for _ in 0..copies {
            let began = std::time::Instant::now();
            ctx.copy_device(
                stream,
                state.delta.range(0, bytes)?,
                self.buffer.range(0, bytes)?,
            )?;
            stream.synchronize()?;
            times.push(began.elapsed().as_secs_f64() * 1e3);
        }
        Ok(times)
    }

    /// Drops the open window and every slot mark: a reset, an import, a poison or a failure.
    pub(crate) fn discard(&mut self) {
        self.book.discard();
    }
}

#[cfg(test)]
#[path = "rollback_tests.rs"]
mod tests;
