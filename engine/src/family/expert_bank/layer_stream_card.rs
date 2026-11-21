//! The card half of layer-major expert streaming: an [`ExpertTransport`] that copies a layer's
//! expert stacks from pinned host memory, or from the model file read in place, into staging
//! slots on the card, on a copy stream ordered against the compute stream by events.
//!
//! Layout. A layer has one or more stacks (gate, up, down), each `experts` equal records in the
//! file's own quantization; the launches address expert `e` of a stack at `e * stack_bytes /
//! experts`. A staging slot holds one device buffer per stack index, sized for the largest layer's
//! stack at that index, so a whole-layer copy and a subset copy both leave expert `e` at its
//! natural offset and the router's ids index the staged stack unchanged. A layer's stacks of a
//! different size or quantization are read through [`CardTransport::stack`], which returns the
//! exact byte range of this layer in the slot.
//!
//! Order. `begin` makes the copy stream wait for the slot's `used` event (recorded on the compute
//! stream by `release`, so a copy never overwrites bytes a launch still reads), queues the
//! copies, and records the slot's `copied` event. The consumer makes its compute stream wait on
//! [`CardTransport::copied`] before the first launch that reads the slot. `wait` blocks the host
//! until the copies finished, so the seconds it reports are the copy stream's own (timing events
//! around the copies) and the seconds the host spends inside `wait` are the exposed part.
//!
//! Sources. A pinned source is page-locked host memory the caller filled at load (the copy is
//! one asynchronous transfer per stack, or per expert record for a subset). A file source is read
//! in place into the slot's pinned bounce buffer on `begin` (the host is busy for the read, which
//! the report counts in `read_seconds`) and then copied the same way.

use std::{fs::File, os::unix::fs::FileExt, rc::Rc, sync::Arc, time::Instant};

use super::layer_stream::{ExpertTransport, TransferReport, UploadSet};
use crate::{
    EngError,
    kernels::cuda::{
        CudaContext, DeviceBuffer, DeviceEvent, DeviceRange, DeviceStream, PinnedBuf, TimingEvent,
    },
};

fn transport_error(detail: String) -> EngError {
    EngError::BadRequest(format!("expert layer transport: {detail}"))
}

/// Where the bytes of one stack of one layer are held between prompts.
pub(crate) enum StackHome {
    /// In the pinned buffer `buf` of the transport's store, from `offset`.
    Pinned { buf: usize, offset: usize },
    /// In a model file, from `offset`.
    File { file: Arc<File>, offset: u64 },
}

/// One stack of one layer.
pub(crate) struct StackSpan {
    pub(crate) bytes: usize,
    pub(crate) home: StackHome,
}

/// What `begin` hands back: the slot, the bytes queued and the host seconds spent reading files.
pub(crate) struct CardTicket {
    slot: usize,
    bytes: u64,
    read_seconds: f64,
}

struct Slot {
    stacks: Vec<DeviceBuffer>,
    bounce: Option<PinnedBuf>,
    used: DeviceEvent,
    used_recorded: bool,
    copied: DeviceEvent,
    start: TimingEvent,
    end: TimingEvent,
}

/// The transport over one card.
pub(crate) struct CardTransport {
    ctx: Rc<CudaContext>,
    compute: Rc<DeviceStream>,
    copy: DeviceStream,
    experts: usize,
    layers: Vec<Vec<StackSpan>>,
    store: Vec<PinnedBuf>,
    slots: Vec<Slot>,
    read_seconds: f64,
}

impl CardTransport {
    /// Allocates `slots` staging slots on the card (and a bounce buffer per slot when any layer
    /// is read from a file). `layers[l]` lists the stacks of layer `l`; every layer has the same
    /// number of stacks and `experts` equal records per stack. `store` holds the pinned buffers
    /// the [`StackHome::Pinned`] spans index.
    pub(crate) fn new(
        ctx: Rc<CudaContext>,
        compute: Rc<DeviceStream>,
        experts: usize,
        layers: Vec<Vec<StackSpan>>,
        store: Vec<PinnedBuf>,
        slots: usize,
    ) -> Result<Self, EngError> {
        let stacks = layers.first().map_or(0, Vec::len);
        if stacks == 0 || slots == 0 || experts == 0 {
            return Err(transport_error("no stacks, slots or experts".into()));
        }
        for (layer, spans) in layers.iter().enumerate() {
            if spans.len() != stacks {
                return Err(transport_error(format!(
                    "layer {layer} has {} stacks, layer 0 has {stacks}",
                    spans.len()
                )));
            }
            for span in spans {
                if span.bytes == 0 || span.bytes % experts != 0 {
                    return Err(transport_error(format!(
                        "a stack of layer {layer} has {} bytes for {experts} experts",
                        span.bytes
                    )));
                }
                if let StackHome::Pinned { buf, offset } = &span.home {
                    let held = store.get(*buf).map_or(0, PinnedBuf::len);
                    if offset + span.bytes > held {
                        return Err(transport_error(format!(
                            "a stack of layer {layer} lies outside pinned buffer {buf}"
                        )));
                    }
                }
            }
        }
        let widest: Vec<usize> = (0..stacks)
            .map(|stack| {
                layers
                    .iter()
                    .map(|spans| spans[stack].bytes)
                    .max()
                    .unwrap_or(0)
            })
            .collect();
        let any_file = layers
            .iter()
            .flatten()
            .any(|span| matches!(span.home, StackHome::File { .. }));
        let bounce_bytes: usize = layers
            .iter()
            .map(|spans| spans.iter().map(|span| span.bytes).sum::<usize>())
            .max()
            .unwrap_or(0);
        let copy = ctx.stream()?;
        let mut staged = Vec::with_capacity(slots);
        for _ in 0..slots {
            let mut buffers = Vec::with_capacity(stacks);
            for bytes in &widest {
                buffers.push(ctx.allocate(*bytes)?);
            }
            staged.push(Slot {
                stacks: buffers,
                bounce: if any_file {
                    Some(ctx.pinned_buf(bounce_bytes)?)
                } else {
                    None
                },
                used: ctx.event()?,
                used_recorded: false,
                copied: ctx.event()?,
                start: ctx.event_timing()?,
                end: ctx.event_timing()?,
            });
        }
        Ok(Self {
            ctx,
            compute,
            copy,
            experts,
            layers,
            store,
            slots: staged,
            read_seconds: 0.0,
        })
    }

    /// The exact bytes of stack `stack` of `layer` in staging slot `slot`.
    pub(crate) fn stack(
        &self,
        layer: usize,
        slot: usize,
        stack: usize,
    ) -> Result<DeviceRange<'_>, EngError> {
        let bytes = self
            .layers
            .get(layer)
            .and_then(|spans| spans.get(stack))
            .map(|span| span.bytes)
            .ok_or_else(|| {
                transport_error(format!("layer {layer} stack {stack} does not exist"))
            })?;
        self.slots
            .get(slot)
            .and_then(|s| s.stacks.get(stack))
            .ok_or_else(|| transport_error(format!("slot {slot} stack {stack} does not exist")))?
            .range(0, bytes)
    }

    /// The event recorded on the copy stream after the slot's copies: the compute stream waits
    /// on it before the first launch that reads the slot.
    pub(crate) fn copied(&self, slot: usize) -> &DeviceEvent {
        &self.slots[slot].copied
    }

    /// Host seconds spent reading model files in `begin`, over the transport's life.
    pub(crate) fn read_seconds(&self) -> f64 {
        self.read_seconds
    }

    /// Reads `span` from its file into `bounce` at `at`; returns the host seconds it took.
    fn read_into(
        bounce: &mut PinnedBuf,
        at: usize,
        file: &File,
        offset: u64,
        bytes: usize,
    ) -> Result<f64, EngError> {
        let started = Instant::now();
        let mut chunk = vec![0u8; bytes.min(64 << 20)];
        let mut done = 0usize;
        while done < bytes {
            let take = chunk.len().min(bytes - done);
            file.read_exact_at(&mut chunk[..take], offset + done as u64)
                .map_err(|error| transport_error(format!("reading the model file: {error}")))?;
            bounce.write(at + done, &chunk[..take])?;
            done += take;
        }
        Ok(started.elapsed().as_secs_f64())
    }
}

impl ExpertTransport for CardTransport {
    type Ticket = CardTicket;

    fn begin(
        &mut self,
        layer: usize,
        set: &UploadSet,
        slot: usize,
    ) -> Result<Self::Ticket, EngError> {
        let spans = self
            .layers
            .get(layer)
            .ok_or_else(|| transport_error(format!("layer {layer} does not exist")))?;
        if slot >= self.slots.len() {
            return Err(transport_error(format!("slot {slot} does not exist")));
        }
        let state = &mut self.slots[slot];
        if state.used_recorded {
            self.copy.wait(&state.used)?;
        }
        self.copy.record_timing(&state.start)?;
        // File sources are read into the slot's bounce buffer first, every stack before the
        // first copy: the buffer cannot be written while a copy of it is outstanding.
        let mut read_seconds = 0f64;
        let mut bounce_at = 0usize;
        for span in spans {
            if let StackHome::File { file, offset } = &span.home {
                let bounce = state.bounce.as_mut().ok_or_else(|| {
                    transport_error("a file source needs the bounce buffer".into())
                })?;
                match set {
                    UploadSet::Whole => {
                        read_seconds +=
                            Self::read_into(bounce, bounce_at, file, *offset, span.bytes)?;
                    }
                    UploadSet::Subset(experts) => {
                        let record = span.bytes / self.experts;
                        for &expert in experts {
                            let rel = expert as usize * record;
                            read_seconds += Self::read_into(
                                bounce,
                                bounce_at + rel,
                                file,
                                *offset + rel as u64,
                                record,
                            )?;
                        }
                    }
                }
                bounce_at += span.bytes;
            }
        }
        let mut bytes = 0u64;
        let mut bounce_at = 0usize;
        for (index, span) in spans.iter().enumerate() {
            let dst = &state.stacks[index];
            // Where this stack's bytes sit in host memory: a pinned buffer and offset.
            let (from_file, base) = match &span.home {
                StackHome::Pinned { buf, offset } => (false, (*buf, *offset)),
                StackHome::File { .. } => (true, (usize::MAX, bounce_at)),
            };
            let record = span.bytes / self.experts;
            let mut copy_range = |rel: usize, len: usize| -> Result<(), EngError> {
                let target = dst.range(rel, len)?;
                let source = if from_file {
                    state.bounce.as_mut().ok_or_else(|| {
                        transport_error("a file source needs the bounce buffer".into())
                    })?
                } else {
                    self.store
                        .get_mut(base.0)
                        .ok_or_else(|| transport_error("a pinned span lost its buffer".into()))?
                };
                self.ctx
                    .copy_from_pinned(&self.copy, source, base.1 + rel, target)
            };
            match set {
                UploadSet::Whole => {
                    copy_range(0, span.bytes)?;
                    bytes += span.bytes as u64;
                }
                UploadSet::Subset(experts) => {
                    for &expert in experts {
                        copy_range(expert as usize * record, record)?;
                        bytes += record as u64;
                    }
                }
            }
            if from_file {
                bounce_at += span.bytes;
            }
        }
        // The end mark is recorded before the copied event: a host wait on the event then
        // proves the end mark complete, and the elapsed time can be read.
        self.copy.record_timing(&state.end)?;
        self.copy.record(&state.copied)?;
        self.read_seconds += read_seconds;
        Ok(CardTicket {
            slot,
            bytes,
            read_seconds,
        })
    }

    fn wait(&mut self, ticket: Self::Ticket) -> Result<TransferReport, EngError> {
        let state = &mut self.slots[ticket.slot];
        state.copied.synchronize()?;
        // Launches queued on the compute stream after this point read the slot behind its copy.
        self.compute.wait(&state.copied)?;
        // The copies are complete: the bounce buffer may be written again by the host. The pinned
        // stores are only read after load and need no release.
        if let Some(bounce) = state.bounce.as_mut()
            && bounce.pending_stream().is_some()
        {
            bounce.wait(&state.copied)?;
        }
        let millis = state.start.elapsed_ms_since(&state.end)?;
        Ok(TransferReport {
            bytes: ticket.bytes,
            seconds: f64::from(millis) / 1000.0 + ticket.read_seconds,
        })
    }

    fn release(&mut self, slot: usize) -> Result<(), EngError> {
        let state = self
            .slots
            .get_mut(slot)
            .ok_or_else(|| transport_error(format!("slot {slot} does not exist")))?;
        self.compute.record(&state.used)?;
        state.used_recorded = true;
        Ok(())
    }
}
