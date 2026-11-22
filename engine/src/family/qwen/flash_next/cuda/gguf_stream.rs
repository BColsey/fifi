//! The routed experts of a Flash-Next GGUF that do not fit the card: which layers' experts are
//! resident, which stream, and the staging of the picked experts of a streamed layer before its
//! expert launches.
//!
//! The placement is the shared stream plan (`expert_bank::layer_stream::StreamPlan`): layers on
//! the card, layers in pinned host memory, layers read from the model file in place. The copy is
//! the shared card transport (`expert_bank::layer_stream_card::CardTransport`), which stages a
//! layer's three expert stacks in the file's own quantization and leaves expert `e` of a stack at
//! its natural offset, so the router's ids index the staged stack unchanged. This module only
//! turns the file's expert index into that plan and those spans, and stages a decode token's
//! picks.
//!
//! Counters: every staging adds the bytes it moved and the bytes the picks needed, so a receipt
//! can say how much of what crossed the bus the token used.

use std::{rc::Rc, sync::Arc, time::Instant};

use super::super::load::gguf::{GgufModel, placement::LayerStacks};
use crate::{
    EngError,
    family::expert_bank::{
        layer_stream::{ExpertTransport, Home, StreamPlan, UploadSet},
        layer_stream_card::{CardTransport, StackHome, StackShape, StackSpan},
    },
    kernels::cuda::{
        CudaContext, DeviceStream, NativeWeight, PinnedBuf, resident::matvec_q::QuantType,
    },
};

fn refuse(detail: impl AsRef<str>) -> EngError {
    EngError::MemoryPlan(format!(
        "Flash-Next GGUF expert streaming: {}",
        detail.as_ref()
    ))
}

/// Host bytes one pinned fill reads at a time.
const FILL_CHUNK: usize = 64 << 20;

/// What the staging moved and used, over the transport's life.
#[derive(Debug, Default, Clone, Copy)]
pub(in crate::family::qwen::flash_next) struct StreamTotals {
    /// Layer stagings (one per streamed layer per token).
    pub(in crate::family::qwen::flash_next) stagings: u64,
    /// Bytes copied to the card.
    pub(in crate::family::qwen::flash_next) moved: u64,
    /// Bytes of the picked experts: what the launches read.
    pub(in crate::family::qwen::flash_next) needed: u64,
    /// Seconds the copy stream spent copying.
    pub(in crate::family::qwen::flash_next) copy_seconds: f64,
    /// Seconds the host spent in the staging call (the read, the wait).
    pub(in crate::family::qwen::flash_next) host_seconds: f64,
}

/// The streamed layers' experts: the transport and the plan that placed them.
pub(in crate::family::qwen::flash_next) struct ExpertStreams {
    transport: CardTransport<'static>,
    plan: StreamPlan,
    /// The transport's layer number of each model layer, `None` for a resident layer.
    streamed: Vec<Option<usize>>,
    record_bytes: Vec<u64>,
    next_slot: usize,
    totals: StreamTotals,
}

impl ExpertStreams {
    /// The transport for the layers `plan` streams. Pinned layers are read from the file into
    /// pinned host memory now; disk layers keep the file open and are read on a miss.
    pub(in crate::family::qwen::flash_next) fn new(
        model: &GgufModel,
        stacks: &[LayerStacks],
        plan: StreamPlan,
        ctx: &Rc<CudaContext>,
        compute: &Rc<DeviceStream>,
    ) -> Result<Self, EngError> {
        let experts = model.config().experts();
        let weights = &*model.weights;
        let mut files: Vec<Option<(Arc<std::fs::File>, Option<Arc<std::fs::File>>)>> =
            vec![None; weights.paths().len()];
        let mut spans: Vec<Vec<StackSpan>> = Vec::new();
        let mut store: Vec<PinnedBuf> = Vec::new();
        let mut streamed = vec![None; stacks.len()];
        let index = model.expert_index()?;
        for (layer, facts) in stacks.iter().enumerate() {
            let home = plan.home(layer);
            if home == Home::Card {
                continue;
            }
            streamed[layer] = Some(spans.len());
            let slices = index
                .slices(layer, 0)
                .map_err(|error| refuse(format!("layer {layer}: {error:?}")))?;
            let [gate, up, down] = slices
                .three()
                .map_err(|error| refuse(format!("layer {layer}: {error:?}")))?;
            let mut layer_spans = Vec::with_capacity(3);
            for (stack, slice) in facts.stacks.iter().zip([gate, up, down]) {
                let quant = QuantType::from_gguf(stack.ty).ok_or_else(|| {
                    refuse(format!("{} is {}, with no kernel", stack.name, stack.ty.name()))
                })?;
                let shape = StackShape {
                    quant,
                    k: stack.k,
                    n_rows: stack.rows,
                };
                let bytes = usize::try_from(stack.bytes)
                    .map_err(|_| refuse(format!("{} overflows usize", stack.name)))?;
                let span_home = match home {
                    Home::Pinned => {
                        let mut buffer = ctx.pinned_buf(bytes)?;
                        let mut chunk = vec![0u8; FILL_CHUNK.min(bytes)];
                        let mut done = 0usize;
                        while done < bytes {
                            let take = chunk.len().min(bytes - done);
                            weights.read_range(&stack.name, done as u64, &mut chunk[..take])?;
                            buffer.write(done, &chunk[..take])?;
                            done += take;
                        }
                        store.push(buffer);
                        StackHome::Pinned {
                            buf: store.len() - 1,
                            offset: 0,
                        }
                    }
                    Home::Disk => {
                        let (file, direct) = open_shard(weights.paths(), stack.shard, &mut files)?;
                        StackHome::File {
                            file,
                            direct,
                            offset: slice.file_range.start,
                        }
                    }
                    Home::Card => unreachable!("a resident layer is skipped above"),
                };
                layer_spans.push(StackSpan {
                    bytes,
                    home: span_home,
                    shape: Some(shape),
                });
            }
            spans.push(layer_spans);
        }
        let record_bytes = stacks
            .iter()
            .map(|layer| layer.stacks.iter().map(|stack| stack.bytes).sum::<u64>() / experts as u64)
            .collect();
        let transport = CardTransport::new(
            Rc::clone(ctx),
            Rc::clone(compute),
            experts,
            spans,
            store,
            plan.staging_slots(),
        )?;
        Ok(Self {
            transport,
            plan,
            streamed,
            record_bytes,
            next_slot: 0,
            totals: StreamTotals::default(),
        })
    }

    /// The plan the transport follows.
    pub(in crate::family::qwen::flash_next) fn plan(&self) -> &StreamPlan {
        &self.plan
    }

    /// The transport's layer number of `layer`, `None` when its experts are on the card.
    pub(in crate::family::qwen::flash_next) fn is_streamed(&self, layer: usize) -> bool {
        self.streamed.get(layer).copied().flatten().is_some()
    }

    /// The next staging slot for a streamed layer, alternating between the plan's slots.
    pub(in crate::family::qwen::flash_next) fn take_slot(&mut self) -> usize {
        let slot = self.next_slot;
        self.next_slot = (self.next_slot + 1) % self.plan.staging_slots().max(1);
        slot
    }

    /// Copies the `picks` of `layer` into `slot`, and makes the compute stream wait for the copy.
    pub(in crate::family::qwen::flash_next) fn stage(
        &mut self,
        layer: usize,
        slot: usize,
        picks: &[u32],
    ) -> Result<(), EngError> {
        let index = self
            .streamed
            .get(layer)
            .copied()
            .flatten()
            .ok_or_else(|| refuse(format!("layer {layer} is not a streamed layer")))?;
        let started = Instant::now();
        let ticket = self
            .transport
            .begin(index, &UploadSet::Subset(picks.to_vec()), slot)?;
        let report = self.transport.wait(ticket)?;
        self.totals.stagings += 1;
        self.totals.moved += report.bytes;
        self.totals.needed += picks.len() as u64 * self.record_bytes[layer];
        self.totals.copy_seconds += report.seconds;
        self.totals.host_seconds += started.elapsed().as_secs_f64();
        Ok(())
    }

    /// The launches that read `slot` are queued: the next copy into it may follow them.
    pub(in crate::family::qwen::flash_next) fn release(&mut self, slot: usize) -> Result<(), EngError> {
        self.transport.release(slot)
    }

    /// Stack `stack` (0 gate, 1 up, 2 down) of `layer` as staged in `slot`.
    pub(in crate::family::qwen::flash_next) fn staged(
        &self,
        layer: usize,
        slot: usize,
        stack: usize,
    ) -> Result<&NativeWeight, EngError> {
        let index = self
            .streamed
            .get(layer)
            .copied()
            .flatten()
            .ok_or_else(|| refuse(format!("layer {layer} is not a streamed layer")))?;
        self.transport.staged(index, slot, stack)
    }

    /// What the staging has moved and used so far.
    pub(in crate::family::qwen::flash_next) fn totals(&self) -> StreamTotals {
        self.totals
    }
}

/// The shard's buffered file and, when the filesystem takes it, its direct handle (the whole-layer
/// reads use the direct one), opened once.
#[allow(clippy::type_complexity)]
fn open_shard(
    paths: &[std::path::PathBuf],
    shard: usize,
    files: &mut [Option<(Arc<std::fs::File>, Option<Arc<std::fs::File>>)>],
) -> Result<(Arc<std::fs::File>, Option<Arc<std::fs::File>>), EngError> {
    if let Some(Some(open)) = files.get(shard) {
        return Ok(open.clone());
    }
    let path = paths
        .get(shard)
        .ok_or_else(|| refuse(format!("shard {shard} is not open")))?;
    crate::storage_guard::require_local(path, "expert stacks")?;
    let file = Arc::new(
        std::fs::File::open(path)
            .map_err(|error| refuse(format!("cannot open {}: {error}", path.display())))?,
    );
    let direct = crate::family::expert_bank::host_tier::open_direct(path)
        .ok()
        .map(Arc::new);
    files[shard] = Some((Arc::clone(&file), direct.clone()));
    Ok((file, direct))
}
