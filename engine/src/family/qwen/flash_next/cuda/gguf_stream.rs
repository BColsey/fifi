//! The routed experts of a Flash-Next GGUF that do not all fit the card: a cache of the experts
//! each layer picks most, in the file's own quantization, per stack.
//!
//! Every layer holds `slots` experts of its three stacks (gate, up, down) in device buffers that
//! are native weights of `slots` experts each. The shared per-layer least-recently-used policy
//! (`expert_bank::policy::LayerLru`) decides which expert holds which slot: a pick that holds a
//! slot is a hit, a miss takes the least recent slot no pick of the token protects. The records
//! of the missed experts are copied from pinned host memory, or read from the model file in place
//! for the layers the host budget did not take, into their slots. The block then runs on the
//! slot stacks with the router's expert ids replaced by slot ids.
//!
//! Counters: hits and misses per pick, the bytes the misses moved, and the bytes of every picked
//! expert (what the launches read), so a receipt shows how much of a token's expert bytes the
//! card already held.

use std::{fs::File, os::unix::fs::FileExt, rc::Rc, sync::Arc, time::Instant};

use super::super::load::gguf::{
    GgufModel,
    placement::{CachePlan, LayerStacks},
};
use crate::{
    EngError,
    family::expert_bank::{PassPlan, layer_stream::Home, policy::LayerLru},
    kernels::cuda::{
        CudaContext, DeviceEvent, DeviceStream, NativeWeight, PinnedBuf,
        resident::matvec_q::QuantType,
    },
};

fn refuse(detail: impl AsRef<str>) -> EngError {
    EngError::MemoryPlan(format!(
        "Flash-Next GGUF expert cache: {}",
        detail.as_ref()
    ))
}

/// Host bytes one pinned fill reads at a time.
const FILL_CHUNK: usize = 64 << 20;

/// Where the records of one stack of one layer are held between tokens.
enum Source {
    /// In the pinned buffer `buf` of the cache's store, from byte 0.
    Pinned { buf: usize },
    /// In a model file, from `offset`, read in place on a miss.
    File { file: Arc<File>, offset: u64 },
}

struct CacheLayer {
    lru: LayerLru,
    /// The slots of gate, up and down: native weights of `slots` experts each.
    stacks: [NativeWeight; 3],
    sources: [Source; 3],
    /// Bytes of one expert in each stack.
    record: [usize; 3],
}

/// What the cache has done over its life.
#[derive(Debug, Default, Clone, Copy)]
pub(in crate::family::qwen::flash_next) struct StreamTotals {
    /// Layer visits (one per layer per token).
    pub(in crate::family::qwen::flash_next) visits: u64,
    /// Picked experts (distinct per visit) that held a slot.
    pub(in crate::family::qwen::flash_next) hits: u64,
    /// Picked experts copied into a slot.
    pub(in crate::family::qwen::flash_next) misses: u64,
    /// Bytes copied to the card.
    pub(in crate::family::qwen::flash_next) moved: u64,
    /// Bytes of the picked experts: what the launches read.
    pub(in crate::family::qwen::flash_next) needed: u64,
    /// Seconds the host spent staging (the reads, the copies and their wait).
    pub(in crate::family::qwen::flash_next) host_seconds: f64,
}

/// The expert caches of every layer and the copy path that fills them.
pub(in crate::family::qwen::flash_next) struct ExpertCache {
    ctx: Rc<CudaContext>,
    copy: DeviceStream,
    copied: DeviceEvent,
    layers: Vec<CacheLayer>,
    store: Vec<PinnedBuf>,
    /// Host staging of the records a file-backed layer misses.
    bounce: PinnedBuf,
    experts: usize,
    slots: usize,
    totals: StreamTotals,
    /// Why a pass failed after it queued a copy: the slots hold bytes the row does not describe.
    poisoned: Option<String>,
}

impl ExpertCache {
    /// The cache of `plan.slots` experts per layer. The layers `plan` homes in pinned memory are
    /// read into pinned buffers now; the others keep their file open. Every slot starts holding
    /// the expert of its own number, which is uploaded here.
    pub(in crate::family::qwen::flash_next) fn new(
        model: &GgufModel,
        stacks: &[LayerStacks],
        plan: &CachePlan,
        ctx: &Rc<CudaContext>,
    ) -> Result<Self, EngError> {
        let experts = model.config().experts();
        let slots = plan.slots;
        let weights = &*model.weights;
        let mut files: Vec<Option<Arc<File>>> = vec![None; weights.paths().len()];
        let mut store: Vec<PinnedBuf> = Vec::new();
        let index = model.expert_index()?;
        let copy = ctx.stream()?;
        let mut layers = Vec::with_capacity(stacks.len());
        let mut bounce_bytes = 0usize;
        for (layer, facts) in stacks.iter().enumerate() {
            let slices = index
                .slices(layer, 0)
                .map_err(|error| refuse(format!("layer {layer}: {error:?}")))?;
            let [gate, up, down] = slices
                .three()
                .map_err(|error| refuse(format!("layer {layer}: {error:?}")))?;
            let mut record = [0usize; 3];
            let mut sources: Vec<Source> = Vec::with_capacity(3);
            let mut bufs: Vec<NativeWeight> = Vec::with_capacity(3);
            for (part, (stack, slice)) in facts.stacks.iter().zip([gate, up, down]).enumerate() {
                let quant = QuantType::from_gguf(stack.ty).ok_or_else(|| {
                    refuse(format!("{} is {}, with no kernel", stack.name, stack.ty.name()))
                })?;
                let bytes = usize::try_from(stack.bytes)
                    .map_err(|_| refuse(format!("{} overflows usize", stack.name)))?;
                if bytes % experts != 0 || stack.rows % experts != 0 {
                    return Err(refuse(format!(
                        "{} holds {bytes} bytes in {} rows, not whole records of {experts} experts",
                        stack.name, stack.rows
                    )));
                }
                let each = bytes / experts;
                record[part] = each;
                let source = match plan.homes[layer] {
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
                        Source::Pinned {
                            buf: store.len() - 1,
                        }
                    }
                    _ => Source::File {
                        file: open_shard(weights.paths(), stack.shard, &mut files)?,
                        offset: slice.file_range.start,
                    },
                };
                // The slots start as experts 0..slots: the first `slots` records of the stack.
                let mut first = vec![0u8; each * slots];
                weights.read_range(&stack.name, 0, &mut first)?;
                let buffer = ctx.upload(&first)?;
                bufs.push(NativeWeight::from_buffer(
                    buffer,
                    quant,
                    stack.k,
                    stack.rows / experts * slots,
                    Some(slots),
                )?);
                sources.push(source);
            }
            bounce_bytes = bounce_bytes.max(plan.top_k * record.iter().sum::<usize>());
            let stacks3: [NativeWeight; 3] = bufs
                .try_into()
                .map_err(|_| refuse(format!("layer {layer} lost a stack")))?;
            let sources3: [Source; 3] = sources
                .try_into()
                .map_err(|_| refuse(format!("layer {layer} lost a source")))?;
            layers.push(CacheLayer {
                lru: LayerLru::from_identity(experts, slots),
                stacks: stacks3,
                sources: sources3,
                record,
            });
        }
        let bounce = ctx.pinned_buf(bounce_bytes.max(4096))?;
        Ok(Self {
            ctx: Rc::clone(ctx),
            copy,
            copied: ctx.event()?,
            layers,
            store,
            bounce,
            experts,
            slots,
            totals: StreamTotals::default(),
            poisoned: None,
        })
    }

    /// Experts held per layer.
    pub(in crate::family::qwen::flash_next) fn slots(&self) -> usize {
        self.slots
    }

    /// Stack `stack` (0 gate, 1 up, 2 down) of `layer`: `slots` experts.
    pub(in crate::family::qwen::flash_next) fn stack(
        &self,
        layer: usize,
        stack: usize,
    ) -> Result<&NativeWeight, EngError> {
        self.layers
            .get(layer)
            .map(|layer| &layer.stacks[stack])
            .ok_or_else(|| refuse(format!("layer {layer} does not exist")))
    }

    /// Makes every picked expert of `layer` hold a slot, copying the missed ones in, and returns
    /// the slot of each pick in pick order. The copies are complete when this returns.
    pub(in crate::family::qwen::flash_next) fn stage(
        &mut self,
        layer: usize,
        picks: &[u32],
    ) -> Result<Vec<i32>, EngError> {
        let started = Instant::now();
        if let Some(why) = &self.poisoned {
            return Err(refuse(format!(
                "layer {layer}: a pass failed earlier ({why}); the cache is poisoned"
            )));
        }
        let cache = self
            .layers
            .get_mut(layer)
            .ok_or_else(|| refuse(format!("layer {layer} does not exist")))?;
        let (order, plan, next) = plan_pass(&cache.lru, self.experts, picks)?;
        let record_bytes: usize = cache.record.iter().sum();
        // Every record of a file-backed layer is read into the bounce buffer before the first
        // copy queues: the buffer cannot be written while a copy of it is outstanding.
        let mut bounce_at = 0usize;
        let mut from_file = false;
        let mut jobs: Vec<(usize, usize, usize, Option<usize>)> = Vec::new();
        for slot_move in &plan.moves {
            let (slot, incoming) = (usize::from(slot_move.slot), usize::from(slot_move.incoming));
            for part in 0..3 {
                let each = cache.record[part];
                let staged = match &cache.sources[part] {
                    Source::Pinned { .. } => None,
                    Source::File { file, offset } => {
                        let mut host = vec![0u8; each];
                        file.read_exact_at(&mut host, *offset + (incoming * each) as u64)
                            .map_err(|error| refuse(format!("read of expert {incoming}: {error}")))?;
                        self.bounce.write(bounce_at, &host)?;
                        from_file = true;
                        bounce_at += each;
                        Some(bounce_at - each)
                    }
                };
                jobs.push((part, slot, incoming, staged));
            }
        }
        // From the first copy on, a failure leaves copies in flight on `copy` and slots whose bytes
        // the live row does not describe: the row is not committed and the cache is poisoned.
        let copied = (|| -> Result<(), EngError> {
            for (part, slot, incoming, staged) in jobs {
                let each = cache.record[part];
                let dst = cache.stacks[part].buffer().range(slot * each, each)?;
                match (&cache.sources[part], staged) {
                    (Source::Pinned { buf }, _) => {
                        self.ctx.copy_from_pinned(
                            &self.copy,
                            &mut self.store[*buf],
                            incoming * each,
                            dst,
                        )?;
                    }
                    (Source::File { .. }, Some(at)) => {
                        self.ctx
                            .copy_from_pinned(&self.copy, &mut self.bounce, at, dst)?;
                    }
                    (Source::File { .. }, None) => {
                        return Err(refuse("a file record was not staged"));
                    }
                }
            }
            if !plan.moves.is_empty() {
                self.copy.record(&self.copied)?;
                self.copied.synchronize()?;
                if from_file {
                    self.bounce.wait(&self.copied)?;
                }
                for buffer in &mut self.store {
                    if buffer.pending_stream().is_some() {
                        buffer.wait(&self.copied)?;
                    }
                }
            }
            Ok(())
        })();
        if let Err(error) = copied {
            self.poisoned = Some(format!("{error:?}"));
            return Err(error);
        }
        cache.lru = next;
        self.totals.visits += 1;
        self.totals.hits += plan.hits as u64;
        self.totals.misses += plan.moves.len() as u64;
        self.totals.moved += (plan.moves.len() * record_bytes) as u64;
        self.totals.needed += (order.len() * record_bytes) as u64;
        self.totals.host_seconds += started.elapsed().as_secs_f64();
        picks
            .iter()
            .map(|&pick| {
                cache
                    .lru
                    .slot_of(pick as u16)
                    .map(i32::from)
                    .ok_or_else(|| refuse(format!("expert {pick} holds no slot after its pass")))
            })
            .collect()
    }

    /// What the cache has done so far.
    pub(in crate::family::qwen::flash_next) fn totals(&self) -> StreamTotals {
        self.totals
    }
}

impl Drop for ExpertCache {
    fn drop(&mut self) {
        // A failed pass can leave copies in flight on `copy`: they end here, before the slot
        // stacks and the pinned buffers free under them.
        let _ = self.copy.synchronize();
    }
}

/// The pass of `picks` over the row of a layer with `experts` experts, planned on a copy of the
/// row: a refused pass leaves the live row as it was, and the caller commits `next` only once
/// the copies of the misses have landed.
fn plan_pass(
    lru: &LayerLru,
    experts: usize,
    picks: &[u32],
) -> Result<(Vec<u16>, PassPlan, LayerLru), EngError> {
    let mut demanded = vec![false; experts];
    let mut order: Vec<u16> = Vec::with_capacity(picks.len());
    for &pick in picks {
        let expert = u16::try_from(pick)
            .ok()
            .filter(|&expert| usize::from(expert) < experts)
            .ok_or_else(|| refuse(format!("expert {pick} of {experts}")))?;
        if !std::mem::replace(&mut demanded[usize::from(expert)], true) {
            order.push(expert);
        }
    }
    let mut next = lru.clone();
    let plan = next.apply(&order, &demanded).map_err(refuse)?;
    Ok((order, plan, next))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A refused pass leaves the live row as it was; a planned pass moves the picks on the copy
    /// only, so a copy that fails before the commit cannot publish a hit the slots do not hold.
    #[test]
    fn a_pass_is_planned_on_a_copy_of_the_row() {
        let lru = LayerLru::from_identity(4, 2);
        let before = lru.state();
        let refused = plan_pass(&lru, 4, &[0, 1, 2]);
        assert!(refused.is_err(), "two slots cannot take three demanded picks");
        assert_eq!(lru.state(), before);
        let (_, plan, next) = plan_pass(&lru, 4, &[2, 0]).expect("two picks fit");
        assert_eq!((plan.hits, plan.moves.len()), (1, 1));
        assert!(next.slot_of(2).is_some());
        assert_eq!(lru.slot_of(2), None, "the live row waits for the copies");
        assert_eq!(lru.state(), before);
    }
}

/// The shard's file, opened once through the local-storage guard.
fn open_shard(
    paths: &[std::path::PathBuf],
    shard: usize,
    files: &mut [Option<Arc<File>>],
) -> Result<Arc<File>, EngError> {
    if let Some(Some(open)) = files.get(shard) {
        return Ok(Arc::clone(open));
    }
    let path = paths
        .get(shard)
        .ok_or_else(|| refuse(format!("shard {shard} is not open")))?;
    crate::storage_guard::require_local(path, "expert stacks")?;
    let file = Arc::new(
        File::open(path)
            .map_err(|error| refuse(format!("cannot open {}: {error}", path.display())))?,
    );
    files[shard] = Some(Arc::clone(&file));
    Ok(file)
}
