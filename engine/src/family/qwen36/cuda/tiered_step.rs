//! The tiered runtime of the resident step: what the step needs, beyond the plain weights, to run
//! the tiered MoE block of `moe/tiered.rs` in every layer.
//!
//! WHAT: [`TieredRuntime`] owns the per-layer expert banks, the scratch buffers and the host
//! (`TieredHost`: policy, residency ledger, CPU pool), and borrows each layer's router and shared
//! expert weights from the loaded model ([`LayerParts`]); the routed expert stacks are NOT loaded
//! in this mode (the bank holds the hot experts). [`TieredRuntime::run_layer`] runs one layer's
//! block, [`TieredRuntime::end_token`] runs the token boundary: policy P decides the swaps and the
//! bank slots are filled with blocking writes.

use std::ops::Range;
use std::sync::Arc;

use super::mixer_weights::DenseWeight;
use super::moe::tiered::{
    LayerPlan, RouteRead, TieredBlock, TieredLayerHost, TieredScratch, run_moe_block_tiered,
};
use super::moe::{Launch, MoeBuffers, MoeDims, MoeLayerWeights, StackWeights};
use super::run_report::HostSyncs;
use crate::EngError;
use crate::family::qwen36::tiered::cpu_pool::MappedRegion;
use crate::family::qwen36::tiered::host::{BankDevice, ChunkColdStage, HostError, TieredHost};
use crate::family::qwen36::tiered::locator::ExpertRanges;
use crate::family::qwen36::tiered::report::TokenReport;
use crate::kernels::cuda::{
    CudaContext, DeviceBuffer, DeviceEvent, DeviceStream, ExpertBank, ExpertBankGeometry,
    NativeWeight, PinnedBuf,
};

/// The cold staging bank of the chunked tiered prefill: ONE zero-initialised
/// [`ExpertBank`] of `cap` slots reused by every layer — the misses of the layer's
/// hottest `cap` distinct cold experts are gathered from the checkpoint mapping into a
/// pinned ring and DMA'd in stack-major, then served by the same grouped MMQ launches
/// as the banks (numerically identical to a hit; strictly closer than the CPU partial).
/// Zero-filled at attach, so an unstaged slot a bug could still address yields a finite
/// zero row (the `0 * Inf` argument of the dummy slot).
///
/// The chunk path reassigns the whole staging every layer (the channels travel in the
/// plan's `cold_ids`); a decode-side reuse would add an expert→slot directory on top.
pub(super) struct ColdStaging {
    bank: ExpertBank,
    cap: usize,
    ring: Vec<RingSlot>,
}

/// One slot of the pinned staging ring: a slice-sized page-locked buffer, the event
/// recorded behind its last DMA and whether a transfer is still outstanding. The host
/// gather (mmap → pinned) runs ahead of the device inside the ring, so the ~9 ms/layer
/// of memcpy hides under the ~12 ms/layer of DMA.
struct RingSlot {
    buf: PinnedBuf,
    event: DeviceEvent,
    pending: bool,
}

/// Ring depth: eight batch-sized slots (~18.9 MiB at the Qwen slice of 589,824 B and
/// batches of four) — enough ahead-runs to keep the DMA fed while one slot waits its
/// event.
const COLD_RING_DEPTH: usize = 8;

impl ColdStaging {
    /// Builds the zero-filled staging bank of `cap` slots plus its ring. Refuses before
    /// any allocation for a cap the bank geometry refuses (below `top_k`).
    pub(super) fn new(
        ctx: &CudaContext,
        geometry: ExpertBankGeometry,
        cap: usize,
    ) -> Result<Self, EngError> {
        let bank = ExpertBank::new(ctx, geometry, cap)?;
        let (gate_up, down) = bank.slice_bytes()?;
        let slot_bytes = gate_up.max(down) * Self::BATCH;
        let mut ring = Vec::with_capacity(COLD_RING_DEPTH);
        for _ in 0..COLD_RING_DEPTH {
            ring.push(RingSlot {
                buf: ctx.pinned_buf(slot_bytes)?,
                event: ctx.event()?,
                pending: false,
            });
        }
        Ok(Self { bank, cap, ring })
    }

    pub(super) fn cap(&self) -> usize {
        self.cap
    }

    /// The staged gate/up/down stacks for the cold grouped launches.
    pub(super) fn weights(&self) -> (&NativeWeight, &NativeWeight, &NativeWeight) {
        (self.bank.gate(), self.bank.up(), self.bank.down())
    }

    /// Slices per ring batch: consecutive cold channels are contiguous in the staging
    /// stacks (the list is channel-ordered), so one pinned slot carries a batch as one
    /// wide DMA — a 590 KiB copy runs the link at a third of a 2.4 MiB one (measured
    /// 3.4 GB/s per-slice early layers).
    const BATCH: usize = 4;

    /// Stages one chunk layer's cold experts: for each stack (gate, up, down —
    /// stack-major), batches of [`Self::BATCH`] consecutive channels' slices are
    /// gathered mapping → ring slot on the host and queued ring slot → staging as one
    /// DMA on `stream` (enqueue-only; the waits that recycle a ring slot synchronize
    /// only that slot's event). The launches queued behind this on the same stream read
    /// the staged bytes in order.
    pub(super) fn stage(
        &mut self,
        ctx: &CudaContext,
        stream: &DeviceStream,
        cold: &[ChunkColdStage],
        region: &Arc<dyn MappedRegion>,
    ) -> Result<(), EngError> {
        if cold.is_empty() {
            return Ok(());
        }
        let bytes = region.bytes();
        let ColdStaging { bank, cap, ring } = self;
        let (gate_up, down) = bank.slice_bytes()?;
        let stacks: [(&DeviceBuffer, usize, fn(&ChunkColdStage) -> &Range<usize>); 3] = [
            (bank.gate().buffer(), gate_up, |c| &c.gate),
            (bank.up().buffer(), gate_up, |c| &c.up),
            (bank.down().buffer(), down, |c| &c.down),
        ];
        let mut at = 0usize;
        for (stack, slice_bytes, range_of) in stacks {
            for batch in cold.chunks(Self::BATCH) {
                let first = usize::from(batch[0].channel);
                if first + batch.len() > *cap {
                    return Err(EngError::BadRequest(format!(
                        "tiered cold staging: channels {first}..{} of {cap}",
                        first + batch.len()
                    )));
                }
                // Consecutive channels, or the batched DMA would scramble slots.
                for (step, entry) in batch.iter().enumerate() {
                    if usize::from(entry.channel) != first + step {
                        return Err(EngError::BadRequest(format!(
                            "tiered cold staging: channel {} follows {} — the staging list is \
                             not channel-ordered",
                            entry.channel, first
                        )));
                    }
                }
                let at_slot = at % ring.len();
                at += 1;
                let slot = &mut ring[at_slot];
                if slot.pending {
                    slot.buf.wait(&slot.event)?;
                    slot.pending = false;
                }
                for (step, entry) in batch.iter().enumerate() {
                    let range = range_of(entry);
                    let slice = bytes.get(range.clone()).ok_or_else(|| {
                        EngError::Checkpoint(format!(
                            "tiered cold staging: expert {} range {range:?} is outside the mapping",
                            entry.expert
                        ))
                    })?;
                    if slice.len() != slice_bytes {
                        return Err(EngError::Checkpoint(format!(
                            "tiered cold staging: expert {} slice holds {} bytes, a slot holds {slice_bytes}",
                            entry.expert,
                            slice.len()
                        )));
                    }
                    slot.buf.write(step * slice_bytes, slice)?;
                }
                let dst = stack.range(first * slice_bytes, batch.len() * slice_bytes)?;
                ctx.copy_from_pinned(stream, &mut slot.buf, 0, dst)?;
                stream.record(&slot.event)?;
                slot.pending = true;
            }
        }
        Ok(())
    }
}

/// The weights of one layer's MoE block that stay in the loaded model: the router, the shared
/// gate vector and the shared expert's three projections.
pub(super) struct LayerParts<'w> {
    pub(super) router: &'w DeviceBuffer,
    pub(super) router_shared: &'w DeviceBuffer,
    pub(super) shared_gate: DenseWeight<'w>,
    pub(super) shared_up: DenseWeight<'w>,
    pub(super) shared_down: DenseWeight<'w>,
}

fn host_error(error: HostError) -> EngError {
    EngError::Backend {
        operation: "tiered_host",
        message: error.to_string(),
    }
}

/// The block-side view of the host: converts the plain route into the host's slices and its
/// plan and errors back, and issues the router-guess prefetch fills of the advice that
/// arrived for this layer (advice exists to be early: it publishes at the token boundary
/// and serves a later token, so issuing it here cannot change this token's result).
struct HostAdapter<'h, 'd> {
    host: &'h mut TieredHost,
    fills: &'d mut dyn BankDevice,
}

impl TieredLayerHost for HostAdapter<'_, '_> {
    fn begin_layer(&mut self, layer: usize, route: &RouteRead) -> Result<LayerPlan, EngError> {
        let plan = self
            .host
            .begin_layer(
                layer,
                &route.ids,
                &route.weights,
                &route.hidden,
                &route.resid,
            )
            .map_err(host_error)?;
        if let Some((target, advice)) = self.host.take_arrived_advice() {
            self.host
                .prefetch(self.fills, target, &advice)
                .map_err(host_error)?;
        }
        Ok(LayerPlan {
            slot_ids: plan.slot_ids,
            device_weights: plan.device_weights,
        })
    }

    fn finish_layer(&mut self, layer: usize) -> Result<Option<Vec<f32>>, EngError> {
        self.host.finish_layer(layer).map_err(host_error)
    }
}

/// The device half of a bank change: blocking writes of one expert's three slices.
pub(super) struct BankFill<'d> {
    pub(super) banks: &'d [ExpertBank],
    pub(super) stream: &'d DeviceStream,
}

impl BankDevice for BankFill<'_> {
    fn fill_slot(
        &mut self,
        layer: usize,
        slot: usize,
        ranges: &ExpertRanges,
        region: &Arc<dyn MappedRegion>,
    ) -> Result<(), HostError> {
        let device = |what: String| HostError::Device(what);
        let bank = self
            .banks
            .get(layer)
            .ok_or_else(|| device(format!("no bank for layer {layer}")))?;
        let bytes = region.bytes();
        let cut = |range: &Range<usize>| {
            bytes
                .get(range.clone())
                .ok_or_else(|| device(format!("expert range {range:?} is outside the region")))
        };
        bank.fill(
            self.stream,
            slot,
            cut(&ranges.gate)?,
            cut(&ranges.up)?,
            cut(&ranges.down)?,
        )
        .map_err(|error| device(format!("{error:?}")))
    }
}

pub(super) struct TieredRuntime<'w> {
    parts: Vec<LayerParts<'w>>,
    banks: Vec<ExpertBank>,
    scratch: TieredScratch,
    host: TieredHost,
    /// The cold staging bank of the chunked prefill: `None` when admission left no
    /// spare VRAM for it (the chunk path then keeps the CPU pool for every miss).
    cold: Option<ColdStaging>,
    /// The bank depth (slots per layer) every bank was built with.
    slot_count: usize,
    /// The host waits of the token so far: the layers' and, after `end_token`, the bank fills'.
    syncs: HostSyncs,
}

impl<'w> TieredRuntime<'w> {
    pub(super) fn new(
        parts: Vec<LayerParts<'w>>,
        banks: Vec<ExpertBank>,
        scratch: TieredScratch,
        host: TieredHost,
        cold: Option<ColdStaging>,
    ) -> Result<Self, EngError> {
        if parts.len() != banks.len() {
            return Err(EngError::BadRequest(format!(
                "tiered runtime: {} layers of weights but {} banks",
                parts.len(),
                banks.len()
            )));
        }
        let slot_count = banks.first().map(ExpertBank::slots_of).unwrap_or(0);
        Ok(Self {
            parts,
            banks,
            scratch,
            host,
            cold,
            slot_count,
            syncs: HostSyncs::default(),
        })
    }

    /// The weights of `layer` for any MoE launch — the per-token block today, the chunk
    /// arm of batched prefill (MR #20 slice 4) tomorrow: the router and shared parts from
    /// the loaded model, the three BANK stacks in place of the full-depth stacks.
    pub(super) fn layer_weights(&self, layer: usize) -> Result<MoeLayerWeights<'_>, EngError> {
        let missing = || EngError::BadRequest(format!("tiered runtime: layer {layer} has no bank"));
        let (parts, bank) = (
            self.parts.get(layer).ok_or_else(missing)?,
            self.banks.get(layer).ok_or_else(missing)?,
        );
        Ok(MoeLayerWeights {
            router: parts.router,
            router_shared: parts.router_shared,
            gate: StackWeights::Native(bank.gate()),
            up: StackWeights::Native(bank.up()),
            down: StackWeights::Native(bank.down()),
            shared_gate: parts.shared_gate,
            shared_up: parts.shared_up,
            shared_down: parts.shared_down,
        })
    }

    /// The bank depth every layer was built with (the tiered plan's slots per layer).
    pub(super) fn slots(&self) -> usize {
        self.banks.first().map(|_| self.slot_count).unwrap_or(0)
    }

    /// The router buffers of `layer` (the raw F32 matrix and the shared gate),
    /// at the parts' 'w lifetime — the chunk's router stage reads them.
    pub(super) fn router_of(
        &self,
        layer: usize,
    ) -> Result<
        (
            &'w crate::kernels::cuda::DeviceBuffer,
            &'w crate::kernels::cuda::DeviceBuffer,
        ),
        EngError,
    > {
        let parts = self.parts.get(layer).ok_or_else(|| {
            EngError::BadRequest(format!("tiered runtime: layer {layer} has no parts"))
        })?;
        Ok((parts.router, parts.router_shared))
    }

    /// Starts the grouped pool serving of a chunk layer's misses (the delegate to
    /// the host's `start_chunk_pool`): one GroupedExpertJob per DISTINCT missed expert,
    /// started BEFORE the GPU stages so the CPU work overlaps them.
    pub(super) fn start_chunk_pool(
        &mut self,
        layer: usize,
        misses: &[crate::family::qwen36::tiered::host::ChunkMissGroup],
        block: &[f32],
        m: usize,
    ) -> Result<
        Option<crate::family::qwen36::tiered::cpu_pool::PendingChunk>,
        crate::family::qwen36::tiered::host::HostError,
    > {
        self.host.start_chunk_pool(layer, misses, block, m)
    }

    /// The M-row classification delegate: the chunk arm's route readback flows through
    /// the host's `begin_chunk_layer`, with the cold staging's capacity as the cut.
    pub(super) fn begin_chunk(
        &mut self,
        layer: usize,
        rows: &[crate::family::qwen36::tiered::host::ChunkRouteRow],
    ) -> Result<
        (
            crate::family::qwen36::tiered::host::ChunkLayerPlanHost,
            Vec<crate::family::qwen36::tiered::host::ChunkMissGroup>,
            Vec<crate::family::qwen36::tiered::host::ChunkColdStage>,
        ),
        crate::family::qwen36::tiered::host::HostError,
    > {
        let cap = self.cold.as_ref().map_or(0, ColdStaging::cap);
        self.host.begin_chunk_layer(layer, rows, cap)
    }

    /// The cold staging depth (0 when admission built none).
    pub(super) fn cold_cap(&self) -> usize {
        self.cold.as_ref().map_or(0, ColdStaging::cap)
    }

    /// The checkpoint mapping the cold staging fill gathers its slices from.
    pub(super) fn cold_region(&self) -> Arc<dyn MappedRegion> {
        self.host.cold_region()
    }

    /// Queues one layer's cold staging fills (the delegate over [`ColdStaging::stage`]);
    /// a no-op without a staging bank or cold experts.
    pub(super) fn stage_cold(
        &mut self,
        ctx: &CudaContext,
        stream: &DeviceStream,
        cold: &[crate::family::qwen36::tiered::host::ChunkColdStage],
        region: &Arc<dyn MappedRegion>,
    ) -> Result<(), EngError> {
        if let Some(staging) = self.cold.as_mut() {
            staging.stage(ctx, stream, cold, region)?;
        }
        Ok(())
    }

    /// The cold staging stacks for the cold grouped launches, borrowed beside the banks'
    /// own in `layer_weights`.
    pub(super) fn cold_weights(
        &self,
    ) -> Result<(&NativeWeight, &NativeWeight, &NativeWeight), EngError> {
        let staging = self.cold.as_ref().ok_or_else(|| {
            EngError::BadRequest(String::from("tiered runtime: no cold staging bank"))
        })?;
        Ok(staging.weights())
    }

    /// Runs the MoE block of `layer` with the host decision between the router and the experts.
    /// The weights construction stays inline: block borrow of `scratch`/`host` beside an
    /// immutable `self` is the disjoint-field shape the accessor cannot express.
    pub(super) fn run_layer(
        &mut self,
        launch: &Launch<'_>,
        layer: usize,
        bufs: &MoeBuffers<'_>,
        dims: &MoeDims,
    ) -> Result<(), EngError> {
        let missing = || EngError::BadRequest(format!("tiered runtime: layer {layer} has no bank"));
        let (parts, bank) = (
            self.parts.get(layer).ok_or_else(missing)?,
            self.banks.get(layer).ok_or_else(missing)?,
        );
        let weights = MoeLayerWeights {
            router: parts.router,
            router_shared: parts.router_shared,
            gate: StackWeights::Native(bank.gate()),
            up: StackWeights::Native(bank.up()),
            down: StackWeights::Native(bank.down()),
            shared_gate: parts.shared_gate,
            shared_up: parts.shared_up,
            shared_down: parts.shared_down,
        };
        let mut block = TieredBlock {
            weights: &weights,
            bufs,
            scratch: &mut self.scratch,
            dims,
        };
        let mut fills = BankFill {
            banks: &self.banks,
            stream: launch.stream,
        };
        let made = run_moe_block_tiered(
            launch,
            &mut HostAdapter {
                host: &mut self.host,
                fills: &mut fills,
            },
            layer,
            &mut block,
        )?;
        self.syncs.add(made);
        Ok(())
    }

    pub(super) fn preload(&mut self, stream: &DeviceStream) -> Result<usize, EngError> {
        let mut fills = BankFill {
            banks: &self.banks,
            stream,
        };
        self.host.preload_all(&mut fills).map_err(host_error)
    }

    pub(super) fn abort_token(&mut self) {
        self.host.abort_token();
        self.syncs = HostSyncs::default();
    }

    /// The host waits of the token that just ended (layers and fills), and zero for the next one.
    pub(super) fn take_syncs(&mut self) -> HostSyncs {
        std::mem::take(&mut self.syncs)
    }

    pub(super) fn end_token(&mut self, stream: &DeviceStream) -> Result<TokenReport, EngError> {
        let mut fills = BankFill {
            banks: &self.banks,
            stream,
        };
        let report = self.host.end_token(&mut fills).map_err(host_error)?;
        self.syncs.writes += report.swaps * ExpertBank::FILL_WRITES;
        Ok(report)
    }
}
