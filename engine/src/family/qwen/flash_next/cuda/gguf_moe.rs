//! The MoE block of a Flash-Next GGUF on the card: the native block that serves Qwen3.6's GGUF
//! (`qwen36::cuda::moe::run_moe_block`: the router, the merged expert matvecs over device ids,
//! the batched SwiGLU and q8_1 quantize, the shared expert and the combine) over this family's
//! buffers.
//!
//! The Flash-Next step hands the block the hyper-mixed row as F32 (`norm_row`) in the buffers of
//! its own plan; the native block reads that row as a q8_1 pack, so the first launch quantizes it
//! into a pack this struct owns, with the packs of the expert down projections and the shared
//! expert. The routed experts are native stacks held on the card (`trunk::expert_key`).

use super::{
    gguf_stream::{ExpertStreams, StreamTotals},
    step::{MoeBlock, refuse},
};
use crate::family::engine_owner::Handle;
use crate::{
    EngError,
    family::{
        exl3::{
            launch::{Launch, MoeDims, RouteBufs},
            moe::Exl3MoeBufs,
        },
        exl3_pack::layer_name,
        qwen::flash_next::load::gguf::trunk::expert_key,
        project::cuda::{CudaWeight, CudaWeights},
        qwen::qwen36::cuda::{
            mixer_weights::DenseWeight,
            moe::{
                ExpertBufs, MoeBuffers, MoeLayerWeights, Packs, StackWeights, run_moe_block, run_moe_ops,
            },
        },
    },
    kernels::cuda::{
        CudaContext, DeviceBuffer, DeviceRange, NativeWeight,
        resident::quantize_q8_1::{QuantizeCall, q8_1_bytes},
    },
};

/// The block over resident native expert stacks.
pub(in crate::family::qwen::flash_next) struct GgufMoe<'w> {
    weights: Handle<'w, CudaWeights>,
    moe: DeviceBuffer,
    down: DeviceBuffer,
    shared: DeviceBuffer,
    /// The experts of the layers that do not fit the card, staged per token.
    streams: Option<ExpertStreams>,
}

fn pack(ctx: &CudaContext, k: usize, rows: usize) -> Result<DeviceBuffer, EngError> {
    let bytes = q8_1_bytes(k, rows).ok_or_else(|| refuse("a q8_1 pack size overflows".into()))?;
    ctx.upload(&vec![0u8; bytes])
}

impl<'w> GgufMoe<'w> {
    /// The block for a model of `dims` over the uploaded `weights`.
    pub(in crate::family::qwen::flash_next) fn new(
        ctx: &CudaContext,
        weights: impl Into<Handle<'w, CudaWeights>>,
        dims: &MoeDims,
        streams: Option<ExpertStreams>,
    ) -> Result<Self, EngError> {
        Ok(Self {
            weights: weights.into(),
            moe: pack(ctx, dims.hidden, 1)?,
            down: pack(ctx, dims.inter, dims.top_k)?,
            shared: pack(ctx, dims.shared_inter, 1)?,
            streams,
        })
    }

    /// What the staging has moved and used so far, when any layer streams.
    pub(in crate::family::qwen::flash_next) fn stream_totals(&self) -> Option<StreamTotals> {
        self.streams.as_ref().map(ExpertStreams::totals)
    }

    fn native(&self, key: &str) -> Result<&crate::kernels::cuda::NativeWeight, EngError> {
        match self.weights.get(key) {
            Some(CudaWeight::Native { weight, .. }) => Ok(weight),
            _ => Err(refuse(format!("{key} is not an uploaded native matrix"))),
        }
    }

    fn raw(&self, key: &str) -> Result<&DeviceBuffer, EngError> {
        match self.weights.get(key) {
            Some(CudaWeight::Raw { buffer, .. }) => Ok(buffer),
            _ => Err(refuse(format!("{key} is not an uploaded raw tensor"))),
        }
    }
}

fn whole(buffer: &DeviceBuffer) -> Result<DeviceRange<'_>, EngError> {
    buffer.range(0, buffer.bytes())
}

/// The picked experts the router wrote, sorted and without repeats.
fn read_picks(launch: &Launch<'_>, ids: &DeviceRange<'_>, experts: usize) -> Result<Vec<u32>, EngError> {
    let mut raw = vec![0u8; ids.bytes()];
    ids.buffer().read_at(launch.stream, ids.offset(), &mut raw)?;
    let mut picks = Vec::with_capacity(raw.len() / 4);
    for chunk in raw.chunks_exact(4) {
        let id = i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        let id = u32::try_from(id)
            .ok()
            .filter(|&id| (id as usize) < experts)
            .ok_or_else(|| refuse(format!("the router picked expert {id} of {experts}")))?;
        picks.push(id);
    }
    picks.sort_unstable();
    picks.dedup();
    Ok(picks)
}

impl GgufMoe<'_> {
    /// The layer's weights over `stacks`: the three expert stacks resident or staged.
    fn layer_weights<'a>(
        &'a self,
        layer: usize,
        stacks: [&'a NativeWeight; 3],
    ) -> Result<MoeLayerWeights<'a>, EngError> {
        let at = |suffix: &str| layer_name(layer, suffix);
        Ok(MoeLayerWeights {
            router: self.raw(&at("mlp.gate.weight"))?,
            router_shared: self.raw(&at("mlp.shared_expert_gate.weight"))?,
            gate: StackWeights::Native(stacks[0]),
            up: StackWeights::Native(stacks[1]),
            down: StackWeights::Native(stacks[2]),
            shared_gate: DenseWeight::Native(self.native(&at("mlp.shared_expert.gate_proj"))?),
            shared_up: DenseWeight::Native(self.native(&at("mlp.shared_expert.up_proj"))?),
            shared_down: DenseWeight::Native(self.native(&at("mlp.shared_expert.down_proj"))?),
        })
    }

    /// The expert stacks of a layer whose experts are on the card.
    fn resident_stacks(&self, layer: usize) -> Result<[&NativeWeight; 3], EngError> {
        Ok([
            self.native(&expert_key(layer, "gate"))?,
            self.native(&expert_key(layer, "up"))?,
            self.native(&expert_key(layer, "down"))?,
        ])
    }

    fn buffers<'b>(
        &'b self,
        bufs: &'b Exl3MoeBufs<'b>,
        moe: DeviceRange<'b>,
    ) -> Result<MoeBuffers<'b>, EngError> {
        let route = &bufs.route;
        Ok(MoeBuffers {
            route: RouteBufs {
                resid_in: route.resid_in,
                resid_out: route.resid_out,
                norm_row: route.norm_row,
                logits: route.logits,
                ids: route.ids,
                weights: route.weights,
                shared_gate: route.shared_gate,
                status: route.status,
            },
            expert: ExpertBufs {
                packs: Packs::Q8 {
                    moe,
                    down: whole(&self.down)?,
                    shared: whole(&self.shared)?,
                },
                expert_gu: bufs.expert_gu,
                expert_down: bufs.expert_down,
                shared_gu: bufs.shared_gu,
                shared_out: bufs.shared_out,
            },
        })
    }
}

impl MoeBlock for GgufMoe<'_> {
    fn run(
        &mut self,
        launch: &Launch<'_>,
        layer: usize,
        _pos: usize,
        bufs: &Exl3MoeBufs<'_>,
        dims: &MoeDims,
    ) -> Result<(), EngError> {
        let moe = whole(&self.moe)?;
        launch.ctx.quantize_q8_1(
            launch.stream,
            QuantizeCall {
                x: bufs.route.norm_row,
                y: moe,
                k: dims.hidden,
                ncols: 1,
            },
        )?;
        let streamed = self
            .streams
            .as_ref()
            .is_some_and(|streams| streams.is_streamed(layer));
        if !streamed {
            let weights = self.layer_weights(layer, self.resident_stacks(layer)?)?;
            let buffers = self.buffers(bufs, moe)?;
            return run_moe_block(launch, &weights, &buffers, dims);
        }
        // A streamed layer: the router runs first, the host learns the picks, the picked experts
        // are copied into the staging slot, and the launches that read them follow the copy.
        let slot = self
            .streams
            .as_mut()
            .map(ExpertStreams::take_slot)
            .ok_or_else(|| refuse("no expert streams".into()))?;
        let streams = self.streams.as_ref().ok_or_else(|| refuse("no expert streams".into()))?;
        let staged = [
            streams.staged(layer, slot, 0)?,
            streams.staged(layer, slot, 1)?,
            streams.staged(layer, slot, 2)?,
        ];
        {
            let weights = self.layer_weights(layer, staged)?;
            let buffers = self.buffers(bufs, moe)?;
            run_moe_ops(launch, &weights, &buffers, dims, 0..1)?;
        }
        let picks = read_picks(launch, &bufs.route.ids, dims.n_expert)?;
        let streams = self.streams.as_mut().ok_or_else(|| refuse("no expert streams".into()))?;
        streams.stage(layer, slot, &picks)?;
        let streams = self.streams.as_ref().ok_or_else(|| refuse("no expert streams".into()))?;
        let staged = [
            streams.staged(layer, slot, 0)?,
            streams.staged(layer, slot, 1)?,
            streams.staged(layer, slot, 2)?,
        ];
        {
            let weights = self.layer_weights(layer, staged)?;
            let buffers = self.buffers(bufs, moe)?;
            run_moe_ops(launch, &weights, &buffers, dims, 1..usize::MAX)?;
        }
        self.streams
            .as_mut()
            .ok_or_else(|| refuse("no expert streams".into()))?
            .release(slot)
    }
}
