//! The MoE block of a Flash-Next GGUF on the card: the native block that serves Qwen3.6's GGUF
//! (`qwen36::cuda::moe::run_moe_block`: the router, the merged expert matvecs over device ids,
//! the batched SwiGLU and q8_1 quantize, the shared expert and the combine) over this family's
//! buffers.
//!
//! The Flash-Next step hands the block the hyper-mixed row as F32 (`norm_row`) in the buffers of
//! its own plan; the native block reads that row as a q8_1 pack, so the first launch quantizes it
//! into a pack this struct owns, with the packs of the expert down projections and the shared
//! expert. The routed experts are native stacks held on the card (`trunk::expert_key`).

#![cfg_attr(not(test), expect(dead_code, reason = "no caller outside the tests"))]

use super::step::{MoeBlock, refuse};
use crate::{
    EngError,
    family::{
        exl3::{
            launch::{Launch, MoeDims, RouteBufs},
            moe::Exl3MoeBufs,
        },
        exl3_pack::layer_name,
        flash_next::load::gguf::trunk::expert_key,
        project::cuda::{CudaWeight, CudaWeights},
        qwen36::cuda::{
            mixer_weights::DenseWeight,
            moe::{ExpertBufs, MoeBuffers, MoeLayerWeights, Packs, StackWeights, run_moe_block},
        },
    },
    kernels::cuda::{
        CudaContext, DeviceBuffer, DeviceRange,
        resident::quantize_q8_1::{QuantizeCall, q8_1_bytes},
    },
};

/// The block over resident native expert stacks.
pub(in crate::family::qwen::flash_next) struct GgufMoe<'w> {
    weights: &'w CudaWeights,
    moe: DeviceBuffer,
    down: DeviceBuffer,
    shared: DeviceBuffer,
}

fn pack(ctx: &CudaContext, k: usize, rows: usize) -> Result<DeviceBuffer, EngError> {
    let bytes = q8_1_bytes(k, rows).ok_or_else(|| refuse("a q8_1 pack size overflows".into()))?;
    ctx.upload(&vec![0u8; bytes])
}

impl<'w> GgufMoe<'w> {
    /// The block for a model of `dims` over the uploaded `weights`.
    pub(in crate::family::qwen::flash_next) fn new(
        ctx: &CudaContext,
        weights: &'w CudaWeights,
        dims: &MoeDims,
    ) -> Result<Self, EngError> {
        Ok(Self {
            weights,
            moe: pack(ctx, dims.hidden, 1)?,
            down: pack(ctx, dims.inter, dims.top_k)?,
            shared: pack(ctx, dims.shared_inter, 1)?,
        })
    }

    fn native(&self, key: &str) -> Result<&'w crate::kernels::cuda::NativeWeight, EngError> {
        match self.weights.get(key) {
            Some(CudaWeight::Native { weight, .. }) => Ok(weight),
            _ => Err(refuse(format!("{key} is not an uploaded native matrix"))),
        }
    }

    fn raw(&self, key: &str) -> Result<&'w DeviceBuffer, EngError> {
        match self.weights.get(key) {
            Some(CudaWeight::Raw { buffer, .. }) => Ok(buffer),
            _ => Err(refuse(format!("{key} is not an uploaded raw tensor"))),
        }
    }
}

fn whole(buffer: &DeviceBuffer) -> Result<DeviceRange<'_>, EngError> {
    buffer.range(0, buffer.bytes())
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
        let at = |suffix: &str| layer_name(layer, suffix);
        let weights = MoeLayerWeights {
            router: self.raw(&at("mlp.gate.weight"))?,
            router_shared: self.raw(&at("mlp.shared_expert_gate.weight"))?,
            gate: StackWeights::Native(self.native(&expert_key(layer, "gate"))?),
            up: StackWeights::Native(self.native(&expert_key(layer, "up"))?),
            down: StackWeights::Native(self.native(&expert_key(layer, "down"))?),
            shared_gate: DenseWeight::Native(self.native(&at("mlp.shared_expert.gate_proj"))?),
            shared_up: DenseWeight::Native(self.native(&at("mlp.shared_expert.up_proj"))?),
            shared_down: DenseWeight::Native(self.native(&at("mlp.shared_expert.down_proj"))?),
        };
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
        let route = &bufs.route;
        let buffers = MoeBuffers {
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
        };
        run_moe_block(launch, &weights, &buffers, dims)
    }
}
