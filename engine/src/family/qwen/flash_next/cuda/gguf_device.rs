//! The card implementation of [`TokenDevice`] for a Flash-Next GGUF: the decode step with the
//! native MoE block over resident experts (`gguf_moe`). The driver calls are the same as
//! [`CardDevice`](super::device::CardDevice)'s; this device has no expert bank, so it reports no
//! bank counters.

#![cfg_attr(not(test), expect(dead_code, reason = "no caller outside the tests"))]

use std::ops::Range;

use super::{
    driver::{DriverShape, StateIo, TokenDevice, TokenDriver},
    feeds::{StartShape, StartSink},
    gguf_moe::GgufMoe,
    pack_feeds::PackFeeds,
    state::StateSlot,
    step::FlashStep,
    step_plan::Slot,
};
use crate::{
    EngError,
    family::{
        exl3_tensors::f32_le_bytes,
        flash_next::load::gguf::{GgufModel, feeds::gguf_feeds},
        project::cuda::CudaWeights,
    },
    kernels::cuda::{
        CudaContext, DeviceStream,
        prefill::gdn_scan_seq::{GdnForm, GdnGate, GdnKeyHeads, GdnL2},
        resident::step_params::StepParams,
    },
};

/// The DeltaNet scan of a GGUF: the tiled value-head order the converter wrote, the epsilon-added
/// L2 norm and sigmoid gate of the Flash-Next layers.
const GGUF_GDN: GdnForm = GdnForm {
    key_heads: GdnKeyHeads::Tiled,
    l2: GdnL2::AddEps,
    gate: GdnGate::Sigmoid,
};

/// The token driver of a Flash-Next GGUF over resident experts, for `positions` cached positions:
/// the decode step, the native MoE block and the host feeds of the file. `threads` is the host PLE
/// projection's thread count.
pub(in crate::family::qwen::flash_next) fn gguf_driver(
    model: &'static GgufModel,
    ctx: &'static CudaContext,
    stream: &'static DeviceStream,
    weights: &'static CudaWeights,
    positions: usize,
    threads: usize,
) -> Result<TokenDriver<GgufDevice, PackFeeds>, EngError> {
    let config = model.config();
    let layers = config.layers();
    let step = FlashStep::from_shape(
        ctx,
        stream,
        weights,
        (config.model_shape()?, config.kinds().to_vec()),
        positions,
        1,
        0..layers,
    )?
    .with_gdn_form(GGUF_GDN);
    let moe = GgufMoe::new(ctx, weights, &step.moe_dims(1))?;
    let feeds = gguf_feeds(model, threads)?;
    let ple = config.model_shape()?.ple;
    let shape = DriverShape {
        layers,
        ple_layer: config.ple_layer(),
        positions,
        start: StartShape {
            sampler_vocab: step.model.vocab,
            stream: step.geometry.branches * step.geometry.hidden,
            ple_embed: ple.embed(),
        },
    };
    TokenDriver::new(GgufDevice::new(step, moe), feeds, shape)
}

pub(in crate::family::qwen::flash_next) struct GgufDevice {
    step: FlashStep<'static>,
    moe: GgufMoe<'static>,
    /// `branches * hidden`.
    stream_width: usize,
    /// The width of the head's output (the vocabulary).
    logits_width: usize,
}

impl GgufDevice {
    fn new(step: FlashStep<'static>, moe: GgufMoe<'static>) -> Self {
        let stream_width = step.geometry.branches * step.geometry.hidden;
        let logits_width = step.model.vocab;
        Self {
            step,
            moe,
            stream_width,
            logits_width,
        }
    }
}

impl StartSink for GgufDevice {
    fn write_step_params(&mut self, token: i32, pos: i32) -> Result<(), EngError> {
        let params = StepParams::new(token, pos)?;
        let range = self.step.params()?;
        range
            .buffer()
            .write_at(self.step.stream, range.offset(), &params.to_bytes())
    }

    fn write_stream(&mut self, stream: &[f32]) -> Result<(), EngError> {
        let range = self.step.slot(Slot::Stream)?;
        range
            .buffer()
            .write_at(self.step.stream, range.offset(), &f32_le_bytes(stream))
    }

    /// The PLE projections run on the host; the start needs no copy of the embedding.
    fn write_ple_embedding(&mut self, _embedding: &[f32]) -> Result<(), EngError> {
        Ok(())
    }
}

impl TokenDevice for GgufDevice {
    fn begin_token(&mut self) -> Result<(), EngError> {
        self.step.begin_token()
    }

    fn layers(&mut self, range: Range<usize>, pos: usize) -> Result<(), EngError> {
        if range.is_empty() {
            return Ok(());
        }
        self.step.run_layers(range, pos, 1, &mut self.moe)
    }

    fn read_stream(&mut self) -> Result<Vec<f32>, EngError> {
        self.step
            .read_f32(self.step.slot(Slot::Stream)?, self.stream_width)
    }

    fn write_stream_back(&mut self, stream: &[f32]) -> Result<(), EngError> {
        StartSink::write_stream(self, stream)
    }

    fn head(&mut self) -> Result<(), EngError> {
        self.step.head(1)
    }

    fn read_logits(&mut self) -> Result<Vec<f32>, EngError> {
        self.step
            .read_f32(self.step.slot(Slot::Logits)?, self.logits_width)
    }

    fn check_health(&mut self) -> Result<(), EngError> {
        self.step.finish_token()
    }

    fn reset(&mut self) -> Result<(), EngError> {
        self.step.reset()
    }
}

impl StateIo for GgufDevice {
    fn read_fixed_state(&self) -> Result<Vec<(StateSlot, Vec<u8>)>, EngError> {
        self.step.read_fixed_state()
    }

    fn fixed_state_bytes(&self) -> u64 {
        self.step.fixed_state_bytes()
    }

    fn write_fixed_state(&mut self, parts: &[(StateSlot, Vec<u8>)]) -> Result<(), EngError> {
        self.step.write_fixed_state(parts)
    }
}
