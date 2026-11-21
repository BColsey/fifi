//! The card implementation of [`TokenDevice`] for a Flash-Next GGUF: the decode step with the
//! native MoE block over resident experts (`gguf_moe`). The driver calls are the same as
//! [`CardDevice`](super::device::CardDevice)'s; this device has no expert bank, so it reports no
//! bank counters.

use std::{ops::Range, rc::Rc};

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
        engine_owner::{EngineOwner, Handle},
        exl3_tensors::f32_le_bytes,
        flash_next::load::gguf::{GgufModel, feeds::gguf_feeds, trunk::gguf_trunk},
        project::cuda::{CudaProjector, CudaWeights},
    },
    kernels::cuda::{
        CudaContext, DeviceStream, prefill::gdn_scan_seq::GdnForm,
        resident::step_params::StepParams,
    },
};

/// The decode step and the MoE block it runs: the part of the device that is dropped first.
struct GgufRun {
    step: FlashStep<'static>,
    moe: GgufMoe<'static>,
    /// `branches * hidden`.
    stream_width: usize,
    /// The width of the head's output (the vocabulary).
    logits_width: usize,
}

type Owner = EngineOwner<GgufRun, Box<GgufModel>, CudaWeights, DeviceStream, CudaContext>;

/// The token driver of a Flash-Next GGUF over resident experts, for `positions` cached positions:
/// the decode step, the native MoE block and the host feeds of the file. The device is an
/// [`EngineOwner`]: it keeps the card handle, the stream, the uploaded weights and the model, and
/// the step shares them, so dropping the driver quiesces the stream and frees the card in a fixed
/// order. `chunk_limit` is the host bytes one tensor upload reads at a time and `threads` the
/// host PLE projection's thread count.
pub(in crate::family::qwen::flash_next) fn gguf_driver(
    model: GgufModel,
    projector: CudaProjector,
    chunk_limit: usize,
    positions: usize,
    threads: usize,
) -> Result<TokenDriver<GgufDevice, PackFeeds>, EngError> {
    let model = Box::new(model);
    let ctx = projector.shared_context();
    drop(projector);
    let stream = Rc::new(ctx.stream()?);
    let weights = Rc::new(gguf_trunk(&model, &ctx, true, chunk_limit)?);
    let config = model.config();
    let layers = config.layers();
    let step = FlashStep::from_shape(
        Handle::Shared(Rc::clone(&ctx)),
        Handle::Shared(Rc::clone(&stream)),
        Handle::Shared(Rc::clone(&weights)),
        (config.model_shape()?, config.kinds().to_vec()),
        positions,
        1,
        0..layers,
    )?
    .with_gdn_form(GdnForm::FLASH_NEXT_GGUF);
    let moe = GgufMoe::new(&ctx, Handle::Shared(Rc::clone(&weights)), &step.moe_dims(1))?;
    let feeds = gguf_feeds(&model, threads)?;
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
    let run = GgufRun {
        stream_width: step.geometry.branches * step.geometry.hidden,
        logits_width: step.model.vocab,
        step,
        moe,
    };
    let owner = EngineOwner::new(run, model, weights, stream, ctx);
    TokenDriver::new(GgufDevice { owner }, feeds, shape)
}

pub(in crate::family::qwen::flash_next) struct GgufDevice {
    owner: Owner,
}

impl StartSink for GgufDevice {
    fn write_step_params(&mut self, token: i32, pos: i32) -> Result<(), EngError> {
        let params = StepParams::new(token, pos)?;
        let step = &self.owner.step().step;
        let range = step.params()?;
        range
            .buffer()
            .write_at(&step.stream, range.offset(), &params.to_bytes())
    }

    fn write_stream(&mut self, stream: &[f32]) -> Result<(), EngError> {
        let step = &self.owner.step().step;
        let range = step.slot(Slot::Stream)?;
        range
            .buffer()
            .write_at(&step.stream, range.offset(), &f32_le_bytes(stream))
    }

    /// The PLE projections run on the host; the start needs no copy of the embedding.
    fn write_ple_embedding(&mut self, _embedding: &[f32]) -> Result<(), EngError> {
        Ok(())
    }
}

impl TokenDevice for GgufDevice {
    fn begin_token(&mut self) -> Result<(), EngError> {
        self.owner.step().step.begin_token()
    }

    fn layers(&mut self, range: Range<usize>, pos: usize) -> Result<(), EngError> {
        if range.is_empty() {
            return Ok(());
        }
        let GgufRun { step, moe, .. } = self.owner.step_mut();
        step.run_layers(range, pos, 1, moe)
    }

    fn read_stream(&mut self) -> Result<Vec<f32>, EngError> {
        let run = self.owner.step();
        run.step
            .read_f32(run.step.slot(Slot::Stream)?, run.stream_width)
    }

    fn write_stream_back(&mut self, stream: &[f32]) -> Result<(), EngError> {
        StartSink::write_stream(self, stream)
    }

    fn head(&mut self) -> Result<(), EngError> {
        self.owner.step().step.head(1)
    }

    fn read_logits(&mut self) -> Result<Vec<f32>, EngError> {
        let run = self.owner.step();
        run.step
            .read_f32(run.step.slot(Slot::Logits)?, run.logits_width)
    }

    fn check_health(&mut self) -> Result<(), EngError> {
        self.owner.step().step.finish_token()
    }

    fn reset(&mut self) -> Result<(), EngError> {
        self.owner.step().step.reset()
    }
}

impl StateIo for GgufDevice {
    fn read_fixed_state(&self) -> Result<Vec<(StateSlot, Vec<u8>)>, EngError> {
        self.owner.step().step.read_fixed_state()
    }

    fn fixed_state_bytes(&self) -> u64 {
        self.owner.step().step.fixed_state_bytes()
    }

    fn write_fixed_state(&mut self, parts: &[(StateSlot, Vec<u8>)]) -> Result<(), EngError> {
        self.owner.step_mut().step.write_fixed_state(parts)
    }
}
