//! The card implementation of [`TokenDevice`] for a Flash-Next GGUF: the decode step with the
//! native MoE block over resident experts (`gguf_moe`). The driver calls are the same as
//! [`CardDevice`](super::device::CardDevice)'s; this device has no expert bank, so it reports no
//! bank counters.

use std::{ops::Range, rc::Rc};

use super::{
    driver::{DriverShape, StateIo, TokenDevice, TokenDriver},
    feeds::{StartShape, StartSink},
    gguf_moe::GgufMoe,
    gguf_stream::{ExpertCache, StreamTotals},
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
        expert_bank::layer_stream::Home,
        mem::fill::{self, FreeMarks, LayerTier, TierReceipt},
        project::cuda::{CudaProjector, CudaWeights},
        qwen::flash_next::load::gguf::{
            GgufModel,
            feeds::gguf_feeds,
            placement::{ExpertSetup, LayerStacks, layer_stacks},
            trunk::gguf_trunk,
        },
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
    setup: ExpertSetup,
) -> Result<TokenDriver<GgufDevice, PackFeeds>, EngError> {
    let model = Box::new(model);
    let ctx = projector.shared_context();
    drop(projector);
    let stream = Rc::new(ctx.stream()?);
    let stacks = layer_stacks(&model)?;
    let resident = |_layer: usize| matches!(setup, ExpertSetup::Resident);
    let weights = Rc::new(gguf_trunk(&model, &ctx, &resident, chunk_limit)?);
    let cache = match &setup {
        ExpertSetup::Resident => None,
        ExpertSetup::Cached(plan) => Some(ExpertCache::new(&model, &stacks, plan, &ctx)?),
    };
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
    let moe = GgufMoe::new(
        &ctx,
        Handle::Shared(Rc::clone(&weights)),
        &step.moe_dims(1),
        cache,
    )?;
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
    let ctx_for_receipt = Rc::clone(&ctx);
    let receipt = memory_receipt(&ctx_for_receipt, &model, &stacks, &setup);
    let owner = EngineOwner::new(run, model, weights, stream, ctx);
    eprintln!("{}", fill::report_line("flash-next gguf", receipt));
    let mut marks = FreeMarks::default();
    if let Ok(info) = ctx_for_receipt.memory_info()
        && let Some(line) = marks.record("load", info.free as u64)
    {
        eprintln!("flash-next gguf: {line}");
    }
    TokenDriver::new(
        GgufDevice {
            owner,
            ctx: ctx_for_receipt,
            marks,
            reported: StreamTotals::default(),
        },
        feeds,
        shape,
    )
}

/// The shared memory receipt of the start: the card's use, the dense core, the experts on the
/// card and the bytes pinned or streamed, from this model's own per-layer sizes.
fn memory_receipt(
    ctx: &CudaContext,
    model: &GgufModel,
    stacks: &[LayerStacks],
    setup: &ExpertSetup,
) -> Result<TierReceipt, EngError> {
    let experts = model.config().experts() as u64;
    let info = ctx.memory_info()?;
    let (slots, pinned) = match setup {
        ExpertSetup::Resident => (experts, 0),
        ExpertSetup::Cached(plan) => (
            plan.slots as u64,
            stacks
                .iter()
                .zip(&plan.homes)
                .filter(|&(_, &home)| home == Home::Pinned)
                .map(|(layer, _)| layer.stacks.iter().map(|stack| stack.bytes).sum::<u64>())
                .sum(),
        ),
    };
    let tiers: Vec<LayerTier> = stacks
        .iter()
        .map(|layer| LayerTier {
            slots,
            slot_bytes: layer.stacks.iter().map(|stack| stack.bytes).sum::<u64>() / experts,
        })
        .collect();
    Ok(TierReceipt::from_layers(
        (info.total as u64, info.free as u64),
        model.map().device_bytes(),
        experts,
        &tiers,
        pinned,
    ))
}

pub(in crate::family::qwen::flash_next) struct GgufDevice {
    owner: Owner,
    /// The card context, read for the free VRAM at the marks.
    ctx: Rc<CudaContext>,
    marks: FreeMarks,
    /// The staging totals at the last receipt line.
    reported: StreamTotals,
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
        let GgufRun {
            step,
            moe,
            stream_width,
            ..
        } = self.owner.step_mut();
        #[cfg(test)]
        if layer_trace::on() {
            // One layer a call, the residual stream read after each: the comparison with the
            // host's per-layer output.
            for layer in range {
                step.run_layers(layer..layer + 1, pos, 1, moe)?;
                let stream = step.read_f32(step.slot(Slot::Stream)?, *stream_width)?;
                layer_trace::push(stream);
            }
            return Ok(());
        }
        let _ = stream_width;
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
        let ctx = &self.ctx;
        if let Some(line) = self.marks.sample("decode", std::time::Instant::now(), || {
            ctx.memory_info().ok().map(|info| info.free as u64)
        }) {
            eprintln!("flash-next gguf: {line}");
        }
        self.owner.step().step.finish_token()
    }

    fn reset(&mut self) -> Result<(), EngError> {
        if let Some(now) = self.owner.step().moe.stream_totals()
            && now.visits != self.reported.visits
        {
            let (visits, hits, misses) = (
                now.visits - self.reported.visits,
                now.hits - self.reported.hits,
                now.misses - self.reported.misses,
            );
            let mib = |bytes: u64| bytes as f64 / (1u64 << 20) as f64;
            eprintln!(
                "flash-next gguf: expert cache since the last request: {visits} layer visits, {:.1}% of the picked experts held a slot ({hits} hits, {misses} misses); moved {:.1} MiB of the {:.1} MiB the picks read, host in staging {:.2} s",
                100.0 * hits as f64 / (hits + misses).max(1) as f64,
                mib(now.moved - self.reported.moved),
                mib(now.needed - self.reported.needed),
                now.host_seconds - self.reported.host_seconds,
            );
            self.reported = now;
        }
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

/// The residual stream after each layer of the tokens a test runs, for the device-against-host
/// comparison of one layer at a time.
#[cfg(test)]
pub(in crate::family::qwen::flash_next) mod layer_trace {
    use std::cell::RefCell;

    thread_local! {
        static TRACE: RefCell<Option<Vec<Vec<f32>>>> = const { RefCell::new(None) };
    }

    /// Starts recording; the next layers each push their stream.
    pub(in crate::family::qwen::flash_next) fn start() {
        TRACE.with(|trace| *trace.borrow_mut() = Some(Vec::new()));
    }

    pub(super) fn on() -> bool {
        TRACE.with(|trace| trace.borrow().is_some())
    }

    pub(super) fn push(stream: Vec<f32>) {
        TRACE.with(|trace| {
            if let Some(rows) = trace.borrow_mut().as_mut() {
                rows.push(stream);
            }
        });
    }

    /// Takes what was recorded since [`start`] and records again from empty.
    pub(in crate::family::qwen::flash_next) fn take() -> Vec<Vec<f32>> {
        TRACE.with(|trace| trace.borrow_mut().replace(Vec::new()).unwrap_or_default())
    }
}
