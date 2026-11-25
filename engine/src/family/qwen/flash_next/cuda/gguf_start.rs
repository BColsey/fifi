//! Builds the executor of the opt-in Flash-Next GGUF serving path (`INFERENCE_FLASH_NEXT_GGUF_SERVE`,
//! `load/gguf/start.rs`) on the engine thread: the GGUF counterpart of `start.rs::make`.
//!
//! Order, each stage announced to the engine state: the CUDA context and its stream; the GGUF
//! trunk, every layer and the final mixer and head, uploaded from the one open model
//! (`GgufTrunk::upload_with_final`); the single-token decode step over the planned context; the
//! routed-expert owner (`top_k + 1` slots: the ten picks and the filler slot); the host feeds
//! (the embedding row and the PLE block); then the token driver over [`GgufCardDevice`] and the
//! executor on top of it, built with prefix resume off.
//!
//! Lifetime: `spawn_engine` needs an executor without a lifetime and the step borrows its owners,
//! so the context, the stream, the model, the trunk and the n-gram rows are `Box::leak`ed once
//! per engine, the pack's convention (`start.rs`). From the first leak a failure leaves the card
//! loaded, so the one-engine flag is committed there.

use std::{sync::Arc, time::Instant};

use super::{
    driver::{DriverShape, TokenDriver},
    executor::{CudaRules, FlashCudaExecutor},
    feeds::StartShape,
    gguf_device::GgufCardDevice,
    gguf_moe::{GgufMoeOwner, GgufStep},
    start::{BuildGuard, ple_threads, register_served_kernels},
};
use crate::{
    EngError,
    family::{
        executor::ExecutorOps,
        project::cuda::CudaProjector,
        qwen::flash_next::{
            load::gguf::{GgufModel, token_feed::GgufTokenFeeds, trunk::GgufTrunk},
            step::ModelShape,
        },
        session::{DecodeSession, PrefixIdentity},
    },
    kernels::cuda::{CudaContext, DeviceStream},
    sched::worker::{Engine, ReadyNote, spawn_engine_noted},
    tokenizer::TokenizerBundle,
};

/// The executor of one Flash-Next GGUF on a card: the token driver over the GGUF device.
type GgufExecutor =
    FlashCudaExecutor<TokenDriver<GgufCardDevice<'static>, GgufTokenFeeds<'static>>>;

/// What the calling thread hands the engine thread: the model it planned from, the context the
/// caches hold, the tokenizer's limits, the store identity and the expert slots per layer the
/// memory plan sized. Each is read once, on that thread, so they travel as one value.
pub(in super::super) struct PlannedStart {
    pub(in super::super) model: GgufModel,
    pub(in super::super) positions: usize,
    pub(in super::super) rules: CudaRules,
    pub(in super::super) identity: PrefixIdentity,
    pub(in super::super) planned_slots: usize,
}

/// The sizes the driver checks and splits the layers by, for the GGUF step (the pack's rule:
/// the sampler vocabulary is the tokenizer's, the stream and the PLE embedding the model's).
fn driver_shape(
    shape: &ModelShape,
    layers: usize,
    positions: usize,
    sampler_vocab: usize,
) -> DriverShape {
    DriverShape {
        layers,
        ple_layer: shape.ple_layer,
        positions,
        start: StartShape {
            sampler_vocab,
            stream: shape.hyper.branches() * shape.hyper.hidden(),
            ple_embed: shape.ple.embed(),
        },
    }
}

/// The limits the executor reads from the tokenizer bundle, as the pack's start builds them,
/// with the sampler vocabulary checked against the width of the model head: the head is wider
/// (padded ids above the tokenizer's vocabulary) and those ids have no text.
pub(in super::super) fn cuda_rules(
    bundle: &TokenizerBundle,
    head_vocab: usize,
) -> Result<CudaRules, EngError> {
    let sampler_vocab = usize::try_from(bundle.vocab_size())
        .map_err(|_| EngError::Checkpoint("tokenizer vocabulary exceeds usize".to_string()))?;
    if sampler_vocab == 0 || sampler_vocab > head_vocab {
        return Err(EngError::Checkpoint(format!(
            "sampler vocabulary {sampler_vocab} is outside 1..={head_vocab}, the width of the model head"
        )));
    }
    Ok(CudaRules {
        eos: Arc::from(bundle.eos_token_ids()),
        sampler_vocab,
    })
}

/// Builds the executor and its first session over the leaked GGUF owners. No CUDA value crosses
/// a thread boundary: the model is the one the calling thread planned from, moved in.
fn make(
    stage: &dyn Fn(&'static str),
    planned: PlannedStart,
    mut guard: BuildGuard,
) -> Result<(GgufExecutor, DecodeSession), EngError> {
    let PlannedStart {
        model,
        positions,
        rules,
        identity,
        planned_slots,
    } = planned;
    let began = Instant::now();
    stage("the CUDA context");
    // The served admission and the Flash-Next kernel registry: how the pack's start opens its card.
    let projector: &'static CudaProjector = Box::leak(Box::new(CudaProjector::new()?));
    register_served_kernels(projector.context())?;
    let ctx: &'static CudaContext = projector.context();
    let stream: &'static DeviceStream = Box::leak(Box::new(ctx.stream()?));
    // LEAK: the owners are leaked for the process lifetime, giving the step `'static` borrows.
    // From the first leak a failure leaves the card loaded, so the one-engine flag is pinned here.
    guard.commit();
    let model: &'static GgufModel = Box::leak(Box::new(model));
    let shape = model.config().model_shape()?;
    let layers = model.config().layers();
    stage("the GGUF trunk");
    let trunk: &'static GgufTrunk<'static> = Box::leak(Box::new(GgufTrunk::upload_with_final(
        model,
        ctx,
        stream,
        0..layers,
    )?));
    stage("the decode step");
    let step = GgufStep::new(ctx, stream, trunk, positions)?;
    let slots = shape.moe.top_k() + 1;
    let owner = GgufMoeOwner::new(&step, slots)?;
    // The memory plan the calling thread printed sized `planned_slots` experts per layer; this
    // executor builds fewer: an LRU bank of `slots`, refilled from the files on a miss.
    eprintln!(
        "flash-next gguf cuda: {slots} expert slots per layer built (top_k + 1, an LRU bank; a \
         routed expert outside it is refilled from the files at the step); the memory plan sized \
         {planned_slots}"
    );
    stage("the host feeds");
    let ngram = Box::leak(Box::new(model.ngram()?));
    let feeds = GgufTokenFeeds::new(model, ngram, ple_threads())?;
    let driver = TokenDriver::new(
        GgufCardDevice::new(step, owner),
        feeds,
        driver_shape(&shape, layers, positions, rules.sampler_vocab),
    )?;
    let (executor, session) = FlashCudaExecutor::without_prefix_resume(driver, rules, identity)?;
    eprintln!(
        "flash-next gguf cuda: ready in {:.1} s, {positions} positions, prefix resume off \
         (store limits {:?})",
        began.elapsed().as_secs_f64(),
        executor.prefix_limits()
    );
    Ok((executor, session))
}

/// Starts the engine thread, which builds the executor ([`make`]) and its first session there,
/// so no executor type leaves this module: the pack's start spawns from its own module the same
/// way. The planned values move in, and no CUDA value crosses the boundary.
pub(in super::super) fn spawn(
    planned: PlannedStart,
    prefill_chunk: usize,
    ready_note: ReadyNote,
    guard: BuildGuard,
) -> Result<Engine, EngError> {
    spawn_engine_noted(prefill_chunk, ready_note, move |stages| {
        make(&|name| stages.begin(name), planned, guard)
    })
}
