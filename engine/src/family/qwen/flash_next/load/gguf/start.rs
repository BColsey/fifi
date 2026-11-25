//! The CUDA start of a Flash-Next GGUF: opens the card, plans memory and prints the plan, then
//! refuses or serves. Serving is opt-in (`INFERENCE_FLASH_NEXT_GGUF_SERVE`, `settings.rs`): with
//! the switch off the start refuses after the plan, as before; with it on it serves the file in
//! place through the GGUF decode modules (`cuda/gguf_start.rs`). The path is not qualified.
//!
//! Order: the CUDA context (for the card's free memory), the host admission snapshot, the
//! context the plan chooses when the entry sets none, the plan line, then the refusal or the
//! start. Missing decode kernels are reported first, with or without the switch; their
//! availability alone does not enable execution. The plan is preliminary and does not account
//! for the executor's scratch buffers.
//!
//! A served start keeps the pack's tokenizer rule (`cuda/start.rs`): the tokenizer files are
//! pinned through an artifact manifest, found by the pack's own lookup, and a GGUF directory
//! that has none is refused by that lookup's own message.

use std::path::Path;
use std::time::Instant;

use sha2::{Digest, Sha256};

use super::{GgufModel, LIMITS, plan::Card};
use crate::{
    EngError,
    family::{
        mem::Facts,
        project::cuda::CudaProjector,
        qwen::flash_next::{
            artifact::tokenizer_pin,
            cuda::{BuildGuard, gguf_start},
            host_plan::{HOST_LABELS, HostMemory},
            settings::{
                DEFAULT_CUDA_CONTEXT, FlashEnv, FlashOptions, GgufStartSettings, MAX_CUDA_CONTEXT,
                cuda_settings, default_context, ring_budget_bytes,
            },
        },
        session::PrefixIdentity,
    },
    sched::worker::{Engine, ReadyNote},
    tokenizer::{TokenizerBundle, TokenizerPin},
    weights::{GgufHeader, gguf_identity_digest, sha256_hex},
};

/// What the factory hands a GGUF start besides the opened model.
pub(in crate::family::qwen::flash_next) struct StartInput<'a> {
    /// The entry's `model_dir`: where the manifest lookup looks before the tokenizer directory.
    pub(in crate::family::qwen::flash_next) model_dir: &'a Path,
    pub(in crate::family::qwen::flash_next) tokenizer_dir: &'a Path,
    pub(in crate::family::qwen::flash_next) options: Option<&'a FlashOptions>,
    /// The captured environment: the switch and the manifest fallback are read here.
    pub(in crate::family::qwen::flash_next) env: &'a FlashEnv,
    pub(in crate::family::qwen::flash_next) ready_note: ReadyNote,
}

/// Plans the model on this machine, prints the plan, then refuses (the switch off) or serves
/// it (the switch on) and returns the engine, its tokenizer bundle and the context the caches
/// hold. A mis-set switch is refused before the card is opened.
pub(in crate::family::qwen::flash_next) fn start(
    model: GgufModel,
    input: StartInput<'_>,
) -> Result<(Engine, TokenizerBundle, u32), EngError> {
    let serve = input.env.gguf_serve()?;
    let context = plan(&model, input.options)?;
    // A type without a decode kernel refuses with or without the switch; the switch lifts only
    // the executor refusal.
    if !serve || !model.kernel_gaps().is_empty() {
        return Err(execution_refusal(&model));
    }
    let settings = GgufStartSettings::new(
        input.model_dir,
        input.tokenizer_dir,
        input.options,
        input.env,
    )?;
    serve_gguf(
        model,
        context,
        &settings,
        input.tokenizer_dir,
        input.ready_note,
    )
}

/// Opens the card for its free memory and chooses the context and the expert slots per layer.
fn plan(model: &GgufModel, options: Option<&FlashOptions>) -> Result<u32, EngError> {
    let mut options = options.cloned().unwrap_or_default();
    if options.cuda_decode == Some(false) {
        return Err(EngError::BadRequest(
            "engine config: the Flash-Next CUDA decoder is off; set flash_next.cuda_decode to true to serve on a card"
                .to_string(),
        ));
    }
    options.cuda_decode = Some(true);
    let cuda = cuda_settings(&options)?.ok_or_else(|| {
        EngError::BadRequest("engine config: the CUDA decoder is not enabled".to_string())
    })?;
    eprintln!("{}", model.summary());

    let projector = CudaProjector::new()?;
    let facts = Facts::read_card(HOST_LABELS, projector.card_bytes())?;
    let ring_budget = ring_budget_bytes(
        std::env::var("INFERENCE_FLASH_NEXT_RING_BUDGET")
            .ok()
            .as_deref(),
    )?;
    let card = Card {
        free_vram: facts.card_free(),
        host: HostMemory::of(&facts.host),
        ring_budget,
    };
    let kv = cuda.kv_format.format();
    let min_slots = model.config().top_k() + 1;
    let plan = model.load_plan(card)?;
    let floor = min_slots.max(cuda.slots.map_or(0, |asked| asked as usize));
    let context = if options.context.is_some() {
        cuda.context
    } else {
        let cap = model.config().max_positions.min(MAX_CUDA_CONTEXT);
        let slots_at = |context: u32| {
            plan.at_context(u64::from(context), kv, floor)
                .ok()
                .map(|load| load.context.plan.slots_per_layer())
        };
        match default_context(&slots_at, floor, cap) {
            Some(chosen) => {
                eprintln!(
                    "flash-next gguf: context {} chosen from the memory plan: {} expert slots \
                     per layer at {}, {} at {DEFAULT_CUDA_CONTEXT}; set flash_next.context to \
                     override",
                    chosen.context, chosen.slots, chosen.context, chosen.baseline_slots
                );
                chosen.context
            }
            None => cuda.context,
        }
    };
    let load = plan.at_context(u64::from(context), kv, floor)?;
    verify_expert_stack_budget(model, load.context.plan.slots_per_layer())?;
    eprintln!(
        "{load}; serving {} slots per layer; expert tier: {}; {}",
        load.context.plan.slots_per_layer(),
        tier_line(&load),
        facts.host.provenance_note()
    );
    Ok(context)
}

/// The serving half, the pack's `start` (`cuda/start.rs`) over the GGUF modules: the tokenizer
/// pin, the bundle, the rules and the identity on the calling thread, then the engine thread
/// builds everything CUDA inside `gguf_start::make`. The plan chose the context before the
/// spawn, so it is returned directly: there is nothing to announce.
fn serve_gguf(
    model: GgufModel,
    context: u32,
    settings: &GgufStartSettings,
    tokenizer_dir: &Path,
    ready_note: ReadyNote,
) -> Result<(Engine, TokenizerBundle, u32), EngError> {
    let pin = load_pin(&settings.manifest, settings.tokenizer_vocab_size)?;
    let bundle = TokenizerBundle::load(tokenizer_dir, &pin)?;
    let head_vocab = model.config().model_shape()?.vocab;
    let rules = gguf_start::cuda_rules(&bundle, head_vocab)?;
    let identity = gguf_identity(&model, &pin)?;
    let prefill_chunk = usize::try_from(settings.prefill_chunk)
        .map_err(|_| EngError::BadRequest("engine config: prefill_chunk exceeds usize".into()))?;
    let guard = BuildGuard::acquire()?;
    let began = Instant::now();
    // The engine thread builds the executor inside `cuda`, so no executor type leaves that module.
    let engine = gguf_start::spawn(
        model,
        usize::try_from(context)
            .map_err(|_| EngError::BadRequest("engine config: context exceeds usize".into()))?,
        prefill_chunk,
        rules,
        identity,
        ready_note,
        guard,
    )?;
    eprintln!(
        "flash-next gguf cuda: engine thread started at {:.1} s",
        began.elapsed().as_secs_f64()
    );
    Ok((engine, bundle, context))
}

/// The tokenizer pin of the artifact manifest, as the pack's start reads it.
fn load_pin(manifest: &Path, vocab_size: u32) -> Result<TokenizerPin, EngError> {
    let manifest = std::fs::read(manifest).map_err(|error| {
        EngError::Checkpoint(format!(
            "the manifest {} cannot be read: {error}",
            manifest.display()
        ))
    })?;
    tokenizer_pin(&manifest, vocab_size)
}

/// The load identity of a GGUF checkpoint for the prefix paths: the identity digest of every
/// shard (`weights::gguf_identity_digest`: the header, a sample of every tensor's bytes, the
/// file length and its modification time; this family pins no full-file digest), folded into
/// one, plus the pinned tokenizer and chat-template digests of the artifact manifest. Constant
/// for one loaded checkpoint, as the pack's identity is for one pack.
fn gguf_identity(model: &GgufModel, pin: &TokenizerPin) -> Result<PrefixIdentity, EngError> {
    let mut hasher = Sha256::new();
    for path in model.weights.paths() {
        hasher.update(shard_identity(path)?);
    }
    let weights: [u8; 32] = hasher.finalize().into();
    Ok(PrefixIdentity {
        weights: sha256_hex(&weights).into(),
        tokenizer: sha256_hex(&pin.tokenizer_json).into(),
        template: sha256_hex(&pin.chat_template_jinja).into(),
    })
}

/// The identity digest of one shard: its own header, read again with the loader's bounds.
fn shard_identity(path: &Path) -> Result<[u8; 32], EngError> {
    let bad = |error: std::io::Error| {
        EngError::Checkpoint(format!("prefix identity: {}: {error}", path.display()))
    };
    let mut file = std::fs::File::open(path).map_err(bad)?;
    let len = file.metadata().map_err(bad)?.len();
    let max_header = LIMITS.max_header_bytes;
    let header = GgufHeader::parse(&mut file, len, max_header)?;
    gguf_identity_digest(&header, path, None).map_err(bad)
}

/// Check the actual part-major stack bytes against the planner's conservative charge.
/// This covers expert buffers only, not CUDA allocation overhead or other buffers.
fn verify_expert_stack_budget(model: &GgufModel, slots: usize) -> Result<(), EngError> {
    use super::{expert_layout::ExpertAllocation, plan::stack_alignment_reserve};
    let sizes = model.sizes()?;
    let actual = ExpertAllocation::for_model(model, slots)?.device_bytes;
    let charge = sizes
        .shape
        .expert_bytes
        .checked_mul(sizes.shape.layers as u64)
        .and_then(|bytes| bytes.checked_mul(slots as u64))
        .and_then(|bytes| bytes.checked_add(stack_alignment_reserve(sizes.shape.layers).ok()?))
        .ok_or_else(|| EngError::MemoryPlan("GGUF expert stack budget overflows".into()))?;
    if u64::try_from(actual)
        .map_err(|_| EngError::MemoryPlan("GGUF expert allocation overflows".into()))?
        > charge
    {
        return Err(EngError::MemoryPlan(
            "GGUF expert stacks exceed planned budget".into(),
        ));
    }
    Ok(())
}

/// Kernel availability does not imply that the path may serve: without the switch, or with a
/// matrix type no decode kernel reads, this refusal stands between the plan and a start.
fn execution_refusal(model: &GgufModel) -> EngError {
    let gaps = model.kernel_gaps();
    if !gaps.is_empty() {
        return model.kernel_refusal(&gaps);
    }
    EngError::Checkpoint(format!(
        "Flash-Next GGUF {} is planned and every matrix type has a decode kernel, but the \
         executor does not read GGUF weights yet",
        model.first_path().display()
    ))
}

#[cfg(test)]
mod tests {
    use super::super::testkit::{Spec, write_flash};
    use super::*;

    #[test]
    fn expert_stack_budget_checks_slot_limits_and_valid_plans() {
        let fixture = write_flash(&Spec::default());
        let model = GgufModel::open(&fixture.first).unwrap().unwrap();
        assert!(verify_expert_stack_budget(&model, 0).is_err());
        assert!(
            verify_expert_stack_budget(&model, model.config().dims.experts as usize + 1).is_err()
        );
        for slots in [1, 11, model.config().dims.experts as usize] {
            verify_expert_stack_budget(&model, slots).unwrap();
        }
    }

    #[test]
    fn supported_matrix_types_still_refuse_until_executor_reads_gguf() {
        let fixture = write_flash(&Spec::default());
        let model = GgufModel::open(&fixture.first).unwrap().unwrap();
        assert!(model.kernel_gaps().is_empty());
        let error = execution_refusal(&model).describe();
        assert!(
            error.contains("executor does not read GGUF weights yet"),
            "{error}"
        );
    }

    /// The factory routes a GGUF checkpoint to this start, and a mis-set switch is refused there
    /// before the card is opened (`plan` is the first call that needs one). Host-only: the
    /// fixture is a synthetic file and no CUDA context is made.
    #[test]
    fn the_factory_refuses_a_mis_set_switch_before_any_card_is_opened() {
        use std::path::Path;
        let fixture = write_flash(&Spec::default());
        let env = FlashEnv {
            gguf_serve: Some("yes".into()),
            ..FlashEnv::default()
        };
        let error = match crate::family::qwen::flash_next::factory::start_engine_resolved(
            &fixture.first,
            Path::new("/unused-tokenizer-dir"),
            None,
            &env,
            false,
            None,
        ) {
            Err(error) => error,
            Ok(_) => panic!("a mis-set switch must refuse the start"),
        };
        let text = error.describe();
        assert!(
            text.contains("INFERENCE_FLASH_NEXT_GGUF_SERVE") && text.contains("\"yes\""),
            "{text}"
        );
    }
}

fn tier_line(load: &crate::family::qwen::flash_next::moe::load_plan::LoadPlan) -> String {
    let (name, speed) = load.context.plan.store_tier().describe();
    format!("{name} ({speed}); the experts are read from the GGUF files in place")
}
