//! The CUDA start of a Flash-Next GGUF: opens the card, plans the memory, prints the plan,
//! refuses to serve while a tensor type of the file has no decode kernel, and serves a model
//! whose routed experts all fit on the card.
//!
//! Order: the CUDA context (for the card's free memory), the host admission snapshot, the
//! context the plan chooses when the entry sets none, the plan line, the kernel check, the
//! tokenizer, then the engine thread, which uploads the weights and builds the decode step.
//! The plan runs first so that an operator whose model is refused still sees the tiers the
//! start would use once the kernels exist.
//!
//! A second start is opt-in (`INFERENCE_FLASH_NEXT_GGUF_SERVE`, `settings.rs`): the factory
//! calls [`start_staged`] with the switch on, which serves the file in place through the GGUF
//! decode modules (`cuda/gguf_start.rs`). That path is not qualified. Its plan is preliminary
//! and does not account for the executor's scratch buffers. It keeps the pack's tokenizer rule
//! (`cuda/start.rs`): the tokenizer files are pinned through an artifact manifest, found by the
//! pack's own lookup, and a GGUF directory that has none is refused by that lookup's own message.

use std::{path::Path, sync::Arc, time::Instant};

use sha2::{Digest, Sha256};

use super::{
    GgufModel, LIMITS,
    admission::{
        check_cuda_decode, check_opt_in, check_slots, observed_pin, tokenizer_digest,
        weights_digest,
    },
    placement::{ExpertSetup, cache_plan, layer_stacks},
    plan::Card,
};
use crate::{
    EngError,
    family::{
        expert_bank::bank_tier::TierChoice,
        mem::Facts,
        project::cuda::CudaProjector,
        qwen::flash_next::{
            artifact::tokenizer_pin,
            cuda::{
                BuildGuard,
                executor::{CudaRules, FlashCudaExecutor},
                gguf_device::gguf_driver,
                gguf_start,
                start::ple_threads,
            },
            host_plan::{HOST_LABELS, HostMemory},
            settings::{
                DEFAULT_CUDA_CONTEXT, DEFAULT_PREFILL_CHUNK, FlashEnv, FlashOptions,
                GgufStartSettings, MAX_CUDA_CONTEXT, cuda_settings, default_context,
                ring_budget_bytes,
            },
        },
        session::PrefixIdentity,
    },
    sched::worker::{Engine, ReadyNote, spawn_engine_noted},
    tokenizer::{TokenizerBundle, TokenizerPin},
    weights::{GgufHeader, gguf_identity_digest, sha256_hex},
};

/// Host bytes one native tensor read holds while a tensor is uploaded.
const UPLOAD_CHUNK_BYTES: usize = 64 << 20;

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

/// Plans the model on this machine, prints the plan, then serves it and returns the engine,
/// its tokenizer bundle and the context the caches hold. The factory calls it as `start_staged`
/// only with the switch on (a mis-set switch is refused there before the card is opened);
/// [`start`] is the path without it. A type without a decode kernel refuses here as it does there.
pub(in crate::family::qwen::flash_next) fn start_staged(
    model: GgufModel,
    input: StartInput<'_>,
) -> Result<(Engine, TokenizerBundle, u32), EngError> {
    let (context, planned_slots) = plan(&model, input.options)?;
    let gaps = model.kernel_gaps();
    if !gaps.is_empty() {
        return Err(model.kernel_refusal(&gaps));
    }
    serve_gguf(model, context, planned_slots, input)
}

/// Opens the card for its free memory and chooses the context; returns it with the expert slots
/// per layer the plan sized there (the executor prints what it builds beside it).
fn plan(model: &GgufModel, options: Option<&FlashOptions>) -> Result<(u32, usize), EngError> {
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
    let plan = model.load_plan_staged(card)?;
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
    Ok((context, load.context.plan.slots_per_layer()))
}

/// The serving half, the pack's `start` (`cuda/start.rs`) over the GGUF modules: the tokenizer
/// pin, the bundle, the rules and the identity on the calling thread, then the engine thread
/// builds everything CUDA inside `gguf_start::make`. The plan chose the context before the
/// spawn, so it is returned directly: there is nothing to announce.
fn serve_gguf(
    model: GgufModel,
    context: u32,
    planned_slots: usize,
    input: StartInput<'_>,
) -> Result<(Engine, TokenizerBundle, u32), EngError> {
    let settings = GgufStartSettings::new(
        input.model_dir,
        input.tokenizer_dir,
        input.options,
        input.env,
    )?;
    let pin = load_pin(&settings.manifest, settings.tokenizer_vocab_size)?;
    let bundle = TokenizerBundle::load(input.tokenizer_dir, &pin)?;
    let head_vocab = model.config().model_shape()?.vocab;
    let rules = gguf_start::cuda_rules(&bundle, head_vocab)?;
    let identity = gguf_identity(&model, &pin)?;
    let prefill_chunk = usize::try_from(settings.prefill_chunk)
        .map_err(|_| EngError::BadRequest("engine config: prefill_chunk exceeds usize".into()))?;
    let positions = usize::try_from(context)
        .map_err(|_| EngError::BadRequest("engine config: context exceeds usize".into()))?;
    let guard = BuildGuard::acquire()?;
    let began = Instant::now();
    // The engine thread builds the executor inside `cuda`, so no executor type leaves that module.
    let engine = gguf_start::spawn(
        gguf_start::PlannedStart {
            model,
            positions,
            rules,
            identity,
            planned_slots,
        },
        prefill_chunk,
        input.ready_note,
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
    let sizes = model.staged_sizes()?;
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

/// Plans the model on this machine, prints the plan, and starts the engine when every
/// matrix type has a decode kernel and the routed experts all fit on the card. The tokenizer
/// files are read from `tokenizer_dir`. `opt_in` is the `FIFI_FLASH_NEXT_GGUF` switch, read by
/// the caller: without it the start refuses where it refused before it read GGUF weights.
pub(in crate::family::qwen::flash_next) fn start(
    owned: GgufModel,
    options: Option<&FlashOptions>,
    opt_in: bool,
    tokenizer_dir: &Path,
    ready_note: ReadyNote,
) -> Result<(Engine, TokenizerBundle, u32), EngError> {
    let model = &owned;
    // The start exists only on a build with the CUDA decoder, where an unset `cuda_decode` is the
    // card, as it is for a pack start.
    let options = options
        .cloned()
        .unwrap_or_default()
        .card_default(cfg!(all(feature = "native-cuda", target_os = "linux")));
    check_cuda_decode(options.cuda_decode)?;
    let cuda = cuda_settings(&options)?.ok_or_else(|| {
        EngError::BadRequest("engine config: the CUDA decoder is not enabled".to_string())
    })?;
    eprintln!("{}", model.summary());

    // The measuring projector is one of the two card leases a process holds: it is dropped here,
    // before the engine thread takes its own, so a third holder in the process cannot fail it.
    let facts = {
        let projector = CudaProjector::new()?;
        Facts::read_card(HOST_LABELS, projector.card_bytes())?
    };
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
    eprintln!(
        "{load}; serving {} slots per layer; expert tier: {}; {}",
        load.context.plan.slots_per_layer(),
        tier_line(&load),
        facts.host.provenance_note()
    );

    let gaps = model.kernel_gaps();
    if !gaps.is_empty() {
        return Err(model.kernel_refusal(&gaps));
    }
    check_opt_in(opt_in, model.first_path())?;
    // The routed experts that do not all fit beside the trunk and the K/V are cached: the memory
    // plan's slot count per layer sits on the card, and each layer's experts are read from pinned
    // host memory or from the file on a miss.
    let experts = model.config().experts();
    let tier = load.context.plan;
    let setup = if tier.slots_per_layer() >= experts {
        ExpertSetup::Resident
    } else {
        check_slots(tier.slots_per_layer(), model.config().top_k())?;
        let host_bytes = match tier.store_tier() {
            TierChoice::Disk | TierChoice::CardDisk { .. } => tier.host_spare_bytes(),
            _ => tier.cold_bytes() + tier.host_spare_bytes(),
        };
        ExpertSetup::Cached(cache_plan(
            &layer_stacks(model)?,
            model.config().top_k(),
            tier.slots_per_layer(),
            host_bytes,
        ))
    };
    match &setup {
        ExpertSetup::Resident => eprintln!("flash-next gguf: routed experts: all on the card"),
        ExpertSetup::Cached(plan) => eprintln!("flash-next gguf: routed experts: {plan}"),
    }

    let vocab = model.config().model_shape()?.vocab;
    let bundle = TokenizerBundle::load(
        tokenizer_dir,
        &observed_pin(
            tokenizer_dir,
            u32::try_from(vocab).map_err(|_| refuse("the vocabulary overflows u32"))?,
        )?,
    )?;
    let sampler_vocab = usize::try_from(bundle.vocab_size())
        .map_err(|_| refuse("the tokenizer vocabulary overflows usize"))?;
    if sampler_vocab == 0 || sampler_vocab > vocab {
        return Err(refuse(format!(
            "the tokenizer has {sampler_vocab} ids and the model head {vocab}"
        )));
    }
    let rules = CudaRules {
        eos: Arc::from(bundle.eos_token_ids()),
        sampler_vocab,
    };
    let weights_digest = weights_digest(model)?;
    let tokenizer_digest = tokenizer_digest(tokenizer_dir)?;
    let identity = PrefixIdentity {
        weights: crate::weights::sha256_hex(&weights_digest).into(),
        tokenizer: crate::weights::sha256_hex(&tokenizer_digest).into(),
        ..PrefixIdentity::default()
    };
    let prefill_chunk = options.prefill_chunk.unwrap_or(DEFAULT_PREFILL_CHUNK) as usize;
    let engine = spawn_engine_noted(prefill_chunk, ready_note, move |stages| {
        stages.begin("initializing CUDA");
        let projector = CudaProjector::new()?;
        stages.begin("uploading the weights and building the decode step");
        let driver = gguf_driver(
            owned,
            projector,
            UPLOAD_CHUNK_BYTES,
            context as usize,
            ple_threads(),
            setup,
        )?;
        FlashCudaExecutor::new(driver, rules, identity)
    })?;
    Ok((engine, bundle, context))
}

fn refuse(detail: impl AsRef<str>) -> EngError {
    EngError::Checkpoint(format!("Flash-Next GGUF start: {}", detail.as_ref()))
}

fn tier_line(load: &crate::family::qwen::flash_next::moe::load_plan::LoadPlan) -> String {
    let (name, speed) = load.context.plan.store_tier().describe();
    format!("{name} ({speed}); the experts are read from the GGUF files in place")
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
