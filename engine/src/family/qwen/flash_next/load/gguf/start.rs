//! The CUDA start of a Flash-Next GGUF: opens the card, plans the memory, prints the plan,
//! refuses to serve while a tensor type of the file has no decode kernel, and serves a model
//! whose routed experts all fit on the card.
//!
//! Order: the CUDA context (for the card's free memory), the host admission snapshot, the
//! context the plan chooses when the entry sets none, the plan line, the kernel check, the
//! tokenizer, then the engine thread, which uploads the weights and builds the decode step.
//! The plan runs first so that an operator whose model is refused still sees the tiers the
//! start would use once the kernels exist.

use std::{path::Path, sync::Arc};

use sha2::{Digest, Sha256};

use super::{
    GgufModel,
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
            cuda::{
                executor::{CudaRules, FlashCudaExecutor},
                gguf_device::gguf_driver,
                start::ple_threads,
            },
            host_plan::{HOST_LABELS, HostMemory},
            settings::{
                DEFAULT_CUDA_CONTEXT, DEFAULT_PREFILL_CHUNK, FlashOptions, MAX_CUDA_CONTEXT,
                cuda_settings, default_context, ring_budget_bytes,
            },
        },
        session::PrefixIdentity,
    },
    sched::worker::{Engine, ReadyNote, spawn_engine_noted},
    tokenizer::{TokenizerBundle, TokenizerPin},
};

/// Host bytes one native tensor read holds while a tensor is uploaded.
const UPLOAD_CHUNK_BYTES: usize = 64 << 20;

/// The pin of the tokenizer files found in `dir`: their own digests, so the load matches and
/// logs nothing, and the vocabulary the model's head holds. A file that is absent pins zeros
/// and is reported by the load.
fn observed_pin(dir: &Path, vocab: u32) -> TokenizerPin {
    let digest = |name: &str| -> [u8; 32] {
        std::fs::read(dir.join(name)).map_or([0u8; 32], |bytes| Sha256::digest(bytes).into())
    };
    TokenizerPin {
        tokenizer_json: digest("tokenizer.json"),
        tokenizer_config_json: digest("tokenizer_config.json"),
        chat_template_jinja: digest("chat_template.jinja"),
        generation_config_json: digest("generation_config.json"),
        vocab_size: vocab,
    }
}

/// Plans the model on this machine, prints the plan, and starts the engine when every
/// matrix type has a decode kernel and the routed experts all fit on the card. The tokenizer
/// files are read from `tokenizer_dir`.
pub(in crate::family::qwen::flash_next) fn start(
    owned: GgufModel,
    options: Option<&FlashOptions>,
    tokenizer_dir: &Path,
    ready_note: ReadyNote,
) -> Result<(Engine, TokenizerBundle, u32), EngError> {
    let model = &owned;
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
    // The routed experts that do not all fit beside the trunk and the K/V are cached: the memory
    // plan's slot count per layer sits on the card, and each layer's experts are read from pinned
    // host memory or from the file on a miss.
    let experts = model.config().experts();
    let tier = load.context.plan;
    let setup = if tier.slots_per_layer() >= experts {
        ExpertSetup::Resident
    } else {
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
        ),
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
    let header = model.weights.header();
    let weights_digest: [u8; 32] = Sha256::new()
        .chain_update(header.header_sha256())
        .chain_update(header.tensors().len().to_le_bytes())
        .finalize()
        .into();
    let tokenizer_digest: [u8; 32] =
        Sha256::digest(std::fs::read(tokenizer_dir.join("tokenizer.json")).unwrap_or_default())
            .into();
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
