//! The CUDA start of a Flash-Next GGUF: opens the card, plans the memory, prints the plan, and
//! refuses to serve while a tensor type of the file has no decode kernel.
//!
//! Order: the CUDA context (for the card's free memory), the host admission snapshot, the
//! context the plan chooses when the entry sets none, the plan line, then the kernel check.
//! The plan runs first so that an operator whose model is refused still sees the tiers the
//! start would use once the kernels exist.

use std::convert::Infallible;

use super::{GgufModel, plan::Card};
use crate::{
    EngError,
    family::{
        mem::Facts,
        project::cuda::CudaProjector,
        qwen::flash_next::host_plan::{HOST_LABELS, HostMemory},
        qwen::flash_next::settings::{
            DEFAULT_CUDA_CONTEXT, FlashOptions, MAX_CUDA_CONTEXT, cuda_settings, default_context,
            ring_budget_bytes,
        },
    },
};

/// Plans the model on this machine and prints the plan. Returns the refusal that stands
/// between the file and a served start; there is no serving path yet, so `Ok` never occurs.
pub(in crate::family::qwen::flash_next) fn start(
    model: &GgufModel,
    options: Option<&FlashOptions>,
) -> Result<Infallible, EngError> {
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
    Err(EngError::Checkpoint(format!(
        "Flash-Next GGUF {} is planned and every matrix type has a decode kernel, but the \
         executor does not read GGUF weights yet",
        model.first_path().display()
    )))
}

fn tier_line(load: &crate::family::qwen::flash_next::moe::load_plan::LoadPlan) -> String {
    let (name, speed) = load.context.plan.store_tier().describe();
    format!("{name} ({speed}); the experts are read from the GGUF files in place")
}
