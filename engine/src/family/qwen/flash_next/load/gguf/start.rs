//! The CUDA start of a Flash-Next GGUF: opens the card, plans memory, prints the plan, and
//! refuses to serve because the resident executor cannot yet load GGUF weights.
//!
//! Order: the CUDA context (for the card's free memory), the host admission snapshot, the
//! context the plan chooses when the entry sets none, the plan line, then the refusal. Missing
//! decode kernels are reported first; their availability alone does not enable execution.
//! The plan is preliminary and does not account for a future GGUF executor's scratch buffers.

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
pub(in super::super::super) fn start(
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
    verify_expert_stack_budget(model, load.context.plan.slots_per_layer())?;
    eprintln!(
        "{load}; serving {} slots per layer; expert tier: {}; {}",
        load.context.plan.slots_per_layer(),
        tier_line(&load),
        facts.host.provenance_note()
    );

    Err(execution_refusal(model))
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

/// Kernel availability does not imply that the CUDA executor can load GGUF weights.
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
}

fn tier_line(load: &crate::family::qwen::flash_next::moe::load_plan::LoadPlan) -> String {
    let (name, speed) = load.context.plan.store_tier().describe();
    format!("{name} ({speed}); the experts are read from the GGUF files in place")
}
