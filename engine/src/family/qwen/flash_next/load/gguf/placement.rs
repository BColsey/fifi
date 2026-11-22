//! Where the routed experts of a Flash-Next GGUF sit in the file and where the memory plan puts
//! them: the three stacks of every layer, each with its own stored type and size, and the shared
//! layer placement (`expert_bank::layer_stream::StreamPlan`) over those sizes. A model's layers
//! differ in type per stack (the gate and up of a layer share a type, the down has its own), so
//! the sizes are per stack and per layer, never one uniform record.

use super::{
    GgufModel,
    tensors::{Mapped, Slot},
};
use crate::{
    EngError,
    family::{
        expert_bank::layer_stream::{Home, LayerExperts, StreamBudget, StreamPlan},
        recipe::Role as Wanted,
    },
    weights::GgufType,
};

fn refuse(detail: impl AsRef<str>) -> EngError {
    EngError::MemoryPlan(format!(
        "Flash-Next GGUF expert placement: {}",
        detail.as_ref()
    ))
}

/// The three stacks of one layer's routed experts, in the order the block reads them.
pub(in crate::family::qwen::flash_next) struct LayerStacks {
    /// Tensor name, stored type, `(k, rows of the whole stack)` and bytes of gate, up and down.
    pub(in crate::family::qwen::flash_next) stacks: [StackFacts; 3],
}

#[derive(Clone)]
pub(in crate::family::qwen::flash_next) struct StackFacts {
    pub(in crate::family::qwen::flash_next) name: String,
    pub(in crate::family::qwen::flash_next) ty: GgufType,
    pub(in crate::family::qwen::flash_next) k: usize,
    pub(in crate::family::qwen::flash_next) rows: usize,
    pub(in crate::family::qwen::flash_next) bytes: u64,
    pub(in crate::family::qwen::flash_next) shard: usize,
}

/// The stacks of every layer, read from the tensor map: per stack its own type and size.
pub(in crate::family::qwen::flash_next) fn layer_stacks(
    model: &GgufModel,
) -> Result<Vec<LayerStacks>, EngError> {
    let layers = model.config().layers();
    let mut found: Vec<[Option<StackFacts>; 3]> = vec![[None, None, None]; layers];
    for entry in model.map().entries() {
        let (Some(layer), Some(part)) = (entry.layer, expert_part(entry)) else {
            continue;
        };
        let (k, rows) = match entry.dims.as_slice() {
            [k, rows, stack] => (*k as usize, *rows as usize * *stack as usize),
            other => {
                return Err(refuse(format!("{} has dimensions {other:?}", entry.name)));
            }
        };
        found
            .get_mut(layer)
            .ok_or_else(|| refuse(format!("{} names layer {layer}", entry.name)))?[part] =
            Some(StackFacts {
                name: entry.name.clone(),
                ty: entry.ty,
                k,
                rows,
                bytes: entry.bytes,
                shard: entry.shard,
            });
    }
    found
        .into_iter()
        .enumerate()
        .map(|(layer, parts)| {
            let [Some(gate), Some(up), Some(down)] = parts else {
                return Err(refuse(format!("layer {layer} lacks an expert stack")));
            };
            Ok(LayerStacks {
                stacks: [gate, up, down],
            })
        })
        .collect()
}

/// 0, 1, 2 for the gate, up and down stack entries.
fn expert_part(entry: &Mapped) -> Option<usize> {
    match entry.role {
        Slot::Layer(Wanted::ExpertGate) => Some(0),
        Slot::Layer(Wanted::ExpertUp) => Some(1),
        Slot::Layer(Wanted::ExpertDown) => Some(2),
        _ => None,
    }
}

/// Card bytes the transport allocates per staging slot for the layers `plan` streams: one buffer
/// per distinct (stack, size, type, shape), because a staged stack is a native weight over a
/// buffer of exactly its size. A mixed-quant file has several, so this can exceed the plan's
/// charge of one largest layer per slot.
pub(in crate::family::qwen::flash_next) fn staging_bytes_per_slot(
    stacks: &[LayerStacks],
    plan: &StreamPlan,
) -> u64 {
    let mut keys: Vec<(usize, u64, GgufType, usize, usize)> = Vec::new();
    for (layer, facts) in stacks.iter().enumerate() {
        if plan.home(layer) == Home::Card {
            continue;
        }
        for (stack, fact) in facts.stacks.iter().enumerate() {
            let key = (stack, fact.bytes, fact.ty, fact.k, fact.rows);
            if !keys.contains(&key) {
                keys.push(key);
            }
        }
    }
    keys.iter().map(|key| key.1).sum()
}

/// The plan: where every layer's experts live, from the card and host bytes the memory plan
/// left for the routed experts. The staging the transport really allocates is charged: when the
/// layers that stream need more than the plan's one-largest-layer figure, the card budget loses
/// the difference and the placement is made again.
pub(in crate::family::qwen::flash_next) fn stream_plan(
    stacks: &[LayerStacks],
    experts: usize,
    mut budget: StreamBudget,
) -> Result<StreamPlan, EngError> {
    let layers = stacks
        .iter()
        .map(|layer| {
            LayerExperts::new(
                experts,
                layer.stacks.iter().map(|stack| stack.bytes).collect(),
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    for _ in 0..8 {
        let plan = StreamPlan::choose(&layers, budget)?;
        let real = staging_bytes_per_slot(stacks, &plan) * plan.staging_slots() as u64;
        if real <= plan.staging_bytes() {
            return Ok(plan);
        }
        budget.card_bytes = budget.card_bytes.saturating_sub(real - plan.staging_bytes());
    }
    Err(refuse("the staging of the streamed layers does not settle against the card budget"))
}

#[cfg(test)]
mod tests {
    use super::{layer_stacks, stream_plan};
    use crate::{
        family::{
            expert_bank::layer_stream::{Home, StreamBudget},
            qwen::flash_next::load::gguf::testkit::{Spec, write_flash},
        },
        weights::GgufType,
    };

    /// The mixed-quant layout of the real file: the gate and up stacks of a layer share a type,
    /// the down stack has its own, and the three differ in size, so every stack is sized by its
    /// own type and no layer's record is assumed to equal another's.
    #[test]
    fn the_stacks_of_the_mixed_quant_file_are_sized_per_stack_and_per_layer() {
        let written = write_flash(&Spec::default());
        let model = super::super::GgufModel::open(&written.first)
            .expect("open")
            .expect("a qwen4exp model");
        let stacks = layer_stacks(&model).expect("stacks");
        assert_eq!(stacks.len(), 48);
        let types = |layer: usize| {
            let facts = &stacks[layer].stacks;
            [facts[0].ty, facts[1].ty, facts[2].ty]
        };
        assert_eq!(
            types(0),
            [GgufType::Iq2S, GgufType::Iq2S, GgufType::Iq4Nl]
        );
        assert_eq!(types(8), [GgufType::Iq2S, GgufType::Iq2S, GgufType::Q20]);
        assert_eq!(
            types(47),
            [GgufType::Iq4Xs, GgufType::Iq4Xs, GgufType::Iq4Nl]
        );
        // The stacks are `[k, rows, experts]` in the file's order: the gate and up read the
        // hidden width, the down the intermediate width.
        let gate = &stacks[0].stacks[0];
        let down = &stacks[0].stacks[2];
        assert_eq!((gate.k, gate.rows), (2560, 640 * 512));
        assert_eq!((down.k, down.rows), (640, 2560 * 512));
        // Bytes follow each stack's own block size: IQ2_S, IQ4_NL and Q2_0 all differ.
        let record = |layer: usize| {
            stacks[layer]
                .stacks
                .iter()
                .map(|stack| stack.bytes)
                .sum::<u64>()
        };
        assert_ne!(record(0), record(8));
        assert_ne!(record(0), record(47));
        let total: u64 = (0..48).map(record).sum();
        // With a card that holds a third of the experts the plan keeps whole layers on the card,
        // the next ones in pinned memory, and the rest on disk, and charges the staging.
        let plan = stream_plan(
            &stacks,
            512,
            StreamBudget {
                card_bytes: total / 3,
                host_bytes: total / 3,
                staging_slots: 2,
            },
        )
        .expect("plan");
        let homes = |home: Home| (0..48).filter(|&layer| plan.home(layer) == home).count();
        assert!(homes(Home::Card) > 0 && homes(Home::Pinned) > 0 && homes(Home::Disk) > 0);
        assert!(plan.resident_bytes() + plan.staging_bytes() <= total / 3);
        // The staging the transport allocates is within what the plan charged.
        assert!(
            super::staging_bytes_per_slot(&stacks, &plan) * plan.staging_slots() as u64
                <= plan.staging_bytes()
        );
        // A card that holds everything streams nothing.
        let all = stream_plan(
            &stacks,
            512,
            StreamBudget {
                card_bytes: total * 2,
                host_bytes: 0,
                staging_slots: 2,
            },
        )
        .expect("plan");
        assert!(all.streamed().is_empty());
    }
}
