//! Where the routed experts of a Flash-Next GGUF sit in the file and how the memory plan holds
//! them: the three stacks of every layer, each with its own stored type and size, and the slot
//! count and host homes that follow from those sizes. A model's layers differ in type per stack
//! (the gate and up of a layer share a type, the down has its own), so the sizes are per stack
//! and per layer, never one uniform record.

use super::{
    GgufModel,
    tensors::{Mapped, Slot},
};
use crate::{
    EngError,
    family::{
        expert_bank::layer_stream::Home,
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

/// How the routed experts are held: all on the card, or `slots` per layer cached on it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::family::qwen::flash_next) enum ExpertSetup {
    /// Every expert of every layer sits on the card as the file stores it.
    Resident,
    /// A cache of `slots` experts per layer, filled from the host or the file.
    Cached(CachePlan),
}

/// How the routed experts are held when they do not all fit the card: `slots` experts of every
/// layer sit on the card, and each layer's whole set of experts lives in pinned host memory
/// ([`Home::Pinned`]) or is read from the model file in place on a miss ([`Home::Disk`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::family::qwen::flash_next) struct CachePlan {
    /// Experts of each layer held on the card.
    pub(in crate::family::qwen::flash_next) slots: usize,
    /// Where each layer's experts are read from on a miss.
    pub(in crate::family::qwen::flash_next) homes: Vec<Home>,
    /// Experts picked per token and layer: the most one pass can miss.
    pub(in crate::family::qwen::flash_next) top_k: usize,
}

impl CachePlan {
    /// Layers pinned in host memory and layers read from disk.
    pub(in crate::family::qwen::flash_next) fn counts(&self) -> (usize, usize) {
        let pinned = self.homes.iter().filter(|&&home| home == Home::Pinned).count();
        (pinned, self.homes.len() - pinned)
    }
}

impl std::fmt::Display for CachePlan {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (pinned, disk) = self.counts();
        write!(
            out,
            "{} expert slots in every layer on the card, {pinned} layers pinned in host memory, {disk} read from the file",
            self.slots
        )
    }
}

/// The host home of every layer: the largest layers first take pinned memory while `host_bytes`
/// lasts, the rest are read from the file. `slots` is the planner's slot count per layer.
pub(in crate::family::qwen::flash_next) fn cache_plan(
    stacks: &[LayerStacks],
    top_k: usize,
    slots: usize,
    host_bytes: u64,
) -> CachePlan {
    let bytes = |layer: usize| stacks[layer].stacks.iter().map(|stack| stack.bytes).sum::<u64>();
    let mut order: Vec<usize> = (0..stacks.len()).collect();
    order.sort_by_key(|&layer| (std::cmp::Reverse(bytes(layer)), layer));
    let mut homes = vec![Home::Disk; stacks.len()];
    let mut left = host_bytes;
    for layer in order {
        if bytes(layer) <= left {
            left -= bytes(layer);
            homes[layer] = Home::Pinned;
        }
    }
    CachePlan {
        slots,
        homes,
        top_k,
    }
}

#[cfg(test)]
mod tests {
    use super::{cache_plan, layer_stacks};
    use crate::{
        family::{
            expert_bank::layer_stream::Home,
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
        // A host budget of a third of the experts pins the largest layers that fit it and reads
        // the rest from the file; one that holds everything pins every layer.
        let plan = cache_plan(&stacks, 10, 167, total / 3);
        let (pinned, disk) = plan.counts();
        assert!(pinned > 0 && disk > 0 && pinned + disk == 48);
        let pinned_bytes: u64 = (0..48)
            .filter(|&layer| plan.homes[layer] == Home::Pinned)
            .map(record)
            .sum();
        assert!(pinned_bytes <= total / 3);
        assert_eq!(cache_plan(&stacks, 10, 167, total).counts(), (48, 0));
        assert_eq!(cache_plan(&stacks, 10, 167, 0).counts(), (0, 48));
    }
}
