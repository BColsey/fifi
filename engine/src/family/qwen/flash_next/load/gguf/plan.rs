//! The memory plan of a Flash-Next GGUF: the sizes the tensor map gives, joined to the context
//! budget and the expert tier planner the EXL3 pack uses.
//!
//! What differs from the pack is the expert size. A pack stores every routed expert in one
//! format, so every expert has one size. A GGUF quantizes each layer's expert matrices on
//! their own (a 3-bit mixed-quant file mixes IQ2_S, IQ3_XXS, IQ4_XS, IQ4_NL and Q2_0 across layers),
//! so the size of an expert differs between layers. The planner's uniform slot is the rounded-up
//! mean expert size. Part-major stacks can add up to six bytes of alignment padding per layer;
//! the planner reserves that worst case separately from free VRAM. The staged `ExpertAllocation`
//! computes exact stack sizes but is not yet wired into this planner. The disk-tier staging ring
//! is charged for the largest record.
//!
//! The tiers: expert slots in video memory first (the same count in every layer), the rest in
//! pinned host memory while the host's budget carries it, and otherwise read from the GGUF
//! files on a miss through the staging ring. A host that cannot pin the cold experts of an
//! 83.6 GB model therefore plans the disk tier and does not refuse.

use super::{GgufModel, tensors::Place};
use crate::family::{
    qwen::flash_next::host_plan::{HostFrame, HostMemory},
    recipe::{Global, Role as Wanted},
};
use crate::{
    EngError,
    family::{
        expert_bank::bank_tier::rung_ring_placeholder_bytes,
        qwen::flash_next::moe::{
            load_plan::FlashLoadPlan,
            tier_plan::{TierInputs, TierShape},
        },
    },
};

fn refuse(detail: impl AsRef<str>) -> EngError {
    EngError::MemoryPlan(format!("Flash-Next GGUF plan: {}", detail.as_ref()))
}

/// Two additional part starts per layer, each rounded up by at most three bytes.
pub(in super::super::super) fn stack_alignment_reserve(layers: usize) -> Result<u64, EngError> {
    u64::try_from(layers)
        .ok()
        .and_then(|n| n.checked_mul(6))
        .ok_or_else(|| refuse("expert alignment reserve overflows"))
}

/// What the plan reads from the file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in super::super::super) struct GgufSizes {
    pub(in super::super::super) shape: TierShape,
    /// Device bytes of the resident trunk in the forms the executor holds it in.
    pub(in super::super::super) trunk_bytes: u64,
    /// The embedding table, whose rows the host gathers.
    pub(in super::super::super) embedding_bytes: u64,
    /// The n-gram table, read row by row from its file.
    pub(in super::super::super) ngram_bytes: u64,
    /// The largest per-layer expert record: what one staging slot holds.
    pub(in super::super::super) record_bytes_max: u64,
    /// The routed experts of all layers, as stored.
    pub(in super::super::super) expert_bytes: u64,
}

/// What the machine offers: the card's free memory after the CUDA context exists, and the
/// host's total and available memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in super::super::super) struct Card {
    pub(in super::super::super) free_vram: u64,
    pub(in super::super::super) host: HostMemory,
    /// Video memory set aside for the prefill ring before the bank sizes itself.
    pub(in super::super::super) ring_budget: u64,
}

impl GgufModel {
    /// The sizes the memory plan needs.
    pub(in super::super::super) fn sizes(&self) -> Result<GgufSizes, EngError> {
        let layers = self.config.layers();
        let experts = u64::from(self.config.dims.experts);
        let top_k = self.config.dims.top_k as usize;
        let mut total = 0u64;
        let mut record_max = 0u64;
        for layer in 0..layers {
            let mut layer_bytes = 0u64;
            for role in [Wanted::ExpertGate, Wanted::ExpertUp, Wanted::ExpertDown] {
                let tensor = self
                    .map
                    .get(role, Some(layer))
                    .ok_or_else(|| refuse(format!("layer {layer} has no {role:?} tensor")))?;
                if tensor.bytes % experts != 0 {
                    return Err(refuse(format!(
                        "{} has {} bytes, which do not divide over {experts} experts",
                        tensor.name, tensor.bytes
                    )));
                }
                layer_bytes += tensor.bytes;
            }
            total += layer_bytes;
            record_max = record_max.max(layer_bytes / experts);
        }
        let count = layers as u64 * experts;
        let shape = TierShape {
            layers,
            experts: experts as usize,
            top_k,
            expert_bytes: total.div_ceil(count),
        };
        let embedding = self
            .map
            .get(Global::Embedding, None)
            .ok_or_else(|| refuse("no embedding tensor"))?
            .bytes;
        let ngram = self
            .map
            .get(Global::NgramTable, None)
            .ok_or_else(|| refuse("no n-gram table"))?
            .bytes;
        debug_assert_eq!(total, self.map.file_bytes(Place::Experts));
        Ok(GgufSizes {
            shape,
            trunk_bytes: self.map.device_bytes(),
            embedding_bytes: embedding,
            ngram_bytes: ngram,
            record_bytes_max: record_max,
            expert_bytes: total,
        })
    }

    /// The load plan on `card`. The disk tier is offered (the staging ring is charged), so a
    /// host that cannot pin the cold experts plans the tier that reads them from the files.
    pub(in super::super::super) fn load_plan(&self, card: Card) -> Result<FlashLoadPlan, EngError> {
        let sizes = self.sizes()?;
        let inputs = TierInputs::measured(
            sizes.shape,
            (0, 0),
            crate::family::device_budget::KvPlan {
                positions: 1,
                bytes_per_position: 1,
            },
            card.free_vram
                .saturating_sub(card.ring_budget)
                .saturating_sub(stack_alignment_reserve(sizes.shape.layers)?),
            HostFrame::with_default_terms(card.host),
            rung_ring_placeholder_bytes(
                usize::try_from(sizes.record_bytes_max)
                    .map_err(|_| refuse("the largest expert record overflows usize"))?,
            ),
        );
        let shape = self.config.model_shape()?;
        FlashLoadPlan::from_sizes(
            &shape,
            self.config.kinds(),
            (sizes.shape, sizes.trunk_bytes, sizes.embedding_bytes),
            sizes.ngram_bytes,
            inputs,
        )
    }
}
