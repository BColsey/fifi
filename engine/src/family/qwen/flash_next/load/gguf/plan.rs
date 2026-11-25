//! The memory plan of a Flash-Next GGUF: the sizes the tensor map gives, joined to the context
//! budget and the expert tier planner the EXL3 pack uses.
//!
//! What differs from the pack is the expert size. A pack stores every routed expert in one
//! format, so every expert has one size. A GGUF quantizes each layer's expert matrices on
//! their own (a 3-bit mixed-quant file mixes IQ2_S, IQ3_XXS, IQ4_XS, IQ4_NL and Q2_0 across layers),
//! so the size of an expert differs between layers. The planner's uniform slot is the mean
//! expert size (the bank of a layer is sized by that layer's own records, so the bank's total
//! bytes are exact), and the staging ring of the disk tier is charged for the largest record.
//!
//! The tiers: expert slots in video memory first (the same count in every layer), the rest in
//! pinned host memory while the host's budget carries it, and otherwise read from the GGUF
//! files on a miss through the staging ring. A host that cannot pin the cold experts of an
//! 83.6 GB model therefore plans the disk tier and does not refuse.

use super::{GgufModel, tensors::Place};
use crate::family::{
    qwen::flash_next::host_plan::{HostFrame, HostMemory, HostTerms},
    recipe::Global,
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

/// What the plan reads from the file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::family::qwen::flash_next) struct GgufSizes {
    pub(in crate::family::qwen::flash_next) shape: TierShape,
    /// Device bytes of the resident trunk in the forms the executor holds it in.
    pub(in crate::family::qwen::flash_next) trunk_bytes: u64,
    /// The embedding table, whose rows the host gathers.
    pub(in crate::family::qwen::flash_next) embedding_bytes: u64,
    /// The n-gram table, read row by row from its file.
    pub(in crate::family::qwen::flash_next) ngram_bytes: u64,
    /// The largest per-layer expert record: what one staging slot holds.
    pub(in crate::family::qwen::flash_next) record_bytes_max: u64,
    /// The routed experts of all layers, as stored.
    pub(in crate::family::qwen::flash_next) expert_bytes: u64,
    /// Bytes of one expert record in each layer. The tier shape's uniform expert is the mean
    /// of these, so the plan's bank bytes are exact only for a bank whose slots in each layer
    /// hold that layer's own record size.
    pub(in crate::family::qwen::flash_next) layer_record_bytes: Vec<u64>,
}

/// What the machine offers: the card's free memory after the CUDA context exists, and the
/// host's total and available memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::family::qwen::flash_next) struct Card {
    pub(in crate::family::qwen::flash_next) free_vram: u64,
    pub(in crate::family::qwen::flash_next) host: HostMemory,
    /// Video memory set aside for the prefill ring before the bank sizes itself.
    pub(in crate::family::qwen::flash_next) ring_budget: u64,
}

impl GgufModel {
    /// The sizes the memory plan needs.
    pub(in crate::family::qwen::flash_next) fn sizes(&self) -> Result<GgufSizes, EngError> {
        let layers = self.config.layers();
        let experts = u64::from(self.config.dims.experts);
        let top_k = self.config.dims.top_k as usize;
        let index = self.expert_index()?;
        let mut per_layer = Vec::with_capacity(layers);
        for layer in 0..layers {
            per_layer.push(
                index
                    .record_bytes(layer)
                    .map_err(|error| refuse(format!("layer {layer}: {error:?}")))?,
            );
        }
        let total = index
            .total_bytes()
            .map_err(|error| refuse(format!("{error:?}")))?;
        let record_max = index
            .max_record_bytes()
            .map_err(|error| refuse(format!("{error:?}")))?;
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
            layer_record_bytes: per_layer,
        })
    }

    /// The host terms of the plan: the default heap, the two host F32 PLE projections (the
    /// bytes the pack start charges for its own) and the expert cache's pinned bounce buffer
    /// (one pass of the largest layer's records), all read from the plan's own numbers.
    pub(in super::super::super) fn host_terms(
        &self,
        sizes: &GgufSizes,
    ) -> Result<HostTerms, EngError> {
        let dims = &self.config.dims;
        let ple_rows = (u64::from(dims.hidden) * u64::from(dims.branches) + u64::from(dims.hidden))
            * self.config.model_shape()?.ple.embed() as u64;
        let largest = sizes.layer_record_bytes.iter().copied().max().unwrap_or(0);
        let engine_heap = HostTerms::DEFAULT
            .engine_heap
            .checked_add(ple_rows * 4)
            .and_then(|bytes| bytes.checked_add(self.config.top_k() as u64 * largest))
            .ok_or_else(|| refuse("the host terms overflow"))?;
        Ok(HostTerms {
            engine_heap,
            ..HostTerms::DEFAULT
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
            card.free_vram.saturating_sub(card.ring_budget),
            HostFrame::new(card.host, self.host_terms(&sizes)?),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::family::qwen::flash_next::load::gguf::testkit::{Spec, write_flash};

    /// The terms grow by the PLE projections' F32 bytes and the bounce buffer, and by nothing
    /// else.
    #[test]
    fn the_host_terms_charge_the_ple_projections_and_the_bounce_buffer() {
        let written = write_flash(&Spec::default());
        let model = GgufModel::open(&written.first)
            .expect("open")
            .expect("a model");
        let sizes = model.sizes().expect("sizes");
        let terms = model.host_terms(&sizes).expect("terms");
        let dims = &model.config().dims;
        let ple = (u64::from(dims.hidden) * u64::from(dims.branches) + u64::from(dims.hidden))
            * model.config().model_shape().expect("shape").ple.embed() as u64
            * 4;
        let largest = sizes.layer_record_bytes.iter().copied().max().unwrap_or(0);
        assert_eq!(
            terms.engine_heap,
            HostTerms::DEFAULT.engine_heap + ple + model.config().top_k() as u64 * largest
        );
        assert_eq!(
            terms,
            HostTerms {
                engine_heap: terms.engine_heap,
                ..HostTerms::DEFAULT
            }
        );
    }
}
