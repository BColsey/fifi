//! The per-card weight-upload manifest: the load plan turned into exact
//! device bytes and slice plans. This is the host side of the real
//! GGUF-to-card loading — the loader drives `load` for whole tensors and
//! `load_sliced` for the planned slices, consuming [`TensorSlice::segments`]
//! and nothing else, so the file bytes it uploads are exactly the bytes
//! these plans describe.
//!
//! The manifest never decides placement: it walks [`shard_loads`], which
//! derives every range from the SAME [`ShardTable`] the planner selected
//! and the executor consumes. Byte counts are the planner's exact
//! [`plan_slice_rows`] figures, not the proportional estimate admission
//! uses while comparing candidates, so a receipt built from these totals
//! can be diffed against CUDA's observed allocations without estimation
//! slack. The DeltaNet `attn_qkv` and `conv1d` slices over-cover the file
//! by one replicated key copy per card (the kernel pairs value head `h`
//! with key head `h % 16`); their `sliced_bytes` deliberately count the
//! replicated copy, because that is what the card uploads and holds.

use crate::EngError;
use crate::family::gguf_model::Inventory;
use crate::family::qwen36::cuda::placement::plan_type;
use crate::family::qwen36::cuda::shard::{ShardLoad, ShardTable, shard_loads};
use crate::kernels::cuda::plan::{TensorSlice, native_device_bytes, plan_slice_rows};
use crate::weights::GgufType;

/// One inventory tensor as one card loads it.
#[derive(Debug)]
pub(crate) struct CardEntry {
    pub(crate) name: String,
    /// The planning type: the file quant where the role allows it, else
    /// the role's first allowed type.
    pub(crate) ty: GgufType,
    /// GGUF dims of the FULL tensor, GGUF order.
    pub(crate) dims: Vec<u64>,
    pub(crate) load: ShardLoad,
    /// Exact device-resident bytes: the whole tensor's, or the planned
    /// slice's. Whole tensors count [`native_device_bytes`], the same
    /// function the slice planner builds on, so the two can never
    /// disagree.
    pub(crate) device_bytes: u64,
    /// The planned slice when the card loads a slice.
    pub(crate) slice: Option<TensorSlice>,
}

/// The upload manifest of one card over the whole inventory: every tensor
/// the model holds, in inventory order.
#[derive(Debug)]
pub(crate) struct CardLoad {
    pub(crate) entries: Vec<CardEntry>,
    /// Exact bytes of this card's sliced tensors.
    pub(crate) sliced_bytes: u64,
    /// Exact bytes replicated on every card (norms, router, per-head
    /// vectors, shared-expert gate scalars).
    pub(crate) whole_bytes: u64,
    /// Exact bytes only card 0 uploads (the embedding table); 0 on every
    /// other card.
    pub(crate) solo_bytes: u64,
}

impl CardLoad {
    /// The bytes this card uploads in total.
    pub(crate) fn upload_bytes(&self) -> u64 {
        self.sliced_bytes + self.whole_bytes + self.solo_bytes
    }
}

/// The manifest of every card under `table`, in card order.
pub(crate) fn card_loads(
    table: &ShardTable,
    inventory: &Inventory,
    quant: GgufType,
) -> Result<Vec<CardLoad>, EngError> {
    (0..table.cards())
        .map(|card| card_load(table, inventory, quant, card))
        .collect()
}

fn card_load(
    table: &ShardTable,
    inventory: &Inventory,
    quant: GgufType,
    card: usize,
) -> Result<CardLoad, EngError> {
    let mut sliced = 0u64;
    let mut whole = 0u64;
    let mut solo = 0u64;
    let mut entries = Vec::with_capacity(inventory.tensors().len());
    for (name, load) in shard_loads(table, inventory, card)? {
        let tensor = inventory
            .get(&name)
            .ok_or_else(|| EngError::Checkpoint(format!("the load plan lost tensor {name}")))?;
        let ty = plan_type(tensor.allowed_types(), quant);
        let (device_bytes, slice) = match &load {
            ShardLoad::Whole | ShardLoad::CardZeroOnly => {
                (native_device_bytes(&name, ty, &tensor.dims)?, None)
            }
            ShardLoad::Sliced(spec) => {
                let plan = plan_slice_rows(&name, ty, &tensor.dims, spec.cols.clone(), &spec.rows)?;
                (plan.bytes, Some(plan))
            }
        };
        match load {
            ShardLoad::Whole => whole += device_bytes,
            // The entry stays on every card's manifest (the load plan is
            // card-shaped), but only card 0 counts and uploads it.
            ShardLoad::CardZeroOnly => solo += u64::from(card == 0) * device_bytes,
            ShardLoad::Sliced(_) => sliced += device_bytes,
        }
        entries.push(CardEntry {
            name,
            ty,
            dims: tensor.dims.clone(),
            load,
            device_bytes,
            slice,
        });
    }
    Ok(CardLoad {
        entries,
        sliced_bytes: sliced,
        whole_bytes: whole,
        solo_bytes: solo,
    })
}
