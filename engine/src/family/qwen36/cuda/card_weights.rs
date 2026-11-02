//! The device half of the manifest: one card's [`CardLoad`] turned into
//! the single `load_sliced` call that uploads it. Every sliced tensor
//! lands under its own name with its sliced shape, every other named
//! tensor loads whole, and card-zero-only tensors are simply not named on
//! the other cards — the loader never decides placement, it only repeats
//! what the manifest (built from the one authoritative `ShardTable`)
//! says.
//!
//! The whole request is one call, so it is all-or-nothing the same way
//! `load_sliced` is: on any refusal or device error the partially built
//! `CudaWeights` drops with the error and every allocation it already
//! made is released — no partially usable card survives to a session.

use crate::EngError;
use crate::family::project::cuda::{CudaProjector, CudaWeights, LoadOptions, SliceSpec};
use crate::family::qwen36::cuda::card_load::CardLoad;
use crate::family::qwen36::cuda::shard::ShardLoad;
use crate::weights::GgufWeights;

/// One card's upload request: the tensors to name, and which of those to
/// slice (every other named tensor loads whole through the same call).
#[derive(Debug, Default)]
pub(crate) struct CardUploadPlan {
    pub(crate) names: Vec<String>,
    pub(crate) slices: Vec<(String, SliceSpec)>,
}

/// The `load_sliced` arguments of `card`'s manifest: whole tensors named
/// whole, sliced tensors named with their slice, and the card-zero-only
/// tensor (the embedding) named only on card 0.
pub(crate) fn card_upload_plan(load: &CardLoad, card: usize) -> CardUploadPlan {
    let mut plan = CardUploadPlan::default();
    for entry in &load.entries {
        match &entry.load {
            // Absent from every other card: not named, so nothing uploads
            // and no launcher can reach for it there.
            ShardLoad::CardZeroOnly if card != 0 => continue,
            ShardLoad::Sliced(spec) => {
                plan.slices.push((entry.name.clone(), spec.clone()));
                plan.names.push(entry.name.clone());
            }
            ShardLoad::Whole | ShardLoad::CardZeroOnly => {
                plan.names.push(entry.name.clone());
            }
        }
    }
    plan
}

/// Uploads `card`'s manifest from the GGUF file in one all-or-nothing
/// `load_sliced` call (Native layout; the caller's admission governs the
/// device).
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "staged for the two-device start and the card block rigs, its first callers"
    )
)]
pub(crate) fn load_card_weights(
    projector: &CudaProjector,
    weights: &GgufWeights,
    load: &CardLoad,
    card: usize,
    options: LoadOptions<'_>,
) -> Result<CudaWeights, EngError> {
    let plan = card_upload_plan(load, card);
    let names: Vec<&str> = plan.names.iter().map(String::as_str).collect();
    let slices: Vec<(&str, SliceSpec)> = plan
        .slices
        .iter()
        .map(|(name, spec)| (name.as_str(), spec.clone()))
        .collect();
    projector.load_sliced(weights, &names, options, &slices)
}
