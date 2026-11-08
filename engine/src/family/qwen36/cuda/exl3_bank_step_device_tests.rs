//! The banked EXL3 step against the resident step, byte for byte, over a full decode.
//!
//! `FIFI_EXL3_CHECKPOINT` names the pack directory. The test runs the same prompt through a
//! resident step (every expert on the device) and a banked step (`slots = 8`, the admission
//! floor — every layer starts empty, so the first token misses everywhere and later tokens
//! miss wherever the routing walks), and requires the same token ids and the same hidden rows
//! at every position. The banked slot fill moves the store's staged bytes unchanged, so a
//! difference of even one rounding step is a wiring bug.
//!
//! The resident leg needs the whole pack on the device, so this test runs where that fits
//! (the 3090); the block-level banked parity in `exl3_moe_device_tests` covers the same
//! property on smaller GPUs.

use std::{cell::RefCell, path::Path};

use sha2::{Digest, Sha256};

use super::{
    common_tests::{dims, inventory},
    exl3_bank::Exl3SlotResolver,
    exl3_load::{BankedModel, load_exl3_banked_dir, load_exl3_dir},
    exl3_moe::Exl3MoeWeights,
    exl3_step_device_tests::{EOS, GENERATED, HEAD_ROWS, PROMPT, SAMPLER_VOCAB, options},
    layer_table::LayerTable,
    native_device_rig::device_present,
    resources::{StepResources, moe_dims},
    rules::QwenRules,
    step::{CaptureConfig, QwenResidentStep},
};
use crate::{
    EngError,
    family::project::cuda::{CudaProjector, CudaWeights, Reserve},
};

/// The tokens and the hidden rows of one decode, plus the resolver's hit/miss counters
/// (`(0, 0)` for the resident leg, which has no resolver).
struct Decoded {
    next: Vec<u32>,
    digest: String,
    hits: u64,
    misses: u64,
}

/// Decodes the fixed prompt plus `GENERATED` tokens on `step`, eager, capture off, recording
/// every sampled id and the hidden rows.
fn decode(step: &mut QwenResidentStep<'_>) -> Result<Decoded, EngError> {
    step.set_capture_config(CaptureConfig {
        capture: false,
        section_marks: false,
    })?;
    let mut next = Vec::new();
    let mut rows = Vec::new();
    let mut id = 0;
    for &token in &PROMPT {
        id = step.run_token(token, true)?;
        next.push(id);
        rows.push(step.row_bytes().to_vec());
    }
    for _ in 0..GENERATED {
        id = step.run_token(id, true)?;
        next.push(id);
        rows.push(step.row_bytes().to_vec());
    }
    let mut hash = Sha256::new();
    for id in &next {
        hash.update(&id.to_le_bytes());
    }
    for row in &rows {
        hash.update(row);
    }
    Ok(Decoded {
        digest: format!("{:x}", hash.finalize()),
        next,
        hits: 0,
        misses: 0,
    })
}

/// The resident leg: every expert on the device.
fn resident_leg(projector: &CudaProjector, weights: &CudaWeights) -> Result<Decoded, EngError> {
    let context = projector.context();
    let mut step_options = options();
    step_options.context_tokens = PROMPT.len() + GENERATED + 1;
    let rules = QwenRules::new(EOS.to_vec(), SAMPLER_VOCAB, HEAD_ROWS)?;
    let resources = StepResources::new(context, dims(), &inventory(), &step_options)?;
    let mut step = QwenResidentStep::new(context, &resources, weights, &step_options, rules)?;
    decode(&mut step)
}

/// The banked leg: consumes the model, every layer's slot table starting empty.
fn banked_leg(projector: &CudaProjector, model: BankedModel) -> Result<Decoded, EngError> {
    let context = projector.context();
    let mut step_options = options();
    step_options.context_tokens = PROMPT.len() + GENERATED + 1;
    let rules = QwenRules::new(EOS.to_vec(), SAMPLER_VOCAB, HEAD_ROWS)?;
    let resources = StepResources::new(context, dims(), &inventory(), &step_options)?;
    // Destructured so the store can move into the resolver while the blocks borrow the
    // weights.
    let BankedModel {
        weights,
        store,
        slots,
        experts,
        routed_bytes: _,
        dense_bytes: _,
    } = model;
    let table = LayerTable::new(&dims(), &inventory())?;
    let moe = moe_dims(&dims());
    let blocks = (0..table.len())
        .map(|layer| Exl3MoeWeights::resolve_banked(&weights, &table, layer, &moe, slots))
        .collect::<Result<Vec<_>, _>>()?;
    let bank = RefCell::new(Exl3SlotResolver::new(store, &blocks, experts)?);
    let mut step = QwenResidentStep::new_banked(
        context,
        &resources,
        &weights,
        &step_options,
        rules,
        slots,
        &bank,
    )?;
    let mut decoded = decode(&mut step)?;
    (decoded.hits, decoded.misses) = bank.borrow().stats();
    Ok(decoded)
}

#[test]
#[ignore = "requires the pinned EXL3 pack in FIFI_EXL3_CHECKPOINT and a GPU that holds it resident"]
fn banked_step_decodes_byte_identically_to_the_resident_step() {
    let dir = std::env::var_os("FIFI_EXL3_CHECKPOINT").expect("FIFI_EXL3_CHECKPOINT");
    assert!(device_present("banked step parity"), "a GPU is required");
    let projector = CudaProjector::new().expect("projector");
    projector
        .context()
        .set_single_row_default(crate::kernels::cuda::resident::exl3::SingleRow::Wide(1));

    // The resident leg first, then drop its weights: the banked leg loads its own banks and
    // the host store needs the room.
    let resident = load_exl3_dir(&projector, Path::new(&dir), Reserve::default()).expect("pack");
    let want = resident_leg(&projector, &resident).expect("the resident leg");
    drop(resident);

    const SLOTS: usize = 8;
    let banked = load_exl3_banked_dir(&projector, Path::new(&dir), SLOTS, Reserve::default())
        .expect("the banked pack");
    let got = banked_leg(&projector, banked).expect("the banked leg");

    assert_eq!(
        got.next, want.next,
        "the banked step decoded different tokens"
    );
    assert_eq!(
        got.digest, want.digest,
        "the banked rows differ in some byte"
    );
    println!(
        "banked step parity: {} positions, hits {}, misses {}, sha256 {}",
        want.next.len(),
        got.hits,
        got.misses,
        want.digest
    );
    assert!(
        got.misses > 0,
        "the decode never missed; the resolver was not exercised"
    );
}
