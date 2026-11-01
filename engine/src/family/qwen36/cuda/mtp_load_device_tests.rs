//! The MTP head loaded beside the model, on the GPU held exclusively (about 16 GiB).
//!
//! `FIFI_EXL3_CHECKPOINT` names the model's pack and `FIFI_MTP_CHECKPOINT` the pack that holds the
//! head. The test loads both, checks that every tensor of the head is present under its name with
//! the shape the draft step needs, reports the device bytes the head added, and compares one
//! projection of the head (`mtp.fc`, which has the head's own codebook) with the fp64 reference of
//! the stored matrix.

use std::path::Path;

use super::{
    common_tests::{dims, inventory},
    exl3_load::load_exl3_dir_with_head,
    exl3_load_device_tests::{matrix_of, relative_l2},
    layer_table::LayerTable,
    mtp_load::{EMBEDDING_NORM_NAME, FC_NAME, HIDDEN_NORM_NAME, OUTPUT_NORM_NAME},
};
use crate::{
    family::{
        gguf_model::Role,
        project::cuda::{CudaProjector, CudaWeight, Reserve},
    },
    kernels::cuda::device_present,
    weights::exl3::Checkpoint,
};

const HIDDEN: usize = 2048;
const FIRST_FULL_LAYER: usize = 3;

fn input(len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| (((i * 7 + 5) % 29) as f32 - 14.0) / 16.0)
        .collect()
}

#[test]
#[ignore = "the GPU, held exclusively: loads the real 4.00 bpw pack and the MTP head of a second pack"]
fn the_head_loads_beside_the_model() {
    if !device_present("the_head_loads_beside_the_model") {
        println!("SKIPPED: no device");
        return;
    }
    let model = std::env::var_os("FIFI_EXL3_CHECKPOINT").expect("FIFI_EXL3_CHECKPOINT");
    let head = std::env::var_os("FIFI_MTP_CHECKPOINT").expect("FIFI_MTP_CHECKPOINT");
    let projector = CudaProjector::new().expect("projector");
    let free_before = projector.memory_info().expect("memory").free;
    let alone = super::exl3_load::load_exl3_dir(&projector, Path::new(&model), Reserve::default())
        .expect("model load");
    let free_alone = projector.memory_info().expect("memory").free;
    drop(alone);
    let weights = load_exl3_dir_with_head(
        &projector,
        Path::new(&model),
        Some(Path::new(&head)),
        Reserve::default(),
    )
    .expect("load with the head");
    let free_with = projector.memory_info().expect("memory").free;
    println!(
        "DEVICE_BYTES model_alone={} with_head={}",
        free_before.saturating_sub(free_alone),
        free_before.saturating_sub(free_with)
    );

    let table = LayerTable::new(&dims(), &inventory())
        .expect("table")
        .single(FIRST_FULL_LAYER, dims().shape.kinds.len())
        .expect("block table");
    for role in [
        Role::AttnQ,
        Role::AttnK,
        Role::AttnV,
        Role::AttnOutput,
        Role::FfnGateShexp,
    ] {
        let name = table.name(0, role).expect("name");
        assert!(
            matches!(weights.get(name), Some(CudaWeight::Exl3 { .. })),
            "{name}"
        );
    }
    for role in [Role::FfnGateExps, Role::FfnUpExps, Role::FfnDownExps] {
        let name = table.name(0, role).expect("name");
        let Some(CudaWeight::Exl3Bank { bank, .. }) = weights.get(name) else {
            panic!("{name} is not a bank");
        };
        assert_eq!(bank.count(), 256, "{name}");
    }
    for name in [HIDDEN_NORM_NAME, EMBEDDING_NORM_NAME, OUTPUT_NORM_NAME] {
        let Some(CudaWeight::Raw { buffer, .. }) = weights.get(name) else {
            panic!("{name} is not a raw vector");
        };
        assert_eq!(buffer.bytes(), HIDDEN * 4, "{name}");
    }
    let Some(CudaWeight::Exl3 { weight, .. }) = weights.get(FC_NAME) else {
        panic!("{FC_NAME} is missing");
    };
    assert_eq!(weight.shape, [2 * HIDDEN, HIDDEN]);
    println!("FC book={:?} bits_twice={}", weight.book, weight.bits_twice);

    let checkpoint = Checkpoint::open(Path::new(&head)).expect("head pack");
    let original = matrix_of(&checkpoint, "mtp.fc");
    let row = input(2 * HIDDEN);
    let mut expected = vec![0.0; HIDDEN];
    let wide: Vec<f64> = row.iter().map(|&v| f64::from(v)).collect();
    original
        .project_fp64_into(&wide, 1, &mut expected)
        .expect("reference");
    let (actual, flag) = crate::family::project::cuda::exl3::project_one_row(
        &projector,
        &weight.view().expect("view"),
        &row,
    )
    .expect("projection");
    assert_eq!(flag, 0, "arithmetic flag");
    let error = relative_l2(&actual, &expected);
    println!("FC relative_l2={error:.3e}");
    assert!(error < 1e-3, "mtp.fc against fp64: {error}");
}
