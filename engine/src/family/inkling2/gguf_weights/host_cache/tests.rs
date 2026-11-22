use std::{borrow::Cow, io::Write};

use super::*;
use crate::{family::mapped_file::MappedFile, weights::GgufType};

#[test]
fn host_weight_cache_preserves_vector_order_and_row_contract() {
    let names = [
        "model.llm.middle_cycle_cross_pass_queries",
        "model.llm.layers.0.attn.k_sconv.static_w",
        "model.llm.embed.weight",
    ];
    let values = [-0.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0];
    let mut file = tempfile::NamedTempFile::new().unwrap();
    for value in values {
        file.write_all(&value.to_le_bytes()).unwrap();
    }
    let len = size_of_val(&values);
    let mut weights = GgufTextWeights {
        mapping: MappedFile::open(file.path(), len as u64).unwrap(),
        tensors: names
            .into_iter()
            .zip([
                VectorOrder::Stored,
                VectorOrder::ConvTaps,
                VectorOrder::Stored,
            ])
            .map(|(name, vector_order)| {
                (
                    name.into(),
                    Tensor {
                        kind: GgufType::F32,
                        bytes: 0..len,
                        columns: 3,
                        rows: 2,
                        planes: 1,
                        vector_order,
                        decoded: None,
                    },
                )
            })
            .collect(),
        #[cfg(all(feature = "native-cuda", target_os = "linux"))]
        cuda: None,
        #[cfg(all(feature = "native-cuda", target_os = "linux"))]
        ram: None,
    };
    let before = weights.vec(names[1], 6).unwrap().into_owned();
    assert_eq!(before, [2.0, 5.0, 1.0, 4.0, -0.0, 3.0]);
    assert_eq!(populate(&mut weights).unwrap(), (2, len * 2));
    let row = weights.row(names[0], 0, 3).unwrap();
    assert!(matches!(row, Cow::Borrowed(_)));
    assert_eq!(row[0].to_bits(), (-0.0_f32).to_bits());
    assert_eq!(&*weights.row(names[0], 1, 3).unwrap(), &values[3..]);
    let reordered = weights.vec(names[1], 6).unwrap();
    assert!(matches!(reordered, Cow::Borrowed(_)));
    assert!(
        before
            .iter()
            .zip(reordered.iter())
            .all(|(a, b)| a.to_bits() == b.to_bits())
    );
    assert!(matches!(
        weights.row(names[1], 0, 3).unwrap(),
        Cow::Owned(_)
    ));
    assert!(matches!(
        weights.row(names[2], 0, 3).unwrap(),
        Cow::Owned(_)
    ));
    assert!(weights.vec(names[0], 3).is_err());
    assert!(weights.row(names[0], 2, 3).is_err());
    assert!(weights.row(names[0], 1, 2).is_err());
}

#[test]
#[ignore = "requires the Inkling 2 Mini GGUF checkpoint"]
fn host_weight_access_probe() {
    use std::{env, hint::black_box, path::Path, time::Instant};

    let model = env::var_os("FIFI_INKLING_MODEL_DIR").expect("set FIFI_INKLING_MODEL_DIR");
    let path = Path::new(&model);
    let (_, weights) = super::super::GgufPlan::open(path)
        .unwrap()
        .map(path)
        .unwrap();
    let selected: Vec<_> = weights
        .tensors
        .iter()
        .filter(|(name, _)| selected(name))
        .collect();
    let mut samples = Vec::new();
    let rounds = 250;
    for _ in 0..6 {
        let started = Instant::now();
        for _ in 0..rounds {
            for &(name, tensor) in &selected {
                if name.contains("_queries") && matches!(tensor.vector_order, VectorOrder::Stored) {
                    for row in 0..tensor.rows {
                        black_box(
                            weights
                                .row(black_box(name), row, tensor.columns)
                                .unwrap()
                                .as_ref(),
                        );
                    }
                } else {
                    black_box(
                        weights
                            .vec(black_box(name), tensor.rows * tensor.columns)
                            .unwrap()
                            .as_ref(),
                    );
                }
            }
        }
        samples.push(started.elapsed().as_secs_f64() * 1e3);
    }
    eprintln!(
        "Host weight probe: {} tensors, {rounds} traversals per trial, ms {samples:?}",
        selected.len()
    );
}
