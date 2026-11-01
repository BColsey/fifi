//! Device checks of `Exl3Bank`: a member projects exactly like the same matrix uploaded alone.

use std::collections::HashMap;

use super::*;
use crate::{
    EngError,
    kernels::cuda::device_present,
    tensor::{Dtype, Tensor},
    weights::{
        Tier, WeightSource,
        exl3::{Config, Matrix},
    },
};
use half::f16;

const K: usize = 256;
const N: usize = 384;
const WORDS: usize = 48;
const MUL1_MARKER: u32 = 0x83dc_d12d;
const MCG_MARKER: u32 = 0xcbac_1fed;

struct Components(HashMap<String, Tensor>);

impl WeightSource for Components {
    fn tensor(&self, key: &str) -> Option<Tensor> {
        self.0.get(key).cloned()
    }
    fn resident_bytes(&self) -> u64 {
        self.0.values().map(Tensor::byte_len).sum()
    }
    fn tier(&self) -> Tier {
        Tier::Resident
    }
}

fn pseudo(seed: u32, count: usize) -> Vec<u16> {
    let mut state = seed;
    (0..count)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 16) as u16
        })
        .collect()
}

fn half_vector(seed: u32, count: usize) -> Tensor {
    let bytes = pseudo(seed, count)
        .into_iter()
        .flat_map(|word| f16::from_f32((f32::from(word % 255) + 1.0) / 128.0).to_le_bytes())
        .collect();
    Tensor::try_from_host(Dtype::F16, vec![count as u32], bytes).unwrap()
}

/// A matrix with its own trellis and scales per `seed`, `rows * 16` tiles of `words` words, in
/// the `mul1` codebook.
fn member(seed: u32, shape: [usize; 2], words: usize, bias: bool) -> Matrix {
    member_in(seed, shape, words, (bias, false))
}

/// The same with a choice of bias and of the `mcg` codebook; the pack config then declares none.
fn member_in(seed: u32, shape: [usize; 2], words: usize, (bias, mcg): (bool, bool)) -> Matrix {
    let [k, n] = shape;
    let trellis: Vec<u8> = pseudo(seed, k / 16 * (n / 16) * words)
        .into_iter()
        .flat_map(u16::to_le_bytes)
        .collect();
    let mut parts = HashMap::from([
        (
            "m.trellis".to_string(),
            Tensor::try_from_host(
                Dtype::I16,
                vec![(k / 16) as u32, (n / 16) as u32, words as u32],
                trellis,
            )
            .unwrap(),
        ),
        ("m.suh".to_string(), half_vector(seed + 1, k)),
        ("m.svh".to_string(), half_vector(seed + 2, n)),
        (
            if mcg { "m.mcg" } else { "m.mul1" }.to_string(),
            Tensor::try_from_host(
                Dtype::I32,
                vec![],
                if mcg { MCG_MARKER } else { MUL1_MARKER }
                    .to_le_bytes()
                    .to_vec(),
            )
            .unwrap(),
        ),
    ]);
    if bias {
        parts.insert("m.bias".to_string(), half_vector(seed + 3, n));
    }
    let config = Config::from_json(if mcg {
        br#"{"quantization_config":{"quant_method":"exl3","version":"1.4.9"}}"#.as_slice()
    } else {
        br#"{"quantization_config":{"quant_method":"exl3","version":"1.4.9","codebook":"mul1"}}"#
            .as_slice()
    })
    .unwrap();
    Matrix::load(&Components(parts), "m", &config).unwrap()
}

#[test]
fn a_bank_member_projects_exactly_like_the_same_matrix_uploaded_alone() {
    if !device_present("EXL3 bank members") {
        return;
    }
    let projector = CudaProjector::new().expect("projector");
    let members: Vec<Matrix> = (0..3)
        .map(|i| member(100 + 7 * i, [K, N], WORDS, false))
        .collect();
    let bank = projector
        .upload_exl3_bank(
            members.len(),
            Reserve::default(),
            |i| Ok(members[i].clone()),
        )
        .expect("bank upload");
    assert_eq!(
        (bank.count(), bank.shape(), bank.bits_twice()),
        (3, [K, N], 6)
    );
    let input: Vec<f32> = (0..K)
        .map(|i| ((i * 5 % 23) as f32 - 11.0) / 16.0)
        .collect();
    let mut outputs = Vec::new();
    for (index, matrix) in members.iter().enumerate() {
        let alone = projector
            .upload_exl3(matrix, Reserve::default())
            .expect("single upload");
        let (expected, flag) = project_one_row(&projector, &alone.view().unwrap(), &input).unwrap();
        assert_eq!(flag, 0);
        let (actual, flag) =
            project_one_row(&projector, &bank.view(index).unwrap(), &input).unwrap();
        assert_eq!(flag, 0);
        assert!(
            expected.iter().any(|v| *v != 0.0),
            "member {index} is not trivial"
        );
        assert_eq!(
            actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "member {index}"
        );
        outputs.push(actual);
    }
    // The members differ, so a shifted stride could not pass by accident.
    assert_ne!(outputs[0], outputs[1]);
    assert_ne!(outputs[1], outputs[2]);
    assert!(matches!(bank.view(3), Err(EngError::BadRequest(_))));
    let held: usize = bank.device_buffers().map(DeviceBuffer::bytes).sum();
    assert_eq!(held, 3 * (K * N / 256 * WORDS * 2 + (K + N) * 4));
    assert_eq!(bank.device_bytes(), held);
}

#[test]
fn a_bank_refuses_members_that_differ_from_the_first_or_carry_a_bias() {
    if !device_present("EXL3 bank refusals") {
        return;
    }
    let projector = CudaProjector::new().expect("projector");
    let first = member(1, [K, N], WORDS, false);
    let others = [
        ("shape", member(2, [N, K], WORDS, false)),
        ("rate", member(3, [K, N], 80, false)),
        ("bias", member(4, [K, N], WORDS, true)),
        ("codebook", member_in(5, [K, N], WORDS, (false, true))),
    ];
    for (what, other) in others {
        let result = projector.upload_exl3_bank(2, Reserve::default(), |i| {
            Ok(if i == 0 { first.clone() } else { other.clone() })
        });
        assert!(
            matches!(&result, Err(EngError::Checkpoint(text)) if text.contains("member 1")),
            "{what}"
        );
    }
    let empty = projector.upload_exl3_bank(0, Reserve::default(), |_| unreachable!());
    assert!(matches!(empty, Err(EngError::BadRequest(_))));
    let failing = projector.upload_exl3_bank(2, Reserve::default(), |i| {
        if i == 0 {
            Ok(first.clone())
        } else {
            Err(EngError::Checkpoint("member read failed".into()))
        }
    });
    assert!(matches!(failing, Err(EngError::Checkpoint(text)) if text == "member read failed"));
}
