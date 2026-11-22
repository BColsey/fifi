//! The record layout at the real sizes (counted independently from the shapes), its refusals, a
//! packed record against the same bytes handed to the CPU kernel directly, and the widening of a
//! stored matrix into a record part.

use std::{collections::HashMap, time::Duration};

use super::*;
use crate::{
    kernels::{
        exl3_cpu::{Exl3Isa, ExpertCpu, ExpertPick},
        nvfp4_cpu::WaitPolicy,
    },
    tensor::Tensor,
    weights::{Tier, WeightSource, exl3::Config},
};

/// 2560 x 640 experts at 3 bpw (48 words per tile).
fn real() -> SlotBankLayout {
    SlotBankLayout::new(2560, 640, [48, 48, 48]).unwrap()
}

#[test]
fn a_real_expert_record_is_1_881_600_bytes_in_the_order_gate_up_down() {
    let layout = real();
    // Per part: (k/16)(n/16) tiles of 96 bytes; suh 4k and svh 4n bytes (k + n = 3200 for each of the three).
    let tiles = 160 * 40;
    assert_eq!(layout.record_bytes(), 1_881_600);
    assert_eq!(layout.record_bytes(), 3 * tiles * 96 + 3 * 4 * (2560 + 640));
    let gate = layout.part(0);
    assert_eq!(gate.shape, [2560, 640]);
    assert_eq!(gate.trellis, 0..614_400);
    assert_eq!(gate.suh, 614_400..624_640);
    assert_eq!(gate.svh, 624_640..627_200);
    assert_eq!(layout.part(1).trellis.start, 627_200);
    let down = layout.part(2);
    assert_eq!(down.shape, [640, 2560]);
    assert_eq!(down.trellis, 1_254_400..1_868_800);
    assert_eq!(down.suh, 1_868_800..1_871_360);
    assert_eq!(down.svh, 1_871_360..1_881_600);
}

#[test]
fn the_shared_expert_rate_makes_a_larger_record_of_the_same_form() {
    // 5 bpw (80 words): 1,024,000-byte trellises as the real shared expert stores them.
    let layout = SlotBankLayout::new(2560, 640, [80, 80, 80]).unwrap();
    assert_eq!(layout.part(0).trellis.len(), 1_024_000);
    assert_eq!(layout.record_bytes(), 3 * 1_024_000 + 38_400);
}

#[test]
fn equal_byte_strides_let_one_slot_index_address_all_three_components() {
    let layout = real();
    assert_eq!(layout.trellis_stride_bytes(), 1_881_600);
    assert_eq!(
        layout.scale_stride_floats() * 4,
        layout.trellis_stride_bytes()
    );
    assert_eq!(layout.slot_offset(184).unwrap(), 184 * 1_881_600);
    assert_eq!(layout.bank_bytes(184).unwrap(), 346_214_400);
    let [trellis, suh, svh] = layout.bank_ranges(1, 184).unwrap();
    let last = 183 * 1_881_600;
    assert_eq!(trellis, 627_200..627_200 + 614_400 + last);
    assert_eq!(suh.start, 1_241_600);
    assert_eq!(suh.end, 1_251_840 + last);
    assert_eq!(svh, 1_251_840..1_254_400 + last);
    // Every component of every slot lies inside the launch ranges.
    for slot in [0, 1, 183] {
        let at = layout.slot_offset(slot).unwrap();
        let part = layout.part(1);
        assert!(trellis.start <= part.trellis.start + at && part.trellis.end + at <= trellis.end);
        assert!(suh.start <= part.suh.start + at && part.suh.end + at <= suh.end);
        assert!(svh.start <= part.svh.start + at && part.svh.end + at <= svh.end);
    }
    assert!(layout.bank_ranges(0, 0).is_err());
    assert!(layout.bank_ranges(3, 4).is_err());
}

#[test]
fn a_layout_off_the_grid_or_at_an_unknown_rate_is_refused() {
    assert!(SlotBankLayout::new(0, 640, [48; 3]).is_err());
    assert!(SlotBankLayout::new(2560, 100, [48; 3]).is_err());
    assert!(SlotBankLayout::new(2560, 640, [48, 20, 48]).is_err());
    assert!(SlotBankLayout::new(2560, 640, [0, 48, 48]).is_err());
    assert!(SlotBankLayout::new(usize::MAX - 127, 640, [48; 3]).is_err());
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn bytes(&mut self, len: usize) -> Vec<u8> {
        (0..len).map(|_| (self.next() >> 24) as u8).collect()
    }

    /// Finite F32 values in [-1, 1), little-endian.
    fn floats(&mut self, count: usize) -> Vec<u8> {
        (0..count)
            .flat_map(|_| {
                let unit = (self.next() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0;
                unit.to_le_bytes()
            })
            .collect()
    }
}

/// Random components for the three parts of `layout`.
fn random_parts(layout: &SlotBankLayout, rng: &mut Rng) -> Vec<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    (0..PARTS)
        .map(|i| {
            let part = layout.part(i);
            (
                rng.bytes(part.trellis.len()),
                rng.floats(part.suh.len() / 4),
                rng.floats(part.svh.len() / 4),
            )
        })
        .collect()
}

fn borrowed(parts: &[(Vec<u8>, Vec<u8>, Vec<u8>)]) -> [PartBytes<'_>; PARTS] {
    std::array::from_fn(|i| PartBytes {
        trellis: &parts[i].0,
        suh: &parts[i].1,
        svh: &parts[i].2,
    })
}

#[test]
fn packing_places_each_component_at_its_range_and_refuses_a_wrong_length_without_writing() {
    let layout = SlotBankLayout::new(128, 256, [16, 32, 24]).unwrap();
    let mut rng = Rng(0x51_07_BA_4C);
    let parts = random_parts(&layout, &mut rng);
    let mut record = vec![0u8; layout.record_bytes()];
    layout.pack_into(borrowed(&parts), &mut record).unwrap();
    for (i, (trellis, suh, svh)) in parts.iter().enumerate() {
        let part = layout.part(i);
        assert_eq!(&record[part.trellis.clone()], trellis.as_slice());
        assert_eq!(&record[part.suh.clone()], suh.as_slice());
        assert_eq!(&record[part.svh.clone()], svh.as_slice());
    }
    let mut short = parts.clone();
    short[2].2.pop();
    let mut untouched = vec![0xEEu8; layout.record_bytes()];
    assert!(layout.pack_into(borrowed(&short), &mut untouched).is_err());
    assert!(untouched.iter().all(|&b| b == 0xEE), "nothing was written");
    assert!(
        layout
            .pack_into(borrowed(&parts), &mut record[1..])
            .is_err()
    );
    assert!(layout.view(&record[1..]).is_err());
}

#[test]
fn the_cpu_kernel_gives_the_same_bits_from_a_record_as_from_the_components() {
    let cpu = ExpertCpu::new(
        1,
        WaitPolicy::SpinThenPark(Duration::from_micros(100)),
        Some(Exl3Isa::Scalar),
    )
    .unwrap();
    for words in [[16usize, 16, 16], [32, 48, 24]] {
        let layout = SlotBankLayout::new(128, 128, words).unwrap();
        let mut rng = Rng(0xC0DE_0000 + words[1] as u64);
        let parts = random_parts(&layout, &mut rng);
        let mut record = vec![0u8; layout.record_bytes()];
        layout.pack_into(borrowed(&parts), &mut record).unwrap();
        let projection = |i: usize| {
            let part = layout.part(i);
            Exl3Projection::new(
                part.shape,
                part.words,
                &parts[i].0,
                &parts[i].1,
                &parts[i].2,
            )
            .unwrap()
        };
        let direct = ExpertView {
            gate: projection(0),
            up: projection(1),
            down: projection(2),
        };
        let x: Vec<f32> = (0..128).map(|i| (i as f32 - 64.0) / 50.0).collect();
        let run = |view: ExpertView<'_>| {
            let mut out = vec![0f32; 128];
            cpu.forward(
                &x,
                &[ExpertPick {
                    expert: view,
                    weight: 0.75,
                }],
                &mut out,
            )
            .unwrap();
            out.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        };
        assert_eq!(
            run(layout.view(&record).unwrap()),
            run(direct),
            "words {words:?}"
        );
    }
}

/// Every component offset of a record is a multiple of 16 bytes (the launches' vector loads need it),
/// at the real sizes and for any 128-grid shape and stored rate.
#[test]
fn every_component_offset_is_a_multiple_of_16_bytes() {
    let mut layouts = vec![real()];
    for (hidden, inter) in [(128, 128), (256, 384), (2560, 512), (4096, 1024)] {
        for words in [16usize, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128] {
            layouts.push(SlotBankLayout::new(hidden, inter, [words; 3]).unwrap());
        }
    }
    for layout in layouts {
        assert_eq!(layout.record_bytes() % 16, 0);
        for part in 0..3 {
            let p = layout.part(part);
            for offset in [p.trellis.start, p.suh.start, p.svh.start, p.svh.end] {
                assert_eq!(offset % 16, 0, "part {part} offset {offset}");
            }
        }
    }
}

/// The `mul1` marker word that `Matrix::load` checks for the codebook.
const MUL1_MARKER: u32 = 0x83dc_d12d;

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

/// F16 scale values in (0, 2], one per index, as the pack stores `suh` and `svh`.
fn half_scales(seed: usize, count: usize) -> Vec<f16> {
    (0..count)
        .map(|i| f16::from_f32(((i * 7 + seed) % 255 + 1) as f32 / 128.0))
        .collect()
}

fn le_bytes(values: &[f16]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// A stored mul1 matrix of `shape` at `words` per tile, with random trellis bytes and F16 scales.
/// Returns the matrix and its three raw components.
fn stored_matrix(seed: u64, shape: [usize; 2], words: usize) -> (Matrix, [Vec<u8>; 3]) {
    let [k, n] = shape;
    let trellis = Rng(seed).bytes((k / 16) * (n / 16) * words * 2);
    let (suh, svh) = (le_bytes(&half_scales(3, k)), le_bytes(&half_scales(11, n)));
    let tensor = |dtype, dims: Vec<u32>, bytes: &[u8]| {
        Tensor::try_from_host(dtype, dims, bytes.to_vec()).unwrap()
    };
    let parts = HashMap::from([
        (
            "m.trellis".to_string(),
            tensor(
                Dtype::I16,
                vec![(k / 16) as u32, (n / 16) as u32, words as u32],
                &trellis,
            ),
        ),
        (
            "m.suh".to_string(),
            tensor(Dtype::F16, vec![k as u32], &suh),
        ),
        (
            "m.svh".to_string(),
            tensor(Dtype::F16, vec![n as u32], &svh),
        ),
        (
            "m.mul1".to_string(),
            tensor(Dtype::I32, vec![], &MUL1_MARKER.to_le_bytes()),
        ),
    ]);
    let config = Config::from_json(
        br#"{"quantization_config":{"quant_method":"exl3","version":"1.4.9","codebook":"mul1"}}"#
            .as_slice(),
    )
    .unwrap();
    let matrix = Matrix::load(&Components(parts), "m", &config).unwrap();
    (matrix, [trellis, suh, svh])
}

#[test]
fn a_part_from_a_stored_matrix_keeps_the_trellis_and_widens_the_half_scales_exactly() {
    let (matrix, [trellis, suh, svh]) = stored_matrix(0x00A1_1CE5, [256, 128], 48);
    let part = OwnedPart::from_matrix(&matrix).unwrap();
    assert_eq!(part.shape, [256, 128]);
    assert_eq!(
        part.words, 48,
        "twice the bits per weight is 6, eight words each"
    );
    assert_eq!(part.trellis, trellis);
    let widen = |half: &[u8]| -> Vec<u8> {
        half.as_chunks::<2>()
            .0
            .iter()
            .flat_map(|pair| f16::from_le_bytes(*pair).to_f32().to_le_bytes())
            .collect()
    };
    assert_eq!(part.suh.len(), 4 * 256);
    assert_eq!(part.svh.len(), 4 * 128);
    assert_eq!(part.suh, widen(&suh));
    assert_eq!(part.svh, widen(&svh));
    // Spot values: the widening is the exact binary16 value, not a rounded decimal.
    let first = f32::from_le_bytes(part.suh[..4].try_into().unwrap());
    assert_eq!(first, half_scales(3, 256)[0].to_f32());
    let last = f32::from_le_bytes(part.svh[4 * 127..].try_into().unwrap());
    assert_eq!(last, half_scales(11, 128)[127].to_f32());
    let borrowed = part.bytes();
    assert_eq!(
        (
            borrowed.trellis.len(),
            borrowed.suh.len(),
            borrowed.svh.len()
        ),
        (trellis.len(), 4 * 256, 4 * 128)
    );
}

#[test]
fn matrices_pack_into_a_record_part_by_part_and_a_wrong_rate_or_shape_is_refused() {
    let layout = SlotBankLayout::new(128, 256, [48, 32, 24]).unwrap();
    let gate = stored_matrix(1, [128, 256], 48);
    let up = stored_matrix(2, [128, 256], 32);
    let down = stored_matrix(3, [256, 128], 24);
    let record = layout
        .pack_matrices([&gate.0, &up.0, &down.0])
        .expect("matrices of the layout's shapes and rates");
    for (index, matrix) in [&gate.0, &up.0, &down.0].into_iter().enumerate() {
        let part = OwnedPart::from_matrix(matrix).unwrap();
        let at = layout.part(index);
        assert_eq!(&record[at.trellis.clone()], part.trellis.as_slice());
        assert_eq!(&record[at.suh.clone()], part.suh.as_slice());
        assert_eq!(&record[at.svh.clone()], part.svh.as_slice());
    }
    // Another rate for the up part, and the down matrix where the up one belongs.
    let up_other_rate = stored_matrix(4, [128, 256], 48);
    assert!(
        layout
            .pack_matrices([&gate.0, &up_other_rate.0, &down.0])
            .is_err()
    );
    assert!(layout.pack_matrices([&gate.0, &down.0, &up.0]).is_err());
}
