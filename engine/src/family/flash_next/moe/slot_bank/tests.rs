//! The record layout at the real sizes (counted independently from the shapes), its refusals, and
//! a packed record against the same bytes handed to the CPU kernel directly.

use std::time::Duration;

use super::*;
use crate::kernels::{
    exl3_cpu::{Exl3Isa, ExpertCpu, ExpertPick},
    nvfp4_cpu::WaitPolicy,
};

/// 2560 x 640 experts at 3 bpw (48 words per tile).
fn real() -> SlotBankLayout {
    SlotBankLayout::new(2560, 640, [48, 48, 48]).unwrap()
}

#[test]
fn a_real_expert_record_is_1_881_600_bytes_in_the_order_gate_up_down() {
    let layout = real();
    // Per part: (k/16)(n/16) tiles of 96 bytes; suh 4k and svh 4n bytes.
    let tiles = 160 * 40;
    assert_eq!(layout.record_bytes(), 1_881_600);
    assert_eq!(layout.record_bytes(), 3 * tiles * 96 + 2 * 4 * (2560 + 640));
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
