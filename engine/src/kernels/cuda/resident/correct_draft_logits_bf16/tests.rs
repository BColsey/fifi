use super::*;
use crate::kernels::cuda::testkit::{device_present, read_bytes, upload_filled};
use half::bf16;

const CAP: f32 = 30.0;
const CANONICAL_NAN: u32 = 0x7fc00000;

fn cap(merged: f32) -> bf16 {
    bf16::from_f32(CAP * (merged / CAP).tanh())
}

fn reference(base: f32, correction: f32) -> u32 {
    let merged =
        bf16::from_f32(bf16::from_f32(base).to_f32() + bf16::from_f32(correction).to_f32());
    if merged.is_finite() {
        cap(merged.to_f32()).to_f32().to_bits()
    } else {
        CANONICAL_NAN
    }
}

fn lut() -> Vec<u8> {
    (0..=u16::MAX)
        .flat_map(|bits| cap(bf16::from_bits(bits).to_f32()).to_le_bytes())
        .collect()
}

#[test]
fn matches_host_bitwise_at_every_bf16_index_and_rounding_boundary() {
    if !device_present("corrected BF16 draft logits oracle") {
        return;
    }
    // Preserve the sign of zero while sweeping every BF16 payload.
    let mut pairs: Vec<_> = (0..=u16::MAX)
        .map(|bits| {
            let base = bf16::from_bits(bits).to_f32();
            (base, 0.0_f32.copysign(base))
        })
        .collect();
    let bf16_max = bf16::from_bits(0x7f7f).to_f32();
    let half_step = 1.0 / 256.0;
    let halfway = 1.0 + half_step;
    pairs.extend([
        // The input is halfway between two BF16 values. Delaying either
        // operand's rounding until after the add changes the result.
        (halfway, half_step),
        (half_step, halfway),
        (-halfway, -half_step),
        (f32::from_bits(0x0000_8000), f32::from_bits(0x0000_8000)),
        (0.0, -0.0),
        (-0.0, 0.0),
        (-0.0, -0.0),
        (1.0, -1.0),
        (200.0, 0.0),
        (300.0, 0.0),
        (0.0, f32::NAN),
        (0.0, f32::INFINITY),
        (f32::INFINITY, f32::NEG_INFINITY),
        (f32::MAX, 0.0),
        (bf16_max, bf16_max),
        // The F32 addition fits, but the merged F32 value rounds to infinity.
        (bf16_max, f32::from_bits(0x7b00_0000)),
    ]);

    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let lut = lut();
    let table = ctx.upload(&lut).unwrap();
    let logits_bytes = pairs.len() * size_of::<f32>();
    // Both rows are in disjoint subranges of the same allocation.
    let mut input = vec![0xa5; 4];
    input.extend(pairs.iter().flat_map(|(base, _)| base.to_le_bytes()));
    input.extend(
        pairs
            .iter()
            .flat_map(|(_, correction)| correction.to_le_bytes()),
    );
    input.extend([0x5a; 4]);
    let logits = ctx.upload(&input).unwrap();
    ctx.correct_draft_logits_bf16(
        &stream,
        CorrectDraftLogitsCall {
            len: pairs.len(),
            base: logits.range(4, logits_bytes).unwrap(),
            correction: logits.range(4 + logits_bytes, logits_bytes).unwrap(),
            softcap_lut: table.range(0, SOFTCAP_LUT_BYTES).unwrap(),
        },
    )
    .unwrap();
    assert_eq!(stream.enqueued(), 1);
    stream.synchronize().unwrap();

    let actual = read_bytes(&ctx, &logits, input.len());
    assert_eq!(&actual[..4], &input[..4]);
    assert_eq!(&actual[4 + logits_bytes..], &input[4 + logits_bytes..]);
    assert_eq!(read_bytes(&ctx, &table, SOFTCAP_LUT_BYTES), lut);
    let actual = &actual[4..4 + logits_bytes];
    for (i, (&bytes, (base, correction))) in actual.as_chunks::<4>().0.iter().zip(pairs).enumerate()
    {
        assert_eq!(
            u32::from_le_bytes(bytes),
            reference(base, correction),
            "row {i}, base {base:?}, correction {correction:?}",
        );
    }
}

#[test]
fn checked_logit_bytes_reject_empty_and_excess_vocabulary() {
    assert_eq!(logits_bytes(0), None);
    assert_eq!(logits_bytes(1), Some(4));
    assert_eq!(logits_bytes(MAX_LOGITS), Some(MAX_LOGITS * 4));
    assert_eq!(logits_bytes(MAX_LOGITS + 1), None);
}

fn rejects(name: &str, expected: &str, change: impl FnOnce(&mut CorrectDraftLogitsCall<'_>)) {
    if !device_present(name) {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let base = upload_filled(&ctx, 0, 68);
    let correction = upload_filled(&ctx, 0, 68);
    let table = upload_filled(&ctx, 0, SOFTCAP_LUT_BYTES + 2);
    let mut call = CorrectDraftLogitsCall {
        len: 16,
        base: base.range(0, 64).unwrap(),
        correction: correction.range(0, 64).unwrap(),
        softcap_lut: table.range(0, SOFTCAP_LUT_BYTES).unwrap(),
    };
    change(&mut call);
    let Err(EngError::BadRequest(error)) = ctx.correct_draft_logits_bf16(&stream, call) else {
        panic!("expected BadRequest for {name}");
    };
    assert!(error.contains(expected), "{error}");
    assert_eq!(stream.enqueued(), 0);
}

#[test]
fn rejects_short_base_logits() {
    rejects("corrected draft logits short base", "needs 64", |call| {
        call.base = call.base.buffer().range(0, 60).unwrap();
    });
}

#[test]
fn rejects_misaligned_correction_logits() {
    rejects(
        "corrected draft logits misaligned correction",
        "not a multiple of 4",
        |call| {
            call.correction = call.correction.buffer().range(2, 64).unwrap();
        },
    );
}

#[test]
fn rejects_lut_of_inexact_size() {
    rejects(
        "corrected draft logits oversized lookup",
        "exactly 131072 bytes",
        |call| {
            call.softcap_lut = call
                .softcap_lut
                .buffer()
                .range(0, SOFTCAP_LUT_BYTES + 2)
                .unwrap();
        },
    );
}

#[test]
fn rejects_overlap_with_correction() {
    rejects(
        "corrected draft logits overlapping correction",
        "overlap",
        |call| {
            call.correction = call.base.buffer().range(4, 64).unwrap();
        },
    );
}

#[test]
fn rejects_overlap_with_lookup() {
    rejects(
        "corrected draft logits overlapping lookup",
        "overlap",
        |call| {
            call.base = call.softcap_lut.buffer().range(0, 64).unwrap();
        },
    );
}
