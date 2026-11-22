//! Comparisons against references frozen before the candidate implementation.
use super::*;
use crate::family::gemma4::resident::bolt::{ATTN_SPLIT_MAX, ATTN_SPLIT_MIN_KEYS};
use crate::kernels::cuda::{CudaContext, testkit};

#[test]
fn gemma_head_shapes_are_admitted_and_refused() {
    // The Bolt (8,2) shape and the Humanizer (16,1)/(16,8) shapes are admitted;
    // their neighbours are refused before any native call.
    for (q, kv) in [(16, 1), (16, 8), (8, 2)] {
        assert!(
            admit_gemma_heads(q, kv).is_ok(),
            "({q},{kv}) must be admitted"
        );
    }
    for (q, kv) in [
        (8, 1),
        (8, 8),
        (16, 2),
        (16, 4),
        (4, 2),
        (2, 8),
        (0, 0),
        (32, 8),
    ] {
        let error = admit_gemma_heads(q, kv).expect_err("neighbour must be refused");
        assert!(
            matches!(error, crate::EngError::BadRequest(_)),
            "({q},{kv}): {error:?}"
        );
    }
}

const LOCAL: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/testdata/gemma4/kernels/local-gqa2.bin"
));
const GLOBAL: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/testdata/gemma4/kernels/global-gqa16.bin"
));

#[test]
fn matches_the_frozen_reference_in_both_geometries() {
    if !testkit::device_present("attn_decode_gemma frozen fixtures") {
        return;
    }
    for (fixture, kv_heads, dim, window) in
        [(LOCAL, 8usize, 256usize, 16usize), (GLOBAL, 1, 512, 0)]
    {
        let ctx = CudaContext::new().expect("context");
        let stream = ctx.stream().expect("stream");
        let cells = 40;
        let q_bytes = 16 * dim * 4;
        let code_bytes = kv_heads * cells * dim;
        let scale_bytes = kv_heads * cells * (dim / 32) * 2;
        let expected_at = q_bytes + 2 * code_bytes + 2 * scale_bytes;
        assert_eq!(fixture.len(), expected_at + 16 * dim * 8);
        let input = testkit::upload_bytes(&ctx, &fixture[..expected_at]);
        let out = testkit::upload_filled(&ctx, 0xff, q_bytes);
        let len = testkit::upload_bytes(&ctx, &(cells as i32).to_le_bytes());
        let status = testkit::upload_bytes(&ctx, &0u32.to_le_bytes());
        ctx.attn_decode_gemma(
            &stream,
            AttnDecodeGemmaCall {
                out: out.range(0, q_bytes).expect("out"),
                status: Some(status.range(0, 4).expect("status")),
                q: input.range(0, q_bytes).expect("q"),
                k_codes: input.range(q_bytes, code_bytes).expect("k codes"),
                k_scales: input
                    .range(q_bytes + code_bytes, scale_bytes)
                    .expect("k scales"),
                v_codes: input
                    .range(q_bytes + code_bytes + scale_bytes, code_bytes)
                    .expect("v codes"),
                v_scales: input
                    .range(q_bytes + 2 * code_bytes + scale_bytes, scale_bytes)
                    .expect("v scales"),
                kv_len: len.range(0, 4).expect("length"),
                m: 1,
                n_q_head: 16,
                n_kv_head: kv_heads,
                head_dim: dim,
                cells,
                window,
                scale: 1.0,
                split: None,
                ring: 0,
            },
        )
        .expect("attention");
        stream.synchronize().expect("completion");
        assert_eq!(testkit::read_bytes(&ctx, &status, 4), [0, 0, 0, 0]);
        assert_eq!(
            testkit::read_bytes(&ctx, &input, expected_at),
            fixture[..expected_at]
        );
        let got = testkit::floats(&testkit::read_bytes(&ctx, &out, q_bytes));
        let expected = fixture[expected_at..].as_chunks::<8>().0;
        for (index, (actual, bytes)) in got.iter().zip(expected).enumerate() {
            let reference = f64::from_le_bytes(*bytes);
            assert!(
                (f64::from(*actual) - reference).abs() <= 3e-3 + 3e-3 * reference.abs(),
                "kv_heads={kv_heads} dim={dim} element={index}: {actual} versus {reference}"
            );
        }
    }
}

#[test]
fn refuses_an_unsupported_head_dim() {
    if !testkit::device_present("attn_decode_gemma dim refusal") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let q_dev = testkit::upload_f32(&ctx, &[0.0f32; 16 * 128]);
    let codes_dev = testkit::upload_filled(&ctx, 0, 8 * 4 * 128);
    let scales_dev = testkit::upload_filled(&ctx, 0, 8 * 4 * 2 * 2);
    let kv_len_dev = testkit::upload_bytes(&ctx, &0i32.to_le_bytes());
    let out_dev = testkit::upload_filled(&ctx, 0xff, 16 * 128 * 4);
    let err = ctx
        .attn_decode_gemma(
            &stream,
            AttnDecodeGemmaCall {
                out: out_dev.range(0, 16 * 128 * 4).expect("out"),
                status: None,
                q: q_dev.range(0, 16 * 128 * 4).expect("q"),
                k_codes: codes_dev.range(0, 8 * 4 * 128).expect("k"),
                k_scales: scales_dev.range(0, 8 * 4 * 2 * 2).expect("ks"),
                v_codes: codes_dev.range(0, 8 * 4 * 128).expect("v"),
                v_scales: scales_dev.range(0, 8 * 4 * 2 * 2).expect("vs"),
                kv_len: kv_len_dev.range(0, 4).expect("len"),
                m: 1,
                n_q_head: 16,
                n_kv_head: 8,
                head_dim: 128,
                cells: 4,
                window: 0,
                scale: 1.0,
                split: None,
                ring: 0,
            },
        )
        .expect_err("head dim refused");
    assert!(matches!(err, crate::EngError::BadRequest(_)), "{err:?}");
}

/// The split plan FFI and a hand mirror must agree on exact (chunk, n_active)
/// pairs: the wrapper sizes the partial buffer from max_splits, and the two
/// kernel copies of the arithmetic must stay consistent with this host view.
#[test]
fn split_plan_pins_to_hand_values() {
    use crate::kernels::cuda::native::resident::attn_decode_gemma::attn_decode_gemma_split_plan;
    let mirror = |n_keys: usize, max_splits: usize, min_keys: usize| -> Option<(usize, usize)> {
        let want = n_keys.div_ceil(min_keys);
        let n_split = want.clamp(1, max_splits);
        let chunk = n_keys.div_ceil(n_split);
        Some((chunk, n_keys.div_ceil(chunk)))
    };
    // Hand-checked corners: one split under the floor, a forced two-split,
    // the clamp at max_splits, a prime count with an uneven chunk, and the
    // exact floors.
    for (n_keys, max_splits, min_keys, want) in [
        (40_usize, 32_usize, 128_usize, (40_usize, 1_usize)),
        (40, 32, 32, (20, 2)),
        (12_288, 32, 128, (384, 32)),
        (10_007, 32, 128, (313, 32)),
        (128, 32, 128, (128, 1)),
        (129, 32, 128, (65, 2)),
    ] {
        assert_eq!(
            attn_decode_gemma_split_plan(n_keys, max_splits, min_keys),
            Some(want),
            "FFI ({n_keys}, {max_splits}, {min_keys})"
        );
        assert_eq!(mirror(n_keys, max_splits, min_keys), Some(want));
    }
    // Zero keys once reached the arithmetic and divided by its own zero
    // chunk (SIGFPE through the export); the export now refuses it outright.
    assert_eq!(attn_decode_gemma_split_plan(0, 32, 128), None);
}

/// The split path over the same frozen fixtures, forced to two splits by a
/// 32-key floor at 40 cells: the multi-split merge must stay within the
/// frozen tolerance (3e-3 absolute plus 3e-3 relative, as the single-pass
/// test states it).
#[test]
fn split_matches_the_frozen_reference_in_both_geometries() {
    if !testkit::device_present("attn_decode_gemma split fixtures") {
        return;
    }
    for (fixture, kv_heads, dim, window) in
        [(LOCAL, 8usize, 256usize, 16usize), (GLOBAL, 1, 512, 0)]
    {
        let ctx = CudaContext::new().expect("context");
        let stream = ctx.stream().expect("stream");
        let cells = 40;
        let heads = 16;
        let q_bytes = heads * dim * 4;
        let code_bytes = kv_heads * cells * dim;
        let scale_bytes = kv_heads * cells * (dim / 32) * 2;
        let expected_at = q_bytes + 2 * code_bytes + 2 * scale_bytes;
        let input = testkit::upload_bytes(&ctx, &fixture[..expected_at]);
        let out = testkit::upload_filled(&ctx, 0xff, q_bytes);
        let len = testkit::upload_bytes(&ctx, &(cells as i32).to_le_bytes());
        let part_floats = heads * 4 * (2 + dim);
        let part = testkit::upload_filled(&ctx, 0, part_floats * 4);
        ctx.attn_decode_gemma(
            &stream,
            AttnDecodeGemmaCall {
                out: out.range(0, q_bytes).expect("out"),
                status: None,
                q: input.range(0, q_bytes).expect("q"),
                k_codes: input.range(q_bytes, code_bytes).expect("k codes"),
                k_scales: input
                    .range(q_bytes + code_bytes, scale_bytes)
                    .expect("k scales"),
                v_codes: input
                    .range(q_bytes + code_bytes + scale_bytes, code_bytes)
                    .expect("v codes"),
                v_scales: input
                    .range(q_bytes + 2 * code_bytes + scale_bytes, scale_bytes)
                    .expect("v scales"),
                kv_len: len.range(0, 4).expect("length"),
                m: 1,
                n_q_head: heads,
                n_kv_head: kv_heads,
                head_dim: dim,
                cells,
                window,
                scale: 1.0,
                split: Some(AttnDecodeGemmaSplit {
                    part: part.range(0, part_floats * 4).expect("part"),
                    max_splits: 4,
                    // 10-key floor: 40 cells over exactly four active splits, so
                    // the merge order matters on the frozen fixtures.
                    min_keys: 10,
                }),
                ring: 0,
            },
        )
        .expect("split attention");
        stream.synchronize().expect("completion");
        let got = testkit::floats(&testkit::read_bytes(&ctx, &out, q_bytes));
        let expected = fixture[expected_at..].as_chunks::<8>().0;
        for (index, (actual, bytes)) in got.iter().zip(expected).enumerate() {
            let reference = f64::from_le_bytes(*bytes);
            assert!(
                (f64::from(*actual) - reference).abs() <= 3e-3 + 3e-3 * reference.abs(),
                "kv_heads={kv_heads} dim={dim} element={index}: {actual} versus {reference}"
            );
        }
    }
}

/// A long-context Bolt-shaped case (8 q heads, 2 kv heads, 512 dim, no window,
/// 12,288 cells, 32 splits of 384 keys) against an f64 host reference over the
/// same generated int8 codes and half scales. Tolerance: 3e-3 absolute plus
/// 3e-3 relative, the frozen-fixture bound; the single-pass kernel over the
/// same data must sit inside the same bound, which pins the split path to
/// v0's accuracy rather than only to the reference.
#[test]
fn split_matches_the_reference_at_long_context() {
    if !testkit::device_present("attn_decode_gemma long-context split") {
        return;
    }
    let (heads, kv_heads, dim, cells) = (8usize, 2usize, 512usize, 12_288usize);
    let groups = dim / 32;
    // Deterministic small LCG: codes in [-16, 15], half scales cycled over a
    // few exactly representable magnitudes.
    let mut state = 0x2f6e2b1u64;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 33) & 0x1f) as i32 - 16
    };
    // The logits are scaled so every split's maximum stays within about 20
    // nats of the global maximum: with raw dot products the split maxima are
    // hundreds of nats apart and all but the hottest split underflow to zero
    // weight in f32, which the every-split-weights-positive pin below forbids.
    let q: Vec<f32> = (0..heads * dim).map(|_| next() as f32 / 1024.0).collect();
    let k_codes: Vec<i8> = (0..kv_heads * cells * dim).map(|_| next() as i8).collect();
    let v_codes: Vec<i8> = (0..kv_heads * cells * dim).map(|_| next() as i8).collect();
    let magnitudes = [0.25f32, 0.5, 1.0, 2.0];
    let scale_at = |i: usize| magnitudes[i % magnitudes.len()];
    let k_scales: Vec<u16> = (0..kv_heads * cells * groups)
        .map(|i| half::f16::from_f32(scale_at(i)).to_bits())
        .collect();
    let v_scales: Vec<u16> = (0..kv_heads * cells * groups)
        .map(|i| half::f16::from_f32(scale_at(i + 1)).to_bits())
        .collect();

    // f64 reference: per q head, online softmax over the kv head its GQA
    // group maps to, score scale 1.0, exactly the kernel's order per cell.
    let mut reference = vec![0.0f64; heads * dim];
    for qh in 0..heads {
        let kvh = qh * kv_heads / heads;
        let mut run_m = f64::NEG_INFINITY;
        let mut run_l = 0.0;
        let mut acc = vec![0.0f64; dim];
        // The split maxima the device partials will hold: one max per split
        // of the plan this test launches, tracked in the reference's own
        // score arithmetic so the precondition pins the data, not a guess.
        let chunk = cells.div_ceil(32);
        let mut split_max = vec![f64::NEG_INFINITY; 32];
        for cell in 0..cells {
            let plane = (kvh * cells + cell) * dim;
            let sbase = (kvh * cells + cell) * groups;
            let mut score = 0.0;
            for d in 0..dim {
                score += f64::from(q[qh * dim + d])
                    * (f64::from(k_codes[plane + d]) * f64::from(scale_at(sbase + d / 32)));
            }
            let sm = &mut split_max[cell / chunk];
            *sm = sm.max(score);
            let new_m = run_m.max(score);
            let p = (score - new_m).exp();
            let correction = if run_l == 0.0 {
                0.0
            } else {
                (run_m - new_m).exp()
            };
            run_l = run_l * correction + p;
            for d in 0..dim {
                let vscale = f64::from(scale_at(sbase + d / 32 + 1));
                acc[d] = acc[d] * correction + p * (f64::from(v_codes[plane + d]) * vscale);
            }
            run_m = new_m;
        }
        for d in 0..dim {
            reference[qh * dim + d] = acc[d] / run_l;
        }
        let hottest = split_max.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let coldest = split_max.iter().cloned().fold(f64::INFINITY, f64::min);
        assert!(
            hottest - coldest <= 20.0,
            "every split's max must be within 20 nats of the global max: \
             hottest {hottest} coldest {coldest}"
        );
    }

    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let q_bytes = heads * dim * 4;
    let code_bytes = kv_heads * cells * dim;
    let scale_len = kv_heads * cells * groups * 2;
    let q_dev = testkit::upload_f32(&ctx, &q);
    let k_dev = testkit::upload_bytes(&ctx, &k_codes.iter().map(|c| *c as u8).collect::<Vec<_>>());
    let v_dev = testkit::upload_bytes(&ctx, &v_codes.iter().map(|c| *c as u8).collect::<Vec<_>>());
    let ks_dev = testkit::upload_bytes(
        &ctx,
        &k_scales
            .iter()
            .flat_map(|s| s.to_le_bytes())
            .collect::<Vec<_>>(),
    );
    let vs_dev = testkit::upload_bytes(
        &ctx,
        &v_scales
            .iter()
            .flat_map(|s| s.to_le_bytes())
            .collect::<Vec<_>>(),
    );
    let len = testkit::upload_bytes(&ctx, &(cells as i32).to_le_bytes());
    let part_floats = heads * 32 * (2 + dim);
    let part = testkit::upload_filled(&ctx, 0, part_floats * 4);
    let out = testkit::upload_filled(&ctx, 0xff, q_bytes);
    let launch = |split: Option<AttnDecodeGemmaSplit<'_>>| {
        ctx.attn_decode_gemma(
            &stream,
            AttnDecodeGemmaCall {
                out: out.range(0, q_bytes).expect("out"),
                status: None,
                q: q_dev.range(0, q_bytes).expect("q"),
                k_codes: k_dev.range(0, code_bytes).expect("k"),
                k_scales: ks_dev.range(0, scale_len).expect("ks"),
                v_codes: v_dev.range(0, code_bytes).expect("v"),
                v_scales: vs_dev.range(0, scale_len).expect("vs"),
                kv_len: len.range(0, 4).expect("len"),
                m: 1,
                n_q_head: heads,
                n_kv_head: kv_heads,
                head_dim: dim,
                cells,
                window: 0,
                scale: 1.0,
                split,
                ring: 0,
            },
        )
    };
    launch(None).expect("single-pass attention");
    stream.synchronize().expect("single-pass completion");
    let single = testkit::floats(&testkit::read_bytes(&ctx, &out, q_bytes));
    let check = |label: &str, got: &[f32]| {
        for (index, actual) in got.iter().enumerate() {
            let want = reference[index];
            assert!(
                (f64::from(*actual) - want).abs() <= 3e-3 + 3e-3 * want.abs(),
                "{label} element {index}: {actual} versus {want}"
            );
        }
    };
    check("single-pass", &single);
    // A separate output buffer, still 0xff-filled, so a stale single-pass
    // result cannot satisfy the split assertion.
    let out2 = testkit::upload_filled(&ctx, 0xff, q_bytes);
    let launch2 = |split: Option<AttnDecodeGemmaSplit<'_>>| {
        ctx.attn_decode_gemma(
            &stream,
            AttnDecodeGemmaCall {
                out: out2.range(0, q_bytes).expect("out2"),
                status: None,
                q: q_dev.range(0, q_bytes).expect("q"),
                k_codes: k_dev.range(0, code_bytes).expect("k"),
                k_scales: ks_dev.range(0, scale_len).expect("ks"),
                v_codes: v_dev.range(0, code_bytes).expect("v"),
                v_scales: vs_dev.range(0, scale_len).expect("vs"),
                kv_len: len.range(0, 4).expect("len"),
                m: 1,
                n_q_head: heads,
                n_kv_head: kv_heads,
                head_dim: dim,
                cells,
                window: 0,
                scale: 1.0,
                split,
                ring: 0,
            },
        )
    };
    launch2(Some(AttnDecodeGemmaSplit {
        part: part.range(0, part_floats * 4).expect("part"),
        max_splits: 32,
        min_keys: 128,
    }))
    .expect("split attention");
    stream.synchronize().expect("split completion");
    let split_out = testkit::floats(&testkit::read_bytes(&ctx, &out2, q_bytes));
    check("split", &split_out);
    // Every active split must carry weight in f32 on this data: with a
    // spread that underflowed expf(m_s - M) to zero for some split, the
    // reference comparison could pass while the merge ignored that split,
    // so pin the device's own partials: all 32 weights of all heads are
    // strictly positive.
    let part_back = testkit::floats(&testkit::read_bytes(&ctx, &part, part_floats * 4));
    for h in 0..heads {
        let at = |s: usize| (h * 32 + s) * 2;
        let top = (0..32).fold(f32::NEG_INFINITY, |m, s| m.max(part_back[at(s)]));
        for s in 0..32 {
            assert!(
                (part_back[at(s)] - top).exp() > 0.0,
                "head {h} split {s}: weight underflowed; the merge ignores this split"
            );
        }
    }
}

/// The split plan refusals: a zero or oversized split capacity, a floor
/// below 32 keys, and a short partial buffer all fail before any native
/// call.
#[test]
fn split_refuses_bad_plans() {
    if !testkit::device_present("attn_decode_gemma split refusals") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let (heads, kv_heads, dim, cells) = (8usize, 2usize, 512usize, 64usize);
    let q_bytes = heads * dim * 4;
    let code_bytes = kv_heads * cells * dim;
    let scale_len = kv_heads * cells * (dim / 32) * 2;
    let q_dev = testkit::upload_f32(&ctx, &vec![0.0f32; heads * dim]);
    let codes = testkit::upload_filled(&ctx, 0, code_bytes);
    let scales = testkit::upload_filled(&ctx, 0, scale_len);
    let len = testkit::upload_bytes(&ctx, &(cells as i32).to_le_bytes());
    let out = testkit::upload_filled(&ctx, 0xff, q_bytes);
    let part_floats = heads * 32 * (2 + dim);
    let part = testkit::upload_filled(&ctx, 0, part_floats * 4);
    let short = testkit::upload_filled(&ctx, 0, part_floats * 4 - 4);
    let launch = |part: Option<&crate::kernels::cuda::DeviceBuffer>, max: usize, min: usize| {
        ctx.attn_decode_gemma(
            &stream,
            AttnDecodeGemmaCall {
                out: out.range(0, q_bytes).expect("out"),
                status: None,
                q: q_dev.range(0, q_bytes).expect("q"),
                k_codes: codes.range(0, code_bytes).expect("k"),
                k_scales: scales.range(0, scale_len).expect("ks"),
                v_codes: codes.range(0, code_bytes).expect("v"),
                v_scales: scales.range(0, scale_len).expect("vs"),
                kv_len: len.range(0, 4).expect("len"),
                m: 1,
                n_q_head: heads,
                n_kv_head: kv_heads,
                head_dim: dim,
                cells,
                window: 0,
                scale: 1.0,
                split: part.map(|p| AttnDecodeGemmaSplit {
                    part: p.range(0, p.bytes()).expect("part"),
                    max_splits: max,
                    min_keys: min,
                }),
                ring: 0,
            },
        )
    };
    for (label, result) in [
        ("zero capacity", launch(Some(&part), 0, 128)),
        ("oversized capacity", launch(Some(&part), 129, 128)),
        ("floor below the minimum", launch(Some(&part), 32, 7)),
        ("short buffer", launch(Some(&short), 32, 128)),
    ] {
        let err = result.expect_err(label);
        assert!(
            matches!(err, crate::EngError::BadRequest(_)),
            "{label}: {err:?}"
        );
    }
}

/// Zero queries make every score equal, so attention is a plain mean of V. Each lane owns
/// its own head values, so a combine that loses the lane dimension returns one value per
/// 32-value group and fails here.
#[test]
fn combines_per_lane_values_across_warps() {
    if !testkit::device_present("attn per-lane combine") {
        return;
    }
    let (heads, dim, kv_heads, cells) = (16usize, 256usize, 8usize, 40usize);
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let q = testkit::upload_f32(&ctx, &vec![0.0f32; heads * dim]);
    let codes = testkit::upload_filled(&ctx, 0, kv_heads * cells * dim);
    let mut v_codes = vec![0u8; kv_heads * cells * dim];
    for h in 0..kv_heads {
        for c in 0..cells {
            for d in 0..dim {
                v_codes[(h * cells + c) * dim + d] = ((d + c) % 7) as u8;
            }
        }
    }
    let vq = testkit::upload_bytes(&ctx, &v_codes);
    let one = [0u8, 0x3c];
    let scales_len = kv_heads * cells * (dim / 32) * 2;
    let scale_bytes: Vec<u8> = one.iter().cycle().take(scales_len).copied().collect();
    let v_scales = testkit::upload_bytes(&ctx, &scale_bytes);
    let out = testkit::upload_filled(&ctx, 0xff, heads * dim * 4);
    let len = testkit::upload_bytes(&ctx, &(cells as i32).to_le_bytes());
    ctx.attn_decode_gemma(
        &stream,
        AttnDecodeGemmaCall {
            out: out.range(0, heads * dim * 4).expect("out"),
            status: None,
            q: q.range(0, heads * dim * 4).expect("q"),
            k_codes: codes.range(0, kv_heads * cells * dim).expect("k"),
            k_scales: v_scales.range(0, scales_len).expect("ks"),
            v_codes: vq.range(0, kv_heads * cells * dim).expect("v"),
            v_scales: v_scales.range(0, scales_len).expect("vs"),
            kv_len: len.range(0, 4).expect("len"),
            m: 1,
            n_q_head: heads,
            n_kv_head: kv_heads,
            head_dim: dim,
            cells,
            window: 0,
            scale: 1.0,
            split: None,
            ring: 0,
        },
    )
    .expect("attention");
    stream.synchronize().expect("drain");
    let got = testkit::floats(&testkit::read_bytes(&ctx, &out, heads * dim * 4));
    for qh in 0..heads {
        for d in 0..dim {
            let mut want = 0.0f64;
            for c in 0..cells {
                want += ((d + c) % 7) as f64;
            }
            want /= cells as f64;
            let value = f64::from(got[qh * dim + d]);
            assert!(
                (value - want).abs() < 1e-3,
                "head {qh} d {d}: {value} versus {want}"
            );
        }
    }
}

/// The CUDA-event bench of the decode attention at the four depths the card
/// window times (108, 3.3K, 11K, 30K cells): the served single-pass kernel
/// against the split path at the resident's own plan (32 splits max, 128-key
/// floor), printing the (chunk, active splits) pair the plan picks per depth
/// and the effective bytes scanned per call. Ignored like the gdn decode
/// bench: run by name in a release build inside an exclusive GPU window.
#[test]
#[ignore = "timing; run by name in a release build inside an exclusive GPU window"]
fn attn_decode_gemma_timing_at_four_depths() {
    use crate::kernels::cuda::native::resident::attn_decode_gemma::attn_decode_gemma_split_plan;
    const ROUNDS: usize = 51;
    const WARMUPS: usize = 5;
    if !testkit::device_present("attn_decode_gemma timing") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    // The served Bolt geometry, read from the spec rather than typed: 8 q
    // heads over 2 KV heads at the 512-dim global head, no window.
    let spec = crate::family::spec::NEMOTRON_BOLT_E4B;
    let heads = spec.heads_q as usize;
    let kv_heads = spec.heads_kv as usize;
    let dim = spec.global_head_dim as usize;
    let q_bytes = heads * dim * 4;
    for cells in [108_usize, 3_300, 11_000, 30_000] {
        let code_bytes = kv_heads * cells * dim;
        let scale_bytes = kv_heads * cells * (dim / 32) * 2;
        // Deterministic finite payload; the timing reads what the kernels read.
        let codes: Vec<u8> = (0..code_bytes).map(|index| (index % 251) as u8).collect();
        let scales: Vec<u8> = (0..scale_bytes)
            .map(|index| if index % 2 == 0 { 0x00 } else { 0x3C })
            .collect();
        let q: Vec<u8> = (0..q_bytes).map(|index| (index % 199) as u8).collect();
        let input = testkit::upload_bytes(&ctx, &q);
        let k_codes = testkit::upload_bytes(&ctx, &codes);
        let k_scales = testkit::upload_bytes(&ctx, &scales);
        let v_codes = testkit::upload_bytes(&ctx, &codes);
        let v_scales = testkit::upload_bytes(&ctx, &scales);
        let out = testkit::upload_filled(&ctx, 0xff, q_bytes);
        let len = testkit::upload_bytes(&ctx, &(cells as i32).to_le_bytes());
        let part_floats = heads * ATTN_SPLIT_MAX * (2 + dim);
        let part = testkit::upload_filled(&ctx, 0, part_floats * 4);
        let (chunk, n_active) =
            attn_decode_gemma_split_plan(cells, ATTN_SPLIT_MAX, ATTN_SPLIT_MIN_KEYS).expect("plan");
        let scanned = (2 * code_bytes + 2 * scale_bytes) as f64;
        let arms: [(&str, Option<AttnDecodeGemmaSplit>); 2] = [
            ("single-pass", None),
            (
                "split",
                Some(AttnDecodeGemmaSplit {
                    part: part.range(0, part_floats * 4).expect("part"),
                    max_splits: ATTN_SPLIT_MAX,
                    min_keys: ATTN_SPLIT_MIN_KEYS,
                }),
            ),
        ];
        for (label, split) in arms {
            let call = || {
                ctx.attn_decode_gemma(
                    &stream,
                    AttnDecodeGemmaCall {
                        out: out.range(0, q_bytes).expect("out"),
                        status: None,
                        q: input.range(0, q_bytes).expect("q"),
                        k_codes: k_codes.range(0, code_bytes).expect("k codes"),
                        k_scales: k_scales.range(0, scale_bytes).expect("k scales"),
                        v_codes: v_codes.range(0, code_bytes).expect("v codes"),
                        v_scales: v_scales.range(0, scale_bytes).expect("v scales"),
                        kv_len: len.range(0, 4).expect("length"),
                        m: 1,
                        n_q_head: heads,
                        n_kv_head: kv_heads,
                        head_dim: dim,
                        cells,
                        window: 0,
                        scale: 1.0,
                        split,
                        ring: 0,
                    },
                )
            };
            for _ in 0..WARMUPS {
                call().expect("warmup attention");
            }
            let begin = ctx.event_timing().expect("begin event");
            let end = ctx.event_timing().expect("end event");
            stream.synchronize().expect("drain before timing");
            stream.record_timing(&begin).expect("begin mark");
            for _ in 0..ROUNDS {
                call().expect("timed attention");
            }
            stream.record_timing(&end).expect("end mark");
            stream.synchronize().expect("drain after timing");
            let total_us = f64::from(begin.elapsed_ms_since(&end).expect("elapsed")) * 1e3;
            let per_call = total_us / ROUNDS as f64;
            println!(
                "cells={cells} {label}: chunk={chunk} active_splits={n_active} \
                 {per_call:.1} us/call, {:.1} GB/s effective",
                scanned / (per_call * 1e-6) / 1e9
            );
        }
    }
}
