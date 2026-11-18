//! The Q5_K dequantizing GEMM against the CPU oracle: `weights::dequantize` of the same
//! native blocks times an f32 matmul of the same activation, in the tolerance style of
//! `resident/matvec_q/tests.rs` (the tolerance scales with the sum of absolute terms, so a
//! random matrix cannot pass by accident).

use super::*;
use crate::kernels::cuda::testkit;
use crate::weights::GgufType;
use crate::weights::dequantize;

fn f16_bits(v: f32) -> u16 {
    testkit::f64_to_f16_bits(f64::from(v))
}

fn f16_of(bits: u16) -> f32 {
    testkit::f16_to_f32(bits)
}

/// Random-but-valid Q5_K blocks: sane d/dmin halves, random scales, qh and qs (the layout
/// decodes any bits; the oracle reads the same bytes).
fn build_weight(rows: usize, k: usize, seed: u64) -> Vec<u8> {
    let mut rng = testkit::Rng::new(seed);
    let mut bytes = vec![0u8; rows * (k / 256) * 176];
    for block in bytes.chunks_mut(176) {
        let d = 0.05 + rng.below(190) as f64 / 100.0;
        let dmin = 0.01 + rng.below(190) as f64 / 1000.0;
        block[0..2].copy_from_slice(&testkit::f64_to_f16_bits(d).to_le_bytes());
        block[2..4].copy_from_slice(&testkit::f64_to_f16_bits(dmin).to_le_bytes());
        for byte in &mut block[4..] {
            *byte = (rng.next_u64() & 0xFF) as u8;
        }
    }
    bytes
}

/// y_ref[r][n] = sum_k dequant(w)[r][k] * f16(x[k][n]), with the absolute-term sum for the
/// tolerance. The weight side is exact f32 dequant; the activation is the same f16 the kernel
/// reads, so the only rounding is the kernel's f16 staging of the weight tile and f32
/// accumulation order.
fn reference(w: &[u8], x: &[u16], rows: usize, k: usize, ncols: usize) -> (Vec<f64>, Vec<f64>) {
    let mut dequant_row = vec![0.0f32; k];
    let mut y = vec![0.0f64; rows * ncols];
    let mut abs = vec![0.0f64; rows * ncols];
    for r in 0..rows {
        let blocks = &w[r * (k / 256) * 176..];
        for b in 0..k / 256 {
            dequantize_into(
                &blocks[b * 176..(b + 1) * 176],
                &mut dequant_row[b * 256..(b + 1) * 256],
            );
        }
        for n in 0..ncols {
            let mut sum = 0.0f64;
            let mut terms = 0.0f64;
            for c in 0..k {
                let term = f64::from(dequant_row[c]) * f64::from(f16_of(x[c * ncols + n]));
                sum += term;
                terms += term.abs();
            }
            y[r * ncols + n] = sum;
            abs[r * ncols + n] = terms;
        }
    }
    (y, abs)
}

/// One block of the CPU oracle, spelled out from the same bytes (weights::dequantize on the
/// whole row would need the full tensor).
fn dequantize_into(block: &[u8], out: &mut [f32]) {
    let full = dequantize(GgufType::Q5K, block).expect("q5_k dequant");
    out.copy_from_slice(&full);
}

#[test]
fn gemm_dq_q5k_matches_the_cpu_dequant_matmul() {
    if !testkit::device_present("gemm_dq_q5k_matches_the_cpu_dequant_matmul") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    // Rows/ncols cross tile boundaries (64) with remainders; k spans several 256-blocks.
    let mut max_ratio = 0.0f64;
    for (rows, k, ncols, seed) in [
        (64usize, 512usize, 64usize, 0x5eed_0001u64),
        (100, 1024, 40, 0x5eed_0002),
        (13, 256, 7, 0x5eed_0003),
        (128, 768, 129, 0x5eed_0004),
    ] {
        let w_bytes = build_weight(rows, k, seed);
        let mut rng = testkit::Rng::new(seed ^ 0xa5a5);
        let x: Vec<u16> = (0..k * ncols)
            .map(|_| f16_bits((rng.normal() * 0.05) as f32))
            .collect();
        let (y_ref, abs) = reference(&w_bytes, &x, rows, k, ncols);
        let w = testkit::upload_bytes(&ctx, &w_bytes);
        let x_bytes: Vec<u8> = x.iter().flat_map(|b| b.to_le_bytes()).collect();
        let xb = testkit::upload_bytes(&ctx, &x_bytes);
        let y = testkit::upload_filled(&ctx, 0xAA, rows * ncols * 4);
        ctx.gemm_dq_q5k(
            &stream,
            GemmDqQ5kCall {
                w: w.range(0, w_bytes.len()).expect("w"),
                x: xb.range(0, x_bytes.len()).expect("x"),
                y: y.range(0, rows * ncols * 4).expect("y"),
                k,
                rows,
                ncols,
            },
        )
        .expect("gemm dq");
        stream.synchronize().expect("drain");
        let got = testkit::floats(&testkit::read_bytes(&ctx, &y, rows * ncols * 4));
        for idx in 0..rows * ncols {
            // Per-element bound: staging a weight into f16 rounds each term by at most
            // 2^-11 of itself, and the f32 accumulation over k adds at most k * 2^-24 of
            // the absolute sum; 4x is scheduling headroom. (~0.2% of the absolute sum.)
            let tolerance =
                4.0 * (2f64.powi(-11) + f64::from(k as u32) * 2f64.powi(-24)) * abs[idx].max(1e-30);
            let error = (f64::from(got[idx]) - y_ref[idx]).abs();
            let ratio = error / tolerance;
            max_ratio = max_ratio.max(ratio);
            assert!(
                error <= tolerance,
                "rows {rows} k {k} ncols {ncols} idx {idx}: error {error} over tol {tolerance}"
            );
        }
    }
    println!("gemm dq q5k: max tolerance ratio {max_ratio:.3}");
}

/// One wrong group multiplier in one block must move an output past the tolerance: this
/// pins that the bound above is tight enough to catch a single sub-block scale error.
#[test]
fn gemm_dq_q5k_catches_one_wrong_group_scale() {
    if !testkit::device_present("gemm_dq_q5k_catches_one_wrong_group_scale") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let (rows, k, ncols) = (100usize, 1024usize, 40usize);
    let w_bytes = build_weight(rows, k, 0x5eed_0002);
    let mut rng = testkit::Rng::new(0x5eed_0002 ^ 0xa5a5);
    let x: Vec<u16> = (0..k * ncols)
        .map(|_| f16_bits((rng.normal() * 0.05) as f32))
        .collect();
    // The reference reads the original bytes; the device copy flips one bit (0x20, half
    // the 6-bit range) of the group-0 multiplier of row 0, block 0.
    let (y_ref, abs) = reference(&w_bytes, &x, rows, k, ncols);
    let mut w_bad = w_bytes.clone();
    w_bad[4] ^= 0x20;
    let w = testkit::upload_bytes(&ctx, &w_bad);
    let x_bytes: Vec<u8> = x.iter().flat_map(|b| b.to_le_bytes()).collect();
    let xb = testkit::upload_bytes(&ctx, &x_bytes);
    let y = testkit::upload_filled(&ctx, 0xAA, rows * ncols * 4);
    ctx.gemm_dq_q5k(
        &stream,
        GemmDqQ5kCall {
            w: w.range(0, w_bad.len()).expect("w"),
            x: xb.range(0, x_bytes.len()).expect("x"),
            y: y.range(0, rows * ncols * 4).expect("y"),
            k,
            rows,
            ncols,
        },
    )
    .expect("gemm dq");
    stream.synchronize().expect("drain");
    let got = testkit::floats(&testkit::read_bytes(&ctx, &y, rows * ncols * 4));
    let factor = 4.0 * (2f64.powi(-11) + f64::from(k as u32) * 2f64.powi(-24));
    let caught = (0..rows * ncols)
        .any(|idx| (f64::from(got[idx]) - y_ref[idx]).abs() > factor * abs[idx].max(1e-30));
    assert!(
        caught,
        "a single wrong group scale must exceed the tolerance"
    );
}

#[test]
fn gemm_dq_q5k_is_deterministic_and_refuses_bad_shapes() {
    if !testkit::device_present("gemm_dq_q5k_is_deterministic_and_refuses_bad_shapes") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let (rows, k, ncols) = (64usize, 512usize, 64usize);
    let w_bytes = build_weight(rows, k, 0x5eed_0011);
    let x: Vec<u8> = vec![0x3c; k * ncols * 2]; // 1.0 in f16, repeated: a stable product
    let w = testkit::upload_bytes(&ctx, &w_bytes);
    let xb = testkit::upload_bytes(&ctx, &x);
    let run = |y: &crate::kernels::cuda::context::DeviceBuffer| {
        ctx.gemm_dq_q5k(
            &stream,
            GemmDqQ5kCall {
                w: w.range(0, w_bytes.len()).expect("w"),
                x: xb.range(0, x.len()).expect("x"),
                y: y.range(0, rows * ncols * 4).expect("y"),
                k,
                rows,
                ncols,
            },
        )
        .expect("gemm dq");
        stream.synchronize().expect("drain");
        testkit::read_bytes(&ctx, y, rows * ncols * 4)
    };
    let y1 = testkit::upload_filled(&ctx, 0xAA, rows * ncols * 4);
    let y2 = testkit::upload_filled(&ctx, 0xAA, rows * ncols * 4);
    assert_eq!(run(&y1), run(&y2), "the two runs must be byte-identical");

    for (k_bad, rows_bad, ncols_bad, needle) in [
        (100usize, 64usize, 64usize, "multiple of 256"),
        (512, 0, 64, "above zero"),
        (512, 64, 0, "above zero"),
    ] {
        let result = ctx.gemm_dq_q5k(
            &stream,
            GemmDqQ5kCall {
                w: w.range(0, w_bytes.len()).expect("w"),
                x: xb.range(0, x.len()).expect("x"),
                y: y1.range(0, rows * ncols * 4).expect("y"),
                k: k_bad,
                rows: rows_bad,
                ncols: ncols_bad,
            },
        );
        let Err(error) = result else {
            panic!("k {k_bad} rows {rows_bad} ncols {ncols_bad}: expected a refusal");
        };
        assert!(
            matches!(&error, EngError::BadRequest(message) if message.contains(needle)),
            "k {k_bad}: expected BadRequest containing {needle:?}, got {error:?}"
        );
    }
}

/// Kernel-only timing rig for the load-path follow-ups: builds the buffers once and
/// times [queue + sync] of the GEMM alone on one prefill-shaped problem
/// (rows 4096, k 4096, ncols 512 => 17.2 GFLOP per launch). `--ignored` only.
#[test]
#[ignore = "profile timing rig"]
fn gemm_dq_q5k_profile_timing() {
    if !testkit::device_present("gemm_dq_q5k_profile_timing") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let rows = 4096usize;
    let k = 4096usize;
    let ncols = 512usize;
    let w_bytes = build_weight(rows, k, 0x5eed_beef);
    let mut rng = testkit::Rng::new(0x5eed_beef);
    let x: Vec<u16> = (0..k * ncols)
        .map(|_| f16_bits((rng.normal() * 0.05) as f32))
        .collect();
    let x_bytes: Vec<u8> = x.iter().flat_map(|b| b.to_le_bytes()).collect();
    let w = testkit::upload_bytes(&ctx, &w_bytes);
    let xb = testkit::upload_bytes(&ctx, &x_bytes);
    let y = testkit::upload_filled(&ctx, 0, rows * ncols * 4);
    let call = GemmDqQ5kCall {
        w: w.range(0, w_bytes.len()).expect("w"),
        x: xb.range(0, x_bytes.len()).expect("x"),
        y: y.range(0, rows * ncols * 4).expect("y"),
        k,
        rows,
        ncols,
    };
    for _ in 0..3 {
        ctx.gemm_dq_q5k(&stream, call.clone()).expect("warmup");
    }
    stream.synchronize().expect("drain");
    let mut samples = Vec::new();
    for _ in 0..20 {
        let started = std::time::Instant::now();
        ctx.gemm_dq_q5k(&stream, call.clone()).expect("gemm");
        stream.synchronize().expect("drain");
        samples.push(started.elapsed().as_secs_f64() * 1e3);
    }
    samples.sort_by(|a, b| a.total_cmp(b));
    let median = samples[samples.len() / 2];
    let gflops = 2.0 * rows as f64 * k as f64 * ncols as f64 / (median * 1e-3) / 1e9;
    println!(
        "PROFILE-TIMING gemm_dq q5k rows {rows} k {k} ncols {ncols}: median {median:.3} ms, {gflops:.0} GFLOP/s, spread [{:.3} .. {:.3}]",
        samples[0],
        samples[samples.len() - 1]
    );
}
