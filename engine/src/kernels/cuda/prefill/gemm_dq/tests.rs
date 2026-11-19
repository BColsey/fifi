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
fn gemm_dq_q5k_token_rows_match_the_row_major_layout() {
    if !testkit::device_present("gemm_dq_q5k_token_rows_match_the_row_major_layout") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    // The token-row strides change only the addressing: the accumulation order is
    // identical, so the outputs must be bitwise equal to the row-major launch.
    for (rows, k, ncols, seed) in [
        (64usize, 512usize, 64usize, 0x7eed_0001u64),
        (100, 1024, 40, 0x7eed_0002),
        (13, 256, 7, 0x7eed_0003),
    ] {
        let w_bytes = build_weight(rows, k, seed);
        let mut rng = testkit::Rng::new(seed ^ 0x5a5a);
        let x: Vec<u16> = (0..k * ncols)
            .map(|_| f16_bits((rng.normal() * 0.05) as f32))
            .collect();
        let w = testkit::upload_bytes(&ctx, &w_bytes);
        let x_bytes: Vec<u8> = x.iter().flat_map(|b| b.to_le_bytes()).collect();
        let y = testkit::upload_filled(&ctx, 0, rows * ncols * 4);
        ctx.gemm_dq_q5k(
            &stream,
            GemmDqQ5kCall {
                w: w.range(0, w_bytes.len()).expect("w"),
                x: testkit::upload_bytes(&ctx, &x_bytes)
                    .range(0, x_bytes.len())
                    .expect("x"),
                y: y.range(0, rows * ncols * 4).expect("y"),
                k,
                rows,
                ncols,
                x_col_stride: 0,
                y_col_stride: 0,
                x_f32: false,
            },
        )
        .expect("row-major gemm");
        stream.synchronize().expect("drain");
        let row_major = testkit::read_bytes(&ctx, &y, rows * ncols * 4);

        // Token-major x: x_t[col][row_k]; token-major y: y_t[col][row].
        let mut x_t = vec![0u16; ncols * k];
        for c in 0..ncols {
            for r in 0..k {
                x_t[c * k + r] = x[r * ncols + c];
            }
        }
        let x_t_bytes: Vec<u8> = x_t.iter().flat_map(|b| b.to_le_bytes()).collect();
        let y_t = testkit::upload_filled(&ctx, 0, ncols * rows * 4);
        ctx.gemm_dq_q5k(
            &stream,
            GemmDqQ5kCall {
                w: w.range(0, w_bytes.len()).expect("w"),
                x: testkit::upload_bytes(&ctx, &x_t_bytes)
                    .range(0, x_t_bytes.len())
                    .expect("x"),
                y: y_t.range(0, ncols * rows * 4).expect("y"),
                k,
                rows,
                ncols,
                x_col_stride: k,
                y_col_stride: rows,
                x_f32: false,
            },
        )
        .expect("token-row gemm");
        stream.synchronize().expect("drain");
        let mut token_rows = testkit::read_bytes(&ctx, &y_t, ncols * rows * 4);

        // Both layouts write the same logical rectangle; repack y_t to [rows][ncols].
        let mut repacked = vec![0u8; rows * ncols * 4];
        for row in 0..rows {
            for col in 0..ncols {
                let at = ((col * rows + row) * 4, (row * ncols + col) * 4);
                repacked[at.1..at.1 + 4].copy_from_slice(&token_rows[at.0..at.0 + 4]);
            }
        }
        let _ = &mut token_rows;
        assert_eq!(
            repacked, row_major,
            "rows {rows} k {k} ncols {ncols}: token-row strides must be layout-only"
        );

        // f32 activations: every f16 value converts to f32 and back exactly, so a buffer
        // of the f32 images under x_f32 must reproduce the same bits.
        let x_f32s: Vec<f32> = x.iter().map(|b| f16_of(*b)).collect();
        let x_f32_bytes: Vec<u8> = x_f32s.iter().flat_map(|v| v.to_le_bytes()).collect();
        let y3 = testkit::upload_filled(&ctx, 0, rows * ncols * 4);
        ctx.gemm_dq_q5k(
            &stream,
            GemmDqQ5kCall {
                w: w.range(0, w_bytes.len()).expect("w"),
                x: testkit::upload_bytes(&ctx, &x_f32_bytes)
                    .range(0, x_f32_bytes.len())
                    .expect("x"),
                y: y3.range(0, rows * ncols * 4).expect("y"),
                k,
                rows,
                ncols,
                x_col_stride: 0,
                y_col_stride: 0,
                x_f32: true,
            },
        )
        .expect("f32 gemm");
        stream.synchronize().expect("drain");
        assert_eq!(
            testkit::read_bytes(&ctx, &y3, rows * ncols * 4),
            row_major,
            "rows {rows} k {k} ncols {ncols}: f32 activations must round-trip the f16 bits"
        );
    }
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
                x_col_stride: 0,
                y_col_stride: 0,
                x_f32: false,
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
            x_col_stride: 0,
            y_col_stride: 0,
            x_f32: false,
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
                x_col_stride: 0,
                y_col_stride: 0,
                x_f32: false,
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
                x_col_stride: 0,
                y_col_stride: 0,
                x_f32: false,
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
        x_col_stride: 0,
        y_col_stride: 0,
        x_f32: false,
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

/// The Q4_0 decoder of the same GEMM against the CPU oracle: the same GGUF
/// blocks through `weights::dequantize`, whose Q4_0 arm is written from the
/// ggml reference layout (low nibbles of `qs[0..16]` are values 0..16,
/// high nibbles values 16..32).
fn build_weight_q40(rows: usize, k: usize, seed: u64) -> Vec<u8> {
    let mut rng = testkit::Rng::new(seed);
    let mut bytes = vec![0u8; rows * (k / 32) * 18];
    for block in bytes.chunks_mut(18) {
        let d = 0.05 + rng.below(190) as f64 / 100.0;
        block[0..2].copy_from_slice(&testkit::f64_to_f16_bits(d).to_le_bytes());
        for byte in &mut block[2..] {
            *byte = (rng.next_u64() & 0xFF) as u8;
        }
    }
    bytes
}

fn reference_q40(w: &[u8], x: &[u16], rows: usize, k: usize, ncols: usize) -> (Vec<f64>, Vec<f64>) {
    let mut dequant_row = vec![0.0f32; k];
    let mut y = vec![0.0f64; rows * ncols];
    let mut abs = vec![0.0f64; rows * ncols];
    for r in 0..rows {
        let blocks = &w[r * (k / 32) * 18..];
        for b in 0..k / 32 {
            let block = &blocks[b * 18..(b + 1) * 18];
            dequant_row[b * 32..(b + 1) * 32]
                .copy_from_slice(&dequantize(GgufType::Q40, block).expect("q4_0 dequant"));
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

#[test]
fn gemm_dq_q40_matches_the_cpu_dequant_matmul() {
    if !testkit::device_present("gemm_dq_q40_matches_the_cpu_dequant_matmul") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let mut max_ratio = 0.0f64;
    // k stays a multiple of 64 (the chunk); rows/ncols cross the 64 tile with remainders.
    for (rows, k, ncols, seed) in [
        (64usize, 512usize, 64usize, 0x5eed_0011u64),
        (100, 1024, 40, 0x5eed_0012),
        (13, 256, 7, 0x5eed_0013),
        (128, 768, 129, 0x5eed_0014),
    ] {
        let w_bytes = build_weight_q40(rows, k, seed);
        let mut rng = testkit::Rng::new(seed ^ 0xa5a5);
        let x: Vec<u16> = (0..k * ncols)
            .map(|_| f16_bits((rng.normal() * 0.05) as f32))
            .collect();
        let (y_ref, abs) = reference_q40(&w_bytes, &x, rows, k, ncols);
        let w = testkit::upload_bytes(&ctx, &w_bytes);
        let x_bytes: Vec<u8> = x.iter().flat_map(|b| b.to_le_bytes()).collect();
        let xb = testkit::upload_bytes(&ctx, &x_bytes);
        let y = testkit::upload_filled(&ctx, 0xAA, rows * ncols * 4);
        ctx.gemm_dq_q40(
            &stream,
            GemmDqQ40Call {
                w: w.range(0, w_bytes.len()).expect("w"),
                x: xb.range(0, x_bytes.len()).expect("x"),
                y: y.range(0, rows * ncols * 4).expect("y"),
                k,
                rows,
                ncols,
                x_col_stride: 0,
                y_col_stride: 0,
                x_f32: false,
            },
        )
        .expect("gemm dq");
        stream.synchronize().expect("drain");
        let got = testkit::floats(&testkit::read_bytes(&ctx, &y, rows * ncols * 4));
        for idx in 0..rows * ncols {
            // The bound mirrors the Q5_K test: f16 weight staging rounds each term by at
            // most 2^-11, the f32 accumulation over k adds at most k * 2^-24 of the
            // absolute sum, 4x headroom.
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
    println!("gemm dq q40: max tolerance ratio {max_ratio:.3}");
}

/// One flipped code nibble in one block must move an output past the tolerance: the value
/// shifts by 8 * d * x, which the absolute-term bound cannot hide.
#[test]
fn gemm_dq_q40_catches_one_flipped_code() {
    if !testkit::device_present("gemm_dq_q40_catches_one_flipped_code") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let rows = 64usize;
    let k = 512usize;
    let ncols = 64usize;
    let mut w_bytes = build_weight_q40(rows, k, 0x5eed_0021);
    let mut rng = testkit::Rng::new(0x5eed_0021);
    let x: Vec<u16> = (0..k * ncols)
        .map(|_| f16_bits((rng.normal() * 0.05) as f32))
        .collect();
    let (y_ref, abs) = reference_q40(&w_bytes, &x, rows, k, ncols);
    // Row 3, block 1, the high nibble of its first code byte: ^0xF0 flips the code of the
    // block's value 16 by up to 15, a swing of 15 * d * x per output that the bound cannot
    // hide.
    w_bytes[(3 * (k / 32) + 1) * 18 + 2] ^= 0xF0;
    let w = testkit::upload_bytes(&ctx, &w_bytes);
    let x_bytes: Vec<u8> = x.iter().flat_map(|b| b.to_le_bytes()).collect();
    let xb = testkit::upload_bytes(&ctx, &x_bytes);
    let y = testkit::upload_filled(&ctx, 0, rows * ncols * 4);
    ctx.gemm_dq_q40(
        &stream,
        GemmDqQ40Call {
            w: w.range(0, w_bytes.len()).expect("w"),
            x: xb.range(0, x_bytes.len()).expect("x"),
            y: y.range(0, rows * ncols * 4).expect("y"),
            k,
            rows,
            ncols,
            x_col_stride: 0,
            y_col_stride: 0,
            x_f32: false,
        },
    )
    .expect("gemm dq");
    stream.synchronize().expect("drain");
    let got = testkit::floats(&testkit::read_bytes(&ctx, &y, rows * ncols * 4));
    let tolerance = 4.0 * (2f64.powi(-11) + f64::from(k as u32) * 2f64.powi(-24));
    let moved = (0..rows * ncols)
        .filter(|idx| {
            ((f64::from(got[*idx]) - y_ref[*idx]).abs()) > tolerance * abs[*idx].max(1e-30)
        })
        .count();
    assert!(moved > 0, "a flipped code nibble must move some output");
}

#[test]
fn gemm_dq_q40_is_deterministic_and_refuses_bad_shapes() {
    if !testkit::device_present("gemm_dq_q40_is_deterministic_and_refuses_bad_shapes") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let rows = 70usize;
    let k = 256usize;
    let ncols = 70usize;
    let w_bytes = build_weight_q40(rows, k, 0x5eed_0031);
    let x_bytes = vec![0x11u8; k * ncols * 2];
    let w = testkit::upload_bytes(&ctx, &w_bytes);
    let xb = testkit::upload_bytes(&ctx, &x_bytes);
    let y1 = testkit::upload_filled(&ctx, 0, rows * ncols * 4);
    let y2 = testkit::upload_filled(&ctx, 0, rows * ncols * 4);
    // The sizes come from the enclosing buffers, so the closure takes only the ranges.
    fn call<'b>(
        w: &'b crate::kernels::cuda::DeviceBuffer,
        xb: &'b crate::kernels::cuda::DeviceBuffer,
        y: &'b crate::kernels::cuda::DeviceBuffer,
        shape: (usize, usize, usize),
        w_len: usize,
        x_len: usize,
    ) -> GemmDqQ40Call<'b> {
        let (k, rows, ncols) = shape;
        GemmDqQ40Call {
            w: w.range(0, w_len).expect("w"),
            x: xb.range(0, x_len).expect("x"),
            y: y.range(0, rows * ncols * 4).expect("y"),
            k,
            rows,
            ncols,
            x_col_stride: 0,
            y_col_stride: 0,
            x_f32: false,
        }
    }
    ctx.gemm_dq_q40(
        &stream,
        call(&w, &xb, &y1, (k, rows, ncols), w_bytes.len(), x_bytes.len()),
    )
    .expect("first");
    ctx.gemm_dq_q40(
        &stream,
        call(&w, &xb, &y2, (k, rows, ncols), w_bytes.len(), x_bytes.len()),
    )
    .expect("second");
    stream.synchronize().expect("drain");
    assert_eq!(
        testkit::read_bytes(&ctx, &y1, rows * ncols * 4),
        testkit::read_bytes(&ctx, &y2, rows * ncols * 4),
        "the same call twice must write identical bytes"
    );
    // k not a multiple of 64, and a y overlapping x, are refused before any launch.
    let bad_k = GemmDqQ40Call {
        k: 96,
        ..call(&w, &xb, &y1, (k, rows, ncols), w_bytes.len(), x_bytes.len())
    };
    assert!(ctx.gemm_dq_q40(&stream, bad_k).is_err());
    let overlap = GemmDqQ40Call {
        y: xb.range(0, rows * ncols * 4).expect("y"),
        ..call(&w, &xb, &y1, (k, rows, ncols), w_bytes.len(), x_bytes.len())
    };
    assert!(ctx.gemm_dq_q40(&stream, overlap).is_err());
    let short_w = GemmDqQ40Call {
        w: w.range(0, w_bytes.len() - 18).expect("w short"),
        ..call(&w, &xb, &y1, (k, rows, ncols), w_bytes.len(), x_bytes.len())
    };
    assert!(ctx.gemm_dq_q40(&stream, short_w).is_err());
}

/// The Q3_K reference decodes the same GGUF blocks through `weights::dequantize`'s
/// Q3_K arm, written from the ggml reference layout of the vendored `block_q3_K`
/// (packed 6-bit group scales, low-2-bit codes, high-bit plane, f16 super scale).
fn build_weight_q3k(rows: usize, k: usize, seed: u64) -> Vec<u8> {
    let mut rng = testkit::Rng::new(seed);
    let mut bytes = vec![0u8; rows * (k / 256) * 110];
    for block in bytes.chunks_mut(110) {
        let d = 0.05 + rng.below(190) as f64 / 100.0;
        block[108..110].copy_from_slice(&testkit::f64_to_f16_bits(d).to_le_bytes());
        for byte in &mut block[..108] {
            *byte = (rng.next_u64() & 0xFF) as u8;
        }
    }
    bytes
}

fn reference_q3k(w: &[u8], x: &[u16], rows: usize, k: usize, ncols: usize) -> (Vec<f64>, Vec<f64>) {
    let mut dequant_row = vec![0.0f32; k];
    let mut y = vec![0.0f64; rows * ncols];
    let mut abs = vec![0.0f64; rows * ncols];
    for r in 0..rows {
        let blocks = &w[r * (k / 256) * 110..];
        for b in 0..k / 256 {
            let block = &blocks[b * 110..(b + 1) * 110];
            dequant_row[b * 256..(b + 1) * 256]
                .copy_from_slice(&dequantize(GgufType::Q3K, block).expect("q3_k dequant"));
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

#[test]
fn gemm_dq_q3k_matches_the_cpu_dequant_matmul() {
    if !testkit::device_present("gemm_dq_q3k_matches_the_cpu_dequant_matmul") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let mut max_ratio = 0.0f64;
    // rows/ncols cross the 64 tile with remainders; k spans several 256-value blocks.
    for (rows, k, ncols, seed) in [
        (64usize, 512usize, 64usize, 0x5eed_0031u64),
        (100, 1024, 40, 0x5eed_0032),
        (13, 256, 7, 0x5eed_0033),
        (128, 768, 129, 0x5eed_0034),
    ] {
        let w_bytes = build_weight_q3k(rows, k, seed);
        let mut rng = testkit::Rng::new(seed ^ 0xa5a5);
        let x: Vec<u16> = (0..k * ncols)
            .map(|_| f16_bits((rng.normal() * 0.05) as f32))
            .collect();
        let (y_ref, abs) = reference_q3k(&w_bytes, &x, rows, k, ncols);
        let w = testkit::upload_bytes(&ctx, &w_bytes);
        let x_bytes: Vec<u8> = x.iter().flat_map(|b| b.to_le_bytes()).collect();
        let xb = testkit::upload_bytes(&ctx, &x_bytes);
        let y = testkit::upload_filled(&ctx, 0xAA, rows * ncols * 4);
        ctx.gemm_dq_q3k(
            &stream,
            GemmDqQ3kCall {
                w: w.range(0, w_bytes.len()).expect("w"),
                x: xb.range(0, x_bytes.len()).expect("x"),
                y: y.range(0, rows * ncols * 4).expect("y"),
                k,
                rows,
                ncols,
                x_col_stride: 0,
                y_col_stride: 0,
                x_f32: false,
            },
        )
        .expect("gemm dq");
        stream.synchronize().expect("drain");
        let got = testkit::floats(&testkit::read_bytes(&ctx, &y, rows * ncols * 4));
        for idx in 0..rows * ncols {
            // The bound mirrors the Q5_K/Q4_0 tests: f16 weight staging rounds each term
            // by at most 2^-11, the f32 accumulation over k adds at most k * 2^-24 of the
            // absolute sum, 4x headroom.
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
    println!("gemm dq q3k: max tolerance ratio {max_ratio:.3}");
}

fn build_weight_iq4xs(rows: usize, k: usize, seed: u64) -> Vec<u8> {
    let mut rng = testkit::Rng::new(seed);
    let mut bytes = vec![0u8; rows * (k / 256) * 136];
    for block in bytes.chunks_mut(136) {
        let d = 0.05 + rng.below(190) as f64 / 100.0;
        block[0..2].copy_from_slice(&testkit::f64_to_f16_bits(d).to_le_bytes());
        for byte in &mut block[2..] {
            *byte = (rng.next_u64() & 0xFF) as u8;
        }
    }
    bytes
}

fn reference_iq4xs(
    w: &[u8],
    x: &[u16],
    rows: usize,
    k: usize,
    ncols: usize,
) -> (Vec<f64>, Vec<f64>) {
    let mut dequant_row = vec![0.0f32; k];
    let mut y = vec![0.0f64; rows * ncols];
    let mut abs = vec![0.0f64; rows * ncols];
    for r in 0..rows {
        let blocks = &w[r * (k / 256) * 136..];
        for b in 0..k / 256 {
            let block = &blocks[b * 136..(b + 1) * 136];
            dequant_row[b * 256..(b + 1) * 256]
                .copy_from_slice(&dequantize(GgufType::Iq4Xs, block).expect("iq4_xs dequant"));
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

#[test]
fn gemm_dq_iq4xs_probe_crafted() {
    if !testkit::device_present("gemm_dq_iq4xs_probe_crafted") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let (rows, k, ncols) = (2usize, 256usize, 3usize);
    let xval = |c: usize, col: usize| ((c % 7) as f32 - 3.0) * 0.25 + col as f32;
    let x: Vec<u16> = (0..k * ncols)
        .map(|i| f16_bits(xval(i / ncols, i % ncols)))
        .collect();
    // Sub-block column sums: the hi scale lanes cover only sub-blocks 8..15,
    // so a craft case is a per-sub-block weight, never one uniform weight.
    let sub_sum = |sb: usize, col: usize| -> f64 {
        (sb * 32..sb * 32 + 32)
            .map(|c| f64::from(xval(c, col)))
            .sum()
    };
    let sub_abs = |sb: usize, col: usize| -> f64 {
        (sb * 32..sb * 32 + 32)
            .map(|c| f64::from(xval(c, col)).abs())
            .sum()
    };
    // craft: d=1, all scales 0, qs 0 -> every weight = (0-32)*1*kv[0] = +4064
    let craft = |scales_l: u8, scales_h: u16, qs: u8| -> Vec<u8> {
        let mut w = vec![0u8; rows * (k / 256) * 136];
        for b in w.chunks_mut(136) {
            b[0..2].copy_from_slice(&0x3C00u16.to_le_bytes()); // d = 1.0
            b[2..4].copy_from_slice(&scales_h.to_le_bytes());
            for byte in &mut b[4..8] {
                *byte = scales_l;
            }
            for q in &mut b[8..] {
                *q = qs;
            }
        }
        w
    };
    let run = |name: &str, w_bytes: &[u8], weight_of: &[f64; 16]| {
        let w = testkit::upload_bytes(&ctx, w_bytes);
        let xb = testkit::upload_bytes(
            &ctx,
            &x.iter().flat_map(|b| b.to_le_bytes()).collect::<Vec<u8>>(),
        );
        let y = testkit::upload_filled(&ctx, 0xAA, rows * ncols * 4);
        ctx.gemm_dq_iq4xs(
            &stream,
            GemmDqIq4xsCall {
                w: w.range(0, w_bytes.len()).expect("w"),
                x: xb.range(0, k * ncols * 2).expect("x"),
                y: y.range(0, rows * ncols * 4).expect("y"),
                k,
                rows,
                ncols,
                x_col_stride: 0,
                y_col_stride: 0,
                x_f32: false,
            },
        )
        .expect("gemm dq");
        stream.synchronize().expect("drain");
        let got = testkit::floats(&testkit::read_bytes(&ctx, &y, rows * ncols * 4));
        // Every weight inside a sub-block is identical by construction, so
        // column n must sit at sum(weight[sb] * sub_sum(sb, n)) inside the
        // oracle test's tolerance model (f16 staging plus f32 accumulation,
        // 4x headroom): a wrong table entry or scale moves an output past it
        // by orders of magnitude.
        for (i, value) in got.iter().enumerate() {
            let col = i % ncols;
            let mut want = 0.0f64;
            let mut abs_terms = 0.0f64;
            for (sb, weight) in weight_of.iter().enumerate().take(k / 32) {
                want += weight * sub_sum(sb, col);
                abs_terms += weight.abs() * sub_abs(sb, col);
            }
            let bound = 4.0 * (2f64.powi(-11) + f64::from(k as u32) * 2f64.powi(-24)) * abs_terms;
            assert!(
                (f64::from(*value) - want).abs() <= bound,
                "craft {name}: got[{i}] = {value}, want {want:.4}, bound {bound:.4}"
            );
        }
    };
    let uniform = |weight: f64| [weight; 16];
    // scales_l feeds the low 4 bits of sub-blocks 0..7's 6-bit scales.
    run("zero-scales", &craft(0, 0, 0), &uniform(4064.0));
    run("lo1-scales", &craft(0x11, 0, 0), &uniform(-31.0 * -127.0));
    // scales_h lanes 01 lift every group scale from 0 to 16.
    run(
        "hi1-scales",
        &craft(0, 0x5555, 0),
        &uniform((16.0 - 32.0) * -127.0),
    );
    run("qs1", &craft(0, 0, 0x11), &uniform(-32.0 * -104.0));
    run("qs2", &craft(0, 0, 0x22), &uniform(-32.0 * -83.0));
    run("qs4", &craft(0, 0, 0x44), &uniform(-32.0 * -49.0));
    run("qs8", &craft(0, 0, 0x88), &uniform(-32.0 * 1.0));
    run("qsf", &craft(0, 0, 0xff), &uniform(-32.0 * 113.0));
    run("lo2-scales", &craft(0x22, 0, 0), &uniform(-30.0 * -127.0));
}

#[test]
fn gemm_dq_iq4xs_matches_the_cpu_dequant_matmul() {
    if !testkit::device_present("gemm_dq_iq4xs_matches_the_cpu_dequant_matmul") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let mut max_ratio = 0.0f64;
    // rows/ncols cross the 64 tile with remainders; k spans several 256-value super-blocks.
    for (rows, k, ncols, seed) in [
        (64usize, 512usize, 64usize, 0x5eed_0041u64),
        (100, 1024, 40, 0x5eed_0042),
        (13, 256, 7, 0x5eed_0043),
        (128, 768, 129, 0x5eed_0044),
    ] {
        let w_bytes = build_weight_iq4xs(rows, k, seed);
        let mut rng = testkit::Rng::new(seed ^ 0xa5a5);
        let x: Vec<u16> = (0..k * ncols)
            .map(|_| f16_bits((rng.normal() * 0.05) as f32))
            .collect();
        let (y_ref, abs) = reference_iq4xs(&w_bytes, &x, rows, k, ncols);
        let w = testkit::upload_bytes(&ctx, &w_bytes);
        let x_bytes: Vec<u8> = x.iter().flat_map(|b| b.to_le_bytes()).collect();
        let xb = testkit::upload_bytes(&ctx, &x_bytes);
        let y = testkit::upload_filled(&ctx, 0xAA, rows * ncols * 4);
        ctx.gemm_dq_iq4xs(
            &stream,
            GemmDqIq4xsCall {
                w: w.range(0, w_bytes.len()).expect("w"),
                x: xb.range(0, x_bytes.len()).expect("x"),
                y: y.range(0, rows * ncols * 4).expect("y"),
                k,
                rows,
                ncols,
                x_col_stride: 0,
                y_col_stride: 0,
                x_f32: false,
            },
        )
        .expect("gemm dq");
        stream.synchronize().expect("drain");
        let got = testkit::floats(&testkit::read_bytes(&ctx, &y, rows * ncols * 4));
        let mut shape_ratio = 0.0f64;
        for idx in 0..rows * ncols {
            // The bound mirrors the Q3_K/Q5_K tests: f16 weight staging rounds each term
            // by at most 2^-11, the f32 accumulation over k adds at most k * 2^-24 of the
            // absolute sum, 4x headroom.
            let tolerance =
                4.0 * (2f64.powi(-11) + f64::from(k as u32) * 2f64.powi(-24)) * abs[idx].max(1e-30);
            let error = (f64::from(got[idx]) - y_ref[idx]).abs();
            let ratio = error / tolerance;
            max_ratio = max_ratio.max(ratio);
            shape_ratio = shape_ratio.max(ratio);
        }
        eprintln!("probe shape rows {rows} k {k} ncols {ncols}: max ratio {shape_ratio:.3}");
    }
    println!("gemm dq iq4xs: max tolerance ratio {max_ratio:.3}");
}

/// One wrong group scale in one block must move an output past the tolerance: this pins
/// that the bound above is tight enough to catch a single sub-block scale error.
#[test]
fn gemm_dq_q3k_catches_one_wrong_group_scale() {
    if !testkit::device_present("gemm_dq_q3k_catches_one_wrong_group_scale") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let (rows, k, ncols) = (100usize, 1024usize, 40usize);
    let w_bytes = build_weight_q3k(rows, k, 0x5eed_0041);
    let mut rng = testkit::Rng::new(0x5eed_0041 ^ 0xa5a5);
    let x: Vec<u16> = (0..k * ncols)
        .map(|_| f16_bits((rng.normal() * 0.05) as f32))
        .collect();
    // The reference reads the original bytes; the device copy flips one bit of the low
    // nibble of the group-0 scale of row 0, block 0 (a quarter of the 6-bit range).
    let (y_ref, abs) = reference_q3k(&w_bytes, &x, rows, k, ncols);
    let mut w_bad = w_bytes.clone();
    w_bad[96] ^= 0x02;
    let w = testkit::upload_bytes(&ctx, &w_bad);
    let x_bytes: Vec<u8> = x.iter().flat_map(|b| b.to_le_bytes()).collect();
    let xb = testkit::upload_bytes(&ctx, &x_bytes);
    let y = testkit::upload_filled(&ctx, 0xAA, rows * ncols * 4);
    ctx.gemm_dq_q3k(
        &stream,
        GemmDqQ3kCall {
            w: w.range(0, w_bad.len()).expect("w"),
            x: xb.range(0, x_bytes.len()).expect("x"),
            y: y.range(0, rows * ncols * 4).expect("y"),
            k,
            rows,
            ncols,
            x_col_stride: 0,
            y_col_stride: 0,
            x_f32: false,
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

/// One wrong sub-block scale in one block must move an output past the tolerance: this pins
/// that the bound above is tight enough to catch a single sub-block scale error.
#[test]
fn gemm_dq_iq4xs_catches_one_wrong_group_scale() {
    if !testkit::device_present("gemm_dq_iq4xs_catches_one_wrong_group_scale") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let (rows, k, ncols) = (100usize, 1024usize, 40usize);
    let w_bytes = build_weight_iq4xs(rows, k, 0x5eed_0041);
    let mut rng = testkit::Rng::new(0x5eed_0041 ^ 0xa5a5);
    let x: Vec<u16> = (0..k * ncols)
        .map(|_| f16_bits((rng.normal() * 0.05) as f32))
        .collect();
    // The reference reads the original bytes; the device copy flips one bit of the high
    // 2-bit scale lane of sub-block 1 of row 0, block 0 (a quarter of the 6-bit range,
    // one sub-block only: bytes 2..4 hold the high lanes, byte 2 bit 2 is lane 1).
    let (y_ref, abs) = reference_iq4xs(&w_bytes, &x, rows, k, ncols);
    let mut w_bad = w_bytes.clone();
    w_bad[2] ^= 0x04;
    let w = testkit::upload_bytes(&ctx, &w_bad);
    let x_bytes: Vec<u8> = x.iter().flat_map(|b| b.to_le_bytes()).collect();
    let xb = testkit::upload_bytes(&ctx, &x_bytes);
    let y = testkit::upload_filled(&ctx, 0xAA, rows * ncols * 4);
    ctx.gemm_dq_iq4xs(
        &stream,
        GemmDqIq4xsCall {
            w: w.range(0, w_bad.len()).expect("w"),
            x: xb.range(0, x_bytes.len()).expect("x"),
            y: y.range(0, rows * ncols * 4).expect("y"),
            k,
            rows,
            ncols,
            x_col_stride: 0,
            y_col_stride: 0,
            x_f32: false,
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
        "a single wrong sub-block scale must exceed the tolerance"
    );
}

#[test]
fn gemm_dq_q3k_is_deterministic_and_refuses_bad_shapes() {
    if !testkit::device_present("gemm_dq_q3k_is_deterministic_and_refuses_bad_shapes") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let (rows, k, ncols) = (64usize, 512usize, 64usize);
    let w_bytes = build_weight_q3k(rows, k, 0x5eed_0051);
    let x: Vec<u8> = vec![0x3c; k * ncols * 2]; // 1.0 in f16, repeated: a stable product
    let w = testkit::upload_bytes(&ctx, &w_bytes);
    let xb = testkit::upload_bytes(&ctx, &x);
    let run = |y: &crate::kernels::cuda::context::DeviceBuffer| {
        ctx.gemm_dq_q3k(
            &stream,
            GemmDqQ3kCall {
                w: w.range(0, w_bytes.len()).expect("w"),
                x: xb.range(0, x.len()).expect("x"),
                y: y.range(0, rows * ncols * 4).expect("y"),
                k,
                rows,
                ncols,
                x_col_stride: 0,
                y_col_stride: 0,
                x_f32: false,
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
        let result = ctx.gemm_dq_q3k(
            &stream,
            GemmDqQ3kCall {
                w: w.range(0, w_bytes.len()).expect("w"),
                x: xb.range(0, x.len()).expect("x"),
                y: y1.range(0, rows * ncols * 4).expect("y"),
                k: k_bad,
                rows: rows_bad,
                ncols: ncols_bad,
                x_col_stride: 0,
                y_col_stride: 0,
                x_f32: false,
            },
        );
        let msg = format!("{result:?}");
        assert!(
            msg.contains(needle),
            "refusal {k_bad}/{rows_bad}/{ncols_bad}: {msg}"
        );
    }
}

/// The serve-path offsets: the fused in-projection reads its weight SEGMENT at a
/// non-zero row offset and writes its output at a non-zero column offset inside the
/// destination slot, with the f32 activation rows (x_f32) the family caller feeds.
/// Both in-house dense types must match the CPU reference there.
#[test]
fn gemm_dq_inhouse_matches_the_reference_at_serve_offsets() {
    const NAME: &str = "gemm_dq_inhouse_matches_the_reference_at_serve_offsets";
    if !testkit::device_present(NAME) {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    // One fused buffer: 8 canary rows before the segment under test, and the y
    // slot starts 16 floats in, so a wrong offset cannot pass by luck.
    let (seg_rows, k, ncols) = (72usize, 512usize, 24usize);
    let lead_rows = 8usize;
    let y_lead = 16usize;
    // The y column stride of the serve slot is at least the weight-row count
    // (a token column holds all `rows` results), padded past it.
    let ld = seg_rows + 12;
    for q5k in [true, false] {
        let w_all = if q5k {
            build_weight(seg_rows + lead_rows, k, 0x6eed_0001)
        } else {
            build_weight_q40(seg_rows + lead_rows, k, 0x6eed_0002)
        };
        let ty = if q5k { GgufType::Q5K } else { GgufType::Q40 };
        // The type's own block numbers, not a hand-copied per-row byte count.
        let seg_bytes = (ty.block_bytes() * (k as u64 / ty.block_elems())) as usize;
        let seg = &w_all[lead_rows * seg_bytes..];
        let mut rng = testkit::Rng::new(0x6eed_0003);
        // The serve x layout: token column c at c*x_stride with x_stride >= k
        // (hidden-strided rows), padded past k to prove the stride is honored.
        let x_stride = k + 8;
        let mut x: Vec<f32> = vec![0.0; (ncols - 1) * x_stride + k];
        for c in 0..ncols {
            for i in 0..k {
                x[c * x_stride + i] = rng.normal() as f32 * 0.05;
            }
        }
        // The CPU reference speaks the zero-stride layout: x k-major as
        // [row_k * ncols + col], y row-major as [row * ncols + col]. Plain
        // loops: nested closures cannot borrow the outer index under -D warnings.
        let mut x_kmajor = vec![0.0_f32; k * ncols];
        for i in 0..k {
            for c in 0..ncols {
                x_kmajor[i * ncols + c] = x[c * x_stride + i];
            }
        }
        let (y_ref, abs) = if q5k {
            reference(
                seg,
                &x_kmajor.iter().map(|v| f16_bits(*v)).collect::<Vec<_>>(),
                seg_rows,
                k,
                ncols,
            )
        } else {
            reference_q40(
                seg,
                &x_kmajor.iter().map(|v| f16_bits(*v)).collect::<Vec<_>>(),
                seg_rows,
                k,
                ncols,
            )
        };
        let w = testkit::upload_bytes(&ctx, &w_all);
        let x_bytes: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
        let xb = testkit::upload_bytes(&ctx, &x_bytes);
        // The y slot is column-major per the serve layout: token column c at
        // c*y_col_stride, weight row r within it, so the span is
        // (ncols-1)*ld + seg_rows floats past the canary head.
        let y_span = (ncols - 1) * ld + seg_rows;
        let y_bytes = (y_lead + y_span) * 4;
        let y = testkit::upload_filled(&ctx, 0xEE, y_bytes);
        let call = GemmDqQ5kCall {
            w: w.range(lead_rows * seg_bytes, seg_rows * seg_bytes)
                .expect("w seg"),
            x: xb.range(0, x_bytes.len()).expect("x"),
            y: y.range(y_lead * 4, y_span * 4).expect("y slot"),
            k,
            rows: seg_rows,
            ncols,
            x_col_stride: x_stride,
            y_col_stride: ld,
            x_f32: true,
        };
        if q5k {
            ctx.gemm_dq_q5k(&stream, call).expect("gemm dq q5k");
        } else {
            ctx.gemm_dq_q40(
                &stream,
                GemmDqQ40Call {
                    w: w.range(lead_rows * seg_bytes, seg_rows * seg_bytes)
                        .expect("w seg"),
                    x: xb.range(0, x_bytes.len()).expect("x"),
                    y: y.range(y_lead * 4, y_span * 4).expect("y slot"),
                    k,
                    rows: seg_rows,
                    ncols,
                    x_col_stride: x_stride,
                    y_col_stride: ld,
                    x_f32: true,
                },
            )
            .expect("gemm dq q40");
        }
        stream.synchronize().expect("drain");
        let got_all = testkit::read_bytes(&ctx, &y, y_bytes);
        // The canary head of the slot survives untouched.
        for byte in &got_all[..y_lead * 4] {
            assert_eq!(*byte, 0xEE, "the slot head canary must survive");
        }
        let got = testkit::floats(&got_all[y_lead * 4..]);
        for row in 0..seg_rows {
            for col in 0..ncols {
                let idx = col * ld + row;
                let ridx = row * ncols + col;
                let tolerance = 4.0
                    * (2f64.powi(-11) + f64::from(k as u32) * 2f64.powi(-24))
                    * abs[ridx].max(1e-30);
                let error = (f64::from(got[idx]) - y_ref[ridx]).abs();
                assert!(
                    error <= tolerance,
                    "{} row {row} col {col}: error {error} over tol {tolerance}",
                    if q5k { "q5k" } else { "q40" }
                );
            }
        }
    }
    println!("{NAME}: q5k and q40 match the CPU reference at serve offsets");
}
