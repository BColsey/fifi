//! Device tests (3090) of the in-house codebook matvec (`native/resident/matvec_gguf.cu`):
//! IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, Q2_0 densely and through the expert launch,
//! and IQ4_XS through the expert launch. The CPU oracle is `weights/gguf_dequant.rs`. Device
//! tests are `#[ignore]`d (run them with `--ignored --test-threads=1`) and skip when no device
//! is visible. The host tests at the top need no card.
//!
//! Accumulation order and the matvec tolerance. A lane adds one term per window to a running
//! f32 sum, `acc = fma(d8, fma(s0, i0, s1 * i1), acc)`, where `i0` and `i1` are exact integer
//! sums of 16 int8 products (below 2^24, so exact in f32). Each window therefore adds three
//! roundings (the `s1 * i1` product, the inner fma, the outer fma), the 32 lane sums meet in a
//! five-level butterfly (five more roundings per output), and the CPU reference weights are
//! themselves one f32 rounding away from the exact `scale * grid` product the kernel
//! multiplies through. Each rounding is at most 2^-24 of the magnitude it rounds, which is at
//! most the sum of the absolute terms. The per-output bound used here is
//! `(4 * steps + 8) * 2^-24 * sum(|w * x|)` with `steps = ceil(k / 32 / 32)` windows per lane,
//! which covers `3 * steps + 5 + 1` roundings with margin; the measured max error over the
//! bound is printed.

use super::matvec_expert::ExpertMatvecCall;
use super::matvec_q::tests::{assert_bad_request, bench, nan_filled};
use super::matvec_q::{MatvecCall, QuantType};
use super::quantize_q8_1::{QuantizeCall, q8_1_bytes};
use crate::EngError;
use crate::kernels::cuda::native::resident::matvec_gguf::dequant_gguf_launch;
use crate::kernels::cuda::testkit;
use crate::kernels::cuda::{CapturePlan, CudaContext, DeviceBuffer, DeviceStream};
use crate::weights::{GgufType, dequantize};

/// The types of the dense and expert matvec, with the oracle's type of each.
const DENSE_TYPES: [(QuantType, GgufType); 7] = [
    (QuantType::Iq4Nl, GgufType::Iq4Nl),
    (QuantType::Iq2S, GgufType::Iq2S),
    (QuantType::Iq3Xxs, GgufType::Iq3Xxs),
    (QuantType::Iq2Xxs, GgufType::Iq2Xxs),
    (QuantType::Q20, GgufType::Q20),
    (QuantType::Iq2Xs, GgufType::Iq2Xs),
    (QuantType::Iq3S, GgufType::Iq3S),
];

/// IQ4_XS joins the list for the expert matvec (its dense kernel is a different file).
const EXPERT_TYPES: [(QuantType, GgufType); 8] = [
    (QuantType::Iq4Nl, GgufType::Iq4Nl),
    (QuantType::Iq2S, GgufType::Iq2S),
    (QuantType::Iq3Xxs, GgufType::Iq3Xxs),
    (QuantType::Iq2Xxs, GgufType::Iq2Xxs),
    (QuantType::Q20, GgufType::Q20),
    (QuantType::Iq2Xs, GgufType::Iq2Xs),
    (QuantType::Iq3S, GgufType::Iq3S),
    (QuantType::Iq4Xs, GgufType::Iq4Xs),
];

/// The Flash-Next routed-expert shapes (config.json: hidden 2560, moe_intermediate 640, 512
/// experts, 10 used): gate and up are `k = 2560, rows = 640`, down is `k = 640, rows = 2560`.
/// The down shape is off the 256-value block grid, which is why the 32- and 64-value block types
/// appear in the expert tensors of the public dynamic quantizations.
const FN_GATE_UP: (usize, usize) = (2560, 640);
const FN_DOWN: (usize, usize) = (640, 2560);
const FN_EXPERTS: usize = 512;
const FN_USED: usize = 10;

fn block_ok(ty: QuantType, k: usize) -> bool {
    k.is_multiple_of(ty.block_size())
}

// ---------------------------------------------------------------------------------------------
// Host tests: the format facts the kernels rely on.
// ---------------------------------------------------------------------------------------------

/// The kernel computes the sign byte of a 7-bit sign index as the index plus its parity in
/// bit 7 instead of reading `KSIGNS_IQ2XS`.
#[test]
fn sign_bytes_are_the_seven_bit_index_with_its_parity() {
    for (index, &byte) in crate::weights::KSIGNS_IQ2XS.iter().enumerate() {
        let parity = (index.count_ones() & 1) as u8;
        assert_eq!(
            byte,
            index as u8 | (parity << 7),
            "sign index {index} of the oracle table"
        );
    }
}

/// The kernel negates four packed grid bytes as `(g ^ m) + (m & 0x01010101)`, which is exact
/// when no byte is zero and no byte exceeds 127 (the byte is read as a signed int8 by the dot).
#[test]
fn grid_bytes_are_in_the_range_the_sign_step_needs() {
    let mut seen = 0usize;
    let mut check = |name: &str, bytes: &[u8]| {
        for (at, &byte) in bytes.iter().enumerate() {
            assert!(
                (1..=127).contains(&byte),
                "{name} byte {at} is {byte}, outside 1..=127"
            );
        }
        seen += bytes.len();
    };
    let words = |table: &[u64]| {
        table
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect::<Vec<u8>>()
    };
    let words32 = |table: &[u32]| {
        table
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect::<Vec<u8>>()
    };
    check("IQ2_XXS", &words(&crate::weights::IQ2XXS_GRID));
    check("IQ2_XS", &words(&crate::weights::IQ2XS_GRID));
    check("IQ2_S", &words(&crate::weights::IQ2S_GRID));
    check("IQ3_XXS", &words32(&crate::weights::IQ3XXS_GRID));
    check("IQ3_S", &words32(&crate::weights::IQ3S_GRID));
    assert_eq!(seen, 2048 + 4096 + 8192 + 1024 + 2048);
}

/// The block geometry the kernels hard-code equals the oracle's.
#[test]
fn quant_types_match_the_oracle_block_geometry() {
    for (ty, gt) in EXPERT_TYPES {
        assert_eq!(ty.block_size() as u64, gt.block_elems(), "{ty:?} elements");
        assert_eq!(ty.block_bytes() as u64, gt.block_bytes(), "{ty:?} bytes");
        assert_eq!(
            crate::kernels::cuda::native::resident::matvec_gguf::gguf_type_id(ty),
            Some(gt.id() as i32),
            "{ty:?} id"
        );
    }
}

// ---------------------------------------------------------------------------------------------
// Test data and the CPU reference.
// ---------------------------------------------------------------------------------------------

/// `count` blocks of random bytes with a finite f16 scale `d` in `[d_lo, d_hi]` at offset 0
/// (every type here keeps `d` first).
fn random_blocks(
    ty: QuantType,
    count: usize,
    rng: &mut testkit::Rng,
    d_lo: f64,
    d_hi: f64,
) -> Vec<u8> {
    let mut bytes = vec![0u8; count * ty.block_bytes()];
    for chunk in bytes.chunks_mut(8) {
        let word = rng.next_u64().to_le_bytes();
        chunk.copy_from_slice(&word[..chunk.len()]);
    }
    for block in bytes.chunks_mut(ty.block_bytes()) {
        let d = d_lo + rng.unit() * (d_hi - d_lo);
        block[0..2].copy_from_slice(&testkit::f64_to_f16_bits(d).to_le_bytes());
    }
    bytes
}

fn f16_at(bytes: &[u8], at: usize) -> f64 {
    f64::from(testkit::f16_to_f32(u16::from_le_bytes([
        bytes[at],
        bytes[at + 1],
    ])))
}

/// The activation row values `d * q` of one q8_1 row.
fn q8_row(bytes: &[u8], k: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(k);
    for block in 0..k / 32 {
        let at = block * 36;
        let d = f16_at(bytes, at);
        for j in 0..32 {
            out.push(d * f64::from(bytes[at + 4 + j] as i8));
        }
    }
    out
}

/// `sum(w * x)` and `sum(|w * x|)` in f64.
fn dot_ref(w: &[f32], x: &[f64]) -> (f64, f64) {
    let (mut dot, mut abs) = (0.0f64, 0.0f64);
    for (&w, &x) in w.iter().zip(x) {
        let term = f64::from(w) * x;
        dot += term;
        abs += term.abs();
    }
    (dot, abs)
}

/// The tolerance factor of the module doc for rows of `k` values.
fn tolerance_factor(k: usize) -> f64 {
    (4.0 * (k / 32).div_ceil(32) as f64 + 8.0) * 2f64.powi(-24)
}

struct Quantized {
    buffer: DeviceBuffer,
    host: Vec<u8>,
    row_bytes: usize,
}

/// `rows` random activation rows of `k` values, quantized by the device kernel the engine
/// uses.
fn quantize_rows(
    ctx: &CudaContext,
    stream: &DeviceStream,
    rng: &mut testkit::Rng,
    k: usize,
    rows: usize,
) -> Quantized {
    let x: Vec<f32> = (0..rows * k)
        .map(|_| (rng.normal() as f32) * 0.05)
        .collect();
    let x_f32 = testkit::upload_f32(ctx, &x);
    let bytes = q8_1_bytes(k, rows).expect("size");
    let row_bytes = q8_1_bytes(k, 1).expect("size");
    let buffer = testkit::upload_filled(ctx, 0xAA, bytes);
    // The quantize launch takes at most eight rows: quantize in groups of eight.
    for first in (0..rows).step_by(8) {
        let count = (rows - first).min(8);
        ctx.quantize_q8_1(
            stream,
            QuantizeCall {
                x: x_f32.range(first * k * 4, count * k * 4).expect("x range"),
                y: buffer
                    .range(first * row_bytes, count * row_bytes)
                    .expect("y range"),
                k,
                ncols: count,
            },
        )
        .expect("quantize");
    }
    stream.synchronize().expect("drain");
    let host = testkit::read_bytes(ctx, &buffer, bytes);
    Quantized {
        buffer,
        host,
        row_bytes,
    }
}

fn read_f32(ctx: &CudaContext, buffer: &DeviceBuffer, len: usize) -> Vec<f32> {
    testkit::floats(&testkit::read_bytes(ctx, buffer, len * 4))
}

fn ids_buffer(ctx: &CudaContext, ids: &[i32]) -> DeviceBuffer {
    let bytes: Vec<u8> = ids.iter().flat_map(|id| id.to_le_bytes()).collect();
    testkit::upload_bytes(ctx, &bytes)
}

impl CudaContext {
    /// Test support: decodes `n_blocks` blocks of `ty` on the device.
    fn dequant_gguf(
        &self,
        stream: &DeviceStream,
        ty: QuantType,
        w: &DeviceBuffer,
        y: &DeviceBuffer,
        n_blocks: usize,
    ) -> Result<(), EngError> {
        let w = w.range(0, ty.block_bytes() * n_blocks)?;
        let y = y.range(0, n_blocks * ty.block_size() * 4)?;
        let (w_raw, y_raw) = (self.raw_range(&w)?, self.raw_range(&y)?);
        self.launch_on(stream, |raw| {
            dequant_gguf_launch(raw, ty, &w_raw, &y_raw, n_blocks)
        })
    }
}

// ---------------------------------------------------------------------------------------------
// (a) Device dequant against the CPU oracle, bit for bit.
// ---------------------------------------------------------------------------------------------

/// Blocks that reach the corners: all-zero and all-ones bodies (zero scales and indices, maximal
/// indices, scales and sign fields) under several scales, ramp bodies that put every byte value at
/// every position, and f16 scales that are zero, negative zero, the smallest subnormal and the
/// largest finite value.
fn edge_blocks(ty: QuantType, rng: &mut testkit::Rng) -> Vec<u8> {
    let bb = ty.block_bytes();
    let mut out = Vec::new();
    let mut push = |body: &dyn Fn(usize) -> u8, d_bits: u16| {
        let mut block: Vec<u8> = (0..bb).map(body).collect();
        block[0..2].copy_from_slice(&d_bits.to_le_bytes());
        out.extend_from_slice(&block);
    };
    for d_bits in [0x3c00u16, 0x0000, 0x8000, 0x0001, 0x7bff, 0xbc00] {
        push(&|_| 0x00, d_bits);
        push(&|_| 0xFF, d_bits);
        push(&|i| (i * 37) as u8, d_bits);
    }
    for shift in 0..256usize {
        push(&|i| (i + shift) as u8, 0x3c00);
        push(&|i| (255 - ((i + shift) & 0xFF)) as u8, 0xb800);
    }
    // Random bodies under the extreme scales.
    for d_bits in [0x0000u16, 0x8000, 0x0001, 0x7bff, 0xfbff] {
        for _ in 0..8 {
            let mut block: Vec<u8> = (0..bb).map(|_| (rng.next_u64() >> 24) as u8).collect();
            block[0..2].copy_from_slice(&d_bits.to_le_bytes());
            out.extend_from_slice(&block);
        }
    }
    out
}

#[test]
#[ignore = "device: needs the card"]
fn gguf_device_dequant_is_bit_identical_to_the_cpu_oracle() {
    if !testkit::device_present("gguf_device_dequant_is_bit_identical_to_the_cpu_oracle") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    for (ty, gt) in EXPERT_TYPES {
        let mut rng = testkit::Rng::new(0xD0_0000 + u64::from(gt.id()));
        let mut bytes = random_blocks(ty, 4096, &mut rng, 0.001, 8.0);
        bytes.extend(edge_blocks(ty, &mut rng));
        let n_blocks = bytes.len() / ty.block_bytes();
        let w = testkit::upload_bytes(&ctx, &bytes);
        let y = testkit::upload_filled(&ctx, 0xAA, n_blocks * ty.block_size() * 4);
        ctx.dequant_gguf(&stream, ty, &w, &y, n_blocks)
            .expect("device dequant");
        stream.synchronize().expect("drain");
        let got = read_f32(&ctx, &y, n_blocks * ty.block_size());
        let want = dequantize(gt, &bytes).expect("oracle dequant");
        assert_eq!(got.len(), want.len());
        let wrong: Vec<usize> = (0..want.len())
            .filter(|&i| got[i].to_bits() != want[i].to_bits())
            .collect();
        assert!(
            wrong.is_empty(),
            "{ty:?}: {} of {} values differ, first at {} (block {}): device {:e} oracle {:e}",
            wrong.len(),
            want.len(),
            wrong.first().copied().unwrap_or(0),
            wrong.first().copied().unwrap_or(0) / ty.block_size(),
            got[wrong.first().copied().unwrap_or(0)],
            want[wrong.first().copied().unwrap_or(0)],
        );
        println!(
            "dequant {ty:?}: {n_blocks} blocks, {} values, all bit-identical",
            want.len()
        );
    }
}

// ---------------------------------------------------------------------------------------------
// (b) Dense matvec against the CPU f32 reference.
// ---------------------------------------------------------------------------------------------

struct DenseRun {
    got: Vec<f32>,
    w_bytes: Vec<u8>,
    x: Quantized,
}

fn run_dense(
    ctx: &CudaContext,
    stream: &DeviceStream,
    ty: QuantType,
    k: usize,
    rows: usize,
    ncols: usize,
    seed: u64,
) -> DenseRun {
    let mut rng = testkit::Rng::new(seed);
    let w_bytes = random_blocks(ty, rows * (k / ty.block_size()), &mut rng, 0.0005, 0.01);
    let w = testkit::upload_bytes(ctx, &w_bytes);
    let x = quantize_rows(ctx, stream, &mut rng, k, ncols);
    let y_bytes = rows * ncols * 4;
    let y = testkit::upload_filled(ctx, 0xAA, y_bytes);
    ctx.matvec_q(
        stream,
        MatvecCall {
            ty,
            w: w.range(0, w_bytes.len()).expect("w"),
            x_q8: x.buffer.range(0, x.row_bytes * ncols).expect("x"),
            y: y.range(0, y_bytes).expect("y"),
            k,
            rows,
            ncols,
        },
    )
    .expect("matvec");
    stream.synchronize().expect("drain");
    DenseRun {
        got: read_f32(ctx, &y, rows * ncols),
        w_bytes,
        x,
    }
}

/// Compares a dense run with the reference; returns the max of error over tolerance.
fn check_dense(
    ty: QuantType,
    gt: GgufType,
    run: &DenseRun,
    k: usize,
    rows: usize,
    ncols: usize,
) -> f64 {
    let weights = dequantize(gt, &run.w_bytes).expect("oracle dequant");
    let mut worst = 0.0f64;
    for col in 0..ncols {
        let x = q8_row(&run.x.host[col * run.x.row_bytes..], k);
        for row in 0..rows {
            let (dot, abs) = dot_ref(&weights[row * k..(row + 1) * k], &x);
            let tolerance = tolerance_factor(k) * abs;
            let error = (f64::from(run.got[col * rows + row]) - dot).abs();
            let ratio = if tolerance > 0.0 {
                error / tolerance
            } else {
                error
            };
            worst = worst.max(ratio);
            assert!(
                error <= tolerance,
                "{ty:?} k {k} rows {rows} ncols {ncols} col {col} row {row}: error {error:e} \
                 over tolerance {tolerance:e} (reference {dot:e}, got {:e})",
                run.got[col * rows + row]
            );
        }
    }
    worst
}

/// The dense shapes of one type: Flash-Next's two expert shapes where the type's blocks allow,
/// a row count off every grid (13, 1), column counts that are not powers of two, a `k` that is
/// not a multiple of the 512 activation padding, and a `k` large enough that the activation
/// no longer fits shared memory (the global-read path).
fn dense_shapes(ty: QuantType) -> Vec<(usize, usize, usize)> {
    let bs = ty.block_size();
    let mut shapes: Vec<(usize, usize, usize)> = vec![
        (2560, 640, 1),
        (2560, 13, 3),
        (2560, 1, 8),
        (bs, 5, 1),
        (bs * 3, 37, 2),
        (640, 512, 1),
        (640, 64, 5),
        (4096, 40, 8),
        (26_624, 9, 1),
        (2304, 33, 2),
    ];
    shapes.retain(|&(k, _, _)| block_ok(ty, k));
    shapes
}

#[test]
#[ignore = "device: needs the card"]
fn gguf_dense_matvec_matches_the_cpu_reference() {
    if !testkit::device_present("gguf_dense_matvec_matches_the_cpu_reference") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    for (ty, gt) in DENSE_TYPES {
        let mut worst = 0.0f64;
        let shapes = dense_shapes(ty);
        for (i, &(k, rows, ncols)) in shapes.iter().enumerate() {
            let seed = 0x5eed_6000u64 + (gt.id() as u64) * 100 + i as u64;
            let run = run_dense(&ctx, &stream, ty, k, rows, ncols, seed);
            worst = worst.max(check_dense(ty, gt, &run, k, rows, ncols));
        }
        println!(
            "dense {ty:?}: {} shapes, max error over tolerance {worst:.3}",
            shapes.len()
        );
    }
}

/// Column `j` of an `ncols` launch equals a one-column launch over the same bytes, bit for bit
/// (the lane arithmetic of a column does not depend on how many columns ride along).
#[test]
#[ignore = "device: needs the card"]
fn gguf_dense_columns_do_not_depend_on_ncols() {
    if !testkit::device_present("gguf_dense_columns_do_not_depend_on_ncols") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let (k, rows, ncols) = (2560usize, 96usize, 8usize);
    for (ty, _gt) in DENSE_TYPES {
        let mut rng = testkit::Rng::new(0xC01_0000 + ty.block_bytes() as u64);
        let w_bytes = random_blocks(ty, rows * (k / ty.block_size()), &mut rng, 0.0005, 0.01);
        let w = testkit::upload_bytes(&ctx, &w_bytes);
        let x = quantize_rows(&ctx, &stream, &mut rng, k, ncols);
        let run = |first: usize, count: usize| {
            let y = testkit::upload_filled(&ctx, 0xAA, rows * count * 4);
            ctx.matvec_q(
                &stream,
                MatvecCall {
                    ty,
                    w: w.range(0, w_bytes.len()).expect("w"),
                    x_q8: x
                        .buffer
                        .range(first * x.row_bytes, count * x.row_bytes)
                        .expect("x"),
                    y: y.range(0, rows * count * 4).expect("y"),
                    k,
                    rows,
                    ncols: count,
                },
            )
            .expect("matvec");
            stream.synchronize().expect("drain");
            read_f32(&ctx, &y, rows * count)
        };
        let batch = run(0, ncols);
        for col in 0..ncols {
            let single = run(col, 1);
            for row in 0..rows {
                assert_eq!(
                    batch[col * rows + row].to_bits(),
                    single[row].to_bits(),
                    "{ty:?} column {col} row {row}"
                );
            }
        }
    }
}

/// One flipped bit per field of a block must move the output: every field feeds the dot. The
/// flipped byte is chosen by the field map of each type (name, offset in the block).
#[test]
#[ignore = "device: needs the card"]
fn gguf_dense_matvec_catches_one_flipped_field() {
    if !testkit::device_present("gguf_dense_matvec_catches_one_flipped_field") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    // (type, [(field, byte offset in the first window of the block, mask)]). The scale and the
    // index fields sit in different bytes per type; every byte of a block below is read by the
    // window whose lane covers it, and the first window reads the offsets listed.
    type Fields<'a> = &'a [(&'a str, usize, u8)];
    let fields: [(QuantType, Fields); 7] = [
        (QuantType::Iq4Nl, &[("d", 0, 1), ("codes", 2, 1)]),
        (
            QuantType::Iq2S,
            &[
                ("d", 0, 1),
                ("grid index", 2, 1),
                ("sign", 34, 1),
                ("high bits", 66, 1),
                ("scale", 74, 1),
            ],
        ),
        (
            QuantType::Iq3Xxs,
            &[
                ("d", 0, 1),
                ("grid index", 2, 1),
                ("signs and scale", 66, 1),
            ],
        ),
        (
            QuantType::Iq2Xxs,
            &[("d", 0, 1), ("grid index", 2, 1), ("signs and scale", 6, 1)],
        ),
        (QuantType::Q20, &[("d", 0, 1), ("codes", 2, 1)]),
        (
            QuantType::Iq2Xs,
            &[
                ("d", 0, 1),
                ("grid index", 2, 1),
                ("sign", 3, 2),
                ("scale", 66, 1),
            ],
        ),
        (
            QuantType::Iq3S,
            &[
                ("d", 0, 1),
                ("grid index", 2, 1),
                ("high bit", 66, 1),
                ("sign", 74, 1),
                ("scale", 106, 1),
            ],
        ),
    ];
    for (ty, list) in fields {
        let (k, rows, ncols, seed) = (
            512usize.max(ty.block_size()),
            64usize,
            2usize,
            0xF11_0001u64,
        );
        let mut rng = testkit::Rng::new(seed);
        let w_bytes = random_blocks(ty, rows * (k / ty.block_size()), &mut rng, 0.001, 0.01);
        let x = quantize_rows(&ctx, &stream, &mut rng, k, ncols);
        let y_bytes = rows * ncols * 4;
        let run = |bytes: &[u8]| {
            let w = testkit::upload_bytes(&ctx, bytes);
            let y = testkit::upload_filled(&ctx, 0xAA, y_bytes);
            ctx.matvec_q(
                &stream,
                MatvecCall {
                    ty,
                    w: w.range(0, bytes.len()).expect("w"),
                    x_q8: x.buffer.range(0, x.row_bytes * ncols).expect("x"),
                    y: y.range(0, y_bytes).expect("y"),
                    k,
                    rows,
                    ncols,
                },
            )
            .expect("matvec");
            stream.synchronize().expect("drain");
            read_f32(&ctx, &y, rows * ncols)
        };
        let clean = run(&w_bytes);
        let row = 3usize;
        let base = (row * (k / ty.block_size()) + 1) * ty.block_bytes();
        for &(field, offset, mask) in list {
            let mut flipped = w_bytes.clone();
            flipped[base + offset] ^= mask;
            let dirty = run(&flipped);
            let moved = (0..ncols).any(|col| clean[col * rows + row] != dirty[col * rows + row]);
            assert!(moved, "{ty:?}: one flipped {field} bit must move row {row}");
            // No other row moves.
            for other in (0..rows).filter(|&r| r != row) {
                for col in 0..ncols {
                    assert_eq!(
                        clean[col * rows + other].to_bits(),
                        dirty[col * rows + other].to_bits(),
                        "{ty:?}: flipping {field} of row {row} moved row {other}"
                    );
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// (c) Expert matvec.
// ---------------------------------------------------------------------------------------------

struct ExpertCase {
    ty: QuantType,
    gt: GgufType,
    k: usize,
    rows: usize,
    n_expert: usize,
    ids: Vec<i32>,
    x_per_slot: bool,
    y_slot_stride: usize,
    glu: bool,
}

fn silu(x: f64) -> f64 {
    x / (1.0 + (-x).exp())
}

/// One expert launch against the reference; returns the max of error over tolerance.
fn run_expert(ctx: &CudaContext, stream: &DeviceStream, case: &ExpertCase, seed: u64) -> f64 {
    let ExpertCase {
        ty,
        gt,
        k,
        rows,
        n_expert,
        ref ids,
        x_per_slot,
        y_slot_stride,
        glu,
    } = *case;
    let n_used = ids.len();
    let mut rng = testkit::Rng::new(seed);
    let blocks = n_expert * rows * (k / ty.block_size());
    let up_bytes = random_blocks(ty, blocks, &mut rng, 0.0005, 0.01);
    let gate_bytes = random_blocks(ty, blocks, &mut rng, 0.0005, 0.01);
    let up = testkit::upload_bytes(ctx, &up_bytes);
    let gate = glu.then(|| testkit::upload_bytes(ctx, &gate_bytes));
    let x_rows = if x_per_slot { n_used } else { 1 };
    let x = quantize_rows(ctx, stream, &mut rng, k, x_rows);
    let ids_dev = ids_buffer(ctx, ids);
    let y_floats = (n_used - 1) * y_slot_stride + rows;
    let y = testkit::upload_filled(ctx, 0xAA, y_floats * 4);
    ctx.matvec_expert(
        stream,
        ExpertMatvecCall {
            ty,
            w_stack: up.range(0, up_bytes.len()).expect("stack"),
            ids: ids_dev.range(0, n_used * 4).expect("ids"),
            x_q8: x.buffer.range(0, x.row_bytes * x_rows).expect("x"),
            x_per_slot,
            y: y.range(0, y_floats * 4).expect("y"),
            y_slot_stride,
            k,
            rows,
            n_expert,
            n_used,
            glu_gate: gate
                .as_ref()
                .map(|g| g.range(0, gate_bytes.len()).expect("gate")),
        },
    )
    .expect("expert matvec");
    stream.synchronize().expect("drain");
    let got = read_f32(ctx, &y, y_floats);
    let up_w = dequantize(gt, &up_bytes).expect("oracle dequant");
    let gate_w = dequantize(gt, &gate_bytes).expect("oracle dequant");
    let mut worst = 0.0f64;
    let poison = f32::from_bits(0xAAAA_AAAA);
    for (slot, &id) in ids.iter().enumerate() {
        let out = &got[slot * y_slot_stride..slot * y_slot_stride + rows];
        if id < 0 || id as usize >= n_expert {
            assert!(
                out.iter().all(|v| v.to_bits() == 0x7fc0_0000),
                "{ty:?} slot {slot}: id {id} must give NaN rows"
            );
            continue;
        }
        let xrow = q8_row(
            &x.host[(if x_per_slot { slot } else { 0 }) * x.row_bytes..],
            k,
        );
        let base = id as usize * rows * k;
        for (row, &got_row) in out.iter().enumerate() {
            let w_row = base + row * k;
            let (dot, abs) = dot_ref(&up_w[w_row..w_row + k], &xrow);
            let mut tolerance = tolerance_factor(k) * abs;
            let mut want = dot;
            if glu {
                let (gdot, gabs) = dot_ref(&gate_w[w_row..w_row + k], &xrow);
                let gtol = tolerance_factor(k) * gabs;
                // y = up * silu(gate): the up error scales by |silu(gate)|, the gate error by
                // |up| times the largest slope of silu (1.1), plus a few roundings of the
                // product, the exponential and the division.
                tolerance = tolerance * silu(gdot).abs()
                    + dot.abs() * 1.1 * gtol
                    + 8.0 * 2f64.powi(-24) * (dot * silu(gdot)).abs();
                want = dot * silu(gdot);
            }
            let error = (f64::from(got_row) - want).abs();
            let ratio = if tolerance > 0.0 {
                error / tolerance
            } else {
                error
            };
            worst = worst.max(ratio);
            assert!(
                error <= tolerance,
                "{ty:?} slot {slot} id {id} row {row}: error {error:e} over tolerance \
                 {tolerance:e} (reference {want:e}, got {got_row:e})"
            );
        }
        // Columns between the rows and the slot stride are not written.
        if y_slot_stride > rows && slot + 1 < n_used {
            let gap = &got[slot * y_slot_stride + rows..(slot + 1) * y_slot_stride];
            assert!(
                gap.iter().all(|v| v.to_bits() == poison.to_bits()),
                "{ty:?} slot {slot}: the stride gap was written"
            );
        }
    }
    worst
}

#[test]
#[ignore = "device: needs the card"]
fn gguf_expert_matvec_matches_the_cpu_reference() {
    if !testkit::device_present("gguf_expert_matvec_matches_the_cpu_reference") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    for (ty, gt) in EXPERT_TYPES {
        let mut worst = 0.0f64;
        let mut cases = 0usize;
        let mut run = |case: ExpertCase, seed: u64| {
            worst = worst.max(run_expert(&ctx, &stream, &case, seed));
            cases += 1;
        };
        for (shape, (k, rows)) in [FN_GATE_UP, FN_DOWN].into_iter().enumerate() {
            if !block_ok(ty, k) {
                continue;
            }
            let seed = 0xE0_0000 + gt.id() as u64 * 1000 + shape as u64 * 100;
            // The routed shape: ten slots over twelve experts, repeats included, shared x.
            run(
                ExpertCase {
                    ty,
                    gt,
                    k,
                    rows: rows.min(256),
                    n_expert: 12,
                    ids: vec![3, 11, 0, 3, 7, 5, 9, 1, 11, 2],
                    x_per_slot: false,
                    y_slot_stride: rows.min(256),
                    glu: false,
                },
                seed,
            );
            // The down view: one activation row per slot, a stride wider than the rows.
            run(
                ExpertCase {
                    ty,
                    gt,
                    k,
                    rows: 37,
                    n_expert: 6,
                    ids: vec![5, 0, 2, 4],
                    x_per_slot: true,
                    y_slot_stride: 64,
                    glu: false,
                },
                seed + 1,
            );
            // The fused gate: up * silu(gate) over one shared activation row.
            run(
                ExpertCase {
                    ty,
                    gt,
                    k,
                    rows: 50,
                    n_expert: 5,
                    ids: vec![4, 1, 2],
                    x_per_slot: false,
                    y_slot_stride: 50,
                    glu: true,
                },
                seed + 2,
            );
            // Ids outside the stack answer NaN rows next to valid slots.
            run(
                ExpertCase {
                    ty,
                    gt,
                    k,
                    rows: 24,
                    n_expert: 8,
                    ids: vec![2, -1, 3, 4, 0, 100],
                    x_per_slot: true,
                    y_slot_stride: 24,
                    glu: false,
                },
                seed + 3,
            );
        }
        // A single slot of a single expert and the widest launch.
        let k = 2560usize
            .next_multiple_of(ty.block_size())
            .max(ty.block_size());
        let k = if block_ok(ty, k) { k } else { 2560 };
        run(
            ExpertCase {
                ty,
                gt,
                k,
                rows: 8,
                n_expert: 1,
                ids: vec![0],
                x_per_slot: false,
                y_slot_stride: 8,
                glu: false,
            },
            0xE1_0000 + gt.id() as u64,
        );
        run(
            ExpertCase {
                ty,
                gt,
                k,
                rows: 16,
                n_expert: 40,
                ids: (0..32).map(|i| i * 7 % 40).collect(),
                x_per_slot: true,
                y_slot_stride: 16,
                glu: false,
            },
            0xE2_0000 + gt.id() as u64,
        );
        println!("expert {ty:?}: {cases} launches, max error over tolerance {worst:.3}");
    }
}

/// An expert slot equals the dense matvec over that expert's rows, bit for bit: the two
/// launches run the same lane arithmetic.
#[test]
#[ignore = "device: needs the card"]
fn gguf_expert_slot_equals_the_dense_matvec_of_its_expert() {
    if !testkit::device_present("gguf_expert_slot_equals_the_dense_matvec_of_its_expert") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let (rows, n_expert) = (48usize, 5usize);
    for (ty, _gt) in DENSE_TYPES {
        let k = 2560usize;
        let mut rng = testkit::Rng::new(0xE3_0000 + ty.block_bytes() as u64);
        let expert_blocks = rows * (k / ty.block_size());
        let bytes = random_blocks(ty, n_expert * expert_blocks, &mut rng, 0.0005, 0.01);
        let stack = testkit::upload_bytes(&ctx, &bytes);
        let x = quantize_rows(&ctx, &stream, &mut rng, k, 1);
        let ids = [4i32, 1, 3];
        let ids_dev = ids_buffer(&ctx, &ids);
        let y = testkit::upload_filled(&ctx, 0xAA, ids.len() * rows * 4);
        ctx.matvec_expert(
            &stream,
            ExpertMatvecCall {
                ty,
                w_stack: stack.range(0, bytes.len()).expect("stack"),
                ids: ids_dev.range(0, 12).expect("ids"),
                x_q8: x.buffer.range(0, x.row_bytes).expect("x"),
                x_per_slot: false,
                y: y.range(0, ids.len() * rows * 4).expect("y"),
                y_slot_stride: rows,
                k,
                rows,
                n_expert,
                n_used: ids.len(),
                glu_gate: None,
            },
        )
        .expect("expert");
        stream.synchronize().expect("drain");
        let got = read_f32(&ctx, &y, ids.len() * rows);
        for (slot, &id) in ids.iter().enumerate() {
            let one = testkit::upload_filled(&ctx, 0xAA, rows * 4);
            let w_bytes = expert_blocks * ty.block_bytes();
            ctx.matvec_q(
                &stream,
                MatvecCall {
                    ty,
                    w: stack.range(id as usize * w_bytes, w_bytes).expect("w"),
                    x_q8: x.buffer.range(0, x.row_bytes).expect("x"),
                    y: one.range(0, rows * 4).expect("y"),
                    k,
                    rows,
                    ncols: 1,
                },
            )
            .expect("dense");
            stream.synchronize().expect("drain");
            let dense = read_f32(&ctx, &one, rows);
            for row in 0..rows {
                assert_eq!(
                    got[slot * rows + row].to_bits(),
                    dense[row].to_bits(),
                    "{ty:?} slot {slot} row {row}"
                );
            }
        }
    }
}

/// The refusals that happen before any native call.
#[test]
#[ignore = "device: needs the card"]
fn gguf_matvec_refuses_bad_shapes_before_the_adapter() {
    if !testkit::device_present("gguf_matvec_refuses_bad_shapes_before_the_adapter") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let ty = QuantType::Iq2S;
    let mut rng = testkit::Rng::new(0x4e_f001);
    let k = 2560usize;
    let rows = 16usize;
    let bytes = random_blocks(ty, rows * (k / 256), &mut rng, 0.001, 0.01);
    let w = testkit::upload_bytes(&ctx, &bytes);
    let x = quantize_rows(&ctx, &stream, &mut rng, k, 4);
    let y = testkit::upload_filled(&ctx, 0xAA, rows * 4 * 9);
    let dense = |k: usize, rows: usize, ncols: usize| {
        ctx.matvec_q(
            &stream,
            MatvecCall {
                ty,
                w: w.range(0, w.bytes()).expect("w"),
                x_q8: x.buffer.range(0, x.buffer.bytes()).expect("x"),
                y: y.range(0, y.bytes()).expect("y"),
                k,
                rows,
                ncols,
            },
        )
    };
    assert_bad_request(
        dense(100, rows, 1),
        "k off the block grid",
        "multiple of 256",
    );
    assert_bad_request(dense(k, 0, 1), "zero rows", "positive multiple of 4");
    assert_bad_request(dense(k, rows, 9), "nine columns", "1 to 8 columns");
    assert_bad_request(dense(k, rows, 0), "zero columns", "1 to 8 columns");
    let ids = ids_buffer(&ctx, &[0i32; 40]);
    let y_slots = testkit::upload_filled(&ctx, 0xAA, 40 * rows * 4);
    let stack = testkit::upload_bytes(
        &ctx,
        &random_blocks(ty, 2 * rows * (k / 256), &mut rng, 0.001, 0.01),
    );
    let expert = |n_used: usize, n_expert: usize, per_slot: bool, glu: bool, stride: usize| {
        ctx.matvec_expert(
            &stream,
            ExpertMatvecCall {
                ty,
                w_stack: stack.range(0, 2 * rows * (k / 256) * 82).expect("stack"),
                ids: ids.range(0, 40 * 4).expect("ids"),
                x_q8: x.buffer.range(0, x.buffer.bytes()).expect("x"),
                x_per_slot: per_slot,
                y: y_slots.range(0, y_slots.bytes()).expect("y"),
                y_slot_stride: stride,
                k,
                rows,
                n_expert,
                n_used,
                glu_gate: glu.then(|| stack.range(0, 2 * rows * (k / 256) * 82).expect("gate")),
            },
        )
    };
    assert_bad_request(expert(33, 40, false, false, rows), "33 slots", "used slots");
    assert_bad_request(
        expert(3, 2, false, false, rows),
        "more slots than experts",
        "used slots",
    );
    assert_bad_request(expert(0, 2, false, false, rows), "zero slots", "used slots");
    assert_bad_request(
        expert(2, 2, false, false, rows - 1),
        "stride below rows",
        "stride",
    );
    assert_bad_request(
        expert(2, 2, true, true, rows),
        "glu with per-slot x",
        "shared-x view only",
    );
    // A 33-column-wide k that is not on the block grid.
    let off_grid = ctx.matvec_expert(
        &stream,
        ExpertMatvecCall {
            ty,
            w_stack: stack.range(0, stack.bytes()).expect("stack"),
            ids: ids.range(0, 8).expect("ids"),
            x_q8: x.buffer.range(0, x.buffer.bytes()).expect("x"),
            x_per_slot: false,
            y: y_slots.range(0, y_slots.bytes()).expect("y"),
            y_slot_stride: rows,
            k: 300,
            rows,
            n_expert: 2,
            n_used: 2,
            glu_gate: None,
        },
    );
    assert_bad_request(off_grid, "k off the block grid", "multiple of 256");
    // The vendored kernel's row rule does not apply: rows 13 is legal for the expert launch.
    expert(2, 2, false, false, rows).expect("a valid launch is accepted");
    stream.synchronize().expect("drain");
}

/// The dense and the expert launch record into a captured graph and replay to the same bytes
/// as eager launches: the table rides the kernel arguments, which a graph node keeps.
#[test]
#[ignore = "device: needs the card"]
fn gguf_matvec_inside_a_capture_equals_eager() {
    if !testkit::device_present("gguf_matvec_inside_a_capture_equals_eager") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let (k, rows, n_expert) = (2560usize, 64usize, 4usize);
    for (ty, _gt) in [
        (QuantType::Iq2S, GgufType::Iq2S),
        (QuantType::Iq4Nl, GgufType::Iq4Nl),
    ] {
        let mut rng = testkit::Rng::new(0xCA_0000 + ty.block_bytes() as u64);
        let bytes = random_blocks(
            ty,
            n_expert * rows * (k / ty.block_size()),
            &mut rng,
            0.001,
            0.01,
        );
        let stack = testkit::upload_bytes(&ctx, &bytes);
        let x = quantize_rows(&ctx, &stream, &mut rng, k, 1);
        let ids = ids_buffer(&ctx, &[3, 0, 2]);
        let y_bytes = 3 * rows * 4;
        let launch = |y: &DeviceBuffer| -> Result<(), EngError> {
            ctx.matvec_expert(
                &stream,
                ExpertMatvecCall {
                    ty,
                    w_stack: stack.range(0, bytes.len())?,
                    ids: ids.range(0, 12)?,
                    x_q8: x.buffer.range(0, x.row_bytes)?,
                    x_per_slot: false,
                    y: y.range(0, y_bytes)?,
                    y_slot_stride: rows,
                    k,
                    rows,
                    n_expert,
                    n_used: 3,
                    glu_gate: None,
                },
            )
        };
        let eager_y = testkit::upload_filled(&ctx, 0xAA, y_bytes);
        launch(&eager_y).expect("eager");
        stream.synchronize().expect("drain eager");
        let eager = testkit::read_bytes(&ctx, &eager_y, y_bytes);
        let cap_y = testkit::upload_filled(&ctx, 0xAA, y_bytes);
        let plan = CapturePlan {
            pins: &[],
            holds: Vec::new(),
        };
        let (graph, ()) = ctx
            .capture(&stream, plan, || launch(&cap_y))
            .expect("capture");
        assert_eq!(graph.nodes(), 1, "one expert node");
        for replay in 0..3 {
            cap_y
                .write_at(&stream, 0, &vec![0xAA; y_bytes])
                .expect("reset y");
            graph.launch(&stream).expect("replay");
            stream.synchronize().expect("drain the replay");
            assert_eq!(
                testkit::read_bytes(&ctx, &cap_y, y_bytes),
                eager,
                "{ty:?} replay {replay} differs from eager"
            );
        }
    }
}

// ---------------------------------------------------------------------------------------------
// (d) Speed: the Flash-Next expert shapes.
// ---------------------------------------------------------------------------------------------

#[test]
#[ignore = "bench: exclusive timing slot"]
fn gguf_codebook_matvec_report_gbps() {
    if !testkit::device_present("gguf_codebook_matvec_report_gbps") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let (warm_iters, cold_iters) = if cfg!(debug_assertions) {
        (5, 3)
    } else {
        (200, 20)
    };
    // An L2 flush: one dense matvec over a buffer far larger than the 6 MB L2.
    let flush_ty = QuantType::Iq4Nl;
    let (flush_k, flush_rows) = (2560usize, 46_000usize);
    let flush_bytes = flush_ty.weight_bytes(flush_rows, flush_k).expect("size");
    let flush_w = testkit::upload_filled(&ctx, 0x31, flush_bytes);
    let flush_x = testkit::upload_filled(&ctx, 0x3c, q8_1_bytes(flush_k, 1).expect("size"));
    let flush_y = testkit::upload_filled(&ctx, 0, flush_rows * 4);
    let flush = || {
        ctx.matvec_q(
            &stream,
            MatvecCall {
                ty: flush_ty,
                w: flush_w.range(0, flush_bytes).expect("w"),
                x_q8: flush_x
                    .range(0, q8_1_bytes(flush_k, 1).expect("size"))
                    .expect("x"),
                y: flush_y.range(0, flush_rows * 4).expect("y"),
                k: flush_k,
                rows: flush_rows,
                ncols: 1,
            },
        )
        .expect("flush");
    };
    println!(
        "warm = {warm_iters} launches between two events; cold = one launch timed alone after a \
         {} MB flush read (L2 evicted). Bytes counted: the weights the launch reads, the \
         activation row(s) and the output. GPU peak 936 GB/s.",
        flush_bytes / 1_000_000
    );
    println!("type     shape          launch      warm_us  cold_us  GB/s_warm GB/s_cold  %936");
    for (ty, _gt) in EXPERT_TYPES {
        for (name, (k, rows)) in [("gate/up", FN_GATE_UP), ("down", FN_DOWN)] {
            if !block_ok(ty, k) {
                continue;
            }
            let mut rng = testkit::Rng::new(0xBE_0000 + (k + rows) as u64);
            let expert_bytes = ty.weight_bytes(rows, k).expect("size");
            let n_expert = FN_EXPERTS;
            let stack_bytes = random_blocks(
                ty,
                n_expert * rows * (k / ty.block_size()),
                &mut rng,
                0.0005,
                0.01,
            );
            let stack = testkit::upload_bytes(&ctx, &stack_bytes);
            drop(stack_bytes);
            let x = quantize_rows(&ctx, &stream, &mut rng, k, FN_USED);
            let id_sets: Vec<DeviceBuffer> = (0..32)
                .map(|_| {
                    let ids: Vec<i32> = (0..FN_USED)
                        .map(|_| rng.below(n_expert as u64) as i32)
                        .collect();
                    ids_buffer(&ctx, &ids)
                })
                .collect();
            let y_bytes = FN_USED * rows * 4;
            let y = nan_filled(&ctx, y_bytes);
            // The down projection reads one activation row per slot.
            for (launch_name, per_slot) in [
                ("expert", name == "down"),
                ("2x5", name == "down"),
                ("dense", false),
            ] {
                let (n_used, bytes_read) = if launch_name == "dense" {
                    (1, expert_bytes)
                } else {
                    (FN_USED, expert_bytes * FN_USED)
                };
                let x_rows = if per_slot { FN_USED } else { 1 };
                let run = |ids: &DeviceBuffer| {
                    if launch_name == "2x5" {
                        // The same ten slots as two launches of five: the second launch reads
                        // ids 5..10 and writes slots 5..10 (a second view of the same ranges).
                        let half = FN_USED / 2;
                        for part in 0..2 {
                            ctx.matvec_expert(
                                &stream,
                                ExpertMatvecCall {
                                    ty,
                                    w_stack: stack
                                        .range(0, n_expert * expert_bytes)
                                        .expect("stack"),
                                    ids: ids.range(part * half * 4, half * 4).expect("ids"),
                                    x_q8: x
                                        .buffer
                                        .range(
                                            if per_slot {
                                                part * half * x.row_bytes
                                            } else {
                                                0
                                            },
                                            x.row_bytes * if per_slot { half } else { 1 },
                                        )
                                        .expect("x"),
                                    x_per_slot: per_slot,
                                    y: y.range(part * half * rows * 4, half * rows * 4).expect("y"),
                                    y_slot_stride: rows,
                                    k,
                                    rows,
                                    n_expert,
                                    n_used: half,
                                    glu_gate: None,
                                },
                            )
                            .expect("expert half");
                        }
                    } else if launch_name == "expert" {
                        ctx.matvec_expert(
                            &stream,
                            ExpertMatvecCall {
                                ty,
                                w_stack: stack.range(0, n_expert * expert_bytes).expect("stack"),
                                ids: ids.range(0, FN_USED * 4).expect("ids"),
                                x_q8: x.buffer.range(0, x.row_bytes * x_rows).expect("x"),
                                x_per_slot: per_slot,
                                y: y.range(0, y_bytes).expect("y"),
                                y_slot_stride: rows,
                                k,
                                rows,
                                n_expert,
                                n_used,
                                glu_gate: None,
                            },
                        )
                        .expect("expert");
                    } else if ty != QuantType::Iq4Xs {
                        ctx.matvec_q(
                            &stream,
                            MatvecCall {
                                ty,
                                w: stack.range(0, expert_bytes).expect("w"),
                                x_q8: x.buffer.range(0, x.row_bytes).expect("x"),
                                y: y.range(0, rows * 4).expect("y"),
                                k,
                                rows,
                                ncols: 1,
                            },
                        )
                        .expect("dense");
                    }
                };
                if launch_name == "dense" && ty == QuantType::Iq4Xs {
                    continue;
                }
                let warm_us = bench(&ctx, &stream, warm_iters, &|| run(&id_sets[0]));
                let mut cold = Vec::new();
                for set in 0..cold_iters {
                    flush();
                    let start = ctx.bench_mark(&stream).expect("start");
                    run(&id_sets[set % id_sets.len()]);
                    let end = ctx.bench_mark(&stream).expect("end");
                    stream.synchronize().expect("drain");
                    cold.push(ctx.bench_elapsed_ms(start, end).expect("elapsed") * 1000.0);
                    ctx.bench_destroy(start).expect("destroy");
                    ctx.bench_destroy(end).expect("destroy");
                }
                cold.sort_by(|a, b| a.total_cmp(b));
                let cold_us = cold[cold.len() / 2];
                let out = read_f32(&ctx, &y, n_used * rows);
                assert!(
                    out.iter().take(rows).all(|v| !v.is_nan()),
                    "{ty:?} {name} {launch_name}: NaN output"
                );
                let bytes =
                    bytes_read as f64 + (x.row_bytes * x_rows) as f64 + (n_used * rows * 4) as f64;
                let gbs = |us: f32| bytes / (f64::from(us) * 1e-6) / 1e9;
                println!(
                    "{:8} {:7} k{:<5} {launch_name:8} {warm_us:9.2} {cold_us:8.2} {:9.0} {:9.0} {:6.1}",
                    format!("{ty:?}"),
                    name,
                    k,
                    gbs(warm_us),
                    gbs(cold_us),
                    gbs(cold_us) / 936.0 * 100.0
                );
            }
        }
    }
}
