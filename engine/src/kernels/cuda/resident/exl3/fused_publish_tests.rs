//! Bit sweeps of the producer-fusion arm. A fusing producer must store exactly what the
//! same producer followed by the separate BF16 publish kernel stores, status flag included.
//! A host without a device skips these tests.

use super::Exl3LinearCall;
use super::linear_tests::{fixture, upload_scales};
use crate::kernels::cuda::prefill::row_ops::{RowMulCall, RowShape};
use crate::kernels::cuda::resident::row_add::RowAddCall;
use crate::kernels::cuda::resident::row_scale::Bf16PublishCall;
use crate::kernels::cuda::{CudaContext, DeviceBuffer, DeviceStream, testkit};
use crate::weights::exl3::Codebook;

const CLEAR: [u8; 4] = 8u32.to_le_bytes();
const STICKY: [u8; 4] = 9u32.to_le_bytes();

/// Edge values of the publication contract: RNE ties, signed zero, subnormals, the
/// BF16 overflow band below f32::MAX, infinities and NaNs of both signs and payloads.
fn edge_values() -> Vec<f32> {
    [
        0x0000_0000u32,
        0x8000_0000,
        0x3f80_0000,
        0xbf80_0000,
        0x3fc0_0000,
        0x3f80_8000,
        0x3f81_8000,
        0xbf80_8000,
        0xbf81_8000,
        0x0000_0001,
        0x007f_ffff,
        0x0000_8000,
        0x0001_8000,
        0x8000_0001,
        0x0080_0000,
        0x4800_0000,
        0x7f7f_0000,
        0x7f7f_7fff,
        0x7f7f_8000,
        0x7f7f_ffff,
        0xff7f_8000,
        0xff7f_ffff,
        0x7f80_0000,
        0xff80_0000,
        0x7fc0_0000,
        0xffc0_0000,
        0x7fc0_0001,
        0x7f80_0001,
    ]
    .map(f32::from_bits)
    .to_vec()
}

fn flag_buffer(ctx: &CudaContext) -> DeviceBuffer {
    testkit::upload_bytes(ctx, &CLEAR)
}

/// The separate publish kernel over `count` contiguous values of `raw`.
fn publish_separately(
    ctx: &CudaContext,
    stream: &DeviceStream,
    raw: &DeviceBuffer,
    flag: &DeviceBuffer,
    (rows, columns): (usize, usize),
) -> DeviceBuffer {
    let bytes = rows * columns * 4;
    let published = testkit::upload_filled(ctx, 0xa5, bytes);
    ctx.publish_bf16_rows(
        stream,
        Bf16PublishCall {
            x: raw.range(0, bytes).unwrap(),
            out: published.range(0, bytes).unwrap(),
            flag: flag.range(0, 4).unwrap(),
            rows,
            columns,
        },
    )
    .unwrap();
    published
}

/// Both arms must have exercised the contract: some value rounded, some value flagged.
fn assert_sweep_was_live(raw: &[u8], published: &[u8], flags: [&[u8]; 2], label: &str) {
    assert_eq!(flags[0], flags[1], "{label}: flag differs between arms");
    assert!(
        testkit::floats(raw)
            .iter()
            .zip(testkit::floats(published))
            .any(|(raw, published)| raw.is_finite() && raw.to_bits() != published.to_bits()),
        "{label}: the sweep never saw a value the roundtrip changes"
    );
}

fn row_add_call<'a>(
    out: &'a DeviceBuffer,
    [x, y]: [&'a DeviceBuffer; 2],
    flag: Option<&'a DeviceBuffer>,
    n: usize,
    fused: bool,
) -> RowAddCall<'a> {
    let bytes = n * 4;
    RowAddCall {
        out: out.range(0, bytes).unwrap(),
        x: x.range(0, bytes).unwrap(),
        y: y.range(0, bytes).unwrap(),
        n,
        flag: flag.map(|flag| flag.range(0, 4).unwrap()),
        fused,
    }
}

#[test]
fn fused_row_add_equals_row_add_then_publish_bitwise() {
    if !testkit::device_present("fused row_add sweep") {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let (mut x, mut y) = (Vec::new(), Vec::new());
    for &a in &edge_values() {
        for b in [0.0, a, -a, 1.0, f32::MAX] {
            x.push(a);
            y.push(b);
        }
    }
    while !x.len().is_multiple_of(32) {
        x.push(0.0);
        y.push(0.0);
    }
    let (n, bytes) = (x.len(), x.len() * 4);
    let (x, y) = (testkit::upload_f32(&ctx, &x), testkit::upload_f32(&ctx, &y));
    let raw = testkit::upload_filled(&ctx, 0xa5, bytes);
    ctx.row_add(&stream, row_add_call(&raw, [&x, &y], None, n, false))
        .unwrap();
    let reference_flag = flag_buffer(&ctx);
    let published = publish_separately(&ctx, &stream, &raw, &reference_flag, (1, n));
    let fused = testkit::upload_filled(&ctx, 0xa5, bytes);
    let fused_flag = flag_buffer(&ctx);
    ctx.row_add(
        &stream,
        row_add_call(&fused, [&x, &y], Some(&fused_flag), n, true),
    )
    .unwrap();
    stream.synchronize().unwrap();
    let (raw, published, fused) = (
        testkit::read_bytes(&ctx, &raw, bytes),
        testkit::read_bytes(&ctx, &published, bytes),
        testkit::read_bytes(&ctx, &fused, bytes),
    );
    assert_eq!(fused, published, "fused row_add output differs");
    let flags = [
        testkit::read_bytes(&ctx, &reference_flag, 4),
        testkit::read_bytes(&ctx, &fused_flag, 4),
    ];
    assert_eq!(flags[0], STICKY, "the sweep must flag its nonfinite sums");
    assert_sweep_was_live(&raw, &published, [&flags[0], &flags[1]], "row_add");
}

#[test]
fn fused_row_multiply_equals_row_multiply_then_publish_bitwise_and_off_is_unrounded() {
    if !testkit::device_present("fused row_multiply sweep") {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let edges = edge_values();
    let side = edges.len();
    let shape = RowShape {
        rows: side,
        columns: side,
    };
    let bytes = side * side * 4;
    // Cross product: column c of every row holds edges[c]; row r multiplies by edges[r].
    let x: Vec<f32> = (0..side * side).map(|i| edges[i % side]).collect();
    let full: Vec<f32> = (0..side * side).map(|i| edges[i / side]).collect();
    let x = testkit::upload_f32(&ctx, &x);
    for rhs_columns in [side, 1] {
        let rhs_values: Vec<f32> = if rhs_columns == 1 {
            edges.clone()
        } else {
            full.clone()
        };
        let rhs = testkit::upload_f32(&ctx, &rhs_values);
        let rhs_bytes = rhs_values.len() * 4;
        let run = |out: &DeviceBuffer, flag: &DeviceBuffer, fused| {
            ctx.row_multiply(
                &stream,
                RowMulCall {
                    x: x.range(0, bytes).unwrap(),
                    rhs: rhs.range(0, rhs_bytes).unwrap(),
                    out: out.range(0, bytes).unwrap(),
                    flag: flag.range(0, 4).unwrap(),
                    shape,
                    rhs_columns,
                    fused,
                },
            )
            .unwrap();
        };
        // The unfused arm shares one flag between the multiply and the publish, as the
        // walk's frame does.
        let (raw, flag) = (testkit::upload_filled(&ctx, 0xa5, bytes), flag_buffer(&ctx));
        run(&raw, &flag, false);
        let published = publish_separately(&ctx, &stream, &raw, &flag, (side, side));
        let (fused, fused_flag) = (testkit::upload_filled(&ctx, 0xa5, bytes), flag_buffer(&ctx));
        run(&fused, &fused_flag, true);
        stream.synchronize().unwrap();
        let label = format!("row_multiply rhs_columns={rhs_columns}");
        let (raw_bytes, published, fused) = (
            testkit::read_bytes(&ctx, &raw, bytes),
            testkit::read_bytes(&ctx, &published, bytes),
            testkit::read_bytes(&ctx, &fused, bytes),
        );
        assert_eq!(fused, published, "{label}: fused output differs");
        // The unfused arm is the pre-existing arithmetic: the plain f32 product, or
        // zero for a nonfinite input or product. Forcing the arm on for a caller that
        // never asked for it (the prefill path) would break exactly this.
        for (index, got) in testkit::floats(&raw_bytes).into_iter().enumerate() {
            let lhs = edges[index % side];
            let rhs = if rhs_columns == 1 {
                rhs_values[index / side]
            } else {
                rhs_values[index]
            };
            let product = lhs * rhs;
            let want = if lhs.is_finite() && rhs.is_finite() && product.is_finite() {
                product
            } else {
                0.0
            };
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "{label}: unfused element {index}"
            );
        }
        let flags = [
            testkit::read_bytes(&ctx, &flag, 4),
            testkit::read_bytes(&ctx, &fused_flag, 4),
        ];
        assert_eq!(
            flags[0], STICKY,
            "{label}: the sweep must flag nonfinite products"
        );
        assert_sweep_was_live(&raw_bytes, &published, [&flags[0], &flags[1]], &label);
    }
}

/// One `rows`-row projection of a fixture matrix, by the entry the arm is exercised on.
struct Projection<'a> {
    ctx: &'a CudaContext,
    stream: &'a DeviceStream,
    fixture: (Codebook, u8, bool),
    shape: (usize, usize),
}

impl Projection<'_> {
    /// Returns (raw out, published out, flag) for the unfused arm followed by the
    /// separate publish, and (fused out, flag) for the fused arm.
    fn run(
        &self,
        rows: usize,
        split_k: u32,
        midrows: bool,
        input: &[f32],
        bias: Option<&[f32]>,
    ) -> [Vec<u8>; 5] {
        let (ctx, stream) = (self.ctx, self.stream);
        let (k, n) = self.shape;
        let (book, bits, subnormal) = self.fixture;
        let matrix = fixture(subnormal, book, bits, (k, n));
        let trellis = matrix.trellis().host_bytes().unwrap();
        let packed = testkit::upload_bytes(ctx, trellis);
        let suh = upload_scales(ctx, matrix.suh());
        let svh = upload_scales(ctx, matrix.svh());
        let x = testkit::upload_f32(ctx, input);
        let bias = bias.map(|values| testkit::upload_f32(ctx, values));
        let banks = if rows <= 8 { split_k as usize - 1 } else { 1 };
        let z_bytes = rows * n * 4 * banks;
        let out_bytes = rows * n * 4;
        let arm = |fused: bool| {
            // Each arm owns its workspaces so one cannot inherit the other's partials.
            let xh = testkit::upload_filled(ctx, 0xff, rows * k * 4);
            let z = testkit::upload_filled(ctx, 0xa5, z_bytes + 8);
            let out = testkit::upload_filled(ctx, 0xa5, out_bytes);
            let flag = flag_buffer(ctx);
            let call = Exl3LinearCall {
                trellis: packed.range(0, trellis.len()).unwrap(),
                suh: suh.range(0, k * 4).unwrap(),
                svh: svh.range(0, n * 4).unwrap(),
                bias: bias.as_ref().map(|bias| bias.range(0, n * 4).unwrap()),
                x: x.range(0, rows * k * 4).unwrap(),
                xh: xh.range(0, rows * k * 4).unwrap(),
                z: z.range(4, z_bytes).unwrap(),
                out: out.range(0, out_bytes).unwrap(),
                flag: flag.range(0, 4).unwrap(),
                rows,
                split_k,
                k,
                n,
                bits_twice: matrix.bitrate_twice(),
                book: matrix.codebook(),
                fused,
            };
            if midrows {
                ctx.exl3_linear_midrows(stream, call)
            } else {
                ctx.exl3_linear(stream, call)
            }
            .unwrap();
            (out, flag)
        };
        let (raw, raw_flag) = arm(false);
        let published = publish_separately(ctx, stream, &raw, &raw_flag, (rows, n));
        let (fused, fused_flag) = arm(true);
        stream.synchronize().unwrap();
        [
            testkit::read_bytes(ctx, &raw, out_bytes),
            testkit::read_bytes(ctx, &published, out_bytes),
            testkit::read_bytes(ctx, &raw_flag, 4),
            testkit::read_bytes(ctx, &fused, out_bytes),
            testkit::read_bytes(ctx, &fused_flag, 4),
        ]
    }
}

fn pattern(count: usize, amplitude: f32) -> Vec<f32> {
    (0..count)
        .map(|i| ((i * 17 % 251) as f32 - 125.0) / 1024.0 * amplitude)
        .collect()
}

#[test]
fn fused_exl3_projection_equals_projection_then_publish_bitwise() {
    if !testkit::device_present("fused EXL3 projection sweep") {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let mut rounded = 0usize;
    let mut flagged = 0usize;
    let mut cases = 0usize;
    for (fixture, shape) in [
        ((Codebook::ThreeInst, 4u8, false), (256usize, 128usize)),
        ((Codebook::Mcg, 12, false), (384, 128)),
        ((Codebook::Mul1, 3, true), (256, 128)),
        ((Codebook::Mul1, 8, false), (256, 128)),
    ] {
        let projection = Projection {
            ctx: &ctx,
            stream: &stream,
            fixture,
            shape,
        };
        let k = shape.0;
        // Rows 1..=8 on the small-row entry, then above the grouped-pack width: the
        // generic entry and the opt-in 9..=144 entry (tensor cores from 64 rows).
        let entries: Vec<(usize, u32, bool)> = (1..=8)
            .map(|rows| (rows, 2, false))
            .chain([(1, 4, false), (8, 8, false), (9, 2, false), (17, 2, false)])
            .chain([9, 17, 33, 64, 100, 144].map(|rows| (rows, 2, true)))
            .collect();
        for (rows, split_k, midrows) in entries {
            for (name, mut input) in [
                ("normal", pattern(rows * k, 1.0)),
                ("small", pattern(rows * k, 1.0 / 4096.0)),
                ("large", pattern(rows * k, 1.0e30)),
                ("subnormal", pattern(rows * k, 1.0e-39)),
                ("nan", pattern(rows * k, 1.0)),
                ("inf", pattern(rows * k, 1.0)),
                ("overflow", pattern(rows * k, 3.0e38)),
            ] {
                match name {
                    "nan" => input[3] = f32::NAN,
                    "inf" => input[5] = f32::INFINITY,
                    _ => {}
                }
                let [raw, published, raw_flag, fused, fused_flag] =
                    projection.run(rows, split_k, midrows, &input, None);
                let label = format!("{fixture:?} rows={rows} split={split_k} mid={midrows} {name}");
                assert_eq!(fused, published, "{label}: fused output differs");
                assert_eq!(raw_flag, fused_flag, "{label}: flag differs");
                cases += 1;
                flagged += usize::from(fused_flag == STICKY);
                rounded += testkit::floats(&raw)
                    .iter()
                    .zip(testkit::floats(&published))
                    .filter(|(raw, published)| {
                        raw.is_finite() && raw.to_bits() != published.to_bits()
                    })
                    .count();
            }
        }
    }
    assert!(
        cases > 0 && rounded > 0,
        "the sweep never saw a value the roundtrip changes"
    );
    assert!(flagged > 0, "the sweep never saw a flagged projection");
}

const GROUPED_K: usize = 384;
const GROUPED_N: usize = 128;
const GROUPED_MIXED: [(Codebook, u8, usize); 4] = [
    (Codebook::ThreeInst, 4, 8),
    (Codebook::Mcg, 10, 8),
    (Codebook::Mul1, 3, 1),
    (Codebook::Mul1, 8, 2),
];

/// One grouped launch over GROUPED_MIXED, unfused then published separately, and fused.
/// Returns [raw out, published out, unfused flag, fused out, fused flag].
fn grouped_pair(
    ctx: &CudaContext,
    stream: &DeviceStream,
    gather: bool,
    input: &[f32],
    biases: &[Option<Vec<f32>>],
) -> [Vec<u8>; 5] {
    let (k, n) = (GROUPED_K, GROUPED_N);
    let total: usize = GROUPED_MIXED.iter().map(|group| group.2).sum();
    let x = testkit::upload_f32(ctx, input);
    let banks: Vec<_> = GROUPED_MIXED
        .iter()
        .zip(biases)
        .map(|(&(book, bits, _), bias)| {
            let matrix = fixture(false, book, bits, (k, n));
            let bytes = matrix.trellis().host_bytes().unwrap();
            (
                testkit::upload_bytes(ctx, bytes),
                upload_scales(ctx, matrix.suh()),
                upload_scales(ctx, matrix.svh()),
                bias.as_ref().map(|values| testkit::upload_f32(ctx, values)),
            )
        })
        .collect();
    let arm = |fused: bool| {
        let xh = testkit::upload_filled(ctx, 0xff, total * k * 4);
        let z = testkit::upload_filled(ctx, 0xa5, total * n * 4 + 8);
        let out = testkit::upload_filled(ctx, 0xa5, total * n * 4);
        let flag = flag_buffer(ctx);
        let mut first = 0usize;
        let calls: Vec<_> = banks
            .iter()
            .zip(&GROUPED_MIXED)
            .map(|((packed, suh, svh, bias), &(book, bits, rows))| {
                let offset = first;
                first += rows;
                Exl3LinearCall {
                    trellis: packed
                        .range(0, k * n / 256 * usize::from(bits) * 16)
                        .unwrap(),
                    suh: suh.range(0, k * 4).unwrap(),
                    svh: svh.range(0, n * 4).unwrap(),
                    bias: bias.as_ref().map(|bias| bias.range(0, n * 4).unwrap()),
                    x: x.range(offset * k * 4, rows * k * 4).unwrap(),
                    xh: xh.range(offset * k * 4, rows * k * 4).unwrap(),
                    z: z.range(4 + offset * n * 4, rows * n * 4).unwrap(),
                    out: out.range(offset * n * 4, rows * n * 4).unwrap(),
                    flag: flag.range(0, 4).unwrap(),
                    rows,
                    split_k: 2,
                    k,
                    n,
                    bits_twice: bits,
                    book,
                    fused,
                }
            })
            .collect();
        ctx.exl3_grouped(stream, &calls, gather).unwrap();
        (out, flag)
    };
    let (raw, raw_flag) = arm(false);
    let published = publish_separately(ctx, stream, &raw, &raw_flag, (total, n));
    let (fused, fused_flag) = arm(true);
    stream.synchronize().unwrap();
    let bytes = total * n * 4;
    [
        testkit::read_bytes(ctx, &raw, bytes),
        testkit::read_bytes(ctx, &published, bytes),
        testkit::read_bytes(ctx, &raw_flag, 4),
        testkit::read_bytes(ctx, &fused, bytes),
        testkit::read_bytes(ctx, &fused_flag, 4),
    ]
}

#[test]
fn fused_grouped_projection_equals_grouped_then_publish_bitwise() {
    if !testkit::device_present("fused EXL3 grouped sweep") {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let total: usize = GROUPED_MIXED.iter().map(|group| group.2).sum();
    let biases = vec![None; GROUPED_MIXED.len()];
    for gather in [false, true] {
        for (name, amplitude) in [
            ("normal", 1.0f32),
            ("large", 1.0e30),
            ("subnormal", 1.0e-39),
        ] {
            let input = pattern(total * GROUPED_K, amplitude);
            let [raw, published, raw_flag, fused, fused_flag] =
                grouped_pair(&ctx, &stream, gather, &input, &biases);
            let label = format!("grouped gather={gather} {name}");
            assert_eq!(fused, published, "{label}: fused output differs");
            assert_eq!(raw_flag, fused_flag, "{label}: flag differs");
            if name == "normal" {
                assert_sweep_was_live(&raw, &published, [&raw_flag, &fused_flag], &label);
            }
        }
    }
}

/// Final-publication values the zero-input projections isolate: the output of a zero
/// projection is exactly its bias, so one bias element is the only candidate for a flag.
fn exceptional_biases() -> Vec<(&'static str, f32)> {
    vec![
        ("nan", f32::NAN),
        ("inf", f32::INFINITY),
        ("-inf", f32::NEG_INFINITY),
        ("f32-max", f32::MAX),
        ("overflow-tie", f32::from_bits(0x7f7f_8000)),
        ("-overflow-tie", f32::from_bits(0xff7f_8000)),
        ("bf16-max", f32::from_bits(0x7f7f_7fff)),
        ("rne-tie", f32::from_bits(0x3f80_8000)),
        ("subnormal", 1.0e-40),
        ("one", 1.0),
    ]
}

/// What the separate publish kernel stores for a finite-or-not value, and whether it flags.
fn published_value(value: f32) -> (f32, bool) {
    let rounded = half::bf16::from_f32(value).to_f32();
    if value.is_finite() && rounded.is_finite() {
        (rounded, false)
    } else {
        (0.0, true)
    }
}

fn assert_isolated_element(
    out: &[u8],
    flag: &[u8],
    element: &[usize],
    want: (f32, bool),
    label: &str,
) {
    for (index, value) in testkit::floats(out).into_iter().enumerate() {
        if element.contains(&index) {
            assert_eq!(
                value.to_bits(),
                want.0.to_bits(),
                "{label}: element {index}"
            );
        } else {
            assert_eq!(value, 0.0, "{label}: element {index} must stay zero");
        }
    }
    assert_eq!(flag, if want.1 { STICKY } else { CLEAR }, "{label}: flag");
}

#[test]
fn fused_projection_publishes_exceptional_biases_like_the_separate_kernel() {
    if !testkit::device_present("fused EXL3 exceptional final publication") {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let (k, n, column) = (256usize, 128usize, 7usize);
    let projection = Projection {
        ctx: &ctx,
        stream: &stream,
        fixture: (Codebook::Mul1, 8, false),
        shape: (k, n),
    };
    for (name, bias) in exceptional_biases() {
        let mut values = vec![0.0f32; n];
        values[column] = bias;
        let input = vec![0.0f32; k];
        let [_, published, raw_flag, fused, fused_flag] =
            projection.run(1, 2, false, &input, Some(&values));
        let label = format!("single {name}");
        let want = published_value(bias);
        assert_eq!(fused, published, "{label}: fused output differs");
        assert_isolated_element(&fused, &fused_flag, &[column], want, &label);
        assert_eq!(raw_flag, fused_flag, "{label}: unfused flag differs");
    }
}

#[test]
fn fused_grouped_projection_publishes_exceptional_biases_like_the_separate_kernel() {
    if !testkit::device_present("fused EXL3 grouped exceptional final publication") {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let (k, n, column) = (GROUPED_K, GROUPED_N, 7usize);
    let total: usize = GROUPED_MIXED.iter().map(|group| group.2).sum();
    let first_group_rows = GROUPED_MIXED[0].2;
    for gather in [false, true] {
        for (name, bias) in exceptional_biases() {
            // Only the first group's column carries the value; every other bias is zero.
            let mut biases = vec![Some(vec![0.0f32; n]); GROUPED_MIXED.len()];
            biases[0].as_mut().unwrap()[column] = bias;
            let input = vec![0.0f32; total * k];
            let [_, published, raw_flag, fused, fused_flag] =
                grouped_pair(&ctx, &stream, gather, &input, &biases);
            let label = format!("grouped gather={gather} {name}");
            let want = published_value(bias);
            assert_eq!(fused, published, "{label}: fused output differs");
            let elements: Vec<usize> = (0..first_group_rows).map(|row| row * n + column).collect();
            assert_isolated_element(&fused, &fused_flag, &elements, want, &label);
            assert_eq!(raw_flag, fused_flag, "{label}: unfused flag differs");
        }
    }
}

/// A grouped set that mixes fused and unfused calls is refused with BadRequest before the
/// native entry, which would otherwise reject it and latch the backend terminal: an
/// otherwise-valid grouped call in the same context still succeeds afterwards.
#[test]
fn grouped_set_mixing_fused_arms_is_refused_without_latching_the_backend() {
    if !testkit::device_present("fused EXL3 grouped mixed arms") {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let (k, n) = (GROUPED_K, GROUPED_N);
    let total: usize = GROUPED_MIXED.iter().map(|group| group.2).sum();
    let input = pattern(total * k, 1.0);
    let x = testkit::upload_f32(&ctx, &input);
    let banks: Vec<_> = GROUPED_MIXED
        .iter()
        .map(|&(book, bits, _)| {
            let matrix = fixture(false, book, bits, (k, n));
            let bytes = matrix.trellis().host_bytes().unwrap();
            (
                testkit::upload_bytes(&ctx, bytes),
                upload_scales(&ctx, matrix.suh()),
                upload_scales(&ctx, matrix.svh()),
            )
        })
        .collect();
    let xh = testkit::upload_filled(&ctx, 0xff, total * k * 4);
    let z = testkit::upload_filled(&ctx, 0xa5, total * n * 4 + 8);
    let out = testkit::upload_filled(&ctx, 0xa5, total * n * 4);
    let flag = flag_buffer(&ctx);
    let mut first = 0usize;
    // Everything but the arm is valid: the second call alone is fused.
    let calls: Vec<_> = banks
        .iter()
        .zip(&GROUPED_MIXED)
        .enumerate()
        .map(|(index, ((packed, suh, svh), &(book, bits, rows)))| {
            let offset = first;
            first += rows;
            Exl3LinearCall {
                trellis: packed
                    .range(0, k * n / 256 * usize::from(bits) * 16)
                    .unwrap(),
                suh: suh.range(0, k * 4).unwrap(),
                svh: svh.range(0, n * 4).unwrap(),
                bias: None,
                x: x.range(offset * k * 4, rows * k * 4).unwrap(),
                xh: xh.range(offset * k * 4, rows * k * 4).unwrap(),
                z: z.range(4 + offset * n * 4, rows * n * 4).unwrap(),
                out: out.range(offset * n * 4, rows * n * 4).unwrap(),
                flag: flag.range(0, 4).unwrap(),
                rows,
                split_k: 2,
                k,
                n,
                bits_twice: bits,
                book,
                fused: index == 1,
            }
        })
        .collect();
    let refused = ctx
        .exl3_grouped(&stream, &calls, false)
        .expect_err("a mixed-arm set must be refused");
    assert!(
        matches!(&refused, crate::EngError::BadRequest(text) if text.contains("fused-publish arm")),
        "{refused:?}"
    );
    stream.synchronize().unwrap();
    assert_eq!(testkit::read_bytes(&ctx, &flag, 4), CLEAR, "nothing ran");
    // The same context still serves a valid grouped call, in both arms.
    let biases = vec![None; GROUPED_MIXED.len()];
    for gather in [false, true] {
        let [_, published, raw_flag, fused, fused_flag] =
            grouped_pair(&ctx, &stream, gather, &input, &biases);
        assert_eq!(fused, published, "gather={gather}");
        assert_eq!(raw_flag, fused_flag, "gather={gather}");
    }
}
