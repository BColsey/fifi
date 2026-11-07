//! Device tests (3090) of the dense half-precision matvec: the Rust wrapper against the C entry
//! against the kernel of `native/resident/dense_gemv.cu`, compared with an f64 host reference.
//!
//! Weights are drawn as f32 and rounded to binary16 or bfloat16 by the `half` crate; the reference
//! widens the same bits, so a comparison measures the kernel and not the dtype. Each device value
//! must lie within twice a first-order rounding bound of the f32 dot product, computed per value
//! from the magnitudes of its terms (a lane adds `cols / 32` terms, then five butterfly levels).
//! Every buffer carries 1 KiB of sentinel bytes around the used bytes and every `y` is
//! pre-filled with a NaN pattern: elements the call must not write (the stride padding) must keep
//! it, and the inputs must be unchanged. A result must not depend on `m`: token `t` of a call of
//! `m` rows has the bits of the same row run alone. Each refusal is checked by its error text and
//! must leave the output untouched. The suite runs with `--test-threads=1`.

use super::{DenseDtype, DenseGemvCall, DenseGemvShape, dense_gemv_bytes};
use crate::EngError;
use crate::kernels::cuda::testkit::{self, Rng};
use crate::kernels::cuda::{CudaContext, DeviceBuffer, DeviceRange, DeviceStream};

/// Unit roundoff of f32.
const U: f64 = 1.0 / 16_777_216.0;
const SLACK: f64 = 2.0;
const PAD: usize = 1024;
const SENT: [u8; 4] = 0x7fc0_dead_u32.to_le_bytes();

struct Rig {
    ctx: CudaContext,
    stream: DeviceStream,
}

impl Rig {
    fn new() -> Self {
        let ctx = CudaContext::new().expect("context");
        let stream = ctx.stream().expect("stream");
        Self { ctx, stream }
    }
}

fn sentinel(bytes: usize) -> Vec<u8> {
    SENT.iter().copied().cycle().take(bytes).collect()
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Device bytes between two sentinels; `extra` shifts the first data byte past the allocation
/// start to test alignments.
struct Padded {
    buf: DeviceBuffer,
    lead: usize,
    len: usize,
}

impl Padded {
    fn new(ctx: &CudaContext, data: &[u8], extra: usize) -> Self {
        let lead = PAD + extra;
        let mut bytes = sentinel(lead);
        bytes.extend_from_slice(data);
        bytes.extend(sentinel(PAD));
        Self {
            buf: testkit::upload_bytes(ctx, &bytes),
            lead,
            len: data.len(),
        }
    }

    fn range(&self, skip: usize, bytes: usize) -> DeviceRange<'_> {
        self.buf.range(self.lead + skip, bytes).expect("range")
    }

    fn all(&self) -> DeviceRange<'_> {
        self.range(0, self.len)
    }

    /// The data bytes; the sentinels around them must be intact.
    fn read(&self, ctx: &CudaContext, what: &str) -> Vec<u8> {
        let all = testkit::read_bytes(ctx, &self.buf, self.lead + self.len + PAD);
        let (lead, rest) = all.split_at(self.lead);
        let (data, trail) = rest.split_at(self.len);
        assert!(
            lead == sentinel(self.lead),
            "{what}: sentinel before changed"
        );
        assert!(trail == sentinel(PAD), "{what}: sentinel after changed");
        data.to_vec()
    }
}

fn narrow(dtype: DenseDtype, values: &[f32]) -> Vec<u16> {
    values
        .iter()
        .map(|&v| match dtype {
            DenseDtype::F16 => half::f16::from_f32(v).to_bits(),
            DenseDtype::Bf16 => half::bf16::from_f32(v).to_bits(),
        })
        .collect()
}

fn widen(dtype: DenseDtype, codes: &[u16]) -> Vec<f64> {
    codes
        .iter()
        .map(|&c| match dtype {
            DenseDtype::F16 => half::f16::from_bits(c).to_f64(),
            DenseDtype::Bf16 => half::bf16::from_bits(c).to_f64(),
        })
        .collect()
}

fn code_bytes(codes: &[u16]) -> Vec<u8> {
    codes.iter().flat_map(|c| c.to_le_bytes()).collect()
}

fn draw(rng: &mut Rng, n: usize, sd: f64) -> Vec<f32> {
    (0..n).map(|_| (rng.normal() * sd) as f32).collect()
}

struct Case {
    shape: DenseGemvShape,
    dtype: DenseDtype,
    codes: Vec<u16>,
    x: Vec<f32>,
}

impl Case {
    fn new(rng: &mut Rng, shape: DenseGemvShape, dtype: DenseDtype) -> Self {
        let DenseGemvShape {
            m,
            rows,
            cols,
            ld_x,
            ..
        } = shape;
        let weights = draw(rng, rows * cols, 1.0 / (cols as f64).sqrt());
        // Rows of x sit at stride ld_x; the padding between rows holds values the call must ignore.
        let mut x = vec![f32::from_bits(0x7fc0_beef); (m - 1) * ld_x + cols];
        for t in 0..m {
            for (c, v) in draw(rng, cols, 1.0).into_iter().enumerate() {
                x[t * ld_x + c] = v;
            }
        }
        Self {
            shape,
            dtype,
            codes: narrow(dtype, &weights),
            x,
        }
    }
}

/// `(sum, sum of |terms|)` of one result in f64.
fn reference(c: &Case, t: usize, r: usize) -> (f64, f64) {
    let cols = c.shape.cols;
    let w = widen(c.dtype, &c.codes[r * cols..(r + 1) * cols]);
    w.iter()
        .zip(&c.x[t * c.shape.ld_x..t * c.shape.ld_x + cols])
        .fold((0.0, 0.0), |(s, a), (w, v)| {
            let term = w * f64::from(*v);
            (s + term, a + term.abs())
        })
}

fn run(rig: &Rig, c: &Case, (w_extra, x_extra, y_extra): (usize, usize, usize)) -> Vec<f32> {
    let w = Padded::new(&rig.ctx, &code_bytes(&c.codes), w_extra);
    let x = Padded::new(&rig.ctx, &f32_bytes(&c.x), x_extra);
    let need = dense_gemv_bytes(c.shape).expect("bytes");
    let y = Padded::new(&rig.ctx, &sentinel(need.y), y_extra);
    let call = DenseGemvCall {
        shape: c.shape,
        dtype: c.dtype,
        w: w.all(),
        x: x.all(),
        y: y.all(),
    };
    rig.ctx.dense_gemv(&rig.stream, call).expect("a valid call");
    rig.stream.synchronize().expect("drain");
    assert!(w.read(&rig.ctx, "w") == code_bytes(&c.codes), "w changed");
    assert!(x.read(&rig.ctx, "x") == f32_bytes(&c.x), "x changed");
    testkit::floats(&y.read(&rig.ctx, "y"))
}

/// Checks every written element against the f64 reference and every padding element for the
/// sentinel; returns the largest `|d - ref| / bound`.
fn check(c: &Case, y: &[f32]) -> f64 {
    let s = c.shape;
    let depth = (s.cols.div_ceil(32) + 6) as f64;
    let mut worst = 0.0f64;
    for t in 0..s.m {
        for r in 0..s.rows {
            let (value, abs) = reference(c, t, r);
            let d = y[t * s.ld_y + r];
            let bound = (depth + 1.0) * U * abs;
            let ratio = (f64::from(d) - value).abs() / bound.max(f64::MIN_POSITIVE);
            assert!(
                d.is_finite() && ratio <= SLACK,
                "[{t}][{r}] of {s:?}: device {d} reference {value} bound {bound} ratio {ratio}"
            );
            worst = worst.max(ratio);
        }
        if t + 1 < s.m {
            for p in s.rows..s.ld_y {
                assert_eq!(
                    y[t * s.ld_y + p].to_bits(),
                    SENT_BITS,
                    "padding [{t}][{p}] of {s:?} was written"
                );
            }
        }
    }
    worst
}

const SENT_BITS: u32 = 0x7fc0_dead;

#[test]
fn the_matvec_matches_the_f64_reference_at_the_ple_projections() {
    if !testkit::device_present("the_matvec_matches_the_f64_reference_at_the_ple_projections") {
        return;
    }
    let rig = Rig::new();
    let mut rng = Rng::new(0xD3_5E_6E_01);
    // The one user: the PLE key projection `[10240, 2560]` and value projection `[2560, 2560]` of
    // the layer with the PLE block, F16, one token a call (a few for a prefill piece).
    for shape in [
        DenseGemvShape::packed(1, 10_240, 2560),
        DenseGemvShape::packed(1, 2560, 2560),
        DenseGemvShape::packed(4, 10_240, 2560),
        DenseGemvShape::packed(4, 2560, 2560),
    ] {
        let case = Case::new(&mut rng, shape, DenseDtype::F16);
        let worst = check(&case, &run(&rig, &case, (0, 0, 0)));
        println!("F16 {shape:?}: worst error {worst:.3} of the bound");
    }
}

#[test]
fn the_edge_shapes_of_the_kernel_agree_with_the_f64_reference() {
    if !testkit::device_present("the_edge_shapes_of_the_kernel_agree_with_the_f64_reference") {
        return;
    }
    let rig = Rig::new();
    let mut rng = Rng::new(0xD3_5E_6E_05);
    // Strides wider than the rows (the padding between activation rows holds NaN the call must not
    // read, and the result rows keep NaN padding), a column count off the 8-column vector path, the
    // most columns the kernel stages, a single element, and bfloat16 weights.
    let cases = [
        (
            DenseGemvShape {
                m: 3,
                rows: 48,
                cols: 2560,
                ld_x: 2688,
                ld_y: 21_728,
            },
            DenseDtype::F16,
        ),
        (DenseGemvShape::packed(2, 7, 100), DenseDtype::F16),
        (DenseGemvShape::packed(1, 9, 12_288), DenseDtype::F16),
        (DenseGemvShape::packed(1, 1, 1), DenseDtype::F16),
        (DenseGemvShape::packed(2, 48, 2560), DenseDtype::Bf16),
    ];
    for (shape, dtype) in cases {
        let case = Case::new(&mut rng, shape, dtype);
        let worst = check(&case, &run(&rig, &case, (0, 0, 0)));
        println!("{dtype:?} {shape:?}: worst error {worst:.3} of the bound");
    }
}

#[test]
fn a_weight_off_the_16_byte_grid_takes_the_scalar_path_and_agrees() {
    if !testkit::device_present("a_weight_off_the_16_byte_grid_takes_the_scalar_path_and_agrees") {
        return;
    }
    let rig = Rig::new();
    let mut rng = Rng::new(0xD3_5E_6E_02);
    let case = Case::new(
        &mut rng,
        DenseGemvShape::packed(2, 48, 2560),
        DenseDtype::F16,
    );
    // Weight at offset 2 (2-byte aligned only), x and y at 4.
    check(&case, &run(&rig, &case, (2, 4, 4)));
    check(&case, &run(&rig, &case, (0, 0, 0)));
}

#[test]
fn a_result_does_not_depend_on_the_number_of_rows_of_the_call() {
    if !testkit::device_present("a_result_does_not_depend_on_the_number_of_rows_of_the_call") {
        return;
    }
    let rig = Rig::new();
    let mut rng = Rng::new(0xD3_5E_6E_03);
    let many = Case::new(
        &mut rng,
        DenseGemvShape::packed(5, 48, 2560),
        DenseDtype::F16,
    );
    let together = run(&rig, &many, (0, 0, 0));
    for t in 0..5 {
        let alone = Case {
            shape: DenseGemvShape::packed(1, 48, 2560),
            dtype: many.dtype,
            codes: many.codes.clone(),
            x: many.x[t * 2560..(t + 1) * 2560].to_vec(),
        };
        let single = run(&rig, &alone, (0, 0, 0));
        let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
        assert_eq!(
            bits(&single),
            bits(&together[t * 48..(t + 1) * 48]),
            "row {t}"
        );
    }
}

fn refused(rig: &Rig, call: DenseGemvCall<'_>, y: &Padded, expect: &str) {
    let error = rig
        .ctx
        .dense_gemv(&rig.stream, call)
        .expect_err("a refused call");
    let text = match error {
        EngError::BadRequest(text) => text,
        other => panic!("expected BadRequest, got {other:?}"),
    };
    assert!(text.contains(expect), "{text:?} lacks {expect:?}");
    rig.stream.synchronize().expect("drain");
    assert!(
        y.read(&rig.ctx, "y") == sentinel(y.len),
        "a refused call wrote y"
    );
}

#[test]
fn bad_arguments_are_refused_before_any_launch() {
    if !testkit::device_present("bad_arguments_are_refused_before_any_launch") {
        return;
    }
    let rig = Rig::new();
    let mut rng = Rng::new(0xD3_5E_6E_04);
    let shape = DenseGemvShape::packed(2, 16, 128);
    let case = Case::new(&mut rng, shape, DenseDtype::F16);
    let w = Padded::new(&rig.ctx, &code_bytes(&case.codes), 0);
    let x = Padded::new(&rig.ctx, &f32_bytes(&case.x), 0);
    let need = dense_gemv_bytes(shape).unwrap();
    let y = Padded::new(&rig.ctx, &sentinel(need.y), 0);
    let ok = |shape: DenseGemvShape| DenseGemvCall {
        shape,
        dtype: DenseDtype::F16,
        w: w.all(),
        x: x.all(),
        y: y.all(),
    };
    // Short output, short input, short weight.
    refused(
        &rig,
        DenseGemvCall {
            y: y.range(0, need.y - 4),
            ..ok(shape)
        },
        &y,
        "the call needs",
    );
    refused(
        &rig,
        DenseGemvCall {
            x: x.range(0, need.x - 4),
            ..ok(shape)
        },
        &y,
        "the call needs",
    );
    refused(
        &rig,
        DenseGemvCall {
            w: w.range(0, need.w - 2),
            ..ok(shape)
        },
        &y,
        "the call needs",
    );
    // Misaligned x and y (offset 2 is not a multiple of 4).
    refused(
        &rig,
        DenseGemvCall {
            x: x.range(2, need.x),
            ..ok(shape)
        },
        &y,
        "multiple of 4",
    );
    refused(
        &rig,
        DenseGemvCall {
            y: y.range(2, need.y),
            ..ok(shape)
        },
        &y,
        "multiple of 4",
    );
    // Dimensions outside the limits and strides below the row width.
    for bad in [
        DenseGemvShape::packed(0, 16, 128),
        DenseGemvShape::packed(2, 0, 128),
        DenseGemvShape::packed(2, 16, 12_289),
        DenseGemvShape { ld_y: 15, ..shape },
        DenseGemvShape { ld_x: 127, ..shape },
    ] {
        refused(&rig, ok(bad), &y, "the CUDA dense matvec takes");
    }
    // y inside x.
    refused(
        &rig,
        DenseGemvCall {
            y: x.range(0, need.y),
            ..ok(shape)
        },
        &y,
        "overlap",
    );
    // The same call, valid, still runs after the refusals.
    rig.ctx
        .dense_gemv(&rig.stream, ok(shape))
        .expect("valid call");
    rig.stream.synchronize().expect("drain");
}
