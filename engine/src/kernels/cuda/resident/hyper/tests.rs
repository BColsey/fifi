//! Device tests (3090) of the hyper-connection read and apply of Qwen3.8 Flash-Next: the Rust
//! wrapper against the C entries against the kernels of `native/resident/hyper.cu`, compared with
//! an f64 host reference written from the formulas of the Qwen3.8-Next architecture report
//! (arXiv 2608.30320, Eq. 30-34) as `native/resident/hyper.h` states them.
//!
//! The weights are drawn as f32 and rounded to binary16 or bfloat16 by the `half` crate; the
//! reference widens the same bits, so a comparison measures the kernels and not the weight dtype.
//! Each kernel is checked stage by stage: the reference of a stage takes the device output of the
//! stage before it (read back from the caller scratch), and each device value must lie within
//! twice a first-order rounding bound of the f32 arithmetic of that stage, computed per value
//! from the magnitudes of its terms. A whole-read check against the pure f64 chain follows.
//!
//! Every device buffer carries 1 KiB of sentinel bytes before and after the used bytes; the
//! sentinels and the inputs must be unchanged after a launch, and an output still holding the
//! (NaN) sentinel fails its bound. Each refusal is checked by the text of its error and must leave
//! every output untouched. The suite runs with `--test-threads=1`.

use super::{
    HyperApplyCall, HyperDtype, HyperReadCall, HyperScratch, HyperShape, HyperWeights,
    hyper_read_bytes,
};
use crate::EngError;
use crate::kernels::cuda::testkit::{self, Rng};
use crate::kernels::cuda::{CudaContext, DeviceBuffer, DeviceRange, DeviceStream};

/// Unit roundoff of f32.
const U: f64 = 1.0 / 16_777_216.0;
/// Margin over the first-order bound of each stage.
const SLACK: f64 = 2.0;
/// Whole-read class: `max|d| <= 2^-14 * max(max|ref|, 1)` per output vector of a token, against
/// the pure f64 chain.
const WHOLE: f64 = 1.0 / 16_384.0;
/// Largest `|silu'(x)|`, about 1.0998.
const SILU_SLOPE: f64 = 1.1;
const PAD: usize = 1024;
const SENT: [u8; 4] = 0x7fc0_dead_u32.to_le_bytes();
const EPS: f32 = 1e-6;

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

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

/// Device bytes between two sentinels. `extra` moves the first data byte past the 256-byte
/// boundary of the allocation to test alignments.
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

    fn floats(ctx: &CudaContext, values: &[f32], extra: usize) -> Self {
        Self::new(ctx, &f32_bytes(values), extra)
    }

    fn output(ctx: &CudaContext, floats: usize) -> Self {
        Self::new(ctx, &sentinel(floats * 4), 0)
    }

    fn all(&self) -> DeviceRange<'_> {
        self.at(0, self.len)
    }

    /// `bytes` bytes from `skip` bytes past the data start; the trailing sentinel keeps a short
    /// shift inside the allocation.
    fn at(&self, skip: usize, bytes: usize) -> DeviceRange<'_> {
        self.buf.range(self.lead + skip, bytes).expect("range")
    }

    fn read(&self, ctx: &CudaContext, what: &str) -> Vec<u8> {
        let all = testkit::read_bytes(ctx, &self.buf, self.lead + self.len + PAD);
        let (lead, rest) = all.split_at(self.lead);
        let (data, trail) = rest.split_at(self.len);
        assert!(
            lead == sentinel(self.lead),
            "{what}: the sentinel before the data changed"
        );
        assert!(
            trail == sentinel(PAD),
            "{what}: the sentinel after the data changed"
        );
        data.to_vec()
    }

    fn read_f32(&self, ctx: &CudaContext, what: &str) -> Vec<f32> {
        testkit::floats(&self.read(ctx, what))
    }

    fn untouched(&self, ctx: &CudaContext, what: &str) {
        assert!(
            self.read(ctx, what) == sentinel(self.len),
            "{what} was written"
        );
    }
}

fn narrow(dtype: HyperDtype, values: &[f32]) -> Vec<u16> {
    let one = |v: f32| match dtype {
        HyperDtype::F16 => half::f16::from_f32(v).to_bits(),
        HyperDtype::Bf16 => half::bf16::from_f32(v).to_bits(),
    };
    values.iter().map(|&v| one(v)).collect()
}

fn widen(dtype: HyperDtype, codes: &[u16]) -> Vec<f64> {
    let one = |c: u16| match dtype {
        HyperDtype::F16 => half::f16::from_bits(c).to_f64(),
        HyperDtype::Bf16 => half::bf16::from_bits(c).to_f64(),
    };
    codes.iter().map(|&c| one(c)).collect()
}

fn code_bytes(codes: &[u16]) -> Vec<u8> {
    codes.iter().flat_map(|c| c.to_le_bytes()).collect()
}

fn draw(rng: &mut Rng, n: usize, sd: f64) -> Vec<f32> {
    (0..n).map(|_| (rng.normal() * sd) as f32).collect()
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

#[derive(Clone, Copy, Debug)]
struct Spec {
    m: usize,
    branches: usize,
    hidden: usize,
    rank: usize,
    dtype: HyperDtype,
    inject: bool,
}

/// The Flash-Next site: 4 branches of 2560, rank 320 (`hc_count`, `hidden_size`, `hc_lowrank`).
const fn flash_next(m: usize, dtype: HyperDtype, inject: bool) -> Spec {
    Spec {
        m,
        branches: 4,
        hidden: 2560,
        rank: 320,
        dtype,
        inject,
    }
}

const fn small(
    m: usize,
    (branches, hidden, rank): (usize, usize, usize),
    dtype: HyperDtype,
) -> Spec {
    Spec {
        m,
        branches,
        hidden,
        rank,
        dtype,
        inject: true,
    }
}

/// One read: `x` f32 and the weight codes (`dtype` bits).
struct Case {
    shape: HyperShape,
    dtype: HyperDtype,
    eps: f32,
    x: Vec<f32>,
    norm: Vec<u16>,
    down: Vec<u16>,
    up: Vec<u16>,
    inject: Option<Vec<u16>>,
}

impl Case {
    fn new(rng: &mut Rng, spec: Spec) -> Self {
        let Spec {
            m,
            branches,
            hidden,
            rank,
            dtype,
            inject,
        } = spec;
        let width = branches * hidden;
        // Each branch has its own scale, so the per-branch RMS differs between branches.
        let mut x = Vec::with_capacity(m * width);
        for _ in 0..m {
            for b in 0..branches {
                x.extend(draw(rng, hidden, 0.25 + b as f64));
            }
        }
        // `dot / C` of a down or inject row has a standard deviation near 1.5, and a mix value
        // near 2: SiLU and both sigmoids see both signs, away from saturation.
        let down_sd = 1.5 * branches as f64 / (width as f64).sqrt();
        let up_sd = 2.0 / (rank as f64).sqrt();
        let mut weights = |n: usize, sd: f64| narrow(dtype, &draw(rng, n, sd));
        Self {
            shape: HyperShape {
                m,
                branches,
                hidden,
                rank,
            },
            dtype,
            eps: EPS,
            x,
            norm: weights(width, 0.2),
            down: weights(rank * width, down_sd),
            up: weights(width * rank, up_sd),
            inject: inject.then(|| weights(branches * width, down_sd)),
        }
    }

    fn width(&self) -> usize {
        self.shape.branches * self.shape.hidden
    }
}

struct Inputs {
    x: Padded,
    norm: Padded,
    down: Padded,
    up: Padded,
    inject: Option<Padded>,
}

impl Inputs {
    /// `extra` is the byte shift of the f32 input and of the weights.
    fn new(ctx: &CudaContext, c: &Case, (f32_extra, weight_extra): (usize, usize)) -> Self {
        let weight = |codes: &[u16]| Padded::new(ctx, &code_bytes(codes), weight_extra);
        Self {
            x: Padded::floats(ctx, &c.x, f32_extra),
            norm: weight(&c.norm),
            down: weight(&c.down),
            up: weight(&c.up),
            inject: c.inject.as_deref().map(weight),
        }
    }

    fn unchanged(&self, ctx: &CudaContext, c: &Case) {
        assert!(self.x.read(ctx, "x") == f32_bytes(&c.x), "x changed");
        assert!(
            self.norm.read(ctx, "norm") == code_bytes(&c.norm),
            "norm changed"
        );
        assert!(
            self.down.read(ctx, "down") == code_bytes(&c.down),
            "down changed"
        );
        assert!(self.up.read(ctx, "up") == code_bytes(&c.up), "up changed");
        if let (Some(dev), Some(codes)) = (&self.inject, &c.inject) {
            assert!(
                dev.read(ctx, "inject") == code_bytes(codes),
                "inject changed"
            );
        }
    }
}

struct Outputs {
    normed: Padded,
    low: Padded,
    mix: Padded,
    mixed: Padded,
    gate: Option<Padded>,
}

impl Outputs {
    fn new(ctx: &CudaContext, c: &Case) -> Self {
        let HyperShape {
            m,
            branches,
            hidden,
            rank,
        } = c.shape;
        Self {
            normed: Padded::output(ctx, m * c.width()),
            low: Padded::output(ctx, m * rank),
            mix: Padded::output(ctx, m * c.width()),
            mixed: Padded::output(ctx, m * hidden),
            gate: c.inject.as_ref().map(|_| Padded::output(ctx, m * branches)),
        }
    }

    fn read(&self, ctx: &CudaContext) -> Read {
        Read {
            normed: self.normed.read_f32(ctx, "normed"),
            low: self.low.read_f32(ctx, "low"),
            mix: self.mix.read_f32(ctx, "mix"),
            mixed: self.mixed.read_f32(ctx, "mixed"),
            gate: self.gate.as_ref().map(|g| g.read_f32(ctx, "gate")),
        }
    }

    fn untouched(&self, ctx: &CudaContext) {
        self.normed.untouched(ctx, "normed");
        self.low.untouched(ctx, "low");
        self.mix.untouched(ctx, "mix");
        self.mixed.untouched(ctx, "mixed");
        if let Some(gate) = &self.gate {
            gate.untouched(ctx, "gate");
        }
    }
}

/// The device results of one read, the scratch included.
struct Read {
    normed: Vec<f32>,
    low: Vec<f32>,
    mix: Vec<f32>,
    mixed: Vec<f32>,
    gate: Option<Vec<f32>>,
}

fn read_call<'b>(c: &Case, i: &'b Inputs, o: &'b Outputs) -> HyperReadCall<'b> {
    HyperReadCall {
        shape: c.shape,
        eps: c.eps,
        x: i.x.all(),
        weights: HyperWeights {
            dtype: c.dtype,
            norm: i.norm.all(),
            down: i.down.all(),
            up: i.up.all(),
            inject: i.inject.as_ref().map(Padded::all),
        },
        scratch: HyperScratch {
            normed: o.normed.all(),
            low: o.low.all(),
            mix: o.mix.all(),
        },
        mixed: o.mixed.all(),
        gate: o.gate.as_ref().map(Padded::all),
    }
}

fn submit_read(rig: &Rig, call: HyperReadCall<'_>) -> Result<(), EngError> {
    let result = rig.ctx.hyper_read(&rig.stream, call);
    rig.stream.synchronize().expect("drain");
    result
}

fn run_read(rig: &Rig, c: &Case, extra: (usize, usize)) -> Read {
    let inputs = Inputs::new(&rig.ctx, c, extra);
    let outputs = Outputs::new(&rig.ctx, c);
    submit_read(rig, read_call(c, &inputs, &outputs)).expect("a valid read");
    inputs.unchanged(&rig.ctx, c);
    outputs.read(&rig.ctx)
}

/// A reference value and the largest first-order rounding error of the device stage.
struct Bounded {
    value: f64,
    bound: f64,
}

/// Returns the largest `|d - value| / bound`; fails above [`SLACK`] or on a non-finite value.
fn check(what: &str, dev: &[f32], reference: &[Bounded]) -> f64 {
    assert_eq!(dev.len(), reference.len(), "{what}: length");
    let mut worst = 0.0f64;
    for (index, (d, r)) in dev.iter().zip(reference).enumerate() {
        let diff = (f64::from(*d) - r.value).abs();
        let ratio = diff / r.bound.max(f64::MIN_POSITIVE);
        assert!(
            d.is_finite() && ratio <= SLACK,
            "{what}[{index}]: device {d} reference {} bound {} ratio {ratio}",
            r.value,
            r.bound
        );
        worst = worst.max(ratio);
    }
    worst
}

/// Terms of a dot product summed by 32 lanes with a stride of 32, then a 5-level butterfly; FMA
/// or not, each term passes at most this many roundings.
fn lane_depth(terms: usize) -> f64 {
    (terms.div_ceil(32) + 6) as f64
}

/// `(dot, sum of |terms|)` in f64.
fn dot(w: &[f64], v: &[f32]) -> (f64, f64) {
    w.iter().zip(v).fold((0.0, 0.0), |(s, a), (w, v)| {
        let t = w * f64::from(*v);
        (s + t, a + t.abs())
    })
}

fn ref_normed(c: &Case) -> Vec<Bounded> {
    let (width, hidden) = (c.width(), c.shape.hidden);
    let norm = widen(c.dtype, &c.norm);
    // 256 threads, each summing hidden / 256 squares, then two 5-level butterflies; the RMS
    // passes the mean, the eps add and sqrtf, and each value 1 + w, a multiply and a divide.
    let squares = (hidden.div_ceil(256) + 11) as f64;
    let rel = ((squares + 3.0) / 2.0 + 4.0) * U;
    let mut out = Vec::with_capacity(c.x.len());
    for row in c.x.chunks(hidden) {
        let b = (out.len() % width) / hidden;
        let mean = row.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>() / hidden as f64;
        let denom = (mean + f64::from(c.eps)).sqrt();
        for (d, v) in row.iter().enumerate() {
            let value = f64::from(*v) * (1.0 + norm[b * hidden + d]) / denom;
            out.push(Bounded {
                value,
                bound: rel * value.abs(),
            });
        }
    }
    out
}

/// The `low` rows and the gates from the device `normed`.
fn ref_low(c: &Case, normed: &[f32]) -> (Vec<Bounded>, Option<Vec<Bounded>>) {
    let HyperShape { branches, rank, .. } = c.shape;
    let width = c.width();
    let branches_f = branches as f64;
    let depth = lane_depth(width);
    let down = widen(c.dtype, &c.down);
    let inject = c.inject.as_deref().map(|codes| widen(c.dtype, codes));
    // Error of `dot / C` before the activation.
    let scaled = |row: &[f64], n: &[f32]| {
        let (s, abs) = dot(row, n);
        let q = s / branches_f;
        (q, (depth + 1.0) * U * abs / branches_f + U * q.abs())
    };
    let mut low = Vec::with_capacity(c.shape.m * rank);
    let mut gate = inject.as_ref().map(|_| Vec::new());
    for n in normed.chunks(width) {
        for row in down.chunks(width) {
            let (q, err) = scaled(row, n);
            let value = q * sigmoid(q);
            let bound = SILU_SLOPE * err + 6.0 * U * value.abs();
            low.push(Bounded { value, bound });
        }
        if let (Some(rows), Some(gate)) = (&inject, &mut gate) {
            for row in rows.chunks(width) {
                let (q, err) = scaled(row, n);
                let value = 2.0 * sigmoid(q);
                gate.push(Bounded {
                    value,
                    bound: 0.5 * err + 5.0 * U * value,
                });
            }
        }
    }
    (low, gate)
}

/// The `mix` rows from the device `low`.
fn ref_mix(c: &Case, low: &[f32]) -> Vec<Bounded> {
    let rank = c.shape.rank;
    let up = widen(c.dtype, &c.up);
    let depth = lane_depth(rank);
    let mut out = Vec::with_capacity(c.shape.m * c.width());
    for l in low.chunks(rank) {
        for row in up.chunks(rank) {
            let (value, abs) = dot(row, l);
            out.push(Bounded {
                value,
                bound: (depth + 1.0) * U * abs,
            });
        }
    }
    out
}

/// `mixed` from the device `normed` and `mix`.
fn ref_mixed(c: &Case, normed: &[f32], mix: &[f32]) -> Vec<Bounded> {
    let HyperShape {
        branches, hidden, ..
    } = c.shape;
    let width = c.width();
    let mut out = Vec::with_capacity(c.shape.m * hidden);
    for (n, x) in normed.chunks(width).zip(mix.chunks(width)) {
        for d in 0..hidden {
            let (mut sum, mut abs) = (0.0f64, 0.0f64);
            for b in 0..branches {
                let at = b * hidden + d;
                let term = sigmoid(f64::from(x[at])) * f64::from(n[at]);
                sum += term;
                abs += term.abs();
            }
            let value = sum / branches as f64;
            let bound = (branches as f64 + 6.0) * U * abs / branches as f64 + U * value.abs();
            out.push(Bounded { value, bound });
        }
    }
    out
}

/// `mixed` and the gates by the pure f64 chain.
fn whole(c: &Case) -> (Vec<f64>, Option<Vec<f64>>) {
    let HyperShape {
        branches,
        hidden,
        rank,
        ..
    } = c.shape;
    let width = c.width();
    let c_f = branches as f64;
    let (down, up) = (widen(c.dtype, &c.down), widen(c.dtype, &c.up));
    let inject = c.inject.as_deref().map(|codes| widen(c.dtype, codes));
    let normed: Vec<f64> = ref_normed(c).iter().map(|b| b.value).collect();
    let dot64 = |w: &[f64], v: &[f64]| w.iter().zip(v).map(|(a, b)| a * b).sum::<f64>();
    let mut mixed = Vec::new();
    let mut gate = inject.as_ref().map(|_| Vec::new());
    for n in normed.chunks(width) {
        let low: Vec<f64> = down
            .chunks(width)
            .map(|row| dot64(row, n) / c_f)
            .map(|q| q * sigmoid(q))
            .collect();
        let mix: Vec<f64> = up.chunks(rank).map(|row| dot64(row, &low)).collect();
        for d in 0..hidden {
            let sum: f64 = (0..branches)
                .map(|b| sigmoid(mix[b * hidden + d]) * n[b * hidden + d])
                .sum();
            mixed.push(sum / c_f);
        }
        if let (Some(rows), Some(gate)) = (&inject, &mut gate) {
            gate.extend(
                rows.chunks(width)
                    .map(|row| 2.0 * sigmoid(dot64(row, n) / c_f)),
            );
        }
    }
    (mixed, gate)
}

fn check_whole(what: &str, dev: &[f32], reference: &[f64], per_token: usize) {
    for (t, (d, r)) in dev
        .chunks(per_token)
        .zip(reference.chunks(per_token))
        .enumerate()
    {
        let scale = r.iter().fold(1.0f64, |a, v| a.max(v.abs()));
        let worst = d
            .iter()
            .zip(r)
            .map(|(d, r)| (f64::from(*d) - r).abs())
            .fold(
                0.0f64,
                |a, e| if e.is_nan() { f64::INFINITY } else { a.max(e) },
            );
        assert!(
            worst <= WHOLE * scale,
            "{what} token {t}: max|d| {worst} over {} (scale {scale})",
            WHOLE * scale
        );
    }
}

/// Every stage against its reference, then the whole read; prints the worst bound ratios.
fn check_read(label: &str, c: &Case, got: &Read) {
    let normed = check(&format!("{label} normed"), &got.normed, &ref_normed(c));
    let (low_ref, gate_ref) = ref_low(c, &got.normed);
    let low = check(&format!("{label} low"), &got.low, &low_ref);
    let mix = check(&format!("{label} mix"), &got.mix, &ref_mix(c, &got.low));
    let mixed_ref = ref_mixed(c, &got.normed, &got.mix);
    let mixed = check(&format!("{label} mixed"), &got.mixed, &mixed_ref);
    let gate = match (&got.gate, &gate_ref) {
        (Some(dev), Some(reference)) => check(&format!("{label} gate"), dev, reference),
        (None, None) => 0.0,
        _ => panic!("{label}: a gate on one side only"),
    };
    let (whole_mixed, whole_gate) = whole(c);
    check_whole(
        &format!("{label} whole mixed"),
        &got.mixed,
        &whole_mixed,
        c.shape.hidden,
    );
    if let (Some(dev), Some(reference)) = (&got.gate, &whole_gate) {
        check_whole(
            &format!("{label} whole gate"),
            dev,
            reference,
            c.shape.branches,
        );
    }
    eprintln!(
        "{label}: worst error / bound normed {normed:.3} low {low:.3} mix {mix:.3} mixed \
         {mixed:.3} gate {gate:.3}"
    );
}

const READ_SPECS: [Spec; 11] = [
    flash_next(1, HyperDtype::F16, true),
    flash_next(3, HyperDtype::Bf16, true),
    // The final mixer has no inject projection.
    flash_next(2, HyperDtype::F16, false),
    // Three branches (C is no power of two), rank and hidden off every block size.
    small(4, (3, 37, 33), HyperDtype::Bf16),
    small(1, (1, 1, 1), HyperDtype::F16),
    // The branch limit; 7 + 16 down and inject rows end inside a block of 8 warps.
    small(1, (16, 300, 7), HyperDtype::F16),
    small(2, (2, 4096, 64), HyperDtype::Bf16),
    // The limits: width 2^18 (the up grid y at 32768) at hidden 65536 and at 16 branches,
    // rank 4096, and m 32768.
    small(1, (4, 65536, 8), HyperDtype::F16),
    small(1, (16, 16384, 4), HyperDtype::Bf16),
    small(1, (2, 2048, 4096), HyperDtype::F16),
    small(32768, (2, 3, 2), HyperDtype::Bf16),
];

#[test]
fn read_matches_the_f64_reference_stage_by_stage() {
    if !testkit::device_present("read_matches_the_f64_reference_stage_by_stage") {
        return;
    }
    let rig = Rig::new();
    let mut rng = Rng::new(0x4c01_0001);
    for spec in READ_SPECS {
        let c = Case::new(&mut rng, spec);
        let got = run_read(&rig, &c, (0, 0));
        check_read(&format!("{spec:?}"), &c, &got);
    }
}

#[test]
fn a_silent_branch_reads_as_zero() {
    if !testkit::device_present("a_silent_branch_reads_as_zero") {
        return;
    }
    let rig = Rig::new();
    let mut rng = Rng::new(0x4c02_0002);
    let mut c = Case::new(&mut rng, flash_next(2, HyperDtype::F16, true));
    // Branch 2 of token 1 is all zero: its RMS is sqrt(eps) and its normed row is zero.
    let (width, hidden) = (c.width(), c.shape.hidden);
    c.x[width + 2 * hidden..width + 3 * hidden].fill(0.0);
    let got = run_read(&rig, &c, (0, 0));
    let silent = &got.normed[width + 2 * hidden..width + 3 * hidden];
    assert!(
        silent.iter().all(|v| *v == 0.0),
        "a zero row normalizes to zero"
    );
    check_read("silent branch", &c, &got);
}

#[test]
fn shifted_ranges_give_the_same_bits() {
    if !testkit::device_present("shifted_ranges_give_the_same_bits") {
        return;
    }
    let rig = Rig::new();
    let mut rng = Rng::new(0x4c03_0003);
    let c = Case::new(&mut rng, small(3, (4, 640, 96), HyperDtype::Bf16));
    let base = run_read(&rig, &c, (0, 0));
    for extra in [(4, 2), (12, 6), (260, 254)] {
        let got = run_read(&rig, &c, extra);
        assert_eq!(bits(&got.mixed), bits(&base.mixed), "{extra:?}: mixed");
        assert_eq!(bits(&got.mix), bits(&base.mix), "{extra:?}: mix");
        let (gate, base_gate) = (got.gate.expect("gate"), base.gate.as_deref().expect("gate"));
        assert_eq!(bits(&gate), bits(base_gate), "{extra:?}: gate");
    }
}

#[test]
fn each_token_reads_alone_as_in_a_batch() {
    if !testkit::device_present("each_token_reads_alone_as_in_a_batch") {
        return;
    }
    let rig = Rig::new();
    let mut rng = Rng::new(0x4c04_0004);
    let batch = Case::new(&mut rng, flash_next(3, HyperDtype::F16, true));
    let together = run_read(&rig, &batch, (0, 0));
    let width = batch.width();
    for t in 0..3 {
        let one = Case {
            shape: HyperShape {
                m: 1,
                ..batch.shape
            },
            x: batch.x[t * width..(t + 1) * width].to_vec(),
            norm: batch.norm.clone(),
            down: batch.down.clone(),
            up: batch.up.clone(),
            inject: batch.inject.clone(),
            ..batch
        };
        let alone = run_read(&rig, &one, (0, 0));
        let rows = |all: &[f32], n: usize| bits(&all[t * n..(t + 1) * n]);
        assert_eq!(
            bits(&alone.mixed),
            rows(&together.mixed, 2560),
            "token {t}: mixed"
        );
        assert_eq!(bits(&alone.low), rows(&together.low, 320), "token {t}: low");
        let gate = together.gate.as_deref().expect("gate");
        assert_eq!(
            bits(&alone.gate.expect("gate")),
            rows(gate, 4),
            "token {t}: gate"
        );
    }
}

#[test]
fn read_is_deterministic() {
    if !testkit::device_present("read_is_deterministic") {
        return;
    }
    let rig = Rig::new();
    let mut rng = Rng::new(0x4c05_0005);
    let c = Case::new(&mut rng, flash_next(2, HyperDtype::Bf16, true));
    let inputs = Inputs::new(&rig.ctx, &c, (0, 0));
    // One set of outputs for both runs: the second overwrites the first.
    let outputs = Outputs::new(&rig.ctx, &c);
    let mut runs = Vec::new();
    for _ in 0..2 {
        submit_read(&rig, read_call(&c, &inputs, &outputs)).expect("a valid read");
        let got = outputs.read(&rig.ctx);
        runs.push((bits(&got.mixed), bits(&got.gate.expect("gate"))));
    }
    assert!(runs[0] == runs[1], "two reads of one input differ");
}

struct ApplyCase {
    shape: HyperShape,
    x: Vec<f32>,
    gate: Vec<f32>,
    block: Vec<f32>,
}

impl ApplyCase {
    fn new(rng: &mut Rng, (m, branches, hidden): (usize, usize, usize)) -> Self {
        let width = branches * hidden;
        Self {
            shape: HyperShape {
                m,
                branches,
                hidden,
                rank: 1,
            },
            x: draw(rng, m * width, 1.0),
            // A gate is 2 * sigmoid(..), inside (0, 2).
            gate: (0..m * branches)
                .map(|_| (2.0 * rng.unit()) as f32)
                .collect(),
            block: draw(rng, m * hidden, 1.0),
        }
    }

    fn reference(&self) -> Vec<Bounded> {
        let HyperShape {
            branches, hidden, ..
        } = self.shape;
        let width = branches * hidden;
        let mut out = Vec::with_capacity(self.x.len());
        for (i, x) in self.x.iter().enumerate() {
            let (t, b, d) = (i / width, (i % width) / hidden, i % hidden);
            let product =
                f64::from(self.gate[t * branches + b]) * f64::from(self.block[t * hidden + d]);
            let value = f64::from(*x) + product;
            // One rounding with FMA, two without.
            out.push(Bounded {
                value,
                bound: U * (product.abs() + value.abs()),
            });
        }
        out
    }
}

#[test]
fn apply_matches_the_f64_reference() {
    if !testkit::device_present("apply_matches_the_f64_reference") {
        return;
    }
    let rig = Rig::new();
    let ctx = &rig.ctx;
    let mut rng = Rng::new(0x4c06_0006);
    for dims in [(3, 4, 2560), (2, 3, 37), (1, 16, 300), (1, 1, 1)] {
        let c = ApplyCase::new(&mut rng, dims);
        let (x, gate, block) = (
            Padded::floats(ctx, &c.x, 0),
            Padded::floats(ctx, &c.gate, 4),
            Padded::floats(ctx, &c.block, 8),
        );
        let call = HyperApplyCall {
            shape: c.shape,
            x: x.all(),
            gate: gate.all(),
            block: block.all(),
        };
        rig.ctx
            .hyper_apply(&rig.stream, call)
            .expect("a valid apply");
        rig.stream.synchronize().expect("drain");
        assert_eq!(
            bits(&gate.read_f32(ctx, "gate")),
            bits(&c.gate),
            "{dims:?}: gate changed"
        );
        assert_eq!(
            bits(&block.read_f32(ctx, "block")),
            bits(&c.block),
            "{dims:?}: block changed"
        );
        let worst = check(
            &format!("{dims:?} x"),
            &x.read_f32(ctx, "x"),
            &c.reference(),
        );
        eprintln!("apply {dims:?}: worst error / bound {worst:.3}");
    }
}

#[test]
fn hyper_read_bytes_pins_the_size_rules() {
    let bytes = hyper_read_bytes(HyperShape {
        m: 1,
        branches: 4,
        hidden: 2560,
        rank: 320,
    })
    .expect("the Flash-Next site");
    assert_eq!(
        (bytes.x, bytes.norm_w, bytes.down, bytes.up, bytes.inject),
        (40_960, 20_480, 6_553_600, 6_553_600, 81_920)
    );
    assert_eq!(
        (bytes.normed, bytes.low, bytes.mix, bytes.mixed, bytes.gate),
        (40_960, 1_280, 40_960, 10_240, 16)
    );
    let at_limit = HyperShape {
        m: 32_768,
        branches: 4,
        hidden: 65_536,
        rank: 4096,
    };
    assert!(hyper_read_bytes(at_limit).is_ok(), "every limit at once");
    let refused = [
        HyperShape { m: 0, ..at_limit },
        HyperShape {
            m: 32_769,
            ..at_limit
        },
        HyperShape {
            branches: 0,
            ..at_limit
        },
        HyperShape {
            branches: 17,
            hidden: 1,
            ..at_limit
        },
        HyperShape {
            hidden: 65_537,
            branches: 1,
            ..at_limit
        },
        // Width 262_160 is over 2^18.
        HyperShape {
            branches: 16,
            hidden: 16_385,
            ..at_limit
        },
        HyperShape {
            rank: 0,
            ..at_limit
        },
        HyperShape {
            rank: 4097,
            ..at_limit
        },
    ];
    for shape in refused {
        assert!(hyper_read_bytes(shape).is_err(), "{shape:?}");
    }
}

/// One bad argument of a read and the text that must name it.
#[derive(Clone, Copy, Debug)]
enum Bad {
    NegativeEps,
    NanEps,
    InfiniteEps,
    GateWithoutInject,
    InjectWithoutGate,
    ZeroRank,
    TooManyBranches,
    ShortMixed,
    ShortDown,
    MisalignedX,
    MisalignedUp,
    MixedOverNormed,
    NormedOverX,
    GateOverLow,
    MixOverNorm,
    LowOverDown,
    MixedOverUp,
    GateOverInject,
    MixedOverX,
    GateOverMixed,
    MixOverLow,
}

const BAD: [Bad; 21] = [
    Bad::NegativeEps,
    Bad::NanEps,
    Bad::InfiniteEps,
    Bad::GateWithoutInject,
    Bad::InjectWithoutGate,
    Bad::ZeroRank,
    Bad::TooManyBranches,
    Bad::ShortMixed,
    Bad::ShortDown,
    Bad::MisalignedX,
    Bad::MisalignedUp,
    Bad::MixedOverNormed,
    Bad::NormedOverX,
    Bad::GateOverLow,
    Bad::MixOverNorm,
    Bad::LowOverDown,
    Bad::MixedOverUp,
    Bad::GateOverInject,
    Bad::MixedOverX,
    Bad::GateOverMixed,
    Bad::MixOverLow,
];

/// Applies `bad` to `call` and returns the text its refusal must contain.
fn spoil<'b>(
    bad: Bad,
    call: &mut HyperReadCall<'b>,
    i: &'b Inputs,
    o: &'b Outputs,
) -> &'static str {
    let gate = o.gate.as_ref().expect("gate");
    let inject = i.inject.as_ref().expect("inject");
    match bad {
        Bad::NegativeEps => call.eps = -1e-6,
        Bad::NanEps => call.eps = f32::NAN,
        Bad::InfiniteEps => call.eps = f32::INFINITY,
        Bad::GateWithoutInject => call.weights.inject = None,
        Bad::InjectWithoutGate => call.gate = None,
        Bad::ZeroRank => call.shape.rank = 0,
        Bad::TooManyBranches => call.shape.branches = 17,
        Bad::ShortMixed => call.mixed = o.mixed.at(0, o.mixed.len - 4),
        Bad::ShortDown => call.weights.down = i.down.at(0, i.down.len - 2),
        Bad::MisalignedX => call.x = i.x.at(2, i.x.len),
        Bad::MisalignedUp => call.weights.up = i.up.at(1, i.up.len),
        Bad::MixedOverNormed => call.mixed = o.normed.at(4, o.mixed.len),
        Bad::NormedOverX => call.scratch.normed = i.x.all(),
        Bad::GateOverLow => call.gate = Some(o.low.at(0, gate.len)),
        // The trailing sentinel of each host buffer holds the longer written range.
        Bad::MixOverNorm => call.scratch.mix = i.norm.at(0, o.mix.len),
        Bad::LowOverDown => call.scratch.low = i.down.at(0, o.low.len),
        Bad::MixedOverUp => call.mixed = i.up.at(0, o.mixed.len),
        Bad::GateOverInject => call.gate = Some(inject.at(0, gate.len)),
        Bad::MixedOverX => call.mixed = i.x.at(0, o.mixed.len),
        Bad::GateOverMixed => call.gate = Some(o.mixed.at(0, gate.len)),
        Bad::MixOverLow => call.scratch.mix = o.low.at(0, o.mix.len),
    }
    match bad {
        Bad::NegativeEps | Bad::NanEps | Bad::InfiniteEps => "eps must be finite and nonnegative",
        Bad::GateWithoutInject | Bad::InjectWithoutGate => "inject and gate together or neither",
        Bad::ZeroRank => "rank must be 1 to 4096",
        Bad::TooManyBranches => "1 to 16 branches",
        Bad::ShortMixed => "the CUDA mixed range holds",
        Bad::ShortDown => "the CUDA down range holds",
        Bad::MisalignedX => "the CUDA x range starts at offset",
        Bad::MisalignedUp => "the CUDA up range starts at offset",
        Bad::MixedOverNormed => "the CUDA normed and mixed ranges overlap",
        Bad::NormedOverX => "the CUDA normed and x ranges overlap",
        Bad::GateOverLow => "the CUDA low and gate ranges overlap",
        Bad::MixOverNorm => "the CUDA mix and norm ranges overlap",
        Bad::LowOverDown => "the CUDA low and down ranges overlap",
        Bad::MixedOverUp => "the CUDA mixed and up ranges overlap",
        Bad::GateOverInject => "the CUDA gate and inject ranges overlap",
        Bad::MixedOverX => "the CUDA mixed and x ranges overlap",
        Bad::GateOverMixed => "the CUDA mixed and gate ranges overlap",
        Bad::MixOverLow => "the CUDA low and mix ranges overlap",
    }
}

#[test]
fn read_refusals_name_the_argument_and_write_nothing() {
    if !testkit::device_present("read_refusals_name_the_argument_and_write_nothing") {
        return;
    }
    let rig = Rig::new();
    let mut rng = Rng::new(0x4c07_0007);
    let c = Case::new(&mut rng, small(2, (3, 37, 33), HyperDtype::F16));
    let inputs = Inputs::new(&rig.ctx, &c, (0, 0));
    let outputs = Outputs::new(&rig.ctx, &c);
    for bad in BAD {
        let mut call = read_call(&c, &inputs, &outputs);
        let text = spoil(bad, &mut call, &inputs, &outputs);
        match submit_read(&rig, call) {
            Err(EngError::BadRequest(message)) => {
                assert!(message.contains(text), "{bad:?}: refused as {message:?}");
            }
            other => panic!("{bad:?}: {other:?}"),
        }
    }
    outputs.untouched(&rig.ctx);
    inputs.unchanged(&rig.ctx, &c);
    // The refusals left the context usable.
    submit_read(&rig, read_call(&c, &inputs, &outputs)).expect("a valid read");
    check_read("after the refusals", &c, &outputs.read(&rig.ctx));
}

#[test]
fn apply_refusals_name_the_argument_and_write_nothing() {
    if !testkit::device_present("apply_refusals_name_the_argument_and_write_nothing") {
        return;
    }
    let rig = Rig::new();
    let ctx = &rig.ctx;
    let mut rng = Rng::new(0x4c08_0008);
    let c = ApplyCase::new(&mut rng, (2, 3, 37));
    let (x, gate, block) = (
        Padded::floats(ctx, &c.x, 0),
        Padded::floats(ctx, &c.gate, 0),
        Padded::floats(ctx, &c.block, 0),
    );
    let base = || HyperApplyCall {
        shape: c.shape,
        x: x.all(),
        gate: gate.all(),
        block: block.all(),
    };
    let cases = [
        (
            HyperApplyCall {
                block: x.at(0, block.len),
                ..base()
            },
            "the CUDA x and block ranges overlap",
        ),
        (
            HyperApplyCall {
                gate: x.at(x.len - 8, gate.len),
                ..base()
            },
            "the CUDA x and gate ranges overlap",
        ),
        (
            HyperApplyCall {
                gate: gate.at(0, gate.len - 4),
                ..base()
            },
            "the CUDA gate range holds",
        ),
        (
            HyperApplyCall {
                block: block.at(2, block.len),
                ..base()
            },
            "the CUDA block range starts at offset",
        ),
        (
            HyperApplyCall {
                shape: HyperShape { m: 0, ..c.shape },
                ..base()
            },
            "1 to 32768 tokens",
        ),
    ];
    for (index, (call, text)) in cases.into_iter().enumerate() {
        let result = rig.ctx.hyper_apply(&rig.stream, call);
        rig.stream.synchronize().expect("drain");
        match result {
            Err(EngError::BadRequest(message)) => {
                assert!(
                    message.contains(text),
                    "case {index}: refused as {message:?}"
                );
            }
            other => panic!("case {index}: {other:?}"),
        }
    }
    assert_eq!(
        bits(&x.read_f32(ctx, "x")),
        bits(&c.x),
        "a refused apply wrote x"
    );
}

// ---- every weight code, saturation, and a large eps ---------------------------------------

/// A read case of one branch of 65536 with rank 1, `x` all 0.5 (so `x / rms` is one, up to the
/// eps), the norm weights `norm`, the down row all `2^-15` (so `dot / C` is near 2 and `low` near
/// silu(2)), the up column `up`, and no inject.
fn code_case(dtype: HyperDtype, norm: Vec<u16>, up: Vec<u16>) -> Case {
    const HIDDEN: usize = 65536;
    Case {
        shape: HyperShape {
            m: 1,
            branches: 1,
            hidden: HIDDEN,
            rank: 1,
        },
        dtype,
        eps: EPS,
        x: vec![0.5; HIDDEN],
        norm,
        down: narrow(dtype, &vec![1.0 / 32768.0; HIDDEN]),
        up,
        inject: None,
    }
}

fn all_codes() -> Vec<u16> {
    (0..=u16::MAX).collect()
}

/// A device value against a reference that may be NaN, infinite or beyond the f32 range: NaN
/// must stay NaN, an infinity or a value beyond f32::MAX must come back as the infinity of its
/// sign, and a finite value must lie within `SLACK * bound` or within the f32 subnormal spacing
/// 2^-149 (one rounding of a subnormal result).
fn check_code(what: &str, index: usize, dev: f32, r: &Bounded) {
    let near_max = f64::from(f32::MAX) * (1.0 - U);
    if r.value.is_nan() {
        assert!(dev.is_nan(), "{what}[{index}]: device {dev}, reference NaN");
    } else if r.value.abs() > near_max && dev.is_infinite() {
        assert!(
            dev.is_sign_negative() == r.value.is_sign_negative(),
            "{what}[{index}]: device {dev}, reference {}",
            r.value
        );
    } else {
        let diff = (f64::from(dev) - r.value).abs();
        assert!(
            dev.is_finite() && diff <= (SLACK * r.bound).max(f64::from(f32::from_bits(1))),
            "{what}[{index}]: device {dev}, reference {} bound {}",
            r.value,
            r.bound
        );
    }
}

/// All 65536 codes of each dtype as norm weights (normal, subnormal, signed zero, infinite and
/// NaN codes): `normed = x * (1 + w) / rms` with `x / rms` near one exposes the widened value of
/// every code, and the non-finite codes must give the infinity or NaN the formula gives. In the
/// norm `1 + w` absorbs a weight of at most 2^-24 (half an f32 ulp of 1), so the tiniest codes
/// show only in the up-projection test.
/// Only the norm stage is judged: the non-finite codes make every later stage NaN.
#[test]
fn every_weight_code_widens_exactly_in_the_norm() {
    if !testkit::device_present("every_weight_code_widens_exactly_in_the_norm") {
        return;
    }
    let rig = Rig::new();
    for dtype in [HyperDtype::F16, HyperDtype::Bf16] {
        let c = code_case(dtype, all_codes(), narrow(dtype, &[1.0; 65536]));
        let got = run_read(&rig, &c, (0, 0));
        for (index, (d, r)) in got.normed.iter().zip(ref_normed(&c)).enumerate() {
            check_code(&format!("{dtype:?} normed"), index, *d, &r);
        }
    }
}

/// All 65536 codes of each dtype as the up column of rank 1: `mix[i] = up[i] * low[0]`, one
/// product per value, so every code's widened value is exposed, subnormals included (no flush
/// to zero), and a bfloat16 code near its maximum overflows f32 to the infinity of its sign.
#[test]
fn every_weight_code_widens_exactly_in_the_up_projection() {
    if !testkit::device_present("every_weight_code_widens_exactly_in_the_up_projection") {
        return;
    }
    let rig = Rig::new();
    for dtype in [HyperDtype::F16, HyperDtype::Bf16] {
        let c = code_case(dtype, narrow(dtype, &[0.0; 65536]), all_codes());
        let got = run_read(&rig, &c, (0, 0));
        assert!(
            got.low[0].abs() > 1.0,
            "{dtype:?}: low[0] is {}, too small to expose the codes",
            got.low[0]
        );
        for (index, (d, r)) in got.mix.iter().zip(ref_mix(&c, &got.low)).enumerate() {
            check_code(&format!("{dtype:?} mix"), index, *d, &r);
        }
    }
}

/// Down and inject rows 64 times their usual size put `dot / C` near +-96, past the f32
/// overflow of `expf` at 88.7: SiLU must give -0 or the input and both sigmoids 0 or 1, never
/// NaN, and every stage stays within its bound. The up rows are 64 times smaller so `mix`
/// stays moderate. The case must reach the overflow: some `low` values and some gates must be
/// exactly zero.
#[test]
fn saturated_projections_give_exact_limits_and_no_nan() {
    if !testkit::device_present("saturated_projections_give_exact_limits_and_no_nan") {
        return;
    }
    let rig = Rig::new();
    let mut rng = Rng::new(0x4c0b_000b);
    let mut c = Case::new(&mut rng, flash_next(2, HyperDtype::F16, true));
    let rescale = |codes: &[u16], factor: f64| -> Vec<u16> {
        let values: Vec<f32> = widen(HyperDtype::F16, codes)
            .iter()
            .map(|v| (v * factor) as f32)
            .collect();
        narrow(HyperDtype::F16, &values)
    };
    c.down = rescale(&c.down, 64.0);
    c.up = rescale(&c.up, 1.0 / 64.0);
    c.inject = c.inject.as_deref().map(|codes| rescale(codes, 64.0));
    let got = run_read(&rig, &c, (0, 0));
    check_read("saturated", &c, &got);
    let zero_low = got.low.iter().filter(|v| **v == 0.0).count();
    let gates = got.gate.as_deref().expect("gate");
    let zero_gate = gates.iter().filter(|v| **v == 0.0).count();
    eprintln!(
        "saturated: {zero_low} of {} low values and {zero_gate} of {} gates are exactly zero",
        got.low.len(),
        gates.len()
    );
    assert!(zero_low > 0, "no low value reached the expf overflow");
    assert!(zero_gate > 0, "no gate reached the expf overflow");
}

/// eps 1e-2 at the Flash-Next site, near the mean square of the quietest branch (0.0625): the
/// eps sits inside the root, `sqrt(mean + eps)`. A kernel that adds it outside the root
/// (`sqrt(mean) + eps`) would move that branch by about 3 percent, far beyond the norm bound;
/// the host check below confirms the data separates the two forms before the device runs.
#[test]
fn a_large_eps_is_added_inside_the_root() {
    let mut rng = Rng::new(0x4c0c_000c);
    let mut c = Case::new(&mut rng, flash_next(2, HyperDtype::F16, true));
    c.eps = 1e-2;
    let hidden = c.shape.hidden;
    let row = &c.x[..hidden];
    let mean = row.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>() / hidden as f64;
    let inside = (mean + 1e-2).sqrt();
    let outside = mean.sqrt() + 1e-2;
    assert!(
        (inside / outside - 1.0).abs() > 0.01,
        "the data does not separate sqrt(mean + eps) from sqrt(mean) + eps"
    );
    if !testkit::device_present("a_large_eps_is_added_inside_the_root") {
        return;
    }
    let rig = Rig::new();
    let got = run_read(&rig, &c, (0, 0));
    check_read("eps 1e-2", &c, &got);
}
