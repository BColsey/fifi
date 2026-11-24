//! Device tests of the Loop decode attention wrapper against a plain f64 host reference.
//!
//! One query head is spread over 8 lanes of the warp (lane bits 4, 1 and 0) and the reduction
//! sums those lanes into the score. A reduction that missed a lane bit scores every key with
//! part of the dot product, and the first position of a real step cannot show it (one key:
//! the softmax is the identity). So every case here has many keys, and the two half cases put
//! the query's nonzero dimensions in one 16-lane half of the warp only (dims 0..64 and
//! 128..192 are lanes 0..16; dims 64..128 and 192..256 are lanes 16..32): each half must reach
//! the score alone. K and V are read from the same fp16 bits as the device. Cache rows no
//! token reads, `part` and `out` start as NaN, so an out-of-range read or an unwritten output
//! element poisons the result. The suite runs with `--test-threads=1`.

use super::{AttnDecodeLoopCall, AttnLoopShape, AttnLoopSplit, attn_loop_part_bytes};
use crate::kernels::cuda::testkit::{self, Rng, f16_bits, f16_to_f32};
use crate::kernels::cuda::{CudaContext, DeviceBuffer, DeviceRange, DeviceStream};

const GROUP: usize = 4;
const DIM: usize = 256;
/// `1 / sqrt(256)`, a power of two: the device's scaled query is exact.
const SCALE: f64 = 1.0 / 16.0;
/// Class E3: `max|d| <= 2^-16 * max(max|ref|, 1)` per head vector.
const TOL: f64 = 1.0 / 65_536.0;
/// The query standard deviation: scaled scores of standard deviation 1.5.
const QSD: f64 = 1.5;
/// fp16 and f32 quiet NaN.
const F16_NAN: u16 = 0x7e00;
const F32_NAN: u32 = 0x7fc0_0000;

/// Which query dimensions carry values.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Half {
    Both,
    Low,
    High,
}

#[derive(Clone, Copy)]
struct Spec {
    m: usize,
    n_kv_head: usize,
    cache_rows: usize,
    kv_len: i32,
    gate: bool,
    half: Half,
    split: AttnLoopSplit,
}

impl Spec {
    /// SenseNova Loop: 16 query heads over 4 kv heads, head_dim 256, a linear cache.
    fn loop9b(m: usize, kv_len: i32) -> Self {
        Self {
            m,
            n_kv_head: 4,
            cache_rows: kv_len as usize + 37,
            kv_len,
            gate: true,
            half: Half::Both,
            split: AttnLoopSplit::STAGE1,
        }
    }
    fn n_head(&self) -> usize {
        GROUP * self.n_kv_head
    }
    fn ld(&self) -> usize {
        self.n_head() * 2 * DIM
    }
    fn shape(&self) -> AttnLoopShape {
        AttnLoopShape {
            m: self.m,
            n_head: self.n_head(),
            n_kv_head: self.n_kv_head,
            cache_rows: self.cache_rows,
            ld: if self.gate { self.ld() } else { 0 },
        }
    }
    fn label(&self) -> String {
        let (max_splits, min_keys) = (self.split.max_splits, self.split.min_keys);
        format!(
            "m={} kv_len={} rows={} gate={} half={:?} split=({max_splits},{min_keys})",
            self.m, self.kv_len, self.cache_rows, self.gate, self.half
        )
    }
}

fn kept(half: Half, dim: usize) -> bool {
    let low = dim % 128 < 64;
    half == Half::Both || low == (half == Half::Low)
}

struct Problem {
    q: Vec<f32>,
    proj: Vec<f32>,
    k: Vec<u16>,
    v: Vec<u16>,
}

/// The query, the gate rows and the fp16 cache of `spec` from `rng`.
fn problem(spec: &Spec, rng: &mut Rng) -> Problem {
    let heads = spec.n_head();
    let mut q: Vec<f32> = (0..spec.m * heads * DIM)
        .map(|_| (QSD * rng.normal()) as f32)
        .collect();
    for (i, value) in q.iter_mut().enumerate() {
        if !kept(spec.half, i % DIM) {
            *value = 0.0;
        }
    }
    let proj: Vec<f32> = (0..spec.m * spec.ld())
        .map(|_| (1.5 * rng.normal()) as f32)
        .collect();
    let cells = spec.cache_rows * spec.n_kv_head * DIM;
    let k: Vec<u16> = (0..cells).map(|_| f16_bits(rng.normal() as f32)).collect();
    let v: Vec<u16> = (0..cells).map(|_| f16_bits(rng.normal() as f32)).collect();
    Problem { q, proj, k, v }
}

/// `k` or `v` with NaN in the rows past the last one a token reads.
fn cache(spec: &Spec, values: &[u16]) -> Vec<u16> {
    let mut values = values.to_vec();
    let first_nan = (spec.kv_len + spec.m as i32 - 1) as usize;
    for row in first_nan..spec.cache_rows {
        let start = row * spec.n_kv_head * DIM;
        values[start..start + spec.n_kv_head * DIM].fill(F16_NAN);
    }
    values
}

fn le16(values: &[u16]) -> Vec<u8> {
    values.iter().flat_map(|b| b.to_le_bytes()).collect()
}

fn whole(b: &DeviceBuffer) -> DeviceRange<'_> {
    b.range(0, b.bytes()).expect("range")
}

struct Rig {
    ctx: CudaContext,
    stream: DeviceStream,
}

impl Rig {
    fn new() -> Self {
        let ctx = CudaContext::new().expect("the device context");
        let stream = ctx.stream().expect("the stream");
        Self { ctx, stream }
    }
}

/// The `out` bytes of one call on `p`.
fn launch(rig: &Rig, p: &Problem, spec: &Spec) -> Vec<u8> {
    let part_len =
        attn_loop_part_bytes(spec.m, spec.n_kv_head, spec.split.max_splits).expect("the part size");
    let out_len = spec.m * spec.n_head() * DIM * 4;
    let q = testkit::upload_f32(&rig.ctx, &p.q);
    let proj = spec.gate.then(|| testkit::upload_f32(&rig.ctx, &p.proj));
    let k = testkit::upload_bytes(&rig.ctx, &le16(&cache(spec, &p.k)));
    let v = testkit::upload_bytes(&rig.ctx, &le16(&cache(spec, &p.v)));
    let word = testkit::upload_bytes(&rig.ctx, &spec.kv_len.to_le_bytes());
    let part = testkit::upload_bytes(&rig.ctx, &F32_NAN.to_le_bytes().repeat(part_len / 4));
    let out = testkit::upload_filled(&rig.ctx, 0xFF, out_len);
    let call = AttnDecodeLoopCall {
        shape: spec.shape(),
        split: spec.split,
        scale: SCALE as f32,
        q: whole(&q),
        proj: proj.as_ref().map(whole),
        k_cache: whole(&k),
        v_cache: whole(&v),
        kv_len: whole(&word),
        part: whole(&part),
        out: whole(&out),
    };
    rig.ctx
        .attn_decode_loop(&rig.stream, call)
        .expect("queue the attention");
    rig.stream.synchronize().expect("drain");
    testkit::read_bytes(&rig.ctx, &out, out_len)
}

/// Head `h` of token `t` in f64 from the same fp16 bits: softmax attention over the
/// `kv_len + t` keys of its kv head, then `sigmoid(gate)` when `spec` has one.
fn reference(p: &Problem, spec: &Spec, t: usize, h: usize) -> Vec<f64> {
    let kvh = h / GROUP;
    let q = &p.q[(t * spec.n_head() + h) * DIM..][..DIM];
    let keys = (spec.kv_len + t as i32) as usize;
    let at = |row: usize, d: usize| (row * spec.n_kv_head + kvh) * DIM + d;
    let scores: Vec<f64> = (0..keys)
        .map(|row| {
            (0..DIM)
                .map(|d| f64::from(q[d]) * f64::from(f16_to_f32(p.k[at(row, d)])))
                .sum::<f64>()
                * SCALE
        })
        .collect();
    let top = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<f64> = scores.iter().map(|s| (s - top).exp()).collect();
    let total: f64 = weights.iter().sum();
    (0..DIM)
        .map(|d| {
            let mut value = (0..keys)
                .map(|row| weights[row] * f64::from(f16_to_f32(p.v[at(row, d)])))
                .sum::<f64>()
                / total;
            if spec.gate {
                let g = f64::from(p.proj[t * spec.ld() + h * 2 * DIM + DIM + d]);
                value /= 1.0 + (-g).exp();
            }
            value
        })
        .collect()
}

/// The largest `max|d| / max(max|ref|, 1)` over every head vector of `out`.
fn worst(p: &Problem, spec: &Spec, out: &[u8]) -> f64 {
    let got = testkit::floats(out);
    (0..spec.m)
        .flat_map(|t| (0..spec.n_head()).map(move |h| (t, h)))
        .map(|(t, h)| {
            let want = reference(p, spec, t, h);
            let have = &got[(t * spec.n_head() + h) * DIM..][..DIM];
            let top = want.iter().fold(1.0f64, |m, w| m.max(w.abs()));
            want.iter()
                .zip(have)
                .fold(0.0f64, |m, (w, g)| m.max((f64::from(*g) - w).abs()))
                / top
        })
        .fold(0.0f64, f64::max)
}

fn check(rig: &Rig, rng: &mut Rng, spec: &Spec) -> f64 {
    let p = problem(spec, rng);
    let out = launch(rig, &p, spec);
    let ratio = worst(&p, spec, &out);
    println!("attn_decode_loop {:<62} worst {:.3e}", spec.label(), ratio);
    ratio
}

fn check_all(rig: &Rig, rng: &mut Rng, specs: &[Spec]) {
    let failed: Vec<String> = specs
        .iter()
        .map(|spec| (spec, check(rig, rng, spec)))
        .filter(|(_, ratio)| !(ratio.is_finite() && *ratio <= TOL))
        .map(|(spec, ratio)| format!("{} worst {:.3e} > {TOL:.1e}", spec.label(), ratio))
        .collect();
    assert!(
        failed.is_empty(),
        "cases over the reference bound:\n{}",
        failed.join("\n")
    );
}

#[test]
fn attn_decode_loop_matches_the_host_reference_over_many_keys() {
    if !testkit::device_present("attn_decode_loop_matches_the_host_reference_over_many_keys") {
        return;
    }
    let (rig, mut rng) = (Rig::new(), Rng::new(0x100e_0001));
    let mut specs: Vec<Spec> = [1, 2, 31, 32, 33, 70, 100, 1000]
        .map(|kv| Spec::loop9b(1, kv))
        .to_vec();
    for m in [2, 3, 16] {
        for kv in [1, 33, 100] {
            specs.push(Spec::loop9b(m, kv));
        }
    }
    for (max_splits, min_keys) in [(1, 32), (4, 64), (128, 128)] {
        specs.push(Spec {
            split: AttnLoopSplit {
                max_splits,
                min_keys,
            },
            ..Spec::loop9b(1, 100)
        });
    }
    specs.push(Spec {
        gate: false,
        ..Spec::loop9b(3, 70)
    });
    specs.push(Spec {
        n_kv_head: 1,
        ..Spec::loop9b(1, 100)
    });
    check_all(&rig, &mut rng, &specs);
}

#[test]
fn attn_decode_loop_scores_either_half_of_the_query_dimensions_alone() {
    if !testkit::device_present("attn_decode_loop_scores_either_half_of_the_query_dimensions_alone")
    {
        return;
    }
    let (rig, mut rng) = (Rig::new(), Rng::new(0x100f_0002));
    let specs: Vec<Spec> = [Half::Low, Half::High]
        .into_iter()
        .flat_map(|half| {
            [(1, 100, true), (3, 33, true), (1, 70, false)].map(|(m, kv, gate)| Spec {
                half,
                gate,
                ..Spec::loop9b(m, kv)
            })
        })
        .collect();
    check_all(&rig, &mut rng, &specs);
}

#[test]
fn attn_decode_loop_is_deterministic() {
    if !testkit::device_present("attn_decode_loop_is_deterministic") {
        return;
    }
    let (rig, mut rng) = (Rig::new(), Rng::new(0x100d_0003));
    let spec = Spec::loop9b(3, 100);
    let p = problem(&spec, &mut rng);
    let first = launch(&rig, &p, &spec);
    let again = launch(&rig, &p, &spec);
    assert_eq!(first, again, "two launches of one call differ");
}
