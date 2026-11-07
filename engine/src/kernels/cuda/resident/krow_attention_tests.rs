//! The attention of a window of `K` rows, launched once
//! with `m = K`, against `K` launches with `m = 1`, byte for byte.
//!
//! The K-row path runs `attn_prep` (RMSNorm of q and k, rope, KV append) and `attn_decode` once
//! with `m = K` rows. Row `t` must produce exactly what the single-row launch produces on the
//! state after rows `0..t`. The device words that drive the launches are those of a step:
//!
//! - the `m = K` call has `pos` word `p` and `kv_len` word `p + 1`; `attn_prep` writes the cells
//!   `p..p + K`, and `attn_decode` row `t` sees `n_keys = kv_len + t = p + 1 + t` keys;
//! - the single-row call of step `t` has `pos` word `p + t` and `kv_len` word `p + t + 1`.
//!
//! Both runs start from byte-identical inputs and caches (history in the cells below `p`, a
//! sentinel byte above) and must end with byte-identical `q`, byte-identical cache planes (fp16
//! planes, or the f32 K/V rows plus the code and scale planes of the KV8 path), byte-identical
//! `out`. The whole buffers are compared, so a cell above the window that a launch touched shows as
//! a difference. The `part` scratch is not compared: the kernel lays it out as all `(max, sum)`
//! pairs of the launch followed by all accumulators, so its byte layout depends on `m` by design
//! (found on the first device run: row 0 differs at the start of the accumulator block).
//!
//! Two mutants must die here:
//!
//! - A mutant of `attn_decode` (or `attn_decode_q8`) with `n_keys = kv_len + t + 1`. Row `t` of the
//!   `m = K` call then reads the cell of row `t + 1`, which the single-row call of step `t` has
//!   not written yet (it still holds the sentinel), so `out` differs from row 1 on.
//! - A mutant of `attn_prep` that writes row `t` at `pos` instead of `pos + t`. The window
//!   cells above `p` keep the sentinel (checked directly) and the cache planes differ from the
//!   single-row runs.
//!
//! Cases: `K` in 2..=8 and 16, and window starts `p` such that `n_keys` takes each of 1, 31, 32,
//! 33, 2047, 2048, 2049, 4095, 4096, 4097 (the edges of the 32-key tile, of the 64-split plan
//! chunk change at 2048 for fp16, and of 4096 for the KV8 plan), plus windows that cross a
//! multiple of 2048 and of 4096 (the split plan changes inside the window) and windows whose last
//! row is the last cache cell. The geometry is that of Qwen3.6: 16 query heads over 2 kv heads,
//! head size 256, `n_rot` 64, projection rows of 9216 floats, fp16 split plan `STAGE1`, KV8 plan
//! `Q8`.
//!
//! The device is not needed to compile this file; the device legs skip when no CUDA device is
//! present (`testkit::device_present`).

use super::attn_decode::{AttnDecodeCall, AttnShape, AttnSpan, AttnSplit, attn_part_bytes};
use super::attn_decode_q8::{AttnDecodeQ8Call, AttnQ8Shape, attn_q8_part_bytes};
use super::attn_prep::{
    AttnPrepCall, AttnPrepRowsCall, AttnPrepShape, RopeParams, SlotMode, cache_bytes, row_width,
};
use super::kv8_append::{Kv8CodeScale, Kv8Data, Kv8Shape};
use super::step_params::{
    OFFSET_KV_LEN, OFFSET_POS, OFFSET_STATUS, STATUS_INVALID_STEP_STATE, STEP_PARAMS_BYTES,
    StepParams,
};
use crate::EngError;
use crate::kernels::cuda::testkit::{self, Rng, f16_bits};
use crate::kernels::cuda::{CudaContext, DeviceBuffer, DeviceRange, DeviceStream};

const D: usize = 256;
const N_HEAD: usize = 16;
const N_KV: usize = 2;
const N_ROT: usize = 64;
const EPS: f32 = 1e-6;
const BASE: f32 = 1e7;
/// `1 / sqrt(256)`, the Qwen3.6 attention scale.
const SCALE: f32 = 0.0625;
/// Floats of one projection row: per q head `[q 256 | gate 256]`, then 2 k heads, then 2 v heads.
const LD: usize = N_HEAD * 2 * D + N_KV * 2 * D;
const GROUPS: usize = D / 32;
/// Bytes of one q row, and of one out row.
const Q_ROW: usize = N_HEAD * D * 4;
const OUT_ROW: usize = N_HEAD * D * 4;
/// Bytes of one cell of an fp16 plane (both kv heads).
const CELL_F16: usize = N_KV * D * 2;
/// Bytes of one f32 K or V row of the KV8 handoff.
const ROWS_F32: usize = N_KV * D * 4;
/// Bytes of the `part` slice of one row: `n_kv_head * 64 splits * 8 * (2 + 256) * 4`, the same
/// for the fp16 plan `STAGE1` and the KV8 plan `Q8`.
const PART_ROW: usize = N_KV * 64 * 8 * (2 + D) * 4;
/// The byte of every cache cell no launch of the case may write.
const SENT: u8 = 0x5a;
/// The byte that fills `q`, `out` and the f32 rows before a launch: not a NaN.
const FILL: u8 = 0x55;
/// The byte that fills `part`: a NaN, so a partial that was read without being written poisons.
const PART_FILL: u8 = 0xff;
const ROPE: RopeParams = RopeParams {
    eps: EPS,
    rope_base: BASE,
};

/// `(K, p, slack)`: a window of `K` rows from position `p` over a cache of `p + K + slack` rows.
/// Row `t` sees `n_keys = p + 1 + t`. Slack 0 puts the last row in the last cache cell.
const CASES: [(usize, usize, usize); 17] = [
    (4, 0, 2),     // n_keys 1 2 3 4
    (2, 0, 2),     // 1 2
    (4, 30, 2),    // 31 32 33 34
    (3, 31, 2),    // 32 33 34
    (5, 61, 2),    // 62 63 64 65 66 (the KV8 tile edge 64)
    (4, 2046, 2),  // 2047 2048 2049 2050: the plan changes inside the window
    (2, 2047, 2),  // 2048 2049
    (6, 2043, 2),  // 2044 .. 2049
    (8, 2044, 2),  // 2045 .. 2052
    (7, 4091, 2),  // 4092 .. 4098
    (4, 4094, 2),  // 4095 4096 4097 4098: the KV8 plan changes inside the window
    (3, 4095, 2),  // 4096 4097 4098
    (2, 4095, 2),  // 4096 4097
    (16, 2040, 2), // 2041 .. 2056, the most rows a launch takes
    (8, 4090, 2),  // 4091 .. 4098
    (4, 2046, 0),  // 2047 .. 2050, last row in the last cache cell
    (8, 30, 0),    // 31 .. 38, last row in the last cache cell
];

#[derive(Clone, Copy)]
struct Window {
    k: usize,
    p: usize,
    slack: usize,
}

impl Window {
    fn of(case: (usize, usize, usize)) -> Self {
        Self {
            k: case.0,
            p: case.1,
            slack: case.2,
        }
    }

    fn rows(self) -> usize {
        self.p + self.k + self.slack
    }

    fn label(self, kind: &str) -> String {
        format!(
            "{kind}: K={} p={} cache_rows={} (n_keys {}..={})",
            self.k,
            self.p,
            self.rows(),
            self.p + 1,
            self.p + self.k
        )
    }
}

fn word(value: usize) -> i32 {
    i32::try_from(value).expect("a small device word")
}

fn params_bytes(pos: usize, kv_len: usize) -> [u8; STEP_PARAMS_BYTES] {
    StepParams {
        token_id: 0,
        pos: word(pos),
        kv_len: word(kv_len),
        kv_slot: word(pos),
        status: 0,
        pad: [0; 3],
    }
    .to_bytes()
}

fn le(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn seed(tag: u64, w: Window) -> u64 {
    0x4b52_0000_0000 ^ (tag << 40) ^ ((w.p as u64) << 8) ^ (w.k as u64) ^ ((w.slack as u64) << 32)
}

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

// ---- host inputs -------------------------------------------------------------------------

struct Inputs {
    proj: Vec<u8>,
    q_norm: Vec<u8>,
    k_norm: Vec<u8>,
}

fn inputs(w: Window) -> Inputs {
    let mut rng = Rng::new(seed(1, w));
    let proj: Vec<f32> = (0..w.k * LD).map(|_| rng.normal() as f32).collect();
    let mut weight = || -> Vec<f32> { (0..D).map(|_| (1.0 + 0.3 * rng.normal()) as f32).collect() };
    let (q_norm, k_norm) = (weight(), weight());
    Inputs {
        proj: le(&proj),
        q_norm: le(&q_norm),
        k_norm: le(&k_norm),
    }
}

/// The fp16 K and V planes: history in the cells below `p`, the sentinel byte everywhere else.
struct Fp16Cache {
    k: Vec<u8>,
    v: Vec<u8>,
}

fn fp16_cache(w: Window) -> Fp16Cache {
    let mut rng = Rng::new(seed(2, w));
    let mut plane = || {
        let mut bytes = vec![SENT; w.rows() * CELL_F16];
        for pair in bytes[..w.p * CELL_F16].as_chunks_mut::<2>().0.iter_mut() {
            pair.copy_from_slice(&f16_bits(rng.normal() as f32).to_le_bytes());
        }
        bytes
    };
    let k = plane();
    let v = plane();
    Fp16Cache { k, v }
}

/// The KV8 planes: codes `[kv head][cell][256]` and fp16 scales `[kv head][cell][8]`.
struct Kv8Cache {
    k_codes: Vec<u8>,
    k_scales: Vec<u8>,
    v_codes: Vec<u8>,
    v_scales: Vec<u8>,
}

fn kv8_planes(rng: &mut Rng, w: Window) -> (Vec<u8>, Vec<u8>) {
    let rows = w.rows();
    let mut codes = vec![SENT; N_KV * rows * D];
    let mut scales = vec![SENT; N_KV * rows * GROUPS * 2];
    for j in 0..N_KV {
        for cell in 0..w.p {
            let at = (j * rows + cell) * D;
            for byte in &mut codes[at..at + D] {
                let code = i16::try_from(rng.below(255)).expect("below 255") - 127;
                *byte = i8::try_from(code).expect("an i8 code").to_le_bytes()[0];
            }
            let at = (j * rows + cell) * GROUPS * 2;
            for pair in scales[at..at + GROUPS * 2]
                .as_chunks_mut::<2>()
                .0
                .iter_mut()
            {
                let scale = (0.004 + 0.02 * rng.unit()) as f32;
                pair.copy_from_slice(&f16_bits(scale).to_le_bytes());
            }
        }
    }
    (codes, scales)
}

fn kv8_cache(w: Window) -> Kv8Cache {
    let mut rng = Rng::new(seed(3, w));
    let (k_codes, k_scales) = kv8_planes(&mut rng, w);
    let (v_codes, v_scales) = kv8_planes(&mut rng, w);
    Kv8Cache {
        k_codes,
        k_scales,
        v_codes,
        v_scales,
    }
}

// ---- device runs -------------------------------------------------------------------------

/// The device words and the extent of one launch group: rows `first..first + m` of the window.
#[derive(Clone, Copy)]
struct Step {
    first: usize,
    m: usize,
    admit: usize,
    pos: usize,
    kv_len: usize,
}

/// The one `m = K` launch group of a window.
fn rows_step(w: Window) -> Step {
    Step {
        first: 0,
        m: w.k,
        admit: w.p,
        pos: w.p,
        kv_len: w.p + 1,
    }
}

/// The `m = 1` launch group of step `t`: the words advance with the position.
fn single_step(w: Window, t: usize) -> Step {
    Step {
        first: t,
        m: 1,
        admit: w.p + t,
        pos: w.p + t,
        kv_len: w.p + t + 1,
    }
}

/// The buffers both paths share. `part` holds `K` row slices of `part_row` bytes, scratch for the
/// launches (its layout depends on `m`, so it is not compared).
struct Common {
    proj: DeviceBuffer,
    q_norm: DeviceBuffer,
    k_norm: DeviceBuffer,
    q: DeviceBuffer,
    params: DeviceBuffer,
    part: DeviceBuffer,
    out: DeviceBuffer,
    part_row: usize,
}

impl Common {
    fn new(rig: &Rig, w: Window, inp: &Inputs, part_row: usize) -> Self {
        let ctx = &rig.ctx;
        Self {
            proj: testkit::upload_bytes(ctx, &inp.proj),
            q_norm: testkit::upload_bytes(ctx, &inp.q_norm),
            k_norm: testkit::upload_bytes(ctx, &inp.k_norm),
            q: testkit::upload_filled(ctx, FILL, w.k * Q_ROW),
            params: testkit::upload_bytes(ctx, &params_bytes(w.p, w.p + 1)),
            part: testkit::upload_filled(ctx, PART_FILL, w.k * part_row),
            out: testkit::upload_filled(ctx, FILL, w.k * OUT_ROW),
            part_row,
        }
    }

    fn write_params(&self, stream: &DeviceStream, s: Step) -> Result<(), EngError> {
        self.params
            .write_at(stream, 0, &params_bytes(s.pos, s.kv_len))
    }

    fn proj_at(&self, s: Step) -> Result<DeviceRange<'_>, EngError> {
        self.proj.range(s.first * LD * 4, s.m * LD * 4)
    }

    fn q_at(&self, s: Step) -> Result<DeviceRange<'_>, EngError> {
        self.q.range(s.first * Q_ROW, s.m * Q_ROW)
    }

    fn part_at(&self, s: Step) -> Result<DeviceRange<'_>, EngError> {
        self.part
            .range(s.first * self.part_row, s.m * self.part_row)
    }

    fn out_at(&self, s: Step) -> Result<DeviceRange<'_>, EngError> {
        self.out.range(s.first * OUT_ROW, s.m * OUT_ROW)
    }

    fn pos(&self) -> Result<DeviceRange<'_>, EngError> {
        self.params.range(OFFSET_POS, 4)
    }

    fn kv_len(&self) -> Result<DeviceRange<'_>, EngError> {
        self.params.range(OFFSET_KV_LEN, 4)
    }

    fn status(&self) -> Result<DeviceRange<'_>, EngError> {
        self.params.range(OFFSET_STATUS, 4)
    }
}

fn prep_shape(m: usize, rows: usize) -> AttnPrepShape {
    AttnPrepShape {
        m,
        n_head: N_HEAD,
        n_kv_head: N_KV,
        head_dim: D,
        n_rot: N_ROT,
        has_gate: true,
        slot: SlotMode::Position,
        rows,
        ld: LD,
    }
}

/// What a run leaves behind: the named buffers whole, and the status word of the step params.
struct Outputs {
    planes: Vec<(&'static str, Vec<u8>)>,
    status: i32,
}

impl Outputs {
    fn plane(&self, name: &str) -> &[u8] {
        self.planes
            .iter()
            .find(|(plane, _)| *plane == name)
            .map(|(_, bytes)| bytes.as_slice())
            .unwrap_or_else(|| panic!("no plane named {name}"))
    }
}

fn snapshot(
    rig: &Rig,
    items: &[(&'static str, &DeviceBuffer)],
    params: &DeviceBuffer,
) -> Result<Outputs, EngError> {
    rig.stream.synchronize()?;
    let planes = items
        .iter()
        .map(|&(name, buffer)| (name, testkit::read_bytes(&rig.ctx, buffer, buffer.bytes())))
        .collect();
    let block = testkit::read_bytes(&rig.ctx, params, STEP_PARAMS_BYTES);
    let status = i32::from_le_bytes(
        block[OFFSET_STATUS..OFFSET_STATUS + 4]
            .try_into()
            .expect("a status word"),
    );
    Ok(Outputs { planes, status })
}

// ---- the fp16 path -----------------------------------------------------------------------

struct Fp16Run<'r> {
    rig: &'r Rig,
    w: Window,
    c: Common,
    k_cache: DeviceBuffer,
    v_cache: DeviceBuffer,
}

impl<'r> Fp16Run<'r> {
    fn new(rig: &'r Rig, w: Window, inp: &Inputs, cache: &Fp16Cache) -> Result<Self, EngError> {
        let part_row = attn_part_bytes(1, N_KV, D, AttnSplit::STAGE1.max_splits)?;
        Ok(Self {
            rig,
            w,
            c: Common::new(rig, w, inp, part_row),
            k_cache: testkit::upload_bytes(&rig.ctx, &cache.k),
            v_cache: testkit::upload_bytes(&rig.ctx, &cache.v),
        })
    }

    /// One launch group: the device words, `attn_prep`, `attn_decode`, queued in step order.
    fn launch(&self, s: Step) -> Result<(), EngError> {
        let (ctx, stream, c) = (&self.rig.ctx, &self.rig.stream, &self.c);
        let rows = self.w.rows();
        let cache = cache_bytes(rows, N_KV);
        c.write_params(stream, s)?;
        ctx.attn_prep(
            stream,
            AttnPrepCall {
                shape: prep_shape(s.m, rows),
                rope: ROPE,
                admit_pos: s.admit,
                proj: c.proj_at(s)?,
                q_norm: c.q_norm.range(0, D * 4)?,
                k_norm: c.k_norm.range(0, D * 4)?,
                k_cache: self.k_cache.range(0, cache)?,
                v_cache: self.v_cache.range(0, cache)?,
                q: c.q_at(s)?,
                pos: c.pos()?,
                status: c.status()?,
            },
        )?;
        ctx.attn_decode(
            stream,
            AttnDecodeCall {
                shape: AttnShape {
                    m: s.m,
                    n_head: N_HEAD,
                    n_kv_head: N_KV,
                    head_dim: D,
                    cache_rows: rows,
                    ld: LD,
                },
                span: AttnSpan::FULL,
                split: AttnSplit::STAGE1,
                scale: SCALE,
                q: c.q_at(s)?,
                proj: Some(c.proj_at(s)?),
                k_cache: self.k_cache.range(0, cache)?,
                v_cache: self.v_cache.range(0, cache)?,
                kv_len: c.kv_len()?,
                part: c.part_at(s)?,
                out: c.out_at(s)?,
            },
        )
    }

    fn finish(&self) -> Result<Outputs, EngError> {
        let c = &self.c;
        snapshot(
            self.rig,
            &[
                ("q", &c.q),
                ("k_cache", &self.k_cache),
                ("v_cache", &self.v_cache),
                ("out", &c.out),
            ],
            &c.params,
        )
    }
}

/// One full run of the window: a single `m = K` launch group, or `K` single-row groups.
fn run_fp16(
    rig: &Rig,
    w: Window,
    inp: &Inputs,
    cache: &Fp16Cache,
    singles: bool,
) -> Result<Outputs, EngError> {
    let run = Fp16Run::new(rig, w, inp, cache)?;
    if singles {
        for t in 0..w.k {
            run.launch(single_step(w, t))?;
        }
    } else {
        run.launch(rows_step(w))?;
    }
    run.finish()
}

fn check_fp16_state(label: &str, w: Window, cache: &Fp16Cache, run: &Outputs) {
    assert_eq!(run.status, 0, "{label}: the step status word is not 0");
    for (name, history) in [("k_cache", &cache.k), ("v_cache", &cache.v)] {
        let bytes = run.plane(name);
        assert!(
            bytes[..w.p * CELL_F16] == history[..w.p * CELL_F16],
            "{label}: {name} history below p changed"
        );
        for t in 0..w.k {
            let cell = &bytes[(w.p + t) * CELL_F16..][..CELL_F16];
            assert!(
                cell.iter().any(|&b| b != SENT),
                "{label}: {name} cell {} was not written",
                w.p + t
            );
        }
        assert!(
            bytes[(w.p + w.k) * CELL_F16..].iter().all(|&b| b == SENT),
            "{label}: {name} above the window was touched"
        );
    }
    assert_finite(label, run, &["q", "out"]);
}

fn assert_finite(label: &str, run: &Outputs, names: &[&str]) {
    for name in names {
        assert!(
            testkit::floats(run.plane(name))
                .iter()
                .all(|value| value.is_finite()),
            "{label}: {name} holds a non-finite value"
        );
    }
}

/// Byte equality of every plane. A difference names the plane, the byte and, for the row-shaped
/// planes, the row, so a failing mutant shows which row first diverged.
fn assert_identical(label: &str, rows_run: &Outputs, single: &Outputs) {
    assert_eq!(
        rows_run.planes.len(),
        single.planes.len(),
        "{label}: the runs hold different planes"
    );
    for ((name, a), (_, b)) in rows_run.planes.iter().zip(&single.planes) {
        assert_eq!(a.len(), b.len(), "{label}: {name} has different sizes");
        if let Some(at) = a.iter().zip(b).position(|(x, y)| x != y) {
            let row_bytes = match *name {
                "q" => Q_ROW,
                "out" => OUT_ROW,
                "k_rows" | "v_rows" => ROWS_F32,
                "part" => PART_ROW,
                _ => a.len(),
            };
            panic!(
                "{label}: {name} differs at byte {at} (row {}): m=K {:#04x}, K x m=1 {:#04x}",
                at / row_bytes.max(1),
                a[at],
                b[at]
            );
        }
    }
    assert_eq!(
        rows_run.status, single.status,
        "{label}: the status words differ"
    );
}

// ---- the KV8 path ------------------------------------------------------------------------

struct Kv8Run<'r> {
    rig: &'r Rig,
    w: Window,
    c: Common,
    k_rows: DeviceBuffer,
    v_rows: DeviceBuffer,
    k_codes: DeviceBuffer,
    k_scales: DeviceBuffer,
    v_codes: DeviceBuffer,
    v_scales: DeviceBuffer,
}

impl<'r> Kv8Run<'r> {
    fn new(rig: &'r Rig, w: Window, inp: &Inputs, cache: &Kv8Cache) -> Result<Self, EngError> {
        let part_row = attn_q8_part_bytes(1, N_KV, D, AttnSplit::Q8.max_splits)?;
        let ctx = &rig.ctx;
        Ok(Self {
            rig,
            w,
            c: Common::new(rig, w, inp, part_row),
            k_rows: testkit::upload_filled(ctx, FILL, w.k * ROWS_F32),
            v_rows: testkit::upload_filled(ctx, FILL, w.k * ROWS_F32),
            k_codes: testkit::upload_bytes(ctx, &cache.k_codes),
            k_scales: testkit::upload_bytes(ctx, &cache.k_scales),
            v_codes: testkit::upload_bytes(ctx, &cache.v_codes),
            v_scales: testkit::upload_bytes(ctx, &cache.v_scales),
        })
    }

    fn rows_at<'a>(&self, buffer: &'a DeviceBuffer, s: Step) -> Result<DeviceRange<'a>, EngError> {
        buffer.range(s.first * ROWS_F32, s.m * ROWS_F32)
    }

    /// One launch group of the Int8G32 step: the device words, `attn_prep_rows`, `kv8_append`,
    /// `attn_decode_q8`, queued in step order.
    fn launch(&self, s: Step) -> Result<(), EngError> {
        let (ctx, stream, c) = (&self.rig.ctx, &self.rig.stream, &self.c);
        let rows = self.w.rows();
        c.write_params(stream, s)?;
        ctx.attn_prep_rows(
            stream,
            AttnPrepRowsCall {
                shape: prep_shape(s.m, rows),
                rope: ROPE,
                admit_pos: s.admit,
                proj: c.proj_at(s)?,
                q_norm: c.q_norm.range(0, D * 4)?,
                k_norm: c.k_norm.range(0, D * 4)?,
                k_rows: self.rows_at(&self.k_rows, s)?,
                v_rows: self.rows_at(&self.v_rows, s)?,
                q: c.q_at(s)?,
                pos: c.pos()?,
                status: c.status()?,
            },
        )?;
        let shape = Kv8Shape {
            rows: s.m,
            n_kv_head: N_KV,
            head_dim: D,
            cells: rows,
            code_scale: Kv8CodeScale::StoredHalf,
        };
        ctx.kv8_append(
            stream,
            shape,
            Kv8Data {
                k: self.rows_at(&self.k_rows, s)?,
                v: self.rows_at(&self.v_rows, s)?,
                k_codes: self.k_codes.range(0, shape.code_bytes())?,
                k_scales: self.k_scales.range(0, shape.scale_bytes())?,
                v_codes: self.v_codes.range(0, shape.code_bytes())?,
                v_scales: self.v_scales.range(0, shape.scale_bytes())?,
            },
            c.pos()?,
            c.status()?,
        )?;
        ctx.attn_decode_q8(
            stream,
            AttnDecodeQ8Call {
                shape: AttnQ8Shape {
                    m: s.m,
                    n_head: N_HEAD,
                    n_kv_head: N_KV,
                    head_dim: D,
                    cells: rows,
                    ld: LD,
                },
                span: AttnSpan::FULL,
                split: AttnSplit::Q8,
                scale: SCALE,
                q: c.q_at(s)?,
                proj: Some(c.proj_at(s)?),
                k_codes: self.k_codes.range(0, shape.code_bytes())?,
                k_scales: self.k_scales.range(0, shape.scale_bytes())?,
                v_codes: self.v_codes.range(0, shape.code_bytes())?,
                v_scales: self.v_scales.range(0, shape.scale_bytes())?,
                kv_len: c.kv_len()?,
                part: c.part_at(s)?,
                out: c.out_at(s)?,
            },
        )
    }

    fn finish(&self) -> Result<Outputs, EngError> {
        let c = &self.c;
        snapshot(
            self.rig,
            &[
                ("q", &c.q),
                ("k_rows", &self.k_rows),
                ("v_rows", &self.v_rows),
                ("k_codes", &self.k_codes),
                ("k_scales", &self.k_scales),
                ("v_codes", &self.v_codes),
                ("v_scales", &self.v_scales),
                ("out", &c.out),
            ],
            &c.params,
        )
    }
}

fn run_kv8(
    rig: &Rig,
    w: Window,
    inp: &Inputs,
    cache: &Kv8Cache,
    singles: bool,
) -> Result<Outputs, EngError> {
    let run = Kv8Run::new(rig, w, inp, cache)?;
    if singles {
        for t in 0..w.k {
            run.launch(single_step(w, t))?;
        }
    } else {
        run.launch(rows_step(w))?;
    }
    run.finish()
}

fn check_kv8_state(label: &str, w: Window, cache: &Kv8Cache, run: &Outputs) {
    assert_eq!(run.status, 0, "{label}: the step status word is not 0");
    let rows = w.rows();
    let planes = [
        ("k_codes", &cache.k_codes, D),
        ("k_scales", &cache.k_scales, GROUPS * 2),
        ("v_codes", &cache.v_codes, D),
        ("v_scales", &cache.v_scales, GROUPS * 2),
    ];
    for (name, history, cell_bytes) in planes {
        let bytes = run.plane(name);
        for j in 0..N_KV {
            let head = &bytes[j * rows * cell_bytes..][..rows * cell_bytes];
            let hist = &history[j * rows * cell_bytes..][..rows * cell_bytes];
            assert!(
                head[..w.p * cell_bytes] == hist[..w.p * cell_bytes],
                "{label}: {name} head {j} history below p changed"
            );
            for t in 0..w.k {
                let cell = &head[(w.p + t) * cell_bytes..][..cell_bytes];
                assert!(
                    cell.iter().any(|&b| b != SENT),
                    "{label}: {name} head {j} cell {} was not written",
                    w.p + t
                );
            }
            assert!(
                head[(w.p + w.k) * cell_bytes..].iter().all(|&b| b == SENT),
                "{label}: {name} head {j} above the window was touched"
            );
        }
    }
    assert_finite(label, run, &["q", "out", "k_rows", "v_rows"]);
}

// ---- tests -------------------------------------------------------------------------------

/// The host copy of `split_plan` of `attn_decode.cu:65-71`: `(chunk, n_active)` of a key count.
fn split_plan(n_keys: usize, split: AttnSplit, tile: usize) -> (usize, usize) {
    let want = n_keys.div_ceil(split.min_keys);
    let n_split = want.max(1).min(split.max_splits);
    let chunk = n_keys.div_ceil(n_split).div_ceil(tile) * tile;
    (chunk, n_keys.div_ceil(chunk))
}

/// Host check, no device: the case matrix reaches every key count the module doc lists, the
/// geometry constants are the mixer's, and a window crosses the plan change of each split plan.
#[test]
fn krow_attention_case_matrix_covers_the_boundary_key_counts() {
    assert_eq!(LD, row_width(N_HEAD, N_KV), "the projection row width");
    let mut seen = std::collections::BTreeSet::new();
    let mut sizes = std::collections::BTreeSet::new();
    for case in CASES {
        let w = Window::of(case);
        sizes.insert(w.k);
        assert!((2..=16).contains(&w.k), "K of {case:?}");
        assert!(
            w.rows() >= w.p + w.k,
            "{case:?}: the cache must hold the window"
        );
        seen.extend(w.p + 1..=w.p + w.k);
    }
    for n_keys in [1, 31, 32, 33, 2047, 2048, 2049, 4095, 4096, 4097] {
        assert!(seen.contains(&n_keys), "no window reaches n_keys {n_keys}");
    }
    for k in [2, 3, 4, 5, 6, 7, 8] {
        assert!(sizes.contains(&k), "no window of K = {k}");
    }
    let crosses = |split: AttnSplit, tile: usize| {
        CASES.iter().any(|&case| {
            let w = Window::of(case);
            let plans: Vec<(usize, usize)> = (w.p + 1..=w.p + w.k)
                .map(|n| split_plan(n, split, tile))
                .collect();
            plans.iter().any(|plan| plan.0 != plans[0].0)
                && (w.p + 1..=w.p + w.k).any(|n| n % 2048 == 0 || n % 4096 == 0)
        })
    };
    assert!(
        crosses(AttnSplit::STAGE1, 32),
        "no window crosses a fp16 chunk change"
    );
    assert!(
        crosses(AttnSplit::Q8, 64),
        "no window crosses a KV8 chunk change"
    );
}

/// Proves, for the fp16 KV planes and the Qwen3.6 attention shape, that `attn_prep` and
/// `attn_decode` with `m = K` leave the same bytes as `K` launches with `m = 1` whose `pos` and
/// `kv_len` words advance by one per step: `q`, both cache planes (the cells above the window
/// stay at the sentinel), the `part` scratch and `out`. Kills the first mutant (row `t` sees
/// `kv_len + t + 1` keys) and the second (row `t` is written at `pos`).
#[test]
fn attention_on_k_rows_equals_k_single_row_launches_fp16() {
    if !testkit::device_present("attention_on_k_rows_equals_k_single_row_launches_fp16") {
        return;
    }
    let rig = Rig::new();
    for case in CASES {
        let w = Window::of(case);
        let label = w.label("fp16");
        let inp = inputs(w);
        let cache = fp16_cache(w);
        let single = run_fp16(&rig, w, &inp, &cache, true).expect("K single-row launches");
        check_fp16_state(&label, w, &cache, &single);
        let rows = run_fp16(&rig, w, &inp, &cache, false).expect("one m = K launch group");
        check_fp16_state(&label, w, &cache, &rows);
        assert_identical(&label, &rows, &single);
    }
}

/// Proves the same for the Int8G32 KV path: `attn_prep_rows` (f32 K/V rows), `kv8_append` and
/// `attn_decode_q8` with `m = K` against `K` single-row groups: `q`, the f32 rows, the code and
/// scale planes (cells above the window stay at the sentinel), `part` and `out`, under the `Q8`
/// split plan. Kills both mutants on the KV8 kernels (`kv8_append` writing row `t` at `pos` is
/// the append half of the second).
#[test]
fn attention_on_k_rows_equals_k_single_row_launches_kv8() {
    if !testkit::device_present("attention_on_k_rows_equals_k_single_row_launches_kv8") {
        return;
    }
    let rig = Rig::new();
    for case in CASES {
        let w = Window::of(case);
        let label = w.label("kv8");
        let inp = inputs(w);
        let cache = kv8_cache(w);
        let single = run_kv8(&rig, w, &inp, &cache, true).expect("K single-row launches");
        check_kv8_state(&label, w, &cache, &single);
        let rows = run_kv8(&rig, w, &inp, &cache, false).expect("one m = K launch group");
        check_kv8_state(&label, w, &cache, &rows);
        assert_identical(&label, &rows, &single);
    }
}

/// A window that does not fit the cache (`pos` word + K above the cache rows, while the host's
/// advisory `admit_pos` fits) must write no cell, NaN every `q` element, OR mask 4 into the
/// status word, and the decode over a `kv_len` word outside `1..=rows - (K - 1)` must NaN every
/// `out` element: the guard of `attn_prep_k` (`ops.cu:193-199`) and `attn_decode.cu:139`.
#[test]
fn k_row_window_that_does_not_fit_is_poisoned_not_written_fp16() {
    if !testkit::device_present("k_row_window_that_does_not_fit_is_poisoned_not_written_fp16") {
        return;
    }
    let rig = Rig::new();
    let w = Window::of((4, 60, 2));
    let (inp, cache) = (inputs(w), fp16_cache(w));
    let run = Fp16Run::new(&rig, w, &inp, &cache).expect("run");
    // 63 + 4 > 66 rows; kv_len 64 > 66 - 3.
    let step = Step {
        first: 0,
        m: w.k,
        admit: w.p,
        pos: 63,
        kv_len: 64,
    };
    run.launch(step).expect("the launches queue");
    let got = run.finish().expect("snapshot");
    assert_eq!(got.status, STATUS_INVALID_STEP_STATE, "status mask 4");
    assert!(
        got.plane("k_cache") == cache.k.as_slice(),
        "k cache written"
    );
    assert!(
        got.plane("v_cache") == cache.v.as_slice(),
        "v cache written"
    );
    for name in ["q", "out"] {
        assert!(
            testkit::floats(got.plane(name)).iter().all(|v| v.is_nan()),
            "{name} is not all NaN"
        );
    }
}

/// The same for the KV8 path: the rows entry and `kv8_append` write no row, no code and no scale,
/// `q` is NaN, mask 4 is set, and `attn_decode_q8` over the bad `kv_len` word NaNs `out`.
#[test]
fn k_row_window_that_does_not_fit_is_poisoned_not_written_kv8() {
    if !testkit::device_present("k_row_window_that_does_not_fit_is_poisoned_not_written_kv8") {
        return;
    }
    let rig = Rig::new();
    let w = Window::of((4, 60, 2));
    let (inp, cache) = (inputs(w), kv8_cache(w));
    let run = Kv8Run::new(&rig, w, &inp, &cache).expect("run");
    let step = Step {
        first: 0,
        m: w.k,
        admit: w.p,
        pos: 63,
        kv_len: 64,
    };
    run.launch(step).expect("the launches queue");
    let got = run.finish().expect("snapshot");
    assert_eq!(got.status, STATUS_INVALID_STEP_STATE, "status mask 4");
    for (name, history) in [
        ("k_codes", &cache.k_codes),
        ("k_scales", &cache.k_scales),
        ("v_codes", &cache.v_codes),
        ("v_scales", &cache.v_scales),
    ] {
        assert!(got.plane(name) == history.as_slice(), "{name} written");
    }
    for name in ["k_rows", "v_rows"] {
        assert!(
            got.plane(name).iter().all(|&b| b == FILL),
            "{name} written by a refused launch"
        );
    }
    for name in ["q", "out"] {
        assert!(
            testkit::floats(got.plane(name)).iter().all(|v| v.is_nan()),
            "{name} is not all NaN"
        );
    }
}
