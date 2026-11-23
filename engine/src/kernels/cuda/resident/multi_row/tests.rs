//! Device tests of the many-rows entries: N rows in one launch carry the bytes of N launches of
//! one row, for the conv, the delta recurrence, the attention preparation, the KV8 append and the
//! KV8 decode attention, at the Qwen3.6 geometry. A host without a device skips them. The device
//! suite runs with `--test-threads=1`.
//!
//! Every comparison is on bytes: the single-row run is the existing entry of the same name on the
//! row's own inputs, the many-rows run is one launch over all rows, and the two start from copies
//! of the same bytes. The timing test prints the kernel time at 1, 4 and 8 rows.

use super::{
    Append16MultiCall, AppendMultiCall, ConvMultiCall, DecodeGemmaMultiCall, DecodeMultiCall,
    DeltaMultiCall, KvRow, PrepMultiCall,
};
use crate::kernels::cuda::resident::attn_decode::{AttnDecodeCall, AttnShape, AttnSpan, AttnSplit};
use crate::kernels::cuda::resident::attn_decode_gemma::{
    AttnDecodeGemmaCall, AttnDecodeGemmaSplit,
};
use crate::kernels::cuda::resident::attn_decode_q8::{
    AttnDecodeQ8Call, AttnQ8Shape, attn_q8_part_bytes,
};
use crate::kernels::cuda::resident::attn_prep::{
    AttnPrepRowsCall, AttnPrepShape, RopeParams, SlotMode,
};
use crate::kernels::cuda::resident::gdn_conv::{ConvCall, ConvShape};
use crate::kernels::cuda::resident::gdn_delta::{DeltaCall, DeltaShape, DeltaWeights, GDN_EPS};
use crate::kernels::cuda::resident::kv8_append::{Kv8CodeScale, Kv8Data, Kv8Shape};
use crate::kernels::cuda::resident::step_params::StepParams;
use crate::kernels::cuda::testkit::{self, Rng};
use crate::kernels::cuda::{CudaContext, DeviceBuffer, DeviceStream};

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

    fn up(&self, values: &[f32]) -> DeviceBuffer {
        testkit::upload_f32(&self.ctx, values)
    }

    fn up_bytes(&self, bytes: &[u8]) -> DeviceBuffer {
        self.ctx.upload(bytes).expect("upload")
    }

    fn filled(&self, byte: u8, len: usize) -> DeviceBuffer {
        testkit::upload_filled(&self.ctx, byte, len)
    }

    fn read(&self, buffer: &DeviceBuffer, len: usize) -> Vec<u8> {
        self.stream.synchronize().expect("drain");
        testkit::read_bytes(&self.ctx, buffer, len)
    }
}

fn normal(rng: &mut Rng, n: usize, sd: f64) -> Vec<f32> {
    (0..n).map(|_| (rng.normal() * sd) as f32).collect()
}

const SENTINEL: u8 = 0xa7;

// ---- conv ------------------------------------------------------------------------------------

const CH: usize = 8192;
const CONV_LD: usize = CH + 64;

fn conv_case(rig: &Rig, n: usize, seed: u64) {
    let mut rng = Rng::new(seed);
    let proj = normal(&mut rng, n * CONV_LD, 1.5);
    let w = normal(&mut rng, CH * 4, 0.7);
    let states: Vec<Vec<f32>> = (0..n).map(|_| normal(&mut rng, 3 * CH, 1.0)).collect();
    let proj_buf = rig.up(&proj);
    let w_buf = rig.up(&w);

    let many_states: Vec<DeviceBuffer> = states.iter().map(|s| rig.up(s)).collect();
    let ranges: Vec<_> = many_states
        .iter()
        .map(|b| b.range(0, 3 * CH * 4).unwrap())
        .collect();
    let table = rig.ctx.state_rows(&ranges).expect("table");
    let many_out = rig.filled(SENTINEL, n * CH * 4);
    rig.ctx
        .gdn_conv_multi(
            &rig.stream,
            ConvMultiCall {
                shape: ConvShape {
                    m: n,
                    channels: CH,
                    ld: CONV_LD,
                },
                proj: proj_buf.range(0, ((n - 1) * CONV_LD + CH) * 4).unwrap(),
                weights: w_buf.range(0, CH * 16).unwrap(),
                states: &table,
                out: many_out.range(0, n * CH * 4).unwrap(),
            },
        )
        .expect("many rows");
    let many = rig.read(&many_out, n * CH * 4);

    for r in 0..n {
        let state = rig.up(&states[r]);
        let out = rig.filled(SENTINEL, CH * 4);
        rig.ctx
            .gdn_conv(
                &rig.stream,
                ConvCall {
                    shape: ConvShape {
                        m: 1,
                        channels: CH,
                        ld: CONV_LD,
                    },
                    proj: proj_buf.range(r * CONV_LD * 4, CH * 4).unwrap(),
                    weights: w_buf.range(0, CH * 16).unwrap(),
                    state: state.range(0, 3 * CH * 4).unwrap(),
                    out: out.range(0, CH * 4).unwrap(),
                },
            )
            .expect("single row");
        assert_eq!(
            rig.read(&out, CH * 4),
            many[r * CH * 4..(r + 1) * CH * 4],
            "conv out, n={n} row {r}"
        );
        assert_eq!(
            rig.read(&state, 3 * CH * 4),
            rig.read(&many_states[r], 3 * CH * 4),
            "conv state, n={n} row {r}"
        );
    }
}

#[test]
fn conv_rows_in_one_launch_are_the_bytes_of_single_row_launches() {
    if !testkit::device_present("gdn_conv_multi against single-row calls") {
        return;
    }
    let rig = Rig::new();
    for n in [1usize, 2, 5, 8, 33] {
        conv_case(&rig, n, 0x6d52_0100 + n as u64);
    }
}

// ---- delta -----------------------------------------------------------------------------------

const NK: usize = 16;
const NV: usize = 32;
const DELTA_CONV_LD: usize = (2 * NK + NV) * 128;
const DELTA_LD: usize = 4096 + 64 + 32;
const STATE_FLOATS: usize = NV * 128 * 128;

fn delta_shape(m: usize) -> DeltaShape {
    DeltaShape {
        m,
        n_k_heads: NK,
        n_v_heads: NV,
        conv_ld: DELTA_CONV_LD,
        ld: DELTA_LD,
        z_off: 0,
        beta_off: 4096,
        alpha_off: 4128,
    }
}

fn delta_case(rig: &Rig, n: usize, seed: u64) {
    let mut rng = Rng::new(seed);
    let conv = normal(&mut rng, n * DELTA_CONV_LD, 1.0);
    let proj = normal(&mut rng, n * DELTA_LD, 1.0);
    let a: Vec<f32> = (0..NV).map(|_| -(rng.unit() as f32) - 0.1).collect();
    let dt_bias = normal(&mut rng, NV, 0.5);
    let norm_w = normal(&mut rng, 128, 0.3)
        .iter()
        .map(|v| v + 1.0)
        .collect::<Vec<_>>();
    let states: Vec<Vec<f32>> = (0..n)
        .map(|_| normal(&mut rng, STATE_FLOATS, 0.2))
        .collect();
    let (conv_b, proj_b) = (rig.up(&conv), rig.up(&proj));
    let (a_b, dt_b, nw_b) = (rig.up(&a), rig.up(&dt_bias), rig.up(&norm_w));
    let weights = || DeltaWeights {
        a: a_b.range(0, NV * 4).unwrap(),
        dt_bias: dt_b.range(0, NV * 4).unwrap(),
        norm_w: nw_b.range(0, 128 * 4).unwrap(),
    };
    let state_bytes = STATE_FLOATS * 4;
    let out_row = NV * 128 * 4;

    let many_states: Vec<DeviceBuffer> = states.iter().map(|s| rig.up(s)).collect();
    let ranges: Vec<_> = many_states
        .iter()
        .map(|b| b.range(0, state_bytes).unwrap())
        .collect();
    let table = rig.ctx.state_rows(&ranges).expect("table");
    let many_out = rig.filled(SENTINEL, n * out_row);
    let shape = delta_shape(n);
    rig.ctx
        .gdn_delta_multi(
            &rig.stream,
            DeltaMultiCall {
                shape,
                eps: GDN_EPS,
                conv: conv_b
                    .range(0, ((n - 1) * DELTA_CONV_LD + DELTA_CONV_LD) * 4)
                    .unwrap(),
                proj: proj_b
                    .range(0, ((n - 1) * DELTA_LD + DELTA_LD) * 4)
                    .unwrap(),
                weights: weights(),
                states: &table,
                out: many_out.range(0, n * out_row).unwrap(),
            },
        )
        .expect("many rows");
    let many = rig.read(&many_out, n * out_row);

    for r in 0..n {
        let state = rig.up(&states[r]);
        let out = rig.filled(SENTINEL, out_row);
        rig.ctx
            .gdn_delta(
                &rig.stream,
                DeltaCall {
                    shape: delta_shape(1),
                    eps: GDN_EPS,
                    conv: conv_b
                        .range(r * DELTA_CONV_LD * 4, DELTA_CONV_LD * 4)
                        .unwrap(),
                    proj: proj_b.range(r * DELTA_LD * 4, DELTA_LD * 4).unwrap(),
                    weights: weights(),
                    state: state.range(0, state_bytes).unwrap(),
                    out: out.range(0, out_row).unwrap(),
                },
            )
            .expect("single row");
        assert_eq!(
            rig.read(&out, out_row),
            many[r * out_row..(r + 1) * out_row],
            "delta out, n={n} row {r}"
        );
        assert_eq!(
            rig.read(&state, state_bytes),
            rig.read(&many_states[r], state_bytes),
            "delta state, n={n} row {r}"
        );
    }
}

#[test]
fn delta_rows_in_one_launch_are_the_bytes_of_single_row_launches() {
    if !testkit::device_present("gdn_delta_multi against single-row calls") {
        return;
    }
    let rig = Rig::new();
    for n in [1usize, 2, 5, 8] {
        delta_case(&rig, n, 0x6d52_0200 + n as u64);
    }
}

// ---- attention: prep, append, decode ---------------------------------------------------------

const N_HEAD: usize = 16;
const N_KV: usize = 2;
const D: usize = 256;
const CELLS: usize = 192;
const ATTN_LD: usize = (N_HEAD * 2 + N_KV * 2) * D;
const CODE_BYTES: usize = N_KV * CELLS * D;
const SCALE_BYTES: usize = N_KV * CELLS * (D / 32) * 2;
const STEP_BYTES: usize = 32;
const ROPE: RopeParams = RopeParams {
    eps: 1.0e-6,
    rope_base: 1.0e7,
};

/// One row's planes and step block as bytes, shared by the two runs.
struct RowBytes {
    k_codes: Vec<u8>,
    k_scales: Vec<u8>,
    v_codes: Vec<u8>,
    v_scales: Vec<u8>,
    step: Vec<u8>,
}

fn codes(rng: &mut Rng) -> Vec<u8> {
    (0..CODE_BYTES)
        .map(|_| (rng.below(255) as i32 - 127) as i8 as u8)
        .collect()
}

fn scales(rng: &mut Rng) -> Vec<u8> {
    (0..SCALE_BYTES / 2)
        .flat_map(|_| testkit::f16_bits(0.004 + 0.03 * rng.unit() as f32).to_le_bytes())
        .collect()
}

/// A row whose planes hold random codes and scales everywhere and whose step block has `pos`
/// (and `kv_len = pos + 1`), or the words `(pos, kv_len)` given.
fn row_bytes(rng: &mut Rng, pos: i32, kv_len: i32) -> RowBytes {
    let mut step = StepParams::new(1, pos.max(0))
        .expect("step")
        .to_bytes()
        .to_vec();
    step[4..8].copy_from_slice(&pos.to_le_bytes());
    step[8..12].copy_from_slice(&kv_len.to_le_bytes());
    RowBytes {
        k_codes: codes(rng),
        k_scales: scales(rng),
        v_codes: codes(rng),
        v_scales: scales(rng),
        step,
    }
}

struct RowBuffers {
    k_codes: DeviceBuffer,
    k_scales: DeviceBuffer,
    v_codes: DeviceBuffer,
    v_scales: DeviceBuffer,
    step: DeviceBuffer,
}

fn upload_row(rig: &Rig, row: &RowBytes) -> RowBuffers {
    RowBuffers {
        k_codes: rig.up_bytes(&row.k_codes),
        k_scales: rig.up_bytes(&row.k_scales),
        v_codes: rig.up_bytes(&row.v_codes),
        v_scales: rig.up_bytes(&row.v_scales),
        step: rig.up_bytes(&row.step),
    }
}

fn kv_row(buffers: &RowBuffers) -> KvRow<'_> {
    KvRow {
        k_codes: buffers.k_codes.range(0, CODE_BYTES).unwrap(),
        k_scales: buffers.k_scales.range(0, SCALE_BYTES).unwrap(),
        v_codes: buffers.v_codes.range(0, CODE_BYTES).unwrap(),
        v_scales: buffers.v_scales.range(0, SCALE_BYTES).unwrap(),
        step: buffers.step.range(0, STEP_BYTES).unwrap(),
        cells: CELLS,
    }
}

fn kv_shape(rows: usize) -> Kv8Shape {
    Kv8Shape::linear(rows, N_KV, D, CELLS, Kv8CodeScale::StoredHalf)
}

fn decode_shape(m: usize) -> AttnQ8Shape {
    AttnQ8Shape {
        m,
        n_head: N_HEAD,
        n_kv_head: N_KV,
        head_dim: D,
        cells: CELLS,
        ld: ATTN_LD,
    }
}

fn assert_rows_equal(rig: &Rig, many: &[RowBuffers], single: &[RowBuffers], label: &str) {
    for (r, (a, b)) in many.iter().zip(single).enumerate() {
        for (name, x, y, len) in [
            ("k_codes", &a.k_codes, &b.k_codes, CODE_BYTES),
            ("k_scales", &a.k_scales, &b.k_scales, SCALE_BYTES),
            ("v_codes", &a.v_codes, &b.v_codes, CODE_BYTES),
            ("v_scales", &a.v_scales, &b.v_scales, SCALE_BYTES),
            ("step", &a.step, &b.step, STEP_BYTES),
        ] {
            assert_eq!(
                rig.read(x, len),
                rig.read(y, len),
                "{label}: {name}, row {r}"
            );
        }
    }
}

/// prep, append and decode of `positions.len()` rows, in one launch each against row by row.
fn chain_case(rig: &Rig, positions: &[i32], seed: u64) {
    let n = positions.len();
    let mut rng = Rng::new(seed);
    let rows: Vec<RowBytes> = positions
        .iter()
        .map(|&p| row_bytes(&mut rng, p, p + 1))
        .collect();
    let proj = normal(&mut rng, n * ATTN_LD, 1.0);
    let q_norm = normal(&mut rng, D, 0.2)
        .iter()
        .map(|v| v + 1.0)
        .collect::<Vec<_>>();
    let k_norm = normal(&mut rng, D, 0.2)
        .iter()
        .map(|v| v + 1.0)
        .collect::<Vec<_>>();
    let (proj_b, qn_b, kn_b) = (rig.up(&proj), rig.up(&q_norm), rig.up(&k_norm));
    let proj_bytes = ((n - 1) * ATTN_LD + ATTN_LD) * 4;
    let q_row = N_HEAD * D * 4;
    let kv_row_bytes = N_KV * D * 4;
    let split = AttnSplit::Q8;
    let part_bytes = attn_q8_part_bytes(n, N_KV, D, split.max_splits).unwrap();

    // Many rows.
    let many: Vec<RowBuffers> = rows.iter().map(|r| upload_row(rig, r)).collect();
    let table = rig
        .ctx
        .kv_rows(&many.iter().map(kv_row).collect::<Vec<_>>())
        .expect("kv table");
    let (q_m, k_m, v_m) = (
        rig.filled(SENTINEL, n * q_row),
        rig.filled(SENTINEL, n * kv_row_bytes),
        rig.filled(SENTINEL, n * kv_row_bytes),
    );
    let (part_m, out_m) = (
        rig.filled(SENTINEL, part_bytes),
        rig.filled(SENTINEL, n * q_row),
    );
    rig.ctx
        .attn_prep_multi(
            &rig.stream,
            PrepMultiCall {
                n_head: N_HEAD,
                n_kv_head: N_KV,
                n_rot: 64,
                ld: ATTN_LD,
                rope: ROPE,
                proj: proj_b.range(0, proj_bytes).unwrap(),
                q_norm: qn_b.range(0, D * 4).unwrap(),
                k_norm: kn_b.range(0, D * 4).unwrap(),
                k_rows: k_m.range(0, n * kv_row_bytes).unwrap(),
                v_rows: v_m.range(0, n * kv_row_bytes).unwrap(),
                q: q_m.range(0, n * q_row).unwrap(),
                rows: &table,
            },
        )
        .expect("prep many");
    rig.ctx
        .kv8_append_multi(
            &rig.stream,
            AppendMultiCall {
                shape: kv_shape(n),
                k: k_m.range(0, n * kv_row_bytes).unwrap(),
                v: v_m.range(0, n * kv_row_bytes).unwrap(),
                rows: &table,
            },
        )
        .expect("append many");
    rig.ctx
        .attn_decode_q8_multi(
            &rig.stream,
            DecodeMultiCall {
                shape: decode_shape(n),
                span: AttnSpan::FULL,
                split,
                scale: 1.0 / 16.0,
                q: q_m.range(0, n * q_row).unwrap(),
                proj: Some(proj_b.range(0, proj_bytes).unwrap()),
                rows: &table,
                part: part_m.range(0, part_bytes).unwrap(),
                out: out_m.range(0, n * q_row).unwrap(),
            },
        )
        .expect("decode many");
    let (q_many, k_many, v_many, out_many) = (
        rig.read(&q_m, n * q_row),
        rig.read(&k_m, n * kv_row_bytes),
        rig.read(&v_m, n * kv_row_bytes),
        rig.read(&out_m, n * q_row),
    );

    // The same rows, one at a time, on copies of the same bytes.
    let single: Vec<RowBuffers> = rows.iter().map(|r| upload_row(rig, r)).collect();
    for (r, row) in single.iter().enumerate() {
        let (q, k, v) = (
            rig.filled(SENTINEL, q_row),
            rig.filled(SENTINEL, kv_row_bytes),
            rig.filled(SENTINEL, kv_row_bytes),
        );
        let part_bytes_1 = attn_q8_part_bytes(1, N_KV, D, split.max_splits).unwrap();
        let (part, out) = (
            rig.filled(SENTINEL, part_bytes_1),
            rig.filled(SENTINEL, q_row),
        );
        let proj_row = proj_b.range(r * ATTN_LD * 4, ATTN_LD * 4).unwrap();
        rig.ctx
            .attn_prep_rows(
                &rig.stream,
                AttnPrepRowsCall {
                    shape: AttnPrepShape {
                        m: 1,
                        n_head: N_HEAD,
                        n_kv_head: N_KV,
                        head_dim: D,
                        n_rot: 64,
                        has_gate: true,
                        slot: SlotMode::Position,
                        rows: CELLS,
                        ld: ATTN_LD,
                    },
                    rope: ROPE,
                    // The host's advisory position: the refusal under test is the kernel's.
                    admit_pos: 0,
                    proj: proj_row,
                    q_norm: qn_b.range(0, D * 4).unwrap(),
                    k_norm: kn_b.range(0, D * 4).unwrap(),
                    k_rows: k.range(0, kv_row_bytes).unwrap(),
                    v_rows: v.range(0, kv_row_bytes).unwrap(),
                    q: q.range(0, q_row).unwrap(),
                    pos: row.step.range(4, 4).unwrap(),
                    status: row.step.range(16, 4).unwrap(),
                    inv_freq: None,
                },
            )
            .expect("prep single");
        rig.ctx
            .kv8_append(
                &rig.stream,
                kv_shape(1),
                Kv8Data {
                    k: k.range(0, kv_row_bytes).unwrap(),
                    v: v.range(0, kv_row_bytes).unwrap(),
                    k_codes: row.k_codes.range(0, CODE_BYTES).unwrap(),
                    k_scales: row.k_scales.range(0, SCALE_BYTES).unwrap(),
                    v_codes: row.v_codes.range(0, CODE_BYTES).unwrap(),
                    v_scales: row.v_scales.range(0, SCALE_BYTES).unwrap(),
                },
                row.step.range(4, 4).unwrap(),
                row.step.range(16, 4).unwrap(),
            )
            .expect("append single");
        rig.ctx
            .attn_decode_q8(
                &rig.stream,
                AttnDecodeQ8Call {
                    shape: decode_shape(1),
                    span: AttnSpan::FULL,
                    split,
                    scale: 1.0 / 16.0,
                    q: q.range(0, q_row).unwrap(),
                    proj: Some(proj_b.range(r * ATTN_LD * 4, ATTN_LD * 4).unwrap()),
                    k_codes: row.k_codes.range(0, CODE_BYTES).unwrap(),
                    k_scales: row.k_scales.range(0, SCALE_BYTES).unwrap(),
                    v_codes: row.v_codes.range(0, CODE_BYTES).unwrap(),
                    v_scales: row.v_scales.range(0, SCALE_BYTES).unwrap(),
                    kv_len: row.step.range(8, 4).unwrap(),
                    part: part.range(0, part_bytes_1).unwrap(),
                    out: out.range(0, q_row).unwrap(),
                },
            )
            .expect("decode single");
        assert_eq!(
            rig.read(&q, q_row),
            q_many[r * q_row..(r + 1) * q_row],
            "q, row {r}"
        );
        assert_eq!(
            rig.read(&k, kv_row_bytes),
            k_many[r * kv_row_bytes..(r + 1) * kv_row_bytes],
            "k rows, row {r}"
        );
        assert_eq!(
            rig.read(&v, kv_row_bytes),
            v_many[r * kv_row_bytes..(r + 1) * kv_row_bytes],
            "v rows, row {r}"
        );
        assert_eq!(
            rig.read(&out, q_row),
            out_many[r * q_row..(r + 1) * q_row],
            "attention out, row {r}"
        );
    }
    assert_rows_equal(rig, &many, &single, "planes and step words");
}

#[test]
fn attention_rows_in_one_launch_are_the_bytes_of_single_row_launches() {
    if !testkit::device_present("attention many rows against single-row calls") {
        return;
    }
    let rig = Rig::new();
    chain_case(&rig, &[40], 0x6d52_0301);
    chain_case(&rig, &[0, 1, 63, 64], 0x6d52_0302);
    chain_case(&rig, &[5, 100, 129, 191, 0, 63, 64, 150], 0x6d52_0303);
    // Every row at a different length, so the split plan differs per row.
    chain_case(
        &rig,
        &(0..32).map(|r| r * 5 + 1).collect::<Vec<_>>(),
        0x6d52_0304,
    );
}

#[test]
fn a_refused_row_is_refused_alone() {
    if !testkit::device_present("a refused row among valid rows") {
        return;
    }
    let rig = Rig::new();
    // Row 1 is past the last cell: its q row is NaN, it writes nothing and its status word gets
    // mask 4, as a single-row launch; the rows around it carry their own bytes. The chain compares
    // every byte against the single-row runs, which refuse the same row the same way.
    chain_case(&rig, &[10, CELLS as i32, 77, -1, 191], 0x6d52_0305);
}

/// The decode alone with a window and a ring, `kv_len` words chosen directly.
#[test]
fn decode_rows_with_a_window_and_a_ring_are_the_bytes_of_single_row_launches() {
    if !testkit::device_present("decode many rows with window and ring") {
        return;
    }
    let rig = Rig::new();
    let kv_lens = [1, 30, 99, 100, 101, 149, 150, 190];
    let n = kv_lens.len();
    let mut rng = Rng::new(0x6d52_0306);
    let rows: Vec<RowBytes> = kv_lens
        .iter()
        .map(|&len| row_bytes(&mut rng, len - 1, len))
        .collect();
    let q = normal(&mut rng, n * N_HEAD * D, 1.0);
    let q_b = rig.up(&q);
    let q_row = N_HEAD * D * 4;
    let split = AttnSplit::Q8;
    for span in [
        AttnSpan::FULL,
        AttnSpan {
            window: 100,
            ring: 150,
        },
        AttnSpan {
            window: 64,
            ring: 0,
        },
    ] {
        let kv_len_ok = |len: i32| span.ring == 0 || len >= 1;
        assert!(kv_lens.iter().all(|&len| kv_len_ok(len)));
        let many: Vec<RowBuffers> = rows.iter().map(|r| upload_row(&rig, r)).collect();
        let table = rig
            .ctx
            .kv_rows(&many.iter().map(kv_row).collect::<Vec<_>>())
            .expect("table");
        let part_bytes = attn_q8_part_bytes(n, N_KV, D, split.max_splits).unwrap();
        let (part, out) = (
            rig.filled(SENTINEL, part_bytes),
            rig.filled(SENTINEL, n * q_row),
        );
        rig.ctx
            .attn_decode_q8_multi(
                &rig.stream,
                DecodeMultiCall {
                    shape: decode_shape(n),
                    span,
                    split,
                    scale: 1.0 / 16.0,
                    q: q_b.range(0, n * q_row).unwrap(),
                    proj: None,
                    rows: &table,
                    part: part.range(0, part_bytes).unwrap(),
                    out: out.range(0, n * q_row).unwrap(),
                },
            )
            .expect("many rows");
        let many_out = rig.read(&out, n * q_row);
        for (r, row) in many.iter().enumerate() {
            let part_1 = attn_q8_part_bytes(1, N_KV, D, split.max_splits).unwrap();
            let (part, out) = (rig.filled(SENTINEL, part_1), rig.filled(SENTINEL, q_row));
            rig.ctx
                .attn_decode_q8(
                    &rig.stream,
                    AttnDecodeQ8Call {
                        shape: decode_shape(1),
                        span,
                        split,
                        scale: 1.0 / 16.0,
                        q: q_b.range(r * q_row, q_row).unwrap(),
                        proj: None,
                        k_codes: row.k_codes.range(0, CODE_BYTES).unwrap(),
                        k_scales: row.k_scales.range(0, SCALE_BYTES).unwrap(),
                        v_codes: row.v_codes.range(0, CODE_BYTES).unwrap(),
                        v_scales: row.v_scales.range(0, SCALE_BYTES).unwrap(),
                        kv_len: row.step.range(8, 4).unwrap(),
                        part: part.range(0, part_1).unwrap(),
                        out: out.range(0, q_row).unwrap(),
                    },
                )
                .expect("single row");
            assert_eq!(
                rig.read(&out, q_row),
                many_out[r * q_row..(r + 1) * q_row],
                "decode out, span {span:?}, row {r}"
            );
        }
    }
}

// ---- Bolt decode attention and the ring append -----------------------------------------------

/// Rows of `head_dim` planes (Bolt: 2 kv heads, `CELLS` cells), random everywhere.
fn gemma_row(rig: &Rig, rng: &mut Rng, head_dim: usize, kv_len: i32) -> RowBuffers {
    gemma_row_cells(rig, rng, head_dim, kv_len, CELLS)
}

/// A row whose planes hold `cells` cells per head.
fn gemma_row_cells(
    rig: &Rig,
    rng: &mut Rng,
    head_dim: usize,
    kv_len: i32,
    cells: usize,
) -> RowBuffers {
    let codes_bytes = N_KV * cells * head_dim;
    let scales_bytes = N_KV * cells * (head_dim / 32) * 2;
    let codes = |rng: &mut Rng| -> Vec<u8> {
        (0..codes_bytes)
            .map(|_| (rng.below(255) as i32 - 127) as i8 as u8)
            .collect()
    };
    let scales = |rng: &mut Rng| -> Vec<u8> {
        (0..scales_bytes / 2)
            .flat_map(|_| testkit::f16_bits(0.004 + 0.03 * rng.unit() as f32).to_le_bytes())
            .collect()
    };
    let mut step = StepParams::new(1, 0).expect("step").to_bytes().to_vec();
    step[8..12].copy_from_slice(&kv_len.to_le_bytes());
    RowBuffers {
        k_codes: rig.up_bytes(&codes(rng)),
        k_scales: rig.up_bytes(&scales(rng)),
        v_codes: rig.up_bytes(&codes(rng)),
        v_scales: rig.up_bytes(&scales(rng)),
        step: rig.up_bytes(&step),
    }
}

fn gemma_kv_row(buffers: &RowBuffers, head_dim: usize) -> KvRow<'_> {
    gemma_kv_row_cells(buffers, head_dim, CELLS)
}

fn gemma_kv_row_cells(buffers: &RowBuffers, head_dim: usize, cells: usize) -> KvRow<'_> {
    KvRow {
        k_codes: buffers.k_codes.range(0, N_KV * cells * head_dim).unwrap(),
        k_scales: buffers
            .k_scales
            .range(0, N_KV * cells * (head_dim / 32) * 2)
            .unwrap(),
        v_codes: buffers.v_codes.range(0, N_KV * cells * head_dim).unwrap(),
        v_scales: buffers
            .v_scales
            .range(0, N_KV * cells * (head_dim / 32) * 2)
            .unwrap(),
        step: buffers.step.range(0, STEP_BYTES).unwrap(),
        cells,
    }
}

/// (head_dim, window, ring, split as (max_splits, min_keys)).
type GemmaCase = (usize, usize, usize, Option<(usize, usize)>);

#[test]
fn gemma_rows_in_one_launch_are_the_bytes_of_single_row_launches() {
    if !testkit::device_present("attn_decode_gemma_multi against single-row calls") {
        return;
    }
    let rig = Rig::new();
    let kv_lens = [1, 30, 99, 100, 101, 149, 150, 190];
    let mut rng = Rng::new(0x6d52_0500);
    // (head_dim, window, ring, split): local layers (256, window, ring), a global layer (512).
    let cases: [GemmaCase; 7] = [
        (256, 0, 0, None),
        (256, 100, 0, None),
        (256, 100, 150, None),
        (256, 100, 150, Some((8, 16))),
        (256, 0, 0, Some((8, 16))),
        (512, 0, 0, None),
        (512, 0, 0, Some((16, 8))),
    ];
    for (head_dim, window, ring, split) in cases {
        gemma_compare(&rig, &mut rng, &kv_lens, head_dim, window, ring, split);
    }
}

/// One launch of `kv_lens.len()` Bolt rows against the same rows one at a time. The part buffer
/// holds exactly what the call needs and each path has its own sentinels, so a slot one path never
/// wrote reads differently from the other path's.
fn gemma_compare(
    rig: &Rig,
    rng: &mut Rng,
    kv_lens: &[i32],
    head_dim: usize,
    window: usize,
    ring: usize,
    split: Option<(usize, usize)>,
) {
    const N_Q: usize = 8;
    let n = kv_lens.len();
    let rows: Vec<RowBuffers> = kv_lens
        .iter()
        .map(|&len| gemma_row(rig, rng, head_dim, len))
        .collect();
    let q_row = N_Q * head_dim * 4;
    let q = rig.up(&normal(rng, n * N_Q * head_dim, 1.0));
    let table = rig
        .ctx
        .kv_rows(
            &rows
                .iter()
                .map(|r| gemma_kv_row(r, head_dim))
                .collect::<Vec<_>>(),
        )
        .expect("table");
    let part_floats = |m: usize, max_splits: usize| m * N_Q * max_splits * (2 + head_dim);
    let max_splits_of = split.map_or(1, |(max_splits, _)| max_splits);
    let part = rig.filled(0xc3, part_floats(n, max_splits_of) * 4);
    let out = rig.filled(0xa7, n * q_row);
    rig.ctx
        .attn_decode_gemma_multi(
            &rig.stream,
            DecodeGemmaMultiCall {
                n_q_head: N_Q,
                n_kv_head: N_KV,
                head_dim,
                window,
                ring,
                scale: 1.0,
                q: q.range(0, n * q_row).unwrap(),
                rows: &table,
                split: split.map(|(max_splits, min_keys)| AttnDecodeGemmaSplit {
                    part: part.range(0, part_floats(n, max_splits) * 4).unwrap(),
                    max_splits,
                    min_keys,
                }),
                out: out.range(0, n * q_row).unwrap(),
            },
        )
        .expect("many rows");
    let many = rig.read(&out, n * q_row);
    for (r, row) in rows.iter().enumerate() {
        let out1 = rig.filled(0x3d, q_row);
        let part1 = rig.filled(0x5a, part_floats(1, max_splits_of) * 4);
        let status = rig.filled(0, 4);
        rig.ctx
            .attn_decode_gemma(
                &rig.stream,
                AttnDecodeGemmaCall {
                    out: out1.range(0, q_row).unwrap(),
                    status: Some(status.range(0, 4).unwrap()),
                    q: q.range(r * q_row, q_row).unwrap(),
                    k_codes: row.k_codes.range(0, N_KV * CELLS * head_dim).unwrap(),
                    k_scales: row
                        .k_scales
                        .range(0, N_KV * CELLS * (head_dim / 32) * 2)
                        .unwrap(),
                    v_codes: row.v_codes.range(0, N_KV * CELLS * head_dim).unwrap(),
                    v_scales: row
                        .v_scales
                        .range(0, N_KV * CELLS * (head_dim / 32) * 2)
                        .unwrap(),
                    kv_len: row.step.range(8, 4).unwrap(),
                    m: 1,
                    n_q_head: N_Q,
                    n_kv_head: N_KV,
                    head_dim,
                    cells: CELLS,
                    window,
                    ring,
                    scale: 1.0,
                    split: split.map(|(max_splits, min_keys)| AttnDecodeGemmaSplit {
                        part: part1.range(0, part_floats(1, max_splits) * 4).unwrap(),
                        max_splits,
                        min_keys,
                    }),
                },
            )
            .expect("single row");
        assert_eq!(
            rig.read(&out1, q_row),
            many[r * q_row..(r + 1) * q_row],
            "gemma out, head_dim {head_dim} window {window} ring {ring} split {split:?}, row {r}"
        );
    }
}

#[test]
fn ring_append_rows_are_the_bytes_of_single_row_launches() {
    if !testkit::device_present("kv8_append_multi with a ring") {
        return;
    }
    let rig = Rig::new();
    let ring = 150;
    let positions = [3, 149, 150, 151, 389, 1000, 0, 191];
    let n = positions.len();
    let mut rng = Rng::new(0x6d52_0501);
    let rows: Vec<RowBytes> = positions
        .iter()
        .map(|&p| row_bytes(&mut rng, p, p + 1))
        .collect();
    let kv_row_bytes = N_KV * D * 4;
    let k = rig.up(&normal(&mut rng, n * N_KV * D, 1.0));
    let v = rig.up(&normal(&mut rng, n * N_KV * D, 1.0));
    let many: Vec<RowBuffers> = rows.iter().map(|r| upload_row(&rig, r)).collect();
    let table = rig
        .ctx
        .kv_rows(&many.iter().map(kv_row).collect::<Vec<_>>())
        .expect("table");
    let shape = Kv8Shape {
        ring,
        ..kv_shape(n)
    };
    rig.ctx
        .kv8_append_multi(
            &rig.stream,
            AppendMultiCall {
                shape,
                k: k.range(0, n * kv_row_bytes).unwrap(),
                v: v.range(0, n * kv_row_bytes).unwrap(),
                rows: &table,
            },
        )
        .expect("many rows");
    let single: Vec<RowBuffers> = rows.iter().map(|r| upload_row(&rig, r)).collect();
    for (r, row) in single.iter().enumerate() {
        rig.ctx
            .kv8_append(
                &rig.stream,
                Kv8Shape {
                    ring,
                    ..kv_shape(1)
                },
                Kv8Data {
                    k: k.range(r * kv_row_bytes, kv_row_bytes).unwrap(),
                    v: v.range(r * kv_row_bytes, kv_row_bytes).unwrap(),
                    k_codes: row.k_codes.range(0, CODE_BYTES).unwrap(),
                    k_scales: row.k_scales.range(0, SCALE_BYTES).unwrap(),
                    v_codes: row.v_codes.range(0, CODE_BYTES).unwrap(),
                    v_scales: row.v_scales.range(0, SCALE_BYTES).unwrap(),
                },
                row.step.range(4, 4).unwrap(),
                row.step.range(16, 4).unwrap(),
            )
            .expect("single row");
    }
    assert_rows_equal(&rig, &many, &single, "ring append");
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn set_step(rig: &Rig, row: &RowBuffers, pos: i32) {
    let mut step = StepParams::new(1, pos.max(0)).expect("step").to_bytes();
    step[4..8].copy_from_slice(&pos.to_le_bytes());
    step[8..12].copy_from_slice(&(pos + 1).to_le_bytes());
    row.step
        .write_at(&rig.stream, 0, &step)
        .expect("step write");
}

fn zero_row(rig: &Rig) -> RowBuffers {
    RowBuffers {
        k_codes: rig.filled(0, CODE_BYTES),
        k_scales: rig.filled(0, SCALE_BYTES),
        v_codes: rig.filled(0, CODE_BYTES),
        v_scales: rig.filled(0, SCALE_BYTES),
        step: rig.filled(0, STEP_BYTES),
    }
}

/// Bolt window layers: rows append one token per step and attend over a ring that wraps for real
/// (positions run past the ring and past the cells), the position word driving append and decode.
#[test]
fn a_ring_wraps_in_the_rows_path_driven_by_append() {
    if !testkit::device_present("ring wrap through append and decode, many rows") {
        return;
    }
    let rig = Rig::new();
    // A ring shorter than the cells, a ring of exactly the cells, and a window as long as the ring.
    for (window, ring) in [(100, 150), (100, CELLS), (CELLS, CELLS)] {
        ring_wrap_case(&rig, window, ring);
    }
}

fn ring_wrap_case(rig: &Rig, window: usize, ring: usize) {
    const N_Q: usize = 8;
    let starts = [0i32, 30, 90];
    let n = starts.len();
    let mut rng = Rng::new(0x6d52_0600);
    let many: Vec<RowBuffers> = (0..n).map(|_| zero_row(rig)).collect();
    let single: Vec<RowBuffers> = (0..n).map(|_| zero_row(rig)).collect();
    let table = rig
        .ctx
        .kv_rows(&many.iter().map(kv_row).collect::<Vec<_>>())
        .expect("table");
    let kv_bytes = N_KV * D * 4;
    let q_row = N_Q * D * 4;
    let (k, v) = (rig.filled(0, n * kv_bytes), rig.filled(0, n * kv_bytes));
    let q = rig.up(&normal(&mut rng, n * N_Q * D, 1.0));
    let out = rig.filled(SENTINEL, n * q_row);
    let shape = Kv8Shape {
        ring,
        ..kv_shape(n)
    };
    let mut wrapped = false;
    for step in 0..200i32 {
        let (kk, vv) = (
            normal(&mut rng, n * N_KV * D, 1.0),
            normal(&mut rng, n * N_KV * D, 1.0),
        );
        k.write_at(&rig.stream, 0, &f32_bytes(&kk)).unwrap();
        v.write_at(&rig.stream, 0, &f32_bytes(&vv)).unwrap();
        for r in 0..n {
            set_step(rig, &many[r], starts[r] + step);
            set_step(rig, &single[r], starts[r] + step);
            wrapped |= starts[r] + step >= ring as i32 + 5;
        }
        rig.ctx
            .kv8_append_multi(
                &rig.stream,
                AppendMultiCall {
                    shape,
                    k: k.range(0, n * kv_bytes).unwrap(),
                    v: v.range(0, n * kv_bytes).unwrap(),
                    rows: &table,
                },
            )
            .expect("append many");
        rig.ctx
            .attn_decode_gemma_multi(
                &rig.stream,
                DecodeGemmaMultiCall {
                    n_q_head: N_Q,
                    n_kv_head: N_KV,
                    head_dim: D,
                    window,
                    ring,
                    scale: 1.0,
                    q: q.range(0, n * q_row).unwrap(),
                    rows: &table,
                    split: None,
                    out: out.range(0, n * q_row).unwrap(),
                },
            )
            .expect("decode many");
        let many_out = rig.read(&out, n * q_row);
        for (r, row) in single.iter().enumerate() {
            rig.ctx
                .kv8_append(
                    &rig.stream,
                    Kv8Shape {
                        ring,
                        ..kv_shape(1)
                    },
                    Kv8Data {
                        k: k.range(r * kv_bytes, kv_bytes).unwrap(),
                        v: v.range(r * kv_bytes, kv_bytes).unwrap(),
                        k_codes: row.k_codes.range(0, CODE_BYTES).unwrap(),
                        k_scales: row.k_scales.range(0, SCALE_BYTES).unwrap(),
                        v_codes: row.v_codes.range(0, CODE_BYTES).unwrap(),
                        v_scales: row.v_scales.range(0, SCALE_BYTES).unwrap(),
                    },
                    row.step.range(4, 4).unwrap(),
                    row.step.range(16, 4).unwrap(),
                )
                .expect("append single");
            let out1 = rig.filled(SENTINEL, q_row);
            rig.ctx
                .attn_decode_gemma(
                    &rig.stream,
                    AttnDecodeGemmaCall {
                        out: out1.range(0, q_row).unwrap(),
                        status: None,
                        q: q.range(r * q_row, q_row).unwrap(),
                        k_codes: row.k_codes.range(0, CODE_BYTES).unwrap(),
                        k_scales: row.k_scales.range(0, SCALE_BYTES).unwrap(),
                        v_codes: row.v_codes.range(0, CODE_BYTES).unwrap(),
                        v_scales: row.v_scales.range(0, SCALE_BYTES).unwrap(),
                        kv_len: row.step.range(8, 4).unwrap(),
                        m: 1,
                        n_q_head: N_Q,
                        n_kv_head: N_KV,
                        head_dim: D,
                        cells: CELLS,
                        window,
                        ring,
                        scale: 1.0,
                        split: None,
                    },
                )
                .expect("decode single");
            assert_eq!(
                rig.read(&out1, q_row),
                many_out[r * q_row..(r + 1) * q_row],
                "ring decode, step {step}, row {r}"
            );
        }
    }
    assert!(wrapped, "the positions passed the ring");
    assert_rows_equal(rig, &many, &single, "ring planes after the wrap");
}

/// Sessions of different sizes in one window: each row's planes hold its own cells, and the
/// append and the Gemma decode of one launch read the row's own cell count.
#[test]
fn rows_of_different_cells_are_the_bytes_of_single_row_launches() {
    if !testkit::device_present("many rows of different cells") {
        return;
    }
    let rig = Rig::new();
    const N_Q: usize = 8;
    let head_dim = 256;
    let cells = [192usize, 128, 160, 192];
    let positions = [150i32, 100, 140, 40];
    let n = cells.len();
    let mut rng = Rng::new(0x6d52_0c00);
    let q_row = N_Q * head_dim * 4;
    let kv_bytes = N_KV * head_dim * 4;
    let make = |rng: &mut Rng| -> Vec<RowBuffers> {
        (0..n)
            .map(|r| {
                let buffers = gemma_row_cells(&rig, rng, head_dim, positions[r] + 1, cells[r]);
                let mut step = StepParams::new(1, 0).expect("step").to_bytes();
                step[4..8].copy_from_slice(&positions[r].to_le_bytes());
                step[8..12].copy_from_slice(&(positions[r] + 1).to_le_bytes());
                buffers.step.write_at(&rig.stream, 0, &step).unwrap();
                buffers
            })
            .collect()
    };
    let q = rig.up(&normal(&mut rng, n * N_Q * head_dim, 1.0));
    let (k, v) = (
        rig.up(&normal(&mut rng, n * N_KV * head_dim, 1.0)),
        rig.up(&normal(&mut rng, n * N_KV * head_dim, 1.0)),
    );
    // The same random planes twice: one set for the many-rows launches, one for the single ones.
    let mut seed_rng = Rng::new(0x6d52_0c01);
    let many = make(&mut seed_rng);
    let mut seed_rng = Rng::new(0x6d52_0c01);
    let single = make(&mut seed_rng);
    let table = rig
        .ctx
        .kv_rows(
            &many
                .iter()
                .enumerate()
                .map(|(r, b)| gemma_kv_row_cells(b, head_dim, cells[r]))
                .collect::<Vec<_>>(),
        )
        .expect("table");
    for (window, split) in [(0usize, None), (100, Some((8usize, 16usize)))] {
        // Append, then decode, for the many rows.
        rig.ctx
            .kv8_append_multi(
                &rig.stream,
                AppendMultiCall {
                    shape: Kv8Shape {
                        rows: n,
                        n_kv_head: N_KV,
                        head_dim,
                        cells: CELLS,
                        code_scale: Kv8CodeScale::StoredHalf,
                        ring: 0,
                        window: 0,
                    },
                    k: k.range(0, n * kv_bytes).unwrap(),
                    v: v.range(0, n * kv_bytes).unwrap(),
                    rows: &table,
                },
            )
            .expect("append many");
        let part_bytes = n * N_Q * 8 * (2 + head_dim) * 4;
        let part = rig.filled(0xc3, part_bytes);
        let out = rig.filled(0xa7, n * q_row);
        rig.ctx
            .attn_decode_gemma_multi(
                &rig.stream,
                DecodeGemmaMultiCall {
                    n_q_head: N_Q,
                    n_kv_head: N_KV,
                    head_dim,
                    window,
                    ring: 0,
                    scale: 1.0,
                    q: q.range(0, n * q_row).unwrap(),
                    rows: &table,
                    split: split.map(|(max_splits, min_keys)| AttnDecodeGemmaSplit {
                        part: part.range(0, part_bytes).unwrap(),
                        max_splits,
                        min_keys,
                    }),
                    out: out.range(0, n * q_row).unwrap(),
                },
            )
            .expect("decode many");
        let many_out = rig.read(&out, n * q_row);
        for (r, row) in single.iter().enumerate() {
            let shape = Kv8Shape {
                rows: 1,
                n_kv_head: N_KV,
                head_dim,
                cells: cells[r],
                code_scale: Kv8CodeScale::StoredHalf,
                ring: 0,
                window: 0,
            };
            let (codes, scales) = (
                N_KV * cells[r] * head_dim,
                N_KV * cells[r] * (head_dim / 32) * 2,
            );
            rig.ctx
                .kv8_append(
                    &rig.stream,
                    shape,
                    Kv8Data {
                        k: k.range(r * kv_bytes, kv_bytes).unwrap(),
                        v: v.range(r * kv_bytes, kv_bytes).unwrap(),
                        k_codes: row.k_codes.range(0, codes).unwrap(),
                        k_scales: row.k_scales.range(0, scales).unwrap(),
                        v_codes: row.v_codes.range(0, codes).unwrap(),
                        v_scales: row.v_scales.range(0, scales).unwrap(),
                    },
                    row.step.range(4, 4).unwrap(),
                    row.step.range(16, 4).unwrap(),
                )
                .expect("append single");
            let out1 = rig.filled(0x3d, q_row);
            let part1 = rig.filled(0x5a, part1_bytes(8, head_dim));
            rig.ctx
                .attn_decode_gemma(
                    &rig.stream,
                    AttnDecodeGemmaCall {
                        out: out1.range(0, q_row).unwrap(),
                        status: None,
                        q: q.range(r * q_row, q_row).unwrap(),
                        k_codes: row.k_codes.range(0, codes).unwrap(),
                        k_scales: row.k_scales.range(0, scales).unwrap(),
                        v_codes: row.v_codes.range(0, codes).unwrap(),
                        v_scales: row.v_scales.range(0, scales).unwrap(),
                        kv_len: row.step.range(8, 4).unwrap(),
                        m: 1,
                        n_q_head: N_Q,
                        n_kv_head: N_KV,
                        head_dim,
                        cells: cells[r],
                        window,
                        ring: 0,
                        scale: 1.0,
                        split: split.map(|(max_splits, min_keys)| AttnDecodeGemmaSplit {
                            part: part1.range(0, part1_bytes(max_splits, head_dim)).unwrap(),
                            max_splits,
                            min_keys,
                        }),
                    },
                )
                .expect("decode single");
            assert_eq!(
                rig.read(&out1, q_row),
                many_out[r * q_row..(r + 1) * q_row],
                "window {window} split {split:?}: row {r} of {} cells",
                cells[r]
            );
        }
        // The appended planes are the same bytes (the second pass appends the same row again).
        for (r, (a, b)) in many.iter().zip(&single).enumerate() {
            let (codes, scales) = (
                N_KV * cells[r] * head_dim,
                N_KV * cells[r] * (head_dim / 32) * 2,
            );
            for (name, x, y, len) in [
                ("k_codes", &a.k_codes, &b.k_codes, codes),
                ("v_codes", &a.v_codes, &b.v_codes, codes),
                ("k_scales", &a.k_scales, &b.k_scales, scales),
                ("v_scales", &a.v_scales, &b.v_scales, scales),
            ] {
                assert_eq!(rig.read(x, len), rig.read(y, len), "{name}, row {r}");
            }
        }
    }
}

fn set_step_words(rig: &Rig, step: &DeviceBuffer, pos: i32, kv_len: i32) {
    let mut bytes = StepParams::new(1, 0).expect("step").to_bytes();
    bytes[4..8].copy_from_slice(&pos.to_le_bytes());
    bytes[8..12].copy_from_slice(&kv_len.to_le_bytes());
    step.write_at(&rig.stream, 0, &bytes).expect("step words");
}

/// Sessions of different sizes in one window of the Qwen3.6 entries: the attention preparation, the
/// q8 decode and the fp16 append and decode read each row's own cells. A row whose position or
/// length is past ITS cells is refused alone while a larger row at the same position is not.
#[test]
fn rows_of_different_cells_on_the_qwen_entries_are_the_bytes_of_single_row_launches() {
    if !testkit::device_present("many rows of different cells on the qwen entries") {
        return;
    }
    let rig = Rig::new();
    let cells = [192usize, 128, 160, 96];
    // Row 1 is past its 128 cells at 130 where row 0 is not; row 3 is past its 96.
    let positions = [130i32, 130, 100, 100];
    let n = cells.len();
    let mut rng = Rng::new(0x6d52_0d00);
    let q_row = N_HEAD * D * 4;

    // --- the attention preparation and the q8 decode, over q8 planes -------------------------
    let make = |seed: u64| -> Vec<RowBuffers> {
        let mut rng = Rng::new(seed);
        (0..n)
            .map(|r| {
                let buffers = gemma_row_cells(&rig, &mut rng, D, positions[r] + 1, cells[r]);
                set_step_words(&rig, &buffers.step, positions[r], positions[r] + 1);
                buffers
            })
            .collect()
    };
    let many = make(0x6d52_0d01);
    let single = make(0x6d52_0d01);
    let table = rig
        .ctx
        .kv_rows(
            &many
                .iter()
                .enumerate()
                .map(|(r, b)| gemma_kv_row_cells(b, D, cells[r]))
                .collect::<Vec<_>>(),
        )
        .expect("table");
    let proj = rig.up(&normal(&mut rng, n * ATTN_LD, 1.0));
    let q_norm = rig.up(&normal(&mut rng, D, 0.2)
        .iter()
        .map(|v| v + 1.0)
        .collect::<Vec<_>>());
    let k_norm = rig.up(&normal(&mut rng, D, 0.2)
        .iter()
        .map(|v| v + 1.0)
        .collect::<Vec<_>>());
    let kv_row_bytes = N_KV * D * 4;
    let (q_m, k_m, v_m) = (
        rig.filled(SENTINEL, n * q_row),
        rig.filled(SENTINEL, n * kv_row_bytes),
        rig.filled(SENTINEL, n * kv_row_bytes),
    );
    rig.ctx
        .attn_prep_multi(
            &rig.stream,
            PrepMultiCall {
                n_head: N_HEAD,
                n_kv_head: N_KV,
                n_rot: 64,
                ld: ATTN_LD,
                rope: ROPE,
                proj: proj.range(0, ((n - 1) * ATTN_LD + ATTN_LD) * 4).unwrap(),
                q_norm: q_norm.range(0, D * 4).unwrap(),
                k_norm: k_norm.range(0, D * 4).unwrap(),
                k_rows: k_m.range(0, n * kv_row_bytes).unwrap(),
                v_rows: v_m.range(0, n * kv_row_bytes).unwrap(),
                q: q_m.range(0, n * q_row).unwrap(),
                rows: &table,
            },
        )
        .expect("prep many");
    let split = AttnSplit::Q8;
    let part_bytes = attn_q8_part_bytes(n, N_KV, D, split.max_splits).unwrap();
    let (part_m, out_m) = (rig.filled(0xc3, part_bytes), rig.filled(0xa7, n * q_row));
    rig.ctx
        .attn_decode_q8_multi(
            &rig.stream,
            DecodeMultiCall {
                shape: decode_shape(n),
                span: AttnSpan::FULL,
                split,
                scale: 1.0 / 16.0,
                q: q_m.range(0, n * q_row).unwrap(),
                proj: None,
                rows: &table,
                part: part_m.range(0, part_bytes).unwrap(),
                out: out_m.range(0, n * q_row).unwrap(),
            },
        )
        .expect("decode many");
    let (q_many, out_many) = (rig.read(&q_m, n * q_row), rig.read(&out_m, n * q_row));
    for (r, row) in single.iter().enumerate() {
        let (q, k, v) = (
            rig.filled(0x3d, q_row),
            rig.filled(0x3d, kv_row_bytes),
            rig.filled(0x3d, kv_row_bytes),
        );
        rig.ctx
            .attn_prep_rows(
                &rig.stream,
                AttnPrepRowsCall {
                    shape: AttnPrepShape {
                        m: 1,
                        n_head: N_HEAD,
                        n_kv_head: N_KV,
                        head_dim: D,
                        n_rot: 64,
                        has_gate: true,
                        slot: SlotMode::Position,
                        rows: cells[r],
                        ld: ATTN_LD,
                    },
                    rope: ROPE,
                    admit_pos: 0,
                    proj: proj.range(r * ATTN_LD * 4, ATTN_LD * 4).unwrap(),
                    q_norm: q_norm.range(0, D * 4).unwrap(),
                    k_norm: k_norm.range(0, D * 4).unwrap(),
                    k_rows: k.range(0, kv_row_bytes).unwrap(),
                    v_rows: v.range(0, kv_row_bytes).unwrap(),
                    q: q.range(0, q_row).unwrap(),
                    pos: row.step.range(4, 4).unwrap(),
                    status: row.step.range(16, 4).unwrap(),
                    inv_freq: None,
                },
            )
            .expect("prep single");
        assert_eq!(
            rig.read(&q, q_row),
            q_many[r * q_row..(r + 1) * q_row],
            "q, row {r}"
        );
        let (codes, scales) = (N_KV * cells[r] * D, N_KV * cells[r] * 8 * 2);
        let part_1 = attn_q8_part_bytes(1, N_KV, D, split.max_splits).unwrap();
        let (part, out) = (rig.filled(0x5a, part_1), rig.filled(0x3d, q_row));
        rig.ctx
            .attn_decode_q8(
                &rig.stream,
                AttnDecodeQ8Call {
                    shape: AttnQ8Shape {
                        cells: cells[r],
                        ..decode_shape(1)
                    },
                    span: AttnSpan::FULL,
                    split,
                    scale: 1.0 / 16.0,
                    q: q.range(0, q_row).unwrap(),
                    proj: None,
                    k_codes: row.k_codes.range(0, codes).unwrap(),
                    k_scales: row.k_scales.range(0, scales).unwrap(),
                    v_codes: row.v_codes.range(0, codes).unwrap(),
                    v_scales: row.v_scales.range(0, scales).unwrap(),
                    kv_len: row.step.range(8, 4).unwrap(),
                    part: part.range(0, part_1).unwrap(),
                    out: out.range(0, q_row).unwrap(),
                },
            )
            .expect("decode single");
        assert_eq!(
            rig.read(&out, q_row),
            out_many[r * q_row..(r + 1) * q_row],
            "q8 decode out, row {r} of {} cells",
            cells[r]
        );
    }
    // The statuses: the rows past their own cells carry mask 4, the others none.
    for (r, (a, b)) in many.iter().zip(&single).enumerate() {
        assert_eq!(
            rig.read(&a.step, STEP_BYTES),
            rig.read(&b.step, STEP_BYTES),
            "step block, row {r}"
        );
    }

    // --- the fp16 append and decode ----------------------------------------------------------
    let plane = |cells: usize| N_KV * cells * D * 2;
    let make16 = |seed: u64| -> Vec<Row16Buffers> {
        let mut rng = Rng::new(seed);
        (0..n)
            .map(|r| {
                let half_plane = |rng: &mut Rng| -> Vec<u8> {
                    (0..plane(cells[r]) / 2)
                        .flat_map(|_| testkit::f16_bits((rng.normal() * 0.7) as f32).to_le_bytes())
                        .collect()
                };
                let buffers = Row16Buffers {
                    k: rig.up_bytes(&half_plane(&mut rng)),
                    v: rig.up_bytes(&half_plane(&mut rng)),
                    step: rig.filled(0, STEP_BYTES),
                };
                set_step_words(&rig, &buffers.step, positions[r], positions[r] + 1);
                buffers
            })
            .collect()
    };
    fn rec16(b: &Row16Buffers, cells: usize) -> KvRow<'_> {
        let bytes = N_KV * cells * D * 2;
        KvRow {
            k_codes: b.k.range(0, bytes).unwrap(),
            k_scales: b.k.range(0, 4).unwrap(),
            v_codes: b.v.range(0, bytes).unwrap(),
            v_scales: b.v.range(0, 4).unwrap(),
            step: b.step.range(0, STEP_BYTES).unwrap(),
            cells,
        }
    }
    let many16 = make16(0x6d52_0d02);
    let table16 = rig
        .ctx
        .kv_rows(
            &many16
                .iter()
                .enumerate()
                .map(|(r, b)| rec16(b, cells[r]))
                .collect::<Vec<_>>(),
        )
        .expect("table16");
    let (k16, v16) = (
        rig.up(&normal(&mut rng, n * N_KV * D, 1.0)),
        rig.up(&normal(&mut rng, n * N_KV * D, 1.0)),
    );
    rig.ctx
        .kv16_append_multi(
            &rig.stream,
            Append16MultiCall {
                n_kv_head: N_KV,
                head_dim: D,
                k: k16.range(0, n * kv_row_bytes).unwrap(),
                v: v16.range(0, n * kv_row_bytes).unwrap(),
                rows: &table16,
            },
        )
        .expect("append16 many");
    // The expected planes: the rounded cell written at the position of a row that is inside ITS
    // cells, nothing for a row that is not; a refused row carries mask 4.
    let reference = make16(0x6d52_0d02);
    for (r, row) in reference.iter().enumerate() {
        let (mut k_want, mut v_want) = (
            rig.read(&row.k, plane(cells[r])),
            rig.read(&row.v, plane(cells[r])),
        );
        let inside = (0..cells[r] as i32).contains(&positions[r]);
        if inside {
            let at = positions[r] as usize * N_KV * D * 2;
            let floats = |buf: &DeviceBuffer| -> Vec<f32> {
                testkit::floats(&rig.read(buf, n * kv_row_bytes))[r * N_KV * D..(r + 1) * N_KV * D]
                    .to_vec()
            };
            k_want[at..at + N_KV * D * 2].copy_from_slice(&halves(&floats(&k16)));
            v_want[at..at + N_KV * D * 2].copy_from_slice(&halves(&floats(&v16)));
        }
        assert_eq!(
            rig.read(&many16[r].k, plane(cells[r])),
            k_want,
            "fp16 k plane, row {r}"
        );
        assert_eq!(
            rig.read(&many16[r].v, plane(cells[r])),
            v_want,
            "fp16 v plane, row {r}"
        );
        let step = rig.read(&many16[r].step, STEP_BYTES);
        assert_eq!(
            i32::from_le_bytes(step[16..20].try_into().unwrap()),
            if inside { 0 } else { 4 },
            "fp16 status, row {r}"
        );
    }
    // The decode: kv_len = pos + 1, past the cells of the rows that are refused.
    let split16 = AttnSplit::STAGE1;
    let part16 = attn_q8_part_bytes(n, N_KV, D, split16.max_splits).unwrap();
    let (part16_m, out16_m) = (rig.filled(0xc3, part16), rig.filled(0xa7, n * q_row));
    let q16 = rig.up(&normal(&mut rng, n * N_HEAD * D, 1.0));
    rig.ctx
        .attn_decode_multi(
            &rig.stream,
            DecodeMultiCall {
                shape: decode_shape(n),
                span: AttnSpan::FULL,
                split: split16,
                scale: 1.0 / 16.0,
                q: q16.range(0, n * q_row).unwrap(),
                proj: None,
                rows: &table16,
                part: part16_m.range(0, part16).unwrap(),
                out: out16_m.range(0, n * q_row).unwrap(),
            },
        )
        .expect("decode16 many");
    let out16_many = rig.read(&out16_m, n * q_row);
    for (r, row) in many16.iter().enumerate() {
        let part_1 = attn_q8_part_bytes(1, N_KV, D, split16.max_splits).unwrap();
        let (part, out) = (rig.filled(0x5a, part_1), rig.filled(0x3d, q_row));
        rig.ctx
            .attn_decode(
                &rig.stream,
                AttnDecodeCall {
                    shape: AttnShape {
                        cache_rows: cells[r],
                        ..attn16_shape(1)
                    },
                    span: AttnSpan::FULL,
                    split: split16,
                    scale: 1.0 / 16.0,
                    q: q16.range(r * q_row, q_row).unwrap(),
                    proj: None,
                    k_cache: row.k.range(0, plane(cells[r])).unwrap(),
                    v_cache: row.v.range(0, plane(cells[r])).unwrap(),
                    kv_len: row.step.range(8, 4).unwrap(),
                    part: part.range(0, part_1).unwrap(),
                    out: out.range(0, q_row).unwrap(),
                },
            )
            .expect("decode16 single");
        assert_eq!(
            rig.read(&out, q_row),
            out16_many[r * q_row..(r + 1) * q_row],
            "fp16 decode out, row {r} of {} cells",
            cells[r]
        );
    }
}

/// A row whose length is out of range is refused alone: its output is NaN and mask 4 reaches its
/// own step block; the rows around it carry the bytes of their own single-row launches.
#[test]
fn an_out_of_range_gemma_row_is_refused_alone() {
    if !testkit::device_present("an out of range gemma row among valid rows") {
        return;
    }
    let rig = Rig::new();
    const N_Q: usize = 8;
    // 193 is past the cells of a linear plane, 0 is below the first key.
    let kv_lens = [10, 193, 50, 0, 190];
    let n = kv_lens.len();
    let head_dim = 256;
    let q_row = N_Q * head_dim * 4;
    let mut rng = Rng::new(0x6d52_0a00);
    for split in [None, Some((8usize, 16usize))] {
        let rows: Vec<RowBuffers> = kv_lens
            .iter()
            .map(|&len| gemma_row(&rig, &mut rng, head_dim, len))
            .collect();
        let q = rig.up(&normal(&mut rng, n * N_Q * head_dim, 1.0));
        let table = rig
            .ctx
            .kv_rows(
                &rows
                    .iter()
                    .map(|r| gemma_kv_row(r, head_dim))
                    .collect::<Vec<_>>(),
            )
            .expect("table");
        let part_bytes = n * N_Q * 8 * (2 + head_dim) * 4;
        let part = rig.filled(0xc3, part_bytes);
        let out = rig.filled(0xa7, n * q_row);
        rig.ctx
            .attn_decode_gemma_multi(
                &rig.stream,
                DecodeGemmaMultiCall {
                    n_q_head: N_Q,
                    n_kv_head: N_KV,
                    head_dim,
                    window: 0,
                    ring: 0,
                    scale: 1.0,
                    q: q.range(0, n * q_row).unwrap(),
                    rows: &table,
                    split: split.map(|(max_splits, min_keys)| AttnDecodeGemmaSplit {
                        part: part.range(0, part_bytes).unwrap(),
                        max_splits,
                        min_keys,
                    }),
                    out: out.range(0, n * q_row).unwrap(),
                },
            )
            .expect("many rows");
        let many = rig.read(&out, n * q_row);
        for (r, row) in rows.iter().enumerate() {
            let step = rig.read(&row.step, STEP_BYTES);
            let status = i32::from_le_bytes(step[16..20].try_into().unwrap());
            let mine = &many[r * q_row..(r + 1) * q_row];
            if (1..=CELLS as i32).contains(&kv_lens[r]) {
                assert_eq!(
                    status, 0,
                    "split {split:?}: a valid row keeps a clean status, row {r}"
                );
                let out1 = rig.filled(0x3d, q_row);
                let part1 = rig.filled(0x5a, N_Q * 8 * (2 + head_dim) * 4);
                rig.ctx
                    .attn_decode_gemma(
                        &rig.stream,
                        AttnDecodeGemmaCall {
                            out: out1.range(0, q_row).unwrap(),
                            status: None,
                            q: q.range(r * q_row, q_row).unwrap(),
                            k_codes: row.k_codes.range(0, N_KV * CELLS * head_dim).unwrap(),
                            k_scales: row.k_scales.range(0, N_KV * CELLS * 8 * 2).unwrap(),
                            v_codes: row.v_codes.range(0, N_KV * CELLS * head_dim).unwrap(),
                            v_scales: row.v_scales.range(0, N_KV * CELLS * 8 * 2).unwrap(),
                            kv_len: row.step.range(8, 4).unwrap(),
                            m: 1,
                            n_q_head: N_Q,
                            n_kv_head: N_KV,
                            head_dim,
                            cells: CELLS,
                            window: 0,
                            ring: 0,
                            scale: 1.0,
                            split: split.map(|(max_splits, min_keys)| AttnDecodeGemmaSplit {
                                part: part1.range(0, part1_bytes(max_splits, head_dim)).unwrap(),
                                max_splits,
                                min_keys,
                            }),
                        },
                    )
                    .expect("single row");
                assert_eq!(rig.read(&out1, q_row), mine, "split {split:?}: row {r}");
            } else {
                assert_eq!(status, 4, "split {split:?}: mask 4 on the refused row {r}");
                assert!(
                    mine.as_chunks::<4>()
                        .0
                        .iter()
                        .all(|word| *word == [0x00, 0x00, 0xc0, 0x7f]),
                    "split {split:?}: the refused row {r} is all NaN"
                );
            }
        }
    }
}

/// Bytes of the part scratch of one Bolt row at `max_splits`.
fn part1_bytes(max_splits: usize, head_dim: usize) -> usize {
    8 * max_splits * (2 + head_dim) * 4
}

/// The Bolt entry at the row limit, in the single-pass and the split-K paths (the split-K scratch
/// is the largest buffer of any entry: 64 splits for 256 rows).
#[test]
fn the_gemma_entry_runs_at_256_rows() {
    if !testkit::device_present("gemma many rows at the row limit") {
        return;
    }
    let rig = Rig::new();
    let mut rng = Rng::new(0x6d52_0b00);
    let kv_lens: Vec<i32> = (0..256).map(|r| (r * 7 + 1) % 191 + 1).collect();
    for (head_dim, window, ring, split) in [
        (256, 0, 0, None),
        (256, 100, 150, Some((16, 16))),
        (256, 0, 0, Some((64, 8))),
    ] {
        gemma_compare(&rig, &mut rng, &kv_lens, head_dim, window, ring, split);
    }
}

/// The attention entries at the row limit (256), and the table refusal one row past it.
#[test]
fn the_row_limit_is_256_and_257_is_refused() {
    if !testkit::device_present("many rows at the row limit") {
        return;
    }
    let rig = Rig::new();
    let positions: Vec<i32> = (0..256).map(|r| (r * 7 + 3) % 190).collect();
    chain_case(&rig, &positions, 0x6d52_0700);
    conv_case(&rig, 256, 0x6d52_0701);
    let mut rng = Rng::new(0x6d52_0702);
    let rows: Vec<RowBuffers> = (0..257)
        .map(|_| gemma_row(&rig, &mut rng, 256, 5))
        .collect();
    let records: Vec<KvRow<'_>> = rows.iter().map(|r| gemma_kv_row(r, 256)).collect();
    assert!(rig.ctx.kv_rows(&records[..256]).is_ok());
    let error = rig.ctx.kv_rows(&records).err().expect("257 rows");
    assert!(format!("{error:?}").contains("rows"), "{error:?}");
    let states: Vec<DeviceBuffer> = (0..257).map(|_| rig.filled(0, 64)).collect();
    let ranges: Vec<_> = states.iter().map(|b| b.range(0, 64).unwrap()).collect();
    assert!(rig.ctx.state_rows(&ranges).is_err());
}

/// The decode at head_dim 128 (the other instantiation of the q8 kernels).
#[test]
fn decode_rows_at_head_dim_128_are_the_bytes_of_single_row_launches() {
    if !testkit::device_present("decode many rows at head_dim 128") {
        return;
    }
    let rig = Rig::new();
    let kv_lens = [1, 63, 64, 65, 130, 191, 192, 100];
    let n = kv_lens.len();
    let head_dim = 128;
    let mut rng = Rng::new(0x6d52_0800);
    let rows: Vec<RowBuffers> = kv_lens
        .iter()
        .map(|&len| gemma_row(&rig, &mut rng, head_dim, len))
        .collect();
    let q_row = N_HEAD * head_dim * 4;
    let q = rig.up(&normal(&mut rng, n * N_HEAD * head_dim, 1.0));
    let table = rig
        .ctx
        .kv_rows(
            &rows
                .iter()
                .map(|r| gemma_kv_row(r, head_dim))
                .collect::<Vec<_>>(),
        )
        .expect("table");
    let split = AttnSplit::Q8;
    let shape = |m| AttnQ8Shape {
        m,
        n_head: N_HEAD,
        n_kv_head: N_KV,
        head_dim,
        cells: CELLS,
        ld: 0,
    };
    let part_bytes = attn_q8_part_bytes(n, N_KV, head_dim, split.max_splits).unwrap();
    let (part, out) = (
        rig.filled(SENTINEL, part_bytes),
        rig.filled(SENTINEL, n * q_row),
    );
    rig.ctx
        .attn_decode_q8_multi(
            &rig.stream,
            DecodeMultiCall {
                shape: shape(n),
                span: AttnSpan::FULL,
                split,
                scale: 1.0 / 16.0,
                q: q.range(0, n * q_row).unwrap(),
                proj: None,
                rows: &table,
                part: part.range(0, part_bytes).unwrap(),
                out: out.range(0, n * q_row).unwrap(),
            },
        )
        .expect("many rows");
    let many = rig.read(&out, n * q_row);
    for (r, row) in rows.iter().enumerate() {
        let part_1 = attn_q8_part_bytes(1, N_KV, head_dim, split.max_splits).unwrap();
        let (part, out) = (rig.filled(SENTINEL, part_1), rig.filled(SENTINEL, q_row));
        let (codes, scales) = (N_KV * CELLS * head_dim, N_KV * CELLS * (head_dim / 32) * 2);
        rig.ctx
            .attn_decode_q8(
                &rig.stream,
                AttnDecodeQ8Call {
                    shape: shape(1),
                    span: AttnSpan::FULL,
                    split,
                    scale: 1.0 / 16.0,
                    q: q.range(r * q_row, q_row).unwrap(),
                    proj: None,
                    k_codes: row.k_codes.range(0, codes).unwrap(),
                    k_scales: row.k_scales.range(0, scales).unwrap(),
                    v_codes: row.v_codes.range(0, codes).unwrap(),
                    v_scales: row.v_scales.range(0, scales).unwrap(),
                    kv_len: row.step.range(8, 4).unwrap(),
                    part: part.range(0, part_1).unwrap(),
                    out: out.range(0, q_row).unwrap(),
                },
            )
            .expect("single row");
        assert_eq!(
            rig.read(&out, q_row),
            many[r * q_row..(r + 1) * q_row],
            "head_dim 128 decode, row {r}"
        );
    }
}

/// A malformed table is refused when it is built, and a call whose planes do not hold the geometry
/// or whose outputs overlap a plane or a step block is refused before any launch.
#[test]
fn a_malformed_kv_table_or_overlapping_output_is_refused() {
    if !testkit::device_present("malformed kv rows") {
        return;
    }
    let rig = Rig::new();
    let mut rng = Rng::new(0x6d52_0900);
    let text = |error: crate::EngError| format!("{error:?}");
    let a = gemma_row(&rig, &mut rng, 256, 5);
    let b = gemma_row(&rig, &mut rng, 256, 5);
    // A step block of 16 bytes.
    let mut short = gemma_kv_row(&a, 256);
    short.step = a.step.range(0, 16).unwrap();
    assert!(rig.ctx.kv_rows(&[short]).is_err(), "a short step block");
    // A misaligned plane.
    let mut skew = gemma_kv_row(&a, 256);
    skew.k_codes = a.k_codes.range(1, 1024).unwrap();
    assert!(rig.ctx.kv_rows(&[skew]).is_err(), "a misaligned plane");
    // Two rows on one step block.
    let mut shared = gemma_kv_row(&b, 256);
    shared.step = a.step.range(0, STEP_BYTES).unwrap();
    let error = rig
        .ctx
        .kv_rows(&[gemma_kv_row(&a, 256), shared])
        .err()
        .expect("shared step");
    assert!(text(error).contains("overlap"));
    // A step block inside a plane.
    let mut inside = gemma_kv_row(&a, 256);
    inside.step = a.k_codes.range(64, STEP_BYTES).unwrap();
    assert!(
        rig.ctx.kv_rows(&[inside]).is_err(),
        "a step block in a plane"
    );
    // Planes too short for the geometry of the append.
    let scales = N_KV * CELLS * 8 * 2;
    let mut tight = gemma_kv_row(&a, 256);
    tight.v_scales = a.v_scales.range(0, scales - 2).unwrap();
    let table = rig.ctx.kv_rows(&[tight]).expect("the table itself is fine");
    let (k, v) = (rig.filled(0, N_KV * 256 * 4), rig.filled(0, N_KV * 256 * 4));
    let refused = rig.ctx.kv8_append_multi(
        &rig.stream,
        AppendMultiCall {
            shape: kv_shape(1),
            k: k.range(0, N_KV * 256 * 4).unwrap(),
            v: v.range(0, N_KV * 256 * 4).unwrap(),
            rows: &table,
        },
    );
    assert!(text(refused.expect_err("short plane")).contains("v_scales"));
    // An append input that is a plane of its row.
    let table = rig.ctx.kv_rows(&[gemma_kv_row(&a, 256)]).expect("table");
    let refused = rig.ctx.kv8_append_multi(
        &rig.stream,
        AppendMultiCall {
            shape: kv_shape(1),
            k: a.k_codes.range(0, N_KV * 256 * 4).unwrap(),
            v: v.range(0, N_KV * 256 * 4).unwrap(),
            rows: &table,
        },
    );
    assert!(refused.is_err(), "k inside a plane");
}

// ---- refusals --------------------------------------------------------------------------------

#[test]
fn a_malformed_table_or_call_is_refused_before_any_launch() {
    if !testkit::device_present("many rows refusals") {
        return;
    }
    let rig = Rig::new();
    let a = rig.filled(0, 4096);
    let whole = a.range(0, 4096).unwrap();
    let half = a.range(0, 2048).unwrap();
    let text = |error: crate::EngError| format!("{error:?}");
    assert!(text(rig.ctx.state_rows(&[]).err().expect("empty")).contains("rows"));
    assert!(
        text(rig.ctx.state_rows(&[whole, half]).err().expect("overlap")).contains("overlap"),
        "two rows that share a byte"
    );
    let misaligned = a.range(2, 64).unwrap();
    assert!(rig.ctx.state_rows(&[misaligned]).is_err());
    // A table of the wrong row count for the shape.
    let table = rig
        .ctx
        .state_rows(&[a.range(0, 1024).unwrap()])
        .expect("one row");
    let proj = rig.filled(0, CONV_LD * 8 * 4);
    let w = rig.filled(0, CH * 16);
    let out = rig.filled(0, CH * 8 * 4);
    let refused = rig.ctx.gdn_conv_multi(
        &rig.stream,
        ConvMultiCall {
            shape: ConvShape {
                m: 2,
                channels: CH,
                ld: CONV_LD,
            },
            proj: proj.range(0, CONV_LD * 8 * 4).unwrap(),
            weights: w.range(0, CH * 16).unwrap(),
            states: &table,
            out: out.range(0, CH * 8 * 4).unwrap(),
        },
    );
    assert!(text(refused.expect_err("row count")).contains("rows"));
    // A state range too short for the conv window.
    let refused = rig.ctx.gdn_conv_multi(
        &rig.stream,
        ConvMultiCall {
            shape: ConvShape {
                m: 1,
                channels: CH,
                ld: CONV_LD,
            },
            proj: proj.range(0, CONV_LD * 4).unwrap(),
            weights: w.range(0, CH * 16).unwrap(),
            states: &table,
            out: out.range(0, CH * 4).unwrap(),
        },
    );
    assert!(text(refused.expect_err("short state")).contains("state"));
}

// ---- timing ----------------------------------------------------------------------------------

/// Kernel time of one launch of `n` rows against `n` launches of one row, per layer step, at 1, 4
/// and 8 rows. Prints the numbers; asserts nothing about speed.
#[test]
fn kernel_time_at_one_four_and_eight_rows() {
    if !testkit::device_present("many rows kernel time") {
        return;
    }
    let rig = Rig::new();
    let repeats = 200;
    let time = |launch: &dyn Fn()| -> f32 {
        let (start, end) = (
            rig.ctx.event_timing().unwrap(),
            rig.ctx.event_timing().unwrap(),
        );
        for _ in 0..10 {
            launch();
        }
        rig.stream.record_timing(&start).unwrap();
        for _ in 0..repeats {
            launch();
        }
        rig.stream.record_timing(&end).unwrap();
        rig.stream.synchronize().unwrap();
        start.elapsed_ms_since(&end).unwrap() * 1000.0 / repeats as f32
    };
    let mut rng = Rng::new(0x6d52_0400);
    for n in [1usize, 4, 8] {
        // conv
        let proj = rig.up(&normal(&mut rng, n * CONV_LD, 1.0));
        let w = rig.up(&normal(&mut rng, CH * 4, 0.7));
        let states: Vec<DeviceBuffer> = (0..n)
            .map(|_| rig.up(&normal(&mut rng, 3 * CH, 1.0)))
            .collect();
        let ranges: Vec<_> = states
            .iter()
            .map(|b| b.range(0, 3 * CH * 4).unwrap())
            .collect();
        let table = rig.ctx.state_rows(&ranges).unwrap();
        let out = rig.filled(0, n * CH * 4);
        let conv_many = time(&|| {
            rig.ctx
                .gdn_conv_multi(
                    &rig.stream,
                    ConvMultiCall {
                        shape: ConvShape {
                            m: n,
                            channels: CH,
                            ld: CONV_LD,
                        },
                        proj: proj.range(0, ((n - 1) * CONV_LD + CH) * 4).unwrap(),
                        weights: w.range(0, CH * 16).unwrap(),
                        states: &table,
                        out: out.range(0, n * CH * 4).unwrap(),
                    },
                )
                .unwrap();
        });
        let conv_single = time(&|| {
            for (r, state) in states.iter().enumerate() {
                rig.ctx
                    .gdn_conv(
                        &rig.stream,
                        ConvCall {
                            shape: ConvShape {
                                m: 1,
                                channels: CH,
                                ld: CONV_LD,
                            },
                            proj: proj.range(r * CONV_LD * 4, CH * 4).unwrap(),
                            weights: w.range(0, CH * 16).unwrap(),
                            state: state.range(0, 3 * CH * 4).unwrap(),
                            out: out.range(r * CH * 4, CH * 4).unwrap(),
                        },
                    )
                    .unwrap();
            }
        });
        // delta
        let conv = rig.up(&normal(&mut rng, n * DELTA_CONV_LD, 1.0));
        let dproj = rig.up(&normal(&mut rng, n * DELTA_LD, 1.0));
        let a = rig.up(&[-0.5f32; NV]);
        let dt = rig.up(&[0.1f32; NV]);
        let nw = rig.up(&vec![1.0f32; 128]);
        let dstates: Vec<DeviceBuffer> = (0..n)
            .map(|_| rig.up(&normal(&mut rng, STATE_FLOATS, 0.2)))
            .collect();
        let dranges: Vec<_> = dstates
            .iter()
            .map(|b| b.range(0, STATE_FLOATS * 4).unwrap())
            .collect();
        let dtable = rig.ctx.state_rows(&dranges).unwrap();
        let dout = rig.filled(0, n * NV * 128 * 4);
        let weights = || DeltaWeights {
            a: a.range(0, NV * 4).unwrap(),
            dt_bias: dt.range(0, NV * 4).unwrap(),
            norm_w: nw.range(0, 512).unwrap(),
        };
        let delta_many = time(&|| {
            rig.ctx
                .gdn_delta_multi(
                    &rig.stream,
                    DeltaMultiCall {
                        shape: delta_shape(n),
                        eps: GDN_EPS,
                        conv: conv.range(0, n * DELTA_CONV_LD * 4).unwrap(),
                        proj: dproj.range(0, ((n - 1) * DELTA_LD + DELTA_LD) * 4).unwrap(),
                        weights: weights(),
                        states: &dtable,
                        out: dout.range(0, n * NV * 128 * 4).unwrap(),
                    },
                )
                .unwrap();
        });
        let delta_single = time(&|| {
            for (r, state) in dstates.iter().enumerate() {
                rig.ctx
                    .gdn_delta(
                        &rig.stream,
                        DeltaCall {
                            shape: delta_shape(1),
                            eps: GDN_EPS,
                            conv: conv
                                .range(r * DELTA_CONV_LD * 4, DELTA_CONV_LD * 4)
                                .unwrap(),
                            proj: dproj.range(r * DELTA_LD * 4, DELTA_LD * 4).unwrap(),
                            weights: weights(),
                            state: state.range(0, STATE_FLOATS * 4).unwrap(),
                            out: dout.range(r * NV * 128 * 4, NV * 128 * 4).unwrap(),
                        },
                    )
                    .unwrap();
            }
        });
        // decode at 4K keys per row would need more cells; time the geometry of the tests.
        let row_data: Vec<RowBytes> = (0..n)
            .map(|r| row_bytes(&mut rng, 150 + r as i32, 151 + r as i32))
            .collect();
        let bufs: Vec<RowBuffers> = row_data.iter().map(|r| upload_row(&rig, r)).collect();
        let ktable = rig
            .ctx
            .kv_rows(&bufs.iter().map(kv_row).collect::<Vec<_>>())
            .unwrap();
        let q = rig.up(&normal(&mut rng, n * N_HEAD * D, 1.0));
        let split = AttnSplit::Q8;
        let part_bytes = attn_q8_part_bytes(n, N_KV, D, split.max_splits).unwrap();
        let (part, dec_out) = (rig.filled(0, part_bytes), rig.filled(0, n * N_HEAD * D * 4));
        let q_row = N_HEAD * D * 4;
        let decode_many = time(&|| {
            rig.ctx
                .attn_decode_q8_multi(
                    &rig.stream,
                    DecodeMultiCall {
                        shape: decode_shape(n),
                        span: AttnSpan::FULL,
                        split,
                        scale: 1.0 / 16.0,
                        q: q.range(0, n * q_row).unwrap(),
                        proj: None,
                        rows: &ktable,
                        part: part.range(0, part_bytes).unwrap(),
                        out: dec_out.range(0, n * q_row).unwrap(),
                    },
                )
                .unwrap();
        });
        let part_1 = attn_q8_part_bytes(1, N_KV, D, split.max_splits).unwrap();
        let decode_single = time(&|| {
            for (r, row) in bufs.iter().enumerate() {
                rig.ctx
                    .attn_decode_q8(
                        &rig.stream,
                        AttnDecodeQ8Call {
                            shape: decode_shape(1),
                            span: AttnSpan::FULL,
                            split,
                            scale: 1.0 / 16.0,
                            q: q.range(r * q_row, q_row).unwrap(),
                            proj: None,
                            k_codes: row.k_codes.range(0, CODE_BYTES).unwrap(),
                            k_scales: row.k_scales.range(0, SCALE_BYTES).unwrap(),
                            v_codes: row.v_codes.range(0, CODE_BYTES).unwrap(),
                            v_scales: row.v_scales.range(0, SCALE_BYTES).unwrap(),
                            kv_len: row.step.range(8, 4).unwrap(),
                            part: part.range(0, part_1).unwrap(),
                            out: dec_out.range(r * q_row, q_row).unwrap(),
                        },
                    )
                    .unwrap();
            }
        });
        eprintln!(
            "rows={n}: us per layer step, one launch vs {n} launches: conv {conv_many:.1} vs {conv_single:.1}, delta {delta_many:.1} vs {delta_single:.1}, decode {decode_many:.1} vs {decode_single:.1}"
        );
    }
}

// ---- the fp16 cache ----------------------------------------------------------------------------

/// Bytes of one fp16 plane: `[CELLS][N_KV][D]` halves.
const PLANE16: usize = CELLS * N_KV * D * 2;

/// A row of the fp16 cache: the K and V planes (random halves) and the step block.
struct Row16Bytes {
    k: Vec<u8>,
    v: Vec<u8>,
    step: Vec<u8>,
}

fn plane16(rng: &mut Rng) -> Vec<u8> {
    (0..PLANE16 / 2)
        .flat_map(|_| testkit::f16_bits((rng.normal() * 0.7) as f32).to_le_bytes())
        .collect()
}

fn row16(rng: &mut Rng, pos: i32, kv_len: i32) -> Row16Bytes {
    let mut step = StepParams::new(1, pos.max(0))
        .expect("step")
        .to_bytes()
        .to_vec();
    step[4..8].copy_from_slice(&pos.to_le_bytes());
    step[8..12].copy_from_slice(&kv_len.to_le_bytes());
    Row16Bytes {
        k: plane16(rng),
        v: plane16(rng),
        step,
    }
}

struct Row16Buffers {
    k: DeviceBuffer,
    v: DeviceBuffer,
    step: DeviceBuffer,
}

fn upload_row16(rig: &Rig, row: &Row16Bytes) -> Row16Buffers {
    Row16Buffers {
        k: rig.up_bytes(&row.k),
        v: rig.up_bytes(&row.v),
        step: rig.up_bytes(&row.step),
    }
}

/// The table record of an fp16 row; the scale fields are not read and name the first words of
/// the planes.
fn kv_row16(buffers: &Row16Buffers) -> KvRow<'_> {
    KvRow {
        k_codes: buffers.k.range(0, PLANE16).unwrap(),
        k_scales: buffers.k.range(0, 4).unwrap(),
        v_codes: buffers.v.range(0, PLANE16).unwrap(),
        v_scales: buffers.v.range(0, 4).unwrap(),
        step: buffers.step.range(0, STEP_BYTES).unwrap(),
        cells: CELLS,
    }
}

fn halves(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|&value| testkit::f16_bits(value).to_le_bytes())
        .collect()
}

#[test]
fn the_fp16_append_writes_each_rows_rounded_cell_and_refuses_a_bad_row_alone() {
    if !testkit::device_present("kv16_append_multi against the rounded rows") {
        return;
    }
    let rig = Rig::new();
    let positions = [0, 7, 191, CELLS as i32, -1, 100];
    let n = positions.len();
    let mut rng = Rng::new(0x6d52_0401);
    let rows: Vec<Row16Bytes> = positions
        .iter()
        .map(|&p| row16(&mut rng, p, p + 1))
        .collect();
    let row_floats = N_KV * D;
    let k = normal(&mut rng, n * row_floats, 1.0);
    let v = normal(&mut rng, n * row_floats, 1.0);
    let (k_b, v_b) = (rig.up(&k), rig.up(&v));
    let bufs: Vec<Row16Buffers> = rows.iter().map(|r| upload_row16(&rig, r)).collect();
    let table = rig
        .ctx
        .kv_rows(&bufs.iter().map(kv_row16).collect::<Vec<_>>())
        .expect("table");
    rig.ctx
        .kv16_append_multi(
            &rig.stream,
            Append16MultiCall {
                n_kv_head: N_KV,
                head_dim: D,
                k: k_b.range(0, n * row_floats * 4).unwrap(),
                v: v_b.range(0, n * row_floats * 4).unwrap(),
                rows: &table,
            },
        )
        .expect("append");
    for (r, (&pos, row)) in positions.iter().zip(&rows).enumerate() {
        let (mut k_want, mut v_want) = (row.k.clone(), row.v.clone());
        let mut status = 0i32;
        if (0..CELLS as i32).contains(&pos) {
            let at = pos as usize * row_floats * 2;
            let src = r * row_floats..(r + 1) * row_floats;
            k_want[at..at + row_floats * 2].copy_from_slice(&halves(&k[src.clone()]));
            v_want[at..at + row_floats * 2].copy_from_slice(&halves(&v[src]));
        } else {
            status = 4;
        }
        assert_eq!(rig.read(&bufs[r].k, PLANE16), k_want, "k plane, row {r}");
        assert_eq!(rig.read(&bufs[r].v, PLANE16), v_want, "v plane, row {r}");
        let step = rig.read(&bufs[r].step, STEP_BYTES);
        assert_eq!(
            i32::from_le_bytes(step[16..20].try_into().unwrap()),
            status,
            "status word, row {r}"
        );
    }
}

fn attn16_shape(m: usize) -> AttnShape {
    AttnShape {
        m,
        n_head: N_HEAD,
        n_kv_head: N_KV,
        head_dim: D,
        cache_rows: CELLS,
        ld: ATTN_LD,
    }
}

/// The fp16 decode of `kv_lens.len()` rows in one launch against the single-sequence entry row by row,
/// for each span.
#[test]
fn fp16_decode_rows_are_the_bytes_of_single_row_launches() {
    if !testkit::device_present("attn_decode_multi against single-row calls") {
        return;
    }
    let rig = Rig::new();
    // Every row at a different length, so the split plan differs per row; one is past the cache.
    let kv_lens = [1, 2, 31, 32, 33, 63, 64, 65, 129, 190, 192, 0, 400];
    let n = kv_lens.len();
    let mut rng = Rng::new(0x6d52_0402);
    let rows: Vec<Row16Bytes> = kv_lens
        .iter()
        .map(|&len| row16(&mut rng, (len - 1).max(0), len))
        .collect();
    let q = normal(&mut rng, n * N_HEAD * D, 1.0);
    let proj = normal(&mut rng, n * ATTN_LD, 1.0);
    let (q_b, proj_b) = (rig.up(&q), rig.up(&proj));
    let q_row = N_HEAD * D * 4;
    let split = AttnSplit::STAGE1;
    let proj_bytes = ((n - 1) * ATTN_LD + N_HEAD * 2 * D) * 4;
    for (gated, span) in [
        (true, AttnSpan::FULL),
        (false, AttnSpan::FULL),
        (
            false,
            AttnSpan {
                window: 100,
                ring: 150,
            },
        ),
        (
            true,
            AttnSpan {
                window: 64,
                ring: 0,
            },
        ),
    ] {
        let many: Vec<Row16Buffers> = rows.iter().map(|r| upload_row16(&rig, r)).collect();
        let table = rig
            .ctx
            .kv_rows(&many.iter().map(kv_row16).collect::<Vec<_>>())
            .expect("table");
        let part_bytes = attn_q8_part_bytes(n, N_KV, D, split.max_splits).unwrap();
        let (part, out) = (
            rig.filled(SENTINEL, part_bytes),
            rig.filled(SENTINEL, n * q_row),
        );
        let proj_range = || gated.then(|| proj_b.range(0, proj_bytes).unwrap());
        rig.ctx
            .attn_decode_multi(
                &rig.stream,
                DecodeMultiCall {
                    shape: decode_shape(n),
                    span,
                    split,
                    scale: 1.0 / 16.0,
                    q: q_b.range(0, n * q_row).unwrap(),
                    proj: proj_range(),
                    rows: &table,
                    part: part.range(0, part_bytes).unwrap(),
                    out: out.range(0, n * q_row).unwrap(),
                },
            )
            .expect("many rows");
        let many_out = rig.read(&out, n * q_row);
        for (r, row) in many.iter().enumerate() {
            let part_1 = attn_q8_part_bytes(1, N_KV, D, split.max_splits).unwrap();
            let (part, out) = (rig.filled(SENTINEL, part_1), rig.filled(SENTINEL, q_row));
            rig.ctx
                .attn_decode(
                    &rig.stream,
                    AttnDecodeCall {
                        shape: attn16_shape(1),
                        span,
                        split,
                        scale: 1.0 / 16.0,
                        q: q_b.range(r * q_row, q_row).unwrap(),
                        proj: gated.then(|| proj_b.range(r * ATTN_LD * 4, ATTN_LD * 4).unwrap()),
                        k_cache: row.k.range(0, PLANE16).unwrap(),
                        v_cache: row.v.range(0, PLANE16).unwrap(),
                        kv_len: row.step.range(8, 4).unwrap(),
                        part: part.range(0, part_1).unwrap(),
                        out: out.range(0, q_row).unwrap(),
                    },
                )
                .expect("single row");
            assert_eq!(
                rig.read(&out, q_row),
                many_out[r * q_row..(r + 1) * q_row],
                "fp16 decode out, gate {gated}, span {span:?}, row {r} (kv_len {})",
                kv_lens[r]
            );
        }
    }
}

#[test]
fn the_fp16_decode_refuses_a_plane_that_is_not_16_byte_aligned() {
    if !testkit::device_present("attn_decode_multi with a misaligned plane") {
        return;
    }
    let rig = Rig::new();
    let mut rng = Rng::new(0x6d52_0403);
    let row = row16(&mut rng, 10, 11);
    let good = upload_row16(&rig, &row);
    // The K plane starts 4 bytes into its allocation: the table accepts it, the decode must not.
    let wide = rig.up_bytes(&[row.k.clone(), vec![0; 16]].concat());
    let rows = [KvRow {
        k_codes: wide.range(4, PLANE16).unwrap(),
        ..kv_row16(&good)
    }];
    let table = rig
        .ctx
        .kv_rows(&rows)
        .expect("the table takes 4-byte planes");
    let q = rig.up(&normal(&mut rng, N_HEAD * D, 1.0));
    let q_row = N_HEAD * D * 4;
    let split = AttnSplit::STAGE1;
    let part_bytes = attn_q8_part_bytes(1, N_KV, D, split.max_splits).unwrap();
    let (part, out) = (
        rig.filled(SENTINEL, part_bytes),
        rig.filled(SENTINEL, q_row),
    );
    let call = |table: &super::KvRows<'_>| {
        rig.ctx.attn_decode_multi(
            &rig.stream,
            DecodeMultiCall {
                shape: decode_shape(1),
                span: AttnSpan::FULL,
                split,
                scale: 1.0 / 16.0,
                q: q.range(0, q_row).unwrap(),
                proj: None,
                rows: table,
                part: part.range(0, part_bytes).unwrap(),
                out: out.range(0, q_row).unwrap(),
            },
        )
    };
    assert!(call(&table).is_err(), "a plane at offset 4 is refused");
    let aligned = rig.ctx.kv_rows(&[kv_row16(&good)]).expect("table");
    call(&aligned).expect("a 16-byte aligned plane decodes");
    // The append stores halves and accepts a 4-byte plane.
    let k = rig.up(&normal(&mut rng, N_KV * D, 1.0));
    let v = rig.up(&normal(&mut rng, N_KV * D, 1.0));
    rig.ctx
        .kv16_append_multi(
            &rig.stream,
            Append16MultiCall {
                n_kv_head: N_KV,
                head_dim: D,
                k: k.range(0, N_KV * D * 4).unwrap(),
                v: v.range(0, N_KV * D * 4).unwrap(),
                rows: &table,
            },
        )
        .expect("the append takes a 4-byte plane");
}
