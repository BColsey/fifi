//! Device tests of the many-rows entries: N rows in one launch carry the bytes of N launches of
//! one row, for the conv, the delta recurrence, the attention preparation, the KV8 append and the
//! KV8 decode attention, at the Qwen3.6 geometry. A host without a device skips them. The device
//! suite runs with `--test-threads=1`.
//!
//! Every comparison is on bytes: the single-row run is the existing entry of the same name on the
//! row's own inputs, the many-rows run is one launch over all rows, and the two start from copies
//! of the same bytes. The timing test prints the kernel time at 1, 4 and 8 rows.

use super::{
    AppendMultiCall, ConvMultiCall, DecodeGemmaMultiCall, DecodeMultiCall, DeltaMultiCall, KvRow,
    PrepMultiCall,
};
use crate::kernels::cuda::resident::attn_decode::{AttnSpan, AttnSplit};
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
                cells: CELLS,
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
    let codes_bytes = N_KV * CELLS * head_dim;
    let scales_bytes = N_KV * CELLS * (head_dim / 32) * 2;
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
    KvRow {
        k_codes: buffers.k_codes.range(0, N_KV * CELLS * head_dim).unwrap(),
        k_scales: buffers
            .k_scales
            .range(0, N_KV * CELLS * (head_dim / 32) * 2)
            .unwrap(),
        v_codes: buffers.v_codes.range(0, N_KV * CELLS * head_dim).unwrap(),
        v_scales: buffers
            .v_scales
            .range(0, N_KV * CELLS * (head_dim / 32) * 2)
            .unwrap(),
        step: buffers.step.range(0, STEP_BYTES).unwrap(),
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
                cells: CELLS,
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
                    cells: CELLS,
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
                    cells: CELLS,
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
