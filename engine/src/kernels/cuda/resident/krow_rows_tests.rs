//! Unit tests 1, 2 and 7 of the K-row design for the kernels that are already row-batched: a
//! launch of `m = K` rows against K launches of `m = 1`, byte for byte. The K-row verifier relies
//! on this for the embedding gather, the add-and-norm (`add_norm_pack`), the SwiGLU and the MoE
//! combine, which it runs once per layer for the whole window; nothing in those kernels depends on
//! the row count, and these tests are the device proof of that (mutants: a launch whose rows
//! are not independent, a row stride that is not the row's own, a status that mixes rows).
//!
//! Plus the layout rule of the shared expert (mutant M21): the interleaved
//! `[gate_0, up_0, gate_1, up_1, ...]` layout is accepted by the SwiGLU's shape check with
//! `ld_in = 2 s`, and the two-groups layout `[gate K rows][up K rows]` is refused. That test
//! needs no device. A host without a device skips the others.

use crate::EngError;
use crate::kernels::cuda::native::xf_bytes;
use crate::kernels::cuda::prefill::swiglu_rows::{
    SwigluRowsCall, SwigluRowsShape, check_shape as swiglu_check,
};
use crate::kernels::cuda::resident::add_norm_pack::AddNormCall;
use crate::kernels::cuda::resident::embed_bf16::EmbedBf16Call;
use crate::kernels::cuda::resident::moe_combine::{
    CombineCall, CombineOut, CombineShape, SharedTerm,
};
use crate::kernels::cuda::testkit::{self, Rng};
use crate::kernels::cuda::{CudaContext, DeviceBuffer, DeviceStream};

const HIDDEN: usize = 2048;
const SENTINEL: u8 = 0xa7;

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

    fn read(&self, buffer: &DeviceBuffer, bytes: usize) -> Vec<u8> {
        self.stream.synchronize().expect("synchronize");
        testkit::read_bytes(&self.ctx, buffer, bytes)
    }
}

fn normal(rng: &mut Rng, n: usize, sd: f64) -> Vec<f32> {
    (0..n).map(|_| (rng.normal() * sd) as f32).collect()
}

fn words_i32(values: &[i32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

// ---- unit test 1: embed_bf16

/// One embed launch of `ids` (m = ids.len()) into a fresh output; the rows and the status word.
fn embed(rig: &Rig, table: &DeviceBuffer, n_rows: usize, ids: &[i32]) -> (Vec<u8>, i32) {
    let m = ids.len();
    let ids_buf = testkit::upload_bytes(&rig.ctx, &words_i32(ids));
    let x = testkit::upload_filled(&rig.ctx, SENTINEL, m * HIDDEN * 4);
    let status = testkit::upload_bytes(&rig.ctx, &0i32.to_le_bytes());
    rig.ctx
        .embed_bf16(
            &rig.stream,
            EmbedBf16Call {
                table: table.range(0, table.bytes()).unwrap(),
                ids: ids_buf.range(0, m * 4).unwrap(),
                x: x.range(0, m * HIDDEN * 4).unwrap(),
                status: status.range(0, 4).unwrap(),
                m,
                n_rows,
                k: HIDDEN,
            },
        )
        .expect("embed_bf16");
    let word = i32::from_le_bytes(rig.read(&status, 4).try_into().unwrap());
    (rig.read(&x, m * HIDDEN * 4), word)
}

/// Unit test 1: K ids in one launch give the rows of K one-id launches, including an
/// out-of-range id in row j (that row is NaN, the others are not, the status word has bit 4).
#[test]
fn embed_on_k_rows_equals_k_single_row_launches() {
    if !testkit::device_present("embed_bf16 m = K against K x m = 1") {
        return;
    }
    let rig = Rig::new();
    let mut rng = Rng::new(0x4b52_0080);
    let n_rows = 4_096usize;
    let table_bytes: Vec<u8> = (0..n_rows * HIDDEN)
        .flat_map(|_| ((rng.next_u64() >> 16) as u16).to_le_bytes())
        .collect();
    let table = testkit::upload_bytes(&rig.ctx, &table_bytes);
    for k in 2..=8usize {
        let ids: Vec<i32> = (0..k).map(|t| ((t * 977 + 13) % n_rows) as i32).collect();
        let (rows, status) = embed(&rig, &table, n_rows, &ids);
        assert_eq!(status, 0, "k {k}: clean ids");
        for (t, id) in ids.iter().enumerate() {
            let (single, single_status) = embed(&rig, &table, n_rows, &[*id]);
            assert_eq!(single_status, 0);
            assert_eq!(
                &rows[t * HIDDEN * 4..(t + 1) * HIDDEN * 4],
                &single[..],
                "k {k} row {t}"
            );
        }
        for bad in [0, k - 1, k / 2] {
            for bad_id in [-1i32, n_rows as i32, i32::MAX] {
                let mut with_bad = ids.clone();
                with_bad[bad] = bad_id;
                let (rows, status) = embed(&rig, &table, n_rows, &with_bad);
                assert_eq!(status, 4, "k {k} bad row {bad} id {bad_id}: status");
                let (clean, _) = embed(&rig, &table, n_rows, &ids);
                for t in 0..k {
                    let row = &rows[t * HIDDEN * 4..(t + 1) * HIDDEN * 4];
                    if t == bad {
                        assert!(
                            testkit::floats(row).iter().all(|v| v.is_nan()),
                            "k {k}: row {bad} is not all NaN"
                        );
                    } else {
                        assert_eq!(
                            row,
                            &clean[t * HIDDEN * 4..(t + 1) * HIDDEN * 4],
                            "k {k} bad {bad}: row {t}"
                        );
                    }
                }
            }
        }
    }
}

// ---- unit test 2: add_norm_pack

struct NormRun {
    h: Vec<u8>,
    norm_out: Vec<u8>,
    /// The fp16 values of the packed rows, `[m][n]`.
    packed: Vec<u16>,
}

/// One `add_norm_pack` launch of `m` rows of `h` (and `delta`, when given).
fn norm(rig: &Rig, h: &[f32], delta: Option<&[f32]>, weight: &[f32], m: usize) -> NormRun {
    let h_buf = testkit::upload_f32(&rig.ctx, h);
    let delta_buf = delta.map(|d| testkit::upload_f32(&rig.ctx, d));
    let weight_buf = testkit::upload_f32(&rig.ctx, weight);
    let xf_len = xf_bytes(m, HIDDEN).expect("xf bytes");
    let xf = testkit::upload_filled(&rig.ctx, SENTINEL, xf_len);
    let norm_out = testkit::upload_filled(&rig.ctx, SENTINEL, m * HIDDEN * 4);
    rig.ctx
        .add_norm_pack(
            &rig.stream,
            AddNormCall {
                m,
                n: HIDDEN,
                eps: 1e-6,
                h: h_buf.range(0, m * HIDDEN * 4).unwrap(),
                delta: delta_buf
                    .as_ref()
                    .map(|b| b.range(0, m * HIDDEN * 4).unwrap()),
                weight: weight_buf.range(0, HIDDEN * 4).unwrap(),
                xf: xf.range(0, xf_len).unwrap(),
                norm_out: Some(norm_out.range(0, m * HIDDEN * 4).unwrap()),
            },
        )
        .expect("add_norm_pack");
    NormRun {
        h: rig.read(&h_buf, m * HIDDEN * 4),
        norm_out: rig.read(&norm_out, m * HIDDEN * 4),
        packed: testkit::unpack_xf(&rig.read(&xf, xf_len), m, HIDDEN),
    }
}

/// Unit test 2: `add_norm_pack` of K rows (the residual add path and the plain norm) equals K
/// launches of one row: the updated residual, the f32 normalized row and the fp16 pack, row by row.
#[test]
fn add_norm_pack_on_k_rows_equals_k_single_row_launches() {
    if !testkit::device_present("add_norm_pack m = K against K x m = 1") {
        return;
    }
    let rig = Rig::new();
    let mut rng = Rng::new(0x4b52_0081);
    let weight: Vec<f32> = normal(&mut rng, HIDDEN, 0.3)
        .iter()
        .map(|w| 1.0 + w)
        .collect();
    for k in 2..=8usize {
        let h = normal(&mut rng, k * HIDDEN, 1.5);
        let delta = normal(&mut rng, k * HIDDEN, 0.5);
        for with_delta in [true, false] {
            let delta_arg = with_delta.then_some(&delta[..]);
            let batch = norm(&rig, &h, delta_arg, &weight, k);
            for t in 0..k {
                let row = t * HIDDEN..(t + 1) * HIDDEN;
                let single = norm(
                    &rig,
                    &h[row.clone()],
                    delta_arg.map(|d| &d[row.clone()]),
                    &weight,
                    1,
                );
                let bytes = row.start * 4..row.end * 4;
                assert_eq!(
                    &batch.h[bytes.clone()],
                    &single.h[..],
                    "k {k} delta {with_delta} row {t}: residual"
                );
                assert_eq!(
                    &batch.norm_out[bytes],
                    &single.norm_out[..],
                    "k {k} delta {with_delta} row {t}: norm_out"
                );
                assert_eq!(
                    &batch.packed[row],
                    &single.packed[..],
                    "k {k} delta {with_delta} row {t}: fp16 pack"
                );
            }
        }
    }
}

// ---- unit test 7: swiglu_rows, moe_combine, the shared-expert layout

fn swiglu(
    rig: &Rig,
    shape: SwigluRowsShape,
    gu: &DeviceBuffer,
    gu_off: usize,
    gu_len: usize,
) -> Vec<u8> {
    let act_len = (shape.rows - 1) * shape.ld_out + shape.n_ff;
    let act = testkit::upload_filled(&rig.ctx, SENTINEL, act_len * 4);
    rig.ctx
        .swiglu_rows(
            &rig.stream,
            SwigluRowsCall {
                shape,
                gu: gu.range(gu_off, gu_len).unwrap(),
                act: act.range(0, act_len * 4).unwrap(),
            },
        )
        .expect("swiglu_rows");
    rig.read(&act, act_len * 4)
}

/// `swiglu_rows` of K rows equals K one-row launches, on the two shapes the verifier uses: the
/// routed rows (K x top_k rows of `[gate | up]`, width 512) and the shared expert's K rows with
/// `ld_in = 2 s`, `up_off = s`.
#[test]
fn swiglu_on_k_rows_equals_k_single_row_launches() {
    if !testkit::device_present("swiglu_rows m = K against K x m = 1") {
        return;
    }
    let rig = Rig::new();
    let mut rng = Rng::new(0x4b52_0082);
    let s = 512usize;
    for k in 2..=8usize {
        // The routed rows are K * 8 picks; the shared expert's are K.
        for rows in [k, 8 * k] {
            let gu_values = normal(&mut rng, rows * 2 * s, 2.0);
            let gu = testkit::upload_f32(&rig.ctx, &gu_values);
            let shape = |rows: usize| SwigluRowsShape {
                rows,
                n_ff: s,
                up_off: s,
                ld_in: 2 * s,
                ld_out: s,
            };
            let batch = swiglu(&rig, shape(rows), &gu, 0, rows * 2 * s * 4);
            for r in 0..rows {
                let single = swiglu(&rig, shape(1), &gu, r * 2 * s * 4, 2 * s * 4);
                assert_eq!(
                    &batch[r * s * 4..(r + 1) * s * 4],
                    &single[..],
                    "k {k} rows {rows} row {r}"
                );
            }
        }
    }
}

/// Mutant M21: gate and up written as two K-row groups land as `[gate K rows][up K rows]`; one
/// SwiGLU over that would need `up_off = K s` with `ld_in = s`, which the shape check refuses
/// (`ld_in` must reach `up_off + n_ff`), whatever the `ld_in`. The interleaved layout is accepted
/// at `ld_in = 2 s`, `up_off = s` for every K. No device is needed.
#[test]
fn the_shared_expert_layout_is_the_interleaved_one() {
    let s = 512usize;
    for k in 2..=8usize {
        let interleaved = SwigluRowsShape {
            rows: k,
            n_ff: s,
            up_off: s,
            ld_in: 2 * s,
            ld_out: s,
        };
        assert!(
            swiglu_check(&interleaved).is_ok(),
            "k {k}: the interleaved layout is accepted"
        );
        for ld_in in [s, 2 * s, k * s] {
            let two_groups = SwigluRowsShape {
                rows: k,
                n_ff: s,
                up_off: k * s,
                ld_in,
                ld_out: s,
            };
            match swiglu_check(&two_groups) {
                Err(EngError::BadRequest(message)) if message.contains("up_off + n_ff") => {}
                other => {
                    panic!("k {k} ld_in {ld_in}: expected the up_off + n_ff refusal, got {other:?}")
                }
            }
        }
    }
}

struct CombineInputs {
    res: Vec<f32>,
    down: Vec<f32>,
    weights: Vec<f32>,
    gate: Vec<f32>,
    shared: Vec<f32>,
}

fn combine(rig: &Rig, inp: &CombineInputs, m: usize, first: usize, rows: usize) -> Vec<u8> {
    let top_k = 8usize;
    let res = testkit::upload_f32(&rig.ctx, &inp.res[first * HIDDEN..(first + rows) * HIDDEN]);
    let down = testkit::upload_f32(
        &rig.ctx,
        &inp.down[first * top_k * HIDDEN..(first + rows) * top_k * HIDDEN],
    );
    let weights = testkit::upload_f32(
        &rig.ctx,
        &inp.weights[first * top_k..(first + rows) * top_k],
    );
    let gate = testkit::upload_f32(&rig.ctx, &inp.gate[first..first + rows]);
    let shared = testkit::upload_f32(
        &rig.ctx,
        &inp.shared[first * HIDDEN..(first + rows) * HIDDEN],
    );
    let out = testkit::upload_filled(&rig.ctx, SENTINEL, rows * HIDDEN * 4);
    let _ = m;
    rig.ctx
        .moe_combine(
            &rig.stream,
            CombineCall {
                shape: CombineShape {
                    m: rows,
                    n: HIDDEN,
                    top_k,
                    ld_down: HIDDEN,
                },
                res: res.range(0, rows * HIDDEN * 4).unwrap(),
                down: down.range(0, rows * top_k * HIDDEN * 4).unwrap(),
                weights: weights.range(0, rows * top_k * 4).unwrap(),
                shared: Some(SharedTerm {
                    gate: gate.range(0, rows * 4).unwrap(),
                    row: shared.range(0, rows * HIDDEN * 4).unwrap(),
                }),
                out: CombineOut::To(out.range(0, rows * HIDDEN * 4).unwrap()),
            },
        )
        .expect("moe_combine");
    rig.read(&out, rows * HIDDEN * 4)
}

/// `moe_combine` of K rows (each row sums its own 8 picks in pick order, then the shared term and
/// the residual) equals K one-row launches.
#[test]
fn moe_combine_on_k_rows_equals_k_single_row_launches() {
    if !testkit::device_present("moe_combine m = K against K x m = 1") {
        return;
    }
    let rig = Rig::new();
    let mut rng = Rng::new(0x4b52_0083);
    for k in 2..=8usize {
        let inp = CombineInputs {
            res: normal(&mut rng, k * HIDDEN, 1.0),
            down: normal(&mut rng, k * 8 * HIDDEN, 1.0),
            weights: normal(&mut rng, k * 8, 0.3),
            gate: normal(&mut rng, k, 0.5),
            shared: normal(&mut rng, k * HIDDEN, 1.0),
        };
        let batch = combine(&rig, &inp, k, 0, k);
        for t in 0..k {
            let single = combine(&rig, &inp, k, t, 1);
            assert_eq!(
                &batch[t * HIDDEN * 4..(t + 1) * HIDDEN * 4],
                &single[..],
                "k {k} row {t}"
            );
        }
    }
}
