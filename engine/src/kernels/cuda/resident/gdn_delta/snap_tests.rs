//! Tests of the strided state snapshots of the DeltaNet step (`kSnap = true`
//! instantiation of `gated_delta_k`): one call of `m = K` rows with
//! [`CudaContext::gdn_delta_snap`] writes the bytes of K separate single-row `gdn_delta` calls
//! (outputs and final state), and snapshot `t` (`t < K - 1`) is the state after the `t + 1`-th
//! single-row call, stored at `t * stride` bytes into a buffer laid out like the rollback slots.
//! A host without a device skips them.

use super::{DeltaCall, DeltaShape, DeltaWeights, GDN_EPS, GDN_HEAD_DIM};
use crate::EngError;
use crate::kernels::cuda::resident::gdn_conv::SnapRows;
use crate::kernels::cuda::testkit::{self, Rng};
use crate::kernels::cuda::{CudaContext, DeviceBuffer, DeviceStream};

const D: usize = GDN_HEAD_DIM;
const NK: usize = 16;
const NV: usize = 32;
const CONV_LD: usize = (2 * NK + NV) * D;
/// A rollback slot is 30 layers of `[rec 2 MiB | conv 98304 B]`; the delta state of layer `l` sits
/// at `l * LAYER_STRIDE` of a slot.
const STATE_BYTES: usize = NV * D * D * 4;
const LAYER_STRIDE: usize = STATE_BYTES + 98_304;
const SLOT_BYTES: usize = 30 * LAYER_STRIDE;
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
}

fn normal(rng: &mut Rng, n: usize, sd: f64) -> Vec<f32> {
    (0..n).map(|_| (rng.normal() * sd) as f32).collect()
}

/// Host inputs of a window of `m` rows. `proj` rows are `[z (4096) | beta (32) | alpha (32)]` at
/// the production offsets.
struct Inputs {
    m: usize,
    ld: usize,
    conv: Vec<f32>,
    proj: Vec<f32>,
    a: Vec<f32>,
    dt_bias: Vec<f32>,
    norm_w: Vec<f32>,
    state0: Vec<f32>,
}

fn inputs(m: usize, seed: u64) -> Inputs {
    let mut rng = Rng::new(seed);
    let ld = NV * D + 2 * NV;
    let a = (0..NV).map(|_| -(0.1 + 1.9 * rng.unit() as f32)).collect();
    Inputs {
        m,
        ld,
        conv: normal(&mut rng, m * CONV_LD, 1.0),
        proj: normal(&mut rng, m * ld, 1.0),
        a,
        dt_bias: normal(&mut rng, NV, 0.5),
        norm_w: (0..D).map(|_| 0.5 + rng.unit() as f32).collect(),
        state0: normal(&mut rng, NV * D * D, 0.05),
    }
}

fn shape(inp: &Inputs, m: usize) -> DeltaShape {
    DeltaShape {
        m,
        n_k_heads: NK,
        n_v_heads: NV,
        conv_ld: CONV_LD,
        ld: inp.ld,
        z_off: 0,
        beta_off: NV * D,
        alpha_off: NV * D + NV,
    }
}

struct Bufs {
    conv: DeviceBuffer,
    proj: DeviceBuffer,
    a: DeviceBuffer,
    dt_bias: DeviceBuffer,
    norm_w: DeviceBuffer,
}

fn upload(rig: &Rig, inp: &Inputs) -> Bufs {
    Bufs {
        conv: testkit::upload_f32(&rig.ctx, &inp.conv),
        proj: testkit::upload_f32(&rig.ctx, &inp.proj),
        a: testkit::upload_f32(&rig.ctx, &inp.a),
        dt_bias: testkit::upload_f32(&rig.ctx, &inp.dt_bias),
        norm_w: testkit::upload_f32(&rig.ctx, &inp.norm_w),
    }
}

/// K single-row calls on a copy of the start state: the output bytes of every row, the state
/// after every row, and the final state.
fn per_row(rig: &Rig, inp: &Inputs, bufs: &Bufs) -> (Vec<u8>, Vec<Vec<u8>>) {
    let state = testkit::upload_f32(&rig.ctx, &inp.state0);
    let out = testkit::upload_filled(&rig.ctx, SENTINEL, inp.m * NV * D * 4);
    let mut states = Vec::new();
    for t in 0..inp.m {
        rig.ctx
            .gdn_delta(
                &rig.stream,
                DeltaCall {
                    shape: shape(inp, 1),
                    eps: GDN_EPS,
                    conv: bufs.conv.range(t * CONV_LD * 4, CONV_LD * 4).unwrap(),
                    proj: bufs.proj.range(t * inp.ld * 4, inp.ld * 4).unwrap(),
                    weights: DeltaWeights {
                        a: bufs.a.range(0, NV * 4).unwrap(),
                        dt_bias: bufs.dt_bias.range(0, NV * 4).unwrap(),
                        norm_w: bufs.norm_w.range(0, D * 4).unwrap(),
                    },
                    state: state.range(0, STATE_BYTES).unwrap(),
                    out: out.range(t * NV * D * 4, NV * D * 4).unwrap(),
                },
            )
            .expect("single-row gdn_delta");
        states.push(testkit::read_bytes(&rig.ctx, &state, STATE_BYTES));
    }
    (
        testkit::read_bytes(&rig.ctx, &out, inp.m * NV * D * 4),
        states,
    )
}

/// One call of m rows with snapshots at `stride`, into a buffer of `slots` rollback slots whose
/// layer `layer` holds the snapshots; returns the output bytes, the final state, and the slot
/// buffer after the call.
fn snapped(
    rig: &Rig,
    inp: &Inputs,
    bufs: &Bufs,
    (layer, stride): (usize, usize),
) -> (Vec<u8>, Vec<u8>, Vec<u8>, usize) {
    let m = inp.m;
    let total = (m - 2) * stride + layer * LAYER_STRIDE + STATE_BYTES;
    let slots = testkit::upload_filled(&rig.ctx, SENTINEL, total);
    let state = testkit::upload_f32(&rig.ctx, &inp.state0);
    let out = testkit::upload_filled(&rig.ctx, SENTINEL, m * NV * D * 4);
    rig.ctx
        .gdn_delta_snap(
            &rig.stream,
            DeltaCall {
                shape: shape(inp, m),
                eps: GDN_EPS,
                conv: bufs.conv.range(0, m * CONV_LD * 4).unwrap(),
                proj: bufs.proj.range(0, m * inp.ld * 4).unwrap(),
                weights: DeltaWeights {
                    a: bufs.a.range(0, NV * 4).unwrap(),
                    dt_bias: bufs.dt_bias.range(0, NV * 4).unwrap(),
                    norm_w: bufs.norm_w.range(0, D * 4).unwrap(),
                },
                state: state.range(0, STATE_BYTES).unwrap(),
                out: out.range(0, m * NV * D * 4).unwrap(),
            },
            SnapRows {
                range: slots
                    .range(layer * LAYER_STRIDE, total - layer * LAYER_STRIDE)
                    .unwrap(),
                stride,
            },
        )
        .expect("gdn_delta_snap");
    (
        testkit::read_bytes(&rig.ctx, &out, m * NV * D * 4),
        testkit::read_bytes(&rig.ctx, &state, STATE_BYTES),
        testkit::read_bytes(&rig.ctx, &slots, total),
        total,
    )
}

/// m = K against K single-row calls, outputs, final state and every
/// snapshot, for K = 2 to 8, at the real slot stride and layer offsets (the first and the last
/// layer of a slot) and at the contiguous stride.
#[test]
fn the_snapshots_are_the_states_after_each_row_and_the_rest_is_unchanged() {
    if !testkit::device_present("gdn_delta_snap against K single-row calls") {
        return;
    }
    let rig = Rig::new();
    for m in 2..=8usize {
        let inp = inputs(m, 0x4b52_0030 + m as u64);
        let bufs = upload(&rig, &inp);
        let (want_out, want_states) = per_row(&rig, &inp, &bufs);
        for (layer, stride) in [(0usize, SLOT_BYTES), (29, SLOT_BYTES), (3, STATE_BYTES)] {
            let (out, state, slots, total) = snapped(&rig, &inp, &bufs, (layer, stride));
            assert_eq!(
                out, want_out,
                "m {m} layer {layer} stride {stride}: outputs"
            );
            assert_eq!(&state, &want_states[m - 1], "m {m}: final state");
            let base = layer * LAYER_STRIDE;
            let mut written = vec![false; total];
            for (t, want) in want_states.iter().enumerate().take(m - 1) {
                let at = base + t * stride;
                assert_eq!(
                    &slots[at..at + STATE_BYTES],
                    &want[..],
                    "m {m} layer {layer} stride {stride}: snapshot {t}"
                );
                written[at..at + STATE_BYTES].fill(true);
            }
            for (i, byte) in slots.iter().enumerate() {
                if !written[i] {
                    assert_eq!(
                        *byte, SENTINEL,
                        "m {m} layer {layer} stride {stride}: byte {i} was written"
                    );
                }
            }
        }
    }
}

fn refusal(result: Result<(), EngError>, needle: &str, case: &str) {
    match result {
        Err(EngError::BadRequest(message)) if message.contains(needle) => {}
        other => panic!("{case}: expected BadRequest containing {needle:?}, got {other:?}"),
    }
}

#[test]
fn the_snapshot_wrapper_refuses_bad_arguments_before_any_native_call() {
    if !testkit::device_present("gdn_delta_snap refusals") {
        return;
    }
    let rig = Rig::new();
    let inp = inputs(3, 0x4b52_0040);
    let bufs = upload(&rig, &inp);
    let state = testkit::upload_f32(&rig.ctx, &inp.state0);
    let out = testkit::upload_filled(&rig.ctx, 0, 8 * NV * D * 4);
    let slots = testkit::upload_filled(&rig.ctx, 0, 3 * STATE_BYTES);
    let call = |m: usize| DeltaCall {
        shape: shape(&inp, m),
        eps: GDN_EPS,
        conv: bufs.conv.range(0, m.min(3) * CONV_LD * 4).unwrap(),
        proj: bufs.proj.range(0, m.min(3) * inp.ld * 4).unwrap(),
        weights: DeltaWeights {
            a: bufs.a.range(0, NV * 4).unwrap(),
            dt_bias: bufs.dt_bias.range(0, NV * 4).unwrap(),
            norm_w: bufs.norm_w.range(0, D * 4).unwrap(),
        },
        state: state.range(0, STATE_BYTES).unwrap(),
        out: out.range(0, m.min(3) * NV * D * 4).unwrap(),
    };
    let run = |c: DeltaCall<'_>, snap: SnapRows<'_>| rig.ctx.gdn_delta_snap(&rig.stream, c, snap);
    let ok_snap = |stride: usize, bytes: usize| SnapRows {
        range: slots.range(0, bytes).unwrap(),
        stride,
    };
    refusal(
        run(call(1), ok_snap(STATE_BYTES, STATE_BYTES)),
        "at least 2 tokens",
        "m = 1",
    );
    refusal(
        run(call(3), ok_snap(STATE_BYTES + 2, 3 * STATE_BYTES)),
        "multiple of 4",
        "stride not a multiple of 4",
    );
    refusal(
        run(call(3), ok_snap(STATE_BYTES - 4, 3 * STATE_BYTES)),
        "at least one snapshot",
        "stride below one state",
    );
    // 3 rows at stride STATE_BYTES need 2 * STATE_BYTES of snapshots; one state is too short.
    refusal(
        run(call(3), ok_snap(STATE_BYTES, STATE_BYTES)),
        "snap",
        "short snapshot range",
    );
    // The snapshots over the live state.
    let over_state = SnapRows {
        range: state.range(0, STATE_BYTES).unwrap(),
        stride: STATE_BYTES,
    };
    refusal(
        run(call(2), over_state),
        "overlap",
        "snapshots over the state",
    );
    // A range at a 2-byte offset is not 4-byte aligned.
    let odd = SnapRows {
        range: slots.range(2, STATE_BYTES + 4).unwrap(),
        stride: STATE_BYTES,
    };
    refusal(run(call(2), odd), "snap", "misaligned snapshots");
}
