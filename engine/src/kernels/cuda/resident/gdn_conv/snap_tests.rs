//! Tests of the strided state snapshots of the DeltaNet conv step (unit test 4 of the K-row
//! design, `kSnap = true` instantiation of `conv_silu_k`) and of the in-place read of the qkv
//! output (unit test 10): one call of `m = K` rows with [`CudaContext::gdn_conv_snap`] writes the
//! bytes of K separate single-row `gdn_conv` calls, snapshot `t` is the window after the `t + 1`-th
//! single-row call at `t * stride` bytes into a buffer laid out like the rollback slots, and a
//! projection read through its own row stride (`ld = channels`, the tight qkv output of the rows
//! kernel) gives the bytes of the same rows read from a wider row. A host without a device skips
//! them.

use super::{ConvCall, ConvShape, SnapRows};
use crate::EngError;
use crate::kernels::cuda::testkit::{self, Rng};
use crate::kernels::cuda::{CudaContext, DeviceBuffer, DeviceStream};

const CHANNELS: usize = 8192;
const STATE_BYTES: usize = 3 * CHANNELS * 4;
/// Where the conv window of layer `l` sits in a rollback slot: after the layer's 2 MiB recurrent
/// state, in a layer stride of 2 MiB + 96 KiB; a slot has 30 layers.
const REC_BYTES: usize = 2_097_152;
const LAYER_STRIDE: usize = REC_BYTES + 98_304;
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

struct Inputs {
    m: usize,
    /// `[m][ld]` rows, the first CHANNELS columns are the qkv values.
    ld: usize,
    proj: Vec<f32>,
    w: Vec<f32>,
    state0: Vec<f32>,
}

fn inputs(m: usize, ld: usize, seed: u64) -> Inputs {
    let mut rng = Rng::new(seed);
    Inputs {
        m,
        ld,
        proj: normal(&mut rng, m * ld, 1.5),
        w: normal(&mut rng, CHANNELS * 4, 0.7),
        state0: normal(&mut rng, 3 * CHANNELS, 1.0),
    }
}

fn call<'b>(
    inp: &Inputs,
    m: usize,
    (proj, w, state, out): (
        &'b DeviceBuffer,
        &'b DeviceBuffer,
        &'b DeviceBuffer,
        &'b DeviceBuffer,
    ),
    first_row: usize,
) -> ConvCall<'b> {
    ConvCall {
        shape: ConvShape {
            m,
            channels: CHANNELS,
            ld: inp.ld,
        },
        proj: proj
            .range(first_row * inp.ld * 4, ((m - 1) * inp.ld + CHANNELS) * 4)
            .unwrap(),
        weights: w.range(0, CHANNELS * 16).unwrap(),
        state: state.range(0, STATE_BYTES).unwrap(),
        out: out
            .range(first_row * CHANNELS * 4, m * CHANNELS * 4)
            .unwrap(),
    }
}

/// K single-row calls: the output bytes and the window after each row.
fn per_row(
    rig: &Rig,
    inp: &Inputs,
    proj: &DeviceBuffer,
    w: &DeviceBuffer,
) -> (Vec<u8>, Vec<Vec<u8>>) {
    let state = testkit::upload_f32(&rig.ctx, &inp.state0);
    let out = testkit::upload_filled(&rig.ctx, SENTINEL, inp.m * CHANNELS * 4);
    let mut states = Vec::new();
    for t in 0..inp.m {
        rig.ctx
            .gdn_conv(&rig.stream, call(inp, 1, (proj, w, &state, &out), t))
            .expect("single-row gdn_conv");
        states.push(testkit::read_bytes(&rig.ctx, &state, STATE_BYTES));
    }
    (
        testkit::read_bytes(&rig.ctx, &out, inp.m * CHANNELS * 4),
        states,
    )
}

/// Unit test 4 (conv): m = K against K single-row calls, outputs, final window and every
/// snapshot, at the real slot stride and conv offsets and at the contiguous stride.
#[test]
fn the_snapshots_are_the_windows_after_each_row_and_the_rest_is_unchanged() {
    if !testkit::device_present("gdn_conv_snap against K single-row calls") {
        return;
    }
    let rig = Rig::new();
    for m in 2..=8usize {
        let inp = inputs(m, CHANNELS + 12 * 5, 0x4b52_0050 + m as u64);
        let proj = testkit::upload_f32(&rig.ctx, &inp.proj);
        let w = testkit::upload_f32(&rig.ctx, &inp.w);
        let (want_out, want_states) = per_row(&rig, &inp, &proj, &w);
        for (layer, stride) in [(0usize, SLOT_BYTES), (29, SLOT_BYTES), (7, STATE_BYTES)] {
            // The conv window of `layer` follows its recurrent state inside the layer's stride.
            let base = layer * LAYER_STRIDE + REC_BYTES;
            let total = base + (m - 2) * stride + STATE_BYTES;
            let slots = testkit::upload_filled(&rig.ctx, SENTINEL, total);
            let state = testkit::upload_f32(&rig.ctx, &inp.state0);
            let out = testkit::upload_filled(&rig.ctx, SENTINEL, m * CHANNELS * 4);
            rig.ctx
                .gdn_conv_snap(
                    &rig.stream,
                    call(&inp, m, (&proj, &w, &state, &out), 0),
                    SnapRows {
                        range: slots.range(base, total - base).unwrap(),
                        stride,
                    },
                )
                .expect("gdn_conv_snap");
            assert_eq!(
                testkit::read_bytes(&rig.ctx, &out, m * CHANNELS * 4),
                want_out,
                "m {m} layer {layer}: outputs"
            );
            assert_eq!(
                testkit::read_bytes(&rig.ctx, &state, STATE_BYTES),
                want_states[m - 1],
                "m {m} layer {layer}: final window"
            );
            let got = testkit::read_bytes(&rig.ctx, &slots, total);
            let mut written = vec![false; total];
            for (t, want) in want_states.iter().enumerate().take(m - 1) {
                let at = base + t * stride;
                assert_eq!(
                    &got[at..at + STATE_BYTES],
                    &want[..],
                    "m {m} layer {layer} stride {stride}: snapshot {t}"
                );
                written[at..at + STATE_BYTES].fill(true);
            }
            for (i, byte) in got.iter().enumerate() {
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

/// Unit test 10 (conv half): the qkv output of the rows kernel is tight, `[K][8192]`, and the conv
/// kernel reads it through its own `ld = channels`. The same values in rows of a wider
/// projection (`ld = 12352`, the rest not read) give the same outputs and window, and both equal K
/// single-row calls.
#[test]
fn the_conv_reads_the_tight_qkv_rows_in_place() {
    if !testkit::device_present("gdn_conv in-place qkv read") {
        return;
    }
    let rig = Rig::new();
    for m in [2usize, 3, 4, 8] {
        let tight = inputs(m, CHANNELS, 0x4b52_0060 + m as u64);
        let mut wide = inputs(m, 12_352, 0);
        wide.w = tight.w.clone();
        wide.state0 = tight.state0.clone();
        wide.proj = vec![f32::NAN; m * 12_352];
        for t in 0..m {
            wide.proj[t * 12_352..t * 12_352 + CHANNELS]
                .copy_from_slice(&tight.proj[t * CHANNELS..(t + 1) * CHANNELS]);
        }
        let w = testkit::upload_f32(&rig.ctx, &tight.w);
        let mut results = Vec::new();
        for inp in [&tight, &wide] {
            let proj = testkit::upload_f32(&rig.ctx, &inp.proj);
            let state = testkit::upload_f32(&rig.ctx, &inp.state0);
            let out = testkit::upload_filled(&rig.ctx, SENTINEL, m * CHANNELS * 4);
            rig.ctx
                .gdn_conv(&rig.stream, call(inp, m, (&proj, &w, &state, &out), 0))
                .expect("gdn_conv");
            results.push((
                testkit::read_bytes(&rig.ctx, &out, m * CHANNELS * 4),
                testkit::read_bytes(&rig.ctx, &state, STATE_BYTES),
            ));
        }
        assert_eq!(
            results[0], results[1],
            "m {m}: tight rows against wide rows"
        );
        let proj = testkit::upload_f32(&rig.ctx, &tight.proj);
        let (want_out, want_states) = per_row(&rig, &tight, &proj, &w);
        assert_eq!(results[0].0, want_out, "m {m}: outputs against single rows");
        assert_eq!(
            results[0].1,
            want_states[m - 1],
            "m {m}: window against single rows"
        );
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
    if !testkit::device_present("gdn_conv_snap refusals") {
        return;
    }
    let rig = Rig::new();
    let inp = inputs(3, CHANNELS, 0x4b52_0070);
    let proj = testkit::upload_f32(&rig.ctx, &inp.proj);
    let w = testkit::upload_f32(&rig.ctx, &inp.w);
    let state = testkit::upload_f32(&rig.ctx, &inp.state0);
    let out = testkit::upload_filled(&rig.ctx, 0, 3 * CHANNELS * 4);
    let slots = testkit::upload_filled(&rig.ctx, 0, 3 * STATE_BYTES);
    let run = |m: usize, snap: SnapRows<'_>| {
        rig.ctx.gdn_conv_snap(
            &rig.stream,
            call(&inp, m, (&proj, &w, &state, &out), 0),
            snap,
        )
    };
    let snap = |stride: usize, bytes: usize| SnapRows {
        range: slots.range(0, bytes).unwrap(),
        stride,
    };
    refusal(
        run(1, snap(STATE_BYTES, STATE_BYTES)),
        "at least 2 tokens",
        "m = 1",
    );
    refusal(
        run(3, snap(STATE_BYTES + 2, 3 * STATE_BYTES)),
        "multiple of 4",
        "stride not a multiple of 4",
    );
    refusal(
        run(3, snap(STATE_BYTES - 4, 3 * STATE_BYTES)),
        "at least one snapshot",
        "stride below one window",
    );
    refusal(
        run(3, snap(STATE_BYTES, STATE_BYTES)),
        "snap",
        "short snapshot range",
    );
    let over_state = SnapRows {
        range: state.range(0, STATE_BYTES).unwrap(),
        stride: STATE_BYTES,
    };
    refusal(run(2, over_state), "overlap", "snapshots over the state");
    let odd = SnapRows {
        range: slots.range(2, STATE_BYTES + 4).unwrap(),
        stride: STATE_BYTES,
    };
    refusal(run(2, odd), "snap", "misaligned snapshots");
}
