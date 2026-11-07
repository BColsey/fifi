//! Tests of the router launches of a K-row verify window: the 4-row chunking and the strided
//! logits launch against K single-row launches, bit
//! for bit, at the shapes of the real Qwen3.6 routers (the MoE router, 2048 x 256 with a shared
//! gate, and the GDN beta and alpha gates, 2048 x 32). The chunk table is a host test; the rest
//! needs a device and is skipped without one.

use super::{CHUNK_ROWS, RouterLogitsStridedCall, TILED_MIN_ROWS, router_chunks};
use crate::EngError;
use crate::kernels::cuda::resident::router::{
    RouterCall, RouterLogitsCall, RouterOut, RouterShape, RouterWeights,
};
use crate::kernels::cuda::testkit::{self, Rng};
use crate::kernels::cuda::{CudaContext, DeviceBuffer, DeviceStream};

const HIDDEN: usize = 2048;
const SENTINEL: u32 = 0x5a5a_a5a5;

#[test]
fn the_chunks_keep_the_decode_kernel_at_every_row_count() {
    for m in 1..TILED_MIN_ROWS {
        assert_eq!(
            router_chunks(m),
            [(0, m)],
            "m {m}: one launch below the tiled threshold"
        );
    }
    assert_eq!(router_chunks(8), [(0, 4), (4, 4)]);
    assert_eq!(router_chunks(9), [(0, 4), (4, 4), (8, 1)]);
    for m in 1..=40 {
        let chunks = router_chunks(m);
        assert_eq!(chunks.iter().map(|c| c.1).sum::<usize>(), m);
        assert!(
            chunks
                .iter()
                .all(|c| c.1 <= CHUNK_ROWS.max(TILED_MIN_ROWS - 1))
        );
        assert!(
            chunks.iter().all(|c| c.1 < TILED_MIN_ROWS),
            "no launch reaches the tiled kernel"
        );
    }
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

    fn f32s(&self, values: &[f32]) -> DeviceBuffer {
        testkit::upload_f32(&self.ctx, values)
    }

    fn read(&self, buffer: &DeviceBuffer, bytes: usize) -> Vec<u8> {
        self.stream.synchronize().expect("synchronize");
        testkit::read_bytes(&self.ctx, buffer, bytes)
    }
}

fn normal(rng: &mut Rng, n: usize, sd: f64) -> Vec<f32> {
    (0..n).map(|_| (rng.normal() * sd) as f32).collect()
}

/// Outputs of one router run, as bytes.
#[derive(Debug, PartialEq, Eq)]
struct RouterBytes {
    logits: Vec<u8>,
    ids: Vec<u8>,
    weights: Vec<u8>,
    gate: Vec<u8>,
    status: Vec<u8>,
}

/// The MoE router (2048 x 256, 8 picks, shared gate) on `m` rows: one launch of `router_rows`, or
/// `m` launches of the single-row `router`, rows concatenated.
fn run_router(
    rig: &Rig,
    x: &[f32],
    w: &[f32],
    gate_w: &[f32],
    m: usize,
    rows: bool,
) -> RouterBytes {
    let (n_expert, top_k) = (256usize, 8usize);
    let w = rig.f32s(w);
    let gate_w = rig.f32s(gate_w);
    let x = rig.f32s(x);
    let logits_bytes = m * (n_expert + 1) * 4;
    let logits = testkit::upload_filled(&rig.ctx, 0xcd, logits_bytes);
    let ids = testkit::upload_filled(&rig.ctx, 0xee, m * top_k * 4);
    let weights = testkit::upload_filled(&rig.ctx, 0xdd, m * top_k * 4);
    let gate = testkit::upload_filled(&rig.ctx, 0xbb, m * 4);
    let status = testkit::upload_filled(&rig.ctx, 0, 4);
    let shape = |m: usize| RouterShape {
        m,
        hidden: HIDDEN,
        n_expert,
        top_k,
    };
    let call = |first: usize, rows: usize| RouterCall {
        shape: shape(rows),
        x: x.range(first * HIDDEN * 4, rows * HIDDEN * 4).unwrap(),
        weights: RouterWeights {
            w: w.range(0, w.bytes()).unwrap(),
            w_shared: Some(gate_w.range(0, gate_w.bytes()).unwrap()),
        },
        logits: logits
            .range(first * (n_expert + 1) * 4, rows * (n_expert + 1) * 4)
            .unwrap(),
        out: RouterOut {
            ids: ids.range(first * top_k * 4, rows * top_k * 4).unwrap(),
            weights: weights.range(first * top_k * 4, rows * top_k * 4).unwrap(),
            shared_gate: Some(gate.range(first * 4, rows * 4).unwrap()),
            status: status.range(0, 4).unwrap(),
        },
    };
    if rows {
        rig.ctx
            .router_rows(&rig.stream, call(0, m))
            .expect("router_rows");
    } else {
        for t in 0..m {
            rig.ctx.router(&rig.stream, call(t, 1)).expect("router");
        }
    }
    RouterBytes {
        logits: rig.read(&logits, logits_bytes),
        ids: rig.read(&ids, m * top_k * 4),
        weights: rig.read(&weights, m * top_k * 4),
        gate: rig.read(&gate, m * 4),
        status: rig.read(&status, 4),
    }
}

/// The MoE router on m = 1 to 8 rows (8 as two 4-row launches) writes the bytes of m
/// single-row launches: logits, ids, weights, shared gate and status.
#[test]
fn the_router_on_k_rows_equals_k_single_row_launches() {
    if !testkit::device_present("router_rows against single-row launches") {
        return;
    }
    let rig = Rig::new();
    let mut rng = Rng::new(0x4b52_0020);
    let w = normal(&mut rng, 256 * HIDDEN, 0.05);
    let gate_w = normal(&mut rng, HIDDEN, 0.05);
    let x = normal(&mut rng, 8 * HIDDEN, 1.0);
    for m in 1..=8usize {
        let rows = run_router(&rig, &x[..m * HIDDEN], &w, &gate_w, m, true);
        let singles = run_router(&rig, &x[..m * HIDDEN], &w, &gate_w, m, false);
        assert_eq!(rows, singles, "m {m}");
    }
}

/// The control: the single 8-row launch, which runs the tiled
/// kernel, writes the same logit bits as the chunked decode kernel on these routers (hidden / 4 is
/// a multiple of 512), so the choice of two 4-row launches is about identity by construction and
/// speed, not about a known mismatch.
#[test]
fn the_tiled_single_launch_equals_the_chunked_launches_on_these_shapes() {
    if !testkit::device_present("router tiled control") {
        return;
    }
    let rig = Rig::new();
    let mut rng = Rng::new(0x4b52_0021);
    let (n_expert, m) = (256usize, 8usize);
    let w = rig.f32s(&normal(&mut rng, n_expert * HIDDEN, 0.05));
    let x = rig.f32s(&normal(&mut rng, m * HIDDEN, 1.0));
    let tight = |chunks: &[(usize, usize)]| {
        let out = testkit::upload_filled(&rig.ctx, 0xcd, m * n_expert * 4);
        for &(first, rows) in chunks {
            rig.ctx
                .router_logits_rows(
                    &rig.stream,
                    RouterLogitsCall {
                        shape: RouterShape {
                            m: rows,
                            hidden: HIDDEN,
                            n_expert,
                            top_k: 1,
                        },
                        x: x.range(first * HIDDEN * 4, rows * HIDDEN * 4).unwrap(),
                        w: w.range(0, w.bytes()).unwrap(),
                        logits: out
                            .range(first * n_expert * 4, rows * n_expert * 4)
                            .unwrap(),
                    },
                )
                .unwrap();
        }
        rig.read(&out, m * n_expert * 4)
    };
    assert_eq!(tight(&[(0, 8)]), tight(&[(0, 4), (4, 4)]));
}

/// One strided launch of the GDN gate shape (2048 x 32) into a `[m][ld]` row at column `off`,
/// next to sentinel columns; `chunked` is `router_logits_rows_strided`, otherwise m single-row
/// launches of the tight entry copied by the host to the strided positions.
fn run_strided(rig: &Rig, x: &[f32], w: &[f32], m: usize, (ld, off): (usize, usize)) -> Vec<u32> {
    let n_expert = 32usize;
    let x = rig.f32s(x);
    let w = rig.f32s(w);
    let total = (m - 1) * ld + off + n_expert;
    let out = testkit::upload_bytes(
        &rig.ctx,
        &vec![SENTINEL; total]
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect::<Vec<u8>>(),
    );
    rig.ctx
        .router_logits_rows_strided(
            &rig.stream,
            RouterLogitsStridedCall {
                shape: RouterShape {
                    m,
                    hidden: HIDDEN,
                    n_expert,
                    top_k: 1,
                },
                x: x.range(0, m * HIDDEN * 4).unwrap(),
                w: w.range(0, w.bytes()).unwrap(),
                logits: out.range(off * 4, ((m - 1) * ld + n_expert) * 4).unwrap(),
                logit_stride: ld,
            },
        )
        .expect("router_logits_rows_strided");
    rig.read(&out, total)
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| u32::from_le_bytes(*b))
        .collect()
}

/// Beta and alpha written straight into the `[K][ld]` row. Row t's 32 values sit at
/// `t * ld + off`, equal bit for bit to the single-row tight launch on that row, and no other
/// column of the destination is written, for K = 1 to 8 (8 as two 4-row launches) and the real
/// column offsets (beta 12288, alpha 12320 of a 12352-wide row).
#[test]
fn the_strided_logits_land_in_the_row_and_equal_the_tight_launch() {
    if !testkit::device_present("router_logits_rows_strided") {
        return;
    }
    let rig = Rig::new();
    let mut rng = Rng::new(0x4b52_0022);
    let n_expert = 32usize;
    let w = normal(&mut rng, n_expert * HIDDEN, 0.05);
    let x = normal(&mut rng, 8 * HIDDEN, 1.0);
    for m in 1..=8usize {
        for (ld, off) in [
            (12_352usize, 12_288usize),
            (12_352, 12_320),
            (40, 3),
            (32, 0),
        ] {
            let got = run_strided(&rig, &x[..m * HIDDEN], &w, m, (ld, off));
            for t in 0..m {
                let single = {
                    let xs = rig.f32s(&x[t * HIDDEN..(t + 1) * HIDDEN]);
                    let ws = rig.f32s(&w);
                    let out = testkit::upload_filled(&rig.ctx, 0, n_expert * 4);
                    rig.ctx
                        .router_logits_rows(
                            &rig.stream,
                            RouterLogitsCall {
                                shape: RouterShape {
                                    m: 1,
                                    hidden: HIDDEN,
                                    n_expert,
                                    top_k: 1,
                                },
                                x: xs.range(0, HIDDEN * 4).unwrap(),
                                w: ws.range(0, ws.bytes()).unwrap(),
                                logits: out.range(0, n_expert * 4).unwrap(),
                            },
                        )
                        .unwrap();
                    rig.read(&out, n_expert * 4)
                };
                let single: Vec<u32> = single
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| u32::from_le_bytes(*b))
                    .collect();
                assert_eq!(
                    &got[t * ld + off..t * ld + off + n_expert],
                    &single[..],
                    "m {m} ld {ld} off {off} row {t}"
                );
            }
            // Every word outside the rows' value columns is still a sentinel.
            for (i, word) in got.iter().enumerate() {
                let col = i % ld;
                let covered = (off..off + n_expert).contains(&col) && i / ld < m;
                if !covered {
                    assert_eq!(
                        *word, SENTINEL,
                        "m {m} ld {ld} off {off}: word {i} was written"
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
fn the_strided_wrapper_refuses_bad_arguments_before_any_native_call() {
    if !testkit::device_present("router_logits_rows_strided refusals") {
        return;
    }
    let rig = Rig::new();
    let x = rig.f32s(&vec![1.0; 8 * HIDDEN]);
    let w = rig.f32s(&vec![1.0; 32 * HIDDEN]);
    let out = rig.f32s(&vec![0.0; 8 * 64]);
    let call =
        |m: usize, hidden: usize, stride: usize, logits_floats: usize| RouterLogitsStridedCall {
            shape: RouterShape {
                m,
                hidden,
                n_expert: 32,
                top_k: 1,
            },
            x: x.range(0, 8 * HIDDEN * 4).unwrap(),
            w: w.range(0, 32 * HIDDEN * 4).unwrap(),
            logits: out.range(0, logits_floats * 4).unwrap(),
            logit_stride: stride,
        };
    let run = |c: RouterLogitsStridedCall<'_>| rig.ctx.router_logits_rows_strided(&rig.stream, c);
    refusal(run(call(0, HIDDEN, 64, 512)), "m, n_expert >= 1", "m = 0");
    refusal(
        run(call(2, 2046, 64, 512)),
        "multiple of 4",
        "hidden not a multiple of 4",
    );
    refusal(
        run(call(2, HIDDEN, 31, 512)),
        "below n_expert",
        "stride below n_expert",
    );
    // Two rows at stride 64 need 64 + 32 floats.
    refusal(run(call(2, HIDDEN, 64, 80)), "logits", "short output");
    // The output over the input.
    let aliased = RouterLogitsStridedCall {
        shape: RouterShape {
            m: 2,
            hidden: HIDDEN,
            n_expert: 32,
            top_k: 1,
        },
        x: x.range(0, 2 * HIDDEN * 4).unwrap(),
        w: w.range(0, 32 * HIDDEN * 4).unwrap(),
        logits: x.range(0, 96 * 4).unwrap(),
        logit_stride: 64,
    };
    refusal(run(aliased), "overlap", "output over x");
}
