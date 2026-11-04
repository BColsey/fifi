//! Device tests (3090) of the QSA attention wrapper: refusals answer `BadRequest` with their
//! rule's text before any native call and write nothing; a chosen block that is not a completed
//! block, or a position word outside the cache, gives a NaN output and status bits 4; and cache
//! rows past the position (NaN here) are never read. The numerics against the host attention are
//! in `family::flash_next::qsa::cuda_tests`.

use super::{QsaAttnCall, QsaAttnShape};
use crate::EngError;
use crate::kernels::cuda::resident::qsa_index::QsaPosition;
use crate::kernels::cuda::{CudaContext, DeviceBuffer, DeviceStream, device_present};

const HEAD: usize = 256;
const N_HEAD: usize = 24;
const N_KV: usize = 2;
const ROWS: usize = 64;
const RATIO: usize = 4;
const BUDGET: usize = 4;
const SPLITS: usize = 8;
const LD: usize = N_HEAD * 2 * HEAD;
const SENT: u8 = 0xA5;
const POISON_BITS: i32 = 4;
/// fp16 NaN.
const HALF_NAN: u16 = 0x7e00;

fn shape() -> QsaAttnShape {
    QsaAttnShape {
        n_head: N_HEAD,
        n_kv_head: N_KV,
        rows: ROWS,
        ratio: RATIO,
        budget_blocks: BUDGET,
        max_splits: SPLITS,
    }
}

struct Rig {
    ctx: CudaContext,
    stream: DeviceStream,
    q: DeviceBuffer,
    proj: DeviceBuffer,
    k: DeviceBuffer,
    v: DeviceBuffer,
    chosen: DeviceBuffer,
    part: DeviceBuffer,
    out: DeviceBuffer,
    words: DeviceBuffer,
    status: DeviceBuffer,
}

/// fp16 bits of `x`, which is exactly representable here.
fn to_half(x: f32) -> u16 {
    half::f16::from_f32(x).to_bits()
}

impl Rig {
    /// Cache rows `0 ..= last` hold small values, rows past `last` NaN.
    fn new(last: usize, chosen: &[i32]) -> Self {
        let ctx = CudaContext::new().expect("context");
        let stream = ctx.stream().expect("stream");
        let cache: Vec<u8> = (0..ROWS * N_KV * HEAD)
            .flat_map(|i| {
                let row = i / (N_KV * HEAD);
                let bits = if row <= last {
                    to_half(((i % 7) as f32 - 3.0) / 8.0)
                } else {
                    HALF_NAN
                };
                bits.to_le_bytes()
            })
            .collect();
        let floats = |n: usize, f: fn(usize) -> f32| -> Vec<u8> {
            (0..n).flat_map(|i| f(i).to_le_bytes()).collect()
        };
        let words: Vec<u8> = [last as i32, ROWS as i32, -1]
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect();
        let chosen: Vec<u8> = chosen.iter().flat_map(|b| b.to_le_bytes()).collect();
        Self {
            q: ctx
                .upload(&floats(N_HEAD * HEAD, |i| ((i % 5) as f32 - 2.0) / 4.0))
                .expect("q"),
            proj: ctx
                .upload(&floats(LD, |i| ((i % 3) as f32 - 1.0) / 2.0))
                .expect("proj"),
            k: ctx.upload(&cache).expect("k"),
            v: ctx.upload(&cache).expect("v"),
            chosen: ctx.upload(&chosen).expect("chosen"),
            part: ctx
                .upload(&vec![SENT; shape().part_bytes().expect("part")])
                .expect("part"),
            out: ctx.upload(&vec![SENT; N_HEAD * HEAD * 4]).expect("out"),
            words: ctx.upload(&words).expect("words"),
            status: ctx.upload(&[0u8; 4]).expect("status"),
            ctx,
            stream,
        }
    }

    fn call<'r>(&'r self, position: QsaPosition<'r>) -> QsaAttnCall<'r> {
        let range = |b: &'r DeviceBuffer| b.range(0, b.bytes()).expect("range");
        QsaAttnCall {
            shape: shape(),
            ld: LD,
            scale: 1.0 / 16.0,
            position,
            q: range(&self.q),
            proj: range(&self.proj),
            k_cache: range(&self.k),
            v_cache: range(&self.v),
            chosen: range(&self.chosen),
            part: range(&self.part),
            out: range(&self.out),
            status: Some(range(&self.status)),
        }
    }

    fn host(&self, at: usize) -> QsaAttnCall<'_> {
        self.call(QsaPosition::Host(at))
    }

    fn word(&self, index: usize, admit: usize) -> QsaAttnCall<'_> {
        self.call(QsaPosition::Device {
            word: self.words.range(index * 4, 4).expect("range"),
            admit,
        })
    }

    fn out(&self) -> Vec<f32> {
        self.stream.synchronize().expect("synchronize");
        let mut bytes = vec![0u8; N_HEAD * HEAD * 4];
        self.ctx.download(&self.out, &mut bytes).expect("download");
        let (words, _) = bytes.as_chunks::<4>();
        words.iter().map(|w| f32::from_le_bytes(*w)).collect()
    }

    fn status(&self) -> i32 {
        self.stream.synchronize().expect("synchronize");
        let mut bytes = [0u8; 4];
        self.ctx
            .download(&self.status, &mut bytes)
            .expect("download");
        i32::from_le_bytes(bytes)
    }
}

fn refused(result: Result<(), EngError>, expect: &str, case: &str) {
    match result {
        Err(EngError::BadRequest(text)) => {
            assert!(text.contains(expect), "{case}: {text:?} lacks {expect:?}");
        }
        other => panic!("{case}: expected a refusal, got {other:?}"),
    }
}

type Case = fn(&Rig) -> QsaAttnCall<'_>;

const CASES: &[(&str, Case, &str)] = &[
    (
        "heads not a multiple of the kv heads",
        |r| QsaAttnCall {
            shape: QsaAttnShape {
                n_head: 25,
                ..shape()
            },
            ..r.host(5)
        },
        "query heads per kv head",
    ),
    (
        "group of 33",
        |r| QsaAttnCall {
            shape: QsaAttnShape {
                n_head: 66,
                ..shape()
            },
            ..r.host(5)
        },
        "query heads per kv head",
    ),
    (
        "no splits",
        |r| QsaAttnCall {
            shape: QsaAttnShape {
                max_splits: 0,
                ..shape()
            },
            ..r.host(5)
        },
        "splits",
    ),
    (
        "narrow row",
        |r| QsaAttnCall {
            ld: LD - 1,
            ..r.host(5)
        },
        "cannot hold",
    ),
    (
        "scale NaN",
        |r| QsaAttnCall {
            scale: f32::NAN,
            ..r.host(5)
        },
        "scale",
    ),
    ("past the cache", |r| r.host(ROWS), "outside the cache"),
    (
        "admitted word past the cache",
        |r| r.word(0, ROWS),
        "outside the cache",
    ),
    (
        "short q",
        |r| QsaAttnCall {
            q: r.q.range(4, N_HEAD * HEAD * 4 - 4).expect("range"),
            ..r.host(5)
        },
        "q range",
    ),
    (
        "short v cache",
        |r| QsaAttnCall {
            v_cache: r.v.range(0, ROWS * N_KV * HEAD * 2 - 16).expect("range"),
            ..r.host(5)
        },
        "v_cache range",
    ),
    (
        "out over q",
        |r| QsaAttnCall {
            out: r.q.range(0, N_HEAD * HEAD * 4).expect("range"),
            ..r.host(5)
        },
        "out and q ranges overlap",
    ),
    (
        "status in the part",
        |r| QsaAttnCall {
            status: Some(r.part.range(0, 4).expect("range")),
            ..r.host(5)
        },
        "part and status ranges overlap",
    ),
];

#[test]
fn qsa_attn_refuses_bad_calls_and_writes_nothing() {
    if !device_present("qsa_attn_refuses_bad_calls_and_writes_nothing") {
        return;
    }
    let rig = Rig::new(ROWS - 1, &[0, 1, 2, 3]);
    for (case, build, expect) in CASES {
        refused(rig.ctx.qsa_attn(&rig.stream, build(&rig)), expect, case);
    }
    let out = rig.out();
    assert!(
        out.iter()
            .all(|v| v.to_bits() == u32::from_le_bytes([SENT; 4])),
        "a refused call wrote the output"
    );
    assert_eq!(rig.status(), 0, "a refusal raised a status bit");
}

#[test]
fn qsa_attn_reads_no_row_past_the_position_and_poisons_bad_input() -> Result<(), EngError> {
    if !device_present("qsa_attn_reads_no_row_past_the_position_and_poisons_bad_input") {
        return Ok(());
    }
    // Position 22: blocks 0 .. 4 complete (5 blocks), budget 4, tail 20 .. 22; rows past 22 NaN.
    let rig = Rig::new(22, &[0, 2, 3, 4]);
    rig.ctx.qsa_attn(&rig.stream, rig.host(22))?;
    assert!(
        rig.out().iter().all(|v| v.is_finite()),
        "a row past the position was read"
    );
    assert_eq!(rig.status(), 0);
    // The same position from the device word.
    rig.ctx.qsa_attn(&rig.stream, rig.word(0, 22))?;
    assert!(
        rig.out().iter().all(|v| v.is_finite()),
        "word: a row past the position was read"
    );
    // Block 5 is not complete at position 22; block -1 is no block.
    for bad in [5, -1] {
        let rig = Rig::new(22, &[0, 2, bad, 4]);
        rig.ctx.qsa_attn(&rig.stream, rig.host(22))?;
        assert!(
            rig.out().iter().all(|v| v.is_nan()),
            "block {bad}: not poisoned"
        );
        assert_eq!(rig.status(), POISON_BITS, "block {bad}: status");
    }
    // Words past the cache and -1, admitted at 22.
    for index in [1, 2] {
        let rig = Rig::new(22, &[0, 1, 2, 3]);
        rig.ctx.qsa_attn(&rig.stream, rig.word(index, 22))?;
        assert!(
            rig.out().iter().all(|v| v.is_nan()),
            "word {index}: not poisoned"
        );
        assert_eq!(rig.status(), POISON_BITS, "word {index}: status");
    }
    Ok(())
}
