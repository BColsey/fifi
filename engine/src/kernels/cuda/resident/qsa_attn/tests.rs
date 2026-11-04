//! Device tests (3090) of the QSA attention wrapper: refusals answer `BadRequest` with their
//! rule's text before any native call and write nothing (each limit, size and alignment rule on
//! its own); a chosen block that is not a completed block gives its token a NaN output and status
//! bits 4, a position word outside the cache (or a chunk running past it) gives every token a NaN
//! output and status bits 4; and cache rows past a token's position (NaN here) are never read for
//! it. The numerics against the host attention are in `family::flash_next::qsa::cuda_attn_tests`.

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
/// Tokens the buffers hold.
const TOKENS: usize = 2;
const Q_BYTES: usize = N_HEAD * HEAD * 4;
const CACHE_BYTES: usize = ROWS * N_KV * HEAD * 2;
/// Spare bytes so a misaligned range still holds a whole operand.
const SPARE: usize = 16;
const SENT: u8 = 0xA5;
const POISON_BITS: i32 = 4;
/// fp16 NaN.
const HALF_NAN: u16 = 0x7e00;

fn shape() -> QsaAttnShape {
    QsaAttnShape {
        m: 1,
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
    /// Cache rows `0 ..= last` hold small values, rows past `last` NaN. `chosen` holds the blocks
    /// of one token or of two.
    fn new(last: usize, chosen: &[i32]) -> Self {
        let ctx = CudaContext::new().expect("context");
        let stream = ctx.stream().expect("stream");
        let mut cache: Vec<u8> = (0..ROWS * N_KV * HEAD)
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
        cache.extend([0u8; SPARE]);
        let floats = |n: usize, f: fn(usize) -> f32| -> Vec<u8> {
            (0..n).flat_map(|i| f(i).to_le_bytes()).collect()
        };
        // Words: the position, the first row past the cache, -1, the last row of the cache.
        let words: Vec<u8> = [last as i32, ROWS as i32, -1, ROWS as i32 - 1]
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect();
        let chosen: Vec<u8> = chosen.iter().flat_map(|b| b.to_le_bytes()).collect();
        Self {
            q: ctx
                .upload(&floats(TOKENS * N_HEAD * HEAD + SPARE / 4, |i| {
                    ((i % 5) as f32 - 2.0) / 4.0
                }))
                .expect("q"),
            proj: ctx
                .upload(&floats(TOKENS * LD, |i| ((i % 3) as f32 - 1.0) / 2.0))
                .expect("proj"),
            k: ctx.upload(&cache).expect("k"),
            v: ctx.upload(&cache).expect("v"),
            chosen: ctx.upload(&chosen).expect("chosen"),
            part: ctx
                .upload(&vec![SENT; TOKENS * shape().part_bytes().expect("part")])
                .expect("part"),
            out: ctx.upload(&vec![SENT; TOKENS * Q_BYTES]).expect("out"),
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

    /// A call of `m` tokens from `position`, every range sized for the call.
    fn tokens<'r>(&'r self, m: usize, position: QsaPosition<'r>) -> QsaAttnCall<'r> {
        let shape = QsaAttnShape { m, ..shape() };
        let part = shape.part_bytes().expect("part");
        QsaAttnCall {
            shape,
            q: self.q.range(0, m * Q_BYTES).expect("range"),
            proj: self.proj.range(0, m * LD * 4).expect("range"),
            k_cache: self.k.range(0, CACHE_BYTES).expect("range"),
            v_cache: self.v.range(0, CACHE_BYTES).expect("range"),
            chosen: self.chosen.range(0, m * BUDGET * 4).expect("range"),
            part: self.part.range(0, part).expect("range"),
            out: self.out.range(0, m * Q_BYTES).expect("range"),
            ..self.call(position)
        }
    }

    /// The output of token `t`.
    fn out_of(&self, t: usize) -> Vec<f32> {
        self.stream.synchronize().expect("synchronize");
        let mut bytes = vec![0u8; TOKENS * Q_BYTES];
        self.ctx.download(&self.out, &mut bytes).expect("download");
        let (words, _) = bytes[t * Q_BYTES..(t + 1) * Q_BYTES].as_chunks::<4>();
        words.iter().map(|w| f32::from_le_bytes(*w)).collect()
    }

    fn out(&self) -> Vec<f32> {
        self.out_of(0)
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
        "no tokens",
        |r| QsaAttnCall {
            shape: QsaAttnShape { m: 0, ..shape() },
            ..r.host(5)
        },
        "tokens",
    ),
    (
        "too many tokens",
        |r| QsaAttnCall {
            shape: QsaAttnShape {
                m: 32769,
                ..shape()
            },
            ..r.host(5)
        },
        "tokens",
    ),
    (
        "tokens past the cache",
        |r| r.tokens(2, QsaPosition::Host(ROWS - 1)),
        "outside the cache",
    ),
    (
        "65 kv heads",
        |r| QsaAttnCall {
            shape: QsaAttnShape {
                n_head: 65,
                n_kv_head: 65,
                ..shape()
            },
            ..r.host(5)
        },
        "kv heads",
    ),
    (
        "ratio 65",
        |r| QsaAttnCall {
            shape: QsaAttnShape {
                ratio: 65,
                ..shape()
            },
            ..r.host(5)
        },
        "block ratio",
    ),
    (
        "budget past 2^24 blocks",
        |r| QsaAttnCall {
            shape: QsaAttnShape {
                budget_blocks: (1 << 24) + 1,
                ..shape()
            },
            ..r.host(5)
        },
        "budget",
    ),
    (
        "2^30 rows",
        |r| QsaAttnCall {
            shape: QsaAttnShape {
                rows: 1 << 30,
                ..shape()
            },
            ..r.host(5)
        },
        "cache rows",
    ),
    (
        "129 splits",
        |r| QsaAttnCall {
            shape: QsaAttnShape {
                max_splits: 129,
                ..shape()
            },
            ..r.host(5)
        },
        "splits",
    ),
    (
        "row past i32",
        |r| QsaAttnCall {
            ld: i32::MAX as usize + 1,
            ..r.host(5)
        },
        "exceeds i32",
    ),
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
            q: r.q.range(0, Q_BYTES - 16).expect("range"),
            ..r.host(5)
        },
        "q range holds",
    ),
    (
        "q of one token for two",
        |r| QsaAttnCall {
            q: r.q.range(0, Q_BYTES).expect("range"),
            ..r.tokens(2, QsaPosition::Host(5))
        },
        "q range holds",
    ),
    (
        "q at 8 bytes",
        |r| QsaAttnCall {
            q: r.q.range(8, Q_BYTES).expect("range"),
            ..r.host(5)
        },
        "q range starts at offset 8, which is not a multiple of 16",
    ),
    (
        "k cache at 8 bytes",
        |r| QsaAttnCall {
            k_cache: r.k.range(8, CACHE_BYTES).expect("range"),
            ..r.host(5)
        },
        "k_cache range starts at offset 8, which is not a multiple of 16",
    ),
    (
        "v cache at 8 bytes",
        |r| QsaAttnCall {
            v_cache: r.v.range(8, CACHE_BYTES).expect("range"),
            ..r.host(5)
        },
        "v_cache range starts at offset 8, which is not a multiple of 16",
    ),
    (
        "short k cache",
        |r| QsaAttnCall {
            k_cache: r.k.range(0, CACHE_BYTES - 16).expect("range"),
            ..r.host(5)
        },
        "k_cache range holds",
    ),
    (
        "short v cache",
        |r| QsaAttnCall {
            v_cache: r.v.range(0, CACHE_BYTES - 16).expect("range"),
            ..r.host(5)
        },
        "v_cache range holds",
    ),
    (
        "short proj",
        |r| QsaAttnCall {
            proj: r.proj.range(0, N_HEAD * 2 * HEAD * 4 - 4).expect("range"),
            ..r.host(5)
        },
        "proj range holds",
    ),
    (
        "proj of one token for two",
        |r| QsaAttnCall {
            proj: r
                .proj
                .range(0, LD * 4 + N_HEAD * 2 * HEAD * 4 - 4)
                .expect("range"),
            ..r.tokens(2, QsaPosition::Host(5))
        },
        "proj range holds",
    ),
    (
        "short chosen",
        |r| QsaAttnCall {
            chosen: r.chosen.range(0, BUDGET * 4 - 4).expect("range"),
            ..r.host(5)
        },
        "chosen range holds",
    ),
    (
        "short part",
        |r| QsaAttnCall {
            part: r
                .part
                .range(0, shape().part_bytes().expect("part") - 4)
                .expect("range"),
            ..r.host(5)
        },
        "part range holds",
    ),
    (
        "short out",
        |r| QsaAttnCall {
            out: r.out.range(0, Q_BYTES - 4).expect("range"),
            ..r.host(5)
        },
        "out range holds",
    ),
    (
        "short status",
        |r| QsaAttnCall {
            status: Some(r.status.range(0, 2).expect("range")),
            ..r.host(5)
        },
        "status range holds",
    ),
    (
        "short word",
        |r| {
            r.call(QsaPosition::Device {
                word: r.words.range(0, 2).expect("range"),
                admit: 5,
            })
        },
        "pos range holds",
    ),
    (
        "word at 2 bytes",
        |r| {
            r.call(QsaPosition::Device {
                word: r.words.range(2, 4).expect("range"),
                admit: 5,
            })
        },
        "pos range starts at offset 2, which is not a multiple of 4",
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
    let rig = Rig::new(ROWS - 1, &[0, 1, 2, 3, 0, 1, 2, 3]);
    for (case, build, expect) in CASES {
        refused(rig.ctx.qsa_attn(&rig.stream, build(&rig)), expect, case);
    }
    for t in 0..TOKENS {
        assert!(
            rig.out_of(t)
                .iter()
                .all(|v| v.to_bits() == u32::from_le_bytes([SENT; 4])),
            "a refused call wrote the output"
        );
    }
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

#[test]
fn qsa_attn_of_two_tokens_reads_no_row_past_each_position_and_poisons_per_token()
-> Result<(), EngError> {
    if !device_present(
        "qsa_attn_of_two_tokens_reads_no_row_past_each_position_and_poisons_per_token",
    ) {
        return Ok(());
    }
    // Tokens at 20 and 21, rows past 21 NaN. (That token 0 reads no row of token 1 is a value
    // question, checked against the host in `cuda_attn_tests`.)
    let rig = Rig::new(21, &[0, 2, 3, 4, 0, 2, 3, 4]);
    rig.ctx
        .qsa_attn(&rig.stream, rig.tokens(2, QsaPosition::Host(20)))?;
    for t in 0..2 {
        assert!(
            rig.out_of(t).iter().all(|v| v.is_finite()),
            "token {t}: a row past its position was read"
        );
    }
    assert_eq!(rig.status(), 0);
    // A process holds at most two adapter leases, so each rig goes before the next.
    drop(rig);
    // Token 0 at position 20 has 5 completed blocks; token 1's block 5 is not complete at 21.
    let rig = Rig::new(ROWS - 1, &[0, 2, 3, 4, 0, 2, 5, 4]);
    rig.ctx
        .qsa_attn(&rig.stream, rig.tokens(2, QsaPosition::Host(20)))?;
    assert!(
        rig.out_of(0).iter().all(|v| v.is_finite()),
        "token 0 was poisoned by token 1's block"
    );
    assert!(
        rig.out_of(1).iter().all(|v| v.is_nan()),
        "token 1: not poisoned"
    );
    assert_eq!(rig.status(), POISON_BITS, "bad block of token 1: status");
    drop(rig);
    // The last row from a word: one token fits, a chunk of two runs past the cache.
    let rig = Rig::new(ROWS - 1, &[0, 1, 2, 3, 0, 1, 2, 3]);
    let word = || QsaPosition::Device {
        word: rig.words.range(12, 4).expect("range"),
        admit: 5,
    };
    rig.ctx.qsa_attn(&rig.stream, rig.tokens(1, word()))?;
    assert!(
        rig.out_of(0).iter().all(|v| v.is_finite()),
        "the last row alone"
    );
    assert_eq!(rig.status(), 0);
    rig.ctx.qsa_attn(&rig.stream, rig.tokens(2, word()))?;
    for t in 0..2 {
        assert!(
            rig.out_of(t).iter().all(|v| v.is_nan()),
            "chunk past the cache: token {t} not poisoned"
        );
    }
    assert_eq!(rig.status(), POISON_BITS, "chunk past the cache: status");
    Ok(())
}
