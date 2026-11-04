//! Device tests (3090) of the QSA indexer wrapper: every refusal answers `BadRequest` with its
//! rule's text before any native call and writes nothing, and the device poison rule of a
//! position word outside the cache (or a chunk running past it) writes NaN queries, no key, ring,
//! score or selection, and raises status bits 4. The selection is checked against a host ranking
//! at the 256K-token size (65536 blocks, budget 512) with ties at the threshold, and one score and
//! selection call of 32768 tokens equals 32768 single-token calls bit for bit. The numerics
//! against the host indexer are in `family::flash_next::qsa::cuda_tests`.

use super::{QsaIndexCall, QsaIndexer, QsaPosition, QsaScoreCall, QsaSelectCall};
use crate::EngError;
use crate::kernels::cuda::{CudaContext, DeviceBuffer, DeviceRange, DeviceStream, device_present};

const HEADS: usize = 4;
const DIM: usize = 128;
const RATIO: usize = 4;
const CAP_BLOCKS: usize = 8;
const M: usize = 6;
const LD: usize = (HEADS + 1) * DIM;
const SENT: u8 = 0xA5;
const POISON_BITS: i32 = 4;
const QUERY_BYTES: usize = M * HEADS * DIM * 4;
const BUDGET_BLOCKS: usize = 4;

fn indexer() -> QsaIndexer {
    QsaIndexer {
        heads: HEADS,
        dim: DIM,
        rotary_dim: 64,
        theta: 1.0e7,
        eps: 1e-6,
        ratio: RATIO,
        cap_blocks: CAP_BLOCKS,
    }
}

struct Rig {
    ctx: CudaContext,
    stream: DeviceStream,
    proj: DeviceBuffer,
    norm: DeviceBuffer,
    queries: DeviceBuffer,
    ring: DeviceBuffer,
    keys: DeviceBuffer,
    scores: DeviceBuffer,
    chosen: DeviceBuffer,
    words: DeviceBuffer,
    status: DeviceBuffer,
}

impl Rig {
    fn new() -> Self {
        let ctx = CudaContext::new().expect("context");
        let stream = ctx.stream().expect("stream");
        let filled = |bytes: usize| ctx.upload(&vec![SENT; bytes]).expect("upload");
        let ones: Vec<u8> = (0..M * LD)
            .flat_map(|i| (i as f32 / 997.0).to_le_bytes())
            .collect();
        // Words: 0, the first position past the cache, -1, the last position of the cache.
        let words: Vec<u8> = [
            0i32,
            (CAP_BLOCKS * RATIO) as i32,
            -1,
            (CAP_BLOCKS * RATIO - 1) as i32,
        ]
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect();
        Self {
            proj: ctx.upload(&ones).expect("proj"),
            norm: ctx.upload(&vec![0u8; DIM * 4]).expect("norm"),
            // 16 spare bytes so a misaligned range still holds the queries.
            queries: filled(QUERY_BYTES + 16),
            ring: filled(RATIO * DIM * 4),
            keys: filled(CAP_BLOCKS * DIM * 4),
            // Room for two tokens, for the chunk that runs past the cache.
            scores: filled(2 * CAP_BLOCKS * 4),
            chosen: filled(2 * BUDGET_BLOCKS * 4),
            words: ctx.upload(&words).expect("words"),
            status: ctx.upload(&[0u8; 4]).expect("status"),
            ctx,
            stream,
        }
    }

    fn index(&self) -> QsaIndexCall<'_> {
        QsaIndexCall {
            indexer: indexer(),
            m: M,
            ld: LD,
            col_off: 0,
            position: QsaPosition::Host(0),
            proj: self.proj.range(0, M * LD * 4).expect("range"),
            q_norm: self.norm.range(0, DIM * 4).expect("range"),
            k_norm: self.norm.range(0, DIM * 4).expect("range"),
            queries: self.queries.range(0, QUERY_BYTES).expect("range"),
            ring: self.ring.range(0, RATIO * DIM * 4).expect("range"),
            keys: self.keys.range(0, CAP_BLOCKS * DIM * 4).expect("range"),
            status: Some(self.status.range(0, 4).expect("range")),
        }
    }

    fn score(&self) -> QsaScoreCall<'_> {
        QsaScoreCall {
            indexer: indexer(),
            m: 1,
            position: QsaPosition::Host(5),
            queries: self.queries.range(0, HEADS * DIM * 4).expect("range"),
            keys: self.keys.range(0, CAP_BLOCKS * DIM * 4).expect("range"),
            scores: self.scores.range(0, CAP_BLOCKS * 4).expect("range"),
            status: Some(self.status.range(0, 4).expect("range")),
        }
    }

    fn select(&self) -> QsaSelectCall<'_> {
        QsaSelectCall {
            indexer: indexer(),
            m: 1,
            budget_blocks: BUDGET_BLOCKS,
            position: QsaPosition::Host(5),
            scores: self.scores.range(0, CAP_BLOCKS * 4).expect("range"),
            blocks: self.chosen.range(0, BUDGET_BLOCKS * 4).expect("range"),
            status: Some(self.status.range(0, 4).expect("range")),
        }
    }

    fn word(&self, index: usize, admit: usize) -> QsaPosition<'_> {
        QsaPosition::Device {
            word: self.words.range(index * 4, 4).expect("range"),
            admit,
        }
    }

    fn bytes(&self, buffer: &DeviceBuffer) -> Vec<u8> {
        self.stream.synchronize().expect("synchronize");
        let mut out = vec![0u8; buffer.bytes()];
        self.ctx.download(buffer, &mut out).expect("download");
        out
    }

    fn untouched(&self, buffer: &DeviceBuffer, what: &str) {
        assert!(
            self.bytes(buffer).iter().all(|&b| b == SENT),
            "{what} was written"
        );
    }

    fn status(&self) -> i32 {
        let bytes = self.bytes(&self.status);
        i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
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

type IndexEdit = fn(&mut QsaIndexCall<'_>);

const INDEX_CASES: &[(&str, IndexEdit, &str)] = &[
    ("no tokens", |c| c.m = 0, "tokens"),
    ("too many tokens", |c| c.m = 32769, "tokens"),
    ("no heads", |c| c.indexer.heads = 0, "heads"),
    ("odd head size", |c| c.indexer.dim = 48, "multiple"),
    (
        "too many query values",
        |c| {
            c.indexer.heads = 33;
            c.indexer.dim = 256;
        },
        "query values",
    ),
    ("ratio 0", |c| c.indexer.ratio = 0, "ratio"),
    ("ratio 65", |c| c.indexer.ratio = 65, "ratio"),
    ("no cache", |c| c.indexer.cap_blocks = 0, "cache blocks"),
    ("odd rotary", |c| c.indexer.rotary_dim = 63, "rotary"),
    (
        "rotary past the head",
        |c| c.indexer.rotary_dim = 130,
        "rotary",
    ),
    ("theta 0", |c| c.indexer.theta = 0.0, "theta"),
    ("theta NaN", |c| c.indexer.theta = f32::NAN, "theta"),
    ("theta inf", |c| c.indexer.theta = f32::INFINITY, "theta"),
    ("eps negative", |c| c.indexer.eps = -1.0, "eps"),
    ("eps NaN", |c| c.indexer.eps = f32::NAN, "eps"),
    ("narrow row", |c| c.ld = LD - 1, "cannot hold"),
    (
        "past the cache",
        |c| c.position = QsaPosition::Host(27),
        "exceeds the cache",
    ),
    ("short proj", |c| c.m = M + 1, "proj range"),
];

type IndexCase = fn(&Rig) -> QsaIndexCall<'_>;

const RANGE_CASES: &[(&str, IndexCase, &str)] = &[
    (
        "misaligned queries",
        |r| QsaIndexCall {
            queries: r.queries.range(2, QUERY_BYTES).expect("range"),
            ..r.index()
        },
        "not a multiple of 4",
    ),
    (
        "short keys",
        |r| QsaIndexCall {
            keys: r.keys.range(0, CAP_BLOCKS * DIM * 4 - 4).expect("range"),
            ..r.index()
        },
        "keys range",
    ),
    (
        "short ring",
        |r| QsaIndexCall {
            ring: r.ring.range(0, RATIO * DIM * 4 - 4).expect("range"),
            ..r.index()
        },
        "ring range",
    ),
    (
        "queries over the ring",
        |r| QsaIndexCall {
            ring: r.queries.range(0, RATIO * DIM * 4).expect("range"),
            ..r.index()
        },
        "queries and ring ranges overlap",
    ),
    (
        "status in the keys",
        |r| QsaIndexCall {
            status: Some(r.keys.range(0, 4).expect("range")),
            ..r.index()
        },
        "keys and status ranges overlap",
    ),
    (
        "keys over the norm",
        |r| QsaIndexCall {
            k_norm: r.keys.range(0, DIM * 4).expect("range"),
            ..r.index()
        },
        "keys and k_norm ranges overlap",
    ),
    (
        "word in the status",
        |r| QsaIndexCall {
            position: QsaPosition::Device {
                word: r.status.range(0, 4).expect("range"),
                admit: 0,
            },
            ..r.index()
        },
        "status and pos ranges overlap",
    ),
];

type ScoreCase = fn(&Rig) -> QsaScoreCall<'_>;

const SCORE_CASES: &[(&str, ScoreCase, &str)] = &[
    (
        "no tokens",
        |r| QsaScoreCall { m: 0, ..r.score() },
        "tokens",
    ),
    (
        "too many tokens",
        |r| QsaScoreCall {
            m: 32769,
            ..r.score()
        },
        "tokens",
    ),
    (
        "tokens past the cache",
        |r| QsaScoreCall {
            m: 2,
            position: QsaPosition::Host(CAP_BLOCKS * RATIO - 1),
            ..r.score()
        },
        "exceeds the cache",
    ),
    (
        "queries of one token for two",
        |r| QsaScoreCall { m: 2, ..r.score() },
        "queries range",
    ),
    (
        "short scores",
        |r| QsaScoreCall {
            scores: r.scores.range(0, CAP_BLOCKS * 4 - 4).expect("range"),
            ..r.score()
        },
        "scores range",
    ),
    (
        "scores over the keys",
        |r| QsaScoreCall {
            scores: r.keys.range(0, CAP_BLOCKS * 4).expect("range"),
            ..r.score()
        },
        "scores and keys ranges overlap",
    ),
    (
        "past the cache",
        |r| QsaScoreCall {
            position: QsaPosition::Host(CAP_BLOCKS * RATIO),
            ..r.score()
        },
        "exceeds the cache",
    ),
];

type SelectCase = fn(&Rig) -> QsaSelectCall<'_>;

const SELECT_CASES: &[(&str, SelectCase, &str)] = &[
    (
        "no tokens",
        |r| QsaSelectCall { m: 0, ..r.select() },
        "tokens",
    ),
    (
        "too many tokens",
        |r| QsaSelectCall {
            m: 32769,
            ..r.select()
        },
        "tokens",
    ),
    (
        "tokens past the cache",
        |r| QsaSelectCall {
            m: 2,
            position: QsaPosition::Host(CAP_BLOCKS * RATIO - 1),
            ..r.select()
        },
        "exceeds the cache",
    ),
    (
        "scores of one token for two",
        |r| QsaSelectCall { m: 2, ..r.select() },
        "scores range",
    ),
    (
        "no budget",
        |r| QsaSelectCall {
            budget_blocks: 0,
            ..r.select()
        },
        "budget",
    ),
    (
        "budget past the cache",
        |r| QsaSelectCall {
            budget_blocks: CAP_BLOCKS + 1,
            ..r.select()
        },
        "budget",
    ),
    (
        "short blocks",
        |r| QsaSelectCall {
            blocks: r.chosen.range(0, BUDGET_BLOCKS * 4 - 4).expect("range"),
            ..r.select()
        },
        "blocks range",
    ),
    (
        "blocks over the scores",
        |r| QsaSelectCall {
            blocks: r.scores.range(0, BUDGET_BLOCKS * 4).expect("range"),
            ..r.select()
        },
        "blocks and scores ranges overlap",
    ),
    (
        "past the cache",
        |r| QsaSelectCall {
            position: QsaPosition::Host(CAP_BLOCKS * RATIO),
            ..r.select()
        },
        "exceeds the cache",
    ),
];

#[test]
fn qsa_index_refuses_bad_calls_and_writes_nothing() {
    if !device_present("qsa_index_refuses_bad_calls_and_writes_nothing") {
        return;
    }
    let rig = Rig::new();
    for (case, edit, expect) in INDEX_CASES {
        let mut call = rig.index();
        edit(&mut call);
        refused(rig.ctx.qsa_index(&rig.stream, call), expect, case);
    }
    for (case, build, expect) in RANGE_CASES {
        refused(rig.ctx.qsa_index(&rig.stream, build(&rig)), expect, case);
    }
    for (case, build, expect) in SCORE_CASES {
        refused(rig.ctx.qsa_score(&rig.stream, build(&rig)), expect, case);
    }
    for (case, build, expect) in SELECT_CASES {
        refused(rig.ctx.qsa_select(&rig.stream, build(&rig)), expect, case);
    }
    rig.untouched(&rig.queries, "queries");
    rig.untouched(&rig.ring, "ring");
    rig.untouched(&rig.keys, "keys");
    rig.untouched(&rig.scores, "scores");
    rig.untouched(&rig.chosen, "blocks");
    assert_eq!(rig.status(), 0, "a refusal raised a status bit");
}

#[test]
fn a_position_word_outside_the_cache_poisons_and_writes_no_key() -> Result<(), EngError> {
    if !device_present("a_position_word_outside_the_cache_poisons_and_writes_no_key") {
        return Ok(());
    }
    // Word 1 is the first position past the cache, word 2 is -1; the host value admits both.
    for index in [1, 2] {
        let rig = Rig::new();
        let mut call = rig.index();
        call.position = rig.word(index, 0);
        rig.ctx.qsa_index(&rig.stream, call)?;
        let queries = rig.bytes(&rig.queries);
        let (words, _) = queries[..QUERY_BYTES].as_chunks::<4>();
        assert!(
            words.iter().all(|w| u32::from_le_bytes(*w) == 0x7fc0_0000),
            "word {index}: the queries are not all NaN"
        );
        rig.untouched(&rig.ring, "ring");
        rig.untouched(&rig.keys, "keys");
        assert_eq!(rig.status(), POISON_BITS, "word {index}: index status");
        let mut score = rig.score();
        score.position = rig.word(index, 5);
        score.status = Some(rig.status.range(0, 4)?);
        rig.ctx.upload_at(&rig.status, 0, &[0u8; 4])?;
        rig.ctx.qsa_score(&rig.stream, score)?;
        rig.untouched(&rig.scores, "scores");
        assert_eq!(rig.status(), POISON_BITS, "word {index}: score status");
        let mut select = rig.select();
        select.position = rig.word(index, 5);
        rig.ctx.upload_at(&rig.status, 0, &[0u8; 4])?;
        rig.ctx.qsa_select(&rig.stream, select)?;
        rig.untouched(&rig.chosen, "blocks");
        assert_eq!(rig.status(), POISON_BITS, "word {index}: select status");
    }
    // Word 3 is the last position: one token fits, a chunk of two runs past the cache.
    let rig = Rig::new();
    let score = QsaScoreCall {
        m: 2,
        position: rig.word(3, 5),
        queries: rig.queries.range(0, 2 * HEADS * DIM * 4)?,
        scores: rig.scores.range(0, 2 * CAP_BLOCKS * 4)?,
        ..rig.score()
    };
    rig.ctx.qsa_score(&rig.stream, score)?;
    rig.untouched(&rig.scores, "scores of the chunk past the cache");
    assert_eq!(
        rig.status(),
        POISON_BITS,
        "chunk past the cache: score status"
    );
    rig.ctx.upload_at(&rig.status, 0, &[0u8; 4])?;
    let select = QsaSelectCall {
        m: 2,
        position: rig.word(3, 5),
        scores: rig.scores.range(0, 2 * CAP_BLOCKS * 4)?,
        blocks: rig.chosen.range(0, 2 * BUDGET_BLOCKS * 4)?,
        ..rig.select()
    };
    rig.ctx.qsa_select(&rig.stream, select)?;
    rig.untouched(&rig.chosen, "blocks of the chunk past the cache");
    assert_eq!(
        rig.status(),
        POISON_BITS,
        "chunk past the cache: select status"
    );
    // A process holds at most two adapter leases, so each rig goes before the next.
    drop(rig);
    // Word 0 is in range: the same call writes the two completed blocks and no poison.
    let rig = Rig::new();
    let mut call = rig.index();
    call.position = rig.word(0, 0);
    rig.ctx.qsa_index(&rig.stream, call)?;
    assert_eq!(rig.status(), 0, "an in-range word raised status");
    let keys = rig.bytes(&rig.keys);
    assert!(
        keys[..DIM * 4].iter().any(|&b| b != SENT),
        "block 0 was not written"
    );
    assert!(
        keys[DIM * 4..2 * DIM * 4].iter().all(|&b| b == SENT),
        "block 1 is not complete"
    );
    Ok(())
}

/// The host ranking of `scores` (descending, the lower block first on ties, a negative or NaN
/// score as +0), the first `take` blocks, ascending.
fn host_selection(scores: &[f32], take: usize) -> Vec<i32> {
    let key = |s: f32| if s > 0.0 { s } else { 0.0 };
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by(|&a, &b| key(scores[b]).total_cmp(&key(scores[a])).then(a.cmp(&b)));
    let mut chosen: Vec<i32> = order[..take]
        .iter()
        .map(|&b| i32::try_from(b).expect("block index"))
        .collect();
    chosen.sort_unstable();
    chosen
}

#[test]
fn qsa_select_matches_the_host_ranking_at_256k_tokens() -> Result<(), EngError> {
    if !device_present("qsa_select_matches_the_host_ranking_at_256k_tokens") {
        return Ok(());
    }
    const BLOCKS: usize = 65_536;
    const BUDGET: usize = 512;
    let indexer = QsaIndexer {
        cap_blocks: BLOCKS,
        ..indexer()
    };
    // 64 distinct values, so about 1000 blocks share each one and the threshold value is
    // tied; a few negative, -0 and NaN scores stand for zero.
    let mut state = 0x005e_1ec7_u64;
    let mut scores: Vec<f32> = (0..BLOCKS)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            ((state >> 58) as f32) * 0.25
        })
        .collect();
    for (i, bad) in [
        (7, -1.0),
        (300, f32::NAN),
        (4000, -0.0),
        (65_535, f32::NEG_INFINITY),
    ] {
        scores[i] = bad;
    }
    let ctx = CudaContext::new()?;
    let stream = ctx.stream()?;
    let bytes: Vec<u8> = scores.iter().flat_map(|v| v.to_le_bytes()).collect();
    let score_buf = ctx.upload(&bytes)?;
    let chosen = ctx.upload(&vec![SENT; BUDGET * 4])?;
    // Positions: the last of the cache (65536 blocks), one before a block ends (fewer blocks),
    // a position with fewer completed blocks than the budget, and one with none.
    let full = BLOCKS * RATIO - 1;
    for pos in [full, full - 1, 4 * 300 + 2, 2] {
        ctx.upload_at(&chosen, 0, &vec![SENT; BUDGET * 4])?;
        ctx.qsa_select(
            &stream,
            QsaSelectCall {
                indexer,
                m: 1,
                budget_blocks: BUDGET,
                position: QsaPosition::Host(pos),
                scores: score_buf.range(0, BLOCKS * 4)?,
                blocks: chosen.range(0, BUDGET * 4)?,
                status: None,
            },
        )?;
        stream.synchronize()?;
        let mut out = vec![0u8; BUDGET * 4];
        ctx.download(&chosen, &mut out)?;
        let blocks = (pos + 1) / RATIO;
        let take = blocks.min(BUDGET);
        let (words, _) = out.as_chunks::<4>();
        let got: Vec<i32> = words[..take]
            .iter()
            .map(|w| i32::from_le_bytes(*w))
            .collect();
        assert_eq!(
            got,
            host_selection(&scores[..blocks], take),
            "position {pos}: the selection differs from the host ranking"
        );
        assert!(
            out[take * 4..].iter().all(|&b| b == SENT),
            "position {pos}: written past the {take} chosen blocks"
        );
    }
    Ok(())
}

/// The selection over the largest cache, 2^24 blocks at the last position. The scores take 2^20
/// values, so about 16 blocks share each and the threshold is tied; the host ranking is a
/// partial sort by (score descending, block ascending).
#[test]
fn qsa_select_ranks_a_cache_of_2_pow_24_blocks() -> Result<(), EngError> {
    if !device_present("qsa_select_ranks_a_cache_of_2_pow_24_blocks") {
        return Ok(());
    }
    const BLOCKS: usize = 1 << 24;
    const BUDGET: usize = 512;
    let indexer = QsaIndexer {
        cap_blocks: BLOCKS,
        ..indexer()
    };
    let mut state = 0x0bad_5eed_u64;
    let scores_host: Vec<f32> = (0..BLOCKS)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            (state >> 44) as f32 / 1024.0
        })
        .collect();
    let bytes: Vec<u8> = scores_host.iter().flat_map(|v| v.to_le_bytes()).collect();
    let mut order: Vec<i32> = (0..i32::try_from(BLOCKS).expect("blocks")).collect();
    let rank = |a: &i32, b: &i32| {
        let score = |i: i32| scores_host[usize::try_from(i).expect("block")];
        score(*b).total_cmp(&score(*a)).then(a.cmp(b))
    };
    order.select_nth_unstable_by(BUDGET - 1, rank);
    let mut expected = order[..BUDGET].to_vec();
    expected.sort_unstable();
    let ctx = CudaContext::new()?;
    let stream = ctx.stream()?;
    let scores = ctx.upload(&bytes)?;
    let chosen = ctx.upload(&vec![SENT; BUDGET * 4])?;
    ctx.qsa_select(
        &stream,
        QsaSelectCall {
            indexer,
            m: 1,
            budget_blocks: BUDGET,
            position: QsaPosition::Host(BLOCKS * RATIO - 1),
            scores: scores.range(0, BLOCKS * 4)?,
            blocks: chosen.range(0, BUDGET * 4)?,
            status: None,
        },
    )?;
    stream.synchronize()?;
    let mut out = vec![0u8; BUDGET * 4];
    ctx.download(&chosen, &mut out)?;
    let (words, _) = out.as_chunks::<4>();
    let got: Vec<i32> = words.iter().map(|w| i32::from_le_bytes(*w)).collect();
    assert_eq!(
        got, expected,
        "the selection over 2^24 blocks differs from the host ranking"
    );
    Ok(())
}

fn whole(buffer: &DeviceBuffer) -> Result<DeviceRange<'_>, EngError> {
    buffer.range(0, buffer.bytes())
}

/// Uniform values in `[-1, 1)` from an LCG, as f32 bytes.
fn noise(state: &mut u64, count: usize) -> Vec<u8> {
    (0..count)
        .flat_map(|_| {
            *state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((*state >> 40) as f32 / (1u64 << 23) as f32 - 1.0).to_le_bytes()
        })
        .collect()
}

/// One score and one selection call of the most tokens (32768, ratio 64, filling a cache of 512
/// blocks, the position at a device word) against one call per token at a host position: every
/// written score and block, and every byte left unwritten, must agree.
#[test]
fn qsa_score_and_select_of_32768_tokens_equal_single_token_calls() -> Result<(), EngError> {
    if !device_present("qsa_score_and_select_of_32768_tokens_equal_single_token_calls") {
        return Ok(());
    }
    const TOKENS: usize = 32_768;
    const BLOCKS: usize = 512;
    const BUDGET: usize = 8;
    const D: usize = 32;
    let indexer = QsaIndexer {
        heads: 1,
        dim: D,
        rotary_dim: D,
        ratio: 64,
        cap_blocks: BLOCKS,
        ..indexer()
    };
    let ctx = CudaContext::new()?;
    let stream = ctx.stream()?;
    let mut state = 0x0b47_c4ed_u64;
    let queries = ctx.upload(&noise(&mut state, TOKENS * D))?;
    let keys = ctx.upload(&noise(&mut state, BLOCKS * D))?;
    let word = ctx.upload(&0i32.to_le_bytes())?;
    let filled = |bytes: usize| ctx.upload(&vec![SENT; bytes]);
    let (scores, singles) = (filled(TOKENS * BLOCKS * 4)?, filled(TOKENS * BLOCKS * 4)?);
    let (chosen, chosen_singles) = (filled(TOKENS * BUDGET * 4)?, filled(TOKENS * BUDGET * 4)?);
    let position = QsaPosition::Device {
        word: word.range(0, 4)?,
        admit: 0,
    };
    ctx.qsa_score(
        &stream,
        QsaScoreCall {
            indexer,
            m: TOKENS,
            position,
            queries: whole(&queries)?,
            keys: whole(&keys)?,
            scores: whole(&scores)?,
            status: None,
        },
    )?;
    ctx.qsa_select(
        &stream,
        QsaSelectCall {
            indexer,
            m: TOKENS,
            budget_blocks: BUDGET,
            position: QsaPosition::Device {
                word: word.range(0, 4)?,
                admit: 0,
            },
            scores: whole(&scores)?,
            blocks: whole(&chosen)?,
            status: None,
        },
    )?;
    for t in 0..TOKENS {
        let score_row = singles.range(t * BLOCKS * 4, BLOCKS * 4)?;
        ctx.qsa_score(
            &stream,
            QsaScoreCall {
                indexer,
                m: 1,
                position: QsaPosition::Host(t),
                queries: queries.range(t * D * 4, D * 4)?,
                keys: whole(&keys)?,
                scores: score_row,
                status: None,
            },
        )?;
        ctx.qsa_select(
            &stream,
            QsaSelectCall {
                indexer,
                m: 1,
                budget_blocks: BUDGET,
                position: QsaPosition::Host(t),
                scores: singles.range(t * BLOCKS * 4, BLOCKS * 4)?,
                blocks: chosen_singles.range(t * BUDGET * 4, BUDGET * 4)?,
                status: None,
            },
        )?;
    }
    stream.synchronize()?;
    let fetch = |b: &DeviceBuffer| -> Result<Vec<u8>, EngError> {
        let mut out = vec![0u8; b.bytes()];
        ctx.download(b, &mut out)?;
        Ok(out)
    };
    let (batch, single) = (fetch(&scores)?, fetch(&singles)?);
    assert!(
        batch == single,
        "the scores of one call differ from single-token calls"
    );
    // The last token has all 512 blocks completed, so every score row is written somewhere.
    assert!(
        batch[(TOKENS - 1) * BLOCKS * 4..]
            .iter()
            .any(|&b| b != SENT),
        "the last token was not scored"
    );
    assert!(
        fetch(&chosen)? == fetch(&chosen_singles)?,
        "the selection of one call differs from single-token calls"
    );
    Ok(())
}
