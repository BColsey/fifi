//! The Flash-Next QSA block indexer on CUDA (`CudaContext::qsa_index` and `qsa_score`) against
//! the host indexer of this module (`indexer_queries`, `block_key`, `block_scores`,
//! `select_positions`), which the Mac MLX path is checked against, at the Flash-Next indexer
//! geometry: 4 heads of 128, blocks of 4 positions, rope over the first 64 values with base
//! 1e7, eps 1e-6.
//!
//! 203 tokens (50 completed blocks and an open tail of 3) run from an empty cache in five
//! schedules: one chunk (the prefill), 17 + 186, chunks of 3, 1, 6 and 193 (a ring that carries
//! across calls, and a chunk shorter than a block), 203 one-token calls (the decode path), and
//! the one-token calls again with the position read from a device word (a replayed step). Every
//! token is scored after its chunk. Checked:
//! - the queries of every token and the keys of every block against the host, per element within
//!   `2^-18 * max(max|h|, 1)` of the row plus, for a rotated pair, the RoPE angle class of
//!   `attn_prep` (2^-22 per unit of position) for each of the two implementations;
//! - the ring holds the raw keys of the last four positions, exactly;
//! - each score against an f64 dot of the device queries and keys, within a first-order bound
//!   of its f32 sums;
//! - the blocks the device selection chooses (`qsa_select`) are the blocks of the host's
//!   selection for every token, with a budget of 16 blocks so the ranking decides from token 67
//!   on;
//! - all five schedules give the same bits.

use super::{QsaDims, block_key, block_scores, indexer_queries, select_positions};
use crate::EngError;
use crate::kernels::cuda::resident::qsa_index::{
    QsaIndexCall, QsaIndexer, QsaPosition, QsaScoreCall, QsaSelectCall,
};
use crate::kernels::cuda::{CudaContext, DeviceBuffer, DeviceStream, device_present};

const HEADS: usize = 4;
const DIM: usize = 128;
const RATIO: usize = 4;
const ROTARY: usize = 64;
const THETA: f32 = 1.0e7;
const EPS: f32 = 1e-6;
/// 16 blocks of 4.
const BUDGET: usize = 64;
const BUDGET_BLOCKS: usize = BUDGET / RATIO;
const TOKENS: usize = 203;
const BLOCKS: usize = TOKENS / RATIO;
const CAP_BLOCKS: usize = 64;
const COL_OFF: usize = 5;
const Q_WIDTH: usize = HEADS * DIM;
const LD: usize = COL_OFF + Q_WIDTH + DIM + 3;
const U: f64 = 1.0 / 16_777_216.0;
const NORM_CLASS: f64 = 1.0 / 262_144.0;
const ANGLE: f64 = 1.0 / 4_194_304.0;

fn host_dims() -> QsaDims {
    QsaDims {
        heads_q: 24,
        heads_kv: 2,
        head_dim: 256,
        idx_heads: HEADS as u32,
        idx_dim: DIM as u32,
        ratio: RATIO as u32,
        budget: BUDGET as u32,
        rotary_dim: ROTARY as u32,
        theta: THETA,
        hidden: 64,
        eps: EPS,
    }
}

fn indexer() -> QsaIndexer {
    QsaIndexer {
        heads: HEADS,
        dim: DIM,
        rotary_dim: ROTARY,
        theta: THETA,
        eps: EPS,
        ratio: RATIO,
        cap_blocks: CAP_BLOCKS,
    }
}

/// A SplitMix64 stream (the test kit's generator is private to `kernels::cuda`).
struct Rng(u64);

impl Rng {
    fn unit(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        (z >> 40) as f64 / (1u64 << 24) as f64
    }
}

fn uniform(rng: &mut Rng, lo: f64, hi: f64, count: usize) -> Vec<f32> {
    (0..count)
        .map(|_| (lo + (hi - lo) * rng.unit()) as f32)
        .collect()
}

struct Data {
    proj: Vec<f32>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
}

impl Data {
    fn new() -> Self {
        let mut rng = Rng(0x95a_1de7);
        Self {
            proj: uniform(&mut rng, -2.0, 2.0, TOKENS * LD),
            q_norm: uniform(&mut rng, -0.5, 0.5, DIM),
            k_norm: uniform(&mut rng, -0.5, 0.5, DIM),
        }
    }

    fn query_row(&self, t: usize) -> &[f32] {
        &self.proj[t * LD + COL_OFF..t * LD + COL_OFF + Q_WIDTH]
    }

    fn raw_key(&self, t: usize) -> &[f32] {
        let at = t * LD + COL_OFF + Q_WIDTH;
        &self.proj[at..at + DIM]
    }
}

struct Host {
    queries: Vec<f32>,
    keys: Vec<f32>,
    scores: Vec<Vec<f64>>,
}

fn host_run(data: &Data) -> Result<Host, EngError> {
    let dims = host_dims();
    let ik: Vec<f32> = (0..TOKENS).flat_map(|t| data.raw_key(t).to_vec()).collect();
    let mut queries = Vec::with_capacity(TOKENS * Q_WIDTH);
    let mut scores = Vec::with_capacity(TOKENS);
    for t in 0..TOKENS {
        let pos = t as u32;
        let q = indexer_queries(dims, data.query_row(t), &data.q_norm, pos)?;
        scores.push(block_scores(
            dims,
            &q,
            &ik[..(t + 1) * DIM],
            &data.k_norm,
            pos,
        )?);
        queries.extend(q.into_iter().flatten());
    }
    let mut keys = Vec::with_capacity(BLOCKS * DIM);
    for b in 0..BLOCKS {
        keys.extend(block_key(dims, &ik, b, &data.k_norm)?);
    }
    Ok(Host {
        queries,
        keys,
        scores,
    })
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn download(ctx: &CudaContext, buffer: &DeviceBuffer, floats: usize) -> Vec<f32> {
    let mut bytes = vec![0u8; floats * 4];
    ctx.download(buffer, &mut bytes).expect("download");
    let (words, _) = bytes.as_chunks::<4>();
    words.iter().map(|w| f32::from_le_bytes(*w)).collect()
}

struct Device {
    ctx: CudaContext,
    stream: DeviceStream,
    proj: DeviceBuffer,
    q_norm: DeviceBuffer,
    k_norm: DeviceBuffer,
    positions: DeviceBuffer,
    queries: DeviceBuffer,
    ring: DeviceBuffer,
    keys: DeviceBuffer,
    scores: DeviceBuffer,
    chosen: DeviceBuffer,
}

/// What one schedule leaves on the device.
struct Run {
    queries: Vec<f32>,
    keys: Vec<f32>,
    ring: Vec<f32>,
    scores: Vec<f32>,
    chosen: Vec<i32>,
}

impl Device {
    fn new(data: &Data) -> Result<Self, EngError> {
        let ctx = CudaContext::new()?;
        let stream = ctx.stream()?;
        let positions: Vec<u8> = (0..TOKENS as i32).flat_map(i32::to_le_bytes).collect();
        let nan = |floats: usize| ctx.upload(&vec![0xFF_u8; floats * 4]);
        Ok(Self {
            proj: ctx.upload(&f32_bytes(&data.proj))?,
            q_norm: ctx.upload(&f32_bytes(&data.q_norm))?,
            k_norm: ctx.upload(&f32_bytes(&data.k_norm))?,
            positions: ctx.upload(&positions)?,
            queries: nan(TOKENS * Q_WIDTH)?,
            ring: nan(RATIO * DIM)?,
            keys: nan(CAP_BLOCKS * DIM)?,
            scores: nan(TOKENS * CAP_BLOCKS)?,
            chosen: nan(TOKENS * BUDGET_BLOCKS)?,
            ctx,
            stream,
        })
    }

    fn position(&self, t: usize, word: bool) -> Result<QsaPosition<'_>, EngError> {
        Ok(if word {
            QsaPosition::Device {
                word: self.positions.range(t * 4, 4)?,
                admit: t,
            }
        } else {
            QsaPosition::Host(t)
        })
    }

    /// Indexes tokens `row0 .. row0 + m`, then scores each of them.
    fn chunk(&self, row0: usize, m: usize, word: bool) -> Result<(), EngError> {
        self.ctx.qsa_index(
            &self.stream,
            QsaIndexCall {
                indexer: indexer(),
                m,
                ld: LD,
                col_off: COL_OFF,
                position: self.position(row0, word)?,
                proj: self.proj.range(row0 * LD * 4, m * LD * 4)?,
                q_norm: self.q_norm.range(0, DIM * 4)?,
                k_norm: self.k_norm.range(0, DIM * 4)?,
                queries: self.queries.range(row0 * Q_WIDTH * 4, m * Q_WIDTH * 4)?,
                ring: self.ring.range(0, RATIO * DIM * 4)?,
                keys: self.keys.range(0, CAP_BLOCKS * DIM * 4)?,
                status: None,
            },
        )?;
        for t in row0..row0 + m {
            self.ctx.qsa_score(
                &self.stream,
                QsaScoreCall {
                    indexer: indexer(),
                    position: self.position(t, word)?,
                    queries: self.queries.range(t * Q_WIDTH * 4, Q_WIDTH * 4)?,
                    keys: self.keys.range(0, CAP_BLOCKS * DIM * 4)?,
                    scores: self.scores.range(t * CAP_BLOCKS * 4, CAP_BLOCKS * 4)?,
                    status: None,
                },
            )?;
            self.ctx.qsa_select(
                &self.stream,
                QsaSelectCall {
                    indexer: indexer(),
                    budget_blocks: BUDGET_BLOCKS,
                    position: self.position(t, word)?,
                    scores: self.scores.range(t * CAP_BLOCKS * 4, CAP_BLOCKS * 4)?,
                    blocks: self
                        .chosen
                        .range(t * BUDGET_BLOCKS * 4, BUDGET_BLOCKS * 4)?,
                    status: None,
                },
            )?;
        }
        Ok(())
    }

    fn finish(self) -> Result<Run, EngError> {
        self.stream.synchronize()?;
        let ctx = &self.ctx;
        Ok(Run {
            queries: download(ctx, &self.queries, TOKENS * Q_WIDTH),
            keys: download(ctx, &self.keys, BLOCKS * DIM),
            ring: download(ctx, &self.ring, RATIO * DIM),
            scores: download(ctx, &self.scores, TOKENS * CAP_BLOCKS),
            chosen: download(ctx, &self.chosen, TOKENS * BUDGET_BLOCKS)
                .into_iter()
                .map(|v| v.to_bits() as i32)
                .collect(),
        })
    }
}

fn device_run(data: &Data, sizes: &[usize], word: bool) -> Result<Run, EngError> {
    assert_eq!(sizes.iter().sum::<usize>(), TOKENS);
    let device = Device::new(data)?;
    let mut row0 = 0;
    for &m in sizes {
        device.chunk(row0, m, word)?;
        row0 += m;
    }
    device.finish()
}

/// The worst `|d - h| / bound` over rows of `DIM` values, the row of index `r` rotated at
/// position `pos_of(r)`; NaN counts as infinite.
fn worst_rotated(dev: &[f32], host: &[f32], pos_of: impl Fn(usize) -> usize) -> f64 {
    assert_eq!(dev.len(), host.len());
    let half = ROTARY / 2;
    let mut worst = 0.0f64;
    for (r, (d, h)) in dev.chunks(DIM).zip(host.chunks(DIM)).enumerate() {
        let scale = h.iter().fold(1.0f64, |m, v| m.max(f64::from(*v).abs()));
        let turn = 2.0 * ANGLE * (pos_of(r) as f64 + 1.0);
        for i in 0..DIM {
            let mut bound = NORM_CLASS * scale;
            if i < ROTARY {
                let partner = if i < half { i + half } else { i - half };
                bound += turn * f64::from(h[i]).hypot(f64::from(h[partner]));
            }
            let err = (f64::from(d[i]) - f64::from(h[i])).abs();
            worst = worst.max(if err.is_nan() {
                f64::INFINITY
            } else {
                err / bound
            });
        }
    }
    worst
}

/// The worst `|d - ref| / bound` of the scores of token `t`, the reference an f64 sum of the
/// device queries and keys, the bound twice the first-order error of the f32 dots and head sum.
fn worst_score(run: &Run, t: usize) -> f64 {
    let blocks = (t + 1) / RATIO;
    let mut worst = 0.0f64;
    for b in 0..blocks {
        let key = &run.keys[b * DIM..(b + 1) * DIM];
        let (mut reference, mut magnitude) = (0.0f64, 0.0f64);
        for h in 0..HEADS {
            let q = &run.queries[(t * HEADS + h) * DIM..(t * HEADS + h + 1) * DIM];
            let terms = q
                .iter()
                .zip(key)
                .map(|(a, b)| f64::from(*a) * f64::from(*b));
            reference += terms.clone().sum::<f64>().max(0.0);
            magnitude += terms.map(f64::abs).sum::<f64>();
        }
        let bound = 2.0 * (DIM + HEADS) as f64 * U * magnitude + f64::from(f32::MIN_POSITIVE);
        let err = (f64::from(run.scores[t * CAP_BLOCKS + b]) - reference).abs();
        worst = worst.max(if err.is_nan() {
            f64::INFINITY
        } else {
            err / bound
        });
    }
    worst
}

/// The tokens whose chosen blocks on the device differ from the blocks of the host's selection
/// (`select_positions` on the host scores, its blocks taken in ascending order).
fn selection_misses(run: &Run, host: &Host) -> Result<Vec<usize>, EngError> {
    let dims = host_dims();
    let mut misses = Vec::new();
    for t in 0..TOKENS {
        let take = ((t + 1) / RATIO).min(BUDGET_BLOCKS);
        let positions = select_positions(dims, &host.scores[t], t as u32)?;
        let mut expected: Vec<i32> = positions[..take * RATIO]
            .iter()
            .step_by(RATIO)
            .map(|&p| (p / RATIO) as i32)
            .collect();
        expected.sort_unstable();
        let got = &run.chosen[t * BUDGET_BLOCKS..t * BUDGET_BLOCKS + take];
        if got != expected.as_slice() {
            misses.push(t);
        }
    }
    Ok(misses)
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

fn check_run(name: &str, run: &Run, host: &Host, data: &Data) -> Result<(), EngError> {
    let q_err = worst_rotated(&run.queries, &host.queries, |r| r / HEADS);
    let k_err = worst_rotated(&run.keys, &host.keys, |b| b * RATIO);
    let s_err = (0..TOKENS).map(|t| worst_score(run, t)).fold(0.0, f64::max);
    let misses = selection_misses(run, host)?;
    println!(
        "REPORTED: flash-next qsa indexer {name}: queries {q_err:.3}, keys {k_err:.3}, scores \
         {s_err:.3} of their bounds; selection misses {misses:?}"
    );
    assert!(q_err <= 1.0, "{name}: queries {q_err:.3} of the bound");
    assert!(k_err <= 1.0, "{name}: keys {k_err:.3} of the bound");
    assert!(s_err <= 1.0, "{name}: scores {s_err:.3} of the bound");
    assert!(
        misses.is_empty(),
        "{name}: the selection differs at {misses:?}"
    );
    for p in TOKENS - RATIO..TOKENS {
        assert_eq!(
            bits(&run.ring[(p % RATIO) * DIM..(p % RATIO + 1) * DIM]),
            bits(data.raw_key(p)),
            "{name}: the ring slot of position {p}"
        );
    }
    Ok(())
}

#[test]
fn cuda_qsa_indexer_matches_the_host_in_every_schedule() -> Result<(), EngError> {
    let data = Data::new();
    let host = host_run(&data)?;
    if !device_present("cuda_qsa_indexer_matches_the_host_in_every_schedule") {
        return Ok(());
    }
    let schedules: [(&str, Vec<usize>, bool); 5] = [
        ("one chunk", vec![TOKENS], false),
        ("17 + 186", vec![17, 186], false),
        ("3 + 1 + 6 + 193", vec![3, 1, 6, 193], false),
        ("tokens", vec![1; TOKENS], false),
        ("tokens at a device word", vec![1; TOKENS], true),
    ];
    let mut first: Option<Run> = None;
    for (name, sizes, word) in &schedules {
        let run = device_run(&data, sizes, *word)?;
        check_run(name, &run, &host, &data)?;
        if let Some(first) = &first {
            assert_eq!(bits(&run.queries), bits(&first.queries), "{name}: queries");
            assert_eq!(bits(&run.keys), bits(&first.keys), "{name}: keys");
            assert_eq!(bits(&run.ring), bits(&first.ring), "{name}: ring");
            assert_eq!(bits(&run.scores), bits(&first.scores), "{name}: scores");
            assert_eq!(run.chosen, first.chosen, "{name}: chosen blocks");
        } else {
            first = Some(run);
        }
    }
    Ok(())
}
