//! The Flash-Next QSA block indexer on CUDA (`CudaContext::qsa_index`, `qsa_score` and
//! `qsa_select`) against the host indexer of this module (`indexer_queries`, `block_key`,
//! `block_scores`, `select_positions`), which the Mac MLX path is checked against.
//!
//! Each geometry runs a stretch of tokens from an empty ring in several schedules (one chunk,
//! chunks that cross block boundaries, single tokens, single tokens at a device position word)
//! and scores and selects the tokens it names after their chunk. Checked:
//! - the queries of every token and the keys of every block completed inside the stretch
//!   against the host, per element within `2^-18 * max(max|h|, 1)` of the row plus, for a rotated
//!   pair, the RoPE angle class of `attn_prep` (2^-22 per unit of position) for each of the two
//!   implementations;
//! - the ring holds the raw keys of the last positions of the stretch, exactly;
//! - each score against an f64 dot of the device queries and keys, within a first-order bound
//!   of its f32 sums;
//! - the blocks the device selection chooses are the blocks of the host's selection;
//! - all schedules of a geometry give the same bits.
//!
//! The geometries: Flash-Next (4 heads of 128, ratio 4, rope 64, 203 tokens, budget 16 blocks,
//! so the ranking decides from token 67 on); the same with eps 1e-2 on values of scale 0.05, where
//! the eps term and the pooling's division by the ratio dominate the norm (and the host keys at
//! eps 1e-6 must fail the bound, so the test pins eps); the kernel limits: 8 heads of 1024 (8192
//! query values, 1024-thread blocks) with ratio 3 and full rope, a single 32768-token chunk at
//! ratio 64 that fills the cache exactly, and 16 tokens ending at position 2^18 - 1 in a cache of
//! 65536 blocks.

use std::ops::Range;

use super::{QsaDims, block_key, block_scores, indexer_queries, select_positions};
use crate::EngError;
use crate::kernels::cuda::resident::qsa_index::{
    QsaIndexCall, QsaIndexer, QsaPosition, QsaScoreCall, QsaSelectCall,
};
use crate::kernels::cuda::{CudaContext, DeviceBuffer, DeviceStream, device_present};

const COL_OFF: usize = 5;
const U: f64 = 1.0 / 16_777_216.0;
const NORM_CLASS: f64 = 1.0 / 262_144.0;
const ANGLE: f64 = 1.0 / 4_194_304.0;
const THETA: f32 = 1.0e7;

/// One indexer geometry and the stretch of positions a run covers.
#[derive(Clone, Copy, Debug)]
struct Geom {
    heads: usize,
    dim: usize,
    ratio: usize,
    rotary: usize,
    eps: f32,
    cap_blocks: usize,
    budget_blocks: usize,
    /// The position of token 0, a multiple of `ratio`.
    first: usize,
    tokens: usize,
    /// The projection values are uniform in `[-scale, scale]`.
    scale: f64,
    /// Every `score_every`-th token is scored and selected (only with `first == 0`); 0 for none.
    score_every: usize,
}

const FLASH_NEXT: Geom = Geom {
    heads: 4,
    dim: 128,
    ratio: 4,
    rotary: 64,
    eps: 1e-6,
    cap_blocks: 64,
    budget_blocks: 16,
    first: 0,
    tokens: 203,
    scale: 2.0,
    score_every: 1,
};

impl Geom {
    fn ld(self) -> usize {
        COL_OFF + (self.heads + 1) * self.dim + 3
    }

    fn q_width(self) -> usize {
        self.heads * self.dim
    }

    fn host_dims(self) -> QsaDims {
        QsaDims {
            heads_q: 24,
            heads_kv: 2,
            head_dim: 256,
            idx_heads: self.heads as u32,
            idx_dim: self.dim as u32,
            ratio: self.ratio as u32,
            budget: (self.budget_blocks * self.ratio) as u32,
            rotary_dim: self.rotary as u32,
            theta: THETA,
            hidden: 64,
            eps: self.eps,
        }
    }

    fn indexer(self) -> QsaIndexer {
        QsaIndexer {
            heads: self.heads,
            dim: self.dim,
            rotary_dim: self.rotary,
            theta: THETA,
            eps: self.eps,
            ratio: self.ratio,
            cap_blocks: self.cap_blocks,
        }
    }

    /// The tokens (relative to `first`) that are scored and selected.
    fn scored(self) -> Vec<usize> {
        if self.score_every == 0 {
            return Vec::new();
        }
        (0..self.tokens)
            .filter(|t| (t + 1) % self.score_every == 0)
            .collect()
    }

    /// The blocks whose positions all lie in the stretch.
    fn key_blocks(self) -> Range<usize> {
        self.first / self.ratio..(self.first + self.tokens) / self.ratio
    }
}

/// A SplitMix64 stream (the test kit's generator is private to `kernels::cuda`).
pub(super) struct Rng(pub(super) u64);

impl Rng {
    pub(super) fn unit(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        (z >> 40) as f64 / (1u64 << 24) as f64
    }
}

pub(super) fn uniform(rng: &mut Rng, lo: f64, hi: f64, count: usize) -> Vec<f32> {
    (0..count)
        .map(|_| (lo + (hi - lo) * rng.unit()) as f32)
        .collect()
}

pub(super) fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

pub(super) fn download(ctx: &CudaContext, buffer: &DeviceBuffer, floats: usize) -> Vec<f32> {
    let mut bytes = vec![0u8; floats * 4];
    ctx.download(buffer, &mut bytes).expect("download");
    let (words, _) = bytes.as_chunks::<4>();
    words.iter().map(|w| f32::from_le_bytes(*w)).collect()
}

struct Data {
    geom: Geom,
    proj: Vec<f32>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
}

impl Data {
    fn new(geom: Geom, seed: u64) -> Self {
        let mut rng = Rng(seed);
        Self {
            geom,
            proj: uniform(&mut rng, -geom.scale, geom.scale, geom.tokens * geom.ld()),
            q_norm: uniform(&mut rng, -0.5, 0.5, geom.dim),
            k_norm: uniform(&mut rng, -0.5, 0.5, geom.dim),
        }
    }

    fn query_row(&self, t: usize) -> &[f32] {
        let at = t * self.geom.ld() + COL_OFF;
        &self.proj[at..at + self.geom.q_width()]
    }

    fn raw_key(&self, t: usize) -> &[f32] {
        let at = t * self.geom.ld() + COL_OFF + self.geom.q_width();
        &self.proj[at..at + self.geom.dim]
    }
}

struct Host {
    queries: Vec<f32>,
    keys: Vec<f32>,
    /// Per scored token, its block scores.
    scores: Vec<Vec<f64>>,
}

/// The host indexer over `data` with the geometry `geom` (whose eps may differ from the data's).
fn host_run(geom: Geom, data: &Data) -> Result<Host, EngError> {
    let dims = geom.host_dims();
    let dim = geom.dim;
    let ik: Vec<f32> = (0..geom.tokens)
        .flat_map(|t| data.raw_key(t).to_vec())
        .collect();
    let mut queries = Vec::with_capacity(geom.tokens * geom.q_width());
    let mut token_queries = Vec::with_capacity(geom.tokens);
    for t in 0..geom.tokens {
        let pos = (geom.first + t) as u32;
        let q = indexer_queries(dims, data.query_row(t), &data.q_norm, pos)?;
        queries.extend(q.iter().flatten());
        token_queries.push(q);
    }
    let mut keys = Vec::new();
    for b in geom.key_blocks() {
        let at = (b * geom.ratio - geom.first) * dim;
        keys.extend(block_key(
            dims,
            &ik[at..at + geom.ratio * dim],
            b,
            &data.k_norm,
        )?);
    }
    let mut scores = Vec::new();
    for t in geom.scored() {
        let q = &token_queries[t];
        scores.push(block_scores(
            dims,
            q,
            &ik[..(t + 1) * dim],
            &data.k_norm,
            t as u32,
        )?);
    }
    Ok(Host {
        queries,
        keys,
        scores,
    })
}

struct Device {
    geom: Geom,
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
    /// The keys of `geom.key_blocks()`.
    keys: Vec<f32>,
    ring: Vec<f32>,
    /// Per scored token, `cap_blocks` scores.
    scores: Vec<f32>,
    /// Per scored token, `budget_blocks` chosen blocks.
    chosen: Vec<i32>,
}

impl Device {
    fn new(data: &Data) -> Result<Self, EngError> {
        let geom = data.geom;
        let ctx = CudaContext::new()?;
        let stream = ctx.stream()?;
        let positions: Vec<u8> = (0..geom.tokens)
            .flat_map(|t| {
                i32::try_from(geom.first + t)
                    .expect("position")
                    .to_le_bytes()
            })
            .collect();
        let scored = geom.scored().len().max(1);
        let nan = |floats: usize| ctx.upload(&vec![0xFF_u8; floats * 4]);
        Ok(Self {
            proj: ctx.upload(&f32_bytes(&data.proj))?,
            q_norm: ctx.upload(&f32_bytes(&data.q_norm))?,
            k_norm: ctx.upload(&f32_bytes(&data.k_norm))?,
            positions: ctx.upload(&positions)?,
            queries: nan(geom.tokens * geom.q_width())?,
            ring: nan(geom.ratio * geom.dim)?,
            keys: nan(geom.cap_blocks * geom.dim)?,
            scores: nan(scored * geom.cap_blocks)?,
            chosen: nan(scored * geom.budget_blocks)?,
            geom,
            ctx,
            stream,
        })
    }

    /// The position of token `t` (relative to `first`).
    fn position(&self, t: usize, word: bool) -> Result<QsaPosition<'_>, EngError> {
        let at = self.geom.first + t;
        Ok(if word {
            QsaPosition::Device {
                word: self.positions.range(t * 4, 4)?,
                admit: at,
            }
        } else {
            QsaPosition::Host(at)
        })
    }

    /// Indexes tokens `row0 .. row0 + m`, then scores and selects the scored ones among them.
    fn chunk(&self, row0: usize, m: usize, word: bool) -> Result<(), EngError> {
        let g = self.geom;
        let (ld, qw, dim) = (g.ld(), g.q_width(), g.dim);
        self.ctx.qsa_index(
            &self.stream,
            QsaIndexCall {
                indexer: g.indexer(),
                m,
                ld,
                col_off: COL_OFF,
                position: self.position(row0, word)?,
                proj: self.proj.range(row0 * ld * 4, m * ld * 4)?,
                q_norm: self.q_norm.range(0, dim * 4)?,
                k_norm: self.k_norm.range(0, dim * 4)?,
                queries: self.queries.range(row0 * qw * 4, m * qw * 4)?,
                ring: self.ring.range(0, g.ratio * dim * 4)?,
                keys: self.keys.range(0, g.cap_blocks * dim * 4)?,
                status: None,
            },
        )?;
        for (row, t) in g.scored().into_iter().enumerate() {
            if (row0..row0 + m).contains(&t) {
                self.score(row, t, word)?;
            }
        }
        Ok(())
    }

    /// Scores token `t` into scores row `row` and selects its blocks.
    fn score(&self, row: usize, t: usize, word: bool) -> Result<(), EngError> {
        let g = self.geom;
        let (cap, budget) = (g.cap_blocks, g.budget_blocks);
        self.ctx.qsa_score(
            &self.stream,
            QsaScoreCall {
                indexer: g.indexer(),
                position: self.position(t, word)?,
                queries: self.queries.range(t * g.q_width() * 4, g.q_width() * 4)?,
                keys: self.keys.range(0, cap * g.dim * 4)?,
                scores: self.scores.range(row * cap * 4, cap * 4)?,
                status: None,
            },
        )?;
        self.ctx.qsa_select(
            &self.stream,
            QsaSelectCall {
                indexer: g.indexer(),
                budget_blocks: budget,
                position: self.position(t, word)?,
                scores: self.scores.range(row * cap * 4, cap * 4)?,
                blocks: self.chosen.range(row * budget * 4, budget * 4)?,
                status: None,
            },
        )
    }

    fn finish(self) -> Result<Run, EngError> {
        self.stream.synchronize()?;
        let (g, ctx) = (self.geom, &self.ctx);
        let scored = g.scored().len();
        let keys = download(ctx, &self.keys, g.cap_blocks * g.dim);
        let blocks = g.key_blocks();
        let (scores, chosen) = if scored == 0 {
            (Vec::new(), Vec::new())
        } else {
            (
                download(ctx, &self.scores, scored * g.cap_blocks),
                download(ctx, &self.chosen, scored * g.budget_blocks)
                    .into_iter()
                    .map(|v| v.to_bits() as i32)
                    .collect(),
            )
        };
        Ok(Run {
            queries: download(ctx, &self.queries, g.tokens * g.q_width()),
            keys: keys[blocks.start * g.dim..blocks.end * g.dim].to_vec(),
            ring: download(ctx, &self.ring, g.ratio * g.dim),
            scores,
            chosen,
        })
    }
}

fn device_run(data: &Data, sizes: &[usize], word: bool) -> Result<Run, EngError> {
    assert_eq!(sizes.iter().sum::<usize>(), data.geom.tokens);
    let device = Device::new(data)?;
    let mut row0 = 0;
    for &m in sizes {
        device.chunk(row0, m, word)?;
        row0 += m;
    }
    device.finish()
}

/// The worst `|d - h| / bound` over rows of `dim` values, the row of index `r` rotated at
/// position `pos_of(r)`; NaN counts as infinite.
fn worst_rotated(
    geom: Geom,
    (dev, host): (&[f32], &[f32]),
    pos_of: impl Fn(usize) -> usize,
) -> f64 {
    assert_eq!(dev.len(), host.len());
    let half = geom.rotary / 2;
    let mut worst = 0.0f64;
    for (r, (d, h)) in dev.chunks(geom.dim).zip(host.chunks(geom.dim)).enumerate() {
        let scale = h.iter().fold(1.0f64, |m, v| m.max(f64::from(*v).abs()));
        let turn = 2.0 * ANGLE * (pos_of(r) as f64 + 1.0);
        for i in 0..geom.dim {
            let mut bound = NORM_CLASS * scale;
            if i < geom.rotary {
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

/// The worst `|d - ref| / bound` of the scores in row `row` (token `t`), the reference an f64
/// sum of the device queries and keys, the bound twice the first-order error of the f32 dots
/// and head sum.
fn worst_score(geom: Geom, run: &Run, (row, t): (usize, usize)) -> f64 {
    let (dim, heads) = (geom.dim, geom.heads);
    let mut worst = 0.0f64;
    for b in 0..(t + 1) / geom.ratio {
        let key = &run.keys[b * dim..(b + 1) * dim];
        let (mut reference, mut magnitude) = (0.0f64, 0.0f64);
        for h in 0..heads {
            let q = &run.queries[(t * heads + h) * dim..(t * heads + h + 1) * dim];
            let terms = q
                .iter()
                .zip(key)
                .map(|(a, b)| f64::from(*a) * f64::from(*b));
            reference += terms.clone().sum::<f64>().max(0.0);
            magnitude += terms.map(f64::abs).sum::<f64>();
        }
        let bound = 2.0 * (dim + heads) as f64 * U * magnitude + f64::from(f32::MIN_POSITIVE);
        let err = (f64::from(run.scores[row * geom.cap_blocks + b]) - reference).abs();
        worst = worst.max(if err.is_nan() {
            f64::INFINITY
        } else {
            err / bound
        });
    }
    worst
}

/// The scored tokens whose chosen blocks on the device differ from the blocks of the host's
/// selection (`select_positions` on the host scores, its blocks taken in ascending order).
fn selection_misses(geom: Geom, run: &Run, host: &Host) -> Result<Vec<usize>, EngError> {
    let dims = geom.host_dims();
    let (ratio, budget) = (geom.ratio, geom.budget_blocks);
    let mut misses = Vec::new();
    for (row, t) in geom.scored().into_iter().enumerate() {
        let take = ((t + 1) / ratio).min(budget);
        let positions = select_positions(dims, &host.scores[row], t as u32)?;
        let mut expected: Vec<i32> = positions[..take * ratio]
            .iter()
            .step_by(ratio)
            .map(|&p| (p / ratio) as i32)
            .collect();
        expected.sort_unstable();
        if run.chosen[row * budget..row * budget + take] != expected[..] {
            misses.push(t);
        }
    }
    Ok(misses)
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

/// The queries and keys of `run` against `host`, as fractions of their bounds.
fn rotated_errors(geom: Geom, run: &Run, host: &Host) -> (f64, f64) {
    let first_block = geom.key_blocks().start;
    let q_err = worst_rotated(geom, (&run.queries, &host.queries), |r| {
        geom.first + r / geom.heads
    });
    let k_err = worst_rotated(geom, (&run.keys, &host.keys), |b| {
        (first_block + b) * geom.ratio
    });
    (q_err, k_err)
}

fn check_run(name: &str, run: &Run, host: &Host, data: &Data) -> Result<(), EngError> {
    let geom = data.geom;
    let (q_err, k_err) = rotated_errors(geom, run, host);
    let s_err = geom
        .scored()
        .into_iter()
        .enumerate()
        .map(|scored| worst_score(geom, run, scored))
        .fold(0.0, f64::max);
    let misses = selection_misses(geom, run, host)?;
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
    let (ratio, dim) = (geom.ratio, geom.dim);
    for t in geom.tokens.saturating_sub(ratio)..geom.tokens {
        let slot = (geom.first + t) % ratio;
        assert_eq!(
            bits(&run.ring[slot * dim..(slot + 1) * dim]),
            bits(data.raw_key(t)),
            "{name}: the ring slot of position {}",
            geom.first + t
        );
    }
    Ok(())
}

/// Runs every schedule of `geom` and checks each against the host and against the first.
fn check_schedules(
    name: &str,
    data: &Data,
    schedules: &[(&str, Vec<usize>, bool)],
) -> Result<(), EngError> {
    let host = host_run(data.geom, data)?;
    let mut first: Option<Run> = None;
    for (schedule, sizes, word) in schedules {
        let label = format!("{name}, {schedule}");
        let run = device_run(data, sizes, *word)?;
        check_run(&label, &run, &host, data)?;
        let Some(first) = &first else {
            first = Some(run);
            continue;
        };
        assert_eq!(bits(&run.queries), bits(&first.queries), "{label}: queries");
        assert_eq!(bits(&run.keys), bits(&first.keys), "{label}: keys");
        assert_eq!(bits(&run.ring), bits(&first.ring), "{label}: ring");
        assert_eq!(bits(&run.scores), bits(&first.scores), "{label}: scores");
        assert_eq!(run.chosen, first.chosen, "{label}: chosen blocks");
    }
    Ok(())
}

#[test]
fn cuda_qsa_indexer_matches_the_host_in_every_schedule() -> Result<(), EngError> {
    if !device_present("cuda_qsa_indexer_matches_the_host_in_every_schedule") {
        return Ok(());
    }
    let tokens = FLASH_NEXT.tokens;
    let schedules = [
        ("one chunk", vec![tokens], false),
        ("17 + 186", vec![17, 186], false),
        ("3 + 1 + 6 + 193", vec![3, 1, 6, 193], false),
        ("tokens", vec![1; tokens], false),
        ("tokens at a device word", vec![1; tokens], true),
    ];
    check_schedules("flash-next", &Data::new(FLASH_NEXT, 0x95a_1de7), &schedules)
}

#[test]
fn cuda_qsa_indexer_honours_eps_and_the_pooling_ratio() -> Result<(), EngError> {
    if !device_present("cuda_qsa_indexer_honours_eps_and_the_pooling_ratio") {
        return Ok(());
    }
    let geom = Geom {
        eps: 1e-2,
        scale: 0.05,
        ..FLASH_NEXT
    };
    let data = Data::new(geom, 0x0e95_0001);
    let schedules = [
        ("one chunk", vec![geom.tokens], false),
        ("3 + 1 + 6 + 193", vec![3, 1, 6, 193], false),
    ];
    check_schedules("eps 1e-2", &data, &schedules)?;
    // The host at eps 1e-6 must miss the device keys by more than the bound, so a kernel that
    // dropped eps (or the division by the ratio, which only eps makes visible) would fail.
    let run = device_run(&data, &[geom.tokens], false)?;
    let wrong = host_run(Geom { eps: 1e-6, ..geom }, &data)?;
    let (q_err, k_err) = rotated_errors(geom, &run, &wrong);
    println!(
        "REPORTED: flash-next qsa indexer eps 1e-2 against a host at eps 1e-6: queries \
         {q_err:.1}, keys {k_err:.1} of their bounds"
    );
    assert!(
        q_err > 10.0 && k_err > 10.0,
        "eps does not move the result far enough to be pinned"
    );
    Ok(())
}

#[test]
fn cuda_qsa_indexer_matches_the_host_at_the_kernel_limits() -> Result<(), EngError> {
    if !device_present("cuda_qsa_indexer_matches_the_host_at_the_kernel_limits") {
        return Ok(());
    }
    // 8 heads of 1024 (8192 query values, 1024-thread blocks), full rope, ratio 3.
    let wide = Geom {
        heads: 8,
        dim: 1024,
        ratio: 3,
        rotary: 1024,
        cap_blocks: 32,
        budget_blocks: 4,
        tokens: 64,
        ..FLASH_NEXT
    };
    let schedules = [
        ("one chunk", vec![64], false),
        ("33 + 1 + 30", vec![33, 1, 30], false),
        ("tokens at a device word", vec![1; 64], true),
    ];
    check_schedules(
        "8 x 1024, ratio 3",
        &Data::new(wide, 0x0a1d_e001),
        &schedules,
    )?;
    // One chunk of the most tokens, ratio 64, filling the cache of 512 blocks exactly.
    let long = Geom {
        heads: 1,
        dim: 32,
        ratio: 64,
        rotary: 32,
        cap_blocks: 512,
        budget_blocks: 8,
        tokens: 32_768,
        score_every: 4096,
        ..FLASH_NEXT
    };
    let schedules = [
        ("one chunk", vec![32_768], false),
        ("32767 + 1", vec![32_767, 1], false),
    ];
    check_schedules(
        "m 32768, ratio 64",
        &Data::new(long, 0x0a1d_e002),
        &schedules,
    )?;
    // 16 tokens ending at position 2^18 - 1, the last of a cache of 65536 blocks.
    let far = Geom {
        cap_blocks: 65_536,
        first: 262_128,
        tokens: 16,
        score_every: 0,
        ..FLASH_NEXT
    };
    let schedules = [
        ("one chunk", vec![16], false),
        ("5 + 11", vec![5, 11], false),
        ("tokens at a device word", vec![1; 16], true),
    ];
    check_schedules(
        "positions to 2^18",
        &Data::new(far, 0x0a1d_e003),
        &schedules,
    )
}
