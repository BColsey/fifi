//! The CPU reference of query-aware sparse KV attention for the full-attention layers (sparse
//! lane, phase 3): low-rank key sketches, softmax-mass relevance scoring, deterministic top-K
//! selection with sink/recent retention, and exact attention over the selected subset.
//!
//! This module is the numerical backbone the device path must match; it is NOT wired into any
//! serving path. It mirrors the phase-2 audit of sparsekv-hyperqwen (Apache-2.0) — the algorithm,
//! not its code: the score of a position is its approximate attention-probability mass (softmax
//! over the candidate range per query lane, summed over the lanes of one KV head — per-KV-head
//! selection), candidates exclude the sink prefix and the recent window, and the top-K tie rule
//! is "the values above the k-th value in ascending index order, then the lowest-index ties"
//! (the radix-select rule). The union is `[top_k ascending | sinks | window]`, disjoint by
//! construction because candidates live in `[sink, positions - recent)`.
//!
//! Numerics: sketches are stored F16 (the initial configuration) and read back as F32; scores
//! and attention accumulate in F64, the convention of `host.rs`. The reference separates (a)
//! omission error — exact attention over an exact-logit ("oracle") selection versus dense
//! attention — from (b) sketch error — the same over a rank-r approximate selection. Nothing
//! here changes weights, rope, scaling or causal semantics; a selection never names a position
//! at or beyond `positions`.
//!
//! The basis here is a deterministic orthonormal control (`Basis::random_orthonormal`). A
//! fitted basis (offline Gram/eigh utility, hash-bound artifact) is a later, measured decision;
//! the reference treats the basis as data so a fitter can supply rows without touching this file.

use crate::EngError;
use half::f16;

/// The geometry of one full-attention layer's core, the slice of `MixerGeometry` the sparse
/// reference needs. Values come from the model's `AttnDims`; the defaults match
/// `QWEN3_6_35B_A3B` (16 query heads over 2 KV heads of 256, scale 1/16).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct SparseGeometry {
    pub(crate) heads_q: usize,
    pub(crate) heads_kv: usize,
    pub(crate) head_dim: usize,
    pub(crate) scale: f64,
}

/// The knobs of the experimental configuration: experiment parameters of the sparse lane, not
/// production settings. Defaults follow the task brief (rank 64, top-K 2048, sink 64, recent
/// 1024).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct SparseConfig {
    pub(crate) rank: usize,
    pub(crate) top_k: usize,
    pub(crate) sink: usize,
    pub(crate) recent: usize,
}

impl SparseConfig {
    /// Sparse selection pays only above the full selected count; a short context takes the
    /// dense fallback.
    pub(crate) fn sparse(&self, positions: usize) -> bool {
        positions > self.top_k + self.sink + self.recent
    }

    pub(crate) fn check(&self) -> Result<(), EngError> {
        let bad = |what: &str| Err(EngError::BadRequest(format!("sparse config: {what}")));
        if self.rank == 0 || !self.rank.is_multiple_of(2) {
            return bad("rank is not a positive even number");
        }
        if self.top_k == 0 || self.sink == 0 || self.recent == 0 {
            return bad("top_k, sink and recent must be positive");
        }
        Ok(())
    }
}

/// One selection of one KV head: the union `[top_k (ascending) | sinks | window]` when sparse,
/// else `0..positions` (the dense fallback of a short context). `count` is the number of valid
/// entries (`indices.len()`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SparseSelection {
    pub(crate) indices: Vec<u32>,
    pub(crate) sparse: bool,
}

/// One projection basis per KV head: `rank` orthonormal rows of `head_dim` values,
/// `values[(head * rank + row) * head_dim + d]`.
pub(crate) struct Basis {
    pub(crate) rank: usize,
    pub(crate) head_dim: usize,
    values: Vec<f32>,
}

impl Basis {
    /// A deterministic orthonormal control basis: Gram-Schmidt over `fixtures::val` hash values,
    /// so every build fits the same rows. The seed names the basis; a fitted production basis
    /// loads its own values and never uses this constructor.
    pub(crate) fn random_orthonormal(
        heads: usize,
        head_dim: usize,
        rank: usize,
        seed: u64,
    ) -> Self {
        let mut values = vec![0.0f32; heads * rank * head_dim];
        for head in 0..heads {
            let head_base = head * rank * head_dim;
            for row in 0..rank {
                let tag = seed
                    .wrapping_mul(0x9E37_79B9)
                    .wrapping_add((head * rank + row) as u64);
                let mut v: Vec<f64> = (0..head_dim as u64)
                    .map(|i| crate::family::qwen36::fixtures::val(tag, i) as f64)
                    .collect();
                for earlier in 0..row {
                    let base = head_base + earlier * head_dim;
                    let dot: f64 = v
                        .iter()
                        .enumerate()
                        .map(|(i, x)| x * f64::from(values[base + i]))
                        .sum();
                    for (i, x) in v.iter_mut().enumerate() {
                        *x -= dot * f64::from(values[base + i]);
                    }
                }
                let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
                for (i, x) in v.iter().enumerate() {
                    values[head_base + row * head_dim + i] = (*x / norm) as f32;
                }
            }
        }
        Self {
            rank,
            head_dim,
            values,
        }
    }

    /// A basis from fitted rows: `rows[(head * rank + r) * head_dim + d]`. The fitter owns
    /// orthonormality; this constructor only records the shape.
    pub(crate) fn from_rows(
        heads: usize,
        head_dim: usize,
        rank: usize,
        rows: Vec<f32>,
    ) -> Result<Self, EngError> {
        if rows.len() != heads * rank * head_dim {
            return Err(EngError::BadRequest(format!(
                "sparse basis: {} rows values do not fill {} heads x {rank} x {head_dim}",
                rows.len(),
                heads
            )));
        }
        Ok(Self {
            rank,
            head_dim,
            values: rows,
        })
    }

    fn row(&self, head: usize, row: usize) -> &[f32] {
        &self.values[(head * self.rank + row) * self.head_dim..][..self.head_dim]
    }

    /// The projection of one vector onto the basis of one head: `rank` values, f64 accumulation.
    fn project(&self, head: usize, x: &[f32]) -> Vec<f64> {
        (0..self.rank)
            .map(|row| {
                self.row(head, row)
                    .iter()
                    .zip(x)
                    .map(|(b, v)| f64::from(*b) * f64::from(*v))
                    .sum()
            })
            .collect()
    }
}

/// The sketch plane of one full-attention layer: one F16 sketch row of `rank` values per KV head
/// per position, `values[(head * positions + t) * rank + r]`. Built once when keys become
/// available; every stored value passes through F16, the configured precision.
pub(crate) struct SketchPlane {
    positions: usize,
    heads: usize,
    rank: usize,
    values: Vec<f16>,
}

impl SketchPlane {
    /// Builds the plane of `keys` in cache order `[positions][heads_kv][head_dim]` — post-norm,
    /// post-rope keys, exactly what the KV planes hold.
    pub(crate) fn build(basis: &Basis, keys: &[f32], positions: usize, heads_kv: usize) -> Self {
        let head_dim = basis.head_dim;
        let mut values = vec![f16::ZERO; positions * heads_kv * basis.rank];
        for t in 0..positions {
            for head in 0..heads_kv {
                let key = &keys[(t * heads_kv + head) * head_dim..][..head_dim];
                let z = basis.project(head, key);
                for (r, value) in z.iter().enumerate() {
                    values[(head * positions + t) * basis.rank + r] = f16::from_f32(*value as f32);
                }
            }
        }
        Self {
            positions,
            heads: heads_kv,
            rank: basis.rank,
            values,
        }
    }

    /// The stored (F16-rounded) sketch of one position of one head, read back as f32 values.
    fn stored(&self, head: usize, t: usize) -> impl Iterator<Item = f32> + '_ {
        let base = (head * self.positions + t) * self.rank;
        self.values[base..base + self.rank]
            .iter()
            .map(|v| v.to_f32())
    }
}

fn dot_f64(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| f64::from(*x) * f64::from(*y))
        .sum()
}

/// The approximate logits of one KV head from the F16 sketches:
/// `logits[lane * positions + t] = scale * (B q_lane) . z(t)`, `lane` over the query heads of
/// this KV head.
fn sketch_logits(
    q: &[f32],
    sketch: &SketchPlane,
    basis: &Basis,
    head: usize,
    g: &SparseGeometry,
    positions: usize,
) -> Vec<f64> {
    let group = g.heads_q / g.heads_kv;
    let mut logits = vec![0.0f64; group * positions];
    for lane in 0..group {
        let query = &q[(head * group + lane) * g.head_dim..][..g.head_dim];
        let qz = basis.project(head, query);
        for t in 0..positions {
            let mut dot = 0.0f64;
            for (r, qb) in qz.iter().enumerate() {
                let base = (head * positions + t) * sketch.rank + r;
                dot += qb * f64::from(sketch.values[base].to_f32());
            }
            logits[lane * positions + t] = dot * g.scale;
        }
    }
    logits
}

/// The exact logits of one KV head against the true keys — the oracle selector's input.
fn exact_logits(
    q: &[f32],
    keys: &[f32],
    head: usize,
    g: &SparseGeometry,
    positions: usize,
) -> Vec<f64> {
    let group = g.heads_q / g.heads_kv;
    let mut logits = vec![0.0f64; group * positions];
    for lane in 0..group {
        let query = &q[(head * group + lane) * g.head_dim..][..g.head_dim];
        for t in 0..positions {
            let key = &keys[(t * g.heads_kv + head) * g.head_dim..][..g.head_dim];
            logits[lane * positions + t] = dot_f64(query, key) * g.scale;
        }
    }
    logits
}

/// The relevance score of every candidate position of one KV head: the softmax mass over the
/// candidate range `[sink, positions - recent)`, summed over the query lanes of the head.
/// Non-candidates score `None`.
fn relevance_scores(logits: &[f64], positions: usize, config: &SparseConfig) -> Vec<Option<f64>> {
    let lanes = logits.len() / positions;
    let mut scores = vec![None; positions];
    if positions <= config.sink + config.recent {
        return scores;
    }
    let (lo, hi) = (config.sink, positions - config.recent);
    let mut mass = vec![0.0f64; hi - lo];
    for lane in 0..lanes {
        let row = &logits[lane * positions..][..positions];
        let max = row[lo..hi]
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let sum: f64 = row[lo..hi].iter().map(|v| (v - max).exp()).sum();
        for (offset, value) in row[lo..hi].iter().enumerate() {
            mass[offset] += (value - max).exp() / sum;
        }
    }
    for (offset, value) in mass.iter().enumerate() {
        scores[lo + offset] = Some(*value);
    }
    scores
}

/// Deterministic top-K over the candidate scores: the values above the k-th largest in ascending
/// index order, then the lowest-index ties (the radix-select rule). Returns ascending indices.
fn deterministic_top_k(scores: &[Option<f64>], k: usize) -> Vec<u32> {
    let mut candidates: Vec<(usize, f64)> = scores
        .iter()
        .enumerate()
        .filter_map(|(t, s)| s.map(|v| (t, v)))
        .collect();
    candidates.sort_by(|&(ta, sa), &(tb, sb)| {
        sb.partial_cmp(&sa)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(ta.cmp(&tb))
    });
    let mut picked: Vec<u32> = candidates[..k.min(candidates.len())]
        .iter()
        .map(|&(t, _)| t as u32)
        .collect();
    picked.sort_unstable();
    picked
}

/// The selection of one KV head from precomputed logits.
fn select_from_logits(logits: &[f64], positions: usize, config: &SparseConfig) -> SparseSelection {
    if !config.sparse(positions) {
        return SparseSelection {
            indices: (0..positions as u32).collect(),
            sparse: false,
        };
    }
    let scores = relevance_scores(logits, positions, config);
    let mut indices = deterministic_top_k(&scores, config.top_k);
    indices.extend(0..config.sink as u32);
    indices.extend((positions - config.recent) as u32..positions as u32);
    SparseSelection {
        indices,
        sparse: true,
    }
}

/// Query-aware selection of every KV head from the F16 sketches: the approximate selector the
/// device path implements. `q` is `[heads_q][head_dim]` post-norm post-rope.
pub(crate) fn select(
    q: &[f32],
    sketch: &SketchPlane,
    basis: &Basis,
    g: &SparseGeometry,
    positions: usize,
    config: &SparseConfig,
) -> Result<Vec<SparseSelection>, EngError> {
    config.check()?;
    check_inputs(q, g, positions)?;
    Ok((0..g.heads_kv)
        .map(|head| {
            let logits = sketch_logits(q, sketch, basis, head, g, positions);
            select_from_logits(&logits, positions, config)
        })
        .collect())
}

/// The oracle selector: the same rule against the exact logits. Its subset isolates the
/// omission error of top-K attention from the sketch approximation. `keys` is cache order
/// `[positions][heads_kv][head_dim]`.
pub(crate) fn select_exact(
    q: &[f32],
    keys: &[f32],
    g: &SparseGeometry,
    positions: usize,
    config: &SparseConfig,
) -> Result<Vec<SparseSelection>, EngError> {
    config.check()?;
    check_inputs(q, g, positions)?;
    if keys.len() != positions * g.heads_kv * g.head_dim {
        return Err(EngError::BadRequest(format!(
            "sparse select: {} key values do not fill {positions} positions of {} x {}",
            keys.len(),
            g.heads_kv,
            g.head_dim
        )));
    }
    Ok((0..g.heads_kv)
        .map(|head| {
            let logits = exact_logits(q, keys, head, g, positions);
            select_from_logits(&logits, positions, config)
        })
        .collect())
}

fn check_inputs(q: &[f32], g: &SparseGeometry, positions: usize) -> Result<(), EngError> {
    if positions == 0 {
        return Err(EngError::BadRequest(
            "sparse select: no positions to select over".to_owned(),
        ));
    }
    if q.len() != g.heads_q * g.head_dim {
        return Err(EngError::BadRequest(format!(
            "sparse select: q holds {} values, the geometry needs {} x {}",
            q.len(),
            g.heads_q,
            g.head_dim
        )));
    }
    if !g.heads_q.is_multiple_of(g.heads_kv) {
        return Err(EngError::BadRequest(
            "sparse select: the query heads do not divide into the KV heads".to_owned(),
        ));
    }
    Ok(())
}

/// Exact attention over each KV head's selected subset, f64 accumulation: every query head of a
/// KV head attends exactly `selections[head]`, nothing else. This is exact attention over a
/// subset, NOT exact equivalence to dense attention — callers state the difference as the
/// omission error. A dense selection (`sparse == false`) reproduces dense attention over
/// `0..positions`, the comparison baseline of the phase-3 tests.
pub(crate) fn attend_f64(
    q: &[f32],
    keys: &[f32],
    values: &[f32],
    selections: &[SparseSelection],
    g: &SparseGeometry,
    positions: usize,
) -> Result<Vec<f64>, EngError> {
    check_inputs(q, g, positions)?;
    if selections.len() != g.heads_kv {
        return Err(EngError::BadRequest(format!(
            "sparse attend: {} selections do not cover {} KV heads",
            selections.len(),
            g.heads_kv
        )));
    }
    if keys.len() != positions * g.heads_kv * g.head_dim
        || values.len() != positions * g.heads_kv * g.head_dim
    {
        return Err(EngError::BadRequest(format!(
            "sparse attend: keys ({}) or values ({}) do not fill {positions} positions of {} x {}",
            keys.len(),
            values.len(),
            g.heads_kv,
            g.head_dim
        )));
    }
    let group = g.heads_q / g.heads_kv;
    let mut out = vec![0.0f64; g.heads_q * g.head_dim];
    for head in 0..g.heads_kv {
        let selection = &selections[head];
        for lane in 0..group {
            let qh = head * group + lane;
            let query = &q[qh * g.head_dim..][..g.head_dim];
            let mut logits = Vec::with_capacity(selection.indices.len());
            for &t in &selection.indices {
                let key =
                    &keys[t as usize * g.heads_kv * g.head_dim + head * g.head_dim..][..g.head_dim];
                logits.push(dot_f64(query, key) * g.scale);
            }
            let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let sum: f64 = logits.iter().map(|v| (v - max).exp()).sum();
            for (j, &t) in selection.indices.iter().enumerate() {
                let p = (logits[j] - max).exp() / sum;
                let value = &values[t as usize * g.heads_kv * g.head_dim + head * g.head_dim..]
                    [..g.head_dim];
                for (d, v) in value.iter().enumerate() {
                    out[qh * g.head_dim + d] += p * f64::from(*v);
                }
            }
        }
    }
    Ok(out)
}

/// The F32 wrapper the model-side comparisons consume.
pub(crate) fn attend_subset(
    q: &[f32],
    keys: &[f32],
    values: &[f32],
    selections: &[SparseSelection],
    g: &SparseGeometry,
    positions: usize,
) -> Result<Vec<f32>, EngError> {
    Ok(attend_f64(q, keys, values, selections, g, positions)?
        .into_iter()
        .map(|v| v as f32)
        .collect())
}

/// Row-scaled maximum error of `got` against the f64 reference: every element's absolute error
/// over `max(|reference element|, row max)` of its query head, the convention of the f64
/// oracle's tolerance metric. Returns the worst ratio and the offending element.
pub(crate) fn max_row_error(got: &[f64], reference: &[f64], g: &SparseGeometry) -> (f64, usize) {
    let mut worst = 0.0f64;
    let mut offender = 0usize;
    for qh in 0..g.heads_q {
        let row = &reference[qh * g.head_dim..][..g.head_dim];
        let row_max = row.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        for (d, w) in row.iter().enumerate() {
            let index = qh * g.head_dim + d;
            let err = (got[index] - w).abs();
            let ratio = err / w.abs().max(row_max).max(f64::MIN_POSITIVE);
            if ratio > worst {
                worst = ratio;
                offender = index;
            }
        }
    }
    (worst, offender)
}

/// Shared indices of two selections' top-K part (before sinks and window), over the picked
/// count — the recall of one selector against another.
pub(crate) fn topk_overlap(a: &SparseSelection, b: &SparseSelection, top_k: usize) -> usize {
    let a_set: std::collections::HashSet<u32> = a.indices[..top_k.min(a.indices.len())]
        .iter()
        .copied()
        .collect();
    b.indices[..top_k.min(b.indices.len())]
        .iter()
        .filter(|t| a_set.contains(t))
        .count()
}
