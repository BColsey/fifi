//! Flash-Next QSA decode step (design section "QSA").

use super::math::{dot, expect_len, rms0, rope_half, sigmoid};
use crate::EngError;
use crate::family::project::Projector;

#[derive(Debug, Clone, Copy)]
pub(crate) struct QsaDims {
    pub(crate) heads_q: u32,
    pub(crate) heads_kv: u32,
    pub(crate) head_dim: u32,
    pub(crate) idx_heads: u32,
    pub(crate) idx_dim: u32,
    /// Positions per indexer block.
    pub(crate) ratio: u32,
    /// Indexer position budget.
    pub(crate) budget: u32,
    pub(crate) rotary_dim: u32,
    pub(crate) theta: f32,
    pub(crate) hidden: u32,
    pub(crate) eps: f32,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct QsaShape {
    dims: QsaDims,
    /// Query width `heads_q * head_dim`.
    q_width: usize,
    /// Query and gate projection width `2 * q_width`.
    q_proj: usize,
    /// Key and value width `heads_kv * head_dim`.
    kv_width: usize,
    /// Indexer query width `idx_heads * idx_dim`.
    idx_width: usize,
    /// Indexer projection width `(idx_heads + 1) * idx_dim`.
    idx_proj: usize,
}

impl QsaShape {
    pub(crate) fn new(dims: QsaDims) -> Result<Self, EngError> {
        let counts = [
            dims.heads_q,
            dims.heads_kv,
            dims.head_dim,
            dims.idx_heads,
            dims.idx_dim,
            dims.ratio,
            dims.budget,
            dims.rotary_dim,
            dims.hidden,
        ];
        if counts.contains(&0) {
            return Err(EngError::Checkpoint("qsa dimension is zero".to_owned()));
        }
        if !dims.heads_q.is_multiple_of(dims.heads_kv) {
            return Err(EngError::Checkpoint(format!(
                "qsa query heads {} are not a multiple of key-value heads {}",
                dims.heads_q, dims.heads_kv
            )));
        }
        if !dims.rotary_dim.is_multiple_of(2)
            || dims.rotary_dim > dims.head_dim
            || dims.rotary_dim > dims.idx_dim
        {
            return Err(EngError::Checkpoint(format!(
                "qsa rotary dimension {} is invalid",
                dims.rotary_dim
            )));
        }
        if !dims.theta.is_finite() || dims.theta <= 0.0 {
            return Err(EngError::Checkpoint("qsa rope base is invalid".to_owned()));
        }
        if !dims.eps.is_finite() || dims.eps < 0.0 {
            return Err(EngError::Checkpoint(
                "qsa norm epsilon is invalid".to_owned(),
            ));
        }
        if dims.budget < dims.ratio {
            return Err(EngError::Checkpoint(
                "qsa budget must be at least the block ratio".to_owned(),
            ));
        }
        let overflow = || EngError::Checkpoint("qsa dimension product overflows usize".into());
        let wide = |v: u32| usize::try_from(v).map_err(|_| overflow());
        let q_width = wide(dims.heads_q)?
            .checked_mul(wide(dims.head_dim)?)
            .ok_or_else(overflow)?;
        let q_proj = q_width.checked_mul(2).ok_or_else(overflow)?;
        let kv_width = wide(dims.heads_kv)?
            .checked_mul(wide(dims.head_dim)?)
            .ok_or_else(overflow)?;
        let idx_width = wide(dims.idx_heads)?
            .checked_mul(wide(dims.idx_dim)?)
            .ok_or_else(overflow)?;
        let idx_proj = idx_width
            .checked_add(wide(dims.idx_dim)?)
            .ok_or_else(overflow)?;
        Ok(Self {
            dims,
            q_width,
            q_proj,
            kv_width,
            idx_width,
            idx_proj,
        })
    }

    pub(crate) fn dims(&self) -> QsaDims {
        self.dims
    }
}

pub(crate) struct QsaWeights<'a, W> {
    /// Query and gate projection `[2 * heads_q * head_dim, hidden]`, interleaved per head.
    pub(crate) q: &'a W,
    /// Key projection `[heads_kv * head_dim, hidden]`.
    pub(crate) k: &'a W,
    /// Value projection `[heads_kv * head_dim, hidden]`.
    pub(crate) v: &'a W,
    /// Output projection `[hidden, heads_q * head_dim]`.
    pub(crate) o: &'a W,
    /// Indexer projection `[(idx_heads + 1) * idx_dim, hidden]`.
    pub(crate) idx: &'a W,
    /// Query norm, shared by all heads.
    pub(crate) q_norm: &'a [f32],
    /// Key norm, shared by all heads.
    pub(crate) k_norm: &'a [f32],
    pub(crate) idx_q_norm: &'a [f32],
    pub(crate) idx_k_norm: &'a [f32],
}

/// Keys, values, and indexer keys, one entry per position.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct QsaCache {
    k: Vec<f32>,
    v: Vec<f32>,
    ik: Vec<f32>,
    len: usize,
}

impl QsaCache {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn len(&self) -> usize {
        self.len
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub(crate) fn byte_len(&self) -> usize {
        (self.k.len() + self.v.len() + self.ik.len()) * 4
    }
}

pub(crate) fn step<P: Projector>(
    projector: &mut P,
    shape: &QsaShape,
    weights: &QsaWeights<'_, P::Weight>,
    cache: &mut QsaCache,
    pos: u32,
    x: &[f32],
) -> Result<Vec<f32>, EngError> {
    let dims = shape.dims;
    let heads_kv = dims.heads_kv as usize;
    let head_dim = dims.head_dim as usize;
    let idx_dim = dims.idx_dim as usize;
    let hidden = dims.hidden as usize;
    let rd = dims.rotary_dim as usize;
    let eps = dims.eps;
    let theta = dims.theta;

    expect_len(x, hidden, "qsa input")?;
    expect_len(weights.q_norm, head_dim, "qsa query norm")?;
    expect_len(weights.k_norm, head_dim, "qsa key norm")?;
    expect_len(weights.idx_q_norm, idx_dim, "qsa indexer query norm")?;
    expect_len(weights.idx_k_norm, idx_dim, "qsa indexer key norm")?;
    if cache.len != pos as usize {
        return Err(EngError::BadRequest(format!(
            "qsa cache length {} does not match position {pos}",
            cache.len
        )));
    }
    let kv_fits = cache
        .len
        .checked_mul(shape.kv_width)
        .is_some_and(|n| cache.k.len() == n && cache.v.len() == n);
    let ik_fits = cache
        .len
        .checked_mul(idx_dim)
        .is_some_and(|n| cache.ik.len() == n);
    if !kv_fits || !ik_fits {
        return Err(EngError::BadRequest(
            "qsa cache does not match its shape".to_owned(),
        ));
    }

    let qp = projector.linear(weights.q, x)?;
    expect_len(&qp, shape.q_proj, "qsa query projection")?;
    let kp = projector.linear(weights.k, x)?;
    expect_len(&kp, shape.kv_width, "qsa key projection")?;
    let vp = projector.linear(weights.v, x)?;
    expect_len(&vp, shape.kv_width, "qsa value projection")?;
    let ip = projector.linear(weights.idx, x)?;
    expect_len(&ip, shape.idx_proj, "qsa indexer projection")?;

    let mut k_new = Vec::with_capacity(shape.kv_width);
    for g in 0..heads_kv {
        let mut kg = rms0(&kp[g * head_dim..(g + 1) * head_dim], weights.k_norm, eps)?;
        rope_half(&mut kg, rd, pos, theta)?;
        k_new.extend_from_slice(&kg);
    }

    let old = cache.len;
    cache.k.extend_from_slice(&k_new);
    cache.v.extend_from_slice(&vp);
    cache.ik.extend_from_slice(&ip[shape.idx_width..]);
    cache.len += 1;

    match attend(projector, shape, weights, cache, pos, &qp, &ip) {
        Ok(out) => Ok(out),
        Err(err) => {
            cache.k.truncate(old * shape.kv_width);
            cache.v.truncate(old * shape.kv_width);
            cache.ik.truncate(old * idx_dim);
            cache.len = old;
            Err(err)
        }
    }
}

fn attend<P: Projector>(
    projector: &mut P,
    shape: &QsaShape,
    weights: &QsaWeights<'_, P::Weight>,
    cache: &QsaCache,
    pos: u32,
    qp: &[f32],
    ip: &[f32],
) -> Result<Vec<f32>, EngError> {
    let dims = shape.dims;
    let hidden = dims.hidden as usize;

    let queries = indexer_queries(dims, ip, weights.idx_q_norm, pos)?;
    let score = block_scores(dims, &queries, &cache.ik, weights.idx_k_norm, pos)?;
    let selected = select_positions(dims, &score, pos)?;

    let heads = attend_selected(
        dims,
        qp,
        weights.q_norm,
        (&cache.k, &cache.v),
        &selected,
        pos,
    )?;

    let out = projector.linear(weights.o, &heads)?;
    expect_len(&out, hidden, "qsa output projection")?;
    Ok(out)
}

/// The gated attention of the token at `pos` over the cache positions `selected`, in that order:
/// per query head, rms0 with the query norm and rope at `pos` of its slice of the query and gate
/// projection `qp` (`[heads_q][q head_dim | gate head_dim]`), softmax over `q . k / sqrt(head_dim)`
/// against the keys of its kv head (`h / (heads_q / heads_kv)`), the weighted sum of the values,
/// times `sigmoid(gate)`. `kv` holds the keys and values `[position][heads_kv][head_dim]`.
/// Returns `[heads_q][head_dim]`, the input of the output projection.
pub(super) fn attend_selected(
    dims: QsaDims,
    qp: &[f32],
    q_norm: &[f32],
    kv: (&[f32], &[f32]),
    selected: &[usize],
    pos: u32,
) -> Result<Vec<f32>, EngError> {
    let (keys, values) = kv;
    let heads_q = dims.heads_q as usize;
    let heads_kv = dims.heads_kv as usize;
    let head_dim = dims.head_dim as usize;
    let rep = heads_q / heads_kv;
    let scale = (head_dim as f32).sqrt();
    let mut heads = Vec::with_capacity(heads_q * head_dim);
    for h in 0..heads_q {
        let qraw = qp
            .get(h * 2 * head_dim..(h + 1) * 2 * head_dim)
            .ok_or_else(|| EngError::BadRequest("qsa query projection is short".to_owned()))?;
        let mut qh = rms0(&qraw[..head_dim], q_norm, dims.eps)?;
        rope_half(&mut qh, dims.rotary_dim as usize, pos, dims.theta)?;
        let g = h / rep;
        let mut scores = Vec::with_capacity(selected.len());
        for &j in selected {
            scores.push(dot(&qh, kv_row(keys, (j * heads_kv + g) * head_dim, head_dim)?)? / scale);
        }
        let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let mut oh = vec![0.0f32; head_dim];
        for (&e, &j) in exps.iter().zip(selected) {
            let w = e / sum;
            let value = kv_row(values, (j * heads_kv + g) * head_dim, head_dim)?;
            for (o, v) in oh.iter_mut().zip(value) {
                *o += w * v;
            }
        }
        for (i, o) in oh.iter_mut().enumerate() {
            *o *= sigmoid(qraw[head_dim + i]);
        }
        heads.extend_from_slice(&oh);
    }
    Ok(heads)
}

/// The `len` cached values from `at` on, refused when the cache is shorter.
fn kv_row(cache: &[f32], at: usize, len: usize) -> Result<&[f32], EngError> {
    cache
        .get(at..at + len)
        .ok_or_else(|| EngError::BadRequest(format!("qsa cache holds no row at {at}")))
}

/// The indexer queries of the token at `pos`, one per indexer head: rms0 of the head's slice of
/// the indexer projection row `ip` with the indexer query norm, then rope at `pos`.
pub(super) fn indexer_queries(
    dims: QsaDims,
    ip: &[f32],
    q_norm: &[f32],
    pos: u32,
) -> Result<Vec<Vec<f32>>, EngError> {
    let idx_dim = dims.idx_dim as usize;
    let mut queries = Vec::with_capacity(dims.idx_heads as usize);
    for h in 0..dims.idx_heads as usize {
        let raw = ip.get(h * idx_dim..(h + 1) * idx_dim).ok_or_else(|| {
            EngError::BadRequest("qsa indexer projection row is short".to_owned())
        })?;
        let mut qi = rms0(raw, q_norm, dims.eps)?;
        rope_half(&mut qi, dims.rotary_dim as usize, pos, dims.theta)?;
        queries.push(qi);
    }
    Ok(queries)
}

/// The indexer key of block `b` from its `ratio` raw indexer keys `raw` (`[r][idx_dim]`, the
/// block's positions in order): each divided by `ratio` and summed in position order, then rms0
/// with the indexer key norm and rope at the block's first position.
pub(super) fn block_key(
    dims: QsaDims,
    raw: &[f32],
    b: usize,
    k_norm: &[f32],
) -> Result<Vec<f32>, EngError> {
    let ratio = dims.ratio as usize;
    let idx_dim = dims.idx_dim as usize;
    expect_len(raw, ratio * idx_dim, "qsa block raw keys")?;
    let mut pool = vec![0.0f32; idx_dim];
    for r in 0..ratio {
        for (i, p) in pool.iter_mut().enumerate() {
            *p += raw[r * idx_dim + i] / ratio as f32;
        }
    }
    let mut pk = rms0(&pool, k_norm, dims.eps)?;
    let first = b * ratio;
    let block_pos = u32::try_from(first)
        .map_err(|_| EngError::BadRequest(format!("qsa block position {first} exceeds u32")))?;
    rope_half(&mut pk, dims.rotary_dim as usize, block_pos, dims.theta)?;
    Ok(pk)
}

/// The score of every block completed at `pos` (`b < (pos + 1) / ratio`): the f64 sum over the
/// indexer heads of `max(query . key, 0)`, divided by `sqrt(idx_dim)`.
pub(super) fn block_scores(
    dims: QsaDims,
    queries: &[Vec<f32>],
    ik: &[f32],
    k_norm: &[f32],
    pos: u32,
) -> Result<Vec<f64>, EngError> {
    let blocks = (pos as usize + 1) / dims.ratio as usize;
    let mut score = Vec::with_capacity(blocks);
    for b in 0..blocks {
        let width = dims.ratio as usize * dims.idx_dim as usize;
        let raw = ik
            .get(b * width..(b + 1) * width)
            .ok_or_else(|| EngError::BadRequest(format!("qsa block {b} is not complete")))?;
        let pk = block_key(dims, raw, b, k_norm)?;
        let mut total = 0.0f64;
        for qi in queries {
            total += f64::from(dot(qi, &pk)?).max(0.0);
        }
        score.push(total / (dims.idx_dim as f64).sqrt());
    }
    Ok(score)
}

/// The positions the token at `pos` attends to: the `budget / ratio` best blocks by `score`
/// (descending, the lower block first on ties), each as its `ratio` positions in the ranked
/// order, then the open tail of positions after the last completed block.
pub(super) fn select_positions(
    dims: QsaDims,
    score: &[f64],
    pos: u32,
) -> Result<Vec<usize>, EngError> {
    let ratio = dims.ratio as usize;
    let visible = pos as usize + 1;
    let blocks = visible / ratio;
    if score.len() != blocks {
        return Err(EngError::BadRequest(format!(
            "qsa selection takes {blocks} block scores, got {}",
            score.len()
        )));
    }
    let tail = blocks * ratio;
    let take = blocks.min(dims.budget as usize / ratio);
    let mut order: Vec<usize> = (0..blocks).collect();
    order.sort_by(|&a, &b| score[b].total_cmp(&score[a]).then(a.cmp(&b)));
    let mut selected = Vec::new();
    for &block in order.iter().take(take) {
        for r in 0..ratio {
            selected.push(block * ratio + r);
        }
    }
    selected.extend(tail..visible);
    Ok(selected)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::family::project::{DenseProjector, DenseWeight};

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() <= 1e-5
    }

    fn fixture_dims() -> QsaDims {
        QsaDims {
            heads_q: 2,
            heads_kv: 1,
            head_dim: 4,
            idx_heads: 1,
            idx_dim: 2,
            ratio: 2,
            budget: 2,
            rotary_dim: 2,
            theta: 10_000.0,
            hidden: 2,
            eps: 1e-6,
        }
    }

    struct Fixture {
        q: DenseWeight,
        k: DenseWeight,
        v: DenseWeight,
        o: DenseWeight,
        idx: DenseWeight,
        q_norm: Vec<f32>,
        k_norm: Vec<f32>,
        idx_q_norm: Vec<f32>,
        idx_k_norm: Vec<f32>,
    }

    impl Fixture {
        fn weights(&self) -> QsaWeights<'_, DenseWeight> {
            QsaWeights {
                q: &self.q,
                k: &self.k,
                v: &self.v,
                o: &self.o,
                idx: &self.idx,
                q_norm: &self.q_norm,
                k_norm: &self.k_norm,
                idx_q_norm: &self.idx_q_norm,
                idx_k_norm: &self.idx_k_norm,
            }
        }
    }

    fn fixture() -> Fixture {
        let mut q = Vec::new();
        for r in 0..16usize {
            q.push(((7 * r % 11) as f32 - 5.0) / 5.0);
            q.push(((3 * r % 7) as f32 - 3.0) / 4.0);
        }
        let mut k = Vec::new();
        for r in 0..4usize {
            k.push((((5 * r + 1) % 9) as f32 - 4.0) / 4.0);
            k.push((((2 * r + 3) % 5) as f32 - 2.0) / 3.0);
        }
        let mut v = Vec::new();
        for r in 0..4usize {
            v.push((((3 * r + 2) % 7) as f32 - 3.0) / 3.0);
            v.push((((r + 1) % 4) as f32 - 1.5) / 2.0);
        }
        let mut idx = Vec::new();
        for r in 0..4usize {
            idx.push((((4 * r + 1) % 7) as f32 - 1.0) / 2.0);
            idx.push((((5 * r + 2) % 9) as f32 - 4.0) / 3.0);
        }
        let mut o = Vec::new();
        for r in 0..2usize {
            for c in 0..8usize {
                o.push(((((r * 8 + c) * 5) % 13) as f32 - 6.0) / 6.0);
            }
        }
        Fixture {
            q: DenseWeight::matrix(16, 2, q),
            k: DenseWeight::matrix(4, 2, k),
            v: DenseWeight::matrix(4, 2, v),
            o: DenseWeight::matrix(2, 8, o),
            idx: DenseWeight::matrix(4, 2, idx),
            q_norm: vec![0.1, -0.2, 0.3, 0.0],
            k_norm: vec![0.0, 0.5, -0.5, 0.25],
            idx_q_norm: vec![0.2, -0.1],
            idx_k_norm: vec![-0.3, 0.4],
        }
    }

    fn inputs() -> [[f32; 2]; 5] {
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, -1.0], [-0.5, 2.0]]
    }

    #[test]
    fn shape_rejects_invalid_dimensions() {
        let base = fixture_dims();
        let changes = [
            QsaDims {
                heads_q: 3,
                heads_kv: 2,
                ..base
            },
            QsaDims {
                rotary_dim: 3,
                ..base
            },
            QsaDims {
                rotary_dim: 6,
                ..base
            },
            QsaDims { theta: 0.0, ..base },
            QsaDims { ratio: 0, ..base },
        ];
        for dims in changes {
            assert!(
                matches!(QsaShape::new(dims), Err(EngError::Checkpoint(_))),
                "{dims:?} was not rejected as a checkpoint error"
            );
        }
    }

    #[test]
    fn five_steps_match_hand_values() {
        let fx = fixture();
        let shape = QsaShape::new(fixture_dims()).unwrap();
        let weights = fx.weights();
        let mut projector = DenseProjector::default();
        let mut cache = QsaCache::new();
        let expected = [
            [0.214_517_6, -0.444_569_2],
            [0.677_813_1, -0.068_546],
            [0.757_397_5, 0.103_092_4],
            [0.779_888_2, 0.342_250_4],
            [0.542_283_1, 0.104_497_9],
        ];
        for (pos, (x, want)) in inputs().iter().zip(expected).enumerate() {
            let out = step(&mut projector, &shape, &weights, &mut cache, pos as u32, x).unwrap();
            assert_eq!(out.len(), 2, "pos {pos} output length");
            for (a, e) in out.iter().zip(want) {
                assert!(close(*a, e), "pos {pos}: {a} vs {e}");
            }
        }
        assert_eq!(cache.len(), 5);
        assert_eq!(cache.byte_len(), 200);
    }

    #[test]
    fn cache_length_must_match_position() {
        let fx = fixture();
        let shape = QsaShape::new(fixture_dims()).unwrap();
        let weights = fx.weights();
        let mut projector = DenseProjector::default();
        let mut cache = QsaCache::new();
        let result = step(&mut projector, &shape, &weights, &mut cache, 1, &[1.0, 0.0]);
        assert!(
            matches!(result, Err(EngError::BadRequest(_))),
            "empty cache at position 1 was not rejected as a bad request"
        );
        assert_eq!(projector.calls, 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn failed_output_projection_restores_the_cache() {
        let mut fx = fixture();
        fx.o = DenseWeight::matrix(2, 7, vec![1.0; 14]);
        let shape = QsaShape::new(fixture_dims()).unwrap();
        let weights = fx.weights();
        let mut projector = DenseProjector::default();
        let mut cache = QsaCache::new();
        let result = step(&mut projector, &shape, &weights, &mut cache, 0, &[1.0, 0.0]);
        assert!(
            matches!(result, Err(EngError::BadRequest(_))),
            "bad output projection was not rejected as a bad request"
        );
        assert_eq!(cache, QsaCache::new());
    }

    #[test]
    fn norm_weights_have_head_width() {
        let mut fx = fixture();
        fx.q_norm = vec![0.0; 8];
        let shape = QsaShape::new(fixture_dims()).unwrap();
        let weights = fx.weights();
        let mut projector = DenseProjector::default();
        let mut cache = QsaCache::new();
        let result = step(&mut projector, &shape, &weights, &mut cache, 0, &[1.0, 0.0]);
        assert!(
            matches!(result, Err(EngError::BadRequest(_))),
            "query norm of length 8 was not rejected as a bad request"
        );
    }

    fn fixture_two_kv_dims() -> QsaDims {
        QsaDims {
            heads_q: 4,
            heads_kv: 2,
            head_dim: 6,
            idx_heads: 2,
            idx_dim: 4,
            ratio: 2,
            budget: 4,
            rotary_dim: 4,
            theta: 10_000.0,
            hidden: 2,
            eps: 1e-6,
        }
    }

    fn fixture_two_kv() -> Fixture {
        let mut q = Vec::new();
        for r in 0..48usize {
            q.push(((7 * r % 11) as f32 - 5.0) / 5.0);
            q.push(((3 * r % 7) as f32 - 3.0) / 4.0);
        }
        let mut k = Vec::new();
        for r in 0..12usize {
            k.push((((5 * r + 1) % 9) as f32 - 4.0) / 4.0);
            k.push((((2 * r + 3) % 5) as f32 - 2.0) / 3.0);
        }
        let mut v = Vec::new();
        for r in 0..12usize {
            v.push((((3 * r + 2) % 7) as f32 - 3.0) / 3.0);
            v.push((((r + 1) % 4) as f32 - 1.5) / 2.0);
        }
        let mut idx = Vec::new();
        for r in 0..12usize {
            idx.push((((4 * r + 1) % 7) as f32 - 1.0) / 2.0);
            idx.push((((5 * r + 2) % 9) as f32 - 4.0) / 3.0);
        }
        let mut o = Vec::new();
        for r in 0..2usize {
            for c in 0..24usize {
                o.push(((((r * 24 + c) * 5) % 13) as f32 - 6.0) / 6.0);
            }
        }
        Fixture {
            q: DenseWeight::matrix(48, 2, q),
            k: DenseWeight::matrix(12, 2, k),
            v: DenseWeight::matrix(12, 2, v),
            o: DenseWeight::matrix(2, 24, o),
            idx: DenseWeight::matrix(12, 2, idx),
            q_norm: (0..6usize).map(|i| ((i % 3) as f32 - 1.0) * 0.1).collect(),
            k_norm: (0..6usize)
                .map(|i| (((i + 1) % 4) as f32 - 1.5) * 0.2)
                .collect(),
            idx_q_norm: (0..4usize).map(|i| (i % 2) as f32 * 0.2 - 0.1).collect(),
            idx_k_norm: (0..4usize)
                .map(|i| (((i + 2) % 3) as f32 - 1.0) * 0.3)
                .collect(),
        }
    }

    #[test]
    fn shape_rejects_overflowing_products() {
        let dims = QsaDims {
            heads_q: u32::MAX,
            heads_kv: 1,
            head_dim: u32::MAX,
            ..fixture_dims()
        };
        assert!(
            matches!(QsaShape::new(dims), Err(EngError::Checkpoint(_))),
            "{dims:?} was not rejected as a checkpoint error"
        );
    }

    #[test]
    fn shape_rejects_budget_below_ratio() {
        let dims = QsaDims {
            budget: 1,
            ..fixture_dims()
        };
        assert!(
            matches!(QsaShape::new(dims), Err(EngError::Checkpoint(_))),
            "{dims:?} was not rejected as a checkpoint error"
        );
    }

    #[test]
    fn failed_step_restores_a_non_empty_cache() {
        let mut fx = fixture();
        let shape = QsaShape::new(fixture_dims()).unwrap();
        let mut projector = DenseProjector::default();
        let mut cache = QsaCache::new();
        step(
            &mut projector,
            &shape,
            &fx.weights(),
            &mut cache,
            0,
            &[1.0, 0.0],
        )
        .unwrap();
        let before = cache.clone();
        fx.o = DenseWeight::matrix(2, 7, vec![1.0; 14]);
        let result = step(
            &mut projector,
            &shape,
            &fx.weights(),
            &mut cache,
            1,
            &[0.0, 1.0],
        );
        assert!(
            matches!(result, Err(EngError::BadRequest(_))),
            "bad output projection at position 1 was not rejected as a bad request"
        );
        assert_eq!(cache, before, "failed step changed the cache");
    }

    #[test]
    fn two_kv_heads_match_reference_values() {
        let fx = fixture_two_kv();
        let shape = QsaShape::new(fixture_two_kv_dims()).unwrap();
        let weights = fx.weights();
        let mut projector = DenseProjector::default();
        let mut cache = QsaCache::new();
        let inputs = [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [1.0, -1.0],
            [-0.5, 2.0],
            [0.25, -1.5],
        ];
        let expected = [
            [0.569_413_7, 1.595_823_9],
            [0.215_817_5, 1.233_577_4],
            [0.662_463_1, 1.271_704_2],
            [0.505_869_9, 0.885_691],
            [0.022_832_1, 1.347_423],
            [1.233_916_3, 0.290_290_1],
        ];
        for (pos, (x, want)) in inputs.iter().zip(expected).enumerate() {
            let out = step(&mut projector, &shape, &weights, &mut cache, pos as u32, x).unwrap();
            assert_eq!(out.len(), 2, "pos {pos} output length");
            for (a, e) in out.iter().zip(want) {
                assert!(close(*a, e), "pos {pos}: {a} vs {e}");
            }
        }
    }

    #[test]
    fn cache_of_another_shape_is_rejected() {
        let fx = fixture();
        let shape = QsaShape::new(fixture_dims()).unwrap();
        let weights = fx.weights();
        let mut projector = DenseProjector::default();
        let mut cache = QsaCache::new();
        step(&mut projector, &shape, &weights, &mut cache, 0, &[1.0, 0.0]).unwrap();
        let fx2 = fixture_two_kv();
        let shape2 = QsaShape::new(fixture_two_kv_dims()).unwrap();
        let weights2 = fx2.weights();
        let result = step(
            &mut projector,
            &shape2,
            &weights2,
            &mut cache,
            1,
            &[0.0, 1.0],
        );
        assert!(
            matches!(result, Err(EngError::BadRequest(_))),
            "cache of another shape was not rejected as a bad request"
        );
        assert_eq!(cache.len(), 1);
    }
}

#[cfg(all(test, feature = "native-cuda", target_os = "linux"))]
mod cuda_attn_tests;
#[cfg(all(test, feature = "native-cuda", target_os = "linux"))]
mod cuda_tests;
