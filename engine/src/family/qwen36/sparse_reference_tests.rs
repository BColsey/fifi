//! Host tests of the sparse-attention reference (phase 3): selection determinism and bounds,
//! the dense fallback, the error decomposition (omission versus sketch), and the softmax
//! contract of subset attention. Fixtures use `fixtures::val` hash values, so every run sees
//! identical inputs and the pinned bounds below are stable.

use super::sparse_reference::{
    Basis, SketchPlane, SparseConfig, SparseGeometry, attend_f64, max_row_error, select,
    select_exact, topk_overlap,
};
use crate::family::qwen36::fixtures::val;

/// The Qwen3.6 full-attention core geometry.
const G: SparseGeometry = SparseGeometry {
    heads_q: 16,
    heads_kv: 2,
    head_dim: 256,
    scale: 1.0 / 16.0,
};

fn config(top_k: usize) -> SparseConfig {
    SparseConfig {
        rank: 64,
        top_k,
        sink: 64,
        recent: 1024,
    }
}

fn vals(tag: u64, n: usize) -> Vec<f32> {
    (0..n as u64).map(|i| val(tag, i)).collect()
}

/// Cache-order keys or values of `positions` rows.
fn cache(tag: u64, positions: usize) -> Vec<f32> {
    vals(tag, positions * G.heads_kv * G.head_dim)
}

fn dense_selection(positions: usize) -> Vec<super::sparse_reference::SparseSelection> {
    (0..G.heads_kv)
        .map(|_| super::sparse_reference::SparseSelection {
            indices: (0..positions as u32).collect(),
            sparse: false,
        })
        .collect()
}

/// One fixture: hash-noise keys/values plus, per KV head and query lane, `needles_per_lane`
/// positions in the candidate range whose key is a strong multiple of that lane's query (a
/// needle per lane, so every lane's softmax mass concentrates on its own needles and the
/// omission error of a needle-respecting selection is provably tiny). Returns (q, keys, values,
/// planted positions).
fn planted_fixture(
    positions: usize,
    needles_per_lane: usize,
    config: &SparseConfig,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<usize>) {
    let mut q = vals(11, G.heads_q * G.head_dim);
    let group = G.heads_q / G.heads_kv;
    for head in 0..G.heads_kv {
        for d in 0..G.head_dim {
            q[head * group * G.head_dim + d] = val(31 + head as u64, d as u64);
        }
    }
    let mut keys = cache(21, positions);
    let mut planted = Vec::new();
    let lo = config.sink + 1;
    let hi = positions - config.recent - 1;
    let step = (hi - lo).max(1);
    for head in 0..G.heads_kv {
        for lane in 0..group {
            for n in 0..needles_per_lane {
                let t =
                    lo + ((lane * needles_per_lane + n) * step / (group * needles_per_lane)) % step;
                planted.push(t);
                for d in 0..G.head_dim {
                    keys[t * G.heads_kv * G.head_dim + head * G.head_dim + d] =
                        q[(head * group + lane) * G.head_dim + d] * 4.0;
                }
            }
        }
    }
    let values = cache(41, positions);
    let mut distinct = planted;
    distinct.sort_unstable();
    distinct.dedup();
    (q, keys, values, distinct)
}

#[test]
fn selection_is_deterministic_and_within_bounds() {
    let positions = 2048usize;
    let config = config(512);
    let (q, keys, _values, _planted) = planted_fixture(positions, 2, &config);
    let basis = Basis::random_orthonormal(G.heads_kv, G.head_dim, config.rank, 7);
    let sketch = SketchPlane::build(&basis, &keys, positions, G.heads_kv);
    let first = select(&q, &sketch, &basis, &G, positions, &config).unwrap();
    let second = select(&q, &sketch, &basis, &G, positions, &config).unwrap();
    assert_eq!(first, second, "selection must be deterministic");
    for (head, sel) in first.iter().enumerate() {
        assert!(sel.sparse, "head {head} must be sparse at {positions}");
        assert_eq!(
            sel.indices.len(),
            config.top_k + config.sink + config.recent
        );
        // Disjoint union, ascending top_k, exact sinks and window.
        let (top, rest) = sel.indices.split_at(config.top_k);
        assert!(
            top.windows(2).all(|w| w[0] < w[1]),
            "the top_k part must be ascending and unique"
        );
        assert!(
            top.iter()
                .all(|&t| (t as usize) >= config.sink && (t as usize) < positions - config.recent)
        );
        assert_eq!(
            &rest[..config.sink],
            &(0..config.sink as u32).collect::<Vec<_>>(),
            "the sink block is positions 0..sink"
        );
        assert_eq!(
            &rest[config.sink..],
            &((positions - config.recent) as u32..positions as u32).collect::<Vec<_>>(),
            "the window block is the last `recent` positions"
        );
        let mut all = sel.indices.clone();
        all.sort_unstable();
        assert!(
            all.windows(2).all(|w| w[0] < w[1]),
            "the union must be duplicate-free"
        );
    }
}

#[test]
fn oracle_selection_finds_planted_and_omission_error_is_tiny() {
    let positions = 2048usize;
    let config = config(512);
    let (q, keys, values, planted) = planted_fixture(positions, 2, &config);
    let reference = attend_f64(
        &q,
        &keys,
        &values,
        &dense_selection(positions),
        &G,
        positions,
    )
    .unwrap();
    let oracle = select_exact(&q, &keys, &G, positions, &config).unwrap();
    for (head, selection) in oracle.iter().enumerate() {
        let hits = selection.indices[..config.top_k]
            .iter()
            .filter(|t| planted.contains(&(**t as usize)))
            .count();
        assert_eq!(
            hits,
            planted.len(),
            "head {head}: exact top-K must find every needle ({hits}/{})",
            planted.len()
        );
    }
    let oracle_out = attend_f64(&q, &keys, &values, &oracle, &G, positions).unwrap();
    let (omission, at) = max_row_error(&oracle_out, &reference, &G);
    // Every lane's mass concentrates on its own needles (needle logit ~21 versus noise <=~1.1),
    // so the mass the oracle subset can drop is bounded by the noise tail, far below 1e-5.
    assert!(
        omission < 1e-5,
        "omission error {omission} at element {at} exceeds the needle-mass bound"
    );
}

#[test]
fn sketch_selection_runs_and_decomposes_against_the_oracle() {
    let positions = 2048usize;
    let config = config(512);
    let (q, keys, values, planted) = planted_fixture(positions, 2, &config);
    let basis = Basis::random_orthonormal(G.heads_kv, G.head_dim, config.rank, 7);
    let sketch = SketchPlane::build(&basis, &keys, positions, G.heads_kv);
    let reference = attend_f64(
        &q,
        &keys,
        &values,
        &dense_selection(positions),
        &G,
        positions,
    )
    .unwrap();
    let oracle = select_exact(&q, &keys, &G, positions, &config).unwrap();
    let approximate = select(&q, &sketch, &basis, &G, positions, &config).unwrap();
    for head in 0..G.heads_kv {
        let recall = topk_overlap(&oracle[head], &approximate[head], config.top_k);
        let found = approximate[head].indices[..config.top_k]
            .iter()
            .filter(|t| planted.contains(&(**t as usize)))
            .count();
        // Rank 64 of 256 keeps the needle direction far above the hash noise (projected needle
        // logit ~5 versus noise ~0.2); the exact recall is deterministic for this fixture, so
        // pin it with a little slack for the borderline positions.
        assert!(
            found >= planted.len() - 1,
            "head {head}: sketch selection finds {found}/{} needles (oracle overlap {recall})",
            planted.len()
        );
    }
    let sketch_out = attend_f64(&q, &keys, &values, &approximate, &G, positions).unwrap();
    let (sketch_error, _) = max_row_error(&sketch_out, &reference, &G);
    let oracle_out = attend_f64(&q, &keys, &values, &oracle, &G, positions).unwrap();
    let (omission, _) = max_row_error(&oracle_out, &reference, &G);
    // The sketch error is the reported decomposition: omission plus selection loss. It cannot
    // be assumed to dominate the oracle's (value cancellation can order two subsets either
    // way); the meaningful bound is that a needle-respecting selection stays near the dense
    // result, which is what the recall assert above guarantees structurally.
    assert!(
        sketch_error < 1e-2,
        "sketch error {sketch_error} at rank {} (omission {omission}) is far above the needle-mass scale",
        config.rank
    );
}

#[test]
fn subset_attention_is_a_correct_softmax_over_the_subset() {
    // Hand-checkable geometry: 2 query heads over 1 KV head of 4, three keys, the selection
    // keeps keys 0 and 2 only. The expected output is a plain softmax over the two kept keys —
    // what a masked dense result would NOT give (masking after a dense softmax would keep the
    // dropped key's mass in the denominator).
    let g = SparseGeometry {
        heads_q: 2,
        heads_kv: 1,
        head_dim: 4,
        scale: 1.0,
    };
    let q = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let keys = vec![2.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0];
    let values = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0];
    let selection = vec![super::sparse_reference::SparseSelection {
        indices: vec![0, 2],
        sparse: true,
    }];
    let out = attend_f64(&q, &keys, &values, &selection, &g, 3).unwrap();
    // Head 0: q = [1,0,0,0]; kept logits {k0: 2, k2: 0} -> p = [e^2, 1] / (e^2 + 1).
    let e2 = std::f64::consts::E * std::f64::consts::E;
    let p0 = e2 / (1.0 + e2);
    let p2 = 1.0 / (1.0 + e2);
    // v0 = [1,0,0,0], v2 = [0,1,1,0]: out = p0*v0 + p2*v2.
    assert!((out[0] - p0).abs() < 1e-12, "h0 d0 = {}", out[0]);
    assert!((out[1] - p2).abs() < 1e-12, "h0 d1 = {}", out[1]);
    assert!((out[2] - p2).abs() < 1e-12, "h0 d2 = {}", out[2]);
    // Head 1: q = [0,1,0,0]; kept logits {0, 0} -> uniform; out = mean of the kept v rows.
    assert!((out[4] - 0.5).abs() < 1e-12, "h1 d0 = {}", out[4]);
    assert!((out[6] - 0.5).abs() < 1e-12, "h1 d2 = {}", out[6]);
}

#[test]
fn short_context_takes_the_dense_fallback() {
    let config = config(512);
    let positions = config.top_k + config.sink + config.recent; // exactly the cutoff
    let (q, keys, values, _planted) = planted_fixture(positions, 1, &config);
    let basis = Basis::random_orthonormal(G.heads_kv, G.head_dim, config.rank, 7);
    let sketch = SketchPlane::build(&basis, &keys, positions, G.heads_kv);
    let selections = select(&q, &sketch, &basis, &G, positions, &config).unwrap();
    for (head, selection) in selections.iter().enumerate() {
        assert!(!selection.sparse, "head {head} must fall back to dense");
        assert_eq!(
            selections[head].indices,
            (0..positions as u32).collect::<Vec<_>>()
        );
    }
    let dense_out = attend_f64(
        &q,
        &keys,
        &values,
        &dense_selection(positions),
        &G,
        positions,
    )
    .unwrap();
    let subset_out = attend_f64(&q, &keys, &values, &selections, &G, positions).unwrap();
    assert_eq!(dense_out, subset_out, "the dense fallback is exact");
}

#[test]
fn bad_configurations_are_refused() {
    let positions = 2048usize;
    let cfg = config(512);
    let (q, keys, _values, _planted) = planted_fixture(positions, 1, &cfg);
    let mut bad = cfg;
    bad.rank = 0;
    assert!(select_exact(&q, &keys, &G, positions, &bad).is_err());
    bad = cfg;
    bad.top_k = 0;
    assert!(select_exact(&q, &keys, &G, positions, &bad).is_err());
}
