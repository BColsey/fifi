//! Wrapper tests of sparse_topk. Device runs go through gpu_run.sh only.

use super::*;
use crate::kernels::cuda::CudaContext;
use crate::kernels::cuda::testkit::{self, Rng};

/// Qwen36-shaped geometry: 2 KV heads, 1024 cells, sink 16, recent 64, top_k 128.
struct Fixture {
    n_kv_head: usize,
    cells: usize,
    top_k: usize,
    sink: usize,
    recent: usize,
    n_q_head: usize,
    head_dim: usize,
    rank: usize,
    q: Vec<f32>,
    k: Vec<f32>,
    basis: Vec<f32>,
}

fn fixture() -> Fixture {
    let (n_kv_head, cells, top_k, sink, recent) = (2usize, 1024, 128, 16, 64);
    let (n_q_head, head_dim, rank) = (16usize, 256, 32);
    let mut rng = Rng::new(0x7070_0000_7069_636b);
    let draw = |rng: &mut Rng, len: usize| -> Vec<f32> {
        (0..len).map(|_| (rng.unit() * 2.0 - 1.0) as f32).collect()
    };
    Fixture {
        n_kv_head,
        cells,
        top_k,
        sink,
        recent,
        n_q_head,
        head_dim,
        rank,
        q: draw(&mut rng, n_q_head * head_dim),
        k: draw(&mut rng, cells * n_kv_head * head_dim),
        basis: draw(&mut rng, n_kv_head * rank * head_dim),
    }
}

/// The host selection rule over the device's own scores: the top_k largest candidate bit
/// patterns, value descending then ascending index, then sinks and window (the reference
/// `deterministic_top_k` tie rule on identical inputs, so equality is exact).
fn host_selection(
    scores: &[f32],
    head: usize,
    cells: usize,
    kv_len: usize,
    f: &Fixture,
) -> Vec<u32> {
    let (lo, hi) = (f.sink, kv_len - f.recent);
    let mut candidates: Vec<(usize, u32)> = (lo..hi)
        .map(|t| (t, scores[head * cells + t].to_bits()))
        .collect();
    candidates.sort_by(|&(ta, sa), &(tb, sb)| sb.cmp(&sa).then(ta.cmp(&tb)));
    let take = f.top_k.min(candidates.len());
    let mut indices: Vec<u32> = candidates[..take].iter().map(|&(t, _)| t as u32).collect();
    indices.sort_unstable();
    indices.extend(0..f.sink as u32);
    indices.extend((kv_len - f.recent) as u32..kv_len as u32);
    indices
}

#[test]
fn topk_matches_the_host_selection_exactly() {
    if !testkit::device_present("sparse_topk exact selection") {
        return;
    }
    let f = fixture();
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let kv_len = 1000usize;
    // Score with the composed pipeline, then select.
    let k = testkit::upload_f32(&ctx, &f.k[..kv_len * f.n_kv_head * f.head_dim]);
    let basis = testkit::upload_f32(&ctx, &f.basis);
    let q = testkit::upload_f32(&ctx, &f.q);
    let plane_bytes = f.n_kv_head * f.cells * f.rank * 2;
    let sketch = testkit::upload_filled(&ctx, 0xff, plane_bytes);
    let score_bytes = f.n_kv_head * f.cells * 4;
    let scores = testkit::upload_filled(&ctx, 0xff, score_bytes);
    let pos = ctx
        .upload(&((kv_len - 1) as u32).to_ne_bytes())
        .expect("pos");
    let status = ctx.upload(&0u32.to_ne_bytes()).expect("status");
    ctx.sparse_sketch_chunk(
        &stream,
        crate::kernels::cuda::resident::sparse_sketch::SparseSketchShape {
            rows: kv_len,
            n_kv_head: f.n_kv_head,
            head_dim: f.head_dim,
            rank: f.rank,
            cells: f.cells,
        },
        crate::kernels::cuda::resident::sparse_sketch::SparseSketchData {
            k: k.range(0, kv_len * f.n_kv_head * f.head_dim * 4).unwrap(),
            basis: basis.range(0, f.basis.len() * 4).unwrap(),
            sketch: sketch.range(0, plane_bytes).unwrap(),
        },
        0,
    )
    .expect("sketch chunk");
    ctx.sparse_score(
        &stream,
        crate::kernels::cuda::resident::sparse_score::SparseScoreShape {
            n_q_head: f.n_q_head,
            n_kv_head: f.n_kv_head,
            head_dim: f.head_dim,
            rank: f.rank,
            cells: f.cells,
            sink: f.sink,
            recent: f.recent,
            scale: 1.0 / 16.0,
        },
        crate::kernels::cuda::resident::sparse_score::SparseScoreData {
            q: q.range(0, f.q.len() * 4).unwrap(),
            basis: basis.range(0, f.basis.len() * 4).unwrap(),
            sketch: sketch.range(0, plane_bytes).unwrap(),
            scores: scores.range(0, score_bytes).unwrap(),
        },
        pos.range(0, 4).unwrap(),
        status.range(0, 4).unwrap(),
    )
    .expect("score");
    // Selection over the device's own scores: run topk, read back both outputs, and check the
    // rows against the host rule applied to the same score values.
    let row = f.top_k + f.sink + f.recent;
    let index_bytes = f.n_kv_head * row * 4;
    let indices = testkit::upload_filled(&ctx, 0xff, index_bytes);
    let counts = testkit::upload_filled(&ctx, 0xff, f.n_kv_head * 4);
    ctx.sparse_topk(
        &stream,
        SparseTopkShape {
            n_kv_head: f.n_kv_head,
            cells: f.cells,
            top_k: f.top_k,
            sink: f.sink,
            recent: f.recent,
        },
        SparseTopkData {
            scores: scores.range(0, score_bytes).unwrap(),
            indices: indices.range(0, index_bytes).unwrap(),
            counts: counts.range(0, f.n_kv_head * 4).unwrap(),
        },
        pos.range(0, 4).unwrap(),
        status.range(0, 4).unwrap(),
    )
    .expect("topk");
    stream.synchronize().expect("drain");
    assert_eq!(testkit::read_bytes(&ctx, &status, 4), 0u32.to_ne_bytes());
    let got_scores = testkit::floats(&testkit::read_bytes(&ctx, &scores, score_bytes));
    let got_indices = testkit::read_bytes(&ctx, &indices, index_bytes);
    let got_counts = testkit::read_bytes(&ctx, &counts, f.n_kv_head * 4);
    for head in 0..f.n_kv_head {
        let want = host_selection(&got_scores, head, f.cells, kv_len, &f);
        let count =
            u32::from_ne_bytes(got_counts[head * 4..head * 4 + 4].try_into().unwrap()) as usize;
        assert_eq!(count, want.len(), "head {head} count");
        for (slot, w) in want.iter().enumerate() {
            let at = (head * row + slot) * 4;
            let g = u32::from_ne_bytes(got_indices[at..at + 4].try_into().unwrap());
            assert_eq!(g, *w, "head {head} slot {slot}");
        }
    }
}

#[test]
fn tie_break_is_value_desc_then_ascending_index() {
    if !testkit::device_present("sparse_topk tie rule") {
        return;
    }
    let mut f = fixture();
    f.top_k = 5;
    f.sink = 2;
    f.recent = 3;
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    // Crafted scores over kv_len = 16 cells: a tie of 0.5 at cells 5, 3, 9 and distinct values
    // elsewhere; the rule must pick 7.0 (cell 7), 2.0 (cell 4), then the lowest-index 0.5 ties
    // (3, 5) for top_k = 5 - wait, 7.0 and 2.0 are two picks, so ties 3, 5, 9 give 9 candidates
    // ranked; top-5 = {7, 4, 3, 5, 9} ascending {3, 4, 5, 7, 9}.
    let kv_len = 16usize;
    let mut scores = vec![0.125f32; f.n_kv_head * f.cells];
    for head in 0..f.n_kv_head {
        let plane = &mut scores[head * f.cells..][..f.cells];
        plane[4] = 2.0;
        plane[7] = 7.0;
        plane[3] = 0.5;
        plane[5] = 0.5;
        plane[9] = 0.5;
    }
    let score_dev = testkit::upload_f32(&ctx, &scores);
    let row = f.top_k + f.sink + f.recent;
    let index_bytes = f.n_kv_head * row * 4;
    let indices = testkit::upload_filled(&ctx, 0xff, index_bytes);
    let counts = testkit::upload_filled(&ctx, 0xff, f.n_kv_head * 4);
    let pos = ctx
        .upload(&((kv_len - 1) as u32).to_ne_bytes())
        .expect("pos");
    let status = ctx.upload(&0u32.to_ne_bytes()).expect("status");
    ctx.sparse_topk(
        &stream,
        SparseTopkShape {
            n_kv_head: f.n_kv_head,
            cells: f.cells,
            top_k: f.top_k,
            sink: f.sink,
            recent: f.recent,
        },
        SparseTopkData {
            scores: score_dev.range(0, f.n_kv_head * f.cells * 4).unwrap(),
            indices: indices.range(0, index_bytes).unwrap(),
            counts: counts.range(0, f.n_kv_head * 4).unwrap(),
        },
        pos.range(0, 4).unwrap(),
        status.range(0, 4).unwrap(),
    )
    .expect("topk");
    stream.synchronize().expect("drain");
    let got = testkit::read_bytes(&ctx, &indices, index_bytes);
    for head in 0..f.n_kv_head {
        let row_at = |slot: usize| -> u32 {
            let at = (head * row + slot) * 4;
            u32::from_ne_bytes(got[at..at + 4].try_into().unwrap())
        };
        // Ascending picks, then sinks [0, 2), then window [13, 16).
        for (slot, want) in [3u32, 4, 5, 7, 9, 0, 1, 13, 14, 15].iter().enumerate() {
            assert_eq!(row_at(slot), *want, "head {head} slot {slot}");
        }
    }
}

#[test]
fn refused_launches_write_nothing_and_raise_mask_4() {
    if !testkit::device_present("sparse_topk refusals") {
        return;
    }
    let f = fixture();
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let row = f.top_k + f.sink + f.recent;
    let index_bytes = f.n_kv_head * row * 4;
    let shape = SparseTopkShape {
        n_kv_head: f.n_kv_head,
        cells: f.cells,
        top_k: f.top_k,
        sink: f.sink,
        recent: f.recent,
    };
    for (label, pos_value) in [
        ("pos above cells", f.cells as u32),
        // kv_len = 51 <= sink + recent = 80: no candidate, refused on the device.
        ("candidate-free length", 50u32),
    ] {
        let scores = testkit::upload_filled(&ctx, 0x00, f.n_kv_head * f.cells * 4);
        let indices = testkit::upload_filled(&ctx, 0xff, index_bytes);
        let counts = testkit::upload_filled(&ctx, 0xff, f.n_kv_head * 4);
        let pos = ctx.upload(&pos_value.to_ne_bytes()).expect("pos");
        let status = ctx.upload(&0u32.to_ne_bytes()).expect("status");
        ctx.sparse_topk(
            &stream,
            shape,
            SparseTopkData {
                scores: scores.range(0, f.n_kv_head * f.cells * 4).unwrap(),
                indices: indices.range(0, index_bytes).unwrap(),
                counts: counts.range(0, f.n_kv_head * 4).unwrap(),
            },
            pos.range(0, 4).unwrap(),
            status.range(0, 4).unwrap(),
        )
        .expect("queued; the device refuses it");
        stream.synchronize().expect("drain");
        assert_eq!(
            testkit::read_bytes(&ctx, &indices, index_bytes),
            vec![0xffu8; index_bytes],
            "{label}: a refused launch must not write an index"
        );
        assert_eq!(
            testkit::read_bytes(&ctx, &counts, f.n_kv_head * 4),
            vec![0xffu8; f.n_kv_head * 4],
            "{label}: a refused launch must not write a count"
        );
        assert_eq!(
            testkit::read_bytes(&ctx, &status, 4),
            4u32.to_ne_bytes(),
            "{label}: mask 4"
        );
    }
}

#[test]
fn wrapper_rejects_bad_calls_before_any_launch() {
    if !testkit::device_present("sparse_topk rejections") {
        return;
    }
    let f = fixture();
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let score_bytes = f.n_kv_head * f.cells * 4;
    let row = f.top_k + f.sink + f.recent;
    let index_bytes = f.n_kv_head * row * 4;
    let scores = testkit::upload_filled(&ctx, 0x00, score_bytes);
    let indices = testkit::upload_filled(&ctx, 0xff, index_bytes);
    let counts = testkit::upload_filled(&ctx, 0xff, f.n_kv_head * 4);
    let pos = ctx.upload(&999u32.to_ne_bytes()).expect("pos");
    let status = ctx.upload(&0u32.to_ne_bytes()).expect("status");
    let mut zero_topk = SparseTopkShape {
        n_kv_head: f.n_kv_head,
        cells: f.cells,
        top_k: f.top_k,
        sink: f.sink,
        recent: f.recent,
    };
    zero_topk.top_k = 0;
    let mut no_gap = zero_topk;
    no_gap.top_k = f.top_k;
    no_gap.sink = f.cells - f.recent;
    let cases: Vec<(&str, Result<(), EngError>)> = vec![
        (
            "top_k 0",
            ctx.sparse_topk(
                &stream,
                zero_topk,
                SparseTopkData {
                    scores: scores.range(0, score_bytes).unwrap(),
                    indices: indices.range(0, index_bytes).unwrap(),
                    counts: counts.range(0, f.n_kv_head * 4).unwrap(),
                },
                pos.range(0, 4).unwrap(),
                status.range(0, 4).unwrap(),
            ),
        ),
        (
            "no candidate gap",
            ctx.sparse_topk(
                &stream,
                no_gap,
                SparseTopkData {
                    scores: scores.range(0, score_bytes).unwrap(),
                    indices: indices.range(0, index_bytes).unwrap(),
                    counts: counts.range(0, f.n_kv_head * 4).unwrap(),
                },
                pos.range(0, 4).unwrap(),
                status.range(0, 4).unwrap(),
            ),
        ),
        (
            "short indices",
            ctx.sparse_topk(
                &stream,
                SparseTopkShape {
                    n_kv_head: f.n_kv_head,
                    cells: f.cells,
                    top_k: f.top_k,
                    sink: f.sink,
                    recent: f.recent,
                },
                SparseTopkData {
                    scores: scores.range(0, score_bytes).unwrap(),
                    indices: indices.range(0, 8).unwrap(),
                    counts: counts.range(0, f.n_kv_head * 4).unwrap(),
                },
                pos.range(0, 4).unwrap(),
                status.range(0, 4).unwrap(),
            ),
        ),
        (
            "overlap",
            ctx.sparse_topk(
                &stream,
                SparseTopkShape {
                    n_kv_head: f.n_kv_head,
                    cells: f.cells,
                    top_k: f.top_k,
                    sink: f.sink,
                    recent: f.recent,
                },
                SparseTopkData {
                    scores: scores.range(0, score_bytes).unwrap(),
                    // The written index rows sit inside the scores buffer.
                    indices: scores.range(0, index_bytes).unwrap(),
                    counts: counts.range(0, f.n_kv_head * 4).unwrap(),
                },
                pos.range(0, 4).unwrap(),
                status.range(0, 4).unwrap(),
            ),
        ),
    ];
    for (label, result) in cases {
        match result {
            Err(EngError::BadRequest(message)) => {
                assert!(!message.is_empty(), "{label}: expected a refusal text")
            }
            other => panic!("{label}: expected a BadRequest, got {other:?}"),
        }
    }
}
