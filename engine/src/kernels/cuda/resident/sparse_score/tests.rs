//! Wrapper tests of sparse_score. Device runs go through gpu_run.sh only.

use super::*;
use crate::kernels::cuda::testkit::{self, Rng};
use crate::kernels::cuda::{CudaContext, DeviceBuffer, DeviceStream};

/// Qwen36-shaped geometry with small extents: 16 query heads over 2 KV heads (group 8),
/// 256 values per head, rank 32, 1024 cells.
struct Fixture {
    n_q_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    rank: usize,
    cells: usize,
    sink: usize,
    recent: usize,
    scale: f32,
    q: Vec<f32>,
    k: Vec<f32>,
    basis: Vec<f32>,
}

fn fixture() -> Fixture {
    let (n_q_head, n_kv_head, head_dim, rank, cells, sink, recent) =
        (16usize, 2, 256, 32, 1024, 16, 64);
    let mut rng = Rng::new(0x5eed_0000_5c0a_5e01);
    let draw = |rng: &mut Rng, len: usize| -> Vec<f32> {
        (0..len).map(|_| (rng.unit() * 2.0 - 1.0) as f32).collect()
    };
    Fixture {
        n_q_head,
        n_kv_head,
        head_dim,
        rank,
        cells,
        sink,
        recent,
        scale: 1.0 / 16.0,
        q: draw(&mut rng, n_q_head * head_dim),
        k: draw(&mut rng, cells * n_kv_head * head_dim),
        basis: draw(&mut rng, n_kv_head * rank * head_dim),
    }
}

fn shape_of(f: &Fixture) -> SparseScoreShape {
    SparseScoreShape {
        n_q_head: f.n_q_head,
        n_kv_head: f.n_kv_head,
        head_dim: f.head_dim,
        rank: f.rank,
        cells: f.cells,
        sink: f.sink,
        recent: f.recent,
        scale: f.scale,
    }
}

/// The exact sketch cell the sparse_sketch kernel stores (the host reference builds the same
/// plane): `RNE_f16(sum_d basis[h][r][d] * k[t][h][d])` read back as f32, from an f64 sum.
fn sketch_value(f: &Fixture, head: usize, t: usize, r: usize) -> f32 {
    let mut acc = 0f64;
    for d in 0..f.head_dim {
        acc += f.basis[(head * f.rank + r) * f.head_dim + d] as f64
            * f.k[(t * f.n_kv_head + head) * f.head_dim + d] as f64;
    }
    testkit::f16_to_f32(testkit::f16_bits(acc as f32))
}

/// The host reference of `relevance_scores` of `sparse_reference.rs`: the softmax mass over the
/// candidate range, summed over the head's lanes, in f64.
fn reference_scores(f: &Fixture, head: usize, kv_len: usize) -> Vec<f32> {
    let group = f.n_q_head / f.n_kv_head;
    let (lo, hi) = (f.sink, kv_len - f.recent);
    // Per lane: project the query, take logits against the stored sketch, softmax over [lo, hi).
    let mut mass = vec![0f64; hi - lo];
    for lane in 0..group {
        let mut qz = vec![0f64; f.rank];
        for (r, qz_r) in qz.iter_mut().enumerate() {
            let mut acc = 0f64;
            for d in 0..f.head_dim {
                acc += f.basis[(head * f.rank + r) * f.head_dim + d] as f64
                    * f.q[(head * group + lane) * f.head_dim + d] as f64;
            }
            *qz_r = acc;
        }
        let mut logits = vec![0f64; hi - lo];
        for (offset, logit) in logits.iter_mut().enumerate() {
            let t = lo + offset;
            let mut dot = 0f64;
            for (r, qb) in qz.iter().enumerate() {
                dot += qb * f64::from(sketch_value(f, head, t, r));
            }
            *logit = dot * f.scale as f64;
        }
        let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let sum: f64 = logits.iter().map(|v| (v - max).exp()).sum();
        for (offset, value) in logits.iter().enumerate() {
            mass[offset] += (value - max).exp() / sum;
        }
    }
    mass.into_iter().map(|v| v as f32).collect()
}

/// Runs the composed pipeline (sketch chunk append + scoring) for `kv_len` filled cells.
fn run_pipeline(
    f: &Fixture,
    ctx: &CudaContext,
    stream: &DeviceStream,
    kv_len: usize,
) -> (DeviceBuffer, u32) {
    let k = testkit::upload_f32(ctx, &f.k[..kv_len * f.n_kv_head * f.head_dim]);
    let basis = testkit::upload_f32(ctx, &f.basis);
    let plane_bytes = f.n_kv_head * f.cells * f.rank * 2;
    let sketch = testkit::upload_filled(ctx, 0xff, plane_bytes);
    let scores = testkit::upload_filled(ctx, 0xff, f.n_kv_head * f.cells * 4);
    let pos = ctx
        .upload(&((kv_len - 1) as u32).to_ne_bytes())
        .expect("pos");
    let status = ctx.upload(&0u32.to_ne_bytes()).expect("status");
    let sketch_shape = crate::kernels::cuda::resident::sparse_sketch::SparseSketchShape {
        rows: kv_len,
        n_kv_head: f.n_kv_head,
        head_dim: f.head_dim,
        rank: f.rank,
        cells: f.cells,
    };
    ctx.sparse_sketch_chunk(
        stream,
        sketch_shape,
        crate::kernels::cuda::resident::sparse_sketch::SparseSketchData {
            k: k.range(0, kv_len * f.n_kv_head * f.head_dim * 4).unwrap(),
            basis: basis.range(0, f.basis.len() * 4).unwrap(),
            sketch: sketch.range(0, plane_bytes).unwrap(),
        },
        0,
    )
    .expect("sketch chunk");
    ctx.sparse_score(
        stream,
        shape_of(f),
        SparseScoreData {
            q: testkit::upload_f32(ctx, &f.q)
                .range(0, f.q.len() * 4)
                .unwrap(),
            basis: basis.range(0, f.basis.len() * 4).unwrap(),
            sketch: sketch.range(0, plane_bytes).unwrap(),
            scores: scores.range(0, f.n_kv_head * f.cells * 4).unwrap(),
        },
        pos.range(0, 4).unwrap(),
        status.range(0, 4).unwrap(),
    )
    .expect("score");
    stream.synchronize().expect("drain");
    let mask = u32::from_ne_bytes(
        testkit::read_bytes(ctx, &status, 4)[..4]
            .try_into()
            .unwrap(),
    );
    (scores, mask)
}

#[test]
fn scores_match_the_host_reference() {
    if !testkit::device_present("sparse_score reference") {
        return;
    }
    let f = fixture();
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let kv_len = 1000usize;
    let (scores, mask) = run_pipeline(&f, &ctx, &stream, kv_len);
    assert_eq!(
        mask, 0,
        "an accepted launch must leave the status word at 0"
    );
    let got = testkit::floats(&testkit::read_bytes(
        &ctx,
        &scores,
        f.n_kv_head * f.cells * 4,
    ));
    for head in 0..f.n_kv_head {
        let want = reference_scores(&f, head, kv_len);
        for (offset, w) in want.iter().enumerate() {
            let t = f.sink + offset;
            let g = got[head * f.cells + t];
            assert!(
                (g - w).abs() <= w.abs() * 1e-3 + 1e-6,
                "head {head} cell {t}: got {g}, want {w}"
            );
        }
        // Non-candidates are never written: the sentinel fill must survive outside [lo, hi).
        for t in (0..f.sink).chain(kv_len - f.recent..f.cells) {
            assert_eq!(
                got[head * f.cells + t].to_bits(),
                0xffff_ffffu32,
                "head {head} cell {t} must keep the sentinel fill"
            );
        }
    }
}

#[test]
fn refused_position_raises_mask_4_without_writing() {
    if !testkit::device_present("sparse_score refused position") {
        return;
    }
    let f = fixture();
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let k = testkit::upload_f32(&ctx, &f.k[..8 * f.n_kv_head * f.head_dim]);
    let basis = testkit::upload_f32(&ctx, &f.basis);
    let plane_bytes = f.n_kv_head * f.cells * f.rank * 2;
    let sketch = testkit::upload_filled(&ctx, 0xff, plane_bytes);
    let scores = testkit::upload_filled(&ctx, 0xff, f.n_kv_head * f.cells * 4);
    // pos = cells is outside 0..cells, so the device must refuse the launch.
    let pos = ctx.upload(&(f.cells as u32).to_ne_bytes()).expect("pos");
    let status = ctx.upload(&0u32.to_ne_bytes()).expect("status");
    ctx.sparse_score(
        &stream,
        shape_of(&f),
        SparseScoreData {
            q: testkit::upload_f32(&ctx, &f.q)
                .range(0, f.q.len() * 4)
                .unwrap(),
            basis: basis.range(0, f.basis.len() * 4).unwrap(),
            sketch: sketch.range(0, plane_bytes).unwrap(),
            scores: scores.range(0, f.n_kv_head * f.cells * 4).unwrap(),
        },
        pos.range(0, 4).unwrap(),
        status.range(0, 4).unwrap(),
    )
    .expect("queued; the device refuses it");
    stream.synchronize().expect("drain");
    assert_eq!(
        testkit::read_bytes(&ctx, &scores, f.n_kv_head * f.cells * 4),
        vec![0xffu8; f.n_kv_head * f.cells * 4],
        "a refused launch must not write a byte of the scores"
    );
    assert_eq!(testkit::read_bytes(&ctx, &status, 4), 4u32.to_ne_bytes());
    assert!(k.bytes() >= 8 * f.n_kv_head * f.head_dim * 4);
}

#[test]
fn no_candidates_writes_nothing_and_keeps_status_zero() {
    if !testkit::device_present("sparse_score no candidates") {
        return;
    }
    let f = fixture();
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    // kv_len = 51 leaves no cell of [sink, kv_len - recent): [16, -13).
    let kv_len = 51usize;
    let basis = testkit::upload_f32(&ctx, &f.basis);
    let plane_bytes = f.n_kv_head * f.cells * f.rank * 2;
    let sketch = testkit::upload_filled(&ctx, 0xff, plane_bytes);
    let scores = testkit::upload_filled(&ctx, 0xff, f.n_kv_head * f.cells * 4);
    let pos = ctx
        .upload(&((kv_len - 1) as u32).to_ne_bytes())
        .expect("pos");
    let status = ctx.upload(&0u32.to_ne_bytes()).expect("status");
    ctx.sparse_score(
        &stream,
        shape_of(&f),
        SparseScoreData {
            q: testkit::upload_f32(&ctx, &f.q)
                .range(0, f.q.len() * 4)
                .unwrap(),
            basis: basis.range(0, f.basis.len() * 4).unwrap(),
            sketch: sketch.range(0, plane_bytes).unwrap(),
            scores: scores.range(0, f.n_kv_head * f.cells * 4).unwrap(),
        },
        pos.range(0, 4).unwrap(),
        status.range(0, 4).unwrap(),
    )
    .expect("queued");
    stream.synchronize().expect("drain");
    assert_eq!(
        testkit::read_bytes(&ctx, &scores, f.n_kv_head * f.cells * 4),
        vec![0xffu8; f.n_kv_head * f.cells * 4],
        "a candidate-free length must not write a score"
    );
    assert_eq!(testkit::read_bytes(&ctx, &status, 4), 0u32.to_ne_bytes());
}

#[test]
fn wrapper_rejects_bad_calls_before_any_launch() {
    if !testkit::device_present("sparse_score rejections") {
        return;
    }
    let f = fixture();
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let q = testkit::upload_f32(&ctx, &f.q);
    let basis = testkit::upload_f32(&ctx, &f.basis);
    let sketch = testkit::upload_filled(&ctx, 0xff, f.n_kv_head * f.cells * f.rank * 2);
    let scores = testkit::upload_filled(&ctx, 0xff, f.n_kv_head * f.cells * 4);
    let pos = ctx.upload(&0u32.to_ne_bytes()).expect("pos");
    let status = ctx.upload(&0u32.to_ne_bytes()).expect("status");
    let data = |q_bytes: usize| SparseScoreData {
        q: q.range(0, q_bytes).unwrap(),
        basis: basis.range(0, f.basis.len() * 4).unwrap(),
        sketch: sketch.range(0, f.n_kv_head * f.cells * f.rank * 2).unwrap(),
        scores: scores.range(0, f.n_kv_head * f.cells * 4).unwrap(),
    };
    let mut odd_heads = shape_of(&f);
    odd_heads.n_q_head = f.n_q_head + 1;
    let mut no_gap = shape_of(&f);
    no_gap.sink = f.cells - f.recent;
    let mut zero_scale = shape_of(&f);
    zero_scale.scale = 0.0;
    let mut thin_rank = shape_of(&f);
    thin_rank.rank = 1;
    let cases: Vec<(&str, Result<(), EngError>)> = vec![
        (
            "odd heads",
            ctx.sparse_score(
                &stream,
                odd_heads,
                data(f.q.len() * 4),
                pos.range(0, 4).unwrap(),
                status.range(0, 4).unwrap(),
            ),
        ),
        (
            "no candidate gap",
            ctx.sparse_score(
                &stream,
                no_gap,
                data(f.q.len() * 4),
                pos.range(0, 4).unwrap(),
                status.range(0, 4).unwrap(),
            ),
        ),
        (
            "zero scale",
            ctx.sparse_score(
                &stream,
                zero_scale,
                data(f.q.len() * 4),
                pos.range(0, 4).unwrap(),
                status.range(0, 4).unwrap(),
            ),
        ),
        (
            "rank 1",
            ctx.sparse_score(
                &stream,
                thin_rank,
                data(f.q.len() * 4),
                pos.range(0, 4).unwrap(),
                status.range(0, 4).unwrap(),
            ),
        ),
        (
            "short q",
            ctx.sparse_score(
                &stream,
                shape_of(&f),
                data(8),
                pos.range(0, 4).unwrap(),
                status.range(0, 4).unwrap(),
            ),
        ),
        (
            "overlap",
            ctx.sparse_score(
                &stream,
                shape_of(&f),
                SparseScoreData {
                    q: q.range(0, f.q.len() * 4).unwrap(),
                    basis: basis.range(0, f.basis.len() * 4).unwrap(),
                    sketch: sketch.range(0, f.n_kv_head * f.cells * f.rank * 2).unwrap(),
                    // The written scores extent sits inside the q buffer.
                    scores: q.range(0, f.n_kv_head * f.cells * 4).unwrap(),
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
