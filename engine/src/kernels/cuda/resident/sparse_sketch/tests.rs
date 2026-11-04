//! Wrapper tests of sparse_sketch. Device runs go through gpu_run.sh only.

use super::*;
use crate::kernels::cuda::CudaContext;
use crate::kernels::cuda::testkit::{self, Rng, f16_to_f32};

/// The host reference: `want[h][t][r] = sum_d basis[h][r][d] * k[t][h][d]` in f64, the value
/// the kernel stores as `RNE_f16` of its f32 shuffle-tree sum.
fn reference(
    k: &[f32],
    basis: &[f32],
    rows: usize,
    heads: usize,
    head_dim: usize,
    rank: usize,
) -> Vec<f64> {
    let mut want = vec![0f64; heads * rows * rank];
    for t in 0..rows {
        for h in 0..heads {
            for r in 0..rank {
                let mut acc = 0f64;
                for d in 0..head_dim {
                    acc += basis[(h * rank + r) * head_dim + d] as f64
                        * k[(t * heads + h) * head_dim + d] as f64;
                }
                want[(h * rows + t) * rank + r] = acc;
            }
        }
    }
    want
}

/// The kernel's f32 tree can land one fp16 quantum away from the exact f64 sum when the sum
/// straddles a rounding boundary, so allow ~1.5 quanta of slack around it.
fn within(want: f64, got: f32) -> bool {
    (got as f64 - want).abs() <= want.abs() * (1.0 / 512.0) + 1e-4
}

struct Fixture {
    heads: usize,
    head_dim: usize,
    rank: usize,
    cells: usize,
    k: Vec<f32>,
    basis: Vec<f32>,
    want: Vec<f64>,
}

/// Qwen36-shaped geometry with a small cell count: 2 KV heads, 256 values per head, rank 64.
fn fixture(rows: usize) -> Fixture {
    let (heads, head_dim, rank, cells) = (2usize, 256, 64, 128);
    let mut rng = Rng::new(0x5a5a_5a5a_0000_0001);
    let draw = |rng: &mut Rng, len: usize| -> Vec<f32> {
        (0..len).map(|_| (rng.unit() * 2.0 - 1.0) as f32).collect()
    };
    let k = draw(&mut rng, rows * heads * head_dim);
    let basis = draw(&mut rng, heads * rank * head_dim);
    let want = reference(&k, &basis, rows, heads, head_dim, rank);
    Fixture {
        heads,
        head_dim,
        rank,
        cells,
        k,
        basis,
        want,
    }
}

fn shape_of(f: &Fixture, rows: usize) -> SparseSketchShape {
    SparseSketchShape {
        rows,
        n_kv_head: f.heads,
        head_dim: f.head_dim,
        rank: f.rank,
        cells: f.cells,
    }
}

/// Reads the fp16 plane back and checks every cell of every head against the reference.
fn check_plane(f: &Fixture, bytes: &[u8], base: usize, rows: usize) {
    for h in 0..f.heads {
        for t in 0..rows {
            for r in 0..f.rank {
                let cell = base + t;
                let at = ((h * f.cells + cell) * f.rank + r) * 2;
                let got = f16_to_f32(u16::from_ne_bytes([bytes[at], bytes[at + 1]]));
                let want = f.want[(h * rows + t) * f.rank + r];
                assert!(
                    within(want, got),
                    "head {h} cell {cell} rank {r}: got {got}, want {want}"
                );
            }
        }
    }
}

#[test]
fn decode_append_matches_the_host_reference() {
    if !testkit::device_present("sparse_sketch decode reference") {
        return;
    }
    let rows = 3usize;
    let f = fixture(rows);
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let k = testkit::upload_f32(&ctx, &f.k);
    let basis = testkit::upload_f32(&ctx, &f.basis);
    let plane_bytes = f.heads * f.cells * f.rank * 2;
    let sketch = testkit::upload_filled(&ctx, 0xff, plane_bytes);
    let pos = ctx.upload(&5u32.to_ne_bytes()).expect("pos");
    let status = ctx.upload(&0u32.to_ne_bytes()).expect("status");
    ctx.sparse_sketch(
        &stream,
        shape_of(&f, rows),
        SparseSketchData {
            k: k.range(0, f.k.len() * 4).unwrap(),
            basis: basis.range(0, f.basis.len() * 4).unwrap(),
            sketch: sketch.range(0, plane_bytes).unwrap(),
        },
        pos.range(0, 4).unwrap(),
        status.range(0, 4).unwrap(),
    )
    .expect("sketch append");
    stream.synchronize().expect("drain");
    check_plane(
        &f,
        &testkit::read_bytes(&ctx, &sketch, plane_bytes),
        5,
        rows,
    );
    assert_eq!(
        testkit::read_bytes(&ctx, &status, 4),
        0u32.to_ne_bytes(),
        "an accepted launch must leave the status word at 0"
    );
}

#[test]
fn chunk_append_matches_the_host_reference() {
    if !testkit::device_present("sparse_sketch chunk reference") {
        return;
    }
    let rows = 7usize;
    let f = fixture(rows);
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let k = testkit::upload_f32(&ctx, &f.k);
    let basis = testkit::upload_f32(&ctx, &f.basis);
    let plane_bytes = f.heads * f.cells * f.rank * 2;
    let sketch = testkit::upload_filled(&ctx, 0xff, plane_bytes);
    ctx.sparse_sketch_chunk(
        &stream,
        shape_of(&f, rows),
        SparseSketchData {
            k: k.range(0, f.k.len() * 4).unwrap(),
            basis: basis.range(0, f.basis.len() * 4).unwrap(),
            sketch: sketch.range(0, plane_bytes).unwrap(),
        },
        13,
    )
    .expect("sketch chunk");
    stream.synchronize().expect("drain");
    check_plane(
        &f,
        &testkit::read_bytes(&ctx, &sketch, plane_bytes),
        13,
        rows,
    );
}

#[test]
fn refused_decode_position_writes_nothing_and_raises_mask_4() {
    if !testkit::device_present("sparse_sketch refused position") {
        return;
    }
    let rows = 2usize;
    let f = fixture(rows);
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let k = testkit::upload_f32(&ctx, &f.k);
    let basis = testkit::upload_f32(&ctx, &f.basis);
    let plane_bytes = f.heads * f.cells * f.rank * 2;
    let sketch = testkit::upload_filled(&ctx, 0xff, plane_bytes);
    // cells - 1 leaves no room for a second row, so the device must refuse the launch.
    let pos = ctx
        .upload(&((f.cells - 1) as u32).to_ne_bytes())
        .expect("pos");
    let status = ctx.upload(&0u32.to_ne_bytes()).expect("status");
    ctx.sparse_sketch(
        &stream,
        shape_of(&f, rows),
        SparseSketchData {
            k: k.range(0, f.k.len() * 4).unwrap(),
            basis: basis.range(0, f.basis.len() * 4).unwrap(),
            sketch: sketch.range(0, plane_bytes).unwrap(),
        },
        pos.range(0, 4).unwrap(),
        status.range(0, 4).unwrap(),
    )
    .expect("the launch is queued; the device refuses it");
    stream.synchronize().expect("drain");
    assert!(
        testkit::read_bytes(&ctx, &sketch, plane_bytes)
            .iter()
            .all(|b| *b == 0xff),
        "a refused launch must not write a byte of the plane"
    );
    assert_eq!(testkit::read_bytes(&ctx, &status, 4), 4u32.to_ne_bytes());
}

#[test]
fn wrapper_rejects_bad_calls_before_any_launch() {
    if !testkit::device_present("sparse_sketch rejections") {
        return;
    }
    let rows = 1usize;
    let f = fixture(rows);
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let k = testkit::upload_f32(&ctx, &f.k);
    let basis = testkit::upload_f32(&ctx, &f.basis);
    let plane_bytes = f.heads * f.cells * f.rank * 2;
    let sketch = testkit::upload_filled(&ctx, 0xff, plane_bytes);
    let pos = ctx.upload(&0u32.to_ne_bytes()).expect("pos");
    let status = ctx.upload(&0u32.to_ne_bytes()).expect("status");
    let data = SparseSketchData {
        k: k.range(0, f.k.len() * 4).unwrap(),
        basis: basis.range(0, f.basis.len() * 4).unwrap(),
        sketch: sketch.range(0, plane_bytes).unwrap(),
    };
    let mut bad_shape = shape_of(&f, rows);
    bad_shape.rank = 1;
    let rejections: Vec<(&str, Result<(), EngError>)> = vec![
        (
            "rank 1",
            ctx.sparse_sketch(
                &stream,
                bad_shape,
                SparseSketchData {
                    k: data.k,
                    basis: data.basis,
                    sketch: data.sketch,
                },
                pos.range(0, 4).unwrap(),
                status.range(0, 4).unwrap(),
            ),
        ),
        (
            "rows 17",
            ctx.sparse_sketch(
                &stream,
                shape_of(&f, 17),
                SparseSketchData {
                    k: data.k,
                    basis: data.basis,
                    sketch: data.sketch,
                },
                pos.range(0, 4).unwrap(),
                status.range(0, 4).unwrap(),
            ),
        ),
        (
            "short k",
            ctx.sparse_sketch(
                &stream,
                shape_of(&f, rows),
                SparseSketchData {
                    k: k.range(0, 8).unwrap(),
                    basis: data.basis,
                    sketch: data.sketch,
                },
                pos.range(0, 4).unwrap(),
                status.range(0, 4).unwrap(),
            ),
        ),
        (
            "chunk above cells",
            ctx.sparse_sketch_chunk(
                &stream,
                shape_of(&f, rows),
                SparseSketchData {
                    k: data.k,
                    basis: data.basis,
                    sketch: data.sketch,
                },
                f.cells,
            ),
        ),
        (
            "overlap",
            ctx.sparse_sketch(
                &stream,
                shape_of(&f, rows),
                SparseSketchData {
                    k: data.k,
                    basis: data.basis,
                    // The sketch extent written by the call sits inside the basis buffer.
                    sketch: basis
                        .range(0, shape_of(&f, rows).sketch_bytes().min(basis_bytes(&f)))
                        .unwrap(),
                },
                pos.range(0, 4).unwrap(),
                status.range(0, 4).unwrap(),
            ),
        ),
    ];
    for (label, result) in rejections {
        match result {
            Err(EngError::BadRequest(message)) => {
                assert!(!message.is_empty(), "{label}: expected a refusal text")
            }
            other => panic!("{label}: expected a BadRequest, got {other:?}"),
        }
    }
}

fn basis_bytes(f: &Fixture) -> usize {
    f.heads * f.rank * f.head_dim * 4
}
