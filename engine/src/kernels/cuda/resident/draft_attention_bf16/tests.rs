use super::*;
use crate::kernels::cuda::DeviceBuffer;
use crate::kernels::cuda::testkit::{Rng, device_present, read_bytes, upload_filled};
use half::bf16;

const SHAPE: Shape = Shape {
    query_heads: 24,
    kv_heads: 8,
    rows: 8,
    capacity: 19,
    first: 17,
    len: 7,
};

struct Fixture {
    shape: Shape,
    query: Vec<bf16>,
    // Logical oldest-first committed rows, followed by the tentative block.
    keys: Vec<bf16>,
    values: Vec<bf16>,
}

impl Fixture {
    fn random(shape: Shape) -> Self {
        let mut rng = Rng::new(30_249);
        let mut sample = |n| {
            (0..n)
                .map(|_| bf16::from_f32((rng.below(997) as f32 - 498.0) / 265.0))
                .collect()
        };
        let kv = (shape.len + shape.rows) * shape.kv_heads * HEAD_DIM;
        Self {
            shape,
            query: sample(shape.rows * shape.query_heads * HEAD_DIM),
            keys: sample(kv),
            values: sample(kv),
        }
    }

    fn input_bytes(&self) -> [Vec<u8>; RANGES - 1] {
        let Shape {
            kv_heads,
            capacity,
            first,
            len,
            ..
        } = self.shape;
        let width = kv_heads * HEAD_DIM;
        let ring = |plane: &[bf16]| {
            let mut physical = vec![bf16::NAN; capacity * width];
            for (slot, row) in (first..capacity)
                .chain(0..first)
                .zip(plane.chunks_exact(width).take(len))
            {
                physical[slot * width..(slot + 1) * width].copy_from_slice(row);
            }
            bf16_bytes(&physical)
        };
        [
            bf16_bytes(&self.query),
            bf16_bytes(&self.keys[len * width..]),
            bf16_bytes(&self.values[len * width..]),
            ring(&self.keys),
            ring(&self.values),
        ]
    }

    fn reference(&self) -> Vec<f32> {
        let mut result = Vec::with_capacity(self.query.len());
        let width = self.shape.kv_heads * HEAD_DIM;
        for (item, query) in self.query.chunks_exact(HEAD_DIM).enumerate() {
            let head = item % self.shape.query_heads;
            let kv = head * self.shape.kv_heads / self.shape.query_heads;
            let head_range = kv * HEAD_DIM..(kv + 1) * HEAD_DIM;
            let scores: Vec<_> = self
                .keys
                .chunks_exact(width)
                .map(|row| {
                    let dot = query
                        .iter()
                        .zip(&row[head_range.clone()])
                        .map(|(q, k)| q.to_f64() * k.to_f64())
                        .sum::<f64>();
                    dot / (HEAD_DIM as f64).sqrt()
                })
                .collect();
            let peak = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let masses: Vec<_> = scores.iter().map(|score| (score - peak).exp()).collect();
            let sum = masses.iter().sum::<f64>();
            let mut attended = [0.0; HEAD_DIM];
            for (row, mass) in self.values.chunks_exact(width).zip(masses) {
                for (dst, v) in attended.iter_mut().zip(&row[head_range.clone()]) {
                    *dst += mass * v.to_f64() / sum;
                }
            }
            result.extend(attended.map(|value| bf16::from_f64(value).to_f32()));
        }
        result
    }
}

fn bf16_bytes(plane: &[bf16]) -> Vec<u8> {
    plane.iter().flat_map(|value| value.to_le_bytes()).collect()
}

fn whole(buffer: &DeviceBuffer) -> DeviceRange<'_> {
    buffer.range(0, buffer.bytes()).unwrap()
}

struct Buffers {
    inputs: [DeviceBuffer; RANGES - 1],
    out: DeviceBuffer,
}

impl Buffers {
    fn new(ctx: &CudaContext, inputs: &[Vec<u8>; RANGES - 1]) -> Self {
        Self {
            inputs: std::array::from_fn(|i| ctx.upload(&inputs[i]).unwrap()),
            out: upload_filled(ctx, 0xff, inputs[0].len() + 2),
        }
    }

    fn call(&self, shape: Shape) -> DraftAttentionCall<'_> {
        let [query, draft_k, draft_v, committed_k, committed_v] = self.inputs.each_ref().map(whole);
        DraftAttentionCall {
            query_heads: shape.query_heads,
            kv_heads: shape.kv_heads,
            rows: shape.rows,
            query,
            draft: DraftKv {
                keys: draft_k,
                values: draft_v,
            },
            committed: CommittedKv {
                kv: DraftKv {
                    keys: committed_k,
                    values: committed_v,
                },
                capacity: shape.capacity,
                first: shape.first,
                len: shape.len,
            },
            output: whole(&self.out),
        }
    }
}

fn run(ctx: &CudaContext, fixture: &Fixture) -> Vec<f32> {
    let stream = ctx.stream().unwrap();
    let inputs = fixture.input_bytes();
    let buffers = Buffers::new(ctx, &inputs);
    ctx.draft_attention_bf16(&stream, buffers.call(fixture.shape))
        .unwrap();
    assert_eq!(stream.enqueued(), 1);
    stream.synchronize().unwrap();
    for (original, device) in inputs.iter().zip(&buffers.inputs) {
        assert_eq!(&read_bytes(ctx, device, original.len()), original);
    }
    let out = read_bytes(ctx, &buffers.out, buffers.out.bytes());
    let (values, guard) = out.split_at(out.len() - 2);
    assert_eq!(guard, [0xff, 0xff]);
    values
        .chunks_exact(2)
        .map(|pair| bf16::from_le_bytes([pair[0], pair[1]]).to_f32())
        .collect()
}

#[test]
fn wraps_and_matches_independent_fp64_attention() {
    if !device_present("draft BF16 attention oracle") {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    for shape in [
        SHAPE,
        Shape {
            first: 3,
            len: SHAPE.capacity,
            ..SHAPE
        },
        Shape {
            capacity: 1031,
            first: 1027,
            len: 1024,
            ..SHAPE
        },
        Shape {
            query_heads: 2,
            kv_heads: 1,
            rows: 1,
            len: 0,
            ..SHAPE
        },
    ] {
        let fixture = Fixture::random(shape);
        let expected = fixture.reference();
        let actual = run(&ctx, &fixture);
        assert_eq!(actual.len(), expected.len());
        let mut max_error = 0.0_f32;
        for (i, (actual, expected)) in actual.into_iter().zip(expected).enumerate() {
            let error = (actual - expected).abs();
            assert!(
                error <= 0.007_875 * expected.abs() + 5e-7,
                "committed rows {}, element {i}: expected {expected}, got {actual}",
                shape.len
            );
            max_error = max_error.max(error);
        }
        eprintln!(
            "committed rows {}, maximum absolute error {max_error}",
            shape.len
        );
    }
}

#[test]
fn every_query_sees_the_final_tentative_row_without_committed_context() {
    if !device_present("draft BF16 full-block visibility") {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let mut fixture = Fixture::random(Shape { len: 0, ..SHAPE });
    fixture.query.fill(bf16::ONE);
    fixture.keys.fill(bf16::from_f32(-16.0));
    fixture.values.fill(bf16::ZERO);
    let final_row_start = (SHAPE.rows - 1) * SHAPE.kv_heads * HEAD_DIM;
    fixture.keys[final_row_start..].fill(bf16::from_f32(16.0));
    for (head, value) in fixture.values[final_row_start..]
        .chunks_exact_mut(HEAD_DIM)
        .enumerate()
    {
        value.fill(bf16::from_f32((head + 1) as f32 / 8.0));
    }
    let actual = run(&ctx, &fixture);
    for (item, head) in actual.chunks_exact(HEAD_DIM).enumerate() {
        let query_head = item % SHAPE.query_heads;
        let expected = ((query_head / 3) + 1) as f32 / 8.0;
        assert!(head.iter().all(|&value| value == expected), "head {item}");
    }
}

#[test]
fn invalid_shape_and_ring_geometry_have_no_region_plan() {
    assert!(SHAPE.region_bytes().is_some());
    for shape in [
        Shape {
            kv_heads: 0,
            ..SHAPE
        },
        Shape {
            query_heads: 23,
            ..SHAPE
        },
        Shape { rows: 0, ..SHAPE },
        Shape {
            rows: MAX_ROWS + 1,
            ..SHAPE
        },
        Shape {
            capacity: usize::MAX,
            ..SHAPE
        },
        Shape {
            capacity: 0,
            first: 0,
            len: 0,
            ..SHAPE
        },
        Shape {
            first: SHAPE.capacity,
            ..SHAPE
        },
        Shape {
            len: SHAPE.capacity + 1,
            ..SHAPE
        },
    ] {
        assert!(shape.region_bytes().is_none());
    }
}

fn rejects(name: &str, message: &str, change: impl FnOnce(&mut DraftAttentionCall<'_>)) {
    if !device_present(name) {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let buffers = Buffers::new(&ctx, &Fixture::random(SHAPE).input_bytes());
    let mut call = buffers.call(SHAPE);
    change(&mut call);
    let Err(EngError::BadRequest(error)) = ctx.draft_attention_bf16(&stream, call) else {
        panic!("expected BadRequest for {name}");
    };
    assert!(error.contains(message), "{error}");
    assert_eq!(stream.enqueued(), 0);
}

#[test]
fn rejects_short_committed_plane() {
    rejects(
        "draft BF16 short committed plane",
        "committed keys range holds",
        |call| {
            let plane = call.committed.kv.keys;
            call.committed.kv.keys = plane.buffer().range(0, plane.bytes() - 2).unwrap();
        },
    );
}

#[test]
fn rejects_misaligned_output() {
    rejects(
        "draft BF16 misaligned output",
        "not a multiple of 2",
        |call| {
            call.output = call.output.buffer().range(1, call.query.bytes()).unwrap();
        },
    );
}

#[test]
fn rejects_output_input_overlap() {
    rejects("draft BF16 output overlap", "overlap", |call| {
        call.output = call.query
    });
}

#[test]
fn rejects_tentative_committed_overlap() {
    rejects("draft BF16 tentative overlap", "overlap", |call| {
        call.draft.keys = call.committed.kv.keys;
    });
}
