use super::*;
use crate::kernels::cuda::resident::step_params::{POS_MAX, STATUS_INVALID_STEP_STATE};
use crate::kernels::cuda::testkit::{
    Rng, device_present, floats, read_bytes, upload_f32, upload_filled,
};
use crate::kernels::cuda::{CapturePlan, DeviceBuffer};

const SHAPE: RelativeAttentionShape = RelativeAttentionShape {
    query_heads: 24,
    kv_heads: 8,
    relative_dims: 16,
    relative_extent: 13,
    capacity: 19,
};
const QUERY: usize = SHAPE.query_heads * HEAD_DIM;
const KV: usize = SHAPE.kv_heads * HEAD_DIM;

fn pattern(len: usize, seed: usize) -> Vec<f32> {
    let mut rng = Rng::new(seed as u64);
    (0..len)
        .map(|_| (rng.below(1031) as f32 - 515.0) / 431.0)
        .collect()
}

fn bytes(values: &[f32]) -> Vec<u8> {
    values.iter().copied().flat_map(f32::to_le_bytes).collect()
}

struct Fixture {
    q: Vec<f32>,
    relative: Vec<f32>,
    table: Vec<f32>,
    sinks: Vec<f32>,
    gates: Vec<f32>,
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
}

impl Fixture {
    fn new(steps: usize) -> Self {
        Self {
            q: pattern(QUERY, 1),
            relative: pattern(SHAPE.query_heads * SHAPE.relative_dims, 2),
            table: pattern(SHAPE.relative_dims * SHAPE.relative_extent, 3),
            sinks: pattern(SHAPE.query_heads, 4)
                .into_iter()
                .map(|s| s * 8.0)
                .collect(),
            gates: pattern(QUERY, 5).into_iter().map(|x| x * 4.0).collect(),
            keys: (0..steps).map(|p| pattern(KV, p * 2 + 6)).collect(),
            values: (0..steps).map(|p| pattern(KV, p * 2 + 7)).collect(),
        }
    }

    /// Newest-first FP64 reference with the unscaled relative logits and
    /// zero-value sink in the same positions as the CPU family.
    fn reference(&self, pos: usize) -> Vec<f32> {
        let mut out = vec![0.0; QUERY];
        let visible = pos.saturating_sub(SHAPE.capacity - 1)..=pos;
        let (rows, _) = out.as_chunks_mut::<HEAD_DIM>();
        for (head, row) in rows.iter_mut().enumerate() {
            let q = head * HEAD_DIM;
            let kv = head / (SHAPE.query_heads / SHAPE.kv_heads) * HEAD_DIM;
            let scores = self.scores(pos, head);
            let sink = f64::from(self.sinks[head]);
            let peak = scores.iter().copied().fold(sink, f64::max);
            let masses: Vec<_> = scores.iter().map(|score| (score - peak).exp()).collect();
            let normalizer = (sink - peak).exp() + masses.iter().sum::<f64>();
            for (d, result) in row.iter_mut().enumerate() {
                let value: f64 = visible
                    .clone()
                    .rev()
                    .zip(&masses)
                    .map(|(token, mass)| mass * f64::from(self.values[token][kv + d]))
                    .sum();
                let gate = 1.0 / (1.0 + (-f64::from(self.gates[q + d])).exp());
                *result = (value / normalizer * gate) as f32;
            }
        }
        out
    }

    fn scores(&self, pos: usize, head: usize) -> Vec<f64> {
        let q = head * HEAD_DIM;
        let rel = head * SHAPE.relative_dims;
        let kv = head / (SHAPE.query_heads / SHAPE.kv_heads) * HEAD_DIM;
        (0..=(pos.min(SHAPE.capacity - 1)))
            .map(|distance| {
                let qk: f64 = (0..HEAD_DIM)
                    .map(|d| {
                        f64::from(self.q[q + d]) * f64::from(self.keys[pos - distance][kv + d])
                    })
                    .sum();
                let mut bias = 0.0;
                if distance < SHAPE.relative_extent {
                    for c in 0..SHAPE.relative_dims {
                        bias += f64::from(self.relative[rel + c])
                            * f64::from(self.table[c * SHAPE.relative_extent + distance]);
                    }
                }
                qk / (HEAD_DIM as f64).sqrt() + bias
            })
            .collect()
    }
}

struct Buffers {
    q: DeviceBuffer,
    k: DeviceBuffer,
    v: DeviceBuffer,
    relative: DeviceBuffer,
    table: DeviceBuffer,
    sinks: DeviceBuffer,
    gates: DeviceBuffer,
    keys: DeviceBuffer,
    values: DeviceBuffer,
    pos: DeviceBuffer,
    status: DeviceBuffer,
    out: DeviceBuffer,
}

impl Buffers {
    fn new(ctx: &CudaContext, fixture: &Fixture) -> Self {
        let allocate = |bytes| upload_filled(ctx, 0xff, bytes);
        Self {
            q: upload_f32(ctx, &fixture.q),
            k: allocate(KV * 4),
            v: allocate(KV * 4),
            relative: upload_f32(ctx, &fixture.relative),
            table: upload_f32(ctx, &fixture.table),
            sinks: upload_f32(ctx, &fixture.sinks),
            gates: upload_f32(ctx, &fixture.gates),
            keys: allocate(SHAPE.capacity * KV * 4),
            values: allocate(SHAPE.capacity * KV * 4),
            pos: allocate(4),
            status: upload_filled(ctx, 0, 4),
            out: allocate(QUERY * 4),
        }
    }

    fn pins(&self) -> [&DeviceBuffer; RANGES] {
        [
            &self.q,
            &self.k,
            &self.v,
            &self.relative,
            &self.table,
            &self.sinks,
            &self.gates,
            &self.keys,
            &self.values,
            &self.pos,
            &self.status,
            &self.out,
        ]
    }

    fn call(&self) -> RelativeAttentionCall<'_> {
        RelativeAttentionCall {
            shape: SHAPE,
            q: whole(&self.q),
            k: whole(&self.k),
            v: whole(&self.v),
            relative: whole(&self.relative),
            table: whole(&self.table),
            sinks: whole(&self.sinks),
            gates: whole(&self.gates),
            k_cache: whole(&self.keys),
            v_cache: whole(&self.values),
            pos: whole(&self.pos),
            status: whole(&self.status),
            out: whole(&self.out),
        }
    }

    fn set_position(&self, stream: &DeviceStream, fixture: &Fixture, pos: usize) {
        self.k
            .write_at(stream, 0, &bytes(&fixture.keys[pos]))
            .unwrap();
        self.v
            .write_at(stream, 0, &bytes(&fixture.values[pos]))
            .unwrap();
        self.pos
            .write_at(stream, 0, &(pos as i32).to_le_bytes())
            .unwrap();
    }
}

fn whole(buf: &DeviceBuffer) -> DeviceRange<'_> {
    buf.range(0, buf.bytes()).expect("whole device buffer")
}

fn compare(actual: &[u8], expected: &[f32], pos: usize) {
    let actual = floats(actual);
    assert_eq!(actual.len(), expected.len());
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() <= 2e-5,
            "position {pos}, element {i}: expected {e}, got {a}"
        );
    }
}

fn assert_refuses(ctx: &CudaContext, stream: &DeviceStream, buf: &Buffers) {
    let mut call = buf.call();
    call.k = buf.keys.range(0, KV * 4).unwrap();
    let Err(EngError::BadRequest(message)) = ctx.relative_attention(stream, call) else {
        panic!("key/cache overlap must be refused");
    };
    assert!(message.contains("overlap"), "{message}");

    let mut call = buf.call();
    call.relative = buf.relative.range(0, buf.relative.bytes() - 4).unwrap();
    let Err(EngError::BadRequest(message)) = ctx.relative_attention(stream, call) else {
        panic!("short relative coefficients must be refused");
    };
    assert!(message.contains("needs"), "{message}");
}

#[test]
fn geometry_rejects_invalid_shapes() {
    assert!(SHAPE.region_bytes().is_some());
    for shape in [
        RelativeAttentionShape {
            kv_heads: 0,
            ..SHAPE
        },
        RelativeAttentionShape {
            query_heads: 23,
            ..SHAPE
        },
        RelativeAttentionShape {
            relative_dims: 33,
            ..SHAPE
        },
        RelativeAttentionShape {
            relative_extent: 0,
            ..SHAPE
        },
        RelativeAttentionShape {
            capacity: usize::MAX,
            ..SHAPE
        },
    ] {
        assert!(shape.region_bytes().is_none());
    }
}

#[test]
fn rolling_relative_attention_capture_matches_cpu() {
    if !device_present("rolling relative attention") {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let steps = SHAPE.capacity * 2 + 3;
    let fixture = Fixture::new(steps);
    let eager = Buffers::new(&ctx, &fixture);
    let replay = Buffers::new(&ctx, &fixture);
    assert_refuses(&ctx, &stream, &eager);

    eager.set_position(&stream, &fixture, 0);
    ctx.relative_attention(&stream, eager.call()).unwrap();
    stream.synchronize().unwrap();
    let pins = replay.pins();
    let (graph, ()) = ctx
        .capture(
            &stream,
            CapturePlan {
                pins: &pins,
                holds: Vec::new(),
            },
            || ctx.relative_attention(&stream, replay.call()),
        )
        .unwrap();
    assert_eq!(graph.nodes(), 1);

    for pos in 0..steps {
        eager.set_position(&stream, &fixture, pos);
        replay.set_position(&stream, &fixture, pos);
        ctx.relative_attention(&stream, eager.call()).unwrap();
        graph.launch(&stream).unwrap();
        stream.synchronize().unwrap();
        let captured = read_bytes(&ctx, &replay.out, QUERY * 4);
        let direct = read_bytes(&ctx, &eager.out, QUERY * 4);
        assert_eq!(captured, direct, "position {pos}");
        compare(&captured, &fixture.reference(pos), pos);
    }

    assert_ring(&ctx, &replay.keys, &fixture.keys);
    assert_ring(&ctx, &replay.values, &fixture.values);
    assert_invalid_positions(&ctx, &stream, &replay, || graph.launch(&stream).unwrap());
}

fn assert_ring(ctx: &CudaContext, ring: &DeviceBuffer, rows: &[Vec<f32>]) {
    let all = read_bytes(ctx, ring, ring.bytes());
    for (pos, row) in rows.iter().enumerate().skip(rows.len() - SHAPE.capacity) {
        let start = pos % SHAPE.capacity * KV * 4;
        assert_eq!(&all[start..start + KV * 4], bytes(row), "ring row {pos}");
    }
}

fn assert_invalid_positions(
    ctx: &CudaContext,
    stream: &DeviceStream,
    buf: &Buffers,
    launch: impl Fn(),
) {
    let snapshots = [&buf.keys, &buf.values, &buf.out].map(|b| read_bytes(ctx, b, b.bytes()));
    for invalid in [-1, POS_MAX + 1] {
        buf.status
            .write_at(stream, 0, &1_i32.to_le_bytes())
            .unwrap();
        buf.pos.write_at(stream, 0, &invalid.to_le_bytes()).unwrap();
        launch();
        stream.synchronize().unwrap();
        let status = read_bytes(ctx, &buf.status, 4).try_into().unwrap();
        assert_eq!(i32::from_le_bytes(status), 1 | STATUS_INVALID_STEP_STATE);
        for (b, snapshot) in [&buf.keys, &buf.values, &buf.out]
            .into_iter()
            .zip(&snapshots)
        {
            assert_eq!(&read_bytes(ctx, b, b.bytes()), snapshot);
        }
    }
}
