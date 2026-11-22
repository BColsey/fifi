//! The sliding ring against the linear plane at the kernel level. A layer with window W keeps
//! `W + 15` cells with cell = position % ring; the linear plane keeps every position. Both are
//! built by the same append kernels from the same rows, and the decode attention reads the same
//! keys in the same order from either, so the outputs must be BIT IDENTICAL at 4K and 32K
//! positions, for the single-pass and the split path, at one query row and at sixteen, and after
//! decode appends that wrap the ring.

use super::*;
use crate::kernels::cuda::resident::kv8_append::{Kv8CodeScale, Kv8Data, Kv8Shape};
use crate::kernels::cuda::{CudaContext, DeviceBuffer, DeviceStream, testkit};

const ROWS: usize = 16;
const CHUNK: usize = 512;
const MAX_SPLITS: usize = 16;
const MIN_KEYS: usize = 64;

struct Geometry {
    name: &'static str,
    n_q_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    window: usize,
}

const GEOMETRIES: [Geometry; 2] = [
    Geometry {
        name: "bolt local",
        n_q_head: 8,
        n_kv_head: 2,
        head_dim: 256,
        window: 512,
    },
    Geometry {
        name: "humanizer local",
        n_q_head: 16,
        n_kv_head: 8,
        head_dim: 256,
        window: 1024,
    },
];

/// The four planes of one cache.
struct Planes {
    k_codes: DeviceBuffer,
    k_scales: DeviceBuffer,
    v_codes: DeviceBuffer,
    v_scales: DeviceBuffer,
    cells: usize,
}

impl Planes {
    fn new(ctx: &CudaContext, g: &Geometry, cells: usize) -> Self {
        let codes = g.n_kv_head * cells * g.head_dim;
        let scales = g.n_kv_head * cells * (g.head_dim / 32) * 2;
        let zero = |bytes| testkit::upload_filled(ctx, 0, bytes);
        Self {
            k_codes: zero(codes),
            k_scales: zero(scales),
            v_codes: zero(codes),
            v_scales: zero(scales),
            cells,
        }
    }

    fn shape(&self, g: &Geometry, rows: usize, ring: usize) -> Kv8Shape {
        Kv8Shape {
            rows,
            n_kv_head: g.n_kv_head,
            head_dim: g.head_dim,
            cells: self.cells,
            code_scale: Kv8CodeScale::UnroundedF32,
            ring,
        }
    }

    fn data<'a>(&'a self, k: DeviceRange<'a>, v: DeviceRange<'a>) -> Kv8Data<'a> {
        let range = |buffer: &'a DeviceBuffer| buffer.range(0, buffer.bytes()).expect("plane");
        Kv8Data {
            k,
            v,
            k_codes: range(&self.k_codes),
            k_scales: range(&self.k_scales),
            v_codes: range(&self.v_codes),
            v_scales: range(&self.v_scales),
        }
    }
}

fn rows_f32(rng: &mut testkit::Rng, count: usize) -> Vec<f32> {
    (0..count).map(|_| rng.normal() as f32).collect()
}

/// How one attention launch reads a cache.
#[derive(Clone, Copy)]
struct Reading {
    /// The `ring` argument: 0 for the linear plane.
    ring: usize,
    window: usize,
    split: bool,
}

struct Rig {
    ctx: CudaContext,
    stream: DeviceStream,
}

impl Rig {
    /// Appends `rows` rows at `pos0` to the linear plane and, in sixteen-row launches, to the
    /// ring.
    fn append_chunk(
        &self,
        g: &Geometry,
        linear: &Planes,
        ring: &Planes,
        host_k: &[f32],
        host_v: &[f32],
        pos0: usize,
    ) {
        let rows = host_k.len() / (g.n_kv_head * g.head_dim);
        let row_bytes = g.n_kv_head * g.head_dim * 4;
        let k = testkit::upload_f32(&self.ctx, host_k);
        let v = testkit::upload_f32(&self.ctx, host_v);
        self.ctx
            .kv8_chunk(
                &self.stream,
                linear.shape(g, rows, 0),
                linear.data(
                    k.range(0, rows * row_bytes).unwrap(),
                    v.range(0, rows * row_bytes).unwrap(),
                ),
                pos0,
            )
            .expect("linear chunk");
        for first in (0..rows).step_by(ROWS) {
            let n = ROWS.min(rows - first);
            self.ctx
                .kv8_chunk(
                    &self.stream,
                    ring.shape(g, n, ring.cells),
                    ring.data(
                        k.range(first * row_bytes, n * row_bytes).unwrap(),
                        v.range(first * row_bytes, n * row_bytes).unwrap(),
                    ),
                    pos0 + first,
                )
                .expect("ring chunk");
        }
        self.stream.synchronize().expect("append");
    }

    /// One decode append at the device position word, on both caches.
    fn append_one(
        &self,
        g: &Geometry,
        linear: &Planes,
        ring: &Planes,
        host_k: &[f32],
        host_v: &[f32],
        pos: usize,
    ) {
        let row_bytes = g.n_kv_head * g.head_dim * 4;
        let k = testkit::upload_f32(&self.ctx, host_k);
        let v = testkit::upload_f32(&self.ctx, host_v);
        let word = testkit::upload_bytes(&self.ctx, &(pos as i32).to_le_bytes());
        let status = testkit::upload_bytes(&self.ctx, &0u32.to_le_bytes());
        for (planes, ring_cells) in [(linear, 0), (ring, ring.cells)] {
            self.ctx
                .kv8_append(
                    &self.stream,
                    planes.shape(g, 1, ring_cells),
                    planes.data(
                        k.range(0, row_bytes).unwrap(),
                        v.range(0, row_bytes).unwrap(),
                    ),
                    word.range(0, 4).unwrap(),
                    status.range(0, 4).unwrap(),
                )
                .expect("decode append");
        }
        self.stream.synchronize().expect("decode append");
        assert_eq!(testkit::read_bytes(&self.ctx, &status, 4), [0, 0, 0, 0]);
    }

    /// The attention of `m` query rows over `planes`, `kv_len` keys for token 0.
    fn attend(
        &self,
        g: &Geometry,
        planes: &Planes,
        q: &DeviceBuffer,
        m: usize,
        kv_len: usize,
        reading: Reading,
    ) -> Vec<u8> {
        let Reading {
            ring,
            window,
            split,
        } = reading;
        let q_bytes = m * g.n_q_head * g.head_dim * 4;
        let out = testkit::upload_filled(&self.ctx, 0xff, q_bytes);
        let len = testkit::upload_bytes(&self.ctx, &(kv_len as i32).to_le_bytes());
        let status = testkit::upload_bytes(&self.ctx, &0u32.to_le_bytes());
        let part_bytes =
            attn_decode_gemma_part_floats(m, g.n_q_head, g.head_dim, MAX_SPLITS).unwrap() * 4;
        let part = testkit::upload_filled(&self.ctx, 0, part_bytes);
        fn plane(buffer: &DeviceBuffer) -> DeviceRange<'_> {
            buffer.range(0, buffer.bytes()).expect("plane")
        }
        self.ctx
            .attn_decode_gemma(
                &self.stream,
                AttnDecodeGemmaCall {
                    out: out.range(0, q_bytes).unwrap(),
                    status: Some(status.range(0, 4).unwrap()),
                    q: q.range(0, q_bytes).unwrap(),
                    k_codes: plane(&planes.k_codes),
                    k_scales: plane(&planes.k_scales),
                    v_codes: plane(&planes.v_codes),
                    v_scales: plane(&planes.v_scales),
                    kv_len: len.range(0, 4).unwrap(),
                    m,
                    n_q_head: g.n_q_head,
                    n_kv_head: g.n_kv_head,
                    head_dim: g.head_dim,
                    cells: planes.cells,
                    window,
                    ring,
                    scale: 1.0,
                    split: split.then(|| AttnDecodeGemmaSplit {
                        part: part.range(0, part_bytes).unwrap(),
                        max_splits: MAX_SPLITS,
                        min_keys: MIN_KEYS,
                    }),
                },
            )
            .expect("attention");
        self.stream.synchronize().expect("attention");
        assert_eq!(
            testkit::read_bytes(&self.ctx, &status, 4),
            [0, 0, 0, 0],
            "status"
        );
        testkit::read_bytes(&self.ctx, &out, q_bytes)
    }
}

/// Builds a linear plane and a ring from the same `positions` rows, then compares the two.
fn ring_matches_linear(g: &Geometry, positions: usize) {
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let rig = Rig { ctx, stream };
    let decode_steps = 20usize;
    let linear = Planes::new(&rig.ctx, g, positions + decode_steps + ROWS);
    let ring_cells = g.window + ROWS - 1;
    let ring = Planes::new(&rig.ctx, g, ring_cells);
    let mut rng = testkit::Rng::new(0x5eed_0000 + positions as u64 + g.window as u64);
    let row = g.n_kv_head * g.head_dim;
    for pos0 in (0..positions).step_by(CHUNK) {
        let rows = CHUNK.min(positions - pos0);
        let (k, v) = (
            rows_f32(&mut rng, rows * row),
            rows_f32(&mut rng, rows * row),
        );
        rig.append_chunk(g, &linear, &ring, &k, &v, pos0);
    }
    for m in [ROWS, 1] {
        let q = testkit::upload_f32(&rig.ctx, &rows_f32(&mut rng, m * g.n_q_head * g.head_dim));
        // Token 0 of the last step sees `positions - m + 1` keys: the rows are already appended.
        let kv_len = positions - m + 1;
        for split in [false, true] {
            let linear_read = Reading {
                ring: 0,
                window: g.window,
                split,
            };
            let ring_read = Reading {
                ring: ring_cells,
                ..linear_read
            };
            let want = rig.attend(g, &linear, &q, m, kv_len, linear_read);
            let got = rig.attend(g, &ring, &q, m, kv_len, ring_read);
            assert_eq!(
                got, want,
                "{} {positions} positions m={m} split={split}",
                g.name
            );
            let all_keys = rig.attend(
                g,
                &linear,
                &q,
                m,
                kv_len,
                Reading {
                    window: 0,
                    ..linear_read
                },
            );
            assert_ne!(
                all_keys, want,
                "{} the window must cut keys at {positions} positions",
                g.name
            );
            assert!(testkit::floats(&got).iter().all(|x| x.is_finite()));
        }
    }
    // Decode appends past the last chunk wrap the ring; each step must still agree.
    for step in 0..decode_steps {
        let pos = positions + step;
        let (k, v) = (rows_f32(&mut rng, row), rows_f32(&mut rng, row));
        rig.append_one(g, &linear, &ring, &k, &v, pos);
        let q = testkit::upload_f32(&rig.ctx, &rows_f32(&mut rng, g.n_q_head * g.head_dim));
        for split in [false, true] {
            let linear_read = Reading {
                ring: 0,
                window: g.window,
                split,
            };
            let ring_read = Reading {
                ring: ring_cells,
                ..linear_read
            };
            let want = rig.attend(g, &linear, &q, 1, pos + 1, linear_read);
            let got = rig.attend(g, &ring, &q, 1, pos + 1, ring_read);
            assert_eq!(got, want, "{} decode step {step} split={split}", g.name);
        }
    }
}

#[test]
fn the_ring_is_bit_identical_to_the_linear_plane_at_4k() {
    if !testkit::device_present("attn_decode_gemma ring 4K") {
        return;
    }
    for g in &GEOMETRIES {
        ring_matches_linear(g, 4_096);
    }
}

#[test]
fn the_ring_is_bit_identical_to_the_linear_plane_at_32k() {
    if !testkit::device_present("attn_decode_gemma ring 32K") {
        return;
    }
    for g in &GEOMETRIES {
        ring_matches_linear(g, 32_768);
    }
}

#[test]
fn the_ring_rule_is_window_plus_rows_minus_one_up_to_the_cells() {
    use crate::kernels::cuda::native::resident::attn_decode_gemma::ring_admitted;
    assert!(ring_admitted(0, 0, 16, 100));
    assert!(ring_admitted(0, 512, 16, 100));
    assert!(ring_admitted(527, 512, 16, 527));
    assert!(ring_admitted(528, 512, 16, 1000));
    assert!(!ring_admitted(526, 512, 16, 1000), "below window + m - 1");
    assert!(!ring_admitted(1001, 512, 16, 1000), "above the cells");
    assert!(!ring_admitted(600, 0, 1, 1000), "a ring needs a window");
}
