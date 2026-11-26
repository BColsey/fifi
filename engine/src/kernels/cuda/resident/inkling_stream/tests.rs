use std::mem::size_of;

use crate::kernels::cuda::{CudaContext, DeviceBuffer, DeviceStream, device_present, testkit::Rng};

use super::{AddCall, MAX_BLOCKS, MixCall, MixShape, NormCall, SealCall, StreamShape};

const EPS: f32 = 1e-5;

fn bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_ne_bytes()).collect()
}

fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_ne_bytes(*b))
        .collect()
}

fn row_bytes(width: usize) -> usize {
    width * size_of::<f32>()
}

fn upload(buffer: &DeviceBuffer, stream: &DeviceStream, values: &[f32]) {
    buffer.write_at(stream, 0, &bytes(values)).unwrap();
}

fn download(buffer: &DeviceBuffer, stream: &DeviceStream, offset: usize, words: usize) -> Vec<f32> {
    let mut out = vec![0u8; words * size_of::<f32>()];
    buffer.read_at(stream, offset, &mut out).unwrap();
    floats(&out)
}

fn normal_row(rng: &mut Rng, width: usize) -> Vec<f32> {
    (0..width).map(|_| rng.normal() as f32).collect()
}

/// The host algorithm of `cpu_forward/depth_mix.rs`, in f32 and in its order.
mod host {
    fn rms_norm(row: &mut [f32], eps: f32) {
        let dot: f32 = row.iter().map(|v| v * v).sum();
        let scale = (dot / row.len() as f32 + eps).sqrt().recip();
        for value in row {
            *value *= scale;
        }
    }

    /// `Stream::seal(next)`: returns the sealed row, leaves `next` as the current row.
    pub fn seal(current: &mut Vec<f32>, next: Vec<f32>, eps: f32) -> Vec<f32> {
        let mut previous = std::mem::replace(current, next);
        rms_norm(&mut previous, eps);
        previous
    }

    /// `rms_norm_with_scale` of the host: the normalized copy of `src`.
    pub fn norm(src: &[f32], eps: f32) -> Vec<f32> {
        let mut row = src.to_vec();
        rms_norm(&mut row, eps);
        row
    }

    /// `Stream::mix(first_block, query)` of a state with normalized sealed rows.
    pub fn mix(
        sealed: &[Vec<f32>],
        current: &[f32],
        first_block: usize,
        query: &[f32],
        eps: f32,
    ) -> Vec<f32> {
        let mut normalized = current.to_vec();
        rms_norm(&mut normalized, eps);
        let rows: Vec<&[f32]> = sealed[first_block..]
            .iter()
            .rev()
            .map(|row| row.as_slice())
            .chain([normalized.as_slice()])
            .collect();
        let scale = (query.len() as f32).sqrt().recip();
        let scores: Vec<f32> = rows
            .iter()
            .map(|row| row.iter().zip(query).map(|(a, b)| a * b).sum::<f32>() * scale)
            .collect();
        let peak = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut mass = 0.0f32;
        let mut output = vec![0.0f32; query.len()];
        for (score, row) in scores.into_iter().zip(&rows) {
            let weight = (score - peak).exp();
            mass += weight;
            for (out, value) in output.iter_mut().zip(row.iter()) {
                *out += weight * value;
            }
        }
        for value in &mut output {
            *value /= mass;
        }
        output
    }
}

/// One stream state of `MAX_BLOCKS + 2` rows, a spare row, a query, an output and a health word.
struct Buffers {
    state: DeviceBuffer,
    row: DeviceBuffer,
    query: DeviceBuffer,
    out: DeviceBuffer,
    health: DeviceBuffer,
}

impl Buffers {
    fn new(ctx: &CudaContext, width: usize) -> Self {
        let row = row_bytes(width);
        Self {
            state: ctx.allocate(row * (MAX_BLOCKS + 2)).unwrap(),
            row: ctx.allocate(row).unwrap(),
            query: ctx.allocate(row).unwrap(),
            out: ctx.allocate(row).unwrap(),
            health: ctx.allocate(size_of::<f32>()).unwrap(),
        }
    }

    fn clear_health(&self, stream: &DeviceStream) {
        upload(&self.health, stream, &[0.0]);
    }

    fn health(&self, stream: &DeviceStream) -> f32 {
        download(&self.health, stream, 0, 1)[0]
    }

    /// Uploads `state` (the current row, then the sealed rows) and `query`, mixes, reads `out`.
    #[allow(clippy::too_many_arguments, reason = "one test fixture")]
    fn mix(
        &self,
        ctx: &CudaContext,
        stream: &DeviceStream,
        state: &[f32],
        query: &[f32],
        first_block: usize,
        blocks: usize,
        width: usize,
    ) -> Result<Vec<f32>, String> {
        upload(&self.state, stream, state);
        upload(&self.query, stream, query);
        self.clear_health(stream);
        ctx.inkling_stream_mix(
            stream,
            MixCall {
                state: self
                    .state
                    .range(0, row_bytes(width) * (blocks + 1))
                    .unwrap(),
                query: self.query.range(0, row_bytes(width)).unwrap(),
                out: self.out.range(0, row_bytes(width)).unwrap(),
                health: self.health.range(0, size_of::<f32>()).unwrap(),
            },
            MixShape {
                first_block,
                blocks,
                width,
                eps: EPS,
            },
        )
        .map_err(|error| format!("{error:?}"))?;
        Ok(download(&self.out, stream, 0, width))
    }
}

fn max_error(out: &[f32], expected: &[f32]) -> f32 {
    out.iter()
        .zip(expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max)
}

#[test]
fn mix_matches_the_host_algorithm_within_rounding() {
    if !device_present("Inkling stream mix") {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let mut rng = Rng::new(0x5eed_572e);
    for width in [1usize, 33, 2304] {
        let buffers = Buffers::new(&ctx, width);
        let query = normal_row(&mut rng, width);
        let mut current = normal_row(&mut rng, width);
        let mut sealed: Vec<Vec<f32>> = Vec::new();
        for blocks in 0..=12usize {
            for first_block in [0, blocks / 2, blocks] {
                let mut state = current.clone();
                for row in &sealed {
                    state.extend_from_slice(row);
                }
                let out = buffers
                    .mix(&ctx, &stream, &state, &query, first_block, blocks, width)
                    .unwrap();
                let expected = host::mix(&sealed, &current, first_block, &query, EPS);
                let error = max_error(&out, &expected);
                assert!(
                    error <= 1e-4,
                    "width {width} blocks {blocks} first_block {first_block}: max error {error}"
                );
                let health = buffers.health(&stream);
                assert_eq!(health.to_bits(), 0, "width {width} blocks {blocks}");
            }
            let next = normal_row(&mut rng, width);
            sealed.push(host::seal(&mut current, next, EPS));
        }
    }
}

#[test]
fn seal_stores_the_normalized_row_and_add_is_bit_exact() {
    if !device_present("Inkling stream seal and add") {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let mut rng = Rng::new(0x5eed_572f);
    let width = 2304;
    let row = row_bytes(width);
    let buffers = Buffers::new(&ctx, width);
    let mut current = normal_row(&mut rng, width);
    let mut sealed: Vec<Vec<f32>> = Vec::new();
    for blocks in 0..12usize {
        let next = normal_row(&mut rng, width);
        let mut state = current.clone();
        for row in &sealed {
            state.extend_from_slice(row);
        }
        upload(&buffers.state, &stream, &state);
        upload(&buffers.row, &stream, &next);
        buffers.clear_health(&stream);
        ctx.inkling_stream_seal(
            &stream,
            SealCall {
                state: buffers.state.range(0, row * (blocks + 2)).unwrap(),
                next: buffers.row.range(0, row).unwrap(),
                health: buffers.health.range(0, size_of::<f32>()).unwrap(),
            },
            StreamShape {
                blocks,
                width,
                eps: EPS,
            },
        )
        .unwrap();
        let expected = host::seal(&mut current, next.clone(), EPS);
        let stored = download(&buffers.state, &stream, row * (blocks + 1), width);
        for (a, b) in stored.iter().zip(&expected) {
            assert!(
                (a - b).abs() <= 1e-6 * (1.0 + b.abs()),
                "blocks {blocks}: {a} vs {b}"
            );
        }
        assert_eq!(download(&buffers.state, &stream, 0, width), next);
        assert_eq!(buffers.health(&stream).to_bits(), 0, "blocks {blocks}");
        sealed.push(expected);
    }

    let addend = normal_row(&mut rng, width);
    upload(&buffers.state, &stream, &current);
    upload(&buffers.row, &stream, &addend);
    buffers.clear_health(&stream);
    ctx.inkling_stream_add(
        &stream,
        AddCall {
            state: buffers.state.range(0, row).unwrap(),
            row: buffers.row.range(0, row).unwrap(),
            health: buffers.health.range(0, size_of::<f32>()).unwrap(),
        },
        width,
    )
    .unwrap();
    let expected: Vec<f32> = current.iter().zip(&addend).map(|(a, b)| a + b).collect();
    assert_eq!(download(&buffers.state, &stream, 0, width), expected);
    assert_eq!(buffers.health(&stream).to_bits(), 0);
}

#[test]
fn non_finite_arithmetic_is_reported_in_the_health_word() {
    if !device_present("Inkling stream health") {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let width = 64;
    let row = row_bytes(width);
    let buffers = Buffers::new(&ctx, width);
    let query = normal_row(&mut Rng::new(7), width);
    let mut broken = normal_row(&mut Rng::new(8), width);
    broken[17] = f32::INFINITY;
    let out = buffers
        .mix(&ctx, &stream, &broken, &query, 0, 0, width)
        .unwrap();
    assert!(buffers.health(&stream).is_nan());
    assert!(out.iter().any(|value| value.is_nan()));

    let mut current = normal_row(&mut Rng::new(9), width);
    let mut next = normal_row(&mut Rng::new(10), width);
    next[3] = f32::NAN;
    upload(&buffers.state, &stream, &current);
    upload(&buffers.row, &stream, &next);
    buffers.clear_health(&stream);
    ctx.inkling_stream_seal(
        &stream,
        SealCall {
            state: buffers.state.range(0, row * 2).unwrap(),
            next: buffers.row.range(0, row).unwrap(),
            health: buffers.health.range(0, size_of::<f32>()).unwrap(),
        },
        StreamShape {
            blocks: 0,
            width,
            eps: EPS,
        },
    )
    .unwrap();
    assert!(buffers.health(&stream).is_nan());

    current[5] = f32::NEG_INFINITY;
    let addend = vec![f32::INFINITY; width];
    upload(&buffers.state, &stream, &current);
    upload(&buffers.row, &stream, &addend);
    buffers.clear_health(&stream);
    ctx.inkling_stream_add(
        &stream,
        AddCall {
            state: buffers.state.range(0, row).unwrap(),
            row: buffers.row.range(0, row).unwrap(),
            health: buffers.health.range(0, size_of::<f32>()).unwrap(),
        },
        width,
    )
    .unwrap();
    assert!(buffers.health(&stream).is_nan());
}

#[test]
fn wrapper_rejects_bad_shapes_and_shared_bytes_before_launching() {
    if !device_present("Inkling stream rejections") {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let width = 64;
    let row = row_bytes(width);
    let word = size_of::<f32>();
    let buffers = Buffers::new(&ctx, width);
    let shape = |first_block, blocks, eps| MixShape {
        first_block,
        blocks,
        width,
        eps,
    };
    // `mix` names the error of one launch; `state` is an offset and a byte count.
    let mix = |state: (usize, usize), out: usize, health: usize, shape| {
        ctx.inkling_stream_mix(
            &stream,
            MixCall {
                state: buffers.state.range(state.0, state.1).unwrap(),
                query: buffers.query.range(0, row).unwrap(),
                out: buffers.out.range(out, row).unwrap(),
                health: buffers.health.range(health, word).unwrap(),
            },
            shape,
        )
        .map_err(|error| format!("{error:?}"))
    };
    let takes = "the CUDA Inkling stream mix takes";
    for bad in [
        shape(1, 0, EPS),
        shape(0, MAX_BLOCKS + 1, EPS),
        shape(0, 0, 0.0),
        shape(0, 0, f32::NAN),
        MixShape {
            first_block: 0,
            blocks: 0,
            width: 0,
            eps: EPS,
        },
    ] {
        let error = mix((0, row * 2), 0, 0, bad).unwrap_err();
        assert!(error.contains(takes), "{bad:?}: {error}");
    }
    let error = mix((0, row), 0, 0, shape(0, 1, EPS)).unwrap_err();
    assert!(
        error
            .contains("the CUDA Inkling stream state range holds 256 bytes but the call needs 512"),
        "{error}"
    );
    let error = mix((0, row * 2), 0, 0, shape(0, 0, EPS)).unwrap_err();
    assert!(
        error.contains("holds 512 bytes but the call needs exactly 256"),
        "{error}"
    );

    // The output inside the two state rows the launch reads.
    let error = ctx
        .inkling_stream_mix(
            &stream,
            MixCall {
                state: buffers.state.range(0, row * 2).unwrap(),
                query: buffers.query.range(0, row).unwrap(),
                out: buffers.state.range(row, row).unwrap(),
                health: buffers.health.range(0, word).unwrap(),
            },
            shape(0, 1, EPS),
        )
        .map_err(|error| format!("{error:?}"))
        .unwrap_err();
    assert!(
        error
            .contains("the CUDA Inkling stream mix output and Inkling stream state ranges overlap"),
        "{error}"
    );
    // The health word inside the output.
    let error = ctx
        .inkling_stream_mix(
            &stream,
            MixCall {
                state: buffers.state.range(0, row * 2).unwrap(),
                query: buffers.query.range(0, row).unwrap(),
                out: buffers.out.range(0, row).unwrap(),
                health: buffers.out.range(row - word, word).unwrap(),
            },
            shape(0, 1, EPS),
        )
        .map_err(|error| format!("{error:?}"))
        .unwrap_err();
    assert!(
        error.contains(
            "the CUDA Inkling stream mix output and Inkling stream health ranges overlap"
        ),
        "{error}"
    );
    // The same allocation, different bytes: accepted, and the context still launches.
    ctx.inkling_stream_mix(
        &stream,
        MixCall {
            state: buffers.state.range(0, row * 2).unwrap(),
            query: buffers.query.range(0, row).unwrap(),
            out: buffers.state.range(row * 2, row).unwrap(),
            health: buffers.health.range(0, word).unwrap(),
        },
        shape(0, 1, EPS),
    )
    .unwrap();

    let error = ctx
        .inkling_stream_seal(
            &stream,
            SealCall {
                state: buffers.state.range(0, row * 2).unwrap(),
                next: buffers.state.range(row, row).unwrap(),
                health: buffers.health.range(0, word).unwrap(),
            },
            StreamShape {
                blocks: 0,
                width,
                eps: EPS,
            },
        )
        .map_err(|error| format!("{error:?}"))
        .unwrap_err();
    assert!(
        error.contains("the CUDA Inkling stream state and Inkling stream next row ranges overlap"),
        "{error}"
    );
    let error = ctx
        .inkling_stream_add(
            &stream,
            AddCall {
                state: buffers.state.range(0, row).unwrap(),
                row: buffers.state.range(0, row).unwrap(),
                health: buffers.health.range(0, word).unwrap(),
            },
            width,
        )
        .map_err(|error| format!("{error:?}"))
        .unwrap_err();
    assert!(
        error
            .contains("the CUDA Inkling stream state and Inkling stream addend row ranges overlap"),
        "{error}"
    );
    let error = ctx
        .inkling_stream_add(
            &stream,
            AddCall {
                state: buffers.state.range(0, row).unwrap(),
                row: buffers.row.range(0, row).unwrap(),
                health: buffers.health.range(0, word).unwrap(),
            },
            0,
        )
        .map_err(|error| format!("{error:?}"))
        .unwrap_err();
    assert!(
        error.contains("the CUDA Inkling stream add takes"),
        "{error}"
    );
}

#[test]
fn norm_matches_the_host_scale_and_refuses_to_normalize_in_place() {
    if !device_present("Inkling stream norm") {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let width = 2304;
    let row = row_bytes(width);
    let word = size_of::<f32>();
    let buffers = Buffers::new(&ctx, width);
    let src = normal_row(&mut Rng::new(0x5eed_5730), width);
    upload(&buffers.row, &stream, &src);
    buffers.clear_health(&stream);
    ctx.inkling_stream_norm(
        &stream,
        NormCall {
            src: buffers.row.range(0, row).unwrap(),
            dst: buffers.out.range(0, row).unwrap(),
            health: buffers.health.range(0, word).unwrap(),
        },
        (width, EPS),
    )
    .unwrap();
    let expected = host::norm(&src, EPS);
    for (a, b) in download(&buffers.out, &stream, 0, width)
        .iter()
        .zip(&expected)
    {
        assert!((a - b).abs() <= 1e-6 * (1.0 + b.abs()), "{a} vs {b}");
    }
    assert_eq!(buffers.health(&stream).to_bits(), 0);

    let error = ctx
        .inkling_stream_norm(
            &stream,
            NormCall {
                src: buffers.out.range(0, row).unwrap(),
                dst: buffers.out.range(0, row).unwrap(),
                health: buffers.health.range(0, word).unwrap(),
            },
            (width, EPS),
        )
        .map_err(|error| format!("{error:?}"))
        .unwrap_err();
    assert!(
        error.contains(
            "the CUDA Inkling stream norm output and Inkling stream norm source ranges overlap"
        ),
        "{error}"
    );
}
