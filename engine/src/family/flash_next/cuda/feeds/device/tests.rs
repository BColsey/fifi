//! Device tests (3090) of the PLE projections and the token-start sink. They skip when no device
//! is present. Run with `--test-threads=1` under a GPU lease.
//!
//! - The projections at the real shapes (`key_proj` 10,240 x 2,560, `value_proj` 2,560 x 2,560, F16)
//!   against an f64 evaluation of the same stored half-precision values, within twice a first-order
//!   f32 rounding bound per output (a lane adds `cols / 32` terms, then a butterfly).
//! - A second `project` with another embedding must not return the first one's results.
//! - The start sink writes the `StepParams` block, the stream and the PLE embedding to their
//!   offsets and nothing around them; a negative token or position is refused with nothing written.

use super::{DevicePle, DevicePleRoute, DeviceStart, f32_bytes, f32_values};
use crate::{
    EngError,
    family::flash_next::cuda::{
        feeds::{StartShape, write_start},
        ple_feed::{HostMatrix, HostProjection, PleKeyValue, PleProjection},
    },
    kernels::cuda::{CudaContext, device_present},
};

const WIDTH: usize = 10_240;
const HIDDEN: usize = 2_560;
const EMBED: usize = 2_560;
const U: f64 = 1.0 / 16_777_216.0;

/// Deterministic values in about [-scale, scale].
fn values(count: usize, seed: usize, scale: f32) -> Vec<f32> {
    (0..count)
        .map(|i| (((i * 7919 + seed * 613) % 2003) as f32 - 1001.0) / 1001.0 * scale)
        .collect()
}

fn halves(values: &[f32]) -> Vec<u16> {
    values
        .iter()
        .map(|&v| half::f16::from_f32(v).to_bits())
        .collect()
}

fn half_bytes(codes: &[u16]) -> Vec<u8> {
    codes.iter().flat_map(|c| c.to_le_bytes()).collect()
}

/// `(value, bound)` of every row of `w x` in f64: the bound is `2 * u * (cols/32 + 6) * sum |w x|`.
fn reference(codes: &[u16], rows: usize, x: &[f32]) -> Vec<(f64, f64)> {
    (0..rows)
        .map(|r| {
            let (mut sum, mut mag) = (0.0f64, 0.0f64);
            for (c, &xv) in x.iter().enumerate() {
                let term = half::f16::from_bits(codes[r * x.len() + c]).to_f64() * f64::from(xv);
                sum += term;
                mag += term.abs();
            }
            (sum, 2.0 * U * (x.len() as f64 / 32.0 + 6.0) * mag + 1e-30)
        })
        .collect()
}

fn check(name: &str, got: &[f32], want: &[(f64, f64)]) {
    assert_eq!(got.len(), want.len(), "{name} length");
    for (row, (g, (w, bound))) in got.iter().zip(want).enumerate() {
        assert!(
            (f64::from(*g) - w).abs() <= *bound,
            "{name} row {row}: device {g}, f64 {w}, bound {bound}"
        );
    }
}

#[test]
fn the_projections_match_an_f64_evaluation_of_the_stored_half_values() {
    if !device_present("PLE projections against f64") {
        return;
    }
    let context = CudaContext::new().expect("context");
    let stream = context.stream().expect("stream");
    let key = halves(&values(WIDTH * EMBED, 1, 0.05));
    let value = halves(&values(HIDDEN * EMBED, 2, 0.05));
    let ple = DevicePle::upload(
        &context,
        &half_bytes(&key),
        &half_bytes(&value),
        WIDTH,
        HIDDEN,
        EMBED,
    )
    .unwrap();
    assert_eq!(
        ple.device_bytes(),
        (WIDTH + HIDDEN) * EMBED * 2 + (EMBED + WIDTH + HIDDEN) * 4
    );
    let mut previous: Option<PleKeyValue> = None;
    for seed in [3usize, 4] {
        let x = values(EMBED, seed, 1.5);
        let got = ple.project(&context, &stream, &x).unwrap();
        check("keys", &got.keys, &reference(&key, WIDTH, &x));
        check("value", &got.value, &reference(&value, HIDDEN, &x));
        if let Some(before) = &previous {
            assert_ne!(
                before.keys, got.keys,
                "the second embedding returned the first's keys"
            );
        }
        previous = Some(got);
    }
    // Queue first, collect later: the results are those of the last enqueue.
    let x = values(EMBED, 5, 1.0);
    ple.enqueue(&context, &stream, &x).unwrap();
    let got = ple.collect(&stream).unwrap();
    check(
        "keys (enqueue/collect)",
        &got.keys,
        &reference(&key, WIDTH, &x),
    );
}

#[test]
fn the_projections_refuse_a_wrong_size_before_any_launch() {
    if !device_present("PLE projections refuse a wrong size") {
        return;
    }
    let context = CudaContext::new().expect("context");
    let stream = context.stream().expect("stream");
    assert!(DevicePle::upload(&context, &[0; 6], &[0; 8], 2, 2, 2).is_err());
    let ple = DevicePle::upload(&context, &[0; 8], &[0; 8], 2, 2, 2).unwrap();
    assert!(matches!(
        ple.enqueue(&context, &stream, &[0.0; 3]),
        Err(EngError::BadRequest(_))
    ));
}

#[test]
fn the_start_sink_writes_the_block_the_stream_and_the_embedding_at_their_offsets() {
    if !device_present("start sink offsets") {
        return;
    }
    let context = CudaContext::new().expect("context");
    let stream = context.stream().expect("stream");
    let words = context.allocate(1024).unwrap();
    let hyper = context.allocate(4096).unwrap();
    let ple = context.allocate(1024).unwrap();
    for buffer in [&words, &hyper, &ple] {
        let len = buffer_len(buffer);
        context.upload_at(buffer, 0, &vec![0xAB; len]).unwrap();
    }
    let shape = StartShape {
        vocab: 248_320,
        stream: 16,
        ple_embed: 8,
    };
    let stream_values = values(16, 7, 1.0);
    let embedding = values(8, 8, 1.0);
    let mut sink = DeviceStart {
        stream: &stream,
        words: &words,
        params_offset: 256,
        hyper: &hyper,
        hyper_offset: 64,
        ple: &ple,
        ple_offset: 32,
    };
    write_start(&mut sink, &shape, 760, 12, &stream_values, &embedding).unwrap();

    let read = |buffer: &crate::kernels::cuda::DeviceBuffer, offset: usize, len: usize| {
        let mut all = vec![0u8; buffer_len(buffer)];
        buffer.read_at(&stream, 0, &mut all).unwrap();
        (all[offset..offset + len].to_vec(), all)
    };
    let (block, all) = read(&words, 256, 32);
    assert_eq!(i32::from_le_bytes(block[0..4].try_into().unwrap()), 760);
    assert_eq!(i32::from_le_bytes(block[4..8].try_into().unwrap()), 12);
    assert_eq!(
        i32::from_le_bytes(block[8..12].try_into().unwrap()),
        13,
        "kv_len"
    );
    assert!(all[..256].iter().all(|b| *b == 0xAB) && all[288..].iter().all(|b| *b == 0xAB));
    let (got, all) = read(&hyper, 64, 64);
    assert_eq!(f32_values(&got), stream_values);
    assert!(all[..64].iter().all(|b| *b == 0xAB) && all[128..].iter().all(|b| *b == 0xAB));
    let (got, all) = read(&ple, 32, 32);
    assert_eq!(got, f32_bytes(&embedding));
    assert!(all[..32].iter().all(|b| *b == 0xAB) && all[64..].iter().all(|b| *b == 0xAB));

    // A position past `POS_MAX` passes the host checks and is refused by `StepParams`: the stream
    // and the embedding are not written (the params write is first).
    context.upload_at(&hyper, 64, &[0xCD; 64]).unwrap();
    assert!(
        write_start(
            &mut sink,
            &shape,
            1,
            1usize << 30,
            &stream_values,
            &embedding
        )
        .is_err()
    );
    let (got, _) = read(&hyper, 64, 64);
    assert!(got.iter().all(|b| *b == 0xCD));
}

fn buffer_len(buffer: &crate::kernels::cuda::DeviceBuffer) -> usize {
    buffer.bytes()
}

#[test]
fn the_route_agrees_with_the_host_projection_of_the_same_stored_values() {
    if !device_present("PLE route against the host projection") {
        return;
    }
    let context = std::rc::Rc::new(CudaContext::new().expect("context"));
    let stream = std::rc::Rc::new(context.stream().expect("stream"));
    let key = halves(&values(WIDTH * EMBED, 1, 0.05));
    let value = halves(&values(HIDDEN * EMBED, 2, 0.05));
    let widen = |codes: &[u16]| -> Vec<f32> {
        codes
            .iter()
            .map(|&c| half::f16::from_bits(c).to_f32())
            .collect()
    };
    let mut host = HostProjection::new(
        HostMatrix::new(WIDTH, EMBED, widen(&key)).unwrap(),
        HostMatrix::new(HIDDEN, EMBED, widen(&value)).unwrap(),
        4,
    );
    let mut route = DevicePleRoute::new(
        DevicePle::upload(
            &context,
            &half_bytes(&key),
            &half_bytes(&value),
            WIDTH,
            HIDDEN,
            EMBED,
        )
        .unwrap(),
        context,
        stream,
    );
    let x = values(EMBED, 9, 1.0);
    let (device, reference_host) = (route.project(&x).unwrap(), host.project(&x).unwrap());
    // Both are f32 sums of the same products in different orders: each differs from the f64 value
    // by at most the bound, so from each other by at most twice it.
    for (name, got, want, codes, rows) in [
        ("keys", &device.keys, &reference_host.keys, &key, WIDTH),
        (
            "value",
            &device.value,
            &reference_host.value,
            &value,
            HIDDEN,
        ),
    ] {
        for (row, (g, (h, (_, bound)))) in got
            .iter()
            .zip(want.iter().zip(reference(codes, rows, &x)))
            .enumerate()
        {
            assert!(
                (f64::from(*g) - f64::from(*h)).abs() <= 2.0 * bound,
                "{name} row {row}: device {g}, host {h}, bound {bound}"
            );
        }
    }
}
