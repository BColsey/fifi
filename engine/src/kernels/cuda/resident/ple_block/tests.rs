//! Tests of the PLE block launches against the host reference (`family::math` and the
//! dequantizer of `weights`). Device runs need a card; without one each test returns.

use super::*;
use crate::family::math::{dot, rms0, silu};
use crate::kernels::cuda::{CudaContext, DeviceBuffer, testkit};
use crate::weights::{GgufType, dequantize};

fn byte_range(buffer: &DeviceBuffer, offset: usize, bytes: usize) -> DeviceRange<'_> {
    buffer.range(offset, bytes).expect("range")
}

/// A uniform value in `[-scale, scale)`.
fn spread(rng: &mut testkit::Rng, scale: f32) -> f32 {
    ((rng.unit() * 2.0 - 1.0) * f64::from(scale)) as f32
}

/// `len` values of [`spread`].
fn spread_vec(rng: &mut testkit::Rng, len: usize, scale: f32) -> Vec<f32> {
    (0..len).map(|_| spread(rng, scale)).collect()
}

fn close(got: f32, want: f32) -> bool {
    (got - want).abs() <= 2e-4 * want.abs().max(1.0)
}

#[test]
fn the_rows_decode_like_the_host_dequantizer() {
    if !testkit::device_present("ple rows") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let (rows, head_dim) = (5usize, 128usize);
    let mut rng = testkit::Rng::new(7);
    let mut raw = Vec::new();
    for _ in 0..rows * head_dim / 32 {
        let d = 0.05 + spread(&mut rng, 0.02).abs();
        raw.extend_from_slice(&testkit::f16_bits(d).to_le_bytes());
        for _ in 0..16 {
            let low = (spread(&mut rng, 8.0) as i32 + 8) as u8;
            let high = (spread(&mut rng, 8.0) as i32 + 8) as u8;
            raw.push(low | (high << 4));
        }
    }
    let want: Vec<f32> = dequantize(GgufType::Iq4Nl, &raw)
        .expect("host decode")
        .iter()
        .map(|v| v * 0.5)
        .collect();
    let raw_dev = ctx.upload(&raw).expect("raw");
    let out_dev = testkit::upload_filled(&ctx, 0xff, rows * head_dim * 4);
    ctx.ple_rows(
        &stream,
        PleRowsCall {
            out: out_dev.range(0, rows * head_dim * 4).expect("out"),
            raw: raw_dev.range(0, raw.len()).expect("raw"),
            scale: 0.5,
            rows,
            head_dim,
        },
    )
    .expect("rows");
    stream.synchronize().expect("drain");
    let got = testkit::floats(&testkit::read_bytes(&ctx, &out_dev, rows * head_dim * 4));
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert!(close(*g, *w), "value {i}: {g} against {w}");
    }
}

#[test]
fn the_gate_follows_the_host_step() {
    if !testkit::device_present("ple gate") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let (branches, hidden, eps) = (4usize, 256usize, 1e-6f32);
    let width = branches * hidden;
    let mut rng = testkit::Rng::new(11);
    let keys = spread_vec(&mut rng, width, 2.0);
    let hyper = spread_vec(&mut rng, width, 3.0);
    let value = spread_vec(&mut rng, hidden, 1.5);
    let nk = spread_vec(&mut rng, width, 0.3);
    let nq = spread_vec(&mut rng, width, 0.3);
    let nc = spread_vec(&mut rng, width, 0.3);
    let (mut gated, mut normed) = (vec![0f32; width], vec![0f32; width]);
    for b in 0..branches {
        let (lo, hi) = (b * hidden, (b + 1) * hidden);
        let kn = rms0(&keys[lo..hi], &nk[lo..hi], eps).unwrap();
        let qn = rms0(&hyper[lo..hi], &nq[lo..hi], eps).unwrap();
        let d = f64::from(dot(&kn, &qn).unwrap() / (hidden as f32).sqrt());
        let shaped = d.abs().max(1e-6).sqrt().copysign(d);
        let g = (1.0 / (1.0 + (-shaped).exp())) as f32;
        for (out, v) in gated[lo..hi].iter_mut().zip(&value) {
            *out = g * v;
        }
        normed[lo..hi].copy_from_slice(&rms0(&gated[lo..hi], &nc[lo..hi], eps).unwrap());
    }
    let up = |v: &[f32]| testkit::upload_f32(&ctx, v);
    let (keys_d, hyper_d, value_d) = (up(&keys), up(&hyper), up(&value));
    let (nk_d, nq_d, nc_d) = (up(&nk), up(&nq), up(&nc));
    let gated_d = testkit::upload_filled(&ctx, 0xff, width * 4);
    let normed_d = testkit::upload_filled(&ctx, 0xff, width * 4);
    let width_bytes = width * 4;
    ctx.ple_gate(
        &stream,
        PleGateCall {
            keys: byte_range(&keys_d, 0, width_bytes),
            hyper: byte_range(&hyper_d, 0, width_bytes),
            value: byte_range(&value_d, 0, hidden * 4),
            norm_key: byte_range(&nk_d, 0, width_bytes),
            norm_query: byte_range(&nq_d, 0, width_bytes),
            norm_conv: byte_range(&nc_d, 0, width_bytes),
            gated: byte_range(&gated_d, 0, width_bytes),
            normed: byte_range(&normed_d, 0, width_bytes),
            eps,
            branches,
            hidden,
        },
    )
    .expect("gate");
    stream.synchronize().expect("drain");
    let got_gated = testkit::floats(&testkit::read_bytes(&ctx, &gated_d, width * 4));
    let got_normed = testkit::floats(&testkit::read_bytes(&ctx, &normed_d, width * 4));
    for i in 0..width {
        assert!(
            close(got_gated[i], gated[i]),
            "gated {i}: {} against {}",
            got_gated[i],
            gated[i]
        );
        assert!(
            close(got_normed[i], normed[i]),
            "normed {i}: {} against {}",
            got_normed[i],
            normed[i]
        );
    }
}

#[test]
fn the_convolution_and_its_ring_follow_the_host_over_three_tokens() {
    if !testkit::device_present("ple conv") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let (width, conv_kernel, ngram) = (96usize, 4usize, 3usize);
    let (taps, stride) = (conv_kernel - 1, (conv_kernel - 1) * ngram);
    let mut rng = testkit::Rng::new(23);
    let conv = spread_vec(&mut rng, width * conv_kernel, 0.7);
    let mut ring = spread_vec(&mut rng, width * stride, 1.0);
    let conv_d = testkit::upload_f32(&ctx, &conv);
    let ring_d = testkit::upload_f32(&ctx, &ring);
    let out_d = testkit::upload_filled(&ctx, 0xff, width * 4);
    for token in 0..3 {
        let gated = spread_vec(&mut rng, width, 1.0);
        let normed = spread_vec(&mut rng, width, 1.0);
        let mut want = vec![0f32; width];
        for c in 0..width {
            let kernel = &conv[c * conv_kernel..(c + 1) * conv_kernel];
            let slot = &mut ring[c * stride..(c + 1) * stride];
            let mut a = kernel[taps] * normed[c];
            for (k, w) in kernel[..taps].iter().enumerate() {
                a += w * slot[k * ngram];
            }
            want[c] = gated[c] + silu(a);
            slot.copy_within(1.., 0);
            slot[stride - 1] = normed[c];
        }
        let gated_d = testkit::upload_f32(&ctx, &gated);
        let normed_d = testkit::upload_f32(&ctx, &normed);
        ctx.ple_conv(
            &stream,
            PleConvCall {
                gated: gated_d.range(0, width * 4).expect("gated"),
                normed: normed_d.range(0, width * 4).expect("normed"),
                conv: conv_d.range(0, conv.len() * 4).expect("conv"),
                ring: ring_d.range(0, ring.len() * 4).expect("ring"),
                out: out_d.range(0, width * 4).expect("out"),
                width,
                conv_kernel,
                ngram,
            },
        )
        .expect("conv");
        stream.synchronize().expect("drain");
        let got = testkit::floats(&testkit::read_bytes(&ctx, &out_d, width * 4));
        for c in 0..width {
            assert!(
                close(got[c], want[c]),
                "token {token} channel {c}: {} against {}",
                got[c],
                want[c]
            );
        }
        let got_ring = testkit::floats(&testkit::read_bytes(&ctx, &ring_d, ring.len() * 4));
        assert_eq!(got_ring, ring, "the ring after token {token}");
    }
}

#[test]
fn a_short_range_is_refused_by_name_before_any_native_call() {
    if !testkit::device_present("ple refusal") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let small = testkit::upload_filled(&ctx, 0, 64);
    let r = small.range(0, 64).expect("range");
    let error = ctx
        .ple_conv(
            &stream,
            PleConvCall {
                gated: r,
                normed: r,
                conv: r,
                ring: r,
                out: r,
                width: 64,
                conv_kernel: 4,
                ngram: 3,
            },
        )
        .expect_err("every range is short");
    assert!(
        matches!(
            &error,
            EngError::BadRequest(text)
                if text.as_str() == "the CUDA gated range holds 64 bytes but the call needs 256"
        ),
        "{error:?}"
    );
}

/// Every range is large enough and aligned; `out` starts halfway through `gated`. The pair
/// is reported later name first, as `all_apart` visits it.
#[test]
fn an_overlapping_written_range_is_refused_by_name_before_any_native_call() {
    if !testkit::device_present("ple overlap refusal") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let (width, conv_kernel, ngram) = (64usize, 4usize, 3usize);
    let conv_bytes = width * conv_kernel * 4;
    let ring_bytes = width * (conv_kernel - 1) * ngram * 4;
    let normed = testkit::upload_filled(&ctx, 0, width * 4);
    let shared = testkit::upload_filled(&ctx, 0, 2 * width * 4);
    let weights = testkit::upload_filled(&ctx, 0, conv_bytes + ring_bytes);
    let error = ctx
        .ple_conv(
            &stream,
            PleConvCall {
                gated: byte_range(&shared, 0, width * 4),
                normed: byte_range(&normed, 0, width * 4),
                conv: byte_range(&weights, 0, conv_bytes),
                ring: byte_range(&weights, conv_bytes, ring_bytes),
                out: byte_range(&shared, width * 2, width * 4),
                width,
                conv_kernel,
                ngram,
            },
        )
        .expect_err("out shares bytes with gated");
    assert!(
        matches!(
            &error,
            EngError::BadRequest(text) if text.as_str() == "the CUDA out and gated ranges overlap"
        ),
        "{error:?}"
    );
}
