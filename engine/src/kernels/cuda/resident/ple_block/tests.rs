//! Tests of the PLE block launches against the host reference (`family::math` and the
//! dequantizer of `weights`). Device runs need a card; without one each test returns.

use super::*;
use crate::family::math::{dot, rms0, silu};
use crate::kernels::cuda::{CudaContext, testkit};
use crate::weights::{GgufType, dequantize};

struct Lcg(u64);

impl Lcg {
    fn next(&mut self, scale: f32) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as f32 / (1u64 << 31) as f32 - 0.5) * 2.0 * scale
    }

    fn vec(&mut self, len: usize, scale: f32) -> Vec<f32> {
        (0..len).map(|_| self.next(scale)).collect()
    }
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
    let mut rng = Lcg(7);
    let mut raw = Vec::new();
    for _ in 0..rows * head_dim / 32 {
        let d = 0.05 + rng.next(0.02).abs();
        raw.extend_from_slice(&half_bits(d).to_le_bytes());
        for _ in 0..16 {
            raw.push((rng.next(8.0) as i32 + 8) as u8 | (((rng.next(8.0) as i32 + 8) as u8) << 4));
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

/// The f16 bits of `value`, round to nearest.
fn half_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mantissa = bits & 0x7f_ffff;
    if exp <= 0 {
        return sign;
    }
    sign | ((exp as u16) << 10) | ((mantissa + 0x1000) >> 13) as u16
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
    let mut rng = Lcg(11);
    let keys = rng.vec(width, 2.0);
    let hyper = rng.vec(width, 3.0);
    let value = rng.vec(hidden, 1.5);
    let nk = rng.vec(width, 0.3);
    let nq = rng.vec(width, 0.3);
    let nc = rng.vec(width, 0.3);
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
    let r = |b: &crate::kernels::cuda::DeviceBuffer, n: usize| b.range(0, n * 4).expect("range");
    ctx.ple_gate(
        &stream,
        PleGateCall {
            keys: r(&keys_d, width),
            hyper: r(&hyper_d, width),
            value: r(&value_d, hidden),
            norm_key: r(&nk_d, width),
            norm_query: r(&nq_d, width),
            norm_conv: r(&nc_d, width),
            gated: r(&gated_d, width),
            normed: r(&normed_d, width),
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
    let mut rng = Lcg(23);
    let conv = rng.vec(width * conv_kernel, 0.7);
    let mut ring = rng.vec(width * stride, 1.0);
    let conv_d = testkit::upload_f32(&ctx, &conv);
    let ring_d = testkit::upload_f32(&ctx, &ring);
    let out_d = testkit::upload_filled(&ctx, 0xff, width * 4);
    for token in 0..3 {
        let gated = rng.vec(width, 1.0);
        let normed = rng.vec(width, 1.0);
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
fn a_launch_with_a_short_buffer_is_refused_before_any_native_call() {
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
        .expect_err("ranges overlap and are short");
    assert!(matches!(error, EngError::BadRequest(_)), "{error:?}");
}
