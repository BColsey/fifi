//! The CUDA hyper-connection kernels (`kernels::cuda::resident::hyper`) against the host [`read`]
//! and [`apply`] of this module, which the Mac MLX path is checked against, at the Flash-Next site
//! shape: 4 branches of 2560, rank 320, binary16 weights as the EXL3 packs store the `hc_*`
//! tensors. Host and device both accumulate in f32 but in different orders, so the class is
//! `max|d - h| <= 2^-13 * max(max|h|, 1)` per output vector of a token (apply: `2^-20`).

use super::{HyperShape, HyperWeights, apply, read};
use crate::family::project::{DenseProjector, DenseWeight};
use crate::kernels::cuda::resident::hyper as device;
use crate::kernels::cuda::{CudaContext, DeviceBuffer, DeviceRange, device_present};

const BRANCHES: usize = 4;
const HIDDEN: usize = 2560;
const RANK: usize = 320;
const WIDTH: usize = BRANCHES * HIDDEN;
const TOKENS: usize = 3;
const EPS: f32 = 1e-6;
const READ_CLASS: f32 = 1.0 / 8192.0;
const APPLY_CLASS: f32 = 1.0 / 1_048_576.0;

/// A value in `[-scale, scale)` hashed from `key` (the SplitMix64 finalizer), so every tensor is
/// a fixed function of its element index.
fn noise(key: u64, scale: f32) -> f32 {
    let mut z = key.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^= z >> 31;
    let unit = (z >> 40) as f32 / (1u64 << 24) as f32;
    (2.0 * unit - 1.0) * scale
}

fn values(count: usize, salt: u64, scale: f32) -> Vec<f32> {
    (0..count as u64)
        .map(|i| noise(salt << 32 | i, scale))
        .collect()
}

/// Binary16 codes of the values, and the f32 values the codes hold.
fn f16_weights(count: usize, salt: u64, scale: f32) -> (Vec<u8>, Vec<f32>) {
    let codes: Vec<half::f16> = values(count, salt, scale)
        .into_iter()
        .map(half::f16::from_f32)
        .collect();
    let bytes = codes
        .iter()
        .flat_map(|c| c.to_bits().to_le_bytes())
        .collect();
    (bytes, codes.iter().map(|c| c.to_f32()).collect())
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn download(ctx: &CudaContext, buffer: &DeviceBuffer, floats: usize) -> Vec<f32> {
    let mut bytes = vec![0u8; floats * 4];
    ctx.download(buffer, &mut bytes).expect("download");
    let (words, _) = bytes.as_chunks::<4>();
    words.iter().map(|w| f32::from_le_bytes(*w)).collect()
}

fn whole(buffer: &DeviceBuffer, bytes: usize) -> DeviceRange<'_> {
    buffer.range(0, bytes).expect("range")
}

fn assert_class(what: &str, dev: &[f32], host: &[f32], per_token: usize, class: f32) {
    for (t, (d, h)) in dev
        .chunks(per_token)
        .zip(host.chunks(per_token))
        .enumerate()
    {
        let scale = h.iter().fold(1.0f32, |a, v| a.max(v.abs()));
        let worst = d
            .iter()
            .zip(h)
            .map(|(d, h)| (d - h).abs())
            .fold(
                0.0f32,
                |a, e| if e.is_nan() { f32::INFINITY } else { a.max(e) },
            );
        assert!(
            worst <= class * scale,
            "{what} token {t}: max|d - h| {worst} over {}",
            class * scale
        );
    }
}

struct Site {
    norm: (Vec<u8>, Vec<f32>),
    down: (Vec<u8>, Vec<f32>),
    up: (Vec<u8>, Vec<f32>),
    inject: (Vec<u8>, Vec<f32>),
}

impl Site {
    /// Scales that keep `dot / 4` of a down or inject row near unit size, as in a trained site.
    fn new() -> Self {
        let row = 6.0 / (WIDTH as f32).sqrt();
        Self {
            norm: f16_weights(WIDTH, 1, 0.3),
            down: f16_weights(RANK * WIDTH, 2, row),
            up: f16_weights(WIDTH * RANK, 3, 3.0 / (RANK as f32).sqrt()),
            inject: f16_weights(BRANCHES * WIDTH, 4, row),
        }
    }
}

/// The host read of each token: `mixed` rows and gates.
fn host_read(site: &Site, x: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let shape = HyperShape::new(BRANCHES as u32, HIDDEN as u32, RANK as u32, EPS).expect("shape");
    let (down, up, inject) = (
        DenseWeight::matrix(RANK, WIDTH, site.down.1.clone()),
        DenseWeight::matrix(WIDTH, RANK, site.up.1.clone()),
        DenseWeight::matrix(BRANCHES, WIDTH, site.inject.1.clone()),
    );
    let weights = HyperWeights {
        norm: &site.norm.1,
        down: &down,
        up: &up,
        inject: Some(&inject),
    };
    let (mut mixed, mut gate) = (Vec::new(), Vec::new());
    for token in x.chunks(WIDTH) {
        let out = read(&mut DenseProjector { calls: 0 }, &shape, &weights, token).expect("read");
        mixed.extend(out.mixed);
        gate.extend(out.gate.expect("an inject site has a gate"));
    }
    (mixed, gate)
}

#[test]
fn cuda_read_and_apply_match_the_host_site() {
    if !device_present("cuda_read_and_apply_match_the_host_site") {
        return;
    }
    let site = Site::new();
    // Each branch has its own scale, so the four per-branch norms differ.
    let x: Vec<f32> = (0..TOKENS * WIDTH)
        .map(|i| noise(5 << 32 | i as u64, 0.5 + ((i % WIDTH) / HIDDEN) as f32))
        .collect();
    let (host_mixed, host_gate) = host_read(&site, &x);

    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let shape = device::HyperShape {
        m: TOKENS,
        branches: BRANCHES,
        hidden: HIDDEN,
        rank: RANK,
    };
    let need = device::hyper_read_bytes(shape).expect("sizes");
    let upload = |bytes: &[u8]| ctx.upload(bytes).expect("upload");
    let alloc = |bytes: usize| ctx.allocate(bytes).expect("allocate");
    let (x_dev, norm, down, up, inject) = (
        upload(&f32_bytes(&x)),
        upload(&site.norm.0),
        upload(&site.down.0),
        upload(&site.up.0),
        upload(&site.inject.0),
    );
    let (normed, low, mix, mixed, gate) = (
        alloc(need.normed),
        alloc(need.low),
        alloc(need.mix),
        alloc(need.mixed),
        alloc(need.gate),
    );
    let call = device::HyperReadCall {
        shape,
        eps: EPS,
        x: whole(&x_dev, need.x),
        weights: device::HyperWeights {
            dtype: device::HyperDtype::F16,
            norm: whole(&norm, need.norm_w),
            down: whole(&down, need.down),
            up: whole(&up, need.up),
            inject: Some(whole(&inject, need.inject)),
        },
        scratch: device::HyperScratch {
            normed: whole(&normed, need.normed),
            low: whole(&low, need.low),
            mix: whole(&mix, need.mix),
        },
        mixed: whole(&mixed, need.mixed),
        gate: Some(whole(&gate, need.gate)),
    };
    ctx.hyper_read(&stream, call).expect("device read");
    stream.synchronize().expect("drain");
    let dev_mixed = download(&ctx, &mixed, TOKENS * HIDDEN);
    let dev_gate = download(&ctx, &gate, TOKENS * BRANCHES);
    assert_class("mixed", &dev_mixed, &host_mixed, HIDDEN, READ_CLASS);
    assert_class("gate", &dev_gate, &host_gate, BRANCHES, READ_CLASS);

    // The apply of a block output, fed with the device gates on both sides.
    let block = values(TOKENS * HIDDEN, 6, 2.0);
    let block_dev = upload(&f32_bytes(&block));
    let host_shape =
        HyperShape::new(BRANCHES as u32, HIDDEN as u32, RANK as u32, EPS).expect("shape");
    let mut host_x = x.clone();
    for (t, token) in host_x.chunks_mut(WIDTH).enumerate() {
        let gates = &dev_gate[t * BRANCHES..(t + 1) * BRANCHES];
        let rows = &block[t * HIDDEN..(t + 1) * HIDDEN];
        apply(&host_shape, token, gates, rows).expect("host apply");
    }
    let call = device::HyperApplyCall {
        shape,
        x: whole(&x_dev, need.x),
        gate: whole(&gate, need.gate),
        block: whole(&block_dev, TOKENS * HIDDEN * 4),
    };
    ctx.hyper_apply(&stream, call).expect("device apply");
    stream.synchronize().expect("drain");
    let dev_x = download(&ctx, &x_dev, TOKENS * WIDTH);
    assert_class("applied x", &dev_x, &host_x, WIDTH, APPLY_CLASS);
}
