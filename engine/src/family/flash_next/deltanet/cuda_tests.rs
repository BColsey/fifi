//! The Flash-Next DeltaNet core on CUDA (the reused conv `gdn_conv`, then `gdn_scan_seq` in
//! [`GdnForm::FLASH_NEXT`]) against the host [`step`] of this module, which the Mac MLX path
//! is checked against, at the Flash-Next head geometry: 16 key heads and 48 value heads of
//! 128, conv kernel 4, eps 1e-6.
//!
//! The device takes the projection rows the host computes (`qkv | z | b | a`, the same f32
//! matvecs `step` runs), so the comparison covers the conv, the L2 norms, the gates, the
//! recurrence and the gated RMSNorm: the host `o` rows are tapped at the input of the output
//! projection. Three schedules of the same 24 tokens run from a zero state: one chunk (the
//! prefill), two chunks of 17 and 7 (a chunk boundary inside the 16-token tile), and 24
//! one-token calls (the decode path). They must give the same bits, and each must match the
//! host within `max|d - h| <= 2^-12 * max(max|h|, 1)` per (token, head) row of `o`, per head
//! of the recurrent state, and over the conv ring.
//!
//! The head association is the point of the geometry: the host pairs value head `h` with key
//! head `h / 3`; the grouped key-head form must give the same pairing with no weight
//! permutation, and the state comes back in the host layout `[head][key i][value j]`.

use super::{DeltaShape, DeltaState, DeltaWeights, step};
use crate::EngError;
use crate::family::project::{DenseProjector, DenseWeight, Projector};
use crate::kernels::cuda::prefill::gdn_scan_seq::{
    GDN_SCAN_EPS, GdnForm, GdnScanSeqCall, ScanShape, ScanWeights, scan_bytes,
};
use crate::kernels::cuda::resident::gdn_conv::{ConvCall, ConvShape, conv_bytes};
use crate::kernels::cuda::{CudaContext, DeviceBuffer, DeviceStream, device_present};

const KEY_HEADS: usize = 16;
const VALUE_HEADS: usize = 48;
const HEAD: usize = 128;
const CONV_KERNEL: usize = 4;
const HIDDEN: usize = 32;
const TOKENS: usize = 24;
const EPS: f32 = 1e-6;
const CLASS: f64 = 1.0 / 4096.0;

const KEY_WIDTH: usize = KEY_HEADS * HEAD;
const VALUE_WIDTH: usize = VALUE_HEADS * HEAD;
const CHANNELS: usize = 2 * KEY_WIDTH + VALUE_WIDTH;
const Z_OFF: usize = CHANNELS;
const BETA_OFF: usize = Z_OFF + VALUE_WIDTH;
const ALPHA_OFF: usize = BETA_OFF + VALUE_HEADS;
const LD: usize = ALPHA_OFF + VALUE_HEADS;

/// The dense projector of the tests, with the output projection replaced by a tap that keeps
/// its input (the gated `o` row) and returns zeros.
struct Tap<'a> {
    out: &'a DenseWeight,
    dense: DenseProjector,
    taps: Vec<Vec<f32>>,
}

impl Projector for Tap<'_> {
    type Weight = DenseWeight;

    fn linear(&mut self, weight: &DenseWeight, x: &[f32]) -> Result<Vec<f32>, EngError> {
        if std::ptr::eq(weight, self.out) {
            self.taps.push(x.to_vec());
            return Ok(vec![0.0; HIDDEN]);
        }
        self.dense.linear(weight, x)
    }

    fn expert_linear(
        &mut self,
        stack: &DenseWeight,
        expert: u32,
        x: &[f32],
    ) -> Result<Vec<f32>, EngError> {
        self.dense.expert_linear(stack, expert, x)
    }

    fn rows(&mut self, table: &DenseWeight, rows: &[u32]) -> Result<Vec<f32>, EngError> {
        self.dense.rows(table, rows)
    }
}

/// A SplitMix64 stream (the test kit's generator is private to `kernels::cuda`).
struct Rng(u64);

impl Rng {
    /// A value in `[0, 1)` with 24 random bits.
    fn unit(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        (z >> 40) as f64 / (1u64 << 24) as f64
    }
}

fn uniform(rng: &mut Rng, lo: f64, hi: f64, count: usize) -> Vec<f32> {
    (0..count)
        .map(|_| (lo + (hi - lo) * rng.unit()) as f32)
        .collect()
}

/// A layer drawn at trained-model scales: `A_log` in [0, 2.5] and `dt_bias` in [-5, -2], so
/// the per-step decay lies around 0.7 to 0.97, and projections of order one.
struct Layer {
    qkv: DenseWeight,
    z: DenseWeight,
    b: DenseWeight,
    a: DenseWeight,
    out: DenseWeight,
    conv: Vec<f32>,
    a_log: Vec<f32>,
    dt_bias: Vec<f32>,
    norm: Vec<f32>,
}

impl Layer {
    fn new(rng: &mut Rng) -> Self {
        let matrix = |rng: &mut Rng, rows: usize, scale: f64| {
            DenseWeight::matrix(rows, HIDDEN, uniform(rng, -scale, scale, rows * HIDDEN))
        };
        Self {
            qkv: matrix(rng, CHANNELS, 0.3),
            z: matrix(rng, VALUE_WIDTH, 0.5),
            b: matrix(rng, VALUE_HEADS, 0.3),
            a: matrix(rng, VALUE_HEADS, 0.3),
            out: DenseWeight::matrix(HIDDEN, VALUE_WIDTH, vec![0.0; HIDDEN * VALUE_WIDTH]),
            conv: uniform(rng, -0.6, 0.6, CHANNELS * CONV_KERNEL),
            a_log: uniform(rng, 0.0, 2.5, VALUE_HEADS),
            dt_bias: uniform(rng, -5.0, -2.0, VALUE_HEADS),
            norm: uniform(rng, 0.5, 1.5, HEAD),
        }
    }

    fn weights(&self) -> DeltaWeights<'_, DenseWeight> {
        DeltaWeights {
            qkv: &self.qkv,
            z: &self.z,
            b: &self.b,
            a: &self.a,
            out: &self.out,
            conv: &self.conv,
            a_log: &self.a_log,
            dt_bias: &self.dt_bias,
            norm: &self.norm,
        }
    }

    /// One device projection row `qkv | z | b | a` of `x`, from the same matvecs as `step`.
    fn proj_row(&self, x: &[f32]) -> Vec<f32> {
        let mut dense = DenseProjector::default();
        let mut row = Vec::with_capacity(LD);
        for weight in [&self.qkv, &self.z, &self.b, &self.a] {
            row.extend(dense.linear(weight, x).expect("projection"));
        }
        row
    }
}

/// The host run: the tapped `o` rows of every token, the recurrent state and the conv ring
/// `[channel][tap]` after the last token.
fn host_run(layer: &Layer, xs: &[Vec<f32>]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let shape = DeltaShape::new(
        KEY_HEADS as u32,
        HEAD as u32,
        VALUE_HEADS as u32,
        HEAD as u32,
        CONV_KERNEL as u32,
        HIDDEN as u32,
        EPS,
    )
    .expect("shape");
    let weights = layer.weights();
    let mut tap = Tap {
        out: &layer.out,
        dense: DenseProjector::default(),
        taps: Vec::new(),
    };
    let mut state = DeltaState::new(&shape);
    for x in xs {
        step(&mut tap, &shape, &weights, &mut state, x).expect("host step");
    }
    (
        tap.taps.concat(),
        state.rec().to_vec(),
        state.ring().to_vec(),
    )
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn download(ctx: &CudaContext, buffer: &DeviceBuffer) -> Vec<f32> {
    let mut bytes = vec![0u8; buffer.bytes()];
    ctx.download(buffer, &mut bytes).expect("download");
    let (words, _) = bytes.as_chunks::<4>();
    words.iter().map(|w| f32::from_le_bytes(*w)).collect()
}

/// The device buffers of one schedule.
struct Device {
    ctx: CudaContext,
    stream: DeviceStream,
    proj: DeviceBuffer,
    conv_w: DeviceBuffer,
    a: DeviceBuffer,
    dt: DeviceBuffer,
    norm: DeviceBuffer,
    ring: DeviceBuffer,
    conv_out: DeviceBuffer,
    state: DeviceBuffer,
    o_out: DeviceBuffer,
}

impl Device {
    fn new(layer: &Layer, proj: &[f32]) -> Result<Self, EngError> {
        let ctx = CudaContext::new()?;
        let stream = ctx.stream()?;
        // ssm_a as the scan reads it, `-exp(A_log)` in f32: the same value `step` forms.
        let a: Vec<f32> = layer.a_log.iter().map(|v| -(v.exp())).collect();
        let zeros = |floats: usize| ctx.upload(&vec![0u8; floats * 4]);
        Ok(Self {
            proj: ctx.upload(&f32_bytes(proj))?,
            conv_w: ctx.upload(&f32_bytes(&layer.conv))?,
            a: ctx.upload(&f32_bytes(&a))?,
            dt: ctx.upload(&f32_bytes(&layer.dt_bias))?,
            norm: ctx.upload(&f32_bytes(&layer.norm))?,
            ring: zeros(3 * CHANNELS)?,
            conv_out: zeros(TOKENS * CHANNELS)?,
            state: zeros(VALUE_HEADS * HEAD * HEAD)?,
            o_out: ctx.upload(&vec![0xFF_u8; TOKENS * VALUE_WIDTH * 4])?,
            ctx,
            stream,
        })
    }

    /// Queues the conv and the scan of tokens `row0 .. row0 + m`.
    fn chunk(&self, row0: usize, m: usize) -> Result<(), EngError> {
        let conv_shape = ConvShape {
            m,
            channels: CHANNELS,
            ld: LD,
        };
        let cneed = conv_bytes(conv_shape)?;
        self.ctx.gdn_conv(
            &self.stream,
            ConvCall {
                shape: conv_shape,
                proj: self.proj.range(row0 * LD * 4, cneed.proj)?,
                weights: self.conv_w.range(0, cneed.weights)?,
                state: self.ring.range(0, cneed.state)?,
                out: self.conv_out.range(row0 * CHANNELS * 4, cneed.out)?,
            },
        )?;
        let shape = ScanShape {
            m,
            p0: row0 as i64,
            ctx_tokens: TOKENS,
            n_k_heads: KEY_HEADS,
            n_v_heads: VALUE_HEADS,
            head_dim: HEAD,
            conv_ld: CHANNELS,
            ld: LD,
            z_off: Z_OFF,
            beta_off: BETA_OFF,
            alpha_off: ALPHA_OFF,
        };
        let need = scan_bytes(shape)?;
        self.ctx.gdn_scan_seq_form(
            &self.stream,
            GdnScanSeqCall {
                shape,
                eps: GDN_SCAN_EPS,
                conv: self.conv_out.range(row0 * CHANNELS * 4, need.conv)?,
                proj: self.proj.range(row0 * LD * 4, need.proj)?,
                weights: ScanWeights {
                    a: self.a.range(0, need.a)?,
                    dt_bias: self.dt.range(0, need.dt_bias)?,
                    norm_w: self.norm.range(0, need.norm_w)?,
                },
                state: self.state.range(0, need.state)?,
                o_out: self.o_out.range(row0 * VALUE_WIDTH * 4, need.o_out)?,
                status: None,
            },
            GdnForm::FLASH_NEXT,
        )
    }

    /// The `o` rows, the recurrent state and the conv ring `[tap][channel]`, after a drain.
    fn finish(&self) -> Result<[Vec<f32>; 3], EngError> {
        self.stream.synchronize()?;
        Ok([
            download(&self.ctx, &self.o_out),
            download(&self.ctx, &self.state),
            download(&self.ctx, &self.ring),
        ])
    }
}

/// Runs the 24 tokens in chunks of the given sizes, from a zero state.
fn device_run(layer: &Layer, proj: &[f32], sizes: &[usize]) -> Result<[Vec<f32>; 3], EngError> {
    assert_eq!(sizes.iter().sum::<usize>(), TOKENS);
    let device = Device::new(layer, proj)?;
    let mut row0 = 0;
    for &m in sizes {
        device.chunk(row0, m)?;
        row0 += m;
    }
    device.finish()
}

/// The largest `|d - h|` of each `width`-long row of `host`, over `max(max|h|, 1)` of that
/// row; NaN counts as infinite.
fn worst_row(dev: &[f32], host: &[f32], width: usize) -> f64 {
    assert_eq!(dev.len(), host.len());
    dev.chunks(width)
        .zip(host.chunks(width))
        .map(|(d, h)| {
            let scale = h.iter().fold(1.0f64, |m, v| m.max(f64::from(*v).abs()));
            let worst = d.iter().zip(h).fold(0.0f64, |m, (d, h)| {
                let e = (f64::from(*d) - f64::from(*h)).abs();
                if e.is_nan() { f64::INFINITY } else { m.max(e) }
            });
            worst / scale
        })
        .fold(0.0, f64::max)
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

#[test]
fn cuda_deltanet_matches_the_host_step_in_every_schedule() -> Result<(), EngError> {
    let mut rng = Rng(0xf1a5_4e47);
    let layer = Layer::new(&mut rng);
    let xs: Vec<Vec<f32>> = (0..TOKENS)
        .map(|_| uniform(&mut rng, -1.0, 1.0, HIDDEN))
        .collect();
    let proj: Vec<f32> = xs.iter().flat_map(|x| layer.proj_row(x)).collect();
    let (host_o, host_rec, host_ring) = host_run(&layer, &xs);
    // The host ring is [channel][tap]; the device keeps it planar [tap][channel].
    let host_ring_planar: Vec<f32> = (0..CONV_KERNEL - 1)
        .flat_map(|t| (0..CHANNELS).map(move |c| (c, t)))
        .map(|(c, t)| host_ring[c * (CONV_KERNEL - 1) + t])
        .collect();
    if !device_present("cuda_deltanet_matches_the_host_step_in_every_schedule") {
        return Ok(());
    }
    let schedules: [(&str, Vec<usize>); 3] = [
        ("one chunk", vec![TOKENS]),
        ("17 + 7", vec![17, 7]),
        ("tokens", vec![1; TOKENS]),
    ];
    let mut runs = Vec::new();
    for (name, sizes) in &schedules {
        runs.push((*name, device_run(&layer, &proj, sizes)?));
    }
    let (_, first) = &runs[0];
    for (name, [o, rec, ring]) in &runs {
        let o_err = worst_row(o, &host_o, HEAD);
        let rec_err = worst_row(rec, &host_rec, HEAD * HEAD);
        let ring_err = worst_row(ring, &host_ring_planar, ring.len());
        println!(
            "REPORTED: flash-next deltanet {name}: o rows {o_err:.3e}, state {rec_err:.3e}, ring {ring_err:.3e} (class {CLASS:.3e})"
        );
        assert!(o_err <= CLASS, "{name}: o rows {o_err:.3e} above the class");
        assert!(
            rec_err <= CLASS,
            "{name}: state {rec_err:.3e} above the class"
        );
        assert!(
            ring_err == 0.0,
            "{name}: the ring is not the last three inputs"
        );
        assert_eq!(
            bits(o),
            bits(&first[0]),
            "{name}: o rows differ from one chunk"
        );
        assert_eq!(
            bits(rec),
            bits(&first[1]),
            "{name}: state differs from one chunk"
        );
    }
    Ok(())
}
