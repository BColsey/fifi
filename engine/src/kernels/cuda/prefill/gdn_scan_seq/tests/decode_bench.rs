//! The decode benchmark of the Flash-Next DeltaNet path, for an exclusive GPU window: one
//! decode token (m = 1) through the reused conv and this entry in the Flash-Next form, against
//! the vendored decode delta kernel (`gdn_delta`, the Qwen3.6 decode path) after the same conv,
//! on the same geometry and buffers. Every arm is sampled in the same rounds, one after the
//! other, so all arms see the same clocks and the same host. Reported, not gated; ignored, run
//! by name in a release build. The state updates of both timed entries are checked against the
//! f64 reference in the same process.
//!
//! Per arm and round: one layer queued and drained (the latency of one launch set, launch
//! overhead included), and 36 layers queued back to back then drained once (the per-layer cost
//! when launches overlap, the decode throughput figure), with the host time spent queueing the
//! 36 layers recorded separately (the CPU launch overhead). The wall clock around a drained
//! stream stands in for device events, which this tree does not time.

use super::forms::{FN_K, FN_V, opts_of};
use super::{
    E3, E4, RefOpts, Rig, ScanLayout, ScanRig, assert_constraints, generate_scan, judge_scan_with,
    le_bytes, run_decode, seed_of,
};
use crate::EngError;
use crate::kernels::cuda::prefill::gdn_scan_seq::{
    GDN_SCAN_EPS, GdnForm, GdnScanSeqCall, ScanWeights, scan_bytes,
};
use crate::kernels::cuda::resident::gdn_conv::{ConvCall, ConvShape};
use crate::kernels::cuda::resident::gdn_delta::{DeltaCall, DeltaShape, DeltaWeights, delta_bytes};
use crate::kernels::cuda::{DeviceBuffer, testkit};

/// The DeltaNet layers of Flash-Next, the queue depth of one decode token.
const LAYERS: usize = 36;
const ROUNDS: usize = 101;
const WARMUPS: usize = 10;

/// The buffers of one layer at m = 1: the conv weights and ring, the projection row, the conv
/// output, the per-head weights, the state, and the outputs of both scans.
struct Layer {
    rig: Rig,
    lay: ScanLayout,
    nk: usize,
    nv: usize,
    proj: DeviceBuffer,
    w: DeviceBuffer,
    ring: DeviceBuffer,
    conv_out: DeviceBuffer,
    a: DeviceBuffer,
    dt: DeviceBuffer,
    nw: DeviceBuffer,
    state: DeviceBuffer,
    o_out: DeviceBuffer,
    d_out: DeviceBuffer,
}

impl Layer {
    fn new(nk: usize, nv: usize) -> Result<Self, EngError> {
        let lay = ScanLayout::of(nk, nv);
        let inp = generate_scan(nk, nv, 1, lay, seed_of(nk, nv, 1, 0xbe));
        let rig = Rig::new()?;
        let ctx = &rig.ctx;
        let up = |values: &[f32]| ctx.upload(&le_bytes(values));
        let channels = lay.conv_ld;
        let need = scan_bytes(inp.shape(1, 0, 1024))?;
        let dneed = delta_bytes(Self::delta_shape(lay, nk, nv))?;
        Ok(Self {
            // The projection row holds the conv input in its first `channels` columns.
            proj: up(&inp.proj)?,
            w: up(&vec![0.25f32; channels * 4])?,
            ring: up(&vec![0.0f32; 3 * channels])?,
            conv_out: up(&inp.conv)?,
            a: up(&inp.a)?,
            dt: up(&inp.dt)?,
            nw: up(&inp.nw)?,
            state: up(&inp.state0)?,
            o_out: ctx.upload(&vec![0u8; need.o_out])?,
            d_out: ctx.upload(&vec![0u8; dneed.out])?,
            rig,
            lay,
            nk,
            nv,
        })
    }

    fn delta_shape(lay: ScanLayout, nk: usize, nv: usize) -> DeltaShape {
        DeltaShape {
            m: 1,
            n_k_heads: nk,
            n_v_heads: nv,
            conv_ld: lay.conv_ld,
            ld: lay.ld,
            z_off: lay.z_off,
            beta_off: lay.beta_off,
            alpha_off: lay.alpha_off,
        }
    }

    fn conv(&self) -> Result<(), EngError> {
        let channels = self.lay.conv_ld;
        self.rig.ctx.gdn_conv(
            &self.rig.stream,
            ConvCall {
                shape: ConvShape {
                    m: 1,
                    channels,
                    ld: self.lay.ld,
                },
                proj: self.proj.range(0, channels * 4)?,
                weights: self.w.range(0, channels * 16)?,
                state: self.ring.range(0, 3 * channels * 4)?,
                out: self.conv_out.range(0, channels * 4)?,
            },
        )
    }

    fn scan(&self, form: GdnForm) -> Result<(), EngError> {
        let shape = super::ScanShape {
            m: 1,
            p0: 0,
            ctx_tokens: 1024,
            n_k_heads: self.nk,
            n_v_heads: self.nv,
            head_dim: super::D,
            conv_ld: self.lay.conv_ld,
            ld: self.lay.ld,
            z_off: self.lay.z_off,
            beta_off: self.lay.beta_off,
            alpha_off: self.lay.alpha_off,
        };
        let need = scan_bytes(shape)?;
        self.rig.ctx.gdn_scan_seq_form(
            &self.rig.stream,
            GdnScanSeqCall {
                shape,
                eps: GDN_SCAN_EPS,
                conv: self.conv_out.range(0, need.conv)?,
                proj: self.proj.range(0, need.proj)?,
                weights: ScanWeights {
                    a: self.a.range(0, need.a)?,
                    dt_bias: self.dt.range(0, need.dt_bias)?,
                    norm_w: self.nw.range(0, need.norm_w)?,
                },
                state: self.state.range(0, need.state)?,
                o_out: self.o_out.range(0, need.o_out)?,
                status: None,
            },
            form,
        )
    }

    fn delta(&self) -> Result<(), EngError> {
        let shape = Self::delta_shape(self.lay, self.nk, self.nv);
        let need = delta_bytes(shape)?;
        self.rig.ctx.gdn_delta(
            &self.rig.stream,
            DeltaCall {
                shape,
                eps: GDN_SCAN_EPS,
                conv: self.conv_out.range(0, need.conv)?,
                proj: self.proj.range(0, need.proj)?,
                weights: DeltaWeights {
                    a: self.a.range(0, need.a)?,
                    dt_bias: self.dt.range(0, need.dt_bias)?,
                    norm_w: self.nw.range(0, need.norm_w)?,
                },
                state: self.state.range(0, need.state)?,
                out: self.d_out.range(0, need.out)?,
            },
        )
    }

    /// The state after the timed rounds is finite (the rounds feed one token repeatedly).
    fn state_is_finite(&self) -> Result<bool, EngError> {
        self.rig.stream.synchronize()?;
        let mut bytes = vec![0u8; self.state.bytes()];
        self.state.read_at(&self.rig.stream, 0, &mut bytes)?;
        Ok(testkit::floats(&bytes).iter().all(|v| v.is_finite()))
    }
}

/// What one arm queues for one layer.
#[derive(Clone, Copy)]
enum Step {
    Conv,
    Scan(GdnForm),
    Delta,
    ConvScan(GdnForm),
    ConvDelta,
}

struct Arm {
    name: &'static str,
    layer: usize,
    step: Step,
    one: Vec<f64>,
    deep: Vec<f64>,
    enqueue: Vec<f64>,
}

fn queue(layer: &Layer, step: Step) -> Result<(), EngError> {
    match step {
        Step::Conv => layer.conv(),
        Step::Scan(form) => layer.scan(form),
        Step::Delta => layer.delta(),
        Step::ConvScan(form) => {
            layer.conv()?;
            layer.scan(form)
        }
        Step::ConvDelta => {
            layer.conv()?;
            layer.delta()
        }
    }
}

/// `n` layers queued, then a drain: (host queue time, total time), in microseconds.
fn sample(layer: &Layer, step: Step, n: usize) -> Result<(f64, f64), EngError> {
    let t0 = std::time::Instant::now();
    for _ in 0..n {
        queue(layer, step)?;
    }
    let queued = t0.elapsed();
    layer.rig.stream.synchronize()?;
    let total = t0.elapsed();
    Ok((queued.as_secs_f64() * 1e6, total.as_secs_f64() * 1e6))
}

/// (p10, p50, p90) of `values`.
fn spread(values: &[f64]) -> (f64, f64, f64) {
    let mut v = values.to_vec();
    v.sort_by(|a, b| a.total_cmp(b));
    let at = |q: usize| v[(v.len() - 1) * q / 100];
    (at(10), at(50), at(90))
}

fn run_rounds(layers: &[Layer], arms: &mut [Arm]) -> Result<(), EngError> {
    for round in 0..WARMUPS + ROUNDS {
        for arm in arms.iter_mut() {
            let layer = &layers[arm.layer];
            let (_, one) = sample(layer, arm.step, 1)?;
            let (queued, deep) = sample(layer, arm.step, LAYERS)?;
            if round >= WARMUPS {
                arm.one.push(one);
                arm.deep.push(deep / LAYERS as f64);
                arm.enqueue.push(queued / LAYERS as f64);
            }
        }
    }
    Ok(())
}

fn report(arms: &[Arm]) {
    let profile = if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    };
    println!(
        "BENCH gdn decode ({profile}, m = 1, {ROUNDS} interleaved rounds after {WARMUPS} warm-ups; microseconds, p50 [p10, p90])"
    );
    for arm in arms {
        let (o10, o50, o90) = spread(&arm.one);
        let (d10, d50, d90) = spread(&arm.deep);
        let (e10, e50, e90) = spread(&arm.enqueue);
        println!(
            "BENCH {:<34} one layer {o50:8.2} [{o10:.2}, {o90:.2}]  per layer at depth {LAYERS} {d50:8.2} [{d10:.2}, {d90:.2}]  host queue per layer {e50:6.2} [{e10:.2}, {e90:.2}]  {LAYERS} layers {:.3} ms",
            arm.name,
            d50 * LAYERS as f64 / 1e3
        );
    }
}

/// The timed entries update the state correctly, in this process: 16 one-token calls of the
/// Flash-Next form against the f64 reference of that form (E3), and 16 vendored decode steps
/// against the reference of the vendored form (tiled heads, `rsqrt(sum + eps)`, silu; E4, which
/// covers its `__expf` gates and its reduction order).
fn check_state_updates() -> Result<(), EngError> {
    let m = 16;
    let lay = ScanLayout::of(FN_K, FN_V);
    let inp = generate_scan(FN_K, FN_V, m, lay, seed_of(FN_K, FN_V, m, 0xbf));
    assert_constraints(&inp, &[]);
    let got = {
        let scan = ScanRig::new(Rig::new()?, &inp)?;
        for t in 0..m {
            scan.run_form(&inp, (1, t as i64, t), GdnForm::FLASH_NEXT)?;
        }
        scan.finish()?
    };
    let opts = opts_of(GdnForm::FLASH_NEXT);
    let (rows, state) = judge_scan_with(&inp, (&got.out, &got.state), "bench flash-next", opts);
    assert!(
        rows <= E3 && state <= E3,
        "flash-next m = 1: rows {rows:.3e} state {state:.3e}"
    );
    let (out, st) = run_decode(&Rig::new()?, &inp)?;
    let vendored = RefOpts {
        decode_norm_form: true,
        ..Default::default()
    };
    let (rows, state) = judge_scan_with(&inp, (&out, &st), "bench vendored", vendored);
    assert!(
        rows <= E4 && state <= E4,
        "vendored decode: rows {rows:.3e} state {state:.3e}"
    );
    Ok(())
}

#[test]
#[ignore = "timing; run by name in a release build inside an exclusive GPU window"]
fn gdn_decode_bench_flash_next_against_the_vendored_decode() -> Result<(), EngError> {
    if !testkit::device_present("gdn_decode_bench_flash_next_against_the_vendored_decode") {
        return Ok(());
    }
    check_state_updates()?;
    let layers = [Layer::new(FN_K, FN_V)?, Layer::new(16, 32)?];
    let arm = |name: &'static str, layer: usize, step: Step| Arm {
        name,
        layer,
        step,
        one: Vec::with_capacity(ROUNDS),
        deep: Vec::with_capacity(ROUNDS),
        enqueue: Vec::with_capacity(ROUNDS),
    };
    let fnx = GdnForm::FLASH_NEXT;
    let q36 = GdnForm::QWEN36_GGUF;
    let mut arms = [
        arm("16x48 conv", 0, Step::Conv),
        arm("16x48 flash-next scan", 0, Step::Scan(fnx)),
        arm("16x48 vendored delta", 0, Step::Delta),
        arm("16x48 conv + flash-next scan", 0, Step::ConvScan(fnx)),
        arm("16x48 conv + vendored delta", 0, Step::ConvDelta),
        arm("16x32 qwen36-form scan", 1, Step::Scan(q36)),
        arm("16x32 vendored delta", 1, Step::Delta),
    ];
    run_rounds(&layers, &mut arms)?;
    report(&arms);
    for layer in &layers {
        assert!(layer.state_is_finite()?, "a timed state went non-finite");
    }
    Ok(())
}
