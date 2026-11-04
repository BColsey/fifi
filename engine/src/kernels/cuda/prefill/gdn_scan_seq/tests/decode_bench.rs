//! The decode benchmark of the Flash-Next DeltaNet path, for an exclusive GPU window: one
//! decode token (m = 1) through the reused conv and this entry in the Flash-Next form, against
//! the vendored decode delta kernel (`gdn_delta`, the Qwen3.6 decode path) after the same conv,
//! on the same geometry and buffers. Every arm is sampled in the same rounds, one after the
//! other, so all arms see the same clocks and the same host. Reported, not gated; ignored, run
//! by name in a release build. The state updates of both timed entries are checked against the
//! f64 reference in the same process.
//!
//! The headline figure is one token's pass over all 36 DeltaNet layers, queued back to back on
//! one stream. Each queued layer has its own slot, as a real layer does: its recurrent state,
//! its conv ring, its conv weights, its per-head vectors and its projection row, each filled
//! with its own values. Every pass updates the states in place, so each pass reads the states
//! the previous one wrote. 36 Flash-Next states are 108 MiB, far beyond the 3090's 6 MB L2, so
//! no state is still cached when its layer comes round again. The projection rows are per layer
//! too, so they come from memory where a real step finds each in L2 just after its projection
//! wrote it; they are about 1% of a layer's bytes.
//!
//! The one-state arms repeat the earlier microbenchmark in the same window: all 36 queued
//! layers use slot 0, so its 3 MiB state stays in L2 (the method behind the 13.1 and 9.9 us per
//! layer of the first window). The two sets side by side measure what the 36-state working set
//! costs.
//!
//! Per arm and round: one layer queued and drained (latency, launch overhead included); the
//! 36-layer pass queued and drained (wall time, and the host time spent queueing it); and the
//! same pass with a timing event recorded before the first layer and after each layer (the
//! device time of the pass and of every layer in it). Every warm-up round runs every arm's full
//! pass. Effective bandwidth divides the bytes a layer must move (each range touched once, the
//! state and the conv ring read and written) by the device time of the pass.

use super::forms::{FN_K, FN_V, opts_of};
use super::{
    D, E3, E4, RefOpts, Rig, ScanLayout, ScanRig, ScanShape, assert_constraints, generate_scan,
    judge_scan_with, le_bytes, run_decode, seed_of,
};
use crate::EngError;
use crate::kernels::cuda::prefill::gdn_scan_seq::{
    GDN_SCAN_EPS, GdnForm, GdnScanSeqCall, ScanWeights, scan_bytes,
};
use crate::kernels::cuda::resident::gdn_conv::{ConvCall, ConvShape};
use crate::kernels::cuda::resident::gdn_delta::{DeltaCall, DeltaShape, DeltaWeights, delta_bytes};
use crate::kernels::cuda::{DeviceBuffer, TimingEvent, testkit};

/// The DeltaNet layers of Flash-Next, the queue depth of one decode token.
const LAYERS: usize = 36;
const ROUNDS: usize = 101;
const WARMUPS: usize = 10;
const F32: usize = 4;
/// The conv window: the ring holds the previous three rows.
const TAPS: usize = 4;

/// What one queued layer owns: its projection row (the conv input columns, then the per-head
/// columns), its conv weights and ring, its per-head vectors and its recurrent state.
struct Slot {
    proj: DeviceBuffer,
    w: DeviceBuffer,
    ring: DeviceBuffer,
    a: DeviceBuffer,
    dt: DeviceBuffer,
    nw: DeviceBuffer,
    state: DeviceBuffer,
}

/// One geometry at m = 1: a slot per queued layer, the scratch the layers share as a step's
/// layers do (the conv output and both scan outputs), and the timing marks of a pass.
struct Layer {
    rig: Rig,
    lay: ScanLayout,
    nk: usize,
    nv: usize,
    slots: Vec<Slot>,
    conv_out: DeviceBuffer,
    o_out: DeviceBuffer,
    d_out: DeviceBuffer,
    marks: Vec<TimingEvent>,
}

fn scan_shape(lay: ScanLayout, nk: usize, nv: usize) -> ScanShape {
    ScanShape {
        m: 1,
        p0: 0,
        ctx_tokens: 1024,
        n_k_heads: nk,
        n_v_heads: nv,
        head_dim: D,
        conv_ld: lay.conv_ld,
        ld: lay.ld,
        z_off: lay.z_off,
        beta_off: lay.beta_off,
        alpha_off: lay.alpha_off,
    }
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

impl Layer {
    fn new(nk: usize, nv: usize) -> Result<Self, EngError> {
        let lay = ScanLayout::of(nk, nv);
        let rig = Rig::new()?;
        let ctx = &rig.ctx;
        let up = |values: &[f32]| ctx.upload(&le_bytes(values));
        let channels = lay.conv_ld;
        let mut slots = Vec::with_capacity(LAYERS);
        let mut first_conv = Vec::new();
        for at in 0..LAYERS {
            let inp = generate_scan(nk, nv, 1, lay, seed_of(nk, nv, 1, 0xbe00 + at as u64));
            let mut proj = inp.proj.clone();
            proj[..channels].copy_from_slice(&inp.conv);
            let w: Vec<f32> = (0..channels * TAPS)
                .map(|i| 0.22 + 0.01 * ((i + at) % 7) as f32)
                .collect();
            slots.push(Slot {
                proj: up(&proj)?,
                w: up(&w)?,
                ring: up(&vec![0.0f32; (TAPS - 1) * channels])?,
                a: up(&inp.a)?,
                dt: up(&inp.dt)?,
                nw: up(&inp.nw)?,
                state: up(&inp.state0)?,
            });
            if at == 0 {
                first_conv = inp.conv;
            }
        }
        let need = scan_bytes(scan_shape(lay, nk, nv))?;
        let dneed = delta_bytes(delta_shape(lay, nk, nv))?;
        let marks = (0..=LAYERS)
            .map(|_| ctx.event_timing())
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            // Read by the scan-only arms before any conv arm has written it.
            conv_out: up(&first_conv)?,
            o_out: ctx.upload(&vec![0u8; need.o_out])?,
            d_out: ctx.upload(&vec![0u8; dneed.out])?,
            slots,
            marks,
            rig,
            lay,
            nk,
            nv,
        })
    }

    /// The conv of slot `at`.
    fn conv(&self, at: usize) -> Result<(), EngError> {
        let channels = self.lay.conv_ld;
        let slot = &self.slots[at];
        self.rig.ctx.gdn_conv(
            &self.rig.stream,
            ConvCall {
                shape: ConvShape {
                    m: 1,
                    channels,
                    ld: self.lay.ld,
                },
                proj: slot.proj.range(0, channels * F32)?,
                weights: slot.w.range(0, channels * TAPS * F32)?,
                state: slot.ring.range(0, (TAPS - 1) * channels * F32)?,
                out: self.conv_out.range(0, channels * F32)?,
            },
        )
    }

    /// The scan of slot `at` in `form`.
    fn scan(&self, form: GdnForm, at: usize) -> Result<(), EngError> {
        let shape = scan_shape(self.lay, self.nk, self.nv);
        let need = scan_bytes(shape)?;
        let slot = &self.slots[at];
        self.rig.ctx.gdn_scan_seq_form(
            &self.rig.stream,
            GdnScanSeqCall {
                shape,
                eps: GDN_SCAN_EPS,
                conv: self.conv_out.range(0, need.conv)?,
                proj: slot.proj.range(0, need.proj)?,
                weights: ScanWeights {
                    a: slot.a.range(0, need.a)?,
                    dt_bias: slot.dt.range(0, need.dt_bias)?,
                    norm_w: slot.nw.range(0, need.norm_w)?,
                },
                state: slot.state.range(0, need.state)?,
                o_out: self.o_out.range(0, need.o_out)?,
                status: None,
            },
            form,
        )
    }

    /// The vendored decode step of slot `at`.
    fn delta(&self, at: usize) -> Result<(), EngError> {
        let shape = delta_shape(self.lay, self.nk, self.nv);
        let need = delta_bytes(shape)?;
        let slot = &self.slots[at];
        self.rig.ctx.gdn_delta(
            &self.rig.stream,
            DeltaCall {
                shape,
                eps: GDN_SCAN_EPS,
                conv: self.conv_out.range(0, need.conv)?,
                proj: slot.proj.range(0, need.proj)?,
                weights: DeltaWeights {
                    a: slot.a.range(0, need.a)?,
                    dt_bias: slot.dt.range(0, need.dt_bias)?,
                    norm_w: slot.nw.range(0, need.norm_w)?,
                },
                state: slot.state.range(0, need.state)?,
                out: self.d_out.range(0, need.out)?,
            },
        )
    }

    /// The bytes one layer of `step` must move: (the state read and written, every range).
    fn traffic(&self, step: Step) -> (f64, f64) {
        let (nv, channels) = (self.nv, self.lay.conv_ld);
        // The ring read and written, the weights, the conv input columns, the conv output.
        let conv = (2 * (TAPS - 1) * channels + TAPS * channels + 2 * channels) * F32;
        let state = 2 * nv * D * D * F32;
        // The state, the conv output, the z, beta and alpha columns, a and dt_bias, the norm
        // weight and the output row.
        let scan = state + (channels + nv * D + 2 * nv + 2 * nv + D + nv * D) * F32;
        let (state, all) = match step {
            Step::Conv => (0, conv),
            Step::Scan(_) | Step::Delta => (state, scan),
            Step::ConvScan(_) | Step::ConvDelta => (state, conv + scan),
        };
        (state as f64, all as f64)
    }

    /// Every state after the timed rounds is finite (each pass feeds a slot the same token).
    fn states_are_finite(&self) -> Result<bool, EngError> {
        self.rig.stream.synchronize()?;
        for slot in &self.slots {
            let mut bytes = vec![0u8; slot.state.bytes()];
            slot.state.read_at(&self.rig.stream, 0, &mut bytes)?;
            if !testkit::floats(&bytes).iter().all(|v| v.is_finite()) {
                return Ok(false);
            }
        }
        Ok(true)
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

/// Which slots the 36 queued layers of a pass use.
#[derive(Clone, Copy)]
enum Set {
    /// Every queued layer on slot 0: one L2-resident state (the first window's method).
    One,
    /// Queued layer `i` on slot `i`, as a decode token's layers.
    PerLayer,
}

impl Set {
    fn slot(self, at: usize) -> usize {
        match self {
            Self::One => 0,
            Self::PerLayer => at,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::One => "1 state",
            Self::PerLayer => "36 states",
        }
    }
}

/// The samples of one arm, in microseconds.
struct Arm {
    name: &'static str,
    layer: usize,
    step: Step,
    set: Set,
    /// One layer queued and drained.
    one: Vec<f64>,
    /// The pass queued and drained, wall clock.
    wall: Vec<f64>,
    /// Host time to queue the pass, per layer.
    enqueue: Vec<f64>,
    /// The pass from its first to its last timing event.
    device: Vec<f64>,
    /// The device time of queued layer `i`, one sample per round.
    per_layer: Vec<Vec<f64>>,
}

/// Queues `step` on slot `at`.
fn queue(layer: &Layer, step: Step, at: usize) -> Result<(), EngError> {
    match step {
        Step::Conv => layer.conv(at),
        Step::Scan(form) => layer.scan(form, at),
        Step::Delta => layer.delta(at),
        Step::ConvScan(form) => {
            layer.conv(at)?;
            layer.scan(form, at)
        }
        Step::ConvDelta => {
            layer.conv(at)?;
            layer.delta(at)
        }
    }
}

/// `n` layers queued on the slots of `set`, then a drain: (host queue time, total time), in
/// microseconds.
fn sample(layer: &Layer, (step, set): (Step, Set), n: usize) -> Result<(f64, f64), EngError> {
    let t0 = std::time::Instant::now();
    for at in 0..n {
        queue(layer, step, set.slot(at))?;
    }
    let queued = t0.elapsed();
    layer.rig.stream.synchronize()?;
    let total = t0.elapsed();
    Ok((queued.as_secs_f64() * 1e6, total.as_secs_f64() * 1e6))
}

/// The pass with a timing mark before the first layer and after each: the device time of the
/// pass and of each layer, in microseconds.
fn marked(layer: &Layer, (step, set): (Step, Set)) -> Result<(f64, Vec<f64>), EngError> {
    let stream = &layer.rig.stream;
    stream.record_timing(&layer.marks[0])?;
    for at in 0..LAYERS {
        queue(layer, step, set.slot(at))?;
        stream.record_timing(&layer.marks[at + 1])?;
    }
    stream.synchronize()?;
    let us = |from: &TimingEvent, to: &TimingEvent| {
        from.elapsed_ms_since(to).map(|ms| f64::from(ms) * 1e3)
    };
    let each = layer
        .marks
        .windows(2)
        .map(|pair| us(&pair[0], &pair[1]))
        .collect::<Result<Vec<_>, _>>()?;
    Ok((us(&layer.marks[0], &layer.marks[LAYERS])?, each))
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
            let how = (arm.step, arm.set);
            let (_, one) = sample(layer, how, 1)?;
            let (queued, wall) = sample(layer, how, LAYERS)?;
            let (device, each) = marked(layer, how)?;
            if round >= WARMUPS {
                arm.one.push(one);
                arm.wall.push(wall);
                arm.enqueue.push(queued / LAYERS as f64);
                arm.device.push(device);
                for (samples, us) in arm.per_layer.iter_mut().zip(each) {
                    samples.push(us);
                }
            }
        }
    }
    Ok(())
}

/// The pass line of `arm`: device and wall time of the 36 layers, the host queue time, the
/// drained one-layer latency.
fn report_pass(arm: &Arm) {
    let (v10, v50, v90) = spread(&arm.device);
    let (w10, w50, w90) = spread(&arm.wall);
    let (_, e50, _) = spread(&arm.enqueue);
    let (_, o50, _) = spread(&arm.one);
    let ms = |us: f64| us / 1e3;
    println!(
        "BENCH {:<30} {:<9} pass of {LAYERS}: device {:.3} ms [{:.3}, {:.3}]  wall {:.3} ms [{:.3}, {:.3}]  host queue {e50:.2} us/layer  one layer drained {o50:.2} us",
        arm.name,
        arm.set.label(),
        ms(v50),
        ms(v10),
        ms(v90),
        ms(w50),
        ms(w10),
        ms(w90),
    );
}

/// The layer line of `arm`: the device time of a layer over every layer and round, the range of
/// the per-layer medians, and the effective bandwidth of the median pass.
fn report_layers(layer: &Layer, arm: &Arm) {
    let all: Vec<f64> = arm.per_layer.iter().flatten().copied().collect();
    let (l10, l50, l90) = spread(&all);
    let max = all.iter().copied().fold(0.0, f64::max);
    let medians: Vec<f64> = arm.per_layer.iter().map(|s| spread(s).1).collect();
    let lo = medians.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = medians.iter().copied().fold(0.0, f64::max);
    let pass = spread(&arm.device).1;
    let (state, every) = layer.traffic(arm.step);
    let gbs = |bytes: f64| bytes * LAYERS as f64 / (pass * 1e-6) / 1e9;
    println!(
        "BENCH {:<40}   per layer device {l50:.2} us [{l10:.2}, {l90:.2}] max {max:.2}; layer medians {lo:.2} to {hi:.2} us; state {:.0} GB/s, all ranges {:.0} GB/s",
        "",
        gbs(state),
        gbs(every),
    );
}

fn report(layers: &[Layer], arms: &[Arm]) {
    let profile = if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    };
    println!(
        "BENCH gdn decode ({profile}, m = 1, {ROUNDS} interleaved rounds after {WARMUPS} warm-up rounds of every pass; p50 [p10, p90])"
    );
    for arm in arms {
        report_pass(arm);
        report_layers(&layers[arm.layer], arm);
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
    let arm = |name: &'static str, layer: usize, step: Step, set: Set| Arm {
        name,
        layer,
        step,
        set,
        one: Vec::with_capacity(ROUNDS),
        wall: Vec::with_capacity(ROUNDS),
        enqueue: Vec::with_capacity(ROUNDS),
        device: Vec::with_capacity(ROUNDS),
        per_layer: vec![Vec::with_capacity(ROUNDS); LAYERS],
    };
    let fnx = GdnForm::FLASH_NEXT;
    let q36 = GdnForm::QWEN36_GGUF;
    let (one, each) = (Set::One, Set::PerLayer);
    let mut arms = [
        arm("16x48 conv + flash-next scan", 0, Step::ConvScan(fnx), each),
        arm("16x48 conv + vendored delta", 0, Step::ConvDelta, each),
        arm("16x48 conv + flash-next scan", 0, Step::ConvScan(fnx), one),
        arm("16x48 conv + vendored delta", 0, Step::ConvDelta, one),
        arm("16x48 conv", 0, Step::Conv, each),
        arm("16x48 flash-next scan", 0, Step::Scan(fnx), each),
        arm("16x48 vendored delta", 0, Step::Delta, each),
        arm("16x48 flash-next scan", 0, Step::Scan(fnx), one),
        arm("16x48 vendored delta", 0, Step::Delta, one),
        arm("16x32 qwen36-form scan", 1, Step::Scan(q36), each),
        arm("16x32 vendored delta", 1, Step::Delta, each),
    ];
    run_rounds(&layers, &mut arms)?;
    report(&layers, &arms);
    for layer in &layers {
        assert!(layer.states_are_finite()?, "a timed state went non-finite");
    }
    Ok(())
}
