//! EXL3 projection-kernel variants for the routed experts of one real Qwen3.8 Flash-Next layer on
//! one GPU. Measurement only: it adds no execution path and changes no executor, default or limit.
//!
//! The id-indexed projection launch (`CudaContext::exl3_grouped_indexed`: input rotation, the
//! projection kernel, output rotation) is timed for the layer's two expert shapes:
//! - gate/up, 2560 to 640. The pairs alternate the gate and up matrices of each pick, as in the
//!   MoE block.
//! - down, 640 to 2560.
//!
//! Each shape is timed at 1, 2, 4 and 8 column tiles per block (`cols`: the existing wide
//! variants), for several pair counts, and for the block's own launch sequences:
//! - 16 + 4 gate/up pairs, the block's split;
//! - 10 + 10 gate/up pairs, the even split;
//! - 10 down pairs.
//!
//! Cold launches read a new set of ten distinct experts each time. 51 sets rotate through the 512
//! experts, so a launch never finds its weights in the 6 MB L2. Warm launches point every pick at
//! one expert, whose matrices stay in L2 after the first launch. Warm runs the same instructions
//! with the DRAM traffic taken away, so the cold/warm gap bounds what memory costs.
//!
//! Each case runs `LAUNCHES` launches back to back, with a timing mark between launches.
//! - The stream first runs a device copy (the gate, about 2.5 ms) while the host queues the
//!   launches behind it, so the launches run back to back instead of at the host's pace.
//! - A run counts as late when the host finished queueing it after the device could have
//!   reached its start; its sample may hold a host gap. The test is conservative: it can count a
//!   clean run as late, never a gapped run as clean.
//! - Every trial runs all four variants of a case one after another, in an order that rotates
//!   from trial to trial, so clock drift and position spread over them. The first trial is a
//!   warm-up.
//! - Before timing, every variant's output rows are compared bit for bit with cols = 1 for the
//!   first and last cold pick sets and the warm set, for every launch. The output rows, the
//!   rotated inputs and the split-K partials are filled with NaN first, so a skipped write shows.
//! - The arithmetic flag must stay 0.
//!
//! Each sample is one run of a case: its launches (three kernels each) plus a timing mark. A
//! profiled run (Nsight Systems) splits it into kernels.

use std::{collections::HashSet, ops::Range, time::Instant};

use super::{
    exl3_cuda_tests::{Router, input, upload},
    graph_bench::Parts,
    residency_bench::{Res, clock_mark, device_us, emit, fail, open_layer},
};
use crate::{
    family::{
        project::cuda::{CudaProjector, exl3::Exl3Io},
        qwen36::cuda::moe::MoeDims,
    },
    kernels::cuda::{CudaContext, DeviceBuffer, DeviceStream, TimingEvent},
};

const F32: usize = size_of::<f32>();
const COLS: [u32; 4] = [1, 2, 4, 8];
/// Picks per id set: the layer's top-k.
const PICKS: usize = 10;
/// Cold id sets of distinct experts; one more set, the last, points every pick at expert 0.
const SETS: usize = 51;
const LAUNCHES: usize = 48;
const TRIALS: usize = 4;
/// The most pairs one id-indexed launch takes.
const MAX_PAIRS: usize = 16;
/// Bytes of the gate's device copy: about 2.5 ms at the 3090's copy rate, longer than the host
/// takes to queue the largest case.
const GATE_BYTES: usize = 1 << 30;

#[derive(Clone, Copy)]
enum Shape {
    GateUp,
    Down,
}

/// One timed case: launches of consecutive pairs (first pair, pair count; in pick order) on one
/// shape.
struct Case {
    name: &'static str,
    shape: Shape,
    launches: &'static [(usize, usize)],
    warm: bool,
}

const fn case(
    name: &'static str,
    shape: Shape,
    launches: &'static [(usize, usize)],
    warm: bool,
) -> Case {
    Case {
        name,
        shape,
        launches,
        warm,
    }
}

const CASES: [Case; 15] = [
    case("gateup_p2", Shape::GateUp, &[(0, 2)], false),
    case("gateup_p4", Shape::GateUp, &[(0, 4)], false),
    case("gateup_p8", Shape::GateUp, &[(0, 8)], false),
    case("gateup_p12", Shape::GateUp, &[(0, 12)], false),
    case("gateup_p16", Shape::GateUp, &[(0, 16)], false),
    case("gateup_tail_p4", Shape::GateUp, &[(16, 4)], false),
    case("gateup_seq_16_4", Shape::GateUp, &[(0, 16), (16, 4)], false),
    case(
        "gateup_seq_10_10",
        Shape::GateUp,
        &[(0, 10), (10, 10)],
        false,
    ),
    case("down_p2", Shape::Down, &[(0, 2)], false),
    case("down_p4", Shape::Down, &[(0, 4)], false),
    case("down_p8", Shape::Down, &[(0, 8)], false),
    case("down_p10", Shape::Down, &[(0, 10)], false),
    case("gateup_p16_warm", Shape::GateUp, &[(0, 16)], true),
    case("gateup_p4_warm", Shape::GateUp, &[(0, 4)], true),
    case("down_p10_warm", Shape::Down, &[(0, 10)], true),
];

struct Rig<'a> {
    ctx: &'a CudaContext,
    parts: Parts<'a>,
    dims: MoeDims,
    /// The gate/up input row (`hidden` values) and the down input row (`inter` values).
    x_up: DeviceBuffer,
    x_down: DeviceBuffer,
    /// Per-pair rows of the rotated inputs, the split-K partials and the outputs.
    xh: DeviceBuffer,
    z: DeviceBuffer,
    out: DeviceBuffer,
    flag: DeviceBuffer,
    /// `SETS + 1` id sets of `PICKS` ids.
    ids: DeviceBuffer,
    marks: Vec<TimingEvent>,
    /// The gate copy's buffers and its start mark; `marks[0]` ends it.
    gate_from: DeviceBuffer,
    gate_to: DeviceBuffer,
    gate_mark: TimingEvent,
    stream: DeviceStream,
}

/// The samples of one `Rig::time` call, how many of them are late, and the host's queueing time
/// per launch.
struct Timed {
    samples: Vec<f64>,
    late: usize,
    queue_us_per_launch: f64,
}

impl Rig<'_> {
    fn dims_of(&self, shape: Shape) -> (usize, usize, &DeviceBuffer) {
        match shape {
            Shape::GateUp => (self.dims.hidden, self.dims.inter, &self.x_up),
            Shape::Down => (self.dims.inter, self.dims.hidden, &self.x_down),
        }
    }

    /// Queues one id-indexed launch of the pairs `pairs` of `shape` with the id set `set`; pair
    /// number `row` of the launch writes output row `row`.
    fn launch(&self, shape: Shape, pairs: &Range<usize>, set: usize, cols: u32) -> Res<()> {
        let (k, n, x) = self.dims_of(shape);
        let ids = self.ids.range(set * PICKS * 4, PICKS * 4)?;
        let flag = self.flag.range(0, F32)?;
        let mut calls = Vec::with_capacity(pairs.len());
        for (row, pair) in pairs.clone().enumerate() {
            let (bank, slot) = match shape {
                Shape::GateUp => ([self.parts.gate, self.parts.up][pair % 2], pair / 2),
                Shape::Down => (self.parts.down, pair),
            };
            let io = Exl3Io {
                x: x.range(0, k * F32)?,
                xh: self.xh.range(row * k * F32, k * F32)?,
                z: self.z.range(row * n * F32, n * F32)?,
                out: self.out.range(row * n * F32, n * F32)?,
                flag,
                rows: 1,
                fused: false,
            };
            let slot = u32::try_from(slot).map_err(|_| fail("a pick slot past u32"))?;
            calls.push(bank.span()?.indexed_call(&io, ids, slot));
        }
        self.ctx.exl3_grouped_indexed(&self.stream, &calls, cols)
    }

    /// The output rows of one launch of `pairs` with the id set `set` under `cols`. The output
    /// rows, rotated inputs and split-K partials are NaN beforehand, so a skipped write of any of
    /// them reaches the output.
    fn outputs(&self, shape: Shape, pairs: &Range<usize>, set: usize, cols: u32) -> Res<Vec<u8>> {
        let (k, n, _) = self.dims_of(shape);
        let bytes = pairs.len() * n * F32;
        let nan = |values: usize| f32::NAN.to_le_bytes().repeat(values);
        self.out.write_at(&self.stream, 0, &nan(pairs.len() * n))?;
        self.z.write_at(&self.stream, 0, &nan(pairs.len() * n))?;
        self.xh.write_at(&self.stream, 0, &nan(pairs.len() * k))?;
        self.launch(shape, pairs, set, cols)?;
        self.stream.synchronize()?;
        let mut got = vec![0u8; bytes];
        self.out.read_at(&self.stream, 0, &mut got)?;
        Ok(got)
    }

    /// The sticky arithmetic flag; 0 while every projection stayed finite.
    fn flag(&self) -> Res<u32> {
        let mut raw = [0u8; 4];
        self.flag.read_at(&self.stream, 0, &mut raw)?;
        Ok(u32::from_le_bytes(raw))
    }

    /// `LAUNCHES` runs of `case` under `cols` back to back behind the gate; the device time of
    /// each run.
    fn time(&self, case: &Case, cols: u32, trial: usize) -> Res<Timed> {
        // Taken before the gate's start mark is queued: the device cannot pass that mark earlier.
        let gate_queued = Instant::now();
        self.stream.record_timing(&self.gate_mark)?;
        self.ctx.copy_device(
            &self.stream,
            self.gate_from.range(0, GATE_BYTES)?,
            self.gate_to.range(0, GATE_BYTES)?,
        )?;
        let queue_start = Instant::now();
        let mut queued = Vec::with_capacity(LAUNCHES);
        for run in 0..LAUNCHES {
            self.stream.record_timing(&self.marks[run])?;
            let set = if case.warm {
                SETS
            } else {
                (trial * LAUNCHES + run) % SETS
            };
            for &(first, count) in case.launches {
                self.launch(case.shape, &(first..first + count), set, cols)?;
            }
            queued.push(Instant::now());
        }
        self.stream.record_timing(&self.marks[LAUNCHES])?;
        self.stream.synchronize()?;
        // Run `run` is clean when the host had queued all of it before the device could reach
        // its start mark, at the earliest `gate_queued` plus the device's offset of that mark.
        let mut late = 0;
        for (run, at) in queued.iter().enumerate().skip(1) {
            let host = at.duration_since(gate_queued).as_secs_f64() * 1e6;
            if host >= device_us(&self.gate_mark, &self.marks[run])? {
                late += 1;
            }
        }
        let launches = LAUNCHES * case.launches.len();
        let queue_us = queued[LAUNCHES - 1]
            .duration_since(queue_start)
            .as_secs_f64()
            * 1e6;
        // The first run of a warm case still reads its weights from DRAM.
        Ok(Timed {
            samples: (1..LAUNCHES)
                .map(|run| device_us(&self.marks[run], &self.marks[run + 1]))
                .collect::<Res<_>>()?,
            late,
            queue_us_per_launch: queue_us / launches as f64,
        })
    }
}

/// `Err` unless every variant writes the bytes cols = 1 writes, for every launch of every case,
/// with the first and last cold id sets and the warm set.
fn check(rig: &Rig<'_>) -> Res<usize> {
    let mut compared = 0;
    for (case, set) in CASES
        .iter()
        .flat_map(|case| [0, SETS - 1, SETS].map(|set| (case, set)))
    {
        for &(first, count) in case.launches {
            let pairs = &(first..first + count);
            let want = rig.outputs(case.shape, pairs, set, 1)?;
            if want
                .as_chunks::<4>()
                .0
                .iter()
                .any(|b| !f32::from_le_bytes(*b).is_finite())
            {
                return Err(fail(format!(
                    "{}: cols 1 left a non-finite output with id set {set}",
                    case.name
                )));
            }
            for cols in &COLS[1..] {
                if rig.outputs(case.shape, pairs, set, *cols)? != want {
                    return Err(fail(format!(
                        "{}: cols {cols} differs from cols 1 for pairs {pairs:?}, id set {set}",
                        case.name
                    )));
                }
                compared += 1;
            }
        }
    }
    let flag = rig.flag()?;
    if flag != 0 {
        return Err(fail(format!("the arithmetic flag is {flag}")));
    }
    Ok(compared)
}

/// One case and variant over the timed trials.
#[derive(Clone, Default)]
struct Pooled {
    samples: Vec<f64>,
    late: usize,
    queue_us: Vec<f64>,
}

fn timing(rig: &Rig<'_>) -> Res<()> {
    let mut pooled: Vec<Vec<Pooled>> = CASES
        .iter()
        .map(|_| vec![Pooled::default(); COLS.len()])
        .collect();
    for trial in 0..=TRIALS {
        clock_mark(&format!("projection_trial{trial}"));
        for (case, into) in CASES.iter().zip(&mut pooled) {
            for turn in 0..COLS.len() {
                let variant = (turn + trial) % COLS.len();
                let got = rig.time(case, COLS[variant], trial)?;
                if trial > 0 {
                    let all = &mut into[variant];
                    all.samples.extend(got.samples);
                    all.late += got.late;
                    all.queue_us.push(got.queue_us_per_launch);
                }
            }
        }
    }
    for (case, got) in CASES.iter().zip(&pooled) {
        let pairs: usize = case.launches.iter().map(|&(_, count)| count).sum();
        for (cols, all) in COLS.iter().zip(got) {
            let queue = all.queue_us.iter().sum::<f64>() / all.queue_us.len() as f64;
            let extra = format!(
                ",\"pairs\":{pairs},\"launches\":{},\"warm\":{},\"late_runs\":{},\"host_queue_us_per_launch\":{queue:.2}",
                case.launches.len(),
                case.warm,
                all.late
            );
            emit(
                "projection",
                &format!("{}_c{cols}", case.name),
                "us",
                &all.samples,
                &extra,
            );
        }
    }
    let flag = rig.flag()?;
    if flag != 0 {
        return Err(fail(format!("the arithmetic flag is {flag} after timing")));
    }
    Ok(())
}

/// `SETS` sets of `PICKS` experts, no expert twice over all of them, then the warm set, every
/// pick expert 0. `Err` when the bank is too small for that.
fn id_sets(n_expert: usize) -> Res<Vec<u8>> {
    let cold: Vec<usize> = (0..SETS * PICKS).map(|i| (i * 37) % n_expert).collect();
    if cold.iter().collect::<HashSet<_>>().len() != cold.len() {
        return Err(fail(format!(
            "{n_expert} experts cannot fill {SETS} cold sets of {PICKS} distinct experts"
        )));
    }
    let warm = std::iter::repeat_n(0, PICKS);
    cold.into_iter()
        .chain(warm)
        .map(|id| i32::try_from(id).map_err(|_| fail("an expert id past i32")))
        .collect::<Res<Vec<i32>>>()
        .map(|ids| ids.iter().flat_map(|id| id.to_le_bytes()).collect())
}

// ------------------------------------------------------------------ the test

#[test]
#[ignore = "timing: needs an exclusive GPU window (4 GiB) and FIFI_FLASH_NEXT_EXL3_PACK"]
fn flash_next_exl3_projection_variants_microbench() {
    run().expect("the EXL3 projection variant microbenchmark");
}

fn run() -> Res<()> {
    let (checkpoint, layer, moe) = open_layer("Flash-Next EXL3 projection variant microbenchmark")?;
    if moe.top_k != PICKS {
        return Err(fail("the cases assume ten picks"));
    }
    let projector = CudaProjector::new()?;
    let ctx = projector.context();
    let router = Router::read(&checkpoint, layer, &moe)?;
    let weights = upload(&projector, &checkpoint, (layer, &router), &moe)?;
    let floats =
        |values: Vec<f32>| -> Vec<u8> { values.iter().flat_map(|v| v.to_le_bytes()).collect() };
    let zeros = |bytes: usize| ctx.upload(&vec![0u8; bytes]);
    let wide = moe.hidden.max(moe.inter);
    let rig = Rig {
        ctx,
        parts: Parts::resolve(&weights, layer, &moe)?,
        dims: moe,
        x_up: ctx.upload(&floats(input(moe.hidden, 0)))?,
        x_down: ctx.upload(&floats(input(moe.inter, 1)))?,
        xh: zeros(MAX_PAIRS * wide * F32)?,
        z: zeros(MAX_PAIRS * wide * F32)?,
        out: zeros(MAX_PAIRS * wide * F32)?,
        flag: zeros(F32)?,
        ids: ctx.upload(&id_sets(moe.n_expert)?)?,
        marks: (0..=LAUNCHES)
            .map(|_| ctx.event_timing())
            .collect::<Res<_>>()?,
        gate_from: ctx.allocate(GATE_BYTES)?,
        gate_to: ctx.allocate(GATE_BYTES)?,
        gate_mark: ctx.event_timing()?,
        stream: ctx.stream()?,
    };
    let compared = check(&rig)?;
    println!(
        "RESBENCH {{\"phase\":\"projection_check\",\"case\":\"variants_equal_cols1\",\"compared\":{compared}}}"
    );
    timing(&rig)
}
