//! The dense single-row EXL3 projections of one Qwen3.8 Flash-Next decode token on real weights:
//! the Gated DeltaNet in and out projections (layer 0), the QSA q, k, v, indexer and output
//! projections (layer 3), the shared expert (layer 0) and `lm_head`, each timed per call on every
//! single-row kernel path the engine has. Measurement only: no executor, default or limit changes.
//!
//! Paths (`CudaContext::set_single_row_default`, `FIFI_EXL3_SINGLE_GATHER` must be unset):
//! - `legacy`: the shuffle kernel Flash-Next runs today (input rotation, `project_small`, output
//!   rotation);
//! - `gather`: the per-warp shared-table gather kernel;
//! - `wide1`: the one-tile wide kernel that Qwen3.6 and Flint serve;
//! - `wide1_r56`: `wide1` with the compile-time rate 10 and rate 6 body of book 2, the kernel the
//!   Flash-Next decode opts into;
//! - `wide1_r56_dp4a`: `wide1_r56` with the DP4A decode of book 2.
//!
//! Each case reads its matrix from the pack once and uploads it in as many copies as make a pool of
//! up to 96 MiB (2 copies at least, 48 at most; the head is 2), so a call never finds its weights
//! in the 6 MB L2 for the shapes of the table. A trial
//! issues `ROUNDS` passes over the pool between two timing events. Trials rotate the order of the
//! paths, and the first trial is a warm-up.
//!
//! Before timing, every path's output row, rotated input and split-K partial are compared byte for
//! byte with `legacy` (NaN-filled first, so a skipped write shows) for the first and last copy, and
//! the arithmetic flag of every result must stay 0. A path that differs is reported, not timed, and
//! fails the test after the last case.
//!
//! Without a device the test only reads the pack, checks each matrix's byte length against the
//! packed-tile formula and prints the shape table (the dry run).
//!
//! Stock figures (an nsys profile of exllamav3 decoding 32 greedy tokens of a short chat prompt on
//! this pack, one row, in microseconds per call) are printed beside the cases they cover: the in
//! projections `qkv` and `z` together 44 + 3.6 us, `out` 21 us, `q`, `k`, `v` together 50 us, the
//! head 466 us.

use std::{path::Path, time::Instant};

use super::{
    exl3_cuda_tests::{input, matrix},
    residency_bench::{Res, device_us, fail},
};
use crate::{
    family::project::cuda::{CudaProjector, Reserve, exl3::Exl3Io},
    kernels::{
        cuda::{
            CudaContext, DeviceBuffer, DeviceStream, TimingEvent, device_present,
            resident::exl3::IndexedOptions,
        },
        single_row::SingleRow,
    },
    weights::exl3::{Checkpoint, Matrix},
};

const F32: usize = size_of::<f32>();
/// Weight bytes a case's pool spans, at least.
const POOL_BYTES: usize = 96 << 20;
const MAX_COPIES: usize = 48;
const ROUNDS: usize = 4;
const TRIALS: usize = 5;
/// DRAM bandwidth the floor column assumes, bytes per microsecond (870 GB/s).
const FLOOR_BYTES_PER_US: f64 = 870_000.0;

/// One matrix of the decode token: the tensor prefix under `layer`, how many such calls one token
/// makes, and the stock time of the call that covers it (0 when stock folds it into another).
struct Case {
    name: &'static str,
    prefix: &'static str,
    layer: Layer,
    per_token: usize,
    stock_us: f64,
}

#[derive(Clone, Copy)]
enum Layer {
    /// A Gated DeltaNet layer (layer 0 unless `FIFI_FLASH_NEXT_DENSE_GDN_LAYER`).
    Gdn,
    /// A QSA layer (layer 3 unless `FIFI_FLASH_NEXT_EXL3_MOE_LAYER`).
    Qsa,
    /// A tensor outside the layers.
    None,
}

const CASES: [Case; 12] = [
    case("gdn_qkv", "linear_attn.in_proj_qkv", Layer::Gdn, 36, 47.6),
    case("gdn_z", "linear_attn.in_proj_z", Layer::Gdn, 36, 0.0),
    case("gdn_out", "linear_attn.out_proj", Layer::Gdn, 36, 21.0),
    case("qsa_q", "self_attn.q_proj", Layer::Qsa, 12, 50.0),
    case("qsa_k", "self_attn.k_proj", Layer::Qsa, 12, 0.0),
    case("qsa_v", "self_attn.v_proj", Layer::Qsa, 12, 0.0),
    case(
        "qsa_index",
        "self_attn.indexer.index_qk_proj",
        Layer::Qsa,
        12,
        0.0,
    ),
    case("qsa_o", "self_attn.o_proj", Layer::Qsa, 12, 21.0),
    case(
        "shared_gate",
        "mlp.shared_expert.gate_proj",
        Layer::Gdn,
        48,
        0.0,
    ),
    case(
        "shared_up",
        "mlp.shared_expert.up_proj",
        Layer::Gdn,
        48,
        0.0,
    ),
    case(
        "shared_down",
        "mlp.shared_expert.down_proj",
        Layer::Gdn,
        48,
        0.0,
    ),
    case("lm_head", "lm_head", Layer::None, 1, 466.0),
];

/// A grouped input-projection launch of a mixer: the member cases in group order, the layers
/// the group runs in per token, and the stock's microseconds for the whole group (the sum of the
/// members' stock rows).
struct Group {
    name: &'static str,
    members: &'static [&'static str],
    per_token: usize,
    stock_us: f64,
}

const GROUPS: [Group; 2] = [
    Group {
        name: "gdn_in",
        members: &["gdn_qkv", "gdn_z"],
        per_token: 36,
        stock_us: 47.6,
    },
    Group {
        name: "qsa_in",
        members: &["qsa_q", "qsa_k", "qsa_v", "qsa_index"],
        per_token: 12,
        stock_us: 50.0,
    },
];

const fn case(
    name: &'static str,
    prefix: &'static str,
    layer: Layer,
    per_token: usize,
    stock_us: f64,
) -> Case {
    Case {
        name,
        prefix,
        layer,
        per_token,
        stock_us,
    }
}

/// A path: name, kernel, whether the one-tile launch may take the rate 10 and rate 6 body of
/// book 2, and whether that body decodes book 2 with DP4A.
type BenchPath = (&'static str, SingleRow, bool, bool);

const PATHS: [BenchPath; 5] = [
    ("legacy", SingleRow::Legacy, false, false),
    ("gather", SingleRow::Gather, false, false),
    ("wide1", SingleRow::Wide(1), false, false),
    ("wide1_r56", SingleRow::Wide(1), true, false),
    ("wide1_r56_dp4a", SingleRow::Wide(1), true, true),
];

/// Selects path `index` on the context.
fn select(ctx: &CudaContext, index: usize) {
    ctx.set_single_row_default(PATHS[index].1);
    ctx.set_odd_rate_kernel(PATHS[index].2);
    ctx.set_indexed_options(Some(if PATHS[index].3 {
        IndexedOptions::DP4A
    } else {
        IndexedOptions::OFF
    }));
}

fn layer_index(var: &str, default: usize) -> Res<usize> {
    std::env::var_os(var).map_or(Ok(default), |value| {
        value
            .to_str()
            .and_then(|text| text.parse().ok())
            .ok_or_else(|| fail(format!("{var} must be a layer number")))
    })
}

/// The cases `FIFI_FLASH_NEXT_DENSE_CASES` names (comma separated), all of them when it is unset.
/// A value that is not UTF-8 or a name that matches no case is refused.
fn selected_cases() -> Res<Vec<&'static Case>> {
    let Some(value) = std::env::var_os("FIFI_FLASH_NEXT_DENSE_CASES") else {
        return Ok(CASES.iter().collect());
    };
    let names = value
        .to_str()
        .ok_or_else(|| fail("FIFI_FLASH_NEXT_DENSE_CASES must be UTF-8"))?;
    names
        .split(',')
        .map(|name| {
            CASES.iter().find(|case| case.name == name).ok_or_else(|| {
                fail(format!(
                    "FIFI_FLASH_NEXT_DENSE_CASES: no case named {name:?}"
                ))
            })
        })
        .collect()
}

/// The groups `FIFI_FLASH_NEXT_DENSE_CASES` names; without the variable, all. Group names ride
/// the same variable as case names, so a run can pick one group alone or mix cases and groups.
fn selected_groups() -> Res<Vec<&'static Group>> {
    let Some(value) = std::env::var_os("FIFI_FLASH_NEXT_DENSE_CASES") else {
        return Ok(GROUPS.iter().collect());
    };
    let names = value
        .to_str()
        .ok_or_else(|| fail("FIFI_FLASH_NEXT_DENSE_CASES must be UTF-8"))?;
    Ok(names
        .split(',')
        .filter_map(|name| GROUPS.iter().find(|group| group.name == name))
        .collect())
}

/// The tensor prefix of `case`.
fn full_prefix(case: &Case) -> Res<String> {
    Ok(match case.layer {
        Layer::Gdn => format!(
            "model.language_model.layers.{}.{}",
            layer_index("FIFI_FLASH_NEXT_DENSE_GDN_LAYER", 0)?,
            case.prefix
        ),
        Layer::Qsa => format!(
            "model.language_model.layers.{}.{}",
            layer_index("FIFI_FLASH_NEXT_EXL3_MOE_LAYER", 3)?,
            case.prefix
        ),
        Layer::None => case.prefix.to_owned(),
    })
}

/// Facts of one matrix read from the pack.
struct Shape {
    k: usize,
    n: usize,
    bytes: usize,
    bits_twice: usize,
}

fn shape_of(matrix: &Matrix, name: &str) -> Res<Shape> {
    let [k, n] = matrix.shape();
    let bytes = matrix
        .trellis()
        .host_bytes()
        .ok_or_else(|| fail(format!("{name}: no host trellis")))?
        .len();
    let per_rate = k * n / 256 * 16;
    if per_rate == 0 || bytes % per_rate != 0 {
        return Err(fail(format!(
            "{name}: {bytes} trellis bytes do not fit {k} x {n} tiles"
        )));
    }
    Ok(Shape {
        k,
        n,
        bytes,
        bits_twice: bytes / per_rate,
    })
}

fn print_shape(case: &Case, shape: &Shape) {
    let floor = shape.bytes as f64 / FLOOR_BYTES_PER_US;
    println!(
        "DENSEBENCH {{\"phase\":\"shape\",\"case\":\"{}\",\"k\":{},\"n\":{},\"bits_twice\":{},\
         \"bytes\":{},\"per_token\":{},\"floor_us\":{floor:.1},\"stock_us\":{}}}",
        case.name, shape.k, shape.n, shape.bits_twice, shape.bytes, case.per_token, case.stock_us
    );
}

/// One case on the device: the pool of uploaded copies and the buffers a call writes.
struct Rig<'a> {
    ctx: &'a CudaContext,
    stream: DeviceStream,
    weights: Vec<crate::family::project::cuda::exl3::Exl3Weight>,
    x: DeviceBuffer,
    xh: DeviceBuffer,
    z: DeviceBuffer,
    out: DeviceBuffer,
    flag: DeviceBuffer,
    shape: Shape,
    marks: [TimingEvent; 2],
}

impl<'a> Rig<'a> {
    fn new(projector: &'a CudaProjector, matrix: &Matrix, shape: Shape) -> Res<Self> {
        let ctx = projector.context();
        let copies = POOL_BYTES.div_ceil(shape.bytes).clamp(2, MAX_COPIES);
        let weights = (0..copies)
            .map(|_| projector.upload_exl3(matrix, Reserve::default()))
            .collect::<Res<Vec<_>>>()?;
        let floats = |count: usize| ctx.allocate(count * F32);
        let x: Vec<u8> = input(shape.k, 0)
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        Ok(Self {
            ctx,
            stream: ctx.stream()?,
            weights,
            x: ctx.upload(&x)?,
            xh: floats(shape.k)?,
            z: floats(shape.n)?,
            out: floats(shape.n)?,
            flag: ctx.upload(&0u32.to_le_bytes())?,
            marks: [ctx.event_timing()?, ctx.event_timing()?],
            shape,
        })
    }

    fn call(&self, copy: usize) -> Res<()> {
        let io = Exl3Io {
            x: self.x.range(0, self.shape.k * F32)?,
            xh: self.xh.range(0, self.shape.k * F32)?,
            z: self.z.range(0, self.shape.n * F32)?,
            out: self.out.range(0, self.shape.n * F32)?,
            flag: self.flag.range(0, F32)?,
            rows: 1,
            fused: false,
        };
        let view = self.weights[copy].view()?;
        self.ctx.exl3_linear(&self.stream, view.linear_call(&io))
    }

    /// The bytes one call on weight copy `copy` writes: rotated input, split-K partial, output, flag.
    fn written(&self, copy: usize) -> Res<Vec<u8>> {
        let (k, n) = (self.shape.k, self.shape.n);
        let nan = f32::NAN.to_le_bytes();
        for (buffer, len) in [(&self.xh, k), (&self.z, n), (&self.out, n)] {
            buffer.write_at(&self.stream, 0, &nan.repeat(len))?;
        }
        self.call(copy)?;
        self.stream.synchronize()?;
        let mut bytes = vec![0u8; (k + 2 * n) * F32 + F32];
        let mut at = 0;
        for (buffer, len) in [
            (&self.xh, k * F32),
            (&self.z, n * F32),
            (&self.out, n * F32),
            (&self.flag, F32),
        ] {
            buffer.read_at(&self.stream, 0, &mut bytes[at..at + len])?;
            at += len;
        }
        Ok(bytes)
    }

    /// Microseconds per call over `ROUNDS` passes of the pool, and the host's microseconds to queue a call.
    fn time(&self) -> Res<(f64, f64)> {
        let copies = self.weights.len();
        self.stream.record_timing(&self.marks[0])?;
        let queue = Instant::now();
        for _ in 0..ROUNDS {
            for copy in 0..copies {
                self.call(copy)?;
            }
        }
        let queued = queue.elapsed().as_secs_f64() * 1e6;
        self.stream.record_timing(&self.marks[1])?;
        self.stream.synchronize()?;
        let calls = (ROUNDS * copies) as f64;
        Ok((
            device_us(&self.marks[0], &self.marks[1])? / calls,
            queued / calls,
        ))
    }
}

/// One group on the device: pools of the member matrices and the summed buffers a grouped launch
/// writes. The members share one input row (the same `k`); the scratch and output slices follow
/// group order, as the mixer's `project_group` lays them out.
struct GroupRig<'a> {
    ctx: &'a CudaContext,
    stream: DeviceStream,
    weights: Vec<Vec<crate::family::project::cuda::exl3::Exl3Weight>>,
    x: DeviceBuffer,
    xh: DeviceBuffer,
    z: DeviceBuffer,
    out: DeviceBuffer,
    flag: DeviceBuffer,
    shapes: Vec<Shape>,
    marks: [TimingEvent; 2],
}

impl<'a> GroupRig<'a> {
    fn new(projector: &'a CudaProjector, matrices: &[&Matrix], shapes: Vec<Shape>) -> Res<Self> {
        let ctx = projector.context();
        let bytes: usize = shapes.iter().map(|shape| shape.bytes).sum();
        let copies = POOL_BYTES.div_ceil(bytes).clamp(2, MAX_COPIES);
        let k = shapes[0].k;
        if shapes.iter().any(|shape| shape.k != k) {
            return Err(fail(
                "a group's members read different input widths".to_string(),
            ));
        }
        let weights = (0..copies)
            .map(|_| {
                matrices
                    .iter()
                    .map(|&matrix| projector.upload_exl3(matrix, Reserve::default()))
                    .collect::<Res<Vec<_>>>()
            })
            .collect::<Res<Vec<_>>>()?;
        let n_total: usize = shapes.iter().map(|shape| shape.n).sum();
        let x: Vec<u8> = input(k, 0).iter().flat_map(|v| v.to_le_bytes()).collect();
        let floats = |count: usize| ctx.allocate(count * F32);
        Ok(Self {
            ctx,
            stream: ctx.stream()?,
            weights,
            x: ctx.upload(&x)?,
            xh: floats(shapes.len() * k)?,
            z: floats(n_total)?,
            out: floats(n_total)?,
            flag: ctx.upload(&0u32.to_le_bytes())?,
            shapes,
            marks: [ctx.event_timing()?, ctx.event_timing()?],
        })
    }

    /// The call of member `index` on weight copy `copy`, with the group-order scratch slices.
    fn call(&self, copy: usize, index: usize) -> Res<()> {
        let (k, n) = (self.shapes[index].k, self.shapes[index].n);
        let (xh_at, z_at) = self.offsets(index);
        let io = Exl3Io {
            x: self.x.range(0, k * F32)?,
            xh: self.xh.range(xh_at, k * F32)?,
            z: self.z.range(z_at, n * F32)?,
            out: self.out.range(z_at, n * F32)?,
            flag: self.flag.range(0, F32)?,
            rows: 1,
            fused: false,
        };
        let view = self.weights[copy][index].view()?;
        self.ctx.exl3_linear(&self.stream, view.linear_call(&io))
    }

    /// The whole group on weight copy `copy` as one `exl3_grouped_step` launch. The views and
    /// their io outlive the calls that borrow them: two passes, as in `project_group`.
    fn group(&self, copy: usize) -> Res<()> {
        let mut members = Vec::with_capacity(self.shapes.len());
        for index in 0..self.shapes.len() {
            let (k, n) = (self.shapes[index].k, self.shapes[index].n);
            let (xh_at, z_at) = self.offsets(index);
            members.push((
                self.weights[copy][index].view()?,
                Exl3Io {
                    x: self.x.range(0, k * F32)?,
                    xh: self.xh.range(xh_at, k * F32)?,
                    z: self.z.range(z_at, n * F32)?,
                    out: self.out.range(z_at, n * F32)?,
                    flag: self.flag.range(0, F32)?,
                    rows: 1,
                    fused: false,
                },
            ));
        }
        let calls: Vec<_> = members
            .iter()
            .map(|(view, io)| view.linear_call(io))
            .collect();
        self.ctx.exl3_grouped_step(&self.stream, &calls)
    }

    /// The float offsets of member `index` in the scratch buffers.
    fn offsets(&self, index: usize) -> (usize, usize) {
        let xh_at = self.shapes[..index]
            .iter()
            .map(|shape| shape.k)
            .sum::<usize>()
            * F32;
        let z_at = self.shapes[..index]
            .iter()
            .map(|shape| shape.n)
            .sum::<usize>()
            * F32;
        (xh_at, z_at)
    }

    /// The output and flag bytes one run of `mode` on copy `copy` writes, over a NaN fill.
    fn written(&self, copy: usize, grouped: bool) -> Res<Vec<u8>> {
        let n_total: usize = self.shapes.iter().map(|shape| shape.n).sum();
        let nan = f32::NAN.to_le_bytes();
        for (buffer, len) in [
            (&self.xh, self.shapes.len() * self.shapes[0].k),
            (&self.z, n_total),
            (&self.out, n_total),
        ] {
            buffer.write_at(&self.stream, 0, &nan.repeat(len))?;
        }
        if grouped {
            self.group(copy)?;
        } else {
            for index in 0..self.shapes.len() {
                self.call(copy, index)?;
            }
        }
        self.stream.synchronize()?;
        let mut bytes = vec![0u8; n_total * F32 + F32];
        self.out
            .read_at(&self.stream, 0, &mut bytes[..n_total * F32])?;
        self.flag
            .read_at(&self.stream, 0, &mut bytes[n_total * F32..])?;
        Ok(bytes)
    }

    /// Microseconds per whole group (one grouped launch, or every member once) over `ROUNDS`
    /// passes of the pool, and the host's microseconds to queue one pass.
    fn time(&self, grouped: bool) -> Res<(f64, f64)> {
        let copies = self.weights.len();
        self.stream.record_timing(&self.marks[0])?;
        let queue = Instant::now();
        for _ in 0..ROUNDS {
            for copy in 0..copies {
                if grouped {
                    self.group(copy)?;
                } else {
                    for index in 0..self.shapes.len() {
                        self.call(copy, index)?;
                    }
                }
            }
        }
        let queued = queue.elapsed().as_secs_f64() * 1e6;
        self.stream.record_timing(&self.marks[1])?;
        self.stream.synchronize()?;
        let passes = (ROUNDS * copies) as f64;
        Ok((
            device_us(&self.marks[0], &self.marks[1])? / passes,
            queued / passes,
        ))
    }
}

fn run() -> Res<()> {
    let dir = std::env::var_os("FIFI_FLASH_NEXT_EXL3_PACK")
        .ok_or_else(|| fail("FIFI_FLASH_NEXT_EXL3_PACK"))?;
    if std::env::var_os("FIFI_EXL3_SINGLE_GATHER").is_some() {
        return Err(fail(
            "the bench chooses the single-row kernel itself: FIFI_EXL3_SINGLE_GATHER must be unset",
        ));
    }
    let checkpoint = Checkpoint::open(Path::new(&dir))?;
    let cases = selected_cases()?;
    let groups = selected_groups()?;
    let device = device_present("Flash-Next dense projection microbenchmark");
    let projector = device.then(CudaProjector::new).transpose()?;
    let mut mismatched = Vec::new();
    for case in cases {
        let name = full_prefix(case)?;
        let matrix = matrix(&checkpoint, &name)?;
        let shape = shape_of(&matrix, &name)?;
        print_shape(case, &shape);
        let Some(projector) = projector.as_ref() else {
            continue;
        };
        let rig = Rig::new(projector, &matrix, shape)?;
        if !measure(case, &rig)? {
            mismatched.push(case.name);
        }
    }
    for group in groups {
        let Some(projector) = projector.as_ref() else {
            break;
        };
        let member = |name: &str| -> Res<&'static Case> {
            CASES
                .iter()
                .find(|case| case.name == name)
                .ok_or_else(|| fail(format!("{name}: not a case")))
        };
        let matrices = group
            .members
            .iter()
            .map(|&name| matrix(&checkpoint, &full_prefix(member(name)?)?))
            .collect::<Res<Vec<_>>>()?;
        let shapes = matrices
            .iter()
            .map(|matrix| shape_of(matrix, group.name))
            .collect::<Res<Vec<_>>>()?;
        let matrices: Vec<&Matrix> = matrices.iter().collect();
        if !measure_group(group, projector, &matrices, &shapes)? {
            mismatched.push(group.name);
        }
    }
    if mismatched.is_empty() {
        return Ok(());
    }
    Err(fail(format!(
        "a path wrote other bytes than the legacy kernel in: {mismatched:?}"
    )))
}

/// Checks every path against `legacy`, then times the paths that agree. `false` when a path wrote
/// other bytes.
fn measure(case: &Case, rig: &Rig<'_>) -> Res<bool> {
    let last = rig.weights.len() - 1;
    let mut want = Vec::with_capacity(2);
    select(rig.ctx, 0);
    for copy in [0, last] {
        want.push(rig.written(copy)?);
    }
    let mut agree = [true; PATHS.len()];
    for (index, (name, _, _, _)) in PATHS.iter().enumerate().skip(1) {
        select(rig.ctx, index);
        for (slot, copy) in [0, last].into_iter().enumerate() {
            if rig.written(copy)? != want[slot] {
                agree[index] = false;
                println!(
                    "DENSEBENCH {{\"phase\":\"identity\",\"case\":\"{}\",\"path\":\"{name}\",\"equal\":false}}",
                    case.name
                );
            }
        }
    }
    if want.iter().any(|bytes| bytes[bytes.len() - 4..] != [0; 4]) {
        return Err(fail(format!("{}: the arithmetic flag is set", case.name)));
    }
    let mut samples: Vec<Vec<(f64, f64)>> = vec![Vec::new(); PATHS.len()];
    for trial in 0..TRIALS {
        for step in 0..PATHS.len() {
            let index = (trial + step) % PATHS.len();
            if !agree[index] {
                continue;
            }
            select(rig.ctx, index);
            let timed = rig.time()?;
            if trial > 0 {
                samples[index].push(timed);
            }
        }
    }
    report(case, rig, &samples, &agree);
    Ok(agree.iter().all(|&same| same))
}

/// Checks one group: the grouped launch must write the member-wise bytes, and must count as one
/// wide launch of the routed shape. Then times the group against its separate members. `false`
/// when the bytes or the launch counts disagree.
fn measure_group(
    group: &Group,
    projector: &CudaProjector,
    matrices: &[&Matrix],
    shapes: &[Shape],
) -> Res<bool> {
    let rig = GroupRig::new(projector, matrices, shapes.to_vec())?;
    let last = rig.weights.len() - 1;
    let mut ok = true;
    // The wide body comes in three served flavors (plain, rate 10/6 compile-time body, that body
    // with DP4A); the grouped launch must write the member-wise bytes on each of them.
    for index in [2, 3, 4] {
        select(rig.ctx, index);
        for &copy in &[0, last.min(1)] {
            let (name, want) = (PATHS[index].0, rig.written(copy, false)?);
            if rig.written(copy, true)? != want {
                ok = false;
                println!(
                    "DENSEBENCH {{\"phase\":\"identity\",\"case\":\"{}\",\"path\":\"grouped-{name}\",\"equal\":false}}",
                    group.name
                );
            }
        }
    }
    // The launch proof: one grouped call is exactly one wide1 launch of the routed shape; the
    // members separately are `members.len()` of them. `exl3_grouped_step` chooses the kernel the
    // context default picks, like every other launch in this bench.
    select(rig.ctx, 2);
    let launch_delta = |grouped: bool| -> Res<u64> {
        let before = rig.ctx.exl3_launch_counts().wide1;
        if grouped {
            rig.group(0)?;
        } else {
            for index in 0..shapes.len() {
                rig.call(0, index)?;
            }
        }
        rig.stream.synchronize()?;
        Ok(rig.ctx.exl3_launch_counts().wide1 - before)
    };
    let separate_launches = launch_delta(false)?;
    let grouped_launches = launch_delta(true)?;
    if separate_launches != shapes.len() as u64 || grouped_launches != 1 {
        return Err(fail(format!(
            "{}: separate mode took {separate_launches} launches, grouped took {grouped_launches}",
            group.name
        )));
    }
    let (separate_us, separate_queued) = median_of(|| rig.time(false))?;
    let (grouped_us, grouped_queued) = median_of(|| rig.time(true))?;
    println!(
        "DENSEBENCH {{\"phase\":\"group\",\"case\":\"{}\",\"members\":{},\"launches\":{{\"separate\":{separate_launches},\"grouped\":{grouped_launches}}},\"bytes_equal\":{ok},\"separate_us\":{separate_us:.2},\"grouped_us\":{grouped_us:.2},\"separate_queued_us\":{separate_queued:.2},\"grouped_queued_us\":{grouped_queued:.2},\"per_token\":{},\"stock_us\":{}}}",
        group.name,
        group.members.len(),
        group.per_token,
        group.stock_us
    );
    Ok(ok)
}

/// The median `(device, queued)` of `TRIALS - 1` timed samples.
fn median_of<F: FnMut() -> Res<(f64, f64)>>(mut trial: F) -> Res<(f64, f64)> {
    let mut samples = Vec::new();
    for index in 0..TRIALS {
        let timed = trial()?;
        if index > 0 {
            samples.push(timed);
        }
    }
    let pick = |side: fn(&(f64, f64)) -> f64| -> f64 {
        let mut values: Vec<f64> = samples.iter().map(side).collect();
        values.sort_by(|a, b| a.total_cmp(b));
        values[values.len() / 2]
    };
    Ok((pick(|s| s.0), pick(|s| s.1)))
}

fn report(case: &Case, rig: &Rig<'_>, samples: &[Vec<(f64, f64)>], agree: &[bool]) {
    let median = |index: usize| -> Option<(f64, f64)> {
        let mut device: Vec<f64> = samples[index].iter().map(|s| s.0).collect();
        device.sort_by(f64::total_cmp);
        let mut queue: Vec<f64> = samples[index].iter().map(|s| s.1).collect();
        queue.sort_by(f64::total_cmp);
        Some((*device.get(device.len() / 2)?, *queue.get(queue.len() / 2)?))
    };
    let legacy = median(0).map(|m| m.0);
    for (index, (name, _, _, _)) in PATHS.iter().enumerate() {
        let Some((us, queue_us)) = median(index) else {
            continue;
        };
        let gbps = rig.shape.bytes as f64 / us / 1e3;
        let ratio = legacy.map_or(f64::NAN, |l| l / us);
        println!(
            "DENSEBENCH {{\"phase\":\"time\",\"case\":\"{}\",\"path\":\"{name}\",\"identical\":{},\
             \"us\":{us:.2},\"gbps\":{gbps:.1},\"vs_legacy\":{ratio:.2},\"ms_per_token\":{:.3},\
             \"host_queue_us\":{queue_us:.2},\"copies\":{}}}",
            case.name,
            agree[index],
            us * case.per_token as f64 / 1e3,
            rig.weights.len()
        );
    }
}

#[test]
#[ignore = "needs FIFI_FLASH_NEXT_EXL3_PACK; timing needs an exclusive GPU window (about 6 GiB), a host run is the dry run"]
fn flash_next_dense_projection_microbench() {
    run().expect("the dense projection microbenchmark");
}
