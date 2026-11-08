//! The MoE block of a prefill chunk on an EXL3 pack: the shared expert and the routed experts of
//! `m` rows, then the combine.
//!
//! The job table of the routed projections lives on the device but is built on the host, so a
//! chunk reads the router's picks back once per layer (the router has queued them on the device),
//! sorts its `m * top_k` pairs by expert, and uploads the two row lists and the table in one
//! write. Gate, up and down then run each as ONE job-table launch over every expert of the
//! chunk (three kernels, whatever the number of experts), and one SwiGLU runs over every pair.
//! A job is at most [`TILE_ROWS`] rows of one expert; the expert's weights are read once per job,
//! which makes the routed pass layer-major: its weight traffic follows the experts a chunk picks,
//! not its rows.
//!
//! No row is copied to sort it. The input rotation of gate and up reads the rows of the normalized
//! chunk through the first list (the row each sorted pair reads), and the output rotation of down
//! writes each sorted pair straight to its `(row, rank)` place of `paired`, the layout the combine
//! reads, through the second list. `perm` holds the two lists and the job table, written by one
//! host upload.
//!
//! Stream rules the table relies on. Every launch of the block, the upload of the lists and the
//! table (`write_at` waits for the stream, so it never overtakes earlier work) and the read of the
//! picks run on the one stream of the step, in this order; no other stream touches the faces or the
//! scratch while a chunk runs, and the table is read by the kernels when they run, so it must not
//! be rewritten before the stream has passed the down projection (the next layer's upload comes
//! after this layer's combine on the same stream). A second stream, or a graph replay that
//! overlaps two layers, would need one table per layer in flight.
//!
//! Gate and up write their halves of the `[gate | up]` rows of `gu` directly, through the row
//! stride and column offset of the job-table output, the layout the SwiGLU reads.

use super::{Output, Projector, window};
use crate::family::qwen36::cuda::prefill_plan::EXL3_EXPERT_TILE_ROWS as TILE_ROWS;
use crate::{
    EngError,
    family::{
        project::cuda::{Exl3JobsIo, exl3::Exl3Bank},
        qwen36::cuda::{
            exl3_moe::Exl3MoeWeights,
            moe::MoeDims,
            prefill::{chunk_combine_shape, routed_swiglu, shared_swiglu},
        },
    },
    kernels::cuda::{
        DeviceRange,
        prefill::swiglu_rows::SwigluRowsCall,
        resident::{
            exl3::{Exl3Job, Exl3JobsCall},
            moe_combine::{CombineCall, CombineOut, SharedTerm},
        },
    },
};

const I32: usize = size_of::<i32>();
const F32: usize = size_of::<f32>();

pub(super) struct Bufs<'b> {
    /// F32 `[m][hidden]`: the normalized rows, read only.
    pub(super) x2: DeviceRange<'b>,
    /// i32 `[m][top_k]`: the router's picks, read only.
    pub(super) ids: DeviceRange<'b>,
    /// F32 `[m][top_k]`: the router's pick weights, read only.
    pub(super) weights: DeviceRange<'b>,
    /// F32 `[m]`: the shared expert's gate, read only.
    pub(super) shared_gate: DeviceRange<'b>,
    /// F32 `[m][hidden]`: the residual, updated in place by the combine.
    pub(super) hidden: DeviceRange<'b>,
    /// u32 `[2][m * top_k]`: for each sorted pair the row it reads, then the pair row it lands in;
    /// then the job table.
    pub(super) perm: DeviceRange<'b>,
    /// F32 `[m * top_k][2 * inter]`.
    pub(super) gu: DeviceRange<'b>,
    /// F32 `[m * top_k][inter]`.
    pub(super) act: DeviceRange<'b>,
    /// F32 `[m * top_k][hidden]`.
    pub(super) paired: DeviceRange<'b>,
    /// F16 `[m * top_k][max(hidden, inter)]`: the rotated input rows of one routed projection.
    pub(super) expert_xh: DeviceRange<'b>,
    /// F32 `[m * top_k][max(hidden, inter)]`: its partials.
    pub(super) expert_z: DeviceRange<'b>,
    /// F32 `[m][2 * shared_inter]`.
    pub(super) shared_gu: DeviceRange<'b>,
    /// F32 `[m][shared_inter]`.
    pub(super) shared_act: DeviceRange<'b>,
    /// F32 `[m][hidden]`.
    pub(super) shared_out: DeviceRange<'b>,
}

#[derive(Debug, PartialEq, Eq)]
pub(super) struct PairPlan {
    /// Pairs of each expert.
    pub(super) counts: Vec<usize>,
    /// For each sorted position, the row of the chunk it reads.
    pub(super) order: Vec<i32>,
    /// For each sorted position, its pair `row * top_k + rank`: the place its output lands in.
    pub(super) scatter: Vec<i32>,
}

/// Counting sort of the pairs by expert: within an expert the rows keep their chunk order. A pick
/// outside `0..n_expert` is the router's fault mark (-1) or a corrupt word, and refuses the chunk.
pub(super) fn plan_pairs(
    picks: &[i32],
    n_expert: usize,
    top_k: usize,
) -> Result<PairPlan, EngError> {
    let mut counts = vec![0usize; n_expert];
    for &pick in picks {
        let expert = usize::try_from(pick).ok().filter(|&e| e < n_expert);
        let Some(expert) = expert else {
            return Err(EngError::ModelOutput(
                "EXL3 moe chunk: the router picked no valid expert (non-finite logits)".into(),
            ));
        };
        counts[expert] += 1;
    }
    let mut next = Vec::with_capacity(n_expert);
    let mut at = 0;
    for &count in &counts {
        next.push(at);
        at += count;
    }
    let (mut order, mut scatter) = (vec![0i32; picks.len()], vec![0i32; picks.len()]);
    for (pair, &pick) in picks.iter().enumerate() {
        let slot = &mut next[pick as usize];
        order[*slot] = i32::try_from(pair / top_k).map_err(|_| too_many())?;
        scatter[*slot] = i32::try_from(pair).map_err(|_| too_many())?;
        *slot += 1;
    }
    Ok(PairPlan {
        counts,
        order,
        scatter,
    })
}

fn too_many() -> EngError {
    EngError::BadRequest("EXL3 moe chunk: the pair count exceeds i32".into())
}

/// Reads the picks of the chunk back, which waits for the stream, and sorts them.
fn read_plan(
    projector: &Projector<'_>,
    ids: DeviceRange<'_>,
    dims: &MoeDims,
    bank: Option<&mut crate::family::qwen36::cuda::exl3_bank::LayerResolver<'_, '_>>,
) -> Result<PairPlan, EngError> {
    let pairs = dims.rows * dims.top_k;
    let mut raw = vec![0u8; pairs * I32];
    ids.buffer()
        .read_at(projector.launch.stream, ids.offset(), &mut raw)?;
    let picks: Vec<i32> = raw
        .as_chunks::<I32>()
        .0
        .iter()
        .map(|bytes| i32::from_le_bytes(*bytes))
        .collect();
    // A banked block maps its picks through the layer's slot table before the sort: the
    // grouped launches then read the slot members of the bank region exactly as a resident
    // block reads its expert ids. A fault-marked id (-1) falls through to the sort's refusal.
    let picks = match bank {
        Some(bank) if !picks.iter().any(|&id| id < 0) => {
            let wanted = picks.iter().map(|&id| id as usize).collect::<Vec<_>>();
            bank.resolve(projector.launch.ctx, &wanted)?
                .into_iter()
                .map(|slot| slot as i32)
                .collect()
        }
        _ => picks,
    };
    plan_pairs(&picks, dims.n_expert, dims.top_k)
}

/// The jobs of a routed pass: each expert with rows, in expert order, cut into tiles of at most
/// [`TILE_ROWS`] rows. The rows of consecutive jobs are back to back, which is the layout of the
/// sorted order of the lists.
pub(super) fn plan_jobs(counts: &[usize]) -> Vec<Exl3Job> {
    let mut jobs = Vec::new();
    let mut row0 = 0usize;
    for (expert, &count) in counts.iter().enumerate() {
        let mut left = count;
        while left > 0 {
            let rows = left.min(TILE_ROWS);
            jobs.push(Exl3Job {
                member: expert as u32,
                row0: row0 as u32,
                rows: rows as u32,
            });
            row0 += rows;
            left -= rows;
        }
    }
    jobs
}

fn write_lists(
    projector: &Projector<'_>,
    perm: DeviceRange<'_>,
    (plan, jobs): (&PairPlan, &[Exl3Job]),
) -> Result<(), EngError> {
    let mut bytes: Vec<u8> = plan
        .order
        .iter()
        .chain(&plan.scatter)
        .flat_map(|index| index.to_le_bytes())
        .collect();
    bytes.extend(Exl3Job::table(jobs));
    perm.buffer()
        .write_at(projector.launch.stream, perm.offset(), &bytes)
}

/// The shared expert: gate and up into the halves of `shared_gu`, the SwiGLU, down.
fn shared(
    projector: &Projector<'_>,
    weights: &Exl3MoeWeights<'_>,
    bufs: &Bufs<'_>,
    dims: &MoeDims,
) -> Result<(), EngError> {
    let (m, s) = (dims.rows, dims.shared_inter);
    for (half, weight) in [weights.shared_gate, weights.shared_up]
        .into_iter()
        .enumerate()
    {
        let out = Output::Columns {
            dst: bufs.shared_gu,
            ld: 2 * s,
            col_off: half * s,
        };
        projector.project(weight, bufs.x2, out)?;
    }
    projector.launch.ctx.swiglu_rows(
        projector.launch.stream,
        SwigluRowsCall {
            shape: shared_swiglu(m, dims),
            gu: bufs.shared_gu,
            act: bufs.shared_act,
        },
    )?;
    projector.project(
        weights.shared_down,
        bufs.shared_act,
        Output::Rows(bufs.shared_out),
    )
}

struct Routed<'a, 'b> {
    bank: &'a Exl3Bank,
    x: DeviceRange<'b>,
    /// The output face, its row stride and its first column.
    out: (DeviceRange<'b>, usize, usize),
    /// The row lists of the call: `x` is read through the first (over `x_rows` rows) and the
    /// output is written through the second (over `out_rows` rows); `None` is the table order.
    lists: Lists<'b>,
}

/// The row lists of one routed projection (`exl3.h`).
#[derive(Clone, Copy)]
struct Lists<'b> {
    input: Option<DeviceRange<'b>>,
    output: Option<DeviceRange<'b>>,
    x_rows: usize,
    out_rows: usize,
}

#[derive(Clone, Copy)]
struct Table<'a, 'b> {
    jobs: &'a [Exl3Job],
    range: DeviceRange<'b>,
    rows: usize,
}

/// The rotated-row and partial-sum ranges one routed projection works in.
#[derive(Clone, Copy)]
struct Scratch<'b> {
    xh: DeviceRange<'b>,
    z: DeviceRange<'b>,
}

/// One routed projection over every job: three launches whatever the number of experts.
fn project_jobs(
    projector: &Projector<'_>,
    bufs: &Bufs<'_>,
    table: Table<'_, '_>,
    call: Routed<'_, '_>,
) -> Result<(), EngError> {
    #[cfg(test)]
    super::split::count("routed job-table projections");
    let scratch = Scratch {
        xh: bufs.expert_xh,
        z: bufs.expert_z,
    };
    let launch = jobs_call(projector, scratch, table, call)?;
    projector
        .launch
        .ctx
        .exl3_jobs(projector.launch.stream, launch)
}

/// The job-table call of one routed projection, over the ranges of `scratch`.
fn jobs_call<'b>(
    projector: &Projector<'b>,
    scratch: Scratch<'b>,
    table: Table<'_, 'b>,
    call: Routed<'b, 'b>,
) -> Result<Exl3JobsCall<'b>, EngError> {
    let (out, out_ld, out_col) = call.out;
    let io = Exl3JobsIo {
        x: call.x,
        xh: scratch.xh,
        z: scratch.z,
        out,
        flag: projector.scratch.flag,
        jobs: table.range,
        jobs_count: table.jobs.len(),
        max_rows: table
            .jobs
            .iter()
            .map(|job| job.rows as usize)
            .max()
            .unwrap_or(1),
        total_rows: table.rows,
        out_ld,
        out_col,
    };
    let members = call.bank.members()?;
    let mut launch = members.jobs_call(&io);
    launch.in_rows = call.lists.input;
    launch.out_rows = call.lists.output;
    launch.x_rows = call.lists.x_rows;
    launch.out_total_rows = call.lists.out_rows;
    Ok(launch)
}

/// Where the pair rotation puts its scratch inside the partial-sum range of `range_bytes` bytes.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
struct PairLayout {
    /// Bytes of partial sums one projection needs (`pairs * inter` values), at the head.
    z_bytes: usize,
    /// Offset of the second rotated-row range: `z_bytes` rounded up to `PAIR_ALIGN`.
    tail: usize,
    /// Bytes of that range (`pairs * hidden` halves).
    xh_bytes: usize,
}

/// Alignment of the second rotated-row range, in bytes.
const PAIR_ALIGN: usize = 256;

/// The layout of the pair's scratch for `pairs` rows of a projection with `inter` outputs over
/// `hidden` inputs, or `None` when the second rotated-row range does not fit behind the partial
/// sums in `range_bytes` bytes (which is the case when `inter` is more than half of `hidden`
/// for the usual `pairs * max(hidden, inter)` values of F32) or the sizes overflow.
fn pair_layout(
    (pairs, inter, hidden): (usize, usize, usize),
    range_bytes: usize,
) -> Option<PairLayout> {
    const F16: usize = 2;
    const F32: usize = 4;
    let z_bytes = pairs.checked_mul(inter)?.checked_mul(F32)?;
    let tail = z_bytes.checked_next_multiple_of(PAIR_ALIGN)?;
    let xh_bytes = pairs.checked_mul(hidden)?.checked_mul(F16)?;
    (tail.checked_add(xh_bytes)? <= range_bytes).then_some(PairLayout {
        z_bytes,
        tail,
        xh_bytes,
    })
}

/// The scratch ranges of the gate and up projections when they rotate their input rows together:
/// the first projection keeps its usual rotated-row range and the head of the partial-sum range,
/// the second takes its rotated rows from the unused tail of the partial-sum range
/// ([`pair_layout`]), so the pair needs no buffer of its own. `None` when the tail is too small.
fn paired_scratch<'b>(
    bufs: &Bufs<'b>,
    dims: &MoeDims,
) -> Result<Option<(Scratch<'b>, Scratch<'b>)>, EngError> {
    let pairs = dims.rows * dims.top_k;
    let Some(layout) = pair_layout((pairs, dims.inter, dims.hidden), bufs.expert_z.bytes()) else {
        return Ok(None);
    };
    let z = window(bufs.expert_z, 0, layout.z_bytes)?;
    let first = Scratch {
        xh: bufs.expert_xh,
        z,
    };
    let second = Scratch {
        xh: window(bufs.expert_z, layout.tail, layout.xh_bytes)?,
        z,
    };
    Ok(Some((first, second)))
}

/// The routed experts: the sort, gate and up reading the chunk rows through a list, one SwiGLU,
/// down writing each pair row to its place through a list.
fn routed(
    projector: &Projector<'_>,
    weights: &Exl3MoeWeights<'_>,
    bufs: &Bufs<'_>,
    dims: &MoeDims,
    bank: Option<&mut crate::family::qwen36::cuda::exl3_bank::LayerResolver<'_, '_>>,
) -> Result<(), EngError> {
    routed_with(
        projector,
        weights,
        bufs,
        dims,
        projector.launch.ctx.exl3_jobs_pair_enabled()?,
    )
    .map(|_| ())
}

/// [`routed`] with the pair request given; returns whether the gate and up projections rotated
/// their rows together (the request was on, the scratch fit, and the buffers were aligned).
fn routed_with(
    projector: &Projector<'_>,
    weights: &Exl3MoeWeights<'_>,
    bufs: &Bufs<'_>,
    dims: &MoeDims,
    pair: bool,
) -> Result<bool, EngError> {
    let (m, pairs, i) = (dims.rows, dims.rows * dims.top_k, dims.inter);
    let plan = read_plan(projector, bufs.ids, dims, bank)?;
    let jobs = plan_jobs(&plan.counts);
    write_lists(projector, bufs.perm, (&plan, &jobs))?;
    super::split::mark(projector.launch.stream, "ids read and sort");
    let list = |at: usize| window(bufs.perm, at * pairs * I32, pairs * I32);
    let table = Table {
        jobs: &jobs,
        range: window(bufs.perm, 2 * pairs * I32, jobs.len() * 16)?,
        rows: pairs,
    };
    // Gate and up read the rows of the chunk through the first list, each sorted pair once.
    let listed = Lists {
        input: Some(list(0)?),
        output: None,
        x_rows: m,
        out_rows: pairs,
    };
    let gate_up = |half: usize, bank| Routed {
        bank,
        x: bufs.x2,
        out: (bufs.gu, 2 * i, half * i),
        lists: listed,
    };
    // With the pair request on, the rows are read and rotated once for both projections.
    let paired = if pair {
        paired_scratch(bufs, dims)?
    } else {
        None
    };
    let mut taken = false;
    if let Some((first, second)) = paired {
        #[cfg(test)]
        for _ in 0..2 {
            super::split::count("routed job-table projections");
        }
        let calls = (
            jobs_call(projector, first, table, gate_up(0, weights.gate))?,
            jobs_call(projector, second, table, gate_up(1, weights.up))?,
        );
        taken = projector.launch.ctx.exl3_jobs_pair_mode(
            projector.launch.stream,
            calls,
            (true, None),
        )?;
    } else {
        for (half, bank) in [weights.gate, weights.up].into_iter().enumerate() {
            project_jobs(projector, bufs, table, gate_up(half, bank))?;
        }
    }
    super::split::mark(projector.launch.stream, "routed gate/up");
    projector.launch.ctx.swiglu_rows(
        projector.launch.stream,
        SwigluRowsCall {
            shape: routed_swiglu(m, dims),
            gu: bufs.gu,
            act: bufs.act,
        },
    )?;
    super::split::mark(projector.launch.stream, "routed swiglu");
    // Down writes each sorted pair to its (row, rank) place through the second list.
    let call = Routed {
        bank: weights.down,
        x: bufs.act,
        out: (bufs.paired, dims.hidden, 0),
        lists: Lists {
            input: None,
            output: Some(list(1)?),
            x_rows: pairs,
            out_rows: pairs,
        },
    };
    project_jobs(projector, bufs, table, call)?;
    super::split::mark(projector.launch.stream, "routed down");
    Ok(taken)
}

/// `hidden = hidden + (picks in rank order + gate * shared)` for every row, in place.
fn combine(projector: &Projector<'_>, bufs: &Bufs<'_>, dims: &MoeDims) -> Result<(), EngError> {
    projector.launch.ctx.moe_combine(
        projector.launch.stream,
        CombineCall {
            shape: chunk_combine_shape(dims, dims.rows),
            res: bufs.hidden,
            down: bufs.paired,
            weights: bufs.weights,
            shared: Some(SharedTerm {
                gate: bufs.shared_gate,
                row: bufs.shared_out,
            }),
            out: CombineOut::InPlace,
        },
    )
}

/// Stages 10-17 of one layer of a chunk, after the router has written the picks: the shared
/// expert, the routed experts and the combine. It waits for the stream once, to read the picks.
pub(super) fn run(
    projector: &Projector<'_>,
    weights: &Exl3MoeWeights<'_>,
    bufs: &Bufs<'_>,
    dims: &MoeDims,
) -> Result<(), EngError> {
    shared(projector, weights, bufs, dims)?;
    super::split::mark(projector.launch.stream, "shared expert");
    routed(projector, weights, bufs, dims, None)?;
    combine(projector, bufs, dims)?;
    super::split::mark(projector.launch.stream, "combine");
    Ok(())
}

/// The per-row slice of every row-scaled buffer of `bufs` at chunk row `r0`, for `sub.rows`
/// rows. The `perm` scratch stays whole: each sub-block rewrites and consumes it on the one
/// stream before the next sub-block's launches, so no slice is needed.
fn slice_at<'b>(
    bufs: &Bufs<'b>,
    full: &MoeDims,
    sub: &MoeDims,
    r0: usize,
) -> Result<Bufs<'b>, EngError> {
    let (h, i, k, s) = (full.hidden, full.inter, full.top_k, full.shared_inter);
    let wide = h.max(i);
    let f = |range: &DeviceRange<'b>, row: usize| {
        range
            .buffer()
            .range(range.offset() + r0 * row, sub.rows * row)
    };
    Ok(Bufs {
        x2: f(&bufs.x2, h * F32)?,
        ids: f(&bufs.ids, k * I32)?,
        weights: f(&bufs.weights, k * F32)?,
        shared_gate: f(&bufs.shared_gate, F32)?,
        hidden: f(&bufs.hidden, h * F32)?,
        perm: bufs.perm,
        gu: f(&bufs.gu, k * 2 * i * F32)?,
        act: f(&bufs.act, k * i * F32)?,
        paired: f(&bufs.paired, k * h * F32)?,
        expert_xh: f(&bufs.expert_xh, k * wide * 2)?,
        expert_z: f(&bufs.expert_z, k * wide * F32)?,
        shared_gu: f(&bufs.shared_gu, 2 * s * F32)?,
        shared_act: f(&bufs.shared_act, s * F32)?,
        shared_out: f(&bufs.shared_out, h * F32)?,
    })
}

/// Stages 10-17 of one banked chunk layer: the shared expert once for the whole chunk, then the
/// routed experts and the combine per sub-block, because the launches of one block read every
/// pick of its rows together and the bank holds only `slots` members. Each sub-block's picks
/// are read back, mapped through the slot table (refilling missed slots from the host store)
/// and grouped as slot ids. `sub` carries each sub-block's first row and its dims, built by the
/// caller from its own chunk model.
pub(super) fn run_banked(
    projector: &Projector<'_>,
    weights: &Exl3MoeWeights<'_>,
    bufs: &Bufs<'_>,
    dims: &MoeDims,
    sub: &[(usize, MoeDims)],
    bank: &mut crate::family::qwen36::cuda::exl3_bank::LayerResolver<'_, '_>,
) -> Result<(), EngError> {
    shared(projector, weights, bufs, dims)?;
    super::split::mark(projector.launch.stream, "shared expert");
    for &(r0, ref sub_dims) in sub {
        let sub_bufs = slice_at(bufs, dims, sub_dims, r0)?;
        routed(projector, weights, &sub_bufs, sub_dims, Some(bank))?;
        combine(projector, &sub_bufs, sub_dims)?;
    }
    super::split::mark(projector.launch.stream, "combine");
    Ok(())
}

#[cfg(test)]
#[path = "prefill_exl3_moe_device_tests.rs"]
mod device_tests;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_pair_layout_puts_the_second_rotated_rows_behind_the_partial_sums() {
        // Qwen3.6 shape: hidden 2048, inter 512, 16,384 pairs, partial sums of pairs * 2048 F32.
        let range = 16_384 * 2048 * 4;
        let layout = pair_layout((16_384, 512, 2048), range).expect("fits");
        assert_eq!(layout.z_bytes, 16_384 * 512 * 4);
        assert_eq!(layout.tail, layout.z_bytes, "already a multiple of 256");
        assert_eq!(layout.xh_bytes, 16_384 * 2048 * 2);
        assert!(layout.tail + layout.xh_bytes <= range);
        // The tail rounds up to 256 bytes.
        let layout = pair_layout((3, 100, 64), 1 << 20).expect("fits");
        assert_eq!(
            (layout.z_bytes, layout.tail, layout.xh_bytes),
            (1200, 1280, 384)
        );
    }

    #[test]
    fn the_pair_layout_fits_exactly_and_refuses_one_byte_less() {
        let need = 1280 + 384;
        assert!(pair_layout((3, 100, 64), need).is_some());
        assert_eq!(pair_layout((3, 100, 64), need - 1), None);
        assert_eq!(pair_layout((3, 100, 64), 0), None);
    }

    #[test]
    fn the_pair_layout_refuses_an_inter_above_half_of_hidden() {
        // The partial-sum range is pairs * max(hidden, inter) F32: the head takes 4 * inter and the
        // second rotated rows 2 * hidden of the 4 * hidden bytes a pair row owns.
        for (hidden, inter, expect) in [
            (2048, 1024, true),
            (2048, 1025, false),
            (2048, 2048, false),
            (512, 2048, false),
            (4096, 2048, true),
        ] {
            let pairs = 64;
            let range = pairs * hidden.max(inter) * 4;
            assert_eq!(
                pair_layout((pairs, inter, hidden), range).is_some(),
                expect,
                "hidden {hidden} inter {inter}"
            );
        }
    }

    #[test]
    fn the_pair_layout_refuses_sizes_that_overflow() {
        assert_eq!(pair_layout((usize::MAX, 2, 2), usize::MAX), None);
        assert_eq!(pair_layout((2, usize::MAX, 2), usize::MAX), None);
        assert_eq!(pair_layout((2, 2, usize::MAX), usize::MAX), None);
    }

    #[test]
    fn the_sort_groups_pairs_by_expert_and_keeps_chunk_order_within_one() {
        // Three rows, two picks each, over four experts.
        let picks = [2, 0, 2, 3, 0, 2];
        let plan = plan_pairs(&picks, 4, 2).expect("plan");
        assert_eq!(plan.counts, [2, 0, 3, 1]);
        // Expert 0: pairs 1 and 4 (rows 0 and 2); expert 2: pairs 0, 2 and 5 (rows 0, 1, 2);
        // expert 3: pair 3 (row 1).
        assert_eq!(plan.order, [0, 2, 0, 1, 2, 1]);
        // Each sorted position lands in the pair it came from.
        assert_eq!(plan.scatter, [1, 4, 0, 2, 5, 3]);
        for (place, &pair) in plan.scatter.iter().enumerate() {
            assert_eq!(
                plan.order[place] as usize,
                pair as usize / 2,
                "place {place}"
            );
            assert_eq!(
                picks[pair as usize],
                [0, 0, 2, 2, 2, 3][place],
                "place {place}"
            );
        }
        // The scatter is a permutation of the pairs.
        let mut seen = plan.scatter.clone();
        seen.sort_unstable();
        assert_eq!(seen, (0..6).collect::<Vec<i32>>());
    }

    /// The job rows of `counts`: `(member, row0, rows)`.
    fn jobs_of(counts: &[usize]) -> Vec<(u32, u32, u32)> {
        plan_jobs(counts)
            .iter()
            .map(|job| (job.member, job.row0, job.rows))
            .collect()
    }

    /// An expert's rows are cut at the tile height and the cut is exact at the edges of the
    /// kernel's narrowest CTA: 1, 8, 9, 32, 63 and 64 rows are one job, 65 rows are a full tile and
    /// a one-row tail, and the last short tile of a larger expert keeps the remainder.
    /// Mutation: cutting at `TILE_ROWS + 1` (or `TILE_ROWS - 1`) rows fails the 65 and 64 rows.
    #[test]
    fn jobs_cut_at_the_tile_height_and_keep_the_short_tail() {
        assert_eq!(
            TILE_ROWS, 64,
            "the literals below are written for 64-row tiles"
        );
        for rows in [1usize, 8, 9, 32, 63, 64] {
            assert_eq!(jobs_of(&[rows]), [(0, 0, rows as u32)], "{rows} rows");
        }
        assert_eq!(jobs_of(&[65]), [(0, 0, 64), (0, 64, 1)]);
        assert_eq!(jobs_of(&[129]), [(0, 0, 64), (0, 64, 64), (0, 128, 1)]);
        assert_eq!(jobs_of(&[150]), [(0, 0, 64), (0, 64, 64), (0, 128, 22)]);
    }

    /// Experts keep their order, an expert without rows has no job (a zero-row job is outside the
    /// kernel's limits), and the first row of a job is the first sorted row of its expert.
    /// Mutation: pushing a job for an empty expert fails the empty-expert row; restarting `row0`
    /// at zero for each expert fails the boundaries.
    #[test]
    fn jobs_follow_expert_order_skip_empty_experts_and_start_at_the_expert_boundary() {
        let counts = [0, 5, 64, 65, 0, 150, 1];
        assert_eq!(
            jobs_of(&counts),
            [
                (1, 0, 5),
                (2, 5, 64),
                (3, 69, 64),
                (3, 133, 1),
                (5, 134, 64),
                (5, 198, 64),
                (5, 262, 22),
                (6, 284, 1),
            ]
        );
        assert!(plan_jobs(&[0, 0, 0]).is_empty());
        assert_eq!(jobs_of(&[0, 0, 7]), [(2, 0, 7)]);
    }

    /// The jobs of any counts tile the sorted rows exactly once: each job lies inside the rows,
    /// the jobs are back to back and in order (so no two jobs share an input row, a scratch row
    /// or an output row), and each is within the kernel's limits.
    /// Mutation: dropping the `row0 += rows` advance, or letting a tail job repeat its tile,
    /// breaks the back-to-back check.
    #[test]
    fn jobs_tile_the_sorted_rows_exactly_once_for_any_counts() {
        let mut seed = 0x9e37_79b9_u64;
        let mut next = move |limit: u64| {
            seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (seed >> 33) % limit
        };
        for round in 0..200 {
            let experts = 1 + next(256) as usize;
            let counts: Vec<usize> = (0..experts)
                .map(|_| match next(4) {
                    0 => 0,
                    1 => 1 + next(8) as usize,
                    2 => 1 + next(300) as usize,
                    _ => TILE_ROWS * (1 + next(3) as usize),
                })
                .collect();
            let total: usize = counts.iter().sum();
            let jobs = plan_jobs(&counts);
            let mut at = 0usize;
            for job in &jobs {
                assert_eq!(job.row0 as usize, at, "round {round}: back to back");
                assert!((1..=TILE_ROWS as u32).contains(&job.rows), "round {round}");
                assert!((job.member as usize) < experts, "round {round}");
                at += job.rows as usize;
            }
            assert_eq!(at, total, "round {round}: every row once");
            // One member per expert with rows, in order, and its jobs cover its count.
            for (expert, &count) in counts.iter().enumerate() {
                let covered: u32 = jobs
                    .iter()
                    .filter(|job| job.member as usize == expert)
                    .map(|job| job.rows)
                    .sum();
                assert_eq!(covered as usize, count, "round {round}: expert {expert}");
            }
            assert!(
                jobs.windows(2).all(|w| w[0].member <= w[1].member),
                "round {round}"
            );
        }
    }

    /// The table never outgrows the room the plan gives it, nor the kernel's job limit, at the
    /// worst shapes of the largest chunk: every pair on its own expert, every expert one tile and
    /// a row, and all pairs on a few experts.
    /// Mutation: cutting tiles at half the height doubles the jobs and fails the bound.
    #[test]
    fn the_job_count_stays_within_the_planned_table_and_the_kernel_limit() {
        use crate::family::qwen36::cuda::prefill_plan::exl3_max_jobs;
        let (experts, pairs) = (256usize, 2048 * 8);
        let shapes: [Vec<usize>; 4] = [
            vec![pairs / experts; experts],
            {
                // Experts of one tile and a row each, the last one short.
                let mut counts = vec![TILE_ROWS + 1; pairs / (TILE_ROWS + 1)];
                counts.push(pairs - counts.iter().sum::<usize>());
                counts
            },
            {
                let mut counts = vec![0; experts];
                counts[3] = pairs;
                counts
            },
            {
                let mut counts = vec![1; experts];
                counts[0] = pairs - (experts - 1);
                counts
            },
        ];
        for counts in shapes {
            assert_eq!(counts.iter().sum::<usize>(), pairs);
            let jobs = plan_jobs(&counts).len();
            assert!(jobs <= exl3_max_jobs(experts, pairs), "{jobs} jobs");
            assert!(jobs <= 4096, "{jobs} jobs");
        }
    }

    #[test]
    fn a_pick_outside_the_experts_refuses_the_chunk() {
        for bad in [-1, 4, i32::MAX] {
            let refused = plan_pairs(&[0, bad], 4, 2).expect_err("a refusal");
            assert!(matches!(refused, EngError::ModelOutput(_)), "{bad}");
        }
    }

    #[test]
    fn a_full_chunk_sorts_every_pair_exactly_once() {
        let (rows, top_k, experts) = (2048, 8, 256);
        let picks: Vec<i32> = (0..rows * top_k)
            .map(|p| (((p * 2_654_435_761usize) >> 7) % experts) as i32)
            .collect();
        let plan = plan_pairs(&picks, experts, top_k).expect("plan");
        assert_eq!(plan.counts.iter().sum::<usize>(), rows * top_k);
        let mut seen = vec![false; rows * top_k];
        for &pair in &plan.scatter {
            assert!(!std::mem::replace(&mut seen[pair as usize], true));
        }
    }
}
