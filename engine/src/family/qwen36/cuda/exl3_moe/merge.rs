//! The shared expert inside the routed experts' EXL3 launches (`FIFI_EXL3_SHARED_IN_ROUTED=1`).
//!
//! Today a MoE layer queues three groups for the routed experts and three for the shared expert:
//!
//! ```text
//! routed gate/up (16 pairs) | swiglu (8 rows) | routed down (8 pairs)
//! shared gate/up (2 pairs)  | swiglu (1 row)  | shared down (1 pair)
//! ```
//!
//! A group is three kernels (input rotate, projection, output rotate), a swiglu is one, so the
//! block is 14 kernels besides the router and the combine. With the flag on the shared expert
//! rides in the routed launches and the block is 7:
//!
//! ```text
//! gate/up (18 pairs) | swiglu (9 rows) | down (9 pairs)
//! ```
//!
//! No kernel changes. A pair is one block row of every kernel, resolving its own trellis, `suh`,
//! `svh`, rate and codebook from its own entry, so a pair computes the same bytes in any launch
//! it is a member of. Only the addresses of the shared pair move.
//!
//! # Layout (the contract of the launches)
//!
//! `top_k` is the number of picks (8 for Qwen3.6), `h` the hidden width, `i` the width of one
//! expert and of the shared expert (the two must be equal, so the pairs share `k` and `n`).
//! All rows are F32. The merged working buffer ([`SharedMerge`]) owns every intermediate the
//! merged block touches; the step arena and its plan are untouched.
//!
//! ```text
//! gate/up launch, k = h, n = i, 2 top_k + 2 pairs (18), the 20-pair kernels
//!   pair 2 s     routed gate of pick s   bank Gate, ids[s]         out gu row 2 s
//!   pair 2 s + 1 routed up of pick s     bank Up,   ids[s]         out gu row 2 s + 1
//!   pair 2 top_k     shared gate         1-member bank, ids word 0 out gu row 2 top_k
//!   pair 2 top_k + 1 shared up           1-member bank, ids word 0 out gu row 2 top_k + 1
//!   every pair reads norm_row; pair p rotates into xh row p and projects into z row p
//!   (partial 0) and out (partial 1, then the rotated result in place)
//! swiglu, rows = top_k + 1, n_ff = i, ld_in = 2 i, ld_out = i
//!   gu row r = [gate | up] = gu half-rows 2 r and 2 r + 1; row top_k is the shared expert
//!   act row r = swiglu of gu row r
//! down launch, k = i, n = h, top_k + 1 pairs (9), the default kernels
//!   pair s       routed down of pick s   bank Down, ids[s]         x act row s    out down row s
//!   pair top_k   shared down             1-member bank, ids word 0 x act row top_k out down row top_k
//! combine: down rows 0..top_k (the routed term, pick order) and down row top_k (the shared
//!   term, scaled by the router's shared gate); the kernel and its sum order are unchanged
//! ```
//!
//! Strides, as the kernels read them: the routed banks keep their own `trellis_stride` and
//! `scale_stride`; the shared matrix is a bank of one member (`members = 1`) with strides that
//! are only required to be non-zero, and its `ids` word is a zero word of the merged buffer,
//! which nothing writes, so `id * stride` is 0 and the pair reads the matrix's own addresses.
//! The `ids` of the routed pairs are the router's `TopkId` slots.
//!
//! The host path (picks read back, no graph) queues the same groups as plain grouped launches.

use super::{
    EngError, Exl3MoeBufs, Exl3MoeWeights, F32, Launch, MoeDims, Pair, Scratch, combine_ranges,
    grouped, read_picks, refuse, route, row, swiglu,
};
use crate::{
    family::{
        project::cuda::exl3::{Exl3Bank, Exl3Io, Exl3Weight},
        qwen36::exl3_merge_plan::{
            BlockPlan, IdWord, Member, MergeLayout, PlanDims, PlannedPair, Proj, SharedFacts, Span,
            block_plan, check_dims, check_merge,
        },
    },
    kernels::cuda::{CudaContext, DeviceBuffer, DeviceRange},
};

/// The plan sizes of a block.
fn plan_dims(dims: &MoeDims) -> PlanDims {
    PlanDims {
        rows: dims.rows,
        hidden: dims.hidden,
        top_k: dims.top_k,
        inter: dims.inter,
        shared_inter: dims.shared_inter,
    }
}

/// The merged block's working buffer, owned by the step for its life.
pub(in crate::family) struct SharedMerge {
    buffer: DeviceBuffer,
    layout: MergeLayout,
}

struct Ranges<'b> {
    buffer: &'b DeviceBuffer,
    layout: &'b MergeLayout,
    xh: DeviceRange<'b>,
    z: DeviceRange<'b>,
    gu: DeviceRange<'b>,
    act: DeviceRange<'b>,
    zero: DeviceRange<'b>,
}

impl SharedMerge {
    /// Checks that every layer can run merged and allocates the zeroed working buffer. Refuses,
    /// naming the switch, what the merged launches cannot run: a wide single-row kernel of more
    /// than one tile, a block of more than 20 gate/up pairs, unequal widths, a shared matrix with
    /// a bias or a rate other than 4 or 6 bits.
    pub(in crate::family) fn new(
        ctx: &CudaContext,
        dims: &MoeDims,
        layers: &[Exl3MoeWeights<'_>],
    ) -> Result<Self, EngError> {
        let facts: Vec<[SharedFacts; 3]> = layers
            .iter()
            .map(|weights| {
                [
                    ("shared gate", weights.shared_gate),
                    ("shared up", weights.shared_up),
                    ("shared down", weights.shared_down),
                ]
                .map(|(name, weight)| SharedFacts {
                    name,
                    bias: weight.bias.is_some(),
                    bits_twice: weight.bits_twice,
                })
            })
            .collect();
        let plan = plan_dims(dims);
        check_merge(&plan, ctx.single_row_setting()?, &facts).map_err(EngError::BadRequest)?;
        let layout = MergeLayout::new(&plan).map_err(EngError::BadRequest)?;
        let buffer = ctx.upload(&vec![0u8; layout.total])?;
        Ok(Self { buffer, layout })
    }

    fn ranges(&self) -> Result<Ranges<'_>, EngError> {
        let span = |s: Span| self.buffer.range(s.offset, s.bytes);
        Ok(Ranges {
            buffer: &self.buffer,
            layout: &self.layout,
            xh: span(self.layout.xh)?,
            z: span(self.layout.z)?,
            gu: span(self.layout.gu)?,
            act: span(self.layout.act)?,
            zero: span(self.layout.zero)?,
        })
    }

    /// The merged working buffer, for the device tests that compare intermediates.
    #[cfg(test)]
    pub(in crate::family) fn buffer(&self) -> &DeviceBuffer {
        &self.buffer
    }

    #[cfg(test)]
    pub(in crate::family) fn layout_for_tests(&self) -> (usize, usize, usize, usize) {
        (
            self.layout.gu.offset,
            self.layout.act.offset,
            self.layout.down.offset,
            self.layout.zero.offset,
        )
    }
}

fn bank_of<'w>(weights: &Exl3MoeWeights<'w>, proj: Proj) -> &'w Exl3Bank {
    match proj {
        Proj::Gate => weights.gate,
        Proj::Up => weights.up,
        Proj::Down => weights.down,
    }
}

fn single_of<'w>(weights: &Exl3MoeWeights<'w>, proj: Proj) -> &'w Exl3Weight {
    match proj {
        Proj::Gate => weights.shared_gate,
        Proj::Up => weights.shared_up,
        Proj::Down => weights.shared_down,
    }
}

/// The input and output ranges of a planned pair: the plan says where, the buffer makes ranges.
fn pair_ranges<'b>(
    pair: &PlannedPair,
    norm_row: DeviceRange<'b>,
    r: &Ranges<'b>,
    dims: &MoeDims,
) -> Result<(DeviceRange<'b>, DeviceRange<'b>), EngError> {
    let spans = pair.spans(&plan_dims(dims), r.layout);
    let x = match spans.input {
        None => norm_row,
        Some(span) => r.buffer.range(span.offset, span.bytes)?,
    };
    Ok((x, r.buffer.range(spans.output.offset, spans.output.bytes)?))
}

/// The matrix of one id-indexed pair of the merged block.
enum MergedMember<'a> {
    /// Pick `slot` of a routed bank.
    Routed(&'a Exl3Bank, u32),
    /// The shared matrix, a bank of one member.
    Shared(&'a Exl3Weight),
}

impl MergedMember<'_> {
    fn shape(&self) -> [usize; 2] {
        match self {
            Self::Routed(bank, _) => bank.shape(),
            Self::Shared(weight) => weight.shape,
        }
    }
}

/// One id-indexed pair of the merged block.
struct MergedPair<'a> {
    member: MergedMember<'a>,
    /// The `ids` word and slot the member is read from, as the plan says.
    word: (IdWord, u32),
    x: DeviceRange<'a>,
    out: DeviceRange<'a>,
}

/// Queues the id-indexed group `pairs` as ONE launch (the 20-pair kernels above 16 pairs).
fn indexed_group(
    launch: &Launch<'_>,
    pairs: &[MergedPair<'_>],
    (router_ids, zero): (DeviceRange<'_>, DeviceRange<'_>),
    scratch: &Scratch<'_>,
) -> Result<(), EngError> {
    let [k, n] = pairs
        .first()
        .map(|pair| pair.member.shape())
        .ok_or_else(|| refuse("a merged launch without pairs".into()))?;
    let mut calls = Vec::with_capacity(pairs.len());
    for (index, pair) in pairs.iter().enumerate() {
        let io = Exl3Io {
            x: pair.x,
            xh: row(&scratch.xh, k * F32, index)?,
            z: row(&scratch.z, n * F32, index)?,
            out: pair.out,
            flag: scratch.flag,
            rows: 1,
            fused: false,
        };
        let (word, slot) = pair.word;
        let ids = match word {
            IdWord::Router => router_ids,
            IdWord::Zero => zero,
        };
        calls.push(match &pair.member {
            MergedMember::Routed(bank, _) => bank.span()?.indexed_call(&io, ids, slot),
            MergedMember::Shared(weight) => weight.view()?.indexed_call(&io, ids, slot),
        });
    }
    launch
        .ctx
        .exl3_grouped_indexed_wide(launch.stream, &calls, launch.ctx.exl3_indexed_cols(n)?)
}

fn merged_on_device(
    launch: &Launch<'_>,
    (weights, bufs, r): (&Exl3MoeWeights<'_>, &Exl3MoeBufs<'_>, &Ranges<'_>),
    plan: &BlockPlan,
    dims: &MoeDims,
) -> Result<(), EngError> {
    let scratch = Scratch {
        xh: r.xh,
        z: r.z,
        flag: bufs.scratch.flag,
    };
    let ids = (bufs.route.ids, r.zero);
    let build = |pairs: &[PlannedPair]| -> Result<Vec<MergedPair<'_>>, EngError> {
        pairs
            .iter()
            .map(|pair| {
                let (x, out) = pair_ranges(pair, bufs.route.norm_row, r, dims)?;
                let member = match pair.member {
                    Member::Routed(slot) => MergedMember::Routed(bank_of(weights, pair.proj), slot),
                    Member::Shared => MergedMember::Shared(single_of(weights, pair.proj)),
                };
                Ok(MergedPair {
                    member,
                    word: pair.id_word(),
                    x,
                    out,
                })
            })
            .collect()
    };
    indexed_group(launch, &build(&plan.gate_up)?, ids, &scratch)?;
    swiglu(launch, r.gu, r.act, (plan.swiglu_rows, dims.inter))?;
    indexed_group(launch, &build(&plan.down)?, ids, &scratch)
}

fn merged_on_host(
    launch: &Launch<'_>,
    (weights, bufs, r): (&Exl3MoeWeights<'_>, &Exl3MoeBufs<'_>, &Ranges<'_>),
    (plan, picks): (&BlockPlan, &[usize]),
    dims: &MoeDims,
) -> Result<(), EngError> {
    let scratch = Scratch {
        xh: r.xh,
        z: r.z,
        flag: bufs.scratch.flag,
    };
    let build = |pairs: &[PlannedPair]| -> Result<Vec<Pair<'_>>, EngError> {
        pairs
            .iter()
            .map(|pair| {
                let (x, out) = pair_ranges(pair, bufs.route.norm_row, r, dims)?;
                let view = match pair.member {
                    Member::Routed(slot) => {
                        let expert = *picks
                            .get(slot as usize)
                            .ok_or_else(|| refuse(format!("no pick for slot {slot}")))?;
                        bank_of(weights, pair.proj).view(expert)?
                    }
                    Member::Shared => single_of(weights, pair.proj).view()?,
                };
                Ok(Pair { view, x, out })
            })
            .collect()
    };
    grouped(launch, &build(&plan.gate_up)?, &scratch)?;
    swiglu(launch, r.gu, r.act, (plan.swiglu_rows, dims.inter))?;
    grouped(launch, &build(&plan.down)?, &scratch)
}

/// The block of one layer with the shared expert in the routed launches: the same router, the
/// same combine, the groups of [`block_plan`]. With `device_ids` the picks stay on the device and
/// the block makes no host read; otherwise it waits for the stream once, to read the ids.
pub(in crate::family) fn run_exl3_moe_merged(
    launch: &Launch<'_>,
    (weights, bufs, merge): (&Exl3MoeWeights<'_>, &Exl3MoeBufs<'_>, &SharedMerge),
    dims: &MoeDims,
    device_ids: bool,
) -> Result<(), EngError> {
    check_dims(&plan_dims(dims)).map_err(refuse)?;
    let r = merge.ranges()?;
    let plan = block_plan(dims.top_k);
    route(launch, weights, bufs, dims)?;
    if device_ids {
        merged_on_device(launch, (weights, bufs, &r), &plan, dims)?;
    } else {
        let picks = read_picks(launch, &bufs.route.ids, dims)?;
        merged_on_host(launch, (weights, bufs, &r), (&plan, &picks), dims)?;
    }
    let (routed, shared) = r.layout.combine_spans(&plan_dims(dims));
    let down = r.buffer.range(routed.offset, routed.bytes)?;
    let shared = r.buffer.range(shared.offset, shared.bytes)?;
    combine_ranges(launch, &bufs.route, (down, shared), dims)
}
