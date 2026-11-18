//! The MoE block of one Flash-Next layer on a device hidden row, with the routed experts in the
//! expert bank (`bank.rs`) instead of all resident. One call per layer and token.
//!
//! WHAT, in launch order on the one stream:
//!
//! 1. The router (shared kernel) writes the picks, their renormalized weights and the shared gate.
//! 2. The layer's slot row goes to the device and `tiered_classify` maps every pick to its bank
//!    slot: a resident pick to its slot, a missed pick to slot 0 with weight +0.0 (the filler
//!    rule). The classify log (ten ids and weights, 80 bytes) comes back to the host.
//! 3. The resident pass is queued: gate and up of the ten slots (id-indexed launches over the bank
//!    buffer), SwiGLU, down, the shared expert, and the combine when no miss will follow (see 5).
//! 4. The bank plans and runs the copies for the picks (`DeviceBank::step`, all ten picks, hits
//!    included, so the policy sees every use). The copies run on the bank's own stream, which is
//!    not ordered with the block's stream: they overlap the resident pass, and the host waits for
//!    them before it queues the miss pass.
//! 5. With at least one miss, the miss pass is queued: the same launches over the slot row after
//!    the copies, carrying ONLY the missed picks on their new slots ([`miss_pass_inputs`]) — no
//!    +0.0 filler, the launch is as wide as the miss count, and each missed pick's down output
//!    lands in its RANK row of the shared down buffer, overwriting the resident pass's filler
//!    row. The miss pass's combine is the block's only sum: every pick in rank order with the
//!    router's true weights, the shared term, and `resid_in` — the same association one launch
//!    over an all-resident bank has, at any slot count. The resident pass computes its filler
//!    rows but skips its combine when a miss will follow, so no +0.0 filler product exists and a
//!    non-finite filler expert can no longer poison the output.
//!
//! WHY this shape: the resident pass needs nothing a miss brings, so it runs while the host
//! plans and waits for the copies. The overlap is safe because of one rule of the bank's policy:
//! the slot of every pick and slot 0 are never victims in a pass, so a copy only writes a slot
//! that neither the resident pass (the picks' slots and the filler slot 0) nor anything earlier
//! reads. The slot-row upload at the start of the call waits for the block's stream, so no
//! launch of an earlier layer or token is still reading this layer's slots when a copy starts.
//! The shared expert rides in the first pass because it never misses. A layer without a miss is
//! one pass whose result goes straight to the output; a layer with a miss sums once, at the end.
//! Every pick contributes exactly one product to exactly one combine.
//!
//! The split result is bit-identical to one launch over an all-resident bank: both sum the picks
//! in rank order in one combine, so it does not depend on which experts the bank happened to
//! hold (the slot count or the LRU state). The tests state the equality.
//!
//! The pure parts (the classify oracle, the log parser, the spans of the bank buffer and the
//! private scratch layout) are host code with host tests; the launches are native-cuda only.

use std::ops::Range;

use super::bank::miss_pass_inputs;
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
use crate::family::flash_next::cuda::driver::chunk::prof;
use crate::{
    EngError,
    family::{
        flash_next::moe::slot_bank::{PARTS, SlotBankLayout},
        project::exl3_jobs_plan::EXL3_JOB_WORDS,
        tiered_classify::{Classified, ROUTE_INVALID, classify_reference},
    },
};

const F32: usize = size_of::<f32>();
const I32: usize = size_of::<i32>();
/// Bytes of one entry of the classify log: the id and the bits of the weight.
const LOG_ENTRY: usize = 8;
/// Control block, request line: the sizes the classify kernel writes whether or not it is read.
const CTRL_BYTES: usize = 64;
const REQUEST_BYTES: usize = 128;
/// Every scratch region starts at a multiple of this (the arena alignment).
const REGION_ALIGN: usize = 256;

fn refuse(detail: String) -> EngError {
    EngError::BadRequest(format!("Flash-Next split MoE block: {detail}"))
}

fn align_up(value: usize) -> usize {
    value.next_multiple_of(REGION_ALIGN)
}

/// The block's private device scratch: the slot row, the classify outputs and the second pass's
/// inputs. One allocation, regions at aligned offsets: 24,832 bytes at the real sizes and one
/// row (pinned by the host tests), and the row-indexed regions grow with the planned chunk
/// width.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct ScratchLayout {
    /// The chunk row count this layout is sized for; the prefill pass admits at most this many rows.
    pub(super) rows: usize,
    /// i32 `[n_expert]`: the layer's slot row, uploaded before each classification.
    pub(super) slot_row: usize,
    /// i32 `[top_k]`: the bank slot of each pick for the resident pass.
    pub(super) slot_ids: usize,
    /// f32 `[top_k]`: the resident pass's combine weights.
    pub(super) weights: usize,
    /// `8 * top_k` bytes: the raw picks (id, weight bits).
    pub(super) log: usize,
    pub(super) ctrl: usize,
    pub(super) word: usize,
    pub(super) mbox_req: usize,
    /// f32 `[hidden]`: the classify kernel copies the row here when a pick misses.
    pub(super) mbox_row: usize,
    /// i32 `[top_k]` (per chunk row): the miss pass's slot ids.
    /// f32 `[top_k]` (per chunk row): the multi-row path's zero-weight tail combine — the one
    /// that moves a clean row's resident result to the block output (the single-row miss pass
    /// reads the router's device weights instead).
    pub(super) miss_ids: usize,
    pub(super) miss_weights: usize,
    /// f32 `[hidden]` (per chunk row): the multi-row resident pass's result, from which the
    /// per-row tails move clean rows to the block output (a miss row's copy is discarded; the
    /// single-row block does not use it — its resident pass skips the combine on a miss).
    pub(super) partial: usize,
    /// i32 `[(2 + EXL3_JOB_WORDS) * routed]`: the sort lists and the job tables of the chunk's
    /// two routed job-table passes (2 list words plus one job record per pair; a source's
    /// jobs are at most its pairs).
    pub(super) perm: usize,
    pub(super) total: usize,
}

impl ScratchLayout {
    /// `rows` is the chunk row count the multi-row block plans for (`1` for the decode block);
    /// only the regions the block indexes per row scale with it.
    pub(super) fn new(
        n_expert: usize,
        top_k: usize,
        hidden: usize,
        rows: usize,
    ) -> Result<Self, EngError> {
        let overflow = || refuse("the scratch size overflows".into());
        let mul = |a: Option<usize>, b: usize| a.and_then(|at| at.checked_mul(b));
        let routed = rows.checked_mul(top_k).ok_or_else(overflow)?;
        let regions = [
            n_expert.checked_mul(I32),
            mul(rows.checked_mul(top_k), I32),
            mul(rows.checked_mul(top_k), F32),
            top_k.checked_mul(LOG_ENTRY),
            Some(CTRL_BYTES),
            Some(I32),
            Some(REQUEST_BYTES),
            hidden.checked_mul(F32),
            mul(rows.checked_mul(top_k), I32),
            mul(rows.checked_mul(top_k), F32),
            mul(rows.checked_mul(hidden), F32),
            // Two sources (the bank's and the ring's slots) plan separately: their lists sum
            // to `routed` pairs and their job tables to at most one job per pair (an expert
            // with rows owns at least one pair of its source), so 2 list words plus one
            // 4-word job record per pair cover both sources' lists and tables.
            (2 + EXL3_JOB_WORDS)
                .checked_mul(routed)
                .and_then(|words| words.checked_mul(I32)),
        ];
        let mut offsets = [0usize; 12];
        let mut at = 0usize;
        for (offset, size) in offsets.iter_mut().zip(regions) {
            *offset = at;
            at = size
                .ok_or_else(overflow)?
                .checked_next_multiple_of(REGION_ALIGN)
                .and_then(|size| at.checked_add(size))
                .ok_or_else(overflow)?;
        }
        let [
            slot_row,
            slot_ids,
            weights,
            log,
            ctrl,
            word,
            mbox_req,
            mbox_row,
            miss_ids,
            miss_weights,
            partial,
            perm,
        ] = offsets;
        Ok(Self {
            rows,
            slot_row,
            slot_ids,
            weights,
            log,
            ctrl,
            word,
            mbox_req,
            mbox_row,
            miss_ids,
            miss_weights,
            partial,
            perm,
            total: at,
        })
    }
}

/// The picks of the classify log: `(ids, weights)` in rank order.
pub(super) fn parse_log(raw: &[u8], top_k: usize) -> Result<(Vec<i32>, Vec<f32>), EngError> {
    if raw.len() != top_k * LOG_ENTRY {
        return Err(refuse(format!(
            "the classify log has {} bytes, {top_k} picks need {}",
            raw.len(),
            top_k * LOG_ENTRY
        )));
    }
    let (mut ids, mut weights) = (Vec::with_capacity(top_k), Vec::with_capacity(top_k));
    for entry in raw.as_chunks::<LOG_ENTRY>().0 {
        let (id, weight) = entry.split_at(I32);
        ids.push(i32::from_le_bytes(id.try_into().expect("four bytes")));
        weights.push(f32::from_le_bytes(weight.try_into().expect("four bytes")));
    }
    Ok((ids, weights))
}

/// The host's classification of the router's picks against `slot_row` before the copies: what the
/// device kernel computed (the kernel equals this oracle bit for bit). Refuses a route the
/// kernel flagged and a router fault, which have no second pass.
pub(super) fn classify_picks(
    slot_row: &[i32],
    slots: usize,
    ids: &[i32],
    weights: &[f32],
) -> Result<Classified, EngError> {
    let classified = classify_reference(slot_row, slots, ids, weights)
        .map_err(|error| refuse(format!("pick lists of different lengths: {error:?}")))?;
    if classified.word & ROUTE_INVALID != 0 {
        return Err(refuse(format!("the router's picks are invalid: {ids:?}")));
    }
    if let Some(at) = ids.iter().position(|&id| id < 0) {
        return Err(refuse(format!(
            "the router faulted on pick {at} (id {}); non-finite logits",
            ids[at]
        )));
    }
    Ok(classified)
}

/// Where the three parts of an expert sit in a bank buffer of `slots` slots, as the id-indexed
/// launch addresses them: member `i` of a part is the component at its start plus `i` strides.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct SlotSpans {
    pub(super) slots: usize,
    /// Bytes between the trellises of two slots.
    pub(super) trellis_stride: usize,
    /// F32 values between the `suh` (and the `svh`) of two slots.
    pub(super) scale_stride: usize,
    pub(super) parts: [PartSpan; PARTS],
}

/// One part's ranges of the bank buffer, from the start of its component in slot 0 to its end in
/// the last slot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct PartSpan {
    pub(super) trellis: Range<usize>,
    pub(super) suh: Range<usize>,
    pub(super) svh: Range<usize>,
    /// `[k, n]` of the matrix.
    pub(super) shape: [usize; 2],
    pub(super) bits_twice: u8,
}

impl SlotSpans {
    pub(super) fn new(layout: &SlotBankLayout, slots: usize) -> Result<Self, EngError> {
        let mut parts = Vec::with_capacity(PARTS);
        for index in 0..PARTS {
            let part = layout.part(index);
            let [trellis, suh, svh] = layout.bank_ranges(index, slots)?;
            let bits_twice = u8::try_from(part.words / 8)
                .ok()
                .filter(|&bits| usize::from(bits) * 8 == part.words)
                .ok_or_else(|| {
                    refuse(format!("{} words per tile is not a whole rate", part.words))
                })?;
            parts.push(PartSpan {
                trellis,
                suh,
                svh,
                shape: part.shape,
                bits_twice,
            });
        }
        let record = layout.record_bytes();
        if layout.trellis_stride_bytes() != record || layout.scale_stride_floats() * F32 != record {
            return Err(refuse(
                "the trellis and scale strides of a slot differ from the record size".into(),
            ));
        }
        Ok(Self {
            slots,
            trellis_stride: layout.trellis_stride_bytes(),
            scale_stride: layout.scale_stride_floats(),
            parts: parts
                .try_into()
                .map_err(|_| refuse("three parts expected".into()))?,
        })
    }
}

/// The two passes of a layer, as the host decided them: how the router's picks split between the
/// resident pass and the miss pass.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct PassInputs {
    /// The miss pass's slots and the rank rows its down outputs overwrite (`None` without a miss).
    pub(super) miss: Option<(Vec<i32>, Vec<u32>)>,
    pub(super) misses: usize,
}

/// The miss pass's inputs after the copies, from the classification made before them.
pub(super) fn plan_second_pass(
    ids: &[i32],
    classified: &Classified,
    slot_row_after: &[i32],
) -> Result<PassInputs, EngError> {
    let misses = classified.misses.len();
    if misses == 0 {
        return Ok(PassInputs { miss: None, misses });
    }
    let inputs = miss_pass_inputs(ids, classified, slot_row_after)?;
    Ok(PassInputs {
        miss: Some(inputs),
        misses,
    })
}

#[cfg(all(feature = "native-cuda", target_os = "linux"))]
#[allow(unused_imports, reason = "the decode step is the first caller")]
pub(in crate::family::flash_next) use device::{SplitMoe, SplitMoeTrunk};

#[cfg(all(feature = "native-cuda", target_os = "linux"))]
#[allow(
    unused_imports,
    reason = "the host tests import these from the sibling test module"
)]
pub(super) use device::{source_offsets, source_plan};

#[cfg(all(feature = "native-cuda", target_os = "linux"))]
mod device {
    use crate::family::exl3::launch::{Launch, MoeDims, RouteBufs};
    /// The router's picks per row: each row's expert ids and their weights.
    type RoutedPicks = Vec<(Vec<i32>, Vec<f32>)>;
    use super::{
        CTRL_BYTES, F32, I32, LOG_ENTRY, REQUEST_BYTES, ScratchLayout, SlotSpans, classify_picks,
        parse_log, plan_second_pass, prof, refuse,
    };
    use crate::{
        EngError,
        family::{
            exl3::{
                moe::{
                    Exl3MoeBufs, IndexedPair, Scratch, SharedWeights, grouped_indexed, shared_of,
                },
                prefill::{Projector, window},
                prefill_moe,
            },
            flash_next::{
                cuda::{
                    bank::device::DeviceBank,
                    prefill_adapter::device::{LayerStore, PrefillAdapter, RingStore},
                },
                moe::slot_bank::SlotBankLayout,
            },
            project::{
                cuda::{
                    exl3::{Exl3BankSpan, Exl3Members, Exl3Weight},
                    exl3_jobs::{
                        PairPlan, Scratch as JobScratch, pair_layout, plan_jobs, plan_pairs,
                    },
                },
                exl3_jobs_plan::{EXL3_JOB_WORDS, MAX_EXL3_ROWS, exl3_job_bytes},
            },
        },
        kernels::cuda::{
            CudaContext, DeviceBuffer, DeviceRange,
            prefill::swiglu_rows::{SwigluRowsCall, SwigluRowsShape},
            resident::{
                moe_combine::{CombineCall, CombineOut, CombineShape, SharedTerm},
                router::{RouterCall, RouterOut, RouterShape, RouterWeights},
                tiered_classify::{ClassifyCall, ClassifyShape},
            },
        },
        weights::exl3::Codebook,
    };

    /// The `rows` rows of `range` at row `index` of `row_bytes` each.
    fn rows_of<'b>(
        range: &DeviceRange<'b>,
        row_bytes: usize,
        index: usize,
        rows: usize,
    ) -> Result<DeviceRange<'b>, EngError> {
        range
            .buffer()
            .range(range.offset() + index * row_bytes, rows * row_bytes)
    }

    /// The counting sort of one source's pairs: `sub` lists the chunk's pair positions of the
    /// source's picks in chunk order; the shared sort runs over their slot ids and the lists
    /// are translated back to the chunk's pair positions and rows.
    pub(crate) fn source_plan(
        ids: &[i32],
        sub: &[usize],
        slots: usize,
        top_k: usize,
    ) -> Result<PairPlan, EngError> {
        let picks: Vec<i32> = sub.iter().map(|&at| ids[at]).collect();
        let mut plan = plan_pairs(&picks, slots, top_k)?;
        let overflow = || refuse("the chunk's pair count exceeds i32".into());
        for (order, scatter) in plan.order.iter_mut().zip(&mut plan.scatter) {
            let pair = sub[*scatter as usize];
            *scatter = i32::try_from(pair).map_err(|_| overflow())?;
            *order = i32::try_from(pair / top_k).map_err(|_| overflow())?;
        }
        Ok(plan)
    }

    /// The perm-region byte offset of each source's lists and job table, and the offset of its
    /// gate/up segment of the pair faces: source 0 (the bank) at the region's base, source 1
    /// (the ring) after source 0's lists (2 words a pair) and its job table (a 4-word record
    /// a job).
    pub(crate) fn source_offsets(
        perm: usize,
        plans: [&PairPlan; 2],
        sub: [&[usize]; 2],
    ) -> ([usize; 2], [usize; 2]) {
        let mut perm_at = perm;
        let mut gu_at = 0usize;
        let mut at_of = [0usize; 2];
        let mut seg_of = [0usize; 2];
        for source in 0..2 {
            at_of[source] = perm_at;
            seg_of[source] = gu_at;
            let jobs = plan_jobs(&plans[source].counts).len();
            perm_at += (2 * sub[source].len() + EXL3_JOB_WORDS * jobs) * I32;
            gu_at += sub[source].len();
        }
        (at_of, seg_of)
    }

    /// The rotated-row and partial-sum ranges of a fused gate and up launch over `n` pairs,
    /// when the second rotation fits behind the partial sums of the scratch's `z` range;
    /// `None` otherwise, and the caller runs the two projections separately.
    fn pair_scratch_of<'b>(
        scratch: &Scratch<'b>,
        n: usize,
        inter: usize,
        hidden: usize,
    ) -> Result<Option<(JobScratch<'b>, JobScratch<'b>)>, EngError> {
        let Some(layout) = pair_layout((n, inter, hidden), scratch.z.bytes()) else {
            return Ok(None);
        };
        let z = scratch
            .z
            .buffer()
            .range(scratch.z.offset(), layout.z_bytes)?;
        let second_xh = scratch
            .z
            .buffer()
            .range(scratch.z.offset() + layout.tail, layout.xh_bytes)?;
        Ok(Some((
            JobScratch { xh: scratch.xh, z },
            JobScratch { xh: second_xh, z },
        )))
    }

    /// The trunk weights of one layer's MoE block: the router, the shared expert's gate row, and
    /// the shared expert's three matrices. The caller borrows them from the loaded weights.
    pub(in crate::family::flash_next) struct SplitMoeTrunk<'w> {
        /// Raw F32 `[n_expert][hidden]`.
        pub(in crate::family::flash_next) router: &'w DeviceBuffer,
        /// Raw F32 `[hidden]`: the shared expert's gate row.
        pub(in crate::family::flash_next) router_shared: &'w DeviceBuffer,
        pub(in crate::family::flash_next) shared_gate: &'w Exl3Weight,
        pub(in crate::family::flash_next) shared_up: &'w Exl3Weight,
        pub(in crate::family::flash_next) shared_down: &'w Exl3Weight,
    }

    /// The block of every layer: one trunk per layer, the bank and the private scratch.
    pub(in crate::family::flash_next) struct SplitMoe<'w> {
        trunks: Vec<SplitMoeTrunk<'w>>,
        bank: DeviceBank,
        spans: SlotSpans,
        /// The record layout the bank spans were built from; the ring's spans are rebuilt
        /// from it at the ring's own record count, which may be smaller than the bank's.
        bank_layout: SlotBankLayout,
        layout: ScratchLayout,
        scratch: DeviceBuffer,
        book: Codebook,
        /// The prefill expert ring: `None` until [`SplitMoe::plan_ring`] funds one; the
        /// multi-row path streams through it instead of admitting into the bank.
        ring: Option<crate::family::flash_next::cuda::prefill_ring::device::PrefillRing>,
        /// The chunked-prefill row count at and above which an active ring serves the whole
        /// chunk; smaller chunks take the bank's union-admit path. Set when the ring is
        /// planned (`ring_rows`, default 32).
        ring_min_rows: usize,
        /// Latched once a prefill window streams a whole chunk through the ring: the rest of
        /// that window, including its trailing single-row logits step, stays on the ring.
        /// Switching mid-window would read the pinned store on the bank stream while the
        /// ring's copies are outstanding on the copy stream, which `PinnedBuf` refuses (one
        /// outstanding stream per store). Reset at `begin_prefill`, which also drains any ring
        /// copy a reopened window left outstanding; a window whose chunks are all below
        /// `ring_min_rows` never engages the ring and admits against that quiet store.
        ring_engaged: bool,
    }

    /// One pass over the bank.
    struct Pass<'a> {
        layer: usize,
        /// i32 `[picks]`: the bank slot of each pick of this pass — `top_k` entries for the
        /// resident pass, the missed count for the compact miss pass.
        ids: DeviceRange<'a>,
        /// f32 `[weights]`: the combine weights — the resident pass's slot weights, the miss
        /// pass's slice of the router's device weights for its one rank-order combine (the
        /// combine width is `dims.top_k` either way).
        weights: DeviceRange<'a>,
        /// Whether the shared expert rides in this pass (computed here and added to the combine).
        shared: bool,
        /// Add the shared TERM without recomputing it: the resident pass already ran the shared
        /// expert, and the miss pass's combine is the block's only sum.
        shared_term_only: bool,
        /// The down row each pick of `ids` writes (`pick j` writes row `down_rows[j]`), instead
        /// of the compact prefix. The miss pass uses it to overwrite the missed ranks' filler
        /// rows so its combine sums every pick in rank order.
        down_rows: Option<&'a [u32]>,
        /// The combine's `(residual, output)`. `None` runs the expert stages only: the
        /// resident pass of a miss layer skips its combine (the miss pass's combine is the
        /// block's only sum, and its filler rows must not be multiplied by anything).
        combine: Option<(DeviceRange<'a>, DeviceRange<'a>)>,
    }

    /// One pair of an id-indexed launch: the part, the pick whose slot id chooses the member, the
    /// input row and the output row.
    struct Item<'a> {
        part: usize,
        pick: u32,
        x: DeviceRange<'a>,
        out: DeviceRange<'a>,
    }

    impl SlotSpans {
        /// Record-strided members; trellis strides count u16s and scale strides count f32s.
        pub(in crate::family::flash_next) fn members_of<'b>(
            &self,
            buffer: &'b DeviceBuffer,
            part: usize,
            book: Codebook,
        ) -> Result<Exl3Members<'b>, EngError> {
            let span = &self.parts[part];
            if !self.trellis_stride.is_multiple_of(2) {
                return Err(refuse(
                    "the bank's record stride is not a whole u16 count".into(),
                ));
            }
            Ok(Exl3Members {
                trellis: buffer.range(span.trellis.start, span.trellis.len())?,
                suh: buffer.range(span.suh.start, span.suh.len())?,
                svh: buffer.range(span.svh.start, span.svh.len())?,
                trellis_stride: self.trellis_stride / 2,
                scale_stride: self.scale_stride,
                members: self.slots,
                shape: span.shape,
                bits_twice: span.bits_twice,
                book,
            })
        }
    }

    fn whole(buffer: &DeviceBuffer) -> Result<DeviceRange<'_>, EngError> {
        buffer.range(0, buffer.bytes())
    }

    fn row<'b>(
        range: &DeviceRange<'b>,
        width: usize,
        index: usize,
    ) -> Result<DeviceRange<'b>, EngError> {
        range.buffer().range(range.offset() + index * width, width)
    }

    fn le_i32(values: &[i32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn le_f32(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    impl<'w> SplitMoe<'w> {
        /// `trunks[layer]` is layer `layer`'s trunk; `bank` has one row per layer. `layout` is the
        /// bank's record layout, the one its records were packed with. Allocates the private
        /// scratch (24,832 bytes at the decode shape, growing with the planned width), zeroed.
        pub(in crate::family::flash_next) fn new(
            ctx: &CudaContext,
            trunks: Vec<SplitMoeTrunk<'w>>,
            bank: DeviceBank,
            layout: &SlotBankLayout,
            dims: &MoeDims,
        ) -> Result<Self, EngError> {
            let geometry = *bank.geometry();
            if trunks.len() != geometry.layers {
                return Err(refuse(format!(
                    "{} trunks for a bank of {} layers",
                    trunks.len(),
                    geometry.layers
                )));
            }
            if geometry.experts != dims.n_expert
                || geometry.top_k != dims.top_k
                || geometry.record_bytes != layout.record_bytes()
            {
                return Err(refuse(format!(
                    "the bank ({} experts, {} picks, {} byte records) does not match the block \
                     ({} experts, {} picks, {} byte records)",
                    geometry.experts,
                    geometry.top_k,
                    geometry.record_bytes,
                    dims.n_expert,
                    dims.top_k,
                    layout.record_bytes()
                )));
            }
            let (gate, down) = (layout.part(0).shape, layout.part(2).shape);
            if gate != [dims.hidden, dims.inter] || down != [dims.inter, dims.hidden] {
                return Err(refuse(format!(
                    "the record holds {gate:?} and {down:?} matrices; the block is {} wide with \
                     an inner width of {}",
                    dims.hidden, dims.inter
                )));
            }
            let spans = SlotSpans::new(layout, geometry.slots)?;
            let book = trunks
                .first()
                .ok_or_else(|| refuse("a block without layers".into()))?
                .shared_gate
                .view()?
                .book;
            let scratch_layout =
                ScratchLayout::new(dims.n_expert, dims.top_k, dims.hidden, dims.rows)?;
            let scratch = ctx.upload(&vec![0u8; scratch_layout.total])?;
            Ok(Self {
                trunks,
                bank,
                spans,
                bank_layout: layout.clone(),
                layout: scratch_layout,
                scratch,
                book,
                ring: None,
                ring_min_rows: 1,
                ring_engaged: false,
            })
        }

        /// Tries to fund the prefill ring: `budget` record slots of device memory, clamped to
        /// twice the streamed experts per layer (the ideal horizon) and never above it. A ring
        /// smaller than one full layer is not usable (a layer's grouped pass reads every
        /// streamed expert at once) and is not built. Returns the slot count funded, `0` when
        /// none. Call once, after the constructor.
        pub(in crate::family::flash_next) fn plan_ring(
            &mut self,
            ctx: &CudaContext,
            budget: usize,
            min_rows: usize,
        ) -> Result<usize, EngError> {
            let geometry = *self.bank.geometry();
            let streamed = geometry
                .experts
                .checked_sub(geometry.slots)
                .unwrap_or(geometry.experts);
            let want = budget.min(2 * streamed).max(1);
            let mut ring =
                crate::family::flash_next::cuda::prefill_ring::device::PrefillRing::build(
                    ctx, geometry, want,
                )?;
            ring.replan(&self.bank);
            if ring.ring_slots() < ring.per_layer() {
                return Ok(0);
            }
            let slots = ring.ring_slots();
            self.ring = Some(ring);
            self.ring_min_rows = min_rows;
            Ok(slots)
        }

        /// Re-plans the ring against the bank as decode left it, at the start of every prompt's
        /// chunked prefill.
        pub(in crate::family::flash_next) fn begin_prefill(&mut self) -> Result<(), EngError> {
            self.ring_engaged = false;
            if let Some(ring) = &mut self.ring {
                // A save-point segment re-opens the window with no intervening end_prefill, so
                // the prior segment's ring-stream copies may still be outstanding on the shared
                // stores. Drain them before this window's bank-stream union-admit copies refuse
                // a copy outstanding on the copy stream. Only ring-stream pendings are ours;
                // decode's bank-stream copies are left alone.
                let layers = self.bank.geometry().layers;
                for layer in 0..layers {
                    if self.bank.store_pending_stream(layer) == Some(ring.copy_stream().id()) {
                        self.bank.synchronize_store(layer, ring.copy_stream())?;
                    }
                }
                ring.replan(&self.bank);
                ring.set_active(true);
                ring.begin_chunk();
            }
            Ok(())
        }

        /// Closes the prefill window: waits the ring's copies out and hands the pinned stores
        /// back to the decode path, whose miss copies run on the bank's own stream.
        pub(in crate::family::flash_next) fn end_prefill(&mut self) -> Result<(), EngError> {
            let Some(ring) = self.ring.as_mut() else {
                return Ok(());
            };
            if !ring.active() {
                return Ok(());
            }
            let layers = self.bank.geometry().layers;
            for layer in 0..layers {
                self.bank.synchronize_store(layer, ring.copy_stream())?;
            }
            ring.set_active(false);
            Ok(())
        }

        /// Resets the ring's per-chunk cursors (the chunk driver's `begin_chunk`).
        pub(in crate::family::flash_next) fn begin_chunk(&mut self) {
            if let Some(ring) = &mut self.ring {
                ring.begin_chunk();
            }
        }

        pub(in crate::family::flash_next) fn bank(&self) -> &DeviceBank {
            &self.bank
        }

        /// Queues layer `layer`'s MoE block on `launch.stream`. Reads `bufs.route.norm_row` (f32
        /// `[hidden]`, the hyper-mixed row) and `bufs.route.resid_in` (a zero row); writes the
        /// block output, the weighted picks plus the gated shared expert and no residual, to
        /// `bufs.route.resid_out`. A `resid_in` that is not zero is added into the result. Returns
        /// after every launch of the block is queued. It waits on the stream for the slot row, for
        /// the classify log and, on a miss, for the bank copies. A non-finite filler output of
        /// slot 0 never reaches the sum: the miss pass overwrites its rank row before the only
        /// combine, so the block cannot turn a torn slot-0 float into a NaN.
        pub(in crate::family::flash_next) fn run(
            &mut self,
            launch: &Launch<'_>,
            layer: usize,
            bufs: &Exl3MoeBufs<'_>,
            dims: &MoeDims,
        ) -> Result<(), EngError> {
            if dims.rows != 1 {
                return Err(refuse(format!(
                    "the block takes one row, got {}",
                    dims.rows
                )));
            }
            let geometry = *self.bank.geometry();
            if dims.n_expert != geometry.experts
                || dims.top_k != geometry.top_k
                || layer >= geometry.layers
            {
                return Err(refuse(format!(
                    "layer {layer} with {} experts and {} picks does not fit a bank of {} layers, \
                     {} experts and {} picks",
                    dims.n_expert, dims.top_k, geometry.layers, geometry.experts, geometry.top_k
                )));
            }
            let (ctx, stream) = (launch.ctx, launch.stream);
            let sc = self.layout;
            let (h, k) = (dims.hidden, dims.top_k);
            let profile = prof::on();
            let slot_row_before: Vec<i32> = self.bank.slot_row(layer)?.to_vec();
            let scratch = &self.scratch;
            let slot_ids = scratch.range(sc.slot_ids, k * I32)?;
            let slot_weights = scratch.range(sc.weights, k * F32)?;

            prof::blocked(34, profile, || {
                scratch.write_at(stream, sc.slot_row, &le_i32(&slot_row_before))
            })?;
            let trunk = &self.trunks[layer];
            ctx.router(
                stream,
                RouterCall {
                    shape: RouterShape {
                        m: 1,
                        hidden: h,
                        n_expert: dims.n_expert,
                        top_k: k,
                    },
                    x: bufs.route.norm_row,
                    weights: RouterWeights {
                        w: whole(trunk.router)?,
                        w_shared: Some(whole(trunk.router_shared)?),
                    },
                    logits: bufs.route.logits,
                    out: RouterOut {
                        ids: bufs.route.ids,
                        weights: bufs.route.weights,
                        shared_gate: Some(bufs.route.shared_gate),
                        status: bufs.route.status,
                    },
                },
            )?;
            ctx.tiered_classify(
                stream,
                ClassifyCall {
                    shape: ClassifyShape {
                        n_expert: dims.n_expert,
                        slots: geometry.slots,
                        top_k: k,
                        hidden: h,
                        layer,
                    },
                    ids: bufs.route.ids,
                    weights: bufs.route.weights,
                    slot_row: scratch.range(sc.slot_row, dims.n_expert * I32)?,
                    norm_row: bufs.route.norm_row,
                    slot_ids_out: slot_ids,
                    weights_out: slot_weights,
                    log_out: scratch.range(sc.log, k * LOG_ENTRY)?,
                    ctrl: scratch.range(sc.ctrl, CTRL_BYTES)?,
                    word: scratch.range(sc.word, I32)?,
                    mbox_req: scratch.range(sc.mbox_req, REQUEST_BYTES)?,
                    mbox_row: scratch.range(sc.mbox_row, h * F32)?,
                },
            )?;
            let mut raw = vec![0u8; k * LOG_ENTRY];
            prof::blocked(35, profile, || scratch.read_at(stream, sc.log, &mut raw))?;
            let (ids, weights) = parse_log(&raw, k)?;
            let classified = classify_picks(&slot_row_before, geometry.slots, &ids, &weights)?;

            // The resident pass. Without a miss its combine IS the block's output; with one it
            // skips the combine entirely (the miss pass's combine is the block's only sum, and
            // the filler rows must not be multiplied by anything).
            let clean = classified.misses.is_empty();
            self.pass(
                launch,
                bufs,
                dims,
                Pass {
                    layer,
                    ids: slot_ids,
                    weights: slot_weights,
                    shared: true,
                    shared_term_only: false,
                    down_rows: None,
                    combine: clean.then_some((bufs.route.resid_in, bufs.route.resid_out)),
                },
            )?;

            // The copies for every pick, hits included; they run behind the resident pass.
            self.bank.step(ctx, layer, &ids)?;
            let slot_row_after = self.bank.slot_row(layer)?.to_vec();
            let second = plan_second_pass(&ids, &classified, &slot_row_after)?;
            if let Some((miss_slots, miss_ranks)) = second.miss {
                let n = miss_slots.len();
                let scratch = &self.scratch;
                prof::blocked(36, profile, || {
                    scratch.write_at(stream, sc.miss_ids, &le_i32(&miss_slots))
                })?;
                self.pass(
                    launch,
                    bufs,
                    dims,
                    Pass {
                        layer,
                        ids: scratch.range(sc.miss_ids, n * I32)?,
                        // The router's device weights: the router wrote the renormalized
                        // `top_k` weights to `route.weights`, and the combine reads them in
                        // place — the classified scratch weights hold +0.0 at missed ranks and
                        // cannot serve.
                        weights: bufs.route.weights,
                        shared: false,
                        shared_term_only: true,
                        down_rows: Some(&miss_ranks),
                        combine: Some((bufs.route.resid_in, bufs.route.resid_out)),
                    },
                )?;
                // The missed ranks' filler rows are overwritten before the only combine reads
                // them, so no +0.0 filler product exists anywhere in the sum.
            }
            Ok(())
        }

        /// The buffers of row `r` of a multi-row chunk, as the one-row [`Self::run`] sees them:
        /// the miss pass of the prefill store reuses `pass` over them.
        fn row_of<'b>(
            bufs: &Exl3MoeBufs<'b>,
            dims: &MoeDims,
            r: usize,
        ) -> Result<Exl3MoeBufs<'b>, EngError> {
            let (h, i, k) = (dims.hidden, dims.inter, dims.top_k);
            Ok(Exl3MoeBufs {
                route: RouteBufs {
                    resid_in: row(&bufs.route.resid_in, h * F32, r)?,
                    resid_out: row(&bufs.route.resid_out, h * F32, r)?,
                    norm_row: row(&bufs.route.norm_row, h * F32, r)?,
                    logits: row(&bufs.route.logits, (dims.n_expert + 1) * F32, r)?,
                    ids: row(&bufs.route.ids, k * F32, r)?,
                    weights: row(&bufs.route.weights, k * F32, r)?,
                    shared_gate: row(&bufs.route.shared_gate, F32, r)?,
                    status: bufs.route.status,
                },
                expert_gu: row(&bufs.expert_gu, k * 2 * i * F32, r)?,
                expert_act: row(&bufs.expert_act, k * i * F32, r)?,
                expert_down: row(&bufs.expert_down, k * h * F32, r)?,
                shared_gu: row(&bufs.shared_gu, 2 * dims.shared_inter * F32, r)?,
                shared_act: row(&bufs.shared_act, dims.shared_inter * F32, r)?,
                shared_out: row(&bufs.shared_out, h * F32, r)?,
                scratch: bufs.scratch,
            })
        }

        /// One router launch over the chunk; the picks stay on the device for the shared pass.
        fn route(
            &self,
            launch: &Launch<'_>,
            layer: usize,
            bufs: &Exl3MoeBufs<'_>,
            dims: &MoeDims,
        ) -> Result<(), EngError> {
            let m = dims.rows;
            let (h, k, e) = (dims.hidden, dims.top_k, dims.n_expert);
            let (ctx, stream) = (launch.ctx, launch.stream);
            let trunk = &self.trunks[layer];
            ctx.router_rows(
                stream,
                RouterCall {
                    shape: RouterShape {
                        m,
                        hidden: h,
                        n_expert: e,
                        top_k: k,
                    },
                    x: bufs.route.norm_row,
                    weights: RouterWeights {
                        w: whole(trunk.router)?,
                        w_shared: Some(whole(trunk.router_shared)?),
                    },
                    logits: bufs.route.logits,
                    out: RouterOut {
                        ids: bufs.route.ids,
                        weights: bufs.route.weights,
                        shared_gate: Some(bufs.route.shared_gate),
                        status: bufs.route.status,
                    },
                },
            )
        }

        /// The promoted FP16-staged pass. The caller supplies its jobs-sized permutation
        /// region, never the legacy classify/split layout's six-words-per-pair region.
        /// The caller has written the dense job table before the first layer of the chunk.
        pub(in crate::family::flash_next) fn run_prefill(
            &mut self,
            projector: &Projector<'_>,
            layer: usize,
            bufs: &Exl3MoeBufs<'_>,
            dims: &MoeDims,
            perm: DeviceRange<'_>,
        ) -> Result<(), EngError> {
            let geometry = *self.bank.geometry();
            if !(1..=MAX_EXL3_ROWS).contains(&dims.rows)
                || dims.rows > self.layout.rows
                || projector.rows != dims.rows
                || layer >= geometry.layers
                || dims.n_expert != geometry.experts
                || dims.top_k != geometry.top_k
            {
                return Err(refuse(
                    "the shared pass does not fit the planned bank window".into(),
                ));
            }
            let (_, perm_bytes) =
                exl3_job_bytes(dims.rows, projector.tile_rows, dims.n_expert, dims.top_k)
                    .ok_or_else(|| refuse("the shared job sizes overflow".into()))?;
            if perm.bytes() < perm_bytes {
                return Err(refuse("the shared permutation range is too short".into()));
            }
            let ring_active = self.ring.as_ref().is_some_and(|ring| ring.active());
            // A window whose chunks are all below `ring_min_rows` takes the bank's union-admit
            // path: a small chunk's picks usually fit the bank without admission, and the ring
            // re-issues its copies every chunk, which costs more than grouping saves there.
            // Once a window streams a whole chunk it latches onto the ring (`ring_engaged`) for
            // the rest of the window, so the trailing single-row logits step does not switch to
            // the bank stream while the ring's copies are outstanding on the copy stream.
            // Invariant: pieces within a window never grow (the chunk driver feeds equal-sized
            // pieces, a smaller remainder, then the 1-row logits step), so a union admit never
            // follows a whole chunk in the same window and the begin_prefill ring plan stays
            // valid; each scheduler piece reopens its own window.
            let whole_chunk = crate::family::flash_next::settings::ring_whole_chunk(
                ring_active,
                dims.rows,
                self.ring_min_rows,
                self.ring_engaged,
            );
            let sub_rows = if whole_chunk {
                // The frozen bank plus the ring cover every expert; one whole-chunk plan
                // groups each source, without bank admission or row-at-a-time launches.
                dims.rows
            } else {
                geometry
                    .slots
                    .checked_div(dims.top_k)
                    .filter(|&n| n > 0)
                    .ok_or_else(|| refuse("the bank cannot hold one routed row".into()))?
            };
            let sub: Vec<_> = (0..dims.rows)
                .step_by(sub_rows)
                .map(|r0| {
                    (
                        r0,
                        MoeDims {
                            rows: sub_rows.min(dims.rows - r0),
                            ..*dims
                        },
                    )
                })
                .collect();
            let launch = projector.launch;
            if whole_chunk {
                self.ring_engaged = true;
                self.ring.as_mut().expect("active ring").issue_through(
                    launch.ctx,
                    &mut self.bank,
                    layer + 1,
                )?;
            }
            self.route(launch, layer, bufs, dims)?;
            let bytes = dims.rows * dims.hidden * F32;
            launch.ctx.copy_device(
                launch.stream,
                window(bufs.route.resid_in, 0, bytes)?,
                window(bufs.route.resid_out, 0, bytes)?,
            )?;
            let common = prefill_moe::Bufs {
                x2: bufs.route.norm_row,
                ids: bufs.route.ids,
                weights: bufs.route.weights,
                shared_gate: bufs.route.shared_gate,
                hidden: bufs.route.resid_out,
                perm,
                gu: bufs.expert_gu,
                act: bufs.expert_act,
                paired: bufs.expert_down,
                shared_gu: bufs.shared_gu,
                shared_act: bufs.shared_act,
                shared_out: bufs.shared_out,
                expert_xh: bufs.scratch.xh,
                expert_z: bufs.scratch.z,
            };
            let trunk = &self.trunks[layer];
            let weights = SharedWeights {
                gate: trunk.shared_gate,
                up: trunk.shared_up,
                down: trunk.shared_down,
            };
            if whole_chunk {
                let ring = self.ring.as_ref().expect("active ring");
                let mut store = RingStore {
                    adapter: PrefillAdapter::new(&mut self.bank),
                    ring,
                    layer,
                    spans: &self.spans,
                    ring_spans: SlotSpans::new(&self.bank_layout, ring.ring_slots())?,
                    book: self.book,
                };
                prefill_moe::run_store(projector, &weights, &common, dims, &sub, &mut store)?;
                // Record used events only after both sources' reads are queued. A reused
                // ring slot's next copy waits these on the copy stream.
                return self
                    .ring
                    .as_mut()
                    .expect("active ring")
                    .free_layer(launch.stream, layer);
            }
            // No ring drain is needed here: ring copies are issued only by the whole-chunk
            // branch, which latches `ring_engaged`, so a union admit never follows a ring copy
            // in the same window; `begin_prefill` drains a reopened window's outstanding ring
            // copies before the latch resets.
            let ring_funded = self.ring.is_some();
            let mut store = LayerStore {
                adapter: PrefillAdapter::new(&mut self.bank),
                ctx: launch.ctx,
                layer,
                spans: &self.spans,
                book: self.book,
                slot_row: self
                    .scratch
                    .range(self.layout.slot_row, dims.n_expert * I32)?,
            };
            prefill_moe::run_store(projector, &weights, &common, dims, &sub, &mut store).map_err(
                |error| {
                    // The admit-site context is ring-path only; without a ring the union admit
                    // is the stock shared-prefill path and its error text stays unchanged.
                    if ring_funded {
                        match error {
                            EngError::BadRequest(message) => EngError::BadRequest(format!(
                                "prefill union admit at layer {layer}: {message}"
                            )),
                            other => other,
                        }
                    } else {
                        error
                    }
                },
            )
        }

        /// One id-indexed launch group over the bank buffer of `layer`, chunked at the launch
        /// capacity; every chunk starts again at row 0 of the scratch.
        fn grouped_at<'a>(
            &'a self,
            launch: &Launch<'_>,
            items: &[Item<'a>],
            bank: (&DeviceBuffer, u32),
            ids: DeviceRange<'_>,
            scratch: &Scratch<'a>,
        ) -> Result<(), EngError> {
            let first = items
                .first()
                .ok_or_else(|| refuse("an indexed launch without pairs".into()))?;
            let [k, n] = self.spans.parts[first.part].shape;
            let mut pairs = Vec::with_capacity(items.len());
            for item in items {
                let (weights, members, spans) = (bank.0, bank.1, &self.spans);
                let span = &spans.parts[item.part];
                if span.shape != [k, n] {
                    return Err(refuse("a launch group mixes matrix shapes".into()));
                }
                pairs.push(IndexedPair {
                    span: Exl3BankSpan::from_parts(
                        weights.range(span.trellis.start, span.trellis.len())?,
                        weights.range(span.suh.start, span.suh.len())?,
                        weights.range(span.svh.start, span.svh.len())?,
                        members,
                        self.spans.trellis_stride,
                        self.spans.scale_stride,
                        span.shape,
                        span.bits_twice,
                        self.book,
                    ),
                    slot: item.pick,
                    x: item.x,
                    out: item.out,
                });
            }
            grouped_indexed(launch, &pairs, ids, scratch)
        }

        /// One id-indexed launch group over the bank buffer of `layer`, chunked at the launch
        /// capacity; every chunk starts again at row 0 of the scratch.
        fn grouped<'a>(
            &'a self,
            launch: &Launch<'_>,
            layer: usize,
            items: &[Item<'a>],
            (ids, scratch): (DeviceRange<'a>, &Scratch<'a>),
        ) -> Result<(), EngError> {
            let members = u32::try_from(self.spans.slots)
                .map_err(|_| refuse("the bank has too many slots".into()))?;
            self.grouped_at(
                launch,
                items,
                (self.bank.slots(layer)?, members),
                ids,
                scratch,
            )
        }

        /// One pass over the bank of `pass.layer`: gate and up of the picks, SwiGLU, down, the
        /// shared expert when asked, and the combine into `pass.combine`'s output when asked.
        fn pass(
            &self,
            launch: &Launch<'_>,
            bufs: &Exl3MoeBufs<'_>,
            dims: &MoeDims,
            pass: Pass<'_>,
        ) -> Result<(), EngError> {
            let (h, i) = (dims.hidden, dims.inter);
            // The resident pass carries top_k picks; the miss pass only the missed ones, so
            // the width comes from the ids range, not the dims.
            let k = pass.ids.bytes() / I32;
            let mut items = Vec::with_capacity(2 * k);
            for pick in 0..k {
                for part in 0..2usize {
                    items.push(Item {
                        part,
                        pick: pick as u32,
                        x: bufs.route.norm_row,
                        out: row(&bufs.expert_gu, i * F32, 2 * pick + part)?,
                    });
                }
            }
            self.grouped(launch, pass.layer, &items, (pass.ids, &bufs.scratch))?;
            launch.ctx.swiglu_rows(
                launch.stream,
                SwigluRowsCall {
                    shape: SwigluRowsShape {
                        rows: k,
                        n_ff: i,
                        up_off: i,
                        ld_in: 2 * i,
                        ld_out: i,
                    },
                    gu: bufs.expert_gu,
                    act: bufs.expert_act,
                },
            )?;
            let mut items = Vec::with_capacity(k);
            for pick in 0..k {
                let down_row = pass.down_rows.map_or(pick, |rows| rows[pick] as usize);
                items.push(Item {
                    part: 2,
                    pick: pick as u32,
                    x: row(&bufs.expert_act, i * F32, pick)?,
                    out: row(&bufs.expert_down, h * F32, down_row)?,
                });
            }
            self.grouped(launch, pass.layer, &items, (pass.ids, &bufs.scratch))?;
            let shared = if pass.shared {
                let trunk = &self.trunks[pass.layer];
                shared_of(
                    launch,
                    &SharedWeights {
                        gate: trunk.shared_gate,
                        up: trunk.shared_up,
                        down: trunk.shared_down,
                    },
                    bufs,
                    dims,
                )?;
                Some(SharedTerm {
                    gate: bufs.route.shared_gate,
                    row: bufs.shared_out,
                })
            } else if pass.shared_term_only {
                Some(SharedTerm {
                    gate: bufs.route.shared_gate,
                    row: bufs.shared_out,
                })
            } else {
                None
            };
            // The combine width is `dims.top_k` for both passes — the resident pass's slot
            // weights and the miss pass's router weights are both `top_k` wide.
            if let Some((res, out)) = pass.combine {
                let cw = dims.top_k;
                launch.ctx.moe_combine(
                    launch.stream,
                    CombineCall {
                        shape: CombineShape {
                            m: 1,
                            n: h,
                            top_k: cw,
                            ld_down: h,
                        },
                        res,
                        down: bufs.expert_down,
                        weights: pass.weights,
                        shared,
                        out: CombineOut::To(out),
                    },
                )?
            }
            Ok(())
        }
    }
}

/// The distinct expert ids of a chunk's picks, first use first. The order only matters when
/// the union overflows the bank (the oldest uses stay); on the admit path it always fits.
pub(in crate::family::flash_next) fn pick_union(rows: &[&[i32]], n_expert: usize) -> Vec<u16> {
    let mut seen = vec![false; n_expert];
    let mut union = Vec::with_capacity(rows.first().map_or(0, |row| row.len()));
    for row in rows {
        for &id in *row {
            // The prefill store refuses ids outside 0..n_expert before the first pick_union call.
            if !seen[id as usize] {
                seen[id as usize] = true;
                union.push(id as u16);
            }
        }
    }
    union
}
