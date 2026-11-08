//! The step plan: which activation tensors one resident decode step needs, how big they are, and
//! when each is live.
//!
//! Pure host arithmetic. Every size comes from the layout record (`HostShape`), from the output
//! rows of a tensor role in the inventory, or from a [`KernelScratch`] that the caller fills with
//! the byte counts of the kernel wrappers; no model number is written here. The lifetimes are in
//! phases of ONE layer. All layers reuse the same offsets, so a layer is a single time axis: a
//! DeltaNet layer runs the phases `DELTA_IN` to `DELTA_OUT`, an attention layer `ATTN_IN` to
//! `ATTN_OUT`, and the tensors of the two mixers may share addresses because a layer runs only one
//! of them. `StepParams`, the result pair and the prefill status word are NOT in this plan: they
//! live in the words block of `state_layout.rs`, so every plan (every row count) and the prefill
//! path see the same addresses. Prefill has its own plan and arena; nothing here is a prefill size.
//!
//! # The two layouts
//!
//! [`StepOptions::layout`] selects the pack slots: the `Tiled` plan (fp16 `Xf*` packs, pinned by
//! the existing tests) or the `Native` plan, whose `Q8*` slots are the padded `block_q8_1` rows of
//! `q8_1_bytes` at the SAME lifetimes (`Q8Down` holds the `top_k` rows of the expert slots in ONE
//! pack). Every non-pack slot is shared but `NormRow`, which the NATIVE plan holds live across the
//! whole step because the fused `add_rmsnorm_q8` always writes the f32 normalized row (the MIXER
//! PROLOGUE writes it as scratch beside Q8Mix; the final norm writes its own FinalNormRow instead).
//! The Native plan exists at one row only (`check_layout_rows`).
//!
//! # Rows
//!
//! A plan is built for `rows` in `1..=MAX_STEP_ROWS` (8): plain decode is 1, a verify window is
//! 2..=8, and the merged `matmul_expert` takes `m` in `1..=8` (`resident/matmul_expert.rs:36`). The
//! tensors that grow with the row count are the residual and mixer rows, the router and top-k
//! tensors, the expert tensors, `AttnPart` (about 1.06 MB per row), `Logits` (993,280 bytes per
//! row) and, once a per-row argmax exists, the `argmax_part`. The `Xf*` packs are one fixed
//! fragment-order buffer up to 16 rows. A session that runs several row counts keeps one
//! `DeviceArena` sized for its largest plan and reaches the smaller plans through
//! `DeviceArena::view` at their own offsets.
//!
//! # Streams
//!
//! The lifetimes are one time axis and assume stream order. Where work runs on branch streams
//! (`forked_experts`), the branches must join the main stream with events at the phase boundaries
//! the table states, or the aliasing is unsafe (the same holds for `AttnPart`).
//!
//! # Order of the status writers
//!
//! The status word (mask 1 router fault, 2 non-finite logit, 4 invalid step state; a prefill chunk
//! has its own word with mask 8) is written by `step_begin` (first kernel: clears it, then sets 4),
//! by the embedding (ORs 4 for a token id outside the vocabulary), by `attn_prep` (ORs 4 on a
//! refused launch), by the router (ORs 1) and by `argmax` (ORs 2; its `atomicOr` return value is
//! the status half of the result pair). `argmax` MUST be the last kernel that writes status in the
//! step; any later writer breaks the pair. Under Int8G32 the rows entry of `attn_prep` holds the
//! `ATTN_PREP` phase for the same mask-4 rule, and `kv8_append` (phase `KV8_APPEND`) also ORs 4 on
//! a bad pos word — but the rows entry has already refused that same word against the same cell
//! bound one launch earlier, so the append can never be the first to set a bit; the writer list of
//! the composed step now names `KV8Append` itself (`writes_status` is true for it), still before
//! `argmax`.
//!
//! What the compiler checks, and what it does not: [`writes_status`] is an exhaustive `match` over
//! [`Phase`], so adding a phase forces a decision about status; `status_order_ok` (a `const fn`
//! evaluated in a `const` assertion) walks the phase table `PHASES` that the plan's lifetimes are
//! written in and requires the table to be strictly ordered, `StepBegin` to be first and a writer,
//! every writer to be at or before `Argmax`, and `Advance` to be after it and no writer. It cannot
//! see a kernel that is launched but missing from the table: the real order of the launches is the
//! one `step_ops.rs` lists, and its order test is the gate. This is a documented rule with a table
//! check, not a proof.
//!
//! The host maps the status before it reads the token: precedence 4 (`step_params`), then 8
//! (`prefill_chunk`, prefill word only), then 1 (`route_experts`), then 2 (`sample`); the adapter
//! sets no latch, the session state is poisoned until reset, and a mask 4 report has two sources: a
//! defect (the host check `ensure_rows` ran first and missed a case), or the embedding gather,
//! which sets mask 4 for a token id outside the vocabulary. The status word cannot tell the two
//! causes apart.
//!
//! The accounting and the plan print are per component.

use std::fmt;

use super::{
    dims::{KvMode, StepDims, StepOptions},
    mixer_geometry::MixerGeometry,
    shard::ShardTable,
    state_layout::align_up,
};
use crate::{
    EngError,
    family::project::cuda::WeightLayout,
    kernels::cuda::{
        ArenaPlan, ArenaTensor, plan_arena,
        resident::{
            attn_decode::{AttnSplit, attn_part_bytes},
            attn_decode_q8::attn_q8_part_bytes,
            quantize_q8_1::q8_1_bytes,
            router::{RouterShape, router_logits_bytes},
            step_params::ARGMAX_PART_BYTES,
        },
        xf_bytes,
    },
};

const F32: usize = size_of::<f32>();
const I32: usize = size_of::<i32>();

/// Most rows of one decode step: the merged `matmul_expert` takes `m` in `1..=8`
/// (`resident/matmul_expert.rs:36`) and the verify window is capped at 8. The split-KV attention
/// and the fragment-order `xf` buffer would allow 16; a plan for 9 to 16 rows could never run.
pub(crate) const MAX_STEP_ROWS: usize = 8;

/// Scratch bytes of the stage 1 kernels for one row count. `attn_part` and `argmax_part` come from
/// the kernel wrappers (`attn_part_bytes` with `AttnSplit::STAGE1`, `ARGMAX_PART_BYTES`,
/// `router_logits_bytes`), so each size has one owner; under `Int8G32` the part scratch is the KV8
/// kernel's own `attn_q8_part_bytes` with `AttnSplit::Q8`. [`Self::for_dims`] takes every size from
/// its wrapper; [`Self::new`] takes a router size from the caller and the plan refuses anything
/// below `4 * rows * (experts + 1)` (1,028 at one row).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct KernelScratch {
    rows: usize,
    router_logits: usize,
    attn_part: usize,
    /// The split `attn_part` was sized with: STAGE1 under `Fp16`, `Q8` under `Int8G32`. The two
    /// constants share `max_splits` 64, so the byte count alone cannot tell them apart; this
    /// field is what the plan checks against the KV mode.
    attn_split: AttnSplit,
    argmax_part: usize,
}

impl KernelScratch {
    pub(crate) fn for_dims(
        dims: &StepDims,
        rows: usize,
        kv_mode: KvMode,
    ) -> Result<Self, EngError> {
        let moe = &dims.shape.moe;
        let router = router_logits_bytes(RouterShape {
            m: rows,
            hidden: dims.shape.hidden,
            n_expert: moe.experts,
            top_k: moe.top_k,
        })?;
        Self::new(dims, rows, router, kv_mode)
    }

    pub(crate) fn new(
        dims: &StepDims,
        rows: usize,
        router_logits: usize,
        kv_mode: KvMode,
    ) -> Result<Self, EngError> {
        let attn = &dims.shape.attn;
        // One arm per mode: the split constant and its own sizing wrapper stay inseparable, so
        // a scratch cannot carry a split its bytes were not sized with.
        let (attn_split, attn_part) = match kv_mode {
            KvMode::Fp16 => (
                AttnSplit::STAGE1,
                attn_part_bytes(
                    rows,
                    attn.heads_kv,
                    attn.head_dim,
                    AttnSplit::STAGE1.max_splits,
                )?,
            ),
            KvMode::Int8G32 => (
                AttnSplit::Q8,
                attn_q8_part_bytes(rows, attn.heads_kv, attn.head_dim, AttnSplit::Q8.max_splits)?,
            ),
        };
        Ok(Self {
            rows,
            router_logits,
            attn_part,
            attn_split,
            argmax_part: ARGMAX_PART_BYTES,
        })
    }

    pub(crate) fn attn_split(&self) -> AttnSplit {
        self.attn_split
    }

    fn check(&self, dims: &StepDims, rows: usize, kv_mode: KvMode) -> Result<(), EngError> {
        let router_min = product(&[rows, dims.shape.moe.experts + 1, F32])?;
        let want = match kv_mode {
            KvMode::Fp16 => AttnSplit::STAGE1,
            KvMode::Int8G32 => AttnSplit::Q8,
        };
        if self.rows != rows || self.router_logits < router_min {
            return Err(EngError::BadRequest(format!(
                "step plan: the kernel scratch was made for {} rows with {} router bytes, the plan needs {rows} rows and at least {router_min}",
                self.rows, self.router_logits
            )));
        }
        if self.attn_split != want {
            return Err(EngError::BadRequest(format!(
                "step plan: the kernel scratch sized AttnPart with the {:?} split, the {kv_mode:?} KV mode needs the {want:?} split",
                self.attn_split
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Slot {
    ResidA,
    ResidB,
    /// fp16 pack feeding the mixer in-projections (`k` = hidden).
    XfMix,
    /// The padded q8_1 row feeding the Native mixer in-projections (`k` = hidden).
    Q8Mix,
    Exl3Xh,
    Exl3Z,
    Exl3Flag,
    /// f32 `[rows][delta_proj]`: qkv, z, beta, alpha side by side.
    DeltaProj,
    DeltaConv,
    DeltaCore,
    XfDeltaOut,
    /// The padded q8_1 row of the delta-rule output (`k` = value width).
    Q8DeltaOut,
    /// f32 `[rows][attn_proj]`: per query head `[q | gate]`, then k, then v.
    AttnProj,
    AttnQ,
    /// f32 `[rows][n_kv_head][head_dim]` k rows of the KV handoff: written by the
    /// rows entry of `attn_prep` at `ATTN_PREP`, consumed by `kv8_append` or
    /// `kv_f16_append` at `KV8_APPEND`. In single-device plans present only under
    /// `KvMode::Int8G32` (the fused fp16 path writes the caches directly); in CARD
    /// plans present under BOTH modes, and sized at the FULL row width — every
    /// card's buffer receives the whole gathered row, not its own extent.
    KRows,
    /// The v twin of [`Slot::KRows`]: the unchanged, un-roped v rows of the same handoff.
    VRows,
    /// Split-KV attention partials; written by the part kernel and read by the combine kernel, both
    /// in `ATTN_CORE` (under `Int8G32` the KV8 kernel's own partials, same phase).
    AttnPart,
    AttnCore,
    XfAttnOut,
    /// The padded q8_1 row of the attention output (`k` = q width).
    Q8AttnOut,
    /// Mixer output `[rows][hidden]`, added into the residual by the next norm.
    MixOut,
    XfMoe,
    /// The padded q8_1 row of the mixer epilogue norm (`k` = hidden).
    Q8Moe,
    /// f32 normalized row, the router input.
    NormRow,
    /// Router logits `[rows][experts + 1]`; the last column is the shared-gate dot.
    RouterLogits,
    TopkId,
    TopkWeight,
    SharedGate,
    ExpertGu,
    /// Pack feeding the expert down projection; one per routed expert when forked.
    XfDown(usize),
    /// The padded q8_1 rows of the eight expert slots (`k` = inter).
    Q8Down,
    ExpertDown,
    /// f32 `[top_k][inter]`: the SwiGLU of the routed experts, the EXL3 down projections' input.
    ExpertAct,
    SharedGu,
    /// f32 `[shared_inter]`: the SwiGLU of the shared expert (EXL3 plans only).
    SharedAct,
    XfShared,
    /// The padded q8_1 row of the shared expert (`k` = shared inter).
    Q8Shared,
    SharedOut,
    XfFinal,
    /// The padded q8_1 row of the output norm (`k` = hidden).
    Q8Final,
    /// The own f32 scratch row of the NATIVE final norm: the fused producer must write an f32 row
    /// beside the pack. `NormRow` alone would be SOUND here (its native lifetime MIX_PACK..ARGMAX
    /// covers FINAL_NORM, and the planner keeps live slots byte-disjoint) - the own slot is
    /// deliberate isolation instead: the final norm then depends on no other op's slot, and the
    /// write cannot interfere with the mixer's NormRow bytes whatever a later plan changes.
    FinalNormRow,
    Logits,
    ArgmaxPart,
}

/// Phases of one layer plus the prologue and epilogue. A layer runs `MixPack`, then `DeltaIn..=
/// DeltaOut` or `AttnIn..=AttnOut`, then `AddNorm` onward. The discriminant is the lifetime number.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
pub(super) enum Phase {
    StepBegin = 0,
    Embed = 1,
    MixPack = 2,
    DeltaIn = 3,
    DeltaConv = 4,
    DeltaRule = 5,
    DeltaPack = 6,
    DeltaOut = 7,
    AttnIn = 8,
    AttnPrep = 9,
    /// The KV8 append of the Int8G32 step: `kv8_append` quantizes the f32 k and v rows of
    /// `ATTN_PREP` into the code and scale planes, before the decode attention reads them.
    /// A single-device Fp16 plan never enters this phase (the fused prep writes the caches
    /// directly); a CARD Fp16 plan runs `kv_f16_append` here over its gathered rows.
    Kv8Append = 10,
    AttnCore = 11,
    AttnPack = 12,
    AttnOut = 13,
    AddNorm = 14,
    Route = 15,
    GateUp = 16,
    Swiglu = 17,
    Down = 18,
    Combine = 19,
    FinalNorm = 22,
    LmHead = 23,
    Argmax = 24,
    /// `step_advance`: the last kernel of the step. It writes `pos`, `kv_len` and `kv_slot`,
    /// never `status`.
    Advance = 25,
}

/// The phase table the plan's lifetimes are written in, in launch order.
pub(super) const PHASES: [Phase; 24] = [
    Phase::StepBegin,
    Phase::Embed,
    Phase::MixPack,
    Phase::DeltaIn,
    Phase::DeltaConv,
    Phase::DeltaRule,
    Phase::DeltaPack,
    Phase::DeltaOut,
    Phase::AttnIn,
    Phase::AttnPrep,
    Phase::Kv8Append,
    Phase::AttnCore,
    Phase::AttnPack,
    Phase::AttnOut,
    Phase::AddNorm,
    Phase::Route,
    Phase::GateUp,
    Phase::Swiglu,
    Phase::Down,
    Phase::Combine,
    Phase::FinalNorm,
    Phase::LmHead,
    Phase::Argmax,
    Phase::Advance,
];

/// Whether the kernel of `phase` writes the status word. Exhaustive: a new phase needs an answer.
pub(super) const fn writes_status(phase: Phase) -> bool {
    match phase {
        Phase::StepBegin
        | Phase::Embed
        | Phase::AttnPrep
        | Phase::Kv8Append
        | Phase::Route
        | Phase::Argmax => true,
        // The table states what the kernel does: `kv8_append` ORs mask 4 into the status word
        // on a bad pos word. The OR is still redundant — the rows entry of `attn_prep`
        // refused the SAME word against the SAME cell bound one launch earlier, so the append
        // is never the first or only writer of the composed step.
        Phase::MixPack
        | Phase::DeltaIn
        | Phase::DeltaConv
        | Phase::DeltaRule
        | Phase::DeltaPack
        | Phase::DeltaOut
        | Phase::AttnIn
        | Phase::AttnCore
        | Phase::AttnPack
        | Phase::AttnOut
        | Phase::AddNorm
        | Phase::GateUp
        | Phase::Swiglu
        | Phase::Down
        | Phase::Combine
        | Phase::FinalNorm
        | Phase::LmHead
        | Phase::Advance => false,
    }
}

/// The documented status rule over the phase table (see the module doc for what it cannot see).
const fn status_order_ok() -> bool {
    let argmax = Phase::Argmax as usize;
    if PHASES[0] as usize != 0 || !writes_status(PHASES[0]) {
        return false;
    }
    let mut at = 1;
    while at < PHASES.len() {
        let (phase, before) = (PHASES[at] as usize, PHASES[at - 1] as usize);
        if phase <= before || (writes_status(PHASES[at]) && phase > argmax) {
            return false;
        }
        at += 1;
    }
    (Phase::Advance as usize) > argmax
}

const _: () = assert!(
    status_order_ok(),
    "argmax must be the last kernel that writes status in a step"
);

const EMBED: usize = Phase::Embed as usize;
const MIX_PACK: usize = Phase::MixPack as usize;
const DELTA_IN: usize = Phase::DeltaIn as usize;
const DELTA_CONV: usize = Phase::DeltaConv as usize;
const DELTA_RULE: usize = Phase::DeltaRule as usize;
const DELTA_PACK: usize = Phase::DeltaPack as usize;
const DELTA_OUT: usize = Phase::DeltaOut as usize;
const ATTN_IN: usize = Phase::AttnIn as usize;
const ATTN_PREP: usize = Phase::AttnPrep as usize;
const KV8_APPEND: usize = Phase::Kv8Append as usize;
const ATTN_CORE: usize = Phase::AttnCore as usize;
const ATTN_PACK: usize = Phase::AttnPack as usize;
const ATTN_OUT: usize = Phase::AttnOut as usize;
const ADD_NORM: usize = Phase::AddNorm as usize;
const ROUTE: usize = Phase::Route as usize;
const GATE_UP: usize = Phase::GateUp as usize;
const SWIGLU: usize = Phase::Swiglu as usize;
const DOWN: usize = Phase::Down as usize;
const COMBINE: usize = Phase::Combine as usize;
const FINAL_NORM: usize = Phase::FinalNorm as usize;
const LM_HEAD: usize = Phase::LmHead as usize;
const ARGMAX: usize = Phase::Argmax as usize;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PlannedSlot {
    pub(crate) slot: Slot,
    pub(crate) bytes: usize,
    pub(crate) first: usize,
    pub(crate) last: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct StepPlan {
    rows: usize,
    slots: Vec<PlannedSlot>,
    arena: ArenaPlan,
    unaliased: usize,
}

fn product(parts: &[usize]) -> Result<usize, EngError> {
    parts.iter().try_fold(1usize, |total, &part| {
        total
            .checked_mul(part)
            .ok_or_else(|| EngError::MemoryPlan("step plan: a tensor size overflows".to_owned()))
    })
}

impl StepPlan {
    pub(crate) fn new(
        dims: &StepDims,
        rows: usize,
        options: &StepOptions,
        scratch: &KernelScratch,
    ) -> Result<Self, EngError> {
        if !(1..=MAX_STEP_ROWS).contains(&rows) {
            return Err(EngError::BadRequest(format!(
                "step plan: {rows} rows, a decode step takes 1 to {MAX_STEP_ROWS}"
            )));
        }
        check_layout_rows(rows, options)?;
        scratch.check(dims, rows, options.kv_mode)?;
        let slots = list_slots(dims, rows, options, scratch)?;
        Self::from_slots(rows, slots)
    }

    /// Explicit EXL3 mixer-only arena; the MoE and serving step are not admitted by this plan.
    pub(super) fn exl3_mixers(
        dims: &StepDims,
        options: &StepOptions,
        scratch: &KernelScratch,
    ) -> Result<Self, EngError> {
        if options.layout != WeightLayout::Exl3 {
            return Err(EngError::BadRequest(
                "EXL3 mixer plan needs EXL3 weights".into(),
            ));
        }
        scratch.check(dims, 1, options.kv_mode)?;
        let mut list = SlotList {
            rows: 1,
            slots: Vec::new(),
        };
        list_mixers(&mut list, dims, options, scratch)?;
        list.add(
            Slot::NormRow,
            list.row(dims.shape.hidden),
            MIX_PACK,
            ADD_NORM,
        )?;
        // The fused norm still emits its tiled pack, which the EXL3 projections never read.
        list.add(Slot::XfMoe, list.xf(dims.shape.hidden), ADD_NORM, ADD_NORM)?;
        let g = super::mixer_geometry::MixerGeometry::new(dims)?;
        let specs = g
            .dense_specs(WeightLayout::Exl3, crate::family::AttentionKind::DeltaNet)?
            .into_iter()
            .chain(g.dense_specs(WeightLayout::Exl3, crate::family::AttentionKind::Full)?);
        let (k, n) = specs.fold((0, 0), |(k, n), s| (k.max(s.k), n.max(s.rows)));
        list.add(Slot::Exl3Xh, list.row(k), DELTA_IN, ATTN_OUT)?;
        list.add(Slot::Exl3Z, list.row(n), DELTA_IN, ATTN_OUT)?;
        list.add(Slot::Exl3Flag, Ok(4), MIX_PACK, ATTN_OUT)?;
        Self::from_slots(1, list.slots)
    }

    /// Paged attention owns its bounded partials outside the activation arena.
    pub(crate) fn new_paged(
        dims: &StepDims,
        options: &StepOptions,
        scratch: &KernelScratch,
    ) -> Result<Self, EngError> {
        if options.layout != WeightLayout::Native || options.kv_mode != KvMode::Int8G32 {
            return Err(EngError::BadRequest(
                "paged step plan requires Native weights and Int8G32 KV".to_owned(),
            ));
        }
        let mut plan = Self::new(dims, 1, options, scratch)?;
        plan.slots.retain(|slot| slot.slot != Slot::AttnPart);
        Self::from_slots(1, plan.slots)
    }

    /// The slot list of ONE CARD of the multi-device step (Native weights, one row). Replicated
    /// and summed rows keep the full shape's width — the residual stream, `MixOut`, the router
    /// and top-k tensors, the final norm — and the head's logits cover only the card's
    /// vocabulary shard. Sharded widths shrink to the card's extent: the delta and attention
    /// projections and their activations over the card's head ranges, the expert gate/up packs
    /// over its inter rows. [`Slot::KRows`] and [`Slot::VRows`] exist under BOTH KV modes at the
    /// FULL row width: every card's buffer receives the whole gathered row, not its own extent.
    pub(super) fn for_card(
        dims: &StepDims,
        table: &ShardTable,
        card: usize,
        options: &StepOptions,
        scratch: &KernelScratch,
    ) -> Result<Self, EngError> {
        if options.layout != WeightLayout::Native {
            return Err(EngError::BadRequest(
                "card step plan: the multi-device step takes the native layout".to_owned(),
            ));
        }
        scratch.check(dims, 1, options.kv_mode)?;
        let geometry = MixerGeometry::for_card(dims, table, card)?;
        let mut list = SlotList {
            rows: 1,
            slots: Vec::new(),
        };
        list_card_mixers(&mut list, dims, &geometry, scratch)?;
        list_card_moe(&mut list, dims, table, card, scratch)?;
        let h = dims.shape.hidden;
        list.add(Slot::Q8Final, list.q8(h, 1), FINAL_NORM, LM_HEAD)?;
        list.add(Slot::FinalNormRow, list.row(h), FINAL_NORM, FINAL_NORM)?;
        let vocab = table.head_rows(card)?.len();
        list.add(Slot::Logits, list.row(vocab), LM_HEAD, ARGMAX)?;
        list.add(Slot::ArgmaxPart, Ok(scratch.argmax_part), ARGMAX, ARGMAX)?;
        Self::from_slots(1, list.slots)
    }

    fn from_slots(rows: usize, slots: Vec<PlannedSlot>) -> Result<Self, EngError> {
        let arena = plan_arena(&arena_tensors(&slots))?;
        let unaliased = slots
            .iter()
            .try_fold(0usize, |sum, s| {
                align_up(s.bytes).and_then(|bytes| sum.checked_add(bytes))
            })
            .ok_or_else(|| EngError::MemoryPlan("step plan: sizes overflow".to_owned()))?;
        Ok(Self {
            rows,
            slots,
            arena,
            unaliased,
        })
    }

    pub(crate) fn tensors(&self) -> &[PlannedSlot] {
        &self.slots
    }

    pub(crate) fn plan(&self) -> &ArenaPlan {
        &self.arena
    }

    /// The list to give `CudaContext::arena`: the device arena then places every tensor at the
    /// offset of this plan (same list, same order, same planner), and `DeviceArena::tensor(index)`
    /// is the range of [`Self::index`].
    pub(crate) fn arena_tensors(&self) -> Vec<ArenaTensor> {
        arena_tensors(&self.slots)
    }

    pub(crate) fn index(&self, slot: Slot) -> Option<usize> {
        self.slots.iter().position(|s| s.slot == slot)
    }

    pub(crate) fn offset(&self, slot: Slot) -> Option<usize> {
        self.arena.offsets.get(self.index(slot)?).copied()
    }

    pub(crate) fn arena_bytes(&self) -> usize {
        self.arena.total
    }

    pub(crate) fn unaliased_bytes(&self) -> usize {
        self.unaliased
    }
}

fn arena_tensors(slots: &[PlannedSlot]) -> Vec<ArenaTensor> {
    slots
        .iter()
        .map(|s| ArenaTensor {
            bytes: s.bytes,
            first: s.first,
            last: s.last,
        })
        .collect()
}

impl fmt::Display for StepPlan {
    /// The arena-plan report line of the qualification run.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "arena-plan rows={} tensors={} arena_bytes={} unaliased_bytes={}",
            self.rows,
            self.slots.len(),
            self.arena.total,
            self.unaliased
        )
    }
}

struct SlotList {
    rows: usize,
    slots: Vec<PlannedSlot>,
}

impl SlotList {
    fn add(
        &mut self,
        slot: Slot,
        bytes: Result<usize, EngError>,
        first: usize,
        last: usize,
    ) -> Result<(), EngError> {
        self.slots.push(PlannedSlot {
            slot,
            bytes: bytes?,
            first,
            last,
        });
        Ok(())
    }

    fn row(&self, width: usize) -> Result<usize, EngError> {
        product(&[self.rows, width, F32])
    }

    fn xf(&self, k: usize) -> Result<usize, EngError> {
        xf_bytes(self.rows, k)
            .ok_or_else(|| EngError::MemoryPlan("step plan: an xf size overflows".to_owned()))
    }

    /// The padded `block_q8_1` bytes of `ncols` rows of `k` values: the Native step's input packs.
    /// `ncols` is 1 for every activation row and `top_k` for the expert slots.
    fn q8(&self, k: usize, ncols: usize) -> Result<usize, EngError> {
        q8_1_bytes(k, ncols)
            .ok_or_else(|| EngError::MemoryPlan("step plan: a q8 pack size overflows".to_owned()))
    }
}

/// The typed refusals of the NATIVE plan and of the Int8G32 KV mode, before any slot is sized: the
/// q8_1 producers take the step's single row, the Native MoE runs ONE merged down launch, so the
/// forked per-expert packs have no Native form, and the Int8G32 attention ops exist as Native op
/// lists only (Tiled stays the fp16 control path). Stage-1 unreachable through `StepResources` (one
/// row, `check_stage1_options`); the plan refuses them on its own so no caller can build a
/// half-tiled plan.
fn check_layout_rows(rows: usize, options: &StepOptions) -> Result<(), EngError> {
    let single_row = matches!(options.layout, WeightLayout::Native | WeightLayout::Exl3);
    // The GGUF Native refusals keep their original wording.
    let name = match options.layout {
        WeightLayout::Exl3 => "EXL3",
        _ => "native",
    };
    if single_row && rows != 1 {
        return Err(EngError::BadRequest(format!(
            "step plan: the {name} layout takes one row, got {rows}"
        )));
    }
    if single_row && options.forked_experts {
        return Err(EngError::BadRequest(format!(
            "step plan: the {name} layout takes the single merged expert down pack"
        )));
    }
    if options.kv_mode == KvMode::Int8G32 && options.layout == WeightLayout::Tiled {
        return Err(EngError::BadRequest(
            "step plan: the Int8G32 KV mode takes the native layout; Tiled is the fp16 control \
             path"
                .to_owned(),
        ));
    }
    Ok(())
}

fn list_slots(
    dims: &StepDims,
    rows: usize,
    options: &StepOptions,
    scratch: &KernelScratch,
) -> Result<Vec<PlannedSlot>, EngError> {
    let mut list = SlotList {
        rows,
        slots: Vec::new(),
    };
    list_mixers(&mut list, dims, options, scratch)?;
    list_moe(&mut list, dims, options, scratch)?;
    let h = dims.shape.hidden;
    match options.layout {
        WeightLayout::Tiled => list.add(Slot::XfFinal, list.xf(h), FINAL_NORM, LM_HEAD)?,
        WeightLayout::Native => {
            list.add(Slot::Q8Final, list.q8(h, 1), FINAL_NORM, LM_HEAD)?;
            // The fused producer must also write an f32 row beside the pack; the native plan gives
            // the final norm its OWN scratch row - sound in isolation (NormRow's native lifetime
            // covers FINAL_NORM and live slots never share bytes), kept so the final norm depends
            // on no other op's slot.
            list.add(Slot::FinalNormRow, list.row(h), FINAL_NORM, FINAL_NORM)?;
        }
        WeightLayout::Exl3 => {
            // The fused norm still emits its tiled pack, which the EXL3 head never reads; the
            // head reads the F32 row the producer writes beside it.
            list.add(Slot::XfFinal, list.xf(h), FINAL_NORM, FINAL_NORM)?;
            list.add(Slot::FinalNormRow, list.row(h), FINAL_NORM, LM_HEAD)?;
            list_exl3_scratch(&mut list, dims)?;
        }
    }
    list.add(Slot::Logits, list.row(dims.shape.vocab), LM_HEAD, ARGMAX)?;
    list.add(Slot::ArgmaxPart, Ok(scratch.argmax_part), ARGMAX, ARGMAX)?;
    Ok(list.slots)
}

fn list_mixers(
    list: &mut SlotList,
    dims: &StepDims,
    options: &StepOptions,
    scratch: &KernelScratch,
) -> Result<(), EngError> {
    let (h, d, a) = (dims.shape.hidden, &dims.shape.delta, &dims.shape.attn);
    let (value_width, q_width) = (d.value_heads * d.value_dim, a.heads_q * a.head_dim);
    list.add(Slot::ResidA, list.row(h), EMBED, ARGMAX)?;
    list.add(Slot::ResidB, list.row(h), EMBED, ARGMAX)?;
    match options.layout {
        WeightLayout::Tiled => list.add(Slot::XfMix, list.xf(h), MIX_PACK, ATTN_IN)?,
        WeightLayout::Native => list.add(Slot::Q8Mix, list.q8(h, 1), MIX_PACK, ATTN_IN)?,
        WeightLayout::Exl3 => list.add(Slot::XfMix, list.xf(h), MIX_PACK, MIX_PACK)?,
    }
    list.add(
        Slot::DeltaProj,
        list.row(dims.delta_proj),
        DELTA_IN,
        DELTA_RULE,
    )?;
    list.add(
        Slot::DeltaConv,
        list.row(d.conv_dim()),
        DELTA_CONV,
        DELTA_RULE,
    )?;
    list.add(
        Slot::DeltaCore,
        list.row(value_width),
        DELTA_RULE,
        if options.layout == WeightLayout::Exl3 {
            DELTA_OUT
        } else {
            DELTA_PACK
        },
    )?;
    match options.layout {
        WeightLayout::Tiled => list.add(
            Slot::XfDeltaOut,
            list.xf(value_width),
            DELTA_PACK,
            DELTA_OUT,
        )?,
        WeightLayout::Native => list.add(
            Slot::Q8DeltaOut,
            list.q8(value_width, 1),
            DELTA_PACK,
            DELTA_OUT,
        )?,
        WeightLayout::Exl3 => {}
    }
    list.add(Slot::AttnProj, list.row(dims.attn_proj), ATTN_IN, ATTN_CORE)?;
    list.add(Slot::AttnQ, list.row(q_width), ATTN_PREP, ATTN_CORE)?;
    // The f32 rows handoff of the Int8G32 KV mode: written by the rows entry of
    // `attn_prep` at ATTN_PREP, consumed by `kv8_append` at KV8_APPEND, dead after it.
    if options.kv_mode == KvMode::Int8G32 {
        let kv_width = a.heads_kv * a.head_dim;
        list.add(Slot::KRows, list.row(kv_width), ATTN_PREP, KV8_APPEND)?;
        list.add(Slot::VRows, list.row(kv_width), ATTN_PREP, KV8_APPEND)?;
    }
    list.add(Slot::AttnPart, Ok(scratch.attn_part), ATTN_CORE, ATTN_CORE)?;
    list.add(
        Slot::AttnCore,
        list.row(q_width),
        ATTN_CORE,
        if options.layout == WeightLayout::Exl3 {
            ATTN_OUT
        } else {
            ATTN_PACK
        },
    )?;
    match options.layout {
        WeightLayout::Tiled => list.add(Slot::XfAttnOut, list.xf(q_width), ATTN_PACK, ATTN_OUT)?,
        WeightLayout::Native => {
            list.add(Slot::Q8AttnOut, list.q8(q_width, 1), ATTN_PACK, ATTN_OUT)?
        }
        WeightLayout::Exl3 => {}
    }
    list.add(Slot::MixOut, list.row(h), DELTA_OUT, ADD_NORM)
}

/// The MoE block: router, routed experts, shared expert.
///
/// The launch order is the MoE block's contract (phase order, one stream): (1) the router; (2) all
/// expert gate/up matmuls (every slot, gate and up) and the shared gate/up; (3) the shared swiglu;
/// (4) for each slot s, `swiglu_pack(s)` into the one `XfDown`, then `down(s)`; (5) the shared
/// down; (6) the combine. So `XfMoe` is last read in `GateUp`; `ExpertGu` is read by every slot's
/// swiglu inside the `Down` phase and `SharedGu` is kept to `Down` with it; `XfDown` is rewritten
/// per slot (stream order makes swiglu(s), down(s), swiglu(s + 1) safe) and covers the whole `Down`
/// phase.
fn list_moe(
    list: &mut SlotList,
    dims: &StepDims,
    options: &StepOptions,
    scratch: &KernelScratch,
) -> Result<(), EngError> {
    let (h, e) = (dims.shape.hidden, &dims.shape.moe);
    let m = list.rows;
    match options.layout {
        WeightLayout::Tiled => list.add(Slot::XfMoe, list.xf(h), ADD_NORM, GATE_UP)?,
        WeightLayout::Native => list.add(Slot::Q8Moe, list.q8(h, 1), ADD_NORM, GATE_UP)?,
        // The fused norm still emits its tiled pack, which the EXL3 projections never read.
        WeightLayout::Exl3 => list.add(Slot::XfMoe, list.xf(h), ADD_NORM, ADD_NORM)?,
    }
    match options.layout {
        // The fused `add_rmsnorm_q8` always writes the f32 normalized row, so the NATIVE
        // mixer prologue writes `NormRow` as scratch beside its `Q8Mix` pack: the slot is live
        // from that first write to the end of the step (the router reads it at ROUTE). The
        // final norm does NOT touch it - it writes the plan's own FinalNormRow.
        WeightLayout::Tiled => list.add(Slot::NormRow, list.row(h), ADD_NORM, ROUTE)?,
        WeightLayout::Native => list.add(Slot::NormRow, list.row(h), MIX_PACK, ARGMAX)?,
        // The mixer prologue writes the row as the EXL3 projections' input, the epilogue rewrites
        // it for the router and for every expert's gate and up projection.
        // The shared expert's gate and up read the row after the routed experts' down launch, so
        // the row stays live to the combine.
        WeightLayout::Exl3 => list.add(Slot::NormRow, list.row(h), MIX_PACK, COMBINE)?,
    }
    list.add(Slot::RouterLogits, Ok(scratch.router_logits), ROUTE, ROUTE)?;
    list.add(Slot::TopkId, product(&[m, e.top_k, I32]), ROUTE, COMBINE)?;
    list.add(Slot::TopkWeight, list.row(e.top_k), ROUTE, COMBINE)?;
    list.add(Slot::SharedGate, list.row(1), ROUTE, COMBINE)?;
    let gate_up = product(&[m, e.top_k, 2 * e.inter, F32]);
    // The EXL3 block runs the routed experts and then the shared expert as one op, in stream
    // order: every intermediate it touches is live from the first launch to the combine, so no
    // later launch of the block writes over an earlier one's result.
    let (gu_last, block_first) = if options.layout == WeightLayout::Exl3 {
        (COMBINE, GATE_UP)
    } else {
        (DOWN, SWIGLU)
    };
    list.add(Slot::ExpertGu, gate_up, GATE_UP, gu_last)?;
    match options.layout {
        WeightLayout::Tiled => {
            let branches = if options.forked_experts { e.top_k } else { 1 };
            for branch in 0..branches {
                list.add(Slot::XfDown(branch), list.xf(e.inter), SWIGLU, DOWN)?;
            }
        }
        // One merged pack of `top_k` padded q8_1 rows feeds the ONE down launch of the slots; the
        // forked per-expert packs have no Native form (`check_layout_rows`).
        WeightLayout::Native => list.add(Slot::Q8Down, list.q8(e.inter, e.top_k), SWIGLU, DOWN)?,
        // One F32 row per pick feeds the grouped down launch.
        WeightLayout::Exl3 => list.add(
            Slot::ExpertAct,
            product(&[m, e.top_k, e.inter, F32]),
            block_first,
            COMBINE,
        )?,
    }
    let down = product(&[m, e.top_k, h, F32]);
    let down_first = if options.layout == WeightLayout::Exl3 {
        GATE_UP
    } else {
        DOWN
    };
    list.add(Slot::ExpertDown, down, down_first, COMBINE)?;
    let shared_last = if options.layout == WeightLayout::Exl3 {
        COMBINE
    } else {
        DOWN
    };
    list.add(
        Slot::SharedGu,
        list.row(2 * e.shared_inter),
        GATE_UP,
        shared_last,
    )?;
    match options.layout {
        WeightLayout::Tiled => list.add(Slot::XfShared, list.xf(e.shared_inter), SWIGLU, DOWN)?,
        WeightLayout::Native => {
            list.add(Slot::Q8Shared, list.q8(e.shared_inter, 1), SWIGLU, DOWN)?
        }
        WeightLayout::Exl3 => list.add(
            Slot::SharedAct,
            list.row(e.shared_inter),
            block_first,
            COMBINE,
        )?,
    }
    list.add(Slot::SharedOut, list.row(h), down_first, COMBINE)
}

/// The card variant of [`list_mixers`] for the Native multi-device step: every width is the
/// card's own (`MixerGeometry::for_card`) except the residual rows and the gathered K/V rows,
/// which stay at the full shape's width (`Slot::KRows`'s card contract).
fn list_card_mixers(
    list: &mut SlotList,
    dims: &StepDims,
    geometry: &MixerGeometry,
    scratch: &KernelScratch,
) -> Result<(), EngError> {
    let (d, a) = (&geometry.delta, &geometry.attn);
    let h = dims.shape.hidden;
    // The gathered rows are the whole cache row, even on a card that holds fewer KV heads:
    // GatherK/GatherV write the full row and the append consumes it at that width.
    let kv_width = dims.shape.attn.heads_kv * dims.shape.attn.head_dim;
    list.add(Slot::ResidA, list.row(h), EMBED, ARGMAX)?;
    list.add(Slot::ResidB, list.row(h), EMBED, ARGMAX)?;
    list.add(Slot::Q8Mix, list.q8(h, 1), MIX_PACK, ATTN_IN)?;
    list.add(Slot::DeltaProj, list.row(d.ld), DELTA_IN, DELTA_RULE)?;
    list.add(
        Slot::DeltaConv,
        list.row(d.channels),
        DELTA_CONV,
        DELTA_RULE,
    )?;
    list.add(
        Slot::DeltaCore,
        list.row(d.value_width),
        DELTA_RULE,
        DELTA_PACK,
    )?;
    list.add(
        Slot::Q8DeltaOut,
        list.q8(d.value_width, 1),
        DELTA_PACK,
        DELTA_OUT,
    )?;
    list.add(Slot::AttnProj, list.row(a.ld), ATTN_IN, ATTN_CORE)?;
    list.add(Slot::AttnQ, list.row(a.q_width), ATTN_PREP, ATTN_CORE)?;
    // BOTH KV modes: the Int8G32 handoff and the Fp16 card path both gather at full width.
    list.add(Slot::KRows, list.row(kv_width), ATTN_PREP, KV8_APPEND)?;
    list.add(Slot::VRows, list.row(kv_width), ATTN_PREP, KV8_APPEND)?;
    list.add(Slot::AttnPart, Ok(scratch.attn_part), ATTN_CORE, ATTN_CORE)?;
    list.add(Slot::AttnCore, list.row(a.q_width), ATTN_CORE, ATTN_PACK)?;
    list.add(Slot::Q8AttnOut, list.q8(a.q_width, 1), ATTN_PACK, ATTN_OUT)?;
    list.add(Slot::MixOut, list.row(h), DELTA_OUT, ADD_NORM)
}

/// The card variant of [`list_moe`] for the Native multi-device step: the router and top-k
/// tensors stay full (the router runs once, its picks broadcast), the inter-width packs shrink
/// to the card's expert rows, and the down/shared outputs stay full hidden width (each card's
/// partial is summed by the ring exchange).
fn list_card_moe(
    list: &mut SlotList,
    dims: &StepDims,
    table: &ShardTable,
    card: usize,
    scratch: &KernelScratch,
) -> Result<(), EngError> {
    let (h, e) = (dims.shape.hidden, &dims.shape.moe);
    let m = list.rows;
    let inter = table.expert_inter(card)?.len();
    let shared_inter = table.shared_inter(card)?.len();
    list.add(Slot::Q8Moe, list.q8(h, 1), ADD_NORM, GATE_UP)?;
    // The NATIVE NormRow lifetime: written by the mixer prologue at MIX_PACK, last read by the
    // argmax over the head's own shard.
    list.add(Slot::NormRow, list.row(h), MIX_PACK, ARGMAX)?;
    list.add(Slot::RouterLogits, Ok(scratch.router_logits), ROUTE, ROUTE)?;
    list.add(Slot::TopkId, product(&[m, e.top_k, I32]), ROUTE, COMBINE)?;
    list.add(Slot::TopkWeight, list.row(e.top_k), ROUTE, COMBINE)?;
    list.add(Slot::SharedGate, list.row(1), ROUTE, COMBINE)?;
    let gate_up = product(&[m, e.top_k, 2 * inter, F32]);
    list.add(Slot::ExpertGu, gate_up, GATE_UP, DOWN)?;
    list.add(Slot::Q8Down, list.q8(inter, e.top_k), SWIGLU, DOWN)?;
    let down = product(&[m, e.top_k, h, F32]);
    list.add(Slot::ExpertDown, down, DOWN, COMBINE)?;
    list.add(Slot::SharedGu, list.row(2 * shared_inter), GATE_UP, DOWN)?;
    list.add(Slot::Q8Shared, list.q8(shared_inter, 1), SWIGLU, DOWN)?;
    list.add(Slot::SharedOut, list.row(h), DOWN, COMBINE)
}
/// the rotated inputs, the split-K partial and the sticky arithmetic flag. The sizes are the
/// largest any projection of the step needs: mixers, grouped expert launches, the head.
fn list_exl3_scratch(list: &mut SlotList, dims: &StepDims) -> Result<(), EngError> {
    let geometry = super::mixer_geometry::MixerGeometry::new(dims)?;
    let (h, e) = (dims.shape.hidden, &dims.shape.moe);
    // (rows of the launch, input width, output width) of every projection kind.
    let mut launches = vec![
        (1, h, dims.shape.vocab),
        (2 * e.top_k, h, e.inter),
        (e.top_k, e.inter, h),
        (2, h, e.shared_inter),
        (1, e.shared_inter, h),
    ];
    for kind in [
        crate::family::AttentionKind::DeltaNet,
        crate::family::AttentionKind::Full,
    ] {
        for spec in geometry.dense_specs(WeightLayout::Exl3, kind)? {
            launches.push((1, spec.k, spec.rows));
        }
    }
    let xh = launches
        .iter()
        .map(|&(rows, k, _)| rows * k)
        .max()
        .unwrap_or(0);
    let z = launches
        .iter()
        .map(|&(rows, _, n)| rows * n)
        .max()
        .unwrap_or(0);
    list.add(Slot::Exl3Xh, product(&[xh, F32]), MIX_PACK, LM_HEAD)?;
    list.add(Slot::Exl3Z, product(&[z, F32]), MIX_PACK, LM_HEAD)?;
    // The step clears this word at StepBegin, before its declared birth at MIX_PACK, and folds it
    // into the status just before the argmax. That is safe because the plan has no slot whose
    // lifetime ends before MIX_PACK that could share these bytes; the old host write at the start
    // of the token had the same property. A slot with such a lifetime would need the clear moved
    // to the first op that uses the flag.
    list.add(Slot::Exl3Flag, Ok(4), MIX_PACK, ARGMAX)
}

/// The card variant of [`list_mixers`] for the Native multi-device step: every width is the
/// card's own (`MixerGeometry::for_card`) except the residual rows and the gathered K/V rows,
/// which stay at the full shape's width (`Slot::KRows`'s card contract).
fn list_card_mixers(
    list: &mut SlotList,
    dims: &StepDims,
    geometry: &MixerGeometry,
    scratch: &KernelScratch,
) -> Result<(), EngError> {
    let (d, a) = (&geometry.delta, &geometry.attn);
    let h = dims.shape.hidden;
    // The gathered rows are the whole cache row, even on a card that holds fewer KV heads:
    // GatherK/GatherV write the full row and the append consumes it at that width.
    let kv_width = dims.shape.attn.heads_kv * dims.shape.attn.head_dim;
    list.add(Slot::ResidA, list.row(h), EMBED, ARGMAX)?;
    list.add(Slot::ResidB, list.row(h), EMBED, ARGMAX)?;
    list.add(Slot::Q8Mix, list.q8(h, 1), MIX_PACK, ATTN_IN)?;
    list.add(Slot::DeltaProj, list.row(d.ld), DELTA_IN, DELTA_RULE)?;
    list.add(
        Slot::DeltaConv,
        list.row(d.channels),
        DELTA_CONV,
        DELTA_RULE,
    )?;
    list.add(
        Slot::DeltaCore,
        list.row(d.value_width),
        DELTA_RULE,
        DELTA_PACK,
    )?;
    list.add(
        Slot::Q8DeltaOut,
        list.q8(d.value_width, 1),
        DELTA_PACK,
        DELTA_OUT,
    )?;
    list.add(Slot::AttnProj, list.row(a.ld), ATTN_IN, ATTN_CORE)?;
    list.add(Slot::AttnQ, list.row(a.q_width), ATTN_PREP, ATTN_CORE)?;
    // BOTH KV modes: the Int8G32 handoff and the Fp16 card path both gather at full width.
    list.add(Slot::KRows, list.row(kv_width), ATTN_PREP, KV8_APPEND)?;
    list.add(Slot::VRows, list.row(kv_width), ATTN_PREP, KV8_APPEND)?;
    list.add(Slot::AttnPart, Ok(scratch.attn_part), ATTN_CORE, ATTN_CORE)?;
    list.add(Slot::AttnCore, list.row(a.q_width), ATTN_CORE, ATTN_PACK)?;
    list.add(Slot::Q8AttnOut, list.q8(a.q_width, 1), ATTN_PACK, ATTN_OUT)?;
    list.add(Slot::MixOut, list.row(h), DELTA_OUT, ADD_NORM)
}

/// The card variant of [`list_moe`] for the Native multi-device step: the router and top-k
/// tensors stay full (the router runs once, its picks broadcast), the inter-width packs shrink
/// to the card's expert rows, and the down/shared outputs stay full hidden width (each card's
/// partial is summed by the ring exchange).
fn list_card_moe(
    list: &mut SlotList,
    dims: &StepDims,
    table: &ShardTable,
    card: usize,
    scratch: &KernelScratch,
) -> Result<(), EngError> {
    let (h, e) = (dims.shape.hidden, &dims.shape.moe);
    let m = list.rows;
    let inter = table.expert_inter(card)?.len();
    let shared_inter = table.shared_inter(card)?.len();
    list.add(Slot::Q8Moe, list.q8(h, 1), ADD_NORM, GATE_UP)?;
    // The NATIVE NormRow lifetime: written by the mixer prologue at MIX_PACK, last read by the
    // argmax over the head's own shard.
    list.add(Slot::NormRow, list.row(h), MIX_PACK, ARGMAX)?;
    list.add(Slot::RouterLogits, Ok(scratch.router_logits), ROUTE, ROUTE)?;
    list.add(Slot::TopkId, product(&[m, e.top_k, I32]), ROUTE, COMBINE)?;
    list.add(Slot::TopkWeight, list.row(e.top_k), ROUTE, COMBINE)?;
    list.add(Slot::SharedGate, list.row(1), ROUTE, COMBINE)?;
    let gate_up = product(&[m, e.top_k, 2 * inter, F32]);
    list.add(Slot::ExpertGu, gate_up, GATE_UP, DOWN)?;
    list.add(Slot::Q8Down, list.q8(inter, e.top_k), SWIGLU, DOWN)?;
    let down = product(&[m, e.top_k, h, F32]);
    list.add(Slot::ExpertDown, down, DOWN, COMBINE)?;
    list.add(Slot::SharedGu, list.row(2 * shared_inter), GATE_UP, DOWN)?;
    list.add(Slot::Q8Shared, list.q8(shared_inter, 1), SWIGLU, DOWN)?;
    list.add(Slot::SharedOut, list.row(h), DOWN, COMBINE)
}
