//! The pure plan of chunked prefill: the two-term chunk policy and the chunk list, the admission
//! rule, and the arena tensor list of one chunk.
//!
//! Host arithmetic only: no CUDA call. It sits in `family::qwen::qwen36::cuda`, so it compiles with
//! that module (`native-cuda` on Linux) and not in the default feature set; it calls the
//! `kernels::cuda` sizing functions, and `kernels::cuda` never calls back. Every tensor size is
//! the output of the function that owns the formula (the sister wrappers, `mmq_size`, the
//! router and norm wrappers, `mixer_geometry`); the offsets come from `plan_arena`. Lifetimes
//! are phases of one layer, as in `step_plan`: a layer runs the DeltaNet block (`D*`) or the
//! attention block (`A*`), so their tensors may alias.
//!
//! Admission charges the computed `plan_arena` total plus a flat `part` (the design's rows are
//! inputs of the reproduction tests only). The attention `part` buffer is its own allocation
//! outside the arena; its real size is reported in [`ChunkCost`] beside the flat 64 MiB charge,
//! which this plan keeps until a decision on the difference.

use super::dims::{KvMode, StepDims, StepOptions};
use super::prefill::{COLD_CHANNELS_MAX, tiered_bounds_ints};
use crate::family::decode_step::mixer_geometry::MixerGeometry;
use crate::family::decode_step::state_layout::{KvPool, StateLayout};
use crate::{
    EngError,
    family::project::cuda::WeightLayout,
    kernels::cuda::{
        ArenaTensor, plan_arena,
        prefill::{
            attn_prefill::{CPF_ATTN_PART_CAP, CPF_ATTN_Q_TILE, cpf_attn_workspace_bytes_for_sm},
            gdn_scan_seq::{cpf_dn_conv_out_bytes, cpf_dn_scan_out_bytes},
            mmq_size::{
                check_ncols_domain, expert_bounds_ints, fixup_floats_max, ids_dst_ints,
                ids_src1_ints, q8_1_bytes,
            },
        },
        resident::{
            add_norm_pack::row_bytes,
            attn_prep::q_bytes,
            router::{RouterShape, router_logits_bytes},
        },
        xf_bytes,
    },
};

/// The chunk sizes a session arena may be built for, largest first. The largest is also the default
/// of the `qwen36.prefill_chunk` cap: absent means no cap.
pub(crate) const M_LADDER: [usize; 5] = [2048, 1024, 512, 256, 128];
/// Only offered by the default-off tiered experiment; the normal ladder stays unchanged.
pub(super) const LARGE_M_LADDER: [usize; 6] = [4096, 2048, 1024, 512, 256, 128];
/// A chunk size is admitted only if at least this many device bytes stay spare.
/// Canonical value in the shared `admission_sketch`, shared with the preflight estimate.
pub(crate) const SPARE_MIN: usize = crate::family::admission_sketch::PREFILL_SPARE_MIN_BYTES;
/// The flat device charge for the attention `part` buffer in every admission row.
pub(crate) const PART_CHARGE: usize = CPF_ATTN_PART_CAP;
/// The sketch copy of the part charge must equal the kernel cap it mirrors;
/// a retuned `CPF_ATTN_PART_CAP` that leaves the sketch behind fails the build.
const _: () = assert!(PART_CHARGE == crate::family::admission_sketch::PREFILL_PART_CHARGE_BYTES);

fn overflow() -> EngError {
    EngError::MemoryPlan("prefill plan: a size overflows".to_owned())
}

/// Bytes of a table of 16-bit elements, `parts` multiplied.
fn half_bytes(parts: &[usize]) -> Result<usize, EngError> {
    parts
        .iter()
        .try_fold(2usize, |total, &part| total.checked_mul(part))
        .ok_or_else(overflow)
}

/// Bytes of `parts` 4-byte words multiplied together (f32 and i32 tensors).
fn word_bytes(parts: &[usize]) -> Result<usize, EngError> {
    parts
        .iter()
        .try_fold(4usize, |total, &part| total.checked_mul(part))
        .ok_or_else(overflow)
}

// The admission types moved to the shared prefill core (`family::prefill`); these re-exports
// keep the old paths alive for the driver, the step and the plan tests.
pub(crate) use crate::family::prefill::admission::{Admission, ChunkCost, Extra};

impl Admission {
    /// The Qwen3.6 rule: the ladder, the flat part charge and the spare minimum bind here, so
    /// every Qwen3.6 call site (and the plan tests' literals) stays a one-argument
    /// constructor.
    pub(crate) fn qwen36(device_budget: usize) -> Self {
        Self::new(device_budget, PART_CHARGE, SPARE_MIN, &M_LADDER)
    }
}

/// KV bytes of a session of `window` positions: the pool's exact bytes per position, without
/// `KvPool`'s per-plane `STATE_ALIGN` rounding (at most 255 B a plane, 40 planes at `Int8G32` —
/// about 10 KiB; zero at `Fp16` on this model, whose planes are whole multiples of the alignment).
/// The admission state term is the whole layout ([`state_bytes`]); this exact figure is the input
/// of the design-reproduction tests.
pub(crate) fn kv_bytes(dims: &StepDims, mode: KvMode, window: usize) -> Result<usize, EngError> {
    KvPool::bytes_per_token(dims, mode)?
        .checked_mul(window)
        .ok_or_else(overflow)
}

/// The state term of the admission sum: [`StateLayout::bytes`] of a session of `window` positions
/// and `delta_slots` DeltaNet slots — the pool, the KV and the persistent words — because
/// `ResidentState::allocate` allocates all three together after the budget snapshot. The KV mode
/// flows through the layout, so `Fp16` and `Int8G32` keep their own figures; the decode arena, the
/// workspace and a prefill arena are separate terms (the [`StateLayout::bytes`] doc). The options
/// fields no pool reads (`forked_experts`, `tiered`, `layout`) sit at the stage-1 defaults.
pub(crate) fn state_bytes(
    dims: &StepDims,
    mode: KvMode,
    window: usize,
    delta_slots: usize,
) -> Result<usize, EngError> {
    let options = StepOptions {
        kv_mode: mode,
        context_tokens: window,
        delta_slots,
        forked_experts: false,
        tiered: None,
        layout: WeightLayout::Tiled,
    };
    StateLayout::new(dims, &options)?.bytes()
}

/// Phases of one layer of a chunk, in launch order (the `step_plan` style): the lifetime
/// numbers of the tensors. `D*` is the DeltaNet block, `A*` the attention block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Phase {
    Embed,
    NormA,
    QuantA,
    DProj,
    DConv,
    DScan,
    DQuant,
    DOut,
    AProj,
    APrep,
    AFlash,
    AQuant,
    AOut,
    NormF,
    Route,
    Shared,
    Ids,
    Scatter,
    GateUp,
    Swiglu,
    QuantAct,
    Down,
    Combine,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ChunkSlot {
    Hidden,
    /// The card plans' permanent zero row (`m * hidden` f32): the arena is
    /// zero-filled at build and nothing ever writes this slot, so card 1's
    /// MoE combine reads it as the residual of its `CombineOut::To`
    /// partial - the exchange then sums card 0's residual-included output
    /// with card 1's partial alone.
    Zero,
    IdsTok,
    X,
    X2,
    XfA,
    XfF,
    Q8X,
    Q8X2,
    ProjD,
    ConvOut,
    OOut,
    Q8OD,
    Q8OA,
    ProjA,
    Q,
    AttnOut,
    Delta,
    Logits,
    TopkId,
    TopkWeight,
    SharedGate,
    GuSh,
    ActSh,
    Q8Ash,
    ShOut,
    IdsG,
    IdsSrc1,
    IdsDst,
    ExpertBounds,
    Q8S,
    Gu,
    Act,
    Q8Act,
    Down,
    Fixup,
}

type Row = (ChunkSlot, usize, Phase, Phase);

/// The tensors that live across blocks: the residual stream, the norm rows and their `xf`
/// sinks (written, never read in prefill), the first quantize and the stream-k fixup range.
fn common_rows(g: &MixerGeometry, m: usize, nsm: usize, card: bool) -> Result<Vec<Row>, EngError> {
    use {ChunkSlot as S, Phase::*};
    let hidden = row_bytes(m, g.hidden).ok_or_else(overflow)?;
    // `xf_bytes` is not monotone in m: a chunk of at most 16 rows packs the whole fixed fragment
    // (65,537 * 16 B, `native.rs`), so each sink planned at `m` rows also carries the 1-row figure,
    // or a smaller chunk of the session would not fit its slot (the prefix promise of
    // `chunk_tensors`).
    let xf = xf_bytes(m, g.hidden)
        .zip(xf_bytes(1, g.hidden))
        .map(|(bytes, fragment)| bytes.max(fragment))
        .ok_or_else(overflow)?;
    Ok(vec![
        (S::Hidden, hidden, Embed, Combine),
        (S::IdsTok, word_bytes(&[m])?, Embed, Embed),
        // X stays live through the projections: the dequantizing GEMM (the in-house
        // dense types) reads the f32 activation rows directly at DProj/AProj, so its
        // output slots (ProjD, ProjA) must never share X's arena span — which the
        // mmq protocol (X dead after the QuantA quantize) otherwise allows.
        (S::X, hidden, NormA, AProj),
        (S::XfA, xf, NormA, NormA),
        (S::Q8X, q8_1_bytes(g.hidden, m)?, QuantA, AProj),
        // A non-primary card reuses Delta for the output of the MoE combine;
        // its routed and shared inputs must remain disjoint from that output.
        // The single-device path combines into Hidden and keeps its shorter lifetime.
        (S::Delta, hidden, DOut, if card { Combine } else { NormF }),
        // X2 stays live through the routed gate and up launches: a stack on the dequantizing
        // GEMM reads these f32 rows directly (the mmq protocol lets it die after the scatter).
        (S::X2, hidden, NormF, GateUp),
        (S::XfF, xf, NormF, NormF),
        (
            S::Fixup,
            word_bytes(&[fixup_floats_max(nsm)?])?,
            Embed,
            Combine,
        ),
    ])
}

/// The two mixer blocks, on disjoint phase ranges.
fn mixer_rows(g: &MixerGeometry, m: usize) -> Result<Vec<Row>, EngError> {
    use {ChunkSlot as S, Phase::*};
    let (d, a) = (&g.delta, &g.attn);
    Ok(vec![
        (S::ProjD, word_bytes(&[m, d.ld])?, DProj, DScan),
        (
            S::ConvOut,
            cpf_dn_conv_out_bytes(m, d.channels)?,
            DConv,
            DScan,
        ),
        (
            S::OOut,
            cpf_dn_scan_out_bytes(m, d.value_heads, d.value_dim)?,
            DScan,
            DOut,
        ),
        (S::Q8OD, q8_1_bytes(d.value_width, m)?, DQuant, DOut),
        (S::ProjA, word_bytes(&[m, a.ld])?, AProj, AFlash),
        (S::Q, q_bytes(m, a.heads_q), APrep, AFlash),
        (S::AttnOut, word_bytes(&[m, a.q_width])?, AFlash, AOut),
        (S::Q8OA, q8_1_bytes(a.q_width, m)?, AQuant, AOut),
    ])
}

/// The Resident MoE block: router, shared expert, ids, scatter quantize, gate/up, down.
/// `routed` is the grouped column count `M * top_k`.
fn moe_rows_with(
    dims: &StepDims,
    e: &crate::family::qwen::qwen36::host::MoeDims,
    m: usize,
) -> Result<Vec<Row>, EngError> {
    use {ChunkSlot as S, Phase::*};
    let h = dims.shape.hidden;
    let routed = m.checked_mul(e.top_k).ok_or_else(overflow)?;
    let picks = word_bytes(&[m, e.top_k])?;
    let router = RouterShape {
        m,
        hidden: h,
        n_expert: e.experts,
        top_k: e.top_k,
    };
    Ok(vec![
        (S::Logits, router_logits_bytes(router)?, Route, Route),
        (S::TopkId, picks, Route, Combine),
        (S::TopkWeight, picks, Route, Combine),
        (S::SharedGate, word_bytes(&[m])?, Route, Combine),
        (S::Q8X2, q8_1_bytes(h, m)?, Shared, Shared),
        (
            S::GuSh,
            word_bytes(&[m, 2 * e.shared_inter])?,
            Shared,
            Shared,
        ),
        (S::ActSh, word_bytes(&[m, e.shared_inter])?, Shared, Shared),
        (S::Q8Ash, q8_1_bytes(e.shared_inter, m)?, Shared, Shared),
        (S::ShOut, word_bytes(&[m, h])?, Shared, Combine),
        (S::IdsG, picks, Ids, Ids),
        (
            S::IdsSrc1,
            word_bytes(&[ids_src1_ints(routed)])?,
            Ids,
            Scatter,
        ),
        (
            S::IdsDst,
            word_bytes(&[ids_dst_ints(routed, m)?])?,
            Ids,
            Down,
        ),
        (
            S::ExpertBounds,
            // A tiered chunk layer groups its picks over the bank slots and then the cold
            // channels; the slot holds the most any plan asks, every expert on the card plus
            // the deepest staging.
            word_bytes(&[expert_bounds_ints(e.experts)?
                .max(tiered_bounds_ints(e.experts, COLD_CHANNELS_MAX))])?,
            Ids,
            Down,
        ),
        (S::Q8S, q8_1_bytes(h, routed)?, Scatter, GateUp),
        (S::Gu, word_bytes(&[routed, 2 * e.inter])?, GateUp, Swiglu),
        // Act stays live through the down launch: a stack on the dequantizing GEMM reads these f32
        // slot rows directly, so `Down` must not share its arena span.
        (S::Act, word_bytes(&[routed, e.inter])?, Swiglu, Down),
        (S::Q8Act, q8_1_bytes(e.inter, routed)?, QuantAct, Down),
        (S::Down, word_bytes(&[routed, h])?, Down, Combine),
    ])
}

/// The tensors of a chunk of `m` rows, in the order of the plan table, for a device with `nsm`
/// SMs (the runtime device fact that sizes the fixup range; never a literal here). `m` is in
/// `1..=4096` (the chunk cap of `mmq_size`). Smaller chunks of a session use prefixes of the
/// tensors planned for its `M_sess`: every slot grows with `m` except the `xf` sinks, whose
/// planned size carries the 1-row fragment floor because `xf_bytes` is not monotone in `m`.
pub(crate) fn chunk_tensors(
    dims: &StepDims,
    m: usize,
    nsm: usize,
) -> Result<Vec<(ChunkSlot, ArenaTensor)>, EngError> {
    chunk_tensors_inner(
        dims,
        &MixerGeometry::new(dims)?,
        &dims.shape.moe,
        m,
        nsm,
        false,
    )
}

/// The card twin: the same slot plan over the CARD's geometry and the
/// card's inter-dimensional MoE facts (its slices of the projections and
/// expert stacks), so each card's arena carries exactly its narrower
/// buffers.
pub(crate) fn chunk_tensors_for(
    dims: &StepDims,
    g: &MixerGeometry,
    moe: &crate::family::qwen::qwen36::host::MoeDims,
    m: usize,
    nsm: usize,
) -> Result<Vec<(ChunkSlot, ArenaTensor)>, EngError> {
    chunk_tensors_inner(dims, g, moe, m, nsm, true)
}

fn chunk_tensors_inner(
    dims: &StepDims,
    g: &MixerGeometry,
    moe: &crate::family::qwen::qwen36::host::MoeDims,
    m: usize,
    nsm: usize,
    card: bool,
) -> Result<Vec<(ChunkSlot, ArenaTensor)>, EngError> {
    check_ncols_domain(m)?;
    if nsm == 0 {
        return Err(EngError::BadRequest(format!(
            "prefill plan: nsm {nsm} is below 1; the fixup range needs the device SM count"
        )));
    }
    let mut rows: Vec<Row> = [
        common_rows(g, m, nsm, card)?,
        mixer_rows(g, m)?,
        moe_rows_with(dims, moe, m)?,
    ]
    .into_iter()
    .flatten()
    .collect();
    if card {
        let hidden = dims.shape.hidden;
        rows.push((
            ChunkSlot::Zero,
            m.checked_mul(hidden)
                .and_then(|v| v.checked_mul(4))
                .ok_or_else(overflow)?,
            Phase::Embed,
            Phase::Combine,
        ));
    }
    Ok(rows
        .into_iter()
        .map(|(slot, bytes, first, last)| {
            let (first, last) = (first as usize, last as usize);
            (slot, ArenaTensor { bytes, first, last })
        })
        .collect())
}

/// The cost of one chunk size: the `plan_arena` total of [`chunk_tensors`] and the real bytes of
/// the attention `part` buffer, whose split plan depends on `nsm` and not on the key count.
pub(crate) fn chunk_cost(dims: &StepDims, m: usize, nsm: usize) -> Result<ChunkCost, EngError> {
    let tensors: Vec<ArenaTensor> = chunk_tensors(dims, m, nsm)?
        .into_iter()
        .map(|(_, tensor)| tensor)
        .collect();
    let a = &dims.shape.attn;
    let heads = (a.heads_q, a.heads_kv);
    Ok(ChunkCost {
        m,
        arena: plan_arena(&tensors)?.total,
        part: cpf_attn_workspace_bytes_for_sm(m, heads, a.head_dim, m, CPF_ATTN_Q_TILE, nsm)?,
    })
}

/// The costs of every size of [`M_LADDER`]: the candidates [`Admission::admit`] chooses from.
pub(crate) fn chunk_costs(dims: &StepDims, nsm: usize) -> Result<Vec<ChunkCost>, EngError> {
    M_LADDER.iter().map(|&m| chunk_cost(dims, m, nsm)).collect()
}

/// Whether the prefill admission keeps a chunk arena at all for a start whose resident state is
/// `state` bytes: the smallest rung of the ladder passes the admission's own rule with `budget`
/// device bytes (free after the weight load, less the headroom), the decode arena, and the
/// device prefix tip. The card choice asks this before it sizes the context, so it cannot pick a
/// context whose state leaves the prefill one token per step.
pub(crate) fn smallest_chunk_passes(
    dims: &StepDims,
    nsm: usize,
    budget: usize,
    (decode_arena, tip): (usize, usize),
    state: usize,
) -> Result<bool, EngError> {
    let mut admission = Admission::qwen36(budget);
    admission.decode_arena = decode_arena;
    if tip > 0 {
        admission.device_extras.push(Extra {
            name: "device-prefix-delta",
            fixed: tip,
            bytes_per_token: 0,
        });
    }
    let smallest = chunk_costs(dims, nsm)?.pop().ok_or_else(overflow)?;
    admission.passes(state, &smallest)
}

#[cfg(test)]
mod card_tests {
    use super::*;
    use crate::family::gguf_model::Inventory;
    use crate::family::qwen::qwen36::host::HostShape;
    use crate::family::spec::QWEN3_6_35B_A3B;
    use crate::kernels::cuda::plan_arena;

    /// Each card's chunk arena carries its narrower slots: the plan over
    /// the card's geometry and inter facts is strictly smaller than the
    /// single-device plan, keeps every slot the stage list needs, and the
    /// two cards' plans together stay under one single-device arena plus
    /// the shared rows' slack.
    #[test]
    fn card_chunk_arenas_are_narrower_and_complete() {
        let inv = Inventory::for_spec(&QWEN3_6_35B_A3B).expect("inventory");
        let shape = HostShape::from_spec(
            &QWEN3_6_35B_A3B,
            crate::family::qwen::qwen36::cuda::pinned::ROTARY_DIM,
        )
        .expect("shape");
        let dims = super::super::dims::StepDims::new(&shape, &inv).expect("dims");
        let m = 512;
        let nsm = 28; // the slower card's SM count sizes the shared plans
        let full = chunk_tensors(&dims, m, nsm).expect("full plan");
        let full_total: usize = plan_arena(&full.iter().map(|(_, t)| *t).collect::<Vec<_>>())
            .expect("full arena")
            .total;
        let table =
            crate::family::decode_step::shard::ShardTable::new(&shape, 2, 32).expect("table");
        let mut card_totals = Vec::new();
        for card in 0..2 {
            let geometry = crate::family::decode_step::mixer_geometry::MixerGeometry::for_card(
                &dims, &table, card,
            )
            .expect("card geometry");
            let moe = crate::family::qwen::qwen36::host::MoeDims {
                inter: table.expert_inter(card).expect("inter").len(),
                shared_inter: table.shared_inter(card).expect("shared").len(),
                ..shape.moe
            };
            let plan = chunk_tensors_for(&dims, &geometry, &moe, m, nsm).expect("card plan");
            let total: usize = plan_arena(&plan.iter().map(|(_, t)| *t).collect::<Vec<_>>())
                .expect("card arena")
                .total;
            assert!(
                total < full_total,
                "card {card} arena {total} must be smaller than the full {full_total}"
            );
            let slots_full: Vec<_> = full.iter().map(|(s, _)| *s).collect();
            let mut slots_card: Vec<_> = plan.iter().map(|(s, _)| *s).collect();
            slots_card.sort_by_key(|s| format!("{s:?}"));
            let mut slots_full_sorted = slots_full.clone();
            assert!(!slots_full_sorted.contains(&ChunkSlot::Zero));
            slots_full_sorted.push(ChunkSlot::Zero);
            slots_full_sorted.sort_by_key(|s| format!("{s:?}"));
            assert_eq!(
                slots_full_sorted, slots_card,
                "card {card} adds only its permanent zero row"
            );
            card_totals.push(total);
        }
        let both: usize = card_totals.iter().sum();
        assert!(
            both < 2 * full_total,
            "the pair's arenas {both} stay under two full arenas"
        );
    }

    /// The non-primary card writes its partial MoE combine into Delta while all of the
    /// combine's inputs are still live. The arena must keep that output apart from them.
    #[test]
    fn card_combine_output_is_disjoint_from_its_inputs() {
        let inv = Inventory::for_spec(&QWEN3_6_35B_A3B).expect("inventory");
        let shape = HostShape::from_spec(
            &QWEN3_6_35B_A3B,
            crate::family::qwen::qwen36::cuda::pinned::ROTARY_DIM,
        )
        .expect("shape");
        let dims = super::super::dims::StepDims::new(&shape, &inv).expect("dims");
        let table =
            crate::family::decode_step::shard::ShardTable::new(&shape, 2, 32).expect("table");
        let single = chunk_tensors(&dims, 512, 28).expect("single-device plan");
        let single_delta = single
            .iter()
            .find(|(slot, _)| *slot == ChunkSlot::Delta)
            .expect("single-device Delta")
            .1;
        assert_eq!(single_delta.last, Phase::NormF as usize);
        for card in 0..2 {
            let geometry = crate::family::decode_step::mixer_geometry::MixerGeometry::for_card(
                &dims, &table, card,
            )
            .expect("card geometry");
            let moe = crate::family::qwen::qwen36::host::MoeDims {
                inter: table.expert_inter(card).expect("inter").len(),
                shared_inter: table.shared_inter(card).expect("shared").len(),
                ..shape.moe
            };
            for nsm in [28, 68] {
                for m in [1, 15, 16, 31, 32, 64, 96, 128, 256, 512] {
                    let tensors =
                        chunk_tensors_for(&dims, &geometry, &moe, m, nsm).expect("card plan");
                    let plan = plan_arena(&tensors.iter().map(|(_, t)| *t).collect::<Vec<_>>())
                        .expect("card arena");
                    let index = |slot| {
                        tensors
                            .iter()
                            .position(|(candidate, _)| *candidate == slot)
                            .expect("slot in card arena")
                    };
                    let out = index(ChunkSlot::Delta);
                    let output = tensors[out].1;
                    assert!(
                        (output.first..=output.last).contains(&(Phase::Combine as usize)),
                        "card {card}, m {m}, nsm {nsm}: combine output is not live"
                    );
                    for slot in [
                        ChunkSlot::Zero,
                        ChunkSlot::Down,
                        ChunkSlot::TopkWeight,
                        ChunkSlot::SharedGate,
                        ChunkSlot::ShOut,
                    ] {
                        let input = index(slot);
                        let apart = plan.offsets[out] + output.bytes <= plan.offsets[input]
                            || plan.offsets[input] + tensors[input].1.bytes <= plan.offsets[out];
                        assert!(
                            apart,
                            "card {card}, m {m}, nsm {nsm}: Delta overlaps {slot:?}"
                        );
                    }
                }
            }
        }
    }

    /// The incident: a UD-IQ4_XS default start chose fp16 at 243,712 positions with a budget of
    /// 4,864,344,064 bytes after the load, and the prefill kept one token per step. The figures
    /// the admission charged then go through the admission's own rule: the state alone exceeds the
    /// budget at 243,712, and the stepped-down 223,232 passes.
    #[test]
    fn the_incident_context_is_refused_and_the_stepped_down_one_passes() {
        let inv = Inventory::for_spec(&QWEN3_6_35B_A3B).expect("inventory");
        let shape = HostShape::from_spec(
            &QWEN3_6_35B_A3B,
            crate::family::qwen::qwen36::cuda::pinned::ROTARY_DIM,
        )
        .expect("shape");
        let dims = super::super::dims::StepDims::new(&shape, &inv).expect("dims");
        let state_at =
            |context: usize| state_bytes(&dims, KvMode::Fp16, context, 1).expect("state");
        assert_eq!(state_at(243_712), 5_057_086_208);
        assert_eq!(state_at(223_232), 4_637_655_808);
        let budget = 4_864_344_064;
        let tip = 65_863_680;
        let decode = 2_238_724;
        let passes = |context: usize| {
            smallest_chunk_passes(&dims, 82, budget, (decode, tip), state_at(context))
                .expect("the rule answers")
        };
        assert!(!passes(243_712), "the state alone exceeds the budget");
        assert!(passes(223_232), "the chosen context leaves the arena room");
        // The boundary is monotone: no longer context passes where a shorter one is refused.
        let mut refused = false;
        for context in (200_000..262_144).step_by(1024) {
            let ok = passes(context);
            assert!(!(refused && ok), "{context}");
            refused |= !ok;
        }
        assert!(refused);
    }

    /// The default start of every local Qwen3.6-architecture GGUF on a free 3090: the context the
    /// card fit picks without and with the prefill admission asked first, and the figures the
    /// admission charges at each. Reads headers only.
    #[test]
    #[ignore = "reads the real GGUF headers under $HOME/models and $HOME/.ollama"]
    fn the_default_fit_keeps_a_prefill_arena_for_every_local_qwen36_gguf() {
        use crate::family::qwen::qwen36::admission_sketch::{
            CudaFitAsk, choose_cuda_fit, cuda_kv_per_position,
        };
        use crate::family::qwen::qwen36::settings::KvDtype;
        use crate::weights::{GgufHeader, MetaValue};
        use std::{fs::File, io::BufReader, io::Read, path::PathBuf};
        let inv = Inventory::for_spec(&QWEN3_6_35B_A3B).expect("inventory");
        let shape = HostShape::from_spec(
            &QWEN3_6_35B_A3B,
            crate::family::qwen::qwen36::cuda::pinned::ROTARY_DIM,
        )
        .expect("shape");
        let dims = super::super::dims::StepDims::new(&shape, &inv).expect("dims");
        let tip = crate::family::qwen::qwen36::settings::delta_state_bytes(&QWEN3_6_35B_A3B)
            .expect("tip") as usize;
        let home = PathBuf::from(std::env::var("HOME").unwrap());
        let mut files: Vec<PathBuf> = Vec::new();
        for dir in [
            home.join("models/qwen3.6"),
            home.join("models/qwen3.6/iq"),
            home.join("models/qwen3.6/requant"),
            home.join(".ollama/models/blobs"),
        ] {
            let Ok(entries) = std::fs::read_dir(&dir) else {
                continue;
            };
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_file() && entry.metadata().unwrap().len() > (1 << 30) {
                    files.push(path);
                }
            }
        }
        files.sort();
        let floor = |kv: KvDtype, context: u32| {
            let cache = u64::from(context) * cuda_kv_per_position(&QWEN3_6_35B_A3B, kv).ok()?;
            Some(
                cache
                    + if kv == KvDtype::Q8 {
                        256 << 20
                    } else {
                        1 << 30
                    },
            )
        };
        for free in [22_820u64 << 20, 23_416 << 20, 24_000 << 20] {
            eprintln!("--- {} MiB free", free >> 20);
            for path in &files {
                let mut magic = [0u8; 4];
                File::open(path).unwrap().read_exact(&mut magic).unwrap();
                if &magic != b"GGUF" {
                    continue;
                }
                let file = File::open(path).unwrap();
                let Ok(header) = GgufHeader::parse(
                    &mut BufReader::new(&file),
                    file.metadata().unwrap().len(),
                    512 << 20,
                ) else {
                    continue;
                };
                if !matches!(header.meta("general.architecture"),
                    Some(MetaValue::String(arch)) if arch == "qwen35moe")
                {
                    continue;
                }
                let weights: u64 = header.tensors().iter().map(|t| t.byte_len()).sum();
                let budget = free
                    .saturating_sub(weights)
                    .saturating_sub(crate::family::mem::DEVICE.load_unplanned)
                    .saturating_sub(1 << 30) as usize;
                let admits = |kv: KvDtype, context: u32| -> Result<bool, EngError> {
                    let mode = if kv == KvDtype::Q8 {
                        KvMode::Int8G32
                    } else {
                        KvMode::Fp16
                    };
                    if mode != KvMode::Fp16 {
                        return Ok(true);
                    }
                    let options = StepOptions {
                        kv_mode: mode,
                        context_tokens: context as usize,
                        delta_slots: 1,
                        forked_experts: false,
                        tiered: None,
                        layout: WeightLayout::Native,
                    };
                    let state = StateLayout::new(&dims, &options)
                        .and_then(|layout| layout.bytes())
                        .expect("state");
                    smallest_chunk_passes(&dims, 82, budget, (3 << 20, tip), state)
                };
                let fit = |admits: &dyn Fn(KvDtype, u32) -> Result<bool, EngError>| {
                    choose_cuda_fit(
                        &QWEN3_6_35B_A3B,
                        CudaFitAsk {
                            free,
                            weights,
                            int8_usable: true,
                            kv: None,
                            context: None,
                        },
                        &floor,
                        admits,
                    )
                    .unwrap()
                };
                let (without, with) = (fit(&|_, _| Ok(true)), fit(&admits));
                let name = path.file_name().unwrap().to_string_lossy().to_string();
                eprintln!(
                    "{name:60.60} weights {:.2} GiB | without {without:?} | with {with:?}",
                    weights as f64 / (1u64 << 30) as f64
                );
                // Asking the admission never lengthens the same form's context; a form that has no
                // arena at any context gives way to the next one.
                if let (Some(with), Some(without)) = (with, without)
                    && with.form == without.form
                {
                    assert!(with.context <= without.context);
                }
            }
        }
    }
}
