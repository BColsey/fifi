//! A declarative op table of one Flash-Next decode token (the launch order of the S4 draft, PLE
//! at `ple_layer` between layers) replayed against the offsets the planner placed, to find a
//! tensor whose bytes another tensor overwrites before its last read.
//!
//! The lifetimes in `step_plan.rs` are intervals on a one-layer phase axis. This test does not
//! restate them: it runs the real time order (48 layers, the PLE block after layer
//! `ple_layer - 1`) and checks the bytes. A write invalidates every other tensor that shares a
//! byte with it; a read of an invalid tensor is a hazard. Feeds (`EmbedRow`, `PleEmbed`) are
//! written by the host before the step, as `Slot::EmbedRow` and `Slot::PleEmbed` document.

use super::{
    state::FlashGeometry,
    state_tests::real,
    step_plan::{Phase, PlannedSlot, Site, Slot, StepOptions, StepPlan},
};
use crate::{
    family::AttentionKind,
    kernels::cuda::{ArenaTensor, plan_arena},
};

struct Op {
    label: &'static str,
    phase: Phase,
    reads: Vec<Slot>,
    writes: Vec<Slot>,
}

fn op(label: &'static str, phase: Phase, reads: &[Slot], writes: &[Slot]) -> Op {
    Op {
        label,
        phase,
        reads: reads.to_vec(),
        writes: writes.to_vec(),
    }
}

fn hyper_read(site: Site) -> Vec<Slot> {
    let mut written = vec![
        Slot::HyperNormed(site),
        Slot::HyperLow(site),
        Slot::HyperMix(site),
        Slot::Mixed(site),
    ];
    if site != Site::Final {
        written.push(Slot::Gate(site));
    }
    written
}

/// The ops of one layer of `kind`, in launch order.
fn layer_ops(kind: AttentionKind) -> Vec<Op> {
    let mut ops = vec![op(
        "hyper_read attn",
        Phase::HyperReadA,
        &[Slot::Stream],
        &hyper_read(Site::Attn),
    )];
    match kind {
        AttentionKind::DeltaNet => {
            ops.push(op(
                "delta in-projections",
                Phase::MixerIn,
                &[Slot::Mixed(Site::Attn)],
                &[Slot::DeltaProj, Slot::Exl3Xh, Slot::Exl3Z, Slot::Exl3Flag],
            ));
            ops.push(op(
                "gdn_conv",
                Phase::DeltaConv,
                &[Slot::DeltaProj],
                &[Slot::DeltaConv],
            ));
            ops.push(op(
                "gdn_scan",
                Phase::DeltaRule,
                &[Slot::DeltaConv, Slot::DeltaProj],
                &[Slot::DeltaCore],
            ));
            ops.push(op(
                "delta out-projection",
                Phase::MixerOut,
                &[Slot::DeltaCore],
                &[Slot::BlockOut(Site::Attn), Slot::Exl3Xh, Slot::Exl3Z],
            ));
        }
        _ => {
            ops.push(op(
                "qsa in-projections",
                Phase::MixerIn,
                &[Slot::Mixed(Site::Attn)],
                &[Slot::AttnProj, Slot::Exl3Xh, Slot::Exl3Z, Slot::Exl3Flag],
            ));
            ops.push(op(
                "attn_prep",
                Phase::AttnPrep,
                &[Slot::AttnProj],
                &[Slot::AttnQ],
            ));
            ops.push(op(
                "qsa_index",
                Phase::Index,
                &[Slot::AttnProj],
                &[Slot::IndexQueries],
            ));
            ops.push(op(
                "qsa_score",
                Phase::Score,
                &[Slot::IndexQueries],
                &[Slot::Scores],
            ));
            ops.push(op(
                "qsa_select",
                Phase::Select,
                &[Slot::Scores],
                &[Slot::Chosen],
            ));
            ops.push(op(
                "qsa_attn",
                Phase::AttnCore,
                &[Slot::AttnQ, Slot::AttnProj, Slot::Chosen],
                &[Slot::AttnPart, Slot::AttnCore],
            ));
            ops.push(op(
                "qsa out-projection",
                Phase::MixerOut,
                &[Slot::AttnCore],
                &[Slot::BlockOut(Site::Attn), Slot::Exl3Xh, Slot::Exl3Z],
            ));
        }
    }
    ops.push(op(
        "hyper_apply attn",
        Phase::HyperApplyA,
        &[
            Slot::Stream,
            Slot::Gate(Site::Attn),
            Slot::BlockOut(Site::Attn),
        ],
        &[Slot::Stream],
    ));
    ops.push(op(
        "hyper_read mlp",
        Phase::HyperReadM,
        &[Slot::Stream],
        &hyper_read(Site::Mlp),
    ));
    ops.push(op(
        "router",
        Phase::Route,
        &[Slot::Mixed(Site::Mlp)],
        &[
            Slot::RouterLogits,
            Slot::TopkId,
            Slot::TopkWeight,
            Slot::SharedGate,
        ],
    ));
    // The EXL3 MoE block is one op in stream order: gate and up of the picks and the shared
    // expert, SwiGLU, down, combine.
    ops.push(op(
        "moe gate/up",
        Phase::GateUp,
        &[Slot::Mixed(Site::Mlp), Slot::TopkId],
        &[Slot::ExpertGu, Slot::SharedGu, Slot::Exl3Xh, Slot::Exl3Z],
    ));
    ops.push(op(
        "moe swiglu",
        Phase::Swiglu,
        &[Slot::ExpertGu, Slot::SharedGu],
        &[Slot::ExpertAct, Slot::SharedAct],
    ));
    ops.push(op(
        "moe down",
        Phase::Down,
        &[Slot::ExpertAct, Slot::SharedAct, Slot::TopkId],
        &[Slot::ExpertDown, Slot::SharedOut, Slot::Exl3Xh, Slot::Exl3Z],
    ));
    ops.push(op(
        "moe combine",
        Phase::Combine,
        &[
            Slot::ExpertDown,
            Slot::SharedOut,
            Slot::TopkWeight,
            Slot::TopkId,
            Slot::SharedGate,
            Slot::Mixed(Site::Mlp),
        ],
        &[Slot::BlockOut(Site::Mlp)],
    ));
    ops.push(op(
        "hyper_apply mlp",
        Phase::HyperApplyM,
        &[
            Slot::Stream,
            Slot::Gate(Site::Mlp),
            Slot::BlockOut(Site::Mlp),
        ],
        &[Slot::Stream],
    ));
    ops
}

/// One token: the feeds, the embedding, 48 layers with PLE after layer `ple_layer - 1`, the final
/// mixer, the head and the argmax.
fn token_ops(geometry: &FlashGeometry) -> Vec<Op> {
    let mut ops = vec![
        op(
            "feeds and flag clear",
            Phase::StepBegin,
            &[],
            &[Slot::EmbedRow, Slot::PleEmbed, Slot::Exl3Flag],
        ),
        op("embed", Phase::Embed, &[Slot::EmbedRow], &[Slot::Stream]),
    ];
    for (layer, &kind) in geometry.layers.iter().enumerate() {
        if layer == geometry.ple_layer {
            ops.push(op(
                "ple",
                Phase::Ple,
                &[Slot::PleEmbed, Slot::Stream],
                &[
                    Slot::PleKeys,
                    Slot::PleValue,
                    Slot::PleGated,
                    Slot::PleNormed,
                    Slot::PleOut,
                ],
            ));
            ops.push(op(
                "ple update",
                Phase::Ple,
                &[Slot::PleOut],
                &[Slot::Stream],
            ));
        }
        ops.extend(layer_ops(kind));
    }
    ops.push(op(
        "final mixer",
        Phase::FinalMix,
        &[Slot::Stream],
        &hyper_read(Site::Final),
    ));
    ops.push(op(
        "lm_head",
        Phase::LmHead,
        &[Slot::Mixed(Site::Final)],
        &[Slot::Logits, Slot::Exl3Xh, Slot::Exl3Z],
    ));
    ops.push(op(
        "argmax",
        Phase::Argmax,
        &[Slot::Logits, Slot::Exl3Flag],
        &[Slot::ArgmaxPart],
    ));
    ops
}

#[derive(Clone, Copy)]
struct Placed {
    slot: Slot,
    offset: usize,
    bytes: usize,
    first: usize,
    last: usize,
}

fn overlap(a: &Placed, b: &Placed) -> bool {
    a.offset < b.offset + b.bytes && b.offset < a.offset + a.bytes
}

/// The planned slots as placed, with `tweak` applied to each lifetime first (the planner is run
/// again on the tweaked list).
fn placed_with(plan: &StepPlan, tweak: impl Fn(&mut PlannedSlot)) -> Vec<Placed> {
    let mut slots = plan.slots().to_vec();
    slots.iter_mut().for_each(tweak);
    let tensors: Vec<ArenaTensor> = slots
        .iter()
        .map(|p| ArenaTensor {
            bytes: p.bytes,
            first: p.first,
            last: p.last,
        })
        .collect();
    let arena = plan_arena(&tensors).expect("plan the arena");
    slots
        .iter()
        .zip(&arena.offsets)
        .map(|(p, &offset)| Placed {
            slot: p.slot,
            offset,
            bytes: p.bytes,
            first: p.first,
            last: p.last,
        })
        .collect()
}

/// Every hazard of the token: a read of a tensor whose bytes were overwritten, a launch outside
/// the tensor's lifetime, or a launch that writes bytes it also reads.
fn hazards(placed: &[Placed], ops: &[Op]) -> Vec<String> {
    let find = |slot: Slot| {
        placed
            .iter()
            .find(|p| p.slot == slot)
            .unwrap_or_else(|| panic!("{slot:?} is not planned"))
    };
    let mut valid: Vec<Slot> = Vec::new();
    let mut found = Vec::new();
    for (index, step) in ops.iter().enumerate() {
        let at = step.phase as usize;
        for &slot in step.reads.iter().chain(&step.writes) {
            let p = find(slot);
            if !(p.first..=p.last).contains(&at) {
                found.push(format!(
                    "op {index} {}: {slot:?} is used at phase {at} outside its lifetime {}..={}",
                    step.label, p.first, p.last
                ));
            }
        }
        for &slot in &step.reads {
            if !valid.contains(&slot) {
                found.push(format!(
                    "op {index} {}: reads {slot:?}, whose bytes were overwritten or never written",
                    step.label
                ));
            }
        }
        for &w in &step.writes {
            for &r in step.reads.iter().filter(|&&r| r != w) {
                if overlap(find(w), find(r)) {
                    found.push(format!(
                        "op {index} {}: writes {w:?} over its own input {r:?}",
                        step.label
                    ));
                }
            }
        }
        for &w in &step.writes {
            let written = *find(w);
            valid.retain(|&other| other == w || !overlap(&written, find(other)));
            if !valid.contains(&w) {
                valid.push(w);
            }
        }
    }
    found
}

#[test]
fn the_planned_offsets_have_no_hazard_over_a_full_token() {
    let (_, g) = real();
    assert_eq!(g.ple_layer, 1, "the real model runs PLE after layer 0");
    for (rows, positions) in [(1, 32_768), (1, 262_144), (4, 4096)] {
        let plan = StepPlan::new(&g, rows, positions, &StepOptions::default()).unwrap();
        let placed = placed_with(&plan, |_| {});
        let ops = token_ops(&g);
        // 2 feeds/embed, 48 layers of 11 to 17 ops, 2 PLE ops, 3 epilogue ops.
        assert!(ops.len() > 48 * 11, "{}", ops.len());
        let found = hazards(&placed, &ops);
        assert!(
            found.is_empty(),
            "rows {rows}, {positions} positions: {found:#?}"
        );
    }
}

#[test]
fn a_feed_that_dies_before_its_consumer_is_found() {
    // The lifetime this slice first planned for PleEmbed, StepBegin..Ple, puts the n-gram row's
    // bytes up for reuse from phase 3 on, while the PLE op runs after layer 0.
    let (_, g) = real();
    let plan = StepPlan::new(&g, 1, 32_768, &StepOptions::default()).unwrap();
    let placed = placed_with(&plan, |p| {
        if p.slot == Slot::PleEmbed {
            p.last = Phase::Ple as usize;
        }
    });
    let found = hazards(&placed, &token_ops(&g));
    assert!(
        found.iter().any(|line| line.contains("PleEmbed")),
        "the old lifetime must be a hazard: {found:#?}"
    );
}

#[test]
fn a_tensor_cut_short_is_found_in_every_mixer() {
    // Cutting the gate of the attention site at its read breaks the apply of both layer kinds.
    let (_, g) = real();
    let plan = StepPlan::new(&g, 1, 32_768, &StepOptions::default()).unwrap();
    let placed = placed_with(&plan, |p| {
        if p.slot == Slot::Gate(Site::Attn) {
            p.last = Phase::HyperReadA as usize;
        }
    });
    let found = hazards(&placed, &token_ops(&g));
    assert!(
        found.iter().any(|line| line.contains("Gate(Attn)")),
        "{found:#?}"
    );
}
