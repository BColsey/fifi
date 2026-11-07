//! CPU tests of the K-row block lists and of the layout of the K-row buffer: every slot a list
//! names is in the K-row plan and live in the phase of the op that names it, every read follows a
//! write, the pack columns tile the rows that the downstream kernels read, and the buffer's
//! regions are aligned, apart and as wide as the geometry says.
use super::*;
use crate::family::project::cuda::WeightLayout;
use crate::family::qwen36::cuda::{
    common_tests::dims,
    dims::{KvMode, StepOptions},
    step_plan::{KernelScratch, StepPlan},
};

fn geometry() -> MixerGeometry {
    MixerGeometry::new(&dims()).expect("geometry")
}

fn k_row_plan() -> StepPlan {
    let d = dims();
    let options = StepOptions {
        kv_mode: KvMode::Fp16,
        context_tokens: 8,
        delta_slots: 1,
        forked_experts: false,
        tiered: None,
        layout: WeightLayout::Exl3,
    };
    let scratch = KernelScratch::for_verify(&d, MAX_STEP_ROWS, KvMode::Fp16).expect("scratch");
    StepPlan::new_verify(&d, MAX_STEP_ROWS, &options, &scratch).expect("plan")
}

fn lists() -> [(&'static str, Vec<RowsBlockOp>); 2] {
    let ops = RowsKindOps::new(&geometry());
    [
        (
            "delta",
            ops.of(AttentionKind::DeltaNet).expect("delta").to_vec(),
        ),
        ("attn", ops.of(AttentionKind::Full).expect("attn").to_vec()),
    ]
}

#[test]
fn every_slot_of_every_op_is_in_the_k_row_plan_and_live_in_the_phase_of_the_op() {
    let plan = k_row_plan();
    for (name, list) in lists() {
        for op in &list {
            let phase = op.phase() as usize;
            for place in op.reads().into_iter().chain(op.writes()) {
                let Place::Slot(slot) = place else { continue };
                let at = plan
                    .index(slot)
                    .unwrap_or_else(|| panic!("{name}: {slot:?} is not in the K-row plan"));
                let planned = plan.tensors()[at];
                assert!(
                    (planned.first..=planned.last).contains(&phase),
                    "{name}: {slot:?} lives in phases {}..={}, the op {op:?} runs in {phase}",
                    planned.first,
                    planned.last
                );
            }
        }
    }
}

#[test]
fn the_phases_never_go_back_and_the_lists_open_and_close_with_the_norms() {
    for (name, list) in lists() {
        let phases: Vec<usize> = list.iter().map(|op| op.phase() as usize).collect();
        assert!(
            phases.windows(2).all(|w| w[0] <= w[1]),
            "{name}: {phases:?}"
        );
        assert!(matches!(
            list.first(),
            Some(RowsBlockOp::NormPack {
                phase: Phase::MixPack,
                delta: None,
                ..
            })
        ));
        assert!(matches!(
            list.last(),
            Some(RowsBlockOp::NormPack {
                phase: Phase::AddNorm,
                delta: Some(Slot::MixOut),
                ..
            })
        ));
    }
}

#[test]
fn every_place_is_written_before_it_is_read() {
    // The flag word is cleared by the first op of the window, before any block; the residual
    // stream and the weights are not places.
    for (name, list) in lists() {
        let mut written = vec![Place::Slot(Slot::Exl3Flag)];
        for op in &list {
            for read in op.reads() {
                assert!(
                    written.contains(&read),
                    "{name}: {op:?} reads {read:?} first"
                );
            }
            written.extend(op.writes());
        }
    }
}

#[test]
fn the_delta_list_fills_the_row_the_delta_kernel_reads_and_the_conv_reads_the_tight_rows() {
    let g = geometry();
    let d = &g.delta;
    let [(_, delta), _] = lists();
    // z, beta and alpha tile the columns after the qkv columns, up to the row length.
    assert_eq!(d.z_off, d.channels);
    assert_eq!(d.z_off + d.z_rows, d.beta_off);
    assert_eq!(d.beta_off + d.beta_rows, d.alpha_off);
    assert_eq!(d.alpha_off + d.alpha_rows, d.ld);
    let gates: Vec<(Role, usize)> = delta
        .iter()
        .filter_map(|op| match op {
            RowsBlockOp::Gate { role, col, .. } => Some((*role, *col)),
            _ => None,
        })
        .collect();
    assert_eq!(
        gates,
        vec![(Role::SsmBeta, d.beta_off), (Role::SsmAlpha, d.alpha_off)]
    );
    let packs: Vec<&Vec<PackPart>> = delta
        .iter()
        .filter_map(|op| match op {
            RowsBlockOp::Pack { parts, ld, dst, .. } => {
                assert_eq!((*ld, *dst), (d.ld, Slot::DeltaProj));
                Some(parts)
            }
            _ => None,
        })
        .collect();
    assert_eq!(
        packs,
        vec![&vec![PackPart {
            src: Tight::Z,
            n: d.z_rows,
            col: d.z_off
        }]]
    );
    let conv = delta
        .iter()
        .position(|op| {
            matches!(
                op,
                RowsBlockOp::Conv {
                    proj: Tight::Qkv,
                    ..
                }
            )
        })
        .expect("conv over the tight qkv");
    let rule = delta
        .iter()
        .position(|op| matches!(op, RowsBlockOp::Delta { .. }))
        .expect("delta");
    let pack = delta
        .iter()
        .position(|op| matches!(op, RowsBlockOp::Pack { .. }))
        .expect("pack");
    assert!(
        pack < conv && conv < rule,
        "pack {pack} conv {conv} delta {rule}"
    );
    let roles: Vec<Role> = delta
        .iter()
        .filter_map(|op| match op {
            RowsBlockOp::Project { role, .. } => Some(*role),
            _ => None,
        })
        .collect();
    assert_eq!(roles, vec![Role::AttnQkv, Role::AttnGate, Role::SsmOut]);
}

#[test]
fn the_attention_list_interleaves_q_k_and_v_into_the_row_the_attention_kernels_read() {
    let g = geometry();
    let a = &g.attn;
    let [_, (_, attn)] = lists();
    let Some(RowsBlockOp::Pack { parts, ld, dst, .. }) = attn
        .iter()
        .find(|op| matches!(op, RowsBlockOp::Pack { .. }))
    else {
        panic!("the attention list has a pack");
    };
    assert_eq!((*ld, *dst), (a.ld, Slot::AttnProj));
    assert_eq!(
        parts,
        &vec![
            PackPart {
                src: Tight::Q,
                n: a.q_rows,
                col: 0
            },
            PackPart {
                src: Tight::K,
                n: a.kv_rows,
                col: a.k_off
            },
            PackPart {
                src: Tight::V,
                n: a.kv_rows,
                col: a.v_off
            },
        ]
    );
    assert_eq!(a.q_rows, a.k_off);
    assert_eq!(a.k_off + a.kv_rows, a.v_off);
    assert_eq!(a.v_off + a.kv_rows, a.ld);
    let order: Vec<&str> = attn
        .iter()
        .map(|op| match op {
            RowsBlockOp::NormPack { .. } => "norm",
            RowsBlockOp::Project { .. } => "project",
            RowsBlockOp::Pack { .. } => "pack",
            RowsBlockOp::AttnPrep { .. } => "prep",
            RowsBlockOp::AttnDecode { .. } => "decode",
            other => panic!("{other:?} in an attention list"),
        })
        .collect();
    assert_eq!(
        order,
        [
            "norm", "project", "project", "project", "pack", "prep", "decode", "project", "norm"
        ]
    );
}

#[test]
fn the_buffer_regions_are_aligned_apart_and_as_wide_as_the_geometry() {
    let g = geometry();
    for rows in 2..=MAX_STEP_ROWS {
        let io = KrowIoLayout::new(&g, rows).expect("layout");
        let regions = [
            ("ids", io.ids, size_of::<i32>()),
            ("result", io.result, RESULT_PAIR_BYTES),
            ("qkv", io.qkv, g.delta.channels * F32),
            ("z", io.z, g.delta.z_rows * F32),
            ("q", io.q, g.attn.q_rows * F32),
            ("k", io.k, g.attn.kv_rows * F32),
            ("v", io.v, g.attn.kv_rows * F32),
        ];
        let mut end = 0;
        for (name, region, row_bytes) in regions {
            assert_eq!(region.row_bytes, row_bytes, "{rows} rows: {name}");
            assert_eq!(region.offset % 256, 0, "{rows} rows: {name} alignment");
            assert!(
                region.offset >= end,
                "{rows} rows: {name} overlaps the one before"
            );
            end = region.offset
                + if name == "result" {
                    window_full_report_bytes(rows)
                } else {
                    region.bytes(rows)
                };
        }
        assert!(
            io.total >= end && io.total.is_multiple_of(256),
            "{rows} rows: total"
        );
        // The tight places resolve to the regions.
        assert_eq!(io.tight(Tight::Qkv), io.qkv);
        assert_eq!(io.tight(Tight::V), io.v);
    }
}

#[test]
fn the_buffer_serves_two_to_eight_rows_only() {
    for rows in [0, 1, 9, 16] {
        assert!(KrowIoLayout::new(&geometry(), rows).is_err(), "{rows}");
    }
}

#[test]
fn a_window_of_k_rows_uses_the_first_k_rows_of_each_tight_region() {
    let io = KrowIoLayout::new(&geometry(), MAX_STEP_ROWS).expect("layout");
    for rows in 2..=MAX_STEP_ROWS {
        assert_eq!(io.qkv.bytes(rows), rows * io.qkv.row_bytes);
        assert!(io.qkv.bytes(rows) <= io.qkv.bytes(MAX_STEP_ROWS));
    }
}

#[test]
fn the_tail_takes_one_row_count_for_both_kernels() {
    for rows in 2..=MAX_STEP_ROWS {
        check_tail_rows(rows, rows).expect("equal counts");
        // The mutant: a fold over more rows than the argmax wrote reads stale pairs.
        let text = format!(
            "{:?}",
            check_tail_rows(rows, rows + 1).expect_err("refused")
        );
        assert!(text.contains("argmax_rows takes"), "{text}");
        assert!(check_tail_rows(rows + 1, rows).is_err());
    }
}

#[test]
fn a_window_clears_the_status_word_first_and_the_fold_is_the_last_writer_of_it() {
    use super::super::step_ops::{RowsOp, StepOp, token_ops_rows};
    // The sticky bits of an earlier window (the status word is only ever ORed into) cannot reach
    // this window's word: its first op is the one that clears the word, as the first op of a
    // one-row step is, and no op that writes the word runs after the fold.
    for rows in 2..=MAX_STEP_ROWS {
        let ops = token_ops_rows(40, rows).expect("list");
        assert_eq!(ops.first(), Some(&RowsOp::Step(StepOp::Begin)));
        let writers: Vec<usize> = ops
            .iter()
            .enumerate()
            .filter(|(_, op)| {
                op.writes_status(AttentionKind::DeltaNet) || op.writes_status(AttentionKind::Full)
            })
            .map(|(at, _)| at)
            .collect();
        let fold = ops
            .iter()
            .position(|op| *op == RowsOp::StatusFold)
            .expect("fold");
        assert_eq!(writers.last(), Some(&fold), "{rows} rows");
        assert_eq!(
            writers.first(),
            Some(&0),
            "{rows} rows: the clear is the first writer"
        );
        // The argmax and the fold come in this order and nothing is between them.
        let argmax = ops
            .iter()
            .position(|op| *op == RowsOp::ArgmaxRows)
            .expect("argmax");
        assert_eq!(fold, argmax + 1, "{rows} rows");
        // Only the device verify, the shortlist (neither writes the status word) and the advance
        // follow the fold.
        assert_eq!(
            ops[fold + 1..],
            [
                RowsOp::VerifyGreedy,
                RowsOp::ShortlistRows,
                RowsOp::Step(StepOp::Advance)
            ],
            "{rows} rows"
        );
    }
}
