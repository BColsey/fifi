//! Host tests of the shared expert inside the routed launches (`merge.rs`).
//!
//! The block queues kernels on a device, so the host tests pin what a host can see: the switch
//! parses strictly and is off by default; the plan of the merged block (which pair is in which
//! launch, which ids, rows and scratch rows) is the data the executor runs, and it is compared
//! against an independent statement of the reference block; the working buffer layout is
//! disjoint and contiguous where the launches need it; the reference block's source does not
//! mention the merged one, and the step reaches the merged block only through the switch. The
//! byte identity against the reference is the device tests' (`exl3_moe_merge_device_tests`).

use std::ffi::OsStr;

use super::exl3_merge_plan::{
    IdWord, Input, Member, MergeLayout, PairSpans, PlanDims, PlannedPair, Proj,
    SHARED_IN_ROUTED_VAR, SharedFacts, Span, block_plan, check_merge, load_suffix,
    reference_kernel_launches, resolve_switch, shared_in_routed_from_env,
};
use crate::{EngError, kernels::single_row::SingleRow};

/// The shape of the Qwen3.6-35B-A3B block.
fn qwen36() -> PlanDims {
    PlanDims {
        rows: 1,
        hidden: 2048,
        top_k: 8,
        inter: 512,
        shared_inter: 512,
    }
}

#[test]
fn the_switch_parses_strictly_and_is_off_when_unset() {
    assert_eq!(shared_in_routed_from_env(None), Ok(false));
    for (value, on) in [("0", false), ("1", true), (" 1 ", true), (" 0", false)] {
        assert_eq!(
            shared_in_routed_from_env(Some(OsStr::new(value))),
            Ok(on),
            "{value:?}"
        );
    }
    for bad in [
        "", "on", "off", "true", "false", "2", "01", "-1", "yes", "lean", "1 1",
    ] {
        let error = shared_in_routed_from_env(Some(OsStr::new(bad))).expect_err(bad);
        assert!(error.contains(SHARED_IN_ROUTED_VAR), "{bad:?}: {error}");
    }
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        let raw = OsStr::from_bytes(&[0x31, 0xff]);
        assert!(shared_in_routed_from_env(Some(raw)).is_err(), "not UTF-8");
    }
}

#[test]
fn the_load_line_names_the_switch_only_when_it_is_on() {
    assert_eq!(load_suffix(false), "");
    assert_eq!(load_suffix(true), " exl3_shared_in_routed=on");
}

/// The reference block, stated independently of the executor: gate and up of pick `s` are pairs
/// `2 s` and `2 s + 1` of the routed gate/up group; the shared expert follows in its own groups.
fn reference_routed_gate_up(top_k: usize) -> Vec<(Proj, u32, usize)> {
    let mut pairs = Vec::new();
    for slot in 0..top_k {
        pairs.push((Proj::Gate, slot as u32, 2 * slot));
        pairs.push((Proj::Up, slot as u32, 2 * slot + 1));
    }
    pairs
}

#[test]
fn the_merged_gate_up_group_is_the_routed_pairs_then_the_shared_pair_of_each_half() {
    for top_k in [1usize, 4, 8, 9] {
        let plan = block_plan(top_k);
        assert_eq!(plan.gate_up.len(), 2 * top_k + 2, "top_k {top_k}");
        // The first 2 top_k pairs are the reference's routed group, pair for pair.
        let routed: Vec<_> = plan.gate_up[..2 * top_k]
            .iter()
            .map(|pair| match pair.member {
                Member::Routed(slot) => (pair.proj, slot, pair.out_row),
                Member::Shared => panic!("a shared pair among the routed ones"),
            })
            .collect();
        assert_eq!(routed, reference_routed_gate_up(top_k));
        // Then exactly one shared gate and one shared up, in that order, after the routed rows.
        assert_eq!(
            &plan.gate_up[2 * top_k..],
            &[
                PlannedPair {
                    proj: Proj::Gate,
                    member: Member::Shared,
                    input: Input::NormRow,
                    out_row: 2 * top_k,
                },
                PlannedPair {
                    proj: Proj::Up,
                    member: Member::Shared,
                    input: Input::NormRow,
                    out_row: 2 * top_k + 1,
                },
            ]
        );
        // Every pair of the group reads the normalized row.
        assert!(plan.gate_up.iter().all(|p| p.input == Input::NormRow));
    }
}

#[test]
fn the_merged_down_group_is_the_routed_picks_then_one_shared_pair() {
    for top_k in [1usize, 4, 8, 9] {
        let plan = block_plan(top_k);
        assert_eq!(plan.down.len(), top_k + 1);
        for (slot, pair) in plan.down[..top_k].iter().enumerate() {
            assert_eq!(
                *pair,
                PlannedPair {
                    proj: Proj::Down,
                    member: Member::Routed(slot as u32),
                    input: Input::ActRow(slot),
                    out_row: slot,
                }
            );
        }
        assert_eq!(
            plan.down[top_k],
            PlannedPair {
                proj: Proj::Down,
                member: Member::Shared,
                input: Input::ActRow(top_k),
                out_row: top_k,
            }
        );
        assert_eq!(plan.swiglu_rows, top_k + 1);
    }
}

#[test]
fn the_shared_expert_runs_once_per_token_as_one_pair_per_projection() {
    let plan = block_plan(8);
    for proj in [Proj::Gate, Proj::Up, Proj::Down] {
        let shared = plan
            .gate_up
            .iter()
            .chain(&plan.down)
            .filter(|p| p.proj == proj && p.member == Member::Shared)
            .count();
        assert_eq!(shared, 1, "{proj:?}");
        let routed = plan
            .gate_up
            .iter()
            .chain(&plan.down)
            .filter(|p| p.proj == proj && matches!(p.member, Member::Routed(_)))
            .count();
        assert_eq!(
            routed, 8,
            "{proj:?}: one routed pair per pick, not per shared use"
        );
    }
    // Each routed pick appears once per projection, in slot order.
    for proj in [Proj::Gate, Proj::Up, Proj::Down] {
        let slots: Vec<u32> = plan
            .gate_up
            .iter()
            .chain(&plan.down)
            .filter(|p| p.proj == proj)
            .filter_map(|p| match p.member {
                Member::Routed(slot) => Some(slot),
                Member::Shared => None,
            })
            .collect();
        assert_eq!(slots, (0..8).collect::<Vec<_>>(), "{proj:?}");
    }
}

#[test]
fn output_rows_are_distinct_and_dense_so_the_launch_spans_stay_contiguous() {
    for top_k in [1usize, 8, 9] {
        let plan = block_plan(top_k);
        // The launches need the output of pair p at row p: the group contract of the kernels.
        let gate_up: Vec<usize> = plan.gate_up.iter().map(|p| p.out_row).collect();
        assert_eq!(gate_up, (0..2 * top_k + 2).collect::<Vec<_>>());
        let down: Vec<usize> = plan.down.iter().map(|p| p.out_row).collect();
        assert_eq!(down, (0..=top_k).collect::<Vec<_>>());
    }
}

#[test]
fn the_launch_plan_has_the_expected_counts_per_layer() {
    let plan = block_plan(8);
    assert_eq!(plan.gate_up.len(), 18);
    assert_eq!(plan.down.len(), 9);
    // Merged: gate/up (3 kernels), swiglu (1), down (3).
    assert_eq!(plan.kernel_launches(), 7);
    // Reference: routed gate/up (3), swiglu (1), routed down (3), then the shared expert's
    // gate/up (3), swiglu (1) and down (3).
    assert_eq!(reference_kernel_launches(), 14);
    let saved = reference_kernel_launches() - plan.kernel_launches();
    assert_eq!(saved, 7, "per MoE layer");
    // 40 MoE layers: the 280 launches per token the analysis counts.
    assert_eq!(40 * saved, 280);
}

#[test]
fn the_layout_is_disjoint_aligned_and_sized_for_the_launches() {
    let dims = qwen36();
    let layout = MergeLayout::new(&dims).expect("the Qwen3.6 block fits");
    let (h, i, k) = (dims.hidden, dims.inter, dims.top_k);
    assert_eq!(layout.xh.bytes, 18 * h * 4, "18 rotated rows of k = h");
    assert_eq!(
        layout.z.bytes,
        (9 * h * 4).max(18 * i * 4),
        "the larger of the two launches"
    );
    assert_eq!(layout.gu.bytes, (k + 1) * 2 * i * 4);
    assert_eq!(layout.act.bytes, (k + 1) * i * 4);
    assert_eq!(layout.down.bytes, (k + 1) * h * 4);
    assert_eq!(layout.zero.bytes, 4);
    let spans = [
        layout.xh,
        layout.z,
        layout.gu,
        layout.act,
        layout.down,
        layout.zero,
    ];
    for (n, a) in spans.iter().enumerate() {
        assert_eq!(a.offset % 256, 0, "span {n} is 256-byte aligned");
        assert!(a.offset + a.bytes <= layout.total);
        for b in &spans[n + 1..] {
            assert!(
                a.offset + a.bytes <= b.offset || b.offset + b.bytes <= a.offset,
                "{a:?} overlaps {b:?}"
            );
        }
    }
    // The swiglu reads and writes 16-byte aligned rows.
    assert_eq!(layout.gu.offset % 16, 0);
    assert_eq!(layout.act.offset % 16, 0);
    assert_eq!((2 * i * 4) % 16, 0);
}

#[test]
fn a_block_the_launches_cannot_hold_is_refused() {
    let mut dims = qwen36();
    assert!(MergeLayout::new(&dims).is_ok());
    // 9 picks need 20 gate/up pairs: the limit of the 20-pair kernels.
    dims.top_k = 9;
    assert!(MergeLayout::new(&dims).is_ok());
    // Flash-Next's ten picks need 22.
    dims.top_k = 10;
    assert!(MergeLayout::new(&dims).is_err());
    dims.top_k = 0;
    assert!(MergeLayout::new(&dims).is_err());
    // A shared expert of another width cannot join the pairs of the routed launch.
    let mut dims = qwen36();
    dims.shared_inter = 1024;
    assert!(MergeLayout::new(&dims).is_err());
    let mut dims = qwen36();
    dims.rows = 2;
    assert!(MergeLayout::new(&dims).is_err());
}

/// The text without whitespace and without the trailing commas rustfmt adds to wrapped lists.
fn squeeze(text: &str) -> String {
    let flat: String = text.split_whitespace().collect();
    flat.replace(",)", ")").replace(",}", "}")
}

/// Whether the squeezed `text` holds the squeezed `pattern`.
fn has(text: &str, pattern: &str) -> bool {
    text.contains(&squeeze(pattern))
}

/// A function body of `source` from `name` to its closing brace at column 0.
fn section<'s>(source: &'s str, name: &str) -> &'s str {
    let start = source.find(name).unwrap_or_else(|| panic!("{name}"));
    let rest = &source[start..];
    let end = rest.find("\n}\n").map_or(rest.len(), |at| at + 3);
    &rest[..end]
}

/// The reference block is the code it was: it never names the merged block, so the switch cannot
/// reach it, and Flash-Next (which calls it) cannot run the merged launches.
#[test]
fn the_reference_block_does_not_know_the_merged_one() {
    let source = include_str!("cuda/exl3_moe.rs");
    let reference = source;
    for name in [
        "fn run_exl3_moe(",
        "fn routed_on_device(",
        "fn routed(",
        "fn shared(",
        "fn grouped(",
        "fn grouped_indexed(",
        "fn indexed_launch(",
    ] {
        let body = section(reference, name);
        for word in ["merge", "Merge", "wide(", "_wide", "SHARED_IN_ROUTED"] {
            assert!(!body.contains(word), "{name} mentions {word}");
        }
    }
    let run = squeeze(section(source, "pub(in crate::family) fn run_exl3_moe("));
    assert!(
        run.contains("route(launch,weights,bufs,dims)?;ifdevice_ids{routed_on_device(launch,weights,bufs,dims)?;}else{"),
        "{run}"
    );
    assert!(
        run.contains("shared(launch,weights,bufs,dims)?;combine(launch,bufs,dims)"),
        "{run}"
    );
}

/// The id-indexed launch with the 20-pair kernels is reached by the merged block alone.
#[test]
fn only_the_merged_block_asks_for_the_wide_indexed_launch() {
    let merge = include_str!("cuda/exl3_moe/merge.rs");
    assert_eq!(merge.matches("exl3_grouped_indexed_wide(").count(), 1);
    for (file, source) in [
        ("exl3_moe.rs", include_str!("cuda/exl3_moe.rs")),
        ("step.rs", include_str!("cuda/step.rs")),
        ("mixer.rs", include_str!("cuda/mixer.rs")),
        ("mixer_ops.rs", include_str!("cuda/mixer_ops.rs")),
        (
            "prefill_exl3_moe.rs",
            include_str!("cuda/prefill_exl3_moe.rs"),
        ),
    ] {
        assert!(!source.contains("exl3_grouped_indexed_wide"), "{file}");
    }
    let resident = include_str!("../../kernels/cuda/resident/exl3.rs");
    // The default entry keeps its meaning: it forwards without widening.
    let default = squeeze(section(resident, "pub(crate) fn exl3_grouped_indexed("));
    assert!(
        default.contains("self.grouped_indexed_by(stream,calls,cols,false)"),
        "{default}"
    );
    let wide = squeeze(section(
        resident,
        "pub(crate) fn exl3_grouped_indexed_wide(",
    ));
    assert!(
        wide.contains("self.grouped_indexed_by(stream,calls,cols,true)"),
        "{wide}"
    );
    // `widen` only adds the 20-pair option; nothing else reads it.
    let by = section(resident, "fn grouped_indexed_by(");
    assert_eq!(by.matches("widen").count(), 2, "{by}");
    assert!(by.contains("options.pairs20 |= widen;"));
}

/// The step reaches the merged block only through a value that exists when the switch is on, and
/// reads the switch at the start, before the stream, for the EXL3 layout only.
#[test]
fn the_step_reaches_the_merged_block_only_through_the_switch() {
    let step = include_str!("cuda/step.rs");
    let squeezed = squeeze(step);
    assert!(
        squeezed.contains(
            "letexl3_merge=ifoptions.layout==WeightLayout::Exl3&&shared_in_routed_explicit()?{Some(SharedMerge::new("
        ),
        "the merge exists only for the EXL3 layout with the switch on"
    );
    assert!(squeezed.contains("else{None};"));
    let run = squeeze(section(step, "fn run_op("));
    assert!(
        run.contains("ifletSome(merge)=&self.exl3_merge{run_exl3_moe_merged(&launch,(weights,&bufs,merge),&self.moe_dims,self.exl3_device_ids)}else{run_exl3_moe(&launch,weights,&bufs,&self.moe_dims,self.exl3_device_ids)}"),
        "{run}"
    );
    // The merge is built from the weights that exist, before the stream of the session.
    let new = &step[step.find("pub(crate) fn new(").expect("new")..];
    let merge = new.find("shared_in_routed_explicit()").expect("the switch");
    let stream = new.find("context.stream()").expect("the stream");
    assert!(
        new.find("check_single_gather()").expect("check") < stream,
        "the single-row switch check precedes the stream"
    );
    // SharedMerge::new reads the single-row kernel the check validated, after the resolve of
    // the layer weights it checks.
    assert!(new.find("Exl3MoeWeights::resolve").expect("resolve") < merge);
}

/// The merged executor's device-pick path makes no host read and no wait: a captured body cannot.
#[test]
fn the_merged_device_path_never_waits_for_the_host() {
    let source = include_str!("cuda/exl3_moe/merge.rs");
    for name in [
        "fn merged_on_device(",
        "fn indexed_group(",
        "fn pair_ranges",
    ] {
        let body = section(source, name);
        for forbidden in [
            "read_at(",
            "write_at(",
            "synchronize(",
            "read_picks(",
            "upload(",
        ] {
            assert!(!body.contains(forbidden), "{name} calls {forbidden}");
        }
    }
    let block = section(source, "pub(in crate::family) fn run_exl3_moe_merged(");
    let device = block.find("if device_ids {").expect("the device branch");
    let host = block.find("read_picks(").expect("the host branch");
    assert!(device < host, "the host read is in the else arm");
}

/// The combine is the one kernel of the reference block: only its ranges move.
#[test]
fn the_combine_keeps_its_kernel_and_inputs() {
    let source = include_str!("cuda/exl3_moe.rs");
    let combine = squeeze(section(source, "fn combine("));
    assert!(
        combine
            .contains("combine_ranges(launch,&bufs.route,(bufs.expert_down,bufs.shared_out),dims)"),
        "{combine}"
    );
    let ranges = squeeze(section(source, "fn combine_ranges("));
    for part in [
        "down,weights:route.weights",
        "shared:Some(SharedTerm{gate:route.shared_gate,row:shared})",
        "res:route.resid_in",
        "out:CombineOut::To(route.resid_out)",
    ] {
        assert!(ranges.contains(part), "{part}: {ranges}");
    }
    let merge = squeeze(include_str!("cuda/exl3_moe/merge.rs"));
    assert!(merge.contains("combine_ranges(launch,&bufs.route,(down,shared),dims)"));
}

// ---- where each pair reads and writes, and which id word it uses ----

#[test]
fn the_shared_pair_reads_the_zero_word_and_every_routed_pair_its_own_router_slot() {
    for top_k in [1usize, 8, 9] {
        let plan = block_plan(top_k);
        for group in [&plan.gate_up, &plan.down] {
            for pair in group.iter() {
                match pair.member {
                    Member::Shared => assert_eq!(pair.id_word(), (IdWord::Zero, 0)),
                    Member::Routed(slot) => assert_eq!(pair.id_word(), (IdWord::Router, slot)),
                }
            }
            let zero = group
                .iter()
                .filter(|p| p.id_word().0 == IdWord::Zero)
                .count();
            assert_eq!(
                zero,
                group.len() - top_k * if group.len() > top_k + 1 { 2 } else { 1 }
            );
        }
    }
}

#[test]
fn the_spans_of_each_pair_put_the_shared_pair_after_the_routed_rows() {
    let dims = qwen36();
    let layout = MergeLayout::new(&dims).unwrap();
    let plan = block_plan(dims.top_k);
    let (h, i, k) = (dims.hidden * 4, dims.inter * 4, dims.top_k);
    // Gate/up: pair p writes gu half-row p; every pair reads the normalized row.
    for (p, pair) in plan.gate_up.iter().enumerate() {
        assert_eq!(
            pair.spans(&dims, &layout),
            PairSpans {
                input: None,
                output: Span {
                    offset: layout.gu.offset + p * i,
                    bytes: i
                },
            },
            "gate/up pair {p}"
        );
    }
    // Down: pair s reads act row s and writes down row s; the shared pair is row top_k.
    for (s, pair) in plan.down.iter().enumerate() {
        assert_eq!(
            pair.spans(&dims, &layout),
            PairSpans {
                input: Some(Span {
                    offset: layout.act.offset + s * i,
                    bytes: i
                }),
                output: Span {
                    offset: layout.down.offset + s * h,
                    bytes: h
                },
            },
            "down pair {s}"
        );
    }
    assert_eq!(plan.down.len(), k + 1);
    // The 9 down rows and 9 gate/up rows fit the areas the layout reserved.
    assert!(layout.gu.bytes >= (k + 1) * 2 * i);
    assert!(layout.down.bytes >= (k + 1) * h);
}

#[test]
fn the_combine_reads_the_routed_rows_then_the_shared_row() {
    let dims = qwen36();
    let layout = MergeLayout::new(&dims).unwrap();
    let (routed, shared) = layout.combine_spans(&dims);
    let h = dims.hidden * 4;
    assert_eq!(
        routed,
        Span {
            offset: layout.down.offset,
            bytes: dims.top_k * h
        }
    );
    assert_eq!(
        shared,
        Span {
            offset: layout.down.offset + dims.top_k * h,
            bytes: h
        }
    );
    // The shared row is the output of the shared down pair, and the routed term ends where it begins.
    let plan = block_plan(dims.top_k);
    let shared_down = plan.down.last().unwrap().spans(&dims, &layout).output;
    assert_eq!(shared_down, shared);
    assert_eq!(routed.offset + routed.bytes, shared.offset);
    // The swiglu covers the routed rows and the shared row.
    assert_eq!(plan.swiglu_rows, dims.top_k + 1);
}

// ---- the refusals of the merged block ----

fn facts(bias: bool, bits_twice: u8) -> [SharedFacts; 3] {
    ["shared gate", "shared up", "shared down"].map(|name| SharedFacts {
        name,
        bias,
        bits_twice,
    })
}

fn refusal(dims: &PlanDims, kernel: SingleRow, layers: &[[SharedFacts; 3]]) -> String {
    let error = check_merge(dims, kernel, layers).expect_err("a refusal");
    assert!(error.contains(SHARED_IN_ROUTED_VAR), "{error}");
    error
}

#[test]
fn the_merged_block_accepts_the_qwen36_pack_and_refuses_what_it_cannot_run() {
    let dims = qwen36();
    let ok = [facts(false, 8), facts(false, 12)];
    assert_eq!(check_merge(&dims, SingleRow::Wide(1), &ok), Ok(()));
    assert_eq!(check_merge(&dims, SingleRow::Wide(1), &[]), Ok(()));
    // The single-row kernel must be wide1.
    for kernel in [
        SingleRow::Legacy,
        SingleRow::Gather,
        SingleRow::Wide(2),
        SingleRow::Wide(4),
    ] {
        assert!(refusal(&dims, kernel, &ok).contains("wide1"), "{kernel:?}");
    }
    // A shared matrix with a bias, in any of the three roles and any layer.
    for role in 0..3 {
        let mut layer = facts(false, 8);
        layer[role].bias = true;
        let error = refusal(&dims, SingleRow::Wide(1), &[facts(false, 8), layer]);
        assert!(
            error.contains("layer 1") && error.contains("bias"),
            "{error}"
        );
        assert!(error.contains(layer[role].name), "{error}");
    }
    // A rate other than 4 and 6 bits per weight (bits twice 8 and 12).
    for bits in [4u8, 6, 10, 14, 16] {
        let error = refusal(&dims, SingleRow::Wide(1), &[facts(false, bits)]);
        assert!(error.contains("bits twice"), "{bits}: {error}");
    }
    let mut layer = facts(false, 8);
    layer[2].bits_twice = 10;
    assert!(refusal(&dims, SingleRow::Wide(1), &[layer]).contains("shared down"));
    // More picks than 20 pairs hold.
    let mut wide = qwen36();
    wide.top_k = 10;
    assert!(refusal(&wide, SingleRow::Wide(1), &ok).contains("picks"));
    // A shared expert of another width, and more than one row.
    let mut narrow = qwen36();
    narrow.shared_inter = 256;
    assert!(refusal(&narrow, SingleRow::Wide(1), &ok).contains("wide"));
    let mut rows = qwen36();
    rows.rows = 2;
    assert!(refusal(&rows, SingleRow::Wide(1), &ok).contains("one row"));
    // The block is checked before the kernel: both wrong names the block.
    assert!(refusal(&rows, SingleRow::Gather, &ok).contains("one row"));
}

#[test]
fn the_merged_block_checks_run_in_the_constructor_and_name_each_fact() {
    let merge = include_str!("cuda/exl3_moe/merge.rs");
    let new = squeeze(section(merge, "pub(in crate::family) fn new("));
    for part in [
        "bias:weight.bias.is_some()",
        "bits_twice:weight.bits_twice",
        "check_merge(&plan,ctx.single_row_setting()?,&facts)",
        "weights.shared_gate",
        "weights.shared_up",
        "weights.shared_down",
    ] {
        assert!(new.contains(part), "{part}: {new}");
    }
}

// ---- the switch at the session start ----

#[test]
fn the_start_refuses_a_bad_switch_for_every_weight_format_and_one_on_gguf() {
    let bad = || {
        Err(EngError::BadRequest(
            "FIFI_EXL3_SHARED_IN_ROUTED must be unset".into(),
        ))
    };
    for exl3 in [false, true] {
        let error = resolve_switch(bad(), exl3).expect_err("a bad value");
        assert!(format!("{error:?}").contains(SHARED_IN_ROUTED_VAR));
        assert!(!resolve_switch(Ok(false), exl3).unwrap());
    }
    assert!(resolve_switch(Ok(true), true).unwrap());
    let error = resolve_switch(Ok(true), false).expect_err("1 on GGUF");
    let text = format!("{error:?}");
    assert!(
        text.contains(SHARED_IN_ROUTED_VAR) && text.contains("EXL3"),
        "{text}"
    );
}

/// The start resolves the switch before the projector exists and before any weight is opened.
#[test]
fn the_start_resolves_the_switch_before_any_weight_work() {
    let start = include_str!("cuda/start.rs");
    let body = &start[start.find("pub(crate) fn start(").expect("start")..];
    let switch = body.find("resolve_switch(").expect("the switch check");
    for later in [
        "check_tiered_start(",
        "CudaProjector::new()",
        "open_gguf(",
        "load_exl3_dir(",
    ] {
        let at = body.find(later).unwrap_or_else(|| panic!("{later}"));
        assert!(switch < at, "resolve_switch must precede {later}");
    }
    let call = squeeze(&body[switch..switch + 160]);
    assert!(
        call.contains("resolve_switch(shared_in_routed_explicit(),matches!(settings.weights,WeightFormat::Exl3),)?;")
            || call.contains("resolve_switch(shared_in_routed_explicit(),matches!(settings.weights,WeightFormat::Exl3))?;"),
        "{call}"
    );
}

/// The test hook of the step refuses a non-EXL3 layout instead of silently building nothing.
#[test]
fn the_step_hook_refuses_a_layout_that_is_not_exl3() {
    let step = include_str!("cuda/step.rs");
    let hook = squeeze(section(step, "pub(crate) fn set_shared_in_routed("));
    assert!(
        hook.contains("ifon&&self.layout!=WeightLayout::Exl3{returnErr("),
        "{hook}"
    );
    assert!(
        hook.contains("self.exl3_merge=ifon{Some(SharedMerge::new("),
        "{hook}"
    );
    let field = &step[step
        .find("exl3_merge: Option<SharedMerge>,")
        .expect("field")..];
    let comment = &step[..step.len() - field.len()];
    let doc = comment.rsplit("\n\n").next().unwrap_or(comment);
    assert!(doc.contains("graph drops first"), "{doc}");
}

// ---- the executor takes its rows, ids and counts from the plan ----

#[test]
fn the_executor_uses_the_plan_for_ids_rows_swiglu_and_the_combine() {
    let merge = include_str!("cuda/exl3_moe/merge.rs");
    // Which ids word: the plan's, mapped to the router ids or the zero word, nothing else.
    let group = squeeze(section(merge, "fn indexed_group("));
    for part in [
        "let(word,slot)=pair.word;",
        "letids=matchword{IdWord::Router=>router_ids,IdWord::Zero=>zero}",
        "bank.span()?.indexed_call(&io,ids,slot)",
        "weight.view()?.indexed_call(&io,ids,slot)",
    ] {
        assert!(group.contains(part), "{part}: {group}");
    }
    assert!(!group.contains("indexed_call(&io,zero"), "{group}");
    assert!(!group.contains(",0)"), "{group}");
    // The pair carries the plan's word.
    let device = squeeze(section(merge, "fn merged_on_device("));
    assert!(device.contains("word:pair.id_word()"), "{device}");
    // Rows come from the plan's spans.
    let ranges = squeeze(section(merge, "fn pair_ranges"));
    for part in [
        "letspans=pair.spans(&plan_dims(dims),r.layout);",
        "None=>norm_row",
        "Some(span)=>r.buffer.range(span.offset,span.bytes)?",
        "r.buffer.range(spans.output.offset,spans.output.bytes)",
    ] {
        assert!(ranges.contains(part), "{part}: {ranges}");
    }
    // The swiglu covers `plan.swiglu_rows` rows on both paths.
    let host = squeeze(section(merge, "fn merged_on_host("));
    for (name, body) in [("device", &device), ("host", &host)] {
        assert!(
            body.contains("swiglu(launch,r.gu,r.act,(plan.swiglu_rows,dims.inter))?;"),
            "{name}: {body}"
        );
    }
    // The combine's two terms come from the layout's spans.
    let run = squeeze(section(
        merge,
        "pub(in crate::family) fn run_exl3_moe_merged(",
    ));
    for part in [
        "let(routed,shared)=r.layout.combine_spans(&plan_dims(dims));",
        "letdown=r.buffer.range(routed.offset,routed.bytes)?;",
        "letshared=r.buffer.range(shared.offset,shared.bytes)?;",
        "combine_ranges(launch,&bufs.route,(down,shared),dims)",
    ] {
        assert!(run.contains(part), "{part}: {run}");
    }
}
