//! Deletion detectors for the K-row window of the step (`cuda/krow_step.rs`): which failures are
//! the state's to judge and which are failures after the launch, the order of the effects after the
//! launch, and the call order of the body. They read the source and compare normalised code, like
//! `window_pin_tests`: they detect a deleted or moved call, they are not behaviour tests. The
//! behaviour of the rules is tested on the fake model (`window_logic_krow_tests`) and the pure code
//! (`krow_tests`).

const KROW: &str = include_str!("cuda/krow_step.rs");

fn code(source: &str) -> String {
    // Comments and blank space are removed; string literals stay (the pins name no string).
    let mut out = String::new();
    for line in source.lines() {
        let line = line.split("//").next().unwrap_or("");
        out.extend(line.chars().filter(|c| !c.is_whitespace()));
    }
    out
}

/// The body of `fn name` in `code` (braces balanced from the first `{` after the name).
fn body<'a>(code: &'a str, name: &str) -> &'a str {
    let start = ["(", "<"]
        .iter()
        .filter_map(|next| code.find(&format!("fn{name}{next}")))
        .min()
        .unwrap_or_else(|| panic!("`fn {name}` not found"));
    let open = code[start..].find('{').expect("a body") + start;
    let mut depth = 0usize;
    for (offset, c) in code[open..].char_indices() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return &code[open + 1..open + offset];
                }
            }
            _ => {}
        }
    }
    panic!("unbalanced braces in `{name}`")
}

fn at(body: &str, needle: &str) -> usize {
    body.find(&code(needle))
        .unwrap_or_else(|| panic!("`{needle}` not found in:\n{body}"))
}

fn in_order(body: &str, needles: &[&str]) {
    let offsets: Vec<usize> = needles.iter().map(|needle| at(body, needle)).collect();
    assert!(
        offsets.windows(2).all(|pair| pair[0] < pair[1]),
        "{needles:?} are not in this order in:\n{body}"
    );
}

#[test]
fn a_refusal_before_the_first_write_is_the_states_to_judge_and_nothing_after_the_launch_is() {
    let source = code(KROW);
    let window = body(&source, "run_krow_window");
    in_order(
        window,
        &[
            "self.krow_prelaunch(inputs).map_err(RowsFailure::StateDecides)?;",
            "self.krow_start(inputs).map_err(RowsFailure::StateDecides)?;",
            "self.run_body_rows(k, admit_pos, &mut entered)",
            "self.fail_rows(entered, error)",
            "self.krow_finish(k)",
            "RowsFailure::AfterLaunch(error)",
        ],
    );
    // The poison on a failed finish is local and unconditional.
    in_order(
        window,
        &["self.state.poison();", "Err(RowsFailure::AfterLaunch(error))"],
    );
    // `StateDecides` appears only for the two steps before the launch.
    assert_eq!(window.matches("StateDecides").count(), 2);
    let rule = body(&source, "fail_rows");
    in_order(
        rule,
        &[
            "FailureAction::Cancel => match self.state.cancel_enqueue()",
            "FailureAction::Poison => {",
            "self.state.poison();",
            "RowsFailure::AfterLaunch(error)",
        ],
    );
}

#[test]
fn the_prelaunch_checks_come_before_any_write_and_the_start_writes_ids_then_the_start_block() {
    let source = code(KROW);
    let pre = body(&source, "krow_prelaunch");
    in_order(
        pre,
        &["check_rows(", "self.refusal_at(Fault::KRowRefusal)?;"],
    );
    assert!(pre.contains(&code("merged_block: self.exl3_merge.is_some()")));
    assert!(!pre.contains("write_at") && !pre.contains("write_step_params"));
    let start = body(&source, "krow_start");
    in_order(
        start,
        &[
            "self.started = true;",
            ".write_at(stream, runtime.layout.ids.offset, &ids)?;",
            "state.write_step_params(stream, &StepParams::new(first, pos)?)?;",
            "state.enqueue_rows(inputs.len())",
        ],
    );
}

#[test]
fn after_the_launch_the_table_is_read_judged_and_the_logits_read_before_the_rows_commit() {
    let source = code(KROW);
    let finish = body(&source, "krow_finish");
    in_order(
        finish,
        &[
            "self.fault_at(Fault::KRowLaunch)?;",
            "self.fault_at(Fault::KRowReadback)?;",
            "let table = self.read_result_table(k)?;",
            "let argmax = rows_verdict(&table.tokens, &status, self.widths.verdict)?;",
            "let rows = self.read_krow_rows(k)?;",
            "self.state.commit_step(k, 0)?;",
            ".mark_row_slots(stream, state, k - 1)?;",
        ],
    );
    // Every row's pair and the window word go to the verdict; none is dropped.
    assert!(finish.contains(&code("bits: table.bits.iter().map(|&bits| bits as u32).collect()")));
    assert!(finish.contains(&code("window: table.window as u32")));
}

#[test]
fn the_body_is_the_k_row_list_and_the_tail_is_built_from_one_row_count() {
    let source = code(KROW);
    let run = body(&source, "run_body_rows");
    assert!(run.contains(&code("token_ops_rows(self.res.table.len(), k)?")));
    let ops = body(&source, "run_op_rows");
    // The window starts by clearing the status word, so no bit of an earlier window is in this
    // window's word.
    in_order(
        ops,
        &[
            "RowsOp::Step(StepOp::Begin) => {",
            "ctx.step_begin(s, params)?;",
            "ctx.exl3_flag_clear(s, self.kview(Slot::Exl3Flag)?)",
        ],
    );
    assert!(ops.contains(&code("RowsOp::ArgmaxRows => ctx.argmax_rows(s, self.tail_calls(k)?.0)")));
    assert!(ops.contains(&code(
        "RowsOp::StatusFold => ctx.window_status_fold(s, self.tail_calls(k)?.1)"
    )));
    let tail = body(&source, "tail_calls");
    in_order(
        tail,
        &[
            "m: k,",
            "result: runtime.io.range(table, k * RESULT_PAIR_BYTES)?",
            "check_tail_rows(argmax.m, fold.m)?;",
        ],
    );
    // The embedding reads the window ids from the K-row buffer, not the start block.
    let embed = body(&source, "embed_rows");
    assert!(embed.contains(&code("runtime.layout.ids.offset")));
}

#[test]
fn the_slot_stores_and_the_head_are_the_k_row_launches_and_the_per_row_chains_are_the_moe_blocks() {
    let source = code(KROW);
    let block = body(&source, "block_op_rows");
    for call in [
        "ctx.gdn_conv_snap(",
        "ctx.gdn_delta_snap(",
        "ctx.router_logits_rows_strided(",
        "ctx.pack_rows(",
        "ctx.attn_prep(",
        "ctx.attn_decode(",
        "ctx.exl3_linear(",
    ] {
        assert!(block.contains(&code(call)), "{call}");
    }
    assert!(body(&source, "moe_rows").contains(&code("run_exl3_moe_rows(&launch, weights, &bufs, &dims)")));
    assert!(body(&source, "lm_head_rows").contains(&code("rows: k,")));
}

#[test]
fn arming_checks_the_krow_rules_and_the_configuration_before_it_allocates() {
    let source = code(KROW);
    let enable = body(&source, "enable_krow");
    in_order(
        enable,
        &[
            "self.check_krow_arming(rows)?;",
            "self.check_krow_configuration()?;",
            "RowsKindOps::new(&res.geometry)",
            "KrowIoLayout::new(&res.geometry, rows)?",
            "self.context.allocate(layout.total)?",
            "self.krow = Some(KrowRuntime {",
        ],
    );
}
