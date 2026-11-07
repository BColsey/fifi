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
            "self.run_krow_body(k, admit_pos, &mut entered)",
            "self.fail_rows(entered, error)",
            "self.krow_finish(inputs, base)",
            "RowsFailure::AfterLaunch(error)",
        ],
    );
    // The poison on a failed finish is local and unconditional.
    in_order(
        window,
        &[
            "self.state.poison();",
            "Err(RowsFailure::AfterLaunch(error))",
        ],
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
fn after_the_launch_the_report_is_read_once_judged_and_only_the_tap_reads_the_logits() {
    let source = code(KROW);
    let finish = body(&source, "krow_finish");
    in_order(
        finish,
        &[
            "self.fault_at(Fault::KRowLaunch)?;",
            "self.fault_at(Fault::KRowReadback)?;",
            "let report = self.read_report(k)?;",
            "let argmax = rows_verdict(&report.table.tokens, &status, self.widths.verdict)?;",
            "check_verify(",
            "check_shortlists(",
            "let rows = if self.krow_wants_logits() {",
            "self.read_krow_rows(k)?",
            "self.state.commit_step(k, 0)?;",
            ".mark_row_slots(stream, state, k - 1)?;",
            "runtime.draws = Some(DeviceDraws {",
        ],
    );
    // Every row's pair and the window word go to the verdict; none is dropped.
    assert!(finish.contains(&code(
        "bits: report.table.bits.iter().map(|&bits| bits as u32).collect()"
    )));
    assert!(finish.contains(&code("window: report.table.window as u32")));
    // The device verify is checked against the pairs and the window ids.
    assert!(finish.contains(&code("&report.table.tokens,\n inputs,")));
    // The logits rows are read in one place, under the tap, and nowhere else in the window: not
    // before the verdict, not in the read of the report, not on the replay.
    assert_eq!(finish.matches("read_krow_rows(").count(), 1);
    assert_eq!(
        source.matches("read_krow_rows(").count(),
        3,
        "its definition, the call of the tap, and the test hook that reads a failed window's arena"
    );
    assert!(source.matches("fnread_krow_row(").count() == 1);
    let tap = body(&source, "krow_wants_logits");
    assert!(tap.contains(&code("runtime.test.want_logits")));
    assert!(tap.contains("#[cfg(not(test))]{false}"));
    // The report is one read of the table, the verify words and the records, from the result region.
    let report = body(&source, "read_report");
    in_order(
        report,
        &[
            "let mut bytes = vec![0u8; window_read_bytes(k, with_records)];",
            ".read_at(&self.stream, runtime.layout.result.offset, &mut bytes)?;",
            "WindowReport::from_bytes(&bytes, k, with_records)",
        ],
    );
    assert_eq!(report.matches("read_at(").count(), 1, "one copy");
}

#[test]
fn the_window_tail_launches_the_device_verify_and_the_shortlist_from_the_one_row_count() {
    let source = code(KROW);
    let ops = body(&source, "run_op_rows");
    in_order(
        ops,
        &[
            "RowsOp::StatusFold => ctx.window_status_fold(s, self.tail_calls(k)?.1),",
            "RowsOp::VerifyGreedy => ctx.verify_greedy(s, self.verify_call(k)?),",
            "RowsOp::ShortlistRows if !self.shortlist_on => Ok(()),",
            "RowsOp::ShortlistRows => ctx.shortlist_rows(s, self.shortlist_call(k)?),",
        ],
    );
    let verify = body(&source, "verify_call");
    in_order(
        verify,
        &[
            "m: k,",
            "runtime.layout.ids.offset, runtime.layout.ids.bytes(k)",
            "runtime.layout.result.offset, window_report_bytes(k)",
        ],
    );
    let shortlist = body(&source, "shortlist_call");
    in_order(
        shortlist,
        &[
            "m: k,",
            "n: self.widths.verdict,",
            "logits: self.kview(Slot::Logits)?,",
            "scratch: self.kview(Slot::ArgmaxPart)?,",
            "scratch_stride: ARGMAX_PART_BYTES,",
            "runtime.layout.result.offset + window_shortlist_offset(k),",
            "k * SHORTLIST_RECORD_BYTES,",
        ],
    );
}

#[test]
fn a_draw_reads_one_row_only_when_the_record_cannot_serve_it_and_a_failed_read_fails_the_window() {
    let source = code(KROW);
    let draw = body(&source, "draw_from_krow_window");
    in_order(
        draw,
        &[
            "WindowSave::Device",
            "draws.belongs_to(window.base, &window.inputs)",
            "draw_source(",
            "DrawSource::Shortlist =>",
            "let drawn = sampler.sample_shortlist(candidates, sampler_vocab);",
            "if drawn.is_err() {",
            "window_logic::fail_window(self);",
            "DrawSource::Row =>",
            "window.rows.get(row)",
            "self.fault_at(Fault::KRowDraw)",
            "self.read_krow_row(row)",
        ],
    );
    // A failed draw closes the window on all three routes: the sampler refusing a record, the
    // injected fault of the row read, and the real error of the row read.
    assert_eq!(draw.matches("window_logic::fail_window(self);").count(), 3);
    // A single row, not the window's rows.
    assert!(body(&source, "read_krow_row").contains(&code("row * row_bytes")));
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
    assert!(ops.contains(&code(
        "RowsOp::ArgmaxRows => ctx.argmax_rows(s, self.tail_calls(k)?.0)"
    )));
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
    assert!(
        body(&source, "moe_rows")
            .contains(&code("run_exl3_moe_rows(&launch, weights, &bufs, &dims)"))
    );
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

#[test]
fn the_window_runs_in_the_mode_of_its_own_length_and_a_launch_is_judged_after_the_launch() {
    let source = code(KROW);
    let run = body(&source, "run_krow_body");
    in_order(
        run,
        &[
            "self.krow_mode(k)?",
            "Mode::Replay => self.replay_krow(k, entered, false)",
            "Mode::Capture => self.capture_krow(k, admit_pos, entered)",
            "Mode::Eager => self",
            ".run_body_rows(k, admit_pos, entered)",
        ],
    );
    // The mode comes from the table of the window's own length.
    let mode = body(&source, "krow_mode");
    assert!(mode.contains(&code("self.krow_graphs.mode(k, self.capture_on)")));
    // A graph launch counts as a window past its first op, and the deliberate break of the replay
    // comes before the launch, so it changes a device word after the capture.
    let replay = body(&source, "replay_krow");
    in_order(
        replay,
        &[
            "*entered = ENTERED_BY_GRAPH;",
            "break_one_id(runtime, &self.stream, \"ids-replay\")?;",
            ".graph(k)?",
            "graph.launch(stream)?;",
            "krow_graphs.note_replay(k);",
        ],
    );
    // The capture: the arena is held and pinned, the body is the K-row list of `k` rows, a refusal
    // falls back to the same list run eagerly, and the graph is stored under `k` and launched.
    let capture = body(&source, "capture_krow");
    in_order(
        capture,
        &[
            "krow.arena.hold()",
            "self.context.capture(",
            "pins: self.krow_pins,",
            "|| self.run_body_rows(k, admit_pos, entered)",
            "fold_capture(",
            "self.krow_graphs.refused_mut()",
            "self.krow_graphs.store(k, graph, facts)?;",
            "self.replay_krow(k, entered, true)",
            "self.run_body_rows(k, admit_pos, entered)?;",
            "self.krow_graphs.note_eager(k);",
        ],
    );
    // An eager body that was enqueued whole warms its length, and nothing else does: the warm
    // flag is not set on the replay path or by a window that failed in its body.
    assert_eq!(run.matches("note_eager(k)").count(), 1);
    assert_eq!(capture.matches("note_eager(k)").count(), 1);
    assert!(!body(&source, "replay_krow").contains("note_eager"));
}

#[test]
fn arming_leaks_the_pins_of_the_graphs_with_the_k_row_arena_before_the_state_is_published() {
    let source = code(KROW);
    let enable = body(&source, "enable_krow");
    in_order(
        enable,
        &[
            "self.pins.to_vec()",
            "pinned.push(krow.arena.buffer());",
            "self.krow_pins = Box::leak(pinned.into_boxed_slice());",
            "self.krow = Some(KrowRuntime {",
        ],
    );
}
