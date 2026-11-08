//! Deletion detectors for the glue that stays outside `window_logic` (the step's forwards and
//! the call order inside its wrappers), and the completeness count of the restoring entry points.
//!
//! The pin tests read the source with `include_str!` and assert that the required calls are
//! present and in order inside named function bodies. They detect a deleted or moved call; they
//! are NOT behaviour tests, and a reformat that changes the text of a pinned call fails them. The
//! behaviour of every decision is tested through the `WindowHost` seam in `window_logic_tests`.

const STEP: &str = include_str!("cuda/step.rs");
const PREFIX: &str = include_str!("cuda/prefix_runtime.rs");
const WINDOW: &str = include_str!("window_logic.rs");
const STATE: &str = include_str!("cuda/state.rs");
const SLOTS: &str = include_str!("cuda/state/rollback.rs");
const START: &str = include_str!("cuda/start.rs");
const RUNTIME: &str = include_str!("cuda/rollback_runtime.rs");

/// The code of `source` as the pins compare it: comments removed, string and character literals
/// emptied, raw strings emptied, and every whitespace character dropped. A commented-out call
/// vanishes, a call inside a string is not a call, and a reformat that only moves whitespace
/// (rustfmt splitting `f(x)?` over lines) leaves the text unchanged.
fn code(source: &str) -> String {
    let chars: Vec<char> = source.chars().collect();
    let mut out = String::new();
    let mut at = 0;
    let ident = |c: char| c.is_alphanumeric() || c == '_';
    while at < chars.len() {
        let c = chars[at];
        let next = chars.get(at + 1).copied();
        if c == '/' && next == Some('/') {
            while at < chars.len() && chars[at] != '\n' {
                at += 1;
            }
        } else if c == '/' && next == Some('*') {
            at += 2;
            while at + 1 < chars.len() && !(chars[at] == '*' && chars[at + 1] == '/') {
                at += 1;
            }
            at += 2;
        } else if c == 'r' && matches!(next, Some('"' | '#')) && !(at > 0 && ident(chars[at - 1])) {
            // A raw string: r"..." or r#"..."#.
            let mut end = at + 1;
            let mut hashes = 0;
            while chars.get(end) == Some(&'#') {
                hashes += 1;
                end += 1;
            }
            if chars.get(end) != Some(&'"') {
                out.push(c);
                at += 1;
                continue;
            }
            end += 1;
            while end < chars.len() {
                if chars[end] == '"' && (1..=hashes).all(|k| chars.get(end + k) == Some(&'#')) {
                    end += hashes + 1;
                    break;
                }
                end += 1;
            }
            out.push_str("\"\"");
            at = end;
        } else if c == '"' {
            out.push_str("\"\"");
            at += 1;
            while at < chars.len() && chars[at] != '"' {
                at += if chars[at] == '\\' { 2 } else { 1 };
            }
            at += 1;
        } else if c == '\'' && (next == Some('\\') || chars.get(at + 2) == Some(&'\'')) {
            // A character literal ('x' or '\n'), not a lifetime.
            out.push_str("''");
            at += 1;
            while at < chars.len() && chars[at] != '\'' {
                at += if chars[at] == '\\' { 2 } else { 1 };
            }
            at += 1;
        } else {
            if !c.is_whitespace() {
                out.push(c);
            }
            at += 1;
        }
    }
    // A trailing comma before a closing bracket is formatting, not meaning.
    for close in [")", "]", "}"] {
        out = out.replace(&format!(",{close}"), close);
    }
    out
}

/// The body of the function named `name` in `code` (already through [`code`]), without its outer
/// braces. The first `fn name(` is taken.
fn body<'a>(code: &'a str, name: &str) -> &'a str {
    let start = ["(", "<"]
        .iter()
        .filter_map(|next| code.find(&format!("fn{name}{next}")))
        .min()
        .unwrap_or_else(|| panic!("`fn {name}` not found"));
    let open = code[start..]
        .find('{')
        .unwrap_or_else(|| panic!("no body for `{name}`"))
        + start;
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

/// The brace depth of every byte of `text`.
fn depths(text: &str) -> Vec<usize> {
    let mut depth = 0usize;
    text.bytes()
        .map(|b| {
            match b {
                b'{' => depth += 1,
                b'}' => depth = depth.saturating_sub(1),
                _ => {}
            }
            depth
        })
        .collect()
}

/// Every offset of `needle` that is a STATEMENT of the function body: at brace depth zero (not
/// inside an `if`, a closure, a loop or a block, so a dead branch such as `if false { call }`
/// does not count) and starting a statement (after `;`, `{` or `}`, or at the start). `needle` is
/// normalised like the code.
fn statements(body: &str, needle: &str) -> Vec<usize> {
    let needle = code(needle);
    let depth = depths(body);
    body.match_indices(&needle)
        .map(|(at, _)| at)
        .filter(|&at| {
            let starts = at == 0 || matches!(body.as_bytes()[at - 1], b';' | b'{' | b'}');
            depth[at] == 0 && starts
        })
        .collect()
}

/// The offset of the first top-level statement `needle`; panics when there is none.
fn statement(body: &str, needle: &str) -> usize {
    *statements(body, needle)
        .first()
        .unwrap_or_else(|| panic!("`{needle}` is not a top-level statement of:\n{body}"))
}

/// `first` and `then` are both top-level statements, in that order.
fn in_order(body: &str, first: &str, then: &str) {
    let (a, b) = (statement(body, first), statement(body, then));
    assert!(a < b, "`{first}` must precede `{then}` in:\n{body}");
}

/// `needle` occurs in `body` after offset `after` (anywhere: it may be part of a larger
/// expression), as normalised code.
fn occurs_after(body: &str, after: usize, needle: &str) {
    let needle = code(needle);
    assert!(
        body[after..].contains(&needle),
        "`{needle}` does not follow offset {after} in:\n{body}"
    );
}

#[test]
fn the_matcher_rejects_a_comment_a_dead_branch_a_closure_and_a_string_and_accepts_a_reformat() {
    let ok = code("fn f(&mut self) { a(); window_logic::step_gate(self)?; b(); }");
    statement(body(&ok, "f"), "window_logic::step_gate(self)?;");
    // Equivalent reformatting keeps the pin.
    let split = code("fn f() {\n    window_logic::step_gate(\n        self,\n    )?;\n}");
    statement(body(&split, "f"), "window_logic::step_gate(self)?;");
    for (what, wrong) in [
        (
            "a line comment",
            "fn f() { // window_logic::step_gate(self)?;\n a(); }",
        ),
        (
            "a block comment",
            "fn f() { /* window_logic::step_gate(self)?; */ a(); }",
        ),
        (
            "a dead branch",
            "fn f() { if false { window_logic::step_gate(self)?; } a(); }",
        ),
        (
            "a closure",
            "fn f() { let c = || { window_logic::step_gate(self)?; }; a(); }",
        ),
        (
            "a string",
            "fn f() { log(\"window_logic::step_gate(self)?;\"); a(); }",
        ),
        (
            "a non-statement",
            "fn f() { let _ = a(window_logic::step_gate(self)?); }",
        ),
    ] {
        let wrong = code(wrong);
        let result = std::panic::catch_unwind(|| {
            statement(body(&wrong, "f"), "window_logic::step_gate(self)?;");
        });
        assert!(result.is_err(), "{what} must fail the pin");
    }
    // The order is checked between statements, not between mentions.
    let swapped = code("fn f() { b(); window_logic::step_gate(self)?; a(); }");
    let result = std::panic::catch_unwind(|| {
        in_order(
            body(&swapped, "f"),
            "a();",
            "window_logic::step_gate(self)?;",
        );
    });
    assert!(result.is_err());
}

#[test]
fn the_decode_step_asks_the_gate_before_it_touches_the_state() {
    let step = code(STEP);
    let text = body(&step, "run_token_inner");
    in_order(
        text,
        "check_token_id(token, self.widths.verdict)?;",
        "window_logic::step_gate(self)?;",
    );
    let gate = statement(text, "window_logic::step_gate(self)?;");
    occurs_after(text, gate, "self.write_start(");
    occurs_after(text, gate, "self.readback_plan(");
}

#[test]
fn the_export_asks_the_gate_before_it_copies() {
    let step = code(STEP);
    let text = body(&step, "export_state");
    let gate = statement(text, "window_logic::export_gate(self)?;");
    occurs_after(text, gate, "self.state.export_state(");
}

#[test]
fn the_chunk_of_a_prefill_asks_state_usable_first() {
    let step = code(STEP);
    let text = body(&step, "run_chunk");
    let gate = statement(text, "self.state_usable()?;");
    occurs_after(text, gate, "check_chunk_start(");
    // First: nothing but declarations (`use`) precede it.
    assert!(
        !text[..gate].contains("self.state"),
        "no state access before the gate: {}",
        &text[..gate]
    );
}

#[test]
fn import_state_is_the_window_modules_sequence() {
    let step = code(STEP);
    let text = body(&step, "import_state");
    statement(text, "window_logic::import_state(self, exported)");
    // Inside it, the gate and the window close precede the first fallible call.
    let window = code(WINDOW);
    let text = body(&window, "import_state");
    in_order(
        text,
        "restore_gate(host)?;",
        "host.commit_fault(CommitPoint::PrefixSave)",
    );
    let save = statement(text, "host.commit_fault(CommitPoint::PrefixSave)");
    occurs_after(text, save, "host.complete_prefix_save()");
    let check = statement(text, "host.commit_fault(CommitPoint::CheckImport)");
    occurs_after(text, check, "host.check_import(saved)");
    let kv = statement(text, "host.commit_fault(CommitPoint::ResidentKv)");
    occurs_after(text, kv, "host.resident_kv()");
    in_order(
        text,
        "host.commit_fault(CommitPoint::ResidentKv)",
        "host.commit_fault(CommitPoint::ImportDevice)",
    );
    let write = statement(text, "host.commit_fault(CommitPoint::ImportDevice)");
    occurs_after(text, write, "host.overwrite(saved)");
}

#[test]
fn the_pinned_prefix_restore_closes_the_window_before_its_first_fallible_call() {
    let prefix = code(PREFIX);
    let text = body(&prefix, "restore_pinned_prefix");
    in_order(
        text,
        "window_logic::restore_gate(self)?;",
        "self.complete_prefix_save()?;",
    );
    let save = statement(text, "self.complete_prefix_save()?;");
    occurs_after(text, save, ".restore(");
}

#[test]
fn reset_is_exempt_and_clears_the_gate_after_the_state_was_zeroed() {
    let step = code(STEP);
    let text = body(&step, "reset_state");
    in_order(
        text,
        "window_logic::reset_prologue(self)?;",
        "self.state.reset(&self.stream)?;",
    );
    in_order(
        text,
        "self.state.reset(&self.stream)?;",
        "window_logic::reset_epilogue(self);",
    );
    // Comments are gone from `text`: only code can ask `state_usable`.
    assert!(
        !text.contains("state_usable"),
        "reset must not ask state_usable:\n{text}"
    );
}

#[test]
fn the_pinned_save_and_its_device_tip_capture_each_ask_state_usable() {
    let prefix = code(PREFIX);
    let text = body(&prefix, "stage_pinned_prefix");
    in_order(
        text,
        "self.state_usable()?;",
        "let frozen = session.prepare_prefix_save(key, fed)?;",
    );
    // A second, separate gate immediately before the tip capture (the tip copies the live state).
    let gates = statements(text, "self.state_usable()?;");
    assert_eq!(gates.len(), 2, "two top-level gates in:\n{text}");
    let tip = code("if let Some(device) = self.device_tip.as_mut() {");
    let tip_at = text.find(&tip).expect("the tip branch");
    assert!(
        gates[0] < gates[1] && gates[1] < tip_at,
        "the second gate precedes the tip branch"
    );
    occurs_after(text, tip_at, "device.capture(");
}

#[test]
fn the_finish_of_a_pending_save_asks_state_usable_before_it_takes_the_save() {
    let prefix = code(PREFIX);
    let text = body(&prefix, "complete_prefix_save");
    let gate = statement(text, "self.state_usable()?;");
    occurs_after(text, gate, "self.pending_prefix.take()");
}

#[test]
fn the_device_poison_route_clears_the_tip_and_the_slot_copy_poisons() {
    // `fail_window` poisons through `ResidentState::poison`, which drops the resident tip.
    let state = code(STATE);
    let text = body(&state, "poison");
    statement(text, "self.resident_tip = None;");
    statement(text, "self.book.poison();");
    // The slot copy poisons on a guard failure, a verdict mismatch and a failed copy.
    let slots = code(SLOTS);
    let text = body(&slots, "finish");
    assert_eq!(text.matches("state.poison();").count(), 3, "{text}");
}

#[test]
fn the_window_forwards_its_effects_to_the_module_of_decisions() {
    let eager = code(include_str!("cuda/eager_window.rs"));
    let run = body(&eager, "run_window");
    statement(run, "window_logic::run_window(self, inputs)");
    let commit = body(&eager, "commit_window");
    statement(commit, "window_logic::commit_window(self, commit)");
}

/// The block (without its braces) that opens after `marker` in `code`.
fn block<'a>(code: &'a str, marker: &str) -> &'a str {
    let marker = self::code(marker);
    let start = code
        .find(&marker)
        .unwrap_or_else(|| panic!("`{marker}` not found"));
    let open = start + marker.len() - 1;
    assert_eq!(
        &code[open..=open],
        "{",
        "the marker must end at the opening brace"
    );
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
    panic!("unbalanced braces after `{marker}`")
}

/// The names of the functions declared directly in `block` (at brace depth zero).
fn declared(block: &str) -> Vec<String> {
    let depth = depths(block);
    let mut names = Vec::new();
    for (at, _) in block.match_indices("fn") {
        let before_ok = at == 0 || !block.as_bytes()[at - 1].is_ascii_alphanumeric();
        let rest = &block[at + 2..];
        let end = rest
            .find(|c: char| !(c.is_alphanumeric() || c == '_'))
            .unwrap_or(rest.len());
        if before_ok && end > 0 && depth[at] == 0 && matches!(rest.as_bytes()[end], b'(' | b'<') {
            names.push(rest[..end].to_owned());
        }
    }
    names
}

/// Each `(function, statements)` row: the function exists in `block` and every statement is a
/// top-level statement of its body.
fn forwards(block: &str, table: &[(&str, &[&str])]) {
    let mut expected: Vec<&str> = table.iter().map(|(name, _)| *name).collect();
    let mut found = declared(block);
    expected.sort_unstable();
    found.sort_unstable();
    assert_eq!(
        found, expected,
        "the forwards in the source and the table of the pin differ: a new or removed forward needs a row"
    );
    for (name, statements_of) in table {
        let text = body(block, name);
        for expected in *statements_of {
            statement(text, expected);
        }
    }
}

#[test]
fn the_step_forwards_every_primitive_of_the_window_host_to_the_right_effect() {
    let eager = code(include_str!("cuda/eager_window.rs"));
    let host = block(&eager, "impl WindowHost for QwenResidentStep<'_> {");
    forwards(
        host,
        &[
            ("device_wanted", &["self.device_rollback_wanted()"]),
            ("tokens", &["QwenResidentStep::tokens(self)"]),
            ("store_window", &["self.window = Some(window);"]),
            ("take_window", &["self.window.take()"]),
            ("export", &["self.export_state()"]),
            (
                "run_row",
                &[
                    "let drawn = self.run_token(token, true)?;",
                    "Ok((drawn, self.logits_values()))",
                ],
            ),
            ("replay_row", &["self.run_token(token, false).map(|_| ())"]),
            ("overwrite", &["self.overwrite_state(saved)"]),
            ("window_awaits", &["self.window_is_open()"]),
            ("capacity", &["self.kv_capacity_rows()"]),
            ("is_valid", &["self.state_is_valid()"]),
            ("needs_reset", &["self.recovery_gate_set()"]),
            ("set_needs_reset", &["self.set_recovery_gate();"]),
            ("clear_needs_reset", &["self.clear_recovery_gate();"]),
            (
                "run_rows",
                &[
                    "if self.krow_runs(inputs.len()) { return self.run_krow_window(inputs); }",
                    "eager_rows(self, inputs).map_err(RowsFailure::StateDecides)",
                ],
            ),
            ("poison", &["self.poison_window_state();"]),
            (
                "drop_pending_prefix",
                &["QwenResidentStep::drop_pending_prefix(self);"],
            ),
            ("invalidate_pool", &["self.invalidate_prefix_pool();"]),
            ("close_window", &["self.close_stored_window();"]),
            ("check_import", &["self.check_saved_state(saved)"]),
            ("resident_kv", &["self.check_resident_kv()"]),
            ("require_armed", &["self.require_armed_for_window()"]),
            (
                "complete_prefix_save",
                &["QwenResidentStep::complete_prefix_save(self)"],
            ),
            (
                "rollback_begin",
                &["QwenResidentStep::rollback_begin(self, rows)"],
            ),
            ("capture", &["self.rollback_capture(row)?;"]),
            (
                "rollback_discard",
                &["QwenResidentStep::rollback_discard(self);"],
            ),
            (
                "rollback_window_complete",
                &["QwenResidentStep::rollback_window_complete(self)"],
            ),
            (
                "rollback_finish",
                &["QwenResidentStep::rollback_finish(self, kept, committed_rows)"],
            ),
            ("trace_mark", &["QwenResidentStep::trace_mark(self, mark);"]),
            ("feed_fault", &["QwenResidentStep::feed_fault(self, row)"]),
            (
                "replay_fault",
                &["QwenResidentStep::replay_fault(self, row)"],
            ),
            ("commit_fault", &["self.commit_point_fault(at)"]),
            (
                "capture_fault",
                &["QwenResidentStep::capture_fault(self, row)"],
            ),
            ("restore_fault", &["QwenResidentStep::restore_fault(self)"]),
        ],
    );
}

#[test]
fn the_effects_of_the_recovery_gate_on_the_step_are_what_the_window_logic_assumes() {
    let gate = code(include_str!("cuda/window_gate.rs"));
    let methods = block(&gate, "impl QwenResidentStep<'_> {");
    forwards(
        methods,
        &[
            (
                "window_is_open",
                &[
                    "self.window.is_some() || self.rollback.as_ref().is_some_and(super::super::state::rollback::RollbackSlots::awaiting_verdict)",
                ],
            ),
            ("kv_capacity_rows", &["self.state.layout().kv.capacity"]),
            ("state_is_valid", &["self.state.health() == Health::Valid"]),
            ("recovery_gate_set", &["self.needs_reset"]),
            ("set_recovery_gate", &["self.needs_reset = true;"]),
            ("clear_recovery_gate", &["self.needs_reset = false;"]),
            ("state_usable", &["window_logic::state_usable(self)"]),
            ("poison_window_state", &["self.state.poison();"]),
            ("drop_pending_prefix", &["self.pending_prefix = None;"]),
            (
                "invalidate_prefix_pool",
                &[
                    "if let Some(pool) = self.prefix_pins.as_mut() { let _exhausted = pool.invalidate(); }",
                ],
            ),
            (
                "close_stored_window",
                &["self.window = None;", "self.rollback_discard();"],
            ),
            ("check_saved_state", &["self.state.check_exported(saved)"]),
            ("check_resident_kv", &["self.state.check_resident_kv()"]),
        ],
    );
}

/// The call sites of `begin_import(` and `begin_reset(` outside tests under `qwen36/cuda`, as
/// (file, call). A new restoring site must be added to the table of the specification (17.2) and
/// to this list together with its gate and its window close.
#[test]
fn the_restoring_entry_points_are_exactly_the_six_of_the_table() {
    fn walk(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
        let mut entries: Vec<_> = std::fs::read_dir(dir)
            .expect("read_dir")
            .map(|entry| entry.expect("entry").path())
            .collect();
        entries.sort();
        for path in entries {
            if path.is_dir() {
                walk(&path, out);
            } else if path.extension().is_some_and(|ext| ext == "rs") {
                out.push(path);
            }
        }
    }
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/family/qwen36/cuda");
    let mut files = Vec::new();
    walk(&root, &mut files);
    let mut sites = Vec::new();
    for path in files {
        let name = path.file_name().and_then(|n| n.to_str()).expect("name");
        if name.ends_with("tests.rs") || name == "book.rs" {
            continue;
        }
        let source = std::fs::read_to_string(&path).expect("read");
        // An inline test module sits at the end of a file: cut at its last marker.
        let cut = ["#[cfg(test)]\nmod tests {", "#[cfg(test)]\n#[path"]
            .iter()
            .filter_map(|marker| source.rfind(marker))
            .max()
            .unwrap_or(source.len());
        // Comments and strings do not count; a method call, a path call
        // (`StateBook::begin_import(..)`) and a plain call all do. Only a definition does not.
        let live = code(&source[..cut]);
        for call in ["begin_import(", "begin_reset("] {
            for (at, _) in live.match_indices(call) {
                if live[..at].ends_with("fn") {
                    continue;
                }
                let relative = path.strip_prefix(&root).expect("under root");
                sites.push(format!(
                    "{} {}",
                    relative.display(),
                    call.trim_end_matches('(')
                ));
            }
        }
    }
    sites.sort();
    assert_eq!(
        sites,
        [
            "state.rs begin_reset",
            "state/device_tip.rs begin_import",
            "state/export.rs begin_import",
            "state/pinned.rs begin_import",
            "state/prefix.rs begin_import",
            "state/rollback.rs begin_import",
        ],
        "a restoring site was added or removed: extend the table of DESIGN 17.2 and its gate"
    );
}

// ---- the K-row seam ----

#[test]
fn the_device_window_runs_its_rows_through_run_rows_and_not_through_a_loop_of_its_own() {
    let window = code(WINDOW);
    let device = body(&window, "run_window_device");
    for gone in [
        "host.run_row(",
        "host.capture(",
        "host.feed_fault(",
        "host.capture_fault(",
    ] {
        assert!(
            !device.contains(&code(gone)),
            "`{gone}` belongs to the eager default of run_rows, not to the window"
        );
    }
    statement(
        device,
        "let RowsOut { argmax, rows } = match host.run_rows(inputs) {",
    );
    // A state-decided failure closes the book, then the old rule decides.
    let decides = block(device, "Err(RowsFailure::StateDecides(error)) => {");
    in_order(
        decides,
        "host.rollback_discard();",
        "return Err(row_failure(host, base, error));",
    );
    // A failure after the launch closes the book, then fails closed unconditionally.
    let launched = block(device, "Err(RowsFailure::AfterLaunch(error)) => {");
    in_order(
        launched,
        "host.rollback_discard();",
        "return Err(failed(host, error));",
    );
    assert!(
        !launched.contains(&code("row_failure")),
        "a failure after the launch must not go through the counters rule"
    );
    // The completion mark and the store stay in the window, after the rows, for both paths.
    let at = |needle: &str| {
        device
            .find(&code(needle))
            .unwrap_or_else(|| panic!("`{needle}`"))
    };
    assert!(at("host.run_rows(inputs)") < at("host.rollback_window_complete()"));
    assert!(at("host.rollback_window_complete()") < at("host.store_window(Window{"));
}

#[test]
fn the_default_of_run_rows_is_the_eager_loop_with_its_capture_after_every_row_but_the_last() {
    let window = code(WINDOW);
    let default = body(&window, "run_rows");
    // The eager rows report only state-decided failures: the default path keeps today's rule.
    statement(
        default,
        "eager_rows(self, inputs).map_err(RowsFailure::StateDecides)",
    );
    assert!(!default.contains(&code("AfterLaunch")));
    let eager = body(&window, "eager_rows");
    let at = |needle: &str| {
        eager
            .find(&code(needle))
            .unwrap_or_else(|| panic!("`{needle}`"))
    };
    assert!(at("host.feed_fault(at)?;") < at("host.run_row(token)?;"));
    assert!(at("host.run_row(token)?;") < at("host.trace_mark(Mark::RowDone);"));
    assert!(at("host.trace_mark(Mark::RowDone);") < at("host.capture_fault(at)?;"));
    assert!(at("host.capture_fault(at)?;") < at("host.capture(at)?;"));
    assert!(
        eager.contains(&code("if at + 1 < inputs.len() {")),
        "the last row is not captured"
    );
}

#[test]
fn the_step_runs_the_k_row_body_for_an_armed_window_and_the_eager_loop_for_every_other() {
    let eager = code(include_str!("cuda/eager_window.rs"));
    let host = block(&eager, "impl WindowHost for QwenResidentStep<'_> {");
    assert!(declared(host).iter().any(|name| name == "run_rows"));
    let rows = body(host, "run_rows");
    in_order(
        rows,
        "if self.krow_runs(inputs.len()) { return self.run_krow_window(inputs); }",
        "eager_rows(self, inputs).map_err(RowsFailure::StateDecides)",
    );
}

#[test]
fn the_start_checks_the_krow_switches_before_the_weights_and_arms_the_k_row_path_after_the_slots() {
    let start = code(START);
    let at = |needle: &str| {
        start
            .find(&code(needle))
            .unwrap_or_else(|| panic!("`{needle}`"))
    };
    assert!(
        !start.contains(&code("not_built")),
        "no start refuses a K-row session any more"
    );
    // The early block: guard, then the switches; all before the first weight read.
    assert!(at("settings.draft_verify == VerifyMode::Krow") < at("check_krow_env("));
    assert!(at("check_krow_env(") < at("load_exl3_dir_decided(&projector"));
    // The K-row arena is added to the resources only for a K-row session, before they are leaked.
    assert!(
        at("res.with_krow(") < at("let res: &'static StepResources = Box::leak(Box::new(res));")
    );
    // After the slots are armed: ask for krow, check the step, arm the path.
    let armed = at("step.enable_device_rollback(");
    assert!(armed < at("step.request_krow_verify();"));
    assert!(at("step.request_krow_verify();") < at("step.check_krow_arming("));
    assert!(at("step.check_krow_arming(") < at("step.enable_krow()?;"));
    // The load line is built after the request, so it names the block the step really runs.
    assert!(at("step.enable_krow()?;") < at("step.shared_in_routed_suffix()"));
}

/// A merged block the DEFAULT built yields to krow: dropped before the first token with the reason
/// logged and the source recorded; an explicit `1` is left for `check_krow_arming` to refuse.
#[test]
fn a_default_merged_block_yields_to_krow_and_an_explicit_one_stays_to_be_refused() {
    let runtime = code(RUNTIME);
    let request = body(&runtime, "request_krow_verify");
    for part in [
        "self.krow_wanted = true;",
        "if self.merge_intent == Intent::Default && self.exl3_merge.is_some() {",
        "self.exl3_merge = None;",
        "self.merge_source = Source::Krow;",
    ] {
        assert!(
            request.contains(&code(part)),
            "`{part}` missing from request_krow_verify"
        );
    }
    // The whole block, exactly: the drop sits INSIDE the condition on the default intent, so an
    // explicit `1` is never touched, and nothing else runs in the function.
    assert_eq!(
        request,
        code(
            "self.krow_wanted = true; \
             if self.merge_intent == Intent::Default && self.exl3_merge.is_some() { \
                 eprintln!(\"\"); \
                 self.exl3_merge = None; \
                 self.merge_source = Source::Krow; \
             }"
        ),
        "request_krow_verify is exactly: ask for krow, then drop a default-built merge"
    );
    // The drop is conditioned on the default intent only, so an explicit `1` is not touched.
    assert!(!request.contains(&code("Intent::Explicit")));
    assert!(
        request.find(&code("self.krow_wanted = true;")).unwrap()
            < request.find(&code("self.exl3_merge = None;")).unwrap()
    );
    // The reason is logged (a string literal, so it is read from the raw source).
    assert!(RUNTIME.contains("unset and draft_verify is krow, using the reference block"));
    // The step keeps the resolved intent that decision reads.
    let step = code(STEP);
    assert!(step.contains(&code("merge_intent: Intent,")));
    assert!(step.contains(&code("merge_source: Source,")));
}

#[test]
fn the_step_reads_the_merged_block_from_its_own_field_and_the_hook_meets_the_same_refusal() {
    let runtime = code(RUNTIME);
    let arming = body(&runtime, "check_krow_arming");
    statement(
        arming,
        "let single_gather = std::env::var_os(SINGLE_GATHER_NAME);",
    );
    for field in [
        "rollback_armed: self.rollback.is_some(),",
        "shared_in_routed: self.exl3_merge.is_some()",
        "single_gather: single_gather.as_deref(),",
    ] {
        assert!(
            arming.contains(&code(field)),
            "`{field}` missing from check_krow_arming"
        );
    }
    let step = code(STEP);
    let hook = body(&step, "set_shared_in_routed");
    assert!(
        hook.contains(&code(
            "if on && self.krow_wanted { crate::family::qwen36::krow::check_krow_switches(None, true)?; }"
        )),
        "the test hook must refuse the merged block on a krow step with the arming's own refusal"
    );
}

/// Where the field `name` of the struct body `fields` starts, if it is declared.
fn field_at(fields: &str, name: &str) -> Option<usize> {
    [format!(",{name}:"), format!("){name}:")]
        .iter()
        .find_map(|pattern| fields.find(&code(pattern)))
        .or_else(|| fields.starts_with(&format!("{name}:")).then_some(0))
}

#[test]
fn the_captured_graphs_are_declared_before_the_buffers_whose_addresses_they_bake_in() {
    // Fields drop in declaration order. By NAME, never by line: the lines keep moving.
    let step = code(STEP);
    let fields = block(&step, "pub(crate) struct QwenResidentStep<'a> {");
    let at = |name: &str| field_at(fields, name).unwrap_or_else(|| panic!("field `{name}`"));
    assert!(
        at("graph") < at("state"),
        "the graph drops before the state it bakes in"
    );
    assert!(
        at("graph") < at("rollback"),
        "the graph drops before the slots"
    );
    assert!(
        at("graph") < at("exl3_merge"),
        "the precedent: the merged block's buffer"
    );
    // The K-row runtime holds the buffer a captured K-row body bakes in: it sits with the graph,
    // before the state and the slots.
    assert!(at("graph") < at("krow") && at("krow") < at("state") && at("krow") < at("rollback"));
    // The per-K graphs obey the same rule as soon as they exist.
    if let Some(krow) = field_at(fields, "krow_graphs") {
        assert!(krow < at("state") && krow < at("rollback"));
    }
}
