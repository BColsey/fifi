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
