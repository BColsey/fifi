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

/// The text of the function that starts at `signature`, up to the closing brace at `indent`.
fn body<'a>(source: &'a str, signature: &str, indent: &str) -> &'a str {
    let start = source
        .find(signature)
        .unwrap_or_else(|| panic!("`{signature}` not found"));
    let rest = &source[start..];
    let end = rest
        .find(&format!("\n{indent}}}\n"))
        .unwrap_or_else(|| panic!("the end of `{signature}` not found"));
    &rest[..end]
}

/// `first` appears in `text`, and `then` appears after it.
fn in_order(text: &str, first: &str, then: &str) {
    let at = text
        .find(first)
        .unwrap_or_else(|| panic!("`{first}` missing in:\n{text}"));
    let after = text[at..]
        .find(then)
        .unwrap_or_else(|| panic!("`{then}` does not follow `{first}` in:\n{text}"));
    assert!(after > 0);
}

#[test]
fn the_decode_step_asks_the_gate_before_it_touches_the_state() {
    let text = body(STEP, "    fn run_token_inner(", "    ");
    in_order(text, "check_token_id(", "window_logic::step_gate(self)?;");
    in_order(text, "window_logic::step_gate(self)?;", "self.write_start(");
    in_order(
        text,
        "window_logic::step_gate(self)?;",
        "self.readback_plan(",
    );
}

#[test]
fn the_export_asks_the_gate_before_it_copies() {
    let text = body(STEP, "    pub(crate) fn export_state(", "    ");
    in_order(
        text,
        "window_logic::export_gate(self)?;",
        "self.state.export_state(",
    );
}

#[test]
fn the_chunk_of_a_prefill_asks_state_usable_first() {
    let text = body(STEP, "    pub(super) fn run_chunk(", "    ");
    in_order(text, "self.state_usable()?;", "check_chunk_start(");
}

#[test]
fn import_state_is_the_window_modules_sequence() {
    let text = body(STEP, "    pub(crate) fn import_state(", "    ");
    assert!(
        text.contains("window_logic::import_state(self, exported)"),
        "{text}"
    );
    // Inside it, the gate and the window close precede the first fallible call.
    let text = body(WINDOW, "pub(crate) fn import_state<", "");
    in_order(text, "restore_gate(host)?;", "host.complete_prefix_save()");
    in_order(
        text,
        "host.complete_prefix_save()",
        "host.check_import(saved)",
    );
    in_order(text, "host.check_import(saved)", "host.resident_kv()");
    in_order(text, "host.resident_kv()", "host.overwrite(saved)");
}

#[test]
fn the_pinned_prefix_restore_closes_the_window_before_its_first_fallible_call() {
    let text = body(PREFIX, "    pub(crate) fn restore_pinned_prefix(", "    ");
    in_order(
        text,
        "window_logic::restore_gate(self)?;",
        "self.complete_prefix_save()?;",
    );
    in_order(text, "self.complete_prefix_save()?;", ".restore(");
}

#[test]
fn reset_is_exempt_and_clears_the_gate_after_the_state_was_zeroed() {
    let text = body(STEP, "    pub(crate) fn reset_state(", "    ");
    in_order(
        text,
        "window_logic::reset_prologue(self)?;",
        "self.state.reset(",
    );
    in_order(
        text,
        "self.state.reset(",
        "window_logic::reset_epilogue(self);",
    );
    let code = text
        .lines()
        .filter(|line| !line.trim_start().starts_with("//"));
    assert!(
        !code.into_iter().any(|line| line.contains("state_usable")),
        "reset must not ask state_usable:\n{text}"
    );
}

#[test]
fn the_pinned_save_and_its_device_tip_capture_each_ask_state_usable() {
    let text = body(PREFIX, "    pub(crate) fn stage_pinned_prefix(", "    ");
    in_order(
        text,
        "self.state_usable()?;",
        "session.prepare_prefix_save(",
    );
    in_order(
        text,
        "self.state_usable()?;\n        if let Some(device) = self.device_tip.as_mut() {",
        "device.capture(",
    );
}

#[test]
fn the_finish_of_a_pending_save_asks_state_usable_before_it_takes_the_save() {
    let text = body(PREFIX, "    pub(crate) fn complete_prefix_save(", "    ");
    in_order(text, "self.state_usable()?;", "self.pending_prefix.take()");
}

#[test]
fn the_device_poison_route_clears_the_tip_and_the_slot_copy_poisons() {
    // `fail_window` poisons through `ResidentState::poison`, which drops the resident tip.
    let text = body(STATE, "    pub(crate) fn poison(", "    ");
    assert!(text.contains("self.resident_tip = None;"), "{text}");
    assert!(text.contains("self.book.poison();"), "{text}");
    // The slot copy poisons on a guard failure, a verdict mismatch and a failed copy.
    let text = body(SLOTS, "    pub(crate) fn finish(", "    ");
    assert_eq!(text.matches("state.poison();").count(), 3, "{text}");
}

#[test]
fn the_window_forwards_its_effects_to_the_module_of_decisions() {
    let text = include_str!("cuda/eager_window.rs");
    assert!(text.contains("window_logic::run_window(self, inputs)"));
    assert!(text.contains("window_logic::commit_window(self, commit)"));
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
        let live = &source[..cut];
        for call in [".begin_import(", ".begin_reset("] {
            for _ in live.matches(call) {
                let relative = path.strip_prefix(&root).expect("under root");
                sites.push(format!(
                    "{} {}",
                    relative.display(),
                    call.trim_matches(['.', '('])
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
