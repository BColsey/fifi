//! The default path is the path of main. K-row work (`draft_verify = krow`) must not change a byte of
//! what the eager and host windows and the one-row step do, so the bodies of their functions are
//! pinned here by hash. The pin reads the source like `krow_step_pin_tests`: comments and white space
//! are removed, the body between the braces of `fn <name>` is hashed with SHA-256, and the hash is
//! the one of the same body at main `12a06a7da` (the table is the same normalization applied to
//! `git show 12a06a7da:<file>`). A failing line is a changed default-path function: either a mistake of
//! the change under review or a deliberate edit, which then updates the table in its own commit
//! and says why.
//!
//! The one function of the default path that the K-row stage edits is `draw` of the step, which
//! gains one call that is taken only for an open K-row window; its pin below checks that the greedy
//! return comes first, that the host rows are still drawn as before, and that the call is the only
//! addition.

use sha2::{Digest, Sha256};

const WINDOW_LOGIC: &str = include_str!("window_logic.rs");
const STEP: &str = include_str!("cuda/step.rs");
const EAGER_WINDOW: &str = include_str!("cuda/eager_window.rs");

fn code(source: &str) -> String {
    let mut out = String::new();
    for line in source.lines() {
        let line = line.split("//").next().unwrap_or("");
        out.extend(line.chars().filter(|c| !c.is_whitespace()));
    }
    out
}

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

/// (file, function, SHA-256 of the normalized body at main 12a06a7da).
const PINNED: &[(&str, &str, &str)] = &[
    (
        "window_logic.rs",
        "run_window",
        "a8cffb75452336aae40bae64a9b3808e97aeb95e88e1e7e87a68192d54cb3a6b",
    ),
    (
        "window_logic.rs",
        "commit_window",
        "f949fac2d9b5778cb6809ab18e91cce079855b5cc1527a48b05a73fdbee45dfd",
    ),
    (
        "window_logic.rs",
        "eager_rows",
        "113020d19cb9d45ff59bf8c9753cbb0bf7012fef6f9e3cfcefb512e16162f8c7",
    ),
    (
        "window_logic.rs",
        "run_window_device",
        "24da29cd1c9e0fbdffd9d3ff6e8bbb22457d5051964f6482de528f141b006823",
    ),
    (
        "window_logic.rs",
        "commit_window_device",
        "39a62bcd44a1998c352f6bb0022b8431b1a48d9d6795609c61d7e4b4c36e843c",
    ),
    (
        "window_logic.rs",
        "fail_window",
        "f2a2285de405df35f16d94ce82c8543cdc603503500c3abe95333b418e85094c",
    ),
    (
        "window_logic.rs",
        "state_usable",
        "e212807aa4a474a80d8326b81e27ab35f5d2bbea6363a0d79727d3ce4954f2f3",
    ),
    (
        "window_logic.rs",
        "step_gate",
        "a49899e2b66d20c7c8ffab7f13887cb854fd054d02ca61015de5f4b7fd3b62da",
    ),
    (
        "window_logic.rs",
        "restore_gate",
        "b55b17d2781bcc47d384f879d3d02efa52f562a2e2dea93d8ed04a783e9b5134",
    ),
    (
        "window_logic.rs",
        "import_state",
        "845b8dd9168123bc39632c0eb1f80aa7f9dc72a4d7858ced0f40edb738975e71",
    ),
    (
        "cuda/step.rs",
        "try_device_shortlist_token",
        "46b29a773f91d5b4a9bed3ecbc0d368be09cc7753f2d8f105887116a0ebf5489",
    ),
    (
        "cuda/step.rs",
        "shortlist_values",
        "6b6ae4ddadcfea2d152093560cfe151fe766ed920251afe13c4be1b0f9a49f81",
    ),
    (
        "cuda/step.rs",
        "read_back",
        "4590c1e147ff4185a97e0be88ce9ee19589fead89ab84ccaac94666731783b9d",
    ),
    (
        "cuda/step.rs",
        "logits_values",
        "09c6dcdfca5b85fa75d60f987454cc002d1693b0b24c928dff2a3bda03a43974",
    ),
    (
        "cuda/eager_window.rs",
        "run_window",
        "3a4e149aa7ae9f01d6263a1d5f3c81908ac7175a5074b76c7813aa10c3fd2960",
    ),
    (
        "cuda/eager_window.rs",
        "commit_window",
        "69fcb2de9f7c6412a4037918b2911e9d86816cd7f11996794edbd65883476f64",
    ),
    (
        "cuda/eager_window.rs",
        "export",
        "cf2a8568148c2c19645d66096d87e0fd54b74b8783744208453c7fe26322e2d6",
    ),
    (
        "cuda/eager_window.rs",
        "run_row",
        "b5e20e06e09c22ff83d44d74eb7a2d75815e6edee777d7faae08e956ac81e788",
    ),
    (
        "cuda/eager_window.rs",
        "replay_row",
        "b408811ab19afb2a7fc981f00f83a85ab7d6cf8db1a64feb83ea934f894f625e",
    ),
    (
        "cuda/eager_window.rs",
        "overwrite",
        "cb75923a69b0ad1c093c04fa33679dd6b5688490f152df2cfc5466a482d6a054",
    ),
    (
        "cuda/eager_window.rs",
        "capture",
        "08097e01876e6d01ba69ede7455059b8eb89fad941cf4ac91a29d7567bee4d37",
    ),
    (
        "cuda/eager_window.rs",
        "run_rows",
        "0a95fd8362bc599f906dee55ab942ef0f0e296ffd4b378ffd2d0994958b8f84b",
    ),
    (
        "cuda/eager_window.rs",
        "rollback_finish",
        "7c8b97b97c6667bd7755637661fe36268f1662d114a6e2533763786bb4b55ad0",
    ),
];

fn source_of(file: &str) -> &'static str {
    match file {
        "window_logic.rs" => WINDOW_LOGIC,
        "cuda/step.rs" => STEP,
        "cuda/eager_window.rs" => EAGER_WINDOW,
        other => panic!("no source for {other}"),
    }
}

#[test]
fn the_functions_of_the_eager_and_host_windows_and_the_one_row_shortlist_are_those_of_main() {
    for &(file, name, want) in PINNED {
        let source = code(source_of(file));
        let got = format!("{:x}", Sha256::digest(body(&source, name).as_bytes()));
        assert_eq!(
            got, want,
            "{file}: fn {name} is not the body of main 12a06a7da"
        );
    }
}

#[test]
fn the_step_draw_gains_one_call_and_keeps_the_greedy_return_and_the_host_rows_as_they_were() {
    let source = code(EAGER_WINDOW);
    let draw = body(&source, "draw");
    let at = |needle: &str| {
        draw.find(&code(needle))
            .unwrap_or_else(|| panic!("`{needle}` not found in the draw"))
    };
    // The greedy request still answers from the device argmax, first.
    let greedy = at("if sampler.is_greedy() {
            return window.argmax.get(row).copied().ok_or_else(no_window);
        }");
    // The addition: one call, and only that call, before the host rows.
    let added = at(
        "if let Some(drawn) = self.draw_from_krow_window(row, sampler, sampler_vocab) {
            return drawn;
        }",
    );
    // The host rows of an eager or host window, as before.
    let host = at("let logits = window.rows.get(row).ok_or_else(no_window)?;
        sample_logits(Self::OPERATION, logits, sampler, sampler_vocab)");
    assert!(greedy < added && added < host);
    assert_eq!(draw.matches("draw_from_krow_window").count(), 1);
    // Nothing else of the K-row stage is reachable from the default draw.
    assert!(
        !draw.replace("draw_from_krow_window", "").contains("krow"),
        "{draw}"
    );
}

/// The files that hold the default path name no K-row item beyond the one call above: the K-row
/// stage lives in its own files and in `krow.rs`.
#[test]
fn the_window_logic_and_the_one_row_step_know_nothing_of_the_k_row_stage() {
    for (name, source) in [("window_logic.rs", WINDOW_LOGIC), ("cuda/step.rs", STEP)] {
        let source = code(source);
        for item in [
            "verify_greedy",
            "shortlist_rows",
            "DeviceDraws",
            "draw_from_krow_window",
            "read_report",
        ] {
            assert!(!source.contains(item), "{name} mentions {item}");
        }
    }
}
