//! Cardless guard: no new initialized `__constant__` arrays in the CUDA sources.
//!
//! The at-load constant-bank layout of this .so's multi-cubin image aliases per-TU
//! initialized `__constant__` arrays by position, not by name or linkage. Observed
//! 2026-11-18 on the 3090: the gemm_dq TU's uniquely named IQ4_NL table read nvfp4
//! kE2M1's initializer, and renaming and external linkage both still aliased. The
//! fix pattern is a BY-VALUE kernel argument (iq4_nl_values.cuh's
//! `make_iq4nl_table`); no cardless gate can see this class of bug once it is in
//! the tree, so this test keeps the tree free of new instances of it.
//!
//! Coverage: everything the build compiles as CUDA — `src/kernels/cuda/native/**`
//! (CUDA_SOURCES and the MMQ glue), `vendor/ggml-cuda-mmq/**` (the vendored
//! llama.cpp MMQ sources) and `vendor/xavier-qwen/**` (the vendored repack
//! kernels). Comments are stripped (newlines kept, so line numbers stay true)
//! and each `__constant__` is scanned to the next `;`, so an initializer any
//! distance from the declaration — brace on its own line, declares-then-
//! initializes — is still found. An initializer this test cannot read as
//! literal floats (a macro, an expression) is flagged, not skipped: what the
//! guard cannot pin, it refuses.
//!
//! The two landed kE2M1 twins stay allowed, pinned BY FILE AND BY VALUE: if one
//! twin's sixteen values change, the two tables can disagree again and this guard
//! fails that file too.
//!
//! NOT covered: `static const __device__` tables (the vendored MMQ kernels keep
//! an IQ4_NL table that way). The observed aliasing was specific to the module
//! `__constant__` bank, whose at-load layout overlaid per-TU initialized arrays
//! by position; whether per-TU `__device__` statics can alias the same way was
//! never probed, and this guard claims nothing about them either way.

/// The only files allowed an initialized `__constant__` array, with the exact
/// sixteen initializer values the file must carry.
const ALLOWED: &[(&str, [f32; 16])] = &[("nvfp4_gemv.cu", KE2M1), ("nvfp4_gemv_batch.cu", KE2M1)];

const KE2M1: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

fn cuda_source_files(root: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
    let Ok(entries) = std::fs::read_dir(root) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let is_source = path
            .extension()
            .is_some_and(|e| e == "cu" || e == "cuh" || e == "h");
        if path.is_dir() {
            cuda_source_files(&path, out);
        } else if is_source {
            out.push(path);
        }
    }
}

/// The text with `// ...` and `/* ... */` comments blanked to spaces. Newlines
/// inside comments are kept, so byte offsets and line numbers match the
/// original file and findings quote the reader's lines.
fn strip_comments(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    let mut in_line_comment = false;
    let mut in_block_comment = false;
    while i < bytes.len() {
        let c = bytes[i];
        let next = bytes.get(i + 1).copied();
        if in_line_comment {
            if c == b'\n' {
                in_line_comment = false;
                out.push('\n');
            } else {
                out.push(' ');
            }
        } else if in_block_comment {
            if c == b'*' && next == Some(b'/') {
                in_block_comment = false;
                out.push_str("  ");
                i += 1;
            } else {
                out.push(if c == b'\n' { '\n' } else { ' ' });
            }
        } else if c == b'/' && next == Some(b'/') {
            in_line_comment = true;
            out.push_str("  ");
            i += 1;
        } else if c == b'/' && next == Some(b'*') {
            in_block_comment = true;
            out.push_str("  ");
            i += 1;
        } else {
            out.push(c as char);
        }
        i += 1;
    }
    out
}

/// The initializer floats of the declaration that `constant_at` points into
/// (the `__constant__` occurrence), scanned through the joined multi-line text
/// to the declaration's `;`. `Ok(None)` = no initializer (runtime symbol init,
/// unaffected by the at-load aliasing); `Err(())` = initialized, but not
/// readable as literal floats (macro or expression) — the caller refuses it.
fn initializer_values(text: &str, constant_at: usize) -> Result<Option<Vec<f32>>, ()> {
    let decl = &text[constant_at..];
    let end = decl
        .find(';')
        .map(|e| constant_at + e)
        .unwrap_or(text.len());
    let decl = &text[constant_at..end];
    let Some(eq) = decl.find('=') else {
        return Ok(None);
    };
    let Some(open) = decl[eq..].find('{') else {
        return Err(()); // `= MACRO;` or `= expr;`: initialized, not pinnable
    };
    let body_start = constant_at + eq + open;
    let Some(close_rel) = text[body_start..].find('}') else {
        return Err(());
    };
    // +1: the body starts after the '{' itself, not on it.
    let values = parse_floats(&text[body_start + 1..body_start + close_rel]);
    if values.is_empty() {
        Err(()) // braces present but not literal floats (nested or macro use)
    } else {
        Ok(Some(values))
    }
}

fn parse_floats(body: &str) -> Vec<f32> {
    let mut out = Vec::new();
    for term in body.split(',') {
        let term = term
            .trim()
            .trim_end_matches(['f', 'F', 'u', 'U', 'l', 'L'])
            .trim();
        if term.is_empty() {
            continue; // trailing comma
        }
        // A non-numeric term (macro, expression) reads as no values: the caller
        // refuses the declaration rather than skipping it.
        match term.parse::<f32>() {
            Ok(value) => out.push(value),
            Err(_) => return Vec::new(),
        }
    }
    out
}

/// Fixture classifier tests: the exact behaviors the tree scan relies on.
/// Written after the 2026-11-18 rewrite bug (body slice started on the '{',
/// every literal read as non-literal) — these pin each classification so a
/// rewrite cannot pass the tree test vacuously.
#[test]
fn the_classifier_reads_the_declaration_forms() {
    // A literal table on one line reads as its values.
    let one_line = "__constant__ float t[2] = { 0.5f, -1.0f };";
    let at = one_line.find("__constant__").unwrap();
    assert_eq!(
        initializer_values(one_line, at),
        Ok(Some(vec![0.5, -1.0])),
        "a one-line literal initializer must read as values"
    );

    // The brace may open on a later line than the declaration.
    let next_line = "__constant__ float t[2] =\n    { 0.5f, -1.0f };";
    let at = next_line.find("__constant__").unwrap();
    assert_eq!(
        initializer_values(next_line, at),
        Ok(Some(vec![0.5, -1.0])),
        "a brace on the next line must read as values"
    );

    // A macro initializer is initialized but not pinnable: refused, not skipped.
    let macro_init = "__constant__ float t[16] = KVALUES;";
    let at = macro_init.find("__constant__").unwrap();
    assert_eq!(
        initializer_values(macro_init, at),
        Err(()),
        "a macro initializer must be refused"
    );

    // A declaration with no initializer is runtime symbol init: None.
    let no_init = "__constant__ float t[16];";
    let at = no_init.find("__constant__").unwrap();
    assert_eq!(
        initializer_values(no_init, at),
        Ok(None),
        "no initializer must classify as runtime symbol init"
    );

    // A __constant__ inside a comment disappears under strip_comments.
    let commented = "/* __constant__ float t[16] = { 0.f }; */\nint x;";
    assert!(
        !strip_comments(commented).contains("__constant__"),
        "strip_comments must remove commented declarations"
    );
}

/// The classifier must read the two pinned declarations as allowed literals
/// (regression guard for the 2026-11-18 rewrite bug). Feeding the real files
/// keeps this test from drifting from what the scan sees.
#[test]
fn the_classifier_accepts_the_two_pinned_declarations() {
    for file in ["nvfp4_gemv.cu", "nvfp4_gemv_batch.cu"] {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src/kernels/cuda/native")
            .join(file);
        let raw = std::fs::read_to_string(&path).expect("the pinned twin is readable");
        let text = strip_comments(&raw);
        let Some(at) = text.find("__constant__") else {
            panic!("{file} carries no __constant__ declaration");
        };
        match initializer_values(&text, at) {
            Ok(Some(values)) => assert_eq!(
                values.as_slice(),
                &KE2M1[..],
                "{file} must classify as the pinned allowed literal table"
            ),
            Ok(None) => panic!("{file}: the pinned declaration must carry an initializer"),
            Err(()) => {
                panic!("{file}: the classifier must read the pinned initializer as literal floats")
            }
        }
    }
}

#[test]
fn cuda_sources_add_no_initialized_constant_tables() {
    let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut files = Vec::new();
    cuda_source_files(&manifest.join("src/kernels/cuda/native"), &mut files);
    cuda_source_files(&manifest.join("vendor/ggml-cuda-mmq"), &mut files);
    cuda_source_files(&manifest.join("vendor/xavier-qwen"), &mut files);
    assert!(
        !files.is_empty(),
        "the native CUDA source trees must be found"
    );

    let mut offenders = Vec::new();
    let mut pinned_seen: Vec<&str> = Vec::new();
    for path in &files {
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        let raw = std::fs::read_to_string(path).expect("CUDA source is readable");
        let text = strip_comments(&raw);
        let mut cursor = 0;
        while let Some(found) = text[cursor..].find("__constant__") {
            let at = cursor + found;
            cursor = at + "__constant__".len();
            let line_no = text[..at].matches('\n').count() + 1;
            let site = format!("{}:{}", path.display(), line_no);
            match initializer_values(&text, at) {
                Ok(None) => continue, // no initializer: runtime symbol init
                Ok(Some(values)) => match ALLOWED.iter().find(|(file, _)| *file == name) {
                    Some((file, pinned)) if values == *pinned => pinned_seen.push(file),
                    Some((file, pinned)) => offenders.push(format!(
                        "{site}: {file} carries {} values, pinned {pinned:?}",
                        values.len()
                    )),
                    None => offenders.push(format!(
                        "{site}: initialized __constant__ array ({} values)",
                        values.len()
                    )),
                },
                Err(()) => offenders.push(format!(
                    "{site}: initialized __constant__ via a non-literal initializer \
                     (macro or expression); the guard only pins literal floats"
                )),
            }
        }
    }
    // The scan must actually see the pinned twins: a form change in their
    // declarations would otherwise make this test pass while pinning nothing.
    let mut want: Vec<&str> = ALLOWED.iter().map(|(file, _)| *file).collect();
    let mut got = pinned_seen;
    want.sort_unstable();
    got.sort_unstable();
    got.dedup();
    assert_eq!(
        got, want,
        "the pinned tables must each be found exactly once in the scan"
    );
    assert!(
        offenders.is_empty(),
        "initialized __constant__ arrays alias by position in this multi-cubin .so \
         (2026-11-18: gemm_dq's IQ4_NL table read nvfp4 kE2M1's initializer; renaming \
         and external linkage did not help). Pass tables BY VALUE as kernel arguments \
         instead (iq4_nl_values.cuh's make_iq4nl_table). New offenders:\n{}",
        offenders.join("\n")
    );
}
