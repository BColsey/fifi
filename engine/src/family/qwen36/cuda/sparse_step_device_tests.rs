//! Executor-level sparse qualification (4e-1c close): the resident step with
//! `SparseExec` injected into `StepResources`, against the real Qwen3.6 weights.
//!
//! Rig: every test is `#[ignore]`d and run by name on the exclusive 3090 (the
//! `kv8_step_device_tests` pattern). Sparse injection is DIRECT
//! (`res.sparse = Some(...)`) so the `armed()` env cache cannot order these
//! tests; the env gate itself is covered by the host suite.
//!
//! F-coverage in this file: F4 planted full-coverage equality, F2 geometric
//! forced refusal (and the no-refusal-band arm), F3 restore generation binding,
//! exactly-once append, and the flag-off determinism anchor. F1 (poison recovery)
//! is covered by the poison test; the empty-selection arm of F2 is a kernel-level
//! contract (a zero count under a ZERO status word writes zeros, never a
//! fallback) pinned by the kernel suite - at the executor a selection always
//! holds take >= 1 plus the bands.

use super::common_tests;
use super::dims;
use super::kv8_step_device_tests::{full_options, load_full_native, rules};
use super::native_device_rig::device_present;
use super::resources::StepResources;
use super::sparse_exec::{SparseConfig, SparseExec, SparseGeometry};
use crate::EngError;
use crate::family::qwen36::cuda::step::QwenResidentStep;

fn geometry_for(dims: &dims::StepDims, cells: usize) -> SparseGeometry {
    let attn = &dims.shape.attn;
    SparseGeometry {
        n_q_head: attn.heads_q,
        n_kv_head: attn.heads_kv,
        head_dim: attn.head_dim,
        layers: dims.layers_of(crate::family::spec::AttentionKind::Full),
        capacity: cells,
    }
}

/// Per-row observation hook: the session's sparse state (None on a dense session).
type SparseNote<'a> = Option<&'a mut dyn FnMut(Option<&SparseExec>)>;

/// Runs `ids` through a fresh session with an optional sparse geometry (the
/// default config) and returns the mixer row after each token.
fn rows_with_sparse(
    ids: &[u32],
    cells: Option<usize>,
    note: SparseNote<'_>,
) -> Result<Vec<Vec<u8>>, EngError> {
    rows_with_cfg(ids, cells, SparseConfig::default(), note)
}

/// The config-general form of [`rows_with_sparse`].
fn rows_with_cfg(
    ids: &[u32],
    cells: Option<usize>,
    cfg: SparseConfig,
    mut note: SparseNote<'_>,
) -> Result<Vec<Vec<u8>>, EngError> {
    let (projector, weights) = load_full_native()?;
    let context = projector.context();
    let dims = common_tests::dims();
    let inventory = common_tests::inventory();
    let options = full_options(dims::KvMode::Int8G32);
    let mut res = StepResources::new(context, dims.clone(), &inventory, &options)?;
    if let Some(cells) = cells {
        // Direct injection: armed() stays unset for the rest of the process.
        res.sparse = Some(SparseExec::new(
            context,
            geometry_for(&dims, cells),
            cfg,
            0,
        )?);
    }
    let mut step = QwenResidentStep::new(context, &res, &weights, &options, rules()?)?;
    let mut rows = Vec::with_capacity(ids.len());
    for &id in ids {
        step.run_token(id, true)?;
        rows.push(step.row_bytes().to_vec());
        if let Some(note) = note.as_mut() {
            note(res.sparse.as_ref());
        }
    }
    drop(step);
    drop(res);
    drop(weights);
    drop(projector);
    Ok(rows)
}

fn same_prompt() -> Vec<u32> {
    // Deterministic, in-vocab ids (the rig's prompt file is not needed).
    (0..48).map(|i| 1_000 + (i * 7919) % 200_000).collect()
}

fn rows_close(a: &[u8], b: &[u8], tol: f32) -> bool {
    if a.len() != b.len() || !a.len().is_multiple_of(4) {
        return false;
    }
    (0..a.len()).step_by(4).all(|i| {
        let word = |r: &[u8]| f32::from_le_bytes([r[i], r[i + 1], r[i + 2], r[i + 3]]);
        let (x, y) = (word(a), word(b));
        (x - y).abs() <= tol * x.abs().max(1.0).max(y.abs())
    })
}

/// The largest element-wise relative drift between two f32 rows (the chaos metric
/// of the chain arms: healthy quantization-flip noise vs catastrophic corruption).
fn max_rel_drift(a: &[u8], b: &[u8]) -> f32 {
    a.as_chunks::<4>()
        .0
        .iter()
        .zip(b.as_chunks::<4>().0)
        .map(|(aw, bw)| {
            let (x, y) = (f32::from_le_bytes(*aw), f32::from_le_bytes(*bw));
            (x - y).abs() / x.abs().max(1.0).max(y.abs())
        })
        .fold(0.0f32, f32::max)
}

/// F4, full-coverage plant: with `top_k` at its default 256 >= every visible key
/// of a 48-token session, the selection provably covers the complete visible
/// attention set, so the sparse chain's output must equal the dense output.
#[test]
#[ignore = "the 3090, held exclusively: loads the real model; run by name"]
fn sparse_full_coverage_selection_matches_dense() {
    if !device_present("sparse_full_coverage_selection_matches_dense") {
        return;
    }
    // 48 tokens: every row stays below the sink+recent basis, so this end-to-end
    // pair never launches a chain - it pins the bypass band only. The chain rows
    // are exercised by the 160-token matched-geometry test below.
    let ids = same_prompt();
    let dense = rows_with_sparse(&ids, None, None).expect("dense session");
    let sparse = rows_with_sparse(&ids, Some(1024), None).expect("sparse session");
    assert_eq!(dense.len(), sparse.len());
    for (n, (d, s)) in dense.iter().zip(&sparse).enumerate() {
        assert!(
            rows_close(d, s, 2e-4),
            "row {n}: full-coverage sparse selection must reproduce dense output"
        );
    }
}

/// R2 pre-launch bypass: a half-capacity geometry with only 48 visible tokens
/// never launches a chain (`basis_valid_for` is false below sink+recent), so
/// every row is the dense path verbatim - bitwise the dense session's rows.
#[test]
#[ignore = "the 3090, held exclusively: loads the real model; run by name"]
fn sparse_short_context_takes_the_host_bypass_bitwise() {
    if !device_present("sparse_short_context_takes_the_host_bypass_bitwise") {
        return;
    }
    let ids = same_prompt();
    let dense = rows_with_sparse(&ids, None, None).expect("dense session");
    let refused = rows_with_sparse(&ids, Some(512), None).expect("refused session");
    for (n, (d, r)) in dense.iter().zip(&refused).enumerate() {
        assert_eq!(d, r, "row {n}: a refused chain must leave the dense output");
    }
}

/// F1, sketch refusal poisons the session: a cells-100 geometry over a 160-token
/// session refuses the SKETCH launch at pos 100 (after the authoritative append),
/// and the session must fail explicitly with the typed invalid-step-state error -
/// never committing the token whose sketch is invalid - and stay poisoned. Only
/// the poison contract is asserted: the shrunken geometry deliberately misstrides
/// the decode bands, which the stride guard in the mixer now refuses to run.
#[test]
#[ignore = "the 3090, held exclusively: loads the real model; run by name"]
fn sparse_sketch_refusal_poisons_the_session() {
    if !device_present("sparse_sketch_refusal_poisons_the_session") {
        return;
    }
    let ids160: Vec<u32> = (0..160).map(|i| 1_000 + (i * 7919) % 200_000).collect();
    let (projector, weights) = load_full_native().expect("load");
    let context = projector.context();
    let dims = common_tests::dims();
    let inventory = common_tests::inventory();
    let options = full_options(dims::KvMode::Int8G32);
    let mut res = StepResources::new(context, dims.clone(), &inventory, &options).expect("res");
    res.sparse = Some(
        SparseExec::new(
            context,
            geometry_for(&dims, 100),
            SparseConfig::default(),
            0,
        )
        .expect("admit"),
    );
    let mut step =
        QwenResidentStep::new(context, &res, &weights, &options, rules().expect("rules"))
            .expect("step");
    let mut first_err = None;
    for (n, &id) in ids160.iter().enumerate() {
        if let Err(e) = step.run_token(id, true) {
            first_err = Some((n, e));
            break;
        }
    }
    let (n_err, err) = first_err.expect("the sketch refusal must surface as an explicit error");
    eprintln!("first failing token {n_err}: {err:?}");
    assert_eq!(n_err, 100, "the refusal starts exactly at pos == cells");
    assert!(
        matches!(err, EngError::Backend { .. }),
        "the poisoned commit is a typed backend refusal"
    );
    assert!(
        step.run_token(ids160[0], true).is_err(),
        "a poisoned session must keep failing explicitly"
    );
    drop(step);
    drop(res);
    drop(weights);
    drop(projector);
}

/// F2 + F4 with a stride-matched geometry (cells == capacity): a 160-token session
/// walks two regimes. Below visible > sink+recent (rows 0..=67) the host bypass
/// gives bitwise dense rows — the host gate sits one row AHEAD of the kernel's
/// empty-band refusal, so row 67 (visible == sink+recent, empty candidate band)
/// never launches at all. From row 68 on, the chain runs every row with a
/// provably complete selection (take == candidates, so picks + sinks +
/// recent cover every visible key) and must stay near dense: the sparse kernels'
/// fp-order deviation is far below the pipeline's inter-block quantization step,
/// so rows are bitwise until a value sits on an int8 rounding boundary - one flip
/// amplifies through the remaining blocks to ~0.1 logits, chaotically and at
/// unpredictable rows. A wrong cell SET (the defect class this test exists for)
/// shifts logits by O(1) units, an order of magnitude above the flip noise, so the
/// chain arithmetic itself stays pinned by the kernel-level oracle (subset
/// exactness, f64). Row 68 - the first chain row, entering with bitwise-identical
/// state - must itself be bitwise: sub-quantum deviation is structural, not luck.
/// Past the first flip the two sessions ride different quantization trajectories
/// and the drift compounds without bound, so later rows gate only against
/// catastrophic corruption. The fallback counter pins the wiring: no row ever
/// takes a post-launch fallback (the band never launches a doomed chain).
///
/// Fixture notes. (1) The 5.0 cap is a REGRESSION characterization for this rig,
/// not a quality acceptance threshold - application-level non-inferiority is
/// 4e-2's judgment, not this test's. (2) Row 68's bitwise assert is a pinned
/// fixture of the current binary, model pack and numerical configuration (B1's
/// fix engages all 10 full-attention layers, ~5x the per-row chain exposure of
/// the pre-fix wiring): a moved first flip is a requalification trigger for the
/// numerical receipts, never a tolerance change. (3) The band asserts are the
/// regression fixture for the historical pos/kv_len word bug: if a never-appended
/// cell becomes selectable, the chain rows stop being near-dense; if the refusal
/// boundary shifts by one, row 67 or row 68 starts taking post-launch fallbacks
/// (counter > 0) - both fail loudly here.
#[test]
#[ignore = "the 3090, held exclusively: loads the real model; run by name"]
fn sparse_matched_geometry_bypass_refuse_fullcover() {
    if !device_present("sparse_matched_geometry_bypass_refuse_fullcover") {
        return;
    }
    let ids160: Vec<u32> = (0..160).map(|i| 1_000 + (i * 7919) % 200_000).collect();
    let dense = rows_with_sparse(&ids160, None, None).expect("dense session");
    let mut fallbacks = Vec::with_capacity(ids160.len());
    let mut last_launches: Vec<u64> = Vec::new();
    let sparse = rows_with_cfg(
        &ids160,
        Some(1024),
        SparseConfig::default(),
        Some(&mut |sp| {
            fallbacks.push(sp.map_or(0, SparseExec::fallbacks));
            last_launches = sp.map_or_else(Vec::new, SparseExec::plane_launches);
        }),
    )
    .expect("sparse session");
    assert_eq!(dense.len(), sparse.len());
    let dump = |n: usize, d: &[u8], s: &[u8]| {
        let mut el: Vec<(usize, f32, f32)> = d
            .as_chunks::<4>()
            .0
            .iter()
            .zip(s.as_chunks::<4>().0)
            .enumerate()
            .filter_map(|(i, (dw, sw))| {
                let (x, y) = (f32::from_le_bytes(*dw), f32::from_le_bytes(*sw));
                (x != y).then_some((i, x, y))
            })
            .collect();
        el.sort_by(|a, b| (b.1 - b.2).abs().partial_cmp(&(a.1 - a.2).abs()).unwrap());
        eprintln!(
            "row {n}: {} elements differ; top: {:?}",
            el.len(),
            &el[..el.len().min(6)]
        );
    };
    let mut prev_fb = 0u64;
    for (n, (d, s)) in dense.iter().zip(&sparse).enumerate() {
        let fb_delta = fallbacks.get(n).copied().unwrap_or(0) - prev_fb;
        prev_fb = fallbacks.get(n).copied().unwrap_or(0);
        if n < 68 {
            assert_eq!(d, s, "bypass row {n} must be bitwise dense");
            if n == 67 {
                assert_eq!(
                    fb_delta, 0,
                    "row 67 (empty candidate band) must not launch: the host gate \
                    stays one row ahead of the kernel refusal"
                );
            }
        } else {
            // Wiring: past the refusal band every chain row must RUN (zero fallbacks).
            assert_eq!(
                fb_delta, 0,
                "row {n} fell back post-launch: the chain must run past the refusal band"
            );
            let drift = max_rel_drift(d, s);
            eprintln!("row {n}: max rel drift {drift:.3e}, fallbacks this row {fb_delta}");
            if drift > 0.0 {
                dump(n, d, s);
            }
            if n == 68 {
                assert_eq!(
                    d, s,
                    "row 68 (first chain row, bitwise entry state) must be bitwise: \
                    a deviation here is a real selection or arithmetic defect"
                );
            } else {
                // Past the first int8 flip the sessions ride different quantization
                // trajectories and drift compounds without bound (observed 1.2 by
                // row 103); only catastrophic corruption is still distinguishable.
                assert!(
                    drift <= 5.0,
                    "row {n} drift {drift:.3e} is catastrophic, past quantization \
                    chaos: the selection or the chain arithmetic is wrong"
                );
            }
        }
    }
    // Invocation coverage (B1's blind spot): every admitted plane must have launched on
    // EVERY chain row - one chain per Full layer per row, no partial engagement, and no
    // launches outside the admitted plane count.
    let chain_rows = ids160.len() - 68;
    assert_eq!(
        last_launches.len(),
        10,
        "admitted plane count must be the 10 Full layers"
    );
    for (plane, count) in last_launches.iter().enumerate() {
        assert_eq!(
            *count, chain_rows as u64,
            "plane {plane} launched {count} times: every chain row must launch every \
            admitted plane exactly once (partial engagement is the B1 defect class)"
        );
    }
}

/// Flag-off anchor: two sparse-less sessions over the same ids are bitwise
/// identical (the wiring changed nothing when the resources are absent).
#[test]
#[ignore = "the 3090, held exclusively: loads the real model; run by name"]
fn without_sparse_resources_rows_are_deterministic() {
    if !device_present("without_sparse_resources_rows_are_deterministic") {
        return;
    }
    let ids = same_prompt();
    let a = rows_with_sparse(&ids, None, None).expect("session a");
    let b = rows_with_sparse(&ids, None, None).expect("session b");
    assert_eq!(a, b);
}

/// The shared 160-token dense control of the restore and band arms (one model load
/// for both tests).
static DENSE160: std::sync::OnceLock<Result<Vec<Vec<u8>>, String>> = std::sync::OnceLock::new();

fn dense160(ids: &[u32]) -> Result<Vec<Vec<u8>>, EngError> {
    DENSE160
        .get_or_init(|| rows_with_sparse(ids, None, None).map_err(|e| format!("{e:?}")))
        .clone()
        .map_err(EngError::BadRequest)
}

/// F3 + exactly-once, the generation binding across a restore: a checkpoint export
/// restores the KV planes and the position but never the sketch (the sketch belongs
/// to the session's SparseExec), so a selection may only consume cells the session
/// itself appended. top_k = 8 makes rows past kv_len = 76 score-dependent, so a
/// stale sketch cell is not masked by a complete selection.
///
/// Arms: (i) transparency - a session that exports at 60 and continues must be
/// bitwise a session that imports its own export at 60 and continues (the restore
/// changes nothing, and every re-appended cell's sketch is rewritten exactly once:
/// a duplicate or skipped sketch write shifts the scores and moves a pick).
/// (ii) fail-closed - a FRESH session that imports the checkpoint (sketch cells
/// [0, 60) never written by it, physically valid KV) must never launch the chain:
/// every row after the import is bitwise the dense control. The KV append side of
/// exactly-once is pinned by the bitwise bands of the matched-geometry test (a
/// duplicate or missing append misstrides the next row), and the fallback path's
/// append happened once before the fallback by construction (the sketch and KV
/// append enqueue ahead of the attention op that may refuse).
#[test]
#[ignore = "the 3090, held exclusively: loads the real model; run by name"]
fn sparse_restore_binds_the_selection_generation() {
    if !device_present("sparse_restore_binds_the_selection_generation") {
        return;
    }
    let ids160: Vec<u32> = (0..160).map(|i| 1_000 + (i * 7919) % 200_000).collect();
    let cfg = |top_k| SparseConfig {
        top_k,
        ..SparseConfig::default()
    };

    // The checkpoint of the 60-token prefix, exported from a session that ran it.
    let exported = {
        let (projector, weights) = load_full_native().expect("load");
        let context = projector.context();
        let dims = common_tests::dims();
        let inventory = common_tests::inventory();
        let options = full_options(dims::KvMode::Int8G32);
        let mut res = StepResources::new(context, dims.clone(), &inventory, &options).expect("res");
        res.sparse =
            Some(SparseExec::new(context, geometry_for(&dims, 1024), cfg(8), 0).expect("admit"));
        let mut step =
            QwenResidentStep::new(context, &res, &weights, &options, rules().expect("rules"))
                .expect("step");
        for &id in &ids160[..60] {
            step.run_token(id, true).expect("prefix token");
        }
        let exported = step.export_state().expect("export");
        drop(step);
        drop(res);
        drop(weights);
        drop(projector);
        exported
    };

    // (i) transparency: continue straight vs import-own-export at 60, then continue.
    let mut straight_fb = 0u64;
    let straight = rows_with_cfg(
        &ids160,
        Some(1024),
        cfg(8),
        Some(&mut |sp| straight_fb = sp.map_or(0, SparseExec::fallbacks)),
    )
    .expect("straight");
    let rewound = {
        let (projector, weights) = load_full_native().expect("load");
        let context = projector.context();
        let dims = common_tests::dims();
        let inventory = common_tests::inventory();
        let options = full_options(dims::KvMode::Int8G32);
        let mut res = StepResources::new(context, dims.clone(), &inventory, &options).expect("res");
        res.sparse =
            Some(SparseExec::new(context, geometry_for(&dims, 1024), cfg(8), 0).expect("admit"));
        let mut step =
            QwenResidentStep::new(context, &res, &weights, &options, rules().expect("rules"))
                .expect("step");
        for &id in &ids160[..60] {
            step.run_token(id, true).expect("prefix token");
        }
        step.import_state(&exported).expect("import");
        let mut rows = Vec::with_capacity(100);
        let mut prev_fb = res.sparse.as_ref().expect("sparse").fallbacks();
        for &id in &ids160[60..] {
            step.run_token(id, true).expect("post-restore token");
            rows.push(step.row_bytes().to_vec());
            let fb = res.sparse.as_ref().expect("sparse").fallbacks();
            if fb != prev_fb {
                eprintln!(
                    "rewound: row {} took {} post-launch fallbacks",
                    59 + rows.len(),
                    fb - prev_fb
                );
                prev_fb = fb;
            }
        }
        assert_eq!(
            res.sparse.as_ref().expect("sparse").fallbacks(),
            straight_fb,
            "a contiguous same-session restore takes exactly the straight session's \
             band fallbacks: both sessions route the band rows identically"
        );
        drop(step);
        drop(res);
        drop(weights);
        drop(projector);
        rows
    };
    for (n, (a, b)) in straight[60..].iter().zip(&rewound).enumerate() {
        assert_eq!(
            a,
            b,
            "row {}: a same-session restore must be transparent",
            n + 60
        );
    }

    // (ii) fail-closed: a fresh session imports the checkpoint and never appends
    // cells [0, 60) itself - the stale sketch must not be selectable.
    let imported = {
        let (projector, weights) = load_full_native().expect("load");
        let context = projector.context();
        let dims = common_tests::dims();
        let inventory = common_tests::inventory();
        let options = full_options(dims::KvMode::Int8G32);
        let mut res = StepResources::new(context, dims.clone(), &inventory, &options).expect("res");
        res.sparse =
            Some(SparseExec::new(context, geometry_for(&dims, 1024), cfg(8), 0).expect("admit"));
        let mut step =
            QwenResidentStep::new(context, &res, &weights, &options, rules().expect("rules"))
                .expect("step");
        step.import_state(&exported).expect("import");
        let mut rows = Vec::with_capacity(100);
        for &id in &ids160[60..] {
            step.run_token(id, true).expect("post-import token");
            rows.push(step.row_bytes().to_vec());
        }
        assert_eq!(
            res.sparse.as_ref().expect("sparse").fallbacks(),
            0,
            "the chain must not launch at all: no refusal, just the R3 bypass"
        );
        drop(step);
        drop(res);
        drop(weights);
        drop(projector);
        rows
    };
    let dense = dense160(&ids160).expect("dense control");
    for (n, (d, s)) in dense[60..].iter().zip(&imported).enumerate() {
        assert_eq!(
            d,
            s,
            "row {}: a session with a stale sketch must stay bitwise dense",
            n + 60
        );
    }
}

/// F2's band arm: sink = 0 and recent = 0 remove the refusal band entirely -
/// `basis_valid_for` holds from the first row, so the chain runs from row 0 with a
/// selection that is complete at every length below top_k + 1, and the dense
/// fallback must NEVER fire (the fallback contract is mask-4 only; a legitimate
/// zero-count selection is a kernel-level outcome, pinned by the kernel suite,
/// and cannot occur at the executor with sink + recent > 0 or take >= 1). Row 0's
/// selection is the single appended cell - structurally the same set and order as
/// the dense decode - so row 0 must be bitwise. Later rows keep the chaos cap of
/// the matched-geometry test.
#[test]
#[ignore = "the 3090, held exclusively: loads the real model; run by name"]
fn sparse_zero_band_runs_the_chain_from_row_zero() {
    if !device_present("sparse_zero_band_runs_the_chain_from_row_zero") {
        return;
    }
    let ids160: Vec<u32> = (0..160).map(|i| 1_000 + (i * 7919) % 200_000).collect();
    let cfg = SparseConfig {
        sink: 0,
        recent: 0,
        ..SparseConfig::default()
    };
    let dense = dense160(&ids160).expect("dense control");
    let mut fallbacks = Vec::with_capacity(ids160.len());
    let sparse = rows_with_cfg(
        &ids160,
        Some(1024),
        cfg,
        Some(&mut |f| fallbacks.push(f.map_or(0, SparseExec::fallbacks))),
    )
    .expect("sparse session");
    assert_eq!(dense.len(), sparse.len());
    let mut prev_fb = 0u64;
    for (n, (d, s)) in dense.iter().zip(&sparse).enumerate() {
        let fb = fallbacks.get(n).copied().unwrap_or(0);
        assert_eq!(
            fb - prev_fb,
            0,
            "row {n}: with no refusal band the fallback must never fire"
        );
        prev_fb = fb;
        if n == 0 {
            assert_eq!(d, s, "row 0 selects the single appended cell: bitwise");
        } else if d != s {
            let drift = max_rel_drift(d, s);
            eprintln!("row {n}: max rel drift {drift:.3e}");
            assert!(
                drift <= 5.0,
                "row {n} drift {drift:.3e} is catastrophic, past quantization \
                chaos: the selection or the chain arithmetic is wrong"
            );
        }
    }
}
