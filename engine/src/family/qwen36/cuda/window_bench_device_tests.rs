//! What the eager verify window costs on the whole 4.00 bpw EXL3 pack against captured single-row
//! decode: a MEASUREMENT harness, written for one future GPU window and NOT run. It gates nothing
//! about speed; it writes one JSON record. The only hard failures are its inline D1 checks (a
//! window that is not equal to one-row decode has no meaningful cost) and its memory admission.
//!
//! One test per process on the GPU held exclusively; `verify_window_cost_record`.
//!
//! **What is compared.** At each depth (default 8,192 and 32,768 committed rows) and each window
//! size `r` (default 2 and 4), per round and in a rotating order so that drift hits all arms:
//!
//! * `captured_single_row`: `r` steps of `run_token` without logits (the captured graph, as an
//!   EXL3 entry serves; `FIFI_D1_CAPTURE=0` runs the eager body instead);
//! * `window_accept_all`: one eager window of `r` rows, every row kept: the export, `r` steps with
//!   a logits row read back each, the greedy draws, and no rollback;
//! * `window_reject_first`: the same window with only row 0 kept: after the rows, the import of
//!   the saved state and one replayed row.
//!
//! Text is teacher forced: every arm consumes the next tokens of the same fixed stream, so the
//! routing is that of real code. The arms advance the committed length by different amounts, so
//! depth drifts by about `2r + 1` rows per round (at most a few hundred rows over a cell).
//!
//! **What is recorded, per window.** The host wall time of the whole window (`run_window` plus
//! the draws plus `commit_window`), and inside it, from clock reads at marks that exist in test
//! builds only (`eager_window::WindowTrace`): the export, each row, the import, each replayed
//! row. The GPU time of each row is the step's in-graph event pair (section marks, on by
//! default, `FIFI_MRV_MARKS=0` for wall times only): a GPU event time alone would miss the export
//! and the import (copies with no events), the host sync of every row, the logits read-back, the
//! acceptance and the replay, which is why the host wall time is the figure of cost and the GPU
//! time is context. The marks add a few in-graph event nodes to every step in both arms: that
//! is an observer effect the record states and does not remove.
//!
//! **What this tooling cannot isolate.** A row's host time holds its launch, the wait for the
//! device and the logits read-back together: there is no event between the compute and the
//! read-back, so the read-back's own cost is only bounded (host row time minus GPU row time,
//! which also holds launch gaps and the sync). The export's host time inside the window is
//! measured exactly (mark to mark); the export alone, repeated at the same state outside a
//! window, is recorded beside it as `export_standalone_ms`, as is the import. The split of the
//! export into device copy time and host allocation time is not available.
//!
//! **Asymmetries the record does not remove.** (1) With the marks on, the window arm reads the
//! event times (`section_times`) inside its wall time, after every row and every replayed row: the
//! read comes after the row's own clock read, so it is in `total_wall_ms` but in no row's time.
//! The single-row arm reads them after its timed region, so it does not pay it. The cost is two
//! event queries per row; `FIFI_MRV_MARKS=0` removes it from the window arm. (2) The state shape,
//! and with it the computed bytes and call counts, is taken once at the start of a cell; the
//! committed length then drifts by about `2r + 1` rows per round, under 1% of the depth over a
//! cell, so the bytes are those of the depth at the start.
//!
//! **Bytes and calls** per window are COMPUTED from the state layout and the logits width
//! (`speculate::window_cost`), not counted by a counter in the serving path; the harness
//! checks the computed state bytes against the real exported state.
//!
//! **Identities.** The record carries the commit (`FIFI_MRV_COMMIT`, given by the runner), the
//! SHA-256 of this test binary (read from `current_exe`), the pack manifest (every file's name
//! and size with the strongest cheap hash; `FIFI_MRV_PACK_MANIFEST_SHA256` pins it), the KV mode,
//! the capture mode, the text and tokenizer files with their SHA-256 and the SHA-256 of the token
//! ids used. Nothing is stamped with a time: the runner records dates with `date -u`.
//!
//! **Not forecasts.** The record reports measured ratios of this window design. The W(2) 1.6 to
//! 1.9 and W(4) 2.4 to 3.0 figures of the audit are hypotheses for a K-row graph that does not
//! exist; nothing here predicts them.
//!
//! Environment: `FIFI_EXL3_CHECKPOINT`, `FIFI_SPEC_TEXT`, `FIFI_SPEC_TOKENIZER`, `FIFI_MRV_COMMIT`
//! (required), `FIFI_MRV_OUT` (the JSON path), `FIFI_MRV_DEPTHS` (8192,32768), `FIFI_MRV_ROWS`
//! (2,4), `FIFI_MRV_ROUNDS` (12), `FIFI_MRV_WARM_FROM` (2: rounds before this are cold and left
//! out of the medians), `FIFI_MRV_MARKS` (1), `FIFI_MRV_DEADLINE_S` (780: after this many seconds
//! from the test's start no new cell or round begins and the record says it stopped early),
//! `FIFI_MRV_LEASE_VRAM_GIB` (21), `FIFI_D1_KV` (fp16), `FIFI_D1_CAPTURE` (1).

use std::{fs, path::PathBuf, time::Instant};

use serde_json::{Value, json};

use super::{
    d1_window_device_tests::{
        EOS, HEAD_ROWS, Loaded, SAMPLER_VOCAB, build_step, capture_from_env, exported_difference,
        kv_mode_from_env, load_pack, step_options, text_stream,
    },
    dims::KvMode,
    eager_window::WindowTrace,
    prefill_driver::{FIXED_ADMISSION_MARGIN, PrefillBudget},
    start::prefill_headroom,
    state::export::ExportedState,
    state_layout::Word,
    step::{CaptureConfig, QwenResidentStep},
};
use crate::{
    EngError,
    family::{
        qwen36::settings::DEFAULT_CUDA_PREFILL_CHUNK,
        sampler::Sampler,
        session::TokenStepper,
        speculate::{
            d1_oracle::{D1Failure, D1Subject, Tally, Verdict, full_logits},
            driver::WindowModel,
            window::Commit,
            window_cost::{
                PackManifest, RECORD_SCHEMA, StateShape, WindowTransfers, check_identity,
                ids_sha256, median, sha256_file, summary,
            },
        },
    },
};

const GIB: u64 = 1 << 30;
/// Free VRAM that must remain after the build and the arena, for the exports and the graph.
const MIN_FREE_AFTER_BUILD: u64 = GIB;
/// Positions past the deepest depth that the cache holds, for the rounds of the last cell.
const CONTEXT_SLACK: usize = 1_536;

struct Config {
    depths: Vec<usize>,
    rows: Vec<usize>,
    rounds: usize,
    warm_from: usize,
    out: Option<PathBuf>,
    commit: String,
    marks: bool,
    deadline_s: f64,
    lease_gib: u64,
}

fn env_list(name: &str, default: &str) -> Vec<usize> {
    std::env::var(name)
        .unwrap_or_else(|_| default.to_owned())
        .split(',')
        .map(|part| {
            part.trim()
                .parse()
                .unwrap_or_else(|_| panic!("{name}: {part:?}"))
        })
        .collect()
}

fn env_num<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name).ok().map_or(default, |text| {
        text.parse().unwrap_or_else(|_| panic!("{name}: {text:?}"))
    })
}

impl Config {
    fn from_env() -> Self {
        let commit = std::env::var("FIFI_MRV_COMMIT").unwrap_or_default();
        assert!(
            !commit.is_empty(),
            "FIFI_MRV_COMMIT: the runner names the commit"
        );
        let config = Self {
            depths: env_list("FIFI_MRV_DEPTHS", "8192,32768"),
            rows: env_list("FIFI_MRV_ROWS", "2,4"),
            rounds: env_num("FIFI_MRV_ROUNDS", 12),
            warm_from: env_num("FIFI_MRV_WARM_FROM", 2),
            out: std::env::var_os("FIFI_MRV_OUT").map(PathBuf::from),
            commit,
            marks: env_num::<u8>("FIFI_MRV_MARKS", 1) == 1,
            deadline_s: env_num("FIFI_MRV_DEADLINE_S", 780.0),
            lease_gib: env_num("FIFI_MRV_LEASE_VRAM_GIB", 21),
        };
        assert!(
            config.depths.windows(2).all(|w| w[0] < w[1]),
            "depths rise: the prefill continues"
        );
        assert!(
            config.rows.iter().all(|&r| (2..=8).contains(&r)),
            "windows of 2 to 8 rows"
        );
        assert!(config.rounds > config.warm_from, "at least one warm round");
        config
    }
}

fn ms(began: Instant) -> f64 {
    began.elapsed().as_secs_f64() * 1e3
}

/// The GPU time of the last token from the step's event pair, or `None` with the marks off.
fn gpu_ms(step: &QwenResidentStep<'_>) -> Option<f64> {
    step.section_times()
        .ok()
        .flatten()
        .map(|times| f64::from(times.whole_ms()))
}

/// The session, admitted for chunked prefill the way the serve path admits it for `kv`.
fn bench_step(
    loaded: &Loaded,
    kv: KvMode,
    context: usize,
    capture: bool,
    marks: bool,
) -> Result<QwenResidentStep<'static>, EngError> {
    let mut step = build_step(
        loaded,
        &step_options(kv, context),
        CaptureConfig {
            capture,
            section_marks: marks,
        },
    )?;
    // As `start::make`: the Q8 profile chunks at 1,024 rows with fixed admission; the fp16 profile
    // keeps the default chunk and the legacy headroom.
    let (kv8, cap, headroom, flags) = match kv {
        KvMode::Int8G32 => (true, 1_024, FIXED_ADMISSION_MARGIN, (true, true, false)),
        KvMode::Fp16 => (
            false,
            DEFAULT_CUDA_PREFILL_CHUNK,
            prefill_headroom(false),
            (true, false, true),
        ),
    };
    step.set_kv8_chunked_prefill(kv8)?;
    step.set_m_sess_cap(cap);
    let free = loaded.projector.memory_info()?.free;
    let verdict = step.admit_prefill(
        PrefillBudget {
            device_free: free.saturating_sub(headroom),
            extras: Vec::new(),
        },
        flags.0,
        flags.1,
        flags.2,
    )?;
    assert!(
        format!("{verdict:?}").starts_with("Ready"),
        "the chunk path must be admitted, or the prefill runs one token at a time: {verdict:?}"
    );
    Ok(step)
}

/// `FIFI_*`, `Q36_*` and `INFERENCE_CUDA_*` variables: switches that can change which kernel runs.
fn switches() -> Value {
    let mut pairs: Vec<(String, String)> = std::env::vars()
        .filter(|(name, _)| {
            ["FIFI_", "Q36_", "INFERENCE_CUDA_"]
                .iter()
                .any(|prefix| name.starts_with(prefix))
        })
        .collect();
    pairs.sort();
    Value::Object(
        pairs
            .into_iter()
            .map(|(name, value)| (name, Value::String(value)))
            .collect(),
    )
}

fn proc_status_kib(key: &str) -> Option<u64> {
    let text = fs::read_to_string("/proc/self/status").ok()?;
    let line = text.lines().find(|line| line.starts_with(key))?;
    line.split_whitespace().nth(1)?.parse().ok()
}

fn host_json() -> Value {
    json!({
        "loadavg": fs::read_to_string("/proc/loadavg").unwrap_or_default().trim(),
        "rss_kib": proc_status_kib("VmRSS:"),
        "rss_high_water_kib": proc_status_kib("VmHWM:"),
        "mem_available_kib": fs::read_to_string("/proc/meminfo").ok().and_then(|text| {
            text.lines().find(|l| l.starts_with("MemAvailable:"))?.split_whitespace().nth(1)?.parse::<u64>().ok()
        }),
    })
}

struct Identity {
    json: Value,
}

fn identity(config: &Config, kv: KvMode, capture: bool, stream: &[u32]) -> Identity {
    let pack_dir =
        PathBuf::from(std::env::var_os("FIFI_EXL3_CHECKPOINT").expect("FIFI_EXL3_CHECKPOINT"));
    let pack = PackManifest::read(&pack_dir).expect("pack manifest");
    if let Ok(expected) = std::env::var("FIFI_MRV_PACK_MANIFEST_SHA256") {
        assert_eq!(
            pack.manifest_sha256, expected,
            "the pack is not the one the run plan froze"
        );
    }
    let exe = std::env::current_exe().expect("current_exe");
    let text = PathBuf::from(std::env::var_os("FIFI_SPEC_TEXT").expect("FIFI_SPEC_TEXT"));
    let tokenizer =
        PathBuf::from(std::env::var_os("FIFI_SPEC_TOKENIZER").expect("FIFI_SPEC_TOKENIZER"));
    let ids_sha256 = ids_sha256(stream);
    if let Ok(expected) = std::env::var("FIFI_MRV_IDS_SHA256") {
        assert_eq!(
            ids_sha256, expected,
            "the token ids are not the ones the run plan froze"
        );
    }
    let json = json!({
        "commit": config.commit,
        "binary_path": exe.display().to_string(),
        "binary_sha256": sha256_file(&exe).expect("binary sha256"),
        "pack": pack.to_json(),
        "kv_mode": format!("{kv:?}"),
        "capture": capture,
        "section_marks": config.marks,
        "switches": switches(),
        "token_ids": {
            "source": format!("encode({}) with {}", text.display(), tokenizer.display()),
            "text_path": text.display().to_string(),
            "text_sha256": sha256_file(&text).expect("text sha256"),
            "tokenizer_path": tokenizer.display().to_string(),
            "tokenizer_sha256": sha256_file(&tokenizer).expect("tokenizer sha256"),
            "ids_used": stream.len(),
            "ids_sha256": ids_sha256,
        },
    });
    Identity { json }
}

/// Bytes of an exported state: the DeltaNet blocks and the KV planes.
fn exported_bytes(state: &ExportedState) -> u64 {
    let delta: usize = state
        .delta
        .iter()
        .map(|layer| (layer.recurrent.len() + layer.conv.len()) * 4)
        .sum();
    let kv: usize = state
        .kv
        .iter()
        .flat_map(|layer| layer.planes.iter().map(Vec::len))
        .sum();
    (delta + kv) as u64
}

/// The copy structure of the state at its current length, checked against a real export.
fn state_shape(
    step: &QwenResidentStep<'_>,
    exported: &ExportedState,
) -> Result<StateShape, EngError> {
    let layout = step.state_layout();
    let kv_runs = (0..layout.kv.layers)
        .map(|ordinal| {
            let ranges = layout.kv.prefix_ranges(ordinal, exported.tokens)?;
            Ok(layout
                .kv
                .planes()
                .iter()
                .map(|&plane| {
                    ranges
                        .iter()
                        .filter(|range| range.plane == plane && range.bytes > 0)
                        .map(|range| range.bytes as u64)
                        .collect()
                })
                .collect())
        })
        .collect::<Result<Vec<_>, EngError>>()?;
    let shape = StateShape {
        delta_layers: layout.delta.layers,
        recurrent_bytes: (layout.delta.rec_values * 4) as u64,
        conv_bytes: (layout.delta.conv_values * 4) as u64,
        kv_runs,
        words_bytes: layout.words.total as u64,
        // The row buffer is allocated head-wide at build (`vec![0; f32_bytes(vocab)]`), and the
        // read-back plan reads that many bytes when logits are wanted.
        logits_row_values: (step.row_bytes().len() / 4) as u64,
        start_block_bytes: Word::StepParams.bytes() as u64,
        result_pair_bytes: Word::ResultPair.bytes() as u64,
    };
    assert_eq!(
        shape.logits_row_values, HEAD_ROWS as u64,
        "the logits row the step reads back is the padded head width"
    );
    assert_eq!(
        shape.state_bytes(),
        exported_bytes(exported),
        "the computed state bytes must equal the exported state's"
    );
    Ok(shape)
}

fn restoration_failure(detail: String) -> D1Failure {
    D1Failure {
        test: "R state restoration",
        detail,
        violation: true,
    }
}

/// After a window that keeps `kept` rows the state must equal `kept` one-row steps from the same
/// base, bit for bit: a full accept (no rollback) and a first-row accept (import and replay).
fn restoration(step: &mut QwenResidentStep<'_>, inputs: &[u32]) -> Verdict<Tally> {
    let on = |result: Result<(), EngError>| {
        result.map_err(|error| D1Failure {
            test: "R state restoration",
            detail: format!("{error:?}"),
            violation: false,
        })
    };
    let rows = inputs.len();
    let base = step
        .snapshot()
        .map_err(|e| restoration_failure(format!("{e:?}")))?;
    let mut tally = Tally::default();
    for kept in [rows, 1] {
        on(step.restore(&base))?;
        for &token in &inputs[..kept] {
            on(step.run_token(token, false).map(|_| ()))?;
        }
        let reference = step
            .snapshot()
            .map_err(|e| restoration_failure(format!("{e:?}")))?;
        on(step.restore(&base))?;
        on(step.run_window(inputs))?;
        on(step.commit_window(&Commit {
            committed_rows: base.tokens + kept,
            kept_rows: kept,
            discarded_rows: rows - kept,
            state_slot: kept - 1,
        }))?;
        let after = step
            .snapshot()
            .map_err(|e| restoration_failure(format!("{e:?}")))?;
        tally.comparisons += 1;
        if let Some(difference) = exported_difference(&after, &reference) {
            return Err(restoration_failure(format!(
                "a {rows}-row window that keeps {kept} differs from {kept} one-row steps: {difference}"
            )));
        }
    }
    on(step.restore(&base))?;
    Ok(tally)
}

struct WindowRun {
    total_ms: f64,
    trace: WindowTrace,
}

/// One window of `tokens` keeping `kept` rows, with the greedy draws of the kept rows.
fn window_arm(
    step: &mut QwenResidentStep<'_>,
    tokens: &[u32],
    kept: usize,
) -> Result<WindowRun, EngError> {
    let base = step.tokens();
    let began = Instant::now();
    step.run_window(tokens)?;
    let mut greedy = Sampler::greedy();
    for row in 0..kept {
        step.draw(row, &mut greedy, SAMPLER_VOCAB)?;
    }
    step.commit_window(&Commit {
        committed_rows: base + kept,
        kept_rows: kept,
        discarded_rows: tokens.len() - kept,
        state_slot: kept - 1,
    })?;
    Ok(WindowRun {
        total_ms: ms(began),
        trace: step.window_trace.clone(),
    })
}

struct SingleRun {
    wall_ms: Vec<f64>,
    gpu_ms: Vec<Option<f64>>,
}

fn single_arm(step: &mut QwenResidentStep<'_>, tokens: &[u32]) -> Result<SingleRun, EngError> {
    let mut run = SingleRun {
        wall_ms: Vec::new(),
        gpu_ms: Vec::new(),
    };
    for &token in tokens {
        let began = Instant::now();
        step.run_token(token, false)?;
        run.wall_ms.push(ms(began));
        run.gpu_ms.push(gpu_ms(step));
    }
    Ok(run)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Arm {
    Single,
    WindowAll,
    WindowFirst,
}

/// One round's numbers, by arm.
#[derive(Default)]
struct Rounds {
    single_step_wall: Vec<f64>,
    single_step_gpu: Vec<f64>,
    single_total: Vec<f64>,
    all: Vec<WindowRun>,
    first: Vec<WindowRun>,
    export_standalone: Vec<f64>,
    import_standalone: Vec<f64>,
}

fn present(values: impl Iterator<Item = Option<f64>>) -> Vec<f64> {
    values.flatten().collect()
}

fn sum_gpu(values: &[Option<f32>]) -> Option<f64> {
    (!values.is_empty() && values.iter().all(Option::is_some))
        .then(|| values.iter().flatten().map(|v| f64::from(*v)).sum())
}

fn window_json(
    runs: &[WindowRun],
    warm_from: usize,
    rows: usize,
    kept: usize,
    shape: &StateShape,
) -> Value {
    let warm = &runs[warm_from.min(runs.len())..];
    let col = |f: &dyn Fn(&WindowRun) -> f64| -> Vec<f64> { warm.iter().map(f).collect() };
    let gpu_rows = present(warm.iter().map(|w| sum_gpu(&w.trace.row_gpu_ms)));
    let gpu_replay = present(warm.iter().map(|w| sum_gpu(&w.trace.replay_gpu_ms)));
    let host_rows = col(&|w| w.trace.row_ms.iter().sum());
    json!({
        "kept_rows": kept,
        "total_wall_ms": summary(&col(&|w| w.total_ms)),
        "export_wall_ms": summary(&col(&|w| w.trace.export_ms)),
        "rows_wall_ms": summary(&host_rows),
        "rows_gpu_event_ms": summary(&gpu_rows),
        "rows_host_minus_gpu_ms": summary(
            &warm.iter().filter_map(|w| Some(w.trace.row_ms.iter().sum::<f64>() - sum_gpu(&w.trace.row_gpu_ms)?)).collect::<Vec<_>>()
        ),
        "import_wall_ms": summary(&col(&|w| w.trace.import_ms)),
        "replay_wall_ms": summary(&col(&|w| w.trace.replay_ms.iter().sum())),
        "replay_gpu_event_ms": summary(&gpu_replay),
        "raw_total_wall_ms_all_rounds": runs.iter().map(|w| w.total_ms).collect::<Vec<_>>(),
        "transfers_per_window": WindowTransfers::new(shape, rows, kept).to_json(),
    })
}

/// Rotates the arm order each round so that no arm always runs first.
fn arm_order(round: usize) -> [Arm; 3] {
    let all = [Arm::Single, Arm::WindowAll, Arm::WindowFirst];
    [all[round % 3], all[(round + 1) % 3], all[(round + 2) % 3]]
}

struct Cell<'a> {
    text: &'a [u32],
    config: &'a Config,
    started: Instant,
}

impl Cell<'_> {
    fn over_deadline(&self) -> bool {
        self.started.elapsed().as_secs_f64() > self.config.deadline_s
    }

    fn take(&self, step: &QwenResidentStep<'_>, rows: usize) -> Vec<u32> {
        let at = step.tokens();
        self.text[at..at + rows].to_vec()
    }
}

fn run_cell(
    cell: &Cell<'_>,
    step: &mut QwenResidentStep<'static>,
    rows: usize,
) -> Result<Value, EngError> {
    let depth_start = step.tokens();
    let d1_inputs = cell.take(step, rows);
    let d1_a = full_logits(step, &d1_inputs);
    let d1_r = restoration(step, &d1_inputs);
    let d1_pass = d1_a.is_ok() && d1_r.is_ok();
    let d1 = json!({
        "pass": d1_pass,
        "full_logits": d1_a.as_ref().map(|t| t.comparisons).map_err(ToString::to_string),
        "state_restoration": d1_r.as_ref().map(|t| t.comparisons).map_err(ToString::to_string),
    });
    println!("MRV_D1 depth={depth_start} rows={rows} {d1}");
    if !d1_pass {
        return Ok(json!({"depth_start": depth_start, "rows": rows, "d1": d1, "timed": false}));
    }
    let exported = step.export_state()?;
    let shape = state_shape(step, &exported)?;
    drop(exported);
    let mut rounds = Rounds::default();
    let mut completed = 0;
    for round in 0..cell.config.rounds {
        if cell.over_deadline() {
            break;
        }
        for arm in arm_order(round) {
            let tokens = cell.take(step, rows);
            match arm {
                Arm::Single => {
                    let began = Instant::now();
                    let run = single_arm(step, &tokens)?;
                    rounds.single_total.push(ms(began));
                    rounds
                        .single_step_wall
                        .push(run.wall_ms.iter().sum::<f64>() / rows as f64);
                    if let Some(gpu) = run.gpu_ms.iter().copied().sum::<Option<f64>>() {
                        rounds.single_step_gpu.push(gpu / rows as f64);
                    }
                }
                Arm::WindowAll => rounds.all.push(window_arm(step, &tokens, rows)?),
                Arm::WindowFirst => rounds.first.push(window_arm(step, &tokens, 1)?),
            }
        }
        let began = Instant::now();
        let state = step.export_state()?;
        rounds.export_standalone.push(ms(began));
        let began = Instant::now();
        step.import_state(&state)?;
        rounds.import_standalone.push(ms(began));
        completed += 1;
    }
    Ok(cell_json(
        cell,
        step,
        rows,
        depth_start,
        &shape,
        &rounds,
        d1,
        completed,
    ))
}

#[allow(clippy::too_many_arguments)]
fn cell_json(
    cell: &Cell<'_>,
    step: &QwenResidentStep<'_>,
    rows: usize,
    depth_start: usize,
    shape: &StateShape,
    rounds: &Rounds,
    d1: Value,
    completed: usize,
) -> Value {
    let warm_from = cell.config.warm_from;
    let warm = |v: &[f64]| v[warm_from.min(v.len())..].to_vec();
    let single = median(&warm(&rounds.single_step_wall));
    let ratio = |window: &[WindowRun]| -> Value {
        let totals: Vec<f64> = window[warm_from.min(window.len())..]
            .iter()
            .map(|w| w.total_ms)
            .collect();
        match (median(&totals), single) {
            (Some(w), Some(s)) if s > 0.0 => json!({
                "over_one_captured_step": w / s,
                "over_r_captured_steps": w / (s * rows as f64),
            }),
            _ => Value::Null,
        }
    };
    json!({
        "depth_start": depth_start,
        "depth_end": step.tokens(),
        "rows": rows,
        "rounds_completed": completed,
        "warm_from": warm_from,
        "timed": true,
        "d1": d1,
        "state_bytes": shape.state_bytes(),
        "arms": {
            "captured_single_row": {
                "step_wall_ms": summary(&warm(&rounds.single_step_wall)),
                "step_gpu_event_ms": summary(&warm(&rounds.single_step_gpu)),
                "r_steps_wall_ms": summary(&warm(&rounds.single_total)),
                "raw_step_wall_ms_all_rounds": rounds.single_step_wall,
            },
            "window_accept_all": window_json(&rounds.all, warm_from, rows, rows, shape),
            "window_reject_first": window_json(&rounds.first, warm_from, rows, 1, shape),
        },
        "standalone_at_this_state": {
            "export_wall_ms": summary(&warm(&rounds.export_standalone)),
            "import_wall_ms": summary(&warm(&rounds.import_standalone)),
        },
        "ratios_measured": {
            "window_accept_all": ratio(&rounds.all),
            "window_reject_first": ratio(&rounds.first),
        },
    })
}

fn write_record(config: &Config, record: &Value) {
    let text = serde_json::to_string_pretty(record).expect("record");
    println!("MRV_RECORD {record}");
    if let Some(path) = &config.out {
        fs::write(path, text).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    }
}

/// The record so far, with identity checked.
fn record(
    config: &Config,
    identity: &Identity,
    memory: Value,
    cold: Value,
    cells: &[Value],
    stopped_early: bool,
) -> Value {
    let record = json!({
        "schema": RECORD_SCHEMA,
        "kind": "fifi-multirow-verifier-window-cost",
        "status": "measurement only; the D1 verdict of the eager window is PENDING DEVICE QUALIFICATION until the d1_window_device_tests have run",
        "identity": identity.json,
        "settings": {
            "depths": config.depths, "rows": config.rows, "rounds": config.rounds,
            "warm_from": config.warm_from, "deadline_s": config.deadline_s,
            "eos": EOS,
        },
        "memory": memory,
        "host": host_json(),
        "cold": cold,
        "cells": cells,
        "stopped_early": stopped_early,
        "notes": [
            "host wall time is the figure of cost; GPU event time is context and misses export, import, sync, read-back, acceptance and replay",
            "a row's host time cannot be split into compute and logits read-back: no event lies between them",
            "bytes and calls are computed from sizes, not counted at run time",
            "the arms advance the committed length by different amounts; depth drifts by about 2r+1 rows per round",
            "W(2) 1.6-1.9 and W(4) 2.4-3.0 of the audit are hypotheses for a K-row graph, not forecasts, and not measured here",
        ],
    });
    check_identity(&record).expect("a record needs its identity");
    record
}

fn vram_json(loaded: &Loaded, baseline_used: u64, lease_gib: u64, what: &str) -> Value {
    let info = loaded.projector.memory_info().expect("memory_info");
    let used = (info.total - info.free) as u64;
    json!({
        "when": what, "total": info.total, "free": info.free,
        "used_by_all_processes": used,
        "used_by_this_process": used.saturating_sub(baseline_used),
        "lease_bytes": lease_gib * GIB,
    })
}

/// Admission: this process stays inside the lease and leaves room for the exports.
fn admit(loaded: &Loaded, baseline_used: u64, lease_gib: u64, what: &str) -> Value {
    let vram = vram_json(loaded, baseline_used, lease_gib, what);
    let ours = vram["used_by_this_process"].as_u64().unwrap();
    assert!(
        ours <= lease_gib * GIB,
        "memory admission ({what}): {ours} B exceed the {lease_gib} GiB lease"
    );
    assert!(
        vram["free"].as_u64().unwrap() >= MIN_FREE_AFTER_BUILD,
        "memory admission ({what}): under 1 GiB of VRAM free"
    );
    vram
}

#[test]
#[ignore = "the GPU, held exclusively (21 GiB lease): loads the real 4.00 bpw pack; a measurement, one future window"]
fn verify_window_cost_record() {
    let started = Instant::now();
    let config = Config::from_env();
    let kv = kv_mode_from_env();
    let capture = capture_from_env();
    let deepest = *config.depths.last().expect("a depth");
    // The text must outlast the deepest prefill and the drift of every cell: at most 17 rows per
    // round per cell, far under the slack.
    let need = deepest + CONTEXT_SLACK;
    let (_tokenizer, stream) = text_stream(need);
    let text = &stream[..need];
    let identity = identity(&config, kv, capture, text);
    let loaded = load_pack();
    let baseline_used = loaded.baseline_used;
    let mut memory = Vec::new();
    let mut step =
        bench_step(&loaded, kv, deepest + CONTEXT_SLACK, capture, config.marks).expect("step");
    memory.push(admit(
        &loaded,
        baseline_used,
        config.lease_gib,
        "after load, state and arena",
    ));
    let cell = Cell {
        text,
        config: &config,
        started,
    };
    let mut cells = Vec::new();
    let mut cold = Value::Null;
    let mut stopped_early = false;
    let mut failed: Option<String> = None;
    'depths: for &depth in &config.depths {
        let at = step.tokens();
        let began = Instant::now();
        step.feed_rows(&text[at..depth]).expect("prefill");
        let prefill_s = began.elapsed().as_secs_f64();
        println!("MRV_PREFILL from={at} to={depth} wall_s={prefill_s:.2}");
        memory.push(admit(
            &loaded,
            baseline_used,
            config.lease_gib,
            &format!("after prefill to {depth}"),
        ));
        if cold.is_null() {
            cold = cold_record(&cell, &mut step, prefill_s);
        }
        for &rows in &config.rows {
            if cell.over_deadline() {
                stopped_early = true;
                break 'depths;
            }
            let value = run_cell(&cell, &mut step, rows).expect("cell");
            if value["timed"] == json!(false) {
                failed = Some(format!(
                    "D1 failed at depth {depth}, rows {rows}: {}",
                    value["d1"]
                ));
            }
            cells.push(value);
            write_record(
                &config,
                &record(
                    &config,
                    &identity,
                    json!(memory),
                    cold.clone(),
                    &cells,
                    stopped_early,
                ),
            );
            if failed.is_some() {
                break 'depths;
            }
        }
    }
    write_record(
        &config,
        &record(
            &config,
            &identity,
            json!(memory),
            cold,
            &cells,
            stopped_early,
        ),
    );
    assert!(failed.is_none(), "{}", failed.unwrap_or_default());
    println!(
        "MRV_COMPLETE cells={} stopped_early={stopped_early}",
        cells.len()
    );
}

/// The first single-row step (it captures the graph) and the first window of the process, before
/// any warm-up: the cold figures.
fn cold_record(cell: &Cell<'_>, step: &mut QwenResidentStep<'static>, prefill_s: f64) -> Value {
    let one = cell.take(step, 1);
    let single = single_arm(step, &one).expect("cold single step");
    let two = cell.take(step, 2);
    let window = window_arm(step, &two, 2).expect("cold window");
    json!({
        "first_prefill_wall_s": prefill_s,
        "first_single_row_step_ms": single.wall_ms[0],
        "first_window_2_rows_total_ms": window.total_ms,
        "first_window_2_rows_export_ms": window.trace.export_ms,
        "note": "the first single-row step captures the graph; the first window follows it",
    })
}
