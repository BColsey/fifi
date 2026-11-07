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
//!   the saved state and one replayed row;
//! * `window_keep_middle`: the same with `r / 2` rows kept (windows of 4 rows or more: with 3 rows
//!   `r / 2` is 1, the reject-first verdict);
//! * the same three verdicts on the device rollback (`device_*`, the eager rows and the slot copies);
//! * with `FIFI_MRV_KROW=1` (device rollback, the K-row path armed), the same three verdicts for the
//!   K-row window as the eager K-row body (`krow_eager_*`) and as the K-row graph (`krow_graph_*`,
//!   timed only after the length captured its graph and replays it; the counters of every timed
//!   window are checked and a window that did not take its path stops the run).
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
//! `FIFI_MRV_LEASE_VRAM_GIB` (21), `FIFI_D1_KV` (fp16), `FIFI_D1_CAPTURE` (1),
//! `FIFI_MRV_ROLLBACK` (`host`, the default; `device` or `both` arm the rollback slots and add the
//! device arms `device_accept_all`, `device_reject_first`, `device_keep_middle`; `single` runs the
//! single-row arm alone, for the plain decode baseline), `FIFI_MRV_KROW` (0; 1 arms the K-row path
//! and adds the K-row arms; needs `device` or `both`), `FIFI_MRV_SYNC_CAPTURE` (0; 1 drains the
//! stream after each capture so the copy time shows in the trace).
//!
//! **The split of a window.** Every window record carries `run_window_wall_ms` (the verification:
//! the rows, or the K-row launch, the sync and the logits read-back), `draw_wall_ms` (the greedy
//! draws) and `commit_wall_ms` (the verdict: the rollback). The K-row arms copy the logits of every
//! row to the host inside `run_window` (K x 993,280 bytes, about 7.9 MB at K = 8) until the
//! device-side comparison exists, so their time is not the floor of that verifier.
//!
//! **The shared-expert block.** The single-row arm runs the block the step runs. A step armed for
//! the K-row path runs the separate shared block (the merged default yields to it); a plain step
//! runs the merged default unless `FIFI_EXL3_SHARED_IN_ROUTED=0`. The record names the block
//! (`settings.shared_expert_block`), so the plain decode step is reported both ways by two runs.
//!
//! **Timing accounting.** Besides the phases, every window record carries `drop_wall_ms` (the drop of
//! the saved host state, a `Dropped` mark in test builds), `mark_overhead_ms` (the marks' own cost),
//! the minor page faults of the whole window (`minflt_window`, read outside the timed region), the
//! faults at the marks, and `residual_ms`: the total minus every attributed phase. The record
//! also carries the launch counts of each single-row EXL3 variant (`single_row_launches`) and
//! `identity.single_row_kernel`; a cell fails if the selected kernel never launched or a routed
//! call fell back to the shuffle kernel.

use std::{fs, path::PathBuf, time::Instant};

use serde_json::{Value, json};

use super::{
    d1_window_device_tests::{
        EOS, HEAD_ROWS, Loaded, SAMPLER_VOCAB, arm_verify, build_step_for, capture_from_env,
        exported_difference, krow_graph_from_env, kv_mode_from_env, load_pack, step_options,
        text_stream,
    },
    dims::KvMode,
    eager_window::{WindowTrace, minor_faults},
    prefill_driver::{FIXED_ADMISSION_MARGIN, PrefillBudget},
    prefill_plan::Extra,
    start::prefill_headroom,
    state::export::ExportedState,
    state_layout::Word,
    step::{CaptureConfig, QwenResidentStep},
};
use crate::{
    EngError,
    family::{
        qwen36::{
            rollback::Cover,
            settings::{DEFAULT_CUDA_PREFILL_CHUNK, VerifyMode},
        },
        sampler::Sampler,
        session::TokenStepper,
        speculate::{
            d1_oracle::{D1Failure, D1Subject, Tally, Verdict, full_logits},
            driver::WindowModel,
            window::Commit,
            window_cost::{
                DeviceWindowTransfers, PackManifest, RECORD_SCHEMA, StateShape, WindowTransfers,
                check_identity, ids_sha256, median, sha256_file, summary,
            },
        },
    },
    kernels::single_row::require_kernel_ran,
};

const GIB: u64 = 1 << 30;
/// Free VRAM that must remain after the build and the arena, for the exports and the graph.
const MIN_FREE_AFTER_BUILD: u64 = GIB;
/// Positions past the deepest depth that the cache holds. Every cell starts at its depth and every
/// round returns to the position the cell started its rounds at (`run_cell`), so a cell reaches at
/// most its depth, the rows of its prepare windows, one round of every arm and one more window of
/// look-ahead: under 100 rows for 8 rows and four window paths. The slack is far above that, and
/// `a_cell_stays_inside_the_cache_and_the_text` pins the arithmetic with the real arms.
const CONTEXT_SLACK: usize = 1_536;
/// The depth the token ids of the identity run to, whatever the depths of the run.
const IDENTITY_DEPTH: usize = 32_768;

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
    /// Which rollbacks the windows rotate through: `host` (the default), `device` (slots armed,
    /// device arms only besides the single-row arm) or `both`.
    rollback: BenchRollback,
    /// Drain the stream after each capture so the copy's time shows in the trace.
    sync_capture: bool,
    /// `FIFI_MRV_KROW`: the K-row path is armed and its arms are timed.
    krow: bool,
    /// The K-row graphs may be captured (`FIFI_D1_KROW_GRAPH`, which follows `FIFI_D1_CAPTURE`).
    krow_graph: bool,
}

/// `FIFI_MRV_ROLLBACK`.
#[derive(Clone, Copy, PartialEq, Eq)]
enum BenchRollback {
    Host,
    Device,
    Both,
    /// The single-row arm alone: the plain decode baseline.
    Single,
}

impl BenchRollback {
    fn host(self) -> bool {
        matches!(self, Self::Host | Self::Both)
    }

    fn device(self) -> bool {
        matches!(self, Self::Device | Self::Both)
    }

    fn name(self) -> &'static str {
        match self {
            Self::Host => "host",
            Self::Device => "device",
            Self::Both => "both",
            Self::Single => "single",
        }
    }
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
            rollback: match std::env::var("FIFI_MRV_ROLLBACK").as_deref() {
                Ok("host") | Err(_) => BenchRollback::Host,
                Ok("device") => BenchRollback::Device,
                Ok("both") => BenchRollback::Both,
                Ok("single") => BenchRollback::Single,
                Ok(other) => {
                    panic!("FIFI_MRV_ROLLBACK must be host, device, both or single, got {other:?}")
                }
            },
            sync_capture: env_num::<u8>("FIFI_MRV_SYNC_CAPTURE", 0) == 1,
            krow: env_num::<u8>("FIFI_MRV_KROW", 0) == 1,
            krow_graph: krow_graph_from_env(capture_from_env()),
        };
        assert!(
            !config.krow || config.rollback.device(),
            "FIFI_MRV_KROW=1 needs FIFI_MRV_ROLLBACK=device or both: the K-row window rolls back on the device"
        );
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
pub(super) fn bench_step(
    loaded: &Loaded,
    kv: KvMode,
    context: usize,
    capture: bool,
    marks: bool,
    device_rows: Option<usize>,
) -> Result<QwenResidentStep<'static>, EngError> {
    bench_step_with(loaded, kv, context, capture, marks, device_rows, false)
}

/// [`bench_step`] with the K-row path armed when `krow` (it needs the device rollback slots): the
/// resources hold the K-row arena, and the arming follows the served start (the request, then the
/// slots, then `enable_krow`).
pub(super) fn bench_step_with(
    loaded: &Loaded,
    kv: KvMode,
    context: usize,
    capture: bool,
    marks: bool,
    device_rows: Option<usize>,
    krow: bool,
) -> Result<QwenResidentStep<'static>, EngError> {
    assert!(
        !krow || device_rows.is_some(),
        "the K-row path needs the device rollback slots"
    );
    let mut step = build_step_for(
        loaded,
        &step_options(kv, context),
        CaptureConfig {
            capture,
            section_marks: marks,
        },
        if krow {
            VerifyMode::Krow
        } else {
            VerifyMode::Eager
        },
    )?;
    // As `start::make`: `load_pack` registers the served single-row kernel (wide1) on the context;
    // the Q8 profile chunks at 1,024 rows with fixed admission; the fp16 profile
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
    // The device rollback slots are armed before the arena admission and charged to it as an extra,
    // as the served start does. The windows start on the host rollback; the device arms switch it.
    let mut extras = Vec::new();
    if let Some(rows) = device_rows {
        step.request_device_rollback();
        let bytes = step.enable_device_rollback(rows, Cover::Arena, 0)?;
        step.use_device_rollback(false)?;
        if krow {
            arm_verify(&mut step, VerifyMode::Krow)?;
        }
        extras.push(Extra {
            name: "device-rollback-slots",
            fixed: bytes,
            bytes_per_token: 0,
        });
    }
    let free = loaded.projector.memory_info()?.free;
    let verdict = step.admit_prefill(
        PrefillBudget {
            device_free: free.saturating_sub(headroom),
            extras,
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
    /// `run_window`: the verification (the rows or the K-row launch, the sync, the logits read).
    run_ms: f64,
    /// The greedy draws of the kept rows.
    draw_ms: f64,
    /// `commit_window`: the verdict and the rollback.
    commit_ms: f64,
    trace: WindowTrace,
    /// Minor page faults of the process across the whole window, read outside the timed region.
    minflt: Option<u64>,
    /// K-row windows, eager K-row bodies, captures and graph launches this window caused.
    krow: PathCounts,
}

/// What a window did on the K-row path, from the counters before and after it.
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
struct PathCounts {
    windows: u64,
    eager: u64,
    captures: u64,
    replays: u64,
}

impl PathCounts {
    fn now(step: &QwenResidentStep<'_>) -> Self {
        let stats = step.krow_graph_stats();
        Self {
            windows: step.krow_windows(),
            eager: stats.eager_total(),
            captures: stats.capture_total(),
            replays: stats.replay_total(),
        }
    }

    fn since(self, before: Self) -> Self {
        Self {
            windows: self.windows - before.windows,
            eager: self.eager - before.eager,
            captures: self.captures - before.captures,
            replays: self.replays - before.replays,
        }
    }

    fn to_json(self) -> Value {
        json!({
            "krow_windows": self.windows, "eager_krow_bodies": self.eager,
            "graph_captures": self.captures, "graph_launches": self.replays,
        })
    }
}

/// Which window a round times: the host rollback, the device rollback with the eager rows, or the
/// K-row window as the eager body or as the graph.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum WindowPath {
    Host,
    Device,
    KrowEager,
    KrowGraph,
}

impl WindowPath {
    fn name(self) -> &'static str {
        match self {
            Self::Host => "host",
            Self::Device => "device_eager",
            Self::KrowEager => "krow_eager",
            Self::KrowGraph => "krow_graph",
        }
    }

    fn krow(self) -> bool {
        matches!(self, Self::KrowEager | Self::KrowGraph)
    }

    /// Selects the path on a step: the rollback, the K-row body, the graphs.
    fn select(self, step: &mut QwenResidentStep<'_>) -> Result<(), EngError> {
        step.use_device_rollback(self != Self::Host)?;
        step.use_krow(self.krow());
        step.use_krow_graphs(self == Self::KrowGraph);
        Ok(())
    }

    /// What the counters of one timed window must show for this path, or the reason they do not.
    fn check(self, counts: PathCounts) -> Result<(), String> {
        let expected = match self {
            Self::Host | Self::Device => PathCounts::default(),
            Self::KrowEager => PathCounts {
                windows: 1,
                eager: 1,
                ..PathCounts::default()
            },
            Self::KrowGraph => PathCounts {
                windows: 1,
                replays: 1,
                ..PathCounts::default()
            },
        };
        if counts == expected {
            Ok(())
        } else {
            Err(format!(
                "a {} window counted {counts:?}, expected {expected:?}",
                self.name()
            ))
        }
    }
}

/// One window of `tokens` keeping `kept` rows, with the greedy draws of the kept rows, on the path
/// `path`.
fn window_arm(
    step: &mut QwenResidentStep<'_>,
    tokens: &[u32],
    kept: usize,
    path: WindowPath,
) -> Result<WindowRun, EngError> {
    path.select(step)?;
    let base = step.tokens();
    let counted = PathCounts::now(step);
    let faults_before = minor_faults();
    let began = Instant::now();
    step.run_window(tokens)?;
    let run_ms = ms(began);
    let drawing = Instant::now();
    let mut greedy = Sampler::greedy();
    for row in 0..kept {
        step.draw(row, &mut greedy, SAMPLER_VOCAB)?;
    }
    let draw_ms = ms(drawing);
    let committing = Instant::now();
    step.commit_window(&Commit {
        committed_rows: base + kept,
        kept_rows: kept,
        discarded_rows: tokens.len() - kept,
        state_slot: kept - 1,
    })?;
    let commit_ms = ms(committing);
    let total_ms = ms(began);
    let minflt = faults_before
        .zip(minor_faults())
        .map(|(a, b)| b.saturating_sub(a));
    Ok(WindowRun {
        total_ms,
        run_ms,
        draw_ms,
        commit_ms,
        trace: step.window_trace.clone(),
        minflt,
        krow: PathCounts::now(step).since(counted),
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

/// The verdict a window ends with: how many of its rows are kept.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Keep {
    /// Every row.
    All,
    /// Row 0 only.
    First,
    /// `rows / 2` rows (windows of 4 rows or more; with 3 rows it equals the first-row verdict).
    Middle,
}

impl Keep {
    fn kept(self, rows: usize) -> usize {
        match self {
            Self::All => rows,
            Self::First => 1,
            Self::Middle => rows / 2,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Arm {
    Single,
    /// A window on a path, with a verdict.
    Window(WindowPath, Keep),
}

/// One round's numbers, by arm.
#[derive(Default)]
struct Rounds {
    single_step_wall: Vec<f64>,
    single_step_gpu: Vec<f64>,
    single_total: Vec<f64>,
    windows: Vec<((WindowPath, Keep), Vec<WindowRun>)>,
    export_standalone: Vec<f64>,
    import_standalone: Vec<f64>,
}

impl Rounds {
    fn runs(&self, path: WindowPath, verdict: Keep) -> &[WindowRun] {
        self.windows
            .iter()
            .find(|(key, _)| *key == (path, verdict))
            .map_or(&[], |(_, runs)| runs.as_slice())
    }

    fn push(&mut self, path: WindowPath, verdict: Keep, run: WindowRun) {
        match self
            .windows
            .iter_mut()
            .find(|(key, _)| *key == (path, verdict))
        {
            Some((_, runs)) => runs.push(run),
            None => self.windows.push(((path, verdict), vec![run])),
        }
    }
}

fn present(values: impl Iterator<Item = Option<f64>>) -> Vec<f64> {
    values.flatten().collect()
}

fn sum_gpu(values: &[Option<f32>]) -> Option<f64> {
    (!values.is_empty() && values.iter().all(Option::is_some))
        .then(|| values.iter().flatten().map(|v| f64::from(*v)).sum())
}

/// The three parts of a window the record splits, and what its path counters showed.
fn split_json(warm: &[WindowRun]) -> Value {
    let col = |f: &dyn Fn(&WindowRun) -> f64| -> Vec<f64> { warm.iter().map(f).collect() };
    let sum = |f: &dyn Fn(&PathCounts) -> u64| -> u64 { warm.iter().map(|w| f(&w.krow)).sum() };
    json!({
        "run_window_wall_ms": summary(&col(&|w| w.run_ms)),
        "draw_wall_ms": summary(&col(&|w| w.draw_ms)),
        "commit_wall_ms": summary(&col(&|w| w.commit_ms)),
        // The total without the host draws: the greedy draws read the copied logits on the host,
        // work that a device-side comparison removes, so the K-row figure is also given without it.
        "total_without_draws_wall_ms": summary(&col(&|w| w.total_ms - w.draw_ms)),
        "path_counters_of_the_timed_windows": {
            "windows": warm.len(),
            "krow_windows": sum(&|c| c.windows),
            "eager_krow_bodies": sum(&|c| c.eager),
            "graph_captures": sum(&|c| c.captures),
            "graph_launches": sum(&|c| c.replays),
        },
    })
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
    let mut record = json!({
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
        "drop_wall_ms": summary(&col(&|w| w.trace.drop_ms)),
        "mark_overhead_ms": summary(&col(&|w| w.trace.mark_overhead_ms)),
        "minflt_window": summary(
            &warm.iter().filter_map(|w| w.minflt.map(|f| f as f64)).collect::<Vec<_>>()
        ),
        "minflt_at_marks": warm.last().map_or(Value::Null, |w| json!(w.trace.minflt)),
        "residual_ms": summary(&col(&|w| {
            w.total_ms
                - w.trace.export_ms
                - w.trace.row_ms.iter().sum::<f64>()
                - w.trace.import_ms
                - w.trace.replay_ms.iter().sum::<f64>()
                - w.trace.drop_ms
                - w.trace.mark_overhead_ms
        })),
        "replay_gpu_event_ms": summary(&gpu_replay),
        "raw_total_wall_ms_all_rounds": runs.iter().map(|w| w.total_ms).collect::<Vec<_>>(),
        "transfers_per_window": WindowTransfers::new(shape, rows, kept).to_json(),
    });
    merge(&mut record, split_json(warm));
    record
}

/// Adds the keys of `extra` to the object `record`.
fn merge(record: &mut Value, extra: Value) {
    if let (Some(into), Value::Object(from)) = (record.as_object_mut(), extra) {
        into.extend(from);
    }
}

/// The record of a device-rollback arm: the same phases, with the captures and the restore in place
/// of the export, the import and the replay, and the D2D bytes computed from the slot size.
fn device_window_json(
    runs: &[WindowRun],
    warm_from: usize,
    rows: usize,
    kept: usize,
    shape: &StateShape,
    slot_bytes: u64,
) -> Value {
    let warm = &runs[warm_from.min(runs.len())..];
    let col = |f: &dyn Fn(&WindowRun) -> f64| -> Vec<f64> { warm.iter().map(f).collect() };
    let gpu_rows = present(warm.iter().map(|w| sum_gpu(&w.trace.row_gpu_ms)));
    let mut record = json!({
        "kept_rows": kept,
        "total_wall_ms": summary(&col(&|w| w.total_ms)),
        "rows_wall_ms": summary(&col(&|w| w.trace.row_ms.iter().sum())),
        "rows_gpu_event_ms": summary(&gpu_rows),
        "capture_wall_ms": summary(&col(&|w| w.trace.capture_ms.iter().sum())),
        "restore_wall_ms": summary(&col(&|w| w.trace.import_ms)),
        "mark_overhead_ms": summary(&col(&|w| w.trace.mark_overhead_ms)),
        "minflt_window": summary(
            &warm.iter().filter_map(|w| w.minflt.map(|f| f as f64)).collect::<Vec<_>>()
        ),
        "residual_ms": summary(&col(&|w| {
            w.total_ms
                - w.trace.row_ms.iter().sum::<f64>()
                - w.trace.capture_ms.iter().sum::<f64>()
                - w.trace.import_ms
                - w.trace.mark_overhead_ms
        })),
        "capture_synced": "the captures are timed to their completion only with FIFI_MRV_SYNC_CAPTURE=1; otherwise to their enqueue",
        "raw_total_wall_ms_all_rounds": runs.iter().map(|w| w.total_ms).collect::<Vec<_>>(),
        "transfers_per_window": DeviceWindowTransfers::new(shape, slot_bytes, rows, kept).to_json(),
    });
    merge(&mut record, split_json(warm));
    record
}

/// The record of a K-row arm: the split of the window, the total, and the transfers computed from
/// sizes (the logits of every row come to the host inside `run_window`; the kernels store the slots,
/// so a verdict that keeps fewer rows than the window copies one slot back and nothing else).
fn krow_window_json(
    runs: &[WindowRun],
    warm_from: usize,
    rows: usize,
    kept: usize,
    shape: &StateShape,
    slot_bytes: u64,
    path: WindowPath,
) -> Value {
    let warm = &runs[warm_from.min(runs.len())..];
    let col = |f: &dyn Fn(&WindowRun) -> f64| -> Vec<f64> { warm.iter().map(f).collect() };
    let mut record = json!({
        "path": path.name(),
        "kept_rows": kept,
        "total_wall_ms": summary(&col(&|w| w.total_ms)),
        "restore_wall_ms": summary(&col(&|w| w.trace.import_ms)),
        "minflt_window": summary(
            &warm.iter().filter_map(|w| w.minflt.map(|f| f as f64)).collect::<Vec<_>>()
        ),
        "raw_total_wall_ms_all_rounds": runs.iter().map(|w| w.total_ms).collect::<Vec<_>>(),
        "raw_run_window_wall_ms_all_rounds": runs.iter().map(|w| w.run_ms).collect::<Vec<_>>(),
        "transfers_per_window": {
            "computed_from_sizes": true,
            "logits_d2h_bytes": (rows * HEAD_ROWS * 4) as u64,
            "result_table_d2h_bytes": (8 * rows + 8) as u64,
            "window_ids_h2d_bytes": (4 * rows) as u64,
            "start_block_h2d_bytes": shape.start_block_bytes,
            "slot_d2d_copies_for_the_verdict": u64::from(kept < rows),
            "slot_d2d_bytes": if kept < rows { slot_bytes } else { 0 },
            "graph_launches": u64::from(path == WindowPath::KrowGraph),
        },
    });
    merge(&mut record, split_json(warm));
    record
}

/// The arms of one cell, in a fixed list that each round rotates.
fn arms_of(config: &Config, rows: usize) -> Vec<Arm> {
    let mut arms = vec![Arm::Single];
    let mut paths = Vec::new();
    if config.rollback.host() {
        paths.push(WindowPath::Host);
    }
    if config.rollback.device() {
        paths.push(WindowPath::Device);
    }
    if config.krow {
        paths.push(WindowPath::KrowEager);
        if config.krow_graph {
            paths.push(WindowPath::KrowGraph);
        }
    }
    for path in paths {
        for verdict in [Keep::All, Keep::First, Keep::Middle] {
            if verdict == Keep::Middle && rows < 4 {
                continue;
            }
            arms.push(Arm::Window(path, verdict));
        }
    }
    arms
}

/// The arm order of a round: the list rotated by one step every second round and walked backwards
/// on odd rounds, so no arm always runs first and no arm always follows the same arm (a plain
/// rotation keeps every neighbour fixed, so whatever one arm leaves behind would always land on
/// the same next arm).
fn arm_order(arms: &[Arm], round: usize) -> Vec<Arm> {
    let turn = round / 2;
    let mut order: Vec<Arm> = (0..arms.len())
        .map(|i| arms[(turn + i) % arms.len()])
        .collect();
    if round % 2 == 1 {
        order.reverse();
    }
    order
}

struct Cell<'a> {
    /// The single-row kernel the context reports; the launch counts must show it.
    kernel: &'static str,
    text: &'a [u32],
    config: &'a Config,
    started: Instant,
    loaded: &'a Loaded,
}

impl Cell<'_> {
    fn over_deadline(&self) -> bool {
        self.started.elapsed().as_secs_f64() > self.config.deadline_s
    }

    fn take(&self, step: &QwenResidentStep<'_>, rows: usize) -> Vec<u32> {
        let at = step.tokens();
        self.text[at..at + rows].to_vec()
    }

    /// Free device memory now, when the device answers.
    fn free_vram(&self) -> Option<u64> {
        self.loaded
            .projector
            .memory_info()
            .ok()
            .map(|info| info.free as u64)
    }
}

/// The paths whose windows the inline D1 checks of a cell run, in order.
fn d1_paths(config: &Config) -> Vec<WindowPath> {
    let mut paths = Vec::new();
    if config.rollback.host() {
        paths.push(WindowPath::Host);
    }
    if config.rollback.device() {
        paths.push(WindowPath::Device);
    }
    if config.krow {
        paths.push(WindowPath::KrowEager);
        if config.krow_graph {
            paths.push(WindowPath::KrowGraph);
        }
    }
    paths
}

/// Untimed K-row windows of `rows` rows until the next one replays the graph of its length: the
/// first of a length is eager (it warms), the second captures and launches. Reports each window's
/// mode and time, the nodes and the device bytes of the graph, and the drop in free memory over the
/// windows. A length that already has its graph (an earlier, shallower cell) needs none.
fn prepare_graph(
    cell: &Cell<'_>,
    step: &mut QwenResidentStep<'static>,
    rows: usize,
) -> Result<Value, EngError> {
    let before = step.krow_graph_stats();
    let free_before = cell.free_vram();
    let mut windows = Vec::new();
    while step.krow_next_mode(rows)? != "replay" {
        assert!(
            windows.len() < 3,
            "the graph of {rows} rows needs more than three windows to replay"
        );
        let mode = step.krow_next_mode(rows)?;
        let tokens = cell.take(step, rows);
        let run = window_arm(step, &tokens, rows, WindowPath::KrowGraph)?;
        windows.push(json!({
            "mode": mode, "total_wall_ms": run.total_ms, "run_window_wall_ms": run.run_ms,
            "counters": run.krow.to_json(),
        }));
    }
    assert!(
        !step.krow_graph_refused(),
        "the capture of the {rows}-row graph was refused: the graph arm would time the eager body"
    );
    let after = step.krow_graph_stats();
    let drop = free_before
        .zip(cell.free_vram())
        .map(|(then, now)| then.saturating_sub(now));
    println!(
        "MRV_GRAPH rows={rows} nodes={} capture_vram_bytes={} free_vram_drop_bytes={drop:?}",
        after.nodes[rows], after.vram_bytes[rows]
    );
    Ok(json!({
        "captured_in_this_cell": after.captures[rows] > before.captures[rows],
        "windows_before_the_first_timed_replay": windows,
        "nodes": after.nodes[rows],
        "capture_instantiate_vram_bytes": after.vram_bytes[rows],
        "free_vram_drop_over_these_windows_bytes": drop,
        "refused": step.krow_graph_refused(),
    }))
}

fn run_cell(
    cell: &Cell<'_>,
    step: &mut QwenResidentStep<'static>,
    rows: usize,
) -> Result<Value, EngError> {
    let depth_start = step.tokens();
    let graph_build = if cell.config.krow && cell.config.krow_graph {
        prepare_graph(cell, step, rows)?
    } else {
        Value::Null
    };
    let d1_inputs = cell.take(step, rows);
    // The inline D1 checks run for every path in the cell: a window that is not equal to the
    // one-row steps has no meaningful cost. A K-row path must also have run K-row windows, and the
    // graph path must have launched graphs and run no eager K-row body.
    let mut d1_pass = true;
    let mut d1_modes = Vec::new();
    for path in d1_paths(cell.config) {
        path.select(step)?;
        let counted = PathCounts::now(step);
        let d1_a = full_logits(step, &d1_inputs);
        let d1_r = restoration(step, &d1_inputs);
        let seen = PathCounts::now(step).since(counted);
        let ran = match path {
            WindowPath::Host | WindowPath::Device => seen.windows == 0,
            WindowPath::KrowEager => seen.windows > 0 && seen.replays == 0 && seen.captures == 0,
            WindowPath::KrowGraph => {
                seen.windows > 0
                    && seen.eager == 0
                    && seen.captures == 0
                    && seen.replays == seen.windows
            }
        };
        d1_pass &= d1_a.is_ok() && d1_r.is_ok() && ran;
        d1_modes.push(json!({
            "rollback": if path == WindowPath::Host { "host" } else { "device" },
            "path": path.name(),
            "full_logits": d1_a.as_ref().map(|t| t.comparisons).map_err(ToString::to_string),
            "state_restoration": d1_r.as_ref().map(|t| t.comparisons).map_err(ToString::to_string),
            "the_path_ran": ran,
            "path_counters": seen.to_json(),
        }));
    }
    let d1 = if d1_modes.is_empty() {
        json!({"pass": true, "skipped": "single-row arm only"})
    } else {
        json!({
            "pass": d1_pass,
            "full_logits": d1_modes[0]["full_logits"],
            "state_restoration": d1_modes[0]["state_restoration"],
            "by_rollback": d1_modes,
        })
    };
    println!("MRV_D1 depth={depth_start} rows={rows} {d1}");
    if !d1_pass {
        return Ok(json!({"depth_start": depth_start, "rows": rows, "d1": d1, "timed": false}));
    }
    // The cell's start: every round returns to it, so each round measures at the same depth and the
    // cache and the text never run out (see `CONTEXT_SLACK`).
    let base = step.export_state()?;
    let shape = state_shape(step, &base)?;
    let mut rounds = Rounds::default();
    let mut completed = 0;
    let arms = arms_of(cell.config, rows);
    for round in 0..cell.config.rounds {
        if cell.over_deadline() {
            break;
        }
        for arm in arm_order(&arms, round) {
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
                Arm::Window(path, keep) => {
                    let run = window_arm(step, &tokens, keep.kept(rows), path)?;
                    if let Err(why) = path.check(run.krow) {
                        return Err(EngError::BadRequest(format!(
                            "depth {depth_start}, {rows} rows, round {round}: {why}"
                        )));
                    }
                    rounds.push(path, keep, run);
                }
            }
        }
        if cell.config.rollback.host() {
            let began = Instant::now();
            let state = step.export_state()?;
            rounds.export_standalone.push(ms(began));
            let began = Instant::now();
            step.import_state(&state)?;
            rounds.import_standalone.push(ms(began));
        }
        step.restore(&base)?;
        completed += 1;
    }
    // The execution-path half of the receipt: the selected kernel launched and nothing routed fell
    // back to the shuffle kernel, for the reference steps and the windows of this cell.
    let launches = step.kernel_launches();
    println!(
        "MRV_KERNEL_LAUNCHES depth={depth_start} rows={rows} {}",
        launches.describe()
    );
    require_kernel_ran(&launches, cell.kernel).unwrap_or_else(|error| panic!("{error}"));
    let mut cell = cell_json(
        cell,
        step,
        rows,
        depth_start,
        &shape,
        &rounds,
        d1,
        completed,
    );
    cell["graph_build"] = graph_build;
    Ok(cell)
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
    let slot_bytes =
        step.state_layout().delta.total as u64 / step.state_layout().delta.slots as u64;
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
    let host = |keep: Keep| {
        window_json(
            rounds.runs(WindowPath::Host, keep),
            warm_from,
            rows,
            keep.kept(rows),
            shape,
        )
    };
    let device = |keep: Keep| {
        device_window_json(
            rounds.runs(WindowPath::Device, keep),
            warm_from,
            rows,
            keep.kept(rows),
            shape,
            slot_bytes,
        )
    };
    let krow = |path: WindowPath, keep: Keep| {
        krow_window_json(
            rounds.runs(path, keep),
            warm_from,
            rows,
            keep.kept(rows),
            shape,
            slot_bytes,
            path,
        )
    };
    let ratios = |path: WindowPath, keep: Keep| ratio(rounds.runs(path, keep));
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
            "window_accept_all": host(Keep::All),
            "window_reject_first": host(Keep::First),
            "window_keep_middle": host(Keep::Middle),
            "device_accept_all": device(Keep::All),
            "device_reject_first": device(Keep::First),
            "device_keep_middle": device(Keep::Middle),
            "krow_eager_accept_all": krow(WindowPath::KrowEager, Keep::All),
            "krow_eager_reject_first": krow(WindowPath::KrowEager, Keep::First),
            "krow_eager_keep_middle": krow(WindowPath::KrowEager, Keep::Middle),
            "krow_graph_accept_all": krow(WindowPath::KrowGraph, Keep::All),
            "krow_graph_reject_first": krow(WindowPath::KrowGraph, Keep::First),
            "krow_graph_keep_middle": krow(WindowPath::KrowGraph, Keep::Middle),
        },
        "single_row_launches": step.kernel_launches().describe(),
        "standalone_at_this_state": {
            "export_wall_ms": summary(&warm(&rounds.export_standalone)),
            "import_wall_ms": summary(&warm(&rounds.import_standalone)),
        },
        "ratios_measured": {
            "window_accept_all": ratios(WindowPath::Host, Keep::All),
            "window_reject_first": ratios(WindowPath::Host, Keep::First),
            "window_keep_middle": ratios(WindowPath::Host, Keep::Middle),
            "device_accept_all": ratios(WindowPath::Device, Keep::All),
            "device_reject_first": ratios(WindowPath::Device, Keep::First),
            "device_keep_middle": ratios(WindowPath::Device, Keep::Middle),
            "krow_eager_accept_all": ratios(WindowPath::KrowEager, Keep::All),
            "krow_eager_reject_first": ratios(WindowPath::KrowEager, Keep::First),
            "krow_eager_keep_middle": ratios(WindowPath::KrowEager, Keep::Middle),
            "krow_graph_accept_all": ratios(WindowPath::KrowGraph, Keep::All),
            "krow_graph_reject_first": ratios(WindowPath::KrowGraph, Keep::First),
            "krow_graph_keep_middle": ratios(WindowPath::KrowGraph, Keep::Middle),
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
            "rollback": config.rollback.name(), "sync_capture": config.sync_capture,
            "krow": config.krow, "krow_graph": config.krow && config.krow_graph,
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
    if config.krow {
        // A K-row run is qualified without the switches that select another kernel or MoE body.
        super::krow_device_tests::refuse_switches().unwrap_or_else(|error| panic!("{error}"));
    }
    let kv = kv_mode_from_env();
    let capture = capture_from_env();
    let deepest = *config.depths.last().expect("a depth");
    // The text must outlast the deepest prefill and the rows a cell reaches past its depth (under
    // 100, see `CONTEXT_SLACK`).
    // The ids hashed into the identity are the same whatever depths a run takes (a run of the
    // shallow cells and a run of the deep cells carry one `ids_sha256`): the text is hashed to the
    // deepest depth the plans use.
    let need = deepest.max(IDENTITY_DEPTH) + CONTEXT_SLACK;
    let (_tokenizer, stream) = text_stream(need);
    let text = &stream[..need];
    let mut identity = identity(&config, kv, capture, text);
    let loaded = load_pack();
    // The effective single-row kernel (registered default or explicit variable), so an unset
    // variable never leaves it unsaid; `check_record.py` requires the field.
    identity.json["single_row_kernel"] = Value::String(loaded.kernel.to_owned());
    println!("MRV_CONFIG single_row_kernel={}", loaded.kernel);
    let baseline_used = loaded.baseline_used;
    let mut memory = Vec::new();
    let mut step = bench_step_with(
        &loaded,
        kv,
        deepest + CONTEXT_SLACK,
        capture,
        config.marks,
        config
            .rollback
            .device()
            .then(|| config.rows.iter().copied().max().unwrap_or(2).max(2)),
        config.krow,
    )
    .expect("step");
    step.trace_sync_capture = config.sync_capture;
    step.use_krow_graphs(config.krow_graph);
    // The block the step runs, so the single-row figure is never read for the other block.
    identity.json["shared_expert_block"] =
        Value::String(step.shared_in_routed_suffix().trim().to_owned());
    identity.json["krow"] = json!({
        "armed": config.krow, "graphs": config.krow && step.krow_graphs_on(),
    });
    println!(
        "MRV_CONFIG shared_expert_block={:?} krow={} krow_graphs={}",
        step.shared_in_routed_suffix().trim(),
        config.krow,
        step.krow_graphs_on()
    );
    memory.push(admit(
        &loaded,
        baseline_used,
        config.lease_gib,
        "after load, state and arena",
    ));
    let cell = Cell {
        kernel: loaded.kernel,
        text,
        config: &config,
        started,
        loaded: &loaded,
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
        // The state at the depth: every cell of the depth starts from it, so each cell is measured
        // at its stated depth and the cells do not push each other past the cache.
        let depth_base = step.export_state().expect("the state at the depth");
        if cold.is_null() {
            cold = cold_record(&cell, &mut step, prefill_s);
        }
        for &rows in &config.rows {
            if cell.over_deadline() {
                stopped_early = true;
                break 'depths;
            }
            step.restore(&depth_base).expect("back to the depth");
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
    memory.push(admit(
        &loaded,
        baseline_used,
        config.lease_gib,
        "after the cells (every K-row graph captured)",
    ));
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
    let path = if cell.config.rollback == BenchRollback::Device {
        WindowPath::Device
    } else {
        WindowPath::Host
    };
    let window = (cell.config.rollback != BenchRollback::Single)
        .then(|| window_arm(step, &two, 2, path).expect("cold window"));
    // The calibration of the device copy: 200 drained copies of one slot (GPU, outside any window).
    let copy = if cell.config.rollback.device() {
        step.use_device_rollback(true).expect("armed");
        let times = step
            .device_copy_microbench(200)
            .expect("copy microbenchmark");
        let slot = step.state_layout().delta.total as f64 / step.state_layout().delta.slots as f64;
        json!({
            "slot_bytes": slot,
            "host_wall_ms_each_drained": summary(&times),
            "gb_per_s_counting_the_copy_once": median(&times).filter(|m| *m > 0.0).map(|m| slot / 1e9 / (m / 1e3)),
        })
    } else {
        Value::Null
    };
    json!({
        "d2d_slot_copy": copy,
        "first_prefill_wall_s": prefill_s,
        "first_single_row_step_ms": single.wall_ms[0],
        "first_window_2_rows_total_ms": window.as_ref().map(|w| w.total_ms),
        "first_window_2_rows_export_ms": window.as_ref().map(|w| w.trace.export_ms),
        "note": "the first single-row step captures the graph; the first window follows it",
    })
}

fn test_config(rollback: BenchRollback, krow: bool, krow_graph: bool) -> Config {
    Config {
        depths: vec![8192],
        rows: vec![2],
        rounds: 12,
        warm_from: 2,
        out: None,
        commit: "test".to_owned(),
        marks: false,
        deadline_s: 1.0,
        lease_gib: 21,
        rollback,
        sync_capture: false,
        krow,
        krow_graph,
    }
}

#[test]
fn the_verdicts_keep_all_the_first_or_the_middle_rows() {
    for rows in 2..=8 {
        assert_eq!(Keep::All.kept(rows), rows);
        assert_eq!(Keep::First.kept(rows), 1);
        assert_eq!(Keep::Middle.kept(rows), rows / 2);
    }
    // The middle verdict is only run where it differs from the first row.
    assert_eq!(Keep::Middle.kept(3), 1);
}

#[test]
fn a_cell_times_the_host_device_and_k_row_windows_with_every_verdict() {
    let config = test_config(BenchRollback::Both, true, true);
    let arms = arms_of(&config, 4);
    // The single-row arm, and four paths with three verdicts each.
    assert_eq!(arms.len(), 1 + 4 * 3);
    for path in [
        WindowPath::Host,
        WindowPath::Device,
        WindowPath::KrowEager,
        WindowPath::KrowGraph,
    ] {
        for keep in [Keep::All, Keep::First, Keep::Middle] {
            assert!(arms.contains(&Arm::Window(path, keep)));
        }
    }
    // Windows of 2 and 3 rows have no middle verdict: with 3 rows it keeps one row, the first-row
    // verdict.
    for rows in [2, 3] {
        let short = arms_of(&config, rows);
        assert_eq!(short.len(), 1 + 4 * 2);
        assert!(!short.contains(&Arm::Window(WindowPath::KrowGraph, Keep::Middle)));
    }
}

#[test]
fn the_graph_arm_exists_only_when_graphs_may_be_captured_and_the_single_arm_stands_alone() {
    let eager_only = arms_of(&test_config(BenchRollback::Both, true, false), 3);
    assert!(eager_only.contains(&Arm::Window(WindowPath::KrowEager, Keep::All)));
    assert!(!eager_only.contains(&Arm::Window(WindowPath::KrowGraph, Keep::All)));
    let single = arms_of(&test_config(BenchRollback::Single, false, true), 8);
    assert_eq!(single, vec![Arm::Single]);
    // No K-row arm without the K-row path.
    let plain = arms_of(&test_config(BenchRollback::Both, false, true), 4);
    assert!(
        !plain
            .iter()
            .any(|arm| matches!(arm, Arm::Window(path, _) if path.krow()))
    );
}

#[test]
fn every_round_runs_every_arm_once_and_no_arm_keeps_its_neighbour() {
    let arms = arms_of(&test_config(BenchRollback::Both, true, true), 4);
    let rounds = 12;
    let mut firsts = std::collections::HashSet::new();
    let mut followers: std::collections::HashMap<String, std::collections::HashSet<String>> =
        std::collections::HashMap::new();
    for round in 0..rounds {
        let order = arm_order(&arms, round);
        assert_eq!(order.len(), arms.len());
        for arm in &arms {
            assert_eq!(order.iter().filter(|other| *other == arm).count(), 1);
        }
        firsts.insert(format!("{:?}", order[0]));
        for pair in order.windows(2) {
            followers
                .entry(format!("{:?}", pair[0]))
                .or_default()
                .insert(format!("{:?}", pair[1]));
        }
    }
    assert!(firsts.len() >= 4, "the first arm changes: {firsts:?}");
    // No arm always follows the same arm: a cyclic shift would leave every arm exactly one
    // follower (the carry-over of one arm would always land on the same next arm).
    for (arm, after) in followers {
        if arm != format!("{:?}", arms[arms.len() - 1]) {
            assert!(after.len() >= 2, "{arm} always precedes {after:?}");
        }
    }
}

/// The rows a round advances the committed length by, summed over the arms of a cell: the single-row
/// arm runs `rows` steps, a window commits the rows it keeps.
fn rows_per_round(arms: &[Arm], rows: usize) -> usize {
    arms.iter()
        .map(|arm| match arm {
            Arm::Single => rows,
            Arm::Window(_, keep) => keep.kept(rows),
        })
        .sum()
}

/// The furthest position a cell reaches under the policy of the run (every round returns to the
/// start of the cell): it starts at its depth plus the rows of its prepare windows (three windows of
/// `rows` rows at most), one round advances `rows_per_round`, and the last window of the round
/// needs `rows` more positions of look-ahead.
fn cell_extent(depth: usize, arms: &[Arm], rows: usize) -> usize {
    depth + 3 * rows + rows_per_round(arms, rows) + rows
}

/// The old defect, computed with the real arms: cells run back to back, every round continuing from
/// the last, the position of the next cell being the end of the previous one. At 8K the 8-row cell
/// then starts about 100 positions before the end of the cache and runs out of it within two rounds.
#[test]
fn without_the_return_to_the_start_the_cells_of_a_depth_overrun_the_cache() {
    let config = test_config(BenchRollback::Both, true, true);
    for depth in [8_192_usize, 32_768] {
        let capacity = depth + CONTEXT_SLACK;
        let mut at = depth;
        let mut over = false;
        for rows in [2_usize, 3, 4, 6, 8] {
            let arms = arms_of(&config, rows);
            at += 3 * rows + config.rounds * rows_per_round(&arms, rows);
            over |= at + rows > capacity;
        }
        assert!(
            over,
            "depth {depth}: the continuing cells end at {at}, capacity {capacity}"
        );
    }
}

/// The policy of the run: every cell starts at its depth and every round returns to the start of
/// the cell. Each cell then stays inside the cache and inside the text the identity run hashes.
#[test]
fn a_cell_stays_inside_the_cache_and_the_text() {
    let config = test_config(BenchRollback::Both, true, true);
    let text = IDENTITY_DEPTH.max(32_768) + CONTEXT_SLACK;
    for depth in [8_192_usize, 32_768] {
        let capacity = depth + CONTEXT_SLACK;
        for rows in [2_usize, 3, 4, 6, 8] {
            let arms = arms_of(&config, rows);
            let extent = cell_extent(depth, &arms, rows);
            assert!(
                extent <= capacity,
                "depth {depth}, {rows} rows: {extent} against {capacity}"
            );
            assert!(
                extent <= text,
                "depth {depth}, {rows} rows: {extent} against {text} ids"
            );
            assert!(
                extent - depth < 100,
                "depth {depth}, {rows} rows: reaches {} past its depth",
                extent - depth
            );
        }
    }
}

#[test]
fn a_timed_window_must_show_the_counters_of_its_own_path() {
    let one = |windows, eager, captures, replays| PathCounts {
        windows,
        eager,
        captures,
        replays,
    };
    // The host and device windows run no K-row window.
    assert!(WindowPath::Host.check(PathCounts::default()).is_ok());
    assert!(WindowPath::Device.check(PathCounts::default()).is_ok());
    assert!(WindowPath::Device.check(one(1, 1, 0, 0)).is_err());
    // The eager K-row body: one window, one eager body, nothing captured or launched.
    assert!(WindowPath::KrowEager.check(one(1, 1, 0, 0)).is_ok());
    assert!(WindowPath::KrowEager.check(PathCounts::default()).is_err());
    assert!(WindowPath::KrowEager.check(one(1, 0, 0, 1)).is_err());
    // The graph: one window, one launch, no eager body and no capture in the timed window. A
    // window that fell back to the eager body, or that captured, is not a replay timing.
    assert!(WindowPath::KrowGraph.check(one(1, 0, 0, 1)).is_ok());
    assert!(WindowPath::KrowGraph.check(one(1, 1, 0, 0)).is_err());
    assert!(WindowPath::KrowGraph.check(one(1, 0, 1, 1)).is_err());
    assert!(WindowPath::KrowGraph.check(one(2, 0, 0, 2)).is_err());
}
