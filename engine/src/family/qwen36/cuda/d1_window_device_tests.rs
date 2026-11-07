//! The D1 oracle against the eager verify window of the real 4.00 bpw EXL3 pack: device tests
//! that COMPILE on the host and RUN only on a device, one process per test on the GPU held
//! exclusively (about 17 GiB with a 4,096-position context).
//!
//! **Status: PENDING DEVICE QUALIFICATION.** Nothing here has run. The eager window is believed
//! bitwise by construction, because every row goes through the one-row step (`run_token`); that is
//! a design argument, not a measurement. The checks themselves are the host-tested functions of
//! `speculate::d1_oracle`, which are shown on a fake model to pass a sound model and to fail
//! each flawed one. A pass here would qualify the eager window on this pack, KV mode and capture
//! mode only, at the row counts and prefixes that ran; it says nothing about the K-row paths that
//! do not exist yet, and nothing about 8-row windows in general (8 is a cap, not a qualification).
//!
//! Letters follow the D1 invariance design of the acceptance study (test B, per-layer taps,
//! is not covered):
//!
//! | test | what | function |
//! |---|---|---|
//! | A | every window row's full logits row equals a one-row step, bit for bit | `d1_a_full_logits` |
//! | C | rows after row `i` do not move rows `0..=i`; row count does not matter; control | `d1_c_later_row_isolation` |
//! | D | for every accepted prefix `a`: state, tokens, pending, generator, next logits | `d1_d_accepted_prefix_commit_and_rollback` |
//! | E | greedy draws on rows and on the near-tie fixture | `d1_e_ties`, `d1_e_near_tie_fixture` |
//! | F | an injected fault leaves the pre-window state or a refusing model | `d1_f_failure_recovery` |
//! | all | A, C, D, E and F in one process, one pack load | `d1_all_checks` |
//!
//! State compared in D and F is the whole `ExportedState`: the recurrent state and the
//! convolution state of every DeltaNet layer, the KV planes of every attention layer for the
//! committed length, and that length. The sampler's generator state, the pending token, the
//! session history and the output budget are compared by the driver-level check.
//!
//! Environment: `FIFI_EXL3_CHECKPOINT` (the pack), `FIFI_SPEC_TEXT` and `FIFI_SPEC_TOKENIZER`
//! (real text, as in the other window tests), `FIFI_D1_PREFIX_ROWS` (comma separated prefix
//! lengths, default 600; use 255,256,257 and 4095,4096,4097 for page and chunk boundaries with
//! `FIFI_D1_CONTEXT` raised), `FIFI_D1_KV` (`fp16`, the current CUDA default, or `int8g32`),
//! `FIFI_D1_CAPTURE` (`1`, the serving default for an EXL3 entry, or `0` for the eager body),
//! `FIFI_D1_NEAR_TIE_FIXTURE` (the fixture JSON, for `d1_e_near_tie_fixture`).
//!
//! Two behaviours the checks meet, stated so a failure is read correctly:
//!
//! * A failed window does not restore the state it exported. `run_window` returns the error with
//!   the rows already fed still in the device state, and the injected fault poisons the state the
//!   way a failed step does, so every later step and snapshot is refused until a reset or an
//!   import. Check F therefore accepts a refusing model and a rolled-back state, and rejects only
//!   a state that differs from the pre-window one and still answers.
//! * The first one-row steps of a process may run the eager body, the capture and then graph
//!   replays, while a window's rows may run later ones. If the eager body, the capture and the
//!   replay differ bitwise, A fails on the first window; the failing row and element name it.
//!
//! The stop tokens are not set in the driver-level check (D), so a stop id among the reference
//! draws cannot cut a scripted window; stops have their own tests.

use std::{cell::RefCell, path::Path, rc::Rc};

use super::{
    common_tests::{dims, inventory},
    dims::{KvMode, StepOptions},
    exl3_load::load_exl3_dir,
    resources::StepResources,
    rules::QwenRules,
    state::export::ExportedState,
    step::{CaptureConfig, QwenResidentStep},
};
use crate::{
    EngError,
    family::{
        project::cuda::{CudaProjector, CudaWeights, Reserve, WeightLayout},
        qwen36::{
            rollback::Cover,
            settings::{
                QWEN36_MAX_WINDOW_ROWS, RollbackMode, VerifyMode, rollback_mode_name,
                verify_mode_name,
            },
        },
        sampler::{Sampler, SamplingParams},
        session::{DecodeSession, Limits, TokenStepper},
        speculate::{
            d1_oracle::{
                D1Subject, Fault, Setup, Tally, Verdict, accepted_prefix, describe_pass,
                failure_recovery, first_bit_difference, first_byte_difference, full_logits,
                host_faults_for, host_greedy, krow_faults_for, krow_matches_eager,
                later_rows_isolated, near_tie, repeated_rollbacks, slot_faults_for, ties_break_low,
            },
            driver::WindowModel,
            window::Commit,
        },
    },
    kernels::{
        cuda::resident::exl3::{SERVED_SINGLE_ROW, SINGLE_GATHER_VAR, SingleRow},
        single_row::{require_kernel_ran, require_served_kernel},
    },
    tokenizer::Tokenizer,
};

pub(super) const SAMPLER_VOCAB: usize = 248_077;
pub(super) const HEAD_ROWS: usize = 248_320;
pub(super) const EOS: [u32; 2] = [248_046, 248_044];
const DEFAULT_CONTEXT: usize = 4_096;
const DEFAULT_PREFIX_ROWS: usize = 600;
/// Row counts of A and E, and the drafts of D: windows of 2, 4 and 8 rows (8 is the cap).
const WINDOW_ROWS: [usize; 4] = [2, 3, 4, 8];
const DRAFTS: [usize; 3] = [1, 3, 7];

pub(super) fn rules() -> QwenRules {
    QwenRules::new(EOS.to_vec(), SAMPLER_VOCAB, HEAD_ROWS).expect("rules")
}

/// The KV mode named by `FIFI_D1_KV`: `fp16` (the CUDA default: absent or `f32` is the fp16
/// control cache) unless `int8g32` is named.
pub(super) fn kv_mode_from_env() -> KvMode {
    match std::env::var("FIFI_D1_KV").as_deref() {
        Ok("int8g32") => KvMode::Int8G32,
        Ok("fp16") | Err(_) => KvMode::Fp16,
        Ok(other) => panic!("FIFI_D1_KV must be fp16 or int8g32, got {other:?}"),
    }
}

/// `FIFI_D1_CAPTURE`: `1` (the default) captures the one-row body into a graph, as an EXL3
/// entry serves; `0` runs the eager body.
pub(super) fn capture_from_env() -> bool {
    match std::env::var("FIFI_D1_CAPTURE").as_deref() {
        Ok("0") => false,
        Ok("1") | Err(_) => true,
        Ok(other) => panic!("FIFI_D1_CAPTURE must be 0 or 1, got {other:?}"),
    }
}

/// `FIFI_D1_ROLLBACK`: `host` (the default: export, import, replay) or `device` (the per-row
/// slots of `state/rollback.rs`).
pub(super) fn rollback_from_env() -> RollbackMode {
    match std::env::var("FIFI_D1_ROLLBACK").as_deref() {
        Ok("device") => RollbackMode::Device,
        Ok("host") | Err(_) => RollbackMode::Host,
        Ok(other) => panic!("FIFI_D1_ROLLBACK must be host or device, got {other:?}"),
    }
}

/// `FIFI_D1_VERIFY`: `eager` (the default: the eager window) or `krow` (the K-row window, which
/// needs `FIFI_D1_ROLLBACK=device`).
pub(super) fn verify_from_env() -> VerifyMode {
    match std::env::var("FIFI_D1_VERIFY").as_deref() {
        Ok("krow") => VerifyMode::Krow,
        Ok("eager") | Err(_) => VerifyMode::Eager,
        Ok(other) => panic!("FIFI_D1_VERIFY must be eager or krow, got {other:?}"),
    }
}

/// Arms the device rollback on a freshly built step (windows of up to the 8-row cap), as the
/// served start does for `draft_rollback = device`. A test rig runs no prefill admission, so the
/// slots are not charged to one.
pub(super) fn arm_rollback(step: &mut QwenResidentStep<'_>, mode: RollbackMode) {
    if mode == RollbackMode::Device {
        step.request_device_rollback();
        step.enable_device_rollback(QWEN36_MAX_WINDOW_ROWS, Cover::Arena, 0)
            .expect("device rollback slots");
    }
}

/// Arms the K-row window on a step whose resources hold the K-row arena and whose device rollback
/// is armed, as the served start does for `draft_verify = krow`. The arming's own refusals are
/// the start's: `Err` when a refused switch is set.
pub(super) fn arm_verify(
    step: &mut QwenResidentStep<'_>,
    verify: VerifyMode,
) -> Result<(), EngError> {
    if verify == VerifyMode::Krow {
        step.request_krow_verify();
        step.check_krow_arming(QWEN36_MAX_WINDOW_ROWS)?;
        step.enable_krow()?;
    }
    Ok(())
}

/// The pack on the device: loaded once, leaked for the process.
pub(super) struct Loaded {
    pub(super) projector: &'static CudaProjector,
    pub(super) weights: &'static CudaWeights,
    /// VRAM in use on the card (by every process) when the projector came up, before the load:
    /// what this process adds is measured against it.
    pub(super) baseline_used: u64,
    /// The single-row EXL3 kernel the context runs (`wide1`, `legacy`, ...): recorded so that a
    /// result never leaves the kernel unsaid.
    pub(super) kernel: &'static str,
}

pub(super) fn load_pack() -> Loaded {
    load_pack_with(SERVED_SINGLE_ROW, true)
}

/// `load_pack` with the kernel registered on the context named by the caller, and the served-kernel
/// guard optional: only the negative control registers the shuffle kernel and keeps the guard off
/// so that it can show the guard's refusal.
pub(super) fn load_pack_with(default: SingleRow, enforce_served: bool) -> Loaded {
    let dir = std::env::var_os("FIFI_EXL3_CHECKPOINT").expect("FIFI_EXL3_CHECKPOINT");
    let projector: &'static CudaProjector =
        Box::leak(Box::new(CudaProjector::new().expect("projector")));
    let idle = projector.memory_info().expect("memory_info");
    let baseline_used = (idle.total - idle.free) as u64;
    // The kernel a served pack runs: the start registers wide1 on its context
    // (`start.rs`, before `load_exl3_dir`). Without it an unset FIFI_EXL3_SINGLE_GATHER leaves
    // the shuffle kernel, which the served engine does not run. An explicit value of the variable
    // still wins, as it does in the served path.
    projector.context().set_single_row_default(default);
    let kernel = projector
        .context()
        .single_row_kernel_name()
        .expect("single-row kernel setting");
    // A run on the shuffle kernel speaks for a kernel the served engine does not run: refuse it
    // here, before anything is loaded or measured, unless the operator named a kernel.
    if enforce_served {
        require_served_kernel(kernel, std::env::var_os(SINGLE_GATHER_VAR).is_some())
            .unwrap_or_else(|error| panic!("{error}"));
    }
    let weights: &'static CudaWeights = Box::leak(Box::new(
        load_exl3_dir(projector, Path::new(&dir), Reserve::default()).expect("load"),
    ));
    Loaded {
        projector,
        weights,
        baseline_used,
        kernel,
    }
}

pub(super) fn step_options(kv_mode: KvMode, context_tokens: usize) -> StepOptions {
    StepOptions {
        kv_mode,
        context_tokens,
        delta_slots: 1,
        forked_experts: false,
        tiered: None,
        layout: WeightLayout::Exl3,
    }
}

/// A session over the loaded weights.
pub(super) fn build_step(
    loaded: &Loaded,
    opts: &StepOptions,
    config: CaptureConfig,
) -> Result<QwenResidentStep<'static>, EngError> {
    build_step_for(loaded, opts, config, VerifyMode::Eager)
}

/// [`build_step`] with the K-row arena added to the resources when `verify` is `Krow`. The step
/// still has to be armed (`arm_rollback`, then `arm_verify`).
pub(super) fn build_step_for(
    loaded: &Loaded,
    opts: &StepOptions,
    config: CaptureConfig,
    verify: VerifyMode,
) -> Result<QwenResidentStep<'static>, EngError> {
    let mut resources = StepResources::new(loaded.projector.context(), dims(), &inventory(), opts)?;
    if verify == VerifyMode::Krow {
        resources = resources.with_krow(loaded.projector.context(), opts)?;
    }
    let res: &'static StepResources = Box::leak(Box::new(resources));
    let mut step = QwenResidentStep::new(
        loaded.projector.context(),
        res,
        loaded.weights,
        opts,
        rules(),
    )?;
    step.set_capture_config(config)?;
    Ok(step)
}

/// The text of `FIFI_SPEC_TEXT`, encoded by `FIFI_SPEC_TOKENIZER`: a teacher-forced stream of
/// real source text, so the routing is that of real code.
pub(super) fn text_stream(need: usize) -> (Tokenizer, Vec<u32>) {
    let path = std::env::var("FIFI_SPEC_TOKENIZER").expect("FIFI_SPEC_TOKENIZER");
    let tokenizer =
        Tokenizer::from_bytes(&std::fs::read(&path).expect("tokenizer")).expect("parse");
    let text = std::fs::read_to_string(std::env::var("FIFI_SPEC_TEXT").expect("FIFI_SPEC_TEXT"))
        .expect("text");
    let ids = tokenizer.encode(&text).expect("encode");
    assert!(
        ids.len() >= need,
        "the text holds {} tokens, the run needs {need}",
        ids.len()
    );
    assert!(
        ids.iter().all(|&id| (id as usize) < SAMPLER_VOCAB),
        "an id outside the sampler vocabulary"
    );
    (tokenizer, ids)
}

/// The first difference between two exported states, compared bit for bit: the identity, the
/// committed length, the recurrent and convolution state of each DeltaNet layer and each KV
/// plane. `None` when they are equal. (`ExportedState`'s own `==` compares floats by value, so
/// it cannot see a signed zero or a NaN payload.)
pub(super) fn exported_difference(left: &ExportedState, right: &ExportedState) -> Option<String> {
    if left.identity != right.identity {
        return Some("state identity".to_owned());
    }
    if left.tokens != right.tokens {
        return Some(format!(
            "committed length {} against {}",
            left.tokens, right.tokens
        ));
    }
    if left.delta.len() != right.delta.len() || left.kv.len() != right.kv.len() {
        return Some("layer counts".to_owned());
    }
    for (layer, (l, r)) in left.delta.iter().zip(&right.delta).enumerate() {
        if let Some(d) = first_bit_difference(&l.recurrent, &r.recurrent) {
            return Some(format!("DeltaNet layer {layer} recurrent state, {d}"));
        }
        if let Some(d) = first_bit_difference(&l.conv, &r.conv) {
            return Some(format!("DeltaNet layer {layer} convolution state, {d}"));
        }
    }
    for (layer, (l, r)) in left.kv.iter().zip(&right.kv).enumerate() {
        for (plane, (lp, rp)) in l.planes.iter().zip(&r.planes).enumerate() {
            if let Some(d) = first_byte_difference(lp, rp) {
                return Some(format!("attention layer {layer} KV plane {plane}, {d}"));
            }
        }
    }
    None
}

impl D1Subject for QwenResidentStep<'_> {
    type State = ExportedState;

    fn snapshot(&mut self) -> Result<ExportedState, EngError> {
        self.export_state()
    }

    fn restore(&mut self, state: &ExportedState) -> Result<(), EngError> {
        // A failed commit sets the recovery gate and only `reset_state` clears it, so the restore
        // is a reset followed by the import. The reset also closes a window left over from before
        // the restore (it describes a state that is gone).
        self.reset_state()?;
        self.import_state(state)
    }

    fn models_recovery_gate(&self) -> bool {
        true
    }

    fn import_without_reset(&mut self, state: &ExportedState) -> Option<Result<(), EngError>> {
        Some(self.import_state(state))
    }

    fn state_difference(&self, left: &ExportedState, right: &ExportedState) -> Option<String> {
        exported_difference(left, right)
    }

    fn window_rows(&self) -> Result<Vec<Vec<f32>>, EngError> {
        self.window_logits_rows()
    }

    fn inject_fault(&mut self, fault: Fault) {
        self.window_fault = Some(fault);
    }

    fn window_slots(&self) -> Option<Vec<Vec<u32>>> {
        self.rollback_slot_words().ok()
    }

    fn fault_plan(&self, rows: usize) -> Vec<(Fault, Commit)> {
        if self.krow_runs(rows) {
            krow_faults_for(self.tokens(), rows)
        } else if self.device_rollback_wanted() {
            slot_faults_for(self.tokens(), rows)
        } else {
            host_faults_for(self.tokens(), rows)
        }
    }
}

struct Rig {
    step: QwenResidentStep<'static>,
    tokenizer: Tokenizer,
    stream: Vec<u32>,
    context: usize,
    prefixes: Vec<usize>,
    /// The single-row kernel the context reports; the launch counts must show it.
    kernel: &'static str,
    /// How the windows of the step run: the eager window or the K-row window.
    verify: VerifyMode,
}

fn prefix_rows_from_env() -> Vec<usize> {
    std::env::var("FIFI_D1_PREFIX_ROWS")
        .unwrap_or_else(|_| DEFAULT_PREFIX_ROWS.to_string())
        .split(',')
        .map(|part| part.trim().parse().expect("a prefix length"))
        .collect()
}

fn rig() -> Rig {
    let context = std::env::var("FIFI_D1_CONTEXT")
        .ok()
        .and_then(|text| text.parse().ok())
        .unwrap_or(DEFAULT_CONTEXT);
    let prefixes = prefix_rows_from_env();
    let need = prefixes.iter().max().copied().unwrap_or(0) + 64;
    let (tokenizer, stream) = text_stream(need);
    let kv = kv_mode_from_env();
    let capture = capture_from_env();
    let loaded = load_pack();
    let rollback = rollback_from_env();
    let verify = verify_from_env();
    assert!(
        verify == VerifyMode::Eager || rollback == RollbackMode::Device,
        "FIFI_D1_VERIFY=krow needs FIFI_D1_ROLLBACK=device"
    );
    println!(
        "D1_CONFIG kv={kv:?} capture={capture} context={context} prefixes={prefixes:?} \
         single_row_kernel={} rollback={} verify={}",
        loaded.kernel,
        rollback_mode_name(rollback),
        verify_mode_name(verify)
    );
    let step = build_step_for(
        &loaded,
        &step_options(kv, context),
        CaptureConfig {
            capture,
            section_marks: false,
        },
        verify,
    )
    .expect("step");
    let mut step = step;
    arm_rollback(&mut step, rollback);
    arm_verify(&mut step, verify).unwrap_or_else(|error| panic!("{error}"));
    Rig {
        step,
        tokenizer,
        stream,
        context,
        prefixes,
        kernel: loaded.kernel,
        verify,
    }
}

/// Resets the step and prefills `prompt`: its last token stays pending.
fn prefilled(
    step: &mut QwenResidentStep<'static>,
    prompt: &[u32],
    context: usize,
) -> DecodeSession {
    let mut session = DecodeSession::default();
    session.reset(step).expect("reset");
    session.prefill(step, prompt, context).expect("prefill");
    session
}

/// The window inputs of `rows` rows after a prefix: the pending token, then the next tokens of
/// the text (teacher forced).
fn window_inputs(stream: &[u32], prefix: usize, rows: usize) -> Vec<u32> {
    stream[prefix - 1..prefix - 1 + rows].to_vec()
}

fn seeded() -> Sampler {
    Sampler::new(SamplingParams::new(0.8, 0.95, 20).expect("params"), 7)
}

/// Prints the result line and panics with the failure text. Called with no device work pending.
/// A tie check that met no exact tie prints `TIE RULE NOT EXERCISED` (no `PASS` in it), not a plain PASS.
fn report(test: &str, label: &str, verdict: Verdict<Tally>) {
    match verdict {
        Ok(tally) => println!(
            "D1_RESULT test={test} {label} {}",
            describe_pass(test, &tally)
        ),
        Err(failure) => panic!("D1_RESULT test={test} {label} FAIL {failure}"),
    }
}

/// Replacement tokens of C: ids at both ends, a stop token, the sentinel, a repeat of row 0.
fn alternatives(inputs: &[u32]) -> Vec<u32> {
    vec![0, 1, (SAMPLER_VOCAB - 1) as u32, EOS[0], 248_055, inputs[0]]
}

/// The execution-path half of a receipt, apart from the output half: the kernel the context reports
/// must have launched, and no routed call may have run the shuffle kernel. Printed as
/// `D1_KERNEL_LAUNCHES` so the log and the record carry the counts. Counts are of launches
/// enqueued or captured; a graph replay re-runs launches that were counted when they were captured.
fn assert_kernel_ran(step: &QwenResidentStep<'_>, kernel: &str, phase: &str) {
    let counts = step.kernel_launches();
    println!(
        "D1_KERNEL_LAUNCHES phase={phase} kernel={kernel} {}",
        counts.describe()
    );
    require_kernel_ran(&counts, kernel).unwrap_or_else(|error| panic!("{phase}: {error}"));
}

/// Check H: which window path the counters say ran. The eager run shows every K-row counter at
/// zero. The K-row run shows one `argmax_rows` and one `window_status_fold` per window it counted,
/// at least one pack, snapshot and strided-router launch per window, and no window that fell back
/// to the eager loop would be counted in `krow_windows`. Only the counters that observe the real
/// enqueue count here; the projection counters echo the dispatch table and are printed for the
/// record, not relied on.
pub(super) fn require_krow_enqueued(
    counts: &crate::kernels::row_launches::RowLaunches,
    windows: u64,
) -> Result<(), String> {
    if windows == 0 {
        return Err("no K-row window ran".to_owned());
    }
    for (name, count) in [
        ("argmax_rows", counts.argmax_rows),
        ("status_fold", counts.status_fold),
    ] {
        if count != windows {
            return Err(format!(
                "{name} ran {count} times in {windows} K-row windows ({})",
                counts.describe()
            ));
        }
    }
    for (name, count) in [
        ("pack_rows", counts.pack_rows),
        ("conv_snap", counts.conv_snap),
        ("delta_snap", counts.delta_snap),
        ("router_strided", counts.router_strided),
    ] {
        if count < windows {
            return Err(format!(
                "{name} ran {count} times in {windows} K-row windows ({})",
                counts.describe()
            ));
        }
    }
    Ok(())
}

fn assert_path_counters(step: &QwenResidentStep<'_>, verify: VerifyMode, phase: &str) {
    let counts = step.row_launches();
    let windows = step.krow_windows();
    println!(
        "D1_KROW_LAUNCHES phase={phase} verify={} krow_windows={windows} {}",
        verify_mode_name(verify),
        counts.describe()
    );
    let verdict = match verify {
        VerifyMode::Eager if windows != 0 => {
            Err(format!("{windows} K-row windows in an eager run"))
        }
        VerifyMode::Eager => counts.require_krow_idle(),
        VerifyMode::Krow => require_krow_enqueued(&counts, windows),
    };
    verdict.unwrap_or_else(|error| panic!("{phase}: {error}"));
}

/// Which checks a test runs.
#[derive(Clone, Copy)]
struct Which {
    a: bool,
    c: bool,
    d: bool,
    e: bool,
    f: bool,
    g: bool,
}

impl Which {
    const NONE: Self = Self {
        a: false,
        c: false,
        d: false,
        e: false,
        f: false,
        g: false,
    };
    const ALL: Self = Self {
        a: true,
        c: true,
        d: true,
        e: true,
        f: true,
        g: true,
    };
}

/// Prefills once, then runs the chosen checks from that state. Every check returns the step to
/// the state it was given, so they share the prefill; the first check starts from the state the
/// prefill built, the later ones from a copy of it that was imported back.
fn check_prefix(rig: &mut Rig, prefix: usize, which: Which) {
    let kernel = rig.kernel;
    let verify = rig.verify;
    let Rig {
        step,
        tokenizer,
        stream,
        context,
        ..
    } = rig;
    let session = prefilled(step, &stream[..prefix], *context);
    // The reference path: the prefill is one-row steps. A correct output can come from a fallback,
    // so the path is checked apart from the output.
    assert_kernel_ran(step, kernel, "reference");
    if which.a {
        for rows in WINDOW_ROWS {
            let inputs = window_inputs(stream, prefix, rows);
            let verdict = full_logits(step, &inputs);
            report("A", &format!("prefix={prefix} rows={rows}"), verdict);
        }
    }
    if which.c {
        for rows in [4, 8] {
            let inputs = window_inputs(stream, prefix, rows);
            let verdict = later_rows_isolated(step, &inputs, &alternatives(&inputs));
            report("C", &format!("prefix={prefix} rows={rows}"), verdict);
        }
    }
    if which.d {
        let setup = Setup {
            session: &session,
            tokenizer: &*tokenizer,
            limits: Limits {
                eos: &[],
                sampler_vocab: SAMPLER_VOCAB,
                context_limit: *context,
            },
        };
        for drafts in DRAFTS {
            let greedy = accepted_prefix(step, &setup, drafts, &Sampler::greedy);
            let label = format!("prefix={prefix} drafts={drafts}");
            report("D", &format!("{label} greedy"), greedy);
            let sampled = accepted_prefix(step, &setup, drafts, &seeded);
            report("D", &format!("{label} seeded"), sampled);
        }
    }
    if which.e {
        let inputs = window_inputs(stream, prefix, 8);
        let verdict = ties_break_low(step, SAMPLER_VOCAB, &inputs);
        report("E", &format!("prefix={prefix} rows=8"), verdict);
    }
    if which.f {
        for rows in [2, 4] {
            let inputs = window_inputs(stream, prefix, rows);
            let verdict = failure_recovery(step, SAMPLER_VOCAB, &inputs);
            report("F", &format!("prefix={prefix} rows={rows}"), verdict);
        }
    }
    if which.g {
        let setup = Setup {
            session: &session,
            tokenizer: &*tokenizer,
            limits: Limits {
                eos: &[],
                sampler_vocab: SAMPLER_VOCAB,
                context_limit: *context,
            },
        };
        for drafts in [3, 7] {
            let verdict = repeated_rollbacks(step, &setup, drafts, &seeded, 100);
            report(
                "G",
                &format!("prefix={prefix} drafts={drafts} windows=100"),
                verdict,
            );
        }
    }
    assert_kernel_ran(step, kernel, "window and reference");
    assert_path_counters(step, verify, "windows");
}

fn check_prefixes(rig: &mut Rig, which: Which) {
    for prefix in rig.prefixes.clone() {
        check_prefix(rig, prefix, which);
    }
}

#[test]
#[ignore = "the GPU, held exclusively: loads the real 4.00 bpw pack in FIFI_EXL3_CHECKPOINT"]
fn d1_a_full_logits() {
    check_prefixes(
        &mut rig(),
        Which {
            a: true,
            ..Which::NONE
        },
    );
}

#[test]
#[ignore = "the GPU, held exclusively: loads the real 4.00 bpw pack in FIFI_EXL3_CHECKPOINT"]
fn d1_c_later_row_isolation() {
    check_prefixes(
        &mut rig(),
        Which {
            c: true,
            ..Which::NONE
        },
    );
}

#[test]
#[ignore = "the GPU, held exclusively: loads the real 4.00 bpw pack in FIFI_EXL3_CHECKPOINT"]
fn d1_d_accepted_prefix_commit_and_rollback() {
    check_prefixes(
        &mut rig(),
        Which {
            d: true,
            ..Which::NONE
        },
    );
}

#[test]
#[ignore = "the GPU, held exclusively: loads the real 4.00 bpw pack in FIFI_EXL3_CHECKPOINT"]
fn d1_e_ties() {
    check_prefixes(
        &mut rig(),
        Which {
            e: true,
            ..Which::NONE
        },
    );
}

#[test]
#[ignore = "the GPU, held exclusively: loads the real 4.00 bpw pack in FIFI_EXL3_CHECKPOINT"]
fn d1_f_failure_recovery() {
    check_prefixes(
        &mut rig(),
        Which {
            f: true,
            ..Which::NONE
        },
    );
}

#[test]
#[ignore = "the GPU, held exclusively: loads the real 4.00 bpw pack in FIFI_EXL3_CHECKPOINT"]
fn d1_all_checks() {
    check_prefixes(&mut rig(), Which::ALL);
    println!("D1_ALL_CHECKS_COMPLETE");
}

/// The exllamav3 near tie on fifi's own numbers. The prompt and the forced prefix are the
/// fixture's 3,853 ids; row 0 of every window below is the row that predicts the tied position.
/// fifi's one-row logits need not tie there (the fixture is stock exllamav3's), so the hard
/// requirement is the D1 one: the window agrees with fifi's own one-row path, bit for bit and
/// in the draw. When fifi's one-row path also draws 494, the window must draw 494 too.
#[test]
#[ignore = "the GPU, held exclusively: loads the real pack; needs FIFI_D1_NEAR_TIE_FIXTURE"]
fn d1_e_near_tie_fixture() {
    let path = std::env::var("FIFI_D1_NEAR_TIE_FIXTURE").expect("FIFI_D1_NEAR_TIE_FIXTURE");
    let fixture = near_tie::parse(&std::fs::read_to_string(&path).expect("fixture file"))
        .expect("the fixture holds its stated facts");
    let mut rig = rig();
    let ids: Vec<u32> = [fixture.prompt.as_slice(), fixture.forced_prefix.as_slice()].concat();
    assert!(
        rig.context > ids.len() + 16,
        "FIFI_D1_CONTEXT must exceed the fixture"
    );
    let session = prefilled(&mut rig.step, &ids, rig.context);
    let pending = session.pending().expect("pending");
    let sentinel = near_tie::SENTINEL_ID;
    for drafts in [
        vec![near_tie::FORCED_TOKEN, sentinel, sentinel],
        vec![near_tie::FORCED_TOKEN],
    ] {
        let inputs: Vec<u32> = std::iter::once(pending).chain(drafts).collect();
        report(
            "A",
            &format!("near-tie rows={}", inputs.len()),
            full_logits(&mut rig.step, &inputs),
        );
        report(
            "E",
            &format!("near-tie rows={}", inputs.len()),
            ties_break_low(&mut rig.step, SAMPLER_VOCAB, &inputs),
        );
    }
    near_tie_diagnostic(&mut rig, pending);
}

/// Prints fifi's top two at the tied position and asserts the 494 expectation where it applies.
fn near_tie_diagnostic(rig: &mut Rig, pending: u32) {
    let base = rig.step.snapshot().expect("snapshot");
    let row = rig.step.logits(pending).expect("one-row logits");
    rig.step.restore(&base).expect("restore");
    let top = host_greedy(&row, SAMPLER_VOCAB);
    let tied_value = row[near_tie::TIED_IDS[0] as usize];
    println!(
        "D1_NEAR_TIE one_row_argmax={top} logit_494={} logit_1510={} tied_exactly={}",
        row[near_tie::TIED_IDS[0] as usize],
        row[near_tie::TIED_IDS[1] as usize],
        tied_value == row[near_tie::TIED_IDS[1] as usize],
    );
    if top == near_tie::FORCED_TOKEN {
        let inputs = [pending, near_tie::FORCED_TOKEN];
        rig.step.run_window(&inputs).expect("window");
        let draw = rig
            .step
            .draw(0, &mut Sampler::greedy(), SAMPLER_VOCAB)
            .expect("draw");
        rig.step.restore(&base).expect("restore");
        assert_eq!(
            draw,
            near_tie::FORCED_TOKEN,
            "the window must draw 494 at the near tie"
        );
    } else {
        println!(
            "D1_NEAR_TIE_NOTE fifi's own one-row path draws {top}, not 494: the fixture's \
             expectation belongs to stock exllamav3's numbers; the D1 comparison above stands"
        );
    }
}

/// `exported_difference` is the only comparator of real state, and it needs no device: an
/// `ExportedState` is a host value, built here the way `export_tests` builds one. One bit flipped
/// in any component must be seen, and a state equal to itself must not.
#[cfg(test)]
mod comparator_tests {
    use super::super::{dims::KvMode, export_tests::export_for};
    use super::exported_difference;

    fn state() -> super::ExportedState {
        // Fp16 KV at 10 committed rows: every attention layer has non-empty planes.
        export_for(&super::dims(), KvMode::Fp16, 64, 10).2
    }

    fn named(difference: Option<String>, part: &str) {
        let text = difference.unwrap_or_else(|| panic!("a flipped bit in {part} was not seen"));
        assert!(text.contains(part), "{part}: {text}");
    }

    #[test]
    fn equal_states_have_no_difference() {
        assert_eq!(exported_difference(&state(), &state()), None);
    }

    #[test]
    fn one_flipped_bit_is_seen_in_each_component() {
        let base = state();
        let mut m = base.clone();
        m.delta[5].recurrent[3] = f32::from_bits(m.delta[5].recurrent[3].to_bits() ^ 1);
        named(
            exported_difference(&base, &m),
            "DeltaNet layer 5 recurrent state",
        );
        let mut m = base.clone();
        m.delta[2].conv[0] = f32::from_bits(m.delta[2].conv[0].to_bits() ^ 1);
        named(
            exported_difference(&base, &m),
            "DeltaNet layer 2 convolution state",
        );
        let mut m = base.clone();
        let last = m.kv.len() - 1;
        let plane = m.kv[last].planes.len() - 1;
        m.kv[last].planes[plane][0] ^= 1;
        named(
            exported_difference(&base, &m),
            &format!("attention layer {last} KV plane {plane}"),
        );
        let mut m = base.clone();
        m.tokens += 1;
        named(exported_difference(&base, &m), "committed length");
    }

    #[test]
    fn a_signed_zero_and_a_nan_payload_are_differences() {
        let mut base = state();
        base.delta[0].recurrent[0] = 0.0;
        let mut m = base.clone();
        m.delta[0].recurrent[0] = -0.0;
        named(exported_difference(&base, &m), "recurrent state");
        base.delta[0].conv[0] = f32::from_bits(0x7fc0_0001);
        let mut m = base.clone();
        m.delta[0].conv[0] = f32::from_bits(0x7fc0_0002);
        named(exported_difference(&base, &m), "convolution state");
        assert_eq!(
            exported_difference(&base, &base.clone()),
            None,
            "equal NaN payloads agree"
        );
    }

    #[test]
    fn a_state_of_another_layer_count_is_a_difference() {
        let base = state();
        let mut m = base.clone();
        m.delta.pop();
        named(exported_difference(&base, &m), "layer counts");
    }
}

#[test]
#[ignore = "the GPU, held exclusively: loads the real 4.00 bpw pack in FIFI_EXL3_CHECKPOINT"]
fn d1_g_repeated_rollbacks() {
    check_prefixes(
        &mut rig(),
        Which {
            g: true,
            ..Which::NONE
        },
    );
}

/// The negative control of the kernel guards: with the shuffle kernel registered on the context,
/// the selection guard and the execution-path guard must both REJECT the run, and the counters must
/// see the shuffle kernel (a counter that stays at zero would make the positive result worthless).
/// A pass of this test is what makes "wide1 launched, nothing fell back" in the other tests mean
/// something. Skipped when the operator named a kernel in FIFI_EXL3_SINGLE_GATHER: the variable
/// would then win over the registration this test depends on.
#[test]
#[ignore = "the GPU, held exclusively: loads the real 4.00 bpw pack in FIFI_EXL3_CHECKPOINT"]
fn d1_negative_control_the_shuffle_kernel_is_rejected() {
    assert!(
        std::env::var_os(SINGLE_GATHER_VAR).is_none(),
        "unset {SINGLE_GATHER_VAR}: the variable would win over the registration of this control"
    );
    let loaded = load_pack_with(SingleRow::Legacy, false);
    assert_eq!(
        loaded.kernel, "legacy",
        "the control must run the shuffle kernel"
    );
    let context = DEFAULT_CONTEXT;
    let (_tokenizer, stream) = text_stream(80);
    let mut step = build_step(
        &loaded,
        &step_options(kv_mode_from_env(), context),
        CaptureConfig {
            capture: capture_from_env(),
            section_marks: false,
        },
    )
    .expect("step");
    let _session = prefilled(&mut step, &stream[..64], context);
    let counts = step.kernel_launches();
    println!("D1_NEGATIVE_CONTROL {}", counts.describe());
    assert!(
        require_served_kernel(loaded.kernel, false).is_err(),
        "the selection guard must reject the shuffle kernel"
    );
    assert!(
        require_kernel_ran(&counts, "wide1").is_err(),
        "the execution-path guard must reject a run on the shuffle kernel"
    );
    assert!(
        require_kernel_ran(&counts, "legacy").is_ok(),
        "the counters must see the shuffle kernel they were forced onto: {}",
        counts.describe()
    );
    assert_eq!(counts.wide1, 0, "{}", counts.describe());
}

/// One path of one step: the K-row window or the eager device window on the same weights and the
/// same device state. The two arms share the step, so check K compares the paths on one set of
/// weights (a second copy of the pack would not fit the card). Every call selects the arm first.
struct Arm {
    step: Rc<RefCell<QwenResidentStep<'static>>>,
    krow: bool,
}

impl Arm {
    fn with<T>(&self, f: impl FnOnce(&mut QwenResidentStep<'static>) -> T) -> T {
        let mut step = self.step.borrow_mut();
        step.use_krow(self.krow);
        f(&mut step)
    }
}

impl TokenStepper for Arm {
    const OPERATION: &'static str = <QwenResidentStep<'static> as TokenStepper>::OPERATION;

    fn feed(&mut self, token: u32) -> Result<(), EngError> {
        self.with(|step| step.feed(token))
    }

    fn logits(&mut self, token: u32) -> Result<Vec<f32>, EngError> {
        self.with(|step| step.logits(token))
    }

    fn fed(&self) -> usize {
        self.with(|step| step.fed())
    }

    fn reset(&mut self) -> Result<(), EngError> {
        self.with(|step| step.reset())
    }
}

impl WindowModel for Arm {
    fn run_window(&mut self, inputs: &[u32]) -> Result<(), EngError> {
        self.with(|step| step.run_window(inputs))
    }

    fn draw(
        &mut self,
        row: usize,
        sampler: &mut Sampler,
        sampler_vocab: usize,
    ) -> Result<u32, EngError> {
        self.with(|step| step.draw(row, sampler, sampler_vocab))
    }

    fn commit_window(&mut self, commit: &Commit) -> Result<(), EngError> {
        self.with(|step| step.commit_window(commit))
    }
}

impl D1Subject for Arm {
    type State = ExportedState;

    fn snapshot(&mut self) -> Result<ExportedState, EngError> {
        self.with(|step| step.snapshot())
    }

    fn restore(&mut self, state: &ExportedState) -> Result<(), EngError> {
        self.with(|step| step.restore(state))
    }

    fn state_difference(&self, left: &ExportedState, right: &ExportedState) -> Option<String> {
        exported_difference(left, right)
    }

    fn window_rows(&self) -> Result<Vec<Vec<f32>>, EngError> {
        self.with(|step| step.window_rows())
    }

    fn window_slots(&self) -> Option<Vec<Vec<u32>>> {
        self.with(|step| step.window_slots())
    }

    fn inject_fault(&mut self, fault: Fault) {
        self.with(|step| step.inject_fault(fault));
    }

    fn fault_plan(&self, rows: usize) -> Vec<(Fault, Commit)> {
        self.with(|step| step.fault_plan(rows))
    }

    fn models_recovery_gate(&self) -> bool {
        true
    }

    fn import_without_reset(&mut self, state: &ExportedState) -> Option<Result<(), EngError>> {
        self.with(|step| step.import_without_reset(state))
    }
}

/// `FIFI_D1_KROW_ROWS`: the window lengths check K runs (comma separated, default 2,3,4,8).
fn krow_rows_from_env() -> Vec<usize> {
    std::env::var("FIFI_D1_KROW_ROWS")
        .unwrap_or_else(|_| "2,3,4,8".to_owned())
        .split(',')
        .map(|part| part.trim().parse().expect("a window length"))
        .collect()
}

/// Check K: the K-row window against the eager device window on the same prefix and inputs, every
/// logits row, every armed slot and, after each verdict, the committed state and the next step,
/// bit for bit. Needs `FIFI_D1_VERIFY=krow` and `FIFI_D1_ROLLBACK=device`.
#[test]
#[ignore = "the GPU, held exclusively: loads the real 4.00 bpw pack in FIFI_EXL3_CHECKPOINT"]
fn d1_k_krow_against_eager() {
    super::krow_device_tests::refuse_switches().unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(
        verify_from_env(),
        VerifyMode::Krow,
        "check K needs FIFI_D1_VERIFY=krow"
    );
    let Rig {
        step,
        stream,
        context,
        prefixes,
        kernel,
        verify,
        ..
    } = rig();
    let mut step = step;
    for prefix in prefixes {
        let _session = prefilled(&mut step, &stream[..prefix], context);
        assert_kernel_ran(&step, kernel, "reference");
        let base_rows = step.tokens();
        let shared = Rc::new(RefCell::new(step));
        let mut krow = Arm {
            step: Rc::clone(&shared),
            krow: true,
        };
        let mut eager = Arm {
            step: Rc::clone(&shared),
            krow: false,
        };
        for rows in krow_rows_from_env() {
            let inputs = window_inputs(&stream, prefix, rows);
            let verdict = krow_matches_eager(&mut krow, &mut eager, &inputs, base_rows);
            report("K", &format!("prefix={prefix} rows={rows}"), verdict);
        }
        drop((krow, eager));
        step = Rc::try_unwrap(shared)
            .unwrap_or_else(|_| panic!("the arms were dropped"))
            .into_inner();
        assert_path_counters(&step, verify, "check K");
    }
}

/// A K-row window of `rows` rows with `injection` made non-finite inside its body. The window must
/// fail closed; returns the table the host read.
fn failed_window_table(
    step: &mut QwenResidentStep<'static>,
    inputs: &[u32],
    injection: super::step::Injection,
) -> crate::kernels::cuda::resident::argmax_rows::WindowResult {
    step.inject_krow(injection);
    step.run_window(inputs)
        .expect_err("a non-finite value must fail the window closed");
    step.last_krow_table().expect("the table was read")
}

/// Needs a prefilled K-row step; returns it with its session, the pre-window state and the inputs.
fn krow_rig(rows: usize) -> (Rig, DecodeSession, ExportedState, Vec<u32>) {
    super::krow_device_tests::refuse_switches().unwrap_or_else(|error| panic!("{error}"));
    let mut rig = rig();
    assert_eq!(rig.verify, VerifyMode::Krow, "needs FIFI_D1_VERIFY=krow");
    let prefix = rig.prefixes[0];
    let session = prefilled(&mut rig.step, &rig.stream[..prefix], rig.context);
    let base = rig.step.snapshot().expect("snapshot");
    let inputs = window_inputs(&rig.stream, prefix, rows);
    (rig, session, base, inputs)
}

/// The row-local status rule on the device: a non-finite value made in logits row `j` alone
/// shows in pair `j` and nowhere else, the window word carries it, and the window fails closed.
/// A pair that stored the return value of an `atomicOr` on a shared word would show the bit in
/// rows after `j` in some of the 100 windows.
#[test]
#[ignore = "the GPU, held exclusively: loads the real 4.00 bpw pack in FIFI_EXL3_CHECKPOINT"]
fn d1_krow_only_the_bad_row_reports_its_bit_over_100_windows() {
    let (mut rig, _session, base, inputs) = krow_rig(8);
    for j in [0, 3, 7] {
        for window in 0..100 {
            let table =
                failed_window_table(&mut rig.step, &inputs, super::step::Injection::LogitsNan(j));
            for (row, bits) in table.bits.iter().enumerate() {
                assert_eq!(
                    *bits,
                    if row == j { 2 } else { 0 },
                    "window {window} bad row {j}: pair {row}"
                );
            }
            assert_eq!(
                table.window & 2,
                2,
                "window {window} bad row {j}: window word"
            );
            rig.step.restore(&base).expect("restore");
        }
    }
    println!("D1_RESULT test=K-status krow_row_local_bits PASS windows=300");
}

/// A value made non-finite in the embedding of row `j` fails the window closed through the window
/// word (the EXL3 projections turn a non-finite input into zeros and raise the arithmetic flag, so
/// the pairs need not show it) and never reaches the rows before `j`: their logits equal those of
/// the clean window bit for bit.
#[test]
#[ignore = "the GPU, held exclusively: loads the real 4.00 bpw pack in FIFI_EXL3_CHECKPOINT"]
fn d1_krow_a_corrupt_hidden_state_fails_the_window_and_leaves_the_rows_before_it_clean() {
    let (mut rig, _session, base, inputs) = krow_rig(4);
    rig.step.run_window(&inputs).expect("the clean window");
    let clean = rig.step.window_logits_rows().expect("the clean rows");
    rig.step.restore(&base).expect("restore");
    for j in 0..4 {
        let table =
            failed_window_table(&mut rig.step, &inputs, super::step::Injection::HiddenNan(j));
        assert_ne!(
            table.window & 16,
            0,
            "bad row {j}: the arithmetic flag is in the window word ({:#x})",
            table.window
        );
        let rows = rig.step.last_krow_rows(inputs.len()).expect("the rows");
        for row in 0..j {
            assert_eq!(
                first_bit_difference(&rows[row], &clean[row]),
                None,
                "bad row {j}: row {row} before it moved"
            );
        }
        rig.step.restore(&base).expect("restore");
    }
}

/// A window after a failed one on the same result table: a clean 3-row window after an 8-row
/// window that left a bad pair and a window word reads none of the old bits, and after a clean
/// 8-row window a 3-row window agrees with the one-row steps.
#[test]
#[ignore = "the GPU, held exclusively: loads the real 4.00 bpw pack in FIFI_EXL3_CHECKPOINT"]
fn d1_krow_a_short_window_after_a_long_one_reads_no_stale_status() {
    let (mut rig, _session, base, inputs) = krow_rig(8);
    let table = failed_window_table(&mut rig.step, &inputs, super::step::Injection::LogitsNan(6));
    assert_eq!(table.bits[6], 2);
    rig.step.restore(&base).expect("restore");
    let step = &mut rig.step;
    report("A", "long then short, after a failed window", {
        let short = &inputs[..3];
        full_logits(step, short)
    });
    report("A", "8 rows, clean", full_logits(step, &inputs));
    report("A", "3 rows, after 8", full_logits(step, &inputs[..3]));
    let clean = step.last_krow_table().expect("table");
    assert_eq!(clean.window, 0);
    assert!(clean.bits.iter().all(|bits| *bits == 0));
}

/// The step reports the reference shared-expert block on a K-row step: the merged block is not
/// built (or was taken off by `request_krow_verify`), and the load line says so.
#[test]
#[ignore = "the GPU, held exclusively: loads the real 4.00 bpw pack in FIFI_EXL3_CHECKPOINT"]
fn d1_krow_the_step_runs_the_reference_shared_expert_block() {
    let (rig, _session, _base, _inputs) = krow_rig(2);
    let suffix = rig.step.shared_in_routed_suffix();
    println!("D1_KROW_LOAD_SUFFIX {suffix:?}");
    assert!(
        suffix.is_empty() || suffix.contains("exl3_shared_in_routed=off source=krow"),
        "a K-row step must not run the merged block: {suffix:?}"
    );
}

/// The refusal of a kernel switch at arming: with a refused switch set, arming fails and no K-row
/// window can run. Run once per value of the matrix with the variable set.
#[test]
#[ignore = "the GPU, held exclusively: loads the real 4.00 bpw pack in FIFI_EXL3_CHECKPOINT"]
fn d1_krow_arming_refuses_a_set_single_row_switch() {
    assert!(
        std::env::var_os(SINGLE_GATHER_VAR).is_some(),
        "set {SINGLE_GATHER_VAR} to one of off, 1, wide1, wide2, wide4 for this control"
    );
    let loaded = load_pack();
    let mut step = build_step_for(
        &loaded,
        &step_options(kv_mode_from_env(), DEFAULT_CONTEXT),
        CaptureConfig {
            capture: capture_from_env(),
            section_marks: false,
        },
        VerifyMode::Krow,
    )
    .expect("step");
    arm_rollback(&mut step, RollbackMode::Device);
    let error = arm_verify(&mut step, VerifyMode::Krow).expect_err("arming must be refused");
    let text = format!("{error:?}");
    assert!(text.contains("unset it"), "{text}");
    assert_eq!(step.krow_windows(), 0);
    println!("D1_RESULT test=K-switch {SINGLE_GATHER_VAR} refused: {text}");
}

/// The deliberate-break control: run with `FIFI_KROW_BREAK=ids`, which changes row 1's token on
/// the device after the host wrote the window ids, inside the K-row path only. Check A must FAIL.
/// A pass of this test shows that the K-row path is the one feeding the checks.
#[test]
#[ignore = "the GPU, held exclusively: loads the real 4.00 bpw pack in FIFI_EXL3_CHECKPOINT"]
fn d1_krow_a_deliberately_broken_window_is_caught_by_check_a() {
    assert_eq!(
        std::env::var("FIFI_KROW_BREAK").as_deref(),
        Ok("ids"),
        "set FIFI_KROW_BREAK=ids for this control"
    );
    let (mut rig, _session, _base, inputs) = krow_rig(4);
    let failure = full_logits(&mut rig.step, &inputs)
        .expect_err("a window with a changed token must differ from the one-row steps");
    println!("D1_RESULT test=K-break caught: {failure}");
    assert!(failure.violation, "{failure}");
}

/// A token outside the sampler vocabulary in any row of a K-row window is refused before the ids
/// are written or anything is enqueued: the state stays valid and unchanged, no window opens, and
/// the next window runs. Without the host check the id reaches the embedding kernel, sets status
/// 4 and poisons the state.
#[test]
#[ignore = "the GPU, held exclusively: loads the real 4.00 bpw pack in FIFI_EXL3_CHECKPOINT"]
fn d1_krow_an_out_of_vocabulary_id_is_refused_before_anything_is_enqueued() {
    let (mut rig, _session, base, inputs) = krow_rig(4);
    let windows = rig.step.krow_windows();
    for bad in [SAMPLER_VOCAB as u32, HEAD_ROWS as u32, u32::MAX] {
        for row in 0..inputs.len() {
            let mut window = inputs.clone();
            window[row] = bad;
            rig.step
                .run_window(&window)
                .expect_err("an id outside the vocabulary must be refused");
            let after = rig.step.snapshot().expect("the state stays valid");
            assert_eq!(
                exported_difference(&after, &base),
                None,
                "id {bad} in row {row} moved the state"
            );
            assert_eq!(rig.step.krow_windows(), windows, "no window ran");
        }
    }
    report(
        "A",
        "after the refusals",
        full_logits(&mut rig.step, &inputs),
    );
}

/// Check B: the residual after every mixer block and every MoE block of a K-row window against the
/// same residual of the one-row steps of the same rows, byte for byte. It is the bisect aid for a
/// failure of A, C, D or K: the first differing tap names the layer, the block and the row. Needs
/// the eager body (`FIFI_D1_CAPTURE=0`: the taps read the device inside the body) and
/// `FIFI_D1_VERIFY=krow`. Off by default and absent from the serving path.
#[test]
#[ignore = "the GPU, held exclusively: loads the real 4.00 bpw pack in FIFI_EXL3_CHECKPOINT"]
fn d1_b_per_layer_taps() {
    assert!(
        !capture_from_env(),
        "check B reads the device inside the body: set FIFI_D1_CAPTURE=0"
    );
    let (mut rig, _session, base, _inputs) = krow_rig(2);
    let prefix = rig.prefixes[0];
    for rows in [2, 4, 8] {
        let inputs = window_inputs(&rig.stream, prefix, rows);
        rig.step.start_taps();
        rig.step.run_window(&inputs).expect("the K-row window");
        let window = rig.step.take_taps();
        rig.step.restore(&base).expect("restore");
        let mut steps = Vec::with_capacity(rows);
        for &token in &inputs {
            rig.step.start_taps();
            rig.step.logits(token).expect("a one-row step");
            steps.push(rig.step.take_taps());
        }
        rig.step.restore(&base).expect("restore");
        let verdict = super::krow_device_tests::tap_difference(&window, &steps);
        match verdict {
            None => println!(
                "D1_RESULT test=B prefix={prefix} rows={rows} PASS taps={} per row",
                window.len()
            ),
            Some(difference) => panic!(
                "D1_RESULT test=B prefix={prefix} rows={rows} FAIL first differing tap: {difference}"
            ),
        }
    }
}
