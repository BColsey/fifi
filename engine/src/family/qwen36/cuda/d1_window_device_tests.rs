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

use std::path::Path;

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
            settings::{QWEN36_MAX_WINDOW_ROWS, RollbackMode, rollback_mode_name},
        },
        sampler::{Sampler, SamplingParams},
        session::{DecodeSession, Limits, TokenStepper},
        speculate::{
            d1_oracle::{
                D1Subject, Fault, Setup, Tally, Verdict, accepted_prefix, describe_pass,
                failure_recovery, first_bit_difference, first_byte_difference, full_logits,
                host_faults_for, host_greedy, later_rows_isolated, near_tie, repeated_rollbacks,
                slot_faults_for, ties_break_low,
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
    let res: &'static StepResources = Box::leak(Box::new(StepResources::new(
        loaded.projector.context(),
        dims(),
        &inventory(),
        opts,
    )?));
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

    fn fault_plan(&self, rows: usize) -> Vec<(Fault, Commit)> {
        if self.device_rollback_wanted() {
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
    println!(
        "D1_CONFIG kv={kv:?} capture={capture} context={context} prefixes={prefixes:?} \
         single_row_kernel={} rollback={}",
        loaded.kernel,
        rollback_mode_name(rollback)
    );
    let step = build_step(
        &loaded,
        &step_options(kv, context),
        CaptureConfig {
            capture,
            section_marks: false,
        },
    )
    .expect("step");
    let mut step = step;
    arm_rollback(&mut step, rollback);
    Rig {
        step,
        tokenizer,
        stream,
        context,
        prefixes,
        kernel: loaded.kernel,
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
