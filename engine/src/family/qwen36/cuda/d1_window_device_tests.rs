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
//! Letters follow `D1-INVARIANCE-SPEC.md` (test B, per-layer taps, is not covered):
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
//! Isolation matters: the stop tokens are not set in the driver-level check (D), so a stop id
//! among the reference draws cannot cut a scripted window; stops have their own tests.

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
        sampler::{Sampler, SamplingParams},
        session::{DecodeSession, Limits, TokenStepper},
        speculate::{
            d1_oracle::{
                D1Subject, Fault, Setup, Tally, Verdict, accepted_prefix, failure_recovery,
                first_bit_difference, first_byte_difference, full_logits, host_greedy,
                later_rows_isolated, near_tie, ties_break_low,
            },
            driver::WindowModel,
        },
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

/// The pack on the device: loaded once, leaked for the process.
pub(super) struct Loaded {
    pub(super) projector: &'static CudaProjector,
    pub(super) weights: &'static CudaWeights,
}

pub(super) fn load_pack() -> Loaded {
    let dir = std::env::var_os("FIFI_EXL3_CHECKPOINT").expect("FIFI_EXL3_CHECKPOINT");
    let projector: &'static CudaProjector =
        Box::leak(Box::new(CudaProjector::new().expect("projector")));
    let weights: &'static CudaWeights = Box::leak(Box::new(
        load_exl3_dir(projector, Path::new(&dir), Reserve::default()).expect("load"),
    ));
    Loaded { projector, weights }
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
        // A window left over from before the restore describes a state that is gone.
        self.window = None;
        self.import_state(state)
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
}

struct Rig {
    step: QwenResidentStep<'static>,
    tokenizer: Tokenizer,
    stream: Vec<u32>,
    context: usize,
    prefixes: Vec<usize>,
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
    println!("D1_CONFIG kv={kv:?} capture={capture} context={context} prefixes={prefixes:?}");
    let step = build_step(
        &load_pack(),
        &step_options(kv, context),
        CaptureConfig {
            capture,
            section_marks: false,
        },
    )
    .expect("step");
    Rig {
        step,
        tokenizer,
        stream,
        context,
        prefixes,
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
fn report(test: &str, label: &str, verdict: Verdict<Tally>) {
    match verdict {
        Ok(tally) => println!(
            "D1_RESULT test={test} {label} PASS comparisons={} exact_ties={}",
            tally.comparisons, tally.exact_ties
        ),
        Err(failure) => panic!("D1_RESULT test={test} {label} FAIL {failure}"),
    }
}

fn check_a(rig: &mut Rig) {
    for &prefix in &rig.prefixes.clone() {
        for rows in WINDOW_ROWS {
            let _session = prefilled(&mut rig.step, &rig.stream[..prefix], rig.context);
            let inputs = window_inputs(&rig.stream, prefix, rows);
            let verdict = full_logits(&mut rig.step, &inputs);
            report("A", &format!("prefix={prefix} rows={rows}"), verdict);
        }
    }
}

/// Replacement tokens of C: ids at both ends, a stop token, the sentinel, a repeat of row 0.
fn alternatives(inputs: &[u32]) -> Vec<u32> {
    vec![0, 1, (SAMPLER_VOCAB - 1) as u32, EOS[0], 248_055, inputs[0]]
}

fn check_c(rig: &mut Rig) {
    for &prefix in &rig.prefixes.clone() {
        for rows in [4, 8] {
            let _session = prefilled(&mut rig.step, &rig.stream[..prefix], rig.context);
            let inputs = window_inputs(&rig.stream, prefix, rows);
            let verdict = later_rows_isolated(&mut rig.step, &inputs, &alternatives(&inputs));
            report("C", &format!("prefix={prefix} rows={rows}"), verdict);
        }
    }
}

fn check_d(rig: &mut Rig) {
    let Rig {
        step,
        tokenizer,
        stream,
        context,
        prefixes,
    } = rig;
    for &prefix in prefixes.iter() {
        let session = prefilled(step, &stream[..prefix], *context);
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
            report(
                "D",
                &format!("prefix={prefix} drafts={drafts} greedy"),
                greedy,
            );
            let sampled = accepted_prefix(step, &setup, drafts, &seeded);
            report(
                "D",
                &format!("prefix={prefix} drafts={drafts} seeded"),
                sampled,
            );
        }
    }
}

fn check_e(rig: &mut Rig) {
    for &prefix in &rig.prefixes.clone() {
        let _session = prefilled(&mut rig.step, &rig.stream[..prefix], rig.context);
        let inputs = window_inputs(&rig.stream, prefix, 8);
        let verdict = ties_break_low(&mut rig.step, SAMPLER_VOCAB, &inputs);
        report("E", &format!("prefix={prefix} rows=8"), verdict);
    }
}

fn check_f(rig: &mut Rig) {
    for &prefix in &rig.prefixes.clone() {
        for rows in [2, 4] {
            let _session = prefilled(&mut rig.step, &rig.stream[..prefix], rig.context);
            let inputs = window_inputs(&rig.stream, prefix, rows);
            let verdict = failure_recovery(&mut rig.step, SAMPLER_VOCAB, &inputs);
            report("F", &format!("prefix={prefix} rows={rows}"), verdict);
        }
    }
}

#[test]
#[ignore = "the GPU, held exclusively: loads the real 4.00 bpw pack in FIFI_EXL3_CHECKPOINT"]
fn d1_a_full_logits() {
    check_a(&mut rig());
}

#[test]
#[ignore = "the GPU, held exclusively: loads the real 4.00 bpw pack in FIFI_EXL3_CHECKPOINT"]
fn d1_c_later_row_isolation() {
    check_c(&mut rig());
}

#[test]
#[ignore = "the GPU, held exclusively: loads the real 4.00 bpw pack in FIFI_EXL3_CHECKPOINT"]
fn d1_d_accepted_prefix_commit_and_rollback() {
    check_d(&mut rig());
}

#[test]
#[ignore = "the GPU, held exclusively: loads the real 4.00 bpw pack in FIFI_EXL3_CHECKPOINT"]
fn d1_e_ties() {
    check_e(&mut rig());
}

#[test]
#[ignore = "the GPU, held exclusively: loads the real 4.00 bpw pack in FIFI_EXL3_CHECKPOINT"]
fn d1_f_failure_recovery() {
    check_f(&mut rig());
}

#[test]
#[ignore = "the GPU, held exclusively: loads the real 4.00 bpw pack in FIFI_EXL3_CHECKPOINT"]
fn d1_all_checks() {
    let mut rig = rig();
    check_a(&mut rig);
    check_c(&mut rig);
    check_d(&mut rig);
    check_e(&mut rig);
    check_f(&mut rig);
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
