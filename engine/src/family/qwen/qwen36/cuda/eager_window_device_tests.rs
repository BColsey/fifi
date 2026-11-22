//! The eager verify window against the one-row step, on the whole 4.00 bpw EXL3 model, one test
//! per process on the GPU held exclusively (about 16 GiB).
//!
//! `FIFI_EXL3_CHECKPOINT` names the pack, `FIFI_SPEC_TEXT` a UTF-8 file of real source text and
//! `FIFI_SPEC_TOKENIZER` the `tokenizer.json` that encodes it. The prompt is the first 600 tokens
//! of the text. A one-row run draws `GENERATED` tokens; the same job then runs again from the same
//! saved state with a scripted drafter whose windows end in a full accept, a partial accept and a
//! full reject in turn. The gate: the same tokens, and a session state after the run that equals
//! the state of the one-row run bit for bit (KV rows, recurrent and convolution state), which is
//! the proof that a partial accept restores exactly what it should. It runs for greedy and for a
//! seeded sampler.

use std::path::Path;

use super::{
    common_tests::{dims, inventory},
    dims::{KvMode, StepOptions},
    exl3_load::load_exl3_dir,
    resources::StepResources,
    rules::QwenRules,
    step::{CaptureConfig, QwenResidentStep},
};
use crate::{
    EngError,
    family::{
        executor::GenerationState,
        project::cuda::{CudaProjector, CudaWeights, Reserve, WeightLayout},
        sampler::{Sampler, SamplingParams},
        session::{DecodeSession, Limits},
        speculate::{
            accept::Accepted,
            driver::{Drafter, Proposal, RepeatDrafter, Speculator},
            window::Window,
        },
    },
    tokenizer::{IncrementalDecoder, Tokenizer},
};

const SAMPLER_VOCAB: usize = 248_077;
const HEAD_ROWS: usize = 248_320;
const EOS: [u32; 2] = [248_046, 248_044];
const PROMPT_ROWS: usize = 600;
const GENERATED: u32 = 24;
const CONTEXT: usize = 2_048;

/// Drafts the tokens the one-row run drew, with a planned mistake in some windows.
struct Scripted {
    reference: Vec<u32>,
    prompt_len: usize,
    windows: usize,
}

/// Per window: how many drafts and which one is wrong (`None`: all right).
const PLAN: [(usize, Option<usize>); 4] = [(3, None), (3, Some(1)), (2, Some(0)), (3, Some(2))];

impl Drafter for Scripted {
    fn propose(&mut self, history: &[u32], room: usize) -> Result<Proposal, EngError> {
        let emitted = history.len() - self.prompt_len;
        let (count, wrong) = PLAN[self.windows % PLAN.len()];
        self.windows += 1;
        let count = count
            .min(room)
            .min(self.reference.len().saturating_sub(emitted));
        let mut tokens: Vec<u32> = self.reference[emitted..emitted + count].to_vec();
        if let Some(at) = wrong.filter(|&at| at < count) {
            tokens[at] = (tokens[at] + 1) % SAMPLER_VOCAB as u32;
        }
        Ok(Proposal {
            tokens,
            scores: Vec::new(),
        })
    }

    fn observe(&mut self, _window: &Window, _accepted: &Accepted) -> Result<(), EngError> {
        Ok(())
    }
}

fn rules() -> QwenRules {
    QwenRules::new(EOS.to_vec(), SAMPLER_VOCAB, HEAD_ROWS).expect("rules")
}

fn options() -> StepOptions {
    StepOptions {
        kv_mode: KvMode::Int8G32,
        context_tokens: CONTEXT,
        delta_slots: 1,
        forked_experts: false,
        tiered: None,
        layout: WeightLayout::Exl3,
    }
}

fn prompt_ids() -> (Tokenizer, Vec<u32>) {
    let path = std::env::var("FIFI_SPEC_TOKENIZER").expect("FIFI_SPEC_TOKENIZER");
    let tokenizer =
        Tokenizer::from_bytes(&std::fs::read(&path).expect("tokenizer")).expect("parse");
    let text = std::fs::read_to_string(std::env::var("FIFI_SPEC_TEXT").expect("FIFI_SPEC_TEXT"))
        .expect("text");
    let ids = tokenizer.encode(&text).expect("encode");
    (tokenizer, ids.into_iter().take(PROMPT_ROWS).collect())
}

fn generation(tokenizer: &Tokenizer, sampler: Sampler) -> GenerationState {
    GenerationState::with_sampler(IncrementalDecoder::new(tokenizer), sampler)
        .with_max_tokens(GENERATED)
}

/// Runs the job to its limit; `speculator` replaces the one-row step when present.
fn run(
    step: &mut QwenResidentStep<'_>,
    session: &mut DecodeSession,
    generation: &mut GenerationState,
    mut speculator: Option<&mut Speculator<impl Drafter>>,
) -> Result<Vec<u32>, EngError> {
    let limits = Limits {
        eos: &EOS,
        sampler_vocab: SAMPLER_VOCAB,
        context_limit: CONTEXT,
    };
    let mut tokens = Vec::new();
    for _ in 0..GENERATED {
        let batch = match speculator.as_deref_mut() {
            Some(speculator) => speculator.decode(session, step, generation, &limits)?,
            None => session.decode(step, generation, &limits)?,
        };
        tokens.extend(batch.tokens);
        if batch.finish.is_some() {
            break;
        }
    }
    Ok(tokens)
}

fn compare(
    step: &mut QwenResidentStep<'_>,
    tokenizer: &Tokenizer,
    prompt: &[u32],
    sampler: impl Fn() -> Sampler,
    label: &str,
) -> Result<(), EngError> {
    let mut session = DecodeSession::default();
    session.reset(step)?;
    session.prefill(step, prompt, CONTEXT)?;
    let start_session = session.clone();
    let start_state = step.export_state()?;
    let reference = run(
        step,
        &mut session,
        &mut generation(tokenizer, sampler()),
        None::<&mut Speculator<RepeatDrafter>>,
    )?;
    let reference_state = step.export_state()?;

    step.import_state(&start_state)?;
    let mut session = start_session;
    let mut speculator = Speculator::new(
        Scripted {
            reference: reference.clone(),
            prompt_len: prompt.len(),
            windows: 0,
        },
        3,
    );
    let tokens = run(
        step,
        &mut session,
        &mut generation(tokenizer, sampler()),
        Some(&mut speculator),
    )?;
    println!("GATE {label}: {}", speculator.stats.line());
    assert_eq!(tokens, reference, "{label}: the committed tokens");
    assert!(
        speculator
            .stats
            .acceptance()
            .is_some_and(|a| a > 0.0 && a < 1.0),
        "{label}: the script must accept some drafts and reject others"
    );
    assert!(
        step.export_state()? == reference_state,
        "{label}: the session state after the run differs from the one-row run"
    );
    Ok(())
}

/// The same job with the n-gram drafter of the engine: drafts come from the prompt and the
/// output, whatever the model does with them.
fn compare_natural(
    step: &mut QwenResidentStep<'_>,
    tokenizer: &Tokenizer,
    prompt: &[u32],
    sampler: impl Fn() -> Sampler,
    label: &str,
) -> Result<(), EngError> {
    let mut session = DecodeSession::default();
    session.reset(step)?;
    session.prefill(step, prompt, CONTEXT)?;
    let start_session = session.clone();
    let start_state = step.export_state()?;
    let reference = run(
        step,
        &mut session,
        &mut generation(tokenizer, sampler()),
        None::<&mut Speculator<RepeatDrafter>>,
    )?;
    let reference_state = step.export_state()?;

    step.import_state(&start_state)?;
    let mut session = start_session;
    let mut speculator = Speculator::new(RepeatDrafter::new((2, 8), (4, 7)), 7);
    let tokens = run(
        step,
        &mut session,
        &mut generation(tokenizer, sampler()),
        Some(&mut speculator),
    )?;
    println!("GATE_NATURAL {label}: {}", speculator.stats.line());
    assert_eq!(tokens, reference, "{label}: the committed tokens");
    assert!(
        step.export_state()? == reference_state,
        "{label}: the session state after the run differs from the one-row run"
    );
    Ok(())
}

#[test]
#[ignore = "the GPU, held exclusively: loads the real 4.00 bpw pack in FIFI_EXL3_CHECKPOINT"]
fn windows_commit_the_one_row_tokens_and_leave_the_one_row_state() {
    let dir = std::env::var_os("FIFI_EXL3_CHECKPOINT").expect("FIFI_EXL3_CHECKPOINT");
    let (tokenizer, prompt) = prompt_ids();
    let projector: &'static CudaProjector =
        Box::leak(Box::new(CudaProjector::new().expect("projector")));
    let weights: &'static CudaWeights = Box::leak(Box::new(
        load_exl3_dir(projector, Path::new(&dir), Reserve::default()).expect("load"),
    ));
    let opts = options();
    let res: &'static StepResources = Box::leak(Box::new(
        StepResources::new(projector.context(), dims(), &inventory(), &opts).expect("resources"),
    ));
    let mut step =
        QwenResidentStep::new(projector.context(), res, weights, &opts, rules()).expect("step");
    step.set_capture_config(CaptureConfig {
        capture: false,
        section_marks: false,
    })
    .expect("capture config");
    compare(&mut step, &tokenizer, &prompt, Sampler::greedy, "greedy").expect("greedy");
    let seeded = || Sampler::new(SamplingParams::new(0.8, 0.95, 20).expect("params"), 7);
    compare(&mut step, &tokenizer, &prompt, seeded, "seeded").expect("seeded");
    compare_natural(&mut step, &tokenizer, &prompt, Sampler::greedy, "greedy").expect("greedy");
    compare_natural(&mut step, &tokenizer, &prompt, seeded, "seeded").expect("seeded");
    // A prompt that ends in a repeated passage, so the n-gram drafts have something to copy.
    let mut repeated = prompt[..300].to_vec();
    repeated.extend_from_slice(&prompt[100..300]);
    compare_natural(
        &mut step,
        &tokenizer,
        &repeated,
        Sampler::greedy,
        "repeated prompt",
    )
    .expect("repeated prompt");
}
