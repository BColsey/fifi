//! The banked K-row verify window against the one-row banked decode, over the real pack.
//!
//! `FIFI_EXL3_CHECKPOINT` names the pack and `FIFI_SPEC_TOKENIZER` the tokenizer (the generation
//! state needs its decoder). The test loads the model banked (`slots = 8`, the floor — every
//! layer starts empty, so windows refill as they go), decodes the fixed prompt one row at a time
//! as the reference, then rewinds to the same saved state and decodes again through a
//! [`Speculator`] whose windows run the K-row body over the bank (fp16 KV, device rollback,
//! `draft_verify = krow` armed). The gate: the same committed tokens and an exported session
//! state that equals the one-row run's bit for bit — the K-row body's host-mapped slot picks
//! must round every layer exactly as the one-row banked step does, on a bank that misses.
//! A scripted drafter forces a full accept, a partial accept and a full reject in turn; a second
//! leg runs the engine's real n-gram drafter over the same prompt.

use std::{cell::RefCell, path::Path};

use super::{
    common_tests::{dims, inventory},
    d1_window_device_tests::{arm_rollback, arm_verify},
    dims::{KvMode, StepOptions},
    exl3_bank::Exl3SlotResolver,
    exl3_load::{BankedModel, load_exl3_banked_dir},
    exl3_moe::Exl3MoeWeights,
    exl3_step_device_tests::{EOS, GENERATED, HEAD_ROWS, PROMPT, SAMPLER_VOCAB},
    layer_table::LayerTable,
    native_device_rig::device_present,
    resources::{StepResources, moe_dims},
    rules::QwenRules,
    step::{CaptureConfig, QwenResidentStep},
};
use crate::{
    EngError,
    family::{
        executor::GenerationState,
        project::cuda::{CudaProjector, Reserve, WeightLayout},
        qwen36::settings::{RollbackMode, VerifyMode},
        sampler::Sampler,
        session::{DecodeSession, Limits},
        speculate::{
            accept::Accepted,
            driver::{Drafter, Proposal, RepeatDrafter, Speculator},
            window::Window,
        },
    },
    tokenizer::{IncrementalDecoder, Tokenizer},
};

const CONTEXT: usize = 1_024;

/// Drafts the tokens the one-row run drew, with a planned mistake in some windows, so the
/// accept path runs a full accept, a partial accept and a full reject in turn.
struct Scripted {
    reference: Vec<u32>,
    prompt_len: usize,
    windows: usize,
}

/// Per window: how many drafts and which one is wrong (`None`: all right).
const PLAN: [(usize, Option<usize>); 3] = [(2, None), (2, Some(1)), (1, Some(0))];

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

fn options() -> StepOptions {
    StepOptions {
        // The K-row body is qualified on the fp16 KV plane (`check_krow_configuration`).
        kv_mode: KvMode::Fp16,
        context_tokens: CONTEXT,
        delta_slots: 1,
        forked_experts: false,
        tiered: None,
        layout: WeightLayout::Exl3,
    }
}

fn generation(tokenizer: &Tokenizer) -> GenerationState {
    GenerationState::with_sampler(IncrementalDecoder::new(tokenizer), Sampler::greedy())
        .with_max_tokens(GENERATED as u32)
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

/// One gate: the one-row reference from the saved start, then the speculator's tokens from the
/// same start must equal it, and the session state after both must be equal too.
fn gate<D: Drafter>(
    step: &mut QwenResidentStep<'_>,
    tokenizer: &Tokenizer,
    drafter: impl FnOnce() -> Speculator<D>,
    label: &str,
) -> Result<(), EngError> {
    let mut session = DecodeSession::default();
    session.reset(step)?;
    session.prefill(step, &PROMPT, CONTEXT)?;
    let start_session = session.clone();
    let start_state = step.export_state()?;
    let reference = run(
        step,
        &mut session,
        &mut generation(tokenizer),
        None::<&mut Speculator<RepeatDrafter>>,
    )?;
    let reference_state = step.export_state()?;

    step.import_state(&start_state)?;
    let mut session = start_session;
    let mut speculator = drafter();
    let tokens = run(
        step,
        &mut session,
        &mut generation(tokenizer),
        Some(&mut speculator),
    )?;
    println!("banked krow gate {label}: {}", speculator.stats.line());
    assert_eq!(tokens, reference, "{label}: the committed tokens");
    assert!(
        step.export_state()? == reference_state,
        "{label}: the session state after the run differs from the one-row run"
    );
    Ok(())
}

#[test]
#[ignore = "requires the pinned EXL3 pack in FIFI_EXL3_CHECKPOINT, the tokenizer in \
            FIFI_SPEC_TOKENIZER and a GPU"]
fn the_banked_krow_window_commits_the_one_row_tokens_and_state() {
    let dir = std::env::var_os("FIFI_EXL3_CHECKPOINT").expect("FIFI_EXL3_CHECKPOINT");
    assert!(
        device_present("banked krow window parity"),
        "a GPU is required"
    );
    let tokenizer = Tokenizer::from_bytes(
        &std::fs::read(std::env::var_os("FIFI_SPEC_TOKENIZER").expect("FIFI_SPEC_TOKENIZER"))
            .expect("tokenizer"),
    )
    .expect("parse");

    // The one-row reference the scripted drafter replays with planned mistakes.
    let scripted = |reference: Vec<u32>| {
        Speculator::new(
            Scripted {
                reference,
                prompt_len: PROMPT.len(),
                windows: 0,
            },
            2,
        )
    };

    const SLOTS: usize = 8;
    let projector: &'static CudaProjector =
        Box::leak(Box::new(CudaProjector::new().expect("projector")));
    let model = load_exl3_banked_dir(projector, Path::new(&dir), SLOTS, Reserve::default())
        .expect("the banked pack");
    // Destructured so the store can move into the resolver while the blocks borrow the
    // weights.
    let BankedModel {
        weights,
        store,
        slots,
        experts,
        routed_bytes: _,
        dense_bytes: _,
    } = model;
    let context = projector.context();
    let mut step_options = options();
    step_options.context_tokens = (PROMPT.len() + GENERATED + 1).max(CONTEXT);
    let rules = QwenRules::new(EOS.to_vec(), SAMPLER_VOCAB, HEAD_ROWS).expect("rules");
    let resources: &'static StepResources = Box::leak(Box::new(
        StepResources::new(context, dims(), &inventory(), &step_options)
            .expect("resources")
            .with_krow(context, &step_options)
            .expect("the K-row arena"),
    ));
    let table = LayerTable::new(&dims(), &inventory()).expect("table");
    let moe = moe_dims(&dims());
    let blocks = (0..table.len())
        .map(|layer| Exl3MoeWeights::resolve_banked(&weights, &table, layer, &moe, slots))
        .collect::<Result<Vec<_>, _>>()
        .expect("bank views");
    let bank = RefCell::new(Exl3SlotResolver::new(store, &blocks, experts, 0).expect("resolver"));
    let mut step = QwenResidentStep::new_banked(
        context,
        resources,
        &weights,
        &step_options,
        rules,
        slots,
        &bank,
    )
    .expect("the banked step");
    step.set_capture_config(CaptureConfig {
        capture: false,
        section_marks: false,
    })
    .expect("capture config");
    arm_rollback(&mut step, RollbackMode::Device);
    arm_verify(&mut step, VerifyMode::Krow).expect("arm krow");
    assert!(
        step.krow_runs(3),
        "the K-row window does not run on the armed banked step"
    );

    let ngram = || Speculator::new(RepeatDrafter::new((2, 8), (4, 7)), 3);
    // A plain leg hands the scripted drafter its reference tokens.
    let mut warm = DecodeSession::default();
    warm.reset(&mut step).expect("reset");
    warm.prefill(&mut step, &PROMPT, step_options.context_tokens)
        .expect("prefill");
    let reference = run(
        &mut step,
        &mut warm,
        &mut generation(&tokenizer),
        None::<&mut Speculator<RepeatDrafter>>,
    )
    .expect("the reference leg");

    gate(
        &mut step,
        &tokenizer,
        || scripted(reference.clone()),
        "scripted",
    )
    .expect("scripted");
    gate(&mut step, &tokenizer, ngram, "ngram").expect("ngram");
}
