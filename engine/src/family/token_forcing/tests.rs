//! Host tests of the whole gate on the toy model: a lossless candidate passes at 4,096 tokens,
//! and each injected defect is rejected by the rule that should catch it.

use super::fixture::{Bug, LAG, RING, Toy, VOCAB, generate, prompt};
use super::gate::{
    Audit, EventStatus, ExecutionIdentity, GateConfig, Generation, MarginClass, Report,
    ScoredRequest, ScoredSystem, Verdict, calibrate, evaluate, score_system,
};
use super::run::{
    Identities, Outputs, Run, RunSettings, qualify, read_scored, verdict_from_files, write_new_json,
};
use super::score::{ForcedModel, LogitBiasFault, StepperModel, TokenRecord};
use std::collections::BTreeMap;

const PROMPT: usize = 64;
const REQUESTS: u64 = 4;
const TOKENS: usize = 4096;
const AUDIT_CAP: usize = 200;

fn identity(label: &str) -> ExecutionIdentity {
    ExecutionIdentity {
        label: label.to_owned(),
        family: "toy".to_owned(),
        weights: "toy-weights-1".to_owned(),
        weight_format: "toy-f32".to_owned(),
        profile: BTreeMap::from([("execution".to_owned(), label.to_owned())]),
    }
}

fn identities() -> Identities {
    Identities {
        scorer: identity("scorer-exact"),
        reference: identity("reference-exact"),
        baseline: identity("baseline-rounded-1"),
        candidate: identity("candidate"),
    }
}

fn prompts() -> Vec<Vec<u32>> {
    (0..REQUESTS).map(|seed| prompt(seed, PROMPT)).collect()
}

fn outputs_of(model: Toy, prompts: &[Vec<u32>], tokens: usize) -> Vec<Generation> {
    outputs_with(&mut StepperModel(model), prompts, tokens)
}

fn outputs_with(
    model: &mut dyn ForcedModel,
    prompts: &[Vec<u32>],
    tokens: usize,
) -> Vec<Generation> {
    prompts
        .iter()
        .enumerate()
        .map(|(index, prompt)| Generation {
            name: format!("request-{index}"),
            prompt: prompt.clone(),
            generated: generate(model, prompt, tokens),
        })
        .collect()
}

struct Case {
    reference: Vec<Generation>,
    baseline: Vec<Generation>,
    candidate: Vec<Generation>,
}

impl Case {
    fn new(candidate: Toy, tokens: usize) -> Self {
        let prompts = prompts();
        Self {
            reference: outputs_of(Toy::exact(), &prompts, tokens),
            baseline: outputs_of(Toy::rounded(1), &prompts, tokens),
            candidate: outputs_of(candidate, &prompts, tokens),
        }
    }

    fn qualify(&self, config: GateConfig, audit_cap: usize) -> Run {
        let mut scorer = StepperModel(Toy::exact());
        let mut baseline_model = StepperModel(Toy::rounded(1));
        let outputs = Outputs {
            reference: &self.reference,
            baseline: &self.baseline,
            candidate: &self.candidate,
        };
        let settings = RunSettings {
            config,
            vocab: VOCAB,
            audit_cap,
        };
        qualify(
            (&mut scorer, Some(&mut baseline_model)),
            &identities(),
            outputs,
            settings,
        )
        .expect("the gate runs")
    }
}

fn gate(candidate: Toy) -> (Case, Report) {
    let case = Case::new(candidate, TOKENS);
    let report = case.qualify(GateConfig::QUALIFICATION, AUDIT_CAP).report;
    (case, report)
}

fn rejected(report: &Report) -> bool {
    matches!(report.verdict, Verdict::Reject { .. })
}

/// The share of requests whose first 32 generated tokens equal the reference's: a task metric
/// that only looks at the start of each answer.
fn task_accuracy(case: &Case, outputs: &[Generation]) -> f64 {
    let correct = outputs
        .iter()
        .zip(&case.reference)
        .filter(|(output, reference)| output.generated[..32] == reference.generated[..32])
        .count();
    correct as f64 / outputs.len() as f64
}

#[test]
fn a_lossless_candidate_passes_at_4096_tokens() {
    let (_, report) = gate(Toy::rounded(2));
    assert_eq!(report.verdict, Verdict::Pass, "{:#?}", report.verdict);
    assert_eq!(report.calibration.delta, 0.125);
    assert!(!report.calibration.departures.is_empty());
    assert!(report.calibration.largest_margin < 0.01);
    // The reference runs the scorer's own path; the two rounded executions flip near ties only.
    assert_eq!(report.reference.mismatches, 0);
    assert!(report.baseline.mismatches > 0 && report.candidate.mismatches > 0);
    assert_eq!(report.candidate.events, 0);
    assert_eq!(
        report.candidate.mismatches,
        report.candidate.near_tie_mismatches
    );
    assert_eq!(report.candidate.shortest_request, TOKENS);
    assert_eq!(report.candidate.tokens, REQUESTS * TOKENS as u64);
    let depth_tokens: u64 = report.candidate.depth.iter().map(|bin| bin.tokens).sum();
    assert_eq!(depth_tokens, report.candidate.tokens);
    assert!(!report.rate_rule.reject && !report.event_rule.reject);
    assert!(report.rate_rule.lower_bound < report.rate_rule.limit);
    assert!(report.rate_rule.clustered.is_some());
}

#[test]
fn a_late_logit_bias_is_rejected_though_task_accuracy_misses_it() {
    let bug = Bug::Bias {
        from: PROMPT + 2048,
        stride: 7,
        bias: 2.0,
    };
    let (case, report) = gate(Toy::new(0, bug));
    assert!(rejected(&report), "{:#?}", report.verdict);
    assert!(report.rate_rule.reject && report.event_rule.reject);
    // Every mismatch is past the 2,048th output token, so a short run would miss the fault.
    let early = &report.candidate.depth[..2];
    assert!(early.iter().all(|bin| bin.mismatches == 0), "{early:?}");
    assert!(
        report
            .candidate
            .mismatch_list
            .iter()
            .all(|m| m.position >= 2048)
    );
    // A task metric on the start of each answer cannot tell the faulty system from the baseline.
    assert!(task_accuracy(&case, &case.candidate) >= task_accuracy(&case, &case.baseline));
}

#[test]
fn the_specpilot_fault_around_a_real_model_interface_is_rejected() {
    let mut case = Case::new(Toy::exact(), TOKENS);
    let mut faulty = LogitBiasFault::specpilot(StepperModel(Toy::exact()));
    case.candidate = outputs_with(&mut faulty, &prompts(), TOKENS);
    let report = case.qualify(GateConfig::QUALIFICATION, AUDIT_CAP).report;
    assert!(
        report.rate_rule.reject && report.event_rule.reject,
        "{:#?}",
        report.verdict
    );
    assert!(
        report
            .candidate
            .mismatch_list
            .iter()
            .all(|m| m.position >= 2048)
    );
    // The wrapper resets with the model: a second request starts clean again.
    let early = &report.candidate.depth[..2];
    assert!(early.iter().all(|bin| bin.mismatches == 0), "{early:?}");
}

#[test]
fn a_ring_wraparound_fault_is_rejected() {
    let (_, report) = gate(Toy::new(0, Bug::RingWrap));
    assert!(report.rate_rule.reject, "{:#?}", report.verdict);
    let list = &report.candidate.mismatch_list;
    assert!(list.iter().all(|m| m.context_position >= RING));
}

#[test]
fn a_sliding_window_boundary_fault_is_rejected() {
    let from = PROMPT + 1500;
    let (_, report) = gate(Toy::new(0, Bug::ShortWindow { from }));
    assert!(report.rate_rule.reject, "{:#?}", report.verdict);
    let list = &report.candidate.mismatch_list;
    assert!(list.iter().all(|m| m.context_position >= from));
}

#[test]
fn a_context_position_transition_fault_is_rejected() {
    let from = PROMPT + 3000;
    let (_, report) = gate(Toy::new(0, Bug::PositionShift { from }));
    assert!(rejected(&report), "{:#?}", report.verdict);
    let list = &report.candidate.mismatch_list;
    assert!(!list.is_empty());
    assert!(list.iter().all(|m| m.context_position >= from));
}

#[test]
fn a_prefix_restore_that_loses_a_block_is_caught_by_the_event_rule() {
    let (at, span) = (PROMPT + 1000, 24);
    let (_, report) = gate(Toy::new(0, Bug::RestoreLosesBlock { at, span }));
    assert!(report.event_rule.reject, "{:#?}", report.event_rule);
    // A burst this short stays under the rate limit: only the event rule sees it.
    assert!(!report.rate_rule.reject, "{:#?}", report.rate_rule);
    let affected = at - span + LAG..at + LAG;
    let list = &report.candidate.mismatch_list;
    assert!(list.iter().all(|m| affected.contains(&m.context_position)));
}

#[test]
fn rare_large_margin_divergence_is_caught_by_the_event_rule_alone() {
    let bug = Bug::RareSpike {
        period: 997,
        boost: 6.0,
    };
    let (_, report) = gate(Toy::new(0, bug));
    assert!(!report.rate_rule.reject, "{:#?}", report.rate_rule);
    assert!(report.event_rule.reject, "{:#?}", report.event_rule);
    assert!(report.candidate.rate < report.rate_rule.limit);
    assert!(report.candidate.unexplained_events >= 8);
    assert_eq!(report.event_rule.baseline_unexplained, 0);
}

#[test]
fn a_persistent_wrong_token_is_rejected() {
    let (_, report) = gate(Toy::new(0, Bug::PersistentSwap { modulus: 4 }));
    assert!(report.rate_rule.reject, "{:#?}", report.rate_rule);
    let list = &report.candidate.mismatch_list;
    assert!(list.iter().all(|m| m.argmax % 4 == 0));
}

#[test]
fn events_past_the_audit_cap_count_against_the_candidate() {
    let bug = Bug::RareSpike {
        period: 997,
        boost: 6.0,
    };
    let case = Case::new(Toy::new(0, bug), TOKENS);
    let report = case.qualify(GateConfig::QUALIFICATION, 0).report;
    assert!(report.event_rule.reject, "{:#?}", report.event_rule);
    assert_eq!(report.candidate.unexplained_events, 0);
    assert!(report.candidate.unaudited_events >= 8);
    let events = report
        .candidate
        .mismatch_list
        .iter()
        .filter_map(|m| m.event);
    assert!(
        events
            .clone()
            .all(|status| status == EventStatus::Unaudited)
    );
    assert_eq!(events.count() as u64, report.candidate.events);
}

#[test]
fn short_requests_make_a_lossless_run_inconclusive() {
    let case = Case::new(Toy::rounded(2), 1024);
    let report = case.qualify(GateConfig::QUALIFICATION, AUDIT_CAP).report;
    let Verdict::Inconclusive { reasons } = &report.verdict else {
        panic!("{:#?}", report.verdict);
    };
    assert_eq!(reasons.len(), 2, "{reasons:?}");
    assert!(reasons.iter().all(|reason| reason.contains("1024 tokens")));
}

#[test]
fn identical_executions_cannot_calibrate_delta() {
    let prompts = prompts();
    let outputs = outputs_of(Toy::exact(), &prompts, 512);
    let mut scorer = StepperModel(Toy::exact());
    let ids = identities();
    let reference =
        score_system(&mut scorer, (&ids.reference, &ids.scorer), &outputs, VOCAB).unwrap();
    let error = calibrate(&reference, &reference).unwrap_err();
    assert!(format!("{error:?}").contains("never departed"), "{error:?}");
}

#[test]
fn other_weights_or_formats_are_refused() {
    let case = Case::new(Toy::rounded(2), 256);
    let mut scorer = StepperModel(Toy::exact());
    let ids = identities();
    let score = |scorer: &mut StepperModel<Toy>, system: &ExecutionIdentity, outputs| {
        score_system(scorer, (system, &ids.scorer), outputs, VOCAB).unwrap()
    };
    let reference = score(&mut scorer, &ids.reference, &case.reference);
    let baseline = score(&mut scorer, &ids.baseline, &case.baseline);
    let changes: [fn(&mut ExecutionIdentity); 2] = [
        |identity| identity.weights = "gguf-q4".to_owned(),
        |identity| identity.weight_format = "gguf".to_owned(),
    ];
    for change in changes {
        let mut other = ids.candidate.clone();
        change(&mut other);
        let candidate = score(&mut scorer, &other, &case.candidate);
        let error =
            evaluate(GateConfig::QUALIFICATION, &reference, &baseline, &candidate).unwrap_err();
        assert!(
            format!("{error:?}").contains("non-inferiority"),
            "{error:?}"
        );
    }
}

#[test]
fn different_prompts_or_scorers_are_refused() {
    let case = Case::new(Toy::rounded(2), 256);
    let mut scorer = StepperModel(Toy::exact());
    let ids = identities();
    let reference = score_system(
        &mut scorer,
        (&ids.reference, &ids.scorer),
        &case.reference,
        VOCAB,
    )
    .unwrap();
    let baseline = score_system(
        &mut scorer,
        (&ids.baseline, &ids.scorer),
        &case.baseline,
        VOCAB,
    )
    .unwrap();
    let candidate = score_system(
        &mut scorer,
        (&ids.candidate, &ids.scorer),
        &case.candidate,
        VOCAB,
    )
    .unwrap();
    let config = GateConfig::QUALIFICATION;
    let mut renamed = candidate.clone();
    renamed.requests[1].prompt_sha256 = "0".repeat(64);
    assert!(evaluate(config, &reference, &baseline, &renamed).is_err());
    let mut rescored = candidate;
    rescored.scorer.label = "another-scorer".to_owned();
    assert!(evaluate(config, &reference, &baseline, &rescored).is_err());
}

fn record(position: usize, generated: u32, argmax: u32, logprob: f64, top: f64) -> TokenRecord {
    TokenRecord {
        position,
        generated,
        argmax,
        runner_up: if argmax == 0 { 1 } else { 0 },
        logprob,
        argmax_logprob: top,
        top2_gap: 0.5,
    }
}

fn scored(label: &str, records: Vec<TokenRecord>) -> ScoredSystem {
    let ids = identities();
    ScoredSystem {
        system: identity(label),
        scorer: ids.scorer,
        requests: vec![ScoredRequest {
            name: "request-0".to_owned(),
            prompt_tokens: 3,
            prompt_sha256: "a".repeat(64),
            records,
            audits: Vec::new(),
        }],
    }
}

#[test]
fn delta_is_the_departure_margin_rounded_up_to_an_eighth() {
    let reference = scored(
        "reference",
        vec![record(0, 1, 1, -0.1, -0.1), record(1, 2, 2, -0.2, -0.2)],
    );
    let baseline = scored(
        "baseline",
        vec![record(0, 1, 1, -0.1, -0.1), record(1, 3, 2, -0.33, -0.2)],
    );
    let calibration = calibrate(&reference, &baseline).unwrap();
    assert_eq!(calibration.departures.len(), 1);
    assert_eq!(calibration.departures[0].position, 1);
    assert!((calibration.largest_margin - 0.13).abs() < 1e-12);
    assert_eq!(calibration.delta, 0.25);
    // An exact tie still takes the first grid step.
    let tie = scored(
        "baseline",
        vec![record(0, 1, 1, -0.1, -0.1), record(1, 3, 2, -0.2, -0.2)],
    );
    assert_eq!(calibrate(&reference, &tie).unwrap().delta, 0.125);
}

#[test]
fn calibration_refuses_a_scorer_that_gave_two_rows_for_one_context() {
    let reference = scored("reference", vec![record(0, 1, 1, -0.1, -0.1)]);
    let baseline = scored("baseline", vec![record(0, 2, 1, -0.3, -0.11)]);
    let error = calibrate(&reference, &baseline).unwrap_err();
    assert!(format!("{error:?}").contains("deterministic"), "{error:?}");
}

#[test]
fn margin_classes_split_at_delta_and_twice_delta() {
    let delta = 0.125;
    let reference = scored("reference", vec![record(0, 1, 1, -0.5, -0.5)]);
    let baseline = scored("baseline", vec![record(0, 2, 1, -0.5625, -0.5)]);
    // Margins on the binary grid, so each class edge is exact.
    let margins = [0.125, 0.1875, 0.25, 0.375];
    let records = margins
        .iter()
        .enumerate()
        .map(|(position, &margin)| record(position, 2, 1, -0.5 - margin, -0.5))
        .collect();
    let candidate = scored("candidate", records);
    let report = evaluate(
        GateConfig {
            min_tokens_per_request: 1,
        },
        &reference,
        &baseline,
        &candidate,
    )
    .unwrap();
    assert_eq!(report.calibration.delta, delta);
    let classes: Vec<_> = report
        .candidate
        .mismatch_list
        .iter()
        .map(|m| m.class)
        .collect();
    assert_eq!(
        classes,
        [
            MarginClass::NearTie,
            MarginClass::Intermediate,
            MarginClass::Intermediate,
            MarginClass::LargeMargin
        ]
    );
    assert_eq!(
        report.candidate.mismatch_list[3].event,
        Some(EventStatus::Unaudited)
    );
}

#[test]
fn an_audit_explains_tokens_the_reference_itself_produces_or_cannot_decide() {
    let event = record(9, 4, 7, -3.0, -0.1);
    let audit = |prefill_argmax, baseline_argmax| Audit {
        position: 9,
        prefill_argmax,
        baseline_argmax,
    };
    assert!(!audit(7, None).explains(&event));
    assert!(!audit(7, Some(7)).explains(&event));
    // One prefill of the whole context produces the generated token.
    assert!(audit(4, None).explains(&event));
    // The reference's choice depends on the prefill length.
    assert!(audit(2, None).explains(&event));
    // The baseline execution produces the generated token, or chooses differently.
    assert!(audit(7, Some(4)).explains(&event));
    assert!(audit(7, Some(3)).explains(&event));
}

#[test]
fn the_verdict_is_reproducible_from_the_written_files() {
    let bug = Bug::RareSpike {
        period: 997,
        boost: 6.0,
    };
    let case = Case::new(Toy::new(0, bug), TOKENS);
    let run = case.qualify(GateConfig::QUALIFICATION, AUDIT_CAP);
    let directory = tempfile::tempdir().unwrap();
    let path = |name: &str| directory.path().join(name);
    write_new_json(&path("reference.json"), &run.reference).unwrap();
    write_new_json(&path("baseline.json"), &run.baseline).unwrap();
    write_new_json(&path("candidate.json"), &run.candidate).unwrap();
    write_new_json(&path("report.json"), &run.report).unwrap();
    assert!(write_new_json(&path("report.json"), &run.report).is_err());
    assert_eq!(read_scored(&path("candidate.json")).unwrap(), run.candidate);
    let again = verdict_from_files(
        &path("reference.json"),
        &path("baseline.json"),
        &path("candidate.json"),
        GateConfig::QUALIFICATION,
    )
    .unwrap();
    assert_eq!(again, run.report);
    let json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path("report.json")).unwrap()).unwrap();
    assert_eq!(json["schema"], "fifi.token_forcing.v1");
    assert_eq!(json["verdict"]["verdict"], "reject");
    assert_eq!(json["scorer"]["label"], "scorer-exact");
    let first = &json["candidate"]["mismatch_list"][0];
    assert_eq!(first["class"], "large_margin");
    assert_eq!(first["event"], "unexplained");
}
