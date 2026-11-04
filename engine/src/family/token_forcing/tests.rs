//! Host tests of the whole gate on the toy model: a lossless candidate passes at 4,096 tokens,
//! and each injected defect is rejected by the rule that should catch it.

use super::fixture::{Bug, LAG, RING, Toy, VOCAB, generate, prompt};
use super::gate::{
    Audit, AuditPlan, EventStatus, ExecutionIdentity, GateConfig, Generation, MarginClass, Report,
    Reread, ScoredRequest, ScoredSystem, Verdict, audit_system, calibrate, evaluate, score_system,
};
use super::run::{
    Candidate, Comparison, Identities, Outputs, Run, RunSettings, input_digests, read_scored,
    verdict_from_files, write_new_json,
};
use super::score::{ForcedModel, LogitBiasFault, StepperModel, TokenRecord, prefill_record};
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
    }
}

/// The coverage rules off: one request of any length passes, delta up to 0.5 nats.
const ANY_COVERAGE: GateConfig = GateConfig {
    min_requests: 1,
    min_tokens_per_request: 1,
    max_delta: 0.5,
};

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

    /// The reference and the baseline scored, calibrated and audited, and the candidate judged
    /// against them; both audits stop at `audit_cap`.
    fn qualify(&self, config: GateConfig, audit_cap: usize) -> (Comparison, Run) {
        let mut scorer = StepperModel(Toy::exact());
        let mut baseline_model = StepperModel(Toy::rounded(1));
        let outputs = Outputs {
            reference: &self.reference,
            baseline: &self.baseline,
        };
        let settings = RunSettings {
            config,
            vocab: VOCAB,
            audit_cap,
        };
        let comparison = Comparison::prepare(
            (&mut scorer, Some(&mut baseline_model)),
            &identities(),
            outputs,
            settings,
        )
        .expect("the reference and the baseline calibrate");
        let candidate = Candidate {
            identity: &identity("candidate"),
            generations: &self.candidate,
            audit_cap,
        };
        let run = comparison
            .judge((&mut scorer, Some(&mut baseline_model)), candidate)
            .expect("the gate runs");
        (comparison, run)
    }

    fn report(&self, config: GateConfig, audit_cap: usize) -> Report {
        self.qualify(config, audit_cap).1.report
    }
}

fn gate(candidate: Toy) -> (Case, Report) {
    let case = Case::new(candidate, TOKENS);
    let report = case.report(GateConfig::QUALIFICATION, AUDIT_CAP);
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
    // Both rules compare with the baseline execution, never with the scorer's own path.
    assert_eq!(report.rate_rule.baseline_rate, report.baseline.rate);
    assert_eq!(report.event_rule.baseline_tokens, report.baseline.tokens);
    assert_ne!(report.rate_rule.baseline_rate, report.reference.rate);
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
    let report = case.report(GateConfig::QUALIFICATION, AUDIT_CAP);
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
    let report = case.report(GateConfig::QUALIFICATION, 0);
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
    let report = case.report(GateConfig::QUALIFICATION, AUDIT_CAP);
    let Verdict::Inconclusive { reasons } = &report.verdict else {
        panic!("{:#?}", report.verdict);
    };
    // The reference, the baseline and the candidate each fall short.
    assert_eq!(reasons.len(), 3, "{reasons:?}");
    assert!(reasons.iter().all(|reason| reason.contains("1024 tokens")));
}

#[test]
fn a_bitwise_identical_baseline_sets_the_smallest_delta_and_the_strictest_gate() {
    // An executor whose legal reconfigurations agree bitwise: the baseline never departs.
    let mut case = Case::new(Toy::exact(), TOKENS);
    case.baseline = outputs_of(Toy::exact(), &prompts(), TOKENS);
    let report = case.report(GateConfig::QUALIFICATION, AUDIT_CAP);
    assert_eq!(report.verdict, Verdict::Pass, "{:#?}", report.verdict);
    assert!(report.calibration.departures.is_empty());
    assert_eq!(report.calibration.largest_margin, 0.0);
    assert_eq!(report.calibration.delta, 0.125);
    assert_eq!(report.baseline.mismatches, 0);
    assert_eq!(report.candidate.mismatches, 0);
    // A lossless rounding change is still a near-tie flip, not an event.
    case.candidate = outputs_of(Toy::rounded(2), &prompts(), TOKENS);
    let report = case.report(GateConfig::QUALIFICATION, AUDIT_CAP);
    assert_eq!(report.candidate.events, 0, "{:#?}", report.candidate);
    // The injected fault is rejected by both rules.
    let mut faulty = LogitBiasFault::specpilot(StepperModel(Toy::exact()));
    case.candidate = outputs_with(&mut faulty, &prompts(), TOKENS);
    let report = case.report(GateConfig::QUALIFICATION, AUDIT_CAP);
    assert!(report.rate_rule.reject && report.event_rule.reject);
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
        let mut other = identity("candidate");
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
        (&identity("candidate"), &ids.scorer),
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

/// A consistent record: a mismatched token is the runner-up, a match has a 0.5-nat lead.
fn record(position: usize, generated: u32, argmax: u32, logprob: f64, top: f64) -> TokenRecord {
    let matched = generated == argmax;
    TokenRecord {
        position,
        generated,
        argmax,
        runner_up: match (matched, argmax) {
            (false, _) => generated,
            (true, 0) => 1,
            (true, _) => 0,
        },
        logprob,
        argmax_logprob: top,
        top2_gap: if matched { 0.5 } else { top - logprob },
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
    let report = evaluate(ANY_COVERAGE, &reference, &baseline, &candidate).unwrap();
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

/// `tokens` matched records with near-tie mismatches (margin 1/16) at the first `mismatches`
/// positions.
fn near_ties(label: &str, tokens: usize, mismatches: usize) -> ScoredSystem {
    let records = (0..tokens)
        .map(|position| {
            let generated = if position < mismatches { 2 } else { 1 };
            record(
                position,
                generated,
                1,
                -0.5 - 0.0625 * f64::from(generated - 1),
                -0.5,
            )
        })
        .collect();
    scored(label, records)
}

#[test]
fn the_rate_rule_rejects_only_on_the_lower_bound() {
    let config = ANY_COVERAGE;
    let reference = near_ties("reference", 10_000, 0);
    let baseline = near_ties("baseline", 10_000, 10);
    // 0.7% against 0.1%: the observed excess is above 0.5 points, but its one-sided 95% lower
    // bound is not, so the rule does not reject.
    let close = near_ties("candidate", 10_000, 70);
    let report = evaluate(config, &reference, &baseline, &close).unwrap();
    assert!(report.rate_rule.difference > report.rate_rule.limit);
    assert!(report.rate_rule.lower_bound < report.rate_rule.limit);
    assert!(!report.rate_rule.reject);
    assert_eq!(report.verdict, Verdict::Pass);
    // 1.5% against 0.1%: the lower bound clears the limit.
    let far = near_ties("candidate", 10_000, 150);
    let report = evaluate(config, &reference, &baseline, &far).unwrap();
    assert!(report.rate_rule.lower_bound > report.rate_rule.limit);
    assert!(report.rate_rule.reject && !report.event_rule.reject);
    assert!(matches!(report.verdict, Verdict::Reject { .. }));
}

#[test]
fn an_audit_explains_only_tokens_a_legal_re_execution_does_not_see_as_events() {
    // An event: generated 4 against argmax 7 at 2.9 nats on the step path; delta 0.125.
    let threshold = 0.25;
    let reread = |argmax, margin| Reread { argmax, margin };
    let audit = |prefill, baseline| Audit {
        position: 9,
        prefill,
        baseline,
    };
    assert!(!audit(reread(7, 2.9), None).explains(threshold));
    assert!(!audit(reread(7, 2.9), Some(reread(7, 3.0))).explains(threshold));
    // One prefill of the whole context chooses the generated token, or keeps it within twice
    // delta of its own choice.
    assert!(audit(reread(4, 0.0), None).explains(threshold));
    assert!(audit(reread(7, 0.25), None).explains(threshold));
    assert!(!audit(reread(7, 0.2501), None).explains(threshold));
    // The baseline execution does.
    assert!(audit(reread(7, 2.9), Some(reread(4, 0.0))).explains(threshold));
    assert!(audit(reread(7, 2.9), Some(reread(3, 0.2))).explains(threshold));
    // An argmax that flips between the reference's own top tokens but stays far from the
    // generated token explains nothing.
    assert!(!audit(reread(2, 6.0), None).explains(threshold));
    assert!(!audit(reread(7, 2.9), Some(reread(3, 6.0))).explains(threshold));
}

#[test]
fn the_verdict_is_reproducible_from_the_written_files() {
    let bug = Bug::RareSpike {
        period: 997,
        boost: 6.0,
    };
    let case = Case::new(Toy::new(0, bug), TOKENS);
    let (comparison, mut run) = case.qualify(GateConfig::QUALIFICATION, AUDIT_CAP);
    let directory = tempfile::tempdir().unwrap();
    let path = |name: &str| directory.path().join(name);
    run.report.inputs = input_digests([
        write_new_json(&path("reference.json"), &comparison.reference).unwrap(),
        write_new_json(&path("baseline.json"), &comparison.baseline).unwrap(),
        write_new_json(&path("candidate.json"), &run.candidate).unwrap(),
    ]);
    write_new_json(&path("report.json"), &run.report).unwrap();
    assert!(write_new_json(&path("report.json"), &run.report).is_err());
    let (read_back, digest) = read_scored(&path("candidate.json")).unwrap();
    assert_eq!(read_back, run.candidate);
    assert_eq!(run.report.inputs["candidate"], digest);
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
    assert_eq!(json["inputs"]["baseline"].as_str().unwrap().len(), 64);
    assert_eq!(json["verdict"]["verdict"], "reject");
    assert_eq!(json["scorer"]["label"], "scorer-exact");
    let first = &json["candidate"]["mismatch_list"][0];
    assert_eq!(first["class"], "large_margin");
    assert_eq!(first["event"], "unexplained");
}

/// Synthetic files of one request: delta 0.125 from a baseline departure at 1/16 nat, and one
/// candidate event at position 2 (margin 3 nats) whose audit leaves it unexplained.
fn synthetic() -> (ScoredSystem, ScoredSystem, ScoredSystem) {
    let matched = |position| record(position, 1, 1, -0.5, -0.5);
    let reference = scored("reference", (0..4).map(matched).collect());
    let mut records: Vec<_> = (0..4).map(matched).collect();
    records[1] = record(1, 2, 1, -0.5625, -0.5);
    let baseline = scored("baseline", records);
    let mut records: Vec<_> = (0..4).map(matched).collect();
    records[2] = record(2, 5, 1, -3.5, -0.5);
    let mut candidate = scored("candidate", records);
    candidate.requests[0].audits.push(Audit {
        position: 2,
        prefill: Reread {
            argmax: 1,
            margin: 3.0,
        },
        baseline: None,
    });
    (reference, baseline, candidate)
}

#[test]
fn the_synthetic_files_are_valid() {
    let (reference, baseline, candidate) = synthetic();
    let report = evaluate(ANY_COVERAGE, &reference, &baseline, &candidate).unwrap();
    assert_eq!(report.calibration.delta, 0.125);
    assert_eq!(report.candidate.unexplained_events, 1);
    assert_eq!(report.verdict, Verdict::Pass);
}

/// An edit of one scored request, as a damaged or hand-edited file would carry it.
type Tamper = fn(&mut ScoredRequest);

#[test]
fn files_whose_records_or_audits_contradict_themselves_are_refused() {
    let tampers: [(&str, Tamper); 10] = [
        ("audited twice", |request| {
            let mut explaining = request.audits[0];
            explaining.prefill = Reread {
                argmax: 5,
                margin: 0.0,
            };
            request.audits.push(explaining);
        }),
        ("past the 4 records", |request| {
            request.audits[0].position = 4
        }),
        ("not a large-margin event", |request| {
            request.audits[0] = Audit {
                position: 1,
                prefill: Reread {
                    argmax: 1,
                    margin: 0.0,
                },
                baseline: None,
            };
        }),
        ("disagree about generated", |request| {
            request.audits[0].prefill = Reread {
                argmax: 5,
                margin: 1.0,
            };
        }),
        ("not a finite gap", |request| {
            request.audits[0].prefill.margin = -1.0;
        }),
        // A raised log-probability that hides the event as a near tie.
        ("sits above the runner-up", |request| {
            request.records[2].logprob = -0.5;
        }),
        ("not ordered below zero", |request| {
            request.records[2].argmax_logprob = 0.5;
        }),
        ("a matched token has margin", |request| {
            request.records[0].logprob = -0.75;
        }),
        ("both argmax and runner-up", |request| {
            request.records[0].runner_up = 1;
        }),
        ("names position", |request| request.records[3].position = 2),
    ];
    for (expected, tamper) in tampers {
        let (reference, baseline, mut candidate) = synthetic();
        tamper(&mut candidate.requests[0]);
        let error = evaluate(ANY_COVERAGE, &reference, &baseline, &candidate).unwrap_err();
        assert!(
            format!("{error:?}").contains(expected),
            "{expected}: {error:?}"
        );
    }
    // The reference and the baseline are checked the same way.
    let (mut reference, baseline, candidate) = synthetic();
    reference.requests[0].records[1].top2_gap = f64::NAN;
    assert!(evaluate(ANY_COVERAGE, &reference, &baseline, &candidate).is_err());
}

#[test]
fn record_checks_follow_the_tie_rule() {
    // Ties go to the lower id: a mismatch at zero margin must name a higher id.
    let tie = record(0, 3, 2, -0.2, -0.2);
    assert_eq!(tie.check(), Ok(()));
    let lower = record(0, 1, 2, -0.2, -0.2);
    assert!(lower.check().unwrap_err().contains("won the tie"));
    let mut infinite = record(0, 1, 1, -0.5, -0.5);
    infinite.top2_gap = f64::INFINITY;
    assert!(infinite.check().is_err());
}

#[test]
fn a_run_with_fewer_requests_than_required_is_inconclusive() {
    let (reference, baseline, mut candidate) = synthetic();
    candidate.requests[0].audits.clear();
    candidate.requests[0].records[2] = record(2, 1, 1, -0.5, -0.5);
    assert_eq!(
        evaluate(ANY_COVERAGE, &reference, &baseline, &candidate)
            .unwrap()
            .verdict,
        Verdict::Pass
    );
    let config = GateConfig {
        min_requests: 4,
        ..ANY_COVERAGE
    };
    let report = evaluate(config, &reference, &baseline, &candidate).unwrap();
    let Verdict::Inconclusive { reasons } = &report.verdict else {
        panic!("{:#?}", report.verdict);
    };
    assert_eq!(reasons.len(), 1, "{reasons:?}");
    assert!(
        reasons[0].contains("1 requests, below the 4"),
        "{reasons:?}"
    );
}

#[test]
fn a_delta_above_the_cap_makes_the_run_inconclusive() {
    let (reference, mut baseline, mut candidate) = synthetic();
    // A baseline departure at 1.4 nats sets delta 1.5: more than reduction order explains.
    baseline.requests[0].records[1] = record(1, 2, 1, -1.9, -0.5);
    candidate.requests[0].audits.clear();
    let report = evaluate(ANY_COVERAGE, &reference, &baseline, &candidate).unwrap();
    assert_eq!(report.calibration.delta, 1.5);
    // At that delta the 3-nat mismatch is an intermediate one, not an event.
    assert_eq!(report.candidate.events, 0);
    let Verdict::Inconclusive { reasons } = &report.verdict else {
        panic!("{:#?}", report.verdict);
    };
    assert!(reasons[0].contains("exceeds the 0.5"), "{reasons:?}");
    let config = GateConfig {
        max_delta: 1.5,
        ..ANY_COVERAGE
    };
    let report = evaluate(config, &reference, &baseline, &candidate).unwrap();
    assert_eq!(report.verdict, Verdict::Pass);
}

#[test]
fn a_reference_that_disagrees_with_its_scorer_is_refused() {
    let (mut reference, baseline, candidate) = synthetic();
    reference.requests[0].records[3] = record(3, 2, 1, -0.5625, -0.5);
    let error = evaluate(ANY_COVERAGE, &reference, &baseline, &candidate).unwrap_err();
    assert!(
        format!("{error:?}").contains("disagrees with its own scorer at 1 tokens"),
        "{error:?}"
    );
}

#[test]
fn one_file_cannot_take_two_roles() {
    let (reference, baseline, candidate) = synthetic();
    let cases = [
        (&reference, &baseline, &baseline),
        (&reference, &reference, &candidate),
        (&candidate, &baseline, &candidate),
    ];
    for (reference, baseline, candidate) in cases {
        let error = evaluate(ANY_COVERAGE, reference, baseline, candidate).unwrap_err();
        assert!(
            format!("{error:?}").contains("distinct labels"),
            "{error:?}"
        );
    }
}

#[test]
fn an_audit_refuses_generations_other_than_the_scored_ones() {
    let prompts = prompts();
    let outputs = outputs_of(Toy::exact(), &prompts, 64);
    let mut scorer = StepperModel(Toy::exact());
    let ids = identities();
    let mut scored =
        score_system(&mut scorer, (&ids.reference, &ids.scorer), &outputs, VOCAB).unwrap();
    let mut other = outputs.clone();
    other[2].generated[10] = (other[2].generated[10] + 1) % VOCAB as u32;
    let plan = AuditPlan {
        threshold: 0.25,
        vocab: VOCAB,
        cap: AUDIT_CAP,
    };
    let error = audit_system(&mut scored, (&mut scorer, None), &other, plan).unwrap_err();
    assert!(format!("{error:?}").contains("do not match"), "{error:?}");
    assert!(audit_system(&mut scored, (&mut scorer, None), &outputs, plan).is_ok());
}

#[test]
fn a_one_prefill_reread_judges_an_existing_token_only() {
    let prompt = prompt(0, PROMPT);
    let mut scorer = StepperModel(Toy::exact());
    let generated = generate(&mut scorer, &prompt, 32);
    let reread = prefill_record(&mut scorer, &prompt, &generated, 31, VOCAB).unwrap();
    assert_eq!((reread.position, reread.generated), (31, generated[31]));
    // The scorer's own greedy token is its argmax in one prefill as well.
    assert!(reread.matched());
    let error = prefill_record(&mut scorer, &prompt, &generated, 32, VOCAB).unwrap_err();
    assert!(format!("{error:?}").contains("past the 32"), "{error:?}");
}

#[test]
fn a_comparison_rebuilt_from_its_files_judges_like_the_original() {
    let case = Case::new(Toy::rounded(2), TOKENS);
    let (comparison, run) = case.qualify(GateConfig::QUALIFICATION, AUDIT_CAP);
    let directory = tempfile::tempdir().unwrap();
    let path = |name: &str| directory.path().join(name);
    write_new_json(&path("reference.json"), &comparison.reference).unwrap();
    write_new_json(&path("baseline.json"), &comparison.baseline).unwrap();
    let settings = RunSettings {
        config: GateConfig::QUALIFICATION,
        vocab: VOCAB,
        audit_cap: AUDIT_CAP,
    };
    let reread = |name: &str| read_scored(&path(name)).unwrap().0;
    let ids = identities();
    let rebuild = |scorer: &ExecutionIdentity| {
        Comparison::from_scored(
            reread("reference.json"),
            reread("baseline.json"),
            scorer,
            settings,
        )
    };
    // Another scorer, such as another build, cannot judge against these files.
    let mut other = ids.scorer.clone();
    other
        .profile
        .insert("executable_sha256".to_owned(), "0".repeat(64));
    let error = rebuild(&other).unwrap_err();
    assert!(
        format!("{error:?}").contains("not by this run's"),
        "{error:?}"
    );
    let rebuilt = rebuild(&ids.scorer).unwrap();
    let mut scorer = StepperModel(Toy::exact());
    let mut baseline_model = StepperModel(Toy::rounded(1));
    let candidate = Candidate {
        identity: &identity("candidate"),
        generations: &case.candidate,
        audit_cap: AUDIT_CAP,
    };
    let again = rebuilt
        .judge((&mut scorer, Some(&mut baseline_model)), candidate)
        .unwrap();
    assert_eq!(again.report, run.report);
    assert_eq!(again.report.verdict, Verdict::Pass);
}

#[test]
fn a_reread_explains_an_event_up_to_twice_delta_and_no_further() {
    // Delta 0.125: one prefill that brings the 3-nat event to 0.25 nats explains it; 0.375 nats,
    // inside four delta but outside two, leaves it unexplained.
    for (margin, explained) in [(0.25, 1), (0.375, 0)] {
        let (reference, baseline, mut candidate) = synthetic();
        candidate.requests[0].audits[0].prefill.margin = margin;
        let report = evaluate(ANY_COVERAGE, &reference, &baseline, &candidate).unwrap();
        assert_eq!(report.candidate.explained_events, explained, "{margin}");
        assert_eq!(
            report.candidate.unexplained_events,
            1 - explained,
            "{margin}"
        );
    }
}
