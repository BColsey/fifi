//! Host tests of the whole gate on the toy model: a lossless candidate passes at 4,096 tokens,
//! and each injected defect is rejected by the rule that should catch it.

use super::fixture::{Bug, LAG, RING, Toy, VOCAB, generate, prompt};
use super::gate::{
    Audit, AuditPlan, EXECUTABLE_SHA256, EventStatus, ExecutionIdentity, GateConfig, Generation,
    MarginClass, Report, Reread, ScoredRequest, ScoredSystem, Verdict, audit_system, calibrate,
    evaluate, ids_sha256, score_system,
};
use super::probe::{EVENT_FLOOR, ProbeFile, ProbeSettings, check_probe, probe_request};
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

/// One toy build: every execution differs from the others in its generation path only.
fn identity(label: &str) -> ExecutionIdentity {
    ExecutionIdentity {
        label: label.to_owned(),
        family: "toy".to_owned(),
        weights: "toy-weights-1".to_owned(),
        weight_format: "toy-f32".to_owned(),
        profile: BTreeMap::from([("generation_path".to_owned(), label.to_owned())]),
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
    assert_eq!(json["schema"], "fifi.token_forcing.v2");
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
    let tampers: [(&str, Tamper); 12] = [
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
        // A zero gap to a higher argmax: the lower generated id would have won the tie.
        ("disagree about generated", |request| {
            request.audits[0].prefill = Reread {
                argmax: 7,
                margin: 0.0,
            };
        }),
        // The baseline reread is checked like the prefill one.
        ("disagree about generated", |request| {
            request.audits[0].baseline = Some(Reread {
                argmax: 5,
                margin: 1.0,
            });
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

#[test]
fn a_repeated_prompt_is_refused() {
    let (reference, baseline, mut candidate) = synthetic();
    let mut again = candidate.requests[0].clone();
    again.name = "request-1".to_owned();
    candidate.requests.push(again);
    let error = evaluate(ANY_COVERAGE, &reference, &baseline, &candidate).unwrap_err();
    assert!(
        format!("{error:?}").contains("repeats the prompt"),
        "{error:?}"
    );
}

#[test]
fn a_departure_hidden_by_unequal_lengths_is_refused() {
    let (reference, mut baseline, candidate) = synthetic();
    // The baseline agrees with the reference on all four tokens and has a fifth.
    baseline.requests[0].records[1] = record(1, 1, 1, -0.5, -0.5);
    baseline.requests[0]
        .records
        .push(record(4, 2, 1, -0.5625, -0.5));
    let error = evaluate(ANY_COVERAGE, &reference, &baseline, &candidate).unwrap_err();
    assert!(
        format!("{error:?}").contains("cannot see where they depart"),
        "{error:?}"
    );
    // The other way round: the reference is the longer one.
    let (mut reference, mut baseline, candidate) = synthetic();
    baseline.requests[0].records[1] = record(1, 1, 1, -0.5, -0.5);
    reference.requests[0]
        .records
        .push(record(4, 1, 1, -0.5, -0.5));
    let error = evaluate(ANY_COVERAGE, &reference, &baseline, &candidate).unwrap_err();
    assert!(
        format!("{error:?}").contains("the reference scored 5 tokens and the baseline 4"),
        "{error:?}"
    );
}

#[test]
fn an_unaudited_baseline_event_makes_the_run_inconclusive() {
    let (reference, mut baseline, mut candidate) = synthetic();
    // A 3-nat baseline mismatch after the departure, never audited.
    baseline.requests[0].records[3] = record(3, 6, 1, -3.5, -0.5);
    candidate.requests[0].audits.clear();
    candidate.requests[0].records[2] = record(2, 1, 1, -0.5, -0.5);
    let report = evaluate(ANY_COVERAGE, &reference, &baseline, &candidate).unwrap();
    assert_eq!(report.baseline.unaudited_events, 1);
    let Verdict::Inconclusive { reasons } = &report.verdict else {
        panic!("{:#?}", report.verdict);
    };
    assert!(
        reasons[0].contains("baseline events were not audited"),
        "{reasons:?}"
    );
}

/// A model whose every row is one fixed logits row.
struct FixedRow(Vec<f32>);

impl ForcedModel for FixedRow {
    fn reset(&mut self) -> Result<(), crate::EngError> {
        Ok(())
    }

    fn prefill_logits(&mut self, _context: &[u32]) -> Result<Vec<f32>, crate::EngError> {
        Ok(self.0.clone())
    }

    fn step_logits(&mut self, _token: u32) -> Result<Vec<f32>, crate::EngError> {
        Ok(self.0.clone())
    }
}

/// A fault around one fixed row, biasing every 2nd id by 1.5 from output position 2 on.
fn small_fault(row: &[f32]) -> LogitBiasFault<FixedRow> {
    let mut fault = LogitBiasFault::specpilot(FixedRow(row.to_vec()));
    fault.from = 2;
    fault.stride = 2;
    fault.bias = 1.5;
    fault
}

#[test]
fn the_fault_reads_each_row_at_its_output_position() {
    // The prefill row is output position 0, and each step adds one.
    let base = vec![0.0, 2.0, 1.0, -1.0];
    let mut fault = small_fault(&base);
    let mut rows = vec![fault.prefill_logits(&[3]).unwrap()];
    for token in [1, 1, 1] {
        rows.push(fault.step_logits(token).unwrap());
    }
    for (position, row) in rows.iter().enumerate() {
        assert_eq!(
            *row,
            fault.bias_at(position, base.clone()),
            "position {position}"
        );
    }
    assert_eq!(rows[1], base);
    assert_eq!(rows[2], vec![1.5, 2.0, 2.5, -1.0]);
    // A new request starts at position 0 again.
    assert_eq!(fault.prefill_logits(&[3]).unwrap(), base);
}

#[test]
fn a_fault_departure_is_the_biased_choice_judged_by_the_unbiased_row() {
    let row = [0.0, 2.0, 1.0, -1.0];
    let fault = small_fault(&row);
    // Before the fault's start the argmax stays chosen.
    assert_eq!(fault.departure_at(1, &row, 4).unwrap(), None);
    // From it on, id 2 gains 1.5 and passes id 1 by 0.5; the scorer reads it 1 nat below.
    let departure = fault
        .departure_at(2, &row, 4)
        .unwrap()
        .expect("a departure");
    assert_eq!((departure.generated, departure.argmax), (2, 1));
    assert!((departure.margin() - 1.0).abs() < 1e-9, "{departure:?}");
    // A gap wider than the bias, or a biased argmax, leaves the choice alone.
    assert_eq!(
        fault.departure_at(2, &[0.0, 2.0, 0.4, -1.0], 4).unwrap(),
        None
    );
    assert_eq!(
        fault.departure_at(2, &[2.0, 1.0, 1.0, -1.0], 4).unwrap(),
        None
    );
    // Only the tokenizer-visible ids count.
    assert_eq!(
        fault
            .departure_at(2, &[0.0, 2.0, -9.0, 9.0, 0.9], 3)
            .unwrap(),
        None
    );
}

#[test]
fn an_audit_records_each_re_execution_in_its_own_place() {
    // The scorer prefers id 0 by 3 logits over id 1; the baseline model prefers id 1.
    let mut scorer = FixedRow(vec![3.0, 0.0, -1.0, -1.0]);
    let mut baseline_model = FixedRow(vec![0.0, 3.0, -1.0, -1.0]);
    let generations = vec![Generation {
        name: "request-0".to_owned(),
        prompt: vec![2, 3],
        generated: vec![0, 1],
    }];
    let ids = identities();
    let mut scored =
        score_system(&mut scorer, (&ids.baseline, &ids.scorer), &generations, 4).unwrap();
    let plan = AuditPlan {
        threshold: 0.25,
        vocab: 4,
        cap: AUDIT_CAP,
    };
    audit_system(
        &mut scored,
        (&mut scorer, Some(&mut baseline_model)),
        &generations,
        plan,
    )
    .unwrap();
    let audits = &scored.requests[0].audits;
    assert_eq!(audits.len(), 1, "{audits:?}");
    let audit = audits[0];
    assert_eq!(audit.position, 1);
    // One prefill on the scorer still sees the 3-logit gap; the baseline model chooses id 1.
    assert_eq!(audit.prefill.argmax, 0);
    assert!((audit.prefill.margin - 3.0).abs() < 1e-6, "{audit:?}");
    assert_eq!(
        audit.baseline,
        Some(Reread {
            argmax: 1,
            margin: 0.0
        })
    );
    assert!(audit.explains(plan.threshold));
}

#[test]
fn explained_events_stay_out_of_the_event_rule() {
    // 10,000 tokens a system, a baseline departure at 1/16 nat (delta 0.125), and six 3-nat
    // candidate events: few enough to stay under the rate limit, enough for the event rule
    // (p = 1/64 against none).
    let matched = |position| record(position, 1, 1, -0.5, -0.5);
    let reference = scored("reference", (0..10_000).map(matched).collect());
    let mut records: Vec<_> = (0..10_000).map(matched).collect();
    records[1] = record(1, 2, 1, -0.5625, -0.5);
    let baseline = scored("baseline", records);
    let events = [100, 200, 300, 400, 500, 600];
    let candidate = |reread_margin: f64| {
        let mut records: Vec<_> = (0..10_000).map(matched).collect();
        for &position in &events {
            records[position] = record(position, 5, 1, -3.5, -0.5);
        }
        let mut candidate = scored("candidate", records);
        candidate.requests[0].audits = events
            .iter()
            .map(|&position| Audit {
                position,
                prefill: Reread {
                    argmax: if reread_margin == 0.0 { 5 } else { 1 },
                    margin: reread_margin,
                },
                baseline: None,
            })
            .collect();
        candidate
    };
    let explained = evaluate(ANY_COVERAGE, &reference, &baseline, &candidate(0.0)).unwrap();
    assert_eq!(explained.candidate.explained_events, 6);
    assert_eq!(explained.event_rule.candidate_unexplained, 0);
    assert!(!explained.rate_rule.reject);
    assert_eq!(explained.verdict, Verdict::Pass);
    let unexplained = evaluate(ANY_COVERAGE, &reference, &baseline, &candidate(3.0)).unwrap();
    assert_eq!(unexplained.event_rule.candidate_unexplained, 6);
    assert!(unexplained.event_rule.reject && !unexplained.rate_rule.reject);
    assert!(matches!(unexplained.verdict, Verdict::Reject { .. }));
}

#[test]
fn the_reference_and_the_baseline_must_run_the_scorers_build() {
    let (reference, baseline, candidate) = synthetic();
    let judge = |reference: &ScoredSystem, baseline: &ScoredSystem, candidate: &ScoredSystem| {
        evaluate(ANY_COVERAGE, reference, baseline, candidate).map_err(|e| format!("{e:?}"))
    };
    let set = |system: &mut ScoredSystem, key: &str, value: &str| {
        system
            .system
            .profile
            .insert(key.to_owned(), value.to_owned());
    };
    // Another executable or family for the reference or the baseline is refused.
    let mut other_build = baseline.clone();
    set(&mut other_build, EXECUTABLE_SHA256, &"0".repeat(64));
    let error = judge(&reference, &other_build, &candidate).unwrap_err();
    assert!(error.contains("scorer's build"), "{error}");
    let mut other_family = reference.clone();
    other_family.system.family = "another-toy".to_owned();
    let error = judge(&other_family, &baseline, &candidate).unwrap_err();
    assert!(error.contains("scorer's build"), "{error}");
    // The reference runs the scorer's numeric path; the baseline differs from it by design.
    let mut other_path = reference.clone();
    set(&mut other_path, "kv_profile", "q8");
    let error = judge(&other_path, &baseline, &candidate).unwrap_err();
    assert!(error.contains("numeric path"), "{error}");
    let mut other_chunk = baseline.clone();
    set(&mut other_chunk, "prefill_chunk_size", "4");
    assert!(judge(&reference, &other_chunk, &candidate).is_ok());
    // A candidate from another build is what an A/B between builds judges.
    let mut other_candidate = candidate;
    set(&mut other_candidate, EXECUTABLE_SHA256, &"0".repeat(64));
    assert!(judge(&reference, &baseline, &other_candidate).is_ok());
}

/// A model that reads out fixed rows: the first for the prompt, then one per step. It keeps
/// the tokens it was fed.
struct Rows {
    rows: Vec<Vec<f32>>,
    next: usize,
    fed: Vec<u32>,
}

impl ForcedModel for Rows {
    fn reset(&mut self) -> Result<(), crate::EngError> {
        self.next = 0;
        self.fed.clear();
        Ok(())
    }

    fn prefill_logits(&mut self, _context: &[u32]) -> Result<Vec<f32>, crate::EngError> {
        self.next = 1;
        Ok(self.rows[0].clone())
    }

    fn step_logits(&mut self, token: u32) -> Result<Vec<f32>, crate::EngError> {
        self.fed.push(token);
        self.next += 1;
        Ok(self.rows[self.next - 1].clone())
    }
}

#[test]
fn the_probe_counts_the_faults_reach_up_to_the_first_end_id() {
    // Every 2nd id gains 1.5 from output position 2; id 3 ends an answer.
    let mut fault = LogitBiasFault::starting_at((), 2);
    assert_eq!((fault.from, fault.stride, fault.bias), (2, 7, 2.0));
    (fault.stride, fault.bias) = (2, 1.5);
    let rows = vec![
        vec![0.0, 2.0, 1.4, -5.0], // 0: the bias would flip id 2 here, before its start
        vec![0.0, 2.0, 0.0, -5.0], // 1
        vec![1.3, 2.5, 0.0, -5.0], // 2: a flip to id 0 at margin 1.2, an event
        vec![2.0, 2.3, 0.0, -5.0], // 3: a flip to id 0 at margin 0.3, no event
        vec![2.0, 1.0, 0.0, -5.0], // 4: close, but the argmax is biased itself
        vec![0.0, 1.0, 0.0, 5.0],  // 5: the end id
        vec![1.3, 2.5, 0.0, -5.0], // 6: after the end, not counted
    ];
    let settings = ProbeSettings {
        fault: &fault,
        end_ids: &[3],
        tokens: rows.len(),
        vocab: 4,
    };
    let mut model = Rows {
        rows: rows.clone(),
        next: 0,
        fed: Vec::new(),
    };
    let row = probe_request(&mut model, ("task", &[9, 9]), settings).unwrap();
    let generated = [1, 1, 1, 1, 0, 3, 1];
    // The probe feeds each greedy choice back, as the reference generates.
    assert_eq!(model.fed, generated[..6]);
    assert_eq!(row.generated_sha256, ids_sha256(&generated));
    assert_eq!(
        (row.prompt_sha256, row.prompt_tokens),
        (ids_sha256(&[9, 9]), 2)
    );
    assert_eq!(row.first_end, Some(5));
    let counts = (
        row.late_tokens,
        row.late_distinct,
        row.late_close,
        row.fault_flips,
        row.fault_events,
    );
    assert_eq!(counts, (3, 2, 3, 2, 1));
    // An end id before the fault's start leaves nothing to count.
    let mut early_end = rows;
    early_end[1] = vec![0.0, 1.0, 0.0, 5.0];
    let mut model = Rows {
        rows: early_end,
        next: 0,
        fed: Vec::new(),
    };
    let row = probe_request(&mut model, ("task", &[9, 9]), settings).unwrap();
    assert_eq!(row.first_end, Some(1));
    assert_eq!(
        (row.late_tokens, row.fault_flips, row.fault_events),
        (0, 0, 0)
    );
}

/// The probe of every toy request on the exact model, as a run writes it.
fn toy_probe(prompts: &[Vec<u32>], tokens: usize) -> ProbeFile {
    let fault = LogitBiasFault::specpilot(());
    let settings = ProbeSettings {
        fault: &fault,
        end_ids: &[],
        tokens,
        vocab: VOCAB,
    };
    let mut model = StepperModel(Toy::exact());
    let rows = prompts
        .iter()
        .enumerate()
        .map(|(index, prompt)| {
            let name = format!("request-{index}");
            probe_request(&mut model, (&name, prompt), settings).unwrap()
        })
        .collect();
    // The probe runs the reference's execution under its own label.
    let mut system = identities().reference;
    system.label = "probe".to_owned();
    ProbeFile {
        system,
        tokens,
        event_floor: EVENT_FLOOR,
        rows,
    }
}

#[test]
fn a_probe_reads_the_reference_output_and_nothing_else() {
    let tokens = 2100;
    let prompts = prompts();
    let outputs = outputs_of(Toy::exact(), &prompts, tokens);
    let ids = identities();
    let mut scorer = StepperModel(Toy::exact());
    let reference =
        score_system(&mut scorer, (&ids.reference, &ids.scorer), &outputs, VOCAB).unwrap();
    let probe = toy_probe(&prompts, tokens);
    assert_eq!(probe.rows[0].late_tokens, tokens - 2048);
    check_probe(&probe, &reference).unwrap();
    let refused = |probe: &ProbeFile| format!("{:?}", check_probe(probe, &reference).unwrap_err());
    let mut other_build = probe.clone();
    other_build
        .system
        .profile
        .insert(EXECUTABLE_SHA256.to_owned(), "0".repeat(64));
    assert!(refused(&other_build).contains("the probe ran"));
    let shorter = toy_probe(&prompts, tokens - 1);
    assert!(refused(&shorter).contains("another output"));
    let mut other_output = probe.clone();
    other_output.rows[2].generated_sha256 = ids_sha256(&[1]);
    assert!(refused(&other_output).contains("another output"));
    let mut other_prompt = probe.clone();
    other_prompt.rows[1].prompt_sha256 = ids_sha256(&[1]);
    assert!(refused(&other_prompt).contains("another prompt"));
    let mut missing = probe;
    missing.rows.remove(3);
    assert!(refused(&missing).contains("no probe row"));
}
