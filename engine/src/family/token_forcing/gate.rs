//! Calibration, the two rejection rules, and the qualification report.

use super::score::{ForcedModel, TokenRecord, prefill_argmax, score_sequence};
use super::stats::{
    Interval, Z_ONE_SIDED_95, Z_TWO_SIDED_95, clustered_rate, newcombe_difference,
    rate_excess_p_value, wilson,
};
use crate::EngError;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap};

type Result<T> = std::result::Result<T, EngError>;

/// The report format. A change to any field's meaning gets a new version.
pub(crate) const REPORT_SCHEMA: &str = "fifi.token_forcing.v1";
/// The rate rule: a candidate whose mismatch rate exceeds the baseline's by more than this,
/// at one-sided 95% confidence, is rejected (SpecPilot, arXiv 2611.02501, section 3.3).
pub(crate) const RATE_EXCESS_LIMIT: f64 = 0.005;
/// The significance level of the event rule.
pub(crate) const EVENT_ALPHA: f64 = 0.05;
/// The near-tie threshold is the largest calibration margin rounded up to this grid.
pub(crate) const DELTA_GRID: f64 = 0.125;
/// Output depths at which the report splits its counts, so a fault that starts late is visible.
const DEPTH_EDGES: [usize; 3] = [1024, 2048, 4096];

/// What ran: the weights, their format, and the execution settings that can change numerics.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct ExecutionIdentity {
    pub(crate) label: String,
    pub(crate) family: String,
    /// The identity of the weight files, such as the SHA-256 of the pack manifest.
    pub(crate) weights: String,
    pub(crate) weight_format: String,
    /// Settings that select a numeric path: KV dtype, chunk rows, kernel switches, build.
    #[serde(default)]
    pub(crate) profile: BTreeMap<String, String>,
}

/// One prompt and the tokens a system generated after it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Generation {
    pub(crate) name: String,
    pub(crate) prompt: Vec<u32>,
    pub(crate) generated: Vec<u32>,
}

/// The audit of one large-margin mismatch.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct Audit {
    pub(crate) position: usize,
    /// The scorer's argmax when the whole context up to `position` is one prefill.
    pub(crate) prefill_argmax: u32,
    /// The argmax of the baseline execution on the same one-prefill context, when the harness
    /// has a model of it.
    pub(crate) baseline_argmax: Option<u32>,
}

impl Audit {
    /// An event is explained when the reference itself produces the token under another legal
    /// execution, or when the reference's own choice there depends on the execution.
    pub(crate) fn explains(&self, record: &TokenRecord) -> bool {
        let fragile = |argmax: u32| argmax == record.generated || argmax != record.argmax;
        fragile(self.prefill_argmax) || self.baseline_argmax.is_some_and(fragile)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct ScoredRequest {
    pub(crate) name: String,
    pub(crate) prompt_tokens: usize,
    pub(crate) prompt_sha256: String,
    pub(crate) records: Vec<TokenRecord>,
    #[serde(default)]
    pub(crate) audits: Vec<Audit>,
}

/// One system's outputs, scored by one reference. This is the file a device run writes.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct ScoredSystem {
    pub(crate) system: ExecutionIdentity,
    pub(crate) scorer: ExecutionIdentity,
    pub(crate) requests: Vec<ScoredRequest>,
}

/// Scores every generation of one system with the reference.
pub(crate) fn score_system(
    scorer: &mut dyn ForcedModel,
    identities: (&ExecutionIdentity, &ExecutionIdentity),
    generations: &[Generation],
    vocab: usize,
) -> Result<ScoredSystem> {
    let (system, scorer_identity) = identities;
    let requests = generations
        .iter()
        .map(|generation| {
            Ok(ScoredRequest {
                name: generation.name.clone(),
                prompt_tokens: generation.prompt.len(),
                prompt_sha256: ids_sha256(&generation.prompt),
                records: score_sequence(scorer, &generation.prompt, &generation.generated, vocab)?,
                audits: Vec::new(),
            })
        })
        .collect::<Result<_>>()?;
    Ok(ScoredSystem {
        system: system.clone(),
        scorer: scorer_identity.clone(),
        requests,
    })
}

/// Which mismatches an audit re-reads, and how many at most.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct AuditPlan {
    /// Mismatches with a margin above this are audited: twice the calibrated delta.
    pub(crate) threshold: f64,
    pub(crate) vocab: usize,
    /// Each audit is one prefill of up to the whole context; SpecPilot audited 200.
    pub(crate) cap: usize,
}

/// Audits every mismatch above the plan's threshold, in request and position order, up to the
/// cap. Events past the cap stay unaudited: the report counts them, and the gate treats them as
/// unexplained for a candidate and as an unknown baseline.
pub(crate) fn audit_system(
    scored: &mut ScoredSystem,
    models: (&mut dyn ForcedModel, Option<&mut dyn ForcedModel>),
    generations: &[Generation],
    plan: AuditPlan,
) -> Result<()> {
    let (scorer, mut baseline) = models;
    check_generations(scored, generations)?;
    for request in &mut scored.requests {
        request.audits.clear();
    }
    let mut audited = 0;
    for (request, generation) in scored.requests.iter_mut().zip(generations) {
        let (prompt, generated) = (&generation.prompt, &generation.generated);
        let events = request
            .records
            .iter()
            .filter(|record| record.margin() > plan.threshold);
        for record in events {
            if audited == plan.cap {
                return Ok(());
            }
            let at = record.position;
            let baseline_argmax = match baseline.as_deref_mut() {
                Some(model) => Some(prefill_argmax(model, prompt, generated, at, plan.vocab)?),
                None => None,
            };
            request.audits.push(Audit {
                position: at,
                prefill_argmax: prefill_argmax(scorer, prompt, generated, at, plan.vocab)?,
                baseline_argmax,
            });
            audited += 1;
        }
    }
    Ok(())
}

fn check_generations(scored: &ScoredSystem, generations: &[Generation]) -> Result<()> {
    let same = scored.requests.len() == generations.len()
        && scored
            .requests
            .iter()
            .zip(generations)
            .all(|(request, generation)| {
                request.name == generation.name
                    && request.prompt_sha256 == ids_sha256(&generation.prompt)
                    && request.records.len() == generation.generated.len()
            });
    if same {
        Ok(())
    } else {
        Err(gate_error(format!(
            "the generations do not match the scored requests of {}",
            scored.system.label
        )))
    }
}

/// The SHA-256 of ids as little-endian `u32`, the encoding of the repository's id dumps.
pub(crate) fn ids_sha256(ids: &[u32]) -> String {
    let mut hasher = Sha256::new();
    for id in ids {
        hasher.update(id.to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}

/// One first departure between the reference and the baseline execution.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub(crate) struct Departure {
    pub(crate) request: String,
    pub(crate) position: usize,
    pub(crate) reference_token: u32,
    pub(crate) baseline_token: u32,
    /// `|log p(reference_token) - log p(baseline_token)|` under the scorer, in nats.
    pub(crate) margin: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(crate) struct Calibration {
    /// The near-tie threshold: the largest departure margin rounded up to `DELTA_GRID`.
    pub(crate) delta: f64,
    pub(crate) largest_margin: f64,
    pub(crate) requests: usize,
    pub(crate) departures: Vec<Departure>,
}

/// Sets the near-tie threshold from the reference's own behaviour. Two legal executions of the
/// reference (for example two prefill chunk sizes) generate from the same prompts; where they
/// first depart, their histories still agree, so the scorer judges both tokens in one context,
/// and the gap between their log-probabilities is a margin that reduction order alone produced.
/// Without a departure there is no evidence to set a threshold, and calibration fails.
pub(crate) fn calibrate(reference: &ScoredSystem, baseline: &ScoredSystem) -> Result<Calibration> {
    check_pair(reference, baseline)?;
    let mut departures = Vec::new();
    for (left, right) in reference.requests.iter().zip(&baseline.requests) {
        let first = left
            .records
            .iter()
            .zip(&right.records)
            .find(|(a, b)| a.generated != b.generated);
        let Some((a, b)) = first else { continue };
        if a.argmax != b.argmax || a.argmax_logprob != b.argmax_logprob {
            return Err(gate_error(format!(
                "request {} position {}: the scorer gave two rows for one context; \
                 calibration needs a deterministic scorer",
                left.name, a.position
            )));
        }
        departures.push(Departure {
            request: left.name.clone(),
            position: a.position,
            reference_token: a.generated,
            baseline_token: b.generated,
            margin: (a.logprob - b.logprob).abs(),
        });
    }
    let largest_margin = departures
        .iter()
        .map(|departure| departure.margin)
        .fold(f64::NEG_INFINITY, f64::max);
    if departures.is_empty() {
        return Err(gate_error(
            "the reference and the baseline never departed; use executions that differ in \
             reduction order, or more or longer requests"
                .into(),
        ));
    }
    // A zero margin is an exact tie; the threshold still takes the first grid step so that a
    // legal tie is not a large-margin event.
    let steps = (largest_margin / DELTA_GRID).ceil().max(1.0);
    Ok(Calibration {
        delta: steps * DELTA_GRID,
        largest_margin,
        requests: reference.requests.len(),
        departures,
    })
}

/// The class of a mismatch by its margin.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum MarginClass {
    /// Margin at most delta.
    NearTie,
    /// Margin above delta, at most twice delta.
    Intermediate,
    /// Margin above twice delta: an event for the event rule.
    LargeMargin,
}

impl MarginClass {
    fn of(margin: f64, delta: f64) -> Self {
        if margin <= delta {
            Self::NearTie
        } else if margin <= 2.0 * delta {
            Self::Intermediate
        } else {
            Self::LargeMargin
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum EventStatus {
    Explained,
    Unexplained,
    Unaudited,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(crate) struct Mismatch {
    pub(crate) request: String,
    pub(crate) position: usize,
    /// The position in the whole context: prompt tokens plus `position`.
    pub(crate) context_position: usize,
    pub(crate) generated: u32,
    pub(crate) argmax: u32,
    pub(crate) margin: f64,
    pub(crate) class: MarginClass,
    /// Set for large-margin events only.
    pub(crate) event: Option<EventStatus>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(crate) struct DepthBin {
    pub(crate) from: usize,
    pub(crate) to: Option<usize>,
    pub(crate) tokens: u64,
    pub(crate) mismatches: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(crate) struct SystemSummary {
    pub(crate) system: ExecutionIdentity,
    pub(crate) requests: usize,
    pub(crate) tokens: u64,
    pub(crate) shortest_request: usize,
    pub(crate) mismatches: u64,
    pub(crate) rate: f64,
    pub(crate) rate_interval_95: Interval,
    pub(crate) near_tie_mismatches: u64,
    pub(crate) intermediate_mismatches: u64,
    pub(crate) events: u64,
    pub(crate) explained_events: u64,
    pub(crate) unexplained_events: u64,
    pub(crate) unaudited_events: u64,
    /// Positions where the scorer's own top two are within delta, matched or not.
    pub(crate) reference_near_ties: u64,
    pub(crate) depth: Vec<DepthBin>,
    pub(crate) mismatch_list: Vec<Mismatch>,
    /// Per request: (mismatches, tokens), the clusters of the clustered check.
    #[serde(skip)]
    clusters: Vec<(u64, u64)>,
}

impl SystemSummary {
    fn new(scored: &ScoredSystem, delta: f64) -> Result<Self> {
        let mut summary = Self::empty(scored);
        for request in &scored.requests {
            summary.add_request(request, delta)?;
        }
        summary.rate = summary.mismatches as f64 / summary.tokens.max(1) as f64;
        summary.rate_interval_95 = wilson(summary.mismatches, summary.tokens, Z_TWO_SIDED_95)
            .map_err(|_| gate_error(format!("{} scored no tokens", scored.system.label)))?;
        Ok(summary)
    }

    fn empty(scored: &ScoredSystem) -> Self {
        let mut edges = vec![0];
        edges.extend(DEPTH_EDGES);
        let depth = edges
            .iter()
            .enumerate()
            .map(|(index, &from)| DepthBin {
                from,
                to: edges.get(index + 1).copied(),
                tokens: 0,
                mismatches: 0,
            })
            .collect();
        Self {
            system: scored.system.clone(),
            requests: scored.requests.len(),
            tokens: 0,
            shortest_request: usize::MAX,
            mismatches: 0,
            rate: 0.0,
            rate_interval_95: Interval {
                lower: 0.0,
                upper: 1.0,
            },
            near_tie_mismatches: 0,
            intermediate_mismatches: 0,
            events: 0,
            explained_events: 0,
            unexplained_events: 0,
            unaudited_events: 0,
            reference_near_ties: 0,
            depth,
            mismatch_list: Vec::new(),
            clusters: Vec::new(),
        }
    }

    fn add_request(&mut self, request: &ScoredRequest, delta: f64) -> Result<()> {
        let audits: HashMap<usize, &Audit> = request
            .audits
            .iter()
            .map(|audit| (audit.position, audit))
            .collect();
        let mut mismatches = 0;
        for (index, record) in request.records.iter().enumerate() {
            if record.position != index {
                return Err(gate_error(format!(
                    "request {}: record {index} names position {}",
                    request.name, record.position
                )));
            }
            self.reference_near_ties += u64::from(record.top2_gap <= delta);
            let bin = self.depth.iter_mut().rev().find(|bin| bin.from <= index);
            let bin = bin.expect("the first depth bin starts at zero");
            bin.tokens += 1;
            if record.matched() {
                continue;
            }
            bin.mismatches += 1;
            mismatches += 1;
            self.add_mismatch(request, record, audits.get(&index).copied(), delta);
        }
        let tokens = request.records.len() as u64;
        self.tokens += tokens;
        self.mismatches += mismatches;
        self.shortest_request = self.shortest_request.min(request.records.len());
        self.clusters.push((mismatches, tokens));
        Ok(())
    }

    fn add_mismatch(
        &mut self,
        request: &ScoredRequest,
        record: &TokenRecord,
        audit: Option<&Audit>,
        delta: f64,
    ) {
        let margin = record.margin();
        let class = MarginClass::of(margin, delta);
        let event = (class == MarginClass::LargeMargin).then(|| match audit {
            None => EventStatus::Unaudited,
            Some(audit) if audit.explains(record) => EventStatus::Explained,
            Some(_) => EventStatus::Unexplained,
        });
        match class {
            MarginClass::NearTie => self.near_tie_mismatches += 1,
            MarginClass::Intermediate => self.intermediate_mismatches += 1,
            MarginClass::LargeMargin => self.events += 1,
        }
        match event {
            Some(EventStatus::Explained) => self.explained_events += 1,
            Some(EventStatus::Unexplained) => self.unexplained_events += 1,
            Some(EventStatus::Unaudited) => self.unaudited_events += 1,
            None => {}
        }
        self.mismatch_list.push(Mismatch {
            request: request.name.clone(),
            position: record.position,
            context_position: request.prompt_tokens + record.position,
            generated: record.generated,
            argmax: record.argmax,
            margin,
            class,
            event,
        });
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(crate) struct ClusteredCheck {
    pub(crate) lower_bound: f64,
    pub(crate) exceeds_limit: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(crate) struct RateRule {
    pub(crate) limit: f64,
    pub(crate) z: f64,
    pub(crate) baseline_rate: f64,
    pub(crate) candidate_rate: f64,
    pub(crate) difference: f64,
    /// The one-sided 95% lower bound of `candidate_rate - baseline_rate` (Newcombe).
    pub(crate) lower_bound: f64,
    pub(crate) reject: bool,
    /// The same bound with sequences as clusters. A diagnostic: it does not decide the verdict,
    /// and it is absent with fewer than two requests.
    pub(crate) clustered: Option<ClusteredCheck>,
}

impl RateRule {
    fn new(baseline: &SystemSummary, candidate: &SystemSummary) -> Result<Self> {
        let interval = newcombe_difference(
            (candidate.mismatches, candidate.tokens),
            (baseline.mismatches, baseline.tokens),
            Z_ONE_SIDED_95,
        )?;
        let clustered = match (
            clustered_rate(&candidate.clusters),
            clustered_rate(&baseline.clusters),
        ) {
            (Ok((pc, vc)), Ok((pb, vb))) => {
                let lower_bound = pc - pb - Z_ONE_SIDED_95 * (vc + vb).sqrt();
                Some(ClusteredCheck {
                    lower_bound,
                    exceeds_limit: lower_bound > RATE_EXCESS_LIMIT,
                })
            }
            _ => None,
        };
        Ok(Self {
            limit: RATE_EXCESS_LIMIT,
            z: Z_ONE_SIDED_95,
            baseline_rate: baseline.rate,
            candidate_rate: candidate.rate,
            difference: candidate.rate - baseline.rate,
            lower_bound: interval.lower,
            reject: interval.lower > RATE_EXCESS_LIMIT,
            clustered,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(crate) struct EventRule {
    /// Twice the near-tie threshold.
    pub(crate) threshold: f64,
    pub(crate) alpha: f64,
    pub(crate) baseline_unexplained: u64,
    pub(crate) baseline_tokens: u64,
    /// Unexplained plus unaudited: an event past the audit cap is not given the benefit of
    /// the doubt.
    pub(crate) candidate_unexplained: u64,
    pub(crate) candidate_tokens: u64,
    pub(crate) p_value: f64,
    pub(crate) reject: bool,
}

impl EventRule {
    fn new(baseline: &SystemSummary, candidate: &SystemSummary, delta: f64) -> Result<Self> {
        let candidate_unexplained = candidate.unexplained_events + candidate.unaudited_events;
        let p_value = rate_excess_p_value(
            (candidate_unexplained, candidate.tokens),
            (baseline.unexplained_events, baseline.tokens),
        )?;
        Ok(Self {
            threshold: 2.0 * delta,
            alpha: EVENT_ALPHA,
            baseline_unexplained: baseline.unexplained_events,
            baseline_tokens: baseline.tokens,
            candidate_unexplained,
            candidate_tokens: candidate.tokens,
            p_value,
            reject: p_value < EVENT_ALPHA,
        })
    }
}

/// What a run must cover before a pass means anything.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub(crate) struct GateConfig {
    /// Every request of the baseline and the candidate must score at least this many tokens.
    pub(crate) min_tokens_per_request: usize,
}

impl GateConfig {
    /// The qualification setting: 4,096 generated tokens, twice SpecPilot's development length,
    /// so that a fault that starts after 2,048 tokens is inside every request.
    pub(crate) const QUALIFICATION: Self = Self {
        min_tokens_per_request: 4096,
    };
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "verdict", rename_all = "snake_case")]
pub(crate) enum Verdict {
    Pass,
    Reject { reasons: Vec<String> },
    Inconclusive { reasons: Vec<String> },
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(crate) struct Report {
    pub(crate) schema: &'static str,
    pub(crate) config: GateConfig,
    pub(crate) scorer: ExecutionIdentity,
    pub(crate) calibration: Calibration,
    pub(crate) reference: SystemSummary,
    pub(crate) baseline: SystemSummary,
    pub(crate) candidate: SystemSummary,
    pub(crate) rate_rule: RateRule,
    pub(crate) event_rule: EventRule,
    pub(crate) verdict: Verdict,
}

/// The gate. All three executions were scored by one scorer on the same weights.
///
/// - `reference` runs the scorer's own numeric path (SpecPilot's autoregressive run). Its
///   mismatch rate is near zero by construction, so it only calibrates delta.
/// - `baseline` is a second legal execution of the accepted build that differs from the scorer
///   in reduction order only, such as another prefill chunk size (SpecPilot's stock system).
///   It calibrates delta with the reference and is the baseline of both rules: a lossless
///   candidate's mismatches are near ties of the same kind.
/// - `candidate` is the execution under test.
pub(crate) fn evaluate(
    config: GateConfig,
    reference: &ScoredSystem,
    baseline: &ScoredSystem,
    candidate: &ScoredSystem,
) -> Result<Report> {
    check_pair(baseline, candidate)?;
    let calibration = calibrate(reference, baseline)?;
    let delta = calibration.delta;
    let reference_summary = SystemSummary::new(reference, delta)?;
    let baseline_summary = SystemSummary::new(baseline, delta)?;
    let candidate_summary = SystemSummary::new(candidate, delta)?;
    let rate_rule = RateRule::new(&baseline_summary, &candidate_summary)?;
    let event_rule = EventRule::new(&baseline_summary, &candidate_summary, delta)?;
    let verdict = verdict(
        config,
        (&baseline_summary, &candidate_summary),
        &rate_rule,
        &event_rule,
    );
    Ok(Report {
        schema: REPORT_SCHEMA,
        config,
        scorer: reference.scorer.clone(),
        calibration,
        reference: reference_summary,
        baseline: baseline_summary,
        candidate: candidate_summary,
        rate_rule,
        event_rule,
        verdict,
    })
}

fn verdict(
    config: GateConfig,
    (baseline, candidate): (&SystemSummary, &SystemSummary),
    rate_rule: &RateRule,
    event_rule: &EventRule,
) -> Verdict {
    let mut rejections = Vec::new();
    if rate_rule.reject {
        rejections.push(format!(
            "rate rule: the candidate's mismatch rate {:.4}% exceeds the baseline's {:.4}% by at \
             least {:.4} points at one-sided 95% confidence (limit {} points)",
            100.0 * rate_rule.candidate_rate,
            100.0 * rate_rule.baseline_rate,
            100.0 * rate_rule.lower_bound,
            100.0 * RATE_EXCESS_LIMIT
        ));
    }
    if event_rule.reject {
        rejections.push(format!(
            "event rule: {} unexplained mismatches above {} nats against the baseline's {} \
             (one-sided p = {:.3e} < {EVENT_ALPHA})",
            event_rule.candidate_unexplained,
            event_rule.threshold,
            event_rule.baseline_unexplained,
            event_rule.p_value
        ));
    }
    if !rejections.is_empty() {
        return Verdict::Reject {
            reasons: rejections,
        };
    }
    let mut gaps = Vec::new();
    for summary in [baseline, candidate] {
        if summary.shortest_request < config.min_tokens_per_request {
            gaps.push(format!(
                "{}: a request scored {} tokens, below the {} the gate requires",
                summary.system.label, summary.shortest_request, config.min_tokens_per_request
            ));
        }
    }
    if baseline.unaudited_events > 0 {
        gaps.push(format!(
            "{}: {} baseline events were not audited, so the event baseline is not known",
            baseline.system.label, baseline.unaudited_events
        ));
    }
    if gaps.is_empty() {
        Verdict::Pass
    } else {
        Verdict::Inconclusive { reasons: gaps }
    }
}

/// Two scored systems are comparable when one reference scored both, on the same weights as the
/// systems themselves, over the same prompts.
fn check_pair(first: &ScoredSystem, second: &ScoredSystem) -> Result<()> {
    if first.scorer != second.scorer {
        return Err(gate_error(format!(
            "{} and {} were scored by different references",
            first.system.label, second.system.label
        )));
    }
    for system in [&first.system, &second.system] {
        let scorer = &first.scorer;
        if system.weights != scorer.weights || system.weight_format != scorer.weight_format {
            return Err(gate_error(format!(
                "{} runs {} ({}) but the reference runs {} ({}): token forcing compares \
                 executions of the same weights; judge a change of weights or format with the \
                 non-inferiority gate",
                system.label,
                system.weights,
                system.weight_format,
                scorer.weights,
                scorer.weight_format
            )));
        }
    }
    let same_prompts = first.requests.len() == second.requests.len()
        && first.requests.iter().zip(&second.requests).all(|(a, b)| {
            a.name == b.name
                && a.prompt_sha256 == b.prompt_sha256
                && a.prompt_tokens == b.prompt_tokens
        });
    if !same_prompts {
        return Err(gate_error(format!(
            "{} and {} did not run the same prompts",
            first.system.label, second.system.label
        )));
    }
    Ok(())
}

fn gate_error(message: String) -> EngError {
    EngError::BadRequest(format!("token forcing gate: {message}"))
}
