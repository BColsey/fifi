//! The fault's reach on a reference output, measured before a qualification spends its leases.
//!
//! The fault self-test proves the gate only when the fault changes tokens: on a repetitive
//! output, where the top two logits lie far apart, a logit bias changes nothing and the test
//! passes a faulty candidate. The probe generates each task's reference output on the
//! reference's own path, as `greedy_generate` does, and reads every row of it from the fault's
//! start up to the first id that would end the answer: the distinct ids, the positions whose
//! top two logits lie closer than the bias, the positions where the bias changes the argmax
//! (`fault_flips`), and those of them whose margin exceeds `EVENT_FLOOR` (`fault_events`).
//!
//! The counts read the reference's history; the faulty candidate leaves that history at its
//! first flip, so they estimate the fault's reach rather than predict its mismatches. The probe
//! file records what ran it and the SHA-256 of each generated output, and `check_probe` ties it
//! to the scored reference of the qualification that relies on it.

use super::gate::{ExecutionIdentity, GateConfig, ScoredSystem, ids_sha256};
use super::score::{ForcedModel, LogitBiasFault, TokenRecord};
use crate::EngError;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

type Result<T> = std::result::Result<T, EngError>;

/// Twice the largest delta the gate accepts: a fault flip with a larger margin is an event
/// under every calibration that qualifies.
pub(crate) const EVENT_FLOOR: f64 = 2.0 * GateConfig::QUALIFICATION.max_delta;

/// One task's reference output, read for what the fault self-test needs.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct ProbeRow {
    pub(crate) task: String,
    pub(crate) prompt_sha256: String,
    pub(crate) prompt_tokens: usize,
    /// The SHA-256 of the generated ids (`ids_sha256`): the output the counts read.
    pub(crate) generated_sha256: String,
    /// The first output position whose argmax is an end id. A harness that runs past end ids
    /// generates text after the answer; the counts stop here.
    pub(crate) first_end: Option<usize>,
    /// Positions from the fault's start up to `first_end`.
    pub(crate) late_tokens: usize,
    pub(crate) late_distinct: usize,
    /// Late positions where the top two logits lie closer than the fault's bias.
    pub(crate) late_close: usize,
    /// Late positions where the fault's bias changes the argmax.
    pub(crate) fault_flips: usize,
    /// Flips whose margin exceeds `EVENT_FLOOR`.
    pub(crate) fault_events: usize,
}

/// The probe's file: what ran it, then one row per task in request order.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct ProbeFile {
    pub(crate) system: ExecutionIdentity,
    pub(crate) tokens: usize,
    pub(crate) event_floor: f64,
    pub(crate) rows: Vec<ProbeRow>,
}

/// What one probe reads: the fault whose reach it measures, the ids that end an answer, the
/// output length and the tokenizer-visible vocabulary.
#[derive(Clone, Copy)]
pub(crate) struct ProbeSettings<'a> {
    pub(crate) fault: &'a LogitBiasFault<()>,
    pub(crate) end_ids: &'a [u32],
    pub(crate) tokens: usize,
    pub(crate) vocab: usize,
}

/// Generates one task's output greedily through the model's prompt path and then its step path,
/// ties to the lower id, and counts the fault's reach in it.
pub(crate) fn probe_request(
    model: &mut dyn ForcedModel,
    (task, prompt): (&str, &[u32]),
    settings: ProbeSettings<'_>,
) -> Result<ProbeRow> {
    if prompt.is_empty() {
        return Err(probe_error("a probe needs a non-empty prompt".into()));
    }
    let mut tally = Tally::default();
    let mut generated = Vec::with_capacity(settings.tokens);
    if settings.tokens > 0 {
        model.reset()?;
        let mut row = model.prefill_logits(prompt)?;
        for position in 0..settings.tokens {
            generated.push(tally.read(position, &row, settings)?);
            if position + 1 < settings.tokens {
                row = model.step_logits(generated[position])?;
            }
        }
    }
    Ok(ProbeRow {
        task: task.to_owned(),
        prompt_sha256: ids_sha256(prompt),
        prompt_tokens: prompt.len(),
        generated_sha256: ids_sha256(&generated),
        first_end: tally.first_end,
        late_tokens: tally.late_tokens,
        late_distinct: tally.late.len(),
        late_close: tally.late_close,
        fault_flips: tally.fault_flips,
        fault_events: tally.fault_events,
    })
}

#[derive(Default)]
struct Tally {
    first_end: Option<usize>,
    late: BTreeSet<u32>,
    late_tokens: usize,
    late_close: usize,
    fault_flips: usize,
    fault_events: usize,
}

impl Tally {
    /// Counts the row at output position `position` and returns its greedy choice.
    fn read(&mut self, position: usize, row: &[f32], settings: ProbeSettings<'_>) -> Result<u32> {
        let record = TokenRecord::from_logits(position, row, 0, settings.vocab)?;
        let token = record.argmax;
        if self.first_end.is_none() && settings.end_ids.contains(&token) {
            self.first_end = Some(position);
        }
        let fault = settings.fault;
        if position < fault.from || self.first_end.is_some() {
            return Ok(token);
        }
        self.late.insert(token);
        self.late_tokens += 1;
        self.late_close += usize::from(record.top2_gap < f64::from(fault.bias));
        if let Some(departure) = fault.departure_at(position, row, settings.vocab)? {
            self.fault_flips += 1;
            self.fault_events += usize::from(departure.margin() > EVENT_FLOOR);
        }
        Ok(token)
    }
}

/// Refuses a probe that did not read the scored reference it is used with: another build,
/// pack or numeric profile, another prompt, or another output. Every reference request needs a
/// probe row, and the probe must have generated exactly the tokens the reference scored.
pub(crate) fn check_probe(probe: &ProbeFile, reference: &ScoredSystem) -> Result<()> {
    let mut claimed = probe.system.clone();
    claimed.label.clone_from(&reference.system.label);
    if claimed != reference.system {
        return Err(probe_error(format!(
            "the probe ran {:?}, not the reference's {:?}",
            probe.system, reference.system
        )));
    }
    for request in &reference.requests {
        let row = probe
            .rows
            .iter()
            .find(|row| row.task == request.name)
            .ok_or_else(|| probe_error(format!("no probe row for request {}", request.name)))?;
        if row.prompt_sha256 != request.prompt_sha256 || row.prompt_tokens != request.prompt_tokens
        {
            return Err(probe_error(format!(
                "request {}: the probe read another prompt",
                request.name
            )));
        }
        let generated: Vec<u32> = request
            .records
            .iter()
            .map(|record| record.generated)
            .collect();
        if probe.tokens != generated.len() || row.generated_sha256 != ids_sha256(&generated) {
            return Err(probe_error(format!(
                "request {}: the probe read another output than the {} tokens the reference \
                 scored",
                request.name,
                generated.len()
            )));
        }
    }
    Ok(())
}

fn probe_error(message: String) -> EngError {
    EngError::BadRequest(format!("token forcing probe: {message}"))
}
