//! How the host reads one token of the resident step back, and the mode switches around it. No
//! device type appears here, so the rules are tested on the CPU without a `CudaContext`.
//!
//! # Read-back
//!
//! One host wait per token. The pair copy is queued and an event recorded behind it; the logits
//! row (when wanted) is read next, which waits for the stream; the pair is read last and its
//! event is complete by then. See [`ReadbackPlan`]. The start block of the token is one blocking
//! host-to-device copy before the launches ([`HOST_WRITES_PER_TOKEN`]); the report counts it
//! apart from the wait.
//!
//! # Modes
//!
//! `INFERENCE_CUDA_SYNC_EACH=1` makes the eager step synchronize after every launch, for
//! compute-sanitizer. A synchronize inside a capture window is refused, so the mode and a capture
//! exclude each other.

use super::run_report::{CaptureOutcome, HostSyncs, RunReport};
use crate::EngError;
use crate::family::sampler::SamplingParams;

pub(crate) const SYNC_EACH_VAR: &str = "INFERENCE_CUDA_SYNC_EACH";
pub(crate) const SHORTLIST_VAR: &str = "Q36_CUDA_DEVICE_SHORTLIST";

pub(crate) fn device_shortlist_enabled(value: Option<&str>) -> bool {
    value.is_none() || value == Some("1")
}

/// Only these modes preserve the host draw from the device's complete top-20 certificate.
pub(crate) fn device_sampling_supported(
    params: SamplingParams,
    sync_each: bool,
    vocab_matches: bool,
) -> bool {
    vocab_matches && (params.is_greedy() || (!sync_each && (1..=20).contains(&params.top_k())))
}

/// The open-vocabulary draw (`top_k` 0 at a positive temperature) rides the same device
/// shortlist with the softmax mass of the whole row beside it; the sampler certifies each draw
/// against the mass bound and reads the full row only when it cannot.
pub(crate) fn device_open_supported(
    params: SamplingParams,
    sync_each: bool,
    vocab_matches: bool,
) -> bool {
    vocab_matches && !sync_each && !params.is_greedy() && params.top_k() == 0
}
const HOST_WRITES_PER_TOKEN: u32 = 1;

pub(crate) fn parse_sync_each(value: Option<&str>) -> Result<bool, EngError> {
    match value {
        None | Some("" | "0") => Ok(false),
        Some("1") => Ok(true),
        Some(other) => Err(EngError::BadRequest(format!(
            "{SYNC_EACH_VAR} must be 0 or 1, got {other:?}"
        ))),
    }
}

pub(crate) fn check_mode(sync_each: bool, capture_requested: bool) -> Result<(), EngError> {
    if sync_each && capture_requested {
        return Err(EngError::BadRequest(format!(
            "{SYNC_EACH_VAR}=1 synchronizes after every launch, which a graph capture forbids; \
             set \"capture\": false in the qwen36 settings to run the eager step with it"
        )));
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ReadStep {
    QueueShortlist,
    QueueMass,
    QueuePairCopy,
    RecordEvent,
    ReadLogits,
    ReadShortlist,
    ReadMass,
    ReadPair,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ReadbackPlan {
    pub(crate) logits_bytes: Option<usize>,
    pub(crate) shortlist: bool,
    /// The softmax mass of the row is queued after the shortlist and its partials are read.
    pub(crate) mass: bool,
}

impl ReadbackPlan {
    pub(crate) fn new(vocab: usize, want_logits: bool) -> Result<Self, EngError> {
        if vocab == 0 {
            return Err(EngError::BadRequest(
                "the read-back plan needs a vocabulary of at least one value".to_owned(),
            ));
        }
        let logits_bytes = if want_logits {
            let bytes = vocab
                .checked_mul(size_of::<f32>())
                .ok_or_else(|| EngError::BadRequest("the logits row size overflows".to_owned()))?;
            Some(bytes)
        } else {
            None
        };
        Ok(Self {
            logits_bytes,
            shortlist: false,
            mass: false,
        })
    }

    pub(crate) fn shortlist(vocab: usize) -> Result<Self, EngError> {
        let mut plan = Self::new(vocab, false)?;
        plan.shortlist = true;
        Ok(plan)
    }

    /// The shortlist with the softmax mass of the row: what an open-vocabulary draw reads.
    pub(crate) fn open_shortlist(vocab: usize) -> Result<Self, EngError> {
        let mut plan = Self::shortlist(vocab)?;
        plan.mass = true;
        Ok(plan)
    }

    /// The pair copy is queued and its event recorded before any
    /// blocking read, so the first blocking read waits for everything the token queued.
    pub(crate) fn steps(&self) -> Vec<ReadStep> {
        let mut steps = Vec::new();
        if self.shortlist {
            steps.push(ReadStep::QueueShortlist);
        }
        if self.mass {
            steps.push(ReadStep::QueueMass);
        }
        steps.extend([ReadStep::QueuePairCopy, ReadStep::RecordEvent]);
        if self.logits_bytes.is_some() {
            steps.push(ReadStep::ReadLogits);
        }
        if self.shortlist {
            steps.push(ReadStep::ReadShortlist);
        }
        if self.mass {
            steps.push(ReadStep::ReadMass);
        }
        steps.push(ReadStep::ReadPair);
        steps
    }

    /// Host waits that can block: the first read after the record. The reads behind it find the
    /// work complete.
    pub(crate) fn blocking_waits(&self) -> u32 {
        let first_read = self.steps().into_iter().position(|step| {
            matches!(
                step,
                ReadStep::ReadLogits
                    | ReadStep::ReadShortlist
                    | ReadStep::ReadMass
                    | ReadStep::ReadPair
            )
        });
        u32::from(first_read.is_some())
    }
}

/// The report of a tiered token: `report` of the plain path, plus the host waits of the layers and
/// of the bank fills. Every one of them blocks the host, so each counts as a sync, and each
/// blocking write also counts as a host write. WHY: the plain report says one wait per token, and
/// the tiered step makes about four per layer.
pub(crate) fn with_host_syncs(mut report: RunReport, extra: HostSyncs) -> RunReport {
    report.syncs += extra.waits + extra.writes;
    report.host_writes += extra.writes;
    report
}

pub(crate) fn eager_report(plan: &ReadbackPlan) -> RunReport {
    RunReport {
        capture: CaptureOutcome::default(),
        syncs: plan.blocking_waits(),
        host_writes: HOST_WRITES_PER_TOKEN,
    }
}

/// The read-back and the start block of a token are outside the graph, so capture changes
/// neither the waits nor the host writes.
pub(crate) fn captured_report(plan: &ReadbackPlan) -> RunReport {
    RunReport {
        capture: CaptureOutcome::Captured,
        syncs: plan.blocking_waits(),
        host_writes: HOST_WRITES_PER_TOKEN,
    }
}

pub(crate) fn refused_report(plan: &ReadbackPlan) -> RunReport {
    RunReport {
        capture: CaptureOutcome::RefusedFellBackToEager,
        syncs: plan.blocking_waits(),
        host_writes: HOST_WRITES_PER_TOKEN,
    }
}
