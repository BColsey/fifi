//! The planned-decision types the capability preflight renders and the
//! families' load plans fill: one row type, shared, so no second row
//! shape grows beside it.

use serde::Serialize;

/// When a planned decision is settled: at engine start (from the
/// configuration) or at load (from the measured device budget).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PlanFixity {
    /// Settled when the engine starts: changing it needs a new config.
    Start,
    /// Re-decided at load against the measured free device bytes.
    Load,
}

impl PlanFixity {
    pub fn name(self) -> &'static str {
        match self {
            PlanFixity::Start => "engine start",
            PlanFixity::Load => "load",
        }
    }
}

/// One alternative a planned decision weighed.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PlanAlternative {
    pub option: String,
    /// `chosen`, `fits`, `does not fit`, or `refused`.
    pub verdict: String,
    pub reason: String,
}

/// One planned decision the estimate embodies.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PlanDecision {
    pub name: String,
    pub chosen: String,
    pub alternatives: Vec<PlanAlternative>,
    pub fixed_at: PlanFixity,
}
