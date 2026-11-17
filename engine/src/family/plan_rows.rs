//! Decision rows: the one shape a family's load plan renders — the same
//! PlanDecision the capability preflight prints — with the load log line
//! as the consumer that exists.

// The plain host build compiles no caller of the trait: the CUDA preflight
// arm and the tests are the callers that exist.
#![cfg_attr(
    not(test),
    allow(dead_code, reason = "the plain host build compiles no caller")
)]

use crate::family::plan_decision::PlanDecision;

/// A load plan that renders its decisions as rows.
pub(crate) trait PlanRows {
    fn plan_rows(&self) -> Vec<PlanDecision>;
}

/// Facts in, rows out: the shape a family's preflight entry implements —
/// the rows the plan rendered, a one-line note the report appends, and the
/// configured context's fit as data for the report, never text to parse
/// back.
/// One borrowed request for a family's preflight rows: the directories
/// the candidate names, the candidate's own context and K/V dtype
/// choices, and the probed hardware. A family reads the fields it needs;
/// the shape never grows a field for one family alone.
pub(crate) struct RowsRequest<'a> {
    pub(crate) model_dir: &'a std::path::Path,
    #[allow(
        dead_code,
        reason = "read only by families whose settings resolution takes a tokenizer directory"
    )]
    pub(crate) tokenizer_dir: &'a std::path::Path,
    pub(crate) context: Option<u32>,
    pub(crate) kv_dtype: Option<&'a str>,
    #[allow(
        dead_code,
        reason = "read only by families whose settings resolution takes a prefill width"
    )]
    pub(crate) prefill_rows: Option<u32>,
    pub(crate) facts: &'a crate::capability::probe::HardwareFacts,
}

pub(crate) type RowsSource = fn(
    &RowsRequest,
) -> Result<
    (
        Vec<crate::family::plan_decision::PlanDecision>,
        String,
        RowsFit,
    ),
    crate::EngError,
>;

/// What the rows say about the configured context against the probed card.
#[derive(Debug)]
pub(crate) enum RowsFit {
    /// The plan admitted the configured context at the probed free bytes.
    Fits,
    /// The configured context does not fit. `largest` is the largest
    /// multiple of 1024 the same plan admits — a proposal, never applied.
    Constrained { requested: u32, largest: u32 },
    /// The probed facts could not carry a plan.
    Unknown,
}
