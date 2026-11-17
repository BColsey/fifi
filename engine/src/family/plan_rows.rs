//! Decision rows: the one shape a family's load plan renders — the same
//! PlanDecision the capability preflight prints — with the load log line
//! as the consumer that exists.

// No plain-lib caller yet: the callers are the loader paths and the tests.
#![cfg_attr(not(test), allow(dead_code))]

use crate::family::plan_decision::PlanDecision;

/// A load plan that renders its decisions as rows.
pub(crate) trait PlanRows {
    fn plan_rows(&self) -> Vec<PlanDecision>;
}
