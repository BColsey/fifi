//! The host rules of the Qwen3.6 K-row verifier (`qwen36.draft_verify = krow`), as pure code.
//!
//! `krow` runs one verify window of K rows (2 to 8) as one captured K-row graph instead of K
//! one-row steps. This file holds what needs no device: the arming refusals, the check that
//! precedes every enqueue, and the row-status rules. The eager window stays the reference and the
//! fallback (`WindowHost::run_rows` defaults to the eager loop); the K-row device path (stage S2)
//! is not built, so a start that asks for `krow` is refused after these checks pass
//! ([`not_built`]).
//!
//! Arming. `krow` is one qualified configuration: the served `wide1` single-row kernel, the
//! reference MoE block, the device rollback. Everything that selects another kernel or another MoE
//! body is refused, not run:
//!
//! * Any value of `FIFI_EXL3_SINGLE_GATHER`, an explicit `wide1` included. The documented served
//!   default is `wide1`, so an operator script may set it; that is refused too, on purpose: one
//!   qualified configuration is simpler than a check on the effective kernel, and the message
//!   says "unset it". (Under `wide2`/`wide4` the multi-row groups never reach the rows kernel, and
//!   under `off`/`1` they take the legacy or the gather kernel: the cost model and the counters
//!   would describe another configuration.)
//! * `FIFI_EXL3_SHARED_IN_ROUTED=1`, and the step's test-only `set_shared_in_routed` hook, which
//!   builds the same merged block: that block takes one row and owns a one-row working buffer, so
//!   a K-row body over it would be a second body with its own tests. The step reads its own
//!   field, not only the environment, so the hook meets the same refusal as the variable.
//!
//! The KV8 switches belong to stage S4 and are not decided here.

use std::ffi::OsStr;

use super::{
    exl3_merge_plan::{SHARED_IN_ROUTED_VAR, shared_in_routed_from_env},
    rollback::{MAX_ROWS, check_window_fits},
};
use crate::EngError;

/// The environment variable that selects the single-row EXL3 kernel
/// (`kernels/cuda/resident/exl3.rs`, `SINGLE_GATHER_VAR`; a test pins that the two agree).
pub(crate) const SINGLE_GATHER_NAME: &str = "FIFI_EXL3_SINGLE_GATHER";

fn refuse(why: &str) -> EngError {
    EngError::BadRequest(format!("draft_verify krow: {why}"))
}

/// What the arming of the K-row path knows. The step builds it from its own fields so that a
/// test-only switch flip is seen, not only the process environment.
#[derive(Clone, Copy, Debug)]
pub(crate) struct KrowFacts<'a> {
    /// The device rollback is armed (its slots are where the K-row kernels store the states).
    pub(crate) rollback_armed: bool,
    /// The weights are an EXL3 pack.
    pub(crate) exl3: bool,
    /// Rows of the longest window.
    pub(crate) rows: usize,
    /// The value of `FIFI_EXL3_SINGLE_GATHER`, `None` when unset.
    pub(crate) single_gather: Option<&'a OsStr>,
    /// The merged MoE block is built: the environment says `1`, or the test hook built it.
    pub(crate) shared_in_routed: bool,
}

/// The refusals of arming the K-row path, in the order the step reports them. Every one is
/// answered before any allocation, with the step untouched.
pub(crate) fn check_krow_arming(facts: KrowFacts<'_>) -> Result<(), EngError> {
    if !facts.rollback_armed {
        return Err(refuse(
            "needs the device rollback slots; set draft_rollback to device or draft_verify to eager",
        ));
    }
    if !facts.exl3 {
        return Err(refuse("needs the EXL3 pack"));
    }
    if !uses_krow(facts.rows) {
        return Err(refuse(&format!(
            "a window has 2 to {MAX_ROWS} rows, not {}",
            facts.rows
        )));
    }
    check_krow_switches(facts.single_gather, facts.shared_in_routed)
}

/// The two kernel switches. `single_gather` is the raw value of `FIFI_EXL3_SINGLE_GATHER`.
pub(crate) fn check_krow_switches(
    single_gather: Option<&OsStr>,
    shared_in_routed: bool,
) -> Result<(), EngError> {
    if let Some(value) = single_gather {
        return Err(refuse(&format!(
            "{SINGLE_GATHER_NAME} is set to {value:?}; unset it (the K-row path is qualified on the \
             served wide1 kernel only, an explicit wide1 included)"
        )));
    }
    if shared_in_routed {
        return Err(refuse(&format!(
            "the merged MoE block ({SHARED_IN_ROUTED_VAR}=1) takes one row; unset it"
        )));
    }
    Ok(())
}

/// [`check_krow_switches`] from the environment values: `shared` is the raw value of
/// [`SHARED_IN_ROUTED_VAR`]. A bad value is the refusal of the variable's own parser.
pub(crate) fn check_krow_env(
    single_gather: Option<&OsStr>,
    shared: Option<&OsStr>,
) -> Result<(), EngError> {
    let shared_in_routed = shared_in_routed_from_env(shared).map_err(EngError::BadRequest)?;
    check_krow_switches(single_gather, shared_in_routed)
}

/// The refusal of a start that asks for `krow` while the K-row device path does not exist (stage
/// S0 ships the seam only). The step is never built on this setting.
pub(crate) fn not_built() -> EngError {
    refuse(
        "the K-row device path is not built in this engine (stage S0 is the host seam); set draft_verify to eager",
    )
}

/// Whether a window of `rows` rows runs as a K-row window when `draft_verify` is `krow`. A one-row
/// window is the single-row step, which has its own captured graph, and runs the eager loop.
pub(crate) fn uses_krow(rows: usize) -> bool {
    (2..=MAX_ROWS).contains(&rows)
}

/// What the host knows of the state when a K-row window is about to be enqueued.
#[derive(Clone, Copy, Debug)]
pub(crate) struct RowsFacts {
    /// Committed rows now.
    pub(crate) base: usize,
    /// Rows the KV cache can hold.
    pub(crate) capacity: usize,
    /// The merged MoE block is built on this step (its field, not the environment: the test hook
    /// can build it after the arming). Read again before every window.
    pub(crate) merged_block: bool,
}

/// The check that precedes every enqueue of a K-row window: the merged block is not built (the
/// arming refusal, read again), the row count, every id of the window through `check_token` (the
/// step passes its `check_token_id` over the sampler width), then the KV capacity. Nothing is
/// written or launched before all of it passed, so a refusal leaves the state valid and unchanged.
/// One bad id among K is found here, not by the embedding kernel, which would set status 4 on the
/// device and poison the state.
pub(crate) fn check_rows(
    inputs: &[u32],
    check_token: impl Fn(u32) -> Result<(), EngError>,
    facts: &RowsFacts,
) -> Result<(), EngError> {
    check_krow_switches(None, facts.merged_block)?;
    if !uses_krow(inputs.len()) {
        return Err(refuse(&format!(
            "a window has 2 to {MAX_ROWS} rows, not {}",
            inputs.len()
        )));
    }
    for &token in inputs {
        check_token(token)?;
    }
    check_window_fits(facts.base, inputs.len(), facts.capacity)
}

/// The window word and the result pairs of a K-row window, as read after its one synchronization.
/// `bits[i]` is row `i`'s own status bits; `window` is the window word (`p->status` after the
/// fold). Status bits: 1 router fault, 2 non-finite logit, 4 invalid state, 16 arithmetic flag.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct RowStatus {
    pub(crate) bits: Vec<u32>,
    pub(crate) window: u32,
}

/// The verdict on the status of a window: every row's own bits AND the window word must be zero.
/// A non-zero value anywhere fails the whole window, as a failing row fails the eager window. The
/// pairs are checked one by one because the window word alone would not name the row; they are
/// checked in addition to the word because router and arithmetic bits live in the word only.
pub(crate) fn check_status(status: &RowStatus) -> Result<(), EngError> {
    if let Some((row, bits)) = status.bits.iter().enumerate().find(|(_, bits)| **bits != 0) {
        return Err(refuse(&format!("row {row} reports status {bits:#x}")));
    }
    if status.window != 0 {
        return Err(refuse(&format!(
            "the window word reports status {:#x}",
            status.window
        )));
    }
    Ok(())
}

#[cfg(test)]
#[path = "krow_tests.rs"]
mod tests;
