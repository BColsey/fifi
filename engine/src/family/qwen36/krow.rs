//! The host rules of the Qwen3.6 K-row verifier (`qwen36.draft_verify = krow`), as pure code.
//!
//! `krow` runs one verify window of K rows (2 to 8) as one captured K-row graph instead of K
//! one-row steps. This file holds what needs no device: the arming refusals, the check that
//! precedes every enqueue, and the row-status rules. The eager window stays the reference and the
//! fallback (`WindowHost::run_rows` defaults to the eager loop); the step overrides it with the
//! K-row body when `krow` is armed.
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
//! The KV8 switches belong to the long-context stage and are not decided here.
//!
//! The logits of a window stay on the device. The window reads one small report (the pairs, the
//! window word, the device comparison of the pair tokens with the window ids, and one top-20
//! record per row); this file holds the host rules that judge it: [`greedy_run`] and
//! [`check_verify`] (the accepted run and the bonus token, re-derived on the host from the pairs
//! and enforced against the device's), [`decode_shortlist`] and [`check_shortlists`] (the records
//! a seeded draw uses, checked against the pairs), [`DeviceDraws`] (what an open window keeps for
//! its draws) and [`draw_source`] (when a draw needs a full logits row after all).

use std::ffi::OsStr;

use super::{
    exl3_merge_plan::{SHARED_IN_ROUTED_VAR, shared_in_routed_from_env},
    rollback::{MAX_ROWS, check_window_fits},
};
use crate::{
    EngError,
    family::sampler::{SamplingParams, check_shortlist},
};

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
/// answered before any window runs, with the step untouched (the slots are already armed when
/// the step asks).
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
/// [`SHARED_IN_ROUTED_VAR`]. A bad value is the refusal of the variable's own parser; only `1`
/// refuses (unset is the default and yields to krow).
pub(crate) fn check_krow_env(
    single_gather: Option<&OsStr>,
    shared: Option<&OsStr>,
) -> Result<(), EngError> {
    // Only an explicit `1` is a refusal here: unset is the default, which yields to krow at
    // `request_krow_verify`, and `0` is off.
    let shared_in_routed =
        shared_in_routed_from_env(shared).map_err(EngError::BadRequest)? == Some(true);
    check_krow_switches(single_gather, shared_in_routed)
}

/// What the step knows of its own build beyond [`KrowFacts`]: the K-row body is written for one
/// configuration only.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ConfigFacts {
    /// The KV is the fp16 plane (the int8 cache has no K-row attention path yet).
    pub(crate) fp16_kv: bool,
    /// The step is neither tiered, nor in the per-op synchronization mode, nor forked.
    pub(crate) plain_step: bool,
    /// The embedding is the pack's bf16 table.
    pub(crate) bf16_embedding: bool,
    /// The head is an EXL3 matrix.
    pub(crate) exl3_head: bool,
}

/// The refusals of the build the K-row body does not cover, in the order the step reports them.
pub(crate) fn check_krow_configuration(facts: ConfigFacts) -> Result<(), EngError> {
    if !facts.fp16_kv {
        return Err(refuse(
            "runs the fp16 KV cache; the int8 cache is not built",
        ));
    }
    if !facts.plain_step {
        return Err(refuse(
            "needs the plain EXL3 step: not tiered, not in per-op synchronization, no forked experts",
        ));
    }
    if !facts.bf16_embedding || !facts.exl3_head {
        return Err(refuse("needs the bf16 embedding table and the EXL3 head"));
    }
    Ok(())
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

/// The tokens a K-row window drew, once the one read of its result table is judged: the pairs'
/// own bits and the window word are checked first ([`check_status`], so one bad row or one word
/// bit fails the whole window), then every token must lie inside the sampler vocabulary. `tokens`
/// are the table's token words, one per row (-1 for a row with no comparable value). The caller
/// treats any error as a failure after the launch.
pub(crate) fn rows_verdict(
    tokens: &[i32],
    status: &RowStatus,
    sampler_vocab: usize,
) -> Result<Vec<u32>, EngError> {
    if tokens.len() != status.bits.len() {
        return Err(refuse(&format!(
            "the result table holds {} tokens and {} status words",
            tokens.len(),
            status.bits.len()
        )));
    }
    check_status(status)?;
    tokens
        .iter()
        .enumerate()
        .map(|(row, &token)| {
            u32::try_from(token)
                .ok()
                .filter(|&id| (id as usize) < sampler_vocab)
                .ok_or_else(|| {
                    refuse(&format!(
                        "row {row} drew token {token}, outside the sampler vocabulary of {sampler_vocab}"
                    ))
                })
        })
        .collect()
}

/// Entries of a shortlist record (`kernels/cuda/native/resident/shortlist.h`).
pub(crate) const SHORTLIST_K: usize = 20;
/// Bytes of a shortlist record: 20 entries of (f32 logit, u32 id), the non-finite word, three pad
/// words. A test pins the value against the kernel wrapper's.
pub(crate) const SHORTLIST_RECORD_BYTES: usize = 176;
/// Byte of the record that holds the non-finite word.
const SHORTLIST_NON_FINITE_AT: usize = 160;

/// The accepted run of a greedy window and the token after it, as a host reference of
/// `verify_greedy`: `accepted` is the number of leading rows `i < K - 1` whose token equals the
/// next window id (`ids[i + 1]`), and `bonus` is the token of row `accepted`. A row with no token
/// (negative) never extends the run, whatever the window ids hold, as in the kernel. `tokens` and
/// `ids` have one entry per row (`K >= 1`).
pub(crate) fn greedy_run(tokens: &[i32], ids: &[u32]) -> (usize, i32) {
    let mut accepted = 0;
    while accepted + 1 < tokens.len()
        && tokens[accepted] >= 0
        && tokens[accepted] as u32 == ids[accepted + 1]
    {
        accepted += 1;
    }
    (accepted, tokens[accepted])
}

/// The device's comparison, judged: the run and the bonus the device stored must be the host's
/// own reference of the pairs. A device that disagrees with its own pair table is a window that
/// cannot be trusted, so the caller fails the window closed on any error.
pub(crate) fn check_verify(
    tokens: &[i32],
    ids: &[u32],
    accepted: i32,
    bonus: i32,
) -> Result<(usize, u32), EngError> {
    if tokens.is_empty() || tokens.len() != ids.len() {
        return Err(refuse(&format!(
            "the device verify judged {} tokens against {} window ids",
            tokens.len(),
            ids.len()
        )));
    }
    let (want, want_bonus) = greedy_run(tokens, ids);
    let got = usize::try_from(accepted)
        .ok()
        .filter(|run| *run < tokens.len())
        .ok_or_else(|| {
            refuse(&format!(
                "the device verify reports an accepted run of {accepted} in a window of {} rows",
                tokens.len()
            ))
        })?;
    if got != want || bonus != want_bonus {
        return Err(refuse(&format!(
            "the device verify says run {got} and bonus {bonus}; the pairs say run {want} and \
             bonus {want_bonus}"
        )));
    }
    let bonus = u32::try_from(bonus).map_err(|_| {
        refuse(&format!(
            "the device verify reports the bonus token {bonus}"
        ))
    })?;
    Ok((got, bonus))
}

/// The candidates of one shortlist record, ranked (larger logit first, then lower id), as
/// `Sampler::sample_shortlist` takes them: the same decoding the one-row step applies to its own
/// record. A non-zero non-finite word is the one-row step's error. `vocab` is the sampler
/// vocabulary; the first `min(vocab, 20)` entries are read.
pub(crate) fn decode_shortlist(record: &[u8], vocab: usize) -> Result<Vec<(u32, f32)>, EngError> {
    if record.len() != SHORTLIST_RECORD_BYTES {
        return Err(refuse(&format!(
            "a shortlist record is {SHORTLIST_RECORD_BYTES} bytes, got {}",
            record.len()
        )));
    }
    if record[SHORTLIST_NON_FINITE_AT..SHORTLIST_NON_FINITE_AT + 4] != [0; 4] {
        return Err(EngError::Backend {
            operation: "sample",
            message: "device shortlist found non-finite logits".to_owned(),
        });
    }
    Ok(record[..vocab.min(SHORTLIST_K) * 8]
        .as_chunks::<8>()
        .0
        .iter()
        .map(|entry| {
            (
                u32::from_le_bytes(entry[4..8].try_into().expect("candidate id word")),
                f32::from_le_bytes(entry[..4].try_into().expect("candidate logit word")),
            )
        })
        .collect())
}

/// Whether the record of row `row` has its non-finite word set.
fn record_non_finite(record: &[u8]) -> bool {
    record[SHORTLIST_NON_FINITE_AT..SHORTLIST_NON_FINITE_AT + 4] != [0; 4]
}

/// The shortlist records against the pairs, for a window whose status already passed: a row with
/// no status bit has a clean record, and the first entry of its record (the largest logit, the
/// lowest id among equals) is the token the argmax chose. Two independent reductions of the same
/// row must agree; a window where they do not fails closed. Returns each row's decoded candidates.
pub(crate) fn check_shortlists(
    tokens: &[i32],
    bits: &[i32],
    records: &[[u8; SHORTLIST_RECORD_BYTES]],
    vocab: usize,
) -> Result<Vec<Vec<(u32, f32)>>, EngError> {
    if records.len() != tokens.len() || bits.len() != tokens.len() {
        return Err(refuse(&format!(
            "the report holds {} shortlist records for {} rows",
            records.len(),
            tokens.len()
        )));
    }
    let mut out = Vec::with_capacity(records.len());
    for (row, record) in records.iter().enumerate() {
        if bits[row] == 0 {
            if record_non_finite(record) {
                return Err(refuse(&format!(
                    "row {row}: the shortlist saw a non-finite value that the argmax did not"
                )));
            }
            let candidates = decode_shortlist(record, vocab)?;
            let first = candidates.first().map(|candidate| i64::from(candidate.0));
            if first != Some(i64::from(tokens[row])) {
                return Err(refuse(&format!(
                    "row {row}: the shortlist starts at {first:?} and the argmax chose {}",
                    tokens[row]
                )));
            }
            // Every entry, by the predicate the sampler applies before it draws: a corrupt later
            // entry fails the window here, not a draw of it.
            check_shortlist(&candidates, vocab)
                .map_err(|error| refuse(&format!("row {row}: {error:?}")))?;
            out.push(candidates);
        } else {
            // The window fails on this row's bit; its record is not read.
            out.push(Vec::new());
        }
    }
    Ok(out)
}

/// What an open K-row window keeps for its draws, read once with the report.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct DeviceDraws {
    /// The committed rows when the window began.
    pub(crate) base: usize,
    /// The window ids.
    pub(crate) inputs: Vec<u32>,
    /// The accepted run the device compared, checked against the pairs.
    pub(crate) accepted: usize,
    /// The token after the accepted run.
    pub(crate) bonus: u32,
    /// The ranked candidates of each row; empty when the shortlist did not run.
    pub(crate) shortlists: Vec<Vec<(u32, f32)>>,
}

impl DeviceDraws {
    /// Whether the draws belong to the open window that began at `base` over `inputs`.
    pub(crate) fn belongs_to(&self, base: usize, inputs: &[u32]) -> bool {
        self.base == base && self.inputs == inputs
    }
}

/// Where the draw of one row of a lazy window gets its logits when the sampler is not greedy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DrawSource {
    /// The row's top-20 record: `Sampler::sample_shortlist`.
    Shortlist,
    /// The complete row, read from the device (or the tap's copy): `Sampler::sample`.
    Row,
}

/// The source of a non-greedy draw. The shortlist serves it when the one-row step would use its own
/// (`readback::device_sampling_supported`, which a test pins against this rule): the sampler keeps
/// 1 to 20 candidates, the step does not synchronize after every launch, and the sampler vocabulary
/// is the width the shortlist covers; and the record exists for the row.
pub(crate) fn draw_source(
    params: SamplingParams,
    sync_each: bool,
    vocab_matches: bool,
    record_present: bool,
) -> DrawSource {
    if record_present
        && vocab_matches
        && !sync_each
        && !params.is_greedy()
        && (1..=SHORTLIST_K as u32).contains(&params.top_k())
    {
        DrawSource::Shortlist
    } else {
        DrawSource::Row
    }
}

/// What the draw hook of an open window knows when one non-greedy draw is asked for. The step and
/// the fake model of the tests build it from their own state and take their route from
/// [`draw_route`], so that the routing is one function, tested on the host.
#[derive(Clone, Copy, Debug)]
pub(crate) struct DrawFacts<'a> {
    /// The K-row path runs a window of the stored window's length (armed, not switched off).
    pub(crate) krow_runs: bool,
    /// The stored window keeps its state in the device slots (`WindowSave::Device`).
    pub(crate) device_window: bool,
    /// What the last K-row window kept for its draws, if any.
    pub(crate) draws: Option<&'a DeviceDraws>,
    /// The committed rows when the stored window began, and its inputs.
    pub(crate) base: usize,
    pub(crate) inputs: &'a [u32],
    /// The row asked for.
    pub(crate) row: usize,
    /// The window holds a host copy of this row's logits (the test tap read them).
    pub(crate) tap_row: bool,
    /// The sampler's settings, whether the step synchronizes after every launch, and whether the
    /// sampler vocabulary is the width the shortlist covers.
    pub(crate) params: SamplingParams,
    pub(crate) sync_each: bool,
    pub(crate) vocab_matches: bool,
}

/// Where one non-greedy draw of a stored window gets its values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DrawRoute {
    /// Not a window of the K-row path, or not the one `draws` belongs to: the default path draws.
    NotKrow,
    /// The row is past the window.
    OutOfWindow,
    /// The row's top-20 record.
    Shortlist,
    /// The host copy of the row's logits that the test tap read.
    TapRow,
    /// One row read from the device.
    DeviceRow,
}

/// The route of one non-greedy draw (a greedy draw is the pair token and never asks). Only a window
/// that the K-row path ran, whose records are those of this window (same base and inputs), is
/// served by the K-row routes; anything else is the default path's.
pub(crate) fn draw_route(facts: &DrawFacts<'_>) -> DrawRoute {
    let Some(draws) = facts.draws else {
        return DrawRoute::NotKrow;
    };
    if !facts.krow_runs
        || !facts.device_window
        || !draws.belongs_to(facts.base, facts.inputs)
    {
        return DrawRoute::NotKrow;
    }
    if facts.row >= draws.inputs.len() {
        return DrawRoute::OutOfWindow;
    }
    let record = draws
        .shortlists
        .get(facts.row)
        .is_some_and(|record| !record.is_empty());
    match draw_source(facts.params, facts.sync_each, facts.vocab_matches, record) {
        DrawSource::Shortlist => DrawRoute::Shortlist,
        DrawSource::Row if facts.tap_row => DrawRoute::TapRow,
        DrawSource::Row => DrawRoute::DeviceRow,
    }
}

#[cfg(test)]
#[path = "krow_tests.rs"]
mod tests;
