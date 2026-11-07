//! The K-row verifier (`draft_verify = krow`) against a fake device model whose window lifecycle IS
//! `window_logic`: the fake implements `WindowHost` with the real `RollbackBook` and overrides
//! `run_rows` with a native K-row body, or leaves it to the eager default. So the seam
//! (`run_rows`, `run_window_device`, `mark_captured_range`) and the host rules of `krow.rs` are
//! the production code; only the arithmetic and the device are modelled.
//!
//! What this proves: the order of the effects, the fail-closed routes at each new fault point, the
//! row-local status rules, the A/B equality of the two paths on the same model, and that every
//! flaw below is caught by the check meant to catch it. What it cannot prove: the kernels, the
//! slot byte layout and the stream order (device-only, in the later kernel and graph stages).

use super::{
    RowsFailure, RowsOut, WindowHost, commit_window, eager_rows, export_gate, fail_window,
    import_state, reset_epilogue, reset_prologue, run_window, state_usable, step_gate,
};
use crate::{
    EngError,
    family::{
        qwen36::{
            krow::{self, DeviceDraws, DrawFacts, DrawRoute, RowStatus, RowsFacts},
            rollback::{RollbackBook, RollbackPlan, StateFacts},
        },
        sampler::{Sampler, SamplingParams},
        session::{DecodeSession, Limits, TokenStepper},
        speculate::{
            d1_oracle::{
                D1Failure, D1Subject, FAKE_TOKENIZER_JSON, Fault, Setup, Tally, Verdict,
                accepted_prefix, failure_recovery, first_bit_difference, full_logits,
                krow_faults_for, krow_matches_eager, later_rows_isolated, repeated_rollbacks,
                slot_faults_for, ties_break_low,
            },
            driver::WindowModel,
            window::Commit,
        },
    },
    tokenizer::Tokenizer,
};

type Result<T> = std::result::Result<T, EngError>;

const VOCAB: usize = 8;
const CONTEXT: usize = 4_096;
const PROMPT: [u32; 12] = [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3];
const ALTERNATIVES: [u32; 4] = [0, 7, 3, 5];

/// Which path the fake runs a window on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Path {
    /// The trait default: one `run_row` and one `capture` per row.
    Eager,
    /// The native K-row body.
    Krow,
}

/// What is wrong with the fake. `Sound` is the correct one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Flaw {
    Sound,
    /// The snapshot of row `t` goes into slot `t + 1` (slot 0 is never written).
    SnapShifted,
    /// A partial accept restores slot `kept` instead of `kept - 1`.
    SlotOffByOne,
    /// The frontier after a rollback is one row too far.
    FrontierPlusOne,
    /// The first K-row graph is replayed for every later window, whatever its K.
    StaleGraph,
    /// The fake claims the K-row path but runs the eager default.
    EagerWhileClaimingKrow,
    /// The pair of a row stores the OR of the rows before it too (the `atomicOr` return).
    RacyStatus,
    /// The window keeps going after a non-zero status.
    IgnoresStatus,
    /// The host looks at the last row's pair only.
    LastPairOnly,
    /// The host looks at the pairs only and ignores the window word.
    PairsOnly,
    /// No host token check before the enqueue.
    NoTokenCheck,
    /// The switch re-check before every window is missing.
    NoSwitchRecheck,
    /// The snapshot kernel stores one row too many: it writes the slot after the last snapshot of
    /// the window, which no verdict of the window restores.
    SnapWritesPastTheWindow,
    /// A failure after the launch is reported as a state-decided one, so only the host counters
    /// and the validity of the state decide whether the window fails closed.
    AfterLaunchAsStateDecides,
    /// The host does not compare the device verify with the pairs.
    NoVerifyCheck,
    /// The window copies the logits rows to the host whether or not the tap asked (the lazy
    /// logits read silently on every window).
    ReadsLogitsEveryWindow,
    /// The K-row shortlist of row `r` is the record of row `r + 1` (the grid indexes the wrong row).
    ShortlistFromWrongRow,
    /// A failed read of a full row at draw time leaves the window open and the state valid.
    LazyReadNotGated,
    /// Entry 7 of the record of row 0 repeats the id of entry 3 (the first entry and the
    /// non-finite word are right, so only a check of every entry can see it).
    CorruptRecordEntry,
    /// The draw hook serves the records of the last K-row window to any window with the same base
    /// and inputs, also when the K-row path does not run it.
    ServesStaleDraws,
    /// A seeded draw that the sampler refuses (a record that changed after the report was judged)
    /// leaves the window open and the state valid.
    ShortlistDrawErrorNotGated,
}

/// The state a snapshot carries: the committed cells, never the stale ones above the frontier.
#[derive(Clone, Debug, PartialEq)]
struct SlotState {
    cells: Vec<u32>,
    recurrent: [f32; 3],
    conv: [f32; 2],
}

/// What the K-row body leaves: the logits rows, the window word and the live state after the last
/// row (recurrent, convolution, committed rows), not yet committed.
struct Body {
    logits: Vec<Vec<f32>>,
    word: u32,
    after: ([f32; 3], [f32; 2], usize),
}

/// How often each path ran: the negative control that the K-row path really was the one used.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Counters {
    krow_windows: usize,
    krow_replays: [usize; 9],
    eager_windows: usize,
}

struct DevFake {
    path: Path,
    flaw: Flaw,
    ties: bool,
    cells: Vec<u32>,
    len: usize,
    recurrent: [f32; 3],
    conv: [f32; 2],
    slots: Vec<([f32; 3], [f32; 2])>,
    book: RollbackBook,
    window: Option<super::Window<SlotState, Vec<f32>>>,
    poisoned: bool,
    gate: bool,
    pending: bool,
    fault: Option<Fault>,
    /// The merged MoE block is built (the step's `exl3_merge`).
    merged: bool,
    counters: Counters,
    /// K of the first K-row window: the one graph that `Flaw::StaleGraph` keeps replaying.
    graph_k: Option<usize>,
    /// The pairs and the window word the last K-row window read back.
    last_status: Option<RowStatus>,
    /// The tap: a K-row window also copies its logits rows to the host (`want_logits`). On by
    /// default so that the checks of the rows run; the lazy-logits tests turn it off.
    want_logits: bool,
    /// The logits rows that stay on the device after a K-row window.
    device_rows: Vec<Vec<f32>>,
    /// Full rows read from the device at draw time.
    row_reads: usize,
    /// Draws made from a shortlist record.
    shortlist_draws: usize,
    /// Windows that copied their logits rows to the host.
    logit_copies: usize,
    /// What the open K-row window keeps for its draws.
    draws: Option<DeviceDraws>,
}

fn refused(what: &str) -> EngError {
    EngError::BadRequest(format!("dev fake: {what}"))
}

/// The host check of one id, as the step's `check_token_id` over the sampler width.
fn check_token(token: u32) -> Result<()> {
    if token as usize >= VOCAB {
        return Err(refused(&format!(
            "token id {token} is outside the vocabulary"
        )));
    }
    Ok(())
}

/// One row of the model: attention over the cells below `pos`, the two state components and the
/// token. The arithmetic of the slot fake of `d1_slot_tests`.
fn model_row(
    cells: &[u32],
    pos: usize,
    recurrent: &mut [f32; 3],
    conv: &mut [f32; 2],
    token: u32,
    ties: bool,
) -> Vec<f32> {
    let attention: f32 = cells[..pos]
        .iter()
        .enumerate()
        .map(|(at, cell)| *cell as f32 * (0.01 + at as f32 * 0.0007))
        .sum();
    for (j, value) in recurrent.iter_mut().enumerate() {
        *value = *value * 0.75 + token as f32 * 0.125 * (j as f32 + 1.0);
    }
    for (j, value) in conv.iter_mut().enumerate() {
        *value = *value * 0.5 + token as f32 * 0.0625 * (j as f32 + 2.0);
    }
    let [r0, r1, r2] = *recurrent;
    let [c0, c1] = *conv;
    let mut row: Vec<f32> = (0..VOCAB)
        .map(|v| {
            let phase = r0 * 1.3
                + r1 * 0.7
                + r2 * 0.3
                + c0 * 0.9
                + c1 * 0.4
                + attention * 0.11
                + v as f32 * 0.9
                + token as f32 * 0.11;
            phase.sin() * 2.0 + ((pos + 1) as f32 * 0.01).cos()
        })
        .collect();
    if ties {
        row[2] = 9.0;
        row[6] = 9.0;
    }
    row
}

/// The device argmax: the largest finite value, ties to the lower index, a NaN beats nothing.
fn device_argmax(row: &[f32]) -> u32 {
    let mut best = 0;
    for (at, value) in row.iter().enumerate() {
        if value.is_finite() && (!row[best].is_finite() || *value > row[best]) {
            best = at;
        }
    }
    best as u32
}

/// The device comparison, written another way than the host reference (`krow::greedy_run`): the
/// first row whose token is not the next window id ends the run.
fn fake_verify(tokens: &[i32], ids: &[u32]) -> (i32, i32) {
    let ended = (0..tokens.len() - 1)
        .find(|&row| tokens[row] != ids[row + 1] as i32)
        .unwrap_or(tokens.len() - 1);
    (ended as i32, tokens[ended])
}

/// The K-row shortlist record of one row as the device writes it: the ranked finite candidates
/// (larger value first, then the lower index), the empty candidate after them, and the non-finite
/// word.
fn fake_record(row: &[f32]) -> [u8; krow::SHORTLIST_RECORD_BYTES] {
    let mut ranked: Vec<(u32, f32)> = row
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, value)| value.is_finite())
        .map(|(at, value)| (at as u32, value))
        .collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
    ranked.truncate(krow::SHORTLIST_K);
    let mut bytes = [0u8; krow::SHORTLIST_RECORD_BYTES];
    for at in 0..krow::SHORTLIST_K {
        let (id, value) = ranked
            .get(at)
            .copied()
            .unwrap_or((u32::MAX, f32::NEG_INFINITY));
        bytes[at * 8..at * 8 + 4].copy_from_slice(&value.to_le_bytes());
        bytes[at * 8 + 4..at * 8 + 8].copy_from_slice(&id.to_le_bytes());
    }
    let flag = u32::from(!row.iter().all(|value| value.is_finite()));
    bytes[160..164].copy_from_slice(&flag.to_le_bytes());
    bytes
}

fn slot_bits(slot: &([f32; 3], [f32; 2])) -> Vec<u32> {
    slot.0.iter().chain(&slot.1).map(|v| v.to_bits()).collect()
}

impl DevFake {
    fn new(path: Path, flaw: Flaw, ties: bool) -> Self {
        let plan = RollbackPlan::new(8, 1024).expect("plan");
        Self {
            path,
            flaw,
            ties,
            cells: Vec::new(),
            len: 0,
            recurrent: [0.0; 3],
            conv: [0.0; 2],
            // Garbage, not zeros: a slot that was never written must not look right.
            slots: vec![([7.5; 3], [7.5; 2]); plan.slots()],
            book: RollbackBook::new(plan),
            window: None,
            poisoned: false,
            gate: false,
            pending: false,
            fault: None,
            merged: false,
            counters: Counters::default(),
            graph_k: None,
            last_status: None,
            want_logits: true,
            device_rows: Vec::new(),
            row_reads: 0,
            shortlist_draws: 0,
            logit_copies: 0,
            draws: None,
        }
    }

    fn facts(&self) -> StateFacts {
        StateFacts {
            owner: true,
            valid: !self.poisoned,
            idle: true,
            resident_kv: true,
        }
    }

    fn live(&self) -> Result<()> {
        if self.poisoned {
            return Err(refused("the state is poisoned"));
        }
        Ok(())
    }

    fn write_cell(&mut self, pos: usize, token: u32) {
        if pos < self.cells.len() {
            self.cells[pos] = token;
        } else {
            self.cells.push(token);
        }
    }

    /// One one-row step: the eager row.
    fn advance(&mut self, token: u32) -> Vec<f32> {
        let pos = self.len;
        let row = model_row(
            &self.cells,
            pos,
            &mut self.recurrent,
            &mut self.conv,
            token,
            self.ties,
        );
        self.write_cell(pos, token);
        self.len += 1;
        row
    }

    /// A fault that fails as a device failure: the state is poisoned.
    fn fires(&mut self, at: Fault) -> Result<()> {
        if self.fault != Some(at) {
            return Ok(());
        }
        self.fault = None;
        self.poisoned = true;
        Err(refused("injected fault"))
    }

    /// A fault that refuses before any device work: the state stays valid.
    fn refusal_at(&mut self, at: Fault) -> Result<()> {
        if self.fault != Some(at) {
            return Ok(());
        }
        self.fault = None;
        Err(refused("injected refusal"))
    }

    /// A fault that fails without touching the state: what the host sees when a read-back or a
    /// launch error comes back. The window's own rule must gate the state, not the fake.
    fn plain_fault(&mut self, at: Fault) -> Result<()> {
        if self.fault != Some(at) {
            return Ok(());
        }
        self.fault = None;
        Err(refused("injected failure"))
    }

    /// The K-row window. Before the launch every failure is a state-decided refusal; from the
    /// launch on, every failure is `AfterLaunch`. The live state is committed only after the
    /// results were read and judged (as the step commits after its read-back), so a lost
    /// read-back leaves `tokens()` at the base and the state valid: only the unconditional
    /// fail-closed of `AfterLaunch` gates it.
    fn krow_rows(&mut self, inputs: &[u32]) -> std::result::Result<RowsOut<Vec<f32>>, RowsFailure> {
        self.krow_checks(inputs)
            .map_err(RowsFailure::StateDecides)?;
        let rows = inputs.len();
        self.draws = None;
        self.counters.krow_windows += 1;
        self.counters.krow_replays[rows] += 1;
        let executed = if self.flaw == Flaw::StaleGraph {
            (*self.graph_k.get_or_insert(rows)).min(rows)
        } else {
            rows
        };
        let ran = self.krow_launched(&inputs[..executed], rows);
        if self.flaw == Flaw::AfterLaunchAsStateDecides {
            return ran.map_err(RowsFailure::StateDecides);
        }
        ran.map_err(RowsFailure::AfterLaunch)
    }

    /// Everything that precedes the enqueue: the host rules of `krow.rs`, then the refusal fault.
    fn krow_checks(&mut self, inputs: &[u32]) -> Result<()> {
        let token_check = |token: u32| {
            if self.flaw == Flaw::NoTokenCheck {
                Ok(())
            } else {
                check_token(token)
            }
        };
        let merged = self.merged && self.flaw != Flaw::NoSwitchRecheck;
        let facts = RowsFacts {
            base: self.len,
            capacity: CONTEXT,
            merged_block: merged,
        };
        krow::check_rows(inputs, token_check, &facts)?;
        self.refusal_at(Fault::KRowRefusal)
    }

    /// The launch and what follows it: the body, the read-back, the status judgement, the slot
    /// mark, and last the commit of the live state.
    fn krow_launched(&mut self, inputs: &[u32], window: usize) -> Result<RowsOut<Vec<f32>>> {
        let base = self.len;
        let Body {
            mut logits,
            mut word,
            after,
        } = self.krow_body(inputs, window);
        self.plain_fault(Fault::KRowLaunch)?;
        // The logits stage is over; the argmax stage has not begun.
        if let Some(Fault::KRowRowPair(j)) = self.fault
            && let Some(row) = logits.get_mut(j)
        {
            self.fault = None;
            row[0] = f32::NAN;
        }
        let (tokens, bits) = self.argmax_rows(&logits);
        word |= bits.iter().fold(0, |all, own| all | own);
        if self.fault == Some(Fault::KRowStatus) {
            self.fault = None;
            word |= 16;
        }
        self.last_status = Some(RowStatus {
            bits: bits.clone(),
            window: word,
        });
        self.plain_fault(Fault::KRowReadback)?;
        let seen = self.host_view(bits, word);
        if self.flaw != Flaw::IgnoresStatus {
            krow::check_status(&seen)?;
        }
        // The device comparison of the pair tokens with the window ids, and the K-row shortlist.
        let pairs: Vec<i32> = tokens.iter().map(|&token| token as i32).collect();
        let (mut accepted, bonus) = fake_verify(&pairs, inputs);
        if self.fault == Some(Fault::KRowVerify) {
            self.fault = None;
            accepted = (accepted + 1) % inputs.len() as i32;
        }
        let (run, bonus) = if self.flaw == Flaw::NoVerifyCheck {
            (accepted as usize, bonus as u32)
        } else {
            krow::check_verify(&pairs, inputs, accepted, bonus)?
        };
        let mut records: Vec<[u8; krow::SHORTLIST_RECORD_BYTES]> =
            logits.iter().map(|row| fake_record(row)).collect();
        if self.flaw == Flaw::ShortlistFromWrongRow {
            records.rotate_left(1);
        }
        if self.flaw == Flaw::CorruptRecordEntry {
            let id: [u8; 4] = records[0][3 * 8 + 4..3 * 8 + 8].try_into().expect("an id");
            records[0][7 * 8 + 4..7 * 8 + 8].copy_from_slice(&id);
        }
        let bits_i32: Vec<i32> = self
            .last_status
            .as_ref()
            .map(|status| status.bits.iter().map(|&bits| bits as i32).collect())
            .unwrap_or_default();
        let shortlists = krow::check_shortlists(&pairs, &bits_i32, &records, VOCAB)?;
        self.book
            .mark_captured_range(inputs.len().saturating_sub(1))?;
        (self.recurrent, self.conv, self.len) = after;
        self.device_rows = logits.clone();
        let copied = self.want_logits || self.flaw == Flaw::ReadsLogitsEveryWindow;
        self.logit_copies += usize::from(copied);
        self.draws = Some(DeviceDraws {
            base,
            inputs: inputs.to_vec(),
            accepted: run,
            bonus,
            shortlists,
        });
        Ok(RowsOut {
            argmax: tokens,
            rows: if copied { logits } else { Vec::new() },
        })
    }

    /// The K-row body: the state stays in locals across the rows and each row's snapshot goes to
    /// its slot. Returns the logits rows, the window word (bit 4 for an id outside the vocabulary,
    /// as the embedding kernel sets it) and the live state after the last row, NOT yet committed.
    fn krow_body(&mut self, inputs: &[u32], window: usize) -> Body {
        let (mut recurrent, mut conv) = (self.recurrent, self.conv);
        let base = self.len;
        let mut logits = Vec::with_capacity(inputs.len());
        let mut word = 0;
        for (t, &token) in inputs.iter().enumerate() {
            if token as usize >= VOCAB {
                word |= 4;
            }
            let row = model_row(
                &self.cells,
                base + t,
                &mut recurrent,
                &mut conv,
                token,
                self.ties,
            );
            self.write_cell(base + t, token);
            logits.push(row);
            if t + 1 < window {
                let at = if self.flaw == Flaw::SnapShifted {
                    t + 1
                } else {
                    t
                };
                if let Some(slot) = self.slots.get_mut(at) {
                    *slot = (recurrent, conv);
                }
            }
        }
        if self.flaw == Flaw::SnapWritesPastTheWindow
            && let Some(slot) = self.slots.get_mut(window.saturating_sub(1))
        {
            *slot = ([-1.0; 3], [-1.0; 2]);
        }
        Body {
            logits,
            word,
            after: (recurrent, conv, base + inputs.len()),
        }
    }

    /// `argmax_rows`: row `i`'s token and its own status bits (2 for a non-finite logit). The
    /// racy flaw stores what an `atomicOr` return would: the bits of the rows before it as well.
    fn argmax_rows(&self, logits: &[Vec<f32>]) -> (Vec<u32>, Vec<u32>) {
        let mut before = 0;
        let mut tokens = Vec::with_capacity(logits.len());
        let mut bits = Vec::with_capacity(logits.len());
        for row in logits {
            let own = if row.iter().all(|v| v.is_finite()) {
                0
            } else {
                2
            };
            tokens.push(device_argmax(row));
            if self.flaw == Flaw::RacyStatus {
                bits.push(before | own);
                before |= own;
            } else {
                bits.push(own);
            }
        }
        (tokens, bits)
    }

    /// What the host looks at: all of it, except for the flawed hosts.
    fn host_view(&self, mut bits: Vec<u32>, window: u32) -> RowStatus {
        match self.flaw {
            Flaw::LastPairOnly => {
                let last = bits.pop().unwrap_or(0);
                bits.fill(0);
                bits.push(last);
                RowStatus { bits, window: 0 }
            }
            Flaw::PairsOnly => RowStatus { bits, window: 0 },
            _ => RowStatus { bits, window },
        }
    }
}

impl TokenStepper for DevFake {
    const OPERATION: &'static str = "dev fake";

    fn feed(&mut self, token: u32) -> Result<()> {
        step_gate(self)?;
        self.live()?;
        check_token(token)?;
        self.advance(token);
        Ok(())
    }

    fn logits(&mut self, token: u32) -> Result<Vec<f32>> {
        step_gate(self)?;
        self.live()?;
        check_token(token)?;
        Ok(self.advance(token))
    }

    fn fed(&self) -> usize {
        self.len
    }

    fn reset(&mut self) -> Result<()> {
        reset_prologue(self)?;
        self.len = 0;
        self.recurrent = [0.0; 3];
        self.conv = [0.0; 2];
        self.poisoned = false;
        reset_epilogue(self);
        Ok(())
    }
}

impl WindowHost for DevFake {
    type Saved = SlotState;
    type Row = Vec<f32>;

    fn device_wanted(&self) -> bool {
        true
    }
    fn tokens(&self) -> usize {
        self.len
    }
    fn store_window(&mut self, window: super::Window<SlotState, Vec<f32>>) {
        self.window = Some(window);
    }
    fn take_window(&mut self) -> Option<super::Window<SlotState, Vec<f32>>> {
        self.window.take()
    }
    fn export(&mut self) -> Result<SlotState> {
        Err(refused("the device path never exports"))
    }
    fn run_row(&mut self, token: u32) -> Result<(u32, Vec<f32>)> {
        step_gate(self)?;
        self.live()?;
        check_token(token)?;
        let row = self.advance(token);
        Ok((device_argmax(&row), row))
    }
    fn replay_row(&mut self, _token: u32) -> Result<()> {
        Err(refused("the device path never replays"))
    }
    fn overwrite(&mut self, saved: &SlotState) -> Result<()> {
        // As the device import: the committed cells are written, the stale ones stay.
        for (at, &cell) in saved.cells.iter().enumerate() {
            self.write_cell(at, cell);
        }
        self.len = saved.cells.len();
        self.recurrent = saved.recurrent;
        self.conv = saved.conv;
        self.poisoned = false;
        Ok(())
    }
    fn require_armed(&mut self) -> Result<()> {
        Ok(())
    }
    fn complete_prefix_save(&mut self) -> Result<()> {
        state_usable(self)?;
        self.pending = false;
        Ok(())
    }
    fn window_awaits(&self) -> bool {
        self.window.is_some() || self.book.awaiting_verdict()
    }
    fn capacity(&self) -> usize {
        CONTEXT
    }
    fn is_valid(&self) -> bool {
        !self.poisoned
    }
    fn needs_reset(&self) -> bool {
        self.gate
    }
    fn set_needs_reset(&mut self) {
        self.gate = true;
    }
    fn clear_needs_reset(&mut self) {
        self.gate = false;
    }
    fn poison(&mut self) {
        self.poisoned = true;
    }
    fn drop_pending_prefix(&mut self) {
        self.pending = false;
    }
    fn invalidate_pool(&mut self) {}
    fn close_window(&mut self) {
        self.window = None;
        self.book.discard();
    }
    fn check_import(&mut self, _saved: &SlotState) -> Result<()> {
        Ok(())
    }
    fn resident_kv(&mut self) -> Result<()> {
        Ok(())
    }
    fn rollback_begin(&mut self, rows: usize) -> Result<()> {
        self.book.begin_window(self.len, rows)
    }
    fn capture(&mut self, row: usize) -> Result<()> {
        let slot = self.book.begin_capture(row)?;
        *self
            .slots
            .get_mut(slot)
            .ok_or_else(|| refused("slot out of range"))? = (self.recurrent, self.conv);
        self.book.note_captured(row)
    }
    fn rollback_discard(&mut self) {
        self.book.discard();
    }
    fn rollback_window_complete(&mut self) -> Result<()> {
        self.book.mark_complete()
    }
    fn rollback_finish(&mut self, kept: usize, committed_rows: usize) -> Result<bool> {
        self.facts().require_copyable("commit_window")?;
        let Some(restore) = self.book.finish_window(kept, committed_rows, self.len)? else {
            return Ok(false);
        };
        let slot = match self.flaw {
            Flaw::SlotOffByOne => restore.slot + 1,
            _ => restore.slot,
        };
        let (recurrent, conv) = *self
            .slots
            .get(slot)
            .ok_or_else(|| refused("slot out of range"))?;
        self.recurrent = recurrent;
        self.conv = conv;
        self.len = match self.flaw {
            Flaw::FrontierPlusOne => restore.frontier + 1,
            _ => restore.frontier,
        };
        Ok(true)
    }

    fn run_rows(&mut self, inputs: &[u32]) -> std::result::Result<RowsOut<Vec<f32>>, RowsFailure> {
        let krow = self.path == Path::Krow
            && self.flaw != Flaw::EagerWhileClaimingKrow
            && krow::uses_krow(inputs.len());
        if krow {
            return self.krow_rows(inputs);
        }
        self.counters.eager_windows += 1;
        eager_rows(self, inputs).map_err(RowsFailure::StateDecides)
    }

    fn feed_fault(&mut self, row: usize) -> Result<()> {
        self.fires(Fault::Feed(row))
    }
    fn capture_fault(&mut self, row: usize) -> Result<()> {
        self.fires(Fault::Capture(row))
    }
    fn restore_fault(&mut self) -> Result<()> {
        self.fires(Fault::Restore)
    }
}

impl WindowModel for DevFake {
    fn run_window(&mut self, inputs: &[u32]) -> Result<()> {
        run_window(self, inputs)
    }

    fn draw(&mut self, row: usize, sampler: &mut Sampler, vocab: usize) -> Result<u32> {
        let window = self.window.as_ref().ok_or_else(|| refused("no window"))?;
        if sampler.is_greedy() {
            return window
                .argmax
                .get(row)
                .copied()
                .ok_or_else(|| refused("no such row"));
        }
        // A K-row window that kept its logits on the device draws from its report (the shortlist
        // record) or reads the one row it needs, as the step does; the routing is production's.
        let krow_runs = self.flaw == Flaw::ServesStaleDraws
            || (self.path == Path::Krow
                && self.flaw != Flaw::EagerWhileClaimingKrow
                && krow::uses_krow(window.inputs.len()));
        let route = krow::draw_route(&DrawFacts {
            krow_runs,
            device_window: true,
            draws: self.draws.as_ref(),
            base: window.base,
            inputs: &window.inputs,
            row,
            tap_row: window.rows.get(row).is_some(),
            params: sampler.params(),
            sync_each: false,
            vocab_matches: vocab == VOCAB,
        });
        match route {
            DrawRoute::NotKrow => {}
            DrawRoute::OutOfWindow => return Err(refused("no such row")),
            DrawRoute::Shortlist => {
                let record = self
                    .draws
                    .as_ref()
                    .and_then(|draws| draws.shortlists.get(row))
                    .cloned()
                    .ok_or_else(|| refused("no record"))?;
                self.shortlist_draws += 1;
                let drawn = sampler.sample_shortlist(&record, vocab);
                if drawn.is_err() && self.flaw != Flaw::ShortlistDrawErrorNotGated {
                    fail_window(self);
                }
                return drawn;
            }
            DrawRoute::TapRow => {
                let values = window.rows.get(row).ok_or_else(|| refused("no such row"))?;
                return sampler.sample(&values[..vocab]);
            }
            DrawRoute::DeviceRow => {
                self.row_reads += 1;
                if self.fault == Some(Fault::KRowDraw) {
                    self.fault = None;
                    if self.flaw != Flaw::LazyReadNotGated {
                        fail_window(self);
                    }
                    return Err(refused("injected failure of the row read"));
                }
                let values = self
                    .device_rows
                    .get(row)
                    .cloned()
                    .ok_or_else(|| refused("no such row"))?;
                return sampler.sample(&values[..vocab]);
            }
        }
        let logits = window.rows.get(row).ok_or_else(|| refused("no such row"))?;
        sampler.sample(&logits[..vocab])
    }

    fn commit_window(&mut self, commit: &Commit) -> Result<()> {
        commit_window(self, commit)
    }
}

impl D1Subject for DevFake {
    type State = SlotState;

    fn snapshot(&mut self) -> Result<SlotState> {
        export_gate(self)?;
        self.live()?;
        Ok(SlotState {
            cells: self.cells[..self.len].to_vec(),
            recurrent: self.recurrent,
            conv: self.conv,
        })
    }

    fn restore(&mut self, state: &SlotState) -> Result<()> {
        // The restore of a step with a recovery gate: a reset, then the import.
        self.reset()?;
        import_state(self, state)
    }

    fn models_recovery_gate(&self) -> bool {
        true
    }

    fn import_without_reset(&mut self, state: &SlotState) -> Option<Result<()>> {
        Some(import_state(self, state))
    }

    fn state_difference(&self, left: &SlotState, right: &SlotState) -> Option<String> {
        if left.cells != right.cells {
            return Some(format!(
                "KV rows {:?} against {:?}",
                left.cells, right.cells
            ));
        }
        first_bit_difference(&left.recurrent, &right.recurrent)
            .map(|d| format!("recurrent state, {d}"))
            .or_else(|| {
                first_bit_difference(&left.conv, &right.conv).map(|d| format!("conv state, {d}"))
            })
    }

    fn window_rows(&self) -> Result<Vec<Vec<f32>>> {
        let window = self.window.as_ref().ok_or_else(|| refused("no window"))?;
        if window.rows.len() != window.inputs.len() {
            return Err(refused("the logits rows were not copied (the tap is off)"));
        }
        Ok(window.rows.clone())
    }

    fn window_slots(&self) -> Option<Vec<Vec<u32>>> {
        // Every armed slot, not only the ones the window wrote: a store past the window shows.
        Some(self.slots.iter().map(slot_bits).collect())
    }

    fn inject_fault(&mut self, fault: Fault) {
        self.fault = Some(fault);
    }

    fn fault_plan(&self, rows: usize) -> Vec<(Fault, Commit)> {
        match self.path {
            Path::Krow => krow_faults_for(self.len, rows),
            Path::Eager => slot_faults_for(self.len, rows),
        }
    }
}

struct Rig {
    fake: DevFake,
    session: DecodeSession,
    tokenizer: Tokenizer,
}

impl Rig {
    fn new(path: Path, flaw: Flaw, ties: bool) -> Self {
        let mut fake = DevFake::new(path, flaw, ties);
        let mut session = DecodeSession::default();
        session
            .prefill(&mut fake, &PROMPT, CONTEXT)
            .expect("prefill");
        Self {
            fake,
            session,
            tokenizer: Tokenizer::from_bytes(FAKE_TOKENIZER_JSON.as_bytes()).expect("tokenizer"),
        }
    }

    fn krow(flaw: Flaw) -> Self {
        Self::new(Path::Krow, flaw, false)
    }

    fn split(&mut self) -> (&mut DevFake, Setup<'_>) {
        let setup = Setup {
            session: &self.session,
            tokenizer: &self.tokenizer,
            limits: Limits {
                eos: &[],
                sampler_vocab: VOCAB,
                context_limit: CONTEXT,
            },
        };
        (&mut self.fake, setup)
    }
}

fn seeded() -> Sampler {
    Sampler::new(SamplingParams::new(0.9, 0.95, 5).expect("params"), 7)
}

fn rows(n: usize) -> Vec<u32> {
    [3, 1, 4, 1, 5, 2, 6, 0].into_iter().take(n).collect()
}

fn caught(result: Verdict<Tally>, what: &str) -> D1Failure {
    result.expect_err(what)
}

// ---- the sound K-row fake passes every check ----

#[test]
fn a_sound_k_row_fake_passes_every_check() {
    let mut rig = Rig::krow(Flaw::Sound);
    for n in [2, 3, 4, 8] {
        full_logits(&mut rig.fake, &rows(n)).expect("A");
        later_rows_isolated(&mut rig.fake, &rows(n), &ALTERNATIVES).expect("C");
    }
    let (fake, setup) = rig.split();
    for drafts in [1, 3, 7] {
        accepted_prefix(fake, &setup, drafts, &Sampler::greedy).expect("D greedy");
        accepted_prefix(fake, &setup, drafts, &seeded).expect("D seeded");
    }
    repeated_rollbacks(fake, &setup, 3, &Sampler::greedy, 40).expect("G greedy");
    repeated_rollbacks(fake, &setup, 7, &seeded, 40).expect("G seeded");
    let mut tied = Rig::new(Path::Krow, Flaw::Sound, true);
    let tally = ties_break_low(&mut tied.fake, VOCAB, &rows(8)).expect("E");
    assert_eq!(tally.exact_ties, 8);
}

#[test]
fn the_k_row_window_equals_the_eager_window_for_every_row_count_and_verdict() {
    for n in 2..=8 {
        let mut krow = Rig::krow(Flaw::Sound);
        let mut eager = Rig::new(Path::Eager, Flaw::Sound, false);
        let at = krow.fake.len;
        let tally = krow_matches_eager(&mut krow.fake, &mut eager.fake, &rows(n), at).expect("K");
        assert_eq!(tally.comparisons, n, "one comparison per verdict");
        assert_eq!(krow.fake.counters.krow_windows, n, "one window per verdict");
        assert_eq!(eager.fake.counters.krow_windows, 0);
    }
}

#[test]
fn the_counters_show_which_path_ran_and_the_eager_run_keeps_the_k_row_counters_at_zero() {
    let mut krow = Rig::krow(Flaw::Sound);
    for n in [2, 3, 3, 8] {
        full_logits(&mut krow.fake, &rows(n)).expect("A");
    }
    let c = krow.fake.counters;
    assert_eq!(c.krow_windows, 4);
    assert_eq!(
        (c.krow_replays[2], c.krow_replays[3], c.krow_replays[8]),
        (1, 2, 1)
    );
    assert_eq!(
        c.eager_windows, 0,
        "no K-row window fell back to the eager loop"
    );
    let mut eager = Rig::new(Path::Eager, Flaw::Sound, false);
    full_logits(&mut eager.fake, &rows(4)).expect("A");
    assert_eq!(eager.fake.counters.krow_windows, 0);
    assert_eq!(eager.fake.counters.krow_replays, [0; 9]);
    assert_eq!(eager.fake.counters.eager_windows, 1);
}

#[test]
fn a_one_row_window_runs_the_eager_loop_even_when_krow_is_asked_for() {
    let mut krow = Rig::krow(Flaw::Sound);
    krow.fake.run_window(&rows(1)).expect("one row");
    assert_eq!(krow.fake.counters.krow_windows, 0);
    assert_eq!(krow.fake.counters.eager_windows, 1);
}

// ---- check F at every new fault point ----

#[test]
fn the_k_row_fake_fails_closed_at_every_new_fault_point_and_recovers() {
    for n in [2, 3, 4, 8] {
        let mut rig = Rig::krow(Flaw::Sound);
        let tally = failure_recovery(&mut rig.fake, VOCAB, &rows(n)).expect("F");
        assert_eq!(
            tally.comparisons,
            krow_faults_for(rig.fake.len, n).len(),
            "{n} rows"
        );
    }
}

#[test]
fn the_k_row_plan_lists_the_five_points_and_the_restore_and_no_capture() {
    let plan = krow_faults_for(10, 4);
    let faults: Vec<Fault> = plan.iter().map(|(fault, _)| *fault).collect();
    for want in [
        Fault::KRowRefusal,
        Fault::KRowLaunch,
        Fault::KRowReadback,
        Fault::KRowStatus,
        Fault::KRowVerify,
        Fault::KRowRowPair(0),
        Fault::KRowRowPair(2),
        Fault::KRowRowPair(3),
        Fault::Restore,
    ] {
        assert!(faults.contains(&want), "{want:?} missing from {faults:?}");
    }
    assert!(
        !faults
            .iter()
            .any(|f| matches!(f, Fault::Capture(_) | Fault::Feed(_))),
        "the K-row path has no per-row capture or feed"
    );
    assert!(
        plan.iter().all(|(_, commit)| commit.committed_rows > 10),
        "every verdict names rows above the base"
    );
}

#[test]
fn a_refusal_before_the_launch_leaves_the_state_valid_and_the_other_new_points_poison_and_gate() {
    for (fault, poisons) in [
        (Fault::KRowRefusal, false),
        (Fault::KRowLaunch, true),
        (Fault::KRowReadback, true),
        (Fault::KRowStatus, true),
        (Fault::KRowVerify, true),
        (Fault::KRowRowPair(1), true),
    ] {
        let mut rig = Rig::krow(Flaw::Sound);
        let base = rig.fake.len;
        rig.fake.inject_fault(fault);
        assert!(rig.fake.run_window(&rows(4)).is_err(), "{fault:?}");
        assert_eq!(rig.fake.poisoned, poisons, "{fault:?}");
        assert_eq!(rig.fake.gate, poisons, "{fault:?}");
        assert!(
            rig.fake.window.is_none() && !rig.fake.book.is_open(),
            "{fault:?}"
        );
        if !poisons {
            assert_eq!(rig.fake.len, base, "{fault:?}: a refusal moved the state");
            rig.fake
                .run_window(&rows(4))
                .expect("the window may run again");
        }
    }
}

#[test]
fn a_failed_k_row_window_is_recovered_only_by_a_reset() {
    let mut rig = Rig::krow(Flaw::Sound);
    let base = rig.fake.snapshot().expect("snapshot");
    rig.fake.inject_fault(Fault::KRowStatus);
    rig.fake.run_window(&rows(3)).expect_err("status");
    assert!(rig.fake.snapshot().is_err() && rig.fake.logits(1).is_err());
    assert!(
        rig.fake
            .import_without_reset(&base)
            .expect("modelled")
            .is_err()
    );
    rig.fake.restore(&base).expect("reset and import");
    rig.fake
        .run_window(&rows(3))
        .expect("a window after the recovery");
}

fn f_catches(flaw: Flaw, n: usize) -> D1Failure {
    let mut rig = Rig::krow(flaw);
    caught(failure_recovery(&mut rig.fake, VOCAB, &rows(n)), "F")
}

#[test]
fn a_k_row_window_that_continues_after_a_non_zero_status_is_caught_by_check_f() {
    let failure = f_catches(Flaw::IgnoresStatus, 4);
    assert!(failure.violation, "{failure}");
    assert!(
        failure.detail.contains("did not fail the call"),
        "{failure}"
    );
}

/// Whether the window fails closed with `fault` armed, on a fake with `flaw`.
fn fails_closed(flaw: Flaw, fault: Fault, n: usize) -> bool {
    let mut rig = Rig::krow(flaw);
    rig.fake.inject_fault(fault);
    let failed = rig.fake.run_window(&rows(n)).is_err();
    failed && rig.fake.poisoned && rig.fake.gate
}

#[test]
fn a_host_that_looks_at_the_last_pair_only_misses_a_bad_row_before_the_last() {
    // The sound host fails the window for every row; the flawed one only for the last.
    for n in 2..=8 {
        for j in 0..n {
            assert!(
                fails_closed(Flaw::Sound, Fault::KRowRowPair(j), n),
                "n {n} j {j}"
            );
            assert_eq!(
                fails_closed(Flaw::LastPairOnly, Fault::KRowRowPair(j), n),
                j == n - 1,
                "n {n} j {j}"
            );
        }
    }
    let failure = f_catches(Flaw::LastPairOnly, 4);
    assert!(
        failure.violation && failure.detail.contains("did not fail the call"),
        "{failure}"
    );
}

#[test]
fn a_host_that_ignores_the_window_word_misses_a_status_that_lives_in_the_word_only() {
    // Mask 16 (and 1, 4) are in the window word and in no pair. A host that checks the word
    // only would NOT be a mutant: the word is the OR of every pair, so it still fails closed.
    assert!(fails_closed(Flaw::Sound, Fault::KRowStatus, 4));
    assert!(!fails_closed(Flaw::PairsOnly, Fault::KRowStatus, 4));
    let failure = f_catches(Flaw::PairsOnly, 4);
    assert!(
        failure.violation && failure.detail.contains("KRowStatus"),
        "{failure}"
    );
}

// ---- the row-local status rule, deterministically ----

fn pairs_after_row_fault(flaw: Flaw, n: usize, j: usize) -> RowStatus {
    let mut rig = Rig::krow(flaw);
    rig.fake.inject_fault(Fault::KRowRowPair(j));
    rig.fake
        .run_window(&rows(n))
        .expect_err("the window fails closed");
    assert!(rig.fake.poisoned && rig.fake.gate, "n {n} j {j}");
    rig.fake.last_status.clone().expect("the pairs were read")
}

/// Whether every pair other than the bad row's is clean, for every window size and bad row.
fn pairs_are_row_local(flaw: Flaw) -> bool {
    (2..=8).all(|n| {
        (0..n).all(|j| {
            let status = pairs_after_row_fault(flaw, n, j);
            status
                .bits
                .iter()
                .enumerate()
                .all(|(i, bits)| *bits == if i == j { 2 } else { 0 })
                && status.window & 2 == 2
        })
    })
}

#[test]
fn only_the_pair_of_the_bad_row_reports_it_and_the_window_word_holds_the_fold() {
    assert!(pairs_are_row_local(Flaw::Sound));
}

#[test]
fn a_pair_that_stores_the_return_of_the_shared_or_shows_one_row_in_another_pair() {
    // Deterministic: the flawed fake folds in index order, so the pair after the bad row
    // always carries its bit. No dependence on scheduling, no 100-window lottery.
    assert!(!pairs_are_row_local(Flaw::RacyStatus));
    for n in 2..=8 {
        for j in 0..n - 1 {
            let status = pairs_after_row_fault(Flaw::RacyStatus, n, j);
            assert_ne!(status.bits[j + 1], 0, "n {n} j {j}");
        }
    }
}

#[test]
fn the_fault_enters_after_the_logits_stage_so_no_other_row_can_see_it() {
    // Injected upstream, a bad value would reach later rows through the K/V cells and the state;
    // here the bad row is the only one, in the pair table and in the logits that were kept.
    let mut rig = Rig::krow(Flaw::Sound);
    rig.fake.inject_fault(Fault::KRowRowPair(1));
    rig.fake.run_window(&rows(5)).expect_err("row 1");
    let status = rig.fake.last_status.clone().expect("pairs");
    assert_eq!(status.bits, [0, 2, 0, 0, 0]);
    assert_eq!(
        status.window, 2,
        "the window word is the fold and nothing else"
    );
}

// ---- the host pre-launch checks ----

#[test]
fn an_out_of_vocabulary_id_is_a_clean_refusal_with_the_state_valid_and_nothing_enqueued() {
    for at in 0..4 {
        let mut inputs = rows(4);
        inputs[at] = VOCAB as u32 + 3;
        let mut rig = Rig::krow(Flaw::Sound);
        let before = rig.fake.snapshot().expect("snapshot");
        let launched = rig.fake.counters.krow_windows;
        let error = rig.fake.run_window(&inputs).expect_err("refused");
        assert!(
            format!("{error:?}").contains("outside the vocabulary"),
            "{error:?}"
        );
        assert!(!rig.fake.poisoned && !rig.fake.gate, "id at row {at}");
        assert_eq!(
            rig.fake.counters.krow_windows, launched,
            "nothing was enqueued"
        );
        let after = rig.fake.snapshot().expect("still valid");
        assert!(rig.fake.state_difference(&before, &after).is_none());
        assert!(!rig.fake.book.is_open() && rig.fake.window.is_none());
        rig.fake.run_window(&rows(4)).expect("the next window runs");
    }
}

#[test]
fn without_the_host_token_check_the_bad_id_reaches_the_device_and_poisons_the_state() {
    // The embedding sets status 4 on the device, the window fails closed.
    let mut inputs = rows(4);
    inputs[2] = VOCAB as u32;
    let mut rig = Rig::krow(Flaw::NoTokenCheck);
    rig.fake.run_window(&inputs).expect_err("status 4");
    assert!(rig.fake.poisoned && rig.fake.gate);
    assert_eq!(rig.fake.last_status.as_ref().map(|s| s.window & 4), Some(4));
}

#[test]
fn a_window_that_does_not_fit_the_kv_is_refused_before_anything_is_enqueued() {
    let mut rig = Rig::krow(Flaw::Sound);
    rig.fake.len = CONTEXT - 2;
    rig.fake.cells.resize(CONTEXT, 0);
    let error = rig.fake.run_window(&rows(3)).expect_err("capacity");
    assert!(matches!(error, EngError::ContextLimit(_)), "{error:?}");
    assert_eq!(rig.fake.counters.krow_windows, 0);
    assert!(!rig.fake.poisoned && !rig.fake.gate);
}

#[test]
fn the_merged_moe_block_is_refused_before_every_window_even_when_it_appears_later() {
    let mut rig = Rig::krow(Flaw::Sound);
    rig.fake.run_window(&rows(3)).expect("first window");
    let commit = Commit {
        committed_rows: rig.fake.len - 2,
        kept_rows: 1,
        discarded_rows: 2,
        state_slot: 0,
    };
    rig.fake.commit_window(&commit).expect("verdict");
    // The test hook builds the merged block between two windows.
    rig.fake.merged = true;
    let before = rig.fake.len;
    let error = rig.fake.run_window(&rows(3)).expect_err("merged block");
    assert!(format!("{error:?}").contains("unset it"), "{error:?}");
    assert_eq!(rig.fake.len, before);
    assert!(!rig.fake.poisoned);
    assert_eq!(
        rig.fake.counters.krow_windows, 1,
        "no second window was enqueued"
    );

    let mut flawed = Rig::krow(Flaw::NoSwitchRecheck);
    flawed.fake.merged = true;
    flawed.fake.run_window(&rows(3)).expect("the flaw runs it");
    assert_eq!(flawed.fake.counters.krow_windows, 1);
}

// ---- every data-path flaw is caught by the check meant to catch it ----

fn d_catches(flaw: Flaw) -> D1Failure {
    let mut rig = Rig::krow(flaw);
    let (fake, setup) = rig.split();
    caught(accepted_prefix(fake, &setup, 3, &Sampler::greedy), "D")
}

#[test]
fn a_snapshot_stored_in_the_wrong_slot_is_caught_by_check_d_and_by_check_k() {
    let failure = d_catches(Flaw::SnapShifted);
    assert!(failure.test.starts_with('D'), "{failure}");
    let mut krow = Rig::krow(Flaw::SnapShifted);
    let mut eager = Rig::new(Path::Eager, Flaw::Sound, false);
    let at = krow.fake.len;
    let failure = caught(
        krow_matches_eager(&mut krow.fake, &mut eager.fake, &rows(4), at),
        "K",
    );
    assert!(failure.test.starts_with('K'), "{failure}");
    assert!(failure.violation, "{failure}");
}

#[test]
fn a_wrong_restore_slot_or_frontier_is_caught_by_check_d() {
    for flaw in [Flaw::SlotOffByOne, Flaw::FrontierPlusOne] {
        let failure = d_catches(flaw);
        assert!(failure.test.starts_with('D'), "{flaw:?}: {failure}");
    }
}

#[test]
fn a_graph_replayed_for_another_row_count_is_caught_by_check_a_at_every_row_count() {
    // The graph of the first window (2 rows) is replayed for every later window.
    let stale = |n: usize| {
        let mut rig = Rig::krow(Flaw::StaleGraph);
        rig.fake.graph_k = Some(2);
        full_logits(&mut rig.fake, &rows(n))
    };
    stale(2).expect("the first K is its own graph");
    for n in [3, 4, 8] {
        stale(n).expect_err("a graph of another K");
    }
}

#[test]
fn a_fake_that_claims_krow_and_runs_the_eager_loop_is_caught_by_the_counters() {
    // The bytes are equal, so only the counters tell.
    let mut claimed = Rig::krow(Flaw::EagerWhileClaimingKrow);
    let mut eager = Rig::new(Path::Eager, Flaw::Sound, false);
    let at = claimed.fake.len;
    krow_matches_eager(&mut claimed.fake, &mut eager.fake, &rows(4), at)
        .expect("same bytes: check K cannot tell");
    assert_eq!(claimed.fake.counters.krow_windows, 0, "no K-row window ran");
    assert!(claimed.fake.counters.eager_windows > 0);
}

#[test]
fn the_eager_default_of_run_rows_is_the_pinned_eager_device_window() {
    // The eager fake overrides `run_rows` only to count the window, then delegates to the function
    // the trait default calls (`eager_rows`): its fault points are the old ones and a capture
    // fault at row i still poisons.
    let mut rig = Rig::new(Path::Eager, Flaw::Sound, false);
    let tally = failure_recovery(&mut rig.fake, VOCAB, &rows(4)).expect("F, eager");
    assert_eq!(tally.comparisons, slot_faults_for(rig.fake.len, 4).len());
}

#[test]
fn a_lost_read_back_leaves_the_counters_at_the_base_and_the_window_still_gates() {
    // The step commits its rows only after the read-back, so a failed read-back leaves tokens() at
    // the base and the state valid. The old rule alone would let that window go; the unconditional
    // fail-closed of a failure after the launch gates it.
    for fault in [
        Fault::KRowLaunch,
        Fault::KRowReadback,
        Fault::KRowStatus,
        Fault::KRowVerify,
    ] {
        let mut rig = Rig::krow(Flaw::Sound);
        let base = rig.fake.len;
        rig.fake.inject_fault(fault);
        rig.fake.run_window(&rows(4)).expect_err("fails");
        assert_eq!(rig.fake.len, base, "{fault:?}: nothing was committed");
        assert!(rig.fake.poisoned && rig.fake.gate, "{fault:?}: gated");
        assert!(
            rig.fake.window.is_none() && !rig.fake.book.is_open(),
            "{fault:?}"
        );
    }
}

#[test]
fn a_failure_after_the_launch_that_is_judged_by_the_counters_leaves_a_usable_state() {
    // The mutant: the read-back is lost after the launch, the host counters stay at the base and
    // the state stays valid. Reported as a state-decided failure, the window would not fail
    // closed; reported as after-launch (the sound fake) it does.
    for fault in [Fault::KRowLaunch, Fault::KRowReadback, Fault::KRowStatus] {
        assert!(fails_closed(Flaw::Sound, fault, 4), "{fault:?}");
        assert!(
            !fails_closed(Flaw::AfterLaunchAsStateDecides, fault, 4),
            "{fault:?}: the mutant must leave the gate open"
        );
    }
}

#[test]
fn a_failure_after_the_launch_judged_by_the_counters_is_caught_by_check_f() {
    let failure = f_catches(Flaw::AfterLaunchAsStateDecides, 4);
    assert!(failure.violation, "{failure}");
    assert!(failure.detail.contains("was accepted"), "{failure}");
}

#[test]
fn a_snapshot_stored_past_the_window_is_caught_by_the_slot_comparison_of_check_k_alone() {
    // The slot after the window's last snapshot is restored by no verdict of the window, so the
    // committed state, the rows and the next step all agree with the eager window. Only the
    // comparison of the slots sees it.
    let mut rig = Rig::krow(Flaw::SnapWritesPastTheWindow);
    let (fake, setup) = rig.split();
    assert!(
        accepted_prefix(fake, &setup, 3, &Sampler::greedy).is_ok(),
        "check D cannot see a store no verdict restores"
    );
    let mut krow = Rig::krow(Flaw::SnapWritesPastTheWindow);
    let mut eager = Rig::new(Path::Eager, Flaw::Sound, false);
    let at = krow.fake.len;
    let failure = caught(
        krow_matches_eager(&mut krow.fake, &mut eager.fake, &rows(4), at),
        "K",
    );
    assert!(failure.violation, "{failure}");
    assert!(failure.detail.contains("state slots differ"), "{failure}");
}

// ---- the logits stay on the device ----

/// A window of `n` rows over the prefilled fake, the draws of its first `drawn` rows with
/// `sampler`, the verdict that keeps them. Returns the tokens drawn.
fn serve(fake: &mut DevFake, n: usize, drawn: usize, sampler: &mut Sampler) -> Vec<u32> {
    let base = fake.len;
    fake.run_window(&rows(n)).expect("the window");
    let tokens: Vec<u32> = (0..drawn)
        .map(|row| fake.draw(row, sampler, VOCAB).expect("a draw"))
        .collect();
    fake.commit_window(&Commit {
        committed_rows: base + drawn,
        kept_rows: drawn,
        discarded_rows: n - drawn,
        state_slot: drawn - 1,
    })
    .expect("the verdict");
    tokens
}

/// The sampler of `top_k` candidates, seeded.
fn keeping(top_k: u32, seed: u64) -> Sampler {
    Sampler::new(SamplingParams::new(0.9, 0.95, top_k).expect("params"), seed)
}

/// What the eager window draws for the same rows and sampler.
fn eager_tokens(n: usize, drawn: usize, sampler: &mut Sampler) -> Vec<u32> {
    let mut eager = Rig::new(Path::Eager, Flaw::Sound, false);
    serve(&mut eager.fake, n, drawn, sampler)
}

#[test]
fn a_serving_window_copies_no_logits_and_its_draws_are_the_eager_windows() {
    for n in 2..=8 {
        for drawn in [1, n / 2 + 1, n] {
            for (name, make) in [
                ("greedy", Sampler::greedy as fn() -> Sampler),
                ("keeps 5", (|| keeping(5, 7)) as fn() -> Sampler),
                ("keeps 1", (|| keeping(1, 3)) as fn() -> Sampler),
            ] {
                let mut rig = Rig::krow(Flaw::Sound);
                rig.fake.want_logits = false;
                let (mut from_report, mut from_rows) = (make(), make());
                let tokens = serve(&mut rig.fake, n, drawn, &mut from_report);
                assert_eq!(
                    tokens,
                    eager_tokens(n, drawn, &mut from_rows),
                    "{name} n {n} drawn {drawn}"
                );
                assert_eq!(
                    from_report.rng_state(),
                    from_rows.rng_state(),
                    "{name}: the generator"
                );
                assert_eq!(
                    rig.fake.logit_copies, 0,
                    "{name} n {n}: no copy of the logits"
                );
                assert_eq!(
                    rig.fake.row_reads, 0,
                    "{name} n {n}: no row was read either"
                );
            }
        }
    }
}

#[test]
fn a_sampler_the_record_cannot_serve_reads_one_row_per_drawn_row_and_draws_what_the_eager_window_draws()
 {
    for top_k in [0, 21, 64] {
        for (n, drawn) in [(4, 1), (4, 3), (8, 8)] {
            let mut rig = Rig::krow(Flaw::Sound);
            rig.fake.want_logits = false;
            let (mut from_device, mut from_rows) = (keeping(top_k, 11), keeping(top_k, 11));
            let tokens = serve(&mut rig.fake, n, drawn, &mut from_device);
            assert_eq!(
                tokens,
                eager_tokens(n, drawn, &mut from_rows),
                "top_k {top_k}"
            );
            assert_eq!(from_device.rng_state(), from_rows.rng_state());
            assert_eq!(
                rig.fake.row_reads, drawn,
                "top_k {top_k}: one read per drawn row"
            );
            assert_eq!(rig.fake.logit_copies, 0);
        }
    }
}

#[test]
fn with_the_tap_on_the_window_copies_its_rows_once_and_draws_the_same() {
    let mut rig = Rig::krow(Flaw::Sound);
    assert!(
        rig.fake.want_logits,
        "the fake runs with the tap on by default"
    );
    let tokens = serve(&mut rig.fake, 6, 6, &mut keeping(5, 7));
    assert_eq!(rig.fake.logit_copies, 1);
    assert_eq!(rig.fake.row_reads, 0, "the record still serves the draw");
    let mut off = Rig::krow(Flaw::Sound);
    off.fake.want_logits = false;
    assert_eq!(tokens, serve(&mut off.fake, 6, 6, &mut keeping(5, 7)));
    // The rows of the tap are the checks' input: a window without it gives none.
    off.fake.run_window(&rows(3)).expect("window");
    assert!(off.fake.window_rows().is_err());
    off.fake.close_window();
    rig.fake.run_window(&rows(3)).expect("window");
    assert_eq!(rig.fake.window_rows().expect("rows").len(), 3);
}

#[test]
fn a_window_that_copies_the_logits_without_the_tap_is_caught_by_the_copy_counter() {
    let copies = |flaw: Flaw| {
        let mut rig = Rig::krow(flaw);
        rig.fake.want_logits = false;
        for n in [2, 5, 8] {
            serve(&mut rig.fake, n, 1, &mut Sampler::greedy());
        }
        rig.fake.logit_copies
    };
    assert_eq!(copies(Flaw::Sound), 0);
    assert_eq!(copies(Flaw::ReadsLogitsEveryWindow), 3);
}

#[test]
fn the_state_left_by_the_report_draws_equals_the_state_left_by_the_eager_window() {
    // Check D and G with the tap off: the committed state, the next logits, the generator.
    let mut rig = Rig::krow(Flaw::Sound);
    rig.fake.want_logits = false;
    let (fake, setup) = rig.split();
    for drafts in [1, 3, 7] {
        accepted_prefix(fake, &setup, drafts, &Sampler::greedy).expect("D greedy");
        accepted_prefix(fake, &setup, drafts, &seeded).expect("D seeded");
    }
    repeated_rollbacks(fake, &setup, 3, &Sampler::greedy, 40).expect("G greedy");
    repeated_rollbacks(fake, &setup, 7, &seeded, 40).expect("G seeded");
    assert_eq!(fake.logit_copies, 0);
    assert_eq!(fake.row_reads, 0);
}

// ---- the device verify and the records are judged ----

#[test]
fn the_device_verify_is_the_accepted_run_of_the_scripted_window() {
    for drafts in [1usize, 3, 7] {
        for accepted in 0..=drafts {
            let mut rig = Rig::krow(Flaw::Sound);
            let mut model = Rig::new(Path::Eager, Flaw::Sound, false);
            // The model's own tokens after the rows, then a window whose first `accepted` drafts
            // are right and the next is not.
            let pending = 3;
            let mut reference = Vec::new();
            let mut token = pending;
            for _ in 0..=drafts {
                let row = model.fake.advance(token);
                token = device_argmax(&row);
                reference.push(token);
            }
            let mut inputs = vec![pending];
            inputs.extend_from_slice(&reference[..accepted]);
            while inputs.len() < drafts + 1 {
                inputs.push((reference[inputs.len() - 1] + 1) % VOCAB as u32);
            }
            rig.fake.run_window(&inputs).expect("the window");
            let draws = rig.fake.draws.clone().expect("draws");
            assert_eq!(
                (draws.accepted, draws.bonus),
                (accepted, reference[accepted])
            );
        }
    }
}

#[test]
fn a_device_verify_that_disagrees_with_the_pairs_fails_the_window_closed() {
    for n in 2..=8 {
        assert!(fails_closed(Flaw::Sound, Fault::KRowVerify, n), "n {n}");
        assert!(
            !fails_closed(Flaw::NoVerifyCheck, Fault::KRowVerify, n),
            "n {n}"
        );
    }
    let failure = f_catches(Flaw::NoVerifyCheck, 4);
    assert!(
        failure.violation && failure.detail.contains("KRowVerify"),
        "{failure}"
    );
}

#[test]
fn a_shortlist_of_another_row_is_caught_by_the_pairs_or_by_the_seeded_draws() {
    let mut sound = Rig::krow(Flaw::Sound);
    let (fake, setup) = sound.split();
    accepted_prefix(fake, &setup, 3, &seeded).expect("sound");
    let mut flawed = Rig::krow(Flaw::ShortlistFromWrongRow);
    let (fake, setup) = flawed.split();
    accepted_prefix(fake, &setup, 3, &seeded).expect_err("a record of another row");
}

#[test]
fn a_failed_row_read_at_draw_time_fails_the_window_closed() {
    let run = |flaw: Flaw| {
        let mut rig = Rig::krow(flaw);
        rig.fake.want_logits = false;
        rig.fake.run_window(&rows(4)).expect("the window");
        rig.fake.inject_fault(Fault::KRowDraw);
        let error = rig
            .fake
            .draw(0, &mut keeping(0, 5), VOCAB)
            .expect_err("the read fails");
        assert!(format!("{error:?}").contains("injected"), "{error:?}");
        rig
    };
    let sound = run(Flaw::Sound);
    assert!(sound.fake.poisoned && sound.fake.gate && sound.fake.window.is_none());
    let mut sound = sound;
    assert!(sound.fake.logits(1).is_err(), "a step on a failed window");
    // The flaw leaves a valid state with the window open: it is not failed closed.
    let flawed = run(Flaw::LazyReadNotGated);
    assert!(!flawed.fake.poisoned && !flawed.fake.gate && flawed.fake.window.is_some());
}

#[test]
fn a_record_the_sampler_refuses_at_draw_time_fails_the_window_closed() {
    let run = |flaw: Flaw| {
        let mut rig = Rig::krow(flaw);
        rig.fake.want_logits = false;
        rig.fake.run_window(&rows(4)).expect("the window");
        // The record was sound when the report was read; entry 7 repeats entry 3 before the draw.
        let draws = rig.fake.draws.as_mut().expect("draws");
        let entry = draws.shortlists[0][3];
        draws.shortlists[0][7] = entry;
        rig.fake
            .draw(0, &mut keeping(5, 7), VOCAB)
            .expect_err("a record the sampler refuses");
        rig
    };
    let sound = run(Flaw::Sound);
    assert!(sound.fake.poisoned && sound.fake.gate && sound.fake.window.is_none());
    // The flaw returns the error and leaves the window open with a valid state.
    let flawed = run(Flaw::ShortlistDrawErrorNotGated);
    assert!(!flawed.fake.poisoned && !flawed.fake.gate && flawed.fake.window.is_some());
}

#[test]
fn a_record_with_a_corrupt_later_entry_fails_the_window_closed_when_the_report_is_read() {
    // Sound: the same window runs and a seeded draw is made from the record.
    let mut sound = Rig::krow(Flaw::Sound);
    sound.fake.want_logits = false;
    serve(&mut sound.fake, 4, 2, &mut keeping(5, 7));
    // Corrupt entry 7 of row 0: the window fails closed at the read of the report, before any
    // draw can meet it (the state is poisoned, the gate is set, no window is stored).
    let mut rig = Rig::krow(Flaw::CorruptRecordEntry);
    rig.fake.want_logits = false;
    let error = rig.fake.run_window(&rows(4)).expect_err("a corrupt record");
    assert!(format!("{error:?}").contains("row 0"), "{error:?}");
    assert!(rig.fake.poisoned && rig.fake.gate && rig.fake.window.is_none());
}

#[test]
fn an_eager_window_with_the_base_and_inputs_of_a_k_row_window_is_not_served_its_records() {
    let run = |flaw: Flaw| {
        let mut rig = Rig::krow(flaw);
        rig.fake.want_logits = false;
        let base = rig.fake.snapshot().expect("snapshot");
        let first = serve(&mut rig.fake, 4, 4, &mut keeping(5, 7));
        assert_eq!(
            rig.fake.shortlist_draws, 4,
            "the K-row window draws from its records"
        );
        // The same base and inputs, run by the eager loop: the records of the first window are
        // still stored, and must not be used.
        rig.fake.restore(&base).expect("restore");
        rig.fake.path = Path::Eager;
        let second = serve(&mut rig.fake, 4, 4, &mut keeping(5, 7));
        assert_eq!(first, second, "the same draws either way");
        rig.fake.shortlist_draws
    };
    assert_eq!(
        run(Flaw::Sound),
        4,
        "the eager window drew from its own rows"
    );
    assert_eq!(
        run(Flaw::ServesStaleDraws),
        8,
        "the flaw serves the stale records"
    );
}

#[test]
fn with_the_k_row_path_off_a_draw_takes_the_one_row_path_and_the_draws_are_never_consulted() {
    let mut eager = Rig::new(Path::Eager, Flaw::Sound, false);
    for n in [2, 5, 8] {
        let tokens = serve(&mut eager.fake, n, n, &mut keeping(5, 7));
        assert_eq!(tokens.len(), n);
    }
    assert!(
        eager.fake.draws.is_none(),
        "an eager window stores no draws"
    );
    assert_eq!((eager.fake.shortlist_draws, eager.fake.row_reads), (0, 0));
    assert_eq!(eager.fake.logit_copies, 0);
}
