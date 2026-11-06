//! The D1 oracle against a fake model that rolls back the way the device path does: per-row state
//! slots for the recurrent and the convolution state (two components), a KV frontier that
//! truncates and never clears, and the real `RollbackBook` for the bookkeeping. A sound fake passes
//! every check; each flawed copy must fail the check meant to catch the flaw.
//!
//! What this proves: the indices, the frontier, the component split, the guards and the order of
//! the copies. What it cannot prove: the byte range of the device copy, the slot offsets in the
//! buffer and the stream order. Those are device-only (`cuda/state/rollback.rs`).

use super::{ALTERNATIVES, CONTEXT, FAKE_TOKENIZER_JSON, PROMPT, VOCAB, rows, seeded};
use crate::{
    EngError,
    family::{
        qwen36::rollback::{RollbackBook, RollbackPlan, StateFacts},
        sampler::Sampler,
        session::{DecodeSession, Limits, TokenStepper},
        speculate::{
            d1_oracle::{
                D1Failure, D1Subject, Fault, Setup, Tally, Verdict, accepted_prefix,
                failure_recovery, first_bit_difference, full_logits, later_rows_isolated,
                repeated_rollbacks, slot_faults_for, ties_break_low,
            },
            driver::WindowModel,
            window::Commit,
        },
    },
    tokenizer::Tokenizer,
};

type Result<T> = std::result::Result<T, EngError>;

/// What is wrong with the slot fake. `Sound` is the correct one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mutant {
    Sound,
    /// A partial accept restores slot `kept` instead of `kept - 1`.
    SlotOffByOne,
    /// The frontier after a rollback is one row too far.
    FrontierPlusOne,
    /// The frontier after a rollback is one row short.
    FrontierMinusOne,
    /// The frontier is not truncated: the discarded rows stay committed.
    NoTruncate,
    /// The copy after row `i` is taken before the row runs.
    CaptureBeforeRow,
    /// Every capture writes slot 0.
    CaptureAlwaysSlot0,
    /// The recurrent state is restored, the convolution state is not.
    RecurrentOnly,
    /// The convolution state is restored, the recurrent state is not.
    ConvOnly,
    /// The restore copies the live state over the slot, not the slot over the live state.
    ReverseCopy,
    /// The commit does not check that the state is valid.
    NoValidCheck,
    /// A failed commit leaves the state valid.
    NoPoisonOnError,
    /// A row writes the cell below the frontier too.
    RowWritesBelowFrontier,
    /// An import leaves the window open.
    ImportKeepsWindow,
    /// A decode step or an export is allowed while a window awaits its verdict.
    RunsWhileAwaiting,
    /// A window over an open one is refused but leaves the state valid.
    OverlapKeepsValid,
    /// A guard failure other than health leaves the state valid.
    GuardFailureKeepsValid,
    /// A snapshot is accepted through the recovery gate.
    SnapshotIgnoresGate,
    /// A snapshot and a decode step are both accepted through the recovery gate.
    SnapshotAndStepIgnoreGate,
    /// An import without a reset is accepted through the recovery gate.
    ImportIgnoresGate,
}

/// The state a snapshot carries: the committed cells, never the stale ones above the frontier.
#[derive(Clone, Debug, PartialEq)]
struct SlotState {
    cells: Vec<u32>,
    recurrent: [f32; 3],
    conv: [f32; 2],
}

struct Open {
    inputs: Vec<u32>,
    rows: Vec<Vec<f32>>,
}

struct SlotFake {
    mutant: Mutant,
    ties: bool,
    /// The physical KV: grows, is overwritten, is never cleared.
    cells: Vec<u32>,
    /// Committed rows: the frontier.
    len: usize,
    recurrent: [f32; 3],
    conv: [f32; 2],
    slots: Vec<([f32; 3], [f32; 2])>,
    book: RollbackBook,
    open: Option<Open>,
    poisoned: bool,
    /// The recovery gate: set with every poison, cleared by a reset or by `restore`, which is a
    /// reset followed by the import. An import without the reset is refused while it is set.
    gate: bool,
    fault: Option<Fault>,
    /// A step is pending: the device words are not the host's.
    busy: bool,
}

fn refused(what: &str) -> EngError {
    EngError::BadRequest(format!("slot fake: {what}"))
}

impl SlotFake {
    fn new(mutant: Mutant, ties: bool, rows: usize) -> Self {
        let plan = RollbackPlan::new(rows, 1024).expect("plan");
        Self {
            mutant,
            ties,
            cells: Vec::new(),
            len: 0,
            recurrent: [0.0; 3],
            conv: [0.0; 2],
            // Garbage, not zeros: a slot that was never captured must not look right.
            slots: vec![([7.5; 3], [7.5; 2]); plan.slots()],
            book: RollbackBook::new(plan),
            open: None,
            poisoned: false,
            gate: false,
            fault: None,
            busy: false,
        }
    }

    fn facts(&self) -> StateFacts {
        StateFacts {
            owner: true,
            valid: !self.poisoned,
            idle: !self.busy,
            resident_kv: true,
        }
    }

    fn live(&self) -> Result<()> {
        if self.poisoned {
            return Err(refused("the state is poisoned"));
        }
        Ok(())
    }

    /// The rule of the step: a decode step through a state whose window awaits its verdict poisons
    /// it, and an export is refused.
    fn refuse_while_awaiting(&mut self, poison: bool) -> Result<()> {
        if self.mutant == Mutant::RunsWhileAwaiting || !self.book.awaiting_verdict() {
            return Ok(());
        }
        if poison {
            self.fail();
        }
        Err(refused("a window waits for its verdict"))
    }

    fn fail(&mut self) {
        self.open = None;
        self.book.discard();
        self.poisoned = true;
        self.gate = true;
    }

    fn fires(&mut self, at: Fault) -> Result<()> {
        if self.fault != Some(at) {
            return Ok(());
        }
        self.fault = None;
        self.poisoned = self.mutant != Mutant::NoPoisonOnError;
        self.gate = self.poisoned;
        Err(refused("injected fault"))
    }

    /// Feeds one token and returns the logits row: attention over the committed cells, the two
    /// state components and the token.
    fn advance(&mut self, token: u32) -> Vec<f32> {
        if self.mutant == Mutant::RowWritesBelowFrontier && self.len > 0 {
            self.cells[self.len - 1] += 1;
        }
        let attention: f32 = self.cells[..self.len]
            .iter()
            .enumerate()
            .map(|(at, cell)| *cell as f32 * (0.01 + at as f32 * 0.0007))
            .sum();
        for (j, value) in self.recurrent.iter_mut().enumerate() {
            *value = *value * 0.75 + token as f32 * 0.125 * (j as f32 + 1.0);
        }
        for (j, value) in self.conv.iter_mut().enumerate() {
            *value = *value * 0.5 + token as f32 * 0.0625 * (j as f32 + 2.0);
        }
        if self.len < self.cells.len() {
            self.cells[self.len] = token;
        } else {
            self.cells.push(token);
        }
        self.len += 1;
        let [r0, r1, r2] = self.recurrent;
        let [c0, c1] = self.conv;
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
                phase.sin() * 2.0 + (self.len as f32 * 0.01).cos()
            })
            .collect();
        if self.ties {
            row[2] = 9.0;
            row[6] = 9.0;
        }
        row
    }

    fn capture(&mut self, row: usize) -> Result<()> {
        let slot = self.book.begin_capture(row)?;
        let at = if self.mutant == Mutant::CaptureAlwaysSlot0 {
            0
        } else {
            slot
        };
        *self
            .slots
            .get_mut(at)
            .ok_or_else(|| refused("slot out of range"))? = (self.recurrent, self.conv);
        self.book.note_captured(row)
    }

    /// Poisons the state and drops the window: any error after the first row ran.
    fn failing<T>(&mut self, result: Result<T>) -> Result<T> {
        if result.is_err() && self.mutant != Mutant::NoPoisonOnError {
            self.fail();
        }
        result
    }

    fn window_body(&mut self, inputs: &[u32]) -> Result<Vec<Vec<f32>>> {
        let mut out = Vec::with_capacity(inputs.len());
        for (at, &token) in inputs.iter().enumerate() {
            self.fires(Fault::Feed(at))?;
            let last = at + 1 == inputs.len();
            if self.mutant == Mutant::CaptureBeforeRow && !last {
                self.capture(at)?;
            }
            out.push(self.advance(token));
            if !last && self.mutant != Mutant::CaptureBeforeRow {
                self.fires(Fault::Capture(at))?;
                self.capture(at)?;
            }
        }
        Ok(out)
    }
}

impl TokenStepper for SlotFake {
    const OPERATION: &'static str = "slot fake";

    fn feed(&mut self, token: u32) -> Result<()> {
        if self.mutant != Mutant::SnapshotAndStepIgnoreGate {
            self.live()?;
            self.refuse_while_awaiting(true)?;
        }
        self.advance(token);
        Ok(())
    }

    fn logits(&mut self, token: u32) -> Result<Vec<f32>> {
        if self.mutant != Mutant::SnapshotAndStepIgnoreGate {
            self.live()?;
            self.refuse_while_awaiting(true)?;
        }
        Ok(self.advance(token))
    }

    fn fed(&self) -> usize {
        self.len
    }

    fn reset(&mut self) -> Result<()> {
        self.len = 0;
        self.recurrent = [0.0; 3];
        self.conv = [0.0; 2];
        self.open = None;
        self.book.discard();
        self.poisoned = false;
        self.gate = false;
        Ok(())
    }
}

impl WindowModel for SlotFake {
    fn run_window(&mut self, inputs: &[u32]) -> Result<()> {
        self.open = None;
        self.live()?;
        self.facts().require_copyable("run_window")?;
        if self.book.is_open() {
            // The previous window has no verdict: the state is ahead of the lineage.
            if self.mutant != Mutant::OverlapKeepsValid {
                self.fail();
            }
            return Err(refused("the previous window has no verdict"));
        }
        self.book.begin_window(self.len, inputs.len())?;
        let result = self.window_body(inputs);
        let rows = self.failing(result)?;
        self.open = Some(Open {
            inputs: inputs.to_vec(),
            rows,
        });
        self.book.mark_complete()?;
        Ok(())
    }

    fn draw(&mut self, row: usize, sampler: &mut Sampler, vocab: usize) -> Result<u32> {
        let open = self.open.as_ref().ok_or_else(|| refused("no window"))?;
        let logits = open.rows.get(row).ok_or_else(|| refused("no such row"))?;
        sampler.sample(&logits[..vocab])
    }

    fn commit_window(&mut self, commit: &Commit) -> Result<()> {
        let open = self.open.take().ok_or_else(|| refused("no window"))?;
        if self.mutant != Mutant::NoValidCheck {
            // A poisoned state is refused here and stays poisoned; the window is dropped.
            if let Err(error) = self.facts().require_copyable("commit_window") {
                self.book.discard();
                if self.mutant != Mutant::GuardFailureKeepsValid {
                    self.poisoned = true;
                    self.gate = true;
                }
                return Err(error);
            }
        }
        let verdict = self
            .book
            .finish_window(commit.kept_rows, commit.committed_rows, self.len);
        let restore = self.failing(verdict)?;
        let Some(restore) = restore else {
            return Ok(());
        };
        let faulted = self.fires(Fault::Restore);
        self.failing(faulted)?;
        let slot = match self.mutant {
            Mutant::SlotOffByOne => restore.slot + 1,
            _ => restore.slot,
        };
        let copied = self.slots.get(slot).copied();
        let (recurrent, conv) = match copied {
            Some(pair) => pair,
            None => {
                self.fail();
                return Err(refused("slot out of range"));
            }
        };
        match self.mutant {
            Mutant::ReverseCopy => self.slots[slot] = (self.recurrent, self.conv),
            Mutant::RecurrentOnly => self.recurrent = recurrent,
            Mutant::ConvOnly => self.conv = conv,
            _ => {
                self.recurrent = recurrent;
                self.conv = conv;
            }
        }
        self.len = match self.mutant {
            Mutant::FrontierPlusOne => restore.frontier + 1,
            Mutant::FrontierMinusOne => restore.frontier - 1,
            Mutant::NoTruncate => self.len,
            _ => restore.frontier,
        };
        // A poisoned state that was not refused above is healed here: only the missing check.
        self.poisoned = false;
        drop(open);
        Ok(())
    }
}

impl D1Subject for SlotFake {
    type State = SlotState;

    fn snapshot(&mut self) -> Result<SlotState> {
        if !matches!(
            self.mutant,
            Mutant::SnapshotIgnoresGate | Mutant::SnapshotAndStepIgnoreGate
        ) {
            self.live()?;
            self.refuse_while_awaiting(false)?;
        }
        Ok(SlotState {
            cells: self.cells[..self.len].to_vec(),
            recurrent: self.recurrent,
            conv: self.conv,
        })
    }

    fn restore(&mut self, state: &SlotState) -> Result<()> {
        // As the device import: the committed cells are written, the stale ones stay.
        for (at, &cell) in state.cells.iter().enumerate() {
            if at < self.cells.len() {
                self.cells[at] = cell;
            } else {
                self.cells.push(cell);
            }
        }
        self.len = state.cells.len();
        self.recurrent = state.recurrent;
        self.conv = state.conv;
        self.poisoned = false;
        self.gate = false;
        self.open = None;
        if self.mutant != Mutant::ImportKeepsWindow {
            self.book.discard();
        }
        Ok(())
    }

    fn models_recovery_gate(&self) -> bool {
        true
    }

    fn import_without_reset(&mut self, state: &SlotState) -> Option<Result<()>> {
        if self.gate && self.mutant != Mutant::ImportIgnoresGate {
            return Some(Err(refused("reset first")));
        }
        Some(self.restore(state))
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
        let open = self.open.as_ref().ok_or_else(|| refused("no window"))?;
        Ok(open.rows.clone())
    }

    fn inject_fault(&mut self, fault: Fault) {
        self.fault = Some(fault);
    }

    fn fault_plan(&self, rows: usize) -> Vec<(Fault, Commit)> {
        slot_faults_for(self.len, rows)
    }
}

struct SlotRig {
    fake: SlotFake,
    session: DecodeSession,
    tokenizer: Tokenizer,
}

impl SlotRig {
    fn new(mutant: Mutant, ties: bool) -> Self {
        let mut fake = SlotFake::new(mutant, ties, 8);
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

    fn split(&mut self) -> (&mut SlotFake, Setup<'_>) {
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

/// A check that must fail, as a violation or as an engine error (a flaw can end in either).
fn caught(result: Verdict<Tally>, what: &str) -> D1Failure {
    result.expect_err(what)
}

// ---- the sound fake passes every check ----

#[test]
fn a_sound_slot_fake_passes_every_check() {
    let mut rig = SlotRig::new(Mutant::Sound, false);
    for n in [2, 3, 4, 8] {
        full_logits(&mut rig.fake, &rows(n)).expect("A");
        later_rows_isolated(&mut rig.fake, &rows(n), &ALTERNATIVES).expect("C");
    }
    let (fake, setup) = rig.split();
    for drafts in [1, 3, 7] {
        accepted_prefix(fake, &setup, drafts, &Sampler::greedy).expect("D greedy");
        accepted_prefix(fake, &setup, drafts, &seeded).expect("D seeded");
    }
    let tally = repeated_rollbacks(fake, &setup, 3, &Sampler::greedy, 40).expect("G greedy");
    assert_eq!(tally.comparisons, 5, "windows 1, 2, 3, 20 and 40");
    repeated_rollbacks(fake, &setup, 7, &seeded, 40).expect("G seeded");
    let mut tied = SlotRig::new(Mutant::Sound, true);
    let tally = ties_break_low(&mut tied.fake, VOCAB, &rows(8)).expect("E");
    assert_eq!(tally.exact_ties, 8);
}

#[test]
fn the_slot_fake_fails_closed_at_every_fault_point_and_recovers() {
    for n in [2, 3, 4, 8] {
        let mut rig = SlotRig::new(Mutant::Sound, false);
        let tally = failure_recovery(&mut rig.fake, VOCAB, &rows(n)).expect("F");
        assert_eq!(
            tally.comparisons,
            slot_faults_for(rig.fake.len, n).len(),
            "{n} rows"
        );
    }
    // The restore fault fires at slot 0 (kept 1) and at the last slot (kept rows - 1).
    let plan = slot_faults_for(10, 4);
    let restores: Vec<_> = plan
        .iter()
        .filter(|(fault, _)| *fault == Fault::Restore)
        .map(|(_, commit)| (commit.kept_rows, commit.committed_rows))
        .collect();
    assert_eq!(restores, [(3, 13), (1, 11)]);
}

// ---- every flaw is caught by the check meant to catch it ----

fn d_catches(mutant: Mutant) -> D1Failure {
    let mut rig = SlotRig::new(mutant, false);
    let (fake, setup) = rig.split();
    caught(accepted_prefix(fake, &setup, 3, &Sampler::greedy), "D")
}

#[test]
fn a_wrong_slot_index_is_caught_by_the_accepted_prefix_check() {
    d_catches(Mutant::SlotOffByOne);
    d_catches(Mutant::CaptureBeforeRow);
    d_catches(Mutant::CaptureAlwaysSlot0);
}

#[test]
fn a_wrong_frontier_is_caught_by_the_accepted_prefix_check() {
    for mutant in [
        Mutant::FrontierPlusOne,
        Mutant::FrontierMinusOne,
        Mutant::NoTruncate,
    ] {
        let failure = d_catches(mutant);
        assert!(failure.test.starts_with('D'), "{mutant:?}: {failure}");
    }
}

#[test]
fn a_restore_of_one_state_component_is_caught_by_the_accepted_prefix_check() {
    for mutant in [Mutant::RecurrentOnly, Mutant::ConvOnly, Mutant::ReverseCopy] {
        let failure = d_catches(mutant);
        assert!(failure.test.starts_with('D'), "{mutant:?}: {failure}");
    }
}

#[test]
fn a_row_that_writes_below_the_frontier_is_caught() {
    let failure = d_catches(Mutant::RowWritesBelowFrontier);
    assert!(failure.violation, "{failure}");
}

#[test]
fn the_cycle_check_catches_flaws_that_a_single_window_may_miss() {
    for mutant in [
        Mutant::SlotOffByOne,
        Mutant::FrontierMinusOne,
        Mutant::ConvOnly,
        Mutant::RowWritesBelowFrontier,
    ] {
        let mut rig = SlotRig::new(mutant, false);
        let (fake, setup) = rig.split();
        let failure = caught(
            repeated_rollbacks(fake, &setup, 3, &Sampler::greedy, 24),
            "G",
        );
        assert!(
            failure.test.starts_with('G') || failure.test.starts_with('D'),
            "{failure}"
        );
    }
}

#[test]
fn a_failed_commit_that_leaves_the_state_valid_is_caught_by_the_failure_check() {
    let mut rig = SlotRig::new(Mutant::NoPoisonOnError, false);
    let failure = caught(failure_recovery(&mut rig.fake, VOCAB, &rows(4)), "F");
    // The recovery gate is checked after every fault: a failed commit that leaves a usable state is
    // refused there, ahead of the silent-partial-state rule.
    assert!(
        failure.detail.contains("a decode step was accepted"),
        "{failure}"
    );
}

/// The detail of the violation that check F reports for the slot fake with `mutant`.
fn f_detail(mutant: Mutant) -> String {
    let mut rig = SlotRig::new(mutant, false);
    let failure = caught(failure_recovery(&mut rig.fake, VOCAB, &rows(4)), "F");
    assert!(failure.violation, "{failure}");
    failure.detail
}

#[test]
fn each_gate_assertion_of_check_f_has_a_flawed_slot_fake_that_only_it_reports() {
    // The step assertion runs first, so a subject that accepts a step and a snapshot is reported
    // for the step; dropping it would change the report.
    assert!(
        f_detail(Mutant::SnapshotAndStepIgnoreGate).contains("a decode step was accepted"),
        "step"
    );
    assert!(
        f_detail(Mutant::SnapshotIgnoresGate).contains("a snapshot was accepted"),
        "snapshot"
    );
    assert!(
        f_detail(Mutant::ImportIgnoresGate).contains("an import without a reset was accepted"),
        "import"
    );
}

#[test]
fn an_import_that_leaves_the_window_open_is_caught() {
    // The next window finds the old one without a verdict.
    let mut rig = SlotRig::new(Mutant::ImportKeepsWindow, false);
    let base = rig.fake.snapshot().unwrap();
    rig.fake.run_window(&rows(4)).unwrap();
    rig.fake.restore(&base).unwrap();
    // The sound fake: the same sequence runs.
    let mut sound = SlotRig::new(Mutant::Sound, false);
    let base = sound.fake.snapshot().unwrap();
    sound.fake.run_window(&rows(4)).unwrap();
    sound.fake.restore(&base).unwrap();
    sound
        .fake
        .run_window(&rows(4))
        .expect("a window after an import");
    assert!(
        rig.fake.run_window(&rows(4)).is_err(),
        "the window left open by the import blocks the next one"
    );
}

/// A state poisoned between `run_window` and `commit_window` must stay poisoned.
fn stays_poisoned_after_a_late_poison(mutant: Mutant) -> bool {
    let mut rig = SlotRig::new(mutant, false);
    let base = rig.fake.len;
    rig.fake.run_window(&rows(4)).unwrap();
    rig.fake.poisoned = true;
    let commit = Commit {
        committed_rows: base + 2,
        kept_rows: 2,
        discarded_rows: 2,
        state_slot: 1,
    };
    let committed = rig.fake.commit_window(&commit);
    committed.is_err() && rig.fake.poisoned && rig.fake.snapshot().is_err()
}

#[test]
fn a_state_poisoned_between_the_window_and_its_commit_stays_poisoned() {
    assert!(stays_poisoned_after_a_late_poison(Mutant::Sound));
    assert!(
        !stays_poisoned_after_a_late_poison(Mutant::NoValidCheck),
        "the missing Valid check heals a poisoned state: the test sees it"
    );
}

#[test]
fn a_capture_or_restore_on_a_busy_or_poisoned_state_is_refused_by_the_guard() {
    let mut rig = SlotRig::new(Mutant::Sound, false);
    rig.fake.poisoned = true;
    assert!(rig.fake.run_window(&rows(4)).is_err());
    assert!(!rig.fake.book.is_open(), "a refused window opens nothing");
}

#[test]
fn a_verdict_that_commits_the_wrong_row_count_is_refused_and_poisons() {
    let mut rig = SlotRig::new(Mutant::Sound, false);
    let base = rig.fake.len;
    rig.fake.run_window(&rows(4)).unwrap();
    let wrong = Commit {
        committed_rows: base + 3,
        kept_rows: 2,
        discarded_rows: 2,
        state_slot: 1,
    };
    assert!(rig.fake.commit_window(&wrong).is_err());
    assert!(rig.fake.snapshot().is_err(), "the state is closed");
}

#[test]
fn the_slot_fake_keeps_stale_cells_above_the_frontier() {
    // The stale cells are the point: a frontier that is too long reads them.
    let mut rig = SlotRig::new(Mutant::Sound, false);
    let before = rig.fake.len;
    rig.fake.run_window(&rows(4)).unwrap();
    let commit = Commit {
        committed_rows: before + 1,
        kept_rows: 1,
        discarded_rows: 3,
        state_slot: 0,
    };
    rig.fake.commit_window(&commit).unwrap();
    assert_eq!(rig.fake.len, before + 1);
    assert_eq!(rig.fake.cells.len(), before + 4, "the discarded cells stay");
}

#[test]
fn a_step_or_an_export_through_a_window_that_awaits_its_verdict_is_refused() {
    let mut rig = SlotRig::new(Mutant::Sound, false);
    rig.fake.run_window(&rows(4)).unwrap();
    assert!(rig.fake.snapshot().is_err(), "an export is refused");
    assert!(!rig.fake.poisoned, "an export only refuses");
    assert!(rig.fake.logits(1).is_err(), "a decode step is refused ...");
    assert!(rig.fake.poisoned, "... and poisons the state");
    assert!(!rig.fake.book.is_open(), "and drops the window");
    // The way back is a restore.
    let mut rig = SlotRig::new(Mutant::Sound, false);
    let base = rig.fake.snapshot().unwrap();
    rig.fake.run_window(&rows(4)).unwrap();
    rig.fake.restore(&base).unwrap();
    rig.fake.logits(1).expect("an import closes the window");
}

#[test]
fn the_awaiting_rule_is_what_blocks_a_step_through_speculative_rows() {
    let mut sound = SlotRig::new(Mutant::Sound, false);
    sound.fake.run_window(&rows(4)).unwrap();
    assert!(sound.fake.feed(1).is_err());
    let mut flawed = SlotRig::new(Mutant::RunsWhileAwaiting, false);
    flawed.fake.run_window(&rows(4)).unwrap();
    assert!(
        flawed.fake.feed(1).is_ok(),
        "without the rule the step continues from the speculative rows: the sound fake differs"
    );
}

#[test]
fn a_window_over_an_open_one_poisons_the_state() {
    for (mutant, poisoned) in [(Mutant::Sound, true), (Mutant::OverlapKeepsValid, false)] {
        let mut rig = SlotRig::new(mutant, false);
        rig.fake.run_window(&rows(4)).unwrap();
        assert!(rig.fake.run_window(&rows(4)).is_err());
        assert_eq!(rig.fake.poisoned, poisoned, "{mutant:?}");
    }
}

#[test]
fn a_guard_failure_of_any_fact_poisons_the_state_at_the_commit() {
    for (mutant, poisoned) in [
        (Mutant::Sound, true),
        (Mutant::GuardFailureKeepsValid, false),
    ] {
        let mut rig = SlotRig::new(mutant, false);
        let base = rig.fake.len;
        rig.fake.run_window(&rows(4)).unwrap();
        rig.fake.busy = true;
        let commit = Commit {
            committed_rows: base + 2,
            kept_rows: 2,
            discarded_rows: 2,
            state_slot: 1,
        };
        assert!(rig.fake.commit_window(&commit).is_err());
        assert_eq!(rig.fake.poisoned, poisoned, "{mutant:?}");
        assert!(!rig.fake.book.is_open());
    }
}
