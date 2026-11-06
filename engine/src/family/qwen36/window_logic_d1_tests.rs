//! The D1 oracle against a fake model whose window lifecycle IS `window_logic`: the fake
//! implements `WindowHost` and `WindowModel` by forwarding to the same generic functions the step
//! uses, so fake and step cannot diverge on the host path. It models the recovery gate (the
//! `TokenStepper` and snapshot paths go through `step_gate` and `export_gate`) and the
//! reset-then-import restore that a gated step needs.

use super::{
    CommitPoint, Window, WindowHost, commit_window, export_gate, import_state, reset_epilogue,
    reset_prologue, run_window, state_usable, step_gate,
};
use crate::{
    EngError,
    family::{
        sampler::{Sampler, SamplingParams},
        session::{DecodeSession, Limits, TokenStepper},
        speculate::{
            d1_oracle::{
                D1Failure, D1Subject, FAKE_TOKENIZER_JSON, Fault, Setup, Tally, Verdict,
                accepted_prefix, failure_recovery, first_bit_difference, full_logits,
                host_faults_for, host_greedy, later_rows_isolated, repeated_rollbacks,
                ties_break_low,
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

/// What is wrong with the fake. `Sound` is the correct one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Flaw {
    Sound,
    /// `restore` imports without the reset first (mutant: the oracle must fail on the gate).
    RestoreWithoutReset,
}

#[derive(Clone, Debug, PartialEq)]
struct State {
    kv: Vec<u32>,
    recurrent: [f32; 3],
}

struct HostFake {
    flaw: Flaw,
    ties: bool,
    state: State,
    window: Option<Window<State, Vec<f32>>>,
    poisoned: bool,
    gate: bool,
    pending: bool,
    fault: Option<Fault>,
}

fn refused(what: &str) -> EngError {
    EngError::BadRequest(format!("host fake: {what}"))
}

impl HostFake {
    fn new(flaw: Flaw, ties: bool) -> Self {
        Self {
            flaw,
            ties,
            state: State {
                kv: Vec::new(),
                recurrent: [0.0; 3],
            },
            window: None,
            poisoned: false,
            gate: false,
            pending: false,
            fault: None,
        }
    }

    fn live(&self) -> Result<()> {
        if self.poisoned {
            return Err(refused("the state is poisoned"));
        }
        Ok(())
    }

    fn advance(&mut self, token: u32) -> Vec<f32> {
        for (j, value) in self.state.recurrent.iter_mut().enumerate() {
            *value = *value * 0.75 + token as f32 * 0.125 * (j as f32 + 1.0);
        }
        self.state.kv.push(token);
        let [r0, r1, r2] = self.state.recurrent;
        let position = self.state.kv.len() as f32;
        let mut row: Vec<f32> = (0..VOCAB)
            .map(|v| {
                let phase = r0 * 1.3 + r1 * 0.7 + r2 * 0.3 + v as f32 * 0.9 + token as f32 * 0.11;
                phase.sin() * 2.0 + (position * 0.01).cos()
            })
            .collect();
        if self.ties {
            row[2] = 9.0;
            row[6] = 9.0;
        }
        row
    }

    /// Fires the armed fault when it is `at`; `poison` says whether it is a device failure (the
    /// state is poisoned) or a refusal before any device work (the state stays valid).
    fn fires(&mut self, at: Fault, poison: bool) -> Result<()> {
        if self.fault != Some(at) {
            return Ok(());
        }
        self.fault = None;
        self.poisoned |= poison;
        Err(refused("injected fault"))
    }
}

impl TokenStepper for HostFake {
    const OPERATION: &'static str = "host fake";

    fn feed(&mut self, token: u32) -> Result<()> {
        step_gate(self)?;
        self.live()?;
        self.advance(token);
        Ok(())
    }

    fn logits(&mut self, token: u32) -> Result<Vec<f32>> {
        step_gate(self)?;
        self.live()?;
        Ok(self.advance(token))
    }

    fn fed(&self) -> usize {
        self.state.kv.len()
    }

    fn reset(&mut self) -> Result<()> {
        reset_prologue(self)?;
        self.state = State {
            kv: Vec::new(),
            recurrent: [0.0; 3],
        };
        self.poisoned = false;
        reset_epilogue(self);
        Ok(())
    }
}

impl WindowHost for HostFake {
    type Saved = State;
    type Row = Vec<f32>;

    fn device_wanted(&self) -> bool {
        false
    }
    fn tokens(&self) -> usize {
        self.state.kv.len()
    }
    fn window_awaits(&self) -> bool {
        self.window.is_some()
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
    }
    fn check_import(&mut self, _saved: &State) -> Result<()> {
        Ok(())
    }
    fn resident_kv(&mut self) -> Result<()> {
        Ok(())
    }
    fn store_window(&mut self, window: Window<State, Vec<f32>>) {
        self.window = Some(window);
    }
    fn take_window(&mut self) -> Option<Window<State, Vec<f32>>> {
        self.window.take()
    }
    fn export(&mut self) -> Result<State> {
        export_gate(self)?;
        self.live()?;
        Ok(self.state.clone())
    }
    fn run_row(&mut self, token: u32) -> Result<(u32, Vec<f32>)> {
        step_gate(self)?;
        self.live()?;
        let row = self.advance(token);
        Ok((host_greedy(&row, VOCAB), row))
    }
    fn replay_row(&mut self, token: u32) -> Result<()> {
        step_gate(self)?;
        self.live()?;
        self.advance(token);
        Ok(())
    }
    fn overwrite(&mut self, saved: &State) -> Result<()> {
        // As the device import: it writes the whole state and heals a poisoned book.
        self.state = saved.clone();
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
    fn rollback_begin(&mut self, _rows: usize) -> Result<()> {
        Ok(())
    }
    fn capture(&mut self, _row: usize) -> Result<()> {
        Ok(())
    }
    fn rollback_discard(&mut self) {}
    fn rollback_window_complete(&mut self) -> Result<()> {
        Ok(())
    }
    fn rollback_finish(&mut self, _kept: usize, _committed_rows: usize) -> Result<bool> {
        Ok(false)
    }
    fn feed_fault(&mut self, row: usize) -> Result<()> {
        self.fires(Fault::Feed(row), true)
    }
    fn replay_fault(&mut self, row: usize) -> Result<()> {
        self.fires(Fault::Replay(row), true)?;
        self.fires(Fault::ReplayRefused(row), false)
    }
    fn commit_fault(&mut self, at: CommitPoint) -> Result<()> {
        match at {
            CommitPoint::PrefixSave => self.fires(Fault::PrefixSave, false),
            CommitPoint::CheckImport => self.fires(Fault::CheckImport, false),
            CommitPoint::ResidentKv => self.fires(Fault::ResidentKv, false),
            CommitPoint::ImportDevice => self.fires(Fault::ImportDevice, true),
        }
    }
}

impl WindowModel for HostFake {
    fn run_window(&mut self, inputs: &[u32]) -> Result<()> {
        run_window(self, inputs)
    }

    fn draw(&mut self, row: usize, sampler: &mut Sampler, vocab: usize) -> Result<u32> {
        let window = self.window.as_ref().ok_or_else(|| refused("no window"))?;
        let logits = window.rows.get(row).ok_or_else(|| refused("no such row"))?;
        sampler.sample(&logits[..vocab])
    }

    fn commit_window(&mut self, commit: &Commit) -> Result<()> {
        commit_window(self, commit)
    }
}

impl D1Subject for HostFake {
    type State = State;

    fn snapshot(&mut self) -> Result<State> {
        export_gate(self)?;
        self.live()?;
        Ok(self.state.clone())
    }

    fn restore(&mut self, state: &State) -> Result<()> {
        // The restore of a step with a recovery gate: a reset, then the import.
        if self.flaw != Flaw::RestoreWithoutReset {
            self.reset()?;
        }
        import_state(self, state)
    }

    fn import_without_reset(&mut self, state: &State) -> Option<Result<()>> {
        Some(import_state(self, state))
    }

    fn state_difference(&self, left: &State, right: &State) -> Option<String> {
        if left.kv != right.kv {
            return Some(format!("KV rows {:?} against {:?}", left.kv, right.kv));
        }
        first_bit_difference(&left.recurrent, &right.recurrent)
            .map(|d| format!("recurrent state, {d}"))
    }

    fn window_rows(&self) -> Result<Vec<Vec<f32>>> {
        let window = self.window.as_ref().ok_or_else(|| refused("no window"))?;
        Ok(window.rows.clone())
    }

    fn inject_fault(&mut self, fault: Fault) {
        self.fault = Some(fault);
    }

    fn fault_plan(&self, rows: usize) -> Vec<(Fault, Commit)> {
        host_faults_for(self.state.kv.len(), rows)
    }
}

struct Rig {
    fake: HostFake,
    session: DecodeSession,
    tokenizer: Tokenizer,
}

impl Rig {
    fn new(flaw: Flaw, ties: bool) -> Self {
        let mut fake = HostFake::new(flaw, ties);
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

    fn split(&mut self) -> (&mut HostFake, Setup<'_>) {
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

fn violation(result: Verdict<Tally>) -> D1Failure {
    let failure = result.expect_err("the flaw must be caught");
    assert!(failure.violation, "{failure}");
    failure
}

#[test]
fn a_sound_host_path_passes_every_check() {
    let mut rig = Rig::new(Flaw::Sound, false);
    for n in [2, 4, 8] {
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
    let mut tied = Rig::new(Flaw::Sound, true);
    let tally = ties_break_low(&mut tied.fake, VOCAB, &rows(8)).expect("E");
    assert_eq!(tally.exact_ties, 8);
}

#[test]
fn every_host_fault_leaves_a_gated_state_that_only_a_reset_recovers() {
    for n in [2, 3, 4] {
        let mut rig = Rig::new(Flaw::Sound, false);
        let tally = failure_recovery(&mut rig.fake, VOCAB, &rows(n)).expect("F");
        assert_eq!(
            tally.comparisons,
            host_faults_for(rig.fake.state.kv.len(), n).len(),
            "{n} rows"
        );
    }
}

#[test]
fn the_host_plan_lists_every_fault_of_the_specification() {
    let plan = host_faults_for(10, 4);
    let faults: Vec<Fault> = plan.iter().map(|(fault, _)| *fault).collect();
    for want in [
        Fault::Feed(0),
        Fault::Feed(3),
        Fault::PrefixSave,
        Fault::CheckImport,
        Fault::ResidentKv,
        Fault::ImportDevice,
        Fault::Replay(0),
        Fault::Replay(2),
        Fault::ReplayRefused(0),
        Fault::ReplayRefused(2),
        Fault::Verdict,
    ] {
        assert!(faults.contains(&want), "{want:?} missing from {faults:?}");
    }
    let verdict = plan
        .iter()
        .find(|(f, _)| *f == Fault::Verdict)
        .expect("verdict");
    assert_eq!(
        verdict.1.committed_rows,
        10 + 3 + 1,
        "one row more than base + kept"
    );
}

#[test]
fn a_restore_without_the_reset_is_caught_by_the_gate() {
    let mut rig = Rig::new(Flaw::RestoreWithoutReset, false);
    // The restore itself is refused by the gate, which check F reports as an engine error.
    let failure = failure_recovery(&mut rig.fake, VOCAB, &rows(4)).expect_err("the gate");
    assert!(failure.test.starts_with('F'), "{failure}");
    assert!(format!("{failure}").contains("reset first"), "{failure}");
}

#[test]
fn an_import_without_a_reset_is_refused_after_a_failed_commit_and_a_reset_recovers() {
    let mut rig = Rig::new(Flaw::Sound, false);
    let base = rig.fake.snapshot().expect("snapshot");
    let at = rig.fake.state.kv.len();
    rig.fake.inject_fault(Fault::CheckImport);
    rig.fake.run_window(&rows(3)).expect("run");
    let kept = Commit {
        committed_rows: at + 2,
        kept_rows: 2,
        discarded_rows: 1,
        state_slot: 1,
    };
    rig.fake.commit_window(&kept).expect_err("the check fails");
    assert!(rig.fake.snapshot().is_err() && rig.fake.logits(1).is_err());
    assert!(
        rig.fake
            .import_without_reset(&base)
            .expect("modelled")
            .is_err()
    );
    rig.fake.reset().expect("reset");
    rig.fake.restore(&base).expect("restore");
    assert_eq!(rig.fake.state, base);
}
