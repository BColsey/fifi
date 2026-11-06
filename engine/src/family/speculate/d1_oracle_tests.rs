//! Host tests of the D1 oracle: a fake model that is correct by construction passes every check,
//! and flawed copies of it fail the check that is meant to catch the flaw. The real-pack versions
//! of the checks are device tests that have not run (`qwen36/cuda/d1_window_device_tests.rs`).

use super::{
    d1_oracle::{
        D1Failure, D1Subject, FAKE_TOKENIZER_JSON, Fault, Observed, Setup, Tally, accepted_prefix,
        describe_pass, failure_recovery, first_bit_difference, first_byte_difference, full_logits,
        host_greedy, later_rows_isolated, near_tie, observed_difference, tied_at_maximum,
        ties_break_low,
    },
    driver::WindowModel,
    window::Commit,
};
use crate::{
    EngError,
    family::{
        sampler::{Sampler, SamplingParams},
        session::{DecodeSession, Limits, TokenStepper},
    },
    tokenizer::Tokenizer,
};

type Result<T> = std::result::Result<T, EngError>;

const VOCAB: usize = 8;
const CONTEXT: usize = 4_096;
const PROMPT: [u32; 12] = [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3];
/// Replacement tokens for the isolation check: ids, a repeat of the first row and the edges.
const ALTERNATIVES: [u32; 4] = [0, 7, 3, 5];

/// What is wrong with a fake model. `Sound` is the correct one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Flaw {
    Sound,
    /// A row's logits depend on the rows after it.
    LaterRowLeak,
    /// Windows of five or more rows take another numeric path.
    RowCountKeyed,
    /// Logits ignore the earlier rows: nothing is connected.
    Deaf,
    /// A partial accept restores nothing: the discarded rows stay.
    KeepsDiscardedRows,
    /// A partial accept restores the KV rows but not the recurrent state.
    KeepsRecurrent,
    /// A greedy draw takes the highest id of a tie.
    TieHigh,
    /// A sampled draw consumes one extra value of the generator.
    ExtraDraw,
    /// A fault leaves the partial state in place and keeps answering.
    SilentFault,
    /// After a commit the model reports one row more than it holds.
    FedLies,
    /// The first step after a partial commit answers from a stale logits row.
    StaleLogits,
    /// The one-row step takes the highest id of a tie; the window and the rules take the lowest.
    OneRowTieHigh,
}

/// The state commit and rollback must restore.
#[derive(Clone, Debug, PartialEq)]
struct FakeState {
    /// The token at each committed position: stands for the KV rows and their count.
    kv: Vec<u32>,
    /// Stands for the recurrent and convolution state.
    recurrent: [f32; 3],
}

struct FakeWindow {
    inputs: Vec<u32>,
    rows: Vec<Vec<f32>>,
    saved: FakeState,
}

struct Fake {
    flaw: Flaw,
    /// Logits 2 and 6 tie for the maximum on every row.
    ties: bool,
    state: FakeState,
    window: Option<FakeWindow>,
    poisoned: bool,
    fault: Option<Fault>,
    /// Set by a commit under `FedLies`: `fed` reports one row too many.
    fed_lies: bool,
    /// Set by a partial commit under `StaleLogits`: the next `logits` call is perturbed.
    stale: bool,
}

fn refused(what: &str) -> EngError {
    EngError::BadRequest(format!("fake: {what}"))
}

impl Fake {
    fn new(flaw: Flaw, ties: bool) -> Self {
        Self {
            flaw,
            ties,
            state: FakeState {
                kv: Vec::new(),
                recurrent: [0.0; 3],
            },
            window: None,
            poisoned: false,
            fault: None,
            fed_lies: false,
            stale: false,
        }
    }

    /// Feeds one token and returns the logits row after it.
    fn advance(&mut self, token: u32) -> Vec<f32> {
        if self.flaw != Flaw::Deaf {
            for (j, value) in self.state.recurrent.iter_mut().enumerate() {
                *value = *value * 0.75 + token as f32 * 0.125 * (j as f32 + 1.0);
            }
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

    fn live(&self) -> Result<()> {
        if self.poisoned {
            return Err(refused("the state is poisoned"));
        }
        Ok(())
    }

    /// Fires the armed fault when it matches: poisons unless the flaw is silent.
    fn fires(&mut self, at: Fault) -> Result<()> {
        if self.fault != Some(at) {
            return Ok(());
        }
        self.fault = None;
        self.poisoned = self.flaw != Flaw::SilentFault;
        Err(refused("injected fault"))
    }
}

impl TokenStepper for Fake {
    const OPERATION: &'static str = "fake";

    fn feed(&mut self, token: u32) -> Result<()> {
        self.live()?;
        self.advance(token);
        Ok(())
    }

    fn logits(&mut self, token: u32) -> Result<Vec<f32>> {
        self.live()?;
        let mut row = self.advance(token);
        if std::mem::take(&mut self.stale) {
            row[0] += 0.01;
        }
        Ok(row)
    }

    fn next_token(&mut self, token: u32, sampler: &mut Sampler, vocab: usize) -> Result<u32> {
        let row = self.logits(token)?;
        if self.flaw == Flaw::OneRowTieHigh && sampler.is_greedy() {
            let top = row[host_greedy(&row, vocab) as usize];
            return Ok(row[..vocab]
                .iter()
                .rposition(|value| *value == top)
                .unwrap() as u32);
        }
        sampler.sample(&row[..vocab])
    }

    fn fed(&self) -> usize {
        self.state.kv.len() + usize::from(self.fed_lies)
    }

    fn reset(&mut self) -> Result<()> {
        self.state.kv.clear();
        self.state.recurrent = [0.0; 3];
        self.window = None;
        self.poisoned = false;
        self.fed_lies = false;
        self.stale = false;
        Ok(())
    }
}

impl WindowModel for Fake {
    fn run_window(&mut self, inputs: &[u32]) -> Result<()> {
        self.window = None;
        self.live()?;
        let saved = self.state.clone();
        let mut rows = Vec::with_capacity(inputs.len());
        for (at, &token) in inputs.iter().enumerate() {
            self.fires(Fault::Feed(at))?;
            rows.push(self.advance(token));
        }
        match self.flaw {
            Flaw::LaterRowLeak => {
                for (at, row) in rows.iter_mut().enumerate() {
                    row[0] += 0.001 * inputs[at + 1..].iter().sum::<u32>() as f32;
                }
            }
            Flaw::RowCountKeyed if inputs.len() >= 5 => {
                rows.iter_mut().for_each(|row| row[1] += 0.01);
            }
            _ => {}
        }
        self.window = Some(FakeWindow {
            inputs: inputs.to_vec(),
            rows,
            saved,
        });
        Ok(())
    }

    fn draw(&mut self, row: usize, sampler: &mut Sampler, vocab: usize) -> Result<u32> {
        let window = self.window.as_ref().ok_or_else(|| refused("no window"))?;
        let logits = window.rows.get(row).ok_or_else(|| refused("no such row"))?;
        let visible = &logits[..vocab];
        if self.flaw == Flaw::TieHigh && sampler.is_greedy() {
            let top = visible[host_greedy(visible, vocab) as usize];
            return Ok(visible.iter().rposition(|value| *value == top).unwrap() as u32);
        }
        let token = sampler.sample(visible)?;
        if self.flaw == Flaw::ExtraDraw && !sampler.is_greedy() {
            sampler.sample(visible)?;
        }
        Ok(token)
    }

    fn commit_window(&mut self, commit: &Commit) -> Result<()> {
        let window = self.window.take().ok_or_else(|| refused("no window"))?;
        self.fed_lies = self.flaw == Flaw::FedLies;
        if commit.discarded_rows == 0 {
            return Ok(());
        }
        self.stale = self.flaw == Flaw::StaleLogits;
        match self.flaw {
            Flaw::KeepsDiscardedRows => return Ok(()),
            Flaw::KeepsRecurrent => {
                let recurrent = self.state.recurrent;
                self.state = window.saved.clone();
                self.state.recurrent = recurrent;
            }
            _ => self.state = window.saved.clone(),
        }
        for (at, &token) in window.inputs[..commit.kept_rows].iter().enumerate() {
            self.fires(Fault::Replay(at))?;
            self.advance(token);
        }
        Ok(())
    }
}

impl D1Subject for Fake {
    type State = FakeState;

    fn snapshot(&mut self) -> Result<FakeState> {
        self.live()?;
        Ok(self.state.clone())
    }

    fn restore(&mut self, state: &FakeState) -> Result<()> {
        self.state = state.clone();
        self.window = None;
        self.poisoned = false;
        self.fed_lies = false;
        self.stale = false;
        Ok(())
    }

    fn state_difference(&self, left: &FakeState, right: &FakeState) -> Option<String> {
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
}

struct Rig {
    fake: Fake,
    session: DecodeSession,
    tokenizer: Tokenizer,
}

impl Rig {
    fn new(flaw: Flaw, ties: bool) -> Self {
        let mut fake = Fake::new(flaw, ties);
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

    /// The model and a setup over the prefilled session, as disjoint borrows.
    fn split(&mut self) -> (&mut Fake, Setup<'_>) {
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

/// `n` rows; the first is the pending token of the rig.
fn rows(n: usize) -> Vec<u32> {
    [3, 1, 4, 1, 5, 2, 6, 0].into_iter().take(n).collect()
}

fn expect_violation(result: super::d1_oracle::Verdict<Tally>, test: &str) -> D1Failure {
    let failure = result.expect_err("the flaw must be caught");
    assert!(failure.violation, "{failure}");
    assert!(failure.test.starts_with(test), "{failure}");
    failure
}

#[test]
fn a_sound_model_passes_every_check() {
    let mut rig = Rig::new(Flaw::Sound, false);
    for n in [2, 4, 8] {
        let tally = full_logits(&mut rig.fake, &rows(n)).expect("A");
        assert_eq!(tally.comparisons, n);
        later_rows_isolated(&mut rig.fake, &rows(n), &ALTERNATIVES).expect("C");
    }
    for drafts in [1, 3, 7] {
        let (fake, setup) = rig.split();
        let greedy = accepted_prefix(fake, &setup, drafts, &Sampler::greedy).expect("D greedy");
        let sampled = accepted_prefix(fake, &setup, drafts, &seeded).expect("D seeded");
        assert_eq!(greedy.comparisons, 2 * (drafts + 1));
        assert_eq!(sampled.comparisons, 2 * (drafts + 1));
    }
    for n in [2, 3, 4] {
        failure_recovery(&mut rig.fake, VOCAB, &rows(n)).expect("F");
    }
}

#[test]
fn the_full_logits_check_catches_a_row_that_depends_on_later_rows() {
    let mut rig = Rig::new(Flaw::LaterRowLeak, false);
    let failure = expect_violation(full_logits(&mut rig.fake, &rows(4)), "A");
    assert!(
        failure.detail.contains("row 0 of a 4-row window"),
        "{failure}"
    );
    assert!(failure.detail.contains("element 0"), "{failure}");
}

#[test]
fn a_row_count_keyed_path_hides_in_small_windows_and_isolation_sees_it() {
    let mut rig = Rig::new(Flaw::RowCountKeyed, false);
    // Windows below the key behave: the junk-row style checks at a fixed small m cannot see it.
    full_logits(&mut rig.fake, &rows(2)).expect("two rows are below the key");
    later_rows_isolated(&mut rig.fake, &rows(4), &ALTERNATIVES).expect("four rows too");
    // A five-row window differs from the one-row steps it must equal.
    expect_violation(full_logits(&mut rig.fake, &rows(6)), "A");
    // Isolation also sees it: the window truncated after row i is a different row count.
    let failure = expect_violation(
        later_rows_isolated(&mut rig.fake, &rows(6), &ALTERNATIVES),
        "C",
    );
    assert!(
        failure.detail.contains("the window ended after row"),
        "{failure}"
    );
}

#[test]
fn isolation_without_a_connected_earlier_row_is_refused_as_vacuous() {
    let mut rig = Rig::new(Flaw::Deaf, false);
    // Every comparison passes: nothing depends on anything. The control must not let that count.
    let failure = expect_violation(
        later_rows_isolated(&mut rig.fake, &rows(4), &ALTERNATIVES),
        "C",
    );
    assert!(failure.detail.contains("control"), "{failure}");
    assert!(failure.detail.contains("vacuous"), "{failure}");
}

#[test]
fn a_partial_accept_that_keeps_the_discarded_rows_is_caught() {
    let mut rig = Rig::new(Flaw::KeepsDiscardedRows, false);
    let (fake, setup) = rig.split();
    let failure = expect_violation(accepted_prefix(fake, &setup, 3, &Sampler::greedy), "D");
    assert!(failure.detail.contains("a=0 of 3"), "{failure}");
}

#[test]
fn a_rollback_that_forgets_the_recurrent_state_is_caught_where_the_kv_rows_match() {
    let mut rig = Rig::new(Flaw::KeepsRecurrent, false);
    let (fake, setup) = rig.split();
    let failure = expect_violation(accepted_prefix(fake, &setup, 3, &Sampler::greedy), "D");
    assert!(failure.detail.contains("recurrent state"), "{failure}");
    assert!(!failure.detail.contains("KV rows"), "{failure}");
}

#[test]
fn a_window_that_draws_a_value_too_many_moves_the_sampler_state() {
    let mut rig = Rig::new(Flaw::ExtraDraw, false);
    let (fake, setup) = rig.split();
    // Greedy never draws from the generator, so only a seeded sampler can show it.
    accepted_prefix(fake, &setup, 2, &Sampler::greedy).expect("greedy has no generator");
    let failure = expect_violation(accepted_prefix(fake, &setup, 2, &seeded), "D");
    assert!(
        failure.detail.contains("sampler generator state"),
        "{failure}"
    );
}

#[test]
fn exact_ties_break_to_the_lower_id_and_a_window_that_takes_the_higher_one_is_caught() {
    let mut sound = Rig::new(Flaw::Sound, true);
    let tally = ties_break_low(&mut sound.fake, VOCAB, &rows(4)).expect("E");
    assert_eq!(
        tally.exact_ties, 4,
        "every row ties: the rule was exercised"
    );
    let mut high = Rig::new(Flaw::TieHigh, true);
    let failure = expect_violation(ties_break_low(&mut high.fake, VOCAB, &rows(4)), "E");
    assert!(failure.detail.contains("window draw 6"), "{failure}");
    assert!(failure.detail.contains("lowest-id rule 2"), "{failure}");
    // Without a tie the high-picking flaw is invisible: the data must exercise the rule.
    let mut untied = Rig::new(Flaw::TieHigh, false);
    let tally = ties_break_low(&mut untied.fake, VOCAB, &rows(4)).expect("no ties");
    assert_eq!(tally.exact_ties, 0);
}

#[test]
fn a_fault_that_leaves_a_partial_state_and_keeps_answering_is_caught() {
    let mut rig = Rig::new(Flaw::SilentFault, false);
    let failure = expect_violation(failure_recovery(&mut rig.fake, VOCAB, &rows(4)), "F");
    assert!(failure.detail.contains("silent partial state"), "{failure}");
}

#[test]
fn a_failure_recovery_run_covers_feed_and_replay_faults() {
    // The sound model fails closed (poisoned) at every fault and recovers by restoring.
    let mut rig = Rig::new(Flaw::Sound, false);
    let tally = failure_recovery(&mut rig.fake, VOCAB, &rows(4)).expect("F");
    assert_eq!(
        tally.comparisons, 4,
        "Feed first, Feed last, Replay first, Replay last"
    );
    let two = failure_recovery(&mut rig.fake, VOCAB, &rows(2)).expect("F");
    assert_eq!(two.comparisons, 4);
}

#[test]
fn bit_comparison_separates_signed_zeros_and_keeps_equal_nans() {
    assert_eq!(first_bit_difference(&[1.0, 2.0], &[1.0, 2.0]), None);
    let signed = first_bit_difference(&[0.0], &[-0.0]).expect("0.0 and -0.0 differ in bits");
    assert!(signed.contains("element 0"), "{signed}");
    let nan = f32::from_bits(0x7fc0_0001);
    assert_eq!(first_bit_difference(&[nan], &[nan]), None);
    assert!(
        first_bit_difference(&[nan], &[f32::NAN]).is_some(),
        "other payload"
    );
    assert!(
        first_bit_difference(&[1.0], &[1.0, 2.0])
            .unwrap()
            .contains("lengths")
    );
    assert!(
        first_byte_difference(&[1, 2, 3], &[1, 9, 3])
            .unwrap()
            .contains("byte 1")
    );
    assert_eq!(first_byte_difference(&[], &[]), None);
}

#[test]
fn the_tie_rule_is_stated_twice_and_both_statements_agree_on_constructed_rows() {
    let rows: Vec<(Vec<f32>, u32, usize)> = vec![
        (vec![1.0, 5.0, 5.0, 2.0], 1, 2),
        (vec![5.0, 1.0, 5.0, 5.0], 0, 3),
        (vec![0.0, -0.0], 0, 2),
        (vec![-0.0, 0.0], 0, 2),
        (vec![3.0; 5], 0, 5),
        (vec![1.0, 2.0, 3.0, 4.0], 3, 1),
        (vec![4.0, 3.0, 2.0, 1.0], 0, 1),
    ];
    for (row, want, tied) in rows {
        let vocab = row.len();
        assert_eq!(host_greedy(&row, vocab), want, "{row:?}");
        assert_eq!(
            Sampler::greedy().sample(&row).expect("sample"),
            want,
            "{row:?}"
        );
        assert_eq!(tied_at_maximum(&row, vocab), tied, "{row:?}");
    }
    // An id past the vocabulary never wins, even above everything.
    assert_eq!(host_greedy(&[1.0, 2.0, 99.0], 2), 1);
}

#[test]
fn the_near_tie_of_the_fixture_breaks_to_the_lower_id_on_an_exact_tie() {
    let vocab = 248_077;
    let mut row = vec![0.0f32; vocab];
    row[near_tie::TIED_IDS[0] as usize] = near_tie::EXLLAMAV3_TIED_LOGIT;
    row[near_tie::TIED_IDS[1] as usize] = near_tie::EXLLAMAV3_TIED_LOGIT;
    assert_eq!(host_greedy(&row, vocab), near_tie::FORCED_TOKEN);
    assert_eq!(
        Sampler::greedy().sample(&row).expect("sample"),
        near_tie::FORCED_TOKEN
    );
    assert_eq!(tied_at_maximum(&row, vocab), 2);
    // One fp16 step above the tie (stock exllamav3's row 0 of the 4-row window) is a different
    // token: the window and the one-row step must therefore agree bit for bit, not nearly.
    row[near_tie::TIED_IDS[1] as usize] = 22.6875;
    assert_eq!(host_greedy(&row, vocab), near_tie::TIED_IDS[1]);
    assert_eq!(
        near_tie::ABSOLUTE_POSITION,
        near_tie::PROMPT_LEN + near_tie::FORCED_PREFIX_LEN
    );
}

#[test]
fn the_fixture_loader_refuses_a_file_whose_facts_do_not_hold() {
    assert!(near_tie::parse("not json").is_err());
    let empty = near_tie::parse("{}").expect_err("no ids");
    assert!(empty.contains("prompt_token_ids"), "{empty}");
    let short = r#"{"prefix":{"prompt_token_ids":[1,2],"forced_prefix_token_ids":[3],
        "forced_token_at_index":494},"source":{"absolute_position":3853},
        "paths":{"control_1":{"top2_ids":[494,1510]}}}"#;
    let wrong = near_tie::parse(short).expect_err("wrong lengths");
    assert!(wrong.contains("prompt length"), "{wrong}");
}

/// Run with `FIFI_D1_NEAR_TIE_FIXTURE` naming `d1-near-tie-exllamav3-4row-vs-1row.json`: the
/// file must hold the facts that the constants and the device test state.
#[test]
#[ignore = "needs FIFI_D1_NEAR_TIE_FIXTURE, the path of the fixture JSON"]
fn the_fixture_file_holds_the_facts_the_constants_state() {
    let path = std::env::var("FIFI_D1_NEAR_TIE_FIXTURE").expect("FIFI_D1_NEAR_TIE_FIXTURE");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let fixture = near_tie::parse(&text).expect("the fixture facts");
    assert_eq!(fixture.prompt.len(), near_tie::PROMPT_LEN);
    assert_eq!(fixture.forced_prefix.len(), near_tie::FORCED_PREFIX_LEN);
    assert!(fixture.prompt.iter().all(|&id| id < 248_320));
}

#[test]
fn a_model_that_reports_a_wrong_row_count_after_a_commit_is_caught_by_rows_fed() {
    let mut rig = Rig::new(Flaw::FedLies, false);
    let (fake, setup) = rig.split();
    let failure = expect_violation(accepted_prefix(fake, &setup, 3, &Sampler::greedy), "D");
    assert!(failure.detail.contains("rows fed"), "{failure}");
}

#[test]
fn a_stale_logits_row_after_a_partial_commit_is_caught_by_the_next_step_logits() {
    let mut rig = Rig::new(Flaw::StaleLogits, false);
    let (fake, setup) = rig.split();
    let failure = expect_violation(accepted_prefix(fake, &setup, 3, &Sampler::greedy), "D");
    assert!(failure.detail.contains("next-step logits"), "{failure}");
}

#[test]
fn a_one_row_step_that_breaks_ties_high_is_caught_by_the_one_row_leg_of_e() {
    // The window, the engine rule and the plain rule all say 2; only the one-row step says 6.
    let mut rig = Rig::new(Flaw::OneRowTieHigh, true);
    let failure = expect_violation(ties_break_low(&mut rig.fake, VOCAB, &rows(4)), "E");
    assert!(failure.detail.contains("one-row draw 6"), "{failure}");
    assert!(failure.detail.contains("window draw 2"), "{failure}");
}

/// A change to one field of an observation.
type Change = fn(&mut Observed<FakeState>);

fn observed() -> Observed<FakeState> {
    let state = FakeState {
        kv: vec![1, 2],
        recurrent: [0.5; 3],
    };
    Observed {
        tokens: vec![3, 4],
        history: vec![1, 2, 3, 4],
        pending: Some(4),
        rng: 9,
        remaining: Some(7),
        fed: 5,
        state: state.clone(),
        next_logits: vec![0.25, 0.5],
        state_after_next: state,
    }
}

/// The comparator names the first field that differs. The pending token and the history are
/// functions of the committed tokens in the real driver, so no model flaw moves them alone: each
/// is tested here on the comparator itself.
#[test]
fn the_comparator_sees_each_field_alone() {
    let fake = Fake::new(Flaw::Sound, false);
    assert_eq!(observed_difference(&fake, &observed(), &observed()), None);
    let cases: [(&str, Change); 9] = [
        ("committed tokens", |o| o.tokens.push(1)),
        ("session history", |o| o.history.push(1)),
        ("pending token", |o| o.pending = Some(5)),
        ("sampler generator state", |o| o.rng += 1),
        ("remaining output budget", |o| o.remaining = Some(6)),
        ("rows fed", |o| o.fed += 1),
        ("committed state", |o| o.state.kv.push(9)),
        ("next-step logits", |o| o.next_logits[1] = 0.75),
        ("state after the next step", |o| {
            o.state_after_next.recurrent[2] = 1.5
        }),
    ];
    for (field, change) in cases {
        let mut other = observed();
        change(&mut other);
        let found = observed_difference(&fake, &other, &observed()).expect(field);
        assert!(found.contains(field), "{field}: {found}");
    }
}

#[test]
fn a_tie_check_that_met_no_tie_says_so() {
    let tied = Tally {
        comparisons: 4,
        exact_ties: 4,
    };
    let untied = Tally {
        comparisons: 4,
        exact_ties: 0,
    };
    assert_eq!(describe_pass("E", &tied), "PASS comparisons=4 exact_ties=4");
    let line = describe_pass("E", &untied);
    assert!(line.starts_with("NOT-EXERCISED-TIE-RULE"), "{line}");
    assert!(!line.contains("PASS"), "a PASS grep must not count it: {line}");
    assert!(!describe_pass("E", &untied).starts_with("PASS comparisons"));
    // Other checks have no tie count to report.
    assert_eq!(
        describe_pass("A", &untied),
        "PASS comparisons=4 exact_ties=0"
    );
}
