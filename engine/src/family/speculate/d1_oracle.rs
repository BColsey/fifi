//! The D1 oracle: checks that a verify window equals one-row decode, written once against a model
//! interface so that the same code runs on a fake model on the host and on the real step on a
//! device.
//!
//! D1 is fifi's contract that a verify window is bitwise identical to sequential one-row decode
//! (the test letters of its design study are used here). The checks, by letter:
//!
//! * **A** [`full_logits`]: every row's complete logits row equals the logits row of a one-row
//!   step on the same prefix, bit for bit over the whole row.
//! * **C** [`later_rows_isolated`]: changing the rows after row `i`, or their number, leaves rows
//!   `0..=i` unchanged; a control shows that an earlier row does reach the later ones, so the
//!   check cannot pass because nothing is connected.
//! * **D** [`accepted_prefix`]: for every accepted-prefix length `a`, the whole observable state
//!   after the commit equals the state of a one-row loop through the same `a + 1` tokens: the
//!   model state (recurrent, convolution, KV and its length), the tokens, the pending token, the
//!   sampler's generator and the next step's logits.
//! * **E** [`ties_break_low`]: the window's draw on exact ties is the one-row draw, which takes
//!   the lowest id.
//! * **F** [`failure_recovery`] (not a letter of the spec): a fault injected mid-window or
//!   mid-replay leaves the pre-window state or a refusing model, never a silent partial state.
//!
//! Test B (per-layer taps) needs device-side debug taps and is not covered.
//!
//! **Status.** On the host these checks run against a fake model, including flawed copies of it:
//! each check has a test that it fails on the flaw it is meant to catch, because an oracle that
//! cannot fail proves nothing. Against the real Qwen3.6 step the same checks are device tests
//! (`d1_window_device_tests`) that have NOT run yet: whether the eager window is D1 on the real
//! pack is PENDING DEVICE QUALIFICATION.

use super::{
    accept::Accepted,
    driver::{Drafter, Proposal, Speculator, WindowModel},
    window::{Commit, Window},
};
use crate::{
    EngError,
    family::{
        executor::GenerationState,
        sampler::Sampler,
        session::{DecodeSession, Limits},
    },
    tokenizer::{IncrementalDecoder, Tokenizer},
};
use std::{fmt, ops::AddAssign};

const A: &str = "A full logits";
const C: &str = "C later-row isolation";
const D: &str = "D accepted prefix, commit and rollback";
const E: &str = "E ties";
const F: &str = "F failure recovery";

/// The job limit of the oracle's generations: far above any window.
const MAX_TOKENS: u32 = 4_096;

/// Where a fault is injected.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Fault {
    /// While the window feeds row `i`.
    Feed(usize),
    /// While the commit replays kept row `i`.
    Replay(usize),
}

/// What the checks need from a model, on top of the window interface.
pub(crate) trait D1Subject: WindowModel {
    /// Everything that commit and rollback must restore, compared bitwise.
    type State;

    /// Copies the committed state. The model keeps running from where it was.
    fn snapshot(&mut self) -> Result<Self::State, EngError>;

    /// Makes the model's state equal `state`, also when a failure left it poisoned.
    fn restore(&mut self, state: &Self::State) -> Result<(), EngError>;

    /// Describes the first difference between two states, or `None` when they are bitwise equal.
    fn state_difference(&self, left: &Self::State, right: &Self::State) -> Option<String>;

    /// The complete logits row of every row of the window that `run_window` just ran.
    fn window_rows(&self) -> Result<Vec<Vec<f32>>, EngError>;

    /// Arms one fault for the next window or commit. A test hook: the fault fires once.
    fn inject_fault(&mut self, fault: Fault);
}

/// A failed check.
#[derive(Debug)]
pub(crate) struct D1Failure {
    /// The check, by its letter.
    pub(crate) test: &'static str,
    pub(crate) detail: String,
    /// True when the contract was violated; false when the model returned an error where the
    /// check needed a result.
    pub(crate) violation: bool,
}

impl D1Failure {
    fn violation(test: &'static str, detail: impl Into<String>) -> Self {
        Self {
            test,
            detail: detail.into(),
            violation: true,
        }
    }
}

impl fmt::Display for D1Failure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let kind = if self.violation {
            "violation"
        } else {
            "engine error"
        };
        write!(f, "D1 {kind} in test {}: {}", self.test, self.detail)
    }
}

pub(crate) type Verdict<T> = Result<T, D1Failure>;

trait At<T> {
    fn at(self, test: &'static str) -> Verdict<T>;
}

impl<T> At<T> for Result<T, EngError> {
    fn at(self, test: &'static str) -> Verdict<T> {
        self.map_err(|error| D1Failure {
            test,
            detail: format!("{error:?}"),
            violation: false,
        })
    }
}

/// How much a check compared.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Tally {
    pub(crate) comparisons: usize,
    /// Rows whose maximum is shared by two or more ids (E only).
    pub(crate) exact_ties: usize,
}

impl AddAssign for Tally {
    fn add_assign(&mut self, other: Self) {
        self.comparisons += other.comparisons;
        self.exact_ties += other.exact_ties;
    }
}

/// The result line of a passed check. A tie check that met no exact tie did not exercise the tie
/// rule, and says so instead of a plain PASS. The line does not start with `PASS`, so a grep or a
/// prefix match for PASS never counts it; the test itself still succeeds.
pub(crate) fn describe_pass(test: &str, tally: &Tally) -> String {
    if test == "E" && tally.exact_ties == 0 {
        return format!(
            "NOT-EXERCISED-TIE-RULE comparisons={} exact_ties=0 (no row had an exact tie, \
             so only agreement of the draws was shown)",
            tally.comparisons
        );
    }
    format!(
        "PASS comparisons={} exact_ties={}",
        tally.comparisons, tally.exact_ties
    )
}

/// The first element whose bit pattern differs, or a length mismatch. Bitwise: `0.0` and `-0.0`
/// differ, and two NaNs with the same payload are equal.
pub(crate) fn first_bit_difference(left: &[f32], right: &[f32]) -> Option<String> {
    if left.len() != right.len() {
        return Some(format!("lengths {} and {}", left.len(), right.len()));
    }
    left.iter()
        .zip(right)
        .position(|(a, b)| a.to_bits() != b.to_bits())
        .map(|at| {
            format!(
                "element {at}: {:#010x} ({}) against {:#010x} ({})",
                left[at].to_bits(),
                left[at],
                right[at].to_bits(),
                right[at]
            )
        })
}

/// The first byte that differs, or a length mismatch.
pub(crate) fn first_byte_difference(left: &[u8], right: &[u8]) -> Option<String> {
    if left.len() != right.len() {
        return Some(format!("lengths {} and {}", left.len(), right.len()));
    }
    left.iter()
        .zip(right)
        .position(|(a, b)| a != b)
        .map(|at| format!("byte {at}: {:#04x} against {:#04x}", left[at], right[at]))
}

/// The id of the largest logit among the first `vocab`, the lowest id on an exact tie. Written
/// apart from `Sampler` so that the tie rule is checked against a second statement of it.
pub(crate) fn host_greedy(row: &[f32], vocab: usize) -> u32 {
    let mut best = 0;
    for (id, value) in row[..vocab].iter().enumerate() {
        if *value > row[best] {
            best = id;
        }
    }
    best as u32
}

/// How many of the first `vocab` ids share the largest value (`0.0` and `-0.0` are equal).
pub(crate) fn tied_at_maximum(row: &[f32], vocab: usize) -> usize {
    let top = row[host_greedy(row, vocab) as usize];
    row[..vocab].iter().filter(|value| **value == top).count()
}

/// Check A: the rows of one window against one-row steps on the same prefix, from the state the
/// subject is in now. The subject is back in that state when the check returns.
pub(crate) fn full_logits<S: D1Subject>(subject: &mut S, inputs: &[u32]) -> Verdict<Tally> {
    let base = subject.snapshot().at(A)?;
    let mut reference = Vec::with_capacity(inputs.len());
    for &token in inputs {
        reference.push(subject.logits(token).at(A)?);
    }
    subject.restore(&base).at(A)?;
    let rows = window_rows_of(subject, inputs, A)?;
    subject.restore(&base).at(A)?;
    if rows.len() != reference.len() {
        return Err(D1Failure::violation(
            A,
            format!(
                "the window kept {} rows for {} inputs",
                rows.len(),
                inputs.len()
            ),
        ));
    }
    let mut tally = Tally::default();
    for (row, (got, want)) in rows.iter().zip(&reference).enumerate() {
        tally.comparisons += 1;
        if let Some(difference) = first_bit_difference(got, want) {
            return Err(D1Failure::violation(
                A,
                format!(
                    "row {row} of a {}-row window differs from the one-row step: {difference}",
                    inputs.len()
                ),
            ));
        }
    }
    Ok(tally)
}

fn window_rows_of<S: D1Subject>(
    subject: &mut S,
    inputs: &[u32],
    test: &'static str,
) -> Verdict<Vec<Vec<f32>>> {
    subject.run_window(inputs).at(test)?;
    subject.window_rows().at(test)
}

/// The tokens that replace the rows after row `i`: all one token, and the original tail reversed.
fn later_variants(inputs: &[u32], i: usize, alternatives: &[u32]) -> Vec<Vec<u32>> {
    let mut variants = Vec::new();
    for &alternative in alternatives {
        let mut fill = inputs.to_vec();
        fill[i + 1..].fill(alternative);
        variants.push(fill);
    }
    let mut reversed = inputs.to_vec();
    reversed[i + 1..].reverse();
    variants.push(reversed);
    variants
}

/// Check C: rows `0..=i` do not move when the rows after `i` change or vanish. `alternatives`
/// are the replacement tokens (a stop token, the sentinel, the largest id, a repeat). The
/// control changes row 0 and requires row 1 to move.
pub(crate) fn later_rows_isolated<S: D1Subject>(
    subject: &mut S,
    inputs: &[u32],
    alternatives: &[u32],
) -> Verdict<Tally> {
    let base = subject.snapshot().at(C)?;
    let original = window_rows_of(subject, inputs, C)?;
    let mut tally = Tally::default();
    let mut compare = |label: String, rows: &[Vec<f32>], upto: usize| -> Verdict<()> {
        for (row, want) in original.iter().enumerate().take(upto + 1) {
            tally.comparisons += 1;
            let got = rows.get(row).map_or(&[][..], Vec::as_slice);
            if let Some(difference) = first_bit_difference(got, want) {
                return Err(D1Failure::violation(
                    C,
                    format!("row {row} moved when {label}: {difference}"),
                ));
            }
        }
        Ok(())
    };
    for i in 0..inputs.len().saturating_sub(1) {
        for variant in later_variants(inputs, i, alternatives) {
            subject.restore(&base).at(C)?;
            let rows = window_rows_of(subject, &variant, C)?;
            compare(
                format!("rows after {i} became {:?}", &variant[i + 1..]),
                &rows,
                i,
            )?;
        }
        subject.restore(&base).at(C)?;
        let rows = window_rows_of(subject, &inputs[..=i], C)?;
        compare(format!("the window ended after row {i}"), &rows, i)?;
    }
    control_reaches_later_rows(subject, &base, inputs, alternatives, &original)?;
    subject.restore(&base).at(C)?;
    Ok(tally)
}

/// An earlier row must change a later one, or the isolation above compared unconnected values.
fn control_reaches_later_rows<S: D1Subject>(
    subject: &mut S,
    base: &S::State,
    inputs: &[u32],
    alternatives: &[u32],
    original: &[Vec<f32>],
) -> Verdict<()> {
    if inputs.len() < 2 {
        return Ok(());
    }
    let Some(&other) = alternatives.iter().find(|&&token| token != inputs[0]) else {
        return Err(D1Failure::violation(C, "control: no alternative token"));
    };
    let mut changed = inputs.to_vec();
    changed[0] = other;
    subject.restore(base).at(C)?;
    let rows = window_rows_of(subject, &changed, C)?;
    if first_bit_difference(&rows[1], &original[1]).is_none() {
        return Err(D1Failure::violation(
            C,
            format!(
                "control: changing row 0 to {other} left row 1 unchanged; \
                 the rows are not connected, so the isolation result is vacuous"
            ),
        ));
    }
    Ok(())
}

/// A synthetic tokenizer: ids 0 to 7 are the letters a to h. For fake models.
pub(crate) const FAKE_TOKENIZER_JSON: &str = r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},"post_processor":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},"decoder":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},"model":{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":false,"vocab":{"a":0,"b":1,"c":2,"d":3,"e":4,"f":5,"g":6,"h":7},"merges":[]}}"#;

/// What the driver-level checks need around the subject: the prefilled session (the subject is in
/// the state that goes with it), the tokenizer for the generation state, and the job limits.
pub(crate) struct Setup<'a> {
    pub(crate) session: &'a DecodeSession,
    pub(crate) tokenizer: &'a Tokenizer,
    pub(crate) limits: Limits<'a>,
}

impl Setup<'_> {
    fn generation(&self, sampler: Sampler) -> GenerationState {
        GenerationState::with_sampler(IncrementalDecoder::new(self.tokenizer), sampler)
            .with_max_tokens(MAX_TOKENS)
    }
}

/// Drafts that are accepted exactly `accepted` deep: the first `accepted` drafts are the draws a
/// one-row run makes, the next one is a token that differs from the draw, and the rest (the
/// later rows) are other tokens that `tail` chooses. One window only.
struct ExactAccept {
    reference: Vec<u32>,
    prompt_len: usize,
    drafts: usize,
    accepted: usize,
    tail: u32,
    vocab: u32,
}

impl ExactAccept {
    fn draft(&self, at: usize, emitted: usize) -> u32 {
        let draw = self.reference[emitted + at];
        if at < self.accepted {
            return draw;
        }
        // `1 + tail` is below the vocabulary, so `wrong` never equals the draw.
        let offset = if at == self.accepted {
            1 + self.tail
        } else {
            3 + self.tail * 5 + at as u32
        };
        (draw + offset) % self.vocab
    }
}

impl Drafter for ExactAccept {
    fn propose(&mut self, history: &[u32], room: usize) -> Result<Proposal, EngError> {
        let emitted = history.len() - self.prompt_len;
        let count = self.drafts.min(room);
        Ok(Proposal {
            tokens: (0..count).map(|at| self.draft(at, emitted)).collect(),
            scores: Vec::new(),
        })
    }

    fn observe(&mut self, _window: &Window, _accepted: &Accepted) -> Result<(), EngError> {
        Ok(())
    }
}

/// Everything observable after `steps` committed tokens, plus the next step. Crate-visible so
/// that the comparator can be tested field by field.
pub(crate) struct Observed<St> {
    pub(crate) tokens: Vec<u32>,
    pub(crate) history: Vec<u32>,
    pub(crate) pending: Option<u32>,
    pub(crate) rng: u64,
    pub(crate) remaining: Option<u32>,
    pub(crate) fed: usize,
    pub(crate) state: St,
    pub(crate) next_logits: Vec<f32>,
    pub(crate) state_after_next: St,
}

fn observe<S: D1Subject>(
    subject: &mut S,
    session: &DecodeSession,
    generation: &GenerationState,
    tokens: Vec<u32>,
) -> Verdict<Observed<S::State>> {
    let state = subject.snapshot().at(D)?;
    let fed = subject.fed();
    let pending = session.pending();
    let next =
        pending.ok_or_else(|| D1Failure::violation(D, "no pending token after the commit"))?;
    let next_logits = subject.logits(next).at(D)?;
    Ok(Observed {
        tokens,
        history: session.history().to_vec(),
        pending,
        rng: generation.sampler.rng_state(),
        remaining: generation.remaining_tokens(),
        fed,
        state,
        next_logits,
        state_after_next: subject.snapshot().at(D)?,
    })
}

/// `steps` one-row decodes from the subject's current state.
fn one_row_tokens<S: D1Subject>(
    subject: &mut S,
    setup: &Setup<'_>,
    make_sampler: &dyn Fn() -> Sampler,
    steps: usize,
) -> Verdict<(Vec<u32>, DecodeSession, GenerationState)> {
    let mut session = setup.session.clone();
    let mut generation = setup.generation(make_sampler());
    let mut tokens = Vec::new();
    for _ in 0..steps {
        let batch = session
            .decode(subject, &mut generation, &setup.limits)
            .at(D)?;
        tokens.extend(batch.tokens);
    }
    Ok((tokens, session, generation))
}

/// The first difference between a one-row run and a window, by field.
pub(crate) fn observed_difference<S: D1Subject>(
    subject: &S,
    window: &Observed<S::State>,
    one_row: &Observed<S::State>,
) -> Option<String> {
    let field = |name: &str, same: bool| (!same).then(|| name.to_owned());
    field("committed tokens", window.tokens == one_row.tokens)
        .or_else(|| field("session history", window.history == one_row.history))
        .or_else(|| field("pending token", window.pending == one_row.pending))
        .or_else(|| field("sampler generator state", window.rng == one_row.rng))
        .or_else(|| {
            field(
                "remaining output budget",
                window.remaining == one_row.remaining,
            )
        })
        .or_else(|| field("rows fed", window.fed == one_row.fed))
        .or_else(|| {
            subject
                .state_difference(&window.state, &one_row.state)
                .map(|d| format!("committed state: {d}"))
        })
        .or_else(|| {
            first_bit_difference(&window.next_logits, &one_row.next_logits)
                .map(|d| format!("next-step logits: {d}"))
        })
        .or_else(|| {
            subject
                .state_difference(&window.state_after_next, &one_row.state_after_next)
                .map(|d| format!("state after the next step: {d}"))
        })
}

/// Check D with the real driver: for every `a` in `0..=drafts`, one speculative step whose
/// window of `drafts + 1` rows accepts exactly `a` drafts must leave everything a one-row loop of
/// `a + 1` tokens leaves. Two different tails of later rows per `a` (they must not matter). The
/// subject is in the prefilled state of `setup.session` on entry and on return.
pub(crate) fn accepted_prefix<S: D1Subject>(
    subject: &mut S,
    setup: &Setup<'_>,
    drafts: usize,
    make_sampler: &dyn Fn() -> Sampler,
) -> Verdict<Tally> {
    let base = subject.snapshot().at(D)?;
    let (reference, ..) = one_row_tokens(subject, setup, make_sampler, drafts + 1)?;
    subject.restore(&base).at(D)?;
    let mut tally = Tally::default();
    for accepted in 0..=drafts {
        for tail in 0..2 {
            let (tokens, session, generation) =
                one_row_tokens(subject, setup, make_sampler, accepted + 1)?;
            let one_row = observe(subject, &session, &generation, tokens)?;
            subject.restore(&base).at(D)?;

            let drafter = ExactAccept {
                reference: reference.clone(),
                prompt_len: setup.session.history().len(),
                drafts,
                accepted,
                tail,
                vocab: setup.limits.sampler_vocab as u32,
            };
            let mut speculator = Speculator::new(drafter, drafts);
            let mut session = setup.session.clone();
            let mut generation = setup.generation(make_sampler());
            let batch = speculator
                .decode(&mut session, subject, &mut generation, &setup.limits)
                .at(D)?;
            let ran = format!(
                "windows=1 rows_fed={} drafted={drafts} accepted={accepted} emitted={}",
                drafts + 1,
                accepted + 1
            );
            if !speculator.stats.line().contains(&ran) {
                return Err(D1Failure::violation(
                    D,
                    format!(
                        "the scripted window did not run as planned ({ran}): {}",
                        speculator.stats.line()
                    ),
                ));
            }
            let window = observe(subject, &session, &generation, batch.tokens)?;
            subject.restore(&base).at(D)?;
            tally.comparisons += 1;
            if let Some(difference) = observed_difference(subject, &window, &one_row) {
                return Err(D1Failure::violation(
                    D,
                    format!(
                        "a={accepted} of {drafts} drafts (tail variant {tail}): {difference} \
                         differs from the one-row loop"
                    ),
                ));
            }
        }
    }
    Ok(tally)
}

/// Check E: for every row of a window, the greedy draw of the window, the greedy draw of a
/// one-row step, the engine's sampler and the plain statement of the rule all give one id, the
/// lowest of the tied ones. `tally.exact_ties` counts the rows that had a tie, so a caller can
/// see whether the data exercised the rule.
pub(crate) fn ties_break_low<S: D1Subject>(
    subject: &mut S,
    vocab: usize,
    inputs: &[u32],
) -> Verdict<Tally> {
    let base = subject.snapshot().at(E)?;
    let mut one_row = Vec::with_capacity(inputs.len());
    for &token in inputs {
        let mut greedy = Sampler::greedy();
        one_row.push(subject.next_token(token, &mut greedy, vocab).at(E)?);
    }
    subject.restore(&base).at(E)?;
    let rows = window_rows_of(subject, inputs, E)?;
    let mut tally = Tally::default();
    for (row, logits) in rows.iter().enumerate() {
        let window = subject.draw(row, &mut Sampler::greedy(), vocab).at(E)?;
        let engine = Sampler::greedy().sample(&logits[..vocab]).at(E)?;
        let statement = host_greedy(logits, vocab);
        tally.comparisons += 1;
        tally.exact_ties += usize::from(tied_at_maximum(logits, vocab) > 1);
        if [engine, statement, one_row[row]] != [window; 3] {
            return Err(D1Failure::violation(
                E,
                format!(
                    "row {row}: window draw {window}, one-row draw {}, engine rule {engine}, \
                     lowest-id rule {statement}",
                    one_row[row]
                ),
            ));
        }
    }
    subject.restore(&base).at(E)?;
    Ok(tally)
}

/// The faults of check F for a window of `rows` rows.
fn faults_for(rows: usize) -> Vec<(Fault, Commit)> {
    let kept = rows.saturating_sub(1).max(1);
    let commit = Commit {
        committed_rows: 0,
        kept_rows: kept,
        discarded_rows: rows - kept,
        state_slot: kept - 1,
    };
    let mut faults = vec![(Fault::Feed(0), commit), (Fault::Feed(rows - 1), commit)];
    if commit.discarded_rows > 0 {
        faults.push((Fault::Replay(0), commit));
        faults.push((Fault::Replay(kept - 1), commit));
    }
    faults
}

/// Check F: each fault ends the call with an error, leaves no window to draw from or commit, and
/// leaves either the pre-window state or a model that refuses to go on; restoring the pre-window
/// state then lets the same window run again with the same rows.
///
/// **What this models and what it does not.** A fault is injected at a feed or replay row and
/// poisons the state the way a failed step does, so the "refuses to go on" outcome is what the
/// check sees on the real step. A real `commit_window` can also fail before any step runs, in
/// `complete_prefix_save` (`step.rs:1326`), `check_import` (`state/export.rs:196`) or
/// `resident_kv` (`state/export.rs:197`): those refuse before `begin_import`, so the state is NOT
/// poisoned. It stays a valid state at the end of the window (the rows fed), the window is
/// already taken (`eager_window.rs`, `commit_window`), and a later step would run from that state
/// without error. No check here can show otherwise, and none claims to. This predates the branch
/// and is not changed here (production behaviour is on hold). The safety net is the worker: an
/// error from the executor ends in `reset_state` (`sched/worker.rs:1049`, called at `:832`, `:873`
/// and `:897`; the policy is written at `qwen36/cuda/executor.rs:17`), which resets the session
/// and the model before the next job.
pub(crate) fn failure_recovery<S: D1Subject>(
    subject: &mut S,
    vocab: usize,
    inputs: &[u32],
) -> Verdict<Tally> {
    let base = subject.snapshot().at(F)?;
    let clean = window_rows_of(subject, inputs, F)?;
    subject.restore(&base).at(F)?;
    let mut tally = Tally::default();
    for (fault, commit) in faults_for(inputs.len()) {
        subject.restore(&base).at(F)?;
        subject.inject_fault(fault);
        let ran = subject.run_window(inputs).and_then(|()| {
            subject.commit_window(&commit)?;
            Ok(())
        });
        if ran.is_ok() {
            return Err(D1Failure::violation(
                F,
                format!("{fault:?} did not fail the call"),
            ));
        }
        if subject.draw(0, &mut Sampler::greedy(), vocab).is_ok() {
            return Err(D1Failure::violation(
                F,
                format!("{fault:?}: a draw came from a dead window"),
            ));
        }
        if subject.commit_window(&commit).is_ok() {
            return Err(D1Failure::violation(
                F,
                format!("{fault:?}: a dead window committed"),
            ));
        }
        require_closed_or_rolled_back(subject, &base, inputs[0], fault)?;
        subject.restore(&base).at(F)?;
        let rows = window_rows_of(subject, inputs, F)?;
        subject.restore(&base).at(F)?;
        tally.comparisons += 1;
        for (row, (got, want)) in rows.iter().zip(&clean).enumerate() {
            if let Some(difference) = first_bit_difference(got, want) {
                return Err(D1Failure::violation(
                    F,
                    format!("{fault:?}: after recovery row {row} moved: {difference}"),
                ));
            }
        }
    }
    Ok(tally)
}

/// After a fault the state is the pre-window state, or the model refuses both a snapshot and a
/// step. A state that differs and still answers is a silent partial state.
fn require_closed_or_rolled_back<S: D1Subject>(
    subject: &mut S,
    base: &S::State,
    token: u32,
    fault: Fault,
) -> Verdict<()> {
    match subject.snapshot() {
        Ok(after) if subject.state_difference(&after, base).is_none() => Ok(()),
        Ok(_) | Err(_) => {
            if subject.logits(token).is_ok() {
                Err(D1Failure::violation(
                    F,
                    format!(
                        "{fault:?} left a state that is not the pre-window state and still \
                         answers a step: a silent partial state"
                    ),
                ))
            } else {
                Ok(())
            }
        }
    }
}

/// The fixture of the exllamav3 near tie (`d1-near-tie-exllamav3-4row-vs-1row.json`): the facts
/// the D1 design states about it, and the loader that checks a fixture file against them.
pub(crate) mod near_tie {
    use sha2::{Digest, Sha256};

    /// Prompt ids, forced-prefix ids, and the position of the tie.
    pub(crate) const PROMPT_LEN: usize = 3_839;
    pub(crate) const FORCED_PREFIX_LEN: usize = 14;
    pub(crate) const ABSOLUTE_POSITION: usize = 3_853;
    /// The token the one-row runs commit at the position: the lower id of the exact tie.
    pub(crate) const FORCED_TOKEN: u32 = 494;
    pub(crate) const TIED_IDS: [u32; 2] = [494, 1_510];
    /// The tied logit as stock exllamav3 computed it. fifi's own logits need not tie here: the
    /// fixture says only that a path that disagrees with its own one-row path is row-keyed.
    pub(crate) const EXLLAMAV3_TIED_LOGIT: f32 = 22.671_875;
    pub(crate) const SENTINEL_ID: u32 = 248_055;
    /// SHA-256 of the little-endian int32 packing of the prompt ids.
    pub(crate) const PROMPT_SHA256: &str =
        "838b77beb63c0117ca4b57bc3645c19a6138a107fc4b1ce48aef5864d4170638";
    /// SHA-256 of the same packing of the prompt ids followed by the forced prefix.
    pub(crate) const CONDITIONING_SHA256: &str =
        "6b9aad0b3d0e00e40493bdc375da56f3300749715dc65750e28e622c52222f81";

    /// The ids of the fixture.
    #[derive(Debug, PartialEq, Eq)]
    pub(crate) struct NearTie {
        pub(crate) prompt: Vec<u32>,
        pub(crate) forced_prefix: Vec<u32>,
    }

    fn pack_sha256(ids: &[u32]) -> String {
        let bytes: Vec<u8> = ids.iter().flat_map(|id| id.to_le_bytes()).collect();
        format!("{:x}", Sha256::digest(bytes))
    }

    fn ids(value: &serde_json::Value, key: &str) -> Result<Vec<u32>, String> {
        value[key]
            .as_array()
            .ok_or_else(|| format!("{key}: not an array"))?
            .iter()
            .map(|id| {
                id.as_u64()
                    .and_then(|id| u32::try_from(id).ok())
                    .ok_or_else(|| format!("{key}: not a u32 id"))
            })
            .collect()
    }

    /// Parses the fixture text and checks it against the constants above, including both
    /// content hashes. The error names the first fact that does not hold.
    pub(crate) fn parse(text: &str) -> Result<NearTie, String> {
        let json: serde_json::Value = serde_json::from_str(text).map_err(|e| e.to_string())?;
        let prompt = ids(&json["prefix"], "prompt_token_ids")?;
        let forced_prefix = ids(&json["prefix"], "forced_prefix_token_ids")?;
        let forced = json["prefix"]["forced_token_at_index"].as_u64();
        let position = json["source"]["absolute_position"].as_u64();
        let top2 = &json["paths"]["control_1"]["top2_ids"];
        let checks = [
            ("prompt length", prompt.len() == PROMPT_LEN),
            (
                "forced prefix length",
                forced_prefix.len() == FORCED_PREFIX_LEN,
            ),
            ("forced token", forced == Some(u64::from(FORCED_TOKEN))),
            (
                "absolute position",
                position == Some(ABSOLUTE_POSITION as u64),
            ),
            (
                "position is prompt plus prefix",
                prompt.len() + forced_prefix.len() == ABSOLUTE_POSITION,
            ),
            ("tied ids", top2 == &serde_json::json!(TIED_IDS)),
            ("prompt sha256", pack_sha256(&prompt) == PROMPT_SHA256),
            (
                "conditioning sha256",
                pack_sha256(&[prompt.as_slice(), forced_prefix.as_slice()].concat())
                    == CONDITIONING_SHA256,
            ),
        ];
        match checks.iter().find(|(_, holds)| !holds) {
            Some((name, _)) => Err(format!("fixture fact does not hold: {name}")),
            None => Ok(NearTie {
                prompt,
                forced_prefix,
            }),
        }
    }
}
