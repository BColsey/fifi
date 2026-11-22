//! Host tests of the window driver over a small scripted family. The family's rows are a pure
//! function of the pending token, so a row's tokens cannot depend on the window it rode in, and
//! every test compares a window against solo runs.

use super::{WindowRows, decode_window};
use crate::EngError;
use crate::family::executor::{ExecutorOps, GenerationState};
use crate::family::sampler::{Sampler, SamplingParams};
use crate::family::session::{DecodeSession, Limits, TokenStepper};
use crate::tokenizer::{DecodeBatch, DecodeFinish, IncrementalDecoder, Tokenizer};
use std::cell::Cell;
use std::rc::Rc;
use std::sync::atomic::Ordering;

type Result<T> = std::result::Result<T, EngError>;

const VOCAB: usize = 16;
const EOS: u32 = 15;

fn tokenizer() -> Tokenizer {
    let vocab: Vec<String> = (0..VOCAB)
        .map(|id| format!("\"{}\":{id}", char::from(b'a' + id as u8)))
        .collect();
    let json = format!(
        r#"{{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":{{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false}},"post_processor":{{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false}},"decoder":{{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false}},"model":{{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":false,"vocab":{{{}}},"merges":[]}}}}"#,
        vocab.join(",")
    );
    Tokenizer::from_bytes(json.as_bytes()).expect("tokenizer")
}

fn generation(limit: u32) -> GenerationState {
    GenerationState::new(IncrementalDecoder::new(&tokenizer())).with_max_tokens(limit)
}

fn sampled_generation(limit: u32, seed: u64) -> GenerationState {
    let params = SamplingParams::new(1.0, 1.0, 0).expect("params");
    GenerationState::with_sampler(
        IncrementalDecoder::new(&tokenizer()),
        Sampler::new(params, seed),
    )
    .with_max_tokens(limit)
}

/// The next token of a row: a function of its pending token alone. It never lands on `EOS`
/// from a token below 8, so a script that wants an EOS starts from a token that maps to it.
fn next_of(token: u32) -> u32 {
    (token * 5 + 3) % VOCAB as u32
}

fn logits_of(token: u32) -> Vec<f32> {
    let mut row = vec![0.0; VOCAB];
    row[next_of(token) as usize] = 4.0;
    row
}

#[derive(Default)]
struct RowSession {
    core: DecodeSession,
    fed: usize,
}

struct RowStepper<'a> {
    fed: &'a mut usize,
}

impl TokenStepper for RowStepper<'_> {
    const OPERATION: &'static str = "rows";

    fn feed(&mut self, _token: u32) -> Result<()> {
        *self.fed += 1;
        Ok(())
    }

    fn logits(&mut self, token: u32) -> Result<Vec<f32>> {
        *self.fed += 1;
        Ok(logits_of(token))
    }

    fn fed(&self) -> usize {
        *self.fed
    }

    fn reset(&mut self) -> Result<()> {
        *self.fed = 0;
        Ok(())
    }
}

struct Permit(Rc<Cell<u64>>, u64);

impl Drop for Permit {
    fn drop(&mut self) {
        self.0.set(self.0.get() - self.1);
    }
}

struct Rows {
    max: usize,
    eos: Vec<u32>,
    growth: u64,
    cap: u64,
    used: Rc<Cell<u64>>,
    step_calls: usize,
    stepped_rows: Vec<usize>,
    fail_after_step: Option<EngError>,
    short_result: bool,
    /// The index among the stepped rows whose logits come back too narrow to sample.
    narrow_row: Option<usize>,
    /// Makes `row_growth` fail for a session whose pending token is this one.
    growth_error_on: Option<u32>,
    /// Makes `reserve_growth` fail with a refusal that is not a memory denial.
    reserve_error: bool,
    /// The family draws the next token of every row itself, as a device sampler would.
    device_draw: bool,
    /// Calls of `step_rows_drawn`.
    drawn_calls: usize,
}

impl Rows {
    fn new(max: usize) -> Self {
        Self {
            max,
            eos: vec![EOS],
            growth: 0,
            cap: u64::MAX,
            used: Rc::new(Cell::new(0)),
            step_calls: 0,
            stepped_rows: Vec::new(),
            fail_after_step: None,
            short_result: false,
            narrow_row: None,
            growth_error_on: None,
            reserve_error: false,
            device_draw: false,
            drawn_calls: 0,
        }
    }
}

impl ExecutorOps for Rows {
    type Session = RowSession;

    fn max_seqs(&self) -> usize {
        self.max
    }

    fn history<'s>(&self, st: &'s RowSession) -> &'s [u32] {
        st.core.history()
    }

    fn reset(&mut self, st: &mut RowSession) -> Result<()> {
        st.core.reset(&mut RowStepper { fed: &mut st.fed })
    }

    fn prefill(&mut self, st: &mut RowSession, toks: &[u32]) -> Result<()> {
        st.core
            .prefill(&mut RowStepper { fed: &mut st.fed }, toks, usize::MAX)
    }

    fn decode(
        &mut self,
        st: &mut RowSession,
        generation: &mut GenerationState,
    ) -> Result<DecodeBatch> {
        let limits = Limits {
            eos: &self.eos,
            sampler_vocab: VOCAB,
            context_limit: usize::MAX,
        };
        st.core
            .decode(&mut RowStepper { fed: &mut st.fed }, generation, &limits)
    }

    fn boundary(&mut self, _st: &mut RowSession) {}
}

impl WindowRows for Rows {
    type Stepper<'a> = RowStepper<'a>;
    type Permit = Permit;
    const OPERATION: &'static str = "rows";

    fn split<'a>(
        &'a mut self,
        session: &'a mut RowSession,
    ) -> (&'a mut DecodeSession, RowStepper<'a>, Limits<'a>) {
        (
            &mut session.core,
            RowStepper {
                fed: &mut session.fed,
            },
            Limits {
                eos: &self.eos,
                sampler_vocab: VOCAB,
                context_limit: usize::MAX,
            },
        )
    }

    fn check_row(&self, session: &RowSession, token: u32) -> Result<()> {
        let _ = session;
        if token as usize >= VOCAB {
            return Err(EngError::BadRequest(
                "token is outside the embedding table".into(),
            ));
        }
        Ok(())
    }

    fn row_growth(&self, session: &RowSession) -> Result<u64> {
        if self.growth_error_on.is_some() && session.core.pending() == self.growth_error_on {
            return Err(EngError::BadRequest("the row has no growth plan".into()));
        }
        Ok(self.growth)
    }

    fn reserve_growth(&self, bytes: u64) -> Result<Permit> {
        if self.reserve_error {
            return Err(EngError::BadRequest("the growth ledger is closed".into()));
        }
        if self.used.get() + bytes > self.cap {
            return Err(EngError::MemoryPlan("the rows are out of growth".into()));
        }
        self.used.set(self.used.get() + bytes);
        Ok(Permit(Rc::clone(&self.used), bytes))
    }

    fn draws_on_device(&self) -> bool {
        self.device_draw
    }

    fn step_rows_drawn(
        &mut self,
        sessions: &mut [&mut RowSession],
        tokens: &[u32],
        samplers: &[&crate::family::sampler::Sampler],
    ) -> Result<Vec<u32>> {
        assert_eq!(samplers.len(), sessions.len(), "one sampler per row");
        self.drawn_calls += 1;
        self.step_calls += 1;
        self.stepped_rows.push(sessions.len());
        for session in sessions.iter_mut() {
            session.fed += 1;
        }
        Ok(tokens.iter().map(|&token| next_of(token)).collect())
    }

    fn step_rows(
        &mut self,
        sessions: &mut [&mut RowSession],
        tokens: &[u32],
    ) -> Result<Vec<Vec<f32>>> {
        self.step_calls += 1;
        self.stepped_rows.push(sessions.len());
        let mut rows = Vec::new();
        for (index, (session, &token)) in sessions.iter_mut().zip(tokens).enumerate() {
            session.fed += 1;
            let mut row = logits_of(token);
            if self.narrow_row == Some(index) {
                row.truncate(VOCAB - 1);
            }
            rows.push(row);
        }
        if let Some(error) = self.fail_after_step.take() {
            return Err(error);
        }
        if self.short_result {
            rows.pop();
        }
        Ok(rows)
    }
}

fn session(family: &mut Rows, prompt: &[u32]) -> RowSession {
    let mut session = RowSession::default();
    family.prefill(&mut session, prompt).expect("prefill");
    session
}

/// Runs one prompt alone to its finish and returns the tokens.
fn solo(prompt: &[u32], limit: u32) -> Vec<u32> {
    let mut family = Rows::new(1);
    let mut st = session(&mut family, prompt);
    let mut generation = generation(limit);
    let mut tokens = Vec::new();
    loop {
        let batch = family
            .decode(&mut st, &mut generation)
            .expect("solo decode");
        tokens.extend(&batch.tokens);
        if batch.finish.is_some() {
            return tokens;
        }
    }
}

/// Runs the prompts together, one window per step over the rows that are still live, and
/// returns each row's tokens.
fn windowed(prompts: &[&[u32]], limit: u32) -> Vec<Vec<u32>> {
    windowed_with(Rows::new(8), prompts, limit)
}

fn windowed_with(mut family: Rows, prompts: &[&[u32]], limit: u32) -> Vec<Vec<u32>> {
    let mut sessions: Vec<RowSession> = prompts.iter().map(|p| session(&mut family, p)).collect();
    let mut generations: Vec<GenerationState> = prompts.iter().map(|_| generation(limit)).collect();
    let mut tokens = vec![Vec::new(); prompts.len()];
    let mut live: Vec<usize> = (0..prompts.len()).collect();
    while !live.is_empty() {
        let batches = {
            let mut rows: Vec<&mut RowSession> = Vec::new();
            let mut gens: Vec<&mut GenerationState> = Vec::new();
            let mut sessions_iter = sessions.iter_mut().enumerate();
            let mut generations_iter = generations.iter_mut();
            for (index, session) in &mut sessions_iter {
                let generation = generations_iter.next().expect("generation");
                if live.contains(&index) {
                    rows.push(session);
                    gens.push(generation);
                }
            }
            decode_window(&mut family, &mut rows, &mut gens).expect("window")
        };
        let mut finished = Vec::new();
        for (&index, batch) in live.iter().zip(batches) {
            let batch = batch.expect("row");
            tokens[index].extend(&batch.tokens);
            if batch.finish.is_some() {
                finished.push(index);
            }
        }
        live.retain(|index| !finished.contains(index));
    }
    tokens
}

#[test]
fn every_row_of_a_window_equals_its_solo_run() {
    let prompts: [&[u32]; 4] = [&[1, 2, 3], &[4], &[7, 9, 11, 2], &[0, 5]];
    let together = windowed(&prompts, 12);
    for (prompt, tokens) in prompts.iter().zip(&together) {
        assert_eq!(tokens, &solo(prompt, 12), "prompt {prompt:?}");
        assert!(!tokens.is_empty());
    }
}

#[test]
fn a_family_that_draws_on_the_device_runs_the_same_windows() {
    let prompts: [&[u32]; 3] = [&[1, 2, 3], &[4], &[7, 9, 11, 2]];
    let mut family = Rows::new(8);
    family.device_draw = true;
    let drawn = windowed_with(family, &prompts, 12);
    assert_eq!(drawn, windowed(&prompts, 12));
}

#[test]
fn a_window_with_a_recording_request_takes_the_logits_rows() {
    let mut family = Rows::new(8);
    family.device_draw = true;
    let mut a = session(&mut family, &[1, 2, 3]);
    let mut b = session(&mut family, &[4]);
    let mut first = sampled_generation(4, 7);
    let mut second = sampled_generation(4, 8);
    let _capture = second.sampler.arm_capture(8);
    let batches = decode_window(
        &mut family,
        &mut [&mut a, &mut b],
        &mut [&mut first, &mut second],
    )
    .expect("window");
    assert_eq!(batches.len(), 2);
    assert_eq!(family.drawn_calls, 0, "the host walk records the draw");
    assert_eq!(family.step_calls, 1);
    let mut unarmed = Rows::new(8);
    unarmed.device_draw = true;
    let mut c = session(&mut unarmed, &[1, 2, 3]);
    let mut d = session(&mut unarmed, &[4]);
    let (mut third, mut fourth) = (sampled_generation(4, 7), sampled_generation(4, 8));
    decode_window(
        &mut unarmed,
        &mut [&mut c, &mut d],
        &mut [&mut third, &mut fourth],
    )
    .expect("window");
    assert_eq!(unarmed.drawn_calls, 1);
}

#[test]
fn a_row_does_not_depend_on_which_rows_share_its_window() {
    let a: &[u32] = &[1, 2, 3];
    let b: &[u32] = &[4];
    let c: &[u32] = &[7, 9];
    let alone = windowed(&[a, b], 10);
    let with_third = windowed(&[a, b, c], 10);
    let reordered = windowed(&[c, b, a], 10);
    assert_eq!(alone[0], with_third[0]);
    assert_eq!(alone[1], with_third[1]);
    assert_eq!(reordered[2], alone[0]);
    assert_eq!(reordered[1], alone[1]);
}

#[test]
fn a_row_that_finishes_leaves_the_others_unchanged() {
    // A limit of two ends the first window sequence early; the longer run covers the same prefix.
    let short = windowed(&[&[2], &[7]], 2);
    let long = windowed(&[&[2], &[7]], 9);
    assert_eq!(short[0].len(), 2);
    assert!(long[1].len() > 2);
    assert_eq!(&long[1][..2], &short[1][..]);
    assert_eq!(long[1], solo(&[7], 9));
}

#[test]
fn an_eos_row_ends_alone() {
    // Token 4 maps to 23 % 16 = 7, 7 maps to 38 % 16 = 6, 6 to 33 % 16 = 1, 1 to 8, 8 to 43 % 16
    // = 11, 11 to 58 % 16 = 10, 10 to 53 % 16 = 5, 5 to 28 % 16 = 12, 12 to 63 % 16 = 15 = EOS.
    let rows = windowed(&[&[4], &[0]], 30);
    assert_eq!(rows[0].last(), Some(&EOS), "{:?}", rows[0]);
    assert_eq!(rows[0], solo(&[4], 30));
    assert_eq!(rows[1], solo(&[0], 30));
}

#[test]
fn each_row_draws_from_its_own_sampler() {
    let prompts: [&[u32]; 2] = [&[1, 2], &[3, 4]];
    let seeds = [11, 12];
    let mut family = Rows::new(8);
    let mut sessions: Vec<RowSession> = prompts.iter().map(|p| session(&mut family, p)).collect();
    let mut generations: Vec<GenerationState> = seeds
        .iter()
        .map(|&seed| sampled_generation(8, seed))
        .collect();
    let mut together: Vec<Vec<u32>> = vec![Vec::new(); 2];
    for _ in 0..8 {
        let [s0, s1] = sessions.as_mut_slice() else {
            unreachable!()
        };
        let [g0, g1] = generations.as_mut_slice() else {
            unreachable!()
        };
        let batches = decode_window(&mut family, &mut [s0, s1], &mut [g0, g1]).expect("window");
        for (row, batch) in batches.into_iter().enumerate() {
            together[row].extend(batch.expect("row").tokens);
        }
        if sessions.iter().any(|s| s.core.is_finished()) {
            break;
        }
    }
    for (row, prompt) in prompts.iter().enumerate() {
        let mut family = Rows::new(1);
        let mut st = session(&mut family, prompt);
        let mut generation = sampled_generation(8, seeds[row]);
        let mut tokens = Vec::new();
        for _ in 0..together[row].len() {
            tokens.extend(
                family
                    .decode(&mut st, &mut generation)
                    .expect("solo")
                    .tokens,
            );
        }
        assert_eq!(together[row], tokens, "row {row}");
    }
}

#[test]
fn a_window_of_one_row_calls_the_solo_decode() {
    let mut family = Rows::new(4);
    let mut st = session(&mut family, &[1, 2]);
    let mut first = generation(3);
    let batches =
        decode_window(&mut family, &mut [&mut st], &mut [&mut first]).expect("one-row window");
    assert_eq!(batches.len(), 1);
    assert_eq!(family.step_calls, 0, "the batched step is not entered");
    let mut other = Rows::new(1);
    let mut st2 = session(&mut other, &[1, 2]);
    let mut generation2 = generation(3);
    let want = other.decode(&mut st2, &mut generation2).expect("solo");
    assert_eq!(batches[0].as_ref().expect("row").tokens, want.tokens);
}

#[test]
fn a_bad_shape_is_refused_before_any_change() {
    let mut family = Rows::new(2);
    let mut a = session(&mut family, &[1]);
    let mut b = session(&mut family, &[2]);
    let mut c = session(&mut family, &[3]);
    let (mut ga, mut gb, mut gc) = (generation(4), generation(4), generation(4));
    assert!(matches!(
        decode_window(&mut family, &mut [], &mut []),
        Err(EngError::BadRequest(_))
    ));
    assert!(matches!(
        decode_window(&mut family, &mut [&mut a, &mut b], &mut [&mut ga]),
        Err(EngError::BadRequest(_))
    ));
    assert!(matches!(
        decode_window(
            &mut family,
            &mut [&mut a, &mut b, &mut c],
            &mut [&mut ga, &mut gb, &mut gc]
        ),
        Err(EngError::BadRequest(_))
    ));
    assert_eq!(family.step_calls, 0);
    assert!(a.core.pending().is_some() && b.core.pending().is_some());
}

#[test]
fn a_refused_row_does_not_lose_the_finish_of_another_row() {
    let mut family = Rows::new(4);
    let mut cancelled = session(&mut family, &[1, 2]);
    // A pending token past the embedding table: the family refuses this row.
    let mut wild = session(&mut family, &[VOCAB as u32 + 5]);
    let mut ok = session(&mut family, &[5, 6]);
    let (mut g0, mut g1, mut g2) = (generation(4), generation(4), generation(4));
    g0.cancel.store(true, Ordering::SeqCst);
    let batches = decode_window(
        &mut family,
        &mut [&mut cancelled, &mut wild, &mut ok],
        &mut [&mut g0, &mut g1, &mut g2],
    )
    .expect("the window itself stands");
    assert_eq!(
        batches[0].as_ref().expect("cancelled row").finish,
        Some(DecodeFinish::Cancelled),
        "the earlier row's cancel is kept"
    );
    assert!(
        matches!(batches[1], Err(EngError::BadRequest(_))),
        "{:?}",
        batches[1]
    );
    assert_eq!(batches[2].as_ref().expect("ok row").tokens, [next_of(6)]);
    assert!(
        cancelled.core.history().is_empty(),
        "the cancelled row was reset"
    );
}

#[test]
fn an_earlier_finish_survives_a_later_row_that_fails_preparation() {
    let mut family = Rows::new(4);
    let mut done = session(&mut family, &[1]);
    // No prompt: the token side has no pending token, so preparation fails.
    let mut empty = RowSession::default();
    let (mut limit_zero, mut g1) = (generation(0), generation(4));
    let batches = decode_window(
        &mut family,
        &mut [&mut done, &mut empty],
        &mut [&mut limit_zero, &mut g1],
    )
    .expect("window");
    assert_eq!(
        batches[0].as_ref().expect("finished row").finish,
        Some(DecodeFinish::Length)
    );
    assert!(matches!(batches[1], Err(EngError::BadRequest(_))));
}

#[test]
fn a_growth_plan_error_or_a_closed_ledger_refuses_that_row_only() {
    let mut family = Rows::new(4);
    family.growth_error_on = Some(2);
    let mut a = session(&mut family, &[1]);
    let mut b = session(&mut family, &[2]);
    let mut c = session(&mut family, &[3]);
    let (mut ga, mut gb, mut gc) = (generation(4), generation(0), generation(4));
    let batches = decode_window(
        &mut family,
        &mut [&mut a, &mut b, &mut c],
        &mut [&mut ga, &mut gb, &mut gc],
    )
    .expect("window");
    assert_eq!(batches[0].as_ref().expect("a").tokens, [next_of(1)]);
    assert_eq!(
        batches[1].as_ref().expect("b").finish,
        Some(DecodeFinish::Length),
        "a limit-zero row finishes without a growth plan"
    );
    assert_eq!(batches[2].as_ref().expect("c").tokens, [next_of(3)]);
    // The row whose pending token is 2 has a growth error once it needs a step.
    let mut family = Rows::new(4);
    family.growth_error_on = Some(2);
    let mut a = session(&mut family, &[1]);
    let mut b = session(&mut family, &[2]);
    let (mut ga, mut gb) = (generation(4), generation(4));
    let batches =
        decode_window(&mut family, &mut [&mut a, &mut b], &mut [&mut ga, &mut gb]).expect("window");
    assert_eq!(batches[0].as_ref().expect("a").tokens, [next_of(1)]);
    assert!(matches!(batches[1], Err(EngError::BadRequest(_))));
    assert_eq!(b.core.pending(), Some(2), "refused before its feed");
    // A ledger that refuses every reservation refuses every stepping row, none of them lost.
    let mut family = Rows::new(4);
    family.reserve_error = true;
    let mut a = session(&mut family, &[1]);
    let mut b = session(&mut family, &[2]);
    let mut done = session(&mut family, &[3]);
    let (mut ga, mut gb, mut gd) = (generation(4), generation(4), generation(0));
    let batches = decode_window(
        &mut family,
        &mut [&mut a, &mut b, &mut done],
        &mut [&mut ga, &mut gb, &mut gd],
    )
    .expect("window");
    assert!(matches!(batches[0], Err(EngError::BadRequest(_))));
    assert!(matches!(batches[1], Err(EngError::BadRequest(_))));
    assert_eq!(
        batches[2].as_ref().expect("done").finish,
        Some(DecodeFinish::Length)
    );
}

#[test]
fn cancel_and_limit_finish_their_rows_and_the_rest_step_together() {
    let mut family = Rows::new(8);
    let mut a = session(&mut family, &[1]);
    let mut b = session(&mut family, &[2]);
    let mut c = session(&mut family, &[3]);
    let mut d = session(&mut family, &[4]);
    let mut ga = generation(5);
    ga.cancel.store(true, Ordering::SeqCst);
    let mut gb = generation(0);
    let (mut gc, mut gd) = (generation(5), generation(5));
    let batches = decode_window(
        &mut family,
        &mut [&mut a, &mut b, &mut c, &mut d],
        &mut [&mut ga, &mut gb, &mut gc, &mut gd],
    )
    .expect("window");
    assert_eq!(
        batches[0].as_ref().expect("a").finish,
        Some(DecodeFinish::Cancelled)
    );
    assert_eq!(
        batches[1].as_ref().expect("b").finish,
        Some(DecodeFinish::Length)
    );
    assert_eq!(batches[2].as_ref().expect("c").tokens, [next_of(3)]);
    assert_eq!(batches[3].as_ref().expect("d").tokens, [next_of(4)]);
    assert_eq!(family.stepped_rows, [2], "only the live rows stepped");
    assert!(a.core.history().is_empty(), "a cancelled row is reset");
}

#[test]
fn a_growth_refusal_refuses_that_row_only_and_before_its_feed() {
    let mut family = Rows::new(8);
    family.growth = 10;
    family.cap = 20;
    let mut a = session(&mut family, &[1]);
    let mut b = session(&mut family, &[2]);
    let mut c = session(&mut family, &[3]);
    let (mut ga, mut gb, mut gc) = (generation(5), generation(5), generation(5));
    let fed_c = c.fed;
    let batches = decode_window(
        &mut family,
        &mut [&mut a, &mut b, &mut c],
        &mut [&mut ga, &mut gb, &mut gc],
    )
    .expect("window");
    assert_eq!(batches[0].as_ref().expect("a").tokens, [next_of(1)]);
    assert_eq!(batches[1].as_ref().expect("b").tokens, [next_of(2)]);
    assert!(matches!(batches[2], Err(EngError::MemoryPlan(_))));
    assert_eq!(c.fed, fed_c, "the refused row did not feed");
    assert_eq!(c.core.pending(), Some(3), "and kept its pending token");
    assert_eq!(family.used.get(), 0, "the reservations are released");
}

#[test]
fn growth_that_admits_one_row_runs_the_solo_decode() {
    let mut family = Rows::new(8);
    family.growth = 10;
    family.cap = 10;
    let mut a = session(&mut family, &[1]);
    let mut b = session(&mut family, &[2]);
    let (mut ga, mut gb) = (generation(5), generation(5));
    let batches =
        decode_window(&mut family, &mut [&mut a, &mut b], &mut [&mut ga, &mut gb]).expect("window");
    assert_eq!(batches[0].as_ref().expect("a").tokens, [next_of(1)]);
    assert!(matches!(batches[1], Err(EngError::MemoryPlan(_))));
    assert_eq!(family.step_calls, 0);
}

#[test]
fn a_failure_after_the_step_is_outer_and_consumes_the_inputs() {
    let mut family = Rows::new(8);
    family.fail_after_step = Some(EngError::Backend {
        operation: "rows",
        message: "device lost".into(),
    });
    let mut a = session(&mut family, &[1]);
    let mut b = session(&mut family, &[2]);
    let (mut ga, mut gb) = (generation(5), generation(5));
    let error = decode_window(&mut family, &mut [&mut a, &mut b], &mut [&mut ga, &mut gb])
        .expect_err("the step failed");
    assert!(matches!(error, EngError::Backend { .. }));
    assert_eq!(a.fed, 1);
    assert_eq!(a.core.pending(), None, "the fed token is not retried");
    assert_eq!(b.core.pending(), None);
}

#[test]
fn a_row_that_cannot_be_sampled_ends_alone_and_the_others_stand() {
    let mut family = Rows::new(8);
    family.narrow_row = Some(1);
    let mut a = session(&mut family, &[1]);
    let mut b = session(&mut family, &[2]);
    let mut c = session(&mut family, &[3]);
    let (mut ga, mut gb, mut gc) = (generation(5), generation(5), generation(5));
    let batches = decode_window(
        &mut family,
        &mut [&mut a, &mut b, &mut c],
        &mut [&mut ga, &mut gb, &mut gc],
    )
    .expect("the window stands");
    assert_eq!(batches[0].as_ref().expect("a").tokens, [next_of(1)]);
    assert!(
        matches!(batches[1], Err(EngError::Backend { .. })),
        "{:?}",
        batches[1]
    );
    assert_eq!(batches[2].as_ref().expect("c").tokens, [next_of(3)]);
    assert_eq!(b.fed, 1, "its input was consumed");
    assert_eq!(b.core.pending(), None, "so it is not retried");
}

#[test]
fn a_wrong_row_count_is_an_outer_error() {
    let mut family = Rows::new(8);
    family.short_result = true;
    let mut a = session(&mut family, &[1]);
    let mut b = session(&mut family, &[2]);
    let (mut ga, mut gb) = (generation(5), generation(5));
    assert!(decode_window(&mut family, &mut [&mut a, &mut b], &mut [&mut ga, &mut gb]).is_err());
}
