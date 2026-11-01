//! One speculative decode step: draft, run a window, accept, commit.
//!
//! The step replaces `DecodeSession::decode` for a job whose engine has speculation on. It
//! publishes the same tokens a one-row `decode` loop would, several per call. When there is no
//! draft, or no room for one, it calls the one-row `decode`, so a stand-down is the plain path.

use super::{
    accept::{Accepted, accept_run},
    repeat::RepeatIndex,
    stats::{SpecStats, StandDown},
    window::{Commit, Window, drafts_allowed},
};
use crate::{
    EngError,
    family::{
        executor::GenerationState,
        sampler::Sampler,
        session::{DecodePreparation, DecodeSession, Limits, TokenStepper},
    },
    tokenizer::DecodeBatch,
};

type Result<T> = std::result::Result<T, EngError>;

/// What a drafter offers for the tokens after the pending token.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct Proposal {
    pub(crate) tokens: Vec<u32>,
    /// The drafter's confidence score of each token, or empty when it has none.
    pub(crate) scores: Vec<f32>,
}

/// A source of drafts.
pub(crate) trait Drafter {
    /// Guesses at most `room` tokens that follow the last token of `history`, which is the
    /// pending token. An empty proposal means no guess.
    fn propose(&mut self, history: &[u32], room: usize) -> Result<Proposal>;

    /// Learns the verdict of the window that carried the proposal.
    fn observe(&mut self, window: &Window, accepted: &Accepted) -> Result<()>;
}

/// The model side of a window.
pub(crate) trait WindowModel: TokenStepper {
    /// Feeds `inputs` as rows after the committed ones and keeps the result of every row. The
    /// rows are not committed: `fed` is unchanged until `commit_window`.
    fn run_window(&mut self, inputs: &[u32]) -> Result<()>;

    /// Draws the token of window row `row` with `sampler`, from the first `sampler_vocab` logits.
    fn draw(&mut self, row: usize, sampler: &mut Sampler, sampler_vocab: usize) -> Result<u32>;

    /// Keeps the rows and the state that `commit` names and drops the rest.
    fn commit_window(&mut self, commit: &Commit) -> Result<()>;
}

/// The n-gram and hybrid drafter: it reads only the committed tokens.
pub(crate) struct RepeatDrafter {
    index: RepeatIndex,
    /// The most drafts of an n-gram window and of a hybrid window.
    plain_cap: usize,
    hybrid_cap: usize,
}

impl RepeatDrafter {
    /// `ngram_match_min` and `hybrid_tokens` are the tail lengths; the caps are the draft
    /// ceilings of the two kinds of window.
    pub(crate) fn new(
        (ngram_match_min, hybrid_tokens): (usize, usize),
        (plain_cap, hybrid_cap): (usize, usize),
    ) -> Self {
        Self {
            index: RepeatIndex::new(ngram_match_min, hybrid_tokens),
            plain_cap,
            hybrid_cap,
        }
    }
}

impl Drafter for RepeatDrafter {
    /// A hybrid draft wins when a long passage repeats; otherwise the n-gram draft.
    fn propose(&mut self, history: &[u32], room: usize) -> Result<Proposal> {
        self.index.sync(history);
        let found = self
            .index
            .hybrid(room.min(self.hybrid_cap))
            .or_else(|| self.index.ngram(room.min(self.plain_cap)));
        Ok(found.map_or_else(Proposal::default, |draft| Proposal {
            tokens: draft.tokens,
            scores: Vec::new(),
        }))
    }

    fn observe(&mut self, _window: &Window, _accepted: &Accepted) -> Result<()> {
        Ok(())
    }
}

/// The speculative step of one engine: its drafter, the draft ceiling and the counters.
pub(crate) struct Speculator<D> {
    pub(crate) drafter: D,
    /// The most drafts any window may carry (`draft_max_rows - 1`).
    pub(crate) max_drafts: usize,
    pub(crate) stats: SpecStats,
}

impl<D: Drafter> Speculator<D> {
    pub(crate) fn new(drafter: D, max_drafts: usize) -> Self {
        Self {
            drafter,
            max_drafts,
            stats: SpecStats::default(),
        }
    }

    /// One call of the job's decode loop. It finishes the job exactly where the one-row loop
    /// would, and publishes between one and `max_drafts + 1` tokens.
    pub(crate) fn decode<M: WindowModel>(
        &mut self,
        session: &mut DecodeSession,
        model: &mut M,
        generation: &mut GenerationState,
        limits: &Limits<'_>,
    ) -> Result<DecodeBatch> {
        if let DecodePreparation::Finished(batch) = session.prepare_decode(model, generation)? {
            return Ok(batch);
        }
        let room = drafts_allowed(
            self.max_drafts,
            model.fed(),
            limits.context_limit,
            generation.remaining_tokens(),
        );
        if room == 0 {
            return self.plain(StandDown::NoRoom, session, model, generation, limits);
        }
        let mut proposal = self.drafter.propose(session.history(), room)?;
        proposal.tokens.truncate(room);
        proposal.scores.truncate(room);
        if proposal.tokens.is_empty() {
            return self.plain(StandDown::NoDraft, session, model, generation, limits);
        }
        let pending = session.checked_pending()?;
        let window = Window::new(model.fed(), proposal.tokens, proposal.scores);
        model.run_window(&window.input(pending))?;
        let left = generation
            .remaining_tokens()
            .map_or(usize::MAX, |tokens| tokens as usize);
        let accepted = accept_run(
            window.drafts(),
            left,
            |row| model.draw(row, &mut generation.sampler, limits.sampler_vocab),
            |token| limits.eos.contains(&token),
        )?;
        model.commit_window(&window.commit(&accepted)?)?;
        self.drafter.observe(&window, &accepted)?;
        self.stats.window(window.rows(), accepted.rows_kept());
        session.finish_decode_tokens(model, generation, limits, &accepted.tokens)
    }

    fn plain<M: WindowModel>(
        &mut self,
        why: StandDown,
        session: &mut DecodeSession,
        model: &mut M,
        generation: &mut GenerationState,
        limits: &Limits<'_>,
    ) -> Result<DecodeBatch> {
        self.stats.stand_down(why);
        session.decode(model, generation, limits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        family::sampler::SamplingParams,
        tokenizer::{DecodeFinish, IncrementalDecoder, Tokenizer},
    };

    /// The synthetic tokenizer: ids 0 to 7 are the letters a to h.
    const TOKENIZER_JSON: &str = r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},"post_processor":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},"decoder":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},"model":{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":false,"vocab":{"a":0,"b":1,"c":2,"d":3,"e":4,"f":5,"g":6,"h":7},"merges":[]}}"#;
    const VOCAB: usize = 8;
    const CONTEXT: usize = 4096;

    /// How the fake model scores the next token.
    #[derive(Clone, Copy)]
    enum Shape {
        /// The token after `t` is `(t + 1) % 8`, by a margin of 10.
        Sharp,
        /// The same peak by a margin of 2: sampled runs leave it often.
        Soft,
        /// An irregular function of the last two tokens.
        Hashy,
    }

    struct Lm {
        shape: Shape,
        fed: Vec<u32>,
        window: Vec<u32>,
    }

    impl Lm {
        fn new(shape: Shape) -> Self {
            Self {
                shape,
                fed: Vec::new(),
                window: Vec::new(),
            }
        }

        fn row(&self, seq: &[u32]) -> Vec<f32> {
            let last = *seq.last().expect("a token") as usize;
            let prev = seq.len().checked_sub(2).map_or(0, |i| seq[i] as usize);
            (0..VOCAB)
                .map(|v| match self.shape {
                    Shape::Sharp => {
                        if v == (last + 1) % VOCAB {
                            10.0
                        } else {
                            0.0
                        }
                    }
                    Shape::Soft => {
                        if v == (last + 1) % VOCAB {
                            2.0
                        } else {
                            0.0
                        }
                    }
                    Shape::Hashy => ((last * 31 + prev * 17 + v * 7) % 11) as f32 * 0.3,
                })
                .collect()
        }
    }

    impl TokenStepper for Lm {
        const OPERATION: &'static str = "fake";

        fn feed(&mut self, token: u32) -> Result<()> {
            self.fed.push(token);
            Ok(())
        }

        fn logits(&mut self, token: u32) -> Result<Vec<f32>> {
            self.fed.push(token);
            Ok(self.row(&self.fed))
        }

        fn fed(&self) -> usize {
            self.fed.len()
        }

        fn reset(&mut self) -> Result<()> {
            self.fed.clear();
            self.window.clear();
            Ok(())
        }
    }

    impl WindowModel for Lm {
        fn run_window(&mut self, inputs: &[u32]) -> Result<()> {
            self.window = inputs.to_vec();
            Ok(())
        }

        fn draw(&mut self, row: usize, sampler: &mut Sampler, vocab: usize) -> Result<u32> {
            let mut seq = self.fed.clone();
            seq.extend_from_slice(&self.window[..=row]);
            sampler.sample(&self.row(&seq)[..vocab])
        }

        fn commit_window(&mut self, commit: &Commit) -> Result<()> {
            self.fed.extend_from_slice(&self.window[..commit.kept_rows]);
            self.window.clear();
            Ok(())
        }
    }

    const PROMPT: [u32; 12] = [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3];

    fn generation(sampler: Sampler, max_tokens: u32) -> GenerationState {
        let tokenizer = Tokenizer::from_bytes(TOKENIZER_JSON.as_bytes()).expect("tokenizer");
        GenerationState::with_sampler(IncrementalDecoder::new(&tokenizer), sampler)
            .with_max_tokens(max_tokens)
    }

    fn sampled(seed: u64) -> Sampler {
        Sampler::new(SamplingParams::new(1.0, 1.0, 8).expect("params"), seed)
    }

    struct Outcome {
        tokens: Vec<u32>,
        finish: Option<DecodeFinish>,
        text: String,
    }

    fn drive(
        (context, shape): (usize, Shape),
        eos: &[u32],
        (sampler, max_tokens): (Sampler, u32),
        mut step: impl FnMut(
            &mut DecodeSession,
            &mut Lm,
            &mut GenerationState,
            &Limits<'_>,
        ) -> Result<DecodeBatch>,
    ) -> Outcome {
        let mut lm = Lm::new(shape);
        let mut session = DecodeSession::default();
        session.prefill(&mut lm, &PROMPT, context).expect("prefill");
        let mut generation = generation(sampler, max_tokens);
        let limits = Limits {
            eos,
            sampler_vocab: VOCAB,
            context_limit: context,
        };
        let mut out = Outcome {
            tokens: Vec::new(),
            finish: None,
            text: String::new(),
        };
        for _ in 0..1000 {
            let batch = step(&mut session, &mut lm, &mut generation, &limits).expect("step");
            out.tokens.extend(batch.tokens);
            out.text.push_str(&batch.text);
            if batch.finish.is_some() {
                out.finish = batch.finish;
                break;
            }
        }
        out
    }

    fn solo_in(context: usize, shape: Shape, eos: &[u32], run: (Sampler, u32)) -> Outcome {
        drive(
            (context, shape),
            eos,
            run,
            |session, lm, generation, limits| session.decode(lm, generation, limits),
        )
    }

    fn solo(shape: Shape, eos: &[u32], run: (Sampler, u32)) -> Outcome {
        solo_in(CONTEXT, shape, eos, run)
    }

    fn speculative_in(
        context: usize,
        shape: Shape,
        eos: &[u32],
        run: (Sampler, u32),
    ) -> (Outcome, SpecStats) {
        let drafter = RepeatDrafter::new((2, 0), (4, 0));
        let mut speculator = Speculator::new(drafter, 7);
        let out = drive(
            (context, shape),
            eos,
            run,
            |session, lm, generation, limits| speculator.decode(session, lm, generation, limits),
        );
        (out, speculator.stats)
    }

    fn speculative(shape: Shape, eos: &[u32], run: (Sampler, u32)) -> (Outcome, SpecStats) {
        speculative_in(CONTEXT, shape, eos, run)
    }

    fn same(a: &Outcome, b: &Outcome) {
        assert_eq!(a.tokens, b.tokens);
        assert_eq!(a.finish, b.finish);
        assert_eq!(a.text, b.text);
    }

    #[test]
    fn a_greedy_job_commits_the_same_tokens_and_takes_fewer_steps() {
        let reference = solo(Shape::Sharp, &[], (Sampler::greedy(), 40));
        let (out, stats) = speculative(Shape::Sharp, &[], (Sampler::greedy(), 40));
        same(&out, &reference);
        assert_eq!(out.tokens.len(), 40);
        assert_eq!(out.finish, Some(DecodeFinish::Length));
        let per_step = stats.tokens_per_step().expect("steps");
        assert!(per_step > 3.0, "{per_step}");
        assert_eq!(stats.acceptance(), Some(1.0));
    }

    #[test]
    fn a_seeded_sampler_commits_the_same_tokens() {
        let mut partial = 0;
        for seed in 0..20 {
            let reference = solo(Shape::Soft, &[], (sampled(seed), 60));
            let (out, stats) = speculative(Shape::Soft, &[], (sampled(seed), 60));
            same(&out, &reference);
            if stats.acceptance().is_some_and(|a| a < 1.0) {
                partial += 1;
            }
        }
        assert!(
            partial > 10,
            "the fixture must exercise rejections: {partial}"
        );
    }

    #[test]
    fn an_irregular_model_with_few_matches_still_commits_the_same_tokens() {
        for seed in 0..10 {
            let reference = solo(Shape::Hashy, &[], (sampled(seed), 50));
            let (out, _) = speculative(Shape::Hashy, &[], (sampled(seed), 50));
            same(&out, &reference);
        }
        let reference = solo(Shape::Hashy, &[], (Sampler::greedy(), 50));
        let (out, _) = speculative(Shape::Hashy, &[], (Sampler::greedy(), 50));
        same(&out, &reference);
    }

    #[test]
    fn an_end_token_inside_a_window_ends_the_job_there() {
        // The counting model emits 4, 5, 6, 7, 0 ...; 6 ends the job.
        let reference = solo(Shape::Sharp, &[6], (Sampler::greedy(), 100));
        let (out, _) = speculative(Shape::Sharp, &[6], (Sampler::greedy(), 100));
        same(&out, &reference);
        assert_eq!(out.finish, Some(DecodeFinish::Stop));
        assert_eq!(out.tokens.last(), Some(&6));
    }

    #[test]
    fn the_output_limit_cuts_a_window() {
        for max in 1..12 {
            let reference = solo(Shape::Sharp, &[], (Sampler::greedy(), max));
            let (out, _) = speculative(Shape::Sharp, &[], (Sampler::greedy(), max));
            same(&out, &reference);
            assert_eq!(out.tokens.len(), max as usize);
        }
    }

    #[test]
    fn the_context_limit_cuts_a_window() {
        // The cache ends a few positions after the prompt: a window must stop where the
        // one-row run stops, with the same finish.
        for room in 1..9 {
            let context = PROMPT.len() + room;
            let reference = solo_in(context, Shape::Sharp, &[], (Sampler::greedy(), 100));
            let (out, _) = speculative_in(context, Shape::Sharp, &[], (Sampler::greedy(), 100));
            same(&out, &reference);
            assert_eq!(out.finish, Some(DecodeFinish::Length), "room {room}");
        }
    }

    #[test]
    fn the_model_state_follows_the_committed_rows() {
        let mut lm = Lm::new(Shape::Sharp);
        let mut session = DecodeSession::default();
        session.prefill(&mut lm, &PROMPT, CONTEXT).expect("prefill");
        let mut generation = generation(Sampler::greedy(), 30);
        let limits = Limits {
            eos: &[],
            sampler_vocab: VOCAB,
            context_limit: CONTEXT,
        };
        let mut speculator = Speculator::new(RepeatDrafter::new((2, 0), (4, 0)), 7);
        for _ in 0..5 {
            speculator
                .decode(&mut session, &mut lm, &mut generation, &limits)
                .expect("step");
            // Every committed token but the pending one has been fed.
            assert_eq!(lm.fed(), session.history().len() - 1);
            assert_eq!(lm.fed, session.history()[..session.history().len() - 1]);
        }
    }

    #[test]
    fn the_hybrid_draft_takes_a_long_repeat_in_one_window() {
        let mut lm = Lm::new(Shape::Sharp);
        let mut session = DecodeSession::default();
        let prompt: Vec<u32> = (0..24).map(|i| i % 8).collect();
        session.prefill(&mut lm, &prompt, CONTEXT).expect("prefill");
        let mut generation = generation(Sampler::greedy(), 20);
        let limits = Limits {
            eos: &[],
            sampler_vocab: VOCAB,
            context_limit: CONTEXT,
        };
        let mut speculator = Speculator::new(RepeatDrafter::new((2, 8), (2, 7)), 7);
        let batch = speculator
            .decode(&mut session, &mut lm, &mut generation, &limits)
            .expect("step");
        assert_eq!(batch.tokens.len(), 8, "seven drafts and the bonus");
    }
}
