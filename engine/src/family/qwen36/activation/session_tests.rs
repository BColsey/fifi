//! Exercise the real shared session policy with a synthetic residual-producing step.
use super::*;
use crate::{
    family::{
        executor::GenerationState,
        session::{DecodeSession, Limits, TokenStepper},
    },
    tokenizer::{IncrementalDecoder, Tokenizer},
};
use std::sync::atomic::Ordering;

#[derive(Default)]
struct Step {
    fed: Vec<u32>,
    arena: Vec<f32>,
    capture: Option<Capture>,
    reads: usize,
    forwards: usize,
    fail_after_copy: bool,
}

impl TokenStepper for Step {
    const OPERATION: &'static str = "activation-synthetic";
    fn feed(&mut self, token: u32) -> Result<()> {
        self.fed.push(token);
        Ok(())
    }
    fn logits(&mut self, token: u32) -> Result<Vec<f32>> {
        let position = self.fed.len();
        self.feed(token)?;
        self.forwards += 1;
        self.arena = vec![token as f32, 0.25, -2.0, 3.0];
        if let Some(capture) = self.capture.as_mut() {
            capture.observe(LAYER, position, || {
                self.reads += 1;
                Ok(self.arena.clone())
            })?;
        }
        self.arena.fill(99.0); // The next mixer reuses the raw row before sampling.
        if self.fail_after_copy {
            return Err(refusal("synthetic later-block failure"));
        }
        Ok(vec![0.0, 2.0, 1.0])
    }
    fn fed(&self) -> usize {
        self.fed.len()
    }
    fn reset(&mut self) -> Result<()> {
        self.fed.clear();
        self.arena.fill(0.0);
        Ok(())
    }
}

fn generation() -> GenerationState {
    let tokenizer = Tokenizer::from_bytes(br#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":null,"model":{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":false,"vocab":{"a":0,"b":1,"c":2},"merges":[]}}"#).unwrap();
    GenerationState::new(IncrementalDecoder::new(&tokenizer))
}

fn limits() -> Limits<'static> {
    Limits {
        eos: &[],
        sampler_vocab: 3,
        context_limit: 32,
    }
}

fn arm(step: &mut Step, session: &DecodeSession) {
    let settings = super::tests::settings();
    let request = super::tests::request(session.history());
    request
        .check_pending(PromptBoundary {
            history: session.history(),
            prompt_len: session.prompt_len(),
            pending: session.pending(),
            fed: step.fed(),
        })
        .unwrap();
    step.capture = Some(Capture::new(request, settings).unwrap());
}

#[test]
fn session_chunks_and_synthetic_restored_prefix_feed_final_prompt_once() {
    let prompt = [0, 2, 0, 2];
    for pieces in [vec![4], vec![1, 3], vec![2, 1, 1]] {
        let mut step = Step::default();
        let mut session = DecodeSession::default();
        let mut offset = 0;
        for len in pieces {
            session
                .prefill(&mut step, &prompt[offset..offset + len], 32)
                .unwrap();
            offset += len;
            // Simulate restoring a committed host prefix into fresh storage.
            let saved_session = session.clone();
            step = Step {
                fed: step.fed.clone(),
                ..Step::default()
            };
            session = saved_session;
        }
        assert_eq!(step.fed, prompt[..3]);
        let mut plain_step = Step {
            fed: step.fed.clone(),
            ..Step::default()
        };
        let mut plain_session = session.clone();
        let plain = plain_session
            .decode(&mut plain_step, &mut generation(), &limits())
            .unwrap();
        arm(&mut step, &session);
        let decoded = session.decode(&mut step, &mut generation(), &limits());
        let capture = step.capture.take().unwrap();
        let (batch, record) = capture.finish(decoded).unwrap();
        assert_eq!(batch.tokens, plain.tokens);
        assert_eq!(batch.text, plain.text);
        assert_eq!(step.fed, plain_step.fed);
        assert_eq!(step.arena, plain_step.arena);
        assert_eq!(session.history(), plain_session.history());
        assert_eq!((step.forwards, step.reads), (1, 1));
        assert_eq!((plain_step.forwards, plain_step.reads), (1, 0));
        assert_eq!(record.values(), &[2.0, 0.25, -2.0, 3.0]);
        step.reset().unwrap();
        assert_eq!(record.values(), &[2.0, 0.25, -2.0, 3.0]);
    }
}

#[test]
fn cancellation_and_failed_forward_discard_the_one_shot_observation() {
    for cancelled in [true, false] {
        let mut step = Step {
            fail_after_copy: !cancelled,
            ..Step::default()
        };
        let mut session = DecodeSession::default();
        session.prefill(&mut step, &[0, 2], 32).unwrap();
        arm(&mut step, &session);
        let mut generation = generation();
        generation.cancel.store(cancelled, Ordering::SeqCst);
        let decoded = session.decode(&mut step, &mut generation, &limits());
        let capture = step.capture.take().unwrap();
        assert!(capture.finish(decoded).is_err());
        assert!(step.capture.is_none());
        assert_eq!(step.reads, usize::from(!cancelled));
        assert_eq!(step.forwards, usize::from(!cancelled));
        if cancelled {
            assert!(step.fed.is_empty());
        } else {
            assert!(
                session.pending().is_none(),
                "consumed token must never be retried"
            );
        }
    }
}
