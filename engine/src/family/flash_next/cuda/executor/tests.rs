//! The executor over the fake driver: the worker's call order, the finish rules and the limits.

use std::sync::Arc;

use super::{CudaRules, FlashCudaExecutor};
use crate::{
    EngError,
    family::{
        executor::{ExecutorOps, GenerationState},
        flash_next::cuda::driver::{
            TokenDriver,
            fakes::{FakeDevice, FakeFeeds, POSITIONS, SAMPLER_VOCAB, shape},
        },
        session::{DecodeSession, TokenStepper},
    },
    tokenizer::{DecodeFinish, IncrementalDecoder, Tokenizer},
};

const TOKENIZER_JSON: &str = r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},"post_processor":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},"decoder":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},"model":{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":false,"vocab":{"a":0,"b":1,"c":2,"d":3,"e":4,"f":5,"g":6,"h":7},"merges":[]}}"#;

type Executor = FlashCudaExecutor<FakeDevice, FakeFeeds>;

fn build(eos: &[u32]) -> (Executor, DecodeSession) {
    let driver = TokenDriver::new(FakeDevice::default(), FakeFeeds::default(), shape()).unwrap();
    FlashCudaExecutor::new(
        driver,
        CudaRules {
            eos: Arc::from(eos),
            sampler_vocab: SAMPLER_VOCAB,
        },
    )
    .unwrap()
}

fn generation(max_tokens: u32) -> GenerationState {
    let tokenizer = Tokenizer::from_bytes(TOKENIZER_JSON.as_bytes()).unwrap();
    GenerationState::new(IncrementalDecoder::new(&tokenizer)).with_max_tokens(max_tokens)
}

#[test]
fn a_job_prefills_all_but_the_last_token_and_decodes_greedily_in_the_vocabulary() {
    let (mut executor, mut session) = build(&[]);
    executor.reset(&mut session).unwrap();
    executor.prefill(&mut session, &[1, 2, 3]).unwrap();
    assert_eq!(
        executor.driver.fed(),
        2,
        "the last prompt token stays pending"
    );
    let mut state = generation(2);
    let mut tokens = Vec::new();
    for _ in 0..2 {
        let batch = executor.decode(&mut session, &mut state).unwrap();
        tokens.extend(batch.tokens);
    }
    // The fake head grows with the id: the draw is the last id inside the sampler vocabulary.
    let last = u32::try_from(SAMPLER_VOCAB - 1).unwrap();
    assert_eq!(tokens, [last, last]);
    assert_eq!(executor.driver.fed(), 4);
    assert_eq!(executor.history(&session), [1, 2, 3, last, last]);
}

#[test]
fn an_eos_draw_finishes_the_job() {
    let last = u32::try_from(SAMPLER_VOCAB - 1).unwrap();
    let (mut executor, mut session) = build(&[last]);
    executor.prefill(&mut session, &[1]).unwrap();
    let batch = executor.decode(&mut session, &mut generation(8)).unwrap();
    assert_eq!(batch.finish, Some(DecodeFinish::Stop));
}

#[test]
fn a_prompt_at_the_context_limit_is_refused_before_any_step() {
    let (executor, _) = build(&[]);
    assert_eq!(executor.context_limit(), POSITIONS);
    assert!(executor.check_prompt(POSITIONS - 1).is_ok());
    assert!(matches!(
        executor.check_prompt(POSITIONS),
        Err(EngError::PromptTooLong { .. })
    ));
    assert_eq!(executor.driver.fed(), 0);
}

#[test]
fn reset_starts_a_new_job_from_position_zero() {
    let (mut executor, mut session) = build(&[]);
    executor.prefill(&mut session, &[1, 2, 3]).unwrap();
    executor.reset(&mut session).unwrap();
    assert_eq!(executor.driver.fed(), 0);
    assert!(executor.history(&session).is_empty());
    assert_eq!(executor.prefix_limits().roots, 0);
}

#[test]
fn a_zero_sampler_vocabulary_is_refused() {
    let driver = TokenDriver::new(FakeDevice::default(), FakeFeeds::default(), shape()).unwrap();
    let rules = CudaRules {
        eos: Arc::from([]),
        sampler_vocab: 0,
    };
    assert!(FlashCudaExecutor::new(driver, rules).is_err());
}
