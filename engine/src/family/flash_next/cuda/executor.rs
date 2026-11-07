//! The `ExecutorOps` adapter of the Flash-Next token driver on a CUDA card.
//!
//! `FlashCudaExecutor` is the engine thread's executor. It follows the Qwen3.6 CUDA executor
//! (`qwen36/cuda/executor.rs`): the session is the shared `DecodeSession`, which owns the committed
//! tokens, the pending token, the finish flag, EOS, the output limit, the context limit and
//! cancellation; the model side is the [`TokenDriver`], a `TokenStepper`. This file holds no finish
//! rule of its own.
//!
//! The prompt is fed through the default `TokenStepper::feed_rows` loop, one token per step: the
//! chunked prefill is a later change and replaces that loop without changing this adapter. No prefix
//! is saved or restored (the defaults of `ExecutorOps`: zero roots, nothing saved), because the
//! device state of a Flash-Next sequence has no export yet.
//!
//! Errors. A failure inside the step poisons the driver and the original error comes back
//! unchanged; the worker resets the session after any error, as for the other executors.

use std::sync::Arc;

use super::driver::{TokenDevice, TokenDriver, TokenFeeds};
use crate::{
    EngError,
    family::{
        executor::{ExecutorOps, GenerationState},
        session::{DecodeSession, Limits, check_prompt},
    },
    tokenizer::DecodeBatch,
};

type Result<T> = std::result::Result<T, EngError>;

/// The limits that the caller reads from the tokenizer bundle. The driver supplies the position
/// limit.
pub(in crate::family::flash_next) struct CudaRules {
    /// Every id that ends generation (`TokenizerBundle::eos_token_ids`).
    pub(in crate::family::flash_next) eos: Arc<[u32]>,
    /// The number of logits the sampler may choose from (`TokenizerBundle::vocab_size`). The head
    /// is wider (padded ids above the tokenizer's vocabulary), and those ids have no text.
    pub(in crate::family::flash_next) sampler_vocab: usize,
}

/// The executor of one Flash-Next model on a card. It lives on the engine thread.
pub(in crate::family::flash_next) struct FlashCudaExecutor<D: TokenDevice, F: TokenFeeds> {
    driver: TokenDriver<D, F>,
    rules: CudaRules,
}

impl<D: TokenDevice, F: TokenFeeds> FlashCudaExecutor<D, F> {
    /// The executor and its first session. Refuses a sampler vocabulary of 0.
    pub(in crate::family::flash_next) fn new(
        driver: TokenDriver<D, F>,
        rules: CudaRules,
    ) -> Result<(Self, DecodeSession)> {
        if rules.sampler_vocab == 0 {
            return Err(EngError::Checkpoint(
                "Flash-Next CUDA: the sampler vocabulary must be above 0".to_string(),
            ));
        }
        Ok((Self { driver, rules }, DecodeSession::default()))
    }

    /// The number of positions the caches accept.
    pub(in crate::family::flash_next) fn context_limit(&self) -> usize {
        self.driver.positions()
    }
}

impl<D: TokenDevice, F: TokenFeeds> ExecutorOps for FlashCudaExecutor<D, F> {
    type Session = DecodeSession;

    /// Refuses a prompt that cannot fit before any state changes.
    fn check_prompt(&self, prompt_tokens: usize) -> Result<()> {
        check_prompt(prompt_tokens, self.context_limit())
    }

    fn history<'s>(&self, st: &'s DecodeSession) -> &'s [u32] {
        st.history()
    }

    fn reset(&mut self, st: &mut DecodeSession) -> Result<()> {
        st.reset(&mut self.driver)
    }

    fn prefill(&mut self, st: &mut DecodeSession, toks: &[u32]) -> Result<()> {
        let context_limit = self.context_limit();
        st.prefill(&mut self.driver, toks, context_limit)
    }

    fn decode(
        &mut self,
        st: &mut DecodeSession,
        generation: &mut GenerationState,
    ) -> Result<DecodeBatch> {
        let limits = Limits {
            eos: &self.rules.eos,
            sampler_vocab: self.rules.sampler_vocab,
            context_limit: self.driver.positions(),
        };
        st.decode(&mut self.driver, generation, &limits)
    }

    /// No transient routing or draft state exists between turns.
    fn boundary(&mut self, _st: &mut DecodeSession) {}
}

#[cfg(test)]
mod tests;
