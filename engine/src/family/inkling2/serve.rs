//! Both reference backends use the shared token session and sampler.

use std::path::Path;

use super::{TEXT_CONTEXT_LIMIT, TextBackend, cpu_forward::CpuForward};
use crate::{
    EngError,
    family::{
        executor::{ExecutorOps, GenerationState},
        session::{DecodeSession, Limits, StoreLimits, TokenStepper, check_prompt},
    },
    sched::worker::{Engine, ReadyNote, spawn_engine_noted},
    tokenizer::{DecodeBatch, TokenizerBundle},
};

pub(crate) fn start_engine(
    backend: TextBackend,
    model_dir: &Path,
    tokenizer_dir: &Path,
    ready_note: ReadyNote,
) -> Result<(Engine, TokenizerBundle), EngError> {
    let bundle = crate::tokenizer::inkling2::load(tokenizer_dir)?;
    let vocab_size = bundle.vocab_size();
    let eos = bundle.eos_token_ids().to_vec();
    let model_dir = model_dir.to_path_buf();
    // Bound the time before the worker can observe cancellation during prefill.
    let engine = spawn_engine_noted(1, ready_note, move |stages| {
        stages.begin("weights");
        let forward = CpuForward::open(&model_dir, TEXT_CONTEXT_LIMIT as usize)?;
        forward.validate_tokenizer(vocab_size, &eos)?;
        let forward = match backend {
            TextBackend::Cpu => forward,
            TextBackend::Cuda => {
                stages.begin("CUDA projections");
                cuda_forward(forward)?
            }
        };
        Ok((TextExecutor { forward, eos }, DecodeSession::default()))
    })?;
    Ok((engine, bundle))
}

#[cfg(all(feature = "native-cuda", target_os = "linux"))]
fn cuda_forward(mut forward: CpuForward) -> Result<CpuForward, EngError> {
    forward.enable_cuda()?;
    Ok(forward)
}

#[cfg(not(all(feature = "native-cuda", target_os = "linux")))]
fn cuda_forward(_forward: CpuForward) -> Result<CpuForward, EngError> {
    Err(EngError::BadRequest(
        "Inkling CUDA serving requires a 64-bit Linux build with native-cuda".into(),
    ))
}

impl TokenStepper for CpuForward {
    const OPERATION: &'static str = "inkling2_text";

    fn feed(&mut self, token: u32) -> Result<(), EngError> {
        self.token(token, false).map(|_| ())
    }

    fn logits(&mut self, token: u32) -> Result<Vec<f32>, EngError> {
        self.token(token, true)?.ok_or_else(|| EngError::Backend {
            operation: Self::OPERATION,
            message: "the output projection returned no logits".into(),
        })
    }

    fn fed(&self) -> usize {
        self.position()
    }

    fn reset(&mut self) -> Result<(), EngError> {
        Self::reset(self);
        Ok(())
    }
}

struct TextExecutor {
    forward: CpuForward,
    eos: Vec<u32>,
}

impl ExecutorOps for TextExecutor {
    type Session = DecodeSession;

    fn check_prompt(&self, prompt_tokens: usize) -> Result<(), EngError> {
        check_prompt(prompt_tokens, self.forward.context)
    }

    fn history<'s>(&self, session: &'s Self::Session) -> &'s [u32] {
        session.history()
    }

    fn reset(&mut self, session: &mut Self::Session) -> Result<(), EngError> {
        session.reset(&mut self.forward)
    }

    fn prefill(&mut self, session: &mut Self::Session, tokens: &[u32]) -> Result<(), EngError> {
        let context = self.forward.context;
        session.prefill(&mut self.forward, tokens, context)
    }

    fn decode(
        &mut self,
        session: &mut Self::Session,
        generation: &mut GenerationState,
    ) -> Result<DecodeBatch, EngError> {
        let limits = Limits {
            eos: &self.eos,
            sampler_vocab: self.forward.vocab_size() as usize,
            context_limit: self.forward.context,
        };
        session.decode(&mut self.forward, generation, &limits)
    }

    fn boundary(&mut self, _session: &mut Self::Session) {}

    fn prefix_limits(&self) -> StoreLimits {
        StoreLimits {
            roots: 0,
            entries: 0,
        }
    }
}
