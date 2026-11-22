//! Both reference backends use the shared token session and sampler.

use std::path::Path;

#[cfg(target_os = "linux")]
use super::DEFAULT_TEXT_CONTEXT;
use super::{
    TextBackend,
    cpu_forward::{CpuForward, PrefixState},
};
use crate::{
    EngError,
    family::{
        executor::{ExecutorOps, GenerationState},
        session::{
            DecodeSession, Limits, PrefixStore, SaveKind, StoreLimits, TokenStepper, check_prompt,
        },
    },
    sched::worker::{Engine, ReadyNote, spawn_engine_noted},
    tokenizer::{DecodeBatch, TokenizerBundle},
};

pub(crate) fn start_engine(
    backend: TextBackend,
    model_dir: &Path,
    tokenizer_dir: &Path,
    settings: TextSettings,
    ready_note: ReadyNote,
) -> Result<(Engine, TokenizerBundle), EngError> {
    let bundle = crate::tokenizer::inkling2::load(tokenizer_dir)?;
    let vocab_size = bundle.vocab_size();
    let eos = bundle.eos_token_ids().to_vec();
    let model_dir = model_dir.to_path_buf();
    let requested_rows = prefill_rows(backend)?;
    // Leave cancellation and prefix boundaries with the shared worker.
    let engine = spawn_engine_noted(1, ready_note, move |stages| {
        stages.begin("weights");
        let forward = CpuForward::open(&model_dir, settings.context as usize)?;
        let (planned_rows, host_reserve) = settings.host_prefill_plan(&forward, requested_rows)?;
        let prefix = PrefixStore::with_limits(settings.prefix_limits());
        forward.validate_tokenizer(vocab_size, &eos)?;
        let forward = match backend {
            TextBackend::Cpu => forward,
            TextBackend::Cuda => {
                stages.begin("CUDA projections");
                cuda_forward(forward, host_reserve)?
            }
        };
        let prefill_rows = if supports_grouped_prefill(&forward) {
            planned_rows
        } else {
            1
        };
        if requested_rows > 1 {
            eprintln!(
                "Inkling CUDA prefill rows {}",
                serde_json::json!({
                    "requested": requested_rows,
                    "admitted": prefill_rows,
                    "context": settings.context,
                    "host_reserve_bytes": host_reserve,
                })
            );
        }
        Ok((
            TextExecutor {
                forward,
                eos,
                prefix,
                prefill_rows,
            },
            DecodeSession::default(),
        ))
    })?;
    Ok((engine, bundle))
}

#[cfg(all(feature = "native-cuda", target_os = "linux"))]
fn cuda_forward(mut forward: CpuForward, host_reserve: u64) -> Result<CpuForward, EngError> {
    forward.enable_cuda(host_reserve)?;
    Ok(forward)
}

#[cfg(not(all(feature = "native-cuda", target_os = "linux")))]
fn cuda_forward(_forward: CpuForward, _host_reserve: u64) -> Result<CpuForward, EngError> {
    Err(EngError::BadRequest(
        "Inkling CUDA serving requires a 64-bit Linux build with native-cuda".into(),
    ))
}

fn prefill_rows(backend: TextBackend) -> Result<usize, EngError> {
    const NAME: &str = "FIFI_INKLING_CUDA_PREFILL_ROWS";
    if backend != TextBackend::Cuda {
        return Ok(1);
    }
    let Some(value) = std::env::var_os(NAME) else {
        return Ok(1);
    };
    value
        .to_str()
        .and_then(|text| text.parse::<usize>().ok())
        .filter(|&rows| rows != 0)
        .ok_or_else(|| super::bad(format!("{NAME} must be a positive row count")))
}

fn supports_grouped_prefill(forward: &CpuForward) -> bool {
    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    return forward.supports_grouped_prefill();
    #[cfg(not(all(feature = "native-cuda", target_os = "linux")))]
    {
        let _ = forward;
        false
    }
}

impl TokenStepper for CpuForward {
    const OPERATION: &'static str = "inkling2_text";

    fn feed(&mut self, token: u32) -> Result<(), EngError> {
        self.token(token, false).map(|_| ())
    }

    fn feed_rows(&mut self, tokens: &[u32]) -> Result<(), EngError> {
        #[cfg(all(feature = "native-cuda", target_os = "linux"))]
        if tokens.len() > 1 && self.supports_grouped_prefill() {
            return self.prefill_grouped(tokens);
        }
        tokens.iter().try_for_each(|&token| self.feed(token))
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

pub(crate) struct TextSettings {
    pub(crate) context: u32,
    pub(crate) prefix_reuse: bool,
}

impl TextSettings {
    fn prefix_limits(&self) -> StoreLimits {
        if self.prefix_reuse {
            StoreLimits {
                roots: 1,
                entries: 2,
            }
        } else {
            StoreLimits {
                roots: 0,
                entries: 0,
            }
        }
    }

    fn host_prefill_plan(
        &self,
        forward: &CpuForward,
        requested_rows: usize,
    ) -> Result<(usize, u64), EngError> {
        let limits = self.prefix_limits();
        // Count every saved state at full size irrespective of shared KV rows.
        let state_bytes = forward
            .host_state_bytes()?
            .checked_mul((1 + limits.roots + limits.entries) as u64)
            .ok_or_else(|| EngError::MemoryPlan("Inkling host state size overflows".into()))?;
        #[cfg(target_os = "linux")]
        {
            const LABEL: &str = "Inkling text state";
            let state_needs_admission = self.context > DEFAULT_TEXT_CONTEXT || self.prefix_reuse;
            if state_needs_admission || requested_rows > 1 {
                let admission =
                    match crate::family::mem::HostFacts::read(LABEL, "Inkling") {
                        Ok(admission) => admission,
                        Err(error) if !state_needs_admission => {
                            eprintln!("Inkling grouped prefill unavailable: {error:?}");
                            return Ok((1, state_bytes));
                        }
                        Err(error) => return Err(error),
                    };
                if state_needs_admission {
                    admission.admit_payload(state_bytes, LABEL)?;
                }
                #[cfg(feature = "native-cuda")]
                if requested_rows > 1 {
                    return Ok(fit_prefill(
                        requested_rows,
                        forward.context,
                        state_bytes,
                        forward.prefill_workspace_bytes_per_row()?,
                        admission.payload_spare(),
                    ));
                }
            }
        }
        let _ = requested_rows;
        Ok((1, state_bytes))
    }
}

#[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
fn fit_prefill(
    requested: usize,
    context: usize,
    state_bytes: u64,
    bytes_per_row: u64,
    capacity: u64,
) -> (usize, u64) {
    let fit = capacity
        .saturating_sub(state_bytes)
        .checked_div(bytes_per_row)
        .unwrap_or(0);
    let rows = requested
        .min(context)
        .min(usize::try_from(fit).unwrap_or(usize::MAX));
    if rows < 2 {
        return (1, state_bytes);
    }
    // The product is bounded by the capacity above. Include it in the
    // reserve seen by the expert RAM warmer.
    (rows, state_bytes + rows as u64 * bytes_per_row)
}

struct TextExecutor {
    prefix: PrefixStore<PrefixState, ()>,
    forward: CpuForward,
    eos: Vec<u32>,
    prefill_rows: usize,
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

    fn prefill_piece(&self) -> Option<usize> {
        Some(self.prefill_rows)
    }

    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    fn prefill_bank_report(&self) -> Option<crate::sched::job::PrefillBankReport> {
        match &self.forward.weights {
            super::cpu_weights::Weights::Gguf(weights) => weights.prefill_bank_report(),
            _ => None,
        }
    }

    fn prefix_limits(&self) -> StoreLimits {
        self.prefix.limits()
    }

    fn save_prefix(
        &mut self,
        session: &Self::Session,
        key: &[u32],
        kind: SaveKind,
    ) -> Result<bool, EngError> {
        session.save_prefix(
            &mut self.prefix,
            key,
            (),
            self.forward.position(),
            kind,
            || Ok(self.forward.snapshot()),
        )
    }

    fn restore_prefix(
        &mut self,
        session: &mut Self::Session,
        prompt: &[u32],
    ) -> Result<usize, EngError> {
        session.restore_prefix(&mut self.prefix, prompt, (), |state| {
            self.forward.restore(state);
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::fit_prefill;

    #[test]
    fn grouped_prefill_fits_requested_width_and_context() {
        assert_eq!(fit_prefill(512, 768, 100, 20, 20_000), (512, 10_340));
        assert_eq!(fit_prefill(usize::MAX, 128, 100, 20, 20_000), (128, 2_660));
    }

    #[test]
    fn grouped_prefill_reserves_workspace_and_falls_back_under_pressure() {
        assert_eq!(fit_prefill(512, 768, 200, 80, 2_999), (34, 2_920));
        assert_eq!(fit_prefill(512, 768, 200, 80, 359), (1, 200));
        assert_eq!(fit_prefill(512, 768, 200, 80, 199), (1, 200));
    }
}
