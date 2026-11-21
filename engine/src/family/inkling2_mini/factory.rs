//! Explicit startup gate for the single-sequence Inkling text worker.

use super::{UNPADDED_VOCAB, state_plan};
use crate::{
    EngError,
    sched::worker::{Engine, ReadyNote},
    tokenizer::{TokenizerBundle, TokenizerPin},
};
use std::{ffi::OsStr, path::Path};

const OPT_IN: &str = "FIFI_INKLING2_MINI_MLX";
const GGUF_OPT_IN: &str = "FIFI_INKLING2_MINI_GGUF_MLX";

pub(crate) fn check_gguf_backend() -> Result<(), EngError> {
    check_backend()?;
    if std::env::var_os(GGUF_OPT_IN).as_deref() != Some(OsStr::new("1")) {
        return Err(EngError::BadRequest(format!(
            "Inkling 2 Mini experimental GGUF text serving requires {GGUF_OPT_IN}=1"
        )));
    }
    Ok(())
}

pub(crate) fn check_backend() -> Result<(), EngError> {
    check_backend_with(
        cfg!(all(
            feature = "native-mlx",
            target_os = "macos",
            target_arch = "aarch64"
        )),
        std::env::var_os(OPT_IN).as_deref(),
    )
}

fn check_backend_with(native: bool, opt_in: Option<&OsStr>) -> Result<(), EngError> {
    if !native {
        return Err(EngError::BadRequest(
            "Inkling 2 Mini requires --features native-mlx on Apple Silicon".into(),
        ));
    }
    if opt_in != Some(OsStr::new("1")) {
        return Err(EngError::BadRequest(format!(
            "Inkling 2 Mini experimental text serving requires {OPT_IN}=1"
        )));
    }
    Ok(())
}

pub(crate) fn context_limit(context: Option<u32>) -> Result<u32, EngError> {
    let limit = context.unwrap_or(super::DEFAULT_CONTEXT);
    state_plan(limit)?;
    Ok(limit)
}

// thinkingmachines/Inkling-2-Mini, revision ed40e7da625f594fc9c4990f5f66be24873c22d7.
const PUBLISHER_PIN: TokenizerPin = TokenizerPin {
    tokenizer_json: hex_digest(b"7f332727f9a3f093dd452a39cc978e6757627e5dad1b0673aa4e75174f236dd6"),
    tokenizer_config_json: hex_digest(
        b"bf09342d285139fd37ce10c70878af1d52cb390e9901e10248654293a80ee636",
    ),
    chat_template_jinja: hex_digest(
        b"53a9035f3255945704b5857657cfb00c1165090c0641c4f782a0cbf268cc16dc",
    ),
    generation_config_json: hex_digest(
        b"a08737c07a65a8fdf69f7cf086576210e5aad4b241437090b26d7b0c4f7905f7",
    ),
    vocab_size: UNPADDED_VOCAB as u32,
};

const fn hex_digest(hex: &[u8; 64]) -> [u8; 32] {
    const fn digit(byte: u8) -> u8 {
        match byte {
            b'0'..=b'9' => byte - b'0',
            b'a'..=b'f' => byte - b'a' + 10,
            _ => panic!("invalid tokenizer digest"),
        }
    }
    let mut digest = [0; 32];
    let mut index = 0;
    while index < digest.len() {
        digest[index] = digit(hex[2 * index]) * 16 + digit(hex[2 * index + 1]);
        index += 1;
    }
    digest
}

pub(crate) fn load_tokenizer(dir: &Path) -> Result<TokenizerBundle, EngError> {
    crate::storage_guard::require_local(dir, "Inkling tokenizer")?;
    let bundle = TokenizerBundle::load(dir, &PUBLISHER_PIN)?;
    if !bundle.load_notes().is_empty() {
        return Err(EngError::Checkpoint(
            "Inkling 2 Mini requires the pinned publisher tokenizer bundle".into(),
        ));
    }
    // The chat decoder dispatches these section tokens before ordinary text decoding.
    // Matching vocabulary sizes alone does not establish their numeric identity.
    for (marker, expected) in [
        ("<|content_text|>", 200_004),
        ("<|content_thinking|>", 200_008),
        ("<|end_message|>", 200_010),
    ] {
        if bundle.tokenizer().token_id(marker) != Some(expected) {
            return Err(EngError::Checkpoint(format!(
                "Inkling tokenizer expects {marker} at id {expected}"
            )));
        }
    }
    Ok(bundle)
}

pub(crate) fn start_engine(
    model_dir: &Path,
    tokenizer_dir: &Path,
    context: Option<u32>,
    ready_note: ReadyNote,
) -> Result<(Engine, TokenizerBundle, u32), EngError> {
    check_backend()?;
    let context = context_limit(context)?;
    launch(model_dir, tokenizer_dir, context, ready_note)
}

#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
fn launch(
    model_dir: &Path,
    tokenizer_dir: &Path,
    context: u32,
    ready_note: ReadyNote,
) -> Result<(Engine, TokenizerBundle, u32), EngError> {
    let load_start = executor::Phase::start_if_enabled();
    let root = model_dir.to_owned();
    let inspection = crate::capability::inspect::inspect(model_dir, tokenizer_dir)?;
    let gguf = inspection.inkling_gguf()?.map(|file| root.join(&file.file));
    if let Some(path) = &gguf {
        check_gguf_backend()?;
        eprintln!(
            "Inkling 2 Mini: experimental GGUF expert projections on Metal, model={}",
            path.display()
        );
    }
    let bundle = load_tokenizer(tokenizer_dir)?;
    let eos = bundle.eos_token_ids().to_vec();
    eprintln!("Inkling 2 Mini: experimental single-sequence text serving, context_limit={context}");
    let engine = crate::sched::worker::spawn_engine_noted(1, ready_note, move |_| {
        let decoder = match gguf {
            Some(path) => super::text::Decoder::open_gguf_native(&path, context)?,
            None => super::text::Decoder::open(&root, context)?,
        };
        if let Some(start) = load_start {
            start.report("load", decoder.expert_cache_stats(), &decoder);
        }
        Ok((
            executor::Executor::new(decoder, eos, context as usize),
            crate::family::session::DecodeSession::default(),
        ))
    })?;
    Ok((engine, bundle, context))
}

#[cfg(not(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64")))]
fn launch(
    _model_dir: &Path,
    _tokenizer_dir: &Path,
    _context: u32,
    _ready_note: ReadyNote,
) -> Result<(Engine, TokenizerBundle, u32), EngError> {
    check_backend()?;
    unreachable!("portable Inkling startup is rejected by check_backend")
}

#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
mod executor {
    use crate::{
        EngError,
        family::{
            executor::{ExecutorOps, GenerationState},
            session::{DecodeSession, Limits, TokenStepper, check_prompt},
        },
        tokenizer::DecodeBatch,
    };

    use super::super::{UNPADDED_VOCAB, mlx::MlxExpertCacheStats, text::Decoder, timing};

    pub(super) struct Phase {
        started: std::time::Instant,
        io: Option<(u64, u64)>,
        cache: MlxExpertCacheStats,
    }

    impl Phase {
        fn new(cache: MlxExpertCacheStats) -> Self {
            Self {
                started: std::time::Instant::now(),
                io: crate::kernels::mlx::native::process_io(),
                cache,
            }
        }

        pub(super) fn start_if_enabled() -> Option<Self> {
            Self::enabled().then(|| Self::new(MlxExpertCacheStats::default()))
        }

        fn enabled() -> bool {
            timing::enabled()
                || std::env::var_os("FIFI_INKLING_EXPERT_TIMING").is_some_and(|value| value == "1")
        }

        pub(super) fn report(&self, phase: &str, end: MlxExpertCacheStats, decoder: &Decoder) {
            let metal = decoder.gguf_metal_memory_stats();
            let io = crate::kernels::mlx::native::process_io();
            let disk = self.io.zip(io).and_then(|(a, b)| b.0.checked_sub(a.0));
            eprintln!(
                "inkling_phase {}",
                serde_json::json!({
                    "phase": phase, "elapsed_us": self.started.elapsed().as_micros(),
                    "process_disk_read_bytes": disk, "lifetime_peak_footprint_bytes": io.map(|v| v.1),
                    "expert_read_bytes": end.read_bytes - self.cache.read_bytes,
                    "expert_read_ranges": end.read_ranges - self.cache.read_ranges,
                    "cached_bytes": end.cached_bytes,
                    "mlx_process_active_bytes": metal.as_ref().map(|v| v.active),
                    "mlx_process_pool_bytes": metal.as_ref().map(|v| v.cache),
                    "mlx_process_peak_active_bytes": metal.as_ref().map(|v| v.peak),
                    "hits": end.hits - self.cache.hits, "misses": end.misses - self.cache.misses,
                    "evictions": end.evictions - self.cache.evictions,
                    "staging_allocations": end.staging_allocations - self.cache.staging_allocations,
                })
            );
        }
    }

    pub(super) struct Executor {
        stepper: Stepper,
        eos: Vec<u32>,
        context: usize,
        trace_cache: bool,
        phase: Option<Phase>,
        first_sample: bool,
        trace_start: Option<MlxExpertCacheStats>,
    }

    impl Executor {
        pub(super) fn new(decoder: Decoder, eos: Vec<u32>, context: usize) -> Self {
            Self {
                stepper: Stepper { decoder, fed: 0 },
                eos,
                context,
                trace_cache: Phase::enabled(),
                trace_start: None,
                phase: None,
                first_sample: false,
            }
        }

        fn report_cache_trace(&mut self, completed: bool) {
            let Some(start) = self.trace_start.take() else {
                return;
            };
            let end = self.stepper.decoder.expert_cache_stats();
            if let Some(phase) = self.phase.take() {
                let name = if self.first_sample {
                    "decode"
                } else {
                    "prefill"
                };
                phase.report(name, end, &self.stepper.decoder);
                self.stepper.decoder.report_component_timings(name);
            }
            eprintln!(
                "inkling_expert_timing {}",
                serde_json::json!({
                    "completed": completed,
                    "consumed_tokens": self.stepper.fed,
                    "visits": end.visits - start.visits,
                    "fully_cached_visits": end.fully_cached_visits - start.fully_cached_visits,
                    "hits": end.hits - start.hits,
                    "misses": end.misses - start.misses,
                    "evictions": end.evictions - start.evictions,
                    "read_bytes": end.read_bytes - start.read_bytes,
                    "read_ranges": end.read_ranges - start.read_ranges,
                    "staging_allocations": end.staging_allocations - start.staging_allocations,
                    "staging_fallback_syncs": end.staging_fallback_syncs - start.staging_fallback_syncs,
                    "cached_bytes": end.cached_bytes,
                })
            );
        }
    }

    struct Stepper {
        decoder: Decoder,
        fed: usize,
    }

    impl TokenStepper for Stepper {
        const OPERATION: &'static str = "Inkling 2 Mini text";

        fn feed(&mut self, token: u32) -> Result<(), EngError> {
            self.decoder.advance(token)?;
            self.fed += 1;
            Ok(())
        }

        fn feed_rows(&mut self, tokens: &[u32]) -> Result<(), EngError> {
            let result = self.decoder.advance_rows(tokens);
            self.fed = self.decoder.position();
            result
        }

        fn logits(&mut self, token: u32) -> Result<Vec<f32>, EngError> {
            let logits = self.decoder.step(token)?;
            self.fed += 1;
            Ok(logits)
        }

        fn fed(&self) -> usize {
            self.fed
        }

        fn reset(&mut self) -> Result<(), EngError> {
            self.decoder.reset();
            self.fed = 0;
            Ok(())
        }
    }

    impl ExecutorOps for Executor {
        type Session = DecodeSession;

        fn check_prompt(&self, prompt_tokens: usize) -> Result<(), EngError> {
            check_prompt(prompt_tokens, self.context)
        }

        fn history<'s>(&self, session: &'s Self::Session) -> &'s [u32] {
            session.history()
        }

        fn reset(&mut self, session: &mut Self::Session) -> Result<(), EngError> {
            self.report_cache_trace(false);
            session.reset(&mut self.stepper)
        }

        fn prefill_piece(&self) -> Option<usize> {
            Some(self.stepper.decoder.prefill_rows())
        }

        fn prefill(&mut self, session: &mut Self::Session, tokens: &[u32]) -> Result<(), EngError> {
            let result = session.prefill(&mut self.stepper, tokens, self.context);
            if result.is_err() {
                self.report_cache_trace(false);
            }
            result
        }

        fn decode(
            &mut self,
            session: &mut Self::Session,
            generation: &mut GenerationState,
        ) -> Result<DecodeBatch, EngError> {
            let fed = self.stepper.fed;
            let result = session.decode(
                &mut self.stepper,
                generation,
                &Limits {
                    eos: &self.eos,
                    sampler_vocab: UNPADDED_VOCAB,
                    context_limit: self.context,
                },
            );
            // DecodeSession leaves the final prompt token pending until the first
            // sampler call. Charge this feed and its output head to prefill.
            if !self.first_sample && self.phase.is_some() && self.stepper.fed > fed {
                let end = self.stepper.decoder.expert_cache_stats();
                if let Some(phase) = self.phase.take() {
                    phase.report("prefill", end, &self.stepper.decoder);
                    self.stepper.decoder.report_component_timings("prefill");
                }
                self.phase = Some(Phase::new(end));
                self.first_sample = true;
            }
            match &result {
                Ok(batch) if batch.finish.is_some() => self.report_cache_trace(true),
                Err(_) => self.report_cache_trace(false),
                _ => {}
            }
            result
        }

        fn boundary(&mut self, _session: &mut Self::Session) {
            if self.trace_cache {
                let cache = self.stepper.decoder.expert_cache_stats();
                self.trace_start = Some(cache);
                self.phase = Some(Phase::new(cache));
                self.first_sample = false;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires INKLING_TOKENIZER_FIXTURE with the pinned publisher files"]
    fn publisher_tokenizer_renders_plain_chat_and_thinking_off() {
        use crate::tokenizer::{PromptMessage, Role, Thinking};
        let path = std::env::var("INKLING_TOKENIZER_FIXTURE").unwrap();
        let bundle = load_tokenizer(Path::new(&path)).unwrap();
        let messages = [PromptMessage {
            role: Role::User,
            content: Some("Say hello.".into()),
            reasoning_content: None,
            name: None,
            tool_call_id: None,
            tool_calls: Vec::new(),
        }];
        let rendered = bundle.render_prompt(&messages, &[], Thinking::Off).unwrap();
        assert_eq!(
            rendered,
            "<|message_system|><|content_text|>Thinking effort level: 0<|end_message|><|message_user|><|content_text|>Say hello.<|end_message|>"
        );
        let ids = bundle.encode_prompt(&messages, &[], Thinking::Off).unwrap();
        assert_eq!(&ids[..2], &[200_002, 200_004]);
        assert_eq!(ids.last(), Some(&200_010));
        // The publisher template leaves channel selection to the model at every effort.
        let (off_ids, boundaries) = bundle
            .encode_prompt_with_boundary(&messages, &[], Thinking::Off)
            .unwrap();
        assert_eq!(off_ids, ids);
        assert_eq!(boundaries.tip, None);
        let default_render = bundle
            .render_prompt(&messages, &[], Thinking::Default)
            .unwrap();
        assert_eq!(
            default_render,
            rendered.replace("Thinking effort level: 0<", "Thinking effort level: 0.9<")
        );
        assert_eq!(bundle.eos_token_ids(), &[200_006]);
        let mut decoder = crate::tokenizer::IncrementalDecoder::inkling_chat(bundle.tokenizer());
        assert_eq!(decoder.push(200_008).unwrap(), "<|content_thinking|>");
        assert_eq!(decoder.push(200_010).unwrap(), "<|end_message|>");
        assert_eq!(decoder.push(200_004).unwrap(), "<|content_text|>");
    }

    #[test]
    fn serving_requires_native_mlx_and_explicit_opt_in() {
        assert!(check_backend_with(false, Some(OsStr::new("1"))).is_err());
        assert!(check_backend_with(true, None).is_err());
        assert!(check_backend_with(true, Some(OsStr::new("0"))).is_err());
        assert!(check_backend_with(true, Some(OsStr::new("1"))).is_ok());
    }

    #[test]
    fn context_defaults_to_the_short_plan_and_checks_bounds() {
        assert_eq!(context_limit(None).unwrap(), 1024);
        assert_eq!(context_limit(Some(4096)).unwrap(), 4096);
        assert!(context_limit(Some(1)).is_err());
        assert!(context_limit(Some(1_048_577)).is_err());
    }
}
