//! The serving wrap of the resident Bolt executor.
//!
//! One session per engine. The token side is the shared `DecodeSession`
//! (`family::session`): it owns the committed tokens, the pending token and
//! the finish rules (cancel, length, end of sequence, context limit). The
//! model side is a `TokenStepper` that feeds prompt rows cache-only and reads
//! the capped logits of a decode step; host sampling draws from that row.
//! Prompt rows run the cache-writing layer prefix only (the shared layers
//! after the prefix write no cache cells and feed nothing but the final norm);
//! the pending token's decode runs the head.

#[cfg(any(feature = "native-cuda", test))]
use std::{fs, path::Path, path::PathBuf};

#[cfg(any(feature = "native-cuda", test))]
use {super::config, crate::EngError};

/// Reads and validates the model directory's `config.json` against the Bolt
/// descriptor; the server starter uses this to dispatch inside the gemma4
/// family. A directory that is not a Bolt checkpoint fails the check.
#[cfg(any(feature = "native-cuda", test))]
pub(crate) fn bolt_configured(model_dir: &Path) -> Result<(), EngError> {
    let json = fs::read(model_dir.join("config.json"))
        .map_err(|error| EngError::Checkpoint(format!("gemma4 config.json: {error}")))?;
    config::check_bolt_config(&json)
}

/// The converted weights file: exactly one `.gguf` beside the config.
#[cfg(any(feature = "native-cuda", test))]
fn find_gguf(dir: &Path) -> Result<PathBuf, EngError> {
    let mut found = Vec::new();
    for entry in fs::read_dir(dir)
        .map_err(|error| EngError::Checkpoint(format!("model directory: {error}")))?
    {
        let path = entry
            .map_err(|error| EngError::Checkpoint(format!("model directory: {error}")))?
            .path();
        if path
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"))
        {
            found.push(path);
        }
    }
    match found.len() {
        1 => Ok(found.pop().expect("length checked")),
        0 => Err(bad("no .gguf weights file in the model directory")),
        n => Err(bad(&format!(
            "{n} .gguf files in the model directory; exactly one expected"
        ))),
    }
}

#[cfg(any(feature = "native-cuda", test))]
fn bad(message: &str) -> EngError {
    EngError::BadRequest(format!("gemma4 serve: {message}"))
}

#[cfg(test)]
mod tests {
    use super::{bolt_configured, find_gguf};

    const BOLT_CONFIG: &[u8] = include_bytes!("../../../testdata/gemma4/bolt-config.json");

    fn scratch(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("gemma4-serve-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("scratch");
        dir
    }

    #[test]
    fn find_gguf_requires_exactly_one_file() {
        let dir = scratch("none");
        assert!(find_gguf(&dir).is_err());
        std::fs::write(dir.join("bolt.gguf"), b"xx").expect("write");
        assert!(find_gguf(&dir).is_ok());
        std::fs::write(dir.join("other.gguf"), b"xx").expect("write");
        assert!(find_gguf(&dir).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn bolt_configured_reads_the_pinned_config_shape() {
        let dir = scratch("config");
        assert!(bolt_configured(&dir).is_err());
        std::fs::write(dir.join("config.json"), BOLT_CONFIG).expect("write");
        assert!(bolt_configured(&dir).is_ok());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_bolt_directory_passes_the_config_gate_and_finds_its_gguf() {
        // Checks the two helpers start_gemma4 dispatches on: bolt_configured
        // accepts the pinned Bolt config, and find_gguf returns the single weights
        // file. The server's engine start itself needs a device and is not run here.
        let dir = scratch("dispatch");
        std::fs::write(dir.join("config.json"), BOLT_CONFIG).expect("write config");
        std::fs::write(dir.join("bolt-e4b-q8_0.gguf"), b"weights").expect("write weights");
        bolt_configured(&dir).expect("the Bolt config gate passes");
        let weights = find_gguf(&dir).expect("find_gguf returns the weights file");
        assert_eq!(weights, dir.join("bolt-e4b-q8_0.gguf"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}

#[cfg(feature = "native-cuda")]
mod device {
    use std::path::Path;

    use super::super::resident::bolt::{self, Bolt, CELLS, VOCAB};
    use crate::{
        EngError,
        family::{
            executor::{ExecutorOps, GenerationState},
            flash_next::factory::StartedEngine,
            session::{DecodeSession, Limits, TokenStepper},
        },
        sched::worker::{ReadyNote, spawn_engine_noted},
        tokenizer::DecodeBatch,
    };

    /// The token side is the shared `DecodeSession`; `pos` is the model cache
    /// position (the number of tokens fed). The finish rules live in
    /// `family::session`, not here.
    pub(crate) struct BoltSession {
        core: DecodeSession,
        pos: usize,
    }

    pub(crate) struct BoltServe {
        model: Bolt,
        eos: Vec<u32>,
    }

    /// The model side of one session: feeds prompt rows cache-only and reads
    /// the capped logits of a decode step. The session never holds it.
    struct Step<'a> {
        model: &'a mut Bolt,
        pos: &'a mut usize,
    }

    impl TokenStepper for Step<'_> {
        const OPERATION: &'static str = "gemma4_bolt";

        fn feed(&mut self, token: u32) -> Result<(), EngError> {
            self.model.prefill_batch(&[token], *self.pos, false)?;
            *self.pos += 1;
            Ok(())
        }

        fn feed_rows(&mut self, tokens: &[u32]) -> Result<(), EngError> {
            for chunk in tokens.chunks(Bolt::PREFILL_BATCH) {
                self.model.prefill_batch(chunk, *self.pos, false)?;
                *self.pos += chunk.len();
            }
            Ok(())
        }

        fn logits(&mut self, token: u32) -> Result<Vec<f32>, EngError> {
            self.model.decode(token, *self.pos)?;
            *self.pos += 1;
            self.model.capped_logits()
        }

        fn fed(&self) -> usize {
            *self.pos
        }

        fn reset(&mut self) -> Result<(), EngError> {
            // A replay from position 0 overwrites every cache cell it uses.
            *self.pos = 0;
            Ok(())
        }
    }

    impl ExecutorOps for BoltServe {
        type Session = BoltSession;

        fn history<'s>(&self, st: &'s Self::Session) -> &'s [u32] {
            st.core.history()
        }

        fn reset(&mut self, st: &mut Self::Session) -> Result<(), EngError> {
            st.core.reset(&mut Step {
                model: &mut self.model,
                pos: &mut st.pos,
            })
        }

        fn check_prompt(&self, prompt_tokens: usize) -> Result<(), EngError> {
            crate::family::session::check_prompt(prompt_tokens, CELLS)
        }

        fn prefill(&mut self, st: &mut Self::Session, toks: &[u32]) -> Result<(), EngError> {
            st.core.prefill(
                &mut Step {
                    model: &mut self.model,
                    pos: &mut st.pos,
                },
                toks,
                CELLS,
            )
        }

        fn decode(
            &mut self,
            st: &mut Self::Session,
            generation: &mut GenerationState,
        ) -> Result<DecodeBatch, EngError> {
            let limits = Limits {
                eos: &self.eos,
                sampler_vocab: VOCAB,
                context_limit: CELLS,
            };
            st.core.decode(
                &mut Step {
                    model: &mut self.model,
                    pos: &mut st.pos,
                },
                generation,
                &limits,
            )
        }

        fn boundary(&mut self, _st: &mut Self::Session) {}
    }

    pub(crate) fn start_bolt_engine(
        model_dir: &Path,
        tokenizer_dir: &Path,
        ready_note: ReadyNote,
    ) -> Result<StartedEngine, EngError> {
        let bundle = crate::tokenizer::gemma::load(tokenizer_dir)?;
        let eos = bundle.eos_token_ids().to_vec();
        if eos.is_empty() || eos.iter().any(|&id| id as usize >= VOCAB) {
            return Err(super::bad(
                "EOS ids must be non-empty and within vocabulary",
            ));
        }
        if bundle.vocab_size() as usize != VOCAB {
            return Err(super::bad(
                "the tokenizer vocabulary differs from the checkpoint vocabulary",
            ));
        }
        let model = model_dir.to_path_buf();
        let engine = spawn_engine_noted(Bolt::PREFILL_BATCH, ready_note, move |_| {
            let weights = super::find_gguf(&model)?;
            let serve = BoltServe {
                model: bolt::load(&weights, None)?,
                eos: eos.clone(),
            };
            Ok((
                serve,
                BoltSession {
                    core: DecodeSession::default(),
                    pos: 0,
                },
            ))
        })?;
        Ok(StartedEngine {
            engine,
            bundle,
            context_limit: u32::try_from(CELLS).expect("cells fit u32"),
        })
    }
}

#[cfg(feature = "native-cuda")]
pub(crate) use device::start_bolt_engine;
