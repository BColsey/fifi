//! SenseNova's host step plugs into the shared serial decode session.

use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

use crate::{
    EngError,
    family::{
        executor::SerialDecode,
        gguf_select,
        project::gguf::GgufProjector,
        session::{DecodeSession, TokenStepper},
    },
    sched::worker::{Engine, ReadyNote, spawn_engine_noted},
    tokenizer::{TokenizerBundle, gguf::bundle_from_lone_gguf},
    weights::{GgufLimits, GgufWeights, MetaValue, shared_header},
};

use super::{
    ARCHITECTURE,
    step::{self, LoopState},
    weights::LoopWeights,
};

// Every row is a synchronous host forward; allow cancellation between rows.
const PREFILL_PIECE: usize = 1;
// shortcut: cap the host f32 KV cache at 4K until larger contexts have a memory plan.
const HOST_CONTEXT_LIMIT: u32 = 4096;
const LIMITS: GgufLimits = GgufLimits {
    max_header_bytes: 32 << 20,
    max_tensor_bytes: 8 << 30,
};

pub(crate) struct StartedEngine {
    pub(crate) engine: Engine,
    pub(crate) bundle: TokenizerBundle,
    pub(crate) context_limit: u32,
}

/// An explicit file selects its own shard set. In a directory select only the text architecture.
pub(crate) fn select_gguf(model: &Path) -> Result<PathBuf, EngError> {
    crate::storage_guard::require_local(model, "SenseNova model")?;
    let (_, files) = gguf_select::candidates(model)?;
    let single = files.len() == 1;
    for file in files {
        let header = crate::storage_guard::require_local(&file.path, "SenseNova weights")
            .and_then(|()| shared_header(&file.path, LIMITS.max_header_bytes))
            .and_then(|header| {
                // Discovery may have cached a header with a larger cap.
                if header.header_len() > LIMITS.max_header_bytes {
                    return Err(EngError::Checkpoint(
                        "sensenova: GGUF header exceeds 32 MiB".into(),
                    ));
                }
                Ok(header)
            });
        match header {
            Ok(header) if matches!(header.meta("general.architecture"), Some(MetaValue::String(s)) if s == ARCHITECTURE) =>
            {
                return Ok(file.path);
            }
            Err(error) if single => return Err(error),
            _ => {}
        }
    }
    Err(EngError::Checkpoint(format!(
        "{} has no {ARCHITECTURE} GGUF weights",
        model.display()
    )))
}

fn context_limit(file: &Path) -> Result<u32, EngError> {
    let header = shared_header(file, LIMITS.max_header_bytes)?;
    match header.meta("sensenova_loop.context_length") {
        Some(MetaValue::U32(n)) if *n > 1 => Ok((*n).min(HOST_CONTEXT_LIMIT)),
        _ => Err(EngError::Checkpoint(
            "sensenova: invalid context_length".into(),
        )),
    }
}

pub(crate) fn start_engine(model: &Path, ready_note: ReadyNote) -> Result<StartedEngine, EngError> {
    let file = select_gguf(model)?;
    let context_limit = context_limit(&file)?;
    let bundle = bundle_from_lone_gguf(&file)?;
    let sampler_vocab = bundle.vocab_size() as usize;
    let eos = bundle.eos_token_ids().to_vec();
    let engine = spawn_engine_noted(PREFILL_PIECE, ready_note, move |_| {
        let step = LoopStep::open(&file)?;
        let executor = SerialDecode::new(step, eos, sampler_vocab, context_limit as usize);
        Ok((executor, DecodeSession::default()))
    })?;
    Ok(StartedEngine {
        engine,
        bundle,
        context_limit,
    })
}

pub(super) struct LoopStep {
    weights: LoopWeights,
    projector: GgufProjector,
    state: LoopState,
    pos: u32,
}

impl LoopStep {
    pub(super) fn open(file: &Path) -> Result<Self, EngError> {
        // Check each used matrix and vector in its host role, including BF16 matrices;
        // the GGUF reader's default admission list is for native CUDA kernels.
        let weights = LoopWeights::open(Arc::new(GgufWeights::open_any_type(file, LIMITS, None)?))?;
        let projector = weights.projector();
        let state = LoopState::new(weights.shape(), weights.plan());
        Ok(Self {
            weights,
            projector,
            state,
            pos: 0,
        })
    }

    fn advance(&mut self, token: u32) -> Result<Vec<f32>, EngError> {
        let hidden = step::trunk(
            &mut self.projector,
            &self.weights,
            &mut self.state,
            token,
            self.pos,
        )?;
        self.pos += 1;
        Ok(hidden)
    }
}

impl TokenStepper for LoopStep {
    const OPERATION: &'static str = "sensenova_logits";

    fn feed(&mut self, token: u32) -> Result<(), EngError> {
        self.advance(token).map(drop)
    }

    fn logits(&mut self, token: u32) -> Result<Vec<f32>, EngError> {
        let hidden = self.advance(token)?;
        step::logits(&mut self.projector, &self.weights, &hidden)
    }

    fn fed(&self) -> usize {
        self.pos as usize
    }

    fn reset(&mut self) -> Result<(), EngError> {
        self.state.clear();
        self.pos = 0;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn sensenova_refuses_a_cached_header_over_its_cap() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("large-header.gguf");
        let mut bytes = b"GGUF".to_vec();
        bytes.extend_from_slice(&3_u32.to_le_bytes());
        bytes.extend_from_slice(&0_u64.to_le_bytes()); // no tensors
        bytes.extend_from_slice(&2_u64.to_le_bytes()); // two metadata entries
        fn string(bytes: &mut Vec<u8>, value: &str) {
            bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
            bytes.extend_from_slice(value.as_bytes());
        }
        string(&mut bytes, "general.architecture");
        bytes.extend_from_slice(&8_u32.to_le_bytes()); // string
        string(&mut bytes, ARCHITECTURE);
        string(&mut bytes, "test.padding");
        bytes.extend_from_slice(&9_u32.to_le_bytes()); // array
        bytes.extend_from_slice(&0_u32.to_le_bytes()); // u8 elements
        bytes.extend_from_slice(&LIMITS.max_header_bytes.to_le_bytes());
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(&bytes).unwrap();
        let length = bytes.len() as u64 + LIMITS.max_header_bytes;
        file.set_len(length).unwrap();
        drop(file);

        let header = shared_header(&path, length).unwrap();
        assert_eq!(header.header_len(), length);
        let err = select_gguf(&path).unwrap_err();
        assert!(err.describe().contains("exceeds 32 MiB"), "{err:?}");
    }
}
