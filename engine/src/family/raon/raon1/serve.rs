//! The serving entry of the Raon-1 family: the engine start that `enginesrv` calls for a
//! `raon1` entry. The weights are one GGUF file; the tokenizer bundle comes from the same file.

use std::path::Path;

use crate::{
    EngError, family::flash_next::factory::StartedEngine, sched::worker::ReadyNote,
    tokenizer::TokenizerBundle,
};

/// The server-family key a config names to select this family.
pub(crate) const FAMILY: &str = "raon1";

/// Starts the engine thread for the GGUF at `model_path` (the file, or a directory with one
/// Raon-1 GGUF). `context_limit` is the configured limit; `None` derives it from device memory.
pub(crate) fn start_raon1_engine(
    model_path: &Path,
    context_limit: Option<u32>,
    bundle: TokenizerBundle,
    ready_note: ReadyNote,
) -> Result<StartedEngine, EngError> {
    let _ = (model_path, context_limit, bundle, ready_note);
    Err(EngError::BadRequest(
        "raon1 serve: the device path is not built yet".into(),
    ))
}
