//! Opt-in research capture of block 19's raw residual for the final prompt token.
//!
//! Identities are declared frozen inputs, not proof of the files they describe. A collection
//! harness must verify them against its frozen files before loading a model. Records contain
//! prompt-derived information; this module has no filesystem, network or serving export.

use crate::EngError;
use serde::Serialize;
use sha2::{Digest, Sha256};

type Result<T> = std::result::Result<T, EngError>;
pub const LAYER: usize = 19;
pub const MAX_HIDDEN: usize = 8192;
pub const MAX_PROMPT_TOKENS: usize = 262_144;
pub const MAX_ARTIFACT_BYTES: usize = 256 * 1024;

#[derive(Debug, Clone, Serialize)]
pub struct Fingerprints {
    model_sha256: String,
    tokenizer_sha256: String,
    source_sha256: String,
}

impl Fingerprints {
    pub fn new(model: String, tokenizer: String, source: String) -> Result<Self> {
        for value in [&model, &tokenizer, &source] {
            check_sha256(value)?;
        }
        Ok(Self {
            model_sha256: model,
            tokenizer_sha256: tokenizer,
            source_sha256: source,
        })
    }
}

/// The supported path is always resident, eager, nonspeculative CUDA. The native adapter
/// checks those conditions rather than changing the session's configured execution mode.
#[derive(Debug, Clone, Serialize)]
pub struct ExecutionSettings {
    pub weight_format: String,
    pub kv_dtype: String,
    pub context_tokens: usize,
    pub prefill_chunk_tokens: usize,
    /// The arm the prompt feed took: `chunk`, `per_token`, or `unset` before any feed or after
    /// a state import. Part of the frozen execution fingerprint either way.
    pub prefill_path: String,
    pub prefill_pieces_sha256: String,
    pub sync_each: bool,
    pub section_marks: bool,
    pub device_shortlist: bool,
    pub prefix_reused: bool,
}

impl ExecutionSettings {
    pub fn fingerprint(&self) -> Result<String> {
        if self.weight_format.is_empty()
            || self.weight_format.len() > 32
            || self.kv_dtype.is_empty()
            || self.kv_dtype.len() > 32
            || self.prefill_path.is_empty()
            || self.prefill_path.len() > 32
            || self.context_tokens == 0
            || self.prefill_chunk_tokens == 0
        {
            return Err(refusal("invalid or oversized execution settings"));
        }
        check_sha256(&self.prefill_pieces_sha256)?;
        Ok(sha256(&encode(self)?))
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct CaptureRequest {
    fingerprints: Fingerprints,
    prompt_sha256: String,
    prompt_tokens: usize,
    token_id: u32,
    hidden: usize,
    execution_sha256: String,
}

/// A snapshot of the shared session boundary, before its pending token is forwarded.
#[derive(Clone, Copy)]
pub struct PromptBoundary<'a> {
    pub history: &'a [u32],
    pub prompt_len: usize,
    pub pending: Option<u32>,
    pub fed: usize,
}

impl CaptureRequest {
    pub fn new(
        ids: Fingerprints,
        prompt: &[u32],
        hidden: usize,
        execution: String,
    ) -> Result<Self> {
        if prompt.is_empty()
            || prompt.len() > MAX_PROMPT_TOKENS
            || !(1..=MAX_HIDDEN).contains(&hidden)
        {
            return Err(refusal("prompt or hidden width outside capture bounds"));
        }
        check_sha256(&execution)?;
        Ok(Self {
            fingerprints: ids,
            prompt_sha256: prompt_sha256(prompt),
            prompt_tokens: prompt.len(),
            token_id: prompt[prompt.len() - 1],
            hidden,
            execution_sha256: execution,
        })
    }

    pub fn check_pending(&self, boundary: PromptBoundary<'_>) -> Result<()> {
        if boundary.prompt_len != self.prompt_tokens
            || boundary.history.len() != self.prompt_tokens
            || boundary.pending != Some(self.token_id)
            || boundary.fed != self.prompt_tokens - 1
            || prompt_sha256(boundary.history) != self.prompt_sha256
        {
            return Err(refusal(
                "capture requires the unfed final prompt token, before any answer",
            ));
        }
        Ok(())
    }

    pub fn hidden(&self) -> usize {
        self.hidden
    }
}

/// A one-shot owned observation. Any failed observation invalidates the whole attempt,
/// even if the caller ignores that error or the forward fails after the copy succeeded.
pub struct Capture {
    request: CaptureRequest,
    settings: ExecutionSettings,
    values: Option<Box<[f32]>>,
    failed: bool,
}

impl Capture {
    pub fn new(request: CaptureRequest, settings: ExecutionSettings) -> Result<Self> {
        if settings.fingerprint()? != request.execution_sha256 {
            return Err(refusal(
                "execution settings do not match the frozen request",
            ));
        }
        Ok(Self {
            request,
            settings,
            values: None,
            failed: false,
        })
    }

    pub fn observe(
        &mut self,
        layer: usize,
        position: usize,
        read: impl FnOnce() -> Result<Vec<f32>>,
    ) -> Result<()> {
        if layer != LAYER {
            return Ok(());
        }
        let already_failed = self.failed;
        self.failed = true;
        if already_failed || self.values.is_some() || position != self.request.prompt_tokens - 1 {
            return Err(refusal(
                "duplicate, failed or misaligned activation observation",
            ));
        }
        let values = read()?;
        if values.len() != self.request.hidden || values.iter().any(|v| !v.is_finite()) {
            return Err(refusal(
                "activation has the wrong F32 shape or nonfinite values",
            ));
        }
        self.values = Some(values.into_boxed_slice());
        self.failed = false;
        Ok(())
    }

    pub fn finish<T>(self, forward: Result<T>) -> Result<(T, ActivationRecord)> {
        let output = forward?;
        if self.failed {
            return Err(refusal("activation observation failed"));
        }
        let values = self
            .values
            .ok_or_else(|| refusal("forward did not capture block 19"))?;
        Ok((
            output,
            ActivationRecord {
                schema_version: 1,
                layer: LAYER,
                position: self.request.prompt_tokens - 1,
                representation: "raw_post_block_residual",
                dtype: "f32",
                execution_path: "qwen36_cuda_eager",
                request: self.request,
                settings: self.settings,
                values,
            },
        ))
    }
}

/// Immutable and independent of every reusable model arena and session lifetime.
#[derive(Debug, Serialize)]
pub struct ActivationRecord {
    schema_version: u32,
    layer: usize,
    position: usize,
    representation: &'static str,
    dtype: &'static str,
    execution_path: &'static str,
    request: CaptureRequest,
    settings: ExecutionSettings,
    values: Box<[f32]>,
}

impl ActivationRecord {
    pub fn values(&self) -> &[f32] {
        &self.values
    }
    pub fn to_json(&self) -> Result<Vec<u8>> {
        encode(self)
    }
    pub fn fingerprint(&self) -> Result<String> {
        Ok(sha256(&self.to_json()?))
    }
}

fn encode(value: &impl Serialize) -> Result<Vec<u8>> {
    let bytes = serde_json::to_vec(value).map_err(|e| refusal(&format!("serialize: {e}")))?;
    if bytes.len() > MAX_ARTIFACT_BYTES {
        return Err(refusal("artifact exceeds byte bound"));
    }
    Ok(bytes)
}

fn check_sha256(value: &str) -> Result<()> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return Err(refusal(
            "fingerprints must be 64 lowercase hexadecimal SHA256 characters",
        ));
    }
    Ok(())
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// Fingerprints actual feed_rows call boundaries and the imported starting row.
pub fn prefill_fingerprint(start_rows: usize, feed_rows: &[usize]) -> String {
    let mut digest = Sha256::new();
    digest.update(b"qwen36-activation-prefill-v1\0");
    digest.update((start_rows as u64).to_le_bytes());
    for &rows in feed_rows {
        digest.update((rows as u64).to_le_bytes());
    }
    format!("{:x}", digest.finalize())
}

fn prompt_sha256(tokens: &[u32]) -> String {
    let mut digest = Sha256::new();
    digest.update(b"qwen36-activation-prompt-v1\0");
    for token in tokens {
        digest.update(token.to_le_bytes());
    }
    format!("{:x}", digest.finalize())
}

pub(crate) fn refusal(message: &str) -> EngError {
    EngError::BadRequest(format!("research activation: {message}"))
}

#[cfg(test)]
#[path = "activation/session_tests.rs"]
mod session_tests;
#[cfg(test)]
mod tests;
