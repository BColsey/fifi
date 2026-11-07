//! The embedding row of a token, read on the host.
//!
//! WHAT: [`embedding_row`] reads leading slice `token` of the pack's BF16 `embed_tokens` with
//! `StreamedCheckpoint::read_slice_into` and widens it to F32 (exact); [`stream_row`] repeats it
//! across the hyper-connection branches, the initial residual stream of one token.
//! WHY this form: it is the gather every other family uses (Flint's `embedding_rows`, the Flint EXL3
//! loader, the Qwen3.6 host path): one positioned read of 5 KiB per token, served from the page
//! cache once warm, so the 1.18 GiB table is never staged and needs no second copy in RAM.
//! The step adds `branches` copies of the row (`step.rs`, `run`), which the device takes as one
//! 40 KiB upload.

use crate::{
    EngError,
    family::{exl3_pack::EMBED_NAME, qwen36::exl3_tensors::widen_f32},
    tensor::{Dtype, Tensor},
    weights::StreamedCheckpoint,
};

/// The F32 row of `token`: `hidden` values. Refuses a token outside the vocabulary and a table that
/// is not BF16 `[vocab, hidden]`.
pub(in crate::family) fn embedding_row(
    source: &StreamedCheckpoint,
    token: u32,
    hidden: usize,
) -> Result<Vec<f32>, EngError> {
    let (dtype, shape) = source
        .stored(EMBED_NAME)
        .ok_or_else(|| EngError::Checkpoint(format!("missing tensor {EMBED_NAME}")))?;
    if dtype != Dtype::Bf16 || shape.len() != 2 || shape[1] as usize != hidden {
        return Err(EngError::Checkpoint(format!(
            "{EMBED_NAME} is {dtype:?} {shape:?}, the gather needs BF16 [vocab, {hidden}]"
        )));
    }
    let mut bytes = vec![0u8; hidden * 2];
    source.read_slice_into(EMBED_NAME, token, &mut bytes)?;
    let row = Tensor::try_from_host(Dtype::Bf16, vec![hidden as u32], bytes)?;
    widen_f32(&row, hidden, EMBED_NAME)
}

/// The initial residual stream of `token`: its row once per branch, `branches * hidden` values.
pub(in crate::family) fn stream_row(
    source: &StreamedCheckpoint,
    token: u32,
    hidden: usize,
    branches: usize,
) -> Result<Vec<f32>, EngError> {
    let row = embedding_row(source, token, hidden)?;
    Ok(row.repeat(branches))
}

#[cfg(test)]
mod tests;
