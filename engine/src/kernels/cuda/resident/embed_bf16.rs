//! The BF16 token-embedding gather of the EXL3 step: one launch reading `m` token ids on the
//! device and writing `m` rows of `k` f32 values from the table as the checkpoint stores it.
//!
//! [`CudaContext::embed_bf16`] queues one kernel on a [`DeviceStream`] and returns. It allocates
//! nothing, copies nothing and never synchronizes. The ids are read when the kernel runs, so the
//! token a step writes into its parameter block needs no host copy of the table. The widening is
//! exact. An id outside the table writes a row of NaN, reads no table byte and ORs status bit 4
//! into `status`, the rule of the quantized gather ([`CudaContext::embed`]).
//!
//! Every refusal is a `BadRequest` and happens before any native call: the backend is usable;
//! `m` is in `1..=4096`; `k` is a multiple of 8 in `8..=32768`; `n_rows` is in `1..=2^28`; the
//! ranges hold their bytes at 16-byte (`x`) and 4-byte alignment; `x` and `status` overlap
//! nothing else.

use crate::EngError;
use crate::kernels::cuda::context::{
    Claim, CudaContext, DeviceRange, DeviceStream, require_apart, require_fit,
};
use crate::kernels::cuda::native::resident::embed_bf16::{EmbedBf16Raw, embed_bf16_launch};

/// The most tokens one gather takes.
pub(crate) const EMBED_BF16_MAX_TOKENS: usize = 4096;

/// One gather: the table, the ids, the output rows and the step status word.
pub(crate) struct EmbedBf16Call<'b> {
    /// `n_rows` rows of `k` BF16 codes.
    pub(crate) table: DeviceRange<'b>,
    /// `m` device ints, read when the kernel runs.
    pub(crate) ids: DeviceRange<'b>,
    /// `m` rows of `k` floats, written in full.
    pub(crate) x: DeviceRange<'b>,
    /// The step status word; a bad id ORs bit 4 into it.
    pub(crate) status: DeviceRange<'b>,
    pub(crate) m: usize,
    pub(crate) n_rows: usize,
    pub(crate) k: usize,
}

fn bytes(factors: &[usize]) -> Result<usize, EngError> {
    factors
        .iter()
        .try_fold(1usize, |total, &factor| total.checked_mul(factor))
        .ok_or_else(|| EngError::BadRequest("the embed_bf16 sizes overflow".to_owned()))
}

impl CudaContext {
    /// Queues the gather of `call` on `stream`, writing `call.x`.
    pub(crate) fn embed_bf16(
        &self,
        stream: &DeviceStream,
        call: EmbedBf16Call<'_>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let raw = self.plan_embed_bf16(&call)?;
        self.launch_on(stream, |stream_raw| embed_bf16_launch(stream_raw, &raw))
    }

    fn plan_embed_bf16<'b>(&self, call: &EmbedBf16Call<'b>) -> Result<EmbedBf16Raw<'b>, EngError> {
        let raw = EmbedBf16Raw {
            w: self.raw_range(&call.table)?,
            ids: self.raw_range(&call.ids)?,
            x: self.raw_range(&call.x)?,
            status: self.raw_range(&call.status)?,
            m: call.m,
            n_rows: call.n_rows,
            k: call.k,
        };
        if !(1..=EMBED_BF16_MAX_TOKENS).contains(&call.m) {
            return Err(EngError::BadRequest(format!(
                "the embed_bf16 takes 1 to {EMBED_BF16_MAX_TOKENS} tokens, got {}",
                call.m
            )));
        }
        if !(8..=32_768).contains(&call.k) || !call.k.is_multiple_of(8) {
            return Err(EngError::BadRequest(format!(
                "the embed_bf16 needs k a multiple of 8 in 8..=32768, got {}",
                call.k
            )));
        }
        if !(1..=(1 << 28)).contains(&call.n_rows) {
            return Err(EngError::BadRequest(format!(
                "the embed_bf16 table has {} rows, outside 1..=2^28",
                call.n_rows
            )));
        }
        let (x_bytes, w_bytes, ids_bytes) = (
            bytes(&[call.m, call.k, 4])?,
            bytes(&[call.n_rows, call.k, 2])?,
            bytes(&[call.m, 4])?,
        );
        require_fit("table", &call.table, w_bytes, 4)?;
        require_fit("ids", &call.ids, ids_bytes, 4)?;
        require_fit("x", &call.x, x_bytes, 16)?;
        require_fit("status", &call.status, 4, 4)?;
        let (x, status) = (
            Claim::of("x", &call.x, x_bytes),
            Claim::of("status", &call.status, 4),
        );
        for read in [
            Claim::of("table", &call.table, w_bytes),
            Claim::of("ids", &call.ids, ids_bytes),
        ] {
            require_apart(&x, &read)?;
            require_apart(&status, &read)?;
        }
        require_apart(&x, &status)?;
        Ok(raw)
    }
}

#[cfg(test)]
mod tests;
