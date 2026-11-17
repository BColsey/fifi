//! The split-half RoPE of the Gemma 4 Humanizer resident step: one launch rotating
//! the q and k rows in place at consecutive positions. Local layers use head_dim 256 with base
//! 1e4 over all pairs; global layers use head_dim 512 with base 1e6 over the first
//! 64 pairs and zero frequency for the rest. The optional f32 factor table divides
//! the frequency; a factor of 1e30 or more forces the angle to zero.
//!
//! [CudaContext::rope_gemma] queues the kernel on a [DeviceStream] and returns when
//! it is queued. It allocates nothing, copies nothing and never synchronizes.
//!
//! Every refusal is a BadRequest and happens before any native call: the backend is
//! usable, rows are in 1..=16, query heads 16 with kv heads 1 or 8, or 8 with 2,
//! head_dim matches
//! the mode (256 local, 512 global), the ranges hold their rows at 4-byte alignment,
//! and q shares no byte with k. A refused call changes nothing.

use crate::EngError;
use crate::kernels::cuda::context::{
    Claim, CudaContext, DeviceRange, DeviceStream, require_apart, require_fit,
};
use crate::kernels::cuda::native::resident::rope_gemma::{RopeGemmaRaw, rope_gemma_launch};

/// Q and K rows in place; row t uses position word + t and the shared factor table.
pub(crate) struct RopeGemmaCall<'b> {
    pub(crate) q: DeviceRange<'b>,
    pub(crate) k: DeviceRange<'b>,
    pub(crate) pos: DeviceRange<'b>,
    pub(crate) factors: Option<DeviceRange<'b>>,
    pub(crate) rows_q: usize,
    pub(crate) rows_k: usize,
    pub(crate) n_q_head: usize,
    pub(crate) n_kv_head: usize,
    pub(crate) head_dim: usize,
    pub(crate) global_mode: bool,
}

/// The RoPE head_dim/mode pairs the kernel admits: 256 local or 512 global. Pure
/// geometry checked before any native call, so the admission is testable on the host.
pub(crate) fn admit_rope_head_dim(head_dim: usize, global_mode: bool) -> Result<(), EngError> {
    if !matches!((head_dim, global_mode), (256, false) | (512, true)) {
        return Err(EngError::BadRequest(format!(
            "the rope_gemma needs head_dim 256 local or 512 global, got {head_dim} \
             global_mode {global_mode}"
        )));
    }
    Ok(())
}

impl CudaContext {
    pub(crate) fn rope_gemma(
        &self,
        stream: &DeviceStream,
        call: RopeGemmaCall<'_>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let raw = self.plan_rope_gemma(&call)?;
        self.launch_on(stream, |stream_raw| rope_gemma_launch(stream_raw, &raw))
    }

    fn plan_rope_gemma<'b>(&self, call: &RopeGemmaCall<'b>) -> Result<RopeGemmaRaw<'b>, EngError> {
        let q_raw = self.raw_range(&call.q)?;
        let k_raw = self.raw_range(&call.k)?;
        let pos_raw = self.raw_range(&call.pos)?;
        let factors_raw = match &call.factors {
            Some(range) => Some(self.raw_range(range)?),
            None => None,
        };
        let q_bytes = call
            .rows_q
            .checked_mul(call.n_q_head)
            .and_then(|v| v.checked_mul(call.head_dim))
            .and_then(|v| v.checked_mul(4))
            .ok_or_else(|| EngError::BadRequest("the rope_gemma size overflows".to_owned()))?;
        let k_bytes = call
            .rows_k
            .checked_mul(call.n_kv_head)
            .and_then(|v| v.checked_mul(call.head_dim))
            .and_then(|v| v.checked_mul(4))
            .ok_or_else(|| EngError::BadRequest("the rope_gemma size overflows".to_owned()))?;
        if !(1..=16).contains(&call.rows_q) || !(1..=16).contains(&call.rows_k) {
            return Err(EngError::BadRequest(format!(
                "the rope_gemma takes 1 to 16 rows, got q {} k {}",
                call.rows_q, call.rows_k
            )));
        }
        super::attn_decode_gemma::admit_gemma_heads(call.n_q_head, call.n_kv_head)?;
        admit_rope_head_dim(call.head_dim, call.global_mode)?;
        require_fit("q", &call.q, q_bytes, 4)?;
        require_fit("k", &call.k, k_bytes, 4)?;
        require_fit("pos", &call.pos, 4, 4)?;
        if let Some(factors) = &call.factors {
            require_fit("factors", factors, call.head_dim / 2 * 4, 4)?;
        }
        require_apart(
            &Claim::of("q", &call.q, q_bytes),
            &Claim::of("k", &call.k, k_bytes),
        )?;
        for (name, range, bytes) in [("q", &call.q, q_bytes), ("k", &call.k, k_bytes)] {
            let output = Claim::of(name, range, bytes);
            require_apart(&output, &Claim::of("pos", &call.pos, 4))?;
            if let Some(factors) = &call.factors {
                require_apart(
                    &output,
                    &Claim::of("factors", factors, call.head_dim / 2 * 4),
                )?;
            }
        }
        Ok(RopeGemmaRaw {
            q: q_raw,
            k: k_raw,
            pos: pos_raw,
            factors: factors_raw,
            rows_q: call.rows_q,
            rows_k: call.rows_k,
            n_q_head: call.n_q_head,
            n_kv_head: call.n_kv_head,
            head_dim: call.head_dim,
            global_mode: call.global_mode,
        })
    }
}

#[cfg(test)]
mod tests;
