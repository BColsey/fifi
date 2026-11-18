//! The Gemma 4 Humanizer decode attention over the KV8 planes: one single-pass
//! launch per call. GQA maps query head h to kv head h*n_kv_head/n_q_head (2:1 on
//! local layers, 16:1 on global layers of the Humanizer, 4:1 on the Bolt layers);
//! head_dim is 256 local or 512 global; the
//! window cuts the keys to the last W positions (0 means all); the scale is a
//! parameter (Gemma uses 1.0); there is no output gate. The device kv_len word is
//! authoritative: token t sees keys 0..kv_len+t-1, out is NaN and status bit 4 is
//! set when kv_len is out of range. `split: None` runs the v0 single-pass kernel;
//! `Some((part, max_splits, min_keys))` runs the part+combine split path over
//! `part` (sized by attn_decode_gemma_part_floats), which changes the softmax
//! summation order and is not bit-identical to the single-pass kernel.
//!
//! [CudaContext::attn_decode_gemma] queues the kernel on a [DeviceStream] and
//! returns when it is queued. It allocates nothing, copies nothing and never
//! synchronizes.
//!
//! Every refusal is a BadRequest and happens before any native call: the backend is
//! usable; m is in 1..=16; query heads 16 with kv heads 1 or 8, or 8 with 2;
//! head_dim is 256 or
//! 512; cells is positive; the scale is finite below 1e30 in magnitude; every range
//! holds its bytes at 4-byte alignment; out overlaps none of q, the K code plane or
//! the V code plane. A refused call changes nothing.

use crate::EngError;
use crate::kernels::cuda::context::{
    Claim, CudaContext, DeviceRange, DeviceStream, require_apart, require_fit,
};
use crate::kernels::cuda::native::resident::attn_decode_gemma::{
    AttnDecodeGemmaRaw, attn_decode_gemma_launch, attn_decode_gemma_part_floats,
};

/// The split plan of a call that opted into the split path: the partial buffer,
/// its split capacity and the keys-per-split floor. `part` must be 16-byte
/// aligned and sized by [attn_decode_gemma_part_floats]; the resident owns it.
#[derive(Clone, Copy)]
pub(crate) struct AttnDecodeGemmaSplit<'b> {
    pub(crate) part: DeviceRange<'b>,
    pub(crate) max_splits: usize,
    pub(crate) min_keys: usize,
}

pub(crate) struct AttnDecodeGemmaCall<'b> {
    pub(crate) out: DeviceRange<'b>,
    pub(crate) status: Option<DeviceRange<'b>>,
    pub(crate) q: DeviceRange<'b>,
    pub(crate) k_codes: DeviceRange<'b>,
    pub(crate) k_scales: DeviceRange<'b>,
    pub(crate) v_codes: DeviceRange<'b>,
    pub(crate) v_scales: DeviceRange<'b>,
    pub(crate) kv_len: DeviceRange<'b>,
    pub(crate) m: usize,
    pub(crate) n_q_head: usize,
    pub(crate) n_kv_head: usize,
    pub(crate) head_dim: usize,
    pub(crate) cells: usize,
    pub(crate) window: usize,
    pub(crate) scale: f32,
    pub(crate) split: Option<AttnDecodeGemmaSplit<'b>>,
}

/// The query/kv head shapes the Gemma attention and RoPE kernels admit: 16 query
/// heads with 1 or 8 kv heads (the Humanizer local/global layers), or 8 with 2
/// (the Bolt layers). Pure geometry checked before any native call, so the
/// admission is testable on the host.
pub(crate) fn admit_gemma_heads(n_q_head: usize, n_kv_head: usize) -> Result<(), EngError> {
    if !matches!((n_q_head, n_kv_head), (16, 1) | (16, 8) | (8, 2)) {
        return Err(EngError::BadRequest(format!(
            "the gemma kernels need query heads 16 with kv heads 1 or 8, or 8 with 2, \
             got {n_q_head} and {n_kv_head}"
        )));
    }
    Ok(())
}

impl CudaContext {
    pub(crate) fn attn_decode_gemma(
        &self,
        stream: &DeviceStream,
        call: AttnDecodeGemmaCall<'_>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let raw = self.plan_attn_decode_gemma(&call)?;
        self.launch_on(stream, |stream_raw| {
            attn_decode_gemma_launch(stream_raw, &raw)
        })
    }

    fn plan_attn_decode_gemma<'b>(
        &self,
        call: &AttnDecodeGemmaCall<'b>,
    ) -> Result<AttnDecodeGemmaRaw<'b>, EngError> {
        let out_raw = self.raw_range(&call.out)?;
        let status_raw = match &call.status {
            Some(range) => Some(self.raw_range(range)?),
            None => None,
        };
        let q_raw = self.raw_range(&call.q)?;
        let k_codes = self.raw_range(&call.k_codes)?;
        let k_scales = self.raw_range(&call.k_scales)?;
        let v_codes = self.raw_range(&call.v_codes)?;
        let v_scales = self.raw_range(&call.v_scales)?;
        let kv_len = self.raw_range(&call.kv_len)?;
        let codes_bytes = call
            .n_kv_head
            .checked_mul(call.cells)
            .and_then(|v| v.checked_mul(call.head_dim))
            .ok_or_else(|| EngError::BadRequest("the attention size overflows".to_owned()))?;
        let scales_bytes = call
            .n_kv_head
            .checked_mul(call.cells)
            .and_then(|v| v.checked_mul(call.head_dim / 32))
            .and_then(|v| v.checked_mul(2))
            .ok_or_else(|| EngError::BadRequest("the attention size overflows".to_owned()))?;
        let q_bytes = call
            .m
            .checked_mul(call.n_q_head)
            .and_then(|v| v.checked_mul(call.head_dim))
            .and_then(|v| v.checked_mul(4))
            .ok_or_else(|| EngError::BadRequest("the attention size overflows".to_owned()))?;
        if !(1..=16).contains(&call.m) {
            return Err(EngError::BadRequest(format!(
                "the attention takes 1 to 16 tokens, got {}",
                call.m
            )));
        }
        admit_gemma_heads(call.n_q_head, call.n_kv_head)?;
        if !matches!(call.head_dim, 256 | 512) {
            return Err(EngError::BadRequest(format!(
                "the attention needs head_dim 256 or 512, got {}",
                call.head_dim
            )));
        }
        if !(1..=i32::MAX as usize).contains(&call.cells) || call.window > call.cells {
            return Err(EngError::BadRequest(
                "the attention needs an i32 cell capacity and a window within it".to_owned(),
            ));
        }
        if !call.scale.is_finite() || call.scale.abs() >= 1e30 {
            return Err(EngError::BadRequest(format!(
                "the attention needs a finite scale below 1e30, got {}",
                call.scale
            )));
        }
        require_fit("out", &call.out, q_bytes, 4)?;
        require_fit("q", &call.q, q_bytes, 4)?;
        require_fit("k_codes", &call.k_codes, codes_bytes, 1)?;
        require_fit("v_codes", &call.v_codes, codes_bytes, 1)?;
        require_fit("k_scales", &call.k_scales, scales_bytes, 2)?;
        require_fit("v_scales", &call.v_scales, scales_bytes, 2)?;
        require_fit("kv_len", &call.kv_len, 4, 4)?;
        if let Some(status) = &call.status {
            require_fit("status", status, 4, 4)?;
        }
        let output = Claim::of("out", &call.out, q_bytes);
        let reads = [
            ("q", &call.q, q_bytes),
            ("k_codes", &call.k_codes, codes_bytes),
            ("k_scales", &call.k_scales, scales_bytes),
            ("v_codes", &call.v_codes, codes_bytes),
            ("v_scales", &call.v_scales, scales_bytes),
            ("kv_len", &call.kv_len, 4),
        ];
        for (name, range, bytes) in reads {
            let input = Claim::of(name, range, bytes);
            require_apart(&output, &input)?;
            if let Some(status) = &call.status {
                require_apart(&Claim::of("status", status, 4), &input)?;
            }
        }
        if let Some(status) = &call.status {
            require_apart(&Claim::of("status", status, 4), &output)?;
        }
        let split = match &call.split {
            None => None,
            Some(plan) => {
                let part_bytes = attn_decode_gemma_part_floats(
                    call.m,
                    call.n_q_head,
                    call.head_dim,
                    plan.max_splits,
                )
                .and_then(|floats| floats.checked_mul(4))
                .ok_or_else(|| EngError::BadRequest("the split size overflows".to_owned()))?;
                if !(1..=128).contains(&plan.max_splits) || plan.min_keys < 8 {
                    return Err(EngError::BadRequest(format!(
                        "the split plan needs max_splits 1..=128 and min_keys >= 8, \
                         got {} and {}",
                        plan.max_splits, plan.min_keys
                    )));
                }
                require_fit("part", &plan.part, part_bytes, 16)?;
                let partial = Claim::of("part", &plan.part, part_bytes);
                require_apart(&partial, &output)?;
                for (name, range, bytes) in reads {
                    let input = Claim::of(name, range, bytes);
                    require_apart(&partial, &input)?;
                }
                Some(AttnDecodeGemmaSplit {
                    part: plan.part,
                    max_splits: plan.max_splits,
                    min_keys: plan.min_keys,
                })
            }
        };
        Ok(AttnDecodeGemmaRaw {
            out: out_raw,
            status: status_raw,
            q: q_raw,
            k_codes,
            k_scales,
            v_codes,
            v_scales,
            kv_len,
            m: call.m,
            n_q_head: call.n_q_head,
            n_kv_head: call.n_kv_head,
            head_dim: call.head_dim,
            cells: call.cells,
            window: call.window,
            scale: call.scale,
            part: split
                .as_ref()
                .map(|plan| self.raw_range(&plan.part))
                .transpose()?,
            split: split.is_some(),
            max_splits: split.as_ref().map_or(1, |plan| plan.max_splits),
            min_keys: split.as_ref().map_or(32, |plan| plan.min_keys),
        })
    }
}

#[cfg(test)]
mod tests;
