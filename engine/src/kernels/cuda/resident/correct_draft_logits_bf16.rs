//! Combine the base and recurrent correction for one speculative logits row.

use crate::EngError;
use crate::kernels::cuda::context::{
    Claim, CudaContext, DeviceRange, DeviceStream, require_apart, require_fit,
};
use crate::kernels::cuda::native::resident::correct_draft_logits_bf16::{
    CorrectDraftLogitsRaw, correct_draft_logits_bf16_launch,
};
use crate::kernels::cuda::resident::step_params::MAX_LOGITS;

pub(crate) const SOFTCAP_LUT_BYTES: usize = (1 << u16::BITS) * size_of::<u16>();

pub(crate) struct CorrectDraftLogitsCall<'a> {
    /// Number of live vocabulary IDs.
    pub(crate) len: usize,
    /// F32 base logits, overwritten with corrected capped BF16 values in F32 storage.
    pub(crate) base: DeviceRange<'a>,
    /// F32 recurrent correction, read only.
    pub(crate) correction: DeviceRange<'a>,
    /// Little-endian BF16 cap results, indexed by the rounded sum's BF16 payload.
    pub(crate) softcap_lut: DeviceRange<'a>,
}

pub(in crate::kernels::cuda) fn logits_bytes(len: usize) -> Option<usize> {
    if !(1..=MAX_LOGITS).contains(&len) {
        return None;
    }
    len.checked_mul(size_of::<f32>())
}

impl CudaContext {
    /// Queues the exact BF16 round, add, round sequence and a softcap lookup.
    /// Nonfinite inputs, sums or cap results publish canonical F32 NaNs for argmax.
    pub(crate) fn correct_draft_logits_bf16(
        &self,
        stream: &DeviceStream,
        call: CorrectDraftLogitsCall<'_>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let bytes = logits_bytes(call.len).ok_or_else(|| {
            EngError::BadRequest("corrected draft logits: invalid vocabulary length".into())
        })?;
        if call.softcap_lut.bytes() != SOFTCAP_LUT_BYTES {
            return Err(EngError::BadRequest(format!(
                "corrected draft logits: softcap table must have exactly {SOFTCAP_LUT_BYTES} bytes"
            )));
        }
        let base = "base draft logits";
        let correction = "draft logit correction";
        let table = "draft softcap table";
        require_fit(base, &call.base, bytes, 4)?;
        require_fit(correction, &call.correction, bytes, 4)?;
        require_fit(table, &call.softcap_lut, SOFTCAP_LUT_BYTES, 2)?;
        let written = Claim::of(base, &call.base, bytes);
        require_apart(&written, &Claim::of(correction, &call.correction, bytes))?;
        require_apart(
            &written,
            &Claim::of(table, &call.softcap_lut, SOFTCAP_LUT_BYTES),
        )?;

        let raw = CorrectDraftLogitsRaw {
            len: call.len,
            base: self.raw_range(&call.base)?,
            correction: self.raw_range(&call.correction)?,
            softcap_lut: self.raw_range(&call.softcap_lut)?,
        };
        self.launch_on(stream, |stream| {
            correct_draft_logits_bf16_launch(stream, &raw)
        })
    }
}

#[cfg(test)]
mod tests;
