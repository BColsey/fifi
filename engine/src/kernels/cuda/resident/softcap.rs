//! The logit softcap of the Gemma 4 Humanizer head: one launch computing
//! out = cap * tanh(x / cap) (the pinned model uses cap 30).
//!
//! [CudaContext::softcap] queues the kernel on a [DeviceStream] and returns when it
//! is queued. It allocates nothing, copies nothing and never synchronizes.
//!
//! Every refusal is a BadRequest and happens before any native call: the backend is
//! usable, the cap is finite and above zero, n is a multiple of 32 in 32..=4194304,
//! both ranges hold n floats at 4-byte alignment, and out overlaps no byte of x.
//! [`CudaContext::softcap_in_place`] accepts any nonempty width up to the same
//! maximum and preserves nonfinite inputs for the caller's validation.

use crate::EngError;
use crate::kernels::cuda::context::{
    Claim, CudaContext, DeviceRange, DeviceStream, require_apart, require_fit,
};
use crate::kernels::cuda::native::resident::softcap::{SoftcapRaw, softcap_launch};

pub(crate) struct SoftcapCall<'b> {
    pub(crate) out: DeviceRange<'b>,
    pub(crate) x: DeviceRange<'b>,
    pub(crate) cap: f32,
    pub(crate) n: usize,
}

impl CudaContext {
    pub(crate) fn softcap(
        &self,
        stream: &DeviceStream,
        call: SoftcapCall<'_>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let raw = self.plan_softcap(&call)?;
        self.launch_on(stream, |stream_raw| softcap_launch(stream_raw, &raw))
    }

    /// Caps a row in place, including non-warp-aligned tails. Nonfinite inputs are
    /// preserved so callers can reject invalid raw logits after the readback.
    pub(crate) fn softcap_in_place(
        &self,
        stream: &DeviceStream,
        row: DeviceRange<'_>,
        n: usize,
        cap: f32,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let bytes = softcap_bytes(n, cap, true)?;
        require_fit("logits", &row, bytes, 4)?;
        let row = self.raw_range(&row)?;
        let raw = SoftcapRaw {
            out: row,
            x: row,
            n,
            cap,
        };
        self.launch_on(stream, |stream_raw| softcap_launch(stream_raw, &raw))
    }

    fn plan_softcap<'b>(&self, call: &SoftcapCall<'b>) -> Result<SoftcapRaw<'b>, EngError> {
        let (out_raw, x_raw) = (self.raw_range(&call.out)?, self.raw_range(&call.x)?);
        let bytes = softcap_bytes(call.n, call.cap, false)?;
        require_fit("out", &call.out, bytes, 4)?;
        require_fit("x", &call.x, bytes, 4)?;
        require_apart(
            &Claim::of("out", &call.out, bytes),
            &Claim::of("x", &call.x, bytes),
        )?;
        Ok(SoftcapRaw {
            out: out_raw,
            x: x_raw,
            cap: call.cap,
            n: call.n,
        })
    }
}

fn softcap_bytes(n: usize, cap: f32, in_place: bool) -> Result<usize, EngError> {
    if !cap.is_finite() || cap <= 0.0 {
        return Err(EngError::BadRequest(format!(
            "the softcap needs a finite cap above zero, got {cap}"
        )));
    }
    if !(1..=4_194_304).contains(&n) || (!in_place && !n.is_multiple_of(32)) {
        let width = if in_place {
            "1..=4194304"
        } else {
            "a multiple of 32 up to 4194304"
        };
        return Err(EngError::BadRequest(format!(
            "the softcap needs n in {width}, got {n}"
        )));
    }
    Ok(n * size_of::<f32>())
}

#[cfg(test)]
mod tests;
