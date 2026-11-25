//! The decode dq matvec: one f32 activation row `x` through a native Q4_K or Q6_K weight,
//! `y[r] = sum_k x[k] * w[r][k]`, with the weights decoded per value and each block's sum
//! rounded in f32 (not an exact f32 product: the CPU reference holds within
//! `64 * 2^-24 * sum |w x|`, see `tests.rs`). That reference is `reference::weight_value` of
//! `resident/matvec_q/tests.rs`, a transcription of ggml's Q4_K and Q6_K layouts, not the
//! crate's `weights/gguf_dequant.rs`.
//!
//! [CudaContext::matvec_dq] queues ONE kernel on a [DeviceStream] and returns when it is
//! queued. It allocates nothing, copies nothing and never synchronizes.
//!
//! [CudaContext::matvec_dq_native] is the same launch over a [NativeWeight]: the SenseNova
//! decode step's dq route (`FIFI_SENSENOVA_DECODE_GEMM=dq`) takes it for every projection.
//!
//! Every refusal of the conditions below is a BadRequest and happens before any native call:
//! the backend is usable, the type is Q4_K or Q6_K, k is a positive multiple of 256, rows is
//! above zero, w holds the rows' blocks and x k floats at 16-byte alignment, y holds rows
//! floats at 4-byte alignment, and y overlaps no byte of w or x. One refusal comes later and
//! as a Backend error: a k or a rows at or above 2^31 passes these checks and is refused by
//! the native entry, as the native entries of matvec_q, matvec_q5k, matvec_gguf and
//! matvec_q3k refuse it.

use crate::EngError;
use crate::kernels::cuda::NativeWeight;
use crate::kernels::cuda::context::{
    Claim, CudaContext, DeviceRange, DeviceStream, require_apart, require_fit,
};
use crate::kernels::cuda::native::resident::matvec_dq::{MatvecDqRaw, matvec_dq_launch};
use crate::kernels::cuda::resident::matvec_q::QuantType;

const F32: usize = size_of::<f32>();

/// One row through one native weight: `y[r] = sum_k x[k] * w[r][k]`.
pub(crate) struct MatvecDqCall<'b> {
    pub(crate) ty: QuantType,
    pub(crate) w: DeviceRange<'b>,
    pub(crate) x: DeviceRange<'b>,
    pub(crate) y: DeviceRange<'b>,
    pub(crate) k: usize,
    pub(crate) rows: usize,
}

impl CudaContext {
    /// `x` (k floats) through `w` (`rows` rows of native blocks) into `y` (rows floats).
    pub(crate) fn matvec_dq(
        &self,
        stream: &DeviceStream,
        call: MatvecDqCall<'_>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let raw = self.plan_matvec_dq(&call)?;
        self.launch_on(stream, |stream_raw| matvec_dq_launch(stream_raw, &raw))
    }

    /// One f32 row (k floats) through the resident `weight` into `y` (rows floats): the call
    /// above over the weight's own type, blocks, `k` and rows. A weight the kernel does not
    /// serve (a type other than Q4_K or Q6_K, a `k` that is not a multiple of 256) is the
    /// matvec's own `BadRequest`, before any launch: nothing else runs in its place.
    pub(crate) fn matvec_dq_native(
        &self,
        stream: &DeviceStream,
        weight: &NativeWeight,
        x: DeviceRange<'_>,
        y: DeviceRange<'_>,
    ) -> Result<(), EngError> {
        self.matvec_dq(
            stream,
            MatvecDqCall {
                ty: weight.quant(),
                w: weight.buffer().range(0, weight.buffer().bytes())?,
                x,
                y,
                k: weight.k(),
                rows: weight.n_rows(),
            },
        )
    }

    fn plan_matvec_dq<'b>(&self, call: &MatvecDqCall<'b>) -> Result<MatvecDqRaw<'b>, EngError> {
        let (k, rows) = (call.k, call.rows);
        let shape = || {
            EngError::BadRequest(
                "matvec_dq needs a Q4_K or Q6_K weight, k a positive multiple of 256 and rows above zero"
                    .to_owned(),
            )
        };
        if !matches!(call.ty, QuantType::Q4K | QuantType::Q6K) || k == 0 || rows == 0 {
            return Err(shape());
        }
        let w_bytes = call.ty.weight_bytes(rows, k).ok_or_else(shape)?;
        let x_bytes = k.checked_mul(F32).ok_or_else(shape)?;
        let y_bytes = rows.checked_mul(F32).ok_or_else(shape)?;
        require_fit("matvec_dq w", &call.w, w_bytes, 16)?;
        require_fit("matvec_dq x", &call.x, x_bytes, 16)?;
        require_fit("matvec_dq y", &call.y, y_bytes, 4)?;
        let y = Claim::of("matvec_dq y", &call.y, y_bytes);
        require_apart(&y, &Claim::of("matvec_dq w", &call.w, w_bytes))?;
        require_apart(&y, &Claim::of("matvec_dq x", &call.x, x_bytes))?;
        Ok(MatvecDqRaw {
            ty: call.ty,
            w: self.raw_range(&call.w)?,
            x: self.raw_range(&call.x)?,
            y: self.raw_range(&call.y)?,
            k,
            rows,
        })
    }
}

#[cfg(test)]
mod tests;
