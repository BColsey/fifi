//! The dense half-precision matvec of the Flash-Next step on the device. Its one user is the PLE
//! block: the key projection `[10240, 2560]` and the value projection `[2560, 2560]`, stored F16 in
//! the EXL3 pack, once per token in the layer that has the block. The host matvec of the same
//! projections measures 2.6 to 3.4 ms a token (p99 7 to 14 ms; the S2 feeds bench) against about 0.07 ms on the device. The
//! DeltaNet `b` and `a` gates do not use it: they take the router's strided F32 dot, widened at
//! upload, as Qwen3.6's gates do. BF16 weights (the official checkpoint) are accepted. The formula
//! and the layouts are in `native/resident/dense_gemv.h`.
//!
//! [`CudaContext::dense_gemv`] queues one kernel on a [`DeviceStream`] and returns. It allocates
//! nothing, copies nothing and never synchronizes; a fault in the queued work surfaces at the next
//! synchronize of the stream. `ld_y` lets the result land inside a wider row.
//!
//! Every refusal is a `BadRequest` and happens before any native call, because a status from the
//! native layer other than an allocation failure latches the backend terminal. A refused call
//! changes nothing.
#![cfg_attr(
    not(test),
    expect(dead_code, reason = "staged for the Flash-Next CUDA step")
)]

use crate::EngError;
use crate::kernels::cuda::context::{Claim, require_apart, require_fit};
pub(crate) use crate::kernels::cuda::native::resident::dense_gemv::{DenseDtype, DenseGemvBytes};
use crate::kernels::cuda::native::resident::dense_gemv::{
    DenseGemvRaw, GEMV_MAX_COLS, GEMV_MAX_LD, GEMV_MAX_M, GEMV_MAX_ROWS, GemvDims,
    dense_gemv_launch, gemv_need,
};
use crate::kernels::cuda::{CudaContext, DeviceRange, DeviceStream};

/// `m` activation rows of `cols` values at stride `ld_x`, against a weight of `rows` rows of
/// `cols` values; `rows` results per activation row at stride `ld_y`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DenseGemvShape {
    pub(crate) m: usize,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) ld_x: usize,
    pub(crate) ld_y: usize,
}

impl DenseGemvShape {
    /// One compact activation row and compact results: the stride equals the width.
    pub(crate) fn packed(m: usize, rows: usize, cols: usize) -> Self {
        Self {
            m,
            rows,
            cols,
            ld_x: cols,
            ld_y: rows,
        }
    }

    fn dims(self) -> GemvDims {
        GemvDims {
            m: self.m,
            rows: self.rows,
            cols: self.cols,
            ld_x: self.ld_x,
            ld_y: self.ld_y,
        }
    }
}

/// One launch: `w` `[rows][cols]` of `dtype` and `x` f32 are read; `y` f32 is written, element
/// `[t][r]` at `t * ld_y + r`, and no other element of it is touched.
pub(crate) struct DenseGemvCall<'b> {
    pub(crate) shape: DenseGemvShape,
    pub(crate) dtype: DenseDtype,
    pub(crate) w: DeviceRange<'b>,
    pub(crate) x: DeviceRange<'b>,
    pub(crate) y: DeviceRange<'b>,
}

/// The bytes each range of a launch of `shape` must hold.
pub(crate) fn dense_gemv_bytes(shape: DenseGemvShape) -> Result<DenseGemvBytes, EngError> {
    gemv_need(shape.dims()).ok_or_else(|| {
        EngError::BadRequest(format!(
            "the CUDA dense matvec takes 1 to {GEMV_MAX_M} rows of activations, 1 to \
             {GEMV_MAX_ROWS} weight rows, 1 to {GEMV_MAX_COLS} columns and strides from the row \
             width to {GEMV_MAX_LD}, got {shape:?}"
        ))
    })
}

impl CudaContext {
    /// Queues the matvec of `call.shape.m` activation rows on `stream`.
    ///
    /// Returns `BadRequest` for a stream or buffer of another adapter, for a shape outside the
    /// limits, for a short or misaligned range (the weight needs 2-byte alignment, `x` and `y`
    /// 4-byte), and for a `y` that overlaps `w` or `x`.
    pub(crate) fn dense_gemv(
        &self,
        stream: &DeviceStream,
        call: DenseGemvCall<'_>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let raw = self.plan_dense_gemv(&call)?;
        self.launch_on(stream, move |stream_raw| dense_gemv_launch(stream_raw, raw))
    }

    fn plan_dense_gemv<'b>(&self, call: &DenseGemvCall<'b>) -> Result<DenseGemvRaw<'b>, EngError> {
        let need = dense_gemv_bytes(call.shape)?;
        require_fit("w", &call.w, need.w, 2)?;
        require_fit("x", &call.x, need.x, 4)?;
        require_fit("y", &call.y, need.y, 4)?;
        let y = Claim::of("y", &call.y, need.y);
        require_apart(&y, &Claim::of("w", &call.w, need.w))?;
        require_apart(&y, &Claim::of("x", &call.x, need.x))?;
        Ok(DenseGemvRaw {
            dims: call.shape.dims(),
            dtype: call.dtype,
            w: self.raw_range(&call.w)?,
            x: self.raw_range(&call.x)?,
            y: self.raw_range(&call.y)?,
        })
    }
}

#[cfg(all(test, feature = "native-cuda"))]
mod tests;
