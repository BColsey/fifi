//! Shared matrix-vector projection for F32, F16, BF16 and original packed GGUF weights.
//! Activations and results stay in device buffers between launches.
//!
//! [`CudaContext::dense_gemv`] queues one kernel on a [`DeviceStream`] and returns. It allocates
//! nothing, copies nothing and never synchronizes; a fault in the queued work surfaces at the next
//! synchronize of the stream. `ld_y` lets the result land inside a wider row.
//!
//! Every refusal is a `BadRequest` and happens before any native call, because a status from the
//! native layer other than an allocation failure latches the backend terminal. A refused call
//! changes nothing.

use crate::EngError;
use crate::kernels::cuda::context::{Claim, require_apart, require_fit};
pub(crate) use crate::kernels::cuda::native::resident::dense_gemv::{DenseDtype, DenseGemvBytes};
use crate::kernels::cuda::native::resident::dense_gemv::{
    DenseGemvRaw, GEMV_MAX_COLS, GEMV_MAX_LD, GEMV_MAX_M, GEMV_MAX_ROWS, GemvDims,
    dense_gemv_launch, gguf_wmma_launch, typed_gemv_need, typed_wmma_need,
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
    pub(crate) const MAX_ACTIVATION_ROWS: usize = GEMV_MAX_M;

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

/// Buffer sizes for the matrix's stored dtype and the F32 activation and result rows.
pub(crate) fn dense_gemv_bytes(
    shape: DenseGemvShape,
    dtype: DenseDtype,
) -> Result<DenseGemvBytes, EngError> {
    typed_gemv_need(shape.dims(), dtype).ok_or_else(|| invalid_shape(shape))
}

fn invalid_shape(shape: DenseGemvShape) -> EngError {
    EngError::BadRequest(format!(
        "the CUDA dense matvec takes 1 to {GEMV_MAX_M} rows of activations, 1 to \
             {GEMV_MAX_ROWS} weight rows, 1 to {GEMV_MAX_COLS} columns and strides from the row \
             width to {GEMV_MAX_LD}, got {shape:?}"
    ))
}

impl CudaContext {
    /// Queues the matvec of `call.shape.m` activation rows on `stream`.
    ///
    /// Returns `BadRequest` for a stream or buffer of another adapter, for a shape outside the
    /// limits, for a short or misaligned range (F32 uses 4-byte alignment, other weights use
    /// 2-byte alignment), and for a `y` that overlaps `w` or `x`.
    pub(crate) fn dense_gemv(
        &self,
        stream: &DeviceStream,
        call: DenseGemvCall<'_>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let raw = self.plan_dense_gemv(&call)?;
        self.launch_on(stream, move |stream_raw| dense_gemv_launch(stream_raw, raw))
    }

    /// Validates the complete group before enqueuing any matvec. Device-only dependencies
    /// between calls are ordered by their shared stream.
    pub(crate) fn dense_gemv_group(
        &self,
        stream: &DeviceStream,
        calls: &[DenseGemvCall<'_>],
    ) -> Result<(), EngError> {
        self.check_launch_stream(stream)?;
        let planned = calls
            .iter()
            .map(|call| self.plan_dense_gemv(call))
            .collect::<Result<Vec<_>, _>>()?;
        self.launch_on(stream, |raw_stream| {
            for call in planned {
                dense_gemv_launch(raw_stream, call)?;
            }
            Ok(())
        })
    }

    /// Projects a batch with packed weights and F32 buffers. Operand tiles round to F16
    /// before Tensor Core products with F32 accumulation. Select it explicitly where that
    /// precision has been qualified. The original dense GEMV remains the exact fallback.
    #[allow(
        dead_code,
        reason = "staged pending full-model quality and dispatch qualification"
    )]
    pub(crate) fn gguf_wmma_gemm(
        &self,
        stream: &DeviceStream,
        call: DenseGemvCall<'_>,
    ) -> Result<(), EngError> {
        self.check_launch_stream(stream)?;
        let need = typed_wmma_need(call.shape.dims(), call.dtype).ok_or_else(|| {
            EngError::BadRequest(format!(
                "GGUF Tensor Core projection cannot use {:?} at {:?}",
                call.dtype, call.shape
            ))
        })?;
        let raw = self.plan_dense_projection(&call, need)?;
        if self.mmq_device_facts()?.cc < 700 {
            return Err(EngError::BadRequest(
                "GGUF Tensor Core projection needs SM70 or later".into(),
            ));
        }
        self.launch_on(stream, move |stream_raw| gguf_wmma_launch(stream_raw, raw))
    }

    fn plan_dense_gemv<'b>(&self, call: &DenseGemvCall<'b>) -> Result<DenseGemvRaw<'b>, EngError> {
        let need = dense_gemv_bytes(call.shape, call.dtype)?;
        self.plan_dense_projection(call, need)
    }

    fn plan_dense_projection<'b>(
        &self,
        call: &DenseGemvCall<'b>,
        need: DenseGemvBytes,
    ) -> Result<DenseGemvRaw<'b>, EngError> {
        require_fit("w", &call.w, need.w, call.dtype.weight_alignment())?;
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

mod moe;
pub(crate) use moe::{GgufMoeCall, GgufMoePick, GgufMoePicks, GgufMoeShape, GgufMoeWeights};

#[cfg(all(test, feature = "native-cuda"))]
mod tests;
