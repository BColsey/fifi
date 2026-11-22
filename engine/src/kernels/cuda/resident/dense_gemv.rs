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
    dense_gemv_launch, gguf_exact_expert_tile_launch, gguf_wmma_launch, typed_gemv_need,
    typed_wmma_need,
};
use crate::kernels::cuda::{CudaContext, DeviceRange, DeviceStream};
use crate::weights::GgufType;

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

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct DenseGemvGroupCounts {
    pub(crate) wmma: usize,
    pub(crate) exact_tiles: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GroupKernel {
    Gemv,
    Wmma,
    ExactTile,
}

fn group_kernel(
    shape: DenseGemvShape,
    dtype: DenseDtype,
    min_wmma_rows: Option<usize>,
    exact_tile: bool,
) -> Result<(GroupKernel, DenseGemvBytes), EngError> {
    if let Some(need) = min_wmma_rows
        .filter(|&rows| shape.m >= rows)
        .and_then(|_| typed_wmma_need(shape.dims(), dtype))
    {
        return Ok((GroupKernel::Wmma, need));
    }
    let need = dense_gemv_bytes(shape, dtype)?;
    // Mirror the native tile bounds to keep known fallbacks on the original entry.
    // The native ABI still validates and rechecks eligibility before enqueueing.
    let kernel = if exact_tile
        && matches!(
            dtype,
            DenseDtype::PackedGguf(GgufType::Iq2Xxs | GgufType::Q20)
        )
        && (2..=512).contains(&shape.m)
        && shape.rows <= 2304
        && shape.cols <= 2304
    {
        GroupKernel::ExactTile
    } else {
        GroupKernel::Gemv
    };
    Ok((kernel, need))
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
        self.dense_gemv_group_with_tiles(stream, calls, None, false)
            .map(|_| ())
    }

    /// Queues the group in order. Where supported, an explicit row threshold permits WMMA
    /// (F16 operands, F32 accumulation); only the remaining ordinary GEMVs may opt into exact
    /// FP32 expert tiling. Validates all calls before the first launch, including capacity,
    /// alignment, overlap and adapter ownership. Counts actual enqueues, and never retries a
    /// native launch failure.
    pub(crate) fn dense_gemv_group_with_tiles(
        &self,
        stream: &DeviceStream,
        calls: &[DenseGemvCall<'_>],
        min_wmma_rows: Option<usize>,
        exact_tile: bool,
    ) -> Result<DenseGemvGroupCounts, EngError> {
        self.check_launch_stream(stream)?;
        let min_wmma_rows = match min_wmma_rows {
            Some(rows) if self.mmq_device_facts()?.cc >= 700 => Some(rows),
            _ => None,
        };
        let planned = calls
            .iter()
            .map(|call| {
                let (kernel, need) =
                    group_kernel(call.shape, call.dtype, min_wmma_rows, exact_tile)?;
                self.plan_dense_projection(call, need)
                    .map(|raw| (kernel, raw))
            })
            .collect::<Result<Vec<_>, _>>()?;
        self.launch_on(stream, |raw_stream| {
            let mut counts = DenseGemvGroupCounts::default();
            for (kernel, call) in planned {
                match kernel {
                    GroupKernel::Gemv => dense_gemv_launch(raw_stream, call)?,
                    GroupKernel::Wmma => {
                        gguf_wmma_launch(raw_stream, call)?;
                        counts.wmma += 1;
                    }
                    GroupKernel::ExactTile => {
                        counts.exact_tiles +=
                            usize::from(gguf_exact_expert_tile_launch(raw_stream, call)?);
                    }
                }
            }
            Ok(counts)
        })
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
pub(crate) use crate::kernels::cuda::native::resident::dense_gemv::moe::{
    GgufGate, GgufMoeSchedule,
};
pub(crate) use moe::{GgufMoeCall, GgufMoePick, GgufMoePicks, GgufMoeShape, GgufMoeWeights};

#[cfg(all(test, feature = "native-cuda"))]
mod tests;

#[cfg(test)]
mod group_tests {
    use super::*;

    #[test]
    fn exact_expert_tile_only_replaces_ordinary_eligible_gemv() {
        let iq2 = DenseDtype::PackedGguf(GgufType::Iq2Xxs);
        let q2 = DenseDtype::PackedGguf(GgufType::Q20);
        let q8 = DenseDtype::PackedGguf(GgufType::Q80);
        let gate = |m| DenseGemvShape::packed(m, 2304, 2304);
        let down = |m| DenseGemvShape::packed(m, 2304, 1152);
        let kernel = |shape, dtype, wmma, tile| group_kernel(shape, dtype, wmma, tile).unwrap().0;
        assert_eq!(kernel(gate(15), iq2, Some(16), false), GroupKernel::Gemv);
        assert_eq!(
            kernel(gate(15), iq2, Some(16), true),
            GroupKernel::ExactTile
        );
        for tile in [false, true] {
            assert_eq!(kernel(gate(16), iq2, Some(16), tile), GroupKernel::Wmma);
            assert_eq!(kernel(down(32), q2, Some(32), tile), GroupKernel::Wmma);
        }
        assert_eq!(kernel(down(31), q2, Some(32), true), GroupKernel::ExactTile);
        assert_eq!(kernel(down(512), q2, None, true), GroupKernel::ExactTile);
        assert_eq!(kernel(gate(1), iq2, None, true), GroupKernel::Gemv);
        assert_eq!(kernel(gate(513), iq2, None, true), GroupKernel::Gemv);
        assert_eq!(kernel(gate(4), q8, Some(2), true), GroupKernel::Gemv);
        for shape in [
            DenseGemvShape::packed(4, 2305, 2304),
            DenseGemvShape::packed(4, 2304, 2560),
        ] {
            assert_eq!(kernel(shape, iq2, None, true), GroupKernel::Gemv);
        }
        assert!(group_kernel(gate(0), iq2, None, true).is_err());
        assert!(group_kernel(DenseGemvShape::packed(4, 2304, 2303), iq2, None, true).is_err());
    }
}
