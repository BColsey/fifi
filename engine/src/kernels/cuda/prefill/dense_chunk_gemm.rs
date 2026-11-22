//! The dense prompt-chunk GEMM of a family whose weights rest as the file's native blocks: the
//! q8_1 input buffer and the stream-k fixup range the shared MMQ GEMM needs, and the two calls
//! that use them. A family keeps one of these beside its weights and runs a chunk's projections
//! through [`Self::input`] (once per activation) and [`Self::project`] (once per weight).

use crate::EngError;
use crate::kernels::cuda::NativeWeight;
use crate::kernels::cuda::context::{CudaContext, DeviceBuffer, DeviceRange, DeviceStream};
use crate::kernels::cuda::prefill::mmq_dense::{MmqDenseCall, MmqDenseInputCall};
use crate::kernels::cuda::prefill::mmq_quant::MmqPacking;
use crate::kernels::cuda::prefill::mmq_size::q8_1_bytes;

/// The most bytes the stream-k fixup range takes on any device (132 SMs of 128 x 128 floats).
const FIXUP_CAP_BYTES: usize = 16 << 20;

pub(crate) struct DenseChunkGemm {
    input: DeviceBuffer,
    fixup: DeviceBuffer,
}

impl DenseChunkGemm {
    /// The device bytes of a gemm over activations of up to `k_max` values and `max_cols` rows,
    /// for the admission to charge before anything is allocated.
    pub(crate) fn bytes(k_max: usize, max_cols: usize) -> Option<u64> {
        Some((q8_1_bytes(k_max, max_cols).ok()? + FIXUP_CAP_BYTES) as u64)
    }

    pub(crate) fn new(ctx: &CudaContext, k_max: usize, max_cols: usize) -> Result<Self, EngError> {
        Ok(Self {
            input: ctx.allocate(q8_1_bytes(k_max, max_cols)?)?,
            fixup: ctx.allocate(ctx.mmq_fixup_bytes()?)?,
        })
    }

    /// Quantizes the `m` f32 rows `x` of `weight.k()` values into the input of the
    /// [`Self::project`] calls that follow; the weight names the layout, so any weight of the
    /// activation's width does.
    pub(crate) fn input(
        &self,
        ctx: &CudaContext,
        stream: &DeviceStream,
        weight: &NativeWeight,
        x: DeviceRange<'_>,
        m: usize,
    ) -> Result<(), EngError> {
        ctx.mmq_dense_input(
            stream,
            MmqDenseInputCall {
                packing: MmqPacking::CpuReference,
                weight,
                m,
                row_stride: weight.k(),
                x,
                y: self.input.range(0, self.input.bytes())?,
            },
        )
    }

    /// Writes the `m` rows of `weight.n_rows()` f32 outputs of the last [`Self::input`] into `y`,
    /// row after row.
    pub(crate) fn project(
        &self,
        ctx: &CudaContext,
        stream: &DeviceStream,
        weight: &NativeWeight,
        m: usize,
        y: DeviceRange<'_>,
    ) -> Result<(), EngError> {
        ctx.mmq_dense(
            stream,
            MmqDenseCall {
                weight,
                w_seg: None,
                m,
                q8_1: self.input.range(0, self.input.bytes())?,
                dst: y,
                ld: weight.n_rows(),
                col_off: 0,
                fixup: self.fixup.range(0, self.fixup.bytes())?,
            },
        )
    }
}

#[cfg(test)]
mod tests;
