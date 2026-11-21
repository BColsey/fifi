//! A native quantized weight applied to a run of q8_1 activation rows.
//!
//! The matvec launch takes at most eight activation columns. This entry takes any row count: it
//! splits the run into groups of eight and queues one launch per group, so a caller with a chunk
//! of `m` rows (a prefill piece) and a caller with one row (decode) use the same call. It
//! assembles the weight's type, shape and byte range from the [`NativeWeight`] itself, which
//! every user of the matvec otherwise repeats at its call site.

use crate::EngError;
use crate::kernels::cuda::NativeWeight;
use crate::kernels::cuda::context::{CudaContext, DeviceRange, DeviceStream};
use crate::kernels::cuda::resident::{matvec_q::MatvecCall, quantize_q8_1::q8_1_bytes};

/// Activation columns one matvec launch takes.
pub(crate) const MATVEC_COLUMNS: usize = 8;

const F32: usize = size_of::<f32>();

impl CudaContext {
    /// `ncols` rows of `x_q8` (each a q8_1 row of `weight.k()` values, back to back) through
    /// `weight` into `y`, `rows` floats per activation row.
    pub(crate) fn matvec_native(
        &self,
        stream: &DeviceStream,
        weight: &NativeWeight,
        x_q8: DeviceRange<'_>,
        y: DeviceRange<'_>,
        ncols: usize,
    ) -> Result<(), EngError> {
        let (k, rows) = (weight.k(), weight.n_rows());
        let row_bytes = q8_1_bytes(k, 1).ok_or_else(|| overflow("a q8_1 row"))?;
        let w = weight.buffer().range(0, weight.buffer().bytes())?;
        for first in (0..ncols).step_by(MATVEC_COLUMNS) {
            let n = MATVEC_COLUMNS.min(ncols - first);
            self.matvec_q(
                stream,
                MatvecCall {
                    ty: weight.quant(),
                    w,
                    x_q8: x_q8
                        .buffer()
                        .range(x_q8.offset() + first * row_bytes, n * row_bytes)?,
                    y: y.buffer()
                        .range(y.offset() + first * rows * F32, n * rows * F32)?,
                    k,
                    rows,
                    ncols: n,
                },
            )?;
        }
        Ok(())
    }
}

fn overflow(what: &str) -> EngError {
    EngError::BadRequest(format!("matvec_native: the size of {what} overflows"))
}
