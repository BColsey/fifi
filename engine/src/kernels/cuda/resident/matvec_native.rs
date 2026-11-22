//! A native quantized weight applied to a run of q8_1 activation rows.
//!
//! The matvec launch takes at most eight activation columns. This entry takes any row count: it
//! splits the run into groups of eight and queues one launch per group, so a caller with a chunk
//! of `m` rows (a prefill piece) and a caller with one row (decode) use the same call. It
//! assembles the weight's type, shape and byte range from the [`NativeWeight`] itself, which
//! every user of the matvec otherwise repeats at its call site.

use crate::EngError;
use crate::kernels::cuda::NativeWeight;
use crate::kernels::cuda::context::{
    Claim, CudaContext, DeviceRange, DeviceStream, require_apart, require_fit,
};
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
        if ncols == 0 {
            return Ok(());
        }
        // Validate the caller's complete ranges before splitting; rebuilding a chunk from
        // its backing buffer must not enlarge the caller's authorized subrange.
        let x_bytes = row_bytes
            .checked_mul(ncols)
            .ok_or_else(|| overflow("q8_1 activations"))?;
        let y_bytes = rows
            .checked_mul(F32)
            .and_then(|n| n.checked_mul(ncols))
            .ok_or_else(|| overflow("output"))?;
        require_fit("x_q8", &x_q8, x_bytes, 4)?;
        require_fit("y", &y, y_bytes, 4)?;
        let output = Claim::of("y", &y, y_bytes);
        require_apart(&output, &Claim::of("x_q8", &x_q8, x_bytes))?;
        require_apart(&output, &Claim::of("w", &w, w.bytes()))?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::cuda::resident::matvec_q::QuantType;

    #[test]
    #[ignore = "requires an admitted CUDA device"]
    fn native_matvec_rejects_short_subranges_of_larger_buffers() {
        let ctx = CudaContext::new().unwrap();
        let stream = ctx.stream().unwrap();
        let quant = QuantType::Q80;
        let weight_bytes = quant.weight_bytes(4, 32).unwrap();
        let w = ctx.allocate(weight_bytes).unwrap();
        w.write_at(&stream, 0, &vec![0; weight_bytes]).unwrap();
        let weight = NativeWeight::from_buffer(w, quant, 32, 4, None).unwrap();
        let q8_bytes = q8_1_bytes(32, 1).unwrap();
        let x = ctx.allocate(q8_bytes).unwrap();
        x.write_at(&stream, 0, &vec![0; q8_bytes]).unwrap();
        let y = ctx.allocate(16).unwrap();
        y.write_at(&stream, 0, &[0; 16]).unwrap();
        ctx.matvec_native(
            &stream,
            &weight,
            x.range(0, q8_bytes).unwrap(),
            y.range(0, 16).unwrap(),
            1,
        )
        .expect("full ranges must succeed");
        stream.synchronize().unwrap();
        let short_x = ctx.matvec_native(
            &stream,
            &weight,
            x.range(0, 4).unwrap(),
            y.range(0, 16).unwrap(),
            1,
        );
        let short_y = ctx.matvec_native(
            &stream,
            &weight,
            x.range(0, q8_bytes).unwrap(),
            y.range(0, 4).unwrap(),
            1,
        );
        stream.synchronize().unwrap();
        assert!(short_x.is_err(), "short input subrange accepted");
        assert!(short_y.is_err(), "short output subrange accepted");
    }

    #[test]
    #[ignore = "requires an admitted CUDA device"]
    fn native_matvec_checks_whole_multichunk_ranges_and_aliases() {
        let ctx = CudaContext::new().unwrap();
        let stream = ctx.stream().unwrap();
        let quant = QuantType::Q80;
        let weight_bytes = quant.weight_bytes(4, 32).unwrap();
        let w = ctx.allocate(weight_bytes).unwrap();
        w.write_at(&stream, 0, &vec![0; weight_bytes]).unwrap();
        let weight = NativeWeight::from_buffer(w, quant, 32, 4, None).unwrap();
        let x_bytes = q8_1_bytes(32, 9).unwrap();
        let x = ctx.allocate(x_bytes).unwrap();
        x.write_at(&stream, 0, &vec![0; x_bytes]).unwrap();
        let y = ctx.allocate(144).unwrap();
        y.write_at(&stream, 0, &[0; 144]).unwrap();
        ctx.matvec_native(
            &stream,
            &weight,
            x.range(0, x_bytes).unwrap(),
            y.range(0, 144).unwrap(),
            9,
        )
        .unwrap();
        stream.synchronize().unwrap();
        assert!(
            ctx.matvec_native(
                &stream,
                &weight,
                x.range(0, x_bytes - 4).unwrap(),
                y.range(0, 144).unwrap(),
                9
            )
            .is_err()
        );
        assert!(
            ctx.matvec_native(
                &stream,
                &weight,
                x.range(0, x_bytes).unwrap(),
                y.range(0, 140).unwrap(),
                9
            )
            .is_err()
        );
        // The first output chunk overlaps a later input chunk, not its own first input.
        assert!(
            ctx.matvec_native(
                &stream,
                &weight,
                x.range(0, x_bytes).unwrap(),
                x.range(q8_1_bytes(32, 1).unwrap() * 8, 144).unwrap(),
                9
            )
            .is_err()
        );
        stream.synchronize().unwrap();
    }
}
