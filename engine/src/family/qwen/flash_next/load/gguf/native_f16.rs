//! Test-only whole-model precision experiment: native GGUF projections take unquantized
//! activations through the dequantizing GEMM instead of a Q8_1 row and an integer dot product.
//!
//! It exists to measure how much of the decode error the activation quantization explains. The
//! GEMM rounds the activation row and the decoded weights to FP16 and accumulates in F32, so it
//! is more accurate than Q8_1, not exact. It is not a production dispatch.
use super::expert_layout::ExpertProjection;
use crate::{
    EngError,
    kernels::cuda::{
        CudaContext, DeviceRange, DeviceStream, NativeWeight, prefill::gemm_dq::GemmDqTypedCall,
        resident::matvec_q::QuantType,
    },
};

fn bad(message: &str) -> EngError {
    EngError::BadRequest(format!("GGUF test-only unquantized projection: {message}"))
}

/// Whether `FIFI_FLASH_GGUF_TEST_NATIVE_F16` asks every native projection to skip Q8_1.
pub(in super::super::super) fn enabled() -> bool {
    std::env::var_os("FIFI_FLASH_GGUF_TEST_NATIVE_F16").is_some()
}

/// The types the dequantizing GEMM decodes.
fn supported(ty: QuantType) -> bool {
    ty.is_codebook() || matches!(ty, QuantType::Q80 | QuantType::Q6K)
}

/// One matrix, not an expert stack: `input` holds `k` f32 and `output` receives `rows` f32.
pub(in super::super::super) fn project(
    ctx: &CudaContext,
    stream: &DeviceStream,
    weight: &NativeWeight,
    input: DeviceRange<'_>,
    output: DeviceRange<'_>,
) -> Result<(), EngError> {
    let ty = weight.quant();
    if weight.experts().is_some() {
        return Err(bad("requires a matrix, not an expert stack"));
    }
    if !supported(ty) {
        return Err(bad(&format!("unsupported matrix type {ty:?}")));
    }
    ctx.gemm_dq_codebook(
        stream,
        ty,
        GemmDqTypedCall {
            w: weight.buffer().range(0, weight.buffer().bytes())?,
            x: input,
            y: output,
            k: weight.k(),
            rows: weight.n_rows(),
            ncols: 1,
            x_col_stride: 0,
            y_col_stride: 0,
            x_f32: true,
        },
    )
}

fn slot_offsets(
    raw: &[i32],
    slots: usize,
    stride: usize,
    bytes: usize,
) -> Result<Vec<usize>, EngError> {
    if raw.is_empty() || stride == 0 || slots.checked_mul(stride) != Some(bytes) {
        return Err(bad("invalid expert stack size"));
    }
    let mut offsets = Vec::with_capacity(raw.len());
    for &slot in raw {
        let slot = usize::try_from(slot).map_err(|_| bad("negative slot"))?;
        let offset = slot
            .checked_mul(stride)
            .ok_or_else(|| bad("slot offset overflow"))?;
        if slot >= slots || offsets.contains(&offset) {
            return Err(bad("duplicate or out-of-range slot"));
        }
        offsets.push(offset);
    }
    Ok(offsets)
}

/// Byte spans of the ranks' activations and outputs, from the start of each range.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Layout {
    input_row: usize,
    input_len: usize,
    output_row: usize,
    output_slot: usize,
    output_len: usize,
}

/// `used` ranks read either one shared `k`-float row or one such row each, and each writes
/// `rows` floats at a stride of `output_stride` floats.
fn layout(
    k: usize,
    rows: usize,
    used: usize,
    input_per_slot: bool,
    output_stride: usize,
) -> Option<Layout> {
    if used == 0 || output_stride < rows {
        return None;
    }
    let input_row = k.checked_mul(4)?;
    let output_row = rows.checked_mul(4)?;
    let output_slot = output_stride.checked_mul(4)?;
    Some(Layout {
        input_row,
        input_len: if input_per_slot {
            input_row.checked_mul(used)?
        } else {
            input_row
        },
        output_row,
        output_slot,
        output_len: output_slot.checked_mul(used - 1)?.checked_add(output_row)?,
    })
}

/// Check every rank and construct all spans before launching any GEMM. `input_per_slot` reads
/// `[used][k]` activations (the down projection) instead of one shared row (gate and up).
pub(in super::super::super) fn plan<'a>(
    projection: &ExpertProjection<'a>,
    slots: &[i32],
    input: DeviceRange<'a>,
    output: DeviceRange<'a>,
    input_per_slot: bool,
    output_stride: usize,
) -> Result<Vec<GemmDqTypedCall<'a>>, EngError> {
    let ty = projection.ty;
    if !supported(ty) {
        return Err(bad(&format!("unsupported expert type {ty:?}")));
    }
    let (k, rows, used) = (projection.k, projection.rows, slots.len());
    let span = layout(k, rows, used, input_per_slot, output_stride)
        .ok_or_else(|| bad("invalid expert layout"))?;
    if input.bytes() < span.input_len || output.bytes() < span.output_len {
        return Err(bad("activation or output range too short"));
    }
    if std::ptr::eq(input.buffer(), output.buffer())
        && input.offset() < output.offset() + span.output_len
        && output.offset() < input.offset() + span.input_len
    {
        return Err(bad("input and output ranges overlap"));
    }
    if std::ptr::eq(projection.weights.buffer(), output.buffer())
        && projection.weights.offset() < output.offset() + span.output_len
        && output.offset() < projection.weights.offset() + projection.weights.bytes()
    {
        return Err(bad("output overlaps the expert weight stack"));
    }
    let stride = ty
        .weight_bytes(rows, k)
        .ok_or_else(|| bad("invalid matrix size"))?;
    let offsets = slot_offsets(slots, projection.slots, stride, projection.weights.bytes())?;
    let mut calls = Vec::with_capacity(used);
    for (rank, offset) in offsets.into_iter().enumerate() {
        let w = projection
            .weights
            .buffer()
            .range(projection.weights.offset() + offset, stride)?;
        let x_at = if input_per_slot {
            rank * span.input_row
        } else {
            0
        };
        let x = input
            .buffer()
            .range(input.offset() + x_at, span.input_row)?;
        let y = output
            .buffer()
            .range(output.offset() + rank * span.output_slot, span.output_row)?;
        calls.push(GemmDqTypedCall {
            w,
            x,
            y,
            k,
            rows,
            ncols: 1,
            x_col_stride: 0,
            y_col_stride: 0,
            x_f32: true,
        });
    }
    Ok(calls)
}

pub(in super::super::super) fn launch(
    ctx: &CudaContext,
    stream: &DeviceStream,
    ty: QuantType,
    calls: &[GemmDqTypedCall<'_>],
) -> Result<(), EngError> {
    for call in calls {
        ctx.gemm_dq_codebook(stream, ty, call.clone())?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{layout, slot_offsets, supported};
    use crate::kernels::cuda::resident::matvec_q::QuantType;

    #[test]
    fn expert_slot_offsets_reject_invalid_and_duplicate_slots() {
        assert_eq!(
            slot_offsets(&[2, 0, 1], 3, 18, 54).unwrap(),
            vec![36, 0, 18]
        );
        assert!(slot_offsets(&[], 3, 18, 54).is_err());
        assert!(slot_offsets(&[-1], 3, 18, 54).is_err());
        assert!(slot_offsets(&[3], 3, 18, 54).is_err());
        assert!(slot_offsets(&[1, 1], 3, 18, 54).is_err());
        assert!(slot_offsets(&[0], 3, 18, 53).is_err());
        assert!(slot_offsets(&[0], usize::MAX, 18, 0).is_err());
    }

    #[test]
    fn rank_layout_shares_or_splits_the_activation_and_honours_the_output_stride() {
        // Ten experts of a 640-wide gate reading one 2560-wide row, interleaved with `up`.
        let gate = layout(2560, 640, 10, false, 1280).unwrap();
        assert_eq!(
            (gate.input_row, gate.input_len, gate.output_row),
            (10240, 10240, 2560)
        );
        assert_eq!((gate.output_slot, gate.output_len), (5120, 9 * 5120 + 2560));
        // The down projection reads each expert's own 640-wide activation row.
        let down = layout(640, 2560, 10, true, 2560).unwrap();
        assert_eq!(
            (down.input_len, down.output_slot, down.output_len),
            (10 * 2560, 10240, 10 * 10240)
        );
        assert_eq!(layout(640, 2560, 10, true, 2559), None);
        assert_eq!(layout(640, 2560, 0, true, 2560), None);
    }

    #[test]
    fn supported_types_are_the_dequantizing_gemm_decoders() {
        for ty in [
            QuantType::Q80,
            QuantType::Q6K,
            QuantType::Q2K,
            QuantType::Q20,
            QuantType::Iq2S,
            QuantType::Iq4Nl,
        ] {
            assert!(supported(ty), "{ty:?}");
        }
        for ty in [QuantType::Q5K, QuantType::Q40, QuantType::Q3K] {
            assert!(!supported(ty), "{ty:?}");
        }
    }
}
