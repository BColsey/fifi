//! Test-only IQ4_NL expert down-projection plan using the existing FP16-intermediate GEMM.
use super::expert_layout::ExpertProjection;
use crate::{
    EngError,
    kernels::cuda::{
        CudaContext, DeviceRange, DeviceStream, prefill::gemm_dq::GemmDqTypedCall,
        resident::matvec_q::QuantType,
    },
};

fn bad(message: &str) -> EngError {
    EngError::BadRequest(format!("GGUF test-only expert F16 projection: {message}"))
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

/// Check every rank and construct all spans before launching any GEMM.
pub(super) fn plan<'a>(
    projection: &ExpertProjection<'a>,
    slots: &[i32],
    input: DeviceRange<'a>,
    output: DeviceRange<'a>,
) -> Result<Vec<GemmDqTypedCall<'a>>, EngError> {
    if projection.ty != QuantType::Iq4Nl {
        return Err(bad("only IQ4_NL is supported"));
    }
    let (k, rows, used) = (projection.k, projection.rows, slots.len());
    let input_row = k.checked_mul(4).ok_or_else(|| bad("input row overflow"))?;
    let output_row = rows
        .checked_mul(4)
        .ok_or_else(|| bad("output row overflow"))?;
    let input_len = input_row
        .checked_mul(used)
        .ok_or_else(|| bad("input length overflow"))?;
    let output_len = output_row
        .checked_mul(used)
        .ok_or_else(|| bad("output length overflow"))?;
    if input.bytes() < input_len || output.bytes() < output_len || input_len == 0 || output_len == 0
    {
        return Err(bad("activation or output range too short"));
    }
    if std::ptr::eq(input.buffer(), output.buffer())
        && input.offset() < output.offset() + output_len
        && output.offset() < input.offset() + input_len
    {
        return Err(bad("input and output ranges overlap"));
    }
    if std::ptr::eq(projection.weights.buffer(), output.buffer())
        && projection.weights.offset() < output.offset() + output_len
        && output.offset() < projection.weights.offset() + projection.weights.bytes()
    {
        return Err(bad("output overlaps the expert weight stack"));
    }
    let stride = projection
        .ty
        .weight_bytes(rows, k)
        .ok_or_else(|| bad("invalid matrix size"))?;
    let offsets = slot_offsets(slots, projection.slots, stride, projection.weights.bytes())?;
    let mut calls = Vec::with_capacity(used);
    for (rank, offset) in offsets.into_iter().enumerate() {
        let w = projection
            .weights
            .buffer()
            .range(projection.weights.offset() + offset, stride)?;
        let x = input
            .buffer()
            .range(input.offset() + rank * input_row, input_row)?;
        let y = output
            .buffer()
            .range(output.offset() + rank * output_row, output_row)?;
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

pub(super) fn launch(
    ctx: &CudaContext,
    stream: &DeviceStream,
    calls: &[GemmDqTypedCall<'_>],
) -> Result<(), EngError> {
    for call in calls {
        ctx.gemm_dq_codebook(stream, QuantType::Iq4Nl, call.clone())?;
    }
    Ok(())
}
