//! One routed (grouped, per-expert) GEMM of a MoE stage, for any family: the stack's type picks the
//! in-house dequantizing GEMM (it reads the f32 activation rows) or the vendored grouped MMQ (it
//! reads q8_1 columns built from them), so a stage quantizes only for the stacks that need it.

use super::{DqMmidCall, MmidIds, MmidLaunchCall, MmidQuantCall, MmidWeight};
use crate::EngError;
use crate::kernels::cuda::{CudaContext, DeviceRange, DeviceStream, NativeWeight};

/// The `MmidWeight` of one routed stack (stages 13a, 13b, 16): the whole stack buffer, its
/// type, `k` values per expert row, the expert's `n` rows, and the per-expert byte stride
/// derived from the type.
pub(crate) fn routed_weight<'a>(
    stack: &'a NativeWeight,
    k: usize,
    n: usize,
) -> Result<MmidWeight<'a>, EngError> {
    let quant = stack.quant();
    let stride_bytes = quant.weight_bytes(n, k).ok_or_else(|| {
        EngError::BadRequest(format!(
            "routed stack: the {quant:?} stride of {n} rows of {k} values overflows"
        ))
    })?;
    Ok(MmidWeight {
        base: stack.buffer().range(0, stack.buffer().bytes())?,
        quant,
        k,
        n,
        stride_bytes,
    })
}

/// What one routed GEMM of a stage reads: the vendored grouped MMQ reads the q8_1 columns `q8`,
/// the in-house dequantizing GEMM the f32 `rows` (the token rows of the gate and up input, or
/// the slot rows of the down input when `per_slot`), so a stage quantizes only for the stacks
/// that need it.
pub(crate) struct RoutedX<'a> {
    pub(crate) q8: DeviceRange<'a>,
    pub(crate) rows: DeviceRange<'a>,
    pub(crate) row_stride: usize,
    pub(crate) per_slot: bool,
}

/// One routed GEMM of `weight` over `ids`: the dequantizing GEMM for the in-house types, the
/// vendored grouped MMQ for the rest. `dst` receives the f32 slot rows at stride `ld`.
#[allow(clippy::too_many_arguments)] // one launch description: weight, ids, input, output, scratch
pub(crate) fn routed_gemm(
    ctx: &CudaContext,
    stream: &DeviceStream,
    weight: &NativeWeight,
    ids: &MmidIds<'_>,
    x: &RoutedX<'_>,
    ld: usize,
    dst: DeviceRange<'_>,
    fixup: DeviceRange<'_>,
) -> Result<(), EngError> {
    let n = weight.n_rows() / weight.experts().unwrap_or(1);
    let routed = routed_weight(weight, weight.k(), n)?;
    if weight.quant().uses_dequantizing_gemm() {
        ctx.gemm_dq_mmid(
            stream,
            DqMmidCall {
                shape: ids.shape(),
                weight: routed,
                ids,
                x: x.rows,
                x_row_stride: x.row_stride,
                x_per_slot: x.per_slot,
                y: dst,
                ld,
            },
        )
    } else {
        ctx.mmq_mmid_launch(
            stream,
            MmidLaunchCall {
                shape: ids.shape(),
                weight: routed,
                ids,
                y: x.q8,
                dst,
                ld,
                fixup,
            },
        )
    }
}

/// The q8_1 quantize feeding the stacks of one stage (`gather` for the down input, else the
/// scatter of the gate and up input): one quantize at the layout of the first stack that rides
/// the vendored MMQ, none when every stack rides the dequantizing GEMM.
#[allow(clippy::too_many_arguments)] // one quantize description: ids, stacks, packing, shape, buffers
pub(crate) fn routed_quantize(
    ctx: &CudaContext,
    stream: &DeviceStream,
    ids: &MmidIds<'_>,
    stacks: &[&NativeWeight],
    gather: bool,
    packing: crate::kernels::cuda::prefill::mmq_quant::MmqPacking,
    k: usize,
    x: DeviceRange<'_>,
    y: DeviceRange<'_>,
) -> Result<(), EngError> {
    let Some(vendored) = stacks.iter().find(|w| !w.quant().uses_dequantizing_gemm()) else {
        return Ok(());
    };
    let call = MmidQuantCall {
        packing,
        quant: vendored.quant(),
        k,
        row_stride: k,
        x,
        y,
    };
    if gather {
        ctx.mmq_quantize_gather_ids(stream, ids, call)
    } else {
        ctx.mmq_quantize_scatter_ids(stream, ids, call)
    }
}
