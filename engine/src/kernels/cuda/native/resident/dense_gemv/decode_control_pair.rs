//! Pair two *already planned* independent Inkling controller outputs. A valid
//! sequential group can contain dependencies: never infer independence merely
//! from matching shapes, or reorder an intervening call.
use super::*;

unsafe extern "C" {
    fn inference_cuda_decode_control_pair(
        a: *const DenseGemvArgsRaw,
        b: *const DenseGemvArgsRaw,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
}

pub(in crate::kernels::cuda) fn try_control_pair(
    stream: &RawStream,
    a: &DenseGemvRaw<'_>,
    b: &DenseGemvRaw<'_>,
) -> Result<bool, DeviceFailure> {
    let eligible = |p: &DenseGemvRaw<'_>| {
        p.dtype == DenseDtype::F32
            && (p.dims.m, p.dims.cols, p.dims.rows) == (1, 64, 2304)
            && p.w.address().addr().is_multiple_of(16)
    };
    if !eligible(a) || !eligible(b) || a.x.address() != b.x.address() {
        return Ok(false);
    }
    let need = typed_gemv_need(a.dims, a.dtype).ok_or_else(invalid_argument)?;
    check(a, need)?;
    check(
        b,
        typed_gemv_need(b.dims, b.dtype).ok_or_else(invalid_argument)?,
    )?;
    let ya = Span::of(&a.y, need.y);
    let yb = Span::of(&b.y, need.y);
    if !ya.apart(&yb) || !ya.apart(&Span::of(&b.w, need.w)) || !yb.apart(&Span::of(&a.w, need.w)) {
        return Ok(false);
    }
    // M=1, so no input or output row stride is traversed.
    let args = |p: &DenseGemvRaw<'_>| DenseGemvArgsRaw {
        w: p.w.address().cast_const(),
        x: p.x.address().cast_const(),
        y: p.y.address(),
        m: 1,
        rows: 2304,
        cols: 64,
        ld_x: 64,
        ld_y: 2304,
        weight_dtype: DenseDtype::F32.code(),
        flags: super::flags(p),
    };
    let (a, b) = (args(a), args(b));
    let mut stage = STAGE_NONE;
    // SAFETY: both compact ranges passed the ordinary live-allocation extent
    // and alias checks. The common input is read-only, and neither output
    // aliases either weight or the other output. The native entry rechecks
    // addresses. The argument structs live through the call and the device
    // buffers outlive execution (free synchronizes).
    let status =
        unsafe { inference_cuda_decode_control_pair(&a, &b, stream.0.as_ptr(), &mut stage) };
    checked(status, stage)?;
    Ok(true)
}
