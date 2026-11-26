//! Single-row attention group: Q, K, V, relative, sink, gate, optional two conv scales.
//! The ordinary group executor allows dependencies. Require independence of all
//! writes before substituting one unordered grid for its stream-ordered calls.
use super::*;

unsafe extern "C" {
    fn inference_cuda_decode_attention_group(
        calls: *const DenseGemvArgsRaw,
        count: c_int,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
}

pub(in crate::kernels::cuda) fn try_attention_group<'a, 'b: 'a>(
    stream: &RawStream,
    calls: impl ExactSizeIterator<Item = &'a DenseGemvRaw<'b>> + Clone,
) -> Result<bool, DeviceFailure> {
    use GgufType::{Iq3Xxs, Q4K, Q5K, Q6K, Q80};
    let count = calls.len();
    if !matches!(count, 6 | 8) {
        return Ok(false);
    }
    let rows = [3072, 1024, 1024, 384, 24, 3072, 32, 32];
    for (i, p) in calls.clone().enumerate() {
        let dtype_ok = if matches!(i, 4 | 6 | 7) {
            p.dtype == DenseDtype::Bf16 && p.w.address().addr().is_multiple_of(16)
        } else {
            match p.dtype {
                DenseDtype::PackedGguf(Q80) => true,
                DenseDtype::PackedGguf(Iq3Xxs) => rows[i] == 3072,
                DenseDtype::PackedGguf(Q4K | Q5K | Q6K) => p.w.address().addr().is_multiple_of(4),
                _ => false,
            }
        };
        if !dtype_ok
            || (p.dims.m, p.dims.cols, p.dims.rows) != (1, 2304, rows[i])
            || !p.x.address().addr().is_multiple_of(16)
        {
            return Ok(false);
        }
    }
    let mut args = Vec::with_capacity(count);
    let mut spans = Vec::with_capacity(count);
    for p in calls {
        let need = typed_gemv_need(p.dims, p.dtype).ok_or_else(invalid_argument)?;
        check(p, need)?;
        spans.push((
            Span::of(&p.w, need.w),
            Span::of(&p.x, need.x),
            Span::of(&p.y, need.y),
        ));
        args.push(DenseGemvArgsRaw {
            w: p.w.address().cast_const(),
            x: p.x.address().cast_const(),
            y: p.y.address(),
            m: 1,
            rows: p.dims.rows as c_int,
            cols: 2304,
            ld_x: 2304,
            ld_y: p.dims.rows as c_int,
            weight_dtype: p.dtype.code(),
            flags: super::flags(p),
        });
    }
    for (i, (_, _, dst)) in spans.iter().enumerate() {
        for (j, (w, x, y)) in spans.iter().enumerate() {
            if i != j && !(dst.apart(w) && dst.apart(x) && dst.apart(y)) {
                return Ok(false);
            }
        }
    }
    let mut stage = STAGE_NONE;
    // SAFETY: check() validates live ranges, sizes, alignments and intra-call
    // aliasing. The loop above validates all cross-call writes. The entry
    // copies these scalar descriptors before returning and repeats the checks.
    let status = unsafe {
        inference_cuda_decode_attention_group(
            args.as_ptr(),
            count as c_int,
            stream.0.as_ptr(),
            &mut stage,
        )
    };
    checked(status, stage)?;
    Ok(true)
}
