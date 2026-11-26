//! Test-only FFI: live, aligned, sized native Q6_K and F32 ranges; output disjoint from inputs.
//! The caller owns the stream and allocations until queued work completes.
use super::{DeviceFailure, RawRange, RawStream, STAGE_NONE, Span, checked, invalid_argument};
use std::{
    ffi::{c_int, c_void},
    mem::offset_of,
};

#[repr(C)]
struct Args {
    w: *const c_void,
    x: *const c_void,
    y: *mut c_void,
    rows: c_int,
    k: c_int,
}
const _: () = assert!(
    size_of::<Args>() == 32
        && offset_of!(Args, w) == 0
        && offset_of!(Args, x) == 8
        && offset_of!(Args, y) == 16
        && offset_of!(Args, rows) == 24
        && offset_of!(Args, k) == 28
);
unsafe extern "C" {
    #[link_name = "inference_cuda_q6_f32_launch"]
    fn launch_raw(args: *const Args, stream: *mut c_void, stage: *mut c_int) -> c_int;
}

pub(in crate::kernels::cuda) fn sizes(rows: usize, k: usize) -> Option<(usize, usize, usize)> {
    if rows == 0
        || rows > i32::MAX as usize - 7
        || k == 0
        || !k.is_multiple_of(256)
        || i32::try_from(k).is_err()
    {
        return None;
    }
    Some((
        rows.checked_mul(k / 256)?.checked_mul(210)?,
        k.checked_mul(4)?,
        rows.checked_mul(4)?,
    ))
}

pub(in crate::kernels::cuda) struct Q6F32Raw<'a> {
    pub(in crate::kernels::cuda) w: RawRange<'a>,
    pub(in crate::kernels::cuda) x: RawRange<'a>,
    pub(in crate::kernels::cuda) y: RawRange<'a>,
    pub(in crate::kernels::cuda) rows: usize,
    pub(in crate::kernels::cuda) k: usize,
}

pub(in crate::kernels::cuda) fn q6_f32_launch(
    stream: &RawStream,
    call: &Q6F32Raw<'_>,
) -> Result<(), DeviceFailure> {
    let (w_bytes, x_bytes, y_bytes) = sizes(call.rows, call.k).ok_or_else(invalid_argument)?;
    if call.w.bytes < w_bytes
        || call.x.bytes < x_bytes
        || call.y.bytes < y_bytes
        || !call.w.offset.is_multiple_of(2)
        || !call.x.offset.is_multiple_of(4)
        || !call.y.offset.is_multiple_of(4)
        || !Span::of(&call.y, y_bytes).apart(&Span::of(&call.w, w_bytes))
        || !Span::of(&call.y, y_bytes).apart(&Span::of(&call.x, x_bytes))
    {
        return Err(invalid_argument());
    }
    let args = Args {
        w: call.w.address().cast_const(),
        x: call.x.address().cast_const(),
        y: call.y.address(),
        rows: c_int::try_from(call.rows).map_err(|_| invalid_argument())?,
        k: c_int::try_from(call.k).map_err(|_| invalid_argument())?,
    };
    let mut stage = STAGE_NONE;
    // SAFETY: ranges are live, sufficiently sized, aligned, and output-disjoint; the C entry
    // rechecks shape/alignment/overlap and enqueues only on the borrowed live stream.
    let status = unsafe { launch_raw(&args, stream.as_ptr(), &mut stage) };
    checked(status, stage)
}
