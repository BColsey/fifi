//! Raw FFI to `inference_cuda_gemm_dq_q5k_launch` of `native/prefill/gemm_dq.cu` (declared in
//! `native/prefill/gemm_dq.h`). Child of `native::prefill`, so it inherits the
//! `allow(unsafe_code)` of `native.rs`.
//!
//! Safety invariants this module relies on (mirrors `matvec_q5k`):
//! - A device range is proved in bounds by `RawRange::new` before it gets here. This module
//!   proves, again and on the sizes of the call, that `w` holds `rows` rows of `k / 256`
//!   native Q5_K blocks, that `x` holds `k * ncols` f16 values, that `y` holds
//!   `rows * ncols` floats, and that `y`, which the kernel writes, shares no byte with `w`
//!   or `x`. The C entry repeats the checks on the addresses.
//! - The launch entry only queues work on the stream it is given; a fault in the kernel
//!   surfaces at the next synchronize.
//! - No raw pointer and no unsafe function leaves this module.

use std::ffi::{c_int, c_void};
use std::mem::offset_of;

use super::{DeviceFailure, RawRange, RawStream, STAGE_NONE, Span, checked, invalid_argument};

/// Mirror of `inference_cuda_gemm_dq_args` (`gemm_dq.h`), field for field.
/// The layout is pinned as well by the `static_assert` of the header.
#[repr(C)]
struct GemmDqArgsRepr {
    w: *const c_void,
    x: *const c_void,
    y: *mut f32,
    k: c_int,
    rows: c_int,
    ncols: c_int,
}

const _: () = assert!(
    size_of::<GemmDqArgsRepr>() == 40
        && offset_of!(GemmDqArgsRepr, w) == 0
        && offset_of!(GemmDqArgsRepr, x) == 8
        && offset_of!(GemmDqArgsRepr, y) == 16
        && offset_of!(GemmDqArgsRepr, k) == 24
        && offset_of!(GemmDqArgsRepr, rows) == 28
        && offset_of!(GemmDqArgsRepr, ncols) == 32
);

unsafe extern "C" {
    #[allow(dead_code)] // Staged until a model caller is qualified.
    #[link_name = "inference_cuda_gemm_dq_q5k_launch"]
    fn gemm_dq_q5k_launch_raw(
        args: *const GemmDqArgsRepr,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
}

/// The Q5_K dequantizing GEMM call: `w` native blocks (`rows * k / 256 * 176` bytes), `x` the
/// f16 activation row-major by `k` (`k * ncols` values), `y` the f32 output row-major by
/// `rows` (`rows * ncols` values).
#[allow(dead_code)] // Staged until a model caller is qualified.
pub(in crate::kernels::cuda) struct GemmDqQ5kRaw<'a> {
    pub(in crate::kernels::cuda) w: RawRange<'a>,
    pub(in crate::kernels::cuda) x: RawRange<'a>,
    pub(in crate::kernels::cuda) y: RawRange<'a>,
    pub(in crate::kernels::cuda) k: usize,
    pub(in crate::kernels::cuda) rows: usize,
    pub(in crate::kernels::cuda) ncols: usize,
}

#[allow(dead_code)] // Staged until a model caller is qualified.
pub(in crate::kernels::cuda) fn check_gemm_dq_q5k(
    call: &GemmDqQ5kRaw<'_>,
) -> Result<(), DeviceFailure> {
    let (k, rows, ncols) = (call.k, call.rows, call.ncols);
    let need_w = rows.checked_mul(k / 256).and_then(|n| n.checked_mul(176));
    let ints = [
        c_int::try_from(k).ok(),
        c_int::try_from(rows).ok(),
        c_int::try_from(ncols).ok(),
    ];
    let Some(need_w) = need_w else {
        return Err(invalid_argument());
    };
    let need_x = k.checked_mul(ncols).and_then(|n| n.checked_mul(2));
    let need_y = rows.checked_mul(ncols).and_then(|n| n.checked_mul(4));
    let (Some(need_x), Some(need_y)) = (need_x, need_y) else {
        return Err(invalid_argument());
    };
    let aligned = call.w.offset().is_multiple_of(4)
        && call.x.offset().is_multiple_of(2)
        && call.y.offset().is_multiple_of(4);
    let y = Span::of(&call.y, need_y);
    let apart = y.apart(&Span::of(&call.w, need_w)) && y.apart(&Span::of(&call.x, need_x));
    if ints.iter().any(|int| int.is_none())
        || !aligned
        || !apart
        || call.w.bytes() < need_w
        || call.x.bytes() < need_x
        || call.y.bytes() < need_y
        || k == 0
        || rows == 0
        || ncols == 0
        || !k.is_multiple_of(256)
    {
        return Err(invalid_argument());
    }
    Ok(())
}

#[allow(dead_code)] // Staged until a model caller is qualified.
pub(in crate::kernels::cuda) fn gemm_dq_q5k_launch(
    stream: &RawStream,
    call: &GemmDqQ5kRaw<'_>,
) -> Result<(), DeviceFailure> {
    check_gemm_dq_q5k(call)?;
    let mut stage = STAGE_NONE;
    // SAFETY: `check_gemm_dq_q5k` proved the `w`, `x` and `y` ranges inside their live
    // allocations, large enough for the call, aligned, and `y` disjoint from the ranges the
    // kernel reads; the adapter repeats the checks on the addresses. `stream` is live for the
    // call.
    let args = GemmDqArgsRepr {
        w: call.w.address().cast_const(),
        x: call.x.address().cast_const(),
        y: call.y.address().cast::<f32>(),
        k: c_int::try_from(call.k).map_err(|_| invalid_argument())?,
        rows: c_int::try_from(call.rows).map_err(|_| invalid_argument())?,
        ncols: c_int::try_from(call.ncols).map_err(|_| invalid_argument())?,
    };
    let status = unsafe { gemm_dq_q5k_launch_raw(&args, stream.as_ptr(), &mut stage) };
    checked(status, stage)
}
