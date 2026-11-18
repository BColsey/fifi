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
    /// 0 reads x row-major by k; otherwise x is [ncols][x_col_stride] token rows.
    x_col_stride: c_int,
    /// 0 writes y row-major by rows; otherwise y is [ncols][y_col_stride] token rows.
    y_col_stride: c_int,
}

const _: () = assert!(
    size_of::<GemmDqArgsRepr>() == 48
        && offset_of!(GemmDqArgsRepr, w) == 0
        && offset_of!(GemmDqArgsRepr, x) == 8
        && offset_of!(GemmDqArgsRepr, y) == 16
        && offset_of!(GemmDqArgsRepr, k) == 24
        && offset_of!(GemmDqArgsRepr, rows) == 28
        && offset_of!(GemmDqArgsRepr, ncols) == 32
        && offset_of!(GemmDqArgsRepr, x_col_stride) == 36
        && offset_of!(GemmDqArgsRepr, y_col_stride) == 40
);

unsafe extern "C" {
    #[allow(dead_code)] // Staged until a model caller is qualified.
    #[link_name = "inference_cuda_gemm_dq_q5k_launch"]
    fn gemm_dq_q5k_launch_raw(
        args: *const GemmDqArgsRepr,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
    #[allow(dead_code)] // Staged until a model caller is qualified.
    #[link_name = "inference_cuda_gemm_dq_q3k_launch"]
    fn gemm_dq_q3k_launch_raw(
        args: *const GemmDqArgsRepr,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
    #[allow(dead_code)] // Staged until a model caller is qualified.
    #[link_name = "inference_cuda_gemm_dq_iq4xs_launch"]
    fn gemm_dq_iq4xs_launch_raw(
        args: *const GemmDqArgsRepr,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
    #[allow(dead_code)] // Staged until a model caller is qualified.
    #[link_name = "inference_cuda_gemm_dq_q40_launch"]
    fn gemm_dq_q40_launch_raw(
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
    pub(in crate::kernels::cuda) layout: GemmDqLayout,
}

/// The size/apartness contract shared by every gemm_dq type: `w` holds `rows` rows of
/// `k / qk` blocks of `block_bytes`, `x` holds `k * ncols` f16 values, `y` holds
/// `rows * ncols` floats and shares no byte with `w` or `x`, and `k` is a positive
/// multiple of both the block size and the kernel's 64-value chunk.
struct GemmDqGeom {
    qk: usize,
    block_bytes: usize,
    w_align: usize,
}

/// The x and y layouts: 0/0 keeps the row-major [k][ncols] and [rows][ncols]
/// forms; otherwise x holds token rows `[ncols][x_col_stride]` and y holds
/// token rows `[ncols][y_col_stride]`, which is how the prefill buffers are
/// cut.
pub(in crate::kernels::cuda) struct GemmDqLayout {
    pub(in crate::kernels::cuda) x_col_stride: usize,
    pub(in crate::kernels::cuda) y_col_stride: usize,
}

#[allow(clippy::too_many_arguments)] // w/x/y plus the GEMM shape, layout, and geometry
fn check_gemm_dq(
    w: &RawRange<'_>,
    x: &RawRange<'_>,
    y: &RawRange<'_>,
    k: usize,
    rows: usize,
    ncols: usize,
    layout: &GemmDqLayout,
    geom: &GemmDqGeom,
) -> Result<(), DeviceFailure> {
    let GemmDqGeom {
        qk,
        block_bytes,
        w_align,
    } = *geom;
    let GemmDqLayout {
        x_col_stride,
        y_col_stride,
    } = *layout;
    let need_w = rows
        .checked_mul(k / qk)
        .and_then(|n| n.checked_mul(block_bytes));
    let ints = [
        c_int::try_from(k).ok(),
        c_int::try_from(rows).ok(),
        c_int::try_from(ncols).ok(),
        if x_col_stride == 0 {
            Some(0)
        } else {
            c_int::try_from(x_col_stride).ok()
        },
        if y_col_stride == 0 {
            Some(0)
        } else {
            c_int::try_from(y_col_stride).ok()
        },
    ];
    let Some(need_w) = need_w else {
        return Err(invalid_argument());
    };
    let need_x = if x_col_stride == 0 {
        k.checked_mul(ncols).and_then(|n| n.checked_mul(2))
    } else if x_col_stride < k || ncols > i32::MAX as usize / x_col_stride {
        None
    } else {
        (ncols - 1)
            .checked_mul(x_col_stride)
            .and_then(|head| head.checked_add(k))
            .and_then(|n| n.checked_mul(2))
    };
    let need_y = if y_col_stride == 0 {
        rows.checked_mul(ncols).and_then(|n| n.checked_mul(4))
    } else if y_col_stride < rows || ncols > i32::MAX as usize / y_col_stride {
        None
    } else {
        (ncols - 1)
            .checked_mul(y_col_stride)
            .and_then(|head| head.checked_add(rows))
            .and_then(|n| n.checked_mul(4))
    };
    let (Some(need_x), Some(need_y)) = (need_x, need_y) else {
        return Err(invalid_argument());
    };
    let aligned = w.offset().is_multiple_of(w_align)
        && x.offset().is_multiple_of(2)
        && y.offset().is_multiple_of(4);
    let y_span = Span::of(y, need_y);
    let apart = y_span.apart(&Span::of(w, need_w)) && y_span.apart(&Span::of(x, need_x));
    if ints.iter().any(|int| int.is_none())
        || !aligned
        || !apart
        || w.bytes() < need_w
        || x.bytes() < need_x
        || y.bytes() < need_y
        || k == 0
        || rows == 0
        || ncols == 0
        || !k.is_multiple_of(qk)
        || !k.is_multiple_of(64)
    {
        return Err(invalid_argument());
    }
    Ok(())
}

fn gemm_dq_args(
    w: &RawRange<'_>,
    x: &RawRange<'_>,
    y: &RawRange<'_>,
    k: usize,
    rows: usize,
    ncols: usize,
    layout: &GemmDqLayout,
) -> Result<GemmDqArgsRepr, DeviceFailure> {
    let stride_arg = |stride: usize| -> Result<c_int, DeviceFailure> {
        c_int::try_from(stride).map_err(|_| invalid_argument())
    };
    Ok(GemmDqArgsRepr {
        w: w.address().cast_const(),
        x: x.address().cast_const(),
        y: y.address().cast::<f32>(),
        k: c_int::try_from(k).map_err(|_| invalid_argument())?,
        rows: c_int::try_from(rows).map_err(|_| invalid_argument())?,
        ncols: c_int::try_from(ncols).map_err(|_| invalid_argument())?,
        x_col_stride: stride_arg(layout.x_col_stride)?,
        y_col_stride: stride_arg(layout.y_col_stride)?,
    })
}

#[allow(dead_code)] // Staged until a model caller is qualified.
pub(in crate::kernels::cuda) fn check_gemm_dq_q5k(
    call: &GemmDqQ5kRaw<'_>,
) -> Result<(), DeviceFailure> {
    check_gemm_dq(
        &call.w,
        &call.x,
        &call.y,
        call.k,
        call.rows,
        call.ncols,
        &call.layout,
        &GemmDqGeom {
            qk: 256,
            block_bytes: 176,
            w_align: 16,
        },
    )
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
    let args = gemm_dq_args(
        &call.w,
        &call.x,
        &call.y,
        call.k,
        call.rows,
        call.ncols,
        &call.layout,
    )?;
    let status = unsafe { gemm_dq_q5k_launch_raw(&args, stream.as_ptr(), &mut stage) };
    checked(status, stage)
}

/// The Q3_K dequantizing GEMM call: `w` native blocks (`rows * k / 256 * 110` bytes), `x`
/// the f16 activation row-major by `k` (`k * ncols` values), `y` the f32 output row-major by
/// `rows` (`rows * ncols` values).
#[allow(dead_code)] // Staged until a model caller is qualified.
pub(in crate::kernels::cuda) struct GemmDqQ3kRaw<'a> {
    pub(in crate::kernels::cuda) w: RawRange<'a>,
    pub(in crate::kernels::cuda) x: RawRange<'a>,
    pub(in crate::kernels::cuda) y: RawRange<'a>,
    pub(in crate::kernels::cuda) k: usize,
    pub(in crate::kernels::cuda) rows: usize,
    pub(in crate::kernels::cuda) ncols: usize,
    pub(in crate::kernels::cuda) layout: GemmDqLayout,
}

#[allow(dead_code)] // Staged until a model caller is qualified.
pub(in crate::kernels::cuda) fn gemm_dq_q3k_launch(
    stream: &RawStream,
    call: &GemmDqQ3kRaw<'_>,
) -> Result<(), DeviceFailure> {
    check_gemm_dq(
        &call.w,
        &call.x,
        &call.y,
        call.k,
        call.rows,
        call.ncols,
        &call.layout,
        &GemmDqGeom {
            qk: 256,
            block_bytes: 110,
            w_align: 2,
        },
    )?;
    let mut stage = STAGE_NONE;
    // SAFETY: as `gemm_dq_q5k_launch`, with the Q3_K geometry (110-byte blocks, 2-byte
    // weight alignment).
    let args = gemm_dq_args(
        &call.w,
        &call.x,
        &call.y,
        call.k,
        call.rows,
        call.ncols,
        &call.layout,
    )?;
    let status = unsafe { gemm_dq_q3k_launch_raw(&args, stream.as_ptr(), &mut stage) };
    checked(status, stage)
}

/// The IQ4_XS dequantizing GEMM call: `w` native super-blocks (`rows * k / 256 * 136`
/// bytes), `x` the f16 activation row-major by `k` (`k * ncols` values), `y` the f32
/// output row-major by `rows` (`rows * ncols` values).
#[allow(dead_code)] // Staged until a model caller is qualified.
pub(in crate::kernels::cuda) struct GemmDqIq4xsRaw<'a> {
    pub(in crate::kernels::cuda) w: RawRange<'a>,
    pub(in crate::kernels::cuda) x: RawRange<'a>,
    pub(in crate::kernels::cuda) y: RawRange<'a>,
    pub(in crate::kernels::cuda) k: usize,
    pub(in crate::kernels::cuda) rows: usize,
    pub(in crate::kernels::cuda) ncols: usize,
    pub(in crate::kernels::cuda) layout: GemmDqLayout,
}

#[allow(dead_code)] // Staged until a model caller is qualified.
pub(in crate::kernels::cuda) fn gemm_dq_iq4xs_launch(
    stream: &RawStream,
    call: &GemmDqIq4xsRaw<'_>,
) -> Result<(), DeviceFailure> {
    check_gemm_dq(
        &call.w,
        &call.x,
        &call.y,
        call.k,
        call.rows,
        call.ncols,
        &call.layout,
        &GemmDqGeom {
            qk: 256,
            block_bytes: 136,
            w_align: 2,
        },
    )?;
    let mut stage = STAGE_NONE;
    // SAFETY: as `gemm_dq_q5k_launch`, with the IQ4_XS geometry (136-byte super-blocks,
    // 2-byte weight alignment).
    let args = gemm_dq_args(
        &call.w,
        &call.x,
        &call.y,
        call.k,
        call.rows,
        call.ncols,
        &call.layout,
    )?;
    let status = unsafe { gemm_dq_iq4xs_launch_raw(&args, stream.as_ptr(), &mut stage) };
    checked(status, stage)
}

/// The Q4_0 dequantizing GEMM call: `w` native blocks (`rows * k / 32 * 18` bytes), `x` the
/// f16 activation row-major by `k` (`k * ncols` values), `y` the f32 output row-major by
/// `rows` (`rows * ncols` values).
#[allow(dead_code)] // Staged until a model caller is qualified.
pub(in crate::kernels::cuda) struct GemmDqQ40Raw<'a> {
    pub(in crate::kernels::cuda) w: RawRange<'a>,
    pub(in crate::kernels::cuda) x: RawRange<'a>,
    pub(in crate::kernels::cuda) y: RawRange<'a>,
    pub(in crate::kernels::cuda) k: usize,
    pub(in crate::kernels::cuda) rows: usize,
    pub(in crate::kernels::cuda) ncols: usize,
    pub(in crate::kernels::cuda) layout: GemmDqLayout,
}

#[allow(dead_code)] // Staged until a model caller is qualified.
pub(in crate::kernels::cuda) fn gemm_dq_q40_launch(
    stream: &RawStream,
    call: &GemmDqQ40Raw<'_>,
) -> Result<(), DeviceFailure> {
    check_gemm_dq(
        &call.w,
        &call.x,
        &call.y,
        call.k,
        call.rows,
        call.ncols,
        &call.layout,
        &GemmDqGeom {
            qk: 32,
            block_bytes: 18,
            w_align: 2,
        },
    )?;
    let mut stage = STAGE_NONE;
    // SAFETY: as `gemm_dq_q5k_launch`, with the Q4_0 geometry (18-byte blocks, 2-byte
    // weight alignment).
    let args = gemm_dq_args(
        &call.w,
        &call.x,
        &call.y,
        call.k,
        call.rows,
        call.ncols,
        &call.layout,
    )?;
    let status = unsafe { gemm_dq_q40_launch_raw(&args, stream.as_ptr(), &mut stage) };
    checked(status, stage)
}
