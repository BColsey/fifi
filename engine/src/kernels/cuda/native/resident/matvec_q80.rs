//! Raw FFI to `inference_cuda_matvec_q80_launch` of `native/resident/matvec_q80.cu`
//! (declared in `native/resident/matvec_q80.h`). Child of `native::resident`, so it inherits
//! the `allow(unsafe_code)` of `native.rs`.
//!
//! Safety invariants this module relies on (mirrors `matvec_q40`):
//! - A device range is proved in bounds by `RawRange::new` before it gets here. This module
//!   proves, again and on the sizes of the call, that `w` holds `rows` rows of `k/32` native
//!   Q8_0 blocks (34 bytes each), that `x_q8` holds the padded blocks of `ncols` rows, that
//!   `y` holds `ncols * rows` floats, that `y`, which the kernel writes, shares no byte with
//!   `w` or `x_q8`, and that every range is aligned (2 bytes for the weights, whose f16
//!   scales sit at even offsets; 4 for the rest). The C entry repeats the checks on the
//!   addresses.
//! - The launch entry only queues work on the stream it is given; a fault in the kernel
//!   surfaces at the next synchronize.
//! - The allocations stay valid until the queued launch finishes because `inference_cuda_free`
//!   synchronizes the whole device before `cudaFree`.
//! - No raw pointer and no unsafe function leaves this module.

use std::ffi::{c_int, c_void};
use std::mem::offset_of;

use super::{DeviceFailure, RawRange, RawStream, STAGE_NONE, Span, checked, invalid_argument};

/// The Q8_0 block: 32 values, 34 bytes on disk (f16 scale, 32 int8 quants). The inline
/// reference of `resident/matvec_q/tests.rs` is the CPU oracle of the layout.
pub(in crate::kernels::cuda) const Q80_BLOCK: usize = 34;

/// Mirror of `inference_cuda_matvec_q80_args` (`matvec_q80.h`), field for field.
/// The layout is pinned as well by the `static_assert` of the header.
#[repr(C)]
struct MatvecQ80ArgsRepr {
    w: *const c_void,
    x_q8: *const c_void,
    y: *mut f32,
    k: c_int,
    rows: c_int,
    ncols: c_int,
}

const _: () = assert!(
    size_of::<MatvecQ80ArgsRepr>() == 40
        && offset_of!(MatvecQ80ArgsRepr, w) == 0
        && offset_of!(MatvecQ80ArgsRepr, x_q8) == 8
        && offset_of!(MatvecQ80ArgsRepr, y) == 16
        && offset_of!(MatvecQ80ArgsRepr, k) == 24
        && offset_of!(MatvecQ80ArgsRepr, rows) == 28
        && offset_of!(MatvecQ80ArgsRepr, ncols) == 32
);

unsafe extern "C" {
    #[link_name = "inference_cuda_matvec_q80_launch"]
    fn matvec_q80_launch_raw(
        args: *const MatvecQ80ArgsRepr,
        x_row_bytes: usize,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
}

/// The in-house Q8_0 matvec call: `w` native blocks, `x_q8` the padded `quantize_q8_1` rows,
/// `x_row_bytes` one activation row's stride (`q8_1_bytes(k, 1)`), `y` f32 out with column `j`
/// at `y + j * rows`. `rows` may be any positive count: the kernel owns whole rows, unlike
/// the vendored matvec's row groups.
pub(in crate::kernels::cuda) struct MatvecQ80Raw<'a> {
    pub(in crate::kernels::cuda) w: RawRange<'a>,
    pub(in crate::kernels::cuda) x_q8: RawRange<'a>,
    pub(in crate::kernels::cuda) y: RawRange<'a>,
    pub(in crate::kernels::cuda) k: usize,
    pub(in crate::kernels::cuda) rows: usize,
    pub(in crate::kernels::cuda) ncols: usize,
}

pub(in crate::kernels::cuda) fn check_matvec_q80(
    call: &MatvecQ80Raw<'_>,
    x_row_bytes: usize,
) -> Result<(), DeviceFailure> {
    let (k, rows, ncols) = (call.k, call.rows, call.ncols);
    let blocks = k / 32;
    let need_w = rows
        .checked_mul(blocks)
        .and_then(|n| n.checked_mul(Q80_BLOCK));
    let ints = [
        c_int::try_from(k).ok(),
        c_int::try_from(rows).ok(),
        c_int::try_from(ncols).ok(),
    ];
    let Some(need_w) = need_w else {
        return Err(invalid_argument());
    };
    let aligned = call.w.offset.is_multiple_of(2)
        && call.x_q8.offset.is_multiple_of(4)
        && call.y.offset.is_multiple_of(4);
    let y = Span::of(&call.y, rows.saturating_mul(ncols).saturating_mul(4));
    let apart = y.apart(&Span::of(&call.w, need_w)) && y.apart(&Span::of(&call.x_q8, x_row_bytes));
    if ints.iter().any(|int| int.is_none())
        || !aligned
        || !apart
        || x_row_bytes == 0
        || call.x_q8.bytes < x_row_bytes.saturating_mul(ncols)
        || call.y.bytes < rows.saturating_mul(ncols).saturating_mul(4)
        || call.w.bytes < need_w
        || k == 0
        || rows == 0
        || !(1..=8).contains(&ncols)
        || !k.is_multiple_of(32)
    {
        return Err(invalid_argument());
    }
    Ok(())
}

pub(in crate::kernels::cuda) fn matvec_q80_launch(
    stream: &RawStream,
    call: &MatvecQ80Raw<'_>,
    x_row_bytes: usize,
) -> Result<(), DeviceFailure> {
    check_matvec_q80(call, x_row_bytes)?;
    let mut stage = STAGE_NONE;
    // SAFETY: `check_matvec_q80` proved the `w`, `x_q8` and `y` ranges inside their live
    // allocations, large enough for the call, aligned, and `y` disjoint from the ranges the
    // kernel reads; the adapter repeats the checks on the addresses. The allocations stay
    // valid until the queued launch finishes because `inference_cuda_free` synchronizes the
    // device first. `stream` is live for the call.
    let args = MatvecQ80ArgsRepr {
        w: call.w.address().cast_const(),
        x_q8: call.x_q8.address().cast_const(),
        y: call.y.address().cast::<f32>(),
        k: c_int::try_from(call.k).map_err(|_| invalid_argument())?,
        rows: c_int::try_from(call.rows).map_err(|_| invalid_argument())?,
        ncols: c_int::try_from(call.ncols).map_err(|_| invalid_argument())?,
    };
    let status =
        unsafe { matvec_q80_launch_raw(&args, x_row_bytes, stream.0.as_ptr(), &mut stage) };
    checked(status, stage)
}
