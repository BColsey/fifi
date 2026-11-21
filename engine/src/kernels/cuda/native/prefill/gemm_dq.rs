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
use crate::kernels::cuda::native::resident::matvec_gguf::{gguf_type_id, table_of};
use crate::kernels::cuda::resident::matvec_q::QuantType;

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
    /// 1: x holds f32 values converted on load; 0: f16.
    x_f32: c_int,
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
        && offset_of!(GemmDqArgsRepr, x_f32) == 44
);

/// Mirror of `inference_cuda_gemm_dq_mmid_args` (`gemm_dq.h`), field for field.
#[repr(C)]
struct GemmDqMmidArgsRepr {
    w: *const c_void,
    x: *const f32,
    y: *mut f32,
    ids_dst: *const c_int,
    expert_bounds: *const c_int,
    expert_stride: u64,
    k: c_int,
    rows: c_int,
    n_experts: c_int,
    total_cols: c_int,
    x_div: c_int,
    x_row_stride: c_int,
    y_row_stride: c_int,
    col_tile: c_int,
}

const _: () = assert!(
    size_of::<GemmDqMmidArgsRepr>() == 80
        && offset_of!(GemmDqMmidArgsRepr, ids_dst) == 24
        && offset_of!(GemmDqMmidArgsRepr, expert_stride) == 40
        && offset_of!(GemmDqMmidArgsRepr, k) == 48
        && offset_of!(GemmDqMmidArgsRepr, y_row_stride) == 72
);

unsafe extern "C" {
    #[link_name = "inference_cuda_gemm_dq_q5k_launch"]
    fn gemm_dq_q5k_launch_raw(
        args: *const GemmDqArgsRepr,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
    #[link_name = "inference_cuda_gemm_dq_q3k_launch"]
    fn gemm_dq_q3k_launch_raw(
        args: *const GemmDqArgsRepr,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
    #[link_name = "inference_cuda_gemm_dq_codebook_launch"]
    fn gemm_dq_codebook_launch_raw(
        args: *const GemmDqArgsRepr,
        ty: c_int,
        table: *const c_void,
        table_bytes: usize,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
    #[link_name = "inference_cuda_gemm_dq_mmid_launch"]
    fn gemm_dq_mmid_launch_raw(
        args: *const GemmDqMmidArgsRepr,
        ty: c_int,
        table: *const c_void,
        table_bytes: usize,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
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
    /// x holds f32 values (the prefill norm-pack buffers) instead of f16.
    pub(in crate::kernels::cuda) x_f32: bool,
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
        x_f32,
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
    let x_unit = if x_f32 { 4 } else { 2 };
    let need_x = if x_col_stride == 0 {
        k.checked_mul(ncols).and_then(|n| n.checked_mul(x_unit))
    } else if x_col_stride < k || ncols > i32::MAX as usize / x_col_stride {
        None
    } else {
        (ncols - 1)
            .checked_mul(x_col_stride)
            .and_then(|head| head.checked_add(k))
            .and_then(|n| n.checked_mul(x_unit))
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
        && x.offset().is_multiple_of(x_unit)
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
        x_f32: i32::from(layout.x_f32),
    })
}

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
pub(in crate::kernels::cuda) struct GemmDqQ3kRaw<'a> {
    pub(in crate::kernels::cuda) w: RawRange<'a>,
    pub(in crate::kernels::cuda) x: RawRange<'a>,
    pub(in crate::kernels::cuda) y: RawRange<'a>,
    pub(in crate::kernels::cuda) k: usize,
    pub(in crate::kernels::cuda) rows: usize,
    pub(in crate::kernels::cuda) ncols: usize,
    pub(in crate::kernels::cuda) layout: GemmDqLayout,
}

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

/// The Q4_0 dequantizing GEMM call: `w` native blocks (`rows * k / 32 * 18` bytes), `x` the
/// f16 activation row-major by `k` (`k * ncols` values), `y` the f32 output row-major by
/// `rows` (`rows * ncols` values).
pub(in crate::kernels::cuda) struct GemmDqQ40Raw<'a> {
    pub(in crate::kernels::cuda) w: RawRange<'a>,
    pub(in crate::kernels::cuda) x: RawRange<'a>,
    pub(in crate::kernels::cuda) y: RawRange<'a>,
    pub(in crate::kernels::cuda) k: usize,
    pub(in crate::kernels::cuda) rows: usize,
    pub(in crate::kernels::cuda) ncols: usize,
    pub(in crate::kernels::cuda) layout: GemmDqLayout,
}

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

/// The codebook-type dequantizing GEMM call: `w` native blocks of `ty` (`rows * k / block *
/// block_bytes` bytes), `x` and `y` as in the other types.
pub(in crate::kernels::cuda) struct GemmDqCodebookRaw<'a> {
    pub(in crate::kernels::cuda) ty: QuantType,
    pub(in crate::kernels::cuda) w: RawRange<'a>,
    pub(in crate::kernels::cuda) x: RawRange<'a>,
    pub(in crate::kernels::cuda) y: RawRange<'a>,
    pub(in crate::kernels::cuda) k: usize,
    pub(in crate::kernels::cuda) rows: usize,
    pub(in crate::kernels::cuda) ncols: usize,
    pub(in crate::kernels::cuda) layout: GemmDqLayout,
}

pub(in crate::kernels::cuda) fn gemm_dq_codebook_launch(
    stream: &RawStream,
    call: &GemmDqCodebookRaw<'_>,
) -> Result<(), DeviceFailure> {
    let id = gguf_type_id(call.ty).ok_or_else(invalid_argument)?;
    if !call.ty.is_codebook() {
        return Err(invalid_argument());
    }
    check_gemm_dq(
        &call.w,
        &call.x,
        &call.y,
        call.k,
        call.rows,
        call.ncols,
        &call.layout,
        &GemmDqGeom {
            qk: call.ty.block_size(),
            block_bytes: call.ty.block_bytes(),
            w_align: 2,
        },
    )?;
    let table = table_of(call.ty);
    let mut stage = STAGE_NONE;
    // SAFETY: as `gemm_dq_q5k_launch`, with the geometry of the type (2-byte weight
    // alignment). `table` is a `'static` slice the adapter copies before it returns.
    let args = gemm_dq_args(
        &call.w,
        &call.x,
        &call.y,
        call.k,
        call.rows,
        call.ncols,
        &call.layout,
    )?;
    let status = unsafe {
        gemm_dq_codebook_launch_raw(
            &args,
            id,
            table.as_ptr().cast::<c_void>(),
            table.len(),
            stream.as_ptr(),
            &mut stage,
        )
    };
    checked(status, stage)
}

/// The grouped dequantizing GEMM call: `w` holds `n_experts` stacked weights `expert_stride`
/// bytes apart (each `rows` rows of `k` values of `ty`), `x` f32 rows of `x_row_stride`
/// floats (`x_rows` of them), `y` f32 slot rows of `y_row_stride` floats (`total_cols` of
/// them), `ids_dst` the `total_cols` token slots in compact-column order and `expert_bounds`
/// the `n_experts + 1` column bounds of the experts. Slot `s` reads x row `s / x_div` and
/// writes y row `s`.
pub(in crate::kernels::cuda) struct GemmDqMmidRaw<'a> {
    pub(in crate::kernels::cuda) ty: QuantType,
    pub(in crate::kernels::cuda) w: RawRange<'a>,
    pub(in crate::kernels::cuda) x: RawRange<'a>,
    pub(in crate::kernels::cuda) y: RawRange<'a>,
    pub(in crate::kernels::cuda) ids_dst: RawRange<'a>,
    pub(in crate::kernels::cuda) expert_bounds: RawRange<'a>,
    pub(in crate::kernels::cuda) k: usize,
    pub(in crate::kernels::cuda) rows: usize,
    pub(in crate::kernels::cuda) n_experts: usize,
    pub(in crate::kernels::cuda) total_cols: usize,
    pub(in crate::kernels::cuda) x_div: usize,
    pub(in crate::kernels::cuda) x_row_stride: usize,
    pub(in crate::kernels::cuda) y_row_stride: usize,
    pub(in crate::kernels::cuda) expert_stride: usize,
    /// The column tile of the launch: 0 chooses from the expected columns per expert, else 64 or
    /// 128.
    pub(in crate::kernels::cuda) col_tile: usize,
}

/// The GGUF type id the grouped entry switches on: the three K-quant and Q4_0 decoders and
/// every type of the window decoders.
fn grouped_type_id(ty: QuantType) -> Option<c_int> {
    match ty {
        QuantType::Q40 => Some(2),
        QuantType::Q3K => Some(11),
        QuantType::Q5K => Some(13),
        other => gguf_type_id(other),
    }
}

/// The types the grouped dequantizing GEMM serves.
pub(in crate::kernels::cuda) fn gemm_dq_mmid_type(ty: QuantType) -> bool {
    matches!(
        ty,
        QuantType::Q40 | QuantType::Q3K | QuantType::Q5K | QuantType::Iq4Xs
    ) || ty.is_codebook()
}

pub(in crate::kernels::cuda) fn gemm_dq_mmid_launch(
    stream: &RawStream,
    call: &GemmDqMmidRaw<'_>,
) -> Result<(), DeviceFailure> {
    let ty = call.ty;
    let id = grouped_type_id(ty).ok_or_else(invalid_argument)?;
    if !gemm_dq_mmid_type(ty) {
        return Err(invalid_argument());
    }
    let (qk, block_bytes) = (ty.block_size(), ty.block_bytes());
    let w_align = if ty == QuantType::Q5K { 16 } else { 2 };
    let row_bytes = (call.k / qk).checked_mul(block_bytes);
    let need_w = row_bytes
        .and_then(|row| row.checked_mul(call.rows))
        .and_then(|expert| {
            call.n_experts
                .checked_sub(1)?
                .checked_mul(call.expert_stride)?
                .checked_add(expert)
        });
    let x_rows = call.total_cols.div_ceil(call.x_div.max(1));
    let need_x = x_rows
        .checked_sub(1)
        .and_then(|head| head.checked_mul(call.x_row_stride))
        .and_then(|head| head.checked_add(call.k))
        .and_then(|n| n.checked_mul(4));
    let need_y = call
        .total_cols
        .checked_sub(1)
        .and_then(|head| head.checked_mul(call.y_row_stride))
        .and_then(|head| head.checked_add(call.rows))
        .and_then(|n| n.checked_mul(4));
    let need_ids = call.total_cols.checked_mul(4);
    let need_bounds = call.n_experts.checked_add(1).and_then(|n| n.checked_mul(4));
    let (Some(need_w), Some(need_x), Some(need_y), Some(need_ids), Some(need_bounds)) =
        (need_w, need_x, need_y, need_ids, need_bounds)
    else {
        return Err(invalid_argument());
    };
    let ints_fit = [
        call.k,
        call.rows,
        call.n_experts,
        call.total_cols,
        call.x_div,
        call.x_row_stride,
        call.y_row_stride,
    ]
    .iter()
    .all(|&v| c_int::try_from(v).is_ok())
        && call
            .total_cols
            .checked_mul(call.x_row_stride.max(call.y_row_stride))
            .is_some_and(|n| n <= i32::MAX as usize);
    let y_span = Span::of(&call.y, need_y);
    let apart = [
        Span::of(&call.w, need_w),
        Span::of(&call.x, need_x),
        Span::of(&call.ids_dst, need_ids),
        Span::of(&call.expert_bounds, need_bounds),
    ]
    .iter()
    .all(|other| y_span.apart(other));
    if !ints_fit
        || !apart
        || !call.w.offset().is_multiple_of(w_align)
        || !call.expert_stride.is_multiple_of(w_align)
        || !call.x.offset().is_multiple_of(16)
        || !call.x_row_stride.is_multiple_of(4)
        || !matches!(call.col_tile, 0 | 64 | 128)
        || !call.y.offset().is_multiple_of(4)
        || !call.ids_dst.offset().is_multiple_of(4)
        || !call.expert_bounds.offset().is_multiple_of(4)
        || call.w.bytes() < need_w
        || call.x.bytes() < need_x
        || call.y.bytes() < need_y
        || call.ids_dst.bytes() < need_ids
        || call.expert_bounds.bytes() < need_bounds
        || call.expert_stride < row_bytes.unwrap_or(0).saturating_mul(call.rows)
        || call.k == 0
        || call.rows == 0
        || call.n_experts == 0
        || call.total_cols == 0
        || call.x_div == 0
        || call.x_row_stride < call.k
        || call.y_row_stride < call.rows
        || !call.k.is_multiple_of(qk)
        || !call.k.is_multiple_of(64)
    {
        return Err(invalid_argument());
    }
    let table = table_of(ty);
    let mut stage = STAGE_NONE;
    let int = |v: usize| c_int::try_from(v).map_err(|_| invalid_argument());
    let args = GemmDqMmidArgsRepr {
        w: call.w.address().cast_const(),
        x: call.x.address().cast_const().cast::<f32>(),
        y: call.y.address().cast::<f32>(),
        ids_dst: call.ids_dst.address().cast_const().cast::<c_int>(),
        expert_bounds: call.expert_bounds.address().cast_const().cast::<c_int>(),
        expert_stride: call.expert_stride as u64,
        k: int(call.k)?,
        rows: int(call.rows)?,
        n_experts: int(call.n_experts)?,
        total_cols: int(call.total_cols)?,
        x_div: int(call.x_div)?,
        x_row_stride: int(call.x_row_stride)?,
        y_row_stride: int(call.y_row_stride)?,
        col_tile: int(call.col_tile)?,
    };
    // SAFETY: the checks above proved every range inside its live allocation, large enough for
    // the call, aligned, and `y`, the only range the kernel writes, disjoint from the others.
    // The id values are the kernel's to trust: they come from an `MmidIds`, whose constructors
    // prove them. `table` is a `'static` slice the adapter copies before it returns, `stream`
    // is live for the call.
    let status = unsafe {
        gemm_dq_mmid_launch_raw(
            &args,
            id,
            table.as_ptr().cast::<c_void>(),
            table.len(),
            stream.as_ptr(),
            &mut stage,
        )
    };
    checked(status, stage)
}
