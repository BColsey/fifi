//! Raw FFI to the entries of `native/resident/matvec_gguf.cu` (declared in
//! `native/resident/matvec_gguf.h`): the dense and the expert decode matvec of the codebook and
//! 2-bit GGUF types, and the device dequant the tests compare against the CPU oracle. Child of
//! `native::resident`, so it inherits the `allow(unsafe_code)` of `native.rs`.
//!
//! Safety invariants this module relies on (mirrors the other native matvec adapters):
//! - A device range is proved in bounds by `RawRange::new` before it gets here. This module
//!   proves, again and on the sizes of the call, that `w` holds `rows` rows of whole native
//!   blocks, that `x_q8` holds the padded blocks of the activation rows the call reads, that
//!   `y` holds the floats the kernel writes, that every range is 4-byte aligned, and that `y`,
//!   which the kernel writes, shares no byte with a range the kernel reads. The C entry repeats
//!   the checks on the addresses.
//! - The launch entries only queue work on the stream they are given; a fault in the kernel
//!   surfaces at the next synchronize.
//! - The allocations stay valid until the queued launch finishes because `inference_cuda_free`
//!   synchronizes the whole device before `cudaFree`.
//! - The lookup table of a type is read from a `'static` table of `weights/gguf_tables.rs` and
//!   copied by the C entry into the kernel arguments before the entry returns, so the host
//!   pointer is not kept.
//! - No raw pointer and no unsafe function leaves this module.

use std::ffi::{c_int, c_void};
use std::mem::offset_of;

use super::{DeviceFailure, RawRange, RawStream, STAGE_NONE, Span, checked, invalid_argument};
use crate::kernels::cuda::resident::matvec_q::QuantType;
use crate::weights::{
    IQ1S_GRID, IQ2S_GRID, IQ2XS_GRID, IQ2XXS_GRID, IQ3S_GRID, IQ3XXS_GRID, KVALUES_IQ4NL,
};

const _: () = assert!(
    cfg!(target_endian = "little"),
    "the tables are passed as their in-memory little-endian bytes"
);

/// The most expert slots one launch takes.
pub(in crate::kernels::cuda) const MAX_USED: usize = 32;

/// Bytes of one q8_1 block of 32 activation values.
const Q8_1_BLOCK: usize = 36;

/// Mirror of `inference_cuda_matvec_gguf_args` (`matvec_gguf.h`), field for field.
#[repr(C)]
struct DenseArgsRepr {
    ty: c_int,
    k: c_int,
    rows: c_int,
    ncols: c_int,
    w: *const c_void,
    x_q8: *const c_void,
    y: *mut f32,
}

const _: () = assert!(
    size_of::<DenseArgsRepr>() == 40
        && offset_of!(DenseArgsRepr, ty) == 0
        && offset_of!(DenseArgsRepr, k) == 4
        && offset_of!(DenseArgsRepr, rows) == 8
        && offset_of!(DenseArgsRepr, ncols) == 12
        && offset_of!(DenseArgsRepr, w) == 16
        && offset_of!(DenseArgsRepr, x_q8) == 24
        && offset_of!(DenseArgsRepr, y) == 32
);

/// Mirror of `inference_cuda_matvec_gguf_expert_args` (`matvec_gguf.h`), field for field.
#[repr(C)]
struct ExpertArgsRepr {
    ty: c_int,
    k: c_int,
    rows: c_int,
    n_expert: c_int,
    n_used: c_int,
    x_per_slot: c_int,
    y_slot_stride: c_int,
    reserved: c_int,
    w_stack: *const c_void,
    ids: *const c_void,
    x_q8: *const c_void,
    y: *mut f32,
    glu_gate: *const c_void,
}

const _: () = assert!(
    size_of::<ExpertArgsRepr>() == 72
        && offset_of!(ExpertArgsRepr, ty) == 0
        && offset_of!(ExpertArgsRepr, k) == 4
        && offset_of!(ExpertArgsRepr, rows) == 8
        && offset_of!(ExpertArgsRepr, n_expert) == 12
        && offset_of!(ExpertArgsRepr, n_used) == 16
        && offset_of!(ExpertArgsRepr, x_per_slot) == 20
        && offset_of!(ExpertArgsRepr, y_slot_stride) == 24
        && offset_of!(ExpertArgsRepr, reserved) == 28
        && offset_of!(ExpertArgsRepr, w_stack) == 32
        && offset_of!(ExpertArgsRepr, ids) == 40
        && offset_of!(ExpertArgsRepr, x_q8) == 48
        && offset_of!(ExpertArgsRepr, y) == 56
        && offset_of!(ExpertArgsRepr, glu_gate) == 64
);

unsafe extern "C" {
    #[link_name = "inference_cuda_matvec_gguf_launch"]
    fn matvec_gguf_launch_raw(
        args: *const DenseArgsRepr,
        table: *const c_void,
        table_bytes: usize,
        x_row_bytes: usize,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;

    #[link_name = "inference_cuda_matvec_gguf_expert_launch"]
    fn matvec_gguf_expert_launch_raw(
        args: *const ExpertArgsRepr,
        table: *const c_void,
        table_bytes: usize,
        x_row_bytes: usize,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;

    #[cfg(test)]
    #[link_name = "inference_cuda_dequant_gguf_launch"]
    fn dequant_gguf_launch_raw(
        ty: c_int,
        w: *const c_void,
        y: *mut f32,
        n_blocks: usize,
        table: *const c_void,
        table_bytes: usize,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
}

/// The bytes of a `'static` table in memory order (little endian, checked above).
fn bytes_of<T: Copy, const N: usize>(table: &'static [T; N]) -> &'static [u8] {
    // SAFETY: the table is `'static`, its elements are plain integers with no padding, and
    // `size_of_val` is exactly the byte length of the array.
    unsafe { std::slice::from_raw_parts(table.as_ptr().cast::<u8>(), size_of_val(table)) }
}

/// The GGUF tensor type id the kernel switches on, `None` for a type this kernel does not serve.
pub(in crate::kernels::cuda) fn gguf_type_id(ty: QuantType) -> Option<c_int> {
    Some(match ty {
        QuantType::Iq2Xxs => 16,
        QuantType::Iq2Xs => 17,
        QuantType::Iq3Xxs => 18,
        QuantType::Iq4Nl => 20,
        QuantType::Iq3S => 21,
        QuantType::Iq2S => 22,
        QuantType::Iq4Xs => 23,
        QuantType::Q20 => 42,
        QuantType::Q2K => 10,
        QuantType::Iq1S => 19,
        QuantType::Iq1M => 29,
        _ => return None,
    })
}

/// The lookup table of a type, as the bytes the kernel copies into its arguments. Q2_0 has
/// none (the empty slice).
pub(in crate::kernels::cuda) fn table_of(ty: QuantType) -> &'static [u8] {
    match ty {
        QuantType::Iq2Xxs => bytes_of(&IQ2XXS_GRID),
        QuantType::Iq2Xs => bytes_of(&IQ2XS_GRID),
        QuantType::Iq2S => bytes_of(&IQ2S_GRID),
        QuantType::Iq3Xxs => bytes_of(&IQ3XXS_GRID),
        QuantType::Iq3S => bytes_of(&IQ3S_GRID),
        QuantType::Iq1S | QuantType::Iq1M => bytes_of(&IQ1S_GRID),
        QuantType::Iq4Nl | QuantType::Iq4Xs => bytes_of(&KVALUES_IQ4NL),
        _ => &[],
    }
}

/// The in-house dense matvec call: `w` native blocks, `x_q8` the padded `quantize_q8_1` rows,
/// `x_row_bytes` one activation row's stride (`q8_1_bytes(k, 1)`), `y` f32 out with column `j`
/// at `y + j * rows`. `rows` may be any positive count: the kernel owns whole rows.
pub(in crate::kernels::cuda) struct MatvecGgufRaw<'a> {
    pub(in crate::kernels::cuda) ty: QuantType,
    pub(in crate::kernels::cuda) w: RawRange<'a>,
    pub(in crate::kernels::cuda) x_q8: RawRange<'a>,
    pub(in crate::kernels::cuda) y: RawRange<'a>,
    pub(in crate::kernels::cuda) k: usize,
    pub(in crate::kernels::cuda) rows: usize,
    pub(in crate::kernels::cuda) ncols: usize,
}

fn aligned4(range: &RawRange<'_>) -> bool {
    range.offset.is_multiple_of(4)
}

pub(in crate::kernels::cuda) fn check_matvec_gguf(
    call: &MatvecGgufRaw<'_>,
    x_row_bytes: usize,
) -> Result<(), DeviceFailure> {
    let (k, rows, ncols) = (call.k, call.rows, call.ncols);
    gguf_type_id(call.ty).ok_or_else(invalid_argument)?;
    let need_w = call.ty.weight_bytes(rows, k).ok_or_else(invalid_argument)?;
    let ints = [
        c_int::try_from(k).ok(),
        c_int::try_from(rows).ok(),
        c_int::try_from(ncols).ok(),
    ];
    let y_bytes = rows.saturating_mul(ncols).saturating_mul(4);
    let y = Span::of(&call.y, y_bytes);
    let apart = y.apart(&Span::of(&call.w, need_w))
        && y.apart(&Span::of(&call.x_q8, x_row_bytes.saturating_mul(ncols)));
    if ints.iter().any(|int| int.is_none())
        || !aligned4(&call.w)
        || !aligned4(&call.x_q8)
        || !aligned4(&call.y)
        || !apart
        || x_row_bytes == 0
        || !x_row_bytes.is_multiple_of(4)
        || x_row_bytes < (k / 32) * Q8_1_BLOCK
        || call.x_q8.bytes < x_row_bytes.saturating_mul(ncols)
        || call.y.bytes < y_bytes
        || call.w.bytes < need_w
        || k == 0
        || rows == 0
        || !(1..=8).contains(&ncols)
    {
        return Err(invalid_argument());
    }
    Ok(())
}

pub(in crate::kernels::cuda) fn matvec_gguf_launch(
    stream: &RawStream,
    call: &MatvecGgufRaw<'_>,
    x_row_bytes: usize,
) -> Result<(), DeviceFailure> {
    check_matvec_gguf(call, x_row_bytes)?;
    let ty = gguf_type_id(call.ty).ok_or_else(invalid_argument)?;
    let table = table_of(call.ty);
    let mut stage = STAGE_NONE;
    let args = DenseArgsRepr {
        ty,
        k: c_int::try_from(call.k).map_err(|_| invalid_argument())?,
        rows: c_int::try_from(call.rows).map_err(|_| invalid_argument())?,
        ncols: c_int::try_from(call.ncols).map_err(|_| invalid_argument())?,
        w: call.w.address().cast_const(),
        x_q8: call.x_q8.address().cast_const(),
        y: call.y.address().cast::<f32>(),
    };
    // SAFETY: `check_matvec_gguf` proved the `w`, `x_q8` and `y` ranges inside their live
    // allocations, large enough for the call, aligned, and `y` disjoint from the ranges the
    // kernel reads; the adapter repeats the checks on the addresses. `table` is a `'static`
    // slice the adapter copies before returning. The allocations stay valid until the queued
    // launch finishes because `inference_cuda_free` synchronizes the device first. `stream` is
    // live for the call.
    let status = unsafe {
        matvec_gguf_launch_raw(
            &args,
            table.as_ptr().cast::<c_void>(),
            table.len(),
            x_row_bytes,
            stream.0.as_ptr(),
            &mut stage,
        )
    };
    checked(status, stage)
}

/// The in-house expert matvec call: slot `s` multiplies the matrix of expert `ids[s]` (read on
/// the device when the kernel runs) with activation row `s` (or row 0). Fields as in
/// `matvec_expert`'s call, with `n_used` up to [`MAX_USED`].
pub(in crate::kernels::cuda) struct ExpertGgufRaw<'a> {
    pub(in crate::kernels::cuda) ty: QuantType,
    pub(in crate::kernels::cuda) w_stack: RawRange<'a>,
    pub(in crate::kernels::cuda) ids: RawRange<'a>,
    pub(in crate::kernels::cuda) x_q8: RawRange<'a>,
    pub(in crate::kernels::cuda) x_per_slot: bool,
    pub(in crate::kernels::cuda) y: RawRange<'a>,
    pub(in crate::kernels::cuda) y_slot_stride: usize,
    pub(in crate::kernels::cuda) k: usize,
    pub(in crate::kernels::cuda) rows: usize,
    pub(in crate::kernels::cuda) n_expert: usize,
    pub(in crate::kernels::cuda) n_used: usize,
    pub(in crate::kernels::cuda) glu_gate: Option<RawRange<'a>>,
}

struct ExpertSizes {
    w_stack: usize,
    ids: usize,
    x_q8: usize,
    y: usize,
}

fn expert_sizes(call: &ExpertGgufRaw<'_>, x_row_bytes: usize) -> Option<ExpertSizes> {
    let x_rows = if call.x_per_slot { call.n_used } else { 1 };
    let y_floats = call
        .y_slot_stride
        .checked_mul(call.n_used.checked_sub(1)?)?
        .checked_add(call.rows)?;
    Some(ExpertSizes {
        w_stack: call
            .ty
            .weight_bytes(call.rows.checked_mul(call.n_expert)?, call.k)?,
        ids: call.n_used.checked_mul(4)?,
        x_q8: x_row_bytes.checked_mul(x_rows)?,
        y: y_floats.checked_mul(4)?,
    })
}

pub(in crate::kernels::cuda) fn check_matvec_gguf_expert(
    call: &ExpertGgufRaw<'_>,
    x_row_bytes: usize,
) -> Result<(), DeviceFailure> {
    gguf_type_id(call.ty).ok_or_else(invalid_argument)?;
    let need = expert_sizes(call, x_row_bytes).ok_or_else(invalid_argument)?;
    let ints = [
        c_int::try_from(call.k).ok(),
        c_int::try_from(call.rows).ok(),
        c_int::try_from(call.n_expert).ok(),
        c_int::try_from(call.n_used).ok(),
        c_int::try_from(call.y_slot_stride).ok(),
    ];
    let y = Span::of(&call.y, need.y);
    let gate_ok = match &call.glu_gate {
        Some(gate) => {
            !call.x_per_slot
                && gate.bytes >= need.w_stack
                && aligned4(gate)
                && y.apart(&Span::of(gate, need.w_stack))
        }
        None => true,
    };
    let apart = y.apart(&Span::of(&call.w_stack, need.w_stack))
        && y.apart(&Span::of(&call.x_q8, need.x_q8))
        && y.apart(&Span::of(&call.ids, need.ids));
    if ints.iter().any(|int| int.is_none())
        || call.w_stack.bytes < need.w_stack
        || call.ids.bytes < need.ids
        || call.x_q8.bytes < need.x_q8
        || call.y.bytes < need.y
        || !aligned4(&call.w_stack)
        || !aligned4(&call.ids)
        || !aligned4(&call.x_q8)
        || !aligned4(&call.y)
        || !apart
        || !gate_ok
        || x_row_bytes == 0
        || !x_row_bytes.is_multiple_of(4)
        || x_row_bytes < (call.k / 32) * Q8_1_BLOCK
        || call.k == 0
        || call.rows < 1
        || call.n_expert < 1
        || !(1..=MAX_USED).contains(&call.n_used)
        || call.n_used > call.n_expert
        || call.y_slot_stride < call.rows
        || !call.k.is_multiple_of(call.ty.block_size())
    {
        return Err(invalid_argument());
    }
    Ok(())
}

pub(in crate::kernels::cuda) fn matvec_gguf_expert_launch(
    stream: &RawStream,
    call: &ExpertGgufRaw<'_>,
    x_row_bytes: usize,
) -> Result<(), DeviceFailure> {
    check_matvec_gguf_expert(call, x_row_bytes)?;
    let ty = gguf_type_id(call.ty).ok_or_else(invalid_argument)?;
    let table = table_of(call.ty);
    let args = ExpertArgsRepr {
        ty,
        k: c_int::try_from(call.k).map_err(|_| invalid_argument())?,
        rows: c_int::try_from(call.rows).map_err(|_| invalid_argument())?,
        n_expert: c_int::try_from(call.n_expert).map_err(|_| invalid_argument())?,
        n_used: c_int::try_from(call.n_used).map_err(|_| invalid_argument())?,
        x_per_slot: c_int::from(call.x_per_slot),
        y_slot_stride: c_int::try_from(call.y_slot_stride).map_err(|_| invalid_argument())?,
        reserved: 0,
        w_stack: call.w_stack.address().cast_const(),
        ids: call.ids.address().cast_const(),
        x_q8: call.x_q8.address().cast_const(),
        y: call.y.address().cast::<f32>(),
        glu_gate: match &call.glu_gate {
            Some(gate) => gate.address().cast_const(),
            None => std::ptr::null(),
        },
    };
    let mut stage = STAGE_NONE;
    // SAFETY: `check_matvec_gguf_expert` proved every range inside its live allocation, large
    // enough for the call, aligned, and `y` disjoint from the ranges the kernel reads (the
    // adapter repeats the checks on the addresses); the ids are read on the device and an id
    // outside the stack is answered with NaN rows and no weight read. `table` is a `'static`
    // slice copied by the adapter before it returns.
    let status = unsafe {
        matvec_gguf_expert_launch_raw(
            &args,
            table.as_ptr().cast::<c_void>(),
            table.len(),
            x_row_bytes,
            stream.0.as_ptr(),
            &mut stage,
        )
    };
    checked(status, stage)
}

/// Test support: decodes `n_blocks` blocks of `ty` from `w` into f32 `y` with the window
/// decoders of the matvec. The caller sizes the ranges (`n_blocks` blocks in, `n_blocks *
/// block_size` floats out).
#[cfg(test)]
pub(in crate::kernels::cuda) fn dequant_gguf_launch(
    stream: &RawStream,
    ty: QuantType,
    w: &RawRange<'_>,
    y: &RawRange<'_>,
    n_blocks: usize,
) -> Result<(), DeviceFailure> {
    let id = gguf_type_id(ty).ok_or_else(invalid_argument)?;
    let need_w = ty.weight_bytes(1, ty.block_size() * n_blocks);
    let need_y = n_blocks
        .checked_mul(ty.block_size())
        .and_then(|n| n.checked_mul(4));
    let (Some(need_w), Some(need_y)) = (need_w, need_y) else {
        return Err(invalid_argument());
    };
    if w.bytes < need_w
        || y.bytes < need_y
        || !aligned4(w)
        || !aligned4(y)
        || n_blocks == 0
        || !Span::of(y, need_y).apart(&Span::of(w, need_w))
    {
        return Err(invalid_argument());
    }
    let table = table_of(ty);
    let mut stage = STAGE_NONE;
    // SAFETY: the ranges were proved inside their allocations, aligned and apart above; the
    // table is a `'static` slice copied by the adapter; the stream is live for the call.
    let status = unsafe {
        dequant_gguf_launch_raw(
            id,
            w.address().cast_const(),
            y.address().cast::<f32>(),
            n_blocks,
            table.as_ptr().cast::<c_void>(),
            table.len(),
            stream.0.as_ptr(),
            &mut stage,
        )
    };
    checked(status, stage)
}
