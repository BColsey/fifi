//! Raw FFI to the entries of `native/resident/matvec_gguf.cu` (declared in
//! `native/resident/matvec_gguf.h`): the dense and the expert decode matvec of the codebook and
//! 2-bit GGUF types, their exact f32 twins (unquantized activation), and the device dequant the
//! tests compare against the CPU oracle. Child of
//! `native::resident`, so it inherits the `allow(unsafe_code)` of `native.rs`.
//!
//! Safety invariants this module relies on (mirrors the other native matvec adapters):
//! - A device range is proved in bounds by `RawRange::new` before it gets here. This module
//!   proves, again and on the sizes of the call, that `w` holds `rows` rows of whole native
//!   blocks, that the activation (`x_q8`: the padded q8_1 blocks, or the packed f32 rows of
//!   the exact entries) holds the rows the call reads, that `y` holds the floats the kernel
//!   writes, that every range is aligned as its kernel needs, and that `y`,
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

/// Mirror of `inference_cuda_matvec_gguf_args` and of the layout-identical
/// `inference_cuda_matvec_gguf_f32_args` (`matvec_gguf.h`), field for field; `x_q8` is the
/// activation: the q8_1 rows, or the packed f32 rows for the exact entry.
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

/// Mirror of `inference_cuda_matvec_gguf_expert_args` and of the layout-identical
/// `inference_cuda_matvec_gguf_f32_expert_args` (`matvec_gguf.h`), field for field.
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

/// Mirror of `inference_cuda_embed_gguf_args` (`matvec_gguf.h`), field for field.
#[repr(C)]
struct EmbedArgsRepr {
    blocks: *const c_void,
    ids: *const c_int,
    x: *mut f32,
    status: *mut c_int,
    ty: c_int,
    m: c_int,
    n_rows: c_int,
    k: c_int,
}

const _: () = assert!(
    size_of::<EmbedArgsRepr>() == 48
        && offset_of!(EmbedArgsRepr, ids) == 8
        && offset_of!(EmbedArgsRepr, x) == 16
        && offset_of!(EmbedArgsRepr, status) == 24
        && offset_of!(EmbedArgsRepr, ty) == 32
        && offset_of!(EmbedArgsRepr, k) == 44
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

    #[link_name = "inference_cuda_matvec_gguf_f32_launch"]
    fn matvec_gguf_f32_launch_raw(
        args: *const DenseArgsRepr,
        table: *const c_void,
        table_bytes: usize,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;

    #[link_name = "inference_cuda_matvec_gguf_f32_expert_launch"]
    fn matvec_gguf_f32_expert_raw(
        args: *const ExpertArgsRepr,
        table: *const c_void,
        table_bytes: usize,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;

    #[link_name = "inference_cuda_embed_gguf_launch"]
    fn embed_gguf_launch_raw(
        args: *const EmbedArgsRepr,
        table: *const c_void,
        table_bytes: usize,
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
        QuantType::Q40 => 2,
        QuantType::Q50 => 6,
        QuantType::Q80 => 8,
        QuantType::Q3K => 11,
        QuantType::Q4K => 12,
        QuantType::Q5K => 13,
        QuantType::Q6K => 14,
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

/// The activation a matvec entry multiplies: the padded `quantize_q8_1` rows (`x_row_bytes`
/// each, at least 36 bytes per 32 values) or the packed f32 rows of the exact entries (`4 * k`
/// bytes each).
#[derive(Clone, Copy)]
enum Activation {
    Q8 { x_row_bytes: usize },
    F32,
}

impl Activation {
    /// The bytes of `rows` activation rows; `None` for an invalid row size or an overflow.
    fn bytes(self, k: usize, rows: usize) -> Option<usize> {
        match self {
            Self::Q8 { x_row_bytes } => {
                if x_row_bytes == 0
                    || !x_row_bytes.is_multiple_of(4)
                    || x_row_bytes < (k / 32) * Q8_1_BLOCK
                {
                    return None;
                }
                x_row_bytes.checked_mul(rows)
            }
            Self::F32 => k.checked_mul(4)?.checked_mul(rows),
        }
    }

    /// Where the kernel needs the activation rows to start.
    fn alignment(self) -> usize {
        match self {
            Self::Q8 { .. } => 4,
            Self::F32 => 16,
        }
    }
}

/// The ranges and geometry of a dense call, whichever entry takes it: what the guard checks.
#[derive(Clone, Copy)]
struct DenseView<'r, 'a> {
    ty: QuantType,
    w: &'r RawRange<'a>,
    x: &'r RawRange<'a>,
    y: &'r RawRange<'a>,
    k: usize,
    rows: usize,
    ncols: usize,
}

/// The guard of both dense entries: the ranges hold the call's blocks, activation rows and
/// output floats, start aligned, and the output shares no byte with what the kernel reads.
fn check_dense(call: &DenseView<'_, '_>, act: Activation) -> Result<(), DeviceFailure> {
    let (k, rows, ncols) = (call.k, call.rows, call.ncols);
    gguf_type_id(call.ty).ok_or_else(invalid_argument)?;
    let need_w = call.ty.weight_bytes(rows, k).ok_or_else(invalid_argument)?;
    let need_x = act.bytes(k, ncols).ok_or_else(invalid_argument)?;
    let ints = [
        c_int::try_from(k).ok(),
        c_int::try_from(rows).ok(),
        c_int::try_from(ncols).ok(),
    ];
    let y_bytes = rows.saturating_mul(ncols).saturating_mul(4);
    let y = Span::of(call.y, y_bytes);
    let apart = y.apart(&Span::of(call.w, need_w)) && y.apart(&Span::of(call.x, need_x));
    if ints.iter().any(|int| int.is_none())
        || !aligned4(call.w)
        || !call.x.offset.is_multiple_of(act.alignment())
        || !aligned4(call.y)
        || !apart
        || call.x.bytes < need_x
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

impl<'a> MatvecGgufRaw<'a> {
    fn view(&self) -> DenseView<'_, 'a> {
        DenseView {
            ty: self.ty,
            w: &self.w,
            x: &self.x_q8,
            y: &self.y,
            k: self.k,
            rows: self.rows,
            ncols: self.ncols,
        }
    }
}

pub(in crate::kernels::cuda) fn check_matvec_gguf(
    call: &MatvecGgufRaw<'_>,
    x_row_bytes: usize,
) -> Result<(), DeviceFailure> {
    check_dense(&call.view(), Activation::Q8 { x_row_bytes })
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

/// The ranges and geometry of an expert call, whichever entry takes it.
#[derive(Clone, Copy)]
struct ExpertView<'r, 'a> {
    ty: QuantType,
    w_stack: &'r RawRange<'a>,
    ids: &'r RawRange<'a>,
    x: &'r RawRange<'a>,
    x_per_slot: bool,
    y: &'r RawRange<'a>,
    y_slot_stride: usize,
    k: usize,
    rows: usize,
    n_expert: usize,
    n_used: usize,
    glu_gate: Option<&'r RawRange<'a>>,
}

struct ExpertSizes {
    w_stack: usize,
    ids: usize,
    x: usize,
    y: usize,
}

fn expert_sizes(call: &ExpertView<'_, '_>, act: Activation) -> Option<ExpertSizes> {
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
        x: act.bytes(call.k, x_rows)?,
        y: y_floats.checked_mul(4)?,
    })
}

/// The guard of both expert entries: as the dense guard, for the stack, the ids and the slots
/// of the output; a gate stack, when given, has the stack's size and stays apart from the
/// output, and only the shared-activation view may take it.
fn check_expert(call: &ExpertView<'_, '_>, act: Activation) -> Result<(), DeviceFailure> {
    gguf_type_id(call.ty).ok_or_else(invalid_argument)?;
    let need = expert_sizes(call, act).ok_or_else(invalid_argument)?;
    let ints = [
        c_int::try_from(call.k).ok(),
        c_int::try_from(call.rows).ok(),
        c_int::try_from(call.n_expert).ok(),
        c_int::try_from(call.n_used).ok(),
        c_int::try_from(call.y_slot_stride).ok(),
    ];
    let y = Span::of(call.y, need.y);
    let gate_ok = match call.glu_gate {
        Some(gate) => {
            !call.x_per_slot
                && gate.bytes >= need.w_stack
                && aligned4(gate)
                && y.apart(&Span::of(gate, need.w_stack))
        }
        None => true,
    };
    let apart = y.apart(&Span::of(call.w_stack, need.w_stack))
        && y.apart(&Span::of(call.x, need.x))
        && y.apart(&Span::of(call.ids, need.ids));
    if ints.iter().any(|int| int.is_none())
        || call.w_stack.bytes < need.w_stack
        || call.ids.bytes < need.ids
        || call.x.bytes < need.x
        || call.y.bytes < need.y
        || !aligned4(call.w_stack)
        || !aligned4(call.ids)
        || !call.x.offset.is_multiple_of(act.alignment())
        || !aligned4(call.y)
        || !apart
        || !gate_ok
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

impl<'a> ExpertGgufRaw<'a> {
    fn view(&self) -> ExpertView<'_, 'a> {
        ExpertView {
            ty: self.ty,
            w_stack: &self.w_stack,
            ids: &self.ids,
            x: &self.x_q8,
            x_per_slot: self.x_per_slot,
            y: &self.y,
            y_slot_stride: self.y_slot_stride,
            k: self.k,
            rows: self.rows,
            n_expert: self.n_expert,
            n_used: self.n_used,
            glu_gate: self.glu_gate.as_ref(),
        }
    }
}

pub(in crate::kernels::cuda) fn check_matvec_gguf_expert(
    call: &ExpertGgufRaw<'_>,
    x_row_bytes: usize,
) -> Result<(), DeviceFailure> {
    check_expert(&call.view(), Activation::Q8 { x_row_bytes })
}

/// The exact f32 dense call: the fields of [`MatvecGgufRaw`] with `x` the packed f32 rows of
/// the activation (`4 * k` bytes per column).
pub(in crate::kernels::cuda) struct MatvecGgufF32Raw<'a> {
    pub(in crate::kernels::cuda) ty: QuantType,
    pub(in crate::kernels::cuda) w: RawRange<'a>,
    pub(in crate::kernels::cuda) x: RawRange<'a>,
    pub(in crate::kernels::cuda) y: RawRange<'a>,
    pub(in crate::kernels::cuda) k: usize,
    pub(in crate::kernels::cuda) rows: usize,
    pub(in crate::kernels::cuda) ncols: usize,
}

impl<'a> MatvecGgufF32Raw<'a> {
    fn view(&self) -> DenseView<'_, 'a> {
        DenseView {
            ty: self.ty,
            w: &self.w,
            x: &self.x,
            y: &self.y,
            k: self.k,
            rows: self.rows,
            ncols: self.ncols,
        }
    }
}

pub(in crate::kernels::cuda) fn matvec_gguf_f32_launch(
    stream: &RawStream,
    call: &MatvecGgufF32Raw<'_>,
) -> Result<(), DeviceFailure> {
    check_dense(&call.view(), Activation::F32)?;
    let ty = gguf_type_id(call.ty).ok_or_else(invalid_argument)?;
    let table = table_of(call.ty);
    let args = DenseArgsRepr {
        ty,
        k: c_int::try_from(call.k).map_err(|_| invalid_argument())?,
        rows: c_int::try_from(call.rows).map_err(|_| invalid_argument())?,
        ncols: c_int::try_from(call.ncols).map_err(|_| invalid_argument())?,
        w: call.w.address().cast_const(),
        x_q8: call.x.address().cast_const(),
        y: call.y.address().cast::<f32>(),
    };
    let mut stage = STAGE_NONE;
    // SAFETY: `check_dense` proved the `w`, `x` and `y` ranges inside their live allocations,
    // large enough for the call, aligned as the exact kernel needs, and `y` disjoint from the
    // ranges the kernel reads; the adapter repeats the checks on the addresses. `table` is a
    // `'static` slice the adapter copies before returning. The allocations stay valid until the
    // queued launch finishes because `inference_cuda_free` synchronizes the device first.
    // `stream` is live for the call.
    let status = unsafe {
        matvec_gguf_f32_launch_raw(
            &args,
            table.as_ptr().cast::<c_void>(),
            table.len(),
            stream.0.as_ptr(),
            &mut stage,
        )
    };
    checked(status, stage)
}

/// The exact f32 expert call: the fields of [`ExpertGgufRaw`] with `x` the packed f32 rows
/// (one row, or one per slot).
pub(in crate::kernels::cuda) struct ExpertGgufF32Raw<'a> {
    pub(in crate::kernels::cuda) ty: QuantType,
    pub(in crate::kernels::cuda) w_stack: RawRange<'a>,
    pub(in crate::kernels::cuda) ids: RawRange<'a>,
    pub(in crate::kernels::cuda) x: RawRange<'a>,
    pub(in crate::kernels::cuda) x_per_slot: bool,
    pub(in crate::kernels::cuda) y: RawRange<'a>,
    pub(in crate::kernels::cuda) y_slot_stride: usize,
    pub(in crate::kernels::cuda) k: usize,
    pub(in crate::kernels::cuda) rows: usize,
    pub(in crate::kernels::cuda) n_expert: usize,
    pub(in crate::kernels::cuda) n_used: usize,
    pub(in crate::kernels::cuda) glu_gate: Option<RawRange<'a>>,
}

impl<'a> ExpertGgufF32Raw<'a> {
    fn view(&self) -> ExpertView<'_, 'a> {
        ExpertView {
            ty: self.ty,
            w_stack: &self.w_stack,
            ids: &self.ids,
            x: &self.x,
            x_per_slot: self.x_per_slot,
            y: &self.y,
            y_slot_stride: self.y_slot_stride,
            k: self.k,
            rows: self.rows,
            n_expert: self.n_expert,
            n_used: self.n_used,
            glu_gate: self.glu_gate.as_ref(),
        }
    }
}

pub(in crate::kernels::cuda) fn matvec_gguf_f32_expert_launch(
    stream: &RawStream,
    call: &ExpertGgufF32Raw<'_>,
) -> Result<(), DeviceFailure> {
    check_expert(&call.view(), Activation::F32)?;
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
        x_q8: call.x.address().cast_const(),
        y: call.y.address().cast::<f32>(),
        glu_gate: match &call.glu_gate {
            Some(gate) => gate.address().cast_const(),
            None => std::ptr::null(),
        },
    };
    let mut stage = STAGE_NONE;
    // SAFETY: `check_expert` proved every range inside its live allocation, large enough for
    // the call, aligned as the exact kernel needs, and `y` disjoint from the ranges the kernel
    // reads (the adapter repeats the checks on the addresses); the ids are read on the device
    // and an id outside the stack is answered with NaN rows and no weight read. `table` is a
    // `'static` slice copied by the adapter before it returns.
    let status = unsafe {
        matvec_gguf_f32_expert_raw(
            &args,
            table.as_ptr().cast::<c_void>(),
            table.len(),
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

/// The embedding gather call: row `ids[t]` of `blocks` (the native table of `n_rows` rows of `k`
/// values of `ty`) into f32 row `t` of `x`, `status` the step status word.
pub(in crate::kernels::cuda) struct EmbedGgufRaw<'a> {
    pub(in crate::kernels::cuda) ty: QuantType,
    pub(in crate::kernels::cuda) blocks: RawRange<'a>,
    pub(in crate::kernels::cuda) ids: RawRange<'a>,
    pub(in crate::kernels::cuda) x: RawRange<'a>,
    pub(in crate::kernels::cuda) status: RawRange<'a>,
    pub(in crate::kernels::cuda) m: usize,
    pub(in crate::kernels::cuda) n_rows: usize,
    pub(in crate::kernels::cuda) k: usize,
}

pub(in crate::kernels::cuda) fn embed_gguf_launch(
    stream: &RawStream,
    call: &EmbedGgufRaw<'_>,
) -> Result<(), DeviceFailure> {
    let ty = gguf_type_id(call.ty).ok_or_else(invalid_argument)?;
    let need_blocks = call
        .ty
        .weight_bytes(call.n_rows, call.k)
        .ok_or_else(invalid_argument)?;
    let need_ids = call.m.checked_mul(4).ok_or_else(invalid_argument)?;
    let need_x = call
        .m
        .checked_mul(call.k)
        .and_then(|n| n.checked_mul(4))
        .ok_or_else(invalid_argument)?;
    let x_span = Span::of(&call.x, need_x);
    let status_span = Span::of(&call.status, 4);
    let apart = x_span.apart(&Span::of(&call.blocks, need_blocks))
        && x_span.apart(&Span::of(&call.ids, need_ids))
        && x_span.apart(&status_span)
        && status_span.apart(&Span::of(&call.ids, need_ids))
        && status_span.apart(&Span::of(&call.blocks, need_blocks));
    let ints_fit = [call.m, call.n_rows, call.k]
        .iter()
        .all(|&v| c_int::try_from(v).is_ok());
    if !ints_fit
        || !apart
        || !(1..=4096).contains(&call.m)
        || !(1..=(1usize << 28)).contains(&call.n_rows)
        || call.k < 32
        || call.k > 32768
        || !call.k.is_multiple_of(call.ty.block_size())
        || !call.blocks.offset.is_multiple_of(2)
        || !aligned4(&call.ids)
        || !call.x.offset.is_multiple_of(16)
        || !aligned4(&call.status)
        || call.blocks.bytes < need_blocks
        || call.ids.bytes < need_ids
        || call.x.bytes < need_x
        || call.status.bytes < 4
    {
        return Err(invalid_argument());
    }
    let table = table_of(call.ty);
    let mut stage = STAGE_NONE;
    let args = EmbedArgsRepr {
        blocks: call.blocks.address().cast_const(),
        ids: call.ids.address().cast_const().cast::<c_int>(),
        x: call.x.address().cast::<f32>(),
        status: call.status.address().cast::<c_int>(),
        ty,
        m: c_int::try_from(call.m).map_err(|_| invalid_argument())?,
        n_rows: c_int::try_from(call.n_rows).map_err(|_| invalid_argument())?,
        k: c_int::try_from(call.k).map_err(|_| invalid_argument())?,
    };
    // SAFETY: the checks above proved every range inside its live allocation, large enough for
    // the call and aligned, and `x`, the range the kernel writes besides the status word,
    // disjoint from every range it reads; the status word is written only by the kernel's
    // integer atomic. `table` is a `'static` slice the adapter copies before returning;
    // `stream` is live for the call.
    let status = unsafe {
        embed_gguf_launch_raw(
            &args,
            table.as_ptr().cast::<c_void>(),
            table.len(),
            stream.0.as_ptr(),
            &mut stage,
        )
    };
    checked(status, stage)
}
