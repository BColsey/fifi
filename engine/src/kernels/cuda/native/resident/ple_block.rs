//! Raw FFI to the three launch entries of native/resident/ple_block.cu (declared in
//! native/resident/ple_block.h). Child of native::resident, so it inherits the
//! allow(unsafe_code) of native.rs.
//!
//! Safety invariants this module relies on:
//! - A device range is proved in bounds by RawRange::new before it gets here. This module
//!   proves again, on the sizes of the call, that every range holds the bytes the kernel
//!   touches and is aligned; the safe wrapper additionally proves that written ranges share
//!   no byte with any other range of the call, and the C entry repeats the null and
//!   alignment checks on the addresses.
//! - A launch entry only queues work on the stream it is given and returns the launch status;
//!   a fault in the kernel surfaces at the next synchronize.
//! - The allocations stay valid until the queued launch finishes because
//!   inference_cuda_free synchronizes the whole device before cudaFree.
//! - Every call runs under the process dispatch guard and on the thread that owns the context.
//! - No raw pointer and no unsafe function leaves this module.

use std::ffi::{c_int, c_void};
use std::mem::offset_of;

use super::{DeviceFailure, RawRange, RawStream, STAGE_NONE, checked, invalid_argument};

unsafe extern "C" {
    #[link_name = "inference_cuda_ple_rows_launch"]
    fn ple_rows_launch_raw(
        args: *const PleRowsArgsRepr,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
    #[link_name = "inference_cuda_ple_gate_launch"]
    fn ple_gate_launch_raw(
        args: *const PleGateArgsRepr,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
    #[link_name = "inference_cuda_ple_conv_launch"]
    fn ple_conv_launch_raw(
        args: *const PleConvArgsRepr,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
}

/// Mirror of inference_cuda_ple_rows_args (ple_block.h), field for field.
#[repr(C)]
struct PleRowsArgsRepr {
    out: *mut c_void,
    raw: *const c_void,
    scale: f32,
    rows: c_int,
    head_dim: c_int,
}

/// Mirror of inference_cuda_ple_gate_args.
#[repr(C)]
struct PleGateArgsRepr {
    keys: *const c_void,
    hyper: *const c_void,
    value: *const c_void,
    norm_key: *const c_void,
    norm_query: *const c_void,
    norm_conv: *const c_void,
    gated: *mut c_void,
    normed: *mut c_void,
    eps: f32,
    branches: c_int,
    hidden: c_int,
}

/// Mirror of inference_cuda_ple_conv_args.
#[repr(C)]
struct PleConvArgsRepr {
    gated: *const c_void,
    normed: *const c_void,
    conv: *const c_void,
    ring: *mut c_void,
    out: *mut c_void,
    width: c_int,
    conv_kernel: c_int,
    ngram: c_int,
}

const _: () = assert!(
    size_of::<PleRowsArgsRepr>() == 32
        && offset_of!(PleRowsArgsRepr, scale) == 16
        && offset_of!(PleRowsArgsRepr, rows) == 20
        && offset_of!(PleRowsArgsRepr, head_dim) == 24
        && size_of::<PleGateArgsRepr>() == 80
        && offset_of!(PleGateArgsRepr, eps) == 64
        && offset_of!(PleGateArgsRepr, branches) == 68
        && offset_of!(PleGateArgsRepr, hidden) == 72
);
const _: () = assert!(
    size_of::<PleConvArgsRepr>() == 56
        && offset_of!(PleConvArgsRepr, width) == 40
        && offset_of!(PleConvArgsRepr, conv_kernel) == 44
        && offset_of!(PleConvArgsRepr, ngram) == 48
);

pub(in crate::kernels::cuda) struct PleRowsRaw<'a> {
    pub(in crate::kernels::cuda) out: RawRange<'a>,
    pub(in crate::kernels::cuda) raw: RawRange<'a>,
    pub(in crate::kernels::cuda) scale: f32,
    pub(in crate::kernels::cuda) rows: usize,
    pub(in crate::kernels::cuda) head_dim: usize,
}

pub(in crate::kernels::cuda) struct PleGateRaw<'a> {
    pub(in crate::kernels::cuda) keys: RawRange<'a>,
    pub(in crate::kernels::cuda) hyper: RawRange<'a>,
    pub(in crate::kernels::cuda) value: RawRange<'a>,
    pub(in crate::kernels::cuda) norm_key: RawRange<'a>,
    pub(in crate::kernels::cuda) norm_query: RawRange<'a>,
    pub(in crate::kernels::cuda) norm_conv: RawRange<'a>,
    pub(in crate::kernels::cuda) gated: RawRange<'a>,
    pub(in crate::kernels::cuda) normed: RawRange<'a>,
    pub(in crate::kernels::cuda) eps: f32,
    pub(in crate::kernels::cuda) branches: usize,
    pub(in crate::kernels::cuda) hidden: usize,
}

pub(in crate::kernels::cuda) struct PleConvRaw<'a> {
    pub(in crate::kernels::cuda) gated: RawRange<'a>,
    pub(in crate::kernels::cuda) normed: RawRange<'a>,
    pub(in crate::kernels::cuda) conv: RawRange<'a>,
    pub(in crate::kernels::cuda) ring: RawRange<'a>,
    pub(in crate::kernels::cuda) out: RawRange<'a>,
    pub(in crate::kernels::cuda) width: usize,
    pub(in crate::kernels::cuda) conv_kernel: usize,
    pub(in crate::kernels::cuda) ngram: usize,
}

fn holds(range: &RawRange<'_>, bytes: usize, align: usize) -> bool {
    range.bytes >= bytes && range.offset.is_multiple_of(align)
}

fn int(value: usize) -> Result<c_int, DeviceFailure> {
    c_int::try_from(value).map_err(|_| invalid_argument())
}

pub(in crate::kernels::cuda) fn ple_rows_launch(
    stream: &RawStream,
    call: &PleRowsRaw<'_>,
) -> Result<(), DeviceFailure> {
    let row_bytes = call.head_dim / 32 * 18;
    let sized = holds(&call.out, call.rows * call.head_dim * 4, 4)
        && holds(&call.raw, call.rows * row_bytes, 2)
        && call.head_dim.is_multiple_of(32)
        && (32..=8192).contains(&call.head_dim)
        && (1..=65_536).contains(&call.rows)
        && call.scale.is_finite();
    if !sized {
        return Err(invalid_argument());
    }
    let args = PleRowsArgsRepr {
        out: call.out.address(),
        raw: call.raw.address().cast_const(),
        scale: call.scale,
        rows: int(call.rows)?,
        head_dim: int(call.head_dim)?,
    };
    let mut stage = STAGE_NONE;
    // SAFETY: the ranges were proved inside their live allocations, large enough and aligned
    // above; the adapter repeats the null and alignment checks on the addresses. The
    // allocations stay valid until the queued launch finishes because inference_cuda_free
    // synchronizes the device first. stream is live for the call and args is a plain value.
    let status = unsafe { ple_rows_launch_raw(&args, stream.0.as_ptr(), &mut stage) };
    checked(status, stage)
}

pub(in crate::kernels::cuda) fn ple_gate_launch(
    stream: &RawStream,
    call: &PleGateRaw<'_>,
) -> Result<(), DeviceFailure> {
    let width = call
        .branches
        .checked_mul(call.hidden)
        .and_then(|v| v.checked_mul(4))
        .ok_or_else(invalid_argument)?;
    let sized = [
        &call.keys,
        &call.hyper,
        &call.norm_key,
        &call.norm_query,
        &call.norm_conv,
        &call.gated,
        &call.normed,
    ]
    .iter()
    .all(|range| holds(range, width, 4))
        && holds(&call.value, call.hidden * 4, 4)
        && (1..=64).contains(&call.branches)
        && (1..=(1 << 20)).contains(&call.hidden)
        && call.eps.is_finite()
        && call.eps >= 0.0;
    if !sized {
        return Err(invalid_argument());
    }
    let args = PleGateArgsRepr {
        keys: call.keys.address().cast_const(),
        hyper: call.hyper.address().cast_const(),
        value: call.value.address().cast_const(),
        norm_key: call.norm_key.address().cast_const(),
        norm_query: call.norm_query.address().cast_const(),
        norm_conv: call.norm_conv.address().cast_const(),
        gated: call.gated.address(),
        normed: call.normed.address(),
        eps: call.eps,
        branches: int(call.branches)?,
        hidden: int(call.hidden)?,
    };
    let mut stage = STAGE_NONE;
    // SAFETY: as for ple_rows_launch; every range was proved large enough and aligned above.
    let status = unsafe { ple_gate_launch_raw(&args, stream.0.as_ptr(), &mut stage) };
    checked(status, stage)
}

pub(in crate::kernels::cuda) fn ple_conv_launch(
    stream: &RawStream,
    call: &PleConvRaw<'_>,
) -> Result<(), DeviceFailure> {
    let taps = call
        .conv_kernel
        .checked_sub(1)
        .ok_or_else(invalid_argument)?;
    let ring = call
        .width
        .checked_mul(taps)
        .and_then(|v| v.checked_mul(call.ngram))
        .and_then(|v| v.checked_mul(4))
        .ok_or_else(invalid_argument)?;
    let conv = call
        .width
        .checked_mul(call.conv_kernel)
        .and_then(|v| v.checked_mul(4))
        .ok_or_else(invalid_argument)?;
    let sized = holds(&call.gated, call.width * 4, 4)
        && holds(&call.normed, call.width * 4, 4)
        && holds(&call.out, call.width * 4, 4)
        && holds(&call.conv, conv, 4)
        && holds(&call.ring, ring, 4)
        && (1..=(1 << 22)).contains(&call.width)
        && (2..=16).contains(&call.conv_kernel)
        && (1..=16).contains(&call.ngram);
    if !sized {
        return Err(invalid_argument());
    }
    let args = PleConvArgsRepr {
        gated: call.gated.address().cast_const(),
        normed: call.normed.address().cast_const(),
        conv: call.conv.address().cast_const(),
        ring: call.ring.address(),
        out: call.out.address(),
        width: int(call.width)?,
        conv_kernel: int(call.conv_kernel)?,
        ngram: int(call.ngram)?,
    };
    let mut stage = STAGE_NONE;
    // SAFETY: as for ple_rows_launch; every range was proved large enough and aligned above.
    let status = unsafe { ple_conv_launch_raw(&args, stream.0.as_ptr(), &mut stage) };
    checked(status, stage)
}
