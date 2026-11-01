//! Raw FFI to inference_cuda_row_add_launch of native/resident/row_add.cu
//! (declared in native/resident/row_add.h). Child of native::resident.
//!
//! Safety invariants this module relies on:
//! - RawRange::new proved each range in bounds. This module re-proves that out, x and y
//!   hold n floats each at 4-byte alignment, and that out, which the kernel writes,
//!   shares no byte with x or y. The C entry repeats the checks on the addresses.
//! - Launch-only: a fault surfaces at the next synchronize; allocations stay valid until
//!   then because inference_cuda_free synchronizes the device first.
//! - Every call runs under the process dispatch guard on the owning thread
//!   (decisions 0023 and 0024). No raw pointer leaves this module.

use std::ffi::{c_int, c_void};
use std::mem::offset_of;

use super::{DeviceFailure, RawRange, RawStream, STAGE_NONE, Span, checked, invalid_argument};

unsafe extern "C" {
    #[link_name = "inference_cuda_row_add_launch"]
    fn row_add_launch_raw(
        args: *const RowAddArgsRepr,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
}

/// Mirror of inference_cuda_row_add_args, field for field.
#[repr(C)]
struct RowAddArgsRepr {
    out: *mut c_void,
    x: *const c_void,
    y: *const c_void,
    n: c_int,
    flag: *mut u32,
    flag_bytes: usize,
    fused_publish: u32,
}

const _: () = assert!(
    size_of::<RowAddArgsRepr>() == 56
        && offset_of!(RowAddArgsRepr, out) == 0
        && offset_of!(RowAddArgsRepr, x) == 8
        && offset_of!(RowAddArgsRepr, y) == 16
        && offset_of!(RowAddArgsRepr, n) == 24
        && offset_of!(RowAddArgsRepr, flag) == 32
        && offset_of!(RowAddArgsRepr, fused_publish) == 48
);

/// One row_add launch: n values of out, x and y.
pub(in crate::kernels::cuda) struct RowAddRaw<'a> {
    pub(in crate::kernels::cuda) out: RawRange<'a>,
    pub(in crate::kernels::cuda) x: RawRange<'a>,
    pub(in crate::kernels::cuda) y: RawRange<'a>,
    pub(in crate::kernels::cuda) n: usize,
    pub(in crate::kernels::cuda) flag: Option<RawRange<'a>>,
    pub(in crate::kernels::cuda) fused: bool,
}

fn check_row_add(call: &RowAddRaw<'_>) -> Result<(), DeviceFailure> {
    let bytes = call.n.checked_mul(4).ok_or_else(invalid_argument)?;
    let sized = call.out.bytes >= bytes && call.x.bytes >= bytes && call.y.bytes >= bytes;
    let aligned = call.out.offset.is_multiple_of(4)
        && call.x.offset.is_multiple_of(4)
        && call.y.offset.is_multiple_of(4);
    let out = Span::of(&call.out, bytes);
    let apart = out.apart(&Span::of(&call.x, bytes)) && out.apart(&Span::of(&call.y, bytes));
    let dims = (32..=4_194_304).contains(&call.n) && call.n.is_multiple_of(32);
    if !(sized && aligned && apart && dims) {
        return Err(invalid_argument());
    }
    Ok(())
}

/// The flag takes atomicOr from every thread when the launch is fused: four aligned
/// bytes that share nothing with the tensors.
fn check_fused_flag(call: &RowAddRaw<'_>) -> Result<(), DeviceFailure> {
    if !call.fused {
        return Ok(());
    }
    let Some(flag) = call
        .flag
        .as_ref()
        .filter(|flag| flag.bytes() >= 4 && flag.offset.is_multiple_of(4))
    else {
        return Err(invalid_argument());
    };
    let flag = Span::of(flag, 4);
    let bytes = call.n * 4;
    let clear = [&call.out, &call.x, &call.y]
        .into_iter()
        .all(|range| flag.apart(&Span::of(range, bytes)));
    if clear {
        Ok(())
    } else {
        Err(invalid_argument())
    }
}

/// Queues the row_add of call on stream. Returns when the kernel is queued.
pub(in crate::kernels::cuda) fn row_add_launch(
    stream: &RawStream,
    call: &RowAddRaw<'_>,
) -> Result<(), DeviceFailure> {
    check_row_add(call)?;
    check_fused_flag(call)?;
    let args = RowAddArgsRepr {
        out: call.out.address(),
        x: call.x.address().cast_const(),
        y: call.y.address().cast_const(),
        n: c_int::try_from(call.n).map_err(|_| invalid_argument())?,
        flag: call
            .flag
            .as_ref()
            .map_or(std::ptr::null_mut(), |flag| flag.address().cast()),
        flag_bytes: call.flag.as_ref().map_or(0, RawRange::bytes),
        fused_publish: u32::from(call.fused),
    };
    let mut stage = STAGE_NONE;
    // SAFETY: check_row_add proved the ranges inside their live allocations, large enough,
    // aligned, and disjoint in the write direction; the adapter repeats the checks on the
    // addresses. The allocations stay valid until the queued launch finishes because
    // inference_cuda_free synchronizes the device first.
    let status = unsafe { row_add_launch_raw(&args, stream.0.as_ptr(), &mut stage) };
    checked(status, stage)
}

#[cfg(test)]
mod tests {
    use super::super::RawDevice;
    use super::*;
    use crate::kernels::cuda::native::STATUS_INVALID_ARGUMENT;
    use std::ptr::NonNull;

    /// A fused flag that shares a byte with out, x or y is refused by the Rust checks and
    /// by the C entry, with every other argument valid; a clear flag passes the Rust checks.
    #[test]
    fn a_fused_flag_that_shares_a_byte_with_a_tensor_is_refused_before_cuda() {
        let mut memory = vec![0u32; 4096];
        let device = RawDevice {
            ptr: NonNull::new(memory.as_mut_ptr().cast()).unwrap(),
            bytes: memory.len() * 4,
        };
        let range = |offset, bytes| RawRange::new(&device, offset, bytes).unwrap();
        let call = |flag_offset| RowAddRaw {
            out: range(0, 128),
            x: range(1024, 128),
            y: range(2048, 128),
            n: 32,
            flag: Some(range(flag_offset, 4)),
            fused: true,
        };
        let stream = RawStream(NonNull::dangling());
        let base = device.ptr.as_ptr().cast::<u8>();
        for (name, offset) in [("out start", 0), ("out end", 124), ("x", 1040), ("y", 2048)] {
            let aliased = call(offset);
            assert!(check_row_add(&aliased).is_ok(), "{name}: other arguments");
            assert_eq!(
                check_fused_flag(&aliased),
                Err(invalid_argument()),
                "{name}"
            );
            assert_eq!(
                row_add_launch(&stream, &aliased),
                Err(invalid_argument()),
                "{name}"
            );
            // SAFETY: only addresses inside the test allocation are formed, and the C entry
            // refuses before any CUDA call.
            let (status, stage) = unsafe {
                let args = RowAddArgsRepr {
                    out: base.cast(),
                    x: base.add(1024).cast(),
                    y: base.add(2048).cast(),
                    n: 32,
                    flag: base.add(offset).cast(),
                    flag_bytes: 4,
                    fused_publish: 1,
                };
                let mut stage = STAGE_NONE;
                let status = row_add_launch_raw(&args, stream.0.as_ptr(), &mut stage);
                (status, stage)
            };
            assert_eq!(
                (status, stage),
                (STATUS_INVALID_ARGUMENT, STAGE_NONE),
                "{name}"
            );
        }
        let clear = call(8192);
        assert!(check_row_add(&clear).is_ok());
        assert_eq!(check_fused_flag(&clear), Ok(()));
    }
}
