//! Raw FFI to the column scatter of `native/prefill/scatter_cols.cu` (declared in
//! `native/prefill/scatter_cols.h`). Child module of `native/prefill.rs`.
//!
//! Safety invariants this file relies on:
//! - The launch entry is called only through `CudaContext::launch_on`, on the thread that owns
//!   the context and under the dispatch guard. It only enqueues and returns.
//! - Both ranges are `RawRange`s proved inside a live allocation by `raw_range`. Their sizes,
//!   their 16-byte alignment and the disjointness of the written `dst` extent from the read
//!   `src` extent are proved here before the call. The C entry repeats the shape, alignment and
//!   disjointness checks on the addresses; the arguments carry no byte lengths, so the extents
//!   are proved only on this side.
//! - A device allocation outlives the queued work because `inference_cuda_free` synchronizes
//!   the device before `cudaFree`.

use std::ffi::{c_int, c_void};
use std::mem::{offset_of, size_of};

use super::{DeviceFailure, RawRange, RawStream, STAGE_NONE, Span, checked, invalid_argument};

/// Mirror of `inference_cuda_scatter_cols_args` in `scatter_cols.h`, field for field.
#[repr(C)]
struct ScatterColsArgsRaw {
    src: *const c_void,
    dst: *mut c_void,
    rows: c_int,
    n: c_int,
    ld: c_int,
    col_off: c_int,
}

const _: () = assert!(
    size_of::<ScatterColsArgsRaw>() == 32
        && offset_of!(ScatterColsArgsRaw, src) == 0
        && offset_of!(ScatterColsArgsRaw, dst) == 8
        && offset_of!(ScatterColsArgsRaw, rows) == 16
        && offset_of!(ScatterColsArgsRaw, n) == 20
        && offset_of!(ScatterColsArgsRaw, ld) == 24
        && offset_of!(ScatterColsArgsRaw, col_off) == 28
);

unsafe extern "C" {
    #[link_name = "inference_cuda_scatter_cols_launch"]
    fn scatter_cols_launch_raw(
        args: *const ScatterColsArgsRaw,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
}

/// Every integer argument of the launch, already bounded and converted by the wrapper.
#[derive(Debug, Clone, Copy)]
pub(in crate::kernels::cuda) struct ScatterColsInts {
    pub(in crate::kernels::cuda) rows: c_int,
    pub(in crate::kernels::cuda) n: c_int,
    pub(in crate::kernels::cuda) ld: c_int,
    pub(in crate::kernels::cuda) col_off: c_int,
}

/// What one launch reads (`src`) and writes (`dst`), and its integers.
pub(in crate::kernels::cuda) struct ScatterColsRaw<'a> {
    pub(in crate::kernels::cuda) src: RawRange<'a>,
    pub(in crate::kernels::cuda) dst: RawRange<'a>,
    pub(in crate::kernels::cuda) ints: ScatterColsInts,
}

/// The bytes of the two extents the kernel touches: `rows * n` floats of `src` and
/// `(rows - 1) * ld + col_off + n` floats of `dst`.
fn extents(ints: &ScatterColsInts) -> Result<(usize, usize), DeviceFailure> {
    let size = |value: c_int| usize::try_from(value).map_err(|_| invalid_argument());
    let (rows, n) = (size(ints.rows)?, size(ints.n)?);
    let (ld, col_off) = (size(ints.ld)?, size(ints.col_off)?);
    let before = rows.checked_sub(1).ok_or_else(invalid_argument)?;
    let src = rows
        .checked_mul(n)
        .and_then(|floats| floats.checked_mul(4))
        .ok_or_else(invalid_argument)?;
    let dst = before
        .checked_mul(ld)
        .and_then(|floats| floats.checked_add(col_off))
        .and_then(|floats| floats.checked_add(n))
        .and_then(|floats| floats.checked_mul(4))
        .ok_or_else(invalid_argument)?;
    Ok((src, dst))
}

fn check_scatter_cols(io: &ScatterColsRaw<'_>) -> Result<(), DeviceFailure> {
    let (src, dst) = extents(&io.ints)?;
    let sized = io.src.bytes() >= src && io.dst.bytes() >= dst;
    let aligned = io.src.offset().is_multiple_of(16) && io.dst.offset().is_multiple_of(16);
    let apart = Span::of(&io.dst, dst).apart(&Span::of(&io.src, src));
    if !(sized && aligned && apart) {
        return Err(invalid_argument());
    }
    Ok(())
}

/// Queues the scatter on `stream`. Returns when the kernel is queued.
pub(in crate::kernels::cuda) fn scatter_cols_launch(
    stream: &RawStream,
    io: ScatterColsRaw<'_>,
) -> Result<(), DeviceFailure> {
    check_scatter_cols(&io)?;
    let args = ScatterColsArgsRaw {
        src: io.src.address().cast_const(),
        dst: io.dst.address(),
        rows: io.ints.rows,
        n: io.ints.n,
        ld: io.ints.ld,
        col_off: io.ints.col_off,
    };
    let mut stage = STAGE_NONE;
    // SAFETY: `check_scatter_cols` proved both ranges inside their live allocations, large enough
    // for the extents the kernel touches, 16-byte aligned, and the written `dst` extent disjoint
    // from the read `src` extent; the C entry repeats the alignment, shape and disjointness checks
    // on the addresses (it has no lengths, so the extents are proved only here). The
    // allocations stay valid until the queued work finishes because `inference_cuda_free`
    // synchronizes the device first. `args` lives across the call and the entry reads it before
    // returning. `stream` is live for the call.
    let status = unsafe { scatter_cols_launch_raw(&args, stream.as_ptr(), &mut stage) };
    checked(status, stage)
}
