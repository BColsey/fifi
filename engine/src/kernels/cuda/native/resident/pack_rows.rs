//! Raw FFI to `inference_cuda_pack_rows_launch` of `native/resident/pack_rows.cu` (declared in
//! `native/resident/pack_rows.h`): the bit-exact pack of up to three tight `[m][n_i]` f32 sources
//! into the `[m][ld]` row layout of a K-row verify window. Child of `native::resident`, so it
//! inherits the `allow(unsafe_code)` of `native.rs`.
//!
//! Safety invariants this file relies on:
//! - The entry is called only through `CudaContext::launch_on`, on the owning thread and under the
//!   dispatch guard. It only enqueues and returns the launch status.
//! - Every range is a `RawRange`, proved inside a live allocation by `RawRange::new`. Sizes,
//!   alignment and disjointness (`super::Span`) are proved again here, before the call, on the
//!   extents the kernel touches; the C entry repeats the checks on the addresses.
//! - A device allocation outlives the queued work because `inference_cuda_free` synchronizes the
//!   device before `cudaFree`.
//! - The C argument struct is a live local that the entry reads only during the call. Its `repr(C)`
//!   mirror is private here and pinned by a `const` assert to the `static_assert` of `pack_rows.h`.
//!   No raw pointer and no unsafe function leaves this module.
#![cfg_attr(
    not(test),
    expect(dead_code, reason = "staged for the K-row verify body")
)]

use std::ffi::{c_int, c_void};
use std::mem::{offset_of, size_of};

use super::{DeviceFailure, RawRange, RawStream, STAGE_NONE, Span, checked, invalid_argument};

pub(in crate::kernels::cuda) const PACK_MAX_ROWS: usize = 16;
pub(in crate::kernels::cuda) const PACK_MAX_SEGMENTS: usize = 3;
pub(in crate::kernels::cuda) const PACK_MAX_N: usize = 65536;
pub(in crate::kernels::cuda) const PACK_MAX_LD: usize = 1 << 20;

const WORD: usize = size_of::<f32>();

/// Mirror of `inference_cuda_pack_segment`.
#[repr(C)]
#[derive(Clone, Copy)]
struct PackSegmentRaw {
    src: *const c_void,
    n: c_int,
    dst_off: c_int,
}

/// Mirror of `inference_cuda_pack_rows_args`.
#[repr(C)]
struct PackRowsArgsRaw {
    dst: *mut c_void,
    seg: [PackSegmentRaw; 3],
    m: c_int,
    ld: c_int,
    segs: c_int,
}

const _: () = assert!(
    size_of::<PackSegmentRaw>() == 16
        && offset_of!(PackSegmentRaw, n) == 8
        && offset_of!(PackSegmentRaw, dst_off) == 12
);
const _: () = assert!(
    size_of::<PackRowsArgsRaw>() == 72
        && offset_of!(PackRowsArgsRaw, seg) == 8
        && offset_of!(PackRowsArgsRaw, m) == 56
        && offset_of!(PackRowsArgsRaw, ld) == 60
        && offset_of!(PackRowsArgsRaw, segs) == 64
);

unsafe extern "C" {
    #[link_name = "inference_cuda_pack_rows_launch"]
    fn pack_rows_raw(
        args: *const PackRowsArgsRaw,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
}

/// One source of a pack: `n` columns, `m` rows tight, copied to column `dst_off` of every row of
/// the destination.
pub(in crate::kernels::cuda) struct PackSegmentRange<'a> {
    pub(in crate::kernels::cuda) src: RawRange<'a>,
    pub(in crate::kernels::cuda) n: usize,
    pub(in crate::kernels::cuda) dst_off: usize,
}

pub(in crate::kernels::cuda) struct PackRowsRaw<'a> {
    pub(in crate::kernels::cuda) dst: RawRange<'a>,
    pub(in crate::kernels::cuda) segments: Vec<PackSegmentRange<'a>>,
    pub(in crate::kernels::cuda) m: usize,
    pub(in crate::kernels::cuda) ld: usize,
}

/// The columns of one destination row that the segments span, `None` for a bad segment list.
fn reach(call: &PackRowsRaw<'_>) -> Option<usize> {
    let mut reach = 0usize;
    for (i, seg) in call.segments.iter().enumerate() {
        let end = seg.dst_off.checked_add(seg.n)?;
        if seg.n == 0 || seg.n > PACK_MAX_N || end > call.ld {
            return None;
        }
        for other in &call.segments[i + 1..] {
            let other_end = other.dst_off.checked_add(other.n)?;
            if !(end <= other.dst_off || other_end <= seg.dst_off) {
                return None;
            }
        }
        reach = reach.max(end);
    }
    Some(reach)
}

fn check_pack(call: &PackRowsRaw<'_>) -> Result<(usize, [c_int; 3]), DeviceFailure> {
    let in_range = (1..=PACK_MAX_ROWS).contains(&call.m)
        && (1..=PACK_MAX_LD).contains(&call.ld)
        && (1..=PACK_MAX_SEGMENTS).contains(&call.segments.len());
    if !in_range {
        return Err(invalid_argument());
    }
    let reach = reach(call).ok_or_else(invalid_argument)?;
    let dst_bytes = (call.m - 1)
        .checked_mul(call.ld)
        .and_then(|words| words.checked_add(reach))
        .and_then(|words| words.checked_mul(WORD))
        .ok_or_else(invalid_argument)?;
    if call.dst.bytes < dst_bytes || !call.dst.offset.is_multiple_of(WORD) {
        return Err(invalid_argument());
    }
    let dst = Span::of(&call.dst, dst_bytes);
    for seg in &call.segments {
        let src_bytes = call
            .m
            .checked_mul(seg.n)
            .and_then(|words| words.checked_mul(WORD))
            .ok_or_else(invalid_argument)?;
        if seg.src.bytes < src_bytes
            || !seg.src.offset.is_multiple_of(WORD)
            || !Span::of(&seg.src, src_bytes).apart(&dst)
        {
            return Err(invalid_argument());
        }
    }
    let ints = [
        c_int::try_from(call.m).map_err(|_| invalid_argument())?,
        c_int::try_from(call.ld).map_err(|_| invalid_argument())?,
        c_int::try_from(call.segments.len()).map_err(|_| invalid_argument())?,
    ];
    Ok((reach, ints))
}

/// Queues the pack kernel of `call`.
pub(in crate::kernels::cuda) fn pack_rows_launch(
    stream: &RawStream,
    call: PackRowsRaw<'_>,
) -> Result<(), DeviceFailure> {
    let (_, [m, ld, segs]) = check_pack(&call)?;
    let mut seg = [PackSegmentRaw {
        src: std::ptr::null(),
        n: 0,
        dst_off: 0,
    }; 3];
    for (slot, source) in seg.iter_mut().zip(&call.segments) {
        *slot = PackSegmentRaw {
            src: source.src.address().cast_const(),
            n: c_int::try_from(source.n).map_err(|_| invalid_argument())?,
            dst_off: c_int::try_from(source.dst_off).map_err(|_| invalid_argument())?,
        };
    }
    let args = PackRowsArgsRaw {
        dst: call.dst.address(),
        seg,
        m,
        ld,
        segs,
    };
    let mut stage = STAGE_NONE;
    // SAFETY: `check_pack` proved the destination and every source inside their live allocations,
    // large enough for the call (`dst` for `(m - 1) * ld + reach` words, each source for `m * n`
    // words), 4-byte aligned, the segments' column ranges disjoint and inside `ld`, and every
    // source apart from the destination, so the writes to `dst` cannot alias a read; the entry
    // repeats these checks on the addresses. The allocations outlive the queued work
    // (`inference_cuda_free` synchronizes first). `args` is a live local that the entry reads only
    // during the call: the kernel arguments are copied at launch, so nothing keeps the pointer.
    // `stream` is a live handle for the call, and `stage` is a live, writable c_int.
    let status = unsafe { pack_rows_raw(&args, stream.0.as_ptr(), &mut stage) };
    checked(status, stage)
}
