//! Raw FFI to the mHC stream routing entries of `native/resident/mhc_route.cu` (declared in
//! `native/resident/mhc_route.h`): route, merge and average over the two residual streams of
//! the SenseNova Loop mHC layers. Child module of `native/resident.rs`, so it inherits the
//! `allow(unsafe_code)` of `native.rs`.
//!
//! Safety invariants this file relies on (as `attn_decode.rs`):
//! - The entries are called only through `CudaContext::launch_on`, on the thread that owns
//!   the context and under the dispatch guard.
//! - Every range is a `RawRange`, proved inside its live allocation. Sizes, alignment and
//!   disjointness (`super::Span`) are proved again here before the call. The C entries check
//!   pointer/scalar/alignment conditions, not allocation extents or overlap.
//! - The `extern` declarations, the `repr(C)` mirrors and every raw pointer are private to
//!   this file. Nothing is re-exported.

use std::ffi::{c_int, c_void};

use super::{DeviceFailure, RawRange, RawStream, STAGE_NONE, Span, checked, invalid_argument};

#[repr(C)]
struct MhcRouteArgsRaw {
    logits: *const c_void,
    bias: *const c_void,
    a: *const c_void,
    b: *const c_void,
    input: *mut c_void,
    route: *mut c_void,
    hidden: c_int,
    pre_scale: f32,
    iterations: c_int,
    fixed_identity: c_int,
    rows: c_int,
    ld_ab: c_int,
}

#[repr(C)]
struct MhcMergeArgsRaw {
    a: *mut c_void,
    b: *mut c_void,
    branch: *const c_void,
    route: *const c_void,
    hidden: c_int,
    rows: c_int,
    ld_ab: c_int,
}

#[repr(C)]
struct MhcAverageArgsRaw {
    a: *const c_void,
    b: *const c_void,
    out: *mut c_void,
    hidden: c_int,
    rows: c_int,
    ld_ab: c_int,
}

unsafe extern "C" {
    #[link_name = "inference_cuda_mhc_route_launch"]
    fn mhc_route_launch_raw(
        args: *const MhcRouteArgsRaw,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
    #[link_name = "inference_cuda_mhc_merge_launch"]
    fn mhc_merge_launch_raw(
        args: *const MhcMergeArgsRaw,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
    #[link_name = "inference_cuda_mhc_average_launch"]
    fn mhc_average_launch_raw(
        args: *const MhcAverageArgsRaw,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
}

pub(in crate::kernels::cuda) struct MhcRouteRaw<'a> {
    pub(in crate::kernels::cuda) logits: RawRange<'a>,
    pub(in crate::kernels::cuda) bias: Option<RawRange<'a>>,
    pub(in crate::kernels::cuda) a: RawRange<'a>,
    pub(in crate::kernels::cuda) b: RawRange<'a>,
    pub(in crate::kernels::cuda) input: RawRange<'a>,
    pub(in crate::kernels::cuda) route: RawRange<'a>,
    pub(in crate::kernels::cuda) hidden: usize,
    pub(in crate::kernels::cuda) pre_scale: f32,
    pub(in crate::kernels::cuda) iterations: usize,
    pub(in crate::kernels::cuda) fixed_identity: bool,
    pub(in crate::kernels::cuda) rows: usize,
    pub(in crate::kernels::cuda) ld_ab: usize,
}

pub(in crate::kernels::cuda) struct MhcMergeRaw<'a> {
    pub(in crate::kernels::cuda) a: RawRange<'a>,
    pub(in crate::kernels::cuda) b: RawRange<'a>,
    pub(in crate::kernels::cuda) branch: RawRange<'a>,
    pub(in crate::kernels::cuda) route: RawRange<'a>,
    pub(in crate::kernels::cuda) hidden: usize,
    pub(in crate::kernels::cuda) rows: usize,
    pub(in crate::kernels::cuda) ld_ab: usize,
}

pub(in crate::kernels::cuda) struct MhcAverageRaw<'a> {
    pub(in crate::kernels::cuda) a: RawRange<'a>,
    pub(in crate::kernels::cuda) b: RawRange<'a>,
    pub(in crate::kernels::cuda) out: RawRange<'a>,
    pub(in crate::kernels::cuda) hidden: usize,
    pub(in crate::kernels::cuda) rows: usize,
    pub(in crate::kernels::cuda) ld_ab: usize,
}

const F32: usize = size_of::<f32>();
const ROUTE_FLOATS: usize = 6;
/// Rows of one chunk: the mmq chunk cap.
const MAX_ROWS: usize = 4096;

fn dim_int(value: usize) -> Result<c_int, DeviceFailure> {
    c_int::try_from(value).map_err(|_| invalid_argument())
}

fn row(hidden: usize) -> Result<usize, DeviceFailure> {
    hidden.checked_mul(F32).ok_or_else(invalid_argument)
}

/// The bytes from the first row's start through the last row's end: `rows` rows of `row_bytes`,
/// consecutive ones `stride` apart (the last needs only its own bytes).
fn rows_extent(rows: usize, stride: usize, row_bytes: usize) -> Result<usize, DeviceFailure> {
    rows.checked_sub(1)
        .and_then(|lead| lead.checked_mul(stride))
        .and_then(|lead| lead.checked_add(row_bytes))
        .ok_or_else(invalid_argument)
}

/// Checks the row count and the stream stride; returns the bytes one stream half covers. The
/// stride rule alone does not keep the two streams apart: [`streams_apart`] proves that from
/// their starts.
fn streams(rows: usize, ld_ab: usize, hidden: usize) -> Result<usize, DeviceFailure> {
    if rows == 0 || rows > MAX_ROWS || ld_ab < 2 * hidden {
        return Err(invalid_argument());
    }
    let stride = ld_ab.checked_mul(F32).ok_or_else(invalid_argument)?;
    rows_extent(rows, stride, row(hidden)?)
}

/// True when no element of `a`'s rows is an element of `b`'s rows. Each stream is `rows` rows
/// of `hidden` f32 `ld_ab` apart from its start, and the starts are byte offsets inside one
/// allocation (two allocations never share an element). The rows repeat every `ld_ab`, so the
/// streams are apart exactly when the upper start lies past the lower stream's last row, or
/// when the gap between the starts, taken mod `ld_ab`, is at least `hidden` and leaves at
/// least `hidden` before the next row: the `[a || b]` rows of a chunk, with or without
/// padding. The callers hold `rows >= 1` and `ld_ab >= 2 * hidden`.
pub(in crate::kernels::cuda) fn streams_apart(
    same_allocation: bool,
    first: usize,
    second: usize,
    rows: usize,
    ld_ab: usize,
    hidden: usize,
) -> bool {
    if !same_allocation {
        return true;
    }
    let (low, high) = if first <= second {
        (first, second)
    } else {
        (second, first)
    };
    let gap = (high - low) / F32;
    let last = (rows - 1) * ld_ab + hidden;
    gap >= last || (hidden..=ld_ab - hidden).contains(&(gap % ld_ab))
}

pub(in crate::kernels::cuda) fn mhc_route_launch(
    stream: &RawStream,
    io: MhcRouteRaw<'_>,
) -> Result<(), DeviceFailure> {
    let row_bytes = row(io.hidden)?;
    let stream_bytes = streams(io.rows, io.ld_ab, io.hidden)?;
    let logit_row = 8 * F32;
    let logits = logit_row * io.rows;
    let route_bytes = ROUTE_FLOATS * F32 * io.rows;
    let input_bytes = rows_extent(io.rows, row_bytes, row_bytes)?;
    let logits_ok = io.logits.bytes >= logits && io.logits.offset.is_multiple_of(4);
    let mut reads = vec![Span::of(&io.logits, logits)];
    if let Some(bias) = &io.bias {
        if bias.bytes < logit_row || !bias.offset.is_multiple_of(4) {
            return Err(invalid_argument());
        }
        reads.push(Span::of(bias, logit_row));
    }
    let a_ok = io.a.bytes >= stream_bytes && io.a.offset.is_multiple_of(4);
    let b_ok = io.b.bytes >= stream_bytes && io.b.offset.is_multiple_of(4);
    let written = [
        (
            io.input.bytes >= input_bytes,
            io.input.offset.is_multiple_of(4),
            Span::of(&io.input, input_bytes),
        ),
        (
            io.route.bytes >= route_bytes,
            io.route.offset.is_multiple_of(4),
            Span::of(&io.route, route_bytes),
        ),
    ];
    let apart = written.iter().all(|w| {
        reads.iter().all(|read| read.apart(&w.2))
            && Span::of(&io.a, stream_bytes).apart(&w.2)
            && Span::of(&io.b, stream_bytes).apart(&w.2)
    }) && written[0].2.apart(&written[1].2);
    let scalars = io.pre_scale.is_finite() && io.iterations <= 32;
    if !(logits_ok && a_ok && b_ok && apart && scalars) || !written.iter().all(|w| w.0 && w.1) {
        return Err(invalid_argument());
    }
    let args = MhcRouteArgsRaw {
        logits: io.logits.address().cast_const(),
        bias: io
            .bias
            .as_ref()
            .map_or(std::ptr::null(), |bias| bias.address().cast_const()),
        a: io.a.address().cast_const(),
        b: io.b.address().cast_const(),
        input: io.input.address(),
        route: io.route.address(),
        hidden: dim_int(io.hidden)?,
        pre_scale: io.pre_scale,
        iterations: io.iterations as c_int,
        fixed_identity: io.fixed_identity as c_int,
        rows: dim_int(io.rows)?,
        ld_ab: dim_int(io.ld_ab)?,
    };
    let mut stage = STAGE_NONE;
    // SAFETY: the claims above proved every range inside its live allocation, large enough,
    // aligned, and the written ranges disjoint from every other range. The C entry checks
    // pointer/scalar conditions, not allocation extents or overlap. Allocations stay valid
    // until work finishes because `inference_cuda_free` synchronizes the device first.
    let status = unsafe { mhc_route_launch_raw(&raw const args, stream.0.as_ptr(), &mut stage) };
    checked(status, stage)
}

pub(in crate::kernels::cuda) fn mhc_merge_launch(
    stream: &RawStream,
    io: MhcMergeRaw<'_>,
) -> Result<(), DeviceFailure> {
    let row_bytes = row(io.hidden)?;
    let stream_bytes = streams(io.rows, io.ld_ab, io.hidden)?;
    let branch_bytes = rows_extent(io.rows, row_bytes, row_bytes)?;
    let route_bytes = ROUTE_FLOATS * F32 * io.rows;
    if io.a.bytes < stream_bytes
        || io.b.bytes < stream_bytes
        || io.branch.bytes < branch_bytes
        || io.route.bytes < route_bytes
        || !io.a.offset.is_multiple_of(4)
        || !io.b.offset.is_multiple_of(4)
        || !io.branch.offset.is_multiple_of(4)
        || !io.route.offset.is_multiple_of(4)
        || !Span::of(&io.a, stream_bytes).apart(&Span::of(&io.branch, branch_bytes))
        || !Span::of(&io.b, stream_bytes).apart(&Span::of(&io.branch, branch_bytes))
        || !Span::of(&io.route, route_bytes).apart(&Span::of(&io.a, stream_bytes))
        || !Span::of(&io.route, route_bytes).apart(&Span::of(&io.b, stream_bytes))
        || !Span::of(&io.route, route_bytes).apart(&Span::of(&io.branch, branch_bytes))
        // a and b are written in place after one read of both, so no element may be both:
        // one row's halves are two apart ranges, and a chunk's halves interleave inside their
        // [a || b] rows.
        || !streams_apart(
            std::ptr::eq(io.a.device, io.b.device),
            io.a.offset,
            io.b.offset,
            io.rows,
            io.ld_ab,
            io.hidden,
        )
    {
        return Err(invalid_argument());
    }
    let args = MhcMergeArgsRaw {
        a: io.a.address(),
        b: io.b.address(),
        branch: io.branch.address().cast_const(),
        route: io.route.address().cast_const(),
        hidden: dim_int(io.hidden)?,
        rows: dim_int(io.rows)?,
        ld_ab: dim_int(io.ld_ab)?,
    };
    let mut stage = STAGE_NONE;
    // SAFETY: as `mhc_route_launch` above; a and b are the same range in and out (in place).
    let status = unsafe { mhc_merge_launch_raw(&raw const args, stream.0.as_ptr(), &mut stage) };
    checked(status, stage)
}

pub(in crate::kernels::cuda) fn mhc_average_launch(
    stream: &RawStream,
    io: MhcAverageRaw<'_>,
) -> Result<(), DeviceFailure> {
    let row_bytes = row(io.hidden)?;
    let stream_bytes = streams(io.rows, io.ld_ab, io.hidden)?;
    let out_bytes = rows_extent(io.rows, row_bytes, row_bytes)?;
    if io.a.bytes < stream_bytes
        || io.b.bytes < stream_bytes
        || io.out.bytes < out_bytes
        || !io.a.offset.is_multiple_of(4)
        || !io.b.offset.is_multiple_of(4)
        || !io.out.offset.is_multiple_of(4)
        || !Span::of(&io.out, out_bytes).apart(&Span::of(&io.a, stream_bytes))
        || !Span::of(&io.out, out_bytes).apart(&Span::of(&io.b, stream_bytes))
    {
        return Err(invalid_argument());
    }
    let args = MhcAverageArgsRaw {
        a: io.a.address().cast_const(),
        b: io.b.address().cast_const(),
        out: io.out.address(),
        hidden: dim_int(io.hidden)?,
        rows: dim_int(io.rows)?,
        ld_ab: dim_int(io.ld_ab)?,
    };
    let mut stage = STAGE_NONE;
    // SAFETY: as `mhc_route_launch` above.
    let status = unsafe { mhc_average_launch_raw(&raw const args, stream.0.as_ptr(), &mut stage) };
    checked(status, stage)
}
