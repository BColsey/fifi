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
    a: *const c_void,
    b: *const c_void,
    input: *mut c_void,
    route: *mut c_void,
    hidden: c_int,
    pre_scale: f32,
    iterations: c_int,
    fixed_identity: c_int,
}

#[repr(C)]
struct MhcMergeArgsRaw {
    a: *mut c_void,
    b: *mut c_void,
    branch: *const c_void,
    route: *const c_void,
    hidden: c_int,
}

#[repr(C)]
struct MhcAverageArgsRaw {
    a: *const c_void,
    b: *const c_void,
    out: *mut c_void,
    hidden: c_int,
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
    pub(in crate::kernels::cuda) a: RawRange<'a>,
    pub(in crate::kernels::cuda) b: RawRange<'a>,
    pub(in crate::kernels::cuda) input: RawRange<'a>,
    pub(in crate::kernels::cuda) route: RawRange<'a>,
    pub(in crate::kernels::cuda) hidden: usize,
    pub(in crate::kernels::cuda) pre_scale: f32,
    pub(in crate::kernels::cuda) iterations: usize,
    pub(in crate::kernels::cuda) fixed_identity: bool,
}

pub(in crate::kernels::cuda) struct MhcMergeRaw<'a> {
    pub(in crate::kernels::cuda) a: RawRange<'a>,
    pub(in crate::kernels::cuda) b: RawRange<'a>,
    pub(in crate::kernels::cuda) branch: RawRange<'a>,
    pub(in crate::kernels::cuda) route: RawRange<'a>,
    pub(in crate::kernels::cuda) hidden: usize,
}

pub(in crate::kernels::cuda) struct MhcAverageRaw<'a> {
    pub(in crate::kernels::cuda) a: RawRange<'a>,
    pub(in crate::kernels::cuda) b: RawRange<'a>,
    pub(in crate::kernels::cuda) out: RawRange<'a>,
    pub(in crate::kernels::cuda) hidden: usize,
}

const F32: usize = size_of::<f32>();
const ROUTE_FLOATS: usize = 6;

fn hidden_int(hidden: usize) -> Result<c_int, DeviceFailure> {
    c_int::try_from(hidden).map_err(|_| invalid_argument())
}

fn row(hidden: usize) -> Result<usize, DeviceFailure> {
    hidden.checked_mul(F32).ok_or_else(invalid_argument)
}

pub(in crate::kernels::cuda) fn mhc_route_launch(
    stream: &RawStream,
    io: MhcRouteRaw<'_>,
) -> Result<(), DeviceFailure> {
    let row_bytes = row(io.hidden)?;
    let logits = 8 * F32;
    let route_bytes = ROUTE_FLOATS * F32;
    let logits_ok = io.logits.bytes >= logits && io.logits.offset.is_multiple_of(4);
    let a_ok = io.a.bytes >= row_bytes && io.a.offset.is_multiple_of(4);
    let b_ok = io.b.bytes >= row_bytes && io.b.offset.is_multiple_of(4);
    let written = [
        (
            io.input.bytes >= row_bytes,
            io.input.offset.is_multiple_of(4),
            Span::of(&io.input, row_bytes),
        ),
        (
            io.route.bytes >= route_bytes,
            io.route.offset.is_multiple_of(4),
            Span::of(&io.route, route_bytes),
        ),
    ];
    let apart = written.iter().all(|w| {
        Span::of(&io.logits, logits).apart(&w.2)
            && Span::of(&io.a, row_bytes).apart(&w.2)
            && Span::of(&io.b, row_bytes).apart(&w.2)
    }) && written[0].2.apart(&written[1].2);
    let scalars = io.pre_scale.is_finite() && io.iterations <= 16;
    if !(logits_ok && a_ok && b_ok && apart && scalars) || !written.iter().all(|w| w.0 && w.1) {
        return Err(invalid_argument());
    }
    let args = MhcRouteArgsRaw {
        logits: io.logits.address().cast_const(),
        a: io.a.address().cast_const(),
        b: io.b.address().cast_const(),
        input: io.input.address(),
        route: io.route.address(),
        hidden: hidden_int(io.hidden)?,
        pre_scale: io.pre_scale,
        iterations: io.iterations as c_int,
        fixed_identity: io.fixed_identity as c_int,
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
    let route_bytes = ROUTE_FLOATS * F32;
    if io.a.bytes < row_bytes
        || io.b.bytes < row_bytes
        || io.branch.bytes < row_bytes
        || io.route.bytes < route_bytes
        || !io.a.offset.is_multiple_of(4)
        || !io.b.offset.is_multiple_of(4)
        || !io.branch.offset.is_multiple_of(4)
        || !io.route.offset.is_multiple_of(4)
        || !Span::of(&io.a, row_bytes).apart(&Span::of(&io.branch, row_bytes))
        || !Span::of(&io.b, row_bytes).apart(&Span::of(&io.branch, row_bytes))
        || !Span::of(&io.route, route_bytes).apart(&Span::of(&io.a, row_bytes))
        || !Span::of(&io.route, route_bytes).apart(&Span::of(&io.b, row_bytes))
        || !Span::of(&io.route, route_bytes).apart(&Span::of(&io.branch, row_bytes))
        // a and b are written in place after one read of both, so they must not overlap
        || !Span::of(&io.a, row_bytes).apart(&Span::of(&io.b, row_bytes))
    {
        return Err(invalid_argument());
    }
    let args = MhcMergeArgsRaw {
        a: io.a.address(),
        b: io.b.address(),
        branch: io.branch.address().cast_const(),
        route: io.route.address().cast_const(),
        hidden: hidden_int(io.hidden)?,
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
    if io.a.bytes < row_bytes
        || io.b.bytes < row_bytes
        || io.out.bytes < row_bytes
        || !io.a.offset.is_multiple_of(4)
        || !io.b.offset.is_multiple_of(4)
        || !io.out.offset.is_multiple_of(4)
        || !Span::of(&io.out, row_bytes).apart(&Span::of(&io.a, row_bytes))
        || !Span::of(&io.out, row_bytes).apart(&Span::of(&io.b, row_bytes))
    {
        return Err(invalid_argument());
    }
    let args = MhcAverageArgsRaw {
        a: io.a.address().cast_const(),
        b: io.b.address().cast_const(),
        out: io.out.address(),
        hidden: hidden_int(io.hidden)?,
    };
    let mut stage = STAGE_NONE;
    // SAFETY: as `mhc_route_launch` above.
    let status = unsafe { mhc_average_launch_raw(&raw const args, stream.0.as_ptr(), &mut stage) };
    checked(status, stage)
}
