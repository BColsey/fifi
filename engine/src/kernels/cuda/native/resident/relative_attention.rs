//! FFI for the launch-only F32 relative-attention step.
//!
//! RawRange proves allocation bounds. This module checks the precise extents,
//! alignment and write disjointness again before using the device addresses.
//! The context dispatch guard supplies the live stream, and free waits for queued
//! work before freeing a device allocation.

use std::ffi::{c_int, c_void};
use std::mem::{offset_of, size_of};

use super::{DeviceFailure, RawRange, RawStream, STAGE_NONE, Span, checked, invalid_argument};
use crate::kernels::cuda::resident::relative_attention::{RANGES, RelativeAttentionShape, WRITTEN};

#[repr(C)]
struct Args {
    ranges: [*mut c_void; RANGES],
    query_heads: c_int,
    kv_heads: c_int,
    relative_dims: c_int,
    relative_extent: c_int,
    capacity: c_int,
}

const _: () = assert!(
    size_of::<Args>() == 120
        && offset_of!(Args, query_heads) == 96
        && offset_of!(Args, capacity) == 112
);

unsafe extern "C" {
    #[link_name = "inference_cuda_relative_attention_launch"]
    fn raw_launch(args: *const Args, stream: *mut c_void, stage: *mut c_int) -> c_int;
}

pub(in crate::kernels::cuda) struct RelativeAttentionRaw<'a> {
    pub(in crate::kernels::cuda) ranges: [RawRange<'a>; RANGES],
    pub(in crate::kernels::cuda) shape: RelativeAttentionShape,
}

pub(in crate::kernels::cuda) fn relative_attention_launch(
    stream: &RawStream,
    call: &RelativeAttentionRaw<'_>,
) -> Result<(), DeviceFailure> {
    let bytes = call.shape.region_bytes().ok_or_else(invalid_argument)?;
    let spans = std::array::from_fn::<_, RANGES, _>(|i| Span::of(&call.ranges[i], bytes[i]));
    for (i, range) in call.ranges.iter().enumerate() {
        if range.bytes() < bytes[i]
            || !range.offset().is_multiple_of(4)
            || (i < WRITTEN && spans[i + 1..].iter().any(|other| !spans[i].apart(other)))
        {
            return Err(invalid_argument());
        }
    }
    let shape = call.shape;
    let args = Args {
        ranges: call.ranges.map(|range| range.address()),
        query_heads: shape.query_heads as c_int,
        kv_heads: shape.kv_heads as c_int,
        relative_dims: shape.relative_dims as c_int,
        relative_extent: shape.relative_extent as c_int,
        capacity: shape.capacity as c_int,
    };
    let mut stage = STAGE_NONE;
    // SAFETY: region_bytes bounded the geometry and the checks above prove all
    // touched ranges are aligned and in bounds, with writes disjoint from all other
    // regions. The explicit stream is live, and the adapter copies the args into
    // the launch before returning.
    let status = unsafe { raw_launch(&args, stream.as_ptr(), &mut stage) };
    checked(status, stage)
}
