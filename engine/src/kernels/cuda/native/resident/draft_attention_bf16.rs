//! The live stream comes from the context dispatch guard. RawRange proves allocation
//! bounds, and this module checks each access extent, alignment and disjointness
//! before taking addresses. Free synchronizes before releasing device memory.

use std::ffi::{c_int, c_void};
use std::mem::{offset_of, size_of};

use super::{DeviceFailure, RawRange, RawStream, STAGE_NONE, Span, checked, invalid_argument};
use crate::kernels::cuda::resident::draft_attention_bf16::{RANGES, Shape};

#[repr(C)]
struct Args {
    ranges: [*mut c_void; RANGES],
    query_heads: c_int,
    kv_heads: c_int,
    rows: c_int,
    capacity: c_int,
    first: c_int,
    len: c_int,
}

const _: () = assert!(
    size_of::<Args>() == 72 && offset_of!(Args, query_heads) == 48 && offset_of!(Args, len) == 68
);

unsafe extern "C" {
    #[link_name = "inference_cuda_draft_attention_bf16_launch"]
    fn raw_launch(args: *const Args, stream: *mut c_void, stage: *mut c_int) -> c_int;
}

pub(in crate::kernels::cuda) struct DraftAttentionRaw<'a> {
    pub(in crate::kernels::cuda) ranges: [RawRange<'a>; RANGES],
    pub(in crate::kernels::cuda) shape: Shape,
}

pub(in crate::kernels::cuda) fn draft_attention_bf16_launch(
    stream: &RawStream,
    call: &DraftAttentionRaw<'_>,
) -> Result<(), DeviceFailure> {
    let shape = call.shape;
    let bytes = shape.region_bytes().ok_or_else(invalid_argument)?;
    for (range, need) in call.ranges.iter().zip(bytes) {
        if range.bytes() < need || !range.offset().is_multiple_of(2) {
            return Err(invalid_argument());
        }
    }
    let spans = std::array::from_fn::<_, RANGES, _>(|i| Span::of(&call.ranges[i], bytes[i]));
    if spans[1..].iter().any(|other| !spans[0].apart(other))
        || spans[2..4]
            .iter()
            .any(|draft| spans[4..].iter().any(|committed| !draft.apart(committed)))
    {
        return Err(invalid_argument());
    }
    let args = Args {
        ranges: call.ranges.map(|range| range.address()),
        query_heads: shape.query_heads as c_int,
        kv_heads: shape.kv_heads as c_int,
        rows: shape.rows as c_int,
        capacity: shape.capacity as c_int,
        first: shape.first as c_int,
        len: shape.len as c_int,
    };
    let mut stage = STAGE_NONE;
    // SAFETY: geometry and range checks prove all device accesses aligned and
    // in bounds, with output disjoint from inputs. The adapter copies the
    // argument struct into the launch on the live stream.
    let status = unsafe { raw_launch(&args, stream.as_ptr(), &mut stage) };
    checked(status, stage)
}
