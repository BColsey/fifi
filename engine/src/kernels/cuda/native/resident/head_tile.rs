//! FFI for a bit-exact grouped-to-tiled head permutation. Both ranges are rechecked for
//! size, alignment and disjointness; the C entry repeats those checks before launch. The
//! stream and allocations remain live through queued work; no raw pointer escapes this module.
use super::{DeviceFailure, RawRange, RawStream, STAGE_NONE, Span, checked, invalid_argument};
use std::{
    ffi::{c_int, c_void},
    mem::offset_of,
};

#[repr(C)]
struct Args {
    dst: *mut c_void,
    src: *const c_void,
    key_heads: c_int,
    value_heads: c_int,
    head_dim: c_int,
}
const _: () = assert!(
    size_of::<Args>() == 32
        && offset_of!(Args, dst) == 0
        && offset_of!(Args, src) == 8
        && offset_of!(Args, key_heads) == 16
        && offset_of!(Args, value_heads) == 20
        && offset_of!(Args, head_dim) == 24
);
unsafe extern "C" {
    #[link_name = "inference_cuda_head_tile_launch"]
    fn launch_raw(args: *const Args, stream: *mut c_void, stage: *mut c_int) -> c_int;
}
pub(in crate::kernels::cuda) struct HeadTileRaw<'a> {
    pub(in crate::kernels::cuda) dst: RawRange<'a>,
    pub(in crate::kernels::cuda) src: RawRange<'a>,
    pub(in crate::kernels::cuda) key_heads: usize,
    pub(in crate::kernels::cuda) value_heads: usize,
    pub(in crate::kernels::cuda) head_dim: usize,
}

pub(in crate::kernels::cuda) fn head_tile_launch(
    stream: &RawStream,
    call: &HeadTileRaw<'_>,
) -> Result<(), DeviceFailure> {
    let bytes = call
        .value_heads
        .checked_mul(call.head_dim)
        .filter(|&n| n > 0 && n <= 1 << 24)
        .and_then(|n| n.checked_mul(4))
        .ok_or_else(invalid_argument)?;
    if call.key_heads == 0
        || !call.value_heads.is_multiple_of(call.key_heads)
        || call.dst.bytes < bytes
        || call.src.bytes < bytes
        || !call.dst.offset.is_multiple_of(4)
        || !call.src.offset.is_multiple_of(4)
        || !Span::of(&call.dst, bytes).apart(&Span::of(&call.src, bytes))
    {
        return Err(invalid_argument());
    }
    let args = Args {
        dst: call.dst.address(),
        src: call.src.address().cast_const(),
        key_heads: c_int::try_from(call.key_heads).map_err(|_| invalid_argument())?,
        value_heads: c_int::try_from(call.value_heads).map_err(|_| invalid_argument())?,
        head_dim: c_int::try_from(call.head_dim).map_err(|_| invalid_argument())?,
    };
    let mut stage = STAGE_NONE;
    // SAFETY: both ranges are live, sized, aligned and disjoint, and integer geometry is
    // checked above. The C entry rechecks and only enqueues onto the live stream.
    let status = unsafe { launch_raw(&args, stream.0.as_ptr(), &mut stage) };
    checked(status, stage)
}
