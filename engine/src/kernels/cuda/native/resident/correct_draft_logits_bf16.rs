//! The context dispatch guard supplies a live stream. RawRange proves allocation
//! bounds, and this module checks access sizes, alignment and write disjointness.
//! The backing device allocations synchronize before free.

use std::ffi::{c_int, c_void};
use std::mem::offset_of;

use super::{DeviceFailure, RawRange, RawStream, STAGE_NONE, Span, checked, invalid_argument};
use crate::kernels::cuda::resident::correct_draft_logits_bf16::{SOFTCAP_LUT_BYTES, logits_bytes};

#[repr(C)]
struct Args {
    base: *mut c_void,
    correction: *const c_void,
    softcap_lut: *const c_void,
    len: usize,
}

const _: () = assert!(size_of::<Args>() == 32 && offset_of!(Args, len) == 24);

unsafe extern "C" {
    #[link_name = "inference_cuda_correct_draft_logits_bf16_launch"]
    fn raw_launch(args: *const Args, stream: *mut c_void, stage: *mut c_int) -> c_int;
}

pub(in crate::kernels::cuda) struct CorrectDraftLogitsRaw<'a> {
    pub(in crate::kernels::cuda) len: usize,
    pub(in crate::kernels::cuda) base: RawRange<'a>,
    pub(in crate::kernels::cuda) correction: RawRange<'a>,
    pub(in crate::kernels::cuda) softcap_lut: RawRange<'a>,
}

pub(in crate::kernels::cuda) fn correct_draft_logits_bf16_launch(
    stream: &RawStream,
    call: &CorrectDraftLogitsRaw<'_>,
) -> Result<(), DeviceFailure> {
    let bytes = logits_bytes(call.len).ok_or_else(invalid_argument)?;
    let written = Span::of(&call.base, bytes);
    if call.base.bytes() < bytes
        || !call.base.offset().is_multiple_of(4)
        || call.correction.bytes() < bytes
        || !call.correction.offset().is_multiple_of(4)
        || call.softcap_lut.bytes() != SOFTCAP_LUT_BYTES
        || !call.softcap_lut.offset().is_multiple_of(2)
        || !written.apart(&Span::of(&call.correction, bytes))
        || !written.apart(&Span::of(&call.softcap_lut, SOFTCAP_LUT_BYTES))
    {
        return Err(invalid_argument());
    }

    let args = Args {
        base: call.base.address(),
        correction: call.correction.address().cast_const(),
        softcap_lut: call.softcap_lut.address().cast_const(),
        len: call.len,
    };
    let mut stage = STAGE_NONE;
    // SAFETY: the range checks prove every access in bounds and aligned, with
    // writes confined to base. The adapter copies args into the queued launch.
    let status = unsafe { raw_launch(&args, stream.as_ptr(), &mut stage) };
    checked(status, stage)
}
