use std::ffi::{c_int, c_void};

use super::{DeviceFailure, RawRange, RawStream, STAGE_NONE, Span, checked, invalid_argument};
use crate::kernels::cuda::resident::inkling_local_prep::{PrepMath, PrepShape};

unsafe extern "C" {
    fn inference_cuda_inkling_local_prep(
        ranges: *const *mut c_void,
        queries: c_int,
        kv: c_int,
        dims: c_int,
        history: c_int,
        qk_limit: f32,
        rel_limit: f32,
        eps: f32,
        stream: *mut c_void,
        stage: *mut c_int,
    ) -> c_int;
}

pub(in crate::kernels::cuda) fn launch(
    stream: &RawStream,
    ranges: [RawRange<'_>; 10],
    shape: PrepShape,
    history: usize,
    math: PrepMath,
) -> Result<(), DeviceFailure> {
    if !(1..=128).contains(&shape.queries)
        || !(1..=128).contains(&shape.kv)
        || !(1..=PrepShape::WIDTH).contains(&shape.dims)
        || history >= PrepShape::TAPS
        || ![math.qk_limit, math.rel_limit, math.eps]
            .into_iter()
            .all(|v| v.is_finite() && v > 0.0)
    {
        return Err(invalid_argument());
    }
    let words = shape.words();
    for (i, (range, words)) in ranges.iter().zip(words).enumerate() {
        let bytes = words * size_of::<f32>();
        if range.bytes() != bytes
            || !range.offset.is_multiple_of(4)
            || ranges[..i]
                .iter()
                .any(|other| !Span::of(range, bytes).apart(&Span::of(other, other.bytes())))
        {
            return Err(invalid_argument());
        }
    }
    let pointers = ranges.map(|r| r.address());
    let mut stage = STAGE_NONE;
    // SAFETY: the host pointer array lives for the call, and the CUDA spans are
    // aligned, disjoint and large enough for every access by the queued kernel.
    let status = unsafe {
        inference_cuda_inkling_local_prep(
            pointers.as_ptr(),
            shape.queries as c_int,
            shape.kv as c_int,
            shape.dims as c_int,
            history as c_int,
            math.qk_limit,
            math.rel_limit,
            math.eps,
            stream.0.as_ptr(),
            &mut stage,
        )
    };
    checked(status, stage)
}
