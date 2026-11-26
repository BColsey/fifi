//! Launch-only FFI for the resident Inkling depth stream
//! (`native/resident/inkling_stream.cu`, contract in `inkling_stream.h`).

use std::ffi::{c_int, c_void};
use std::mem::size_of;

use super::{DeviceFailure, RawRange, RawStream, STAGE_NONE, Span, checked, invalid_argument};

/// Largest `blocks` of a stream state: the mix reads this many sealed rows and the current row.
pub(crate) const MAX_BLOCKS: usize = 63;
const MAX_WIDTH: usize = 8192;

unsafe extern "C" {
    fn inference_cuda_inkling_stream_mix(
        state: *const c_void,
        query: *const c_void,
        out: *mut c_void,
        health: *mut c_void,
        first_block: c_int,
        blocks: c_int,
        width: c_int,
        eps: f32,
        rows: c_int,
        stride: c_int,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;

    fn inference_cuda_inkling_stream_seal(
        state: *mut c_void,
        next: *const c_void,
        health: *mut c_void,
        blocks: c_int,
        width: c_int,
        eps: f32,
        rows: c_int,
        stride: c_int,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;

    fn inference_cuda_inkling_stream_add(
        state: *mut c_void,
        row: *const c_void,
        health: *mut c_void,
        width: c_int,
        rows: c_int,
        stride: c_int,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;

    fn inference_cuda_inkling_stream_norm(
        src: *const c_void,
        dst: *mut c_void,
        health: *mut c_void,
        width: c_int,
        eps: f32,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
}

/// The most states one launch takes (one CUDA block each).
pub(crate) const MAX_STATES: usize = 65535;

/// `rows` stream states, each starting `stride` rows of the width after the previous one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct States {
    pub(crate) rows: usize,
    pub(crate) stride: usize,
}

impl States {
    /// One state of `used` rows.
    pub(crate) fn one(used: usize) -> Self {
        Self {
            rows: 1,
            stride: used,
        }
    }

    /// The bytes from the first state's start to the end of the last state's `used` rows, when
    /// the count and stride are valid.
    pub(crate) fn bytes(self, used: usize, width: usize) -> Option<usize> {
        if !(1..=MAX_STATES).contains(&self.rows)
            || self.stride < used
            || c_int::try_from(self.stride).is_err()
        {
            return None;
        }
        ((self.rows - 1) * self.stride)
            .checked_add(used)?
            .checked_mul(width)?
            .checked_mul(size_of::<f32>())
    }
}

fn shape(blocks: usize, width: usize, eps: f32) -> bool {
    blocks <= MAX_BLOCKS && (1..=MAX_WIDTH).contains(&width) && eps.is_finite() && eps > 0.0
}

fn sized(range: &RawRange<'_>, bytes: usize) -> bool {
    range.bytes >= bytes && range.offset.is_multiple_of(4)
}

/// `state` holds the current row and `blocks` sealed rows; `out` takes the weighted mix of
/// `Stream::mix(first_block, query)`.
pub(in crate::kernels::cuda) fn mix(
    stream: &RawStream,
    state: RawRange<'_>,
    query: RawRange<'_>,
    out: RawRange<'_>,
    health: RawRange<'_>,
    [first_block, blocks, width]: [usize; 3],
    (eps, states): (f32, States),
) -> Result<(), DeviceFailure> {
    let row = width * size_of::<f32>();
    let state_bytes = states
        .bytes(blocks + 1, width)
        .ok_or_else(invalid_argument)?;
    let out_bytes = states.rows * row;
    let health_bytes = size_of::<f32>();
    let valid = first_block <= blocks
        && shape(blocks, width, eps)
        && sized(&state, state_bytes)
        && sized(&query, row)
        && sized(&out, out_bytes)
        && sized(&health, health_bytes);
    if !valid {
        return Err(invalid_argument());
    }
    let (out_span, health_span, state_span, query_span) = (
        Span::of(&out, out_bytes),
        Span::of(&health, health_bytes),
        Span::of(&state, state_bytes),
        Span::of(&query, row),
    );
    if !out_span.apart(&state_span)
        || !out_span.apart(&query_span)
        || !health_span.apart(&state_span)
        || !health_span.apart(&query_span)
        || !health_span.apart(&out_span)
        || !state_span.apart(&query_span)
    {
        return Err(invalid_argument());
    }
    let mut stage = STAGE_NONE;
    // SAFETY: the ranges are aligned, hold the extents above and share no byte.
    let status = unsafe {
        inference_cuda_inkling_stream_mix(
            state.address() as *const c_void,
            query.address() as *const c_void,
            out.address(),
            health.address(),
            first_block as c_int,
            blocks as c_int,
            width as c_int,
            eps,
            states.rows as c_int,
            states.stride as c_int,
            stream.0.as_ptr(),
            &mut stage,
        )
    };
    checked(status, stage)
}

/// `sealed[blocks] = current * rms_scale(current)`, then `current = next` (`Stream::seal(next)`).
/// `state` must hold the new sealed row: `blocks + 2` rows.
pub(in crate::kernels::cuda) fn seal(
    stream: &RawStream,
    state: RawRange<'_>,
    next: RawRange<'_>,
    health: RawRange<'_>,
    [blocks, width]: [usize; 2],
    eps: f32,
    states: States,
) -> Result<(), DeviceFailure> {
    let row = width * size_of::<f32>();
    let state_bytes = states
        .bytes(blocks + 2, width)
        .ok_or_else(invalid_argument)?;
    let next_bytes = states.rows * row;
    let health_bytes = size_of::<f32>();
    if !shape(blocks, width, eps)
        || !sized(&state, state_bytes)
        || !sized(&next, next_bytes)
        || !sized(&health, health_bytes)
    {
        return Err(invalid_argument());
    }
    let (state_span, next_span, health_span) = (
        Span::of(&state, state_bytes),
        Span::of(&next, next_bytes),
        Span::of(&health, health_bytes),
    );
    if !state_span.apart(&next_span)
        || !health_span.apart(&state_span)
        || !health_span.apart(&next_span)
    {
        return Err(invalid_argument());
    }
    let mut stage = STAGE_NONE;
    // SAFETY: the ranges are aligned, hold the extents above and share no byte.
    let status = unsafe {
        inference_cuda_inkling_stream_seal(
            state.address(),
            next.address() as *const c_void,
            health.address(),
            blocks as c_int,
            width as c_int,
            eps,
            states.rows as c_int,
            states.stride as c_int,
            stream.0.as_ptr(),
            &mut stage,
        )
    };
    checked(status, stage)
}

/// `current = current + row` (`Stream::add_current(row)`).
pub(in crate::kernels::cuda) fn add(
    stream: &RawStream,
    state: RawRange<'_>,
    row: RawRange<'_>,
    health: RawRange<'_>,
    width: usize,
    states: States,
) -> Result<(), DeviceFailure> {
    let state_bytes = states.bytes(1, width).ok_or_else(invalid_argument)?;
    let row_bytes = states.rows * width * size_of::<f32>();
    let health_bytes = size_of::<f32>();
    if !shape(0, width, 1.0)
        || !sized(&state, state_bytes)
        || !sized(&row, row_bytes)
        || !sized(&health, health_bytes)
    {
        return Err(invalid_argument());
    }
    let (state_span, row_span, health_span) = (
        Span::of(&state, state_bytes),
        Span::of(&row, row_bytes),
        Span::of(&health, health_bytes),
    );
    if !state_span.apart(&row_span)
        || !health_span.apart(&state_span)
        || !health_span.apart(&row_span)
    {
        return Err(invalid_argument());
    }
    let mut stage = STAGE_NONE;
    // SAFETY: the ranges are aligned, hold the extents above and share no byte.
    let status = unsafe {
        inference_cuda_inkling_stream_add(
            state.address(),
            row.address() as *const c_void,
            health.address(),
            width as c_int,
            states.rows as c_int,
            states.stride as c_int,
            stream.0.as_ptr(),
            &mut stage,
        )
    };
    checked(status, stage)
}

/// `dst = src * rms_scale(src)`: one gate half of a cross-pass controller.
pub(in crate::kernels::cuda) fn norm(
    stream: &RawStream,
    src: RawRange<'_>,
    dst: RawRange<'_>,
    health: RawRange<'_>,
    width: usize,
    eps: f32,
) -> Result<(), DeviceFailure> {
    let row = width * size_of::<f32>();
    let health_bytes = size_of::<f32>();
    if !shape(0, width, eps)
        || !sized(&src, row)
        || !sized(&dst, row)
        || !sized(&health, health_bytes)
    {
        return Err(invalid_argument());
    }
    let (src_span, dst_span, health_span) = (
        Span::of(&src, row),
        Span::of(&dst, row),
        Span::of(&health, health_bytes),
    );
    if !dst_span.apart(&src_span) || !health_span.apart(&src_span) || !health_span.apart(&dst_span)
    {
        return Err(invalid_argument());
    }
    let mut stage = STAGE_NONE;
    // SAFETY: the ranges are aligned, hold the extents above and share no byte.
    let status = unsafe {
        inference_cuda_inkling_stream_norm(
            src.address() as *const c_void,
            dst.address(),
            health.address(),
            width as c_int,
            eps,
            stream.0.as_ptr(),
            &mut stage,
        )
    };
    checked(status, stage)
}
