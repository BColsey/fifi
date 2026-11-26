//! Launch-only FFI for rows of 32-bit words zeroed or copied in one launch
//! (`native/resident/word_rows.cu`, contract in `word_rows.h`).

use std::ffi::{c_int, c_void};

use super::{DeviceFailure, RawRange, RawStream, STAGE_NONE, Span, checked, invalid_argument};

/// Most rows of one launch (`INFERENCE_CUDA_WORD_ROWS_MAX`).
pub(in crate::kernels::cuda) const MAX_WORD_ROWS: usize = 64;

unsafe extern "C" {
    fn inference_cuda_zero_word_rows(
        rows: *const *mut c_void,
        words: *const c_int,
        count: c_int,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;

    fn inference_cuda_copy_word_rows(
        source: *const *const c_void,
        destination: *const *mut c_void,
        words: *const c_int,
        count: c_int,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
}

/// The word count of each row: 1 to [`MAX_WORD_ROWS`] rows, each a positive whole number of
/// words at a 4-byte aligned offset.
fn word_counts<'r, 'a: 'r>(
    rows: impl ExactSizeIterator<Item = &'r RawRange<'a>>,
) -> Result<Vec<c_int>, DeviceFailure> {
    if !(1..=MAX_WORD_ROWS).contains(&rows.len()) {
        return Err(invalid_argument());
    }
    rows.map(|row| {
        let whole = row.bytes.is_multiple_of(4) && row.offset.is_multiple_of(4);
        whole
            .then(|| c_int::try_from(row.bytes / 4).ok())
            .flatten()
            .filter(|&words| words > 0)
            .ok_or_else(invalid_argument)
    })
    .collect()
}

/// True when no written row shares a byte with another written row or a read row.
fn written_apart(written: &[Span<'_>], read: &[Span<'_>]) -> bool {
    written.iter().enumerate().all(|(at, span)| {
        written[at + 1..].iter().all(|other| span.apart(other))
            && read.iter().all(|other| span.apart(other))
    })
}

/// Zero every row in one launch, a block per row. No two rows share a byte.
pub(in crate::kernels::cuda) fn zero_word_rows(
    stream: &RawStream,
    rows: &[RawRange<'_>],
) -> Result<(), DeviceFailure> {
    let words = word_counts(rows.iter())?;
    let spans: Vec<Span<'_>> = rows.iter().map(|r| Span::of(r, r.bytes)).collect();
    if !written_apart(&spans, &[]) {
        return Err(invalid_argument());
    }
    let pointers: Vec<*mut c_void> = rows.iter().map(RawRange::address).collect();
    let mut stage = STAGE_NONE;
    // SAFETY: every row is aligned, holds its words and shares no byte with another row; the C
    // entry copies the lists into the kernel's parameters before it returns.
    let status = unsafe {
        inference_cuda_zero_word_rows(
            pointers.as_ptr(),
            words.as_ptr(),
            rows.len() as c_int,
            stream.0.as_ptr(),
            &mut stage,
        )
    };
    checked(status, stage)
}

/// Copy every `(source, destination)` pair of equal size in one launch, a block per pair. No
/// destination shares a byte with a source or another destination.
pub(in crate::kernels::cuda) fn copy_word_rows(
    stream: &RawStream,
    pairs: &[(RawRange<'_>, RawRange<'_>)],
) -> Result<(), DeviceFailure> {
    let words = word_counts(pairs.iter().map(|(source, _)| source))?;
    if word_counts(pairs.iter().map(|(_, destination)| destination))? != words {
        return Err(invalid_argument());
    }
    let sources: Vec<Span<'_>> = pairs.iter().map(|(s, _)| Span::of(s, s.bytes)).collect();
    let destinations: Vec<Span<'_>> = pairs.iter().map(|(_, d)| Span::of(d, d.bytes)).collect();
    if !written_apart(&destinations, &sources) {
        return Err(invalid_argument());
    }
    let source: Vec<*const c_void> = pairs
        .iter()
        .map(|(s, _)| s.address() as *const c_void)
        .collect();
    let destination: Vec<*mut c_void> = pairs.iter().map(|(_, d)| d.address()).collect();
    let mut stage = STAGE_NONE;
    // SAFETY: every row is aligned and holds its words, and no destination shares a byte with a
    // source or another destination; the C entry copies the lists into the kernel's parameters
    // before it returns.
    let status = unsafe {
        inference_cuda_copy_word_rows(
            source.as_ptr(),
            destination.as_ptr(),
            words.as_ptr(),
            pairs.len() as c_int,
            stream.0.as_ptr(),
            &mut stage,
        )
    };
    checked(status, stage)
}
