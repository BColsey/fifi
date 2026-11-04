//! Raw FFI to the sparse-attention relevance-score entry of
//! `native/resident/sparse_score.cu` (declared in `native/resident/sparse_score.h`). Child
//! module of `native/resident.rs`, so it inherits the `allow(unsafe_code)` of `native.rs`.
//!
//! Safety invariants this file relies on:
//! - The launch entry is called only through `CudaContext::launch_on`, on the thread that owns
//!   the context and under the dispatch guard. It only enqueues.
//! - Every range is a `RawRange`, proved inside a live allocation. Sizes, alignment and
//!   disjointness (`super::Span`) are proved again here, on the extents the kernel touches,
//!   before the call; the C entry repeats the alignment and shape checks on the addresses.
//! - The C argument struct is built in one `unsafe` block that cites those checks.
//! - A device allocation outlives the queued work because `inference_cuda_free` synchronizes the
//!   device before `cudaFree`.
//! - The `extern` declarations, the `repr(C)` mirrors and every raw pointer are private to this
//!   file. Nothing is re-exported.
use std::ffi::{c_int, c_void};
use std::mem::{align_of, offset_of, size_of};

use super::{DeviceFailure, RawRange, RawStream, STAGE_NONE, Span, checked, invalid_argument};

/// Mirror of `inference_cuda_sparse_score_args` in `sparse_score.h`, field for field.
#[repr(C)]
struct SparseScoreArgsRaw {
    q: *const c_void,
    basis: *const c_void,
    sketch: *const c_void,
    scores: *mut c_void,
    pos: *const c_void,
    status: *mut c_void,
    n_q_head: c_int,
    n_kv_head: c_int,
    head_dim: c_int,
    rank: c_int,
    cells: c_int,
    sink: c_int,
    recent: c_int,
    scale: f32,
}

const _: () = assert!(
    size_of::<SparseScoreArgsRaw>() == 80
        && align_of::<SparseScoreArgsRaw>() == 8
        && offset_of!(SparseScoreArgsRaw, q) == 0
        && offset_of!(SparseScoreArgsRaw, basis) == 8
        && offset_of!(SparseScoreArgsRaw, sketch) == 16
        && offset_of!(SparseScoreArgsRaw, scores) == 24
        && offset_of!(SparseScoreArgsRaw, pos) == 32
        && offset_of!(SparseScoreArgsRaw, status) == 40
        && offset_of!(SparseScoreArgsRaw, n_q_head) == 48
        && offset_of!(SparseScoreArgsRaw, sink) == 68
);

unsafe extern "C" {
    #[link_name = "inference_cuda_sparse_score_launch"]
    fn sparse_score_launch_raw(
        args: *const SparseScoreArgsRaw,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
}

/// Every integer argument of the launch, already bounded and converted by the wrapper before
/// `launch_on`, so no conversion can fail inside the launch closure.
#[derive(Debug, Clone, Copy)]
pub(in crate::kernels::cuda) struct SparseScoreInts {
    pub(in crate::kernels::cuda) n_q_head: c_int,
    pub(in crate::kernels::cuda) n_kv_head: c_int,
    pub(in crate::kernels::cuda) head_dim: c_int,
    pub(in crate::kernels::cuda) rank: c_int,
    pub(in crate::kernels::cuda) cells: c_int,
    pub(in crate::kernels::cuda) sink: c_int,
    pub(in crate::kernels::cuda) recent: c_int,
    pub(in crate::kernels::cuda) scale: f32,
}

pub(in crate::kernels::cuda) struct SparseScoreDataRaw<'a> {
    pub(in crate::kernels::cuda) q: RawRange<'a>,
    pub(in crate::kernels::cuda) basis: RawRange<'a>,
    pub(in crate::kernels::cuda) sketch: RawRange<'a>,
    pub(in crate::kernels::cuda) scores: RawRange<'a>,
    pub(in crate::kernels::cuda) ints: SparseScoreInts,
}

/// The entry's device words: the position of the appended token (read; kv_len = *pos + 1) and
/// the status word (mask 4 is OR-ed in on a refused launch).
pub(in crate::kernels::cuda) struct SparseScoreWordsRaw<'a> {
    pub(in crate::kernels::cuda) pos: RawRange<'a>,
    pub(in crate::kernels::cuda) status: RawRange<'a>,
}

fn size(n: c_int) -> Result<usize, DeviceFailure> {
    usize::try_from(n).map_err(|_| invalid_argument())
}

fn product(factors: &[usize]) -> Result<usize, DeviceFailure> {
    factors
        .iter()
        .try_fold(1usize, |acc, &factor| acc.checked_mul(factor))
        .ok_or_else(invalid_argument)
}

fn needs(ints: &SparseScoreInts) -> Result<(usize, usize, usize, usize), DeviceFailure> {
    let (q_heads, kv_heads, dim, rank) = (
        size(ints.n_q_head)?,
        size(ints.n_kv_head)?,
        size(ints.head_dim)?,
        size(ints.rank)?,
    );
    let cells = size(ints.cells)?;
    Ok((
        product(&[q_heads, dim, 4])?,
        product(&[kv_heads, rank, dim, 4])?,
        product(&[kv_heads, cells, rank, 2])?,
        product(&[kv_heads, cells, 4])?,
    ))
}

/// Alignment 4: every buffer the kernel touches is f32 words, an f16 plane at a 4-byte aligned
/// base, or a device word.
fn fits(range: &RawRange<'_>, need: usize) -> bool {
    range.bytes >= need && range.offset.is_multiple_of(4)
}

pub(in crate::kernels::cuda) fn sparse_score_launch(
    stream: &RawStream,
    io: SparseScoreDataRaw<'_>,
    words: SparseScoreWordsRaw<'_>,
) -> Result<(), DeviceFailure> {
    let (q, basis, sketch, scores) = needs(&io.ints)?;
    if !(fits(&io.q, q)
        && fits(&io.basis, basis)
        && fits(&io.sketch, sketch)
        && fits(&io.scores, scores)
        && fits(&words.pos, 4)
        && fits(&words.status, 4))
    {
        return Err(invalid_argument());
    }
    // `scores` is the only written range; it must be apart from the three reads and `status`
    // must be apart from everything (`pos` is read only).
    let score_span = Span::of(&io.scores, scores);
    let status_span = Span::of(&words.status, 4);
    let apart = [
        Span::of(&io.q, q),
        Span::of(&io.basis, basis),
        Span::of(&io.sketch, sketch),
        Span::of(&words.pos, 4),
    ]
    .iter()
    .all(|a| score_span.apart(a) && status_span.apart(a));
    if !(apart && score_span.apart(&status_span)) {
        return Err(invalid_argument());
    }
    let ints = io.ints;
    let args = SparseScoreArgsRaw {
        q: io.q.address().cast_const(),
        basis: io.basis.address().cast_const(),
        sketch: io.sketch.address().cast_const(),
        scores: io.scores.address(),
        pos: words.pos.address().cast_const(),
        status: words.status.address(),
        n_q_head: ints.n_q_head,
        n_kv_head: ints.n_kv_head,
        head_dim: ints.head_dim,
        rank: ints.rank,
        cells: ints.cells,
        sink: ints.sink,
        recent: ints.recent,
        scale: ints.scale,
    };
    let mut stage = STAGE_NONE;
    // SAFETY: (a) the fit and apart checks above proved every range inside its live allocation,
    // large enough, 4-aligned, and the written `scores` extent and `status` word disjoint from
    // every other range, so each pointer in `args` addresses bytes the kernel may touch; the C
    // entry repeats the checks on the addresses. (b) The allocations outlive the queued work
    // because `inference_cuda_free` synchronizes the device before `cudaFree`. (c) `stream` is
    // borrowed from a live `RawStream`. (d) `args` is a live local that the entry reads only
    // during the call. (e) The dispatch guard is held by `launch_on`, the only caller.
    let status = unsafe { sparse_score_launch_raw(&raw const args, stream.0.as_ptr(), &mut stage) };
    checked(status, stage)
}

#[cfg(test)]
mod tests {
    use super::SparseScoreArgsRaw;
    use std::mem::{align_of, offset_of, size_of};

    #[test]
    fn sparse_score_argument_layout_pins_the_c_struct() {
        assert_eq!(size_of::<SparseScoreArgsRaw>(), 80);
        assert_eq!(offset_of!(SparseScoreArgsRaw, q), 0);
        assert_eq!(offset_of!(SparseScoreArgsRaw, scores), 24);
        assert_eq!(offset_of!(SparseScoreArgsRaw, n_q_head), 48);
        assert_eq!(offset_of!(SparseScoreArgsRaw, sink), 68);
        assert_eq!(align_of::<SparseScoreArgsRaw>(), 8);
    }
}
