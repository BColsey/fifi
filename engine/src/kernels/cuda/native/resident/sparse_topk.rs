//! Raw FFI to the sparse-attention deterministic top-K entry of
//! `native/resident/sparse_topk.cu` (declared in `native/resident/sparse_topk.h`). Child module
//! of `native/resident.rs`, so it inherits the `allow(unsafe_code)` of `native.rs`.
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
//! - The `extern` declarations, the `repr(C)` mirror and every raw pointer are private to this
//!   file. Nothing is re-exported.
use std::ffi::{c_int, c_void};
use std::mem::{align_of, offset_of, size_of};

use super::{DeviceFailure, RawRange, RawStream, STAGE_NONE, Span, checked, invalid_argument};

/// Mirror of `inference_cuda_sparse_topk_args` in `sparse_topk.h`, field for field.
#[repr(C)]
struct SparseTopkArgsRaw {
    scores: *const c_void,
    indices: *mut c_void,
    counts: *mut c_void,
    pos: *const c_void,
    status: *mut c_void,
    n_kv_head: c_int,
    cells: c_int,
    top_k: c_int,
    sink: c_int,
    recent: c_int,
}

const _: () = assert!(
    size_of::<SparseTopkArgsRaw>() == 64
        && align_of::<SparseTopkArgsRaw>() == 8
        && offset_of!(SparseTopkArgsRaw, scores) == 0
        && offset_of!(SparseTopkArgsRaw, indices) == 8
        && offset_of!(SparseTopkArgsRaw, counts) == 16
        && offset_of!(SparseTopkArgsRaw, pos) == 24
        && offset_of!(SparseTopkArgsRaw, status) == 32
        && offset_of!(SparseTopkArgsRaw, n_kv_head) == 40
);

unsafe extern "C" {
    #[link_name = "inference_cuda_sparse_topk_launch"]
    fn sparse_topk_launch_raw(
        args: *const SparseTopkArgsRaw,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
}

/// Every integer argument of the launch, already bounded and converted by the wrapper before
/// `launch_on`, so no conversion can fail inside the launch closure.
#[derive(Debug, Clone, Copy)]
pub(in crate::kernels::cuda) struct SparseTopkInts {
    pub(in crate::kernels::cuda) n_kv_head: c_int,
    pub(in crate::kernels::cuda) cells: c_int,
    pub(in crate::kernels::cuda) top_k: c_int,
    pub(in crate::kernels::cuda) sink: c_int,
    pub(in crate::kernels::cuda) recent: c_int,
}

pub(in crate::kernels::cuda) struct SparseTopkDataRaw<'a> {
    pub(in crate::kernels::cuda) scores: RawRange<'a>,
    pub(in crate::kernels::cuda) indices: RawRange<'a>,
    pub(in crate::kernels::cuda) counts: RawRange<'a>,
    pub(in crate::kernels::cuda) ints: SparseTopkInts,
}

/// The entry's device words: the position of the appended token (read; kv_len = *pos + 1) and
/// the status word (mask 4 is OR-ed in on a refused launch).
pub(in crate::kernels::cuda) struct SparseTopkWordsRaw<'a> {
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

fn needs(ints: &SparseTopkInts) -> Result<(usize, usize, usize), DeviceFailure> {
    let (kv_heads, cells) = (size(ints.n_kv_head)?, size(ints.cells)?);
    let row = size(ints.top_k)? + size(ints.sink)? + size(ints.recent)?;
    Ok((
        product(&[kv_heads, cells, 4])?,
        product(&[kv_heads, row, 4])?,
        product(&[kv_heads, 4])?,
    ))
}

/// Alignment 4: the f32 scores, the u32 index rows, the u32 counts and the device words.
fn fits(range: &RawRange<'_>, need: usize) -> bool {
    range.bytes >= need && range.offset.is_multiple_of(4)
}

pub(in crate::kernels::cuda) fn sparse_topk_launch(
    stream: &RawStream,
    io: SparseTopkDataRaw<'_>,
    words: SparseTopkWordsRaw<'_>,
) -> Result<(), DeviceFailure> {
    let (scores, indices, counts) = needs(&io.ints)?;
    if !(fits(&io.scores, scores)
        && fits(&io.indices, indices)
        && fits(&io.counts, counts)
        && fits(&words.pos, 4)
        && fits(&words.status, 4))
    {
        return Err(invalid_argument());
    }
    // `indices` and `counts` are the written ranges; `status` is written on a refused launch.
    // Each must be apart from every other range of the call; the reads (`scores`, `pos`) may
    // overlap each other.
    let written = [
        Span::of(&io.indices, indices),
        Span::of(&io.counts, counts),
        Span::of(&words.status, 4),
    ];
    let read = [Span::of(&io.scores, scores), Span::of(&words.pos, 4)];
    for (i, a) in written.iter().enumerate() {
        let apart = written[i + 1..]
            .iter()
            .chain(read.iter())
            .all(|b| a.apart(b));
        if !apart {
            return Err(invalid_argument());
        }
    }
    let ints = io.ints;
    let args = SparseTopkArgsRaw {
        scores: io.scores.address().cast_const(),
        indices: io.indices.address(),
        counts: io.counts.address(),
        pos: words.pos.address().cast_const(),
        status: words.status.address(),
        n_kv_head: ints.n_kv_head,
        cells: ints.cells,
        top_k: ints.top_k,
        sink: ints.sink,
        recent: ints.recent,
    };
    let mut stage = STAGE_NONE;
    // SAFETY: (a) the fit and apart checks above proved every range inside its live allocation,
    // large enough, 4-aligned, and the written `indices` and `counts` extents and `status` word
    // disjoint from every other range, so each pointer in `args` addresses bytes the kernel may
    // touch; the C entry repeats the checks on the addresses. (b) The allocations outlive the
    // queued work because `inference_cuda_free` synchronizes the device before `cudaFree`.
    // (c) `stream` is borrowed from a live `RawStream`. (d) `args` is a live local that the
    // entry reads only during the call. (e) The dispatch guard is held by `launch_on`, the only
    // caller.
    let status = unsafe { sparse_topk_launch_raw(&raw const args, stream.0.as_ptr(), &mut stage) };
    checked(status, stage)
}

#[cfg(test)]
mod tests {
    use super::SparseTopkArgsRaw;
    use std::mem::{align_of, offset_of, size_of};

    #[test]
    fn sparse_topk_argument_layout_pins_the_c_struct() {
        assert_eq!(size_of::<SparseTopkArgsRaw>(), 64);
        assert_eq!(offset_of!(SparseTopkArgsRaw, scores), 0);
        assert_eq!(offset_of!(SparseTopkArgsRaw, counts), 16);
        assert_eq!(offset_of!(SparseTopkArgsRaw, n_kv_head), 40);
        assert_eq!(align_of::<SparseTopkArgsRaw>(), 8);
    }
}
