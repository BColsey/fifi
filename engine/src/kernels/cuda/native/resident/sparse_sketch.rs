//! Raw FFI to the sparse-attention sketch append entries of
//! `native/resident/sparse_sketch.cu` (declared in `native/resident/sparse_sketch.h`). Child
//! module of `native/resident.rs`, so it inherits the `allow(unsafe_code)` of `native.rs`.
//!
//! Safety invariants this file relies on:
//! - The launch entries are called only through `CudaContext::launch_on`, on the thread that owns
//!   the context and under the dispatch guard. They only enqueue.
//! - Every range is a `RawRange`, proved inside a live allocation. Sizes, alignment and
//!   disjointness (`super::Span`) are proved again here, on the extents the kernel touches,
//!   before the call; the C entries repeat the alignment and shape checks on the addresses.
//! - The C argument structs are built in one `unsafe` block that cites those checks.
//! - A device allocation outlives the queued work because `inference_cuda_free` synchronizes the
//!   device before `cudaFree`.
//! - The `extern` declarations, the `repr(C)` mirrors and every raw pointer are private to this
//!   file. Nothing is re-exported.
#![cfg_attr(
    not(test),
    expect(dead_code, reason = "staged for the resident decode step")
)]

use std::ffi::{c_int, c_void};
use std::mem::{align_of, offset_of, size_of};

use super::{DeviceFailure, RawRange, RawStream, STAGE_NONE, Span, checked, invalid_argument};

/// Mirror of `inference_cuda_sparse_sketch_args` in `sparse_sketch.h`, field for field.
#[repr(C)]
struct SparseSketchArgsRaw {
    k: *const c_void,
    basis: *const c_void,
    sketch: *mut c_void,
    pos: *const c_void,
    status: *mut c_void,
    m: c_int,
    n_kv_head: c_int,
    head_dim: c_int,
    rank: c_int,
    cells: c_int,
}

/// Mirror of `inference_cuda_sparse_sketch_chunk_args` in `sparse_sketch.h`, field for field.
#[repr(C)]
struct SparseSketchChunkArgsRaw {
    k: *const c_void,
    basis: *const c_void,
    sketch: *mut c_void,
    pos0: c_int,
    t: c_int,
    n_kv_head: c_int,
    head_dim: c_int,
    rank: c_int,
    cells: c_int,
}

const _: () = assert!(
    size_of::<SparseSketchArgsRaw>() == 64
        && align_of::<SparseSketchArgsRaw>() == 8
        && offset_of!(SparseSketchArgsRaw, k) == 0
        && offset_of!(SparseSketchArgsRaw, basis) == 8
        && offset_of!(SparseSketchArgsRaw, sketch) == 16
        && offset_of!(SparseSketchArgsRaw, pos) == 24
        && offset_of!(SparseSketchArgsRaw, status) == 32
        && offset_of!(SparseSketchArgsRaw, m) == 40
);

const _: () = assert!(
    size_of::<SparseSketchChunkArgsRaw>() == 48
        && align_of::<SparseSketchChunkArgsRaw>() == 8
        && offset_of!(SparseSketchChunkArgsRaw, k) == 0
        && offset_of!(SparseSketchChunkArgsRaw, basis) == 8
        && offset_of!(SparseSketchChunkArgsRaw, sketch) == 16
        && offset_of!(SparseSketchChunkArgsRaw, pos0) == 24
        && offset_of!(SparseSketchChunkArgsRaw, t) == 28
);

unsafe extern "C" {
    #[link_name = "inference_cuda_sparse_sketch_launch"]
    fn sparse_sketch_launch_raw(
        args: *const SparseSketchArgsRaw,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
    #[link_name = "inference_cuda_sparse_sketch_chunk_launch"]
    fn sparse_sketch_chunk_launch_raw(
        args: *const SparseSketchChunkArgsRaw,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
}

/// Every integer argument of the launches, already bounded and converted by the wrapper before
/// `launch_on`, so no conversion can fail inside the launch closure.
#[derive(Debug, Clone, Copy)]
pub(in crate::kernels::cuda) struct SparseSketchInts {
    pub(in crate::kernels::cuda) rows: c_int,
    pub(in crate::kernels::cuda) n_kv_head: c_int,
    pub(in crate::kernels::cuda) head_dim: c_int,
    pub(in crate::kernels::cuda) rank: c_int,
    pub(in crate::kernels::cuda) cells: c_int,
}

pub(in crate::kernels::cuda) struct SparseSketchDataRaw<'a> {
    pub(in crate::kernels::cuda) k: RawRange<'a>,
    pub(in crate::kernels::cuda) basis: RawRange<'a>,
    pub(in crate::kernels::cuda) sketch: RawRange<'a>,
    pub(in crate::kernels::cuda) ints: SparseSketchInts,
}

/// The decode entry's device words: the position of token 0 (read) and the status word (mask 4
/// is OR-ed in on a refused launch).
pub(in crate::kernels::cuda) struct SparseSketchWordsRaw<'a> {
    pub(in crate::kernels::cuda) pos: RawRange<'a>,
    pub(in crate::kernels::cuda) status: RawRange<'a>,
}

struct Needs {
    rows: usize,
    basis: usize,
    sketch: usize,
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

fn needs(ints: &SparseSketchInts) -> Result<Needs, DeviceFailure> {
    let (rows, kv_heads, dim, rank) = (
        size(ints.rows)?,
        size(ints.n_kv_head)?,
        size(ints.head_dim)?,
        size(ints.rank)?,
    );
    let cells = size(ints.cells)?;
    Ok(Needs {
        rows: product(&[rows, kv_heads, dim, 4])?,
        basis: product(&[kv_heads, rank, dim, 4])?,
        sketch: product(&[kv_heads, cells, rank, 2])?,
    })
}

/// Alignment 4: the kernel reads f32 words and writes fp16 values in 2-byte cells at a 4-byte
/// aligned base (the plane layout keeps every cell 2-aligned; the allocation is 4-aligned).
fn fits(range: &RawRange<'_>, need: usize) -> bool {
    range.bytes >= need && range.offset.is_multiple_of(4)
}

fn check_data(io: &SparseSketchDataRaw<'_>) -> Result<(), DeviceFailure> {
    let need = needs(&io.ints)?;
    if !(fits(&io.k, need.rows) && fits(&io.basis, need.basis) && fits(&io.sketch, need.sketch)) {
        return Err(invalid_argument());
    }
    let written = Span::of(&io.sketch, need.sketch);
    let read = [Span::of(&io.k, need.rows), Span::of(&io.basis, need.basis)];
    if !read.iter().all(|b| written.apart(b)) {
        return Err(invalid_argument());
    }
    Ok(())
}

fn check_words(
    io: &SparseSketchDataRaw<'_>,
    words: &SparseSketchWordsRaw<'_>,
) -> Result<(), DeviceFailure> {
    if !(fits(&words.pos, 4) && fits(&words.status, 4)) {
        return Err(invalid_argument());
    }
    let need = needs(&io.ints)?;
    let status = Span::of(&words.status, 4);
    let apart = [
        Span::of(&io.sketch, need.sketch),
        Span::of(&io.k, need.rows),
        Span::of(&io.basis, need.basis),
        Span::of(&words.pos, 4),
    ]
    .iter()
    .all(|a| status.apart(a));
    if !apart {
        return Err(invalid_argument());
    }
    Ok(())
}

pub(in crate::kernels::cuda) fn sparse_sketch_launch(
    stream: &RawStream,
    io: SparseSketchDataRaw<'_>,
    words: SparseSketchWordsRaw<'_>,
) -> Result<(), DeviceFailure> {
    check_data(&io)?;
    check_words(&io, &words)?;
    let ints = io.ints;
    let args = SparseSketchArgsRaw {
        k: io.k.address().cast_const(),
        basis: io.basis.address().cast_const(),
        sketch: io.sketch.address(),
        pos: words.pos.address().cast_const(),
        status: words.status.address(),
        m: ints.rows,
        n_kv_head: ints.n_kv_head,
        head_dim: ints.head_dim,
        rank: ints.rank,
        cells: ints.cells,
    };
    let mut stage = STAGE_NONE;
    // SAFETY: (a) `check_data` and `check_words` proved every range inside its live allocation,
    // large enough, 4-aligned, and the written sketch plane and `status` disjoint from every
    // other range, so each pointer in `args` addresses bytes the kernel may touch; the C entry
    // repeats the checks on the addresses. (b) The allocations outlive the queued work because
    // `inference_cuda_free` synchronizes the device before `cudaFree`. (c) `stream` is borrowed
    // from a live `RawStream`. (d) `args` is a live local that the entry reads only during the
    // call. (e) The dispatch guard is held by `launch_on`, which is the only caller.
    let status =
        unsafe { sparse_sketch_launch_raw(&raw const args, stream.0.as_ptr(), &mut stage) };
    checked(status, stage)
}

pub(in crate::kernels::cuda) fn sparse_sketch_chunk_launch(
    stream: &RawStream,
    io: SparseSketchDataRaw<'_>,
    pos0: c_int,
) -> Result<(), DeviceFailure> {
    check_data(&io)?;
    let ints = io.ints;
    let args = SparseSketchChunkArgsRaw {
        k: io.k.address().cast_const(),
        basis: io.basis.address().cast_const(),
        sketch: io.sketch.address(),
        pos0,
        t: ints.rows,
        n_kv_head: ints.n_kv_head,
        head_dim: ints.head_dim,
        rank: ints.rank,
        cells: ints.cells,
    };
    let mut stage = STAGE_NONE;
    // SAFETY: as `sparse_sketch_launch`; `check_data` proved the three data ranges, and the C
    // entry re-checks the addresses plus `pos0 + t <= cells` before any launch.
    let status =
        unsafe { sparse_sketch_chunk_launch_raw(&raw const args, stream.0.as_ptr(), &mut stage) };
    checked(status, stage)
}

#[cfg(test)]
mod tests {
    use super::{SparseSketchArgsRaw, SparseSketchChunkArgsRaw};
    use std::mem::{align_of, offset_of, size_of};

    #[test]
    fn sparse_sketch_argument_layouts_pin_the_c_structs() {
        assert_eq!(size_of::<SparseSketchArgsRaw>(), 64);
        assert_eq!(offset_of!(SparseSketchArgsRaw, m), 40);
        assert_eq!(align_of::<SparseSketchArgsRaw>(), 8);
        assert_eq!(size_of::<SparseSketchChunkArgsRaw>(), 48);
        assert_eq!(offset_of!(SparseSketchChunkArgsRaw, pos0), 24);
        assert_eq!(offset_of!(SparseSketchChunkArgsRaw, t), 28);
        assert_eq!(align_of::<SparseSketchChunkArgsRaw>(), 8);
    }
}
