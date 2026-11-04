//! Safe launch wrapper of the sparse-attention deterministic top-K: turns the relevance scores
//! of [`CudaContext::sparse_score`] into the selected index list of the sparse KV path, per KV
//! head — the `top_k` largest candidate scores ranked by value descending with ties broken by
//! ascending index (the radix-select rule of `sparse_reference.rs::deterministic_top_k`),
//! followed by the sink cells and the recent window, each ascending; `counts[head]` holds the
//! written row length `min(top_k, candidates) + sink + recent`.
//!
//! [`CudaContext::sparse_topk`] queues one kernel on a [`DeviceStream`] and returns; it
//! allocates nothing, copies nothing and never synchronizes.
//!
//! The decode entry's `pos` word (`StepParams.pos`) is authoritative: the selected kv length is
//! `*pos + 1`. A word outside `0..cells` - or a kv length that leaves no candidate
//! (`*pos + 1 <= sink + recent`) - is refused on the device: nothing is written and mask 4 is
//! OR-ed into the status word. The host takes the dense fallback instead of launching then.
//!
//! Scores are the softmax masses of `sparse_score` (finite, non-negative); the kernel ranks
//! their bit patterns, so a negative or NaN score would rank above +inf - garbage in, garbage
//! out, no mask. Every argument is checked here, before any native call, mirroring the
//! `valid_args` function of `sparse_topk.cu`, so the native entry cannot return
//! `INVALID_ARGUMENT` (which would latch the backend). Local rules carry the tag
//! `sparse_topk: <rule>`; size, alignment and overlap are refused by `require_fit` and
//! `require_apart` with their own texts.
use std::ffi::c_int;

use crate::EngError;
use crate::kernels::cuda::context::{
    Claim, CudaContext, DeviceRange, DeviceStream, require_apart, require_fit,
};
use crate::kernels::cuda::native::resident::sparse_topk::{
    SparseTopkDataRaw, SparseTopkInts, SparseTopkWordsRaw, sparse_topk_launch,
};

pub(crate) const MAX_N_KV_HEAD: usize = 64;
pub(crate) const MAX_CELLS: usize = (1 << 30) - 1;
pub(crate) const MAX_TOP_K: usize = 1 << 24;

/// Geometry of one top-K launch: `n_kv_head` heads, `cells` plane cells per head, `top_k`
/// selected candidates, `sink` leading and `recent` trailing forced cells.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SparseTopkShape {
    pub(crate) n_kv_head: usize,
    pub(crate) cells: usize,
    pub(crate) top_k: usize,
    pub(crate) sink: usize,
    pub(crate) recent: usize,
}

impl SparseTopkShape {
    pub(crate) fn score_bytes(&self) -> usize {
        self.n_kv_head.saturating_mul(self.cells).saturating_mul(4)
    }

    pub(crate) fn index_bytes(&self) -> usize {
        self.n_kv_head
            .saturating_mul(self.top_k + self.sink + self.recent)
            .saturating_mul(4)
    }

    pub(crate) fn count_bytes(&self) -> usize {
        self.n_kv_head.saturating_mul(4)
    }
}

/// The three data ranges of one top-K launch. All are device ranges of this context.
///
/// - `scores`: f32 `[n_kv_head][cells]`, 4-byte aligned (the `sparse_score` output).
/// - `indices`: u32 `[n_kv_head][top_k + sink + recent]`, 4-byte aligned.
/// - `counts`: u32 `[n_kv_head]`, 4-byte aligned.
pub(crate) struct SparseTopkData<'b> {
    pub(crate) scores: DeviceRange<'b>,
    pub(crate) indices: DeviceRange<'b>,
    pub(crate) counts: DeviceRange<'b>,
}

fn rule(detail: &str) -> EngError {
    EngError::BadRequest(format!("sparse_topk: {detail}"))
}

fn to_int(value: usize) -> Result<c_int, EngError> {
    c_int::try_from(value).map_err(|_| rule(&format!("an integer is {value}, above i32::MAX")))
}

fn check_shape(shape: &SparseTopkShape) -> Result<SparseTopkInts, EngError> {
    if !(1..=MAX_N_KV_HEAD).contains(&shape.n_kv_head) {
        return Err(rule(&format!(
            "n_kv_head is {}, not 1 to {MAX_N_KV_HEAD}",
            shape.n_kv_head
        )));
    }
    if !(1..=MAX_CELLS).contains(&shape.cells) {
        return Err(rule(&format!(
            "cells is {}, not 1 to {MAX_CELLS}",
            shape.cells
        )));
    }
    if !(1..=MAX_TOP_K).contains(&shape.top_k) {
        return Err(rule(&format!(
            "top_k is {}, not 1 to {MAX_TOP_K}",
            shape.top_k
        )));
    }
    if shape.sink + shape.recent >= shape.cells {
        return Err(rule(&format!(
            "sink {} + recent {} leaves no candidate of cells {}",
            shape.sink, shape.recent, shape.cells
        )));
    }
    Ok(SparseTopkInts {
        n_kv_head: to_int(shape.n_kv_head)?,
        cells: to_int(shape.cells)?,
        top_k: to_int(shape.top_k)?,
        sink: to_int(shape.sink)?,
        recent: to_int(shape.recent)?,
    })
}

impl CudaContext {
    /// Queues the sparse-attention deterministic top-K of `data` on `stream`: one block of 512
    /// threads per KV head selects at cells `*pos + 1` of the scores plane. Returns when the
    /// kernel is queued; `indices` and `counts` are valid after the stream drained.
    ///
    /// The device word `*pos` is authoritative (`kv_len = *pos + 1`). A word outside `0..cells`,
    /// or a kv length that leaves no candidate, makes the launch write nothing and OR mask 4
    /// into the status word.
    ///
    /// Returns `BadRequest`, before any native call and with the backend unchanged, for a buffer
    /// or stream of another context, a freed buffer, a shape the kernel refuses, a range that is
    /// too short or misaligned, and a written range that overlaps another range of the call.
    pub(crate) fn sparse_topk(
        &self,
        stream: &DeviceStream,
        shape: SparseTopkShape,
        data: SparseTopkData<'_>,
        pos: DeviceRange<'_>,
        status: DeviceRange<'_>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let ints = check_shape(&shape)?;
        let (scores, indices, counts) = (
            shape.score_bytes(),
            shape.index_bytes(),
            shape.count_bytes(),
        );
        require_fit("scores", &data.scores, scores, 4)?;
        require_fit("indices", &data.indices, indices, 4)?;
        require_fit("counts", &data.counts, counts, 4)?;
        require_fit("pos", &pos, 4, 4)?;
        require_fit("status", &status, 4, 4)?;
        // The written ranges (`indices`, `counts`, and `status` on a refused launch) must be
        // apart from every range of the call; the reads (`scores`, `pos`) may overlap each other.
        let written = [
            Claim::of("indices", &data.indices, indices),
            Claim::of("counts", &data.counts, counts),
            Claim::of("status", &status, 4),
        ];
        let read = [
            Claim::of("scores", &data.scores, scores),
            Claim::of("pos", &pos, 4),
        ];
        for (i, claim) in written.iter().enumerate() {
            for other in written[i + 1..].iter().chain(read.iter()) {
                require_apart(claim, other)?;
            }
        }
        let raw = SparseTopkDataRaw {
            scores: self.raw_range(&data.scores)?,
            indices: self.raw_range(&data.indices)?,
            counts: self.raw_range(&data.counts)?,
            ints,
        };
        let words = SparseTopkWordsRaw {
            pos: self.raw_range(&pos)?,
            status: self.raw_range(&status)?,
        };
        self.launch_on(stream, move |raw_stream| {
            sparse_topk_launch(raw_stream, raw, words)
        })
    }
}

#[cfg(all(test, feature = "native-cuda"))]
mod tests;
