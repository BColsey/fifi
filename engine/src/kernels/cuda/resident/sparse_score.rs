//! Safe launch wrapper of the sparse-attention relevance score: for every KV head, the softmax
//! mass over the candidate range `[sink, kv_len - recent)` summed over the head's query lanes,
//! from the F16 sketch plane and the basis (`scores f32 [n_kv_head][cells]`, candidates only;
//! the exact contract is in the module docs of `sparse_reference.rs` and
//! `native/resident/sparse_score.h`).
//!
//! [`CudaContext::sparse_score`] queues one kernel on a [`DeviceStream`] and returns; it
//! allocates nothing, copies nothing and never synchronizes; a fault in the queued work
//! surfaces at the next synchronize of the stream and latches the backend there.
//!
//! The decode entry's `pos` word (`StepParams.pos`) is authoritative: the scored kv length is
//! `*pos + 1`. A word outside `0..cells` is refused on the device exactly as `attn_prep` and
//! `kv8_append` refuse it: nothing is written and mask 4 is OR-ed into the status word. A kv
//! length that leaves no candidates (`*pos + 1 <= sink + recent`) writes nothing and raises no
//! mask; the caller takes the dense fallback.
//!
//! Every argument is checked here, before any native call: the rules mirror the `valid_args`
//! function of `sparse_score.cu`, so the native entry cannot return `INVALID_ARGUMENT` (which
//! would latch the backend). Local rules carry the tag `sparse_score: <rule>`; size, alignment
//! and overlap are refused by `require_fit` and `require_apart` with their own texts.
use std::ffi::c_int;

use crate::EngError;
use crate::kernels::cuda::context::{
    Claim, CudaContext, DeviceRange, DeviceStream, require_apart, require_fit,
};
use crate::kernels::cuda::native::resident::sparse_score::{
    SparseScoreDataRaw, SparseScoreInts, SparseScoreWordsRaw, sparse_score_launch,
};

/// The least head size (one warp holds the row in the sketch append).
pub(crate) const MIN_HEAD_DIM: usize = 32;
/// The most values per head (the register budget of the sketch append).
pub(crate) const MAX_HEAD_DIM: usize = 1024;
pub(crate) const MAX_N_KV_HEAD: usize = 64;
/// The most query lanes per KV head (the shared projection budget with the rank bound).
pub(crate) const MAX_GROUP: usize = 16;
/// The least projection outputs (a degenerate rank-1 sketch is refused).
pub(crate) const MIN_RANK: usize = 2;
/// The most (lanes per KV head) * rank values the shared projection holds: 16 KiB.
pub(crate) const MAX_PROJECTION: usize = 4096;
pub(crate) const MAX_CELLS: usize = (1 << 30) - 1;

/// Geometry of one scoring launch.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SparseScoreShape {
    pub(crate) n_q_head: usize,
    pub(crate) n_kv_head: usize,
    pub(crate) head_dim: usize,
    pub(crate) rank: usize,
    pub(crate) cells: usize,
    pub(crate) sink: usize,
    pub(crate) recent: usize,
    pub(crate) scale: f32,
}

impl SparseScoreShape {
    pub(crate) fn q_bytes(&self) -> usize {
        self.n_q_head
            .saturating_mul(self.head_dim)
            .saturating_mul(4)
    }

    pub(crate) fn basis_bytes(&self) -> usize {
        self.n_kv_head
            .saturating_mul(self.rank)
            .saturating_mul(self.head_dim)
            .saturating_mul(4)
    }

    pub(crate) fn sketch_bytes(&self) -> usize {
        self.n_kv_head
            .saturating_mul(self.cells)
            .saturating_mul(self.rank)
            .saturating_mul(2)
    }

    pub(crate) fn score_bytes(&self) -> usize {
        self.n_kv_head.saturating_mul(self.cells).saturating_mul(4)
    }
}

/// The four data ranges of one scoring launch. All are device ranges of this context.
///
/// - `q`: f32 `[n_q_head][head_dim]`, 4-byte aligned (post-norm, post-rope rows).
/// - `basis`: f32 `[n_kv_head][rank][head_dim]`, 4-byte aligned, output-major.
/// - `sketch`: fp16 `[n_kv_head][cells][rank]`, 4-byte aligned, `cell = position`.
/// - `scores`: f32 `[n_kv_head][cells]`, 4-byte aligned; candidate cells only are written.
pub(crate) struct SparseScoreData<'b> {
    pub(crate) q: DeviceRange<'b>,
    pub(crate) basis: DeviceRange<'b>,
    pub(crate) sketch: DeviceRange<'b>,
    pub(crate) scores: DeviceRange<'b>,
}

fn rule(tag: &str, detail: &str) -> EngError {
    EngError::BadRequest(format!("{tag}: {detail}"))
}

fn to_int(tag: &str, value: usize) -> Result<c_int, EngError> {
    c_int::try_from(value).map_err(|_| rule(tag, &format!("{tag} is {value}, above i32::MAX")))
}

/// Checks the geometry rules and converts the integers once, under `tag`, so that no conversion
/// can fail inside the launch closure.
fn check_shape(shape: &SparseScoreShape) -> Result<SparseScoreInts, EngError> {
    const TAG: &str = "sparse_score";
    let group = if shape.n_kv_head >= 1 {
        shape.n_q_head / shape.n_kv_head
    } else {
        0
    };
    if !(1..=MAX_GROUP).contains(&group) || group * shape.n_kv_head != shape.n_q_head {
        return Err(rule(
            TAG,
            &format!(
                "n_q_head {} over n_kv_head {} is not a group of 1 to {MAX_GROUP}",
                shape.n_q_head, shape.n_kv_head
            ),
        ));
    }
    if !(1..=MAX_N_KV_HEAD).contains(&shape.n_kv_head) {
        return Err(rule(
            TAG,
            &format!("n_kv_head is {}, not 1 to {MAX_N_KV_HEAD}", shape.n_kv_head),
        ));
    }
    if !shape.head_dim.is_multiple_of(MIN_HEAD_DIM)
        || !(MIN_HEAD_DIM..=MAX_HEAD_DIM).contains(&shape.head_dim)
    {
        return Err(rule(
            TAG,
            &format!(
                "head_dim is {}, not a multiple of {MIN_HEAD_DIM} in {MIN_HEAD_DIM} to {MAX_HEAD_DIM}",
                shape.head_dim
            ),
        ));
    }
    if !(MIN_RANK..=shape.head_dim).contains(&shape.rank) {
        return Err(rule(
            TAG,
            &format!(
                "rank is {}, not {MIN_RANK} to head_dim {}",
                shape.rank, shape.head_dim
            ),
        ));
    }
    if group * shape.rank > MAX_PROJECTION {
        return Err(rule(
            TAG,
            &format!(
                "group {group} * rank {} is above the shared projection budget {MAX_PROJECTION}",
                shape.rank
            ),
        ));
    }
    if shape.cells < 1 || shape.cells > MAX_CELLS {
        return Err(rule(
            TAG,
            &format!("cells is {}, not 1 to {MAX_CELLS}", shape.cells),
        ));
    }
    if shape.sink + shape.recent >= shape.cells {
        return Err(rule(
            TAG,
            &format!(
                "sink {} + recent {} leaves no candidate of cells {}",
                shape.sink, shape.recent, shape.cells
            ),
        ));
    }
    if !shape.scale.is_finite() || shape.scale <= 0.0 {
        return Err(rule(
            TAG,
            &format!(
                "scale is {scale}, not a positive finite value",
                scale = shape.scale
            ),
        ));
    }
    Ok(SparseScoreInts {
        n_q_head: to_int(TAG, shape.n_q_head)?,
        n_kv_head: to_int(TAG, shape.n_kv_head)?,
        head_dim: to_int(TAG, shape.head_dim)?,
        rank: to_int(TAG, shape.rank)?,
        cells: to_int(TAG, shape.cells)?,
        sink: to_int(TAG, shape.sink)?,
        recent: to_int(TAG, shape.recent)?,
        scale: shape.scale,
    })
}

impl CudaContext {
    /// Queues the sparse-attention relevance scoring of `data` on `stream`: one block of 256
    /// threads per KV head scores the candidate cells `[sink, *pos + 1 - recent)` of the sketch
    /// plane. Returns when the kernel is queued; `scores` is valid after the stream drained.
    ///
    /// The device word `*pos` is authoritative (`kv_len = *pos + 1`). A word outside `0..cells`
    /// makes the launch write nothing and OR mask 4 into the status word. A kv length that
    /// leaves no candidates writes nothing and raises no mask.
    ///
    /// Returns `BadRequest`, before any native call and with the backend unchanged, for a buffer
    /// or stream of another context, a freed buffer, a shape the kernel refuses, a range that is
    /// too short or misaligned, and a written range that overlaps another range of the call.
    pub(crate) fn sparse_score(
        &self,
        stream: &DeviceStream,
        shape: SparseScoreShape,
        data: SparseScoreData<'_>,
        pos: DeviceRange<'_>,
        status: DeviceRange<'_>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let ints = check_shape(&shape)?;
        let (q, basis, sketch, scores) = (
            shape.q_bytes(),
            shape.basis_bytes(),
            shape.sketch_bytes(),
            shape.score_bytes(),
        );
        require_fit("q", &data.q, q, 4)?;
        require_fit("basis", &data.basis, basis, 4)?;
        require_fit("sketch", &data.sketch, sketch, 4)?;
        require_fit("scores", &data.scores, scores, 4)?;
        require_fit("pos", &pos, 4, 4)?;
        require_fit("status", &status, 4, 4)?;
        // `scores` is written, so it must be apart from every other range of the call; `status`
        // is written on a refused launch, so it must be apart from everything as well. The three
        // reads and `pos` may overlap each other.
        let written = [
            Claim::of("scores", &data.scores, scores),
            Claim::of("status", &status, 4),
        ];
        let read = [
            Claim::of("q", &data.q, q),
            Claim::of("basis", &data.basis, basis),
            Claim::of("sketch", &data.sketch, sketch),
            Claim::of("pos", &pos, 4),
        ];
        for (i, claim) in written.iter().enumerate() {
            for other in written[i + 1..].iter().chain(read.iter()) {
                require_apart(claim, other)?;
            }
        }
        let raw = SparseScoreDataRaw {
            q: self.raw_range(&data.q)?,
            basis: self.raw_range(&data.basis)?,
            sketch: self.raw_range(&data.sketch)?,
            scores: self.raw_range(&data.scores)?,
            ints,
        };
        let words = SparseScoreWordsRaw {
            pos: self.raw_range(&pos)?,
            status: self.raw_range(&status)?,
        };
        self.launch_on(stream, move |raw_stream| {
            sparse_score_launch(raw_stream, raw, words)
        })
    }
}

#[cfg(all(test, feature = "native-cuda"))]
mod tests;
