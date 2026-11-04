//! Safe launch wrapper of the sparse-attention decode: attention over the SELECTED cells
//! only — the union of the top-K picks, the sink cells and the recent window written by
//! [`CudaContext::sparse_topk`] — with the exact Q8 arithmetic of the production decode
//! (`attn_decode_q8.cu`): int8 codes with one fp16 scale per 32 values, dequantized in
//! registers as `f32(scale) * code`, q pre-scaled once, f32 `fmaf` dots in the production
//! fixed order, 64-cell tiles under an online softmax. Sparsity changes which tokens are
//! attended, never how Q8 attention computes over them.
//!
//! [`CudaContext::sparse_decode`] queues one kernel — a single block of 256 threads per KV
//! head, one warp per query head (the production GQA mapping `out[kvh * G + g]`) — and
//! returns; it allocates nothing, copies nothing and never synchronizes. `out` is normalized
//! per head (`acc / l`); a zero count writes zeros, not NaN.
//!
//! The selection is trusted as a SET: `sparse_topk` guarantees the picked range is disjoint
//! from sinks and the recent window, so no token contributes twice. The device words decide
//! the rest, under the poison rule: this head's `counts[head]` and every `indices[head][i]`
//! are validated against `row` and `cells` BEFORE any write (block-uniformly — a divergent
//! early return in front of a barrier is undefined); a violation writes nothing except mask 4
//! OR-ed into the status word. `count > row` is refused, never truncated. Range validity of
//! the physical cell is all this proves; logical-position identity (paging, prefix restore)
//! is owned by the selection and pinned by its tests.
//!
//! Every argument is checked here, before any native call, mirroring the `valid_args`
//! function of `sparse_decode.cu`, so the native entry cannot return `INVALID_ARGUMENT`
//! (which would latch the backend). Local rules carry the tag `sparse_decode: <rule>`; size,
//! alignment and overlap are refused by `require_fit` and `require_apart` with their own
//! texts.
use std::ffi::c_int;

use super::sparse_topk::{MAX_CELLS, MAX_N_KV_HEAD, MAX_TOP_K};
use crate::EngError;
use crate::kernels::cuda::context::{
    Claim, CudaContext, DeviceRange, DeviceStream, require_apart, require_fit,
};
use crate::kernels::cuda::native::resident::sparse_decode::{
    SparseDecodeDataRaw, SparseDecodeInts, sparse_decode_launch,
};

/// Geometry of one sparse-decode launch: `n_kv_head` KV heads, `n_head` query heads
/// (`n_head / n_kv_head` per KV head, 1 to 8), `head_dim` 128 or 256, `cells` plane cells per
/// head, `row` indices capacity per head (`top_k + sink + recent` of the selection), and the
/// production attention `scale`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SparseDecodeShape {
    pub(crate) n_kv_head: usize,
    pub(crate) n_head: usize,
    pub(crate) head_dim: usize,
    pub(crate) cells: usize,
    pub(crate) row: usize,
    pub(crate) scale: f32,
}

impl SparseDecodeShape {
    pub(crate) fn code_bytes(&self) -> usize {
        self.n_kv_head
            .saturating_mul(self.cells)
            .saturating_mul(self.head_dim)
    }

    pub(crate) fn scale_bytes(&self) -> usize {
        self.n_kv_head
            .saturating_mul(self.cells)
            .saturating_mul(self.head_dim / 32)
            .saturating_mul(2)
    }

    pub(crate) fn index_bytes(&self) -> usize {
        self.n_kv_head.saturating_mul(self.row).saturating_mul(4)
    }

    pub(crate) fn q_out_bytes(&self) -> usize {
        self.n_head.saturating_mul(self.head_dim).saturating_mul(4)
    }

    pub(crate) fn count_bytes(&self) -> usize {
        self.n_kv_head.saturating_mul(4)
    }
}

/// The nine data ranges of one sparse-decode launch. All are device ranges of this context.
///
/// - `q`: f32 `[n_head][head_dim]`, one decode token, unscaled (`scale` applies inside),
///   16-byte aligned.
/// - `k_codes`/`v_codes`: int8 `[n_kv_head][cells][head_dim]`, 4-byte aligned.
/// - `k_scales`/`v_scales`: fp16 bits u16 `[n_kv_head][cells][head_dim / 32]`, 2-byte aligned.
/// - `indices`: u32 `[n_kv_head][row]` and `counts`: u32 `[n_kv_head]` (the `sparse_topk`
///   output), 4-byte aligned.
/// - `out`: f32 `[n_head * head_dim]`, 16-byte aligned (written).
/// - `status`: one i32 word (mask 4 is OR-ed in on a refused launch).
pub(crate) struct SparseDecodeData<'b> {
    pub(crate) q: DeviceRange<'b>,
    pub(crate) k_codes: DeviceRange<'b>,
    pub(crate) k_scales: DeviceRange<'b>,
    pub(crate) v_codes: DeviceRange<'b>,
    pub(crate) v_scales: DeviceRange<'b>,
    pub(crate) indices: DeviceRange<'b>,
    pub(crate) counts: DeviceRange<'b>,
    pub(crate) out: DeviceRange<'b>,
    /// Optional sigmoid gate `[n_head][2 * head_dim]` f32 (the production `proj` view);
    /// `None` leaves the output ungated.
    pub(crate) proj: Option<DeviceRange<'b>>,
    pub(crate) status: DeviceRange<'b>,
}

fn rule(detail: &str) -> EngError {
    EngError::BadRequest(format!("sparse_decode: {detail}"))
}

/// Post-synchronize interpretation of the status word. Bit 2 (mask 4,
/// `INFERENCE_CUDA_STATUS_INVALID_STEP_STATE`) means at least one head refused
/// its selection — per-head refusal — and the whole result is unusable: the
/// caller takes the dense Q8 fallback. A zero word is the only good outcome;
/// counts of zero under a zero word are the legitimate empty selection (zeros
/// out). Reading `out` or `counts` without this check conflates an invalid
/// selection with an empty one and silently substitutes an empty attention set
/// for an invalid one.
pub(crate) fn status_refused(word: u32) -> bool {
    word & 4 != 0
}

fn to_int(value: usize) -> Result<c_int, EngError> {
    c_int::try_from(value).map_err(|_| rule(&format!("an integer is {value}, above i32::MAX")))
}

fn check_shape(shape: &SparseDecodeShape) -> Result<SparseDecodeInts, EngError> {
    if !(1..=MAX_N_KV_HEAD).contains(&shape.n_kv_head) {
        return Err(rule(&format!(
            "n_kv_head is {}, not 1 to {MAX_N_KV_HEAD}",
            shape.n_kv_head
        )));
    }
    if !shape.n_head.is_multiple_of(shape.n_kv_head) {
        return Err(rule(&format!(
            "n_head {} is not a multiple of n_kv_head {}",
            shape.n_head, shape.n_kv_head
        )));
    }
    let per_head = shape.n_head / shape.n_kv_head;
    if !(1..=8).contains(&per_head) {
        return Err(rule(&format!(
            "n_head / n_kv_head is {per_head}, not 1 to 8"
        )));
    }
    if shape.head_dim != 128 && shape.head_dim != 256 {
        return Err(rule(&format!(
            "head_dim is {}, not 128 or 256",
            shape.head_dim
        )));
    }
    if !(1..=MAX_CELLS).contains(&shape.cells) {
        return Err(rule(&format!(
            "cells is {}, not 1 to {MAX_CELLS}",
            shape.cells
        )));
    }
    if !(1..=MAX_TOP_K).contains(&shape.row) {
        return Err(rule(&format!("row is {}, not 1 to {MAX_TOP_K}", shape.row)));
    }
    Ok(SparseDecodeInts {
        n_kv_head: to_int(shape.n_kv_head)?,
        n_head: to_int(shape.n_head)?,
        head_dim: to_int(shape.head_dim)?,
        cells: to_int(shape.cells)?,
        row: to_int(shape.row)?,
        scale: shape.scale,
    })
}

impl CudaContext {
    /// Queues the sparse-attention decode of `data` on `stream`: one block of 256 threads per
    /// KV head attends over this head's selected indices (picks ascending, then sinks, then
    /// the recent window) at length `counts[head]`. Returns when the kernel is queued; `out`
    /// is valid after the stream drained.
    ///
    /// The device words are authoritative: a count above `row`, or any index at or above
    /// `cells`, makes the launch write nothing except mask 4 OR-ed into the status word (the
    /// host takes the dense fallback instead of launching then).
    ///
    /// Returns `BadRequest`, before any native call and with the backend unchanged, for a
    /// buffer or stream of another context, a freed buffer, a shape the kernel refuses, a
    /// range that is too short or misaligned, and a written range that overlaps another range
    /// of the call.
    pub(crate) fn sparse_decode(
        &self,
        stream: &DeviceStream,
        shape: SparseDecodeShape,
        data: SparseDecodeData<'_>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let ints = check_shape(&shape)?;
        let (codes, scales, indices, q_out, counts) = (
            shape.code_bytes(),
            shape.scale_bytes(),
            shape.index_bytes(),
            shape.q_out_bytes(),
            shape.count_bytes(),
        );
        require_fit("q", &data.q, q_out, 16)?;
        require_fit("k_codes", &data.k_codes, codes, 4)?;
        require_fit("k_scales", &data.k_scales, scales, 2)?;
        require_fit("v_codes", &data.v_codes, codes, 4)?;
        require_fit("v_scales", &data.v_scales, scales, 2)?;
        require_fit("indices", &data.indices, indices, 4)?;
        require_fit("counts", &data.counts, counts, 4)?;
        require_fit("out", &data.out, q_out, 16)?;
        require_fit("status", &data.status, 4, 4)?;
        let gate = shape.n_head * 2 * shape.head_dim * 4;
        if let Some(proj) = data.proj.as_ref() {
            require_fit("proj", proj, gate, 4)?;
        }
        // The written ranges (`out`, and `status` on a refused launch) must be apart from
        // every range of the call; the reads may overlap each other.
        let written = [
            Claim::of("out", &data.out, q_out),
            Claim::of("status", &data.status, 4),
        ];
        let mut read = vec![
            Claim::of("q", &data.q, q_out),
            Claim::of("k_codes", &data.k_codes, codes),
            Claim::of("k_scales", &data.k_scales, scales),
            Claim::of("v_codes", &data.v_codes, codes),
            Claim::of("v_scales", &data.v_scales, scales),
            Claim::of("indices", &data.indices, indices),
            Claim::of("counts", &data.counts, counts),
        ];
        if let Some(proj) = data.proj.as_ref() {
            read.push(Claim::of("proj", proj, gate));
        }
        for (i, claim) in written.iter().enumerate() {
            for other in written[i + 1..].iter().chain(read.iter()) {
                require_apart(claim, other)?;
            }
        }
        let raw = SparseDecodeDataRaw {
            q: self.raw_range(&data.q)?,
            k_codes: self.raw_range(&data.k_codes)?,
            k_scales: self.raw_range(&data.k_scales)?,
            v_codes: self.raw_range(&data.v_codes)?,
            v_scales: self.raw_range(&data.v_scales)?,
            indices: self.raw_range(&data.indices)?,
            counts: self.raw_range(&data.counts)?,
            out: self.raw_range(&data.out)?,
            proj: data.proj.as_ref().map(|p| self.raw_range(p)).transpose()?,
            status: self.raw_range(&data.status)?,
            ints,
        };
        self.launch_on(stream, move |raw_stream| {
            sparse_decode_launch(raw_stream, raw)
        })
    }
}

#[cfg(all(test, feature = "native-cuda"))]
mod tests;

#[cfg(test)]
mod status_tests {
    use super::status_refused;

    #[test]
    fn only_bit_2_means_refused() {
        assert!(!status_refused(0), "a zero word is the only good outcome");
        assert!(status_refused(4), "mask 4 alone is a refusal");
        assert!(status_refused(5), "mask 4 sticks with other bits");
        assert!(!status_refused(1), "other status bits are not refusals");
    }
}
