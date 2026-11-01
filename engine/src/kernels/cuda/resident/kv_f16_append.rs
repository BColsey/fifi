//! Safe launch wrapper of the standalone fp16 KV append: converts the gathered, post-rope
//! `k` rows and post-norm `v` rows of one step to the fp16 caches with the same
//! `__float2half` and the same plane layout the fused single-device append inside
//! `attn_prep` writes (`fp16 [cells][n_kv_head][head_dim]`, row `*pos + t`), so the
//! multi-device card path - whose rows arrive through the peer-ring gathers after the prep
//! kernel already ran - produces bit-identical caches.
//!
//! [`CudaContext::kv_f16_append`] queues one kernel on a [`DeviceStream`] and returns; it
//! allocates nothing, copies nothing and never synchronizes; a fault in the queued work
//! surfaces at the next synchronize of the stream and latches the backend there.
//!
//! The device `pos` word (`StepParams.pos`) is authoritative: token `t` goes to row
//! `*pos + t`. A word outside `0..=cells - m` is refused on the device exactly as
//! `attn_prep` refuses it: no cache byte is written and mask 4 is OR-ed into the status
//! word.
//!
//! Every argument is checked here, before any native call: the rules mirror the
//! `valid_args` of `kv_f16_append.cu`, so the native entry cannot return
//! `INVALID_ARGUMENT` (which would latch the backend). Local rules carry the tag
//! `kv_f16_append: <rule>`; size, alignment and overlap are refused by `require_fit` and
//! `require_apart` with their own texts.

// Staged with the multi-device card step: the wrapper's first production
// caller is the card driver; until then only the device rigs call it,
// which the lib target does not see.
#![cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "staged for the multi-device card step, its first production caller"
    )
)]

use std::ffi::c_int;

use crate::EngError;
use crate::kernels::cuda::context::{
    Claim, CudaContext, DeviceRange, DeviceStream, require_apart, require_fit,
};
use crate::kernels::cuda::native::resident::kv_f16_append::{
    KvF16DataRaw, KvF16Ints, kv_f16_append_launch,
};

/// The fused kernel's only head size (`attn_prep`'s supported configuration).
pub(crate) const HEAD_DIM: usize = 256;
/// The most kv heads.
pub(crate) const MAX_N_KV_HEAD: usize = 64;
/// The most rows of the entry (the decode look-ahead; `kv8_append`'s bound).
pub(crate) const MAX_DECODE_ROWS: usize = 16;
/// The most rows per cache: `2^30 - 1`.
pub(crate) const MAX_CELLS: usize = (1 << 30) - 1;

/// Geometry of one append: `rows` tokens, `n_kv_head` kv heads, `head_dim` values per head
/// (256, the fused kernel's only size), and `cells` cache rows (the KV capacity).
#[derive(Debug, Clone, Copy)]
pub(crate) struct KvF16AppendShape {
    pub(crate) rows: usize,
    pub(crate) n_kv_head: usize,
    pub(crate) head_dim: usize,
    pub(crate) cells: usize,
}

impl KvF16AppendShape {
    /// Bytes of one f32 row block: `rows * n_kv_head * head_dim` values.
    pub(crate) fn row_bytes(&self) -> usize {
        self.rows
            .saturating_mul(self.n_kv_head)
            .saturating_mul(self.head_dim)
            .saturating_mul(4)
    }

    /// Bytes of one fp16 cache: `cells * n_kv_head * head_dim` half-words.
    pub(crate) fn cache_bytes(&self) -> usize {
        self.cells
            .saturating_mul(self.n_kv_head)
            .saturating_mul(self.head_dim)
            .saturating_mul(2)
    }
}

/// The four data ranges of one append. All are device ranges of this context.
///
/// - `k`, `v`: f32 `[rows][n_kv_head][head_dim]`, 4-byte aligned (k roped, v normed).
/// - `k_cache`, `v_cache`: fp16 `[cells][n_kv_head][head_dim]`, 16-byte aligned.
pub(crate) struct KvF16AppendData<'b> {
    pub(crate) k: DeviceRange<'b>,
    pub(crate) v: DeviceRange<'b>,
    pub(crate) k_cache: DeviceRange<'b>,
    pub(crate) v_cache: DeviceRange<'b>,
}

/// A `BadRequest` that names the rule that fired, under `tag`.
fn rule(tag: &str, detail: &str) -> EngError {
    EngError::BadRequest(format!("{tag}: {detail}"))
}

/// `value` as `c_int`, or the "exceeds i32" refusal naming `tag`.
fn to_int(tag: &str, value: usize) -> Result<c_int, EngError> {
    c_int::try_from(value).map_err(|_| rule(tag, &format!("{tag} is {value}, above i32::MAX")))
}

/// Checks the geometry rules and converts the integers once, so that no conversion can fail
/// inside the launch closure.
fn check_shape(shape: &KvF16AppendShape) -> Result<KvF16Ints, EngError> {
    let tag = "kv_f16_append";
    if !(1..=MAX_DECODE_ROWS).contains(&shape.rows) {
        return Err(rule(
            tag,
            &format!("rows is {}, not 1 to {MAX_DECODE_ROWS}", shape.rows),
        ));
    }
    if !(1..=MAX_N_KV_HEAD).contains(&shape.n_kv_head) {
        return Err(rule(
            tag,
            &format!("n_kv_head is {}, not 1 to {MAX_N_KV_HEAD}", shape.n_kv_head),
        ));
    }
    if shape.head_dim != HEAD_DIM {
        return Err(rule(
            tag,
            &format!(
                "head_dim is {}, not {HEAD_DIM} (the fused kernel's only size)",
                shape.head_dim
            ),
        ));
    }
    if shape.cells < shape.rows || shape.cells > MAX_CELLS {
        return Err(rule(
            tag,
            &format!(
                "cells is {}, not {rows} to {MAX_CELLS}",
                shape.cells,
                rows = shape.rows
            ),
        ));
    }
    Ok(KvF16Ints {
        rows: to_int(tag, shape.rows)?,
        n_kv_head: to_int(tag, shape.n_kv_head)?,
        head_dim: to_int(tag, shape.head_dim)?,
        cells: to_int(tag, shape.cells)?,
    })
}

impl CudaContext {
    /// Queues the fp16 append of `data` on `stream`: `shape.rows` tokens (1 to 16) of k and
    /// v, converted to the two caches at rows `*pos .. *pos + rows`. Returns when the kernel
    /// is queued; the caches are valid after the stream drained.
    ///
    /// The device word `*pos` is authoritative. A word outside `0..=cells - rows` makes the
    /// launch write nothing and OR mask 4 into the status word (`StepParams.status`).
    ///
    /// Returns `BadRequest`, before any native call and with the backend unchanged, for a
    /// buffer or stream of another context, a freed buffer, a shape the kernel refuses, a
    /// range that is too short or misaligned, and a written range that overlaps another range
    /// of the call.
    pub(crate) fn kv_f16_append(
        &self,
        stream: &DeviceStream,
        shape: KvF16AppendShape,
        data: KvF16AppendData<'_>,
        pos: DeviceRange<'_>,
        status: DeviceRange<'_>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let ints = check_shape(&shape)?;
        let (rows, cache) = (shape.row_bytes(), shape.cache_bytes());
        require_fit("k", &data.k, rows, 4)?;
        require_fit("v", &data.v, rows, 4)?;
        require_fit("k_cache", &data.k_cache, cache, 16)?;
        require_fit("v_cache", &data.v_cache, cache, 16)?;
        require_fit("pos", &pos, 4, 4)?;
        require_fit("status", &status, 4, 4)?;
        // The three written ranges (both caches, the status word) pairwise and against the
        // three read ranges; the read rows and the words may alias each other.
        let written = [
            Claim::of("k_cache", &data.k_cache, cache),
            Claim::of("v_cache", &data.v_cache, cache),
            Claim::of("status", &status, 4),
        ];
        let read = [
            Claim::of("k", &data.k, rows),
            Claim::of("v", &data.v, rows),
            Claim::of("pos", &pos, 4),
        ];
        for (i, claim) in written.iter().enumerate() {
            for other in written[i + 1..].iter().chain(read.iter()) {
                require_apart(claim, other)?;
            }
        }
        let raw = KvF16DataRaw {
            k: self.raw_range(&data.k)?,
            v: self.raw_range(&data.v)?,
            k_cache: self.raw_range(&data.k_cache)?,
            v_cache: self.raw_range(&data.v_cache)?,
            pos: self.raw_range(&pos)?,
            status: self.raw_range(&status)?,
            ints,
        };
        self.launch_on(stream, move |raw_stream| {
            kv_f16_append_launch(raw_stream, raw)
        })
    }
}
