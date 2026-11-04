//! The sparse attention of Qwen3.8 Flash-Next QSA on the device. The formula and layouts are in
//! `native/resident/qsa_attn.h`; they follow the host `attend_selected` of
//! `family::flash_next::qsa`.
//!
//! [`CudaContext::qsa_attn`] queues, for each of `m` consecutive tokens (one in decode, a chunk in
//! prefill), the gated attention of every query head over the positions its selection names: the
//! `ratio` positions of each chosen block (`qsa_select`'s output) and the open tail. q comes from `attn_prep` (normed, rotated, not
//! scaled), the keys and values from its fp16 cache, the gates from the projection row. The
//! attended positions are split over `max_splits` blocks per kv head; `part` is their scratch.
//!
//! The call allocates nothing, copies nothing and never synchronizes; a fault in the queued work
//! surfaces at the next synchronize of the stream. The output is valid only to work queued later
//! on the same stream, or after it is synchronized.
//!
//! The position (of the first token) follows [`QsaPosition`]: a host value is checked here
//! (`position + m <= rows`), a device word is checked on the device. A word outside the cache
//! makes every output NaN, a chosen block that is not a completed block makes its token's output
//! NaN, and either raises status bits 4; no cache row past a token's position is read for it.
//!
//! Every refusal is a `BadRequest` and happens before any native call, because a status from the
//! native layer other than an allocation failure latches the backend terminal. A refused call
//! changes nothing.
#![cfg_attr(
    not(test),
    expect(dead_code, reason = "staged for the Flash-Next CUDA step")
)]

use std::ffi::c_int;

use crate::EngError;
use crate::kernels::cuda::context::{Claim, require_fit};
use crate::kernels::cuda::native::resident::qsa_attn::{
    QSA_ATTN_MAX_BUDGET, QSA_ATTN_MAX_GROUP, QSA_ATTN_MAX_KV_HEADS, QSA_ATTN_MAX_RATIO,
    QSA_ATTN_MAX_ROWS, QSA_ATTN_MAX_SPLITS, QsaAttnBytes, QsaAttnDims, QsaAttnRaw, qsa_attn_launch,
};
use crate::kernels::cuda::resident::qsa_index::{
    QsaPosition, check_tokens, position_word, require_all_apart,
};
use crate::kernels::cuda::{CudaContext, DeviceRange, DeviceStream};

/// The geometry of one layer's call: `m` tokens, `n_head` query heads over `n_kv_head` kv heads
/// of 256 values, a cache of `rows` positions, blocks of `ratio` positions, at most
/// `budget_blocks` chosen blocks per token, and `max_splits` splits of each token's attended
/// positions per kv head.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct QsaAttnShape {
    pub(crate) m: usize,
    pub(crate) n_head: usize,
    pub(crate) n_kv_head: usize,
    pub(crate) rows: usize,
    pub(crate) ratio: usize,
    pub(crate) budget_blocks: usize,
    pub(crate) max_splits: usize,
}

impl QsaAttnShape {
    fn dims(self) -> QsaAttnDims {
        QsaAttnDims {
            m: self.m,
            n_head: self.n_head,
            n_kv_head: self.n_kv_head,
            rows: self.rows,
            ratio: self.ratio,
            budget_blocks: self.budget_blocks,
            max_splits: self.max_splits,
        }
    }

    /// The scratch `part` of a call, in bytes.
    pub(crate) fn part_bytes(self) -> Result<usize, EngError> {
        Ok(check_shape(self)?.part)
    }
}

/// One call: `q` f32 `[m][n_head][256]`, `proj` the f32 projection rows `[m][ld]` (head `h`'s
/// gate at column `h * 512 + 256`), the fp16 caches `[rows][n_kv_head][256]`, `chosen` i32
/// `[m][budget_blocks]`; `part` is scratch and `out` f32 `[m][n_head][256]` is written.
pub(crate) struct QsaAttnCall<'b> {
    pub(crate) shape: QsaAttnShape,
    pub(crate) ld: usize,
    pub(crate) scale: f32,
    pub(crate) position: QsaPosition<'b>,
    pub(crate) q: DeviceRange<'b>,
    pub(crate) proj: DeviceRange<'b>,
    pub(crate) k_cache: DeviceRange<'b>,
    pub(crate) v_cache: DeviceRange<'b>,
    pub(crate) chosen: DeviceRange<'b>,
    pub(crate) part: DeviceRange<'b>,
    pub(crate) out: DeviceRange<'b>,
    pub(crate) status: Option<DeviceRange<'b>>,
}

fn check_shape(shape: QsaAttnShape) -> Result<QsaAttnBytes, EngError> {
    check_tokens(shape.m)?;
    shape.dims().bytes().ok_or_else(|| {
        EngError::BadRequest(format!(
            "the CUDA QSA attention takes 1 to {QSA_ATTN_MAX_KV_HEADS} kv heads, at most \
             {QSA_ATTN_MAX_GROUP} query heads per kv head, 1 to {QSA_ATTN_MAX_ROWS} cache rows, \
             a block ratio of 1 to {QSA_ATTN_MAX_RATIO}, a budget of 1 to {QSA_ATTN_MAX_BUDGET} \
             blocks and 1 to {QSA_ATTN_MAX_SPLITS} splits, got {shape:?}"
        ))
    })
}

/// The host value of the position (refused unless every token's position is a cached row) and
/// the launch value.
fn admit(shape: QsaAttnShape, position: &QsaPosition<'_>) -> Result<c_int, EngError> {
    let (value, pos0) = match position {
        QsaPosition::Host(value) => (*value, *value),
        QsaPosition::Device { admit, .. } => (*admit, 0),
    };
    if value
        .checked_add(shape.m)
        .is_none_or(|end| end > shape.rows)
    {
        return Err(EngError::BadRequest(format!(
            "the CUDA QSA attention position {value} with {} tokens is outside the cache of {} \
             rows",
            shape.m, shape.rows
        )));
    }
    // Below rows, so below 2^30.
    c_int::try_from(pos0).map_err(|_| EngError::BadRequest("the QSA position overflows".into()))
}

impl CudaContext {
    /// Queues the QSA attention of `call.shape.m` tokens on `stream`.
    ///
    /// Returns `BadRequest` for a stream or buffer of another adapter, a geometry or token count
    /// outside the limits, an `ld` below `n_head * 512` or past i32, a non-finite `scale`, a
    /// position whose tokens do not fit the cache, a short or misaligned range (16 bytes for q and the caches, 4 for the rest),
    /// and a written range that overlaps another range.
    pub(crate) fn qsa_attn(
        &self,
        stream: &DeviceStream,
        call: QsaAttnCall<'_>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let raw = self.plan_qsa_attn(&call)?;
        self.launch_on(stream, move |stream_raw| qsa_attn_launch(stream_raw, raw))
    }

    fn plan_qsa_attn<'b>(&self, call: &QsaAttnCall<'b>) -> Result<QsaAttnRaw<'b>, EngError> {
        let shape = call.shape;
        let need = check_shape(shape)?;
        let proj = shape.dims().proj_bytes(call.ld).ok_or_else(|| {
            EngError::BadRequest(format!(
                "the CUDA QSA attention row of {} floats cannot hold {} heads of query and gate, \
                 or exceeds i32",
                call.ld, shape.n_head
            ))
        })?;
        if !call.scale.is_finite() {
            return Err(EngError::BadRequest(format!(
                "the CUDA QSA attention scale must be finite, got {}",
                call.scale
            )));
        }
        let pos0 = admit(shape, &call.position)?;
        let word = position_word(&call.position)?;
        require_fit("q", &call.q, need.q, 16)?;
        require_fit("proj", &call.proj, proj, 4)?;
        require_fit("k_cache", &call.k_cache, need.cache, 16)?;
        require_fit("v_cache", &call.v_cache, need.cache, 16)?;
        require_fit("chosen", &call.chosen, need.chosen, 4)?;
        require_fit("part", &call.part, need.part, 4)?;
        require_fit("out", &call.out, need.out, 4)?;
        if let Some(status) = &call.status {
            require_fit("status", status, 4, 4)?;
        }
        let mut written = vec![
            Claim::of("part", &call.part, need.part),
            Claim::of("out", &call.out, need.out),
        ];
        written.extend(call.status.as_ref().map(|s| Claim::of("status", s, 4)));
        let mut read = vec![
            Claim::of("q", &call.q, need.q),
            Claim::of("proj", &call.proj, proj),
            Claim::of("k_cache", &call.k_cache, need.cache),
            Claim::of("v_cache", &call.v_cache, need.cache),
            Claim::of("chosen", &call.chosen, need.chosen),
        ];
        read.extend(word.map(|w| Claim::of("pos", w, 4)));
        require_all_apart(&written, &read)?;
        Ok(QsaAttnRaw {
            dims: shape.dims(),
            ld: call.ld,
            scale: call.scale,
            pos0,
            q: self.raw_range(&call.q)?,
            proj: self.raw_range(&call.proj)?,
            k_cache: self.raw_range(&call.k_cache)?,
            v_cache: self.raw_range(&call.v_cache)?,
            chosen: self.raw_range(&call.chosen)?,
            pos: word.map(|w| self.raw_range(w)).transpose()?,
            part: self.raw_range(&call.part)?,
            out: self.raw_range(&call.out)?,
            status: call
                .status
                .as_ref()
                .map(|s| self.raw_range(s))
                .transpose()?,
        })
    }
}

#[cfg(all(test, feature = "native-cuda"))]
mod tests;
