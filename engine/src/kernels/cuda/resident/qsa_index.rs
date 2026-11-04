//! The block indexer of Qwen3.8 Flash-Next QSA attention on the device. The formulas and layouts
//! are in `native/resident/qsa_index.h`; they follow the host indexer of
//! `family::flash_next::qsa` (`indexer_queries`, `block_key`, `block_scores`).
//!
//! QSA keeps, per layer, one pooled indexer key per completed block of `ratio` positions.
//! [`CudaContext::qsa_index`] queues, for `m` tokens of a chunk, the normed and rotated indexer
//! queries of every token, the keys of the blocks the chunk completes (pooled from the chunk's raw
//! keys and the raw-key ring of the open block), and the ring update. [`CudaContext::qsa_score`]
//! queues the score of every completed block for one token's queries; the host's `1 / sqrt(dim)`
//! is left out, since the ranking is the only consumer. [`CudaContext::qsa_select`] queues the
//! choice of the best `budget_blocks` blocks from those scores, a tie going to the lower block as
//! in the host's ranking, written in ascending block order. The attention over the chosen blocks
//! and the open tail is the next kernel's job.
//!
//! The calls allocate nothing, copy nothing and never synchronize; a fault in the queued work
//! surfaces at the next synchronize of the stream. The outputs are valid only to work queued
//! later on the same stream, or after it is synchronized.
//!
//! The position of token 0 is either a host value or the device `pos` word of a replayed step
//! ([`QsaPosition`]). The host value of a device position is advisory: it is the admission
//! check `position + m <= cap_blocks * ratio` here and never reaches the kernel, which reads the
//! word. A word outside the cache poisons on the device: no key or ring value is written, the
//! index queries are NaN, and bits 4 are OR-ed into the status word when one is given.
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
use crate::kernels::cuda::context::{Claim, require_apart, require_fit};
use crate::kernels::cuda::native::resident::qsa_index::{
    QSA_MAX_CAP_BLOCKS, QSA_MAX_DIM, QSA_MAX_HEADS, QSA_MAX_M, QSA_MAX_QUERY_FLOATS, QSA_MAX_RATIO,
    QsaDims, QsaIndexRaw, QsaRope, QsaRows, QsaScoreRaw, index_need, qsa_index_launch,
    qsa_score_launch,
};
use crate::kernels::cuda::native::resident::qsa_select::{
    QsaSelectDims, QsaSelectRaw, qsa_select_launch,
};
use crate::kernels::cuda::{CudaContext, DeviceRange, DeviceStream};

/// The constants of one layer's indexer: `heads` query heads of `dim` values, rope over the
/// first `rotary_dim` values with base `theta`, the rms0 `eps`, blocks of `ratio` positions and a
/// key cache of `cap_blocks` blocks.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct QsaIndexer {
    pub(crate) heads: usize,
    pub(crate) dim: usize,
    pub(crate) rotary_dim: usize,
    pub(crate) theta: f32,
    pub(crate) eps: f32,
    pub(crate) ratio: usize,
    pub(crate) cap_blocks: usize,
}

impl QsaIndexer {
    fn dims(self) -> QsaDims {
        QsaDims {
            heads: self.heads,
            dim: self.dim,
            ratio: self.ratio,
            cap_blocks: self.cap_blocks,
        }
    }

    fn rope(self) -> QsaRope {
        QsaRope {
            rotary_dim: self.rotary_dim,
            theta: self.theta,
            eps: self.eps,
        }
    }
}

/// The position of token 0 of a call.
pub(crate) enum QsaPosition<'b> {
    /// A host value, passed to the kernel.
    Host(usize),
    /// The device i32 `word`, read by the kernel; `admit` is the host's value of it, used only for
    /// the admission check.
    Device { word: DeviceRange<'b>, admit: usize },
}

/// One index call of `m` tokens: `proj` f32 `[m][ld]` holds each token's indexer row
/// (`heads * dim` query values, then `dim` raw key values) from column `col_off` on. `queries`
/// f32 `[m][heads][dim]` is written, `keys` f32 `[cap_blocks][dim]` gets the completed blocks,
/// `ring` f32 `[ratio][dim]` is read and updated in place.
pub(crate) struct QsaIndexCall<'b> {
    pub(crate) indexer: QsaIndexer,
    pub(crate) m: usize,
    pub(crate) ld: usize,
    pub(crate) col_off: usize,
    pub(crate) position: QsaPosition<'b>,
    pub(crate) proj: DeviceRange<'b>,
    pub(crate) q_norm: DeviceRange<'b>,
    pub(crate) k_norm: DeviceRange<'b>,
    pub(crate) queries: DeviceRange<'b>,
    pub(crate) ring: DeviceRange<'b>,
    pub(crate) keys: DeviceRange<'b>,
    pub(crate) status: Option<DeviceRange<'b>>,
}

/// One score call for the token at the position: `queries` f32 `[heads][dim]` (that token's
/// index output) against `keys`; `scores` f32 `[cap_blocks]` gets `[0, (position + 1) / ratio)`.
pub(crate) struct QsaScoreCall<'b> {
    pub(crate) indexer: QsaIndexer,
    pub(crate) position: QsaPosition<'b>,
    pub(crate) queries: DeviceRange<'b>,
    pub(crate) keys: DeviceRange<'b>,
    pub(crate) scores: DeviceRange<'b>,
    pub(crate) status: Option<DeviceRange<'b>>,
}

/// One selection for the token at the position: `scores` f32 `[cap_blocks]` (its score
/// output) is read; `blocks` i32 `[budget_blocks]` gets the chosen block indices in
/// `[0, min((position + 1) / ratio, budget_blocks))`, ascending.
pub(crate) struct QsaSelectCall<'b> {
    pub(crate) indexer: QsaIndexer,
    pub(crate) budget_blocks: usize,
    pub(crate) position: QsaPosition<'b>,
    pub(crate) scores: DeviceRange<'b>,
    pub(crate) blocks: DeviceRange<'b>,
    pub(crate) status: Option<DeviceRange<'b>>,
}

fn check_dims(indexer: QsaIndexer) -> Result<(), EngError> {
    if indexer.dims().inside() {
        return Ok(());
    }
    Err(EngError::BadRequest(format!(
        "the CUDA QSA indexer takes 1 to {QSA_MAX_HEADS} heads, a head size that is a multiple \
         of 32 from 32 to {QSA_MAX_DIM}, at most {QSA_MAX_QUERY_FLOATS} query values, a block \
         ratio of 1 to {QSA_MAX_RATIO} and 1 to {QSA_MAX_CAP_BLOCKS} cache blocks, got \
         {indexer:?}"
    )))
}

/// The host value of the position, refused unless `tokens` tokens from it fit the cache.
fn admit(
    indexer: QsaIndexer,
    position: &QsaPosition<'_>,
    tokens: usize,
) -> Result<c_int, EngError> {
    let (value, pos0) = match position {
        QsaPosition::Host(value) => (*value, *value),
        QsaPosition::Device { admit, .. } => (*admit, 0),
    };
    let cap = indexer.cap_blocks * indexer.ratio;
    if value.checked_add(tokens).is_none_or(|end| end > cap) {
        return Err(EngError::BadRequest(format!(
            "the CUDA QSA indexer position {value} with {tokens} tokens exceeds the cache of {cap} \
             positions"
        )));
    }
    // Inside the cache, so below 2^30.
    c_int::try_from(pos0).map_err(|_| EngError::BadRequest("the QSA position overflows".into()))
}

/// The device word of `position`, required 4 bytes at a 4-byte offset.
pub(super) fn position_word<'p, 'b>(
    position: &'p QsaPosition<'b>,
) -> Result<Option<&'p DeviceRange<'b>>, EngError> {
    match position {
        QsaPosition::Host(_) => Ok(None),
        QsaPosition::Device { word, .. } => {
            require_fit("pos", word, 4, 4)?;
            Ok(Some(word))
        }
    }
}

/// Every written claim apart from the other written claims and from every read claim.
pub(super) fn require_all_apart(written: &[Claim<'_>], read: &[Claim<'_>]) -> Result<(), EngError> {
    for (index, out) in written.iter().enumerate() {
        for input in read {
            require_apart(out, input)?;
        }
        for other in &written[index + 1..] {
            require_apart(out, other)?;
        }
    }
    Ok(())
}

impl CudaContext {
    /// Queues the indexer of `call.m` tokens on `stream`.
    ///
    /// Returns `BadRequest` for a stream or buffer of another adapter, for a geometry outside
    /// the limits, a row narrower than the indexer columns, a rotary size that is odd or outside
    /// `2..=dim`, a non-finite or nonpositive `theta`, a negative or non-finite `eps`, a position
    /// whose tokens do not fit the cache, a short or misaligned range, and a written range that
    /// overlaps another range.
    pub(crate) fn qsa_index(
        &self,
        stream: &DeviceStream,
        call: QsaIndexCall<'_>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let raw = self.plan_qsa_index(&call)?;
        self.launch_on(stream, move |stream_raw| qsa_index_launch(stream_raw, raw))
    }

    /// Queues the block scores of one token on `stream`.
    ///
    /// Returns `BadRequest` for a stream or buffer of another adapter, a geometry outside the
    /// limits, a position outside the cache, a short or misaligned range, and a written range
    /// that overlaps another range.
    pub(crate) fn qsa_score(
        &self,
        stream: &DeviceStream,
        call: QsaScoreCall<'_>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let raw = self.plan_qsa_score(&call)?;
        self.launch_on(stream, move |stream_raw| qsa_score_launch(stream_raw, raw))
    }

    /// Queues the block selection of one token on `stream`.
    ///
    /// Returns `BadRequest` for a stream or buffer of another adapter, a geometry outside the
    /// limits, a budget outside `1..=cap_blocks`, a position outside the cache, a short or
    /// misaligned range, and a written range that overlaps another range.
    pub(crate) fn qsa_select(
        &self,
        stream: &DeviceStream,
        call: QsaSelectCall<'_>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let raw = self.plan_qsa_select(&call)?;
        self.launch_on(stream, move |stream_raw| qsa_select_launch(stream_raw, raw))
    }

    fn plan_qsa_index<'b>(&self, call: &QsaIndexCall<'b>) -> Result<QsaIndexRaw<'b>, EngError> {
        let indexer = call.indexer;
        check_dims(indexer)?;
        if !(1..=QSA_MAX_M).contains(&call.m) {
            return Err(EngError::BadRequest(format!(
                "the CUDA QSA indexer takes 1 to {QSA_MAX_M} tokens, got {}",
                call.m
            )));
        }
        if !indexer.rope().valid_for(indexer.dim) {
            return Err(EngError::BadRequest(format!(
                "the CUDA QSA indexer needs an even rotary size in 2..={}, a finite positive theta \
                 and a finite nonnegative eps, got {indexer:?}",
                indexer.dim
            )));
        }
        let rows = QsaRows {
            m: call.m,
            ld: call.ld,
            col_off: call.col_off,
        };
        let need = index_need(indexer.dims(), rows).ok_or_else(|| {
            EngError::BadRequest(format!(
                "the CUDA QSA indexer row of {} floats from column {} cannot hold {} heads of {} \
                 plus the key, or exceeds i32",
                call.ld, call.col_off, indexer.heads, indexer.dim
            ))
        })?;
        let pos0 = admit(indexer, &call.position, call.m)?;
        let pos = position_word(&call.position)?;
        require_fit("proj", &call.proj, need.proj, 4)?;
        require_fit("q_norm", &call.q_norm, need.norm, 4)?;
        require_fit("k_norm", &call.k_norm, need.norm, 4)?;
        require_fit("queries", &call.queries, need.queries, 4)?;
        require_fit("ring", &call.ring, need.ring, 4)?;
        require_fit("keys", &call.keys, need.keys, 4)?;
        if let Some(status) = &call.status {
            require_fit("status", status, 4, 4)?;
        }
        let mut written = vec![
            Claim::of("queries", &call.queries, need.queries),
            Claim::of("ring", &call.ring, need.ring),
            Claim::of("keys", &call.keys, need.keys),
        ];
        written.extend(call.status.as_ref().map(|s| Claim::of("status", s, 4)));
        let mut read = vec![
            Claim::of("proj", &call.proj, need.proj),
            Claim::of("q_norm", &call.q_norm, need.norm),
            Claim::of("k_norm", &call.k_norm, need.norm),
        ];
        read.extend(pos.map(|w| Claim::of("pos", w, 4)));
        require_all_apart(&written, &read)?;
        Ok(QsaIndexRaw {
            dims: indexer.dims(),
            rows,
            rope: indexer.rope(),
            pos0,
            proj: self.raw_range(&call.proj)?,
            q_norm: self.raw_range(&call.q_norm)?,
            k_norm: self.raw_range(&call.k_norm)?,
            queries: self.raw_range(&call.queries)?,
            ring: self.raw_range(&call.ring)?,
            keys: self.raw_range(&call.keys)?,
            pos: pos.map(|w| self.raw_range(w)).transpose()?,
            status: call
                .status
                .as_ref()
                .map(|s| self.raw_range(s))
                .transpose()?,
        })
    }

    fn plan_qsa_score<'b>(&self, call: &QsaScoreCall<'b>) -> Result<QsaScoreRaw<'b>, EngError> {
        let indexer = call.indexer;
        check_dims(indexer)?;
        let pos0 = admit(indexer, &call.position, 1)?;
        let pos = position_word(&call.position)?;
        let dims = indexer.dims();
        let (queries, keys, scores) = (
            dims.heads * dims.dim * 4,
            dims.keys_bytes(),
            dims.cap_blocks * 4,
        );
        require_fit("queries", &call.queries, queries, 4)?;
        require_fit("keys", &call.keys, keys, 4)?;
        require_fit("scores", &call.scores, scores, 4)?;
        if let Some(status) = &call.status {
            require_fit("status", status, 4, 4)?;
        }
        let mut written = vec![Claim::of("scores", &call.scores, scores)];
        written.extend(call.status.as_ref().map(|s| Claim::of("status", s, 4)));
        let mut read = vec![
            Claim::of("queries", &call.queries, queries),
            Claim::of("keys", &call.keys, keys),
        ];
        read.extend(pos.map(|w| Claim::of("pos", w, 4)));
        require_all_apart(&written, &read)?;
        Ok(QsaScoreRaw {
            dims,
            pos0,
            queries: self.raw_range(&call.queries)?,
            keys: self.raw_range(&call.keys)?,
            scores: self.raw_range(&call.scores)?,
            pos: pos.map(|w| self.raw_range(w)).transpose()?,
            status: call
                .status
                .as_ref()
                .map(|s| self.raw_range(s))
                .transpose()?,
        })
    }

    fn plan_qsa_select<'b>(&self, call: &QsaSelectCall<'b>) -> Result<QsaSelectRaw<'b>, EngError> {
        let indexer = call.indexer;
        check_dims(indexer)?;
        let dims = QsaSelectDims {
            ratio: indexer.ratio,
            cap_blocks: indexer.cap_blocks,
            budget_blocks: call.budget_blocks,
        };
        if !dims.inside() {
            return Err(EngError::BadRequest(format!(
                "the CUDA QSA selection takes a budget of 1 to {} blocks, got {}",
                indexer.cap_blocks, call.budget_blocks
            )));
        }
        let pos0 = admit(indexer, &call.position, 1)?;
        let pos = position_word(&call.position)?;
        let (scores, blocks) = (dims.cap_blocks * 4, dims.budget_blocks * 4);
        require_fit("scores", &call.scores, scores, 4)?;
        require_fit("blocks", &call.blocks, blocks, 4)?;
        if let Some(status) = &call.status {
            require_fit("status", status, 4, 4)?;
        }
        let mut written = vec![Claim::of("blocks", &call.blocks, blocks)];
        written.extend(call.status.as_ref().map(|s| Claim::of("status", s, 4)));
        let mut read = vec![Claim::of("scores", &call.scores, scores)];
        read.extend(pos.map(|w| Claim::of("pos", w, 4)));
        require_all_apart(&written, &read)?;
        Ok(QsaSelectRaw {
            dims,
            pos0,
            scores: self.raw_range(&call.scores)?,
            blocks: self.raw_range(&call.blocks)?,
            pos: pos.map(|w| self.raw_range(w)).transpose()?,
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
