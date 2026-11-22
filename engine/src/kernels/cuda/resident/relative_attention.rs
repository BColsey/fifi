//! One F32 decode step for grouped attention with learned relative logits.
//!
//! Q and relative coefficients arrive scaled and clipped, K arrives convolved and
//! clipped, and V arrives convolved and RMS-normalized. Attention adds a learned bias
//! by distance from the current token, includes one zero-value sink per query head,
//! and applies a sigmoid gate per output channel. Softmax uses the sum of the
//! scaled QK logit and the learned bias directly.
//!
//! The caller owns and orders the rings. Before position p, the previous
//! `min(p, capacity - 1)` rows must be present at their modulo-capacity slots.
//! The step writes p's slot in both rings. Prefix snapshots and rollback across
//! overwritten slots must preserve their own copies of the cache rows.

use crate::EngError;
use crate::kernels::cuda::context::{
    Claim, CudaContext, DeviceRange, DeviceStream, require_apart, require_fit,
};
use crate::kernels::cuda::native::resident::relative_attention::{
    RelativeAttentionRaw, relative_attention_launch,
};

pub(crate) const HEAD_DIM: usize = 128;
const MAX_HEADS: usize = 64;
const MAX_WORD: usize = (1 << 30) - 1;

#[derive(Clone, Copy)]
pub(crate) struct RelativeAttentionShape {
    pub(crate) query_heads: usize,
    pub(crate) kv_heads: usize,
    pub(crate) relative_dims: usize,
    pub(crate) relative_extent: usize,
    pub(crate) capacity: usize,
}

pub(crate) struct RelativeAttentionCall<'a> {
    pub(crate) shape: RelativeAttentionShape,
    pub(crate) q: DeviceRange<'a>,
    pub(crate) k: DeviceRange<'a>,
    pub(crate) v: DeviceRange<'a>,
    pub(crate) relative: DeviceRange<'a>,
    // [relative_dims][relative_extent], with distance zero representing the current token.
    pub(crate) table: DeviceRange<'a>,
    pub(crate) sinks: DeviceRange<'a>,
    pub(crate) gates: DeviceRange<'a>,
    // F32 [capacity][kv_heads][HEAD_DIM] for each plane.
    pub(crate) k_cache: DeviceRange<'a>,
    pub(crate) v_cache: DeviceRange<'a>,
    pub(crate) out: DeviceRange<'a>,
    pub(crate) pos: DeviceRange<'a>,
    pub(crate) status: DeviceRange<'a>,
}

pub(in crate::kernels::cuda) const WRITTEN: usize = 4;
pub(in crate::kernels::cuda) const RANGES: usize = 12;

impl RelativeAttentionShape {
    pub(crate) fn region_bytes(self) -> Option<[usize; RANGES]> {
        if !(1..=MAX_HEADS).contains(&self.kv_heads)
            || !(self.kv_heads..=MAX_HEADS).contains(&self.query_heads)
            || !self.query_heads.is_multiple_of(self.kv_heads)
            || !(1..=32).contains(&self.relative_dims)
            || !(1..=MAX_WORD).contains(&self.relative_extent)
            || !(1..=MAX_WORD).contains(&self.capacity)
        {
            return None;
        }
        let q = self.query_heads * HEAD_DIM * 4;
        let kv = self.kv_heads * HEAD_DIM * 4;
        let cache = self.capacity.checked_mul(kv)?;
        let relative = self.query_heads * self.relative_dims * 4;
        let table = (self.relative_dims * 4).checked_mul(self.relative_extent)?;
        let sinks = self.query_heads * 4;
        // Order matches the ranges below and the ABI. The written ranges come first.
        Some([q, cache, cache, 4, q, kv, kv, relative, table, sinks, q, 4])
    }
}

impl CudaContext {
    /// Queues the relative-attention step in one launch, with no copies or waits.
    /// The device position (`StepParams.pos`) can change between graph replays.
    /// An invalid position writes only the status word, adding
    /// `STATUS_INVALID_STEP_STATE`. Check that status before publishing a result.
    pub(crate) fn relative_attention(
        &self,
        stream: &DeviceStream,
        call: RelativeAttentionCall<'_>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let bytes = call.shape.region_bytes().ok_or_else(|| {
            EngError::BadRequest("relative attention: invalid geometry or overflow".into())
        })?;
        let ranges = [
            ("relative attention output", call.out),
            ("relative attention key cache", call.k_cache),
            ("relative attention value cache", call.v_cache),
            ("relative attention step status", call.status),
            ("relative attention query", call.q),
            ("relative attention key", call.k),
            ("relative attention value", call.v),
            ("relative attention coefficients", call.relative),
            ("relative attention bias table", call.table),
            ("relative attention sinks", call.sinks),
            ("relative attention gates", call.gates),
            ("relative attention step position", call.pos),
        ];
        for (i, (name, range)) in ranges.iter().enumerate() {
            require_fit(name, range, bytes[i], 4)?;
            if i < WRITTEN {
                let claim = Claim::of(name, range, bytes[i]);
                for (j, (other, range)) in ranges.iter().enumerate().skip(i + 1) {
                    require_apart(&claim, &Claim::of(other, range, bytes[j]))?;
                }
            }
        }
        let mut raw_ranges = [self.raw_range(&ranges[0].1)?; RANGES];
        for (raw, (_, range)) in raw_ranges[1..].iter_mut().zip(&ranges[1..]) {
            *raw = self.raw_range(range)?;
        }
        let raw = RelativeAttentionRaw {
            ranges: raw_ranges,
            shape: call.shape,
        };
        self.launch_on(stream, |stream| relative_attention_launch(stream, &raw))
    }
}

#[cfg(test)]
mod tests;
