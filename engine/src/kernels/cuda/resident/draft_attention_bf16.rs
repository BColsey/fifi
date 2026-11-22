//! Grouped-query attention for a full speculative block over immutable committed KV.

use crate::EngError;
use crate::kernels::cuda::context::{
    Claim, CudaContext, DeviceRange, DeviceStream, require_apart, require_fit,
};
use crate::kernels::cuda::native::resident::draft_attention_bf16::{
    DraftAttentionRaw, draft_attention_bf16_launch,
};

const HEAD_DIM: usize = 128;
const MAX_HEADS: usize = 64;
const MAX_ROWS: usize = 65_535; // CUDA grid Y limit.
const MAX_CAPACITY: usize = (1 << 30) - 1;
pub(in crate::kernels::cuda) const RANGES: usize = 6;

pub(crate) struct DraftKv<'a> {
    pub(crate) keys: DeviceRange<'a>,
    pub(crate) values: DeviceRange<'a>,
}

pub(crate) struct CommittedKv<'a> {
    /// Each plane has `capacity` rows, including slots outside the visible range.
    pub(crate) kv: DraftKv<'a>,
    pub(crate) capacity: usize,
    /// Physical slot of the oldest visible row. Visible rows wrap at capacity.
    pub(crate) first: usize,
    pub(crate) len: usize,
}

/// All tensors use contiguous BF16 [row][head][128] layout. Q and output use
/// `query_heads` heads, both KV banks use `kv_heads` heads. The query and draft
/// KV banks contain `rows` rows each. Every query sees every visible committed
/// row and every draft row. The caller orders prior writes on the stream.
pub(crate) struct DraftAttentionCall<'a> {
    pub(crate) query_heads: usize,
    pub(crate) kv_heads: usize,
    pub(crate) rows: usize,
    pub(crate) query: DeviceRange<'a>,
    pub(crate) draft: DraftKv<'a>,
    pub(crate) committed: CommittedKv<'a>,
    pub(crate) output: DeviceRange<'a>,
}

#[derive(Clone, Copy)]
pub(in crate::kernels::cuda) struct Shape {
    pub(in crate::kernels::cuda) query_heads: usize,
    pub(in crate::kernels::cuda) kv_heads: usize,
    pub(in crate::kernels::cuda) rows: usize,
    pub(in crate::kernels::cuda) capacity: usize,
    pub(in crate::kernels::cuda) first: usize,
    pub(in crate::kernels::cuda) len: usize,
}

impl Shape {
    pub(in crate::kernels::cuda) fn region_bytes(self) -> Option<[usize; RANGES]> {
        if !(1..=MAX_HEADS).contains(&self.kv_heads)
            || !(self.kv_heads..=MAX_HEADS).contains(&self.query_heads)
            || !self.query_heads.is_multiple_of(self.kv_heads)
            || !(1..=MAX_ROWS).contains(&self.rows)
            || !(1..=MAX_CAPACITY).contains(&self.capacity)
            || self.first >= self.capacity
            || self.len > self.capacity
        {
            return None;
        }
        let query = (self.query_heads * HEAD_DIM * 2).checked_mul(self.rows)?;
        let kv_row = self.kv_heads * HEAD_DIM * 2;
        let draft = kv_row.checked_mul(self.rows)?;
        let committed = kv_row.checked_mul(self.capacity)?;
        // Output, query, draft key/value, committed key/value.
        Some([query, query, draft, draft, committed, committed])
    }
}

impl CudaContext {
    /// Queues attention without modifying either KV bank or allocating scratch.
    pub(crate) fn draft_attention_bf16(
        &self,
        stream: &DeviceStream,
        call: DraftAttentionCall<'_>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let shape = Shape {
            query_heads: call.query_heads,
            kv_heads: call.kv_heads,
            rows: call.rows,
            capacity: call.committed.capacity,
            first: call.committed.first,
            len: call.committed.len,
        };
        let bytes = shape.region_bytes().ok_or_else(|| {
            EngError::BadRequest("draft attention: invalid geometry or ring bounds".into())
        })?;
        let ranges = [
            ("draft attention output", call.output),
            ("draft attention query", call.query),
            ("draft attention tentative keys", call.draft.keys),
            ("draft attention tentative values", call.draft.values),
            ("draft attention committed keys", call.committed.kv.keys),
            ("draft attention committed values", call.committed.kv.values),
        ];
        for (i, (name, range)) in ranges.iter().enumerate() {
            require_fit(name, range, bytes[i], 2)?;
        }
        let claims =
            std::array::from_fn::<_, RANGES, _>(|i| Claim::of(ranges[i].0, &ranges[i].1, bytes[i]));
        for claim in &claims[1..] {
            require_apart(&claims[0], claim)?;
        }
        for draft in &claims[2..4] {
            for committed in &claims[4..] {
                require_apart(draft, committed)?;
            }
        }
        let mut raw_ranges = [self.raw_range(&ranges[0].1)?; RANGES];
        for (raw, (_, range)) in raw_ranges[1..].iter_mut().zip(&ranges[1..]) {
            *raw = self.raw_range(range)?;
        }
        let raw = DraftAttentionRaw {
            ranges: raw_ranges,
            shape,
        };
        self.launch_on(stream, |stream| draft_attention_bf16_launch(stream, &raw))
    }
}

#[cfg(test)]
mod tests;
