//! Local attention with host-authoritative raw convolution and processed cache rows.

use super::{
    super::super::dense::{LocalInputs, LocalProjection, LocalTargets},
    *,
};
use crate::{
    family::inkling2::cuda_projection::DEVICE_HEADROOM,
    kernels::cuda::resident::inkling_local_prep::{PrepMath, PrepShape},
};

type ReadWeights<'a> = dyn Fn([usize; 4]) -> Result<[Vec<f32>; 4], EngError> + 'a;

pub(super) struct AuxStore {
    buffer: DeviceBuffer,
    transfer: PinnedBuf,
    slot_bytes: usize,
    scratch_at: usize,
    next_position: Vec<Option<usize>>,
}

impl AuxStore {
    fn allocate(
        context: &CudaContext,
        visits: &[Visit],
        (device, host): (usize, usize),
    ) -> Result<Option<Self>, EngError> {
        let slot_bytes = visits
            .iter()
            .map(|v| Layout::new(v.shape, None, false).aux_bytes())
            .max()
            .unwrap_or(0);
        let Some(scratch_at) = slot_bytes.checked_mul(visits.len()) else {
            return Ok(None);
        };
        let Some(bytes) = scratch_at.checked_add(device) else {
            return Ok(None);
        };
        if bytes == 0
            || (context.memory_info()?.free as u64).saturating_sub(bytes as u64) < DEVICE_HEADROOM
        {
            return Ok(None);
        }
        let buffer = match context.allocate(bytes) {
            Ok(buffer) => buffer,
            Err(EngError::MemoryPlan(_)) => return Ok(None),
            Err(error) => return Err(error),
        };
        if (context.memory_info()?.free as u64) < DEVICE_HEADROOM {
            return Ok(None);
        }
        let transfer = match context.pinned_buf(host) {
            Ok(transfer) => transfer,
            Err(EngError::MemoryPlan(_)) => return Ok(None),
            Err(error) => return Err(error),
        };
        Ok(Some(Self {
            buffer,
            transfer,
            slot_bytes,
            scratch_at,
            next_position: vec![None; visits.len()],
        }))
    }

    pub(super) fn invalidate(&mut self) {
        self.next_position.fill(None);
    }
}

#[cfg(test)]
mod tests;

pub(in crate::family::inkling2) struct LocalPrep<'a> {
    pub math: PrepMath,
    // Query scale, relative scale, channel-major K and V kernels.
    pub read_weights: &'a ReadWeights<'a>,
    pub raw: [&'a VecDeque<Vec<f32>>; 2],
}

pub(in crate::family::inkling2) struct LocalResult {
    pub output: Vec<f32>,
    pub projected: bool,
    // Raw K/V followed by clipped K and RMS-normalized V.
    pub rows: [Vec<f32>; 4],
}

struct Layout {
    prep: PrepShape,
    shape: RelativeAttentionShape,
    prep_at: [usize; 10],
    controller_at: usize,
    output_at: usize,
    output_rows: usize,
    readback: usize,
    attention_at: usize,
    end: usize,
}

impl Layout {
    fn new(shape: RelativeAttentionShape, output_rows: Option<usize>, controlled: bool) -> Self {
        let prep = PrepShape {
            queries: shape.query_heads,
            kv: shape.kv_heads,
            dims: shape.relative_dims,
        };
        let words = prep.words().map(|n| n * size_of::<f32>());
        let mut prep_at = [0; 10];
        // Status + prep health, controller health, final output and the four host rows.
        let controller_at = words[9];
        let output_at = controller_at + usize::from(controlled) * 3 * size_of::<f32>();
        let mut end = output_at + output_rows.map_or(words[0], |rows| rows * size_of::<f32>());
        for i in [2, 3, 4, 5, 0, 1, 6, 7, 8] {
            prep_at[i] = end;
            end += words[i];
        }
        let attention_at = if output_rows.is_some() {
            let at = end;
            end += words[0];
            at
        } else {
            output_at
        };
        Self {
            prep,
            shape,
            prep_at,
            controller_at,
            output_at,
            output_rows: output_rows.unwrap_or(prep.queries * PrepShape::WIDTH),
            readback: prep_at[0],
            attention_at,
            end,
        }
    }

    fn aux_bytes(&self) -> usize {
        self.prep.words()[8] * size_of::<f32>()
    }

    fn ranges<'a>(
        &self,
        arena: &'a DeviceBuffer,
        base: usize,
    ) -> Result<[DeviceRange<'a>; 10], EngError> {
        let words = self.prep.words();
        let [a, b, c, d, e, f, g, h, i, j] = std::array::from_fn(|n| {
            arena.range(base + self.prep_at[n], words[n] * size_of::<f32>())
        });
        Ok([a?, b?, c?, d?, e?, f?, g?, h?, i?, j?])
    }

    fn read(&self, transfer: &PinnedBuf) -> Result<LocalResult, EngError> {
        let mut health = vec![0; self.output_at];
        transfer.read(0, &mut health)?;
        if health.into_iter().any(|byte| byte != 0) {
            return Err(bad("invalid CUDA local attention preparation or state"));
        }
        let read = |at, words| {
            let mut values = vec![0.0; words];
            transfer.read_f32(at, &mut values)?;
            if !values.iter().all(|v| v.is_finite()) {
                return Err(bad("non-finite CUDA local attention output or history"));
            }
            Ok(values)
        };
        let [k0, v0, k, v] = [2, 3, 4, 5].map(|i| read(self.prep_at[i], self.prep.words()[i]));
        Ok(LocalResult {
            output: read(self.output_at, self.output_rows)?,
            projected: self.attention_at != self.output_at,
            rows: [k0?, v0?, k?, v?],
        })
    }
}

pub(super) fn workspace(
    shape: RelativeAttentionShape,
    projected_rows: Option<usize>,
    hidden: usize,
) -> (usize, usize) {
    let projected = Layout::new(shape, projected_rows, true);
    (
        projected.end,
        projected
            .readback
            .max(projected.aux_bytes() + 15 * hidden * size_of::<f32>()),
    )
}

impl LocalPrep<'_> {
    fn encode(
        &self,
        layout: &Layout,
        position: i32,
        cached: bool,
        buf: &mut Vec<u8>,
    ) -> Result<usize, EngError> {
        buf.clear();
        buf.extend_from_slice(&position.to_ne_bytes());
        let count = (position as usize).min(PrepShape::TAPS - 1);
        if cached {
            return Ok(count);
        }
        let [q, rel, kv, ..] = layout.prep.words();
        let widths = [q, rel, kv * PrepShape::TAPS, kv * PrepShape::TAPS];
        let weights = (self.read_weights)(widths)?;
        if weights.each_ref().map(Vec::len) != widths
            || self
                .raw
                .iter()
                .any(|raw| raw.len() != count || raw.iter().any(|row| row.len() != kv))
        {
            return Err(bad(
                "wrong CUDA local attention convolution history or weights",
            ));
        }
        for weights in &weights {
            extend_words(buf, weights);
        }
        for raw in self.raw {
            for row in raw {
                extend_words(buf, row);
            }
            buf.resize(
                buf.len() + (PrepShape::TAPS - 1 - count) * kv * size_of::<f32>(),
                0,
            );
        }
        debug_assert_eq!(buf.len(), layout.aux_bytes());
        Ok(count)
    }
}

impl ResidentAttention {
    pub(in crate::family::inkling2) fn can_chain_local(
        &self,
        slot: usize,
        position: usize,
    ) -> bool {
        self.local_workspace.is_some()
            && position <= POS_MAX as usize
            && self.state.borrow().next_position[slot] == Some(position)
    }

    pub(in crate::family::inkling2::cuda_projection) fn step_local(
        &self,
        context: &CudaContext,
        (slot, position): (usize, usize),
        prep: LocalPrep<'_>,
        input: LocalProjection<'_>,
        projection: Option<OutputProjection<'_>>,
    ) -> Result<Option<LocalResult>, EngError> {
        if !self.can_chain_local(slot, position) {
            return Ok(None);
        }
        let visit = self.visits[slot];
        let controlled = matches!(input.inputs, LocalInputs::Controlled(..));
        let layout = Layout::new(visit.shape, projection.as_ref().map(|p| p.rows), controlled);
        let mut store = self.local_aux.borrow_mut();
        if store.is_none()
            && let Some(sizes) = self.local_workspace
        {
            *store = AuxStore::allocate(context, &self.visits, sizes)?;
        }
        let Some(store) = store.as_mut() else {
            return Ok(None);
        };
        let cached = store.next_position[slot].take() == Some(position);
        let aux = store
            .buffer
            .range(slot * store.slot_bytes, layout.aux_bytes())?;
        let mut state = self.state.borrow_mut();
        let count = prep.encode(&layout, position as i32, cached, &mut state.encoded)?;
        state.next_position[slot] = None;
        let transfer = &mut store.transfer;
        transfer.write(0, &state.encoded)?;
        let queued = self.enqueue_local(
            context,
            visit,
            &layout,
            (&store.buffer, store.scratch_at, transfer),
            (count, prep.math, aux, state.encoded.len()),
            input,
            projection,
        );
        // Device attention can write the cache even when the health check fails.
        // Drain every accepted operation and publish the cursor only on success.
        let complete = transfer.synchronize(&self.stream);
        queued?;
        complete?;
        let result = layout.read(transfer)?;
        state.next_position[slot] = Some(position + 1);
        store.next_position[slot] = Some(position + 1);
        if cached {
            static REUSED: std::sync::Once = std::sync::Once::new();
            REUSED.call_once(|| eprintln!("Inkling CUDA reused local decode aux"));
        }
        if result.projected {
            state.mark_chained_projection();
        }
        static USED: std::sync::Once = std::sync::Once::new();
        USED.call_once(|| eprintln!("Inkling CUDA used chained local decode attention"));
        Ok(Some(result))
    }

    pub(super) fn invalidate_local_aux(&self, slot: usize) {
        if let Some(store) = self.local_aux.borrow_mut().as_mut() {
            store.next_position[slot] = None;
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn enqueue_local(
        &self,
        context: &CudaContext,
        visit: Visit,
        layout: &Layout,
        (scratch, base, transfer): (&DeviceBuffer, usize, &mut PinnedBuf),
        (count, math, aux, aux_uploaded): (usize, PrepMath, DeviceRange<'_>, usize),
        input: LocalProjection<'_>,
        projection: Option<OutputProjection<'_>>,
    ) -> Result<(), EngError> {
        let mut ranges = layout.ranges(scratch, base)?;
        ranges[8] = aux;
        let [q, relative, raw_k, raw_v, k, v, sinks, gates, _, _] = ranges;
        let stream = &self.stream;
        input.dense.enqueue_local(
            context,
            input.inputs,
            input.weights,
            LocalTargets {
                projections: [q, raw_k, raw_v, relative, sinks, gates],
                controller_health: (layout.controller_at < layout.output_at)
                    .then(|| scratch.range(base + layout.controller_at, 3 * size_of::<f32>()))
                    .transpose()?,
                stream,
                transfer,
                upload_at: aux_uploaded,
            },
        )?;
        let position = aux.buffer().range(aux.offset(), size_of::<i32>())?;
        context.copy_from_pinned(
            stream,
            transfer,
            0,
            aux.buffer().range(aux.offset(), aux_uploaded)?,
        )?;
        context.inkling_local_prep(stream, ranges, layout.prep, count, math)?;
        let attention_out = scratch.range(base + layout.attention_at, q.bytes())?;
        context.relative_attention(
            stream,
            RelativeAttentionCall {
                shape: layout.shape,
                q,
                k,
                v,
                relative,
                sinks,
                gates,
                out: attention_out,
                table: self.arena.range(visit.table_at, visit.table_bytes)?,
                pos: position,
                status: scratch.range(base, size_of::<u32>())?,
                k_cache: self.arena.range(visit.cache_at, visit.cache_bytes)?,
                v_cache: self
                    .arena
                    .range(visit.cache_at + visit.cache_bytes, visit.cache_bytes)?,
            },
        )?;
        if let Some(projection) = projection {
            context.dense_gemv(
                stream,
                DenseGemvCall {
                    shape: DenseGemvShape::packed(1, projection.rows, q.bytes() / size_of::<f32>()),
                    dtype: projection.dtype,
                    w: projection.weight,
                    x: attention_out,
                    y: scratch.range(base + layout.output_at, f32_bytes(projection.rows)?)?,
                },
            )?;
        }
        context.copy_to_pinned(stream, scratch.range(base, layout.readback)?, transfer, 0)
    }
}
