//! Batch transfer and fences, not causal ring updates: each row still runs in order.

use super::*;
use crate::family::inkling2::cuda_projection::DEVICE_HEADROOM;

const LIMIT: usize = 64;
const ARENA_BYTES: usize = 8 << 20;

pub(in crate::family::inkling2) fn enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("FIFI_INKLING_CUDA_BATCH_RELATIVE_ATTENTION").as_deref() == Ok("1")
    })
}

pub(in crate::family::inkling2) fn host_reserve() -> u64 {
    if enabled() {
        (2 * ARENA_BYTES) as u64
    } else {
        0
    }
}

pub(super) struct Workspace {
    device: DeviceBuffer,
    transfer: PinnedBuf,
    encoded: Vec<u8>,
}

impl Workspace {
    pub(super) fn allocate(context: &CudaContext) -> Result<Option<Self>, EngError> {
        if (context.memory_info()?.free as u64).saturating_sub(ARENA_BYTES as u64) < DEVICE_HEADROOM
        {
            return Ok(None);
        }
        let device = match context.allocate(ARENA_BYTES) {
            Ok(device) => device,
            Err(EngError::MemoryPlan(_)) => return Ok(None),
            Err(error) => return Err(error),
        };
        if (context.memory_info()?.free as u64) < DEVICE_HEADROOM {
            return Ok(None);
        }
        let transfer = match context.pinned_buf(ARENA_BYTES) {
            Ok(transfer) => transfer,
            Err(EngError::MemoryPlan(_)) => return Ok(None),
            Err(error) => return Err(error),
        };
        Ok(Some(Self {
            device,
            transfer,
            encoded: Vec::with_capacity(ARENA_BYTES),
        }))
    }
}

#[derive(Clone, Copy)]
struct Layout {
    input: usize,
    attention: usize,
    output: usize,
    projected: bool,
}

impl Layout {
    fn new(shape: RelativeAttentionShape, output_rows: Option<usize>) -> Option<Self> {
        let [attention, _, _, _, q, k, v, relative, _, sinks, gates, pos] = shape.region_bytes()?;
        Some(Self {
            input: [q, k, v, relative, sinks, gates, pos].into_iter().sum(),
            attention,
            output: output_rows.map_or(Some(attention), |n| n.checked_mul(4))?,
            projected: output_rows.is_some(),
        })
    }

    fn capacity(self) -> usize {
        let projected = if self.projected { self.attention } else { 0 };
        LIMIT
            .min(DenseGemvShape::MAX_ACTIVATION_ROWS)
            .min(ARENA_BYTES / (self.input + 4 + self.output + projected))
    }
}

impl ResidentAttention {
    pub(in crate::family::inkling2) fn prefill_capacity(
        &self,
        context: &CudaContext,
        slot: usize,
        projection: Option<&OutputProjection<'_>>,
    ) -> Result<usize, EngError> {
        if !enabled() {
            return Ok(0);
        }
        let Some(layout) = Layout::new(self.visits[slot].shape, projection.map(|p| p.rows)) else {
            return Ok(0);
        };
        let mut workspace = self.prefill.borrow_mut();
        if workspace.is_none() {
            *workspace = Workspace::allocate(context)?;
        }
        Ok(workspace.as_ref().map_or(0, |_| layout.capacity()))
    }

    pub(in crate::family::inkling2) fn step_prefill(
        &self,
        context: &CudaContext,
        slot: usize,
        rows: &[Row<'_>],
        history: History<'_>,
        projection: Option<&OutputProjection<'_>>,
    ) -> Result<Vec<Vec<f32>>, EngError> {
        let visit = self.visits[slot];
        let layout = Layout::new(visit.shape, projection.map(|p| p.rows))
            .ok_or_else(|| bad("invalid CUDA relative attention batch layout"))?;
        let count = rows.len();
        let position = history.position;
        if count == 0
            || count > layout.capacity()
            || position
                .checked_add(count - 1)
                .is_none_or(|last| last > POS_MAX as usize)
        {
            return Err(bad("invalid CUDA relative attention batch extent"));
        }
        let mut workspace = self.prefill.borrow_mut();
        let work = workspace
            .as_mut()
            .ok_or_else(|| bad("CUDA relative attention batch lacks workspace"))?;
        work.encoded.clear();
        let mut offsets = Vec::with_capacity(count);
        for (i, row) in rows.iter().enumerate() {
            offsets.push(row.append_inputs(
                visit.shape,
                (position + i) as i32,
                &mut work.encoded,
            )?);
        }
        let mut state = self.state.borrow_mut();
        let cursor = state.next_position[slot].take();
        // This path serves ordinary prompts only. A rewind reseeds from host
        // using blocking copies instead of leaving a partial async repair.
        self.restore_rows(
            visit,
            &history,
            cursor.filter(|&cursor| cursor <= position),
            &mut state.encoded,
        )?;
        let result = self.execute_prefill(context, visit, layout, work, &offsets, projection)?;
        state.next_position[slot] = Some(position + count);
        static USED: std::sync::Once = std::sync::Once::new();
        USED.call_once(|| eprintln!("Inkling CUDA used batched relative attention"));
        Ok(result)
    }

    fn execute_prefill(
        &self,
        context: &CudaContext,
        visit: Visit,
        layout: Layout,
        work: &mut Workspace,
        offsets: &[[usize; 8]],
        projection: Option<&OutputProjection<'_>>,
    ) -> Result<Vec<Vec<f32>>, EngError> {
        let count = offsets.len();
        let status_at = layout.input * count;
        let out_at = status_at + 4 * count;
        let attention_at = if layout.projected {
            out_at + layout.output * count
        } else {
            out_at
        };
        debug_assert_eq!(work.encoded.len(), status_at);
        work.encoded.resize(out_at, 0);
        work.transfer.write(0, &work.encoded)?;
        let queued = (|| {
            context.copy_from_pinned(
                &self.stream,
                &mut work.transfer,
                0,
                work.device.range(0, out_at)?,
            )?;
            for (i, &[q, k, v, rel, sinks, gates, pos, end]) in offsets.iter().enumerate() {
                context.relative_attention(
                    &self.stream,
                    RelativeAttentionCall {
                        shape: visit.shape,
                        q: work.device.range(q, k - q)?,
                        k: work.device.range(k, v - k)?,
                        v: work.device.range(v, rel - v)?,
                        relative: work.device.range(rel, sinks - rel)?,
                        sinks: work.device.range(sinks, gates - sinks)?,
                        gates: work.device.range(gates, pos - gates)?,
                        pos: work.device.range(pos, end - pos)?,
                        status: work.device.range(status_at + 4 * i, 4)?,
                        out: work
                            .device
                            .range(attention_at + i * layout.attention, layout.attention)?,
                        table: self.arena.range(visit.table_at, visit.table_bytes)?,
                        k_cache: self.arena.range(visit.cache_at, visit.cache_bytes)?,
                        v_cache: self
                            .arena
                            .range(visit.cache_at + visit.cache_bytes, visit.cache_bytes)?,
                    },
                )?;
            }
            if let Some(p) = projection {
                context.dense_gemv(
                    &self.stream,
                    DenseGemvCall {
                        shape: DenseGemvShape::packed(count, p.rows, layout.attention / 4),
                        dtype: p.dtype,
                        w: p.weight,
                        x: work.device.range(attention_at, count * layout.attention)?,
                        y: work.device.range(out_at, count * layout.output)?,
                    },
                )?;
            }
            context.copy_to_pinned(
                &self.stream,
                work.device.range(status_at, count * (4 + layout.output))?,
                &mut work.transfer,
                0,
            )
        })();
        let complete = work.transfer.synchronize(&self.stream);
        queued?;
        complete?;
        let mut status = vec![0; 4 * count];
        work.transfer.read(0, &mut status)?;
        if status.into_iter().any(|byte| byte != 0) {
            return Err(bad("invalid CUDA relative attention batch state"));
        }
        (0..count)
            .map(|i| {
                let mut output = vec![0.0; layout.output / 4];
                work.transfer
                    .read_f32(4 * count + layout.output * i, &mut output)?;
                if !output.iter().all(|v| v.is_finite()) {
                    return Err(bad("non-finite CUDA relative attention batch output"));
                }
                Ok(output)
            })
            .collect()
    }
}
