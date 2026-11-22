//! Batch transfer and fences, with opt-in immutable-snapshot query tiles.

use super::*;
use crate::{
    family::inkling2::cuda_projection::{DEVICE_HEADROOM, q8_row_reuse},
    kernels::cuda::resident::relative_attention::{QUERY_TILE_LIMIT, RelativeAttentionTile},
};

const ARENA_BYTES: usize = 8 << 20;

pub(in crate::family::inkling2) fn enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("FIFI_INKLING_CUDA_BATCH_RELATIVE_ATTENTION").as_deref() == Ok("1")
    })
}

pub(super) fn query_tiles_enabled(context: &CudaContext) -> bool {
    enabled()
        && std::env::var("FIFI_INKLING_CUDA_PREFILL_QUERY_TILES").as_deref() == Ok("1")
        // Only SM75 has device evidence for the component.
        && context
            .device_attributes()
            .is_ok_and(|attrs| attrs.compute_capability == (7, 5))
}

pub(in crate::family::inkling2) fn device_side_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("FIFI_INKLING_CUDA_PREFILL_ATTENTION_SIDE").as_deref() == Ok("1")
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
        let Some(row_bytes) = (self.input + 4 + projected).checked_add(self.output) else {
            return 0;
        };
        QUERY_TILE_LIMIT
            .min(DenseGemvShape::MAX_ACTIVATION_ROWS)
            .min(ARENA_BYTES / row_bytes)
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
        inputs: (&[Row<'_>], Option<DeviceRange<'_>>),
        history: History<'_>,
        projection: Option<&OutputProjection<'_>>,
    ) -> Result<Vec<Vec<f32>>, EngError> {
        self.step_prefill_dispatch(context, slot, inputs, history, projection, self.query_tiles)
    }

    pub(super) fn step_prefill_dispatch(
        &self,
        context: &CudaContext,
        slot: usize,
        (rows, side): (&[Row<'_>], Option<DeviceRange<'_>>),
        history: History<'_>,
        projection: Option<&OutputProjection<'_>>,
        query_tiles: bool,
    ) -> Result<Vec<Vec<f32>>, EngError> {
        let visit = self.visits[slot];
        let mut layout = Layout::new(visit.shape, projection.map(|p| p.rows))
            .ok_or_else(|| bad("invalid CUDA relative attention batch layout"))?;
        let sink_bytes = visit.shape.query_heads * size_of::<f32>();
        let side_bytes = sink_bytes + layout.attention;
        if let Some(side) = side {
            if rows.len().checked_mul(side_bytes) != Some(side.bytes()) {
                return Err(bad("incorrect CUDA attention sidecar extent"));
            }
            layout.input -= side_bytes;
        }
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
        // No additional workspace. Require resident output projection so the same
        // fenced transaction covers it; singleton tails use the ordinary launch.
        let tile = if query_tiles && projection.is_some() {
            let tile = RelativeAttentionTile {
                first: position,
                count,
                input_stride: layout.input,
                side_stride: if side.is_some() {
                    side_bytes
                } else {
                    layout.input
                },
            };
            tile.region_bytes(visit.shape).map(|bytes| (tile, bytes))
        } else {
            None
        };
        if tile.is_some() {
            context.ensure_uncaptured("an Inkling attention prefill tile")?;
            context.check_launch_stream(&self.stream)?;
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
                side.is_some(),
            )?);
        }
        self.invalidate_local_aux(slot);
        let mut state = self.state.borrow_mut();
        let cursor = state.next_position[slot].take();
        // Ordinary grouped prefill never runs on a speculative clone. Reset and
        // prefix replacement invalidate the visit cursors before replacing host
        // state. Within this lineage an equal cursor is certified; never pass a
        // mismatched one to restore_rows's incremental rewind path here.
        self.restore_rows(
            visit,
            &history,
            cursor.filter(|&cursor| cursor == position),
            &mut state.encoded,
        )?;
        let result = self.execute_prefill(
            context,
            visit,
            layout,
            work,
            (&offsets, side),
            (projection, tile),
        )?;
        // Commit may already have written the ring. Publish its cursor only after
        // the fence, attention status and finite projected readback have passed.
        state.next_position[slot] = Some(position + count);
        if tile.is_some() {
            static TILED: std::sync::Once = std::sync::Once::new();
            TILED.call_once(|| eprintln!("Inkling CUDA used relative attention query tiles"));
        }
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
        (offsets, side): (&[[usize; 8]], Option<DeviceRange<'_>>),
        (projection, tile): (
            Option<&OutputProjection<'_>>,
            Option<(RelativeAttentionTile, [usize; 12])>,
        ),
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
        let device = &work.device;
        let call = |i: usize, [out, kc, vc, status, q, k, v, rel, table, sinks, gates, pos]| {
            let [q_at, k_at, v_at, rel_at, sinks_at, gates_at, pos_at, _] = offsets[i];
            let (sink_in, gate_in) = if let Some(side) = side {
                let sink_bytes = visit.shape.query_heads * size_of::<f32>();
                let at = side.offset() + i * (sink_bytes + layout.attention);
                (
                    side.buffer().range(at, sinks)?,
                    side.buffer().range(at + sink_bytes, gates)?,
                )
            } else {
                (
                    device.range(sinks_at, sinks)?,
                    device.range(gates_at, gates)?,
                )
            };
            Ok::<_, EngError>(RelativeAttentionCall {
                shape: visit.shape,
                q: device.range(q_at, q)?,
                k: device.range(k_at, k)?,
                v: device.range(v_at, v)?,
                relative: device.range(rel_at, rel)?,
                sinks: sink_in,
                gates: gate_in,
                pos: device.range(pos_at, pos)?,
                status: device.range(status_at + 4 * i, status)?,
                out: device.range(attention_at + i * layout.attention, out)?,
                table: self.arena.range(visit.table_at, table)?,
                k_cache: self.arena.range(visit.cache_at, kc)?,
                v_cache: self.arena.range(visit.cache_at + visit.cache_bytes, vc)?,
            })
        };
        let queued = (|| {
            context.copy_from_pinned(
                &self.stream,
                &mut work.transfer,
                0,
                device.range(0, out_at)?,
            )?;
            if let Some((tile, bytes)) = tile {
                context.relative_attention_tile(&self.stream, call(0, bytes)?, tile)?;
            } else {
                let bytes = visit.shape.region_bytes().unwrap(); // validated by Layout
                for i in 0..count {
                    context.relative_attention(&self.stream, call(i, bytes)?)?;
                }
            }
            if let Some(p) = projection {
                q8_row_reuse::project_output(
                    context,
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
        // A later enqueue can fail after queries or commit were accepted. Drain
        // before the borrowed sidecar or pinned workspace can be reused; a fence
        // failure takes precedence. Do not retry serially on a possibly dirty ring.
        work.transfer.synchronize(&self.stream)?;
        queued?;
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
