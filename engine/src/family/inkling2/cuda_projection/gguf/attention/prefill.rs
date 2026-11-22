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

/// Extends the ordinary batched relative-attention path; does not enable
/// batching, grouped controllers, or the resident attention cache on its own.
pub(in crate::family::inkling2) fn resident_query_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("FIFI_INKLING_CUDA_PREFILL_RESIDENT_QUERY").as_deref() == Ok("1")
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
pub(in crate::family::inkling2) struct QueryClip<'a> {
    pub scales: [&'a [f32]; 2],
    pub limits: [f32; 2],
}

#[derive(Clone, Copy)]
pub(in crate::family::inkling2) struct Sidecar<'a> {
    pub range: DeviceRange<'a>,
    pub query: Option<QueryClip<'a>>,
}

pub(in crate::family::inkling2) fn projection_side(query: bool) -> &'static [usize] {
    if query { &[0, 3, 4, 5] } else { &[4, 5] }
}

#[derive(Clone, Copy)]
struct Layout {
    input: usize,
    scales: usize,
    resident: [usize; 6],
    attention: usize,
    output: usize,
    projected: bool,
}

impl Layout {
    fn new(
        shape: RelativeAttentionShape,
        output_rows: Option<usize>,
        keep: &[usize],
    ) -> Option<Self> {
        let [attention, _, _, _, q, k, v, relative, _, sinks, gates, pos] = shape.region_bytes()?;
        let widths = [q, k, v, relative, sinks, gates];
        let mut resident = [0; 6];
        for &i in keep {
            resident[i] = widths[i];
        }
        Some(Self {
            input: widths.iter().sum::<usize>() + pos - resident.iter().sum::<usize>(),
            scales: resident[0] + resident[3],
            resident,
            attention,
            output: output_rows.map_or(Some(attention), |n| n.checked_mul(4))?,
            projected: output_rows.is_some(),
        })
    }

    fn side_stride(self) -> usize {
        self.resident.iter().sum()
    }

    fn row_bytes(self) -> Option<usize> {
        let projected = if self.projected { self.attention } else { 0 };
        (self.input + 4 + projected).checked_add(self.output)
    }

    fn capacity(self, query_tiles: bool) -> usize {
        let Some(row_bytes) = self.row_bytes() else {
            return 0;
        };
        // Keep the combined tip's batching when resident queries are off.
        let limit = if query_tiles || self.scales == 0 {
            QUERY_TILE_LIMIT
        } else {
            512 // resident clip launch limit
        };
        limit
            .min(DenseGemvShape::MAX_ACTIVATION_ROWS)
            .min((ARENA_BYTES - self.scales) / row_bytes)
    }

    fn row_inputs<'a>(
        self,
        work: &'a DeviceBuffer,
        offsets: [usize; 8],
        widths: [usize; 6],
        side: Option<&Sidecar<'a>>,
        row: usize,
    ) -> Result<[DeviceRange<'a>; 6], EngError> {
        let mut side_at = side.map_or(0, |s| s.range.offset()) + row * self.side_stride();
        let mut next = |i: usize| {
            if self.resident[i] == 0 {
                work.range(offsets[i], widths[i])
            } else {
                let side = side.ok_or_else(|| bad("missing CUDA attention sidecar"))?;
                let relative_at = side_at - side.range.offset();
                if relative_at
                    .checked_add(widths[i])
                    .is_none_or(|end| end > side.range.bytes())
                {
                    return Err(bad("incorrect CUDA attention sidecar extent"));
                }
                let range = side.range.buffer().range(side_at, widths[i]);
                side_at += self.resident[i];
                range
            }
        };
        Ok([next(0)?, next(1)?, next(2)?, next(3)?, next(4)?, next(5)?])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resident_query_layout_leaves_only_kv_and_position_in_host_rows() {
        let shape = RelativeAttentionShape {
            query_heads: 64,
            kv_heads: 8,
            relative_dims: 32,
            relative_extent: 1024,
            capacity: 4096,
        };
        let words = |n| n * size_of::<f32>();
        let q = words(64 * 128);
        let kv = words(8 * 128);
        let relative = words(64 * 32);
        let sinks = words(64);
        let pos = size_of::<i32>();
        let projected = Some(16384);
        let host = Layout::new(shape, projected, &[]).unwrap();
        let side = Layout::new(shape, projected, projection_side(false)).unwrap();
        let query = Layout::new(shape, projected, projection_side(true)).unwrap();
        assert_eq!((host.side_stride(), host.scales), (0, 0));
        assert_eq!((side.side_stride(), side.scales), (sinks + q, 0));
        assert_eq!(query.resident, [q, 0, 0, relative, sinks, q]);
        assert_eq!(query.input, 2 * kv + pos);
        assert_eq!(query.scales, q + relative);
        assert_eq!(query.input + query.side_stride(), host.input);
        for layout in [host, side, query] {
            for tiled in [false, true] {
                let limit = if tiled || layout.scales == 0 {
                    QUERY_TILE_LIMIT
                } else {
                    512
                };
                let capacity = layout.capacity(tiled);
                let row_bytes = layout.row_bytes().unwrap();
                assert!(layout.scales + capacity * row_bytes <= ARENA_BYTES);
                assert!(
                    capacity == limit.min(DenseGemvShape::MAX_ACTIVATION_ROWS)
                        || layout.scales + (capacity + 1) * row_bytes > ARENA_BYTES
                );
            }
        }
        assert!(query.capacity(false) > side.capacity(false));
        assert!(side.capacity(false) > host.capacity(false));
    }
}

impl ResidentAttention {
    pub(in crate::family::inkling2) fn prefill_capacity(
        &self,
        context: &CudaContext,
        slot: usize,
        projection: Option<&OutputProjection<'_>>,
    ) -> Result<usize, EngError> {
        self.prefill_tile(context, slot, projection, &[])
            .map(|(capacity, _)| capacity)
    }

    pub(in crate::family::inkling2) fn prefill_tile(
        &self,
        context: &CudaContext,
        slot: usize,
        projection: Option<&OutputProjection<'_>>,
        keep: &[usize],
    ) -> Result<(usize, usize), EngError> {
        if !enabled() {
            return Ok((0, 0));
        }
        let Some(layout) = Layout::new(self.visits[slot].shape, projection.map(|p| p.rows), keep)
        else {
            return Ok((0, 0));
        };
        let mut workspace = self.prefill.borrow_mut();
        if workspace.is_none() {
            *workspace = Workspace::allocate(context)?;
        }
        Ok((
            workspace
                .as_ref()
                .map_or(0, |_| layout.capacity(self.query_tiles)),
            layout.side_stride(),
        ))
    }

    pub(in crate::family::inkling2) fn step_prefill(
        &self,
        context: &CudaContext,
        slot: usize,
        inputs: (&[Row<'_>], Option<Sidecar<'_>>),
        history: History<'_>,
        projection: Option<&OutputProjection<'_>>,
    ) -> Result<Vec<Vec<f32>>, EngError> {
        self.step_prefill_dispatch(context, slot, inputs, history, projection, self.query_tiles)
    }

    pub(super) fn step_prefill_dispatch(
        &self,
        context: &CudaContext,
        slot: usize,
        (rows, side): (&[Row<'_>], Option<Sidecar<'_>>),
        history: History<'_>,
        projection: Option<&OutputProjection<'_>>,
        query_tiles: bool,
    ) -> Result<Vec<Vec<f32>>, EngError> {
        let visit = self.visits[slot];
        let keep = side
            .as_ref()
            .map_or(&[][..], |s| projection_side(s.query.is_some()));
        let layout = Layout::new(visit.shape, projection.map(|p| p.rows), keep)
            .ok_or_else(|| bad("invalid CUDA relative attention batch layout"))?;
        if let Some(side) = &side
            && rows.len().checked_mul(layout.side_stride()) != Some(side.range.bytes())
        {
            return Err(bad("incorrect CUDA attention sidecar extent"));
        }
        let count = rows.len();
        let position = history.position;
        if count == 0
            || count > layout.capacity(query_tiles)
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
                query_stride: if layout.scales != 0 {
                    layout.side_stride()
                } else {
                    layout.input
                },
                side_stride: if side.is_some() {
                    layout.side_stride()
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
                layout.resident.map(|n| n != 0),
            )?);
        }
        if let Some(query) = side.as_ref().and_then(|s| s.query) {
            for (scale, i) in query.scales.into_iter().zip([0, 3]) {
                if scale.len() != layout.resident[i] / size_of::<f32>() {
                    return Err(bad("incorrect CUDA attention query scale extent"));
                }
                extend_words(&mut work.encoded, scale);
            }
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
        (offsets, side): (&[[usize; 8]], Option<Sidecar<'_>>),
        (projection, tile): (
            Option<&OutputProjection<'_>>,
            Option<(RelativeAttentionTile, [usize; 12])>,
        ),
    ) -> Result<Vec<Vec<f32>>, EngError> {
        let count = offsets.len();
        let scales_at = layout.input * count;
        let status_at = scales_at + layout.scales;
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
        let call = |i: usize, bytes: [usize; 12]| {
            let [out, kc, vc, status, q, k, v, rel, table, sinks, gates, pos] = bytes;
            let [q, k, v, relative, sinks, gates] = layout.row_inputs(
                device,
                offsets[i],
                [q, k, v, rel, sinks, gates],
                side.as_ref(),
                i,
            )?;
            Ok::<_, EngError>(RelativeAttentionCall {
                shape: visit.shape,
                q,
                k,
                v,
                relative,
                sinks,
                gates,
                pos: device.range(offsets[i][6], pos)?,
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
            if let Some(side) = &side
                && let Some(query) = side.query
            {
                context.relative_attention_clip_queries(
                    &self.stream,
                    visit.shape,
                    count,
                    query.limits,
                    [
                        side.range,
                        work.device.range(scales_at, layout.scales)?,
                        work.device.range(status_at, 4 * count)?,
                    ],
                )?;
            }
            // Prep ORs into the same status checked by snapshot commit. No
            // resident sidecar is replayed or re-prepared after enqueue.
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
        // before the borrowed sidecar or pinned workspace can be reused. Do not
        // retry serially on a possibly dirty ring.
        work.transfer
            .synchronize_after_enqueue(&self.stream, queued)?;
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
