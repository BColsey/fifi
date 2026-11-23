//! Batch transfer and fences, with opt-in immutable-snapshot query tiles.

use super::*;
use crate::{
    family::inkling2::cuda_projection::{DEVICE_HEADROOM, q8_row_reuse},
    kernels::cuda::resident::{
        inkling_local_prep::{OutputConvShape, PrepShape},
        relative_attention::{QUERY_TILE_LIMIT, RelativeAttentionTile},
    },
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

#[derive(Clone, Copy)]
pub(super) struct ResidentOutput<'a> {
    pub raw: DeviceRange<'a>,
    pub convolution: Option<OutputConvolution<'a>>,
}

#[derive(Clone, Copy)]
pub(super) struct OutputConvolution<'a> {
    pub history: usize,
    // Raw history (right-aligned, oldest first), channel-major weights, output.
    pub planes: [DeviceRange<'a>; 3],
    pub read_convolved: bool,
}

/// Owns the admitted whole visit. Adjacent raw tiles become convolution
/// history by slicing the same buffer, with no intermediate copies.
pub(in crate::family::inkling2) struct PrefillConv {
    raw: DeviceBuffer,
    convolved: DeviceBuffer,
    weights: DeviceBuffer,
    stride: usize,
    initial: usize,
    rows: usize,
    written: usize,
}

impl PrefillConv {
    const PAST: usize = PrepShape::TAPS - 1;

    fn destination(&self, count: usize) -> Result<ResidentOutput<'_>, EngError> {
        if count == 0 || count > self.rows - self.written {
            return Err(bad("CUDA prefill convolution exceeds admitted rows"));
        }
        Ok(ResidentOutput {
            raw: self.raw.range(
                (Self::PAST + self.written) * self.stride,
                count * self.stride,
            )?,
            convolution: Some(OutputConvolution {
                history: (self.initial + self.written).min(Self::PAST),
                planes: [
                    self.raw
                        .range(self.written * self.stride, Self::PAST * self.stride)?,
                    self.weights.range(0, self.weights.bytes())?,
                    self.convolved
                        .range(self.written * self.stride, count * self.stride)?,
                ],
                // Until depth mixing consumes the full convolved plane.
                read_convolved: true,
            }),
        })
    }
}

impl ResidentOutput<'_> {
    fn validate(self, work: &Workspace, layout: Layout, rows: usize) -> Result<(), EngError> {
        let Some(conv) = self.convolution else {
            if layout.projected
                && rows.checked_mul(layout.output) == Some(self.raw.bytes())
                && !std::ptr::eq(self.raw.buffer(), &work.device)
            {
                return Ok(());
            }
            return Err(bad("incorrect resident CUDA attention output destination"));
        };
        let shape = OutputConvShape {
            rows,
            width: layout.output / 4,
            history: conv.history,
        };
        if let Some([raw, past, weight, out, _]) = shape.words()
            && layout.projected
            && [self.raw, conv.planes[0], conv.planes[1], conv.planes[2]]
                .into_iter()
                .zip([raw, past, weight, out])
                .all(|(r, words)| {
                    r.bytes() == 4 * words
                        && r.offset().is_multiple_of(4)
                        && !std::ptr::eq(r.buffer(), &work.device)
                })
        {
            return Ok(());
        }
        Err(bad(
            "incorrect resident CUDA attention convolution destination",
        ))
    }
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

    pub(in crate::family::inkling2) fn admit_prefill_conv(
        &self,
        context: &CudaContext,
        shape: [usize; 2],
        prior: &VecDeque<Vec<f32>>,
        weights: &[f32],
    ) -> Result<Option<PrefillConv>, EngError> {
        let [rows, width] = shape;
        let initial = prior.len();
        let Some([plane, past, taps, _, _]) = (OutputConvShape {
            rows,
            width,
            history: initial,
        })
        .words() else {
            return Ok(None);
        };
        if weights.len() != taps || prior.iter().any(|row| row.len() != width) {
            return Err(bad("invalid CUDA prefill convolution weights or history"));
        }
        let bytes = 4 * (2 * plane + past + taps);
        if (context.memory_info()?.free as u64).saturating_sub(bytes as u64) < DEVICE_HEADROOM {
            return Ok(None);
        }
        let buffers = (|| {
            Ok::<_, EngError>([
                context.allocate(4 * (plane + past))?,
                context.allocate(4 * plane)?,
                context.allocate(4 * taps)?,
            ])
        })();
        let [raw, convolved, kernel] = match buffers {
            Ok(buffers) => buffers,
            Err(EngError::MemoryPlan(_)) => return Ok(None),
            Err(error) => return Err(error),
        };
        if (context.memory_info()?.free as u64) < DEVICE_HEADROOM {
            return Ok(None);
        }
        let mut upload = vec![0; 4 * (past - initial * width)];
        for row in prior.iter().rev() {
            extend_words(&mut upload, row);
        }
        raw.write_at(&self.stream, 0, &upload)?;
        upload.clear();
        extend_words(&mut upload, weights);
        kernel.write_at(&self.stream, 0, &upload)?;
        Ok(Some(PrefillConv {
            raw,
            convolved,
            weights: kernel,
            stride: 4 * width,
            initial,
            rows,
            written: 0,
        }))
    }

    pub(in crate::family::inkling2) fn step_prefill_conv(
        &self,
        context: &CudaContext,
        slot: usize,
        (rows, side): (&[Row<'_>], Option<Sidecar<'_>>),
        history: History<'_>,
        projection: &OutputProjection<'_>,
        conv: &mut PrefillConv,
    ) -> Result<Vec<Vec<f32>>, EngError> {
        let destination = conv.destination(rows.len())?;
        let output = self.step_prefill_to(
            context,
            slot,
            (rows, side),
            history,
            Some(projection),
            (self.query_tiles, Some(destination)),
        )?;
        conv.written += rows.len();
        Ok(output)
    }

    pub(in crate::family::inkling2) fn prefill_conv_tail(
        &self,
        conv: PrefillConv,
    ) -> Result<VecDeque<Vec<f32>>, EngError> {
        if conv.written != conv.rows {
            return Err(bad("incomplete CUDA prefill convolution visit"));
        }
        let count = (conv.written + conv.initial).min(PrefillConv::PAST);
        let mut bytes = vec![0; count * conv.stride];
        let first = PrefillConv::PAST + conv.written - count;
        conv.raw
            .read_at(&self.stream, first * conv.stride, &mut bytes)?;
        let mut tail = VecDeque::with_capacity(count);
        for row in bytes.chunks_exact(conv.stride).rev() {
            let row: Vec<_> = row
                .chunks_exact(4)
                .map(|w| f32::from_ne_bytes(w.try_into().unwrap()))
                .collect();
            if !row.iter().all(|x| x.is_finite()) {
                return Err(bad("non-finite CUDA prefill convolution history"));
            }
            tail.push_back(row);
        }
        Ok(tail)
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
        self.step_prefill_to(
            context,
            slot,
            (rows, side),
            history,
            projection,
            (query_tiles, None),
        )
    }

    /// A supplied destination is owned by the visit, not the reused attention tile.
    /// The kernel validates convolved output through the attention status plane.
    /// Raw-only consumers must check finite values before publishing the visit.
    pub(super) fn step_prefill_to(
        &self,
        context: &CudaContext,
        slot: usize,
        (rows, side): (&[Row<'_>], Option<Sidecar<'_>>),
        history: History<'_>,
        projection: Option<&OutputProjection<'_>>,
        (query_tiles, destination): (bool, Option<ResidentOutput<'_>>),
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
        if let Some(dest) = destination {
            dest.validate(work, layout, count)?;
        }
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
            (projection, tile, destination),
        )?;
        // The ring is independent of the projection output. Publish its cursor
        // after the fence and attention status (and host output check, if read).
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
        (projection, tile, destination): (
            Option<&OutputProjection<'_>>,
            Option<(RelativeAttentionTile, [usize; 12])>,
            Option<ResidentOutput<'_>>,
        ),
    ) -> Result<Vec<Vec<f32>>, EngError> {
        let count = offsets.len();
        let convolved_readback = destination
            .and_then(|d| d.convolution)
            .filter(|c| c.read_convolved)
            .map(|c| c.planes[2]);
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
                let y = match destination {
                    Some(dest) => dest.raw,
                    None => work.device.range(out_at, count * layout.output)?,
                };
                q8_row_reuse::project_output(
                    context,
                    &self.stream,
                    DenseGemvCall {
                        shape: DenseGemvShape::packed(count, p.rows, layout.attention / 4),
                        dtype: p.dtype,
                        w: p.weight,
                        x: work.device.range(attention_at, count * layout.attention)?,
                        y,
                    },
                )?;
            }
            if let Some(dest) = destination
                && let Some(conv) = dest.convolution
            {
                let [past, weight, output] = conv.planes;
                context.inkling_output_conv(
                    &self.stream,
                    [
                        dest.raw,
                        past,
                        weight,
                        output,
                        device.range(status_at, 4 * count)?,
                    ],
                    OutputConvShape {
                        rows: count,
                        width: layout.output / 4,
                        history: conv.history,
                    },
                )?;
            }
            if let Some(convolved) = convolved_readback {
                context.copy_to_pinned(&self.stream, convolved, &mut work.transfer, 4 * count)?;
            }
            let readback = if destination.is_some() {
                4 * count
            } else {
                count * (4 + layout.output)
            };
            context.copy_to_pinned(
                &self.stream,
                work.device.range(status_at, readback)?,
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
        if destination.is_some() && convolved_readback.is_none() {
            return Ok(Vec::new());
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
