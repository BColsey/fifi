//! Keep resident GGUF expert intermediates on the device during grouped prefill.
use std::{
    collections::{BTreeMap, BTreeSet},
    ops::Range,
};

use super::{
    DEVICE_HEADROOM, GgufCuda, bad,
    streaming::{Matrix, PrefillRoute, group_routes},
};
use crate::{
    EngError,
    expert_tier::{ExpertId, ExpertPayloadSource},
    kernels::cuda::{
        CudaContext, DeviceBuffer, DeviceRange, DeviceStream, PinnedBuf,
        prefill::row_ops::{ClippedExpertCall, IndexedRankSumCall, RowGatherCall, RowShape},
        resident::dense_gemv::{DenseDtype, DenseGemvCall, DenseGemvShape, dense_gemv_bytes},
    },
};

#[derive(Clone, Copy, PartialEq, Eq)]
struct Geometry {
    hidden: usize,
    middle: usize,
    rank: usize,
}

struct Layout {
    input: Range<usize>,
    row_ids: Range<usize>,
    inverse_ids: Range<usize>,
    gate_scales: Range<usize>,
    down_scales: Range<usize>,
    weights: Range<usize>,
    flag: Range<usize>,
    output: Range<usize>,
    gathered: Range<usize>,
    gate: Range<usize>,
    middle: Range<usize>,
    down: Range<usize>,
}

impl Layout {
    fn new(rows: usize, shape: Geometry) -> Option<Self> {
        let slots = rows.checked_mul(shape.rank)?;
        let mut offset = 0usize;
        let mut span = |width: usize, height: usize| {
            let start = offset;
            offset = start.checked_add(width.checked_mul(height)?.checked_mul(4)?)?;
            Some(start..offset)
        };
        Some(Self {
            input: span(rows, shape.hidden)?,
            row_ids: span(slots, 1)?,
            inverse_ids: span(slots, 1)?,
            gate_scales: span(slots, 1)?,
            down_scales: span(slots, 1)?,
            weights: span(slots, 1)?,
            flag: span(1, 1)?,
            output: span(rows, shape.hidden)?,
            gathered: span(slots, shape.hidden)?,
            gate: span(slots, shape.middle.checked_mul(2)?)?,
            middle: span(slots, shape.middle)?,
            down: span(slots, shape.hidden)?,
        })
    }

    fn fit(requested: usize, shape: Geometry, capacity: usize) -> Option<usize> {
        let single = Self::new(1, shape)?;
        let bytes_per_row = single.down.end - single.flag.len();
        let admitted = capacity
            .saturating_sub(single.flag.len())
            .checked_div(bytes_per_row)?
            .min(requested)
            .min(DenseGemvShape::MAX_ACTIVATION_ROWS.checked_div(shape.rank)?);
        (admitted >= 2).then_some(admitted)
    }
}

#[derive(Default)]
pub(super) struct Workspace {
    scratch: Option<Scratch>,
}

struct Scratch {
    buffer: DeviceBuffer,
    stream: DeviceStream,
    host: PinnedBuf,
    rows: usize,
    geometry: Geometry,
}

impl Workspace {
    fn fit(
        &mut self,
        cuda: &CudaContext,
        requested: usize,
        shape: Geometry,
    ) -> Result<Option<&mut Scratch>, EngError> {
        if self
            .scratch
            .as_ref()
            .is_some_and(|buffer| buffer.geometry == shape && buffer.rows >= requested)
        {
            return Ok(self
                .scratch
                .as_mut()
                .filter(|buffer| buffer.geometry == shape));
        }
        // This temporary work can consume the existing reserve above the family floor.
        // Check only after expert and optional projection placement has completed.
        let free = cuda
            .memory_info()?
            .free
            .saturating_sub(DEVICE_HEADROOM as usize);
        let Some(rows) = Layout::fit(requested, shape, free) else {
            return Ok(self
                .scratch
                .as_mut()
                .filter(|buffer| buffer.geometry == shape));
        };
        if self
            .scratch
            .as_ref()
            .is_some_and(|buffer| buffer.geometry == shape && buffer.rows >= rows)
        {
            return Ok(self
                .scratch
                .as_mut()
                .filter(|buffer| buffer.geometry == shape));
        }
        let layout =
            Layout::new(rows, shape).ok_or_else(|| bad("expert prefill arena overflows"))?;
        let Some(scratch) = Scratch::allocate(cuda, rows, shape, &layout)? else {
            return Ok(self
                .scratch
                .as_mut()
                .filter(|buffer| buffer.geometry == shape));
        };
        eprintln!(
            "Inkling CUDA device expert workspace {}",
            serde_json::json!({"requested_rows": requested, "admitted_rows": rows,
                "device_bytes": layout.down.end, "pinned_bytes": layout.output.end})
        );
        self.scratch = Some(scratch);
        Ok(self.scratch.as_mut())
    }
}

struct Projections<'a> {
    geometry: Geometry,
    caps: [f32; 2],
    dtypes: [DenseDtype; 2],
    matrices: BTreeMap<ExpertId, [Matrix<'a>; 2]>,
}

impl GgufCuda {
    pub(in crate::family::inkling2) fn supports_grouped_prefill(&self) -> bool {
        self.has_stream_banks()
            || (self.device_prefill.is_some()
                && !self.resident.placement().full_layers().is_empty())
    }

    pub(in crate::family::inkling2) fn prefills_device_layer(&self, layer: u32) -> bool {
        self.device_prefill.is_some() && !self.resident.placement().has_cold(layer)
    }

    pub(in crate::family::inkling2) fn prefill_device_experts(
        &self,
        source: &mut impl ExpertPayloadSource,
        layer: u32,
        routes: &[PrefillRoute<'_>],
        dtypes: [DenseDtype; 2],
        caps: [f32; 2],
        weight: impl Fn(usize, usize) -> f32,
    ) -> Result<Option<Vec<Vec<f32>>>, EngError> {
        let Some(workspace) = &self.device_prefill else {
            return Ok(None);
        };
        let Some(first) = routes.first() else {
            return Ok(None);
        };
        let [rank, hidden] = [first.ids.len(), first.input.len()];
        if routes.len() < 2
            || hidden == 0
            || !(1..=32).contains(&rank)
            || !caps.iter().all(|cap| cap.is_finite() && *cap > 0.0)
            || routes
                .iter()
                .any(|route| route.ids.len() != rank || route.input.len() != hidden)
            || dtypes
                .iter()
                .any(|&dtype| !dtype.is_packed_gguf() || self.step.uses_q8_activations(dtype))
        {
            return Ok(None);
        }
        let Some(projections) =
            self.prepare_device_experts(source, layer, routes, [rank, hidden], dtypes, caps)?
        else {
            return Ok(None);
        };
        let cuda = &self.context;
        let mut workspace = workspace.borrow_mut();
        let Some(scratch) = workspace.fit(cuda, routes.len(), projections.geometry)? else {
            return Ok(None);
        };
        let mut output = Vec::with_capacity(routes.len());
        for (window, routes) in routes.chunks(scratch.rows).enumerate() {
            let first = window * scratch.rows;
            let plan = RoutedWindow::prepare(&projections, layer, routes, |row, id| {
                weight(first + row, id)
            })?;
            output.extend(scratch.execute(cuda, &projections, &plan)?);
        }
        if self.prefill_report.is_some() {
            eprintln!(
                "Inkling CUDA device expert prefill {}",
                serde_json::json!({"layer": layer, "rows": routes.len(),
                    "windows": routes.len().div_ceil(scratch.rows)})
            );
        }
        Ok(Some(output))
    }

    fn prepare_device_experts<'a>(
        &'a self,
        source: &mut impl ExpertPayloadSource,
        layer: u32,
        routes: &[PrefillRoute<'_>],
        [rank, hidden]: [usize; 2],
        dtypes: [DenseDtype; 2],
        caps: [f32; 2],
    ) -> Result<Option<Projections<'a>>, EngError> {
        let experts: BTreeSet<_> = routes.iter().flat_map(|route| route.ids).copied().collect();
        let mut matrices = BTreeMap::new();
        let mut middle = 0;
        for expert in experts {
            let id = ExpertId {
                layer,
                expert: expert as u32,
            };
            let [gate, down] = self.resident_prefill_matrices(source, id)?;
            let width = down.shape[1];
            if width == 0
                || gate.scale == 0.0
                || down.scale == 0.0
                || [hidden, width] != down.shape
                || [2 * width, hidden] != gate.shape
                || (middle != 0 && width != middle)
            {
                return Ok(None);
            }
            middle = width;
            matrices.insert(id, [gate, down]);
        }
        if dense_gemv_bytes(DenseGemvShape::packed(1, 2 * middle, hidden), dtypes[0]).is_err()
            || dense_gemv_bytes(DenseGemvShape::packed(1, hidden, middle), dtypes[1]).is_err()
        {
            return Ok(None);
        }
        Ok(Some(Projections {
            geometry: Geometry {
                hidden,
                middle,
                rank,
            },
            dtypes,
            caps,
            matrices,
        }))
    }
}

struct RoutedWindow {
    upload: Vec<[u8; 4]>,
    groups: Vec<(ExpertId, Range<usize>)>,
    layout: Layout,
    rows: usize,
}

impl RoutedWindow {
    fn prepare(
        projections: &Projections<'_>,
        layer: u32,
        routes: &[PrefillRoute<'_>],
        weight: impl Fn(usize, usize) -> f32,
    ) -> Result<Self, EngError> {
        let shape = projections.geometry;
        let layout = Layout::new(routes.len(), shape)
            .ok_or_else(|| bad("expert prefill arena overflows"))?;
        let mut upload = vec![[0; 4]; layout.flag.end / 4];
        for (dst, value) in upload[..layout.input.end / 4]
            .iter_mut()
            .zip(routes.iter().flat_map(|route| route.input))
        {
            *dst = value.to_ne_bytes();
        }
        let groups = pack_routes(
            &mut upload,
            &layout,
            shape.rank,
            group_routes(layer, routes),
            |id| {
                projections.matrices[&id]
                    .each_ref()
                    .map(|matrix| matrix.scale)
            },
            weight,
        )?;
        Ok(Self {
            upload,
            groups,
            layout,
            rows: routes.len(),
        })
    }
}

fn pack_routes(
    upload: &mut [[u8; 4]],
    layout: &Layout,
    rank: usize,
    groups: BTreeMap<ExpertId, Vec<(usize, usize)>>,
    scales: impl Fn(ExpertId) -> [f32; 2],
    weight: impl Fn(usize, usize) -> f32,
) -> Result<Vec<(ExpertId, Range<usize>)>, EngError> {
    let mut next = 0;
    let mut batches = Vec::with_capacity(groups.len());
    for (id, lanes) in groups {
        let first = next;
        let [gate, down] = scales(id);
        for (row, lane) in lanes {
            let original = row * rank + lane;
            let weight = weight(row, id.expert as usize);
            if !weight.is_finite() {
                return Err(bad("CUDA expert routing requires finite weights"));
            }
            upload[layout.row_ids.start / 4 + next] = (row as i32).to_ne_bytes();
            upload[layout.inverse_ids.start / 4 + original] = (next as i32).to_ne_bytes();
            upload[layout.weights.start / 4 + original] = weight.to_ne_bytes();
            upload[layout.gate_scales.start / 4 + next] = gate.to_ne_bytes();
            upload[layout.down_scales.start / 4 + next] = down.to_ne_bytes();
            next += 1;
        }
        batches.push((id, first..next));
    }
    Ok(batches)
}

impl Scratch {
    fn allocate(
        cuda: &CudaContext,
        rows: usize,
        geometry: Geometry,
        layout: &Layout,
    ) -> Result<Option<Self>, EngError> {
        let allocate = || -> Result<_, EngError> {
            let buffer = cuda.allocate(layout.down.end)?;
            if (cuda.memory_info()?.free as u64) < DEVICE_HEADROOM {
                // Driver allocation granularity can exceed the logical extent.
                return Ok(None);
            }
            Ok(Some(Self {
                buffer,
                stream: cuda.stream()?,
                host: cuda.pinned_buf(layout.output.end)?,
                rows,
                geometry,
            }))
        };
        match allocate() {
            Err(EngError::MemoryPlan(_)) => Ok(None),
            other => other,
        }
    }

    fn execute(
        &mut self,
        cuda: &CudaContext,
        projections: &Projections<'_>,
        plan: &RoutedWindow,
    ) -> Result<Vec<Vec<f32>>, EngError> {
        let l = &plan.layout;
        let upload = self.buffer.range(0, l.flag.end)?;
        self.host.write(0, plan.upload.as_flattened())?;
        cuda.copy_from_pinned(&self.stream, &mut self.host, 0, upload)?;
        let result = self.enqueue(cuda, projections, plan).and_then(|()| {
            // The status word and reduced rows are contiguous; stage them in one copy.
            let readback = self
                .buffer
                .range(l.flag.start, l.output.end - l.flag.start)?;
            cuda.copy_to_pinned(&self.stream, readback, &mut self.host, l.flag.start)
        });
        // A launch can be refused after the upload has been queued.
        let waited = self.host.synchronize(&self.stream);
        result?;
        waited?;
        let mut flag = [0; 4];
        self.host.read(l.flag.start, &mut flag)?;
        if u32::from_ne_bytes(flag) != 0 {
            return Err(bad(
                "CUDA expert prefill produced a nonfinite output or invalid route",
            ));
        }
        let mut out = vec![[0; 4]; l.output.len() / 4];
        self.host.read(l.output.start, out.as_flattened_mut())?;
        Ok(out
            .chunks_exact(projections.geometry.hidden)
            .map(|row| row.iter().copied().map(f32::from_ne_bytes).collect())
            .collect())
    }

    fn enqueue(
        &self,
        cuda: &CudaContext,
        projections: &Projections<'_>,
        plan: &RoutedWindow,
    ) -> Result<(), EngError> {
        let l = &plan.layout;
        let shape = projections.geometry;
        let count = plan.rows * shape.rank;
        let range = |span: &Range<usize>| self.buffer.range(span.start, span.len());
        let flag = range(&l.flag)?;
        cuda.row_gather(
            &self.stream,
            RowGatherCall {
                x: range(&l.input)?,
                indices: range(&l.row_ids)?,
                out: range(&l.gathered)?,
                flag,
                shape: RowShape {
                    rows: count,
                    columns: shape.hidden,
                },
                source_rows: plan.rows,
            },
        )?;
        self.project(cuda, projections, plan, 0)?;
        cuda.row_clipped_expert(
            &self.stream,
            ClippedExpertCall {
                x: range(&l.gate)?,
                scales: range(&l.gate_scales)?,
                out: range(&l.middle)?,
                flag,
                shape: RowShape {
                    rows: count,
                    columns: shape.middle,
                },
                caps: projections.caps,
            },
        )?;
        self.project(cuda, projections, plan, 1)?;
        cuda.row_indexed_rank_sum(
            &self.stream,
            IndexedRankSumCall {
                x: range(&l.down)?,
                inverse_ids: range(&l.inverse_ids)?,
                scales: range(&l.down_scales)?,
                weights: range(&l.weights)?,
                out: range(&l.output)?,
                flag,
                shape: RowShape {
                    rows: plan.rows,
                    columns: shape.hidden,
                },
                rank: shape.rank,
            },
        )
    }

    fn project(
        &self,
        cuda: &CudaContext,
        projections: &Projections<'_>,
        plan: &RoutedWindow,
        stage: usize,
    ) -> Result<(), EngError> {
        let l = &plan.layout;
        let pair = [(&l.gathered, &l.gate), (&l.middle, &l.down)][stage];
        let calls = plan
            .groups
            .iter()
            .map(|(id, slots)| {
                let matrix = &projections.matrices[id][stage];
                let [rows, cols] = matrix.shape;
                let span = |range: &Range<usize>, width| -> Result<DeviceRange<'_>, EngError> {
                    self.buffer.range(
                        range.start + slots.start * width * 4,
                        slots.len() * width * 4,
                    )
                };
                Ok(DenseGemvCall {
                    w: matrix.data,
                    x: span(pair.0, cols)?,
                    y: span(pair.1, rows)?,
                    shape: DenseGemvShape::packed(slots.len(), rows, cols),
                    dtype: projections.dtypes[stage],
                })
            })
            .collect::<Result<Vec<_>, EngError>>()?;
        cuda.dense_gemv_group(&self.stream, &calls)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_prompt_arena_fits_width_by_workspace_and_launch_bounds() {
        let shape = Geometry {
            hidden: 2304,
            middle: 512,
            rank: 8,
        };
        let single = Layout::new(1, shape).unwrap();
        let plan = Layout::new(512, shape).unwrap();
        assert_eq!(plan.input.len(), 512 * 2304 * 4);
        assert_eq!(plan.gate.len(), 512 * 8 * 2 * 512 * 4);
        assert_eq!(plan.down.end, 512 * (single.down.end - 4) + 4);
        assert_eq!(Layout::fit(512, shape, plan.down.end), Some(512));
        assert_eq!(Layout::fit(512, shape, plan.down.end - 1), Some(511));
        assert_eq!(Layout::fit(512, shape, single.down.end), None);
        assert_eq!(
            Layout::fit(usize::MAX, shape, usize::MAX),
            Some(DenseGemvShape::MAX_ACTIVATION_ROWS / 8)
        );
    }

    #[test]
    fn device_prompt_routes_keep_original_rank_and_expert_scale() {
        let shape = Geometry {
            hidden: 2,
            middle: 2,
            rank: 2,
        };
        let input = [0.0; 2];
        let routes = [
            PrefillRoute {
                ids: &[2, 0],
                input: &input,
            },
            PrefillRoute {
                ids: &[1, 2],
                input: &input,
            },
        ];
        let layout = Layout::new(routes.len(), shape).unwrap();
        let groups = group_routes(5, &routes);
        let mut upload = vec![[0; 4]; layout.flag.end / 4];
        let batches = pack_routes(
            &mut upload,
            &layout,
            shape.rank,
            groups,
            |id| {
                let expert = id.expert as f32;
                [expert + 2.0, expert + 3.0]
            },
            |row, id| (10 * (row + 1) + id) as f32,
        )
        .unwrap();
        let slot = |bytes: &Range<usize>, i| upload[bytes.start / 4 + i];
        let ints = |span: &Range<usize>| {
            std::array::from_fn::<_, 4, _>(|i| i32::from_ne_bytes(slot(span, i)))
        };
        let floats = |span: &Range<usize>| {
            std::array::from_fn::<_, 4, _>(|i| f32::from_ne_bytes(slot(span, i)))
        };
        assert_eq!(
            batches
                .iter()
                .map(|(id, slots)| (id.expert, slots.clone()))
                .collect::<Vec<_>>(),
            [(0, 0..1), (1, 1..2), (2, 2..4)]
        );
        assert_eq!(ints(&layout.row_ids), [0, 1, 0, 1]);
        assert_eq!(ints(&layout.inverse_ids), [2, 0, 1, 3]);
        assert_eq!(floats(&layout.gate_scales), [2.0, 3.0, 4.0, 4.0]);
        assert_eq!(floats(&layout.down_scales), [3.0, 4.0, 5.0, 5.0]);
        assert_eq!(floats(&layout.weights), [12.0, 10.0, 21.0, 22.0]);
    }
}
