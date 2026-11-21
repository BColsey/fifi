//! Import one selected expert set from bounded host pages and project in its stored format.

use std::{collections::VecDeque, ops::Range, sync::Arc};

use super::timing::{self, measure};
use super::{
    EXPERTS, HIDDEN, InklingExpertPlan, InklingExpertProjection, InklingExpertSource,
    InklingRoutes, LAYERS, TOP_K,
};
use crate::{
    EngError,
    kernels::{
        BoundedIndices, HostPages, MlxKernels, PackedPages, PageRange, QuantizationMode,
        QuantizationSpec, ResidentPackedWeight,
    },
    tensor::{Dtype, ExecutionTensor, OwnerKey, Tensor},
};

const MAX_ROUTED_ROWS: usize = u32::MAX as usize / HIDDEN / TOP_K;

/// One resident router bound to the packed experts of its physical layer.
pub struct MlxRouter<'a> {
    source: &'a InklingExpertSource,
    layer: usize,
    weight: ExecutionTensor,
    bias: [f32; EXPERTS],
}

impl<'a> MlxRouter<'a> {
    /// Reads the checkpoint's BF16 gate matrix and F32 selection bias once.
    /// Widen and transpose at load so successive visits reuse the F32 matrix.
    pub fn load(
        kernels: &mut MlxKernels,
        source: &'a InklingExpertSource,
        layer: usize,
    ) -> Result<Self, EngError> {
        kernels.check_ready()?;
        let stored = source.read_router(layer)?;
        let weight = kernels.import_tensor(&router_matrix(&stored.weight)?)?;
        Ok(Self {
            source,
            layer,
            weight,
            bias: stored.bias,
        })
    }

    /// Projects normalized F32 [tokens, hidden] inputs to unbiased F32 logits.
    pub fn logits(
        &self,
        kernels: &mut MlxKernels,
        input: &ExecutionTensor,
    ) -> Result<ExecutionTensor, EngError> {
        let &[rows, width] = input.shape() else {
            return Err(bad_router_input());
        };
        if input.dtype() != Dtype::F32
            || width != HIDDEN as u32
            || !(1..=MAX_ROUTED_ROWS).contains(&(rows as usize))
        {
            return Err(bad_router_input());
        }
        kernels.mm(input, &self.weight)
    }

    /// Reads back only the F32 logits. Bias affects expert selection separately
    /// from the unbiased route weights.
    pub fn select(
        &self,
        kernels: &mut MlxKernels,
        input: &ExecutionTensor,
    ) -> Result<InklingRoutes, EngError> {
        let projected = self.logits(kernels, input)?;
        let logits = kernels.readback(&projected)?;
        let bytes = logits.host_bytes().ok_or_else(|| EngError::Backend {
            operation: "Inkling router",
            message: "logits readback returned no host bytes".into(),
        })?;
        let (rows, trailing) = bytes.as_chunks::<{ EXPERTS * 4 }>();
        if !trailing.is_empty() || rows.len() != input.shape()[0] as usize {
            return Err(EngError::Backend {
                operation: "Inkling router",
                message: "logits readback has the wrong size".into(),
            });
        }
        let rows = rows.iter().map(|row| {
            let words = row.as_chunks::<4>().0;
            std::array::from_fn(|expert| f32::from_le_bytes(words[expert]))
        });
        InklingRoutes::from_rows(rows, &self.bias)
    }

    /// Prices the selected expert stage on the same checkpoint and weight layer.
    pub fn plan<'plan>(
        &'plan self,
        routes: &'plan InklingRoutes,
    ) -> Result<MlxRoutedExpertPlan<'plan>, EngError> {
        MlxRoutedExpertPlan::new(self.source, self.layer, routes)
    }
}

fn bad_router_input() -> EngError {
    EngError::BadRequest(
        "Inkling router requires a nonempty F32 [tokens, hidden] batch within Metal limits".into(),
    )
}

fn router_matrix(stored: &[u8]) -> Result<Tensor, EngError> {
    if stored.len() != EXPERTS * HIDDEN * 2 {
        return Err(EngError::Checkpoint(
            "Inkling BF16 router payload has the wrong size".into(),
        ));
    }
    let mut transposed = vec![0; HIDDEN * EXPERTS * 4];
    for (expert, row) in stored.chunks_exact(HIDDEN * 2).enumerate() {
        for (feature, &word) in row.as_chunks::<2>().0.iter().enumerate() {
            if u16::from_le_bytes(word) & 0x7f80 == 0x7f80 {
                return Err(EngError::Checkpoint(
                    "Inkling BF16 router weights must be finite".into(),
                ));
            }
            let offset = (feature * EXPERTS + expert) * 4;
            transposed[offset + 2..offset + 4].copy_from_slice(&word);
        }
    }
    Tensor::try_from_host(Dtype::F32, vec![HIDDEN as u32, EXPERTS as u32], transposed)
}

/// The selected experts for both projections of one physical layer.
pub struct MlxExpertStage {
    gate_up: Projection,
    down: Projection,
}

/// The disk ranges and page capacity for one token-major routed batch.
pub struct MlxRoutedExpertPlan<'a> {
    experts: InklingExpertPlan<'a>,
    routes: &'a InklingRoutes,
    routing: RoutePages,
}

impl<'a> MlxRoutedExpertPlan<'a> {
    /// Plans the selected weights and their compact route metadata before any I/O.
    pub fn new(
        source: &'a InklingExpertSource,
        layer: usize,
        routes: &'a InklingRoutes,
    ) -> Result<Self, EngError> {
        let page_size = HostPages::page_size();
        RoutePages::validate(routes, page_size)?;
        let experts = source.plan(layer, routes.expert_ids(), page_size)?;
        Self::from_experts(experts, routes, page_size)
    }

    fn from_experts(
        experts: InklingExpertPlan<'a>,
        routes: &'a InklingRoutes,
        page_size: usize,
    ) -> Result<Self, EngError> {
        let routing = RoutePages::new(routes, experts.staging_bytes(), page_size)?;
        Ok(Self {
            experts,
            routes,
            routing,
        })
    }

    pub fn staging_bytes(&self) -> usize {
        self.routing.end
    }

    pub fn expert_plan(&self) -> &InklingExpertPlan<'_> {
        &self.experts
    }
}

struct RoutePages {
    ids_offset: usize,
    weight_offset: usize,
    end: usize,
}

impl RoutePages {
    fn validate(routes: &InklingRoutes, page_size: usize) -> Result<(), EngError> {
        // Both packed projections have HIDDEN output values per route.
        if !(1..=MAX_ROUTED_ROWS).contains(&routes.rows()) {
            return Err(EngError::BadRequest(
                "Inkling Metal staging requires a nonempty batch within native index limits".into(),
            ));
        }
        if page_size == 0 {
            return Err(EngError::Backend {
                operation: "Inkling route staging",
                message: "the OS page size query failed".into(),
            });
        }
        Ok(())
    }

    fn new(routes: &InklingRoutes, start: usize, page_size: usize) -> Result<Self, EngError> {
        let [ids_offset, weight_offset, end] =
            super::route_page_offsets(start, routes.expert_ids().len(), page_size)?;
        Ok(Self {
            ids_offset,
            weight_offset,
            end,
        })
    }

    fn write(
        &self,
        pages: &mut [u8],
        routes: &InklingRoutes,
        slots: impl IntoIterator<Item = u32>,
    ) -> [PageRange; 2] {
        let count = routes.expert_ids().len();
        let ids = &mut pages[self.ids_offset..self.ids_offset + count * 4];
        for (word, slot) in ids.as_chunks_mut::<4>().0.iter_mut().zip(slots) {
            *word = slot.to_le_bytes();
        }
        for (word, weight) in pages[self.weight_offset..self.end]
            .as_chunks_mut::<4>()
            .0
            .iter_mut()
            .zip(routes.weights())
        {
            *word = weight.to_le_bytes();
        }
        [
            PageRange {
                offset: self.ids_offset,
                dtype: Dtype::U32,
                shape: vec![count as u32],
            },
            PageRange {
                offset: self.weight_offset,
                dtype: Dtype::F32,
                shape: vec![count as u32],
            },
        ]
    }
}

/// One staged weight-only routed expert block. Route metadata stays on the
/// same pages as the weights. The block accepts F32 input and returns F32
/// storage with BF16-rounded values at the checkpoint's projection boundaries.
pub struct MlxRoutedExpertStage {
    experts: MlxExpertStage,
    routing: StagedRoutes,
}

struct StagedRoutes {
    ids: BoundedIndices,
    weights: ExecutionTensor,
    rows: u32,
}

impl StagedRoutes {
    fn new(
        routing: Vec<ExecutionTensor>,
        routes: &InklingRoutes,
        expert_count: u32,
    ) -> Result<Self, EngError> {
        let [ids, weights] = routing.try_into().map_err(|_| EngError::Backend {
            operation: "Inkling expert staging",
            message: "native import returned a different route tensor count".into(),
        })?;
        Ok(Self {
            ids: BoundedIndices::from_clamped(ids, expert_count),
            weights,
            rows: routes.rows() as u32,
        })
    }
}

impl MlxRoutedExpertStage {
    /// Imports packed weights and route metadata together without synchronization.
    /// Keep the page pins until all consuming graphs have finished. Drop the
    /// stage, release its pins, and synchronize before reusing the arena.
    pub fn read(
        kernels: &mut MlxKernels,
        plan: &MlxRoutedExpertPlan<'_>,
        pages: &mut HostPages,
    ) -> Result<Self, EngError> {
        Self::read_with_fill(kernels, plan, pages, |bytes| plan.experts.read_into(bytes))
    }

    fn read_with_fill(
        kernels: &mut MlxKernels,
        plan: &MlxRoutedExpertPlan<'_>,
        pages: &mut HostPages,
        fill: impl FnOnce(&mut [u8]) -> Result<(), EngError>,
    ) -> Result<Self, EngError> {
        let (experts, routing) =
            MlxExpertStage::read_pages(kernels, &plan.experts, Some(plan), pages, fill)?;
        let routing =
            StagedRoutes::new(routing, plan.routes, plan.experts.gate_up().shape[0] as u32)?;
        Ok(Self { experts, routing })
    }

    pub fn forward(
        &self,
        kernels: &mut MlxKernels,
        input: &ExecutionTensor,
    ) -> Result<ExecutionTensor, EngError> {
        self.experts
            .forward_routed(kernels, input, &self.routing, false)
    }

    /// Applies the layer's activation quantizer at both expert projections.
    /// The Metal kernels reconstruct F32 values for the packed-weight gather.
    /// Routing still uses the unquantized input supplied to the router.
    pub fn forward_quantized(
        &self,
        kernels: &mut MlxKernels,
        input: &ExecutionTensor,
    ) -> Result<ExecutionTensor, EngError> {
        self.experts
            .forward_routed(kernels, input, &self.routing, true)
    }
}

fn validate_routed_input(input: &ExecutionTensor, rows: u32) -> Result<(), EngError> {
    if input.dtype() != Dtype::F32 || input.shape() != [rows, HIDDEN as u32] {
        return Err(EngError::BadRequest(
            "Inkling routed expert input must be F32 [tokens, hidden] for the staged routes".into(),
        ));
    }
    Ok(())
}

impl MlxExpertStage {
    fn forward_routed(
        &self,
        kernels: &mut MlxKernels,
        input: &ExecutionTensor,
        routing: &StagedRoutes,
        quantized: bool,
    ) -> Result<ExecutionTensor, EngError> {
        let width = HIDDEN as u32;
        validate_routed_input(input, routing.rows)?;
        let quantization = quantized.then(|| self.gate_up.weight.quantization().mode);
        let input = match quantization {
            Some(mode) => kernels.inkling_quantized_rows(input, mode)?,
            None => input.share(),
        };
        let input = if routing.rows == 1 {
            input
        } else {
            kernels.repeat(&input, TOP_K as u32, 0)?
        };
        let input = kernels.reshape(&input, &[input.shape()[0], 1, width])?;
        let gate_up = self.project_gate_up(kernels, &input, &routing.ids)?;
        let activation = kernels.inkling_activation(&gate_up, &routing.weights, quantization)?;
        let down = self.project_down(kernels, &activation, &routing.ids)?;
        kernels.inkling_merge(&down, routing.rows, TOP_K as u32)
    }

    /// Reads directly into reusable pages and imports both packed projections together.
    /// Create the plan with `HostPages::page_size()` alignment. Use F32 inputs for
    /// scaled NVFP4 projections. The caller maps routes with the plan's compact slots.
    /// Keep the page pins until all consuming graphs have finished. Drop the stage,
    /// release the pins, and synchronize before reusing the pages.
    pub fn read(
        kernels: &mut MlxKernels,
        plan: &InklingExpertPlan<'_>,
        pages: &mut HostPages,
    ) -> Result<Self, EngError> {
        Ok(Self::read_pages(kernels, plan, None, pages, |bytes| plan.read_into(bytes))?.0)
    }

    fn read_pages(
        kernels: &mut MlxKernels,
        plan: &InklingExpertPlan<'_>,
        routes: Option<&MlxRoutedExpertPlan<'_>>,
        pages: &mut HostPages,
        fill: impl FnOnce(&mut [u8]) -> Result<(), EngError>,
    ) -> Result<(Self, Vec<ExecutionTensor>), EngError> {
        kernels.check_ready()?;
        let page_size = HostPages::page_size();
        if page_size == 0 || !plan.alignment().is_multiple_of(page_size) {
            return Err(EngError::BadRequest(
                "Inkling MLX expert staging requires page-aligned tensor offsets".into(),
            ));
        }
        let projections = [plan.gate_up(), plan.down()];
        let stages = projections.map(packed_pages);
        let bytes = pages.try_write().ok_or_else(|| EngError::Backend {
            operation: "Inkling expert staging",
            message: "host pages are still in use by a previous stage".into(),
        })?;
        if let Some(routes) = routes
            && bytes.len() < routes.staging_bytes()
        {
            return Err(EngError::BadRequest(
                "Inkling expert and route metadata exceed the staging arena".into(),
            ));
        }
        fill(bytes)?;
        let mut scale_ranges = Vec::new();
        for projection in projections {
            if let Some(range) = normalize_multipliers(bytes, projection)? {
                scale_ranges.push(range);
            }
        }
        let scale_count = scale_ranges.len();
        if let Some(routes) = routes {
            scale_ranges.extend(routes.routing.write(
                bytes,
                routes.routes,
                routes.experts.compact_slots().iter().copied(),
            ));
        }
        let (weights, mut multipliers) =
            kernels.import_packed_pages(pages, &stages, &scale_ranges)?;
        let [gate_up, down] = weights.try_into().map_err(|_| EngError::Backend {
            operation: "Inkling expert staging",
            message: "native import returned a different packed tensor count".into(),
        })?;
        // The two projections of a layer share their format.
        let routing = multipliers.split_off(scale_count);
        let mut multipliers = multipliers.into_iter();
        let stage = Self {
            gate_up: Projection {
                weight: gate_up,
                multipliers: multipliers.next(),
            },
            down: Projection {
                weight: down,
                multipliers: multipliers.next(),
            },
        };
        Ok((stage, routing))
    }

    pub fn project_gate_up(
        &self,
        kernels: &mut MlxKernels,
        input: &ExecutionTensor,
        ids: &BoundedIndices,
    ) -> Result<ExecutionTensor, EngError> {
        self.gate_up.run(kernels, input, ids)
    }

    pub fn project_down(
        &self,
        kernels: &mut MlxKernels,
        input: &ExecutionTensor,
        ids: &BoundedIndices,
    ) -> Result<ExecutionTensor, EngError> {
        self.down.run(kernels, input, ids)
    }
}

/// Counters for visits through an opt-in Inkling expert cache.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct MlxExpertCacheStats {
    pub visits: u64,
    pub fully_cached_visits: u64,
    /// Hits, misses and evictions count distinct experts, not routed picks.
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    /// Logical shard bytes and ranges successfully read into staging.
    pub read_bytes: u64,
    pub read_ranges: u64,
    pub staging_allocations: u64,
    pub staging_fallback_syncs: u64,
    pub cached_bytes: usize,
}

#[derive(Default, serde::Serialize)]
pub(super) struct ExpertTimings {
    metal_input_import_ns: u128,
    staging_arena_allocation_ns: u128,
    staging_ns: u128,
    demand_read_ns: u128,
    metal_dispatch_ns: u128,
    metal_readback_ns: u128,
    metal_fallback_sync_ns: u128,
}

/// Byte-bounded LRU of packed checkpoint payloads indexed by physical layer and expert.
/// On each visit, a single Metal staging arena receives cached slabs and coalesced
/// missing reads in compact order. The budget covers retained raw expert payloads.
pub struct MlxExpertCache {
    source: Arc<InklingExpertSource>,
    owner: OwnerKey,
    max_bytes: usize,
    entries: VecDeque<CachedExpert>,
    denominators: [Option<[Option<[u8; 4]>; 2]>; LAYERS],
    staging: Option<HostPages>,
    staging_rows: usize,
    trace_routes: bool,
    timing: Option<ExpertTimings>,
    stats: MlxExpertCacheStats,
}

struct CachedExpert {
    layer: usize,
    expert: usize,
    bytes: Box<[u8]>,
}

impl MlxExpertCache {
    pub fn new(kernels: &MlxKernels, source: Arc<InklingExpertSource>, max_bytes: usize) -> Self {
        Self {
            source,
            owner: kernels.owner_key(),
            max_bytes,
            entries: VecDeque::new(),
            denominators: [None; LAYERS],
            staging: None,
            staging_rows: 1,
            trace_routes: std::env::var_os("FIFI_INKLING_ROUTE_TRACE").as_deref()
                == Some(std::ffi::OsStr::new("1")),
            timing: timing::enabled().then(ExpertTimings::default),
            stats: MlxExpertCacheStats::default(),
        }
    }

    /// Use the working-set headroom already available to the expert cache.
    /// The single-token arena was charged by the base plan.
    pub(super) fn configure_prefill(&mut self, requested: usize) -> Result<usize, EngError> {
        let mut staging = None;
        let (rows, extra) = self.source.reserve_prefill_window(
            requested.min(MAX_ROUTED_ROWS),
            HostPages::page_size(),
            self.max_bytes,
            |stage_bytes| match HostPages::try_new(stage_bytes) {
                Ok(pages) => {
                    staging = Some(pages);
                    Ok(true)
                }
                Err(EngError::MemoryPlan(_)) => Ok(false),
                Err(error) => Err(error),
            },
        )?;
        if let Some(pages) = staging {
            self.staging = Some(pages);
            self.staging_rows = rows;
            self.max_bytes -= extra;
            self.stats.staging_allocations += 1;
        }
        Ok(rows)
    }

    pub(super) fn capacity_bytes(&self) -> usize {
        self.max_bytes
    }

    pub(super) fn take_component_timings(&mut self) -> Option<ExpertTimings> {
        self.timing.as_mut().map(std::mem::take)
    }

    pub fn stats(&self) -> MlxExpertCacheStats {
        self.stats
    }

    pub fn forward(
        &mut self,
        kernels: &mut MlxKernels,
        layer: usize,
        routes: &InklingRoutes,
        input: &ExecutionTensor,
    ) -> Result<ExecutionTensor, EngError> {
        self.forward_impl(kernels, layer, routes, input, false, None)
    }

    pub fn forward_quantized(
        &mut self,
        kernels: &mut MlxKernels,
        layer: usize,
        routes: &InklingRoutes,
        input: &ExecutionTensor,
    ) -> Result<ExecutionTensor, EngError> {
        self.forward_impl(kernels, layer, routes, input, true, None)
    }

    /// Runs routed rows to completion using a shared arena for consecutive calls.
    /// The next input import synchronizes the device and flushes the prior visit's
    /// retired page handles, so arena reuse ordinarily adds no synchronization.
    pub fn forward_quantized_readback(
        &mut self,
        kernels: &mut MlxKernels,
        layer: usize,
        routes: &InklingRoutes,
        input: &Tensor,
    ) -> Result<Tensor, EngError> {
        RoutePages::validate(routes, HostPages::page_size())?;
        if routes.rows() > self.staging_rows {
            return Err(EngError::BadRequest(
                "Inkling routed rows exceed the admitted staging window".into(),
            ));
        }
        if input.dtype() != Dtype::F32 || input.shape() != [routes.rows() as u32, HIDDEN as u32] {
            return Err(EngError::BadRequest(
                "Inkling reusable staging requires F32 input matching the routed rows".into(),
            ));
        }
        if kernels.owner_key() != self.owner {
            return Err(EngError::BadRequest(
                "Inkling cache must belong to the active Metal owner".into(),
            ));
        }
        let input = measure(
            self.timing.as_mut().map(|t| &mut t.metal_input_import_ns),
            || kernels.import_tensor(input),
        )?;
        let mut pages = match self.staging.take() {
            Some(pages) => pages,
            None => {
                let pages = measure(
                    self.timing
                        .as_mut()
                        .map(|t| &mut t.staging_arena_allocation_ns),
                    || {
                        let bytes = self
                            .source
                            .prefill_staging_bytes(self.staging_rows, HostPages::page_size())?;
                        HostPages::try_new(bytes)
                    },
                )?;
                self.stats.staging_allocations += 1;
                pages
            }
        };
        if pages.in_use() {
            self.stats.staging_fallback_syncs += 1;
            measure(
                self.timing.as_mut().map(|t| &mut t.metal_fallback_sync_ns),
                || kernels.evaluate(&[&input]),
            )?;
            if pages.in_use() {
                return Err(EngError::Backend {
                    operation: "Inkling expert staging",
                    message: "previous visit still holds the staging arena after synchronization"
                        .into(),
                });
            }
        }
        let output = self.forward_impl(kernels, layer, routes, &input, true, Some(&mut pages))?;
        let result = measure(
            self.timing.as_mut().map(|t| &mut t.metal_readback_ns),
            || kernels.readback(&output),
        )?;
        drop(output);
        pages.release();
        self.staging = Some(pages);
        Ok(result)
    }

    fn forward_impl(
        &mut self,
        kernels: &mut MlxKernels,
        layer: usize,
        routes: &InklingRoutes,
        input: &ExecutionTensor,
        quantized: bool,
        pages: Option<&mut HostPages>,
    ) -> Result<ExecutionTensor, EngError> {
        kernels.check_ready()?;
        if kernels.owner_key() != self.owner || input.owner_key() != self.owner {
            return Err(EngError::BadRequest(
                "Inkling cache and input must belong to the active Metal owner".into(),
            ));
        }
        let staging_start = self.timing.is_some().then(std::time::Instant::now);
        let page_size = HostPages::page_size();
        RoutePages::validate(routes, page_size)?;
        validate_routed_input(input, routes.rows() as u32)?;
        let selected = routes
            .expert_ids()
            .iter()
            .fold(0_u128, |set, &expert| set | (1_u128 << expert));
        let cached = self
            .entries
            .iter()
            .filter(|entry| entry.layer == layer)
            .fold(0_u128, |set, entry| set | (1_u128 << entry.expert));
        let misses = selected & !cached;
        let is_missing = |expert| misses & (1_u128 << expert) != 0;
        let shared = self.denominators.get(layer).copied().flatten();
        let source = Arc::clone(&self.source);
        let plan = source.plan_with_missing(
            layer,
            routes.expert_ids(),
            page_size,
            is_missing,
            shared.is_none(),
        )?;
        let plan = MlxRoutedExpertPlan::from_experts(plan, routes, page_size)?;
        let mut temporary;
        let pages = match pages {
            Some(pages) => pages,
            None => {
                temporary = HostPages::try_new(plan.staging_bytes())?;
                self.stats.staging_allocations += 1;
                &mut temporary
            }
        };
        self.stats.visits += 1;
        self.stats.fully_cached_visits += u64::from(misses == 0);
        self.stats.hits += u64::from((selected & cached).count_ones());
        self.stats.misses += u64::from(misses.count_ones());
        let stage = MlxRoutedExpertStage::read_with_fill(kernels, &plan, pages, |bytes| {
            let plan = plan.expert_plan();
            let read_start = self.trace_routes.then(std::time::Instant::now);
            if plan.read_count() > 0 {
                measure(self.timing.as_mut().map(|t| &mut t.demand_read_ns), || {
                    plan.read_into(bytes)
                })?;
                self.stats.read_bytes += plan.read_bytes() as u64;
                self.stats.read_ranges += plan.read_count() as u64;
            }

            if let Some(start) = read_start {
                let read_us = start.elapsed().as_micros();
                let ids = |mask: u128| {
                    (0..EXPERTS)
                        .filter(|id| mask & (1_u128 << id) != 0)
                        .collect::<Vec<_>>()
                };
                eprintln!(
                    "inkling_route {}",
                    serde_json::json!({
                        "visit": self.stats.visits, "layer": layer, "rows": routes.rows(),
                        "selected": ids(selected), "missing": ids(misses),
                        "expert_bytes": expert_ranges(plan, 0).map(|r| r.len()).sum::<usize>(),
                        "read_bytes": plan.read_bytes(), "read_us": read_us,
                        "capacity_bytes": self.max_bytes,
                    })
                );
            }

            let denominator_ranges = [plan.gate_up(), plan.down()].map(|p| p.denominator.clone());
            if let Some(shared) = shared {
                for (range, value) in denominator_ranges.into_iter().zip(shared) {
                    if let (Some(range), Some(value)) = (range, value) {
                        bytes[range].copy_from_slice(&value);
                    }
                }
            } else {
                self.denominators[layer] = Some(
                    denominator_ranges
                        .map(|range| range.map(|range| bytes[range].try_into().unwrap())),
                );
            }

            let experts = || {
                (0..EXPERTS)
                    .filter(|expert| selected & (1_u128 << expert) != 0)
                    .enumerate()
            };
            for (slot, expert) in experts().filter(|&(_, expert)| !is_missing(expert)) {
                let position = self
                    .entries
                    .iter()
                    .position(|entry| entry.layer == layer && entry.expert == expert)
                    .unwrap();
                let entry = self.entries.remove(position).unwrap();
                let mut stored = entry.bytes.as_ref();
                for destination in expert_ranges(plan, slot) {
                    let (part, rest) = stored.split_at(destination.len());
                    bytes[destination].copy_from_slice(part);
                    stored = rest;
                }
                self.entries.push_back(entry);
            }
            for (slot, expert) in experts().filter(|&(_, expert)| is_missing(expert)) {
                self.retain_expert(layer, expert, plan, slot, bytes)?;
            }
            Ok(())
        });
        if let Some((timing, start)) = self.timing.as_mut().zip(staging_start) {
            timing.staging_ns += start.elapsed().as_nanos();
        }
        let stage = stage?;
        measure(
            self.timing.as_mut().map(|t| &mut t.metal_dispatch_ns),
            || {
                stage
                    .experts
                    .forward_routed(kernels, input, &stage.routing, quantized)
            },
        )
    }

    fn retain_expert(
        &mut self,
        layer: usize,
        expert: usize,
        plan: &InklingExpertPlan<'_>,
        slot: usize,
        staging: &[u8],
    ) -> Result<(), EngError> {
        let len: usize = expert_ranges(plan, slot).map(|range| range.len()).sum();
        if len > self.max_bytes {
            return Ok(());
        }
        while self.stats.cached_bytes > self.max_bytes - len {
            let entry = self.entries.pop_front().unwrap();
            self.stats.cached_bytes -= entry.bytes.len();
            self.stats.evictions += 1;
        }
        let mut bytes = Vec::new();
        bytes.try_reserve_exact(len).map_err(|_| {
            EngError::MemoryPlan("Inkling packed expert cache allocation failed".into())
        })?;
        for source in expert_ranges(plan, slot) {
            bytes.extend_from_slice(&staging[source]);
        }
        self.stats.cached_bytes += len;
        self.entries.push_back(CachedExpert {
            layer,
            expert,
            bytes: bytes.into_boxed_slice(),
        });
        Ok(())
    }
}

fn expert_ranges(plan: &InklingExpertPlan<'_>, slot: usize) -> impl Iterator<Item = Range<usize>> {
    [plan.gate_up(), plan.down()]
        .into_iter()
        .flat_map(move |p| {
            [Some(&p.weight), Some(&p.scale), p.amax.as_ref()]
                .into_iter()
                .flatten()
                .map(move |range| {
                    let stride = range.len() / p.shape[0];
                    let start = range.start + slot * stride;
                    start..start + stride
                })
        })
}

struct Projection {
    weight: ResidentPackedWeight,
    multipliers: Option<ExecutionTensor>,
}

impl Projection {
    fn run(
        &self,
        kernels: &mut MlxKernels,
        input: &ExecutionTensor,
        ids: &BoundedIndices,
    ) -> Result<ExecutionTensor, EngError> {
        match &self.multipliers {
            Some(scales) => kernels.gather_nvfp4(input, &self.weight, ids, scales),
            None => kernels.gather_qmm_bounded(input, &self.weight, ids),
        }
    }
}

fn packed_pages(projection: &InklingExpertProjection) -> PackedPages {
    let [experts, rows, columns] = projection.shape.map(|dim| dim as u32);
    let (mode, bits, group_size) = if projection.amax.is_some() {
        (QuantizationMode::Nvfp4, 4, 16)
    } else {
        (QuantizationMode::Mxfp8, 8, 32)
    };
    PackedPages {
        payload: PageRange {
            offset: projection.weight.start,
            dtype: Dtype::U32,
            shape: vec![experts, rows, columns * bits / 32],
        },
        scales: PageRange {
            offset: projection.scale.start,
            dtype: Dtype::U8,
            shape: vec![experts, rows, columns / group_size],
        },
        biases: None,
        logical_shape: vec![experts, rows, columns],
        quantization: QuantizationSpec {
            mode,
            bits,
            group_size,
        },
    }
}

fn normalize_multipliers(
    bytes: &mut [u8],
    projection: &InklingExpertProjection,
) -> Result<Option<PageRange>, EngError> {
    let (Some(amax), Some(denominator)) = (&projection.amax, &projection.denominator) else {
        return Ok(None);
    };
    let invalid = || {
        EngError::Checkpoint(
            "Inkling NVFP4 outer scales must be finite and nonnegative with a positive denominator"
                .into(),
        )
    };
    let denominator = f32::from_le_bytes(bytes[denominator.clone()].try_into().unwrap());
    if !denominator.is_finite() || denominator <= 0.0 {
        return Err(invalid());
    }
    // The amax slot already has the required alignment and shape. Reuse it as
    // the native multiplier tensor without making another page or read.
    for encoded in bytes[amax.clone()].as_chunks_mut::<4>().0 {
        let amax = f32::from_le_bytes(*encoded);
        let scale = amax / denominator;
        if amax < 0.0 || !scale.is_finite() {
            return Err(invalid());
        }
        *encoded = scale.to_le_bytes();
    }
    Ok(Some(PageRange {
        offset: amax.start,
        dtype: Dtype::F32,
        shape: vec![projection.shape[0] as u32],
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::PackedWeight;

    #[test]
    fn resident_expert_cache_uses_fresh_routes_and_evicts_by_layer() {
        use std::io::Write;

        use crate::family::inkling2_mini::{ExpertRegion, ProjectionRegions, WeightPlan};

        let _lock = crate::native_test_lock();
        let mut shard = tempfile::tempfile().unwrap();
        let mut offset = 0_u64;
        let mut tensor = |stride: usize, values: &[u8; 2]| {
            let start = offset;
            for expert in 0..EXPERTS {
                shard.write_all(&vec![values[expert % 2]; stride]).unwrap();
            }
            offset += (EXPERTS * stride) as u64;
            ExpertRegion::Slices {
                start,
                stride: stride as u64,
            }
        };
        let mut projection = |rows: usize, cols: usize| ProjectionRegions {
            rows,
            cols,
            weight: tensor(rows * cols / 2, &[0x22, 0x44]),
            scale: tensor(rows * cols / 16, &[0x38, 0x38]),
            // 0x3f3f3f3f encodes the same positive finite F32 in every slot.
            amax: Some(tensor(4, &[0x3f, 0x3f])),
            denominator: None,
        };
        let mut layers: Vec<_> = (0..2)
            .map(|_| [projection(128, HIDDEN), projection(HIDDEN, 64)])
            .collect();
        let denominator = 1.0_f32.to_le_bytes();
        shard.write_all(&denominator).unwrap();
        for layer in &mut layers {
            for projection in layer {
                projection.denominator = Some(ExpertRegion::Shared(offset..offset + 4));
            }
        }
        let source = Arc::new(InklingExpertSource {
            shard,
            len: offset + 4,
            layers,
            routers: Vec::new(),
            weights: WeightPlan::default(),
        });
        let routes = |offset: usize, reverse: bool| {
            let mut row = [-20.0; EXPERTS];
            for i in 0..TOP_K {
                row[offset + i] = if reverse {
                    i as f32 / 4.0
                } else {
                    -(i as f32) / 2.0
                };
            }
            InklingRoutes::from_logits(&[row], &[0.0; EXPERTS]).unwrap()
        };
        let first = routes(0, false);
        let next = routes(0, true);
        let changed_set = routes(1, false);
        assert_ne!(first.expert_ids(), next.expert_ids());

        let mut kernels = MlxKernels::try_new().unwrap();
        let host_input = Tensor::try_from_host(
            Dtype::F32,
            vec![1, HIDDEN as u32],
            std::iter::once(1.0_f32)
                .chain(std::iter::repeat_n(0.0, HIDDEN - 1))
                .flat_map(f32::to_le_bytes)
                .collect(),
        )
        .unwrap();
        let input = kernels.import_tensor(&host_input).unwrap();
        let page = HostPages::page_size();
        let full = source.plan(0, first.expert_ids(), page).unwrap();
        let new_expert = source.plan(0, &[TOP_K], page).unwrap();
        let shared = 2 * size_of::<f32>();
        let capacity = full.read_bytes() - shared;
        let mut cache = MlxExpertCache::new(&kernels, Arc::clone(&source), capacity);

        // Build all four graphs before readback, including graphs built before eviction.
        let previous = cache.forward(&mut kernels, 0, &first, &input).unwrap();
        let hit = cache.forward(&mut kernels, 0, &next, &input).unwrap();
        let before = cache.stats();
        let partial = cache
            .forward(&mut kernels, 0, &changed_set, &input)
            .unwrap();
        let after = cache.stats();
        assert_eq!(after.hits - before.hits, 7);
        assert_eq!(after.misses - before.misses, 1);
        assert_eq!(
            after.read_bytes - before.read_bytes,
            (new_expert.read_bytes() - shared) as u64
        );
        assert_eq!(
            after.read_ranges - before.read_ranges,
            (new_expert.read_count() - 2) as u64
        );

        let different_layer = cache
            .forward(&mut kernels, 1, &changed_set, &input)
            .unwrap();
        assert_eq!(
            cache.stats(),
            MlxExpertCacheStats {
                visits: 4,
                fully_cached_visits: 1,
                hits: 15,
                misses: 17,
                evictions: 9,
                read_bytes: (full.read_bytes() * 2 + new_expert.read_bytes() - shared) as u64,
                read_ranges: (full.read_count() * 2 + new_expert.read_count() - 2) as u64,
                staging_allocations: 4,
                staging_fallback_syncs: 0,
                cached_bytes: capacity,
            },
        );

        let bytes = |kernels: &mut MlxKernels, output: &ExecutionTensor| {
            kernels
                .readback(output)
                .unwrap()
                .host_bytes()
                .unwrap()
                .to_vec()
        };
        let control = |kernels: &mut MlxKernels, routes| {
            let plan = MlxRoutedExpertPlan::new(&source, 0, routes).unwrap();
            let mut pages = HostPages::try_new(plan.staging_bytes()).unwrap();
            let stage = MlxRoutedExpertStage::read(kernels, &plan, &mut pages).unwrap();
            let result = stage.forward(kernels, &input).unwrap();
            bytes(kernels, &result)
        };
        let expected = control(&mut kernels, &next);
        let expected_partial = control(&mut kernels, &changed_set);
        assert_eq!(bytes(&mut kernels, &hit), expected);
        assert_eq!(bytes(&mut kernels, &partial), expected_partial);
        assert_eq!(bytes(&mut kernels, &different_layer), expected_partial);
        assert_ne!(bytes(&mut kernels, &previous), expected);
        assert_ne!(expected_partial, expected);

        let mut disabled = MlxExpertCache::new(&kernels, Arc::clone(&source), capacity / TOP_K - 1);
        let bypassed = disabled.forward(&mut kernels, 0, &next, &input).unwrap();
        assert_eq!(disabled.stats().misses, TOP_K as u64);
        assert_eq!(disabled.stats().read_bytes, full.read_bytes() as u64);
        assert_eq!(disabled.stats().read_ranges, full.read_count() as u64);
        assert_eq!(disabled.stats().cached_bytes, 0);
        assert_eq!(bytes(&mut kernels, &bypassed), expected);
        drop((previous, hit, partial, different_layer, bypassed));

        let mut reuse = MlxExpertCache::new(&kernels, Arc::clone(&source), capacity);
        for routes in [&first, &next, &changed_set] {
            let refreshed = reuse
                .forward_quantized_readback(&mut kernels, 0, routes, &host_input)
                .unwrap();
            let layout = MlxRoutedExpertPlan::new(&reuse.source, 0, routes).unwrap();
            let mut pages = HostPages::try_new(layout.staging_bytes()).unwrap();
            let stage = MlxRoutedExpertStage::read(&mut kernels, &layout, &mut pages).unwrap();
            let reference = stage.forward_quantized(&mut kernels, &input).unwrap();
            assert_eq!(
                refreshed.host_bytes().unwrap(),
                bytes(&mut kernels, &reference)
            );
        }
        assert_eq!(reuse.stats().staging_allocations, 1);
        assert_eq!(reuse.stats().staging_fallback_syncs, 0);

        let setup_row = |offset: usize, logits: f32| {
            let mut row = [-20.0; EXPERTS];
            row[offset..offset + TOP_K].fill(logits);
            row
        };
        let logits = [setup_row(0, 1.0), setup_row(1, 0.5)];
        let bias = [0.; EXPERTS];
        let routes = InklingRoutes::from_logits(&logits, &bias).unwrap();
        let make_input = |values: &[f32]| {
            Tensor::try_from_host(
                Dtype::F32,
                vec![values.len() as u32, HIDDEN as u32],
                values
                    .iter()
                    .flat_map(|&value| {
                        std::iter::once(value)
                            .chain(std::iter::repeat_n(0.0, HIDDEN - 1))
                            .flat_map(f32::to_le_bytes)
                    })
                    .collect(),
            )
            .unwrap()
        };
        let input = make_input(&[1.0, -0.5]);
        let single = source.prefill_staging_bytes(1, page).unwrap();
        let double = source.prefill_staging_bytes(2, page).unwrap();
        let charge = double - single + 2 * super::super::PREFILL_WORKSPACE_PER_ROW;
        let mut fallback = MlxExpertCache::new(&kernels, Arc::clone(&source), charge - 1);
        assert_eq!(fallback.configure_prefill(2).unwrap(), 1);
        assert!(fallback.staging.is_none());
        let mut batch = MlxExpertCache::new(&kernels, Arc::clone(&source), charge);
        assert_eq!(batch.configure_prefill(3).unwrap(), 2);
        assert_eq!(batch.max_bytes, 0);
        let output = batch
            .forward_quantized_readback(&mut kernels, 0, &routes, &input)
            .unwrap();
        let actual = output.host_bytes().unwrap();
        let mut solo = MlxExpertCache::new(&kernels, source, 0);
        for (index, value) in [1.0, -0.5].into_iter().enumerate() {
            let routes = InklingRoutes::from_logits(&logits[index..=index], &bias).unwrap();
            let output = solo
                .forward_quantized_readback(&mut kernels, 0, &routes, &make_input(&[value]))
                .unwrap();
            let width = HIDDEN * size_of::<f32>();
            assert_eq!(
                &actual[index * width..(index + 1) * width],
                output.host_bytes().unwrap()
            );
        }
        assert_eq!(batch.stats().misses, TOP_K as u64 + 1);
        assert_eq!(solo.stats().misses, 2 * TOP_K as u64);
        assert!(batch.stats().read_bytes < solo.stats().read_bytes);
        assert_eq!(batch.stats().staging_allocations, 1);
    }

    #[test]
    fn inkling_quantizers_match_scale_and_midpoint_cases() {
        let _lock = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let check = |kernels: &mut MlxKernels, mode, input: &[f32], expected: &[f32]| {
            let host = Tensor::try_from_host(
                Dtype::F32,
                vec![1, input.len() as u32],
                input.iter().copied().flat_map(f32::to_le_bytes).collect(),
            )
            .unwrap();
            let resident = kernels.import_tensor(&host).unwrap();
            let output = kernels.inkling_quantized_rows(&resident, mode).unwrap();
            let output = kernels.readback(&output).unwrap();
            assert_eq!(output.shape(), host.shape());
            let (words, tail) = output.host_bytes().unwrap().as_chunks::<4>();
            assert!(tail.is_empty() && words.len() == expected.len());
            for (col, (&word, &value)) in words.iter().zip(expected).enumerate() {
                assert_eq!(
                    f32::from_le_bytes(word).to_bits(),
                    value.to_bits(),
                    "{mode:?} col {col}"
                );
            }
        };
        check(
            &mut kernels,
            QuantizationMode::Nvfp4,
            &[
                6., 5., 3.5, 2.5, 1.75, 1.25, 0.75, 0.25, -6., -5., -3.5, -2.5, -1.75, -1.25,
                -0.75, -0.25, 1.125, 0.875, -0.34375, 0.25, -0.8125, 0.375, -0.25, 0.1875, 1.125,
                0.875, -0.34375, 0.25, -0.8125, 0.375, -0.25, 0.1875,
            ],
            &[
                6., 4.5, 3., 2.25, 1.5, 1.5, 0.75, 0., -6., -4.5, -3., -2.25, -1.5, -1.5, -0.75,
                -0.0, 1.125, 0.75, -0.375, 0.28125, -0.75, 0.375, -0.28125, 0.1875, 1.125, 0.75,
                -0.375, 0.28125, -0.75, 0.375, -0.28125, 0.1875,
            ],
        );
        check(
            &mut kernels,
            QuantizationMode::Nvfp4,
            &[
                6., 0.25, -0.75, 1.25, -1.75, 2.5, -3.5, 5., 1., 1., 1., 1., 1., 1., 1., 1.,
            ],
            &[
                6., 0., -1., 1., -2., 2., -4., 4., 1., 1., 1., 1., 1., 1., 1., 1.,
            ],
        );
        check(
            &mut kernels,
            QuantizationMode::Nvfp4,
            &[-0.0; 16],
            &[-0.0; 16],
        );
        let mut input = [0.0; 64];
        input[..10].copy_from_slice(&[448., -448., 432., 416., 400., 384., 368., 352., 0., -0.0]);
        input[32..38].copy_from_slice(&[448., -450., 4.375, 4.125, -4.375, -4.125]);
        let mut expected = [0.0; 64];
        expected[..10]
            .copy_from_slice(&[448., -448., 448., 416., 384., 384., 384., 352., 0., -0.0]);
        expected[32..38].copy_from_slice(&[448., -448., 4.5, 4., -4.5, -4.]);
        check(&mut kernels, QuantizationMode::Mxfp8, &input, &expected);
        check(
            &mut kernels,
            QuantizationMode::Mxfp8,
            &[-0.0; 32],
            &[-0.0; 32],
        );
    }

    #[test]
    fn router_transposes_bf16_columns_for_f32_matmul() {
        let _lock = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let mut stored = vec![0; HIDDEN * EXPERTS * 2];
        for (expert, feature, weight) in [(0, 9, 1.5_f32), (127, 9, -2.0)] {
            let offset = (expert * HIDDEN + feature) * 2;
            stored[offset..offset + 2]
                .copy_from_slice(&((weight.to_bits() >> 16) as u16).to_le_bytes());
        }
        let weight = kernels
            .import_tensor(&router_matrix(&stored).unwrap())
            .unwrap();
        let mut input = vec![0; HIDDEN * 4];
        input[9 * 4..10 * 4].copy_from_slice(&3.0_f32.to_le_bytes());
        let input = Tensor::try_from_host(Dtype::F32, vec![1, HIDDEN as u32], input).unwrap();
        let input = kernels.import_tensor(&input).unwrap();
        let logits = kernels.mm(&input, &weight).unwrap();
        let logits = kernels.readback(&logits).unwrap();
        assert_eq!(logits.shape(), [1, EXPERTS as u32]);
        let words = logits.host_bytes().unwrap().as_chunks::<4>().0;
        assert_eq!(f32::from_le_bytes(words[0]), 4.5);
        assert_eq!(f32::from_le_bytes(words[127]), -6.0);
        assert!(
            words[1..127]
                .iter()
                .all(|&word| f32::from_le_bytes(word) == 0.0)
        );
        stored[..2].copy_from_slice(&0x7fc0_u16.to_le_bytes());
        assert!(router_matrix(&stored).is_err());
    }

    #[test]
    fn routed_activation_and_merge_round_at_the_bf16_boundaries() {
        let _lock = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let import = |kernels: &mut MlxKernels, shape: Vec<u32>, values: &[f32]| {
            let bytes = values.iter().copied().flat_map(f32::to_le_bytes).collect();
            let host = Tensor::try_from_host(Dtype::F32, shape, bytes).unwrap();
            kernels.import_tensor(&host).unwrap()
        };
        let data = |kernels: &mut MlxKernels, tensor: &ExecutionTensor| {
            kernels
                .readback(tensor)
                .unwrap()
                .host_bytes()
                .unwrap()
                .as_chunks::<4>()
                .0
                .iter()
                .copied()
                .map(f32::from_le_bytes)
                .collect::<Vec<_>>()
        };
        let bf16 = |value: f32| {
            let bits = value.to_bits();
            f32::from_bits((bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff_0000)
        };
        let raw = [64.0, -80.0, -12.0, 26.25, 3.5, 1.1, -2.3, -7.2];
        let route_weights = [0.25, 1.75];
        let gate_up = import(&mut kernels, vec![2, 1, 4], &raw);
        let weights = import(&mut kernels, vec![2], &route_weights);
        let output = kernels
            .inkling_activation(&gate_up, &weights, None)
            .unwrap();
        assert_eq!(output.shape(), [2, 1, 2]);
        let expected: Vec<_> = raw
            .chunks_exact(2)
            .enumerate()
            .map(|(i, row)| {
                let g = bf16(row[0]);
                let u = bf16(row[1]);
                let sigmoid = 1.0 / (1.0 + (-g).exp());
                let gate = 4.0 * (g / 4.0).tanh();
                let up = 25.0 * (u / 25.0).tanh();
                bf16(((gate * sigmoid) * up) * route_weights[i / 2])
            })
            .collect();
        assert_eq!(data(&mut kernels, &output), expected);

        let rows = [
            1e8, 0.1, 1.0, 0.2, -1e8, 0.3, 3.0, 0.4, 0.0, 0.5, 0.0, 0.6, 0.0, 0.7, 0.0, 0.8,
        ];
        let down = import(&mut kernels, vec![8, 1, 2], &rows);
        let output = kernels.inkling_merge(&down, 1, 8).unwrap();
        assert_eq!(output.shape(), [1, 2]);
        let expected: Vec<_> = (0..2)
            .map(|col| {
                let sum = rows
                    .chunks_exact(2)
                    .fold(0.0, |acc, row| acc + bf16(row[col]));
                bf16(sum)
            })
            .collect();
        assert_eq!(data(&mut kernels, &output), expected);
    }

    #[test]
    fn scales_reordered_nvfp4_routes_on_metal() {
        let _lock = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let mut codes = vec![0x22; 8 * 32];
        codes.extend(vec![0x44; 8 * 32]);
        let weight = PackedWeight::try_new(
            Tensor::try_from_host(Dtype::U32, vec![2, 8, 8], codes).unwrap(),
            Tensor::try_from_host(Dtype::U8, vec![2, 8, 4], vec![0x38; 64]).unwrap(),
            None,
            vec![2, 8, 64],
            QuantizationSpec {
                mode: QuantizationMode::Nvfp4,
                bits: 4,
                group_size: 16,
            },
        )
        .unwrap();
        let weight = kernels.import_packed(&weight).unwrap();
        let input = Tensor::try_from_host(
            Dtype::F32,
            vec![1, 1, 64],
            [1.0_f32; 64]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect(),
        )
        .unwrap();
        let ids = Tensor::try_from_host(
            Dtype::U32,
            vec![3],
            [1_u32, 0, 1]
                .into_iter()
                .flat_map(u32::to_le_bytes)
                .collect(),
        )
        .unwrap();
        let multipliers = Tensor::try_from_host(
            Dtype::F32,
            vec![2],
            [0.25_f32, 2.0]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect(),
        )
        .unwrap();
        let input = kernels.import_tensor(&input).unwrap();
        let ids = kernels.import_tensor(&ids).unwrap();
        let ids = kernels.bound_indices(&ids, 2).unwrap();
        let multipliers = kernels.import_tensor(&multipliers).unwrap();
        let y = kernels
            .gather_nvfp4(&input, &weight, &ids, &multipliers)
            .unwrap();
        let y = kernels.readback(&y).unwrap();
        assert_eq!(y.shape(), [3, 1, 8]);
        let (bytes, tail) = y.host_bytes().unwrap().as_chunks::<4>();
        assert!(tail.is_empty());
        let values: Vec<_> = bytes.iter().copied().map(f32::from_le_bytes).collect();
        assert_eq!(values, [[256.0; 8], [16.0; 8], [256.0; 8]].concat());
    }
}
