//! GGUF expert projection. Disk and host RAM retain GGUF bytes.
//! Metal projects supported expert planes directly from their packed layout.

use std::{
    collections::{HashSet, VecDeque},
    rc::Rc,
    sync::Arc,
};

use half::bf16;

use super::{
    HIDDEN, INTERMEDIATE, LAYERS, TOP_K,
    gguf::{
        ExpertPlane, GGUF_BATCH_RETIRE_BYTES, LARGEST_PLANE_F32_BYTES, Weights,
        batched_experts_enabled, finish_expert_visit, packed_mlx_enabled,
        single_copy_experts_enabled,
    },
    mlx::MlxExpertCacheStats,
    timing::{self, measure},
};
use crate::{
    EngError,
    kernels::MlxKernels,
    tensor::{Backing, Dtype, ExecutionTensor, Tensor},
    weights::{
        GgufType, IQ1S_GRID, IQ2S_GRID, IQ2XS_GRID, IQ2XXS_GRID, dequantize_into,
        inkling_codebook_words,
    },
};

type Key = (String, usize);

struct Entry<T> {
    key: Key,
    bytes: usize,
    value: Rc<T>,
}

struct Cache<T> {
    entries: VecDeque<Entry<T>>,
    protected: VecDeque<Entry<T>>,
    protected_bytes: usize,
    protected_limit: usize,
    bytes: usize,
    limit: usize,
}

impl<T> Cache<T> {
    fn new(limit: usize) -> Self {
        Self {
            entries: VecDeque::new(),
            protected: VecDeque::new(),
            protected_bytes: 0,
            protected_limit: 0,
            bytes: 0,
            limit,
        }
    }

    fn get(&mut self, name: &str, expert: usize) -> Option<Rc<T>> {
        let matches = |entry: &Entry<T>| entry.key.0 == name && entry.key.1 == expert;
        if let Some(position) = self.protected.iter().position(matches) {
            let entry = self.protected.remove(position)?;
            let value = Rc::clone(&entry.value);
            self.protected.push_back(entry);
            return Some(value);
        }
        let position = self.entries.iter().position(matches)?;
        let entry = self.entries.remove(position)?;
        let value = Rc::clone(&entry.value);
        if entry.bytes <= self.protected_limit {
            self.protected_bytes += entry.bytes;
            self.protected.push_back(entry);
            while self.protected_bytes > self.protected_limit {
                let demoted = self.protected.pop_front().expect("tracked protected bytes");
                self.protected_bytes -= demoted.bytes;
                self.entries.push_back(demoted);
            }
        } else {
            self.entries.push_back(entry);
        }
        Some(value)
    }

    fn put(&mut self, key: Key, bytes: usize, value: Rc<T>) -> u64 {
        if bytes > self.limit {
            return 0;
        }
        let mut evicted = 0;
        while self.bytes > self.limit - bytes {
            evicted += 1;
            let Some(victim) = self.entries.pop_front() else {
                // The incoming plane would itself be the oldest probationary
                // entry. Keep the protected planes and dispatch without caching it.
                return evicted;
            };
            self.bytes -= victim.bytes;
        }
        self.entries.push_back(Entry { key, bytes, value });
        self.bytes += bytes;
        evicted
    }
}

enum Projection {
    Packed(ExecutionTensor, GgufType),
    Expanded(ExecutionTensor),
}

#[derive(Default, serde::Serialize)]
struct ProjectionStats {
    iq1_s: u64,
    iq2_xxs: u64,
    iq2_xs: u64,
    iq2_s: u64,
    q2_0: u64,
    nvfp4: u64,
    expanded_cpu_transpose: u64,
    expanded_metal_transpose: u64,
}

#[derive(Default, serde::Serialize)]
pub(super) struct ExpertTimings {
    demand_read_ns: u128,
    staging_ns: u128,
    metal_input_import_ns: u128,
    metal_dispatch_ns: u128,
    metal_readback_ns: u128,
    metal_readback_calls: u64,
    metal_activation_import_calls: u64,
    gguf_file_read_bytes: u64,
    gguf_unique_file_read_bytes: u64,
    gguf_metal_staged_bytes: u64,
    #[serde(skip)]
    file_planes: HashSet<Key>,
}

impl ExpertTimings {
    fn file_read(&mut self, name: &str, expert: usize, bytes: usize) {
        let bytes = bytes as u64;
        self.gguf_file_read_bytes += bytes;
        if self.file_planes.insert((name.into(), expert)) {
            self.gguf_unique_file_read_bytes += bytes;
        }
    }
}

pub(super) struct Experts {
    source: Arc<Weights>,
    metal: MlxKernels,
    packed: Cache<ExpertPlane>,
    resident: Cache<Projection>,
    packed_enabled: bool,
    batched_enabled: bool,
    batched_visits: u64,
    codes: Vec<(GgufType, ExecutionTensor)>,
    projections: ProjectionStats,
    stats: MlxExpertCacheStats,
    timing: Option<ExpertTimings>,
    trace_routes: bool,
}

impl Experts {
    pub(super) fn open(source: Arc<Weights>, headroom: usize) -> Result<Self, EngError> {
        // Metal and RAM share physical memory. The combined capacity is charged
        // against one admission budget, independent of which tier serves a hit.
        let single_copy = single_copy_experts_enabled() && all_experts_have_packed_kernels(&source);
        let ram = if single_copy { 0 } else { headroom / 4 };
        let mut resident = Cache::new(headroom - ram);
        if single_copy
            && std::env::var_os("FIFI_INKLING_GGUF_SEGMENTED_LRU").is_some_and(|value| value == "1")
        {
            // A second access promotes a plane. New admissions and demotions
            // share the remaining space. Both queues use the same Metal budget.
            resident.protected_limit = resident.limit - resident.limit.div_ceil(10);
        }
        let mut metal = MlxKernels::try_new()?;
        // Bound deferred retirements on this adapter. The allocator pool is
        // shared by the process, so changing its limit would affect other engines.
        let batched_enabled = batched_experts_enabled();
        let retire_cap = if batched_enabled {
            GGUF_BATCH_RETIRE_BYTES
        } else {
            LARGEST_PLANE_F32_BYTES as u64
        };
        metal.set_retire_cap(retire_cap)?;
        Ok(Self {
            source,
            metal,
            packed: Cache::new(ram),
            resident,
            packed_enabled: packed_mlx_enabled(),
            batched_enabled,
            batched_visits: 0,
            codes: Vec::new(),
            projections: ProjectionStats::default(),
            stats: MlxExpertCacheStats::default(),
            timing: timing::enabled().then(ExpertTimings::default),
            trace_routes: std::env::var_os("FIFI_INKLING_ROUTE_TRACE")
                .is_some_and(|value| value == "1"),
        })
    }

    pub(super) fn memory_stats(&self) -> Option<crate::kernels::MemoryStats> {
        self.metal.memory_stats().ok()
    }

    pub(super) fn stats(&self) -> MlxExpertCacheStats {
        self.stats
    }

    pub(super) fn take_component_timings(&mut self) -> Option<ExpertTimings> {
        self.timing.as_mut().map(std::mem::take)
    }

    pub(super) fn report_projections(&mut self, phase: &str) {
        eprintln!(
            "inkling_gguf_projection {}",
            serde_json::json!({
                "phase": phase,
                "packed_enabled": self.packed_enabled,
                "batched_enabled": self.batched_enabled,
                "batched_visits": std::mem::take(&mut self.batched_visits),
                "retained_ram_capacity_bytes": self.packed.limit,
                "retained_metal_capacity_bytes": self.resident.limit,
                "protected_metal_capacity_bytes": self.resident.protected_limit,
                "projections": std::mem::take(&mut self.projections),
                "resident_bytes": self.resident.bytes,
                "packed_ram_bytes": self.packed.bytes,
                "codebook_bytes": self.codes.iter().map(|(_, codes)| codes.byte_len()).sum::<u64>(),
            })
        );
    }

    fn stage_projection(&mut self, plane: &ExpertPlane) -> Result<(Projection, usize), EngError> {
        let needs_codes = inkling_codebook_words(plane.dtype).is_some();
        if self.packed_enabled && packed_format(plane.dtype) {
            if needs_codes && !self.codes.iter().any(|(format, _)| *format == plane.dtype) {
                let table = self
                    .metal
                    .import_tensor(&codebook(plane.dtype)?.expect("format has a codebook"))?;
                self.codes.push((plane.dtype, table));
            }
            let payload = Tensor::from_shared(
                Dtype::U8,
                vec![
                    plane.height as u32,
                    (plane.bytes.len() / plane.height) as u32,
                ],
                Backing::SharedVec(Arc::clone(&plane.bytes)),
                0..plane.bytes.len(),
            )?;
            let projection = self.metal.import_tensor(&payload)?;
            return Ok((
                Projection::Packed(projection, plane.dtype),
                plane.bytes.len(),
            ));
        }

        let expanded = if self.packed_enabled {
            let rows = self.metal.import_tensor(&decode_rows(plane)?)?;
            self.metal.transpose(&rows, &[1, 0])?
        } else {
            self.metal.import_tensor(&decode_transposed(plane)?)?
        };
        Ok((
            Projection::Expanded(expanded),
            plane.width * plane.height * 4,
        ))
    }

    pub(super) fn row(
        &mut self,
        layer: usize,
        input: &[f32],
        selected: &[usize],
        routes: &[f32],
    ) -> Result<Vec<bf16>, EngError> {
        let source = Arc::clone(&self.source);
        self.stats.visits += 1;
        let before = self.stats;
        let grouped = self.batched_enabled && self.can_batch(layer);
        let output = if grouped {
            let output = self.row_batched(layer, input, selected, routes)?;
            self.batched_visits += 1;
            output
        } else {
            source.moe_row_with(
                layer,
                input,
                selected,
                routes,
                |name, expert, height, input| self.project(name, expert, height, input),
            )?
        };
        self.stats.fully_cached_visits += u64::from(before.misses == self.stats.misses);
        if self.trace_routes {
            self.trace_visit(layer, selected, grouped, before);
        }
        Ok(output)
    }

    fn trace_visit(
        &self,
        layer: usize,
        selected: &[usize],
        grouped: bool,
        before: MlxExpertCacheStats,
    ) {
        let plane_bytes = ["ffn_gate_up_exps", "ffn_down_exps"].map(|name| {
            let tensor = self
                .source
                .header()
                .tensor(&format!("blk.{layer}.{name}.weight"))
                .expect("completed visit uses a validated expert tensor");
            let resident = if self.packed_enabled && packed_format(tensor.ggml_type()) {
                tensor.byte_len() / tensor.dims()[2]
            } else {
                tensor.dims()[0] * tensor.dims()[1] * 4
            };
            [tensor.byte_len() / tensor.dims()[2], resident]
        });
        eprintln!(
            "inkling_gguf_route {}",
            serde_json::json!({
                "visit": self.stats.visits,
                "layer": layer,
                "selected": selected,
                "grouped": grouped,
                "plane_bytes": plane_bytes,
                "resident_capacity_bytes": self.resident.limit,
                "ram_capacity_bytes": self.packed.limit,
                "read_bytes": self.stats.read_bytes - before.read_bytes,
                "misses": self.stats.misses - before.misses,
                "evictions": self.stats.evictions - before.evictions,
            })
        );
    }

    fn can_batch(&self, layer: usize) -> bool {
        layer_has_packed_kernels(&self.source, layer)
    }

    fn row_batched(
        &mut self,
        layer: usize,
        input: &[f32],
        selected: &[usize],
        routes: &[f32],
    ) -> Result<Vec<bf16>, EngError> {
        let visit = self.source.expert_visit(layer, selected, routes)?;
        let gate = self.project_group(&visit.gate, selected, 2 * INTERMEDIATE, &[input])?;
        let activated: Vec<_> = selected
            .iter()
            .zip(gate.as_chunks::<{ 2 * INTERMEDIATE }>().0)
            .map(|(&expert, fused)| visit.activate(expert, fused))
            .collect();
        drop(gate);
        let activations: Vec<_> = activated.iter().map(Vec::as_slice).collect();
        let down = self.project_group(&visit.down, selected, HIDDEN, &activations)?;
        let mut merged = vec![0.; HIDDEN];
        for ((&expert, &route), projected) in selected
            .iter()
            .zip(routes)
            .zip(down.as_chunks::<HIDDEN>().0)
        {
            visit.accumulate(&mut merged, expert, route, projected);
        }
        finish_expert_visit(merged)
    }

    fn project_group(
        &mut self,
        name: &str,
        selected: &[usize],
        height: usize,
        inputs: &[&[f32]],
    ) -> Result<Vec<f32>, EngError> {
        // Stage every plane before dispatch so a cache miss cannot split a
        // pending projection group with the synchronous weight import.
        let width = inputs[0].len();
        let weights = selected
            .iter()
            .map(|&expert| self.weight(name, expert, height, width))
            .collect::<Result<Vec<_>, _>>()?;
        let host_inputs = inputs
            .iter()
            .map(|input| host_activation(input))
            .collect::<Result<Vec<_>, _>>()?;
        let activation = measure(
            self.timing.as_mut().map(|t| &mut t.metal_input_import_ns),
            || {
                if host_inputs.len() == 1 {
                    Ok(vec![self.metal.import_tensor(&host_inputs[0])?])
                } else {
                    let inputs: Vec<_> = host_inputs.iter().collect();
                    self.metal.import_many(&inputs)
                }
            },
        )?;
        if let Some(timing) = &mut self.timing {
            timing.metal_activation_import_calls += 1;
        }
        let outputs = weights
            .iter()
            .enumerate()
            .map(|(slot, weight)| {
                let index = if activation.len() == 1 { 0 } else { slot };
                self.dispatch(weight, &activation[index])
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut concatenate_ns = 0;
        let result = measure(self.timing.is_some().then_some(&mut concatenate_ns), || {
            let rows: Vec<_> = outputs.iter().collect();
            self.metal.concatenate(&rows, 0)
        })?;
        if let Some(timing) = &mut self.timing {
            timing.metal_dispatch_ns += concatenate_ns;
        }
        self.read_output(&result, TOP_K, height)
    }

    fn project(
        &mut self,
        name: &str,
        expert: usize,
        height: usize,
        input: &[f32],
    ) -> Result<Vec<f32>, EngError> {
        let weight = self.weight(name, expert, height, input.len())?;
        let activation = measure(
            self.timing.as_mut().map(|t| &mut t.metal_input_import_ns),
            || self.metal.import_tensor(&host_activation(input)?),
        )?;
        if let Some(timing) = &mut self.timing {
            timing.metal_activation_import_calls += 1;
        }
        let projected = self.dispatch(&weight, &activation)?;
        self.read_output(&projected, 1, height)
    }

    fn weight(
        &mut self,
        name: &str,
        expert: usize,
        height: usize,
        width: usize,
    ) -> Result<Rc<Projection>, EngError> {
        let weight = if let Some(weight) = self.resident.get(name, expert) {
            self.stats.hits += 1;
            weight
        } else {
            let plane = if let Some(plane) = self.packed.get(name, expert) {
                self.stats.hits += 1;
                plane
            } else {
                let plane = Rc::new(measure(
                    self.timing.as_mut().map(|t| &mut t.demand_read_ns),
                    || self.source.read_expert(name, expert),
                )?);
                self.stats.misses += 1;
                self.stats.read_bytes += plane.bytes.len() as u64;
                self.stats.read_ranges += 1;
                if let Some(timing) = &mut self.timing {
                    timing.file_read(name, expert, plane.bytes.len());
                }
                if self.packed.limit != 0 {
                    self.stats.evictions += self.packed.put(
                        (name.into(), expert),
                        plane.bytes.len(),
                        Rc::clone(&plane),
                    );
                }
                plane
            };
            if plane.width != width || plane.height != height {
                return Err(super::checkpoint_error(
                    "GGUF Metal expert projection shape mismatch",
                ));
            }
            let mut staging_ns = 0;
            let (projection, bytes) =
                measure(self.timing.is_some().then_some(&mut staging_ns), || {
                    self.stage_projection(&plane)
                })?;
            if let Some(timing) = &mut self.timing {
                timing.staging_ns += staging_ns;
                timing.gguf_metal_staged_bytes += bytes as u64;
            }
            let weight = Rc::new(projection);
            self.stats.evictions +=
                self.resident
                    .put((name.into(), expert), bytes, Rc::clone(&weight));
            self.stats.cached_bytes = self.packed.bytes + self.resident.bytes;
            weight
        };
        Ok(weight)
    }

    fn dispatch(
        &mut self,
        weight: &Projection,
        activation: &ExecutionTensor,
    ) -> Result<ExecutionTensor, EngError> {
        measure(
            self.timing.as_mut().map(|t| &mut t.metal_dispatch_ns),
            || {
                let output = match weight {
                    Projection::Packed(payload, format) => {
                        let codes = self
                            .codes
                            .iter()
                            .find_map(|(kind, table)| (kind == format).then_some(table));
                        let count = match format {
                            GgufType::Iq1S => &mut self.projections.iq1_s,
                            GgufType::Iq2Xxs => &mut self.projections.iq2_xxs,
                            GgufType::Iq2Xs => &mut self.projections.iq2_xs,
                            GgufType::Iq2S => &mut self.projections.iq2_s,
                            GgufType::Q20 => &mut self.projections.q2_0,
                            GgufType::Nvfp4 => &mut self.projections.nvfp4,
                            _ => unreachable!("packed formats were validated when staged"),
                        };
                        let result = self
                            .metal
                            .inkling_gguf_matvec(activation, payload, codes, *format)?;
                        *count += 1;
                        result
                    }
                    Projection::Expanded(expanded) => {
                        let result = self.metal.mm(activation, expanded)?;
                        if self.packed_enabled {
                            self.projections.expanded_metal_transpose += 1;
                        } else {
                            self.projections.expanded_cpu_transpose += 1;
                        }
                        result
                    }
                };
                Ok::<_, EngError>(output)
            },
        )
    }

    fn read_output(
        &mut self,
        projected: &ExecutionTensor,
        rows: usize,
        height: usize,
    ) -> Result<Vec<f32>, EngError> {
        let host = measure(
            self.timing.as_mut().map(|t| &mut t.metal_readback_ns),
            || self.metal.readback(projected),
        )?;
        if let Some(timing) = &mut self.timing {
            timing.metal_readback_calls += 1;
        }
        if host.shape() != [rows as u32, height as u32] || host.dtype() != Dtype::F32 {
            return Err(super::checkpoint_error(
                "GGUF Metal expert output shape mismatch",
            ));
        }
        let bytes = host.host_bytes().ok_or_else(|| {
            super::checkpoint_error("GGUF Metal expert projection has no readback")
        })?;
        Ok(bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&word| f32::from_le_bytes(word))
            .collect())
    }
}

/// Expanded formats can still benefit from keeping a smaller packed host
/// copy. Fully packed checkpoints retain the same bytes in the Metal buffer.
fn all_experts_have_packed_kernels(source: &Weights) -> bool {
    (0..LAYERS).all(|layer| layer_has_packed_kernels(source, layer))
}

fn layer_has_packed_kernels(source: &Weights, layer: usize) -> bool {
    ["ffn_gate_up_exps", "ffn_down_exps"]
        .into_iter()
        .all(|name| {
            source
                .header()
                .tensor(&format!("blk.{layer}.{name}.weight"))
                .is_some_and(|tensor| packed_format(tensor.ggml_type()))
        })
}

fn host_activation(input: &[f32]) -> Result<Tensor, EngError> {
    Tensor::try_from_host(
        Dtype::F32,
        vec![1, input.len() as u32],
        input.iter().flat_map(|value| value.to_le_bytes()).collect(),
    )
}

fn packed_format(format: GgufType) -> bool {
    inkling_codebook_words(format).is_some() || matches!(format, GgufType::Q20 | GgufType::Nvfp4)
}

fn codebook(format: GgufType) -> Result<Option<Tensor>, EngError> {
    let Some(words) = inkling_codebook_words(format) else {
        return Ok(None);
    };
    let bytes = match format {
        GgufType::Iq1S => IQ1S_GRID
            .iter()
            .flat_map(|&word| u32::from(word).to_le_bytes())
            .collect(),
        GgufType::Iq2Xxs => IQ2XXS_GRID
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect(),
        GgufType::Iq2Xs => IQ2XS_GRID
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect(),
        GgufType::Iq2S => IQ2S_GRID
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect(),
        _ => unreachable!("codebook formats were validated"),
    };
    Tensor::try_from_host(Dtype::U32, vec![words], bytes).map(Some)
}

fn decode_rows(plane: &ExpertPlane) -> Result<Tensor, EngError> {
    let stride = plane.bytes.len() / plane.height;
    let mut row = vec![0.; plane.width];
    let mut out = Vec::with_capacity(plane.width * plane.height * 4);
    for block_row in plane.bytes.chunks_exact(stride) {
        dequantize_into(plane.dtype, block_row, &mut row)?;
        out.extend(row.iter().flat_map(|v| v.to_le_bytes()));
    }
    Tensor::try_from_host(
        Dtype::F32,
        vec![plane.height as u32, plane.width as u32],
        out,
    )
}

fn decode_transposed(plane: &ExpertPlane) -> Result<Tensor, EngError> {
    let row_bytes = plane.bytes.len() / plane.height;
    let mut row = vec![0.; plane.width];
    let mut out = vec![0; plane.width * plane.height * 4];
    for (index, bytes) in plane.bytes.chunks_exact(row_bytes).enumerate() {
        dequantize_into(plane.dtype, bytes, &mut row)?;
        for (column, value) in row.iter().enumerate() {
            let at = 4 * (column * plane.height + index);
            out[at..at + 4].copy_from_slice(&value.to_le_bytes());
        }
    }
    Tensor::try_from_host(
        Dtype::F32,
        vec![plane.width as u32, plane.height as u32],
        out,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn segmented_lru_retains_a_reused_plane_during_new_admissions() {
        let mut cache = Cache::new(10);
        cache.protected_limit = 9;
        for id in 0..3 {
            assert_eq!(cache.put(("gate".into(), id), 3, Rc::new(id)), 0);
        }
        assert_eq!(*cache.get("gate", 0).unwrap(), 0);
        for id in 3..6 {
            assert_eq!(cache.put(("gate".into(), id), 3, Rc::new(id)), 1);
        }
        assert_eq!(*cache.get("gate", 0).unwrap(), 0);
        assert!(cache.get("gate", 1).is_none());
        assert_eq!(cache.bytes, 9);
        assert_eq!(cache.protected_bytes, 3);
    }

    #[test]
    fn segmented_lru_demotes_by_bytes_and_can_skip_an_incoming_plane() {
        let mut cache = Cache::new(16);
        cache.protected_limit = 9;
        for (id, bytes) in [(0, 6), (1, 4), (2, 6)] {
            cache.put(("gate".into(), id), bytes, Rc::new(id));
        }
        assert!(cache.get("gate", 0).is_some());
        assert!(cache.get("gate", 1).is_some());
        assert_eq!(cache.protected_bytes, 4);
        assert_eq!(cache.put(("down".into(), 3), 13, Rc::new(3)), 3);
        assert_eq!(cache.bytes, 4);
        assert!(cache.get("down", 3).is_none());
        assert!(cache.get("gate", 1).is_some());
    }

    #[test]
    fn gguf_read_amplification_counts_each_tensor_plane_once_per_phase() {
        let mut timing = ExpertTimings::default();
        timing.file_read("blk.0.ffn_gate_up_exps.weight", 3, 120);
        timing.file_read("blk.0.ffn_gate_up_exps.weight", 3, 120);
        timing.file_read("blk.0.ffn_down_exps.weight", 3, 80);
        timing.file_read("blk.1.ffn_gate_up_exps.weight", 3, 130);
        timing.file_read("blk.1.ffn_gate_up_exps.weight", 4, 130);
        let phase = std::mem::take(&mut timing);
        assert_eq!(phase.gguf_file_read_bytes, 580);
        assert_eq!(phase.gguf_unique_file_read_bytes, 460);
        timing.file_read("blk.0.ffn_gate_up_exps.weight", 3, 120);
        assert_eq!(timing.gguf_unique_file_read_bytes, 120);
    }

    #[test]
    #[ignore = "requires complete Unsloth UD-IQ2_XXS fork weights at INKLING_GGUF_REFERENCE and Apple Metal"]
    fn packed_gguf_matvec_matches_released_expert_blocks() {
        let _lock = crate::native_test_lock();
        let file = std::env::var_os("INKLING_GGUF_REFERENCE").expect("INKLING_GGUF_REFERENCE");
        let source = Weights::open(std::path::Path::new(&file)).unwrap();
        assert!(all_experts_have_packed_kernels(&source));
        let mut metal = MlxKernels::try_new().unwrap();
        for (name, format) in [
            ("blk.0.ffn_gate_up_exps.weight", GgufType::Iq1S),
            ("blk.0.ffn_down_exps.weight", GgufType::Q20),
            ("blk.5.ffn_down_exps.weight", GgufType::Nvfp4),
            ("blk.5.ffn_gate_up_exps.weight", GgufType::Iq2Xxs),
            ("blk.9.ffn_gate_up_exps.weight", GgufType::Iq2Xs),
            ("blk.16.ffn_gate_up_exps.weight", GgufType::Iq2S),
        ] {
            let info = source.header().tensor(name).unwrap();
            assert_eq!(info.ggml_type(), format);
            let full_stride = info.dims()[0] / format.block_elems() * format.block_bytes();
            let stride = 2 * format.block_bytes() as usize;
            let width = 2 * format.block_elems() as usize;
            let mut packed = vec![0; 2 * stride];
            for (row, dst) in packed.chunks_exact_mut(stride).enumerate() {
                source
                    .read_range(name, row as u64 * full_stride, dst)
                    .unwrap();
            }
            let x: Vec<_> = (0..width)
                .map(|i| ((i % 17) as f32 - 8.) * 0.0625)
                .collect();
            let weight =
                Tensor::try_from_host(Dtype::U8, vec![2, stride as u32], packed.clone()).unwrap();
            let activation = Tensor::try_from_host(
                Dtype::F32,
                vec![1, width as u32],
                x.iter().flat_map(|value| value.to_le_bytes()).collect(),
            )
            .unwrap();
            let w = metal.import_tensor(&weight).unwrap();
            let a = metal.import_tensor(&activation).unwrap();
            let codes = codebook(format)
                .unwrap()
                .map(|tensor| metal.import_tensor(&tensor).unwrap());
            let got = metal
                .inkling_gguf_matvec(&a, &w, codes.as_ref(), format)
                .unwrap();
            assert!(
                metal
                    .inkling_gguf_matvec(&a, &w, None, GgufType::F32)
                    .is_err()
            );
            let got = metal.readback(&got).unwrap();
            assert_eq!(got.shape(), [1, 2]);
            let mut decoded = vec![0.; width];
            for (row, bytes) in packed.chunks_exact(stride).enumerate() {
                dequantize_into(format, bytes, &mut decoded).unwrap();
                let want = x.iter().zip(&decoded).map(|(x, w)| x * w).sum::<f32>();
                let bound = x
                    .iter()
                    .zip(&decoded)
                    .map(|(x, w)| (x * w).abs())
                    .sum::<f32>();
                let actual = f32::from_le_bytes(got.host_bytes().unwrap().as_chunks::<4>().0[row]);
                assert!(
                    (actual - want).abs() <= 2e-5 * bound.max(1.0),
                    "{name} row {row}: {actual} != {want}"
                );
            }
        }
    }

    #[cfg(feature = "native-fault-qualification")]
    #[test]
    #[ignore = "requires a complete Unsloth UD-IQ2_XXS GGUF at INKLING_GGUF_REFERENCE and Apple Metal"]
    fn batched_gguf_visit_matches_sequential_packed_bits_and_counts() {
        let _lock = crate::native_test_lock();
        let file = std::env::var_os("INKLING_GGUF_REFERENCE").expect("INKLING_GGUF_REFERENCE");
        let source = Arc::new(Weights::open(std::path::Path::new(&file)).unwrap());
        let x: Vec<_> = (0..HIDDEN)
            .map(|i| ((i % 17) as f32 - 8.) * 0.0625)
            .collect();
        let experts = [17, 1, 21, 0, 43, 5, 87, 127];
        let routes = [0.8125, 0.9375, 1., 1.25, 0.875, 0.6875, 1.125, 1.3125];
        let pairs = [
            (0, GgufType::Iq1S, GgufType::Q20),
            (5, GgufType::Iq2Xxs, GgufType::Nvfp4),
            (7, GgufType::Iq2Xxs, GgufType::Q20),
            (9, GgufType::Iq2Xs, GgufType::Nvfp4),
            (10, GgufType::Iq2Xs, GgufType::Q20),
            (16, GgufType::Iq2S, GgufType::Q20),
            (19, GgufType::Iq2S, GgufType::Nvfp4),
        ];
        let run = |batched, headroom| {
            let mut runtime = Experts::open(Arc::clone(&source), headroom).unwrap();
            runtime.packed_enabled = true;
            runtime.batched_enabled = batched;
            runtime.timing = Some(ExpertTimings::default());
            let retire_cap = if batched {
                GGUF_BATCH_RETIRE_BYTES
            } else {
                LARGEST_PLANE_F32_BYTES as u64
            };
            runtime.metal.set_retire_cap(retire_cap).unwrap();
            let mut values = Vec::new();
            for (layer, gate, down) in pairs {
                let plan = source.expert_visit(layer, &experts, &routes).unwrap();
                for (name, dtype) in [(&plan.gate, gate), (&plan.down, down)] {
                    assert_eq!(source.header().tensor(name).unwrap().ggml_type(), dtype);
                }
                assert!(runtime.can_batch(layer));
                for repeated in [false, true] {
                    let case = match (headroom, repeated) {
                        (0, _) => "zero_retention",
                        (n, _) if n == 8 << 20 => "eviction",
                        (_, false) => "cold",
                        (_, true) => "warm",
                    };
                    let before = runtime.stats();
                    let sync_before = runtime.metal.qualification_sync_count();
                    let flush_before = runtime.metal.qualification_retire_stats().0;
                    runtime.metal.qualification_reset_retire_peak();
                    let result = runtime.row(layer, &x, &experts, &routes).unwrap();
                    let syncs = runtime.metal.qualification_sync_count() - sync_before;
                    let (flush_total, retire_peak) = runtime.metal.qualification_retire_stats();
                    let flushes = flush_total - flush_before;
                    let after = runtime.stats();
                    let timing = runtime.take_component_timings().unwrap();
                    assert_eq!(
                        timing.gguf_file_read_bytes,
                        after.read_bytes - before.read_bytes
                    );
                    assert!(timing.gguf_unique_file_read_bytes <= timing.gguf_file_read_bytes);
                    if case == "warm" {
                        assert_eq!(timing.gguf_metal_staged_bytes, 0);
                    } else {
                        assert!(timing.gguf_metal_staged_bytes > 0);
                    }
                    let calls = if batched { 2 } else { 16 };
                    assert_eq!(timing.metal_activation_import_calls, calls);
                    assert_eq!(timing.metal_readback_calls, calls);
                    assert_eq!(
                        std::mem::take(&mut runtime.batched_visits),
                        u64::from(batched)
                    );
                    let weight_or_table_imports = if case == "warm" {
                        0
                    } else {
                        2 * TOP_K as u64 + 2
                    };
                    // A strided readback may take one extra sync. The shim counts
                    // retire-cap flushes too, so subtract those for the call bound.
                    let minimum = 2 * calls + flushes;
                    let maximum = minimum + calls + weight_or_table_imports;
                    assert!(
                        (minimum..=maximum).contains(&syncs),
                        "layer={layer} batched={batched} case={case} syncs={syncs} expected={minimum}..={maximum}"
                    );
                    assert!(retire_peak <= retire_cap + LARGEST_PLANE_F32_BYTES as u64);
                    if case == "warm" {
                        assert_eq!(after.read_ranges, before.read_ranges);
                        assert_eq!(after.evictions, before.evictions);
                        assert_eq!(flushes, 0);
                    }
                    if case == "cold" {
                        assert!(after.read_ranges > before.read_ranges);
                    }
                    if case == "eviction" {
                        assert!(
                            after.evictions > before.evictions,
                            "eviction case must evict an entry"
                        );
                    }
                    if case == "zero_retention" {
                        assert_eq!(after.cached_bytes, 0);
                    }
                    eprintln!(
                        "gguf batch visit layer={layer} batched={batched} case={case} syncs={syncs} retire_flushes={flushes} retire_peak_bytes={retire_peak} evictions={}",
                        after.evictions - before.evictions
                    );
                    values.extend(result.into_iter().map(bf16::to_bits));
                }
                if layer == 0 && batched && headroom == 64 << 20 {
                    // Both phases of one warm visit also compare projection
                    // values directly, before BF16 narrowing can mask changes.
                    let gates = runtime
                        .project_group(&plan.gate, &experts, 2 * INTERMEDIATE, &[&x])
                        .unwrap();
                    let mut activations = Vec::new();
                    for (&expert, fused) in experts
                        .iter()
                        .zip(gates.as_chunks::<{ 2 * INTERMEDIATE }>().0)
                    {
                        let sequential = runtime
                            .project(&plan.gate, expert, 2 * INTERMEDIATE, &x)
                            .unwrap();
                        assert!(
                            fused
                                .iter()
                                .zip(sequential)
                                .all(|(&a, b)| a.to_bits() == b.to_bits())
                        );
                        activations.push(plan.activate(expert, fused));
                    }
                    let inputs: Vec<_> = activations.iter().map(Vec::as_slice).collect();
                    let downs = runtime
                        .project_group(&plan.down, &experts, HIDDEN, &inputs)
                        .unwrap();
                    for ((&expert, activation), down) in experts
                        .iter()
                        .zip(inputs)
                        .zip(downs.as_chunks::<HIDDEN>().0)
                    {
                        let sequential = runtime
                            .project(&plan.down, expert, HIDDEN, activation)
                            .unwrap();
                        assert!(
                            down.iter()
                                .zip(sequential)
                                .all(|(&a, b)| a.to_bits() == b.to_bits())
                        );
                    }
                    let _ = runtime.take_component_timings();
                }
            }
            values
        };
        for headroom in [0, 8 << 20, 64 << 20] {
            assert_eq!(run(false, headroom), run(true, headroom));
        }
    }

    #[test]
    fn cache_evicts_least_recent_and_respects_byte_limit() {
        let mut cache = Cache::new(12);
        let value = Rc::new(11);
        assert_eq!(cache.put(("down".into(), 2), 5, Rc::clone(&value)), 0);
        assert_eq!(cache.put(("up".into(), 2), 5, Rc::clone(&value)), 0);
        assert_eq!(cache.get("down", 2).as_deref(), Some(&11));
        assert_eq!(cache.put(("down".into(), 5), 5, Rc::clone(&value)), 1);
        assert!(cache.get("up", 2).is_none());
        assert!(cache.get("down", 2).is_some());
        assert_eq!(cache.bytes, 10);
        assert_eq!(cache.put(("too big".into(), 1), 13, value), 0);
        assert_eq!(cache.bytes, 10);
    }
}
