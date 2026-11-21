//! Experimental GGUF expert projection. Disk and host RAM retain GGUF bytes.
//! The opt-in Metal path can project selected expert planes from their packed layout.

use std::{collections::VecDeque, rc::Rc, sync::Arc};

use half::bf16;

use super::{
    gguf::{ExpertPlane, LARGEST_PLANE_F32_BYTES, Weights, packed_mlx_enabled},
    mlx::MlxExpertCacheStats,
    timing::{self, measure},
};
use crate::{
    EngError,
    kernels::MlxKernels,
    tensor::{Dtype, ExecutionTensor, Tensor},
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
    bytes: usize,
    limit: usize,
}

impl<T> Cache<T> {
    fn new(limit: usize) -> Self {
        Self {
            entries: VecDeque::new(),
            bytes: 0,
            limit,
        }
    }

    fn get(&mut self, name: &str, expert: usize) -> Option<Rc<T>> {
        let position = self
            .entries
            .iter()
            .position(|entry| entry.key.0 == name && entry.key.1 == expert)?;
        let entry = self.entries.remove(position)?;
        let value = Rc::clone(&entry.value);
        self.entries.push_back(entry);
        Some(value)
    }

    fn put(&mut self, key: Key, bytes: usize, value: Rc<T>) -> u64 {
        if bytes > self.limit {
            return 0;
        }
        let mut evicted = 0;
        while self.bytes > self.limit - bytes {
            self.bytes -= self.entries.pop_front().expect("tracked cache bytes").bytes;
            evicted += 1;
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
}

pub(super) struct Experts {
    source: Arc<Weights>,
    metal: MlxKernels,
    packed: Cache<ExpertPlane>,
    resident: Cache<Projection>,
    packed_enabled: bool,
    codes: Vec<(GgufType, ExecutionTensor)>,
    projections: ProjectionStats,
    stats: MlxExpertCacheStats,
    timing: Option<ExpertTimings>,
}

impl Experts {
    pub(super) fn open(source: Arc<Weights>, headroom: usize) -> Result<Self, EngError> {
        // Metal and RAM share physical memory. The combined capacity is charged
        // against one admission budget, independent of which tier serves a hit.
        let ram = headroom / 4;
        let mut metal = MlxKernels::try_new()?;
        // Bound deferred retirements on this adapter. The allocator pool is
        // shared by the process, so changing its limit would affect other engines.
        metal.set_retire_cap(LARGEST_PLANE_F32_BYTES as u64)?;
        Ok(Self {
            source,
            metal,
            packed: Cache::new(ram),
            resident: Cache::new(headroom - ram),
            packed_enabled: packed_mlx_enabled(),
            codes: Vec::new(),
            projections: ProjectionStats::default(),
            stats: MlxExpertCacheStats::default(),
            timing: timing::enabled().then(ExpertTimings::default),
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
                "projections": std::mem::take(&mut self.projections),
                "resident_bytes": self.resident.bytes,
                "packed_ram_bytes": self.packed.bytes,
                "codebook_bytes": self.codes.iter().map(|(_, codes)| codes.byte_len()).sum::<u64>(),
            })
        );
    }

    fn stage_projection(&mut self, plane: &ExpertPlane) -> Result<(Projection, usize), EngError> {
        let needs_codes = inkling_codebook_words(plane.dtype).is_some();
        if self.packed_enabled
            && (needs_codes || matches!(plane.dtype, GgufType::Q20 | GgufType::Nvfp4))
        {
            if needs_codes && !self.codes.iter().any(|(format, _)| *format == plane.dtype) {
                let table = self
                    .metal
                    .import_tensor(&codebook(plane.dtype)?.expect("format has a codebook"))?;
                self.codes.push((plane.dtype, table));
            }
            let payload = Tensor::try_from_host(
                Dtype::U8,
                vec![
                    plane.height as u32,
                    (plane.bytes.len() / plane.height) as u32,
                ],
                plane.bytes.clone(),
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
        let misses = self.stats.misses;
        let output = source.moe_row_with(
            layer,
            input,
            selected,
            routes,
            |name, expert, height, input| self.project(name, expert, height, input),
        )?;
        self.stats.fully_cached_visits += u64::from(misses == self.stats.misses);
        Ok(output)
    }

    fn project(
        &mut self,
        name: &str,
        expert: usize,
        height: usize,
        input: &[f32],
    ) -> Result<Vec<f32>, EngError> {
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
                self.stats.evictions +=
                    self.packed
                        .put((name.into(), expert), plane.bytes.len(), Rc::clone(&plane));
                plane
            };
            if plane.width != input.len() || plane.height != height {
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
            }
            let weight = Rc::new(projection);
            self.stats.evictions +=
                self.resident
                    .put((name.into(), expert), bytes, Rc::clone(&weight));
            self.stats.cached_bytes = self.packed.bytes + self.resident.bytes;
            weight
        };
        let activation = measure(
            self.timing.as_mut().map(|t| &mut t.metal_input_import_ns),
            || {
                let activation = Tensor::try_from_host(
                    Dtype::F32,
                    vec![1, input.len() as u32],
                    input.iter().flat_map(|value| value.to_le_bytes()).collect(),
                )?;
                self.metal.import_tensor(&activation)
            },
        )?;
        let projected = measure(
            self.timing.as_mut().map(|t| &mut t.metal_dispatch_ns),
            || {
                let output = match weight.as_ref() {
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
                        let result =
                            self.metal
                                .inkling_gguf_matvec(&activation, payload, codes, *format)?;
                        *count += 1;
                        result
                    }
                    Projection::Expanded(expanded) => {
                        let result = self.metal.mm(&activation, expanded)?;
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
        )?;
        let host = measure(
            self.timing.as_mut().map(|t| &mut t.metal_readback_ns),
            || self.metal.readback(&projected),
        )?;
        if host.shape() != [1, height as u32] || host.dtype() != Dtype::F32 {
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
    #[ignore = "requires complete Unsloth UD-IQ2_XXS fork weights at INKLING_GGUF_REFERENCE and Apple Metal"]
    fn packed_gguf_matvec_matches_released_expert_blocks() {
        let _lock = crate::native_test_lock();
        let file = std::env::var_os("INKLING_GGUF_REFERENCE").expect("INKLING_GGUF_REFERENCE");
        let source = Weights::open(std::path::Path::new(&file)).unwrap();
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
