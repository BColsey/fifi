//! Experimental GGUF expert projection. Disk and host RAM retain GGUF bytes.
//! Metal stores expanded F32 planes so all supported GGUF codes use one math path.

use std::{collections::VecDeque, rc::Rc, sync::Arc};

use half::bf16;

use super::{
    HIDDEN, INTERMEDIATE,
    gguf::{ExpertPlane, Weights},
    mlx::MlxExpertCacheStats,
};
use crate::{
    EngError,
    kernels::MlxKernels,
    tensor::{Dtype, ExecutionTensor, Tensor},
    weights::dequantize_into,
};

const LARGEST_PLANE_F32_BYTES: usize = 2 * HIDDEN * INTERMEDIATE * 4;

pub(super) fn scratch_bytes() -> u64 {
    // Packed expert input, converted F32 plane, resident import and allocator
    // workspace, in addition to the common text decoder's temporary projections.
    super::gguf::projection_scratch_bytes() + 4 * LARGEST_PLANE_F32_BYTES as u64
}

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

pub(super) struct Experts {
    source: Arc<Weights>,
    metal: MlxKernels,
    packed: Cache<ExpertPlane>,
    resident: Cache<ExecutionTensor>,
    stats: MlxExpertCacheStats,
}

impl Experts {
    pub(super) fn open(source: Arc<Weights>, headroom: usize) -> Result<Self, EngError> {
        // Metal and RAM share physical memory. The combined capacity is charged
        // against one admission budget, independent of which tier serves a hit.
        let ram = headroom / 4;
        Ok(Self {
            source,
            metal: MlxKernels::try_new()?,
            packed: Cache::new(ram),
            resident: Cache::new(headroom - ram),
            stats: MlxExpertCacheStats::default(),
        })
    }

    pub(super) fn stats(&self) -> MlxExpertCacheStats {
        self.stats
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
                let plane = Rc::new(self.source.read_expert(name, expert)?);
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
            let weight = Rc::new(self.metal.import_tensor(&decode_transposed(&plane)?)?);
            self.stats.evictions += self.resident.put(
                (name.into(), expert),
                plane.width * plane.height * 4,
                Rc::clone(&weight),
            );
            self.stats.cached_bytes = self.packed.bytes + self.resident.bytes;
            weight
        };
        let activation = Tensor::try_from_host(
            Dtype::F32,
            vec![1, input.len() as u32],
            input.iter().flat_map(|value| value.to_le_bytes()).collect(),
        )?;
        let activation = self.metal.import_tensor(&activation)?;
        let projected = self.metal.mm(&activation, &weight)?;
        let host = self.metal.readback(&projected)?;
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
