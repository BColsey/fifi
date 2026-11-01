//! Host allocation and physical storage policy for the CUDA state.
//! Physical capacity is separate from the causal window and compact checkpoint rows.

use super::{config::FULL_LAYERS, reference::ChunkWindow};
use crate::EngError;

pub(crate) const LAYERS: usize = 49;
pub(super) const WINDOW: usize = 513;
const MODEL_CONTEXT_LIMIT: usize = 262_144;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum KvStorage {
    F16,
    Q8G32,
}

#[allow(
    dead_code,
    reason = "Portable serving does not use the CUDA KV adapter"
)]
impl KvStorage {
    /// Payload and F16 scale bytes per row of one K or V plane.
    pub(crate) fn plane_row_bytes(self) -> (usize, usize) {
        match self {
            Self::F16 => (4 * 128 * 2, 0),
            Self::Q8G32 => (4 * 128, 4 * (128 / 32) * 2),
        }
    }

    pub(crate) fn row_bytes(self) -> usize {
        let (data, scales) = self.plane_row_bytes();
        2 * (data + scales)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvPlan {
    context_rows: usize,
    /// Per-plane storage; K and V may differ (profile parameter: "q8k"
    /// quantizes K and keeps V at F16). `with_storage` sets both together
    /// and stays the default uniform path.
    k_storage: KvStorage,
    v_storage: KvStorage,
    cache_rows: [usize; LAYERS],
    snapshots: Option<SnapshotPlan>,
    /// Most rows one append may carry.
    chunk_rows: usize,
    /// Ring attention window the capacities were computed with; production
    /// [`Self::new`] always passes WINDOW, tests may substitute a smaller one.
    window: usize,
}

impl KvPlan {
    /// The caller admits backend context/chunk limits first. False retains full capacities
    /// and disables these checkpoints; no environment variable or native state is changed.
    pub fn new(context: usize, chunk_rows: usize, ring_enabled: bool) -> Result<Self, EngError> {
        Self::new_with_window(context, chunk_rows, ring_enabled, WINDOW)
    }

    /// Test-only variant of [`Self::new`] substituting the ring window so a short walk
    /// still crosses the wrap boundary; absent from production builds.
    #[cfg(test)]
    pub(crate) fn new_test_window(
        context: usize,
        chunk_rows: usize,
        ring_enabled: bool,
        window: usize,
    ) -> Result<Self, EngError> {
        Self::new_with_window(context, chunk_rows, ring_enabled, window)
    }

    fn new_with_window(
        context: usize,
        chunk_rows: usize,
        ring_enabled: bool,
        window: usize,
    ) -> Result<Self, EngError> {
        if context == 0 || chunk_rows == 0 {
            return Err(EngError::BadRequest(
                "Flint KV context and chunk rows must be positive".into(),
            ));
        }
        if context > MODEL_CONTEXT_LIMIT {
            return Err(EngError::ContextLimit(format!(
                "Flint KV context exceeds {MODEL_CONTEXT_LIMIT} positions"
            )));
        }
        let required = ChunkWindow::new(window, window - 1, chunk_rows)?.required_slots();
        let mut cache_rows = [context; LAYERS];
        if ring_enabled {
            for (layer, capacity) in cache_rows.iter_mut().enumerate() {
                if !FULL_LAYERS.contains(&(layer as u64)) {
                    *capacity = context.min(required);
                }
            }
        }
        Ok(Self {
            context_rows: context,
            k_storage: KvStorage::F16,
            v_storage: KvStorage::F16,
            cache_rows,
            snapshots: ring_enabled.then_some(SnapshotPlan {
                rows: context.min(window),
            }),
            chunk_rows,
            window,
        })
    }

    /// Most rows one append may carry, which the ring capacities were computed with.
    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    pub(crate) fn chunk_rows(&self) -> usize {
        self.chunk_rows
    }

    #[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
    pub(crate) fn window(&self) -> usize {
        self.window
    }

    #[allow(
        dead_code,
        reason = "Only the uniform-profile tests use the both-planes sugar"
    )]
    pub(crate) fn with_storage(mut self, storage: KvStorage) -> Self {
        self.k_storage = storage;
        self.v_storage = storage;
        self
    }

    /// Per-plane profile parameter; only an explicit mixed profile
    /// (settings "q8k") reaches a non-uniform plan.
    pub(crate) fn with_kv_storage(mut self, k: KvStorage, v: KvStorage) -> Self {
        self.k_storage = k;
        self.v_storage = v;
        self
    }

    #[allow(
        dead_code,
        reason = "Portable serving does not use the CUDA KV adapter"
    )]
    pub(crate) fn storage(&self) -> KvStorage {
        self.k_storage
    }

    #[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
    pub(crate) fn k_storage(&self) -> KvStorage {
        self.k_storage
    }

    #[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
    pub(crate) fn v_storage(&self) -> KvStorage {
        self.v_storage
    }

    /// Encoded (K, V) row bytes for one cache position, each plane on its
    /// own storage.
    #[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
    pub(crate) fn per_plane_row_bytes(&self) -> (usize, usize) {
        let k = self.k_storage.plane_row_bytes();
        let v = self.v_storage.plane_row_bytes();
        (k.0 + k.1, v.0 + v.1)
    }

    #[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
    pub(crate) fn context_rows(&self) -> usize {
        self.context_rows
    }

    pub fn cache_rows(&self, layer: usize) -> Result<usize, EngError> {
        self.cache_rows
            .get(layer)
            .copied()
            .ok_or_else(|| EngError::BadRequest("Flint layer is outside 0..49".into()))
    }

    /// Encoded K/V bytes per layer-position include codec scales. Non-row padding,
    /// allocator overhead, and other owners must be charged separately by the caller.
    pub fn bytes(&self, row_bytes: usize) -> Result<usize, EngError> {
        if row_bytes == 0 {
            return Err(byte_error());
        }
        self.cache_rows.iter().try_fold(0usize, |total, rows| {
            rows.checked_mul(row_bytes)
                .and_then(|bytes| total.checked_add(bytes))
                .ok_or_else(byte_error)
        })
    }

    pub fn snapshots(&self) -> Option<SnapshotPlan> {
        self.snapshots
    }
}

/// Maximum compact payload shape. Copy ownership/publication belongs to the later state slice.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SnapshotPlan {
    rows: usize,
}

impl SnapshotPlan {
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Two reusable slots, including an in-progress replacement, never a third staging slot.
    pub fn keep(&self) -> usize {
        2
    }

    /// Charge both maximum payloads, with last logits in their actual encoded format.
    pub fn bytes(&self, row_bytes: usize, logits_bytes: usize) -> Result<usize, EngError> {
        if row_bytes == 0 {
            return Err(byte_error());
        }
        self.rows
            .checked_mul(LAYERS - FULL_LAYERS.len())
            .and_then(|rows| rows.checked_mul(row_bytes))
            .and_then(|bytes| bytes.checked_add(logits_bytes))
            .and_then(|bytes| bytes.checked_mul(self.keep()))
            .ok_or_else(byte_error)
    }
}

fn byte_error() -> EngError {
    EngError::MemoryPlan("Flint KV row byte charge is zero or the total overflows".into())
}
