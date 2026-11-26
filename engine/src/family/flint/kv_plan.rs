//! Host allocation and physical storage policy for the CUDA state.
//! Physical capacity is separate from the causal window and compact checkpoint rows.

use super::{config::FULL_LAYERS, reference::ChunkWindow};
use crate::EngError;
use crate::family::kv_format::{KvFormat, KvRowShape};

pub(crate) const LAYERS: usize = 49;
pub(super) const WINDOW: usize = 513;
const MODEL_CONTEXT_LIMIT: usize = 262_144;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum KvStorage {
    F16,
    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "the int8 K/V layout is built by the staged CUDA views and tests only"
        )
    )]
    Q8G32,
}

/// The geometry every Flint K/V plane allocates: 4 KV heads of 128 each.
#[allow(
    dead_code,
    reason = "Portable serving does not use the CUDA KV adapter"
)]
const FLINT_KV_ROW: KvRowShape = KvRowShape {
    heads_kv: 4,
    head_dim: 128,
};

#[allow(
    dead_code,
    reason = "Portable serving does not use the CUDA KV adapter"
)]
impl KvStorage {
    /// The shared-format identity of this storage: the byte arithmetic lives in
    /// `family::kv_format`; this mapping carries only the family vocabulary.
    pub(crate) fn format(self) -> KvFormat {
        match self {
            Self::F16 => KvFormat::F16,
            // One f16 scale per 32-element group: 4 heads * (128/32) groups * 2 bytes.
            Self::Q8G32 => KvFormat::Int8 {
                group: 32,
                scale_bytes: 2,
            },
        }
    }

    /// Payload and scale bytes per row of one K or V plane, computed by the
    /// shared format. This is the production path; Flint's fixed geometry
    /// (128 divides into 32-wide groups, 4*128 elements) cannot trip the
    /// shared refusals, so the error arm is unreachable and folds onto the
    /// family's byte-error text. The literal test in `kv_plan_tests` keeps
    /// the equality to the old numbers honest.
    pub(crate) fn checked_plane_row_bytes(self) -> Result<(usize, usize), EngError> {
        let format = self.format();
        match (
            format.payload_bytes(FLINT_KV_ROW),
            format.scale_bytes(FLINT_KV_ROW),
        ) {
            (Ok(data), Ok(scales)) => Ok((
                usize::try_from(data).map_err(|_| byte_error())?,
                usize::try_from(scales).map_err(|_| byte_error())?,
            )),
            _ => Err(byte_error()),
        }
    }

    /// Total encoded bytes per row of one K or V plane (test helper).
    #[cfg(test)]
    pub(crate) fn row_bytes(self) -> usize {
        let (data, scales) = self
            .checked_plane_row_bytes()
            .expect("the shared Flint row arithmetic cannot refuse its own geometry");
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

    #[cfg(test)]
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
    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "the int8 K/V layout is built by the staged CUDA views and tests only"
        )
    )]
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

    #[cfg(test)]
    pub(crate) fn k_storage(&self) -> KvStorage {
        self.k_storage
    }

    #[cfg(test)]
    pub(crate) fn v_storage(&self) -> KvStorage {
        self.v_storage
    }

    /// Encoded (K, V) row bytes for one cache position, each plane on its
    /// own storage, computed by the shared format per plane. Returns the
    /// shared refusal if either plane's geometry is rejected; Flint's fixed
    /// geometry never trips it. Tests unwrap, production propagates.
    #[cfg(test)]
    pub(crate) fn per_plane_row_bytes(&self) -> Result<(usize, usize), EngError> {
        let k = self.k_storage.checked_plane_row_bytes()?;
        let v = self.v_storage.checked_plane_row_bytes()?;
        Ok((k.0 + k.1, v.0 + v.1))
    }

    #[cfg(test)]
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
