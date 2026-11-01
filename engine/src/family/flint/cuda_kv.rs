//! Explicit F16/Q8G32 allocation adapter for the Flint CUDA caller; no default selection.

use super::{
    config::FULL_LAYERS,
    kv_plan::{KvPlan, KvStorage},
    reference::ChunkWindow,
};
use crate::EngError;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct LayerKvAllocation {
    storage: KvStorage,
    /// V plane storage; equals `storage` unless a mixed profile ("q8k")
    /// set the planes apart.
    v_storage: KvStorage,
    cache_rows: usize,
    data_bytes: usize,
    scale_bytes: usize,
    v_data_bytes: usize,
    v_scale_bytes: usize,
    context_rows: usize,
    window_rows: usize,
}

impl LayerKvAllocation {
    pub(crate) fn new(plan: &KvPlan, layer: usize) -> Result<Self, EngError> {
        let cache_rows = plan.cache_rows(layer)?;
        let plane = |storage: KvStorage| {
            let (data_row, scale_row) = storage.plane_row_bytes();
            let plane_bytes = cache_rows
                .checked_mul(data_row + scale_row)
                .ok_or_else(|| EngError::MemoryPlan("Flint KV plane size overflows".into()))?;
            Ok::<_, EngError>((plane_bytes - scale_row * cache_rows, scale_row * cache_rows))
        };
        let (data_bytes, scale_bytes) = plane(plan.k_storage())?;
        let (v_data_bytes, v_scale_bytes) = plane(plan.v_storage())?;
        Ok(Self {
            storage: plan.k_storage(),
            v_storage: plan.v_storage(),
            cache_rows,
            data_bytes,
            scale_bytes,
            v_data_bytes,
            v_scale_bytes,
            context_rows: plan.context_rows(),
            window_rows: if FULL_LAYERS.contains(&(layer as u64)) {
                plan.context_rows()
            } else {
                plan.window()
            },
        })
    }

    #[cfg(test)]
    pub(crate) fn cache_rows(&self) -> usize {
        self.cache_rows
    }

    /// One K or V payload including its scales, not the combined K+V charge.
    #[cfg(test)]
    pub(crate) fn plane_bytes(&self) -> usize {
        self.data_bytes + self.scale_bytes
    }

    /// V-plane payload plus scales; equals [`Self::plane_bytes`] on every
    /// uniform profile.
    #[cfg(test)]
    pub(crate) fn v_plane_bytes(&self) -> usize {
        self.v_data_bytes + self.v_scale_bytes
    }

    /// Host bytes of one data plane covering the committed rows
    /// `[0, committed)`: the prefix-snapshot copy bound.
    /// Rows below the first wrap sit at slots `[0, committed)` (slot =
    /// position % cache_rows), so they are a plane-prefix byte range; a
    /// wrapped ring (committed >= cache_rows) needs the whole plane.
    /// Scales are excluded: f16-only snapshots carry none.
    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    pub(crate) fn committed_data_bytes(&self, committed: usize) -> usize {
        self.data_bytes * committed.min(self.cache_rows) / self.cache_rows
    }

    #[cfg(test)]
    pub(crate) fn plane_components(&self) -> (usize, usize) {
        (self.data_bytes, self.scale_bytes)
    }

    pub(crate) fn cursor(&self) -> KvCursor {
        KvCursor {
            allocation: *self,
            committed_prefix: 0,
            poisoned: false,
        }
    }

    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    pub(crate) fn allocate(
        &self,
        ctx: &crate::kernels::cuda::CudaContext,
    ) -> Result<LayerKv, EngError> {
        Ok(LayerKv {
            keys: ctx.allocate(self.data_bytes)?,
            values: ctx.allocate(self.v_data_bytes)?,
            scales: match (self.scale_bytes, self.v_scale_bytes) {
                (0, 0) => None,
                // Uniform: both planes quantized, both scale planes live.
                (same, v) if same == v && same > 0 => Some([
                    ctx.allocate(self.scale_bytes)?,
                    ctx.allocate(self.v_scale_bytes)?,
                ]),
                // Mixed "q8k": the f16 V plane carries no scales; the empty
                // placeholder keeps the pair shape and is never read or written.
                (k, 0) if k > 0 => Some([ctx.allocate(self.scale_bytes)?, ctx.empty_buffer()?]),
                (0, v) if v > 0 => Some([ctx.empty_buffer()?, ctx.allocate(self.v_scale_bytes)?]),
                _ => {
                    return Err(EngError::MemoryPlan(
                        "Flint KV scale planes disagree across K and V".into(),
                    ));
                }
            },
            cursor: self.cursor(),
        })
    }
}

pub(crate) struct KvCursor {
    allocation: LayerKvAllocation,
    committed_prefix: usize,
    poisoned: bool,
}

impl KvCursor {
    pub(crate) fn committed_prefix(&self) -> usize {
        self.committed_prefix
    }

    #[cfg(all(test, feature = "native-cuda", target_os = "linux"))]
    pub(crate) fn test_poisoned(&self) -> bool {
        self.poisoned
    }

    /// The native owner must drain all in-flight users before this metadata reset.
    pub(crate) fn reset_after_quiescence(&mut self) {
        self.committed_prefix = 0;
        self.poisoned = false;
    }

    pub(crate) fn begin_append(
        &mut self,
        start: usize,
        rows: usize,
    ) -> Result<KvAppend<'_>, EngError> {
        if self.poisoned {
            return Err(EngError::Backend {
                operation: "flint_kv_append",
                message: "Flint KV cursor requires reset after an uncommitted append".into(),
            });
        }
        if start != self.committed_prefix {
            return Err(EngError::BadRequest(
                "Flint KV append must continue the committed absolute prefix".into(),
            ));
        }
        let window = ChunkWindow::new(self.allocation.window_rows, start, rows)?;
        if window.live_range().end > self.allocation.context_rows {
            return Err(EngError::ContextLimit(
                "Flint KV append exceeds the admitted logical context".into(),
            ));
        }
        window.slot(start, self.allocation.cache_rows)?;
        // Mark first so even a forgotten append view remains fail-closed.
        self.poisoned = true;
        Ok(KvAppend {
            cursor: self,
            window,
        })
    }
}

/// Exclusive append view; the integration owner accepts stream/status before commit.
pub(crate) struct KvAppend<'a> {
    cursor: &'a mut KvCursor,
    window: ChunkWindow,
}

impl KvAppend<'_> {
    pub(crate) fn live_range(&self) -> std::ops::Range<usize> {
        self.window.live_range()
    }

    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    pub(crate) fn window(&self) -> usize {
        self.window.window()
    }

    pub(crate) fn storage(&self) -> KvStorage {
        self.cursor.allocation.storage
    }

    #[allow(
        dead_code,
        reason = "Portable builds never query the V plane; the mixed profile is CUDA-only"
    )]
    pub(crate) fn v_storage(&self) -> KvStorage {
        self.cursor.allocation.v_storage
    }

    pub(crate) fn cache_rows(&self) -> usize {
        self.cursor.allocation.cache_rows
    }

    pub(crate) fn slot(&self, position: usize) -> Result<usize, EngError> {
        self.window
            .slot(position, self.cursor.allocation.cache_rows)
    }

    /// Physical cell start and source-row range; an empty second run means no wrap.
    pub(crate) fn runs(&self) -> Result<[(usize, std::ops::Range<usize>); 2], EngError> {
        let start = self.cursor.committed_prefix;
        let rows = self.window.live_range().end - start;
        let slot = self.slot(start)?;
        let first = rows.min(self.cache_rows() - slot);
        Ok([(slot, 0..first), (0, first..rows)])
    }

    pub(crate) fn commit(self) {
        self.cursor.committed_prefix = self.window.live_range().end;
        self.cursor.poisoned = false;
    }
}

impl KvCursor {
    /// Installs a restored committed boundary (prefix snapshot restore):
    /// planes were refilled whole, so the cursor only needs the count.
    /// Only the native-cuda restore path calls it (portable test builds
    /// compile the cursor but never restore).
    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    pub(crate) fn restore_committed(&mut self, committed: usize) -> Result<(), EngError> {
        if self.poisoned || self.committed_prefix != 0 {
            return Err(EngError::BadRequest(
                "flint cuda prefix restore needs a reset cursor".into(),
            ));
        }
        self.committed_prefix = committed;
        Ok(())
    }
}

#[cfg(all(feature = "native-cuda", target_os = "linux"))]
pub(crate) struct LayerKv {
    pub(crate) keys: crate::kernels::cuda::DeviceBuffer,
    pub(crate) values: crate::kernels::cuda::DeviceBuffer,
    pub(crate) scales: Option<[crate::kernels::cuda::DeviceBuffer; 2]>,
    pub(crate) cursor: KvCursor,
}

#[cfg(all(feature = "native-cuda", target_os = "linux"))]
impl LayerKv {
    /// The committed-bounds data-plane byte size (see
    /// [`LayerKvAllocation::committed_data_bytes`]) for this layer's planes.
    pub(crate) fn committed_data_bytes(&self, committed: usize) -> usize {
        self.cursor.allocation.committed_data_bytes(committed)
    }
}
