//! Committed checkpoint payloads and numerical path identity for Qwen CUDA prefix reuse.

use super::state::{
    export::ExportedState,
    pinned::{PendingPinned, PinnedSnapshot},
};
use super::{dims::KvMode, state::export::plane_sizes, state_layout::StateLayout};
use crate::{
    EngError,
    family::session::{DecodeSession, PrefixSlot, SaveKind},
    sched::job::PrefixReport,
};

pub(super) enum CudaSavedPrefix {
    Pageable(ExportedState),
    Pinned {
        state: PinnedSnapshot,
        payload_bytes: usize,
    },
}

pub(super) trait CheckpointPayload {
    fn payload_bytes(&self) -> u64;
}

impl CheckpointPayload for CudaSavedPrefix {
    fn payload_bytes(&self) -> u64 {
        match self {
            Self::Pageable(state) => state.payload_bytes(),
            Self::Pinned { payload_bytes, .. } => *payload_bytes as u64,
        }
    }
}

impl CheckpointPayload for Vec<ExportedState> {
    fn payload_bytes(&self) -> u64 {
        self.iter().map(ExportedState::payload_bytes).sum()
    }
}

/// Only published roots and LRU entries contribute. Reading the report never completes a copy.
pub(super) fn stored_prefix_report<S: CheckpointPayload, P>(
    store: &PrefixSlot<S, P>,
    lookup: PrefixReport,
) -> PrefixReport {
    PrefixReport {
        entries: u32::try_from(store.entries_len().saturating_add(store.roots_len()))
            .unwrap_or(u32::MAX),
        bytes: store
            .snapshots()
            .map(CheckpointPayload::payload_bytes)
            .sum(),
        ..lookup
    }
}

pub(super) struct PendingPrefix {
    pub(super) session: DecodeSession,
    pub(super) fed: usize,
    pub(super) kind: SaveKind,
    pub(super) path: CudaPrefillPath,
    pub(super) state: PendingPinned,
}
/// The numeric prefill path that built a saved entry: a state restores only under the path that
/// built it, because a batched prefill matches one-token steps only within a tolerance. The key
/// also binds the actual KV format before native restore validation; chunked prefill must add
/// `Chunked(n)` and pass it here, exactly like the Flash-Next twin.
///
/// Neither this path nor `StateIdentity` covers the WEIGHT LAYOUT: an entry matches the device
/// buffers it was exported from only while it lives in memory, inside one step's lifetime (one
/// process, one loaded model, where the layout cannot change). Persisting an entry, or a store
/// shared by several steps, must add the weight layout to what a restore checks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum CudaPrefillPath {
    Steps(KvMode),
}

fn overflow() -> EngError {
    EngError::MemoryPlan("prefix entry: a size overflows".to_owned())
}

/// Host bytes of one saved entry at `positions` committed positions: the DeltaNet export
/// (recurrent plus convolution, f32, one layer's worth per DeltaNet layer) plus the exported KV
/// planes of the first `positions` rows of every attention layer. Checked arithmetic only: an
/// absurd count is a `MemoryPlan` refusal, never a panic.
pub(super) fn entry_bytes(layout: &StateLayout, positions: usize) -> Result<usize, EngError> {
    let pool = layout.delta;
    let per_layer = pool
        .rec_values
        .checked_add(pool.conv_values)
        .and_then(|values| values.checked_mul(size_of::<f32>()))
        .and_then(|bytes| bytes.checked_mul(pool.layers))
        .ok_or_else(overflow)?;
    let mut total = per_layer;
    for ordinal in 0..layout.kv.layers {
        for bytes in plane_sizes(&layout.kv, ordinal, positions)? {
            total = total.checked_add(bytes).ok_or_else(overflow)?;
        }
    }
    Ok(total)
}
