//! Committed checkpoint payloads and numerical path identity for Qwen CUDA prefix reuse.

use super::state::{
    export::ExportedState,
    pinned::{PendingPinned, PinnedSnapshot},
};
use super::{dims::KvMode, state_layout::StateLayout};
use crate::{
    EngError,
    family::session::{DecodeSession, PrefixIdentity, PrefixSlot, SaveKind},
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
    pub(super) path: (CudaPrefillPath, PrefixIdentity),
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
    /// Diagnostic identity only. The production chunk path still disables reuse.
    #[cfg(test)]
    ChunkedForTest(KvMode, usize),
}

fn overflow() -> EngError {
    EngError::MemoryPlan("prefix entry: a size overflows".to_owned())
}

/// Host bytes of one saved entry at `positions` committed positions: the DeltaNet export
/// (recurrent plus convolution, f32, one layer's worth per DeltaNet layer) plus the exported KV
/// planes of the first `positions` rows of every attention layer. Checked arithmetic only: an
/// absurd count is a `MemoryPlan` refusal, never a panic.
pub(super) fn entry_bytes(layout: &StateLayout, positions: usize) -> Result<usize, EngError> {
    Ok(export_bytes(layout, positions)?.0)
}

// A pageable export retains the committed payload plus one live temporary. DeltaNet
// converts each byte vector to f32. KV readback stages one transfer range at a time.
fn export_bytes(layout: &StateLayout, positions: usize) -> Result<(usize, usize), EngError> {
    let pool = layout.delta;
    let per_layer = pool
        .rec_values
        .checked_add(pool.conv_values)
        .and_then(|values| values.checked_mul(size_of::<f32>()))
        .and_then(|bytes| bytes.checked_mul(pool.layers))
        .ok_or_else(overflow)?;
    let mut total = per_layer;
    let mut temporary = if pool.layers == 0 {
        0
    } else {
        pool.rec_values
            .max(pool.conv_values)
            .checked_mul(size_of::<f32>())
            .ok_or_else(overflow)?
    };
    for ordinal in 0..layout.kv.layers {
        for range in layout.kv.prefix_ranges(ordinal, positions)? {
            total = total.checked_add(range.bytes).ok_or_else(overflow)?;
            temporary = temporary.max(range.bytes.min(crate::kernels::CHUNK_LIMIT));
        }
    }
    Ok((total, temporary))
}

pub(super) const HOST_HEADROOM: usize = crate::family::mem::HOST_HEADROOM_BYTES as usize;

pub(super) fn pageable_required<'a>(
    states: impl Iterator<Item = (&'a StateLayout, usize)>,
) -> Result<usize, EngError> {
    let mut payload = 0usize;
    let mut peak = 0;
    for (layout, positions) in states {
        let (bytes, temporary) = export_bytes(layout, positions)?;
        payload = payload.checked_add(bytes).ok_or_else(overflow)?;
        // Card exports run in order. Only earlier exports remain live beside this shard.
        peak = peak.max(payload.checked_add(temporary).ok_or_else(overflow)?);
    }
    peak.checked_add(HOST_HEADROOM).ok_or_else(overflow)
}

pub(super) fn admit_pageable<'a>(
    states: impl Iterator<Item = (&'a StateLayout, usize)>,
) -> Result<bool, EngError> {
    let required = pageable_required(states)?;
    // The cgroup-aware figure the other admissions size from, with the bare
    // MemAvailable figure when the cgroup tree refuses. A probe that cannot
    // verify anything falls back and logs: it proceeds with the save the
    // engine performed before the check, it does not take a feature away.
    let available = match crate::family::mem::HostFacts::probe_or_bare(
        "q36 pageable prefix",
        "q36 pageable prefix",
    ) {
        Ok((_total, available)) => usize::try_from(available).unwrap_or(usize::MAX),
        Err(error) => {
            eprintln!(
                "q36 cuda: pageable prefix host budget: probe unavailable ({}); proceeding without the check",
                error.describe()
            );
            return Ok(true);
        }
    };
    Ok(admit_or_log(required, available))
}

/// The save decision: one logged line states what a shortfall skip costs -
/// nothing for the request - because the save is optional infrastructure.
pub(super) fn admit_or_log(required: usize, available: usize) -> bool {
    if required > available {
        eprintln!(
            "q36 cuda: pageable prefix host budget: skip save (need {required} bytes, have {available} bytes); the request is unaffected"
        );
        return false;
    }
    true
}
