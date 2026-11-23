//! Routed expert placement measured at startup, without device handles.

use serde::Serialize;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
#[cfg_attr(
    not(any(test, all(feature = "native-cuda", target_os = "linux"))),
    allow(dead_code, reason = "currently reported only by Flash-Next CUDA")
)]
pub(crate) enum ExpertPlacementMode {
    Resident,
    HostStreamed,
    DiskStreamed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub(crate) struct ExpertPlacement {
    pub(crate) mode: ExpertPlacementMode,
    /// The sized device expert bank, excluding other model weights and scratch.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) vram_bytes: Option<u64>,
    /// The pinned backing store, excluding staging and the disk victim cache.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) ram_bytes: Option<u64>,
    /// Stored expert tensor bytes, when disk is a serving source and the size is known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) disk_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) slots_per_layer: Option<usize>,
}
