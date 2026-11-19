//! Qualification counters for disk-backed expert stages in one Flint worker.

use std::{sync::OnceLock, time::Instant};

use super::weights::records::{LayerReads, layer_reads};
use crate::{
    family::stored::{ExpertReadStats, qualification_expert_reads},
    kernels::MlxKernels,
};

pub(super) const EVENT: &str = "flint_expert_timing";

pub(super) fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os("FIFI_FLINT_EXPERT_TIMING").is_some_and(|value| value == "1")
    })
}

pub(super) struct ForwardTiming {
    started: Instant,
    stats: ExpertReadStats,
    page_copies: u64,
    position: u32,
    layer: LayerReads,
}

impl ForwardTiming {
    pub(super) fn begin(k: &MlxKernels, position: u32) -> Option<Self> {
        enabled().then(|| Self {
            started: Instant::now(),
            stats: qualification_expert_reads(),
            page_copies: k.qualification_page_copies(),
            position,
            layer: layer_reads(),
        })
    }

    pub(super) fn finish(self, k: &MlxKernels, rows: u32, full_kv_rows: u32) {
        let forward_ns = self.started.elapsed().as_nanos() as u64;
        let end = qualification_expert_reads();
        let layer = layer_reads();
        eprintln!(
            "{EVENT} {}",
            serde_json::json!({
                "position": self.position,
                "rows": rows,
                "full_kv_rows": full_kv_rows,
                "forward_ns": forward_ns,
                "slice_reads": end.slice_reads - self.stats.slice_reads,
                "read_bytes": end.read_bytes - self.stats.read_bytes,
                "read_ns": end.read_ns - self.stats.read_ns,
                "page_imports": end.page_imports - self.stats.page_imports,
                "import_ns": end.import_ns - self.stats.import_ns,
                "page_copies": k.qualification_page_copies() - self.page_copies,
                "expert_probe_read_ns": layer.read_ns - self.layer.read_ns,
                "expert_probe_stages": layer.stages - self.layer.stages,
                "expert_record_stages": layer.record_stages - self.layer.record_stages,
            })
        );
    }
}
