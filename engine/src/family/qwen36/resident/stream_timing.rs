//! Existing MLX wait counters and host-side staging measurements for a streamed forward.

use std::time::Instant;

use super::{decoder::StepMode, weights::ExpertPages, weights::ResidentWeights};
use crate::{EngError, kernels::MlxKernels};

#[cfg(feature = "native-fault-qualification")]
use crate::{family::stored::ExpertReadStats, kernels::mlx::QualificationTiming};

pub(super) struct ForwardTiming {
    #[cfg(feature = "native-fault-qualification")]
    started: Instant,
    #[cfg(feature = "native-fault-qualification")]
    position: u32,
    #[cfg(feature = "native-fault-qualification")]
    reads: ExpertReadStats,
    #[cfg(feature = "native-fault-qualification")]
    native: QualificationTiming,
    #[cfg(feature = "native-fault-qualification")]
    page_copies: u64,
    #[cfg(feature = "native-fault-qualification")]
    syncs: u64,
    host_route_ns: u64,
    stages: u32,
    stage_fill_ns: u64,
    stage_lifetime_ns: u64,
    peak_stage_bytes: u64,
    peak_stage_experts: u32,
}

fn ns(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

impl ForwardTiming {
    pub(super) fn begin(k: &MlxKernels, weights: &ResidentWeights, position: u32) -> Option<Self> {
        #[cfg(feature = "native-fault-qualification")]
        {
            static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            let enabled = *ENABLED.get_or_init(|| {
                std::env::var_os("FIFI_Q36_MLX_STREAM_TIMING").is_some_and(|value| value == "1")
            });
            (enabled && weights.streams_experts()).then(|| Self {
                started: Instant::now(),
                position,
                reads: crate::family::stored::qualification_expert_reads(),
                native: k.qualification_timing(),
                page_copies: k.qualification_page_copies(),
                syncs: k.qualification_sync_count(),
                host_route_ns: 0,
                stages: 0,
                stage_fill_ns: 0,
                stage_lifetime_ns: 0,
                peak_stage_bytes: 0,
                peak_stage_experts: 0,
            })
        }
        #[cfg(not(feature = "native-fault-qualification"))]
        {
            let _ = (k, weights, position);
            None
        }
    }

    pub(super) fn host_route<T>(timing: Option<&mut Self>, route: impl FnOnce() -> T) -> T {
        let Some(timing) = timing else {
            return route();
        };
        let started = Instant::now();
        let value = route();
        timing.host_route_ns += ns(started);
        value
    }

    pub(super) fn stage_ready(
        timing: Option<&mut Self>,
        started: Option<Instant>,
        pages: Option<&ExpertPages>,
    ) -> Result<(), EngError> {
        if let (Some(timing), Some(started), Some(pages)) = (timing, started, pages) {
            timing.stages += 1;
            timing.stage_fill_ns += ns(started);
            timing.peak_stage_bytes = timing.peak_stage_bytes.max(pages.page_bytes()?);
            timing.peak_stage_experts =
                timing.peak_stage_experts.max(pages.down.logical_shape()[0]);
        }
        Ok(())
    }

    pub(super) fn stage_end(timing: Option<&mut Self>, started: Option<Instant>) {
        if let (Some(timing), Some(started)) = (timing, started) {
            timing.stage_lifetime_ns += ns(started);
        }
    }

    pub(super) fn finish(self, k: &MlxKernels, rows: u32, mode: StepMode) {
        #[cfg(feature = "native-fault-qualification")]
        {
            let forward_ns = ns(self.started);
            let reads = crate::family::stored::qualification_expert_reads();
            let native = k.qualification_timing();
            eprintln!(
                "qwen36_stream_timing {}",
                serde_json::json!({
                    "position": self.position,
                    "rows": rows,
                    "mode": format!("{mode:?}"),
                    "forward_ns": forward_ns,
                    "slice_reads": reads.slice_reads - self.reads.slice_reads,
                    "read_bytes": reads.read_bytes - self.reads.read_bytes,
                    "read_ns": reads.read_ns - self.reads.read_ns,
                    "page_imports": reads.page_imports - self.reads.page_imports,
                    "page_import_ns": reads.import_ns - self.reads.import_ns,
                    "page_copies": k.qualification_page_copies() - self.page_copies,
                    "metal_syncs": k.qualification_sync_count() - self.syncs,
                    "native_import_calls": native.import_calls - self.native.import_calls,
                    "native_import_ns": native.import_ns - self.native.import_ns,
                    "native_readback_calls": native.readback_calls - self.native.readback_calls,
                    "native_readback_ns": native.readback_ns - self.native.readback_ns,
                    "native_evaluate_calls": native.evaluate_calls - self.native.evaluate_calls,
                    "native_evaluate_ns": native.evaluate_ns - self.native.evaluate_ns,
                    "host_route_ns": self.host_route_ns,
                    "stages": self.stages,
                    "stage_fill_ns": self.stage_fill_ns,
                    "stage_lifetime_ns": self.stage_lifetime_ns,
                    "peak_stage_bytes": self.peak_stage_bytes,
                    "peak_stage_experts": self.peak_stage_experts,
                })
            );
        }
        #[cfg(not(feature = "native-fault-qualification"))]
        let _ = (k, rows, mode);
    }
}
