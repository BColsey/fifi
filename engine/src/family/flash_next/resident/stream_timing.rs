use super::StepMode;
use crate::{family::mlx_stream_stats::StreamSnapshot, kernels::MlxKernels};

pub(super) struct ForwardTiming {
    counters: StreamSnapshot,
    position: u32,
}

pub(super) fn begin(kernels: &MlxKernels, position: u32) -> Option<ForwardTiming> {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    ENABLED
        .get_or_init(|| {
            std::env::var_os("FIFI_FLASH_NEXT_MLX_STREAM_TIMING").is_some_and(|value| value == "1")
        })
        .then(|| ForwardTiming {
            counters: StreamSnapshot::begin(kernels),
            position,
        })
}

impl ForwardTiming {
    pub(super) fn finish(self, kernels: &MlxKernels, rows: u32, mode: StepMode, route: &str) {
        let mut row = self.counters.finish(kernels);
        row["position"] = self.position.into();
        row["rows"] = rows.into();
        row["mode"] = format!("{mode:?}").into();
        row["forward_path"] = route.into();
        eprintln!("flash_next_stream_timing {row}");
    }
}
