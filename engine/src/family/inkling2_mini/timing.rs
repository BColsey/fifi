//! Optional wall-clock accounting for the native text executor.
use std::{
    sync::atomic::{AtomicU64, Ordering::Relaxed},
    time::Instant,
};

#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
pub(super) fn enabled() -> bool {
    std::env::var_os("FIFI_INKLING_COMPONENT_TIMING").is_some_and(|value| value == "1")
}

pub(super) fn phase_enabled() -> bool {
    cfg!(all(
        feature = "native-mlx",
        target_os = "macos",
        target_arch = "aarch64"
    )) && [
        "FIFI_INKLING_EXPERT_TIMING",
        "FIFI_INKLING_COMPONENT_TIMING",
    ]
    .into_iter()
    .any(|flag| std::env::var_os(flag).is_some_and(|value| value == "1"))
}

#[derive(Default)]
pub(super) struct DenseIo {
    file_bytes: AtomicU64,
    file_ranges: AtomicU64,
    mapped_span_bytes: AtomicU64,
    mapped_accesses: AtomicU64,
}

impl DenseIo {
    pub(super) fn start_if_enabled() -> Option<Self> {
        phase_enabled().then(Self::default)
    }

    pub(super) fn file_read(&self, bytes: usize) {
        if bytes != 0 {
            self.file_bytes.fetch_add(bytes as u64, Relaxed);
            self.file_ranges.fetch_add(1, Relaxed);
        }
    }

    pub(super) fn mapped_access(&self, bytes: usize) {
        if bytes != 0 {
            self.mapped_span_bytes.fetch_add(bytes as u64, Relaxed);
            self.mapped_accesses.fetch_add(1, Relaxed);
        }
    }

    #[cfg(any(
        test,
        all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64")
    ))]
    pub(super) fn snapshot(&self) -> DenseIoStats {
        DenseIoStats {
            file_bytes: self.file_bytes.load(Relaxed),
            file_ranges: self.file_ranges.load(Relaxed),
            mapped_span_bytes: self.mapped_span_bytes.load(Relaxed),
            mapped_accesses: self.mapped_accesses.load(Relaxed),
        }
    }
}

#[cfg(any(
    test,
    all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64")
))]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct DenseIoStats {
    pub(super) file_bytes: u64,
    pub(super) file_ranges: u64,
    pub(super) mapped_span_bytes: u64,
    pub(super) mapped_accesses: u64,
}

#[cfg(any(
    test,
    all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64")
))]
impl DenseIoStats {
    pub(super) fn since(self, earlier: Self) -> Self {
        Self {
            file_bytes: self.file_bytes - earlier.file_bytes,
            file_ranges: self.file_ranges - earlier.file_ranges,
            mapped_span_bytes: self.mapped_span_bytes - earlier.mapped_span_bytes,
            mapped_accesses: self.mapped_accesses - earlier.mapped_accesses,
        }
    }
}

pub(super) fn measure<T>(total_ns: Option<&mut u128>, work: impl FnOnce() -> T) -> T {
    let Some(total_ns) = total_ns else {
        return work();
    };
    let start = Instant::now();
    let result = work();
    *total_ns += start.elapsed().as_nanos();
    result
}

#[cfg(test)]
mod tests {
    #[test]
    fn dense_io_phase_delta_separates_file_reads_from_mapped_spans() {
        let io = super::DenseIo::default();
        io.file_read(32);
        let earlier = io.snapshot();
        std::thread::scope(|scope| {
            for _ in 0..4 {
                scope.spawn(|| {
                    io.file_read(8);
                    io.mapped_access(16);
                });
            }
        });
        io.file_read(0);
        io.mapped_access(0);
        assert_eq!(
            io.snapshot().since(earlier),
            super::DenseIoStats {
                file_bytes: 32,
                file_ranges: 4,
                mapped_span_bytes: 64,
                mapped_accesses: 4,
            },
        );
    }

    #[test]
    fn measured_and_disabled_scopes_return_the_operation_result() {
        let mut elapsed = 0;
        let mut calls = 0;
        let mut operation = || {
            calls += 1;
            Err::<(), _>("kept")
        };
        assert_eq!(super::measure(None, &mut operation), Err("kept"));
        assert_eq!(super::measure(Some(&mut elapsed), operation), Err("kept"));
        assert_eq!(calls, 2);
    }
}
