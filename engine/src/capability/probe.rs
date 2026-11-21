//! The one hardware probe behind `doctor` and `check`.
//!
//! Device memory comes from `kernels::cuda` (the same queries the loaders
//! use); host memory is parsed from `/proc/meminfo` on Linux. Families keep
//! their own admission-time reads for now; this probe is the user-facing
//! report, and consolidating the loaders onto it is follow-up work, not a
//! silent side effect of adding a report.

use super::Probe;
use serde::Serialize;

/// Which native backends are compiled into this binary. A build fact, not
/// a measurement: a compiled-in backend can still find no device.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CompiledBackends {
    pub cuda: bool,
    pub mlx: bool,
}

impl CompiledBackends {
    pub fn of_this_build() -> Self {
        Self {
            cuda: cfg!(all(feature = "native-cuda", target_os = "linux")),
            mlx: cfg!(all(
                feature = "native-mlx",
                target_os = "macos",
                target_arch = "aarch64"
            )),
        }
    }

    /// True when no GPU backend is compiled in: this binary can serve only
    /// the `fake` family, whatever the machine holds.
    pub fn none(&self) -> bool {
        !self.cuda && !self.mlx
    }
}

/// CUDA facts. Memory figures are for device 0, the default serving
/// device. A Qwen3.6 engine can also span the devices its
/// `cuda_devices` setting names; that placement has its own planner
/// and receipt at load (`family::qwen36::cuda::placement`), and this
/// probe does not reproduce it — multi-device figures are per-device
/// budgets, not one sum.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CudaFacts {
    pub device_count: Probe<usize>,
    pub first_device_integrated: Probe<bool>,
    pub first_device_free_bytes: Probe<u64>,
    pub first_device_total_bytes: Probe<u64>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct HostFacts {
    pub mem_total_bytes: Probe<u64>,
    pub mem_available_bytes: Probe<u64>,
    pub cpu_threads: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct HardwareFacts {
    pub backends: CompiledBackends,
    pub cuda: CudaFacts,
    pub host: HostFacts,
}

#[cfg(all(feature = "native-cuda", target_os = "linux"))]
fn unavailable<T>(error: &crate::EngError) -> Probe<T> {
    Probe::Unavailable(super::describe_error(error))
}

/// Probe this machine once. Never call from a serving process: a failed
/// CUDA query latches the backend terminal (see the module docs).
pub fn probe() -> HardwareFacts {
    let backends = CompiledBackends::of_this_build();
    let cuda = if backends.cuda {
        probe_cuda()
    } else {
        let reason = "the CUDA backend is not compiled into this build".to_string();
        CudaFacts {
            device_count: Probe::Unavailable(reason.clone()),
            first_device_integrated: Probe::Unavailable(reason.clone()),
            first_device_free_bytes: Probe::Unavailable(reason.clone()),
            first_device_total_bytes: Probe::Unavailable(reason),
        }
    };
    HardwareFacts {
        backends,
        cuda,
        host: probe_host(),
    }
}

#[cfg(not(all(feature = "native-cuda", target_os = "linux")))]
fn probe_cuda() -> CudaFacts {
    // `probe` only calls this when the same cfg compiled the backend in;
    // this arm exists so non-CUDA builds (which have no `kernels::cuda`
    // module at all) still type-check.
    let reason = "the CUDA backend is not compiled into this build".to_string();
    CudaFacts {
        device_count: Probe::Unavailable(reason.clone()),
        first_device_integrated: Probe::Unavailable(reason.clone()),
        first_device_free_bytes: Probe::Unavailable(reason.clone()),
        first_device_total_bytes: Probe::Unavailable(reason),
    }
}

#[cfg(all(feature = "native-cuda", target_os = "linux"))]
fn probe_cuda() -> CudaFacts {
    let count = crate::kernels::cuda::device_count();
    let device_count = match &count {
        Ok(n) => Probe::Measured(*n),
        Err(error) => unavailable(error),
    };
    if count != Ok(1) && !matches!(count, Ok(n) if n > 1) {
        let reason = match count {
            Ok(_) => "no CUDA device is present".to_string(),
            Err(error) => super::describe_error(&error),
        };
        return CudaFacts {
            device_count,
            first_device_integrated: Probe::Unavailable(reason.clone()),
            first_device_free_bytes: Probe::Unavailable(reason.clone()),
            first_device_total_bytes: Probe::Unavailable(reason),
        };
    }
    let integrated = match crate::kernels::cuda::device_is_integrated() {
        Ok(value) => Probe::Measured(value),
        Err(error) => unavailable(&error),
    };
    let (free, total) = match crate::kernels::cuda::memory_info_first_device() {
        Ok((free, total)) => (Probe::Measured(free as u64), Probe::Measured(total as u64)),
        Err(error) => (unavailable(&error), unavailable(&error)),
    };
    CudaFacts {
        device_count,
        first_device_integrated: integrated,
        first_device_free_bytes: free,
        first_device_total_bytes: total,
    }
}

/// Host facts only. Unlike `probe`, this is safe inside a serving
/// process: it reads the host memory figures and the CPU count, and makes no
/// CUDA call that could latch the backend. The memory pair comes from the
/// one host reader of `family::mem` (the cgroup-aware budget the families'
/// host guards admit against, so a preflight plan agrees with the start in a
/// capped container, on every Linux build); when the cgroup tree cannot be
/// verified, the pair is the bare `/proc/meminfo` figures with a logged
/// `cgroup budget unknown` line, so nothing that worked inside a container
/// goes dark.
pub(crate) fn probe_host() -> HostFacts {
    let (total, available) = read_meminfo().map_or_else(
        |reason| {
            (
                Probe::Unavailable(reason.clone()),
                Probe::Unavailable(reason),
            )
        },
        |(total, available)| (Probe::Measured(total), Probe::Measured(available)),
    );
    HostFacts {
        mem_total_bytes: total,
        mem_available_bytes: available,
        cpu_threads: std::thread::available_parallelism().map_or(1, usize::from),
    }
}

fn read_meminfo() -> Result<(u64, u64), String> {
    #[cfg(target_os = "linux")]
    {
        crate::family::mem::HostFacts::read("Hardware facts", "the host")
            .map(|host| host.host_memory())
            .map_err(|error| format!("the host memory probe refused: {error:?}"))
    }
    #[cfg(not(target_os = "linux"))]
    {
        Err("host memory probing is implemented on Linux only; on the MLX arm the load-time session plan is the memory authority".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::CompiledBackends;

    /// `check` and the starts read the host through one function, so one machine has one
    /// budget whichever build asks: the report's total is the starts' total.
    #[cfg(target_os = "linux")]
    #[test]
    fn the_report_reads_the_host_through_the_one_reader() {
        let report = super::probe_host();
        let facts = crate::family::mem::HostFacts::read("the test", "the test").unwrap();
        assert_eq!(
            report.mem_total_bytes,
            super::Probe::Measured(facts.total())
        );
        assert!(matches!(
            report.mem_available_bytes,
            super::Probe::Measured(bytes) if bytes > 0 && bytes <= facts.total()
        ));
    }

    #[test]
    fn compiled_backends_match_the_feature_gates() {
        let backends = CompiledBackends::of_this_build();
        assert_eq!(
            backends.cuda,
            cfg!(all(feature = "native-cuda", target_os = "linux"))
        );
        assert_eq!(
            backends.mlx,
            cfg!(all(
                feature = "native-mlx",
                target_os = "macos",
                target_arch = "aarch64"
            ))
        );
        assert_eq!(backends.none(), !backends.cuda && !backends.mlx);
    }
}
