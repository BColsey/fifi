#[cfg(feature = "research-activation")]
pub mod activation;
/// The paper version of the fixed CUDA admission ledger, shared by preflight
/// estimates and the runtime admission so the two cannot drift apart.
pub(crate) mod admission_sketch;
#[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
pub(crate) mod card_short;
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
pub mod cuda;
pub(crate) mod factory;
#[cfg(test)]
mod fixtures;
mod host;
pub(in crate::family) mod host_cpu;
#[cfg(test)]
mod host_defaults_tests;
mod host_load;
pub(crate) mod host_plan;
#[cfg(test)]
mod host_real;
#[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
mod pair_router_report;
#[cfg(test)]
mod recipe_map;
#[cfg(any(
    test,
    all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64")
))]
pub(in crate::family) mod resident;
pub(crate) mod route_capture;
pub(crate) mod route_stage;

/// The opt-in of the section timer's attention-core pair (`Q36_SECTION_CORE_MARKS=1`):
/// those event nodes sit between the block's dependent kernels and cost real replay time
/// on small GPUs (measured +30ms per token on the Xavier), so the default rig records only
/// the boundary events and the core/proj split is an explicit attribution run.
pub(crate) fn section_core_marks() -> bool {
    std::env::var("Q36_SECTION_CORE_MARKS")
        .map(|value| value == "1")
        .unwrap_or(false)
}
pub(crate) mod settings;
#[cfg(test)]
mod sparse_reference;
#[cfg(test)]
mod sparse_reference_tests;
#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
mod start;
pub mod tiered;
#[cfg(test)]
mod tiny;
mod trace;
