#[cfg(feature = "research-activation")]
pub mod activation;
/// The paper version of the fixed CUDA admission ledger, shared by preflight
/// estimates and the runtime admission so the two cannot drift apart.
pub(crate) mod admission_sketch;
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
#[expect(
    dead_code,
    reason = "staged for the resident step's mixers, MoE block and token loop"
)]
pub mod cuda;
#[cfg(test)]
mod exl3_admission;
#[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
#[cfg_attr(
    not(all(feature = "native-cuda", target_os = "linux")),
    allow(
        dead_code,
        reason = "the CUDA step is the caller; host builds test the plan"
    )
)]
pub(in crate::family) mod exl3_merge_plan;
#[cfg(test)]
mod exl3_merge_plan_tests;
#[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
mod exl3_pack;
#[cfg(test)]
mod exl3_pack_tests;
#[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
mod exl3_tensors;
#[cfg(test)]
mod exl3_tokenizer_tests;
pub(crate) mod factory;
#[cfg(test)]
mod fixtures;
mod host;
pub(in crate::family) mod host_cpu;
mod host_load;
#[cfg(test)]
mod host_real;
#[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
#[cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "the K-row step, built with the native CUDA backend only, is the caller of most of it"
    )
)]
pub(crate) mod krow;
#[cfg(test)]
mod krow_step_pin_tests;
#[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
mod mtp_pack;
#[cfg(test)]
mod mtp_pack_tests;
#[cfg(any(
    test,
    all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64")
))]
mod resident;
#[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "first callers: the device rollback slots and the window"
    )
)]
pub(crate) mod rollback;
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
#[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
pub(crate) mod window_logic;
