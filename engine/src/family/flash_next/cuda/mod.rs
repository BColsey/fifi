//! The Qwen3.8 Flash-Next CUDA step, built slice by slice (`docs/plans/flash-next-cuda.md`). The
//! expert bank (S5, host half) and the state and step plans (S3) exist so far; the other slices add
//! their modules beside them.

mod bank;
// The pack reader exists where the pack files do (`load::pack_files`): tests and native-cuda Linux.
#[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
mod bank_source;
#[cfg(test)]
mod bank_source_tests;
#[cfg(test)]
mod bank_tests;
// S3: the state and decode arena plans size device memory with the kernel wrappers, so they build
// with the native CUDA backend only. Staged: the decode step (S4) is the first caller.
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
#[cfg_attr(
    not(test),
    allow(dead_code, reason = "staged for the decode step (S4)")
)]
mod state;
#[cfg(all(test, feature = "native-cuda", target_os = "linux"))]
mod state_device_tests;
#[cfg(all(test, feature = "native-cuda", target_os = "linux"))]
mod state_tests;
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
#[cfg_attr(
    not(test),
    allow(dead_code, reason = "staged for the decode step (S4)")
)]
mod step_plan;
#[cfg(all(test, feature = "native-cuda", target_os = "linux"))]
mod step_plan_hazard_tests;
#[cfg(all(test, feature = "native-cuda", target_os = "linux"))]
mod step_plan_tests;
