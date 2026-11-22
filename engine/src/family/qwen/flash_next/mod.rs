//! Host-orchestrated forward pass for `qwen3_8_flash_next`.
//!
//! Activations stay in host `f32` vectors. Quantized projections go through the
//! [`crate::family::project::Projector`] seam.

mod artifact;
mod cuda;
mod deltanet;
pub(crate) mod factory;
mod host_plan;
mod hyper;
mod load;

/// The preflight rows entry of this build: the family's own function where
/// the CUDA load path is compiled in, `None` everywhere else — those builds
/// keep the shallow report, because a family that
/// exists on one backend must not cost the capability a build split.
pub(crate) fn preflight_rows_entry() -> Option<crate::family::plan_rows::RowsSource> {
    #[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
    {
        Some(load::preflight_rows)
    }
    #[cfg(not(any(test, all(feature = "native-cuda", target_os = "linux"))))]
    {
        None
    }
}
#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
mod mlx;
mod moe;
mod ple;
mod prefill_plan;
mod qsa;
#[cfg(any(
    test,
    all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64")
))]
mod resident;
pub(crate) mod settings;
#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
mod start;
mod step;
