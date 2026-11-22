//! Shared prefill planning and execution primitives. CUDA arena planners supply
//! their costs, spare minimum and chunk ladder to `admission`. MLX resident
//! executors supply a workspace cost to `mlx` and share its row limits and
//! DeltaNet retirement arithmetic. Each family owns its state and memory charges.

/// Host-side chunk admission and DeltaNet retirement planning for MLX.
pub(crate) mod mlx;

// Host (non-CUDA) builds have no production consumer yet: the promoting family's CUDA driver
// re-exports these under native-cuda, and the second consumer imports them ungated later.
#[allow(
    dead_code,
    reason = "the promoting family's CUDA driver consumes this under native-cuda; the second family's ungated planner lands in the shared-pass series"
)]
pub(crate) mod admission;

/// The rows of a prefill piece at a depth, from a latency target and the device.
#[allow(
    dead_code,
    reason = "the CUDA family drivers consume the piece rule; host builds use it only in tests"
)]
pub(crate) mod piece_rows;

// Both CUDA drivers consume these helpers. Host builds use them only in tests.
#[allow(
    dead_code,
    reason = "the CUDA family drivers consume these helpers; host builds use them only in tests"
)]
pub(crate) mod steps;

#[cfg(all(feature = "native-cuda", target_os = "linux"))]
pub(in crate::family) mod arena;

#[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
pub(in crate::family) mod chunk_family;

#[cfg(all(
    test,
    feature = "native-fault-qualification",
    target_os = "macos",
    target_arch = "aarch64"
))]
pub(crate) mod mlx_request_bench;
