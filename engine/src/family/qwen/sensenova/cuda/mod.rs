//! The CUDA path of SenseNova-Loop: host geometry over the shared device-planning
//! shape, the GGUF weight adapter, and the trunk executor (see the lane's CUDA-PLAN).
//!
//! The shared qwen36 CUDA serve path is wired to one model spec; this module rides the
//! shared kernels and planning pieces the way Flash-Next does, owning its step loop.

pub(crate) mod geometry;
pub(crate) mod load;
