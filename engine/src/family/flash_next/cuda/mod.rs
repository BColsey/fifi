//! The Qwen3.8 Flash-Next CUDA step, built slice by slice (`docs/plans/flash-next-cuda.md`). Only the
//! expert bank (S5, host half) exists so far; the other slices add their modules beside it.

mod bank;
#[cfg(test)]
mod bank_tests;
