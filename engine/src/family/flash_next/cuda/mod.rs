//! The Qwen3.8 Flash-Next CUDA step, built slice by slice (`docs/plans/flash-next-cuda.md`). Only the
//! expert bank (S5) exists so far; the other slices add their modules beside it.

mod bank;
// The pack reader exists where the pack files do (`load::pack_files`): tests and native-cuda Linux.
#[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
mod bank_source;
#[cfg(test)]
mod bank_source_tests;
#[cfg(test)]
mod bank_tests;
