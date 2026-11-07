//! The Qwen3.8 Flash-Next CUDA step, built slice by slice (`docs/plans/flash-next-cuda.md`). The
//! expert bank (S5, host half), the state and step plans (S3) and the trunk loader with the host
//! feeds (S1) exist so far; the other slices add their modules beside them. The trunk loader is
//! `load`; the host feeds are `embedding` (the token's embedding row), `ple_feed` (the n-gram rows
//! and the PLE step) and, with a device build, `stream_io` (the residual stream between the host
//! and the card). Nothing here is called by a model step, and a CUDA build of Flash-Next still
//! refuses to serve.

mod bank;
// The pack reader exists where the pack files do (`load::pack_files`): tests and native-cuda Linux.
#[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
mod bank_source;
#[cfg(test)]
mod bank_source_tests;
#[cfg(test)]
mod bank_tests;
mod moe_block;
#[cfg(all(test, feature = "native-cuda", target_os = "linux"))]
mod moe_block_device_tests;
#[cfg(test)]
mod moe_block_tests;
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
// The trunk list needs the pack walk (`load::exl3_pack`), which exists in tests and on native-cuda
// Linux.
#[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
mod embedding;
#[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
pub(in crate::family::flash_next) mod load;
#[cfg(test)]
mod load_tests;
#[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
mod ple_feed;
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
mod stream_io;

#[cfg(test)]
mod testkit {
    //! An in-memory safetensors checkpoint for the host tests of the feeds.

    use crate::weights::{ShardIo, StreamedCheckpoint};

    /// One shard holding `(name, dtype, shape, bytes)` tensors.
    pub(super) fn checkpoint(tensors: &[(&str, &str, Vec<u32>, Vec<u8>)]) -> StreamedCheckpoint {
        let mut header = serde_json::Map::new();
        let mut payload = Vec::new();
        let mut index = serde_json::Map::new();
        for (name, dtype, shape, bytes) in tensors {
            let start = payload.len();
            payload.extend_from_slice(bytes);
            header.insert(
                name.to_string(),
                serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [start, payload.len()]}),
            );
            index.insert(name.to_string(), serde_json::json!("m.safetensors"));
        }
        let header = serde_json::to_vec(&header).unwrap();
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(&header);
        file.extend_from_slice(&payload);
        let index = serde_json::to_vec(&serde_json::json!({"weight_map": index})).unwrap();
        let len = file.len() as u64;
        let reader: Box<dyn ShardIo + Send> = Box::new(std::io::Cursor::new(file));
        StreamedCheckpoint::open(
            &index,
            vec![("m.safetensors".to_string(), reader, len)],
            1 << 20,
        )
        .unwrap()
    }
}

// The decode step. It needs the state and step plans, so it builds with the native CUDA backend
// only. Staged: the executor is the first caller.
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
mod mixer;
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
mod step;
#[cfg(all(test, feature = "native-cuda", target_os = "linux"))]
mod step_tests;
