//! Flash-Next from a GGUF checkpoint (architecture `qwen4exp`): the file or split set is read
//! in place and never copied.
//!
//! - [`config`]: the model dimensions from the metadata, in the structure the EXL3 pack's
//!   `config.json` fills.
//! - [`tensors`]: every tensor of the file mapped to a field of the executor's weight
//!   structures, with the form the device holds it in and the types no CUDA kernel reads yet.
//! - [`ngram`]: the n-gram memory, rows read and decoded on demand.
//! - [`plan`]: the memory plan for the whole file on a card and a host, through the same
//!   tier planner as the EXL3 pack.
//!
//! [`GgufModel::open`] takes a `.gguf` file or a directory holding one. The routed experts
//! (about 50 GB at 3 bpw) and the n-gram table (28.8 GB) stay in the files; the plan places
//! experts in video memory, in pinned host memory, or reads them from the files on a miss,
//! whichever the host memory allows.

// Staged: the Flash-Next CUDA start is the first caller.
#![cfg_attr(not(test), allow(dead_code))]

pub(super) mod config;
#[cfg(test)]
mod host;
pub(super) mod ngram;
pub(super) mod plan;
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
pub(in crate::family::flash_next) mod start;
pub(super) mod tensors;
#[cfg(test)]
mod testkit;

use std::path::Path;

use self::{
    config::{ARCHITECTURE, GgufConfig},
    ngram::GgufNgramRows,
    tensors::{KernelGap, Place, Role, TensorMap, has_decode_kernel},
};
use crate::{
    EngError,
    family::gguf_select,
    weights::{GgufLimits, GgufWeights, MetaValue},
};

/// Bounds of the open: the metadata of a model with a 250,000-token vocabulary is about 10 MB;
/// the largest tensor is the 30 GB n-gram table.
const LIMITS: GgufLimits = GgufLimits {
    max_header_bytes: 256 << 20,
    max_tensor_bytes: 1 << 40,
};

/// An opened Flash-Next GGUF: the checkpoint, its configuration and its tensor map.
#[derive(Debug)]
pub(in crate::family::flash_next) struct GgufModel {
    pub(super) weights: GgufWeights,
    pub(super) config: GgufConfig,
    pub(super) map: TensorMap,
}

impl GgufModel {
    /// Opens the Flash-Next GGUF at `model` (a `.gguf` file, one shard of a split set, or a
    /// directory). `Ok(None)` when the path holds no GGUF of this architecture, so the EXL3
    /// loader can take it; a directory with several GGUF files opens the largest one of the
    /// architecture. A GGUF of this architecture that cannot be used is an error naming why.
    pub(in crate::family::flash_next) fn open(model: &Path) -> Result<Option<Self>, EngError> {
        let Ok((_, files)) = gguf_select::candidates(model) else {
            return Ok(None);
        };
        let mut first_error = None;
        for file in files {
            let weights = match GgufWeights::open_any_type(&file.path, LIMITS, None) {
                Ok(weights) => weights,
                // A file of another model in the directory is not this loader's to judge;
                // its error is reported only when no file of this architecture turns up.
                Err(error) => {
                    first_error.get_or_insert(error);
                    continue;
                }
            };
            if !matches!(
                weights.header().meta("general.architecture"),
                Some(MetaValue::String(arch)) if arch == ARCHITECTURE
            ) {
                continue;
            }
            let config = GgufConfig::read(&weights)?;
            let map = TensorMap::build(weights.header(), &config)?;
            return Ok(Some(Self {
                weights,
                config,
                map,
            }));
        }
        first_error.map_or(Ok(None), Err)
    }

    /// The path of shard 1 (the file itself outside a split set).
    pub(in crate::family::flash_next) fn first_path(&self) -> &Path {
        &self.weights.paths()[0]
    }

    pub(in crate::family::flash_next) fn config(&self) -> &GgufConfig {
        &self.config
    }

    pub(in crate::family::flash_next) fn map(&self) -> &TensorMap {
        &self.map
    }

    /// The n-gram memory of the file, rows decoded on demand.
    pub(in crate::family::flash_next) fn ngram(&self) -> Result<GgufNgramRows, EngError> {
        let table = self
            .map
            .get(Role::NgramTable, None)
            .ok_or_else(|| EngError::Checkpoint("Flash-Next GGUF: no n-gram table".into()))?;
        GgufNgramRows::open(&self.weights, table, self.config.hash.clone())
    }

    /// The tensor types of this file that no CUDA decode kernel reads yet.
    pub(in crate::family::flash_next) fn kernel_gaps(&self) -> Vec<KernelGap> {
        self.map.kernel_gaps(has_decode_kernel)
    }

    /// The refusal for a file whose matrices include types without a decode kernel: it names
    /// each type with its tensor count, bytes and first tensor, and the types that have one.
    pub(in crate::family::flash_next) fn kernel_refusal(&self, gaps: &[KernelGap]) -> EngError {
        let gib = |bytes: u64| bytes as f64 / (1u64 << 30) as f64;
        let listed: Vec<String> = gaps
            .iter()
            .map(|gap| {
                format!(
                    "{} ({} tensors, {:.2} GiB, first {})",
                    gap.ty.name(),
                    gap.tensors,
                    gib(gap.bytes),
                    gap.first
                )
            })
            .collect();
        let readable: Vec<&str> = tensors::DECODE_BLOCK_TYPES
            .iter()
            .map(|ty| ty.name())
            .collect();
        EngError::Checkpoint(format!(
            "Flash-Next GGUF {} cannot be served on CUDA yet: no decode kernel reads {}. The \
             CUDA kernels read {}.",
            self.first_path().display(),
            listed.join(", "),
            readable.join(", ")
        ))
    }

    /// Bytes of the routed experts in the file.
    pub(in crate::family::flash_next) fn expert_bytes(&self) -> u64 {
        self.map.file_bytes(Place::Experts)
    }

    /// One line for the load log: the files, the tensors mapped and the quantization types.
    pub(in crate::family::flash_next) fn summary(&self) -> String {
        let gib = |bytes: u64| bytes as f64 / (1u64 << 30) as f64;
        let mut types: Vec<(&'static str, u64)> = Vec::new();
        for entry in self.map.entries() {
            if entry.dims.len() < 2 {
                continue;
            }
            let name = entry.ty.name();
            match types.iter_mut().find(|(known, _)| *known == name) {
                Some((_, bytes)) => *bytes += entry.bytes,
                None => types.push((name, entry.bytes)),
            }
        }
        types.sort_by_key(|&(_, bytes)| std::cmp::Reverse(bytes));
        let types: Vec<String> = types
            .iter()
            .map(|(name, bytes)| format!("{name} {:.1} GiB", gib(*bytes)))
            .collect();
        let files = self.weights.paths();
        format!(
            "Flash-Next GGUF: {} ({} file{}), {} of {} tensors mapped, {} unmapped; matrices {}",
            self.first_path().display(),
            files.len(),
            if files.len() == 1 { "" } else { "s" },
            self.map.mapped(),
            self.map.total(),
            self.map.unmapped().len(),
            types.join(", ")
        )
    }
}

#[cfg(test)]
mod tests;
