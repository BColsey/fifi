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

#![cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "the n-gram reader and the per-tensor map entries have no caller outside the tests until the executor reads GGUF weights"
    )
)]

pub(super) mod config;
pub(in crate::family::qwen::flash_next) mod admission;
pub(in crate::family::qwen::flash_next) mod feeds;
#[cfg(all(test, feature = "native-cuda", target_os = "linux"))]
mod device_tests;
#[cfg(test)]
mod host;
pub(super) mod ngram;
pub(super) mod plan;
pub(in crate::family::qwen::flash_next) mod placement;
#[cfg(test)]
mod smallmodel;
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
pub(in super::super) mod start;
pub(in crate::family::qwen::flash_next) mod tensors;
#[cfg(test)]
mod testkit;
mod weight_map;
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
pub(in crate::family::qwen::flash_next) mod trunk;

use std::path::Path;

use self::{
    config::{ARCHITECTURE, GgufConfig},
    ngram::GgufNgramRows,
    tensors::{KernelGap, Place, TensorMap, has_decode_kernel},
};
use crate::{
    EngError,
    family::{gguf_select, recipe::Global},
    weights::{
        ExpertTensorNames, GgufExpertIndex, GgufLimits, GgufWeights, MetaValue,
        gguf_experts::GgufRecordStore,
    },
};

/// Bounds of the open: the metadata of a model with a 250,000-token vocabulary is about 10 MB;
/// the largest tensor is the 30 GB n-gram table.
const LIMITS: GgufLimits = GgufLimits {
    max_header_bytes: 256 << 20,
    max_tensor_bytes: 1 << 40,
};

/// An opened Flash-Next GGUF: the checkpoint, its configuration and its tensor map.
#[derive(Debug)]
pub(in super::super) struct GgufModel {
    pub(in crate::family::qwen::flash_next) weights: std::sync::Arc<GgufWeights>,
    pub(super) config: GgufConfig,
    pub(super) map: TensorMap,
}

impl GgufModel {
    /// Opens the Flash-Next GGUF at `model` (a `.gguf` file, one shard of a split set, or a
    /// directory). `Ok(None)` when the path holds no GGUF of this architecture, so the EXL3
    /// loader can take it; a directory with several GGUF files opens the largest one of the
    /// architecture. A GGUF file that cannot be opened is skipped with a log line naming it,
    /// unless its file name or the start of its header says `qwen4exp` (or it is a shard of a
    /// set whose first shard does): that one is an error naming why, because the model is
    /// there and unusable. A partial download of another model beside an EXL3 pack therefore
    /// leaves the pack's start alone.
    pub(in super::super) fn open(model: &Path) -> Result<Option<Self>, EngError> {
        let Ok((_, files)) = gguf_select::candidates(model) else {
            return Ok(None);
        };
        for file in files {
            let weights = match GgufWeights::open_any_type(&file.path, LIMITS, None) {
                Ok(weights) => weights,
                Err(error) if names_this_architecture(&file.path) => return Err(error),
                // A file of another model is not this loader's to judge.
                Err(error) => {
                    eprintln!(
                        "flash-next gguf: skipping {}, which does not open: {}",
                        file.path.display(),
                        error.describe()
                    );
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
                weights: std::sync::Arc::new(weights),
                config,
                map,
            }));
        }
        Ok(None)
    }

    /// The path of shard 1 (the file itself outside a split set).
    pub(in super::super) fn first_path(&self) -> &Path {
        &self.weights.paths()[0]
    }

    pub(in super::super) fn config(&self) -> &GgufConfig {
        &self.config
    }

    pub(in super::super) fn map(&self) -> &TensorMap {
        &self.map
    }

    /// The n-gram memory of the file, rows decoded on demand.
    pub(in super::super) fn ngram(&self) -> Result<GgufNgramRows, EngError> {
        let table = self
            .map
            .get(Global::NgramTable, None)
            .ok_or_else(|| EngError::Checkpoint("Flash-Next GGUF: no n-gram table".into()))?;
        GgufNgramRows::open(&self.weights, table, self.config.hash.clone())
    }

    /// The placement of every routed expert in the files: shard, offset, size and type per
    /// layer, built by the shared GGUF expert index.
    pub(in crate::family::qwen::flash_next) fn expert_index(&self) -> Result<GgufExpertIndex, EngError> {
        let dims = &self.config.dims;
        GgufExpertIndex::new(
            self.weights.header(),
            self.weights.shard_lens(),
            &ExpertTensorNames::standard(),
            self.config.layers(),
            dims.experts as usize,
            (dims.hidden as usize, dims.moe_inter as usize),
            None,
        )
        .map_err(|error| EngError::Checkpoint(format!("Flash-Next GGUF experts: {error:?}")))
    }

    /// The routed experts as an [`ExpertRecordStore`](crate::weights::expert_store): one
    /// positioned read per slice, records of each layer's own size.
    pub(in crate::family::qwen::flash_next) fn record_store(&self) -> Result<GgufRecordStore, EngError> {
        GgufRecordStore::new(std::sync::Arc::clone(&self.weights), self.expert_index()?)
    }

    /// The tensor types of this file that no CUDA decode kernel reads yet.
    pub(in super::super) fn kernel_gaps(&self) -> Vec<KernelGap> {
        self.map.kernel_gaps(has_decode_kernel)
    }

    /// The refusal for a file whose matrices include types without a decode kernel: it names
    /// each type with its tensor count, bytes and first tensor, and the types that have one.
    pub(in super::super) fn kernel_refusal(&self, gaps: &[KernelGap]) -> EngError {
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
        let names = |types: &[crate::weights::GgufType]| {
            types
                .iter()
                .map(|ty| ty.name())
                .collect::<Vec<_>>()
                .join(", ")
        };
        EngError::Checkpoint(format!(
            "Flash-Next GGUF {} cannot be served on CUDA yet: no decode kernel reads {}. The \
             dense and routed-expert kernels read {}.",
            self.first_path().display(),
            listed.join(", "),
            names(&crate::weights::GgufType::native_types())
        ))
    }

    /// Bytes of the routed experts in the file.
    pub(in super::super) fn expert_bytes(&self) -> u64 {
        self.map.file_bytes(Place::Experts)
    }

    /// One line for the load log: the files, the tensors mapped and the quantization types.
    pub(in super::super) fn summary(&self) -> String {
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

/// Whether the file name or the first bytes of the file at `path` name this architecture. The
/// architecture is the first metadata entry of every file the converters write. A shard
/// after the first has only split keys, so it answers through shard 1.
fn names_this_architecture(path: &Path) -> bool {
    use std::io::Read;
    let name = path
        .file_name()
        .map(|name| name.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default();
    if name.contains(ARCHITECTURE) || name.contains("flash-next") {
        return true;
    }
    if let Some(shard) = crate::weights::shard_name_of(path)
        && shard.number != 1
    {
        return names_this_architecture(&shard.sibling(path, 1));
    }
    let Ok(mut file) = std::fs::File::open(path) else {
        return false;
    };
    let mut head = vec![0u8; 4096];
    let Ok(read) = file.read(&mut head) else {
        return false;
    };
    head[..read]
        .windows(ARCHITECTURE.len())
        .any(|window| window == ARCHITECTURE.as_bytes())
}

#[cfg(test)]
mod tests;
