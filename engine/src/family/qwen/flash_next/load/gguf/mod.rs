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

pub(in crate::family::qwen::flash_next) mod admission;
pub(super) mod config;
#[cfg(all(test, feature = "native-cuda", target_os = "linux"))]
mod cuda_probe;
mod delta_columns;
mod delta_rows;
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
mod delta_state;
#[cfg(all(test, feature = "native-cuda", target_os = "linux"))]
mod device_tests;
mod expert_layout;
pub(in crate::family::qwen::flash_next) mod feeds;
#[cfg(test)]
mod host;
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
mod hyper_weights;
#[cfg(all(test, feature = "native-cuda", target_os = "linux"))]
pub(in super::super::super) mod native_f16;
pub(super) mod ngram;
pub(in crate::family::qwen::flash_next) mod placement;
pub(super) mod plan;
mod ple_weights;
pub(super) mod projector;
#[cfg(all(test, feature = "native-cuda", target_os = "linux"))]
mod route_classifier_probe;
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
pub(in super::super) mod routed;
#[cfg(all(test, feature = "native-cuda", target_os = "linux"))]
pub(in super::super) mod routed_probe;
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
mod router_weights;
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
mod shared_weights;
#[cfg(test)]
pub(in crate::family::qwen::flash_next) mod smallmodel;
pub(in super::super) mod staged_budget;
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
pub(in super::super) mod staged_trunk;
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
pub(in super::super) mod start;
pub(in crate::family::qwen::flash_next) mod tensors;
#[cfg(test)]
mod testkit;
#[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
pub(in super::super) mod token_feed;
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
pub(in crate::family::qwen::flash_next) mod trunk;
mod weight_map;

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

/// Uploads a checked rank-two head using bounded reads of the already-open shard.
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
fn upload_native_head(
    weights: &GgufWeights,
    head: &tensors::Mapped,
    name: String,
    ctx: &crate::kernels::cuda::CudaContext,
) -> Result<crate::family::project::cuda::CudaWeight, EngError> {
    use crate::kernels::cuda::{CHUNK_LIMIT, QuantTensor};
    let bad = || {
        EngError::Checkpoint(format!(
            "Flash-Next GGUF: invalid native head {}",
            head.name
        ))
    };
    let [k, rows] = head.dims.as_slice() else {
        return Err(bad());
    };
    let tensor = QuantTensor {
        name: &head.name,
        ty: head.ty,
        k: usize::try_from(*k).map_err(|_| bad())?,
        n_rows: usize::try_from(*rows).map_err(|_| bad())?,
        source_len: usize::try_from(head.bytes).map_err(|_| bad())?,
    };
    let weight = ctx.upload_native(tensor, CHUNK_LIMIT, &mut |offset, buf| {
        weights.read_range(&head.name, offset as u64, buf)
    })?;
    Ok(crate::family::project::cuda::CudaWeight::Native { name, weight })
}

/// Concatenate validated half-precision query then key rows with bounded host staging.
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
fn upload_indexer_pair(
    weights: &GgufWeights,
    pair: [&tensors::Mapped; 2],
    layer: usize,
    ctx: &crate::kernels::cuda::CudaContext,
    stream: &crate::kernels::cuda::DeviceStream,
) -> Result<crate::family::project::cuda::CudaWeight, EngError> {
    use crate::{family::exl3_pack::layer_name, kernels::cuda::CHUNK_LIMIT, weights::GgufType};
    let bad = || {
        EngError::Checkpoint(format!(
            "Flash-Next GGUF: invalid indexer pair at layer {layer}"
        ))
    };
    if !matches!(pair[0].ty, GgufType::F16 | GgufType::Bf16) || pair[0].ty != pair[1].ty {
        return Err(bad());
    }
    let mut sizes = [0usize; 2];
    let mut cols = None;
    for (i, entry) in pair.iter().enumerate() {
        let [k, n] = entry.dims.as_slice() else {
            return Err(bad());
        };
        if *k == 0 || *n == 0 || cols.is_some_and(|v| v != *k) {
            return Err(bad());
        }
        cols = Some(*k);
        let expected = k
            .checked_mul(*n)
            .and_then(|v| v.checked_mul(2))
            .ok_or_else(bad)?;
        if expected != entry.bytes {
            return Err(bad());
        }
        sizes[i] = usize::try_from(entry.bytes).map_err(|_| bad())?;
    }
    let total = sizes[0].checked_add(sizes[1]).ok_or_else(bad)?;
    let chunk = total.min(CHUNK_LIMIT);
    let mut staging = Vec::new();
    staging
        .try_reserve_exact(chunk)
        .map_err(|_| EngError::MemoryPlan("GGUF indexer staging allocation failed".into()))?;
    staging.resize(chunk, 0);
    let buffer = ctx.allocate(total)?;
    let mut dst = 0;
    for (entry, size) in pair.into_iter().zip(sizes) {
        let mut offset = 0;
        while offset < size {
            let take = (size - offset).min(chunk);
            weights.read_range(&entry.name, offset as u64, &mut staging[..take])?;
            buffer.write_at(stream, dst, &staging[..take])?;
            offset += take;
            dst += take;
        }
    }
    Ok(crate::family::project::cuda::CudaWeight::Raw {
        name: layer_name(layer, "self_attn.indexer.index_qk_proj"),
        ty: pair[0].ty,
        buffer,
    })
}

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

    /// Plan logical-to-stored rows for one tiled DeltaNet value-head projection.
    /// The plan does not upload or reorder the projection yet.
    pub(in super::super) fn delta_projection_rows(
        &self,
        layer: usize,
        role: crate::family::recipe::Role,
    ) -> Result<Vec<usize>, EngError> {
        use crate::family::recipe::Role;
        let dims = &self.config.dims;
        let bad = || {
            EngError::Checkpoint(format!(
                "Flash-Next GGUF: invalid DeltaNet row plan for {role:?} at layer {layer}"
            ))
        };
        let key_heads = usize::try_from(dims.key_heads).map_err(|_| bad())?;
        let value_heads = usize::try_from(dims.value_heads).map_err(|_| bad())?;
        let value_dim = usize::try_from(dims.value_dim).map_err(|_| bad())?;
        let key_dim = usize::try_from(dims.key_dim).map_err(|_| bad())?;
        let (skip, unit) = match role {
            Role::DeltaQkv => (
                key_heads
                    .checked_mul(key_dim)
                    .and_then(|n| n.checked_mul(2))
                    .ok_or_else(bad)?,
                value_dim,
            ),
            Role::DeltaGate => (0, value_dim),
            Role::DeltaBeta | Role::DeltaAlpha => (0, 1),
            _ => return Err(bad()),
        };
        let entry = self.map.get(role, Some(layer)).ok_or_else(bad)?;
        let [_, rows] = entry.dims.as_slice() else {
            return Err(bad());
        };
        delta_rows::delta_rows(
            skip,
            unit,
            key_heads,
            value_heads,
            usize::try_from(*rows).map_err(|_| bad())?,
        )
        .map_err(|_| bad())
    }

    /// Refuse QSA and out-of-range layers before staging any DeltaNet buffers.
    fn validate_delta_layer(&self, layer: usize) -> Result<(), EngError> {
        match self.config.kinds().get(layer) {
            Some(false) => Ok(()),
            Some(true) => Err(EngError::Checkpoint(format!(
                "Flash-Next GGUF: layer {layer} is QSA, not DeltaNet"
            ))),
            None => Err(EngError::Checkpoint(format!(
                "Flash-Next GGUF: layer {layer} is outside the model"
            ))),
        }
    }

    /// Stage one complete DeltaNet mixer in the executor's native/F32 layouts.
    /// The existing role-specific loaders validate shapes and layout and own partial
    /// uploads on error. The caller owns the returned buffers and stream ordering.
    /// This helper does not enable GGUF execution or account for admission scratch.
    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    pub(in super::super) fn upload_delta_trunk(
        &self,
        layer: usize,
        ctx: &crate::kernels::cuda::CudaContext,
        stream: &crate::kernels::cuda::DeviceStream,
    ) -> Result<Vec<crate::family::project::cuda::CudaWeight>, EngError> {
        self.validate_delta_layer(layer)?;
        ctx.check_launch_stream(stream)?;
        let mut weights = self.upload_delta_qkv_gate(layer, ctx)?;
        weights.extend(self.upload_delta_beta_alpha(layer, ctx, stream)?);
        weights.push(self.upload_delta_out(layer, ctx)?);
        weights.extend(self.upload_delta_state(layer, ctx, stream)?);
        Ok(weights)
    }

    /// Stage DeltaNet QKV and gate weights in logical value-head row order.
    /// This does not upload the other DeltaNet weights or enable GGUF execution.
    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    pub(in super::super) fn upload_delta_qkv_gate(
        &self,
        layer: usize,
        ctx: &crate::kernels::cuda::CudaContext,
    ) -> Result<Vec<crate::family::project::cuda::CudaWeight>, EngError> {
        use crate::{
            family::{exl3_pack::layer_name, project::cuda::CudaWeight, recipe::Role},
            kernels::cuda::{CHUNK_LIMIT, QuantTensor},
        };
        [
            (Role::DeltaQkv, "linear_attn.in_proj_qkv"),
            (Role::DeltaGate, "linear_attn.in_proj_z"),
        ]
        .into_iter()
        .map(|(role, suffix)| {
            let bad = || {
                EngError::Checkpoint(format!(
                    "Flash-Next GGUF: invalid DeltaNet {role:?} at layer {layer}"
                ))
            };
            let entry = self.map.get(role, Some(layer)).ok_or_else(bad)?;
            let [k, rows] = entry.dims.as_slice() else {
                return Err(bad());
            };
            let k = usize::try_from(*k).map_err(|_| bad())?;
            let rows = usize::try_from(*rows).map_err(|_| bad())?;
            let source_len = usize::try_from(entry.bytes).map_err(|_| bad())?;
            if rows == 0 || !source_len.is_multiple_of(rows) {
                return Err(bad());
            }
            let plan = delta_rows::DeltaRowReader::new(
                self.delta_projection_rows(layer, role)?,
                source_len / rows,
                source_len,
            )
            .map_err(|_| bad())?;
            let tensor = QuantTensor {
                name: &entry.name,
                ty: entry.ty,
                k,
                n_rows: rows,
                source_len,
            };
            let weight = ctx.upload_native(tensor, CHUNK_LIMIT, &mut |offset, out| {
                plan.read(offset, out, |src, dst| {
                    self.weights.read_range(&entry.name, src as u64, dst)
                })
            })?;
            Ok(CudaWeight::Native {
                name: layer_name(layer, suffix),
                weight,
            })
        })
        .collect()
    }

    /// Stage DeltaNet beta and alpha matrices as F32 rows in logical value-head order.
    /// QKV, gate, output, state and other weights remain separate responsibilities.
    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    pub(in super::super) fn upload_delta_beta_alpha(
        &self,
        layer: usize,
        ctx: &crate::kernels::cuda::CudaContext,
        stream: &crate::kernels::cuda::DeviceStream,
    ) -> Result<Vec<crate::family::project::cuda::CudaWeight>, EngError> {
        use crate::{
            family::{exl3_pack::layer_name, project::cuda::CudaWeight, recipe::Role},
            weights::{GgufType, dequantize},
        };
        [
            (Role::DeltaBeta, "linear_attn.in_proj_b.weight"),
            (Role::DeltaAlpha, "linear_attn.in_proj_a.weight"),
        ]
        .into_iter()
        .map(|(role, suffix)| {
            let bad = || {
                EngError::Checkpoint(format!(
                    "Flash-Next GGUF: invalid DeltaNet {role:?} at layer {layer}"
                ))
            };
            let entry = self.map.get(role, Some(layer)).ok_or_else(bad)?;
            let [k, rows] = entry.dims.as_slice() else {
                return Err(bad());
            };
            let k = usize::try_from(*k).map_err(|_| bad())?;
            let rows = usize::try_from(*rows).map_err(|_| bad())?;
            let block = usize::try_from(entry.ty.block_elems()).map_err(|_| bad())?;
            let block_bytes = usize::try_from(entry.ty.block_bytes()).map_err(|_| bad())?;
            if k == 0 || rows == 0 || block == 0 || !k.is_multiple_of(block) {
                return Err(bad());
            }
            let row_bytes = (k / block).checked_mul(block_bytes).ok_or_else(bad)?;
            let source_len = usize::try_from(entry.bytes).map_err(|_| bad())?;
            let plan = delta_rows::DeltaRowReader::new(
                self.delta_projection_rows(layer, role)?,
                row_bytes,
                source_len,
            )
            .map_err(|_| bad())?;
            let row_out = k.checked_mul(4).ok_or_else(bad)?;
            let output_bytes = rows.checked_mul(row_out).ok_or_else(bad)?;
            let mut raw = Vec::new();
            raw.try_reserve_exact(row_bytes).map_err(|_| {
                EngError::MemoryPlan("GGUF DeltaNet scalar staging allocation failed".into())
            })?;
            raw.resize(row_bytes, 0);
            let buffer = ctx.allocate(output_bytes)?;
            for row in 0..rows {
                plan.read(row * row_bytes, &mut raw, |src, dst| {
                    self.weights.read_range(&entry.name, src as u64, dst)
                })?;
                let values = dequantize(entry.ty, &raw)?;
                if values.len() != k {
                    return Err(bad());
                }
                let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();
                buffer.write_at(stream, row * row_out, &bytes)?;
            }
            Ok(CudaWeight::Raw {
                name: layer_name(layer, suffix),
                ty: GgufType::F32,
                buffer,
            })
        })
        .collect()
    }

    /// Stage DeltaNet output in grouped columns when heads align to encoded blocks.
    /// When a block crosses heads, retain the stored tiled order: the GGUF decode mixer
    /// tiles its activation before projection rather than changing quantization scales.
    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    pub(in super::super) fn upload_delta_out(
        &self,
        layer: usize,
        ctx: &crate::kernels::cuda::CudaContext,
    ) -> Result<crate::family::project::cuda::CudaWeight, EngError> {
        use crate::{
            family::{exl3_pack::layer_name, project::cuda::CudaWeight, recipe::Role},
            kernels::cuda::{CHUNK_LIMIT, QuantTensor},
        };
        let bad = || {
            EngError::Checkpoint(format!(
                "Flash-Next GGUF: invalid DeltaOut at layer {layer}"
            ))
        };
        let entry = self.map.get(Role::DeltaOut, Some(layer)).ok_or_else(bad)?;
        let [k, rows] = entry.dims.as_slice() else {
            return Err(bad());
        };
        let k = usize::try_from(*k).map_err(|_| bad())?;
        let rows = usize::try_from(*rows).map_err(|_| bad())?;
        let dims = &self.config.dims;
        let heads = usize::try_from(dims.value_heads).map_err(|_| bad())?;
        let head_dim = usize::try_from(dims.value_dim).map_err(|_| bad())?;
        if heads.checked_mul(head_dim) != Some(k) || head_dim == 0 {
            return Err(bad());
        }
        let block_elements = usize::try_from(entry.ty.block_elems()).map_err(|_| bad())?;
        if !head_dim.is_multiple_of(block_elements) {
            return upload_native_head(
                &self.weights,
                entry,
                layer_name(layer, "linear_attn.out_proj"),
                ctx,
            );
        }
        let source_len = usize::try_from(entry.bytes).map_err(|_| bad())?;
        let plan = delta_columns::DeltaColumnReader::new(
            usize::try_from(dims.key_heads).map_err(|_| bad())?,
            heads,
            head_dim,
            usize::try_from(entry.ty.block_elems()).map_err(|_| bad())?,
            usize::try_from(entry.ty.block_bytes()).map_err(|_| bad())?,
            rows,
            source_len,
        )
        .map_err(|_| bad())?;
        let tensor = QuantTensor {
            name: &entry.name,
            ty: entry.ty,
            k,
            n_rows: rows,
            source_len,
        };
        let weight = ctx.upload_native(tensor, CHUNK_LIMIT, &mut |offset, out| {
            plan.read(offset, out, |src, dst| {
                self.weights.read_range(&entry.name, src as u64, dst)
            })
        })?;
        Ok(CudaWeight::Native {
            name: layer_name(layer, "linear_attn.out_proj"),
            weight,
        })
    }

    /// Upload the mapped output head in native GGUF layout under the executor's key.
    /// This is one staged trunk component; it does not enable GGUF execution.
    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    pub(in super::super) fn upload_native_head(
        &self,
        ctx: &crate::kernels::cuda::CudaContext,
    ) -> Result<crate::family::project::cuda::CudaWeight, EngError> {
        let head = self
            .map
            .get(Global::Head, None)
            .ok_or_else(|| EngError::Checkpoint("Flash-Next GGUF: no output head".into()))?;
        upload_native_head(&self.weights, head, "lm_head".into(), ctx)
    }

    /// Stage the final hyper mixer and native output head together. State, scratch and
    /// admission are separate; this does not enable GGUF execution.
    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    pub(in super::super) fn upload_final_trunk(
        &self,
        ctx: &crate::kernels::cuda::CudaContext,
        stream: &crate::kernels::cuda::DeviceStream,
    ) -> Result<Vec<crate::family::project::cuda::CudaWeight>, EngError> {
        ctx.check_launch_stream(stream)?;
        let mut weights =
            self.upload_hyper(crate::family::recipe::Site::Head, None, ctx, stream)?;
        weights.push(self.upload_native_head(ctx)?);
        Ok(weights)
    }

    /// Stage the four native QSA projections without enabling GGUF execution.
    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    pub(in super::super) fn upload_qsa_projections(
        &self,
        layer: usize,
        ctx: &crate::kernels::cuda::CudaContext,
    ) -> Result<Vec<crate::family::project::cuda::CudaWeight>, EngError> {
        use crate::family::{exl3_pack::layer_name, recipe::Role};
        let projections = [
            (Role::QProj, "self_attn.q_proj"),
            (Role::KProj, "self_attn.k_proj"),
            (Role::VProj, "self_attn.v_proj"),
            (Role::OProj, "self_attn.o_proj"),
        ];
        projections
            .into_iter()
            .map(|(role, name)| {
                let tensor = self.map.get(role, Some(layer)).ok_or_else(|| {
                    EngError::Checkpoint(format!(
                        "Flash-Next GGUF: missing {name} at layer {layer}"
                    ))
                })?;
                upload_native_head(&self.weights, tensor, layer_name(layer, name), ctx)
            })
            .collect()
    }

    /// Upload the four QSA norm vectors in the conventions of their CUDA consumers.
    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    pub(in super::super) fn upload_qsa_norms(
        &self,
        layer: usize,
        ctx: &crate::kernels::cuda::CudaContext,
        stream: &crate::kernels::cuda::DeviceStream,
    ) -> Result<Vec<crate::family::project::cuda::CudaWeight>, EngError> {
        use crate::{
            family::{exl3_pack::layer_name, project::cuda::CudaWeight, recipe::Role},
            weights::{GgufType, dequantize},
        };
        use tensors::Convention;
        let roles = [
            (Role::QNorm, "self_attn.q_norm.weight", false),
            (Role::KNorm, "self_attn.k_norm.weight", false),
            (
                Role::IndexQNorm,
                "self_attn.indexer.q_layernorm.weight",
                true,
            ),
            (
                Role::IndexKNorm,
                "self_attn.indexer.k_layernorm.weight",
                true,
            ),
        ];
        let mut out = Vec::with_capacity(roles.len());
        for (role, suffix, rms0) in roles {
            let entry = self.map.get(role, Some(layer)).ok_or_else(|| {
                EngError::Checkpoint(format!(
                    "Flash-Next GGUF: missing {suffix} at layer {layer}"
                ))
            })?;
            if entry.convention != Convention::GammaOnePlusW || entry.dims.len() != 1 {
                return Err(EngError::Checkpoint(format!(
                    "Flash-Next GGUF: invalid norm {}",
                    entry.name
                )));
            }
            let (ty, bytes) = self.weights.read_tensor(&entry.name)?;
            let mut values = dequantize(ty, &bytes)?;
            if values.len() as u64 != entry.dims[0] {
                return Err(EngError::Checkpoint(format!(
                    "Flash-Next GGUF: invalid norm size {}",
                    entry.name
                )));
            }
            // Attention preparation multiplies by gamma; the indexer computes 1+w.
            if rms0 {
                for v in &mut values {
                    *v -= 1.0;
                }
            }
            let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();
            let buffer = ctx.allocate(bytes.len())?;
            buffer.write_at(stream, 0, &bytes)?;
            out.push(CudaWeight::Raw {
                name: layer_name(layer, suffix),
                ty: GgufType::F32,
                buffer,
            });
        }
        Ok(out)
    }

    /// Upload one GGUF QSA indexer's query and key projections as query-then-key rows.
    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    pub(in super::super) fn upload_indexer(
        &self,
        layer: usize,
        ctx: &crate::kernels::cuda::CudaContext,
        stream: &crate::kernels::cuda::DeviceStream,
    ) -> Result<crate::family::project::cuda::CudaWeight, EngError> {
        use crate::family::recipe::Role;
        let get = |role| {
            self.map.get(role, Some(layer)).ok_or_else(|| {
                EngError::Checkpoint(format!(
                    "Flash-Next GGUF: missing indexer projection at layer {layer}"
                ))
            })
        };
        upload_indexer_pair(
            &self.weights,
            [get(Role::IndexProj)?, get(Role::IndexKey)?],
            layer,
            ctx,
            stream,
        )
    }

    /// Upload one layer's resident hyper, mixer, router and shared-expert weights.
    /// Routed expert banks, PLE, token feeds, persistent state, and admission scratch have
    /// separate owners. This staged bundle does not enable the GGUF executor.
    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    pub(in super::super) fn upload_layer_trunk(
        &self,
        layer: usize,
        ctx: &crate::kernels::cuda::CudaContext,
        stream: &crate::kernels::cuda::DeviceStream,
    ) -> Result<Vec<crate::family::project::cuda::CudaWeight>, EngError> {
        use crate::family::recipe::Site;
        let is_qsa = *self.config.kinds().get(layer).ok_or_else(|| {
            EngError::Checkpoint(format!(
                "Flash-Next GGUF: layer {layer} is outside the model"
            ))
        })?;
        ctx.check_launch_stream(stream)?;
        let mut weights = self.upload_hyper(Site::Mixer, Some(layer), ctx, stream)?;
        if is_qsa {
            weights.extend(self.upload_qsa_trunk(layer, ctx, stream)?);
        } else {
            weights.extend(self.upload_delta_trunk(layer, ctx, stream)?);
        }
        weights.extend(self.upload_hyper(Site::Ffn, Some(layer), ctx, stream)?);
        weights.extend(self.upload_shared_expert(layer, ctx)?);
        weights.extend(self.upload_router(layer, ctx, stream)?);
        Ok(weights)
    }

    /// Stage the QSA projections, indexer and norms for one GGUF attention layer.
    /// Hyper connections, attention state, and MoE weights have separate owners.
    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    pub(in super::super) fn upload_qsa_trunk(
        &self,
        layer: usize,
        ctx: &crate::kernels::cuda::CudaContext,
        stream: &crate::kernels::cuda::DeviceStream,
    ) -> Result<Vec<crate::family::project::cuda::CudaWeight>, EngError> {
        let mut weights = self.upload_qsa_projections(layer, ctx)?;
        weights.push(self.upload_indexer(layer, ctx, stream)?);
        weights.extend(self.upload_qsa_norms(layer, ctx, stream)?);
        Ok(weights)
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
    pub(in crate::family::qwen::flash_next) fn expert_index(
        &self,
    ) -> Result<GgufExpertIndex, EngError> {
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
    pub(in crate::family::qwen::flash_next) fn record_store(
        &self,
    ) -> Result<GgufRecordStore, EngError> {
        GgufRecordStore::new(std::sync::Arc::clone(&self.weights), self.expert_index()?)
    }

    /// Read checked stored rows for one expert into a caller-sized buffer. This does not
    /// convert formats or upload to the CUDA expert bank.
    pub(in super::super) fn read_expert_rows(
        &self,
        layer: usize,
        role: crate::family::recipe::Role,
        expert: u64,
        rows: std::ops::Range<u64>,
        out: &mut [u8],
    ) -> Result<(), EngError> {
        use crate::family::recipe::Role;
        if !matches!(role, Role::ExpertGate | Role::ExpertUp | Role::ExpertDown) {
            return Err(EngError::BadRequest("not a routed expert role".into()));
        }
        let entry = self.map.get(role, Some(layer)).ok_or_else(|| {
            EngError::BadRequest(format!("missing expert role {role:?} at layer {layer}"))
        })?;
        let (offset, bytes) = entry.row_span(Some(expert), rows)?;
        if out.len() != bytes {
            return Err(EngError::BadRequest(format!(
                "expert read needs {bytes} bytes, got {}",
                out.len()
            )));
        }
        self.weights.read_range(&entry.name, offset, out)
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
pub(in super::super) mod tests;
