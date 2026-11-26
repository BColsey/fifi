//! Staged GGUF hyper-connection weights for the resident kernel. A vector (`norm`, `inject`)
//! and any F16, BF16 or F32 tensor is widened to F16. A Q8_0 matrix (`down`, `up`) is kept as
//! stored, 34 bytes a block of 32 values, which the kernels of dtype `Q80AndF16` decode value by
//! value; an F16 copy of the decoded matrix would round every weight a second time.
//! The expanded F16 allocations need their own admission charge before production use.

use super::{
    GgufModel,
    tensors::{Convention, Mapped, Slot},
};
use crate::{
    EngError,
    family::{
        exl3_pack::layer_name,
        project::cuda::CudaWeight,
        recipe::{Site, SiteTensor},
    },
    kernels::cuda::{CHUNK_LIMIT, CudaContext, DeviceStream},
    weights::{GgufType, dequantize},
};

#[cfg(test)]
mod tests;

/// How one hyper tensor reaches the device.
struct Staging<'a> {
    entry: &'a Mapped,
    blocks: usize,
    block_elements: usize,
    block_bytes: usize,
    output_bytes: usize,
    centered: bool,
    /// A Q8_0 matrix kept as stored (`GgufType::Q80` on the device); any other tensor is F16.
    stored: bool,
}

impl<'a> Staging<'a> {
    /// `matrix`: a `down` or `up` tensor, the two a Q8_0 storage keeps as stored.
    fn new(
        entry: &'a Mapped,
        dims: &[u64],
        centered: bool,
        matrix: bool,
    ) -> Result<Self, EngError> {
        let bad = || {
            EngError::Checkpoint(format!(
                "Flash-Next GGUF: invalid hyper tensor {}",
                entry.name
            ))
        };
        let expected_convention = if centered {
            Convention::GammaOnePlusW
        } else {
            Convention::AsStored
        };
        if entry.dims != dims
            || entry.convention != expected_convention
            || !matches!(
                entry.ty,
                GgufType::F32 | GgufType::F16 | GgufType::Bf16 | GgufType::Q80
            )
            || (centered && entry.ty == GgufType::Q80)
        {
            return Err(bad());
        }
        let elements = dims
            .iter()
            .try_fold(1usize, |n, &d| n.checked_mul(usize::try_from(d).ok()?))
            .filter(|&n| n > 0)
            .ok_or_else(bad)?;
        let block_elements = usize::try_from(entry.ty.block_elems()).map_err(|_| bad())?;
        let block_bytes = usize::try_from(entry.ty.block_bytes()).map_err(|_| bad())?;
        let first = dims
            .first()
            .and_then(|&v| usize::try_from(v).ok())
            .ok_or_else(bad)?;
        if block_elements == 0
            || block_bytes == 0
            || !first.is_multiple_of(block_elements)
            || !elements.is_multiple_of(block_elements)
        {
            return Err(bad());
        }
        let blocks = elements / block_elements;
        let source_bytes = blocks.checked_mul(block_bytes).ok_or_else(bad)?;
        if usize::try_from(entry.bytes).map_err(|_| bad())? != source_bytes {
            return Err(bad());
        }
        let stored = matrix && entry.ty == GgufType::Q80;
        let output_bytes = if stored {
            source_bytes
        } else {
            elements.checked_mul(size_of::<u16>()).ok_or_else(bad)?
        };
        Ok(Self {
            entry,
            blocks,
            block_elements,
            block_bytes,
            output_bytes,
            centered,
            stored,
        })
    }

    fn upload(
        &self,
        model: &GgufModel,
        name: String,
        ctx: &CudaContext,
        stream: &DeviceStream,
    ) -> Result<CudaWeight, EngError> {
        if self.stored {
            return self.upload_stored(model, name, ctx, stream);
        }
        let block_footprint = self
            .block_elements
            .checked_mul(size_of::<f32>())
            .map(|n| n.max(self.block_bytes))
            .ok_or_else(|| EngError::Checkpoint("GGUF hyper block overflows".into()))?;
        let chunk_blocks = (CHUNK_LIMIT / block_footprint).max(1).min(self.blocks);
        let mut raw = Vec::new();
        raw.try_reserve_exact(chunk_blocks * self.block_bytes)
            .map_err(|_| EngError::MemoryPlan("GGUF hyper staging allocation failed".into()))?;
        let buffer = ctx.allocate(self.output_bytes)?;
        for first in (0..self.blocks).step_by(chunk_blocks) {
            let count = chunk_blocks.min(self.blocks - first);
            raw.resize(count * self.block_bytes, 0);
            model.weights.read_range(
                &self.entry.name,
                (first * self.block_bytes) as u64,
                &mut raw,
            )?;
            let values = dequantize(self.entry.ty, &raw)?;
            if values.len() != count * self.block_elements {
                return Err(EngError::Checkpoint(format!(
                    "Flash-Next GGUF: invalid hyper data {}",
                    self.entry.name
                )));
            }
            let bytes = half_bytes(&values, self.centered)?;
            buffer.write_at(
                stream,
                first * self.block_elements * size_of::<u16>(),
                &bytes,
            )?;
        }
        Ok(CudaWeight::Raw {
            name,
            ty: GgufType::F16,
            buffer,
        })
    }

    /// The stored blocks, copied as they are.
    fn upload_stored(
        &self,
        model: &GgufModel,
        name: String,
        ctx: &CudaContext,
        stream: &DeviceStream,
    ) -> Result<CudaWeight, EngError> {
        let chunk_blocks = (CHUNK_LIMIT / self.block_bytes).max(1).min(self.blocks);
        let mut raw = Vec::new();
        raw.try_reserve_exact(chunk_blocks * self.block_bytes)
            .map_err(|_| EngError::MemoryPlan("GGUF hyper staging allocation failed".into()))?;
        let buffer = ctx.allocate(self.output_bytes)?;
        for first in (0..self.blocks).step_by(chunk_blocks) {
            let count = chunk_blocks.min(self.blocks - first);
            raw.resize(count * self.block_bytes, 0);
            model.weights.read_range(
                &self.entry.name,
                (first * self.block_bytes) as u64,
                &mut raw,
            )?;
            buffer.write_at(stream, first * self.block_bytes, &raw)?;
        }
        Ok(CudaWeight::Raw {
            name,
            ty: GgufType::Q80,
            buffer,
        })
    }
}

fn half_bytes(values: &[f32], centered: bool) -> Result<Vec<u8>, EngError> {
    let capacity = values
        .len()
        .checked_mul(size_of::<u16>())
        .ok_or_else(|| EngError::MemoryPlan("GGUF hyper conversion size overflows".into()))?;
    let mut out = Vec::new();
    out.try_reserve_exact(capacity)
        .map_err(|_| EngError::MemoryPlan("GGUF hyper conversion allocation failed".into()))?;
    for &value in values {
        let value = if centered { value - 1.0 } else { value };
        let half = half::f16::from_f32(value);
        if !value.is_finite() || !half.is_finite() {
            return Err(EngError::Checkpoint(
                "Flash-Next GGUF: hyper value cannot be represented in F16".into(),
            ));
        }
        out.extend_from_slice(&half.to_bits().to_ne_bytes());
    }
    Ok(out)
}

impl GgufModel {
    /// Stage one hyper site: the vectors as F16, the norm's stored one subtracted before the
    /// rounding, and a Q8_0 pair of matrices as stored. This expands every other quantized tensor
    /// and does not enable GGUF execution.
    pub(in super::super::super) fn upload_hyper(
        &self,
        site: Site,
        layer: Option<usize>,
        ctx: &CudaContext,
        stream: &DeviceStream,
    ) -> Result<Vec<CudaWeight>, EngError> {
        let prefix = match (site, layer) {
            (Site::Mixer, Some(layer)) if layer < self.config.dims.kinds.len() => {
                layer_name(layer, "attn_hyper_connection")
            }
            (Site::Ffn, Some(layer)) if layer < self.config.dims.kinds.len() => {
                layer_name(layer, "mlp_hyper_connection")
            }
            (Site::Head, None) => "model.language_model.hyper_connection_mixer".into(),
            _ => {
                return Err(EngError::Checkpoint(
                    "Flash-Next GGUF: invalid hyper site or layer".into(),
                ));
            }
        };
        let dims = &self.config.dims;
        let width = u64::from(dims.hidden)
            .checked_mul(u64::from(dims.branches))
            .ok_or_else(|| EngError::Checkpoint("GGUF hyper width overflows".into()))?;
        let rank = u64::from(dims.rank);
        let branches = u64::from(dims.branches);
        let mut parts = vec![
            (SiteTensor::Norm, "hc_norm.weight", vec![width], true),
            (
                SiteTensor::Down,
                "input_mix_weight_down.weight",
                vec![width, rank],
                false,
            ),
            (
                SiteTensor::Up,
                "input_mix_weight_up.weight",
                vec![rank, width],
                false,
            ),
        ];
        if site != Site::Head {
            parts.push((
                SiteTensor::Inject,
                "block_inject_weight.weight",
                vec![width, branches],
                false,
            ));
        }
        let plans = parts
            .into_iter()
            .map(|(part, name, dims, centered)| {
                let entry = self
                    .map
                    .get(Slot::hyper(site, part), layer)
                    .ok_or_else(|| {
                        EngError::Checkpoint(format!("Flash-Next GGUF: missing {site:?} {part:?}"))
                    })?;
                let matrix = part.q80_stays_stored();
                Ok((name, Staging::new(entry, &dims, centered, matrix)?))
            })
            .collect::<Result<Vec<_>, EngError>>()?;
        // One dtype serves both matrices: a Q8_0 matrix needs the other one Q8_0 too.
        let stored = |index: usize| plans[index].1.stored;
        if stored(1) != stored(2) {
            return Err(EngError::Checkpoint(format!(
                "Flash-Next GGUF: the hyper matrices of {prefix} are not both Q8_0"
            )));
        }
        plans
            .into_iter()
            .map(|(name, plan)| plan.upload(self, format!("{prefix}.{name}"), ctx, stream))
            .collect()
    }
}
