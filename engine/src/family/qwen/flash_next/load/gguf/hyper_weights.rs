//! Staged GGUF hyper-connection weights for the existing half-precision resident kernel.
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

struct HalfPlan<'a> {
    entry: &'a Mapped,
    blocks: usize,
    block_elements: usize,
    block_bytes: usize,
    output_bytes: usize,
    centered: bool,
}

impl<'a> HalfPlan<'a> {
    fn new(entry: &'a Mapped, dims: &[u64], centered: bool) -> Result<Self, EngError> {
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
        let output_bytes = elements.checked_mul(size_of::<u16>()).ok_or_else(bad)?;
        Ok(Self {
            entry,
            blocks,
            block_elements,
            block_bytes,
            output_bytes,
            centered,
        })
    }

    fn upload(
        &self,
        model: &GgufModel,
        name: String,
        ctx: &CudaContext,
        stream: &DeviceStream,
    ) -> Result<CudaWeight, EngError> {
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
    /// Stage one hyper site as F16, subtracting the stored norm's one before rounding.
    /// This expands quantized matrices and does not enable GGUF execution.
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
                Ok((name, HalfPlan::new(entry, &dims, centered)?))
            })
            .collect::<Result<Vec<_>, EngError>>()?;
        plans
            .into_iter()
            .map(|(name, plan)| plan.upload(self, format!("{prefix}.{name}"), ctx, stream))
            .collect()
    }
}
