//! Staged GGUF router weights in the F32 layout consumed by the resident router.

use super::{
    GgufModel,
    config::is_float,
    tensors::{Convention, Mapped},
};
#[cfg(test)]
mod tests;

use crate::{
    EngError,
    family::{exl3_pack::layer_name, project::cuda::CudaWeight, recipe::Role},
    kernels::cuda::{CHUNK_LIMIT, CudaContext, DeviceStream},
    weights::{GgufType, dequantize},
};

struct FloatPlan<'a> {
    entry: &'a Mapped,
    elements: usize,
    scalar_bytes: usize,
}

impl<'a> FloatPlan<'a> {
    fn new(entry: &'a Mapped, dims: &[u64]) -> Result<Self, EngError> {
        let bad = || {
            EngError::Checkpoint(format!(
                "Flash-Next GGUF: invalid router tensor {}",
                entry.name
            ))
        };
        if entry.dims != dims || !is_float(entry.ty) || entry.convention != Convention::AsStored {
            return Err(bad());
        }
        let elements = dims
            .iter()
            .try_fold(1usize, |n, &d| n.checked_mul(usize::try_from(d).ok()?))
            .filter(|&n| n > 0)
            .ok_or_else(bad)?;
        let scalar_bytes = usize::try_from(entry.ty.block_bytes()).map_err(|_| bad())?;
        let expected = elements.checked_mul(scalar_bytes).ok_or_else(bad)?;
        if usize::try_from(entry.bytes).ok() != Some(expected) || elements.checked_mul(4).is_none()
        {
            return Err(bad());
        }
        Ok(Self {
            entry,
            elements,
            scalar_bytes,
        })
    }

    fn upload(
        &self,
        model: &GgufModel,
        name: String,
        ctx: &CudaContext,
        stream: &DeviceStream,
    ) -> Result<CudaWeight, EngError> {
        let output_bytes = self.elements * size_of::<f32>(); // checked in new
        let chunk_elements = (CHUNK_LIMIT / size_of::<f32>()).min(self.elements);
        let mut raw = Vec::new();
        raw.try_reserve_exact(chunk_elements * self.scalar_bytes)
            .map_err(|_| EngError::MemoryPlan("GGUF router staging allocation failed".into()))?;
        let buffer = ctx.allocate(output_bytes)?;
        for first in (0..self.elements).step_by(chunk_elements) {
            let count = chunk_elements.min(self.elements - first);
            raw.resize(count * self.scalar_bytes, 0);
            model.weights.read_range(
                &self.entry.name,
                (first * self.scalar_bytes) as u64,
                &mut raw,
            )?;
            let values = dequantize(self.entry.ty, &raw)?;
            if values.len() != count {
                return Err(EngError::Checkpoint(format!(
                    "Flash-Next GGUF: invalid router data {}",
                    self.entry.name
                )));
            }
            let mut bytes = Vec::new();
            bytes
                .try_reserve_exact(count * size_of::<f32>())
                .map_err(|_| {
                    EngError::MemoryPlan("GGUF router conversion allocation failed".into())
                })?;
            for value in values {
                bytes.extend_from_slice(&value.to_ne_bytes());
            }
            buffer.write_at(stream, first * size_of::<f32>(), &bytes)?;
        }
        Ok(CudaWeight::Raw {
            name,
            ty: GgufType::F32,
            buffer,
        })
    }
}

impl GgufModel {
    /// Stage the expert router matrix and shared-expert gate as F32. This does not enable serving.
    pub(in super::super::super) fn upload_router(
        &self,
        layer: usize,
        ctx: &CudaContext,
        stream: &DeviceStream,
    ) -> Result<Vec<CudaWeight>, EngError> {
        let bad =
            || EngError::Checkpoint(format!("Flash-Next GGUF: missing router at layer {layer}"));
        let dims = &self.config.dims;
        let hidden = u64::from(dims.hidden);
        let experts = u64::from(dims.experts);
        let router = self.map.get(Role::Router, Some(layer)).ok_or_else(bad)?;
        let shared = self
            .map
            .get(Role::SharedRouterGate, Some(layer))
            .ok_or_else(bad)?;
        // Check both before allocating either; expert-major GGUF rows are the kernel's layout.
        let router = FloatPlan::new(router, &[hidden, experts])?;
        let shared = FloatPlan::new(shared, &[hidden])?;
        Ok(vec![
            router.upload(self, layer_name(layer, "mlp.gate.weight"), ctx, stream)?,
            shared.upload(
                self,
                layer_name(layer, "mlp.shared_expert_gate.weight"),
                ctx,
                stream,
            )?,
        ])
    }
}
