//! Staged DeltaNet state-weight upload from GGUF to the CUDA mixer's F32 layout.

use super::{
    GgufModel,
    config::is_float,
    delta_rows,
    tensors::{Convention, Mapped},
};
use crate::{
    EngError,
    family::{exl3_pack::layer_name, project::cuda::CudaWeight, recipe::Role},
    kernels::cuda::{CHUNK_LIMIT, CudaContext, DeviceStream},
    weights::{GgufType, dequantize},
};

struct StatePlan<'a> {
    entry: &'a Mapped,
    reader: delta_rows::DeltaRowReader,
    rows: usize,
    row_bytes: usize,
    row_elements: usize,
}

impl<'a> StatePlan<'a> {
    fn new(model: &'a GgufModel, layer: usize, role: Role) -> Result<Self, EngError> {
        let bad = || {
            EngError::Checkpoint(format!(
                "Flash-Next GGUF: invalid DeltaNet {role:?} at layer {layer}"
            ))
        };
        let entry = model.map.get(role, Some(layer)).ok_or_else(bad)?;
        if !is_float(entry.ty) || entry.convention != Convention::AsStored {
            return Err(bad());
        }
        let dims = &model.config.dims;
        let kh = usize::try_from(dims.key_heads).map_err(|_| bad())?;
        let vh = usize::try_from(dims.value_heads).map_err(|_| bad())?;
        let kd = usize::try_from(dims.key_dim).map_err(|_| bad())?;
        let vd = usize::try_from(dims.value_dim).map_err(|_| bad())?;
        let (rows, row_elements, mapping) = match (role, entry.dims.as_slice()) {
            (Role::DeltaConv, [kernel, channels]) => {
                let kernel = usize::try_from(*kernel).map_err(|_| bad())?;
                let channels = usize::try_from(*channels).map_err(|_| bad())?;
                if kernel != usize::try_from(dims.conv_kernel).map_err(|_| bad())? {
                    return Err(bad());
                }
                let skip = kh
                    .checked_mul(kd)
                    .and_then(|n| n.checked_mul(2))
                    .ok_or_else(bad)?;
                let mapping =
                    delta_rows::delta_rows(skip, vd, kh, vh, channels).map_err(|_| bad())?;
                (channels, kernel, mapping)
            }
            (Role::DeltaDecay | Role::DeltaDtBias, [heads]) => {
                let heads = usize::try_from(*heads).map_err(|_| bad())?;
                let mapping = delta_rows::delta_rows(0, 1, kh, vh, heads).map_err(|_| bad())?;
                (heads, 1, mapping)
            }
            (Role::DeltaNorm, [width]) if usize::try_from(*width).ok() == Some(vd) && vd > 0 => {
                let mut mapping = Vec::new();
                mapping.try_reserve_exact(vd).map_err(|_| {
                    EngError::MemoryPlan("GGUF DeltaNet norm plan allocation failed".into())
                })?;
                mapping.extend(0..vd);
                (vd, 1, mapping)
            }
            _ => return Err(bad()),
        };
        let scalar_bytes = usize::try_from(entry.ty.block_bytes()).map_err(|_| bad())?;
        let row_bytes = row_elements
            .checked_mul(scalar_bytes)
            .filter(|&n| n > 0)
            .ok_or_else(bad)?;
        let source_len = usize::try_from(entry.bytes).map_err(|_| bad())?;
        let reader =
            delta_rows::DeltaRowReader::new(mapping, row_bytes, source_len).map_err(|_| bad())?;
        Ok(Self {
            entry,
            reader,
            rows,
            row_bytes,
            row_elements,
        })
    }

    fn upload(
        &self,
        model: &GgufModel,
        layer: usize,
        name: &str,
        ctx: &CudaContext,
        stream: &DeviceStream,
    ) -> Result<CudaWeight, EngError> {
        let bad = || {
            EngError::Checkpoint(format!(
                "Flash-Next GGUF: invalid DeltaNet state tensor {}",
                self.entry.name
            ))
        };
        let out_row = self
            .row_elements
            .checked_mul(size_of::<f32>())
            .ok_or_else(bad)?;
        let output_bytes = self.rows.checked_mul(out_row).ok_or_else(bad)?;
        let chunk_rows = (CHUNK_LIMIT / self.row_bytes.max(out_row))
            .max(1)
            .min(self.rows);
        let capacity = chunk_rows.checked_mul(self.row_bytes).ok_or_else(bad)?;
        let mut raw = Vec::new();
        raw.try_reserve_exact(capacity).map_err(|_| {
            EngError::MemoryPlan("GGUF DeltaNet state staging allocation failed".into())
        })?;
        let buffer = ctx.allocate(output_bytes)?;
        for first in (0..self.rows).step_by(chunk_rows) {
            let count = chunk_rows.min(self.rows - first);
            raw.resize(count * self.row_bytes, 0);
            self.reader
                .read(first * self.row_bytes, &mut raw, |src, dst| {
                    model.weights.read_range(&self.entry.name, src as u64, dst)
                })?;
            let values = dequantize(self.entry.ty, &raw)?;
            if values.len() != count * self.row_elements {
                return Err(bad());
            }
            let mut bytes = Vec::new();
            bytes.try_reserve_exact(count * out_row).map_err(|_| {
                EngError::MemoryPlan("GGUF DeltaNet state conversion allocation failed".into())
            })?;
            for value in values {
                bytes.extend_from_slice(&value.to_ne_bytes());
            }
            buffer.write_at(stream, first * out_row, &bytes)?;
        }
        Ok(CudaWeight::Raw {
            name: layer_name(layer, name),
            ty: GgufType::F32,
            buffer,
        })
    }
}

impl GgufModel {
    /// Uploads grouped convolution and per-head state weights; GGUF decay is already in
    /// the negative-exponential form consumed by the scan and must not be transformed again.
    /// This staged helper does not enable GGUF execution.
    pub(in super::super::super) fn upload_delta_state(
        &self,
        layer: usize,
        ctx: &CudaContext,
        stream: &DeviceStream,
    ) -> Result<Vec<CudaWeight>, EngError> {
        let roles = [
            (Role::DeltaConv, "linear_attn.conv1d.weight"),
            (Role::DeltaDecay, "linear_attn.A_log"),
            (Role::DeltaDtBias, "linear_attn.dt_bias"),
            (Role::DeltaNorm, "linear_attn.norm.weight"),
        ];
        let plans = roles
            .into_iter()
            .map(|(role, name)| Ok((StatePlan::new(self, layer, role)?, name)))
            .collect::<Result<Vec<_>, EngError>>()?;
        plans
            .into_iter()
            .map(|(plan, name)| plan.upload(self, layer, name, ctx, stream))
            .collect()
    }
}
