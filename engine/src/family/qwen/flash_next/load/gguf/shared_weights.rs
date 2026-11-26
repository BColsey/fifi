//! Staged native GGUF projections for a Flash-Next shared expert.

use super::{
    GgufModel,
    tensors::{Convention, Mapped},
    upload_native_head,
};
use crate::{
    EngError,
    family::{exl3_pack::layer_name, project::cuda::CudaWeight, recipe::Role},
    kernels::cuda::{CudaContext, resident::matvec_q::QuantType},
};

fn validate<'a>(entry: &'a Mapped, dims: &[u64]) -> Result<&'a Mapped, EngError> {
    let bad = || {
        EngError::Checkpoint(format!(
            "Flash-Next GGUF: invalid shared projection {}",
            entry.name
        ))
    };
    if entry.dims != dims || entry.convention != Convention::AsStored {
        return Err(bad());
    }
    let [k, rows] = dims else { return Err(bad()) };
    let (k, rows) = (
        usize::try_from(*k).map_err(|_| bad())?,
        usize::try_from(*rows).map_err(|_| bad())?,
    );
    let quant = QuantType::from_gguf(entry.ty).ok_or_else(bad)?;
    if k == 0 || rows == 0 || !k.is_multiple_of(quant.block_size()) {
        return Err(bad());
    }
    let bytes = quant.weight_bytes(rows, k).ok_or_else(bad)?;
    if usize::try_from(entry.bytes).map_err(|_| bad())? != bytes {
        return Err(bad());
    }
    Ok(entry)
}

impl GgufModel {
    /// Upload one layer's gate, up, and down projections in their native stored layout.
    /// This is staged and does not connect a GGUF MoE block to the executor.
    pub(in super::super::super) fn upload_shared_expert(
        &self,
        layer: usize,
        ctx: &CudaContext,
    ) -> Result<Vec<CudaWeight>, EngError> {
        let d = &self.config.dims;
        let (hidden, shared) = (u64::from(d.hidden), u64::from(d.shared_inter));
        let roles = [
            (
                Role::SharedGate,
                "mlp.shared_expert.gate_proj",
                [hidden, shared],
            ),
            (
                Role::SharedUp,
                "mlp.shared_expert.up_proj",
                [hidden, shared],
            ),
            (
                Role::SharedDown,
                "mlp.shared_expert.down_proj",
                [shared, hidden],
            ),
        ];
        let checked = roles
            .into_iter()
            .map(|(role, name, dims)| {
                let entry = self.map.get(role, Some(layer)).ok_or_else(|| {
                    EngError::Checkpoint(format!(
                        "Flash-Next GGUF: missing {role:?} at layer {layer}"
                    ))
                })?;
                Ok((name, validate(entry, &dims)?))
            })
            .collect::<Result<Vec<_>, EngError>>()?;
        checked
            .into_iter()
            .map(|(name, entry)| {
                upload_native_head(&self.weights, entry, layer_name(layer, name), ctx)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests;
