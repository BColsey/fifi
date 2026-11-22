//! Host-side GGUF embedding and PLE weights, independent of the transformer trunk.

use super::{
    GgufModel,
    projector::{HostWeight, Remap, refuse},
    tensors::{Convention, Mapped, Slot},
};
use crate::{
    EngError,
    family::{
        qwen::flash_next::ple::PleWeights,
        recipe::{Global, Role},
    },
    weights::{GgufWeights, dequantize},
};

pub(super) fn read_floats(weights: &GgufWeights, entry: &Mapped) -> Result<Vec<f32>, EngError> {
    let (ty, bytes) = weights.read_tensor(&entry.name)?;
    let mut values = dequantize(ty, &bytes)?;
    if entry.convention == Convention::GammaOnePlusW {
        for value in &mut values {
            *value -= 1.0;
        }
    }
    Ok(values)
}

pub(super) struct GgufPleWeights {
    pub(super) embed: HostWeight,
    table: HostWeight,
    rows: [u64; 1],
    key: HostWeight,
    value: HostWeight,
    norms: [Vec<f32>; 3],
    conv: Vec<f32>,
    multipliers: Vec<i64>,
    vocab: Vec<i64>,
    offsets: Vec<i64>,
}

impl GgufPleWeights {
    pub(super) fn load(model: &GgufModel) -> Result<Self, EngError> {
        let layer = Some(model.config().dims.ple_layer);
        let get = |role: Slot, layer: Option<usize>| -> Result<&Mapped, EngError> {
            model
                .map()
                .get(role, layer)
                .ok_or_else(|| refuse(format!("{role:?} of layer {layer:?} is not mapped")))
        };
        let plain = |role, layer| -> Result<HostWeight, EngError> {
            Ok(HostWeight::new(&[get(role, layer)?], Remap::None))
        };
        let floats = |role, layer| -> Result<Vec<f32>, EngError> {
            read_floats(&model.weights, get(role, layer)?)
        };
        let table = get(Global::NgramTable.into(), None)?;
        let hash = &model.config().hash;
        Ok(Self {
            embed: plain(Global::Embedding.into(), None)?,
            table: HostWeight::new(&[table], Remap::None),
            rows: [table.dims[1]],
            key: plain(Role::PleKey.into(), layer)?,
            value: plain(Role::PleValue.into(), layer)?,
            norms: [
                floats(Role::PleNormKey.into(), layer)?,
                floats(Role::PleNormQuery.into(), layer)?,
                floats(Role::PleNormConv.into(), layer)?,
            ],
            conv: floats(Role::PleConv.into(), layer)?,
            multipliers: hash.multipliers.clone(),
            vocab: hash.vocab.clone(),
            offsets: hash.offsets.clone(),
        })
    }
    pub(super) fn view(&self) -> PleWeights<'_, HostWeight> {
        PleWeights {
            multipliers: &self.multipliers,
            vocab: &self.vocab,
            offsets: &self.offsets,
            shards: std::slice::from_ref(&self.table),
            shard_rows: &self.rows,
            weight_scale: 1.0,
            key: &self.key,
            value: &self.value,
            norm_key: &self.norms[0],
            norm_query: &self.norms[1],
            norm_conv: &self.norms[2],
            conv: &self.conv,
        }
    }
}
