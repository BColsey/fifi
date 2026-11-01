//! The whole-model EXL3 load: every tensor of a stock pack that the resident step reads, on the
//! device, under the GGUF names of the inventory.
//!
//! One layer at a time, a mixer (full attention or DeltaNet) and its MoE block; then the embedding
//! table, the output norm and the head. A pack that does not fit the family is refused before the
//! first upload where the headers allow it (config, expert tensors present); every matrix is
//! checked against the geometry before its own upload. The result is an ordinary
//! [`CudaWeights`] of the `Exl3` layout, so the GGUF loads and their readers are untouched.

use std::path::Path;

use super::{
    super::{
        exl3_pack::{EMBED_NAME, HEAD_PREFIX, NORM_NAME, Root, inspect_pack},
        mtp_pack::inspect_mtp_pack,
    },
    dims::StepDims,
    exl3_moe_source::{check_moe_source_geometry, moe_weights},
    exl3_source::{PackReader, attention_weights, delta_weights},
    layer_table::LayerTable,
    mixer_geometry::MixerGeometry,
    moe::MoeDims,
    mtp_load::mtp_weights,
    pinned::ROTARY_DIM,
    resources::{GlobalNames, GlobalSpecs, STEP_ROWS},
};
use crate::{
    EngError,
    family::{
        AttentionKind, FamilySpec, QWEN3_6_35B_A3B,
        gguf_model::Inventory,
        project::cuda::{CudaProjector, CudaWeight, CudaWeights, Reserve},
        qwen36::host::HostShape,
    },
    weights::exl3::Checkpoint,
};

pub(super) struct ModelFacts<'a> {
    pub(super) spec: &'a FamilySpec,
    pub(super) dims: &'a StepDims,
    pub(super) inventory: &'a Inventory,
}

pub(super) fn load_exl3_dir(
    projector: &CudaProjector,
    dir: &Path,
    reserve: Reserve,
) -> Result<CudaWeights, EngError> {
    load_exl3_dir_with_head(projector, dir, None, reserve)
}

/// Like [`load_exl3_dir`], and with `head_dir` set also the MTP head found in that pack, beside
/// the model's weights. The head's pack may differ from the model's in rate and codebook: every
/// matrix carries its own.
pub(super) fn load_exl3_dir_with_head(
    projector: &CudaProjector,
    dir: &Path,
    head_dir: Option<&Path>,
    reserve: Reserve,
) -> Result<CudaWeights, EngError> {
    let checkpoint = Checkpoint::open(dir)?;
    let inventory = Inventory::for_spec(&QWEN3_6_35B_A3B)?;
    let shape = HostShape::from_spec(&QWEN3_6_35B_A3B, ROTARY_DIM)?;
    let dims = StepDims::new(&shape, &inventory)?;
    let facts = ModelFacts {
        spec: &QWEN3_6_35B_A3B,
        dims: &dims,
        inventory: &inventory,
    };
    let reader = PackReader::new(projector, &checkpoint, reserve);
    let mut weights = model_weights(&reader, &facts)?;
    if let Some(head_dir) = head_dir {
        let head = Checkpoint::open(head_dir)?;
        let reader = PackReader::new(projector, &head, reserve).under(Root::Mtp);
        weights.extend(head_weights(&reader, &facts)?);
    }
    CudaWeights::from_exl3(weights)
}

pub(super) fn load_exl3_model(
    reader: &PackReader<'_>,
    facts: &ModelFacts<'_>,
) -> Result<CudaWeights, EngError> {
    CudaWeights::from_exl3(model_weights(reader, facts)?)
}

/// The MTP head of the pack behind `reader` (which reads the MTP root), under the names of a
/// block numbered after the model's own layers.
fn head_weights(
    reader: &PackReader<'_>,
    facts: &ModelFacts<'_>,
) -> Result<Vec<CudaWeight>, EngError> {
    let config = &reader.checkpoint.model_config;
    facts.spec.check_architecture_config(config)?;
    let layers = mtp_layers(config)?;
    let report = inspect_mtp_pack(&reader.checkpoint.source, &facts.dims.shape, layers)?;
    reader.admit(report.device_bytes)?;
    let moe = MoeDims::new(facts.dims, STEP_ROWS);
    check_moe_source_geometry(config, &moe)?;
    let table = LayerTable::new(facts.dims, facts.inventory)?;
    let first_full = (0..table.len())
        .find(|&layer| matches!(table.slot(layer).map(|s| s.kind), Ok(AttentionKind::Full)))
        .ok_or_else(|| {
            EngError::Checkpoint("EXL3 load: the model has no full-attention layer".into())
        })?;
    let block = table.single(first_full, table.len())?;
    let geometry = MixerGeometry::new(facts.dims)?;
    mtp_weights(reader, &block, &geometry, &moe)
}

/// `mtp_num_hidden_layers` of the config, which this load requires to be 1.
fn mtp_layers(config: &[u8]) -> Result<usize, EngError> {
    let value: serde_json::Value = serde_json::from_slice(config)
        .map_err(|error| EngError::Checkpoint(format!("EXL3 MTP config: {error}")))?;
    let layers = value
        .get("text_config")
        .and_then(|text| text.get("mtp_num_hidden_layers"))
        .and_then(serde_json::Value::as_u64);
    match layers {
        Some(1) => Ok(1),
        other => Err(EngError::Checkpoint(format!(
            "EXL3 MTP config: mtp_num_hidden_layers is {other:?}, the load needs 1"
        ))),
    }
}

/// Every tensor of the model's own pack, on the device.
fn model_weights(
    reader: &PackReader<'_>,
    facts: &ModelFacts<'_>,
) -> Result<Vec<CudaWeight>, EngError> {
    let model_config = &reader.checkpoint.model_config;
    if model_config.len() as u64 > crate::weights::exl3::CONFIG_LIMIT {
        return Err(EngError::MemoryPlan(
            "EXL3 model config exceeds 1 MiB".into(),
        ));
    }
    facts.spec.check_architecture_config(model_config)?;
    let moe = MoeDims::new(facts.dims, STEP_ROWS);
    check_moe_source_geometry(model_config, &moe)?;
    let table = LayerTable::new(facts.dims, facts.inventory)?;
    let geometry = MixerGeometry::new(facts.dims)?;
    // Every tensor is checked against the model from the headers alone, and the whole load is
    // admitted against the free memory, before the first upload.
    let report = inspect_pack(&reader.checkpoint.source, &facts.dims.shape)?;
    reader.admit(report.device_bytes)?;
    let mut weights = Vec::new();
    for layer in 0..table.len() {
        let mixer = match table.slot(layer)?.kind {
            AttentionKind::Full => attention_weights(reader, &table, layer, &geometry)?,
            AttentionKind::DeltaNet => delta_weights(reader, &table, layer, &geometry)?,
            other => {
                return Err(EngError::Checkpoint(format!(
                    "EXL3 load: layer {layer} has attention kind {other:?}, which has no mixer"
                )));
            }
        };
        weights.extend(mixer);
        weights.extend(moe_weights(reader, &table, layer, &moe)?);
    }
    weights.extend(global_weights(reader, facts)?);
    Ok(weights)
}

/// The embedding table is kept in its stored BF16.
fn global_weights(
    reader: &PackReader<'_>,
    facts: &ModelFacts<'_>,
) -> Result<Vec<CudaWeight>, EngError> {
    let names = GlobalNames::new(facts.inventory)?;
    let specs = GlobalSpecs::new(facts.dims);
    let embed_shape = [specs.embed.rows as u32, specs.embed.k as u32];
    Ok(vec![
        reader.table(names.embed, EMBED_NAME, &embed_shape)?,
        reader.norm(names.norm, NORM_NAME, specs.norm_values)?,
        reader.exl3(names.head, HEAD_PREFIX, &specs.head, None)?,
    ])
}

#[cfg(test)]
mod tests {
    use super::mtp_layers;
    use crate::EngError;

    fn layers(config: &str) -> Result<usize, EngError> {
        mtp_layers(config.as_bytes())
    }

    #[test]
    fn the_head_needs_exactly_one_block_in_the_config() {
        assert_eq!(
            layers(r#"{"text_config":{"mtp_num_hidden_layers":1}}"#).unwrap(),
            1
        );
        for config in [
            r#"{"text_config":{"mtp_num_hidden_layers":0}}"#,
            r#"{"text_config":{"mtp_num_hidden_layers":2}}"#,
            r#"{"text_config":{"mtp_num_hidden_layers":"1"}}"#,
            r#"{"text_config":{"mtp_num_hidden_layers":-1}}"#,
            r#"{"text_config":{"mtp_num_hidden_layers":1.5}}"#,
            r#"{"text_config":{}}"#,
            r#"{"mtp_num_hidden_layers":1}"#,
            r#"[]"#,
            r#"not json"#,
            "",
        ] {
            assert!(
                matches!(layers(config), Err(EngError::Checkpoint(_))),
                "{config}"
            );
        }
    }
}
