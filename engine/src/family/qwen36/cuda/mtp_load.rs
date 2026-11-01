//! Device upload of the MTP head of a stock pack, beside the model's own weights.
//!
//! The head is an input stage (two norms and the `2 * hidden` to `hidden` projection), one
//! full-attention block with a routed-expert MLP, and an output norm. The block's tensors go to
//! the device under the names of a one-layer table (`LayerTable::single`), the rest under the
//! names below, so they join the model's `CudaWeights` without colliding with it. The head has no
//! embedding table and no output head: the draft step reads the model's.

use super::{
    exl3_moe_source::moe_weights,
    exl3_source::{PackReader, attention_weights},
    layer_table::LayerTable,
    mixer_geometry::{DenseSpec, MixerGeometry},
    moe::MoeDims,
};
use crate::{
    EngError,
    family::{
        gguf_model::Role,
        project::cuda::CudaWeight,
        qwen36::{
            exl3_pack::Root,
            mtp_pack::{FC_PREFIX, NORM_NAME, PRE_FC_NORM_EMBEDDING, PRE_FC_NORM_HIDDEN},
        },
    },
};

/// The logical names of the head's tensors outside the block.
pub(super) const FC_NAME: &str = "mtp.fc.weight";
pub(super) const HIDDEN_NORM_NAME: &str = "mtp.pre_fc_norm_hidden";
pub(super) const EMBEDDING_NORM_NAME: &str = "mtp.pre_fc_norm_embedding";
pub(super) const OUTPUT_NORM_NAME: &str = "mtp.output_norm";

/// Uploads the head read through `reader`, which must read the MTP root. `table` is the
/// one-layer table of the block (`LayerTable::single` of a full-attention layer).
pub(super) fn mtp_weights(
    reader: &PackReader<'_>,
    table: &LayerTable,
    geometry: &MixerGeometry,
    moe: &MoeDims,
) -> Result<Vec<CudaWeight>, EngError> {
    if reader.root() != Root::Mtp {
        return Err(EngError::Checkpoint(
            "EXL3 MTP load: the reader does not read the MTP root".into(),
        ));
    }
    let hidden = geometry.hidden;
    let fc = DenseSpec {
        role: Role::AttnOutput,
        rows: hidden,
        k: 2 * hidden,
    };
    let mut weights = vec![
        reader.exl3(FC_NAME.to_owned(), FC_PREFIX, &fc, None)?,
        reader.norm(HIDDEN_NORM_NAME.to_owned(), PRE_FC_NORM_HIDDEN, hidden)?,
        reader.norm(
            EMBEDDING_NORM_NAME.to_owned(),
            PRE_FC_NORM_EMBEDDING,
            hidden,
        )?,
        reader.norm(OUTPUT_NORM_NAME.to_owned(), NORM_NAME, hidden)?,
    ];
    weights.extend(attention_weights(reader, table, 0, geometry)?);
    weights.extend(moe_weights(reader, table, 0, moe)?);
    Ok(weights)
}
