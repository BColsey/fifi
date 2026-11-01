//! The MTP head of a stock pack: its tensor names and a header-only walk against the model shape.
//!
//! The head is one input stage and `mtp_num_hidden_layers` blocks. The input stage norms the
//! target's final hidden row and the embedding of the drafted token, joins the two (embedding
//! first) and projects `2 * hidden` to `hidden` with `mtp.fc`. Each block is a full-attention
//! layer with a routed-expert MLP, shaped like a full-attention layer of the model, and the
//! stage ends with `mtp.norm`. The head has no embedding table and no output head of its own: it
//! reads the model's.

use super::{
    exl3_pack::{Inspector, PackReport, Root},
    host::HostShape,
};
use crate::{EngError, weights::StreamedCheckpoint};

/// Pack name of the norm of the target hidden row.
pub(super) const PRE_FC_NORM_HIDDEN: &str = "mtp.pre_fc_norm_hidden.weight";
/// Pack name of the norm of the drafted token's embedding.
pub(super) const PRE_FC_NORM_EMBEDDING: &str = "mtp.pre_fc_norm_embedding.weight";
/// Pack prefix of the `2 * hidden` to `hidden` projection.
pub(super) const FC_PREFIX: &str = "mtp.fc";
/// Pack name of the output norm of the head.
pub(super) const NORM_NAME: &str = "mtp.norm.weight";

/// Walks the headers of the MTP tensors of `source` against `shape`, for a head of `layers`
/// blocks. Reads no payload. The report counts the device bytes of the head alone.
pub(super) fn inspect_mtp_pack(
    source: &StreamedCheckpoint,
    shape: &HostShape,
    layers: usize,
) -> Result<PackReport, EngError> {
    if layers == 0 {
        return Err(EngError::Checkpoint(
            "EXL3 pack: the MTP head has no block".into(),
        ));
    }
    let h = shape.hidden;
    let hidden = u32::try_from(h).unwrap_or(u32::MAX);
    let mut walk = Inspector::new(source, Root::Mtp);
    walk.plain(PRE_FC_NORM_HIDDEN, &[hidden], false)?;
    walk.plain(PRE_FC_NORM_EMBEDDING, &[hidden], false)?;
    walk.plain(NORM_NAME, &[hidden], false)?;
    walk.matrix(FC_PREFIX, 2 * h, h)?;
    for layer in 0..layers {
        for norm in ["input_layernorm.weight", "post_attention_layernorm.weight"] {
            walk.plain(&Root::Mtp.name(layer, norm), &[hidden], false)?;
        }
        walk.attention(layer, shape)?;
        walk.moe(layer, shape)?;
    }
    walk.report()
}
