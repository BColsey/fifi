//! Explicit opt-in loading of the separate GGUF drafter for the Inkling target.

use std::path::Path;

use super::serve::InklingDraft;
use crate::{EngError, weights::dspark_gguf::DsparkGguf};

pub(crate) fn configured_draft(context: usize) -> Result<Option<InklingDraft>, EngError> {
    const PATH: &str = "FIFI_INKLING_DSPARK_GGUF";
    let cuda = std::env::var("FIFI_INKLING_DSPARK_CUDA").as_deref() == Ok("1");
    let cuda_head = std::env::var("FIFI_INKLING_DSPARK_CUDA_HEAD").as_deref() == Ok("1");
    let path = std::env::var_os(PATH);
    if cuda_head && (!cuda || path.is_none()) {
        return Err(super::bad(
            "FIFI_INKLING_DSPARK_CUDA_HEAD requires FIFI_INKLING_DSPARK_CUDA=1 and a DSpark GGUF sidecar",
        ));
    }
    let Some(path) = path else {
        return Ok(None);
    };
    let model = DsparkGguf::open(Path::new(&path))?;
    let config = model.config();
    let workspace_bytes = config.workspace_bytes(context)?;
    let (width, vocab, taps, max_drafts) = (
        config.width,
        config.vocab,
        config.target_layers.clone(),
        config.block_size - 1,
    );
    Ok(Some(InklingDraft {
        model,
        cuda,
        cuda_head,
        taps,
        max_drafts,
        width,
        vocab,
        workspace_bytes,
    }))
}
