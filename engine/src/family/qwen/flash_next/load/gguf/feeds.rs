//! The host feeds of a token from a Flash-Next GGUF: the embedding row (decoded from the
//! quantized table) and the PLE block (the norms, the convolution, the two projections and the
//! n-gram rows), assembled into the same feed the EXL3 pack uses.
//!
//! The PLE projections run on the host as F32 matvecs over the decoded matrices (131 MB for the
//! two), as the EXL3 feed does by default. The norms are stored as `1 + w` in the file and the
//! step applies the 1 itself, so they are returned to `w` on read.

use super::{
    GgufModel,
    tensors::{Convention, Mapped, Slot},
};
use crate::{
    EngError,
    family::recipe::{Global, Role as Wanted},
    family::qwen::flash_next::cuda::{
        pack_feeds::{EmbeddingRows, PackFeeds},
        ple_feed::{HostMatrix, HostProjection, PleFeed, PleVectors},
    },
    weights::{GgufWeights, dequantize},
};

fn refuse(detail: impl AsRef<str>) -> EngError {
    EngError::Checkpoint(format!("Flash-Next GGUF feeds: {}", detail.as_ref()))
}

/// The embedding table of the file: one row decoded per token, read in place.
struct GgufEmbedding {
    weights: std::sync::Arc<GgufWeights>,
    name: String,
    ty: crate::weights::GgufType,
    hidden: usize,
    row_bytes: usize,
    vocab: u64,
    branches: usize,
}

impl EmbeddingRows for GgufEmbedding {
    fn stream_row(&self, token: u32) -> Result<Vec<f32>, EngError> {
        if u64::from(token) >= self.vocab {
            return Err(EngError::BadRequest(format!(
                "token {token} is outside the {} embedding rows",
                self.vocab
            )));
        }
        let mut bytes = vec![0u8; self.row_bytes];
        self.weights.read_range(
            &self.name,
            u64::from(token) * self.row_bytes as u64,
            &mut bytes,
        )?;
        let row = dequantize(self.ty, &bytes)?;
        debug_assert_eq!(row.len(), self.hidden);
        Ok(row.repeat(self.branches))
    }
}

fn read_floats(weights: &GgufWeights, entry: &Mapped) -> Result<Vec<f32>, EngError> {
    let (ty, bytes) = weights.read_tensor(&entry.name)?;
    let mut values = dequantize(ty, &bytes)?;
    if entry.convention == Convention::GammaOnePlusW {
        for value in &mut values {
            *value -= 1.0;
        }
    }
    Ok(values)
}

/// The token feeds of `model`: embedding rows from the file, the PLE block from the file's PLE
/// layer, n-gram rows from the file's table. `threads` is the host PLE projection's thread count.
pub(in crate::family::qwen::flash_next) fn gguf_feeds(
    model: &GgufModel,
    threads: usize,
) -> Result<PackFeeds, EngError> {
    let config = model.config();
    let map = model.map();
    let shape = config.model_shape()?.ple;
    let layer = Some(config.ple_layer());
    let get = |role: Slot, layer| {
        map.get(role, layer)
            .ok_or_else(|| refuse(format!("{role:?} is not mapped")))
    };
    let embed = get(Slot::Global(Global::Embedding), None)?;
    let hidden = embed.dims[0] as usize;
    let branches = config.dims.branches as usize;
    let block = embed.ty.block_elems() as usize;
    if !hidden.is_multiple_of(block) {
        return Err(refuse("embedding rows are not whole blocks"));
    }
    let embedding = GgufEmbedding {
        weights: std::sync::Arc::clone(&model.weights),
        name: embed.name.clone(),
        ty: embed.ty,
        hidden,
        row_bytes: hidden / block * embed.ty.block_bytes() as usize,
        vocab: embed.dims[1],
        branches,
    };
    let width = hidden * branches;
    let ple_embed = shape.embed();
    let matrix = |role: Wanted, rows: usize| -> Result<HostMatrix, EngError> {
        let entry = get(Slot::Layer(role), layer)?;
        HostMatrix::new(rows, ple_embed, read_floats(&model.weights, entry)?)
    };
    let projection = HostProjection::new(
        matrix(Wanted::PleKey, width)?,
        matrix(Wanted::PleValue, hidden)?,
        threads,
    );
    let vectors = PleVectors::from_parts(
        read_floats(&model.weights, get(Slot::Layer(Wanted::PleNormKey), layer)?)?,
        read_floats(&model.weights, get(Slot::Layer(Wanted::PleNormQuery), layer)?)?,
        read_floats(&model.weights, get(Slot::Layer(Wanted::PleNormConv), layer)?)?,
        read_floats(&model.weights, get(Slot::Layer(Wanted::PleConv), layer)?)?,
    );
    let ple = PleFeed::new(shape, model.ngram()?, vectors, Box::new(projection));
    Ok(PackFeeds::with_embedding(Box::new(embedding), ple, None))
}
