//! The model configuration of a `qwen4exp` GGUF: the metadata keys of shard 1 read into the
//! same [`Dims`] the `config.json` of the EXL3 pack gives, so every shape check and the memory
//! plan run on one structure.
//!
//! Key by key (GGUF key, then the `config.json` field it replaces):
//!
//! | GGUF key | `config.json` field |
//! |---|---|
//! | `block_count` | `num_hidden_layers` |
//! | `embedding_length` | `hidden_size` |
//! | `attention.layer_norm_rms_epsilon` | `rms_norm_eps` |
//! | `hyper_connection.count`, `.low_rank` | `hc_count`, `hc_lowrank` |
//! | `attention.head_count`, `.head_count_kv` | `num_attention_heads`, `num_key_value_heads` |
//! | `attention.key_length` (and `value_length`) | `head_dim` |
//! | `rope.freq_base`, `rope.dimension_count` | `rope_parameters.rope_theta`, `head_dim * partial_rotary_factor` |
//! | `attention.indexer.head_count`, `.key_length`, `.top_k` | `indexer_n_heads`, `indexer_head_dim`, `indexer_budget` |
//! | `attention.compress_ratios` | `layer_types` (nonzero ratio is a QSA layer) and `indexer_compress_ratio` |
//! | `full_attention_interval` | cross-check of `layer_types` |
//! | `ssm.group_count`, `.state_size`, `.time_step_rank`, `.conv_kernel` | `linear_num_key_heads`, `linear_{key,value}_head_dim`, `linear_num_value_heads`, `linear_conv_kernel_dim` |
//! | `ssm.inner_size` | cross-check: value heads times value head dimension |
//! | `expert_count`, `expert_used_count` | `num_experts`, `num_experts_per_tok` |
//! | `expert_feed_forward_length`, `expert_shared_feed_forward_length` | `moe_intermediate_size`, `shared_expert_intermediate_size` |
//! | `ple.layers[0]` | `ple_layer_ids[0] - 1` (the GGUF list is already zero-based) |
//! | `ple.ngram_size`, `.heads_per_ngram`, `.conv_kernel` | `ngram_size`, `heads_per_ngram`, `ple_conv_kernel_size` |
//! | `ple.eos_token_id` | `eos_token_id` (the id that resets the n-gram hash) |
//! | `embedding_length_per_layer_input` | `ple_embed_dim / ((ngram_size - 1) * heads_per_ngram)` |
//! | `ple.layer_multipliers`, `.head_offsets`, `.head_vocab_sizes` | the three hash tensors of the n-gram file |
//! | `context_length` | `max_position_embeddings` |
//! | token embedding row count | `vocab_size` |
//!
//! `expert_weights_norm`, when the file carries it, must be true (the executor renormalises
//! the top-k weights) and `expert_weights_scale`, when present, must be 0 or 1 (no scaling);
//! a file that says otherwise is refused by key. The real file carries neither key, so the
//! renormalisation stays on, as `norm_topk_prob` absent does in the EXL3 config.
//!
//! Fields the GGUF does not carry take the value the EXL3 path gives them:
//! `split_ngram_parts` has no counterpart (the GGUF holds the n-gram table as one tensor).
//! `rope.dimension_sections` is read and checked but not used: the model decodes text only, where the
//! interleaved multi-axis rotary reduces to the plain rotary the step applies.

use super::super::{Dims, product};
use crate::{
    EngError,
    family::qwen::flash_next::step::ModelShape,
    weights::{GgufHeader, GgufType, GgufWeights, MetaValue},
};

/// The architecture string of the model's GGUF files.
pub(in crate::family::qwen::flash_next) const ARCHITECTURE: &str = "qwen4exp";

fn refuse(detail: impl AsRef<str>) -> EngError {
    EngError::Checkpoint(format!("Flash-Next GGUF: {}", detail.as_ref()))
}

/// The hash constants of the n-gram table: the layer multipliers and the row range of every
/// head. The EXL3 pack keeps them as three small tensors beside the table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::family::qwen::flash_next) struct NgramHash {
    pub(in crate::family::qwen::flash_next) multipliers: Vec<i64>,
    pub(in crate::family::qwen::flash_next) offsets: Vec<i64>,
    pub(in crate::family::qwen::flash_next) vocab: Vec<i64>,
}

/// The configuration read from the metadata.
#[derive(Debug)]
pub(in crate::family::qwen::flash_next) struct GgufConfig {
    pub(super) dims: Dims,
    pub(in crate::family::qwen::flash_next) hash: NgramHash,
    /// Rows of the n-gram table: the largest row any head addresses.
    pub(in crate::family::qwen::flash_next) ngram_rows_needed: u64,
    pub(in crate::family::qwen::flash_next) max_positions: u32,
    pub(in crate::family::qwen::flash_next) rope_sections: [u32; 4],
}

struct Meta<'a> {
    weights: Option<&'a GgufWeights>,
    header: &'a GgufHeader,
}

impl Meta<'_> {
    fn key(suffix: &str) -> String {
        format!("{ARCHITECTURE}.{suffix}")
    }

    fn value(&self, suffix: &str) -> Result<&MetaValue, EngError> {
        let key = Self::key(suffix);
        self.header
            .meta(&key)
            .ok_or_else(|| refuse(format!("metadata key {key} is missing")))
    }

    fn uint(&self, suffix: &str) -> Result<u32, EngError> {
        let key = Self::key(suffix);
        let wide = match *self.value(suffix)? {
            MetaValue::U8(v) => u64::from(v),
            MetaValue::U16(v) => u64::from(v),
            MetaValue::U32(v) => u64::from(v),
            MetaValue::U64(v) => v,
            MetaValue::I8(v) if v >= 0 => v as u64,
            MetaValue::I16(v) if v >= 0 => v as u64,
            MetaValue::I32(v) if v >= 0 => v as u64,
            MetaValue::I64(v) if v >= 0 => v as u64,
            _ => return Err(refuse(format!("metadata key {key} is not a count"))),
        };
        u32::try_from(wide)
            .ok()
            .filter(|&v| v > 0)
            .ok_or_else(|| refuse(format!("metadata key {key} is zero or above 2^32")))
    }

    fn float(&self, suffix: &str) -> Result<f64, EngError> {
        let key = Self::key(suffix);
        match *self.value(suffix)? {
            MetaValue::F32(v) => Ok(f64::from(v)),
            MetaValue::F64(v) => Ok(v),
            _ => Err(refuse(format!("metadata key {key} is not a float"))),
        }
    }

    /// The integers of an array key, read from the file because the header keeps only the
    /// array's extent.
    fn array(&self, suffix: &str) -> Result<Vec<i64>, EngError> {
        let key = Self::key(suffix);
        let MetaValue::Array(array) = self.value(suffix)? else {
            return Err(refuse(format!("metadata key {key} is not an array")));
        };
        let width: u64 = match array.elem_type {
            0 | 1 => 1,
            2 | 3 => 2,
            4 | 5 => 4,
            10 | 11 => 8,
            _ => {
                return Err(refuse(format!(
                    "metadata key {key} is not an integer array"
                )));
            }
        };
        let signed = matches!(array.elem_type, 1 | 3 | 5 | 11);
        let weights = self.weights.ok_or_else(|| {
            refuse(format!(
                "metadata key {key} is an array and no file is open to read it from"
            ))
        })?;
        let bytes = usize::try_from(array.byte_range.end - array.byte_range.start)
            .map_err(|_| refuse(format!("metadata key {key} is too large")))?;
        if bytes as u64 != array.len * width {
            return Err(refuse(format!(
                "metadata key {key} holds {bytes} bytes for {} elements of {width} bytes",
                array.len
            )));
        }
        let mut raw = vec![0u8; bytes];
        weights.read_header_range(array.byte_range.start, &mut raw)?;
        Ok(raw
            .chunks_exact(width as usize)
            .map(|chunk| {
                let mut word = [0u8; 8];
                word[..chunk.len()].copy_from_slice(chunk);
                let unsigned = u64::from_le_bytes(word);
                if signed {
                    let shift = 64 - 8 * width as u32;
                    ((unsigned << shift) as i64) >> shift
                } else {
                    unsigned as i64
                }
            })
            .collect())
    }

    fn present(&self, suffix: &str) -> bool {
        self.header.meta(&Self::key(suffix)).is_some()
    }
}

impl GgufConfig {
    /// Reads the configuration of the open checkpoint. The checkpoint must be a `qwen4exp`
    /// model.
    pub(in super::super::super) fn read(weights: &GgufWeights) -> Result<Self, EngError> {
        Self::read_from(weights.header(), Some(weights))
    }

    fn read_from(header: &GgufHeader, weights: Option<&GgufWeights>) -> Result<Self, EngError> {
        match header.meta("general.architecture") {
            Some(MetaValue::String(arch)) if arch == ARCHITECTURE => {}
            Some(MetaValue::String(arch)) => {
                return Err(refuse(format!(
                    "general.architecture is {arch:?}, expected {ARCHITECTURE:?}"
                )));
            }
            _ => return Err(refuse("general.architecture is missing or not a string")),
        }
        let meta = Meta { weights, header };
        let layers = meta.uint("block_count")?;
        let hidden = meta.uint("embedding_length")?;
        let eps = meta.float("attention.layer_norm_rms_epsilon")? as f32;
        let branches = meta.uint("hyper_connection.count")?;
        let rank = meta.uint("hyper_connection.low_rank")?;
        let width = product(&[branches, hidden]).map_err(|_| refuse("hyper width overflows"))?;

        let vocab = header
            .tensor("token_embd.weight")
            .and_then(|tensor| tensor.dims().get(1).copied())
            .and_then(|rows| u32::try_from(rows).ok())
            .ok_or_else(|| refuse("token_embd.weight is missing or has no row count"))?;
        if let Some(MetaValue::Array(tokens)) = header.meta("tokenizer.ggml.tokens")
            && tokens.len != u64::from(vocab)
        {
            return Err(refuse(format!(
                "the tokenizer lists {} tokens but token_embd.weight has {vocab} rows",
                tokens.len
            )));
        }

        let heads_q = meta.uint("attention.head_count")?;
        let heads_kv = meta.uint("attention.head_count_kv")?;
        let head_dim = meta.uint("attention.key_length")?;
        if meta.uint("attention.value_length")? != head_dim {
            return Err(refuse(
                "attention.value_length differs from attention.key_length",
            ));
        }
        let theta = meta.float("rope.freq_base")? as f32;
        let rotary_dim = meta.uint("rope.dimension_count")?;
        let sections = meta.array("rope.dimension_sections")?;
        let rope_sections: [u32; 4] = sections
            .iter()
            .map(|&v| u32::try_from(v).ok())
            .collect::<Option<Vec<u32>>>()
            .and_then(|v| v.try_into().ok())
            .ok_or_else(|| refuse("rope.dimension_sections must hold four counts"))?;
        let mrope: u32 = rope_sections[..3].iter().sum();
        if mrope * 2 != rotary_dim {
            return Err(refuse(format!(
                "rope.dimension_sections {rope_sections:?} cover {} rotary values, rope.dimension_count is {rotary_dim}",
                mrope * 2
            )));
        }

        let idx_heads = meta.uint("attention.indexer.head_count")?;
        let idx_dim = meta.uint("attention.indexer.key_length")?;
        let budget = meta.uint("attention.indexer.top_k")?;

        let ratios = meta.array("attention.compress_ratios")?;
        if ratios.len() != layers as usize {
            return Err(refuse(format!(
                "attention.compress_ratios has {} entries for {layers} layers",
                ratios.len()
            )));
        }
        let kinds: Vec<bool> = ratios.iter().map(|&ratio| ratio != 0).collect();
        let mut nonzero = ratios.iter().copied().filter(|&ratio| ratio != 0);
        let ratio = nonzero
            .next()
            .and_then(|first| u32::try_from(first).ok())
            .ok_or_else(|| refuse("attention.compress_ratios has no attention layer"))?;
        if nonzero.any(|other| other != i64::from(ratio)) {
            return Err(refuse("attention layers disagree on their compress ratio"));
        }
        if meta.present("full_attention_interval") {
            let interval = meta.uint("full_attention_interval")? as usize;
            if kinds
                .iter()
                .enumerate()
                .any(|(layer, &attention)| attention != ((layer + 1) % interval == 0))
            {
                return Err(refuse(
                    "attention.compress_ratios and full_attention_interval disagree on which layers attend",
                ));
            }
        }

        if let Some(MetaValue::Bool(false)) = header.meta(&Meta::key("expert_weights_norm")) {
            return Err(refuse(
                "metadata key qwen4exp.expert_weights_norm is false, but the executor renormalises the top-k expert weights",
            ));
        }
        if meta.present("expert_weights_scale") {
            let scale = meta.float("expert_weights_scale")?;
            if scale != 0.0 && scale != 1.0 {
                return Err(refuse(format!(
                    "metadata key qwen4exp.expert_weights_scale is {scale}, but the executor applies no expert weight scale"
                )));
            }
        }

        let key_heads = meta.uint("ssm.group_count")?;
        let key_dim = meta.uint("ssm.state_size")?;
        let value_heads = meta.uint("ssm.time_step_rank")?;
        let value_dim = key_dim;
        let conv_kernel = meta.uint("ssm.conv_kernel")?;
        if meta.uint("ssm.inner_size")? != value_heads * value_dim {
            return Err(refuse(
                "ssm.inner_size is not time_step_rank times state_size",
            ));
        }

        let experts = meta.uint("expert_count")?;
        let top_k = meta.uint("expert_used_count")?;
        let moe_inter = meta.uint("expert_feed_forward_length")?;
        let shared_inter = meta.uint("expert_shared_feed_forward_length")?;

        let ple_layers = meta.array("ple.layers")?;
        let [ple_layer] = ple_layers[..] else {
            return Err(refuse(format!(
                "ple.layers lists {} layers; one is supported",
                ple_layers.len()
            )));
        };
        let ple_layer = usize::try_from(ple_layer)
            .ok()
            .filter(|&layer| layer < layers as usize)
            .ok_or_else(|| refuse("ple.layers names a layer outside the model"))?;
        if kinds[ple_layer] {
            return Err(refuse("the PLE layer must be a linear attention layer"));
        }
        let ngram = meta.uint("ple.ngram_size")?;
        let heads_per_ngram = meta.uint("ple.heads_per_ngram")?;
        let ple_conv = meta.uint("ple.conv_kernel")?;
        let eos = meta.uint("ple.eos_token_id")?;
        let ple_head_dim = meta.uint("embedding_length_per_layer_input")?;
        let ple_heads = ngram
            .checked_sub(1)
            .and_then(|n| n.checked_mul(heads_per_ngram))
            .ok_or_else(|| refuse("ple head count overflows"))?;
        let ple_embed = product(&[ple_heads, ple_head_dim])
            .map_err(|_| refuse("ple embedding width overflows"))?;

        let multipliers = meta.array("ple.layer_multipliers")?;
        let offsets = meta.array("ple.head_offsets")?;
        let vocab_sizes = meta.array("ple.head_vocab_sizes")?;
        if multipliers.len() < ngram as usize
            || offsets.len() < ple_heads as usize
            || vocab_sizes.len() < ple_heads as usize
        {
            return Err(refuse(format!(
                "the n-gram hash arrays are too short: {} multipliers for n-gram size {ngram}, {} offsets and {} head sizes for {ple_heads} heads",
                multipliers.len(),
                offsets.len(),
                vocab_sizes.len()
            )));
        }
        let mut rows_needed = 0u64;
        for (head, (&offset, &size)) in offsets.iter().zip(&vocab_sizes).enumerate() {
            if offset < 0 || size <= 0 {
                return Err(refuse(format!("n-gram head {head} has a negative range")));
            }
            rows_needed = rows_needed.max(offset as u64 + size as u64);
        }

        let max_positions = meta.uint("context_length")?;

        let dims = Dims {
            hidden,
            eps,
            branches,
            rank,
            width,
            vocab,
            kinds: kinds.clone(),
            heads_q,
            heads_kv,
            head_dim,
            rotary_dim,
            theta,
            idx_heads,
            idx_dim,
            budget,
            ratio,
            key_heads,
            key_dim,
            value_heads,
            value_dim,
            conv_kernel,
            experts,
            top_k,
            moe_inter,
            shared_inter,
            renorm: true,
            ngram,
            heads_per_ngram,
            ple_heads,
            ple_head_dim,
            ple_embed,
            ple_conv,
            ple_layer,
            ngram_parts: None,
            eos,
        };
        // The same checks the shape constructors apply to the EXL3 pack's configuration.
        dims.model_shape()?;
        Ok(Self {
            dims,
            hash: NgramHash {
                multipliers,
                offsets,
                vocab: vocab_sizes,
            },
            ngram_rows_needed: rows_needed,
            max_positions,
            rope_sections,
        })
    }

    /// The shape of every block and the model-level layout.
    pub(in crate::family::qwen::flash_next) fn model_shape(&self) -> Result<ModelShape, EngError> {
        self.dims.model_shape()
    }

    /// Per layer, whether it is a QSA attention layer (the others are DeltaNet).
    pub(in crate::family::qwen::flash_next) fn kinds(&self) -> &[bool] {
        &self.dims.kinds
    }

    /// Routed experts picked per token and layer.
    pub(in crate::family::qwen::flash_next) fn top_k(&self) -> usize {
        self.dims.top_k as usize
    }

    pub(in crate::family::qwen::flash_next) fn layers(&self) -> usize {
        self.dims.kinds.len()
    }

    /// The layer whose attention read follows the PLE update.
    pub(in crate::family::qwen::flash_next) fn ple_layer(&self) -> usize {
        self.dims.ple_layer
    }
}

/// Whether a tensor type stores floating-point values as plain elements.
pub(in crate::family::qwen::flash_next) fn is_float(ty: GgufType) -> bool {
    matches!(ty, GgufType::F32 | GgufType::F16 | GgufType::Bf16)
}
