//! GGUF loader for dense recurrent DSpark drafters published under the
//! `dflash` architecture. Embeddings and the vocabulary head belong to
//! the target model and are supplied by the caller.

use std::path::Path;

use super::{GgufLimits, GgufType, GgufWeights, MetaValue, dequantize, dequantize_into};
use crate::EngError;

#[cfg(all(feature = "native-cuda", target_os = "linux"))]
mod device;
mod forward;

#[cfg(all(feature = "native-cuda", target_os = "linux"))]
pub(crate) use device::DeviceDraft;

const LIMITS: GgufLimits = GgufLimits {
    max_header_bytes: 32 * 1024 * 1024,
    max_tensor_bytes: 1024 * 1024 * 1024,
};

#[derive(Debug)]
pub(crate) struct DsparkConfig {
    pub(crate) width: usize,
    pub(crate) markov_rank: usize,
    pub(crate) vocab: usize,
    pub(crate) padded_vocab: usize,
    pub(crate) block_size: usize,
    pub(crate) target_layers: Vec<usize>,
    pub(crate) context_length: usize,
    pub(crate) final_logit_softcap: f32,
    feed_forward: usize,
    query_heads: usize,
    kv_heads: usize,
    head_width: usize,
    rope_theta: f32,
    epsilon: f32,
    sliding_window: usize,
    local_layers: Vec<bool>,
}

impl DsparkConfig {
    /// The reference holds the mixed context and its projected K/V for one
    /// draft layer at a time. Bound that peak separately from target state.
    pub(crate) fn workspace_bytes(&self, target_context: usize) -> Result<u64, EngError> {
        let context = target_context
            .saturating_sub(1)
            .min(self.context_length - self.block_size) as u128;
        let width = self.width as u128;
        let query = self.query_heads as u128 * self.head_width as u128;
        let kv = self.kv_heads as u128 * self.head_width as u128;
        let context_values = context * (width + 2 * kv + 2);
        let block_values = self.block_size as u128
            * (8 * width + 6 * query + 6 * kv + 4 * self.feed_forward as u128);
        let logits_and_head =
            2 * self.padded_vocab as u128 + 6 * self.markov_rank as u128 + 4 * width;
        let values = context_values + block_values + logits_and_head;
        let row_handles = 4 * (context + self.block_size as u128) * size_of::<Vec<f32>>() as u128;
        u64::try_from(values * size_of::<f32>() as u128 + row_handles)
            .map_err(|_| EngError::MemoryPlan("DSpark draft workspace size overflows".into()))
    }

    fn read(weights: &GgufWeights) -> Result<Self, EngError> {
        let meta = Metadata(weights);
        if !matches!(
            weights.header().meta("general.architecture"),
            Some(MetaValue::String(arch)) if arch == "dflash"
        ) {
            return Err(checkpoint("expected general.architecture=dflash"));
        }

        for key in [
            "dflash.sample_from_anchor",
            "dflash.has_confidence_head",
            "dflash.attention.causal",
        ] {
            if !matches!(weights.header().meta(key), Some(MetaValue::Bool(false))) {
                return Err(checkpoint(format!(
                    "{key} must be false for dense recurrent DSpark"
                )));
            }
        }

        let block_count = meta.size("dflash.block_count")?;
        let local_layers = meta.array("dflash.attention.sliding_window_pattern", 7, 1)?;
        if local_layers.len() != block_count {
            return Err(checkpoint(
                "sliding window pattern does not match draft layer count",
            ));
        }
        let visits = meta.array("dflash.target_layers", 5, 4)?;
        let target_layers = visits
            .as_chunks::<4>()
            .0
            .iter()
            .map(|bytes| usize::try_from(i32::from_le_bytes(*bytes)))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| checkpoint("target layer visit must be nonnegative"))?;
        if target_layers.is_empty() || target_layers.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(checkpoint(
                "target layers must be nonempty and in visit order",
            ));
        }

        let config = Self {
            width: meta.size("dflash.embedding_length")?,
            markov_rank: meta.size("dflash.dspark.markov_rank")?,
            vocab: meta.size("dflash.vocab_size_unpadded")?,
            padded_vocab: meta.size("dflash.vocab_size")?,
            block_size: meta.size("dflash.block_size")?,
            target_layers,
            context_length: meta.size("dflash.context_length")?,
            final_logit_softcap: meta.positive_float("dflash.final_logit_softcapping")?,
            feed_forward: meta.size("dflash.feed_forward_length")?,
            query_heads: meta.size("dflash.attention.head_count")?,
            kv_heads: meta.size("dflash.attention.head_count_kv")?,
            head_width: meta.size("dflash.attention.key_length")?,
            rope_theta: meta.positive_float("dflash.rope.freq_base")?,
            epsilon: meta.positive_float("dflash.attention.layer_norm_rms_epsilon")?,
            sliding_window: meta.size("dflash.attention.sliding_window")?,
            local_layers: local_layers.into_iter().map(|flag| flag != 0).collect(),
        };
        config.validate(&meta)?;
        Ok(config)
    }

    fn validate(&self, meta: &Metadata<'_>) -> Result<(), EngError> {
        if self.vocab > self.padded_vocab
            || self.block_size < 2
            || self.block_size > self.context_length
        {
            return Err(checkpoint("invalid vocabulary or draft block dimensions"));
        }
        if self.head_width != meta.size("dflash.attention.value_length")?
            || self.head_width != meta.size("dflash.rope.dimension_count")?
            || !self.head_width.is_multiple_of(2)
            || !self.query_heads.is_multiple_of(self.kv_heads)
            || self.query_heads.checked_mul(self.head_width).is_none()
        {
            return Err(checkpoint("invalid grouped attention or rotary dimensions"));
        }
        if self
            .markov_rank
            .checked_add(self.width)
            .and_then(|sum| sum.checked_mul(2))
            .is_none()
            || self.markov_rank.checked_mul(3).is_none()
            || self.width.checked_mul(self.target_layers.len()).is_none()
        {
            return Err(checkpoint("draft dimensions exceed host limits"));
        }
        Ok(())
    }
}

struct Metadata<'a>(&'a GgufWeights);

impl Metadata<'_> {
    fn size(&self, key: &str) -> Result<usize, EngError> {
        match self.0.header().meta(key) {
            Some(MetaValue::U32(value)) if *value != 0 => Ok(*value as usize),
            _ => Err(checkpoint(format!("{key} must be a positive GGUF uint32"))),
        }
    }

    fn positive_float(&self, key: &str) -> Result<f32, EngError> {
        match self.0.header().meta(key) {
            Some(MetaValue::F32(value)) if value.is_finite() && *value > 0.0 => Ok(*value),
            _ => Err(checkpoint(format!(
                "{key} must be a finite positive GGUF float32"
            ))),
        }
    }

    fn array(&self, key: &str, elem_type: u32, elem_bytes: usize) -> Result<Vec<u8>, EngError> {
        let Some(MetaValue::Array(array)) = self.0.header().meta(key) else {
            return Err(checkpoint(format!("{key} must be a GGUF array")));
        };
        // These arrays index the actual draft or target layers, never the tokenizer.
        let max_entries = self.0.header().tensors().len();
        if array.elem_type != elem_type || array.len > max_entries as u64 {
            return Err(checkpoint(format!(
                "{key} has an invalid element type or count"
            )));
        }
        let len = array.len as usize * elem_bytes;
        let mut bytes = vec![0; len];
        self.0
            .read_header_range(array.byte_range.start, &mut bytes)?;
        Ok(bytes)
    }
}

#[derive(Debug)]
pub(crate) struct DsparkBlock {
    /// Normalized draft rows for the mask slots. Slot zero is the anchor.
    pub(crate) hidden: Vec<Vec<f32>>,
    /// The recurrent head's feature mix from the most recent committed token.
    pub(crate) recurrent_context: Vec<f32>,
}

#[derive(Debug)]
pub(crate) struct DsparkGguf {
    config: DsparkConfig,
    mask_embedding: Vec<f32>,
    fusion: Vec<Vec<f32>>,
    output_norm: Vec<f32>,
    recurrent_norm: Vec<f32>,
    layers: Vec<DraftLayer>,
    markov: DsparkMarkovWeights,
}

impl DsparkGguf {
    pub(crate) fn open(path: &Path) -> Result<Self, EngError> {
        let weights = GgufWeights::open_any_type(path, LIMITS, None)?;
        let total = weights
            .header()
            .tensors()
            .iter()
            .try_fold(0u64, |total, tensor| total.checked_add(tensor.byte_len()));
        if total.is_none_or(|bytes| bytes > LIMITS.max_tensor_bytes) {
            return Err(EngError::MemoryPlan(
                "DSpark GGUF exceeds the 1 GiB weight limit".into(),
            ));
        }
        let config = DsparkConfig::read(&weights)?;
        let load = TensorLoader(&weights);
        let width = config.width;
        let rank = config.markov_rank;
        let taps = config.target_layers.len();
        let count = config.local_layers.len();
        let fusion = load.f32("dspark_fuse.weight", &[taps, count + 1])?;
        let fusion = fusion.chunks_exact(taps).map(forward::softmax).collect();
        let layers = (0..count)
            .map(|index| DraftLayer::read(&load, &config, index))
            .collect::<Result<_, _>>()?;
        Ok(Self {
            mask_embedding: load.f32("mask_embd.weight", &[width])?,
            output_norm: load.f32("output_norm.weight", &[width])?,
            recurrent_norm: load.f32("dspark_norm.weight", &[width])?,
            markov: DsparkMarkovWeights {
                previous: load.matrix(
                    "markov_w1.weight",
                    rank,
                    config.padded_vocab,
                    &[GgufType::Bf16],
                )?,
                joint: load.matrix(
                    "joint_proj.weight",
                    2 * (rank + width),
                    3 * rank,
                    &[GgufType::Bf16],
                )?,
                joint_bias: load.f32("joint_proj.bias", &[3 * rank])?,
                output: load.matrix("dspark_hidden_out.weight", rank, width, &[GgufType::Bf16])?,
                output_bias: load.f32("dspark_hidden_out.bias", &[width])?,
            },
            fusion,
            layers,
            config,
        })
    }

    pub(crate) fn config(&self) -> &DsparkConfig {
        &self.config
    }

    pub(crate) fn markov(&self) -> &DsparkMarkovWeights {
        &self.markov
    }
}

#[derive(Debug)]
pub(crate) struct DsparkMarkovWeights {
    pub(crate) previous: DsparkMatrix,
    pub(crate) joint: DsparkMatrix,
    pub(crate) joint_bias: Vec<f32>,
    pub(crate) output: DsparkMatrix,
    pub(crate) output_bias: Vec<f32>,
}

#[derive(Debug)]
struct DraftLayer {
    query: DsparkMatrix,
    key: DsparkMatrix,
    value: DsparkMatrix,
    attention_output: DsparkMatrix,
    attention_norm: Vec<f32>,
    query_norm: Vec<f32>,
    key_norm: Vec<f32>,
    context_key: DsparkMatrix,
    context_value: DsparkMatrix,
    context_key_norm: Vec<f32>,
    context_norm: Vec<f32>,
    ffn_gate: DsparkMatrix,
    ffn_up: DsparkMatrix,
    ffn_down: DsparkMatrix,
    ffn_norm: Vec<f32>,
}

impl DraftLayer {
    fn read(
        load: &TensorLoader<'_>,
        config: &DsparkConfig,
        index: usize,
    ) -> Result<Self, EngError> {
        let width = config.width;
        let heads = config.head_width;
        let query_width = config.query_heads * heads;
        let kv_width = config.kv_heads * heads;
        let ffn = config.feed_forward;
        let name = |suffix| format!("blk.{index}.{suffix}.weight");
        let matrix = |suffix, inputs, outputs| {
            load.matrix(
                &name(suffix),
                inputs,
                outputs,
                &[GgufType::Q80, GgufType::Bf16],
            )
        };
        let norm = |suffix, length| load.f32(&name(suffix), &[length]);
        Ok(Self {
            query: matrix("attn_q", width, query_width)?,
            key: matrix("attn_k", width, kv_width)?,
            value: matrix("attn_v", width, kv_width)?,
            attention_output: matrix("attn_output", query_width, width)?,
            attention_norm: norm("attn_norm", width)?,
            query_norm: norm("attn_q_norm", heads)?,
            key_norm: norm("attn_k_norm", heads)?,
            context_key: matrix("attn_k_ctx", width, kv_width)?,
            context_value: matrix("attn_v_ctx", width, kv_width)?,
            context_key_norm: norm("attn_k_ctx_norm", heads)?,
            context_norm: norm("dflare_norm", width)?,
            ffn_gate: matrix("ffn_gate", width, ffn)?,
            ffn_up: matrix("ffn_up", width, ffn)?,
            ffn_down: matrix("ffn_down", ffn, width)?,
            ffn_norm: norm("ffn_norm", width)?,
        })
    }
}

struct TensorLoader<'a>(&'a GgufWeights);

impl TensorLoader<'_> {
    fn f32(&self, name: &str, dims: &[usize]) -> Result<Vec<f32>, EngError> {
        let (ty, bytes) = self.read(name, dims, &[GgufType::F32])?;
        let values = dequantize(ty, &bytes)?;
        if values.iter().any(|value| !value.is_finite()) {
            return Err(checkpoint(format!("{name} contains nonfinite values")));
        }
        Ok(values)
    }

    fn matrix(
        &self,
        name: &str,
        inputs: usize,
        outputs: usize,
        types: &[GgufType],
    ) -> Result<DsparkMatrix, EngError> {
        let (ty, bytes) = self.read(name, &[inputs, outputs], types)?;
        let row_bytes = inputs / ty.block_elems() as usize * ty.block_bytes() as usize;
        Ok(DsparkMatrix {
            ty,
            inputs,
            row_bytes,
            bytes,
        })
    }

    fn read(
        &self,
        name: &str,
        dims: &[usize],
        types: &[GgufType],
    ) -> Result<(GgufType, Vec<u8>), EngError> {
        let tensor = self
            .0
            .header()
            .tensor(name)
            .ok_or_else(|| checkpoint(format!("missing tensor {name}")))?;
        if !tensor
            .dims()
            .iter()
            .copied()
            .eq(dims.iter().map(|&dim| dim as u64))
        {
            return Err(checkpoint(format!(
                "{name} has shape {:?}, expected {dims:?}",
                tensor.dims()
            )));
        }
        let ty = tensor.ggml_type();
        if !types.contains(&ty) {
            return Err(checkpoint(format!(
                "{name} has unsupported type {}",
                ty.name()
            )));
        }
        self.0.read_tensor(name)
    }
}

/// Retain packed matrices and decode each row once per block projection.
#[derive(Debug)]
pub(crate) struct DsparkMatrix {
    ty: GgufType,
    inputs: usize,
    row_bytes: usize,
    bytes: Vec<u8>,
}

impl DsparkMatrix {
    pub(crate) fn row(&self, index: usize) -> Vec<f32> {
        let bytes = &self.bytes[index * self.row_bytes..][..self.row_bytes];
        let mut result = vec![0.0; self.inputs];
        self.decode(bytes, &mut result);
        result
    }

    pub(crate) fn project(&self, input: &[f32]) -> Vec<f32> {
        self.project_batch(&[input])
            .pop()
            .expect("one projection input")
    }

    fn project_batch<T: AsRef<[f32]>>(&self, inputs: &[T]) -> Vec<Vec<f32>> {
        let outputs = self.bytes.len() / self.row_bytes;
        let mut result = vec![vec![0.0; outputs]; inputs.len()];
        let mut row = vec![0.0; self.inputs];
        for (index, bytes) in self.bytes.chunks_exact(self.row_bytes).enumerate() {
            self.decode(bytes, &mut row);
            for (input, output) in inputs.iter().zip(&mut result) {
                debug_assert_eq!(self.inputs, input.as_ref().len());
                output[index] = forward::dot(&row, input.as_ref());
            }
        }
        result
    }

    fn decode(&self, bytes: &[u8], output: &mut [f32]) {
        dequantize_into(self.ty, bytes, output).expect("GGUF tensor type and row shape validated");
    }
}

fn checkpoint(message: impl std::fmt::Display) -> EngError {
    EngError::Checkpoint(format!("DSpark GGUF: {message}"))
}

#[cfg(test)]
mod tests;
