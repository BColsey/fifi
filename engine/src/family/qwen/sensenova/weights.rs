//! Checked GGUF weight views for SenseNova Loop.
//!
//! Matrices stay in the file as [`TensorRef`] handles and are projected
//! through the shared [`GgufProjector`], which dequantizes rows on demand. The small
//! per-head vectors (conv taps, `ssm_a`, `dt.bias`, norms, router biases)
//! load as F32 at open. Fused tensors keep the vendor layout: the
//! `attn_qkv_gate` matrix is `[q | k | v | z]`, `ssm_beta_alpha` is
//! `[beta | alpha]`, and the full-attention `attn_q` packs each head as
//! `[query | gate]` (hence `2 * head_dim * heads` rows).

use crate::{
    EngError,
    family::project::gguf::{GgufMatrix as TensorRef, GgufProjector},
    weights::{GgufHeader, GgufWeights, MetaValue, dequantize_into, is_plain_float},
};

use super::{ARCHITECTURE, LoopPlan, RopePlan};

/// Model dimensions of the SenseNova Loop 9B checkpoint.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct LoopShape {
    pub(crate) hidden: usize,
    pub(crate) feed_forward: usize,
    pub(crate) heads_q: usize,
    pub(crate) heads_kv: usize,
    pub(crate) head_dim: usize,
    /// GDN key heads (`ssm.group_count`); key dim is `ssm.state_size`.
    pub(crate) delta_key_heads: usize,
    pub(crate) delta_key_dim: usize,
    /// GDN value heads (`ssm.time_step_rank`); value dim is `inner / heads`.
    pub(crate) delta_value_heads: usize,
    pub(crate) delta_value_dim: usize,
    pub(crate) conv_kernel: usize,
    pub(crate) vocab: usize,
    /// `attention.context_length`: the recurrent state has no trim, so the
    /// whole committed history must stay under it.
    pub(crate) context: usize,
    pub(crate) eps: f32,
}

impl LoopShape {
    pub(crate) fn delta_inner(&self) -> usize {
        self.delta_value_heads * self.delta_value_dim
    }
    /// Conv channels `[q | k | v]` of the fused `attn_qkv_gate` projection.
    pub(crate) fn conv_dim(&self) -> usize {
        2 * self.delta_key_heads * self.delta_key_dim + self.delta_inner()
    }
    /// Output rows of `attn_qkv_gate`: conv channels plus the `z` gate.
    pub(crate) fn qkvz_dim(&self) -> usize {
        self.conv_dim() + self.delta_inner()
    }
}

pub(crate) struct LoopWeights {
    gguf: std::sync::Arc<GgufWeights>,
    shape: LoopShape,
    plan: LoopPlan,
    rope: RopePlan,
    layers: Vec<Layer>,
    token_embd: TensorRef,
    output_norm: Vec<f32>,
    output: TensorRef,
}

/// Router logits of one mHC layer: weights over `[a || b]` plus a bias.
#[derive(Debug, Clone)]
pub(crate) struct Router {
    pub(crate) weight: TensorRef,
    pub(crate) bias: Vec<f32>,
}

pub(crate) enum Mixer {
    Delta {
        qkvz: TensorRef,
        beta_alpha: TensorRef,
        out: TensorRef,
        conv: Vec<f32>,
        a: Vec<f32>,
        dt_bias: Vec<f32>,
        norm: Vec<f32>,
    },
    Attn {
        q: TensorRef,
        k: TensorRef,
        v: TensorRef,
        o: TensorRef,
        q_norm: Vec<f32>,
        k_norm: Vec<f32>,
    },
}

pub(crate) struct Layer {
    pub(crate) attn_norm: Vec<f32>,
    pub(crate) post_norm: Vec<f32>,
    pub(crate) mixer: Mixer,
    pub(crate) ffn_gate: TensorRef,
    pub(crate) ffn_up: TensorRef,
    pub(crate) ffn_down: TensorRef,
    pub(crate) mhc_attn: Option<Router>,
    pub(crate) mhc_ffn: Option<Router>,
}

impl LoopWeights {
    pub(crate) fn open(weights: std::sync::Arc<GgufWeights>) -> Result<Self, EngError> {
        let header = weights.header();
        let arch = match header.meta("general.architecture") {
            Some(MetaValue::String(name)) if name == ARCHITECTURE => ARCHITECTURE,
            _ => {
                return Err(EngError::Checkpoint(
                    "sensenova: general.architecture must be sensenova_loop".into(),
                ));
            }
        };
        let shape = shape_from_header(header, arch)?;
        let plan = LoopPlan::from_header(&weights, header, arch)?;
        let rope = RopePlan::from_header(&weights, header, arch)?;
        if rope.dims > shape.head_dim {
            return Err(reject("rotary dimensions exceed the attention head"));
        }
        let mut layers = Vec::with_capacity(super::PHYSICAL_LAYERS);
        for p in 0..super::PHYSICAL_LAYERS {
            layers.push(layer(&weights, header, &shape, &plan, p)?);
        }
        let gguf = std::sync::Arc::clone(&weights);
        Ok(Self {
            gguf,
            shape,
            plan,
            rope,
            layers,
            token_embd: matrix(&weights, "token_embd.weight", shape.vocab, shape.hidden)?,
            output_norm: f32_vector(&weights, header, "output_norm.weight", shape.hidden)?,
            output: matrix(&weights, "output.weight", shape.vocab, shape.hidden)?,
        })
    }

    /// A projector whose scratch buffers outlive one token.
    pub(crate) fn projector(&self) -> GgufProjector {
        GgufProjector::new(std::sync::Arc::clone(&self.gguf))
    }

    pub(crate) fn shape(&self) -> &LoopShape {
        &self.shape
    }
    pub(crate) fn plan(&self) -> &LoopPlan {
        &self.plan
    }
    pub(crate) fn rope(&self) -> &RopePlan {
        &self.rope
    }
    pub(crate) fn layers(&self) -> &[Layer] {
        &self.layers
    }
    pub(crate) fn output_norm(&self) -> &[f32] {
        &self.output_norm
    }
    pub(crate) fn token_embd(&self) -> &TensorRef {
        &self.token_embd
    }
    pub(crate) fn output(&self) -> &TensorRef {
        &self.output
    }
}

fn shape_from_header(header: &GgufHeader, arch: &str) -> Result<LoopShape, EngError> {
    let u32_of = |key: &str| -> Result<usize, EngError> {
        match header.meta(key) {
            Some(MetaValue::U32(v)) => Ok(*v as usize),
            _ => Err(reject(format!(
                "metadata key {key} is missing or not a u32"
            ))),
        }
    };
    let f32_of = |key: &str| -> Result<f32, EngError> {
        match header.meta(key) {
            Some(MetaValue::F32(v)) => Ok(*v),
            _ => Err(reject(format!(
                "metadata key {key} is missing or not an f32"
            ))),
        }
    };
    let head_dim = u32_of(&format!("{arch}.attention.key_length"))?;
    let value_heads = u32_of(&format!("{arch}.ssm.time_step_rank"))?;
    let key_heads = u32_of(&format!("{arch}.ssm.group_count"))?;
    let inner = u32_of(&format!("{arch}.ssm.inner_size"))?;
    if value_heads == 0
        || key_heads == 0
        || inner == 0
        || !inner.is_multiple_of(value_heads)
        || !value_heads.is_multiple_of(key_heads)
    {
        return Err(reject("ssm value heads do not divide evenly"));
    }
    let vocab = match header.meta("tokenizer.ggml.tokens") {
        Some(MetaValue::Array(meta)) => meta.len as usize,
        _ => return Err(reject("tokenizer.ggml.tokens is missing")),
    };
    let context = match header.meta(&format!("{arch}.context_length")) {
        Some(MetaValue::U32(v)) => *v as usize,
        _ => return Err(reject("attention.context_length is missing or not a u32")),
    };
    let shape = LoopShape {
        hidden: u32_of(&format!("{arch}.embedding_length"))?,
        feed_forward: u32_of(&format!("{arch}.feed_forward_length"))?,
        heads_q: u32_of(&format!("{arch}.attention.head_count"))?,
        heads_kv: u32_of(&format!("{arch}.attention.head_count_kv"))?,
        head_dim,
        delta_key_heads: key_heads,
        delta_key_dim: u32_of(&format!("{arch}.ssm.state_size"))?,
        delta_value_heads: value_heads,
        delta_value_dim: inner / value_heads,
        conv_kernel: u32_of(&format!("{arch}.ssm.conv_kernel"))?,
        vocab,
        context,
        eps: f32_of(&format!("{arch}.attention.layer_norm_rms_epsilon"))?,
    };
    // This recipe is for the 9B checkpoint. Bound the recurrent and KV allocations
    // before using header dimensions in products or as buffer sizes.
    if [
        shape.hidden,
        shape.feed_forward,
        shape.heads_q,
        shape.heads_kv,
        shape.head_dim,
        shape.delta_key_heads,
        shape.delta_key_dim,
        shape.delta_value_heads,
        shape.delta_value_dim,
        shape.conv_kernel,
        shape.vocab,
    ] != [4096, 12288, 16, 4, 256, 16, 128, 32, 128, 4, 248_320]
        || shape.context < 2
        || !shape.eps.is_finite()
        || shape.eps <= 0.0
    {
        return Err(reject("expected SenseNova Loop 9B geometry"));
    }
    if head_dim != u32_of(&format!("{arch}.attention.value_length"))? {
        return Err(reject("attention key and value lengths differ"));
    }
    Ok(shape)
}

fn layer(
    weights: &GgufWeights,
    header: &GgufHeader,
    shape: &LoopShape,
    plan: &LoopPlan,
    p: usize,
) -> Result<Layer, EngError> {
    let stem = format!("blk.{}", plan.block(p));
    let attn_norm = f32_vector(
        weights,
        header,
        &format!("{stem}.attn_norm.weight"),
        shape.hidden,
    )?;
    let post_norm = f32_vector(
        weights,
        header,
        &format!("{stem}.post_attention_norm.weight"),
        shape.hidden,
    )?;
    let mixer = if plan.recurrent[plan.block(p)] {
        Mixer::Delta {
            qkvz: matrix(
                weights,
                &format!("{stem}.attn_qkv_gate.weight"),
                shape.qkvz_dim(),
                shape.hidden,
            )?,
            beta_alpha: matrix(
                weights,
                &format!("{stem}.ssm_beta_alpha.weight"),
                2 * shape.delta_value_heads,
                shape.hidden,
            )?,
            out: matrix(
                weights,
                &format!("{stem}.ssm_out.weight"),
                shape.hidden,
                shape.delta_inner(),
            )?,
            conv: f32_tensor(
                weights,
                header,
                &format!("{stem}.ssm_conv1d.weight"),
                &[shape.conv_kernel as u64, shape.conv_dim() as u64],
            )?,
            a: f32_vector(
                weights,
                header,
                &format!("{stem}.ssm_a"),
                shape.delta_value_heads,
            )?,
            dt_bias: f32_vector(
                weights,
                header,
                &format!("{stem}.ssm_dt.bias"),
                shape.delta_value_heads,
            )?,
            norm: f32_vector(
                weights,
                header,
                &format!("{stem}.ssm_norm.weight"),
                shape.delta_value_dim,
            )?,
        }
    } else {
        let qk = shape.head_dim * shape.heads_q;
        let kv = shape.head_dim * shape.heads_kv;
        Mixer::Attn {
            q: matrix(
                weights,
                &format!("{stem}.attn_q.weight"),
                2 * qk,
                shape.hidden,
            )?,
            k: matrix(weights, &format!("{stem}.attn_k.weight"), kv, shape.hidden)?,
            v: matrix(weights, &format!("{stem}.attn_v.weight"), kv, shape.hidden)?,
            o: matrix(
                weights,
                &format!("{stem}.attn_output.weight"),
                shape.hidden,
                qk,
            )?,
            q_norm: f32_vector(
                weights,
                header,
                &format!("{stem}.attn_q_norm.weight"),
                shape.head_dim,
            )?,
            k_norm: f32_vector(
                weights,
                header,
                &format!("{stem}.attn_k_norm.weight"),
                shape.head_dim,
            )?,
        }
    };
    let mhc_router = |part: &str| -> Result<Option<Router>, EngError> {
        if !(plan.mhc_start..plan.mhc_end).contains(&p) {
            return Ok(None);
        }
        let routes = if plan.mhc_fixed_identity { 4 } else { 8 };
        Ok(Some(Router {
            weight: matrix(
                weights,
                &format!("{stem}.{part}.weight"),
                routes,
                2 * shape.hidden,
            )?,
            bias: f32_vector(weights, header, &format!("{stem}.{part}.bias"), routes)?,
        }))
    };
    Ok(Layer {
        attn_norm,
        post_norm,
        mixer,
        ffn_gate: matrix(
            weights,
            &format!("{stem}.ffn_gate.weight"),
            shape.feed_forward,
            shape.hidden,
        )?,
        ffn_up: matrix(
            weights,
            &format!("{stem}.ffn_up.weight"),
            shape.feed_forward,
            shape.hidden,
        )?,
        ffn_down: matrix(
            weights,
            &format!("{stem}.ffn_down.weight"),
            shape.hidden,
            shape.feed_forward,
        )?,
        mhc_attn: mhc_router("hc_attn_fn")?,
        mhc_ffn: mhc_router("hc_ffn_fn")?,
    })
}

fn f32_vector(
    weights: &GgufWeights,
    header: &GgufHeader,
    name: &str,
    len: usize,
) -> Result<Vec<f32>, EngError> {
    f32_tensor(weights, header, name, &[len as u64])
}

fn f32_tensor(
    weights: &GgufWeights,
    header: &GgufHeader,
    name: &str,
    dims: &[u64],
) -> Result<Vec<f32>, EngError> {
    let tensor = header
        .tensor(name)
        .ok_or_else(|| reject(format!("tensor {name} is missing")))?;
    if tensor.dims() != dims {
        return Err(reject(format!(
            "tensor {name} must have dimensions {dims:?}, found {:?}",
            tensor.dims()
        )));
    }
    if !is_plain_float(tensor.ggml_type()) {
        return Err(reject(format!(
            "tensor {name} must use floating-point storage"
        )));
    }
    let (ty, bytes) = weights.read_tensor(name)?;
    let mut out = vec![0.0; tensor.element_count() as usize];
    dequantize_into(ty, &bytes, &mut out)?;
    Ok(out)
}

fn matrix(
    weights: &GgufWeights,
    name: &str,
    rows: usize,
    cols: usize,
) -> Result<TensorRef, EngError> {
    TensorRef::new(weights.header(), name, rows, cols)
}

fn reject(message: impl Into<String>) -> EngError {
    EngError::Checkpoint(format!("sensenova: {}", message.into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn convolution_requires_taps_contiguous_within_each_channel() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("conv.gguf");
        let name = "blk.0.ssm_conv1d.weight";
        let mut bytes = b"GGUF".to_vec();
        bytes.extend_from_slice(&3_u32.to_le_bytes());
        bytes.extend_from_slice(&1_u64.to_le_bytes()); // tensor count
        bytes.extend_from_slice(&0_u64.to_le_bytes()); // metadata count
        bytes.extend_from_slice(&(name.len() as u64).to_le_bytes());
        bytes.extend_from_slice(name.as_bytes());
        bytes.extend_from_slice(&2_u32.to_le_bytes());
        bytes.extend_from_slice(&8_u64.to_le_bytes());
        bytes.extend_from_slice(&4_u64.to_le_bytes());
        bytes.extend_from_slice(&0_u32.to_le_bytes()); // F32
        bytes.extend_from_slice(&0_u64.to_le_bytes()); // tensor offset
        bytes.resize(bytes.len().next_multiple_of(32) + 32 * 4, 0);
        std::fs::write(&path, bytes).unwrap();
        let weights = GgufWeights::open(
            &path,
            crate::weights::GgufLimits {
                max_header_bytes: 1024,
                max_tensor_bytes: 1024,
            },
            None,
        )
        .unwrap();
        assert!(f32_tensor(&weights, weights.header(), name, &[8, 4]).is_ok());
        let error = f32_tensor(&weights, weights.header(), name, &[4, 8]).unwrap_err();
        assert!(error.describe().contains("must have dimensions [4, 8]"));
    }
}
