//! The checked dimensions of a model that mixes gated-DeltaNet and full-attention layers, read
//! from a [`FamilySpec`]: hidden width, vocabulary, the attention kind of every layer, the
//! DeltaNet, attention and mixture-of-experts sizes. Pure host arithmetic, shared by the host
//! reference forward, the card step plans and the settings.

use crate::EngError;
use crate::family::spec::{AttentionKind, FamilySpec, Router};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DeltaDims {
    pub(crate) key_heads: usize,
    pub(crate) key_dim: usize,
    pub(crate) value_heads: usize,
    pub(crate) value_dim: usize,
    pub(crate) conv_kernel: usize,
}

impl DeltaDims {
    pub(crate) fn conv_dim(&self) -> usize {
        2 * self.key_heads * self.key_dim + self.value_heads * self.value_dim
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct AttnDims {
    pub(crate) heads_q: usize,
    pub(crate) heads_kv: usize,
    pub(crate) head_dim: usize,
    /// Leading dimensions of each head that RoPE rotates.
    pub(crate) rotary_dim: usize,
    pub(crate) theta: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MoeDims {
    pub(crate) experts: usize,
    pub(crate) top_k: usize,
    pub(crate) inter: usize,
    pub(crate) shared_inter: usize,
    pub(crate) renorm: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct HostShape {
    pub(crate) hidden: usize,
    pub(crate) vocab: usize,
    /// One entry per layer: `Full` or `DeltaNet`.
    pub(crate) kinds: Vec<AttentionKind>,
    /// Epsilon of every learned-weight RMS norm.
    pub(crate) eps: f32,
    pub(crate) delta: DeltaDims,
    pub(crate) attn: AttnDims,
    /// Dense intermediate width, or `None` for MoE layers.
    pub(crate) dense_inter: Option<usize>,
    /// All four widths/counts are zero sentinels for dense layers. Dense loading never uses them.
    pub(crate) moe: MoeDims,
}

fn invalid(message: &str) -> EngError {
    EngError::Checkpoint(format!("host shape: {message}"))
}

pub(crate) fn to_usize(value: u32) -> usize {
    // `usize` holds a `u32` on every supported target.
    usize::try_from(value).unwrap_or(usize::MAX)
}

impl HostShape {
    pub(crate) fn from_spec(spec: &FamilySpec, rotary_dim: u32) -> Result<Self, EngError> {
        let linear = spec
            .linear
            .ok_or_else(|| invalid("the family has no linear attention dimensions"))?;
        let Router::SoftmaxTopK { renorm } = spec.router else {
            return Err(invalid("the family router is not softmax top-k"));
        };
        let shape = Self {
            hidden: to_usize(spec.hidden),
            vocab: to_usize(spec.vocab),
            kinds: spec.layers_attention.to_vec(),
            eps: spec.rms_norm_eps,
            delta: DeltaDims {
                key_heads: to_usize(linear.key_heads),
                key_dim: to_usize(linear.key_head_dim),
                value_heads: to_usize(linear.value_heads),
                value_dim: to_usize(linear.value_head_dim),
                conv_kernel: to_usize(linear.conv_kernel),
            },
            attn: AttnDims {
                heads_q: to_usize(spec.heads_q),
                heads_kv: to_usize(spec.heads_kv),
                head_dim: to_usize(spec.head_dim),
                rotary_dim: to_usize(rotary_dim),
                theta: spec.rope_theta,
            },
            dense_inter: spec.feed_forward.map(|dense| to_usize(dense.intermediate)),
            moe: MoeDims {
                experts: to_usize(spec.experts),
                top_k: to_usize(spec.top_k),
                inter: to_usize(spec.moe_inter),
                shared_inter: to_usize(spec.shared_inter),
                renorm,
            },
        };
        shape.check()?;
        Ok(shape)
    }

    pub(crate) fn check(&self) -> Result<(), EngError> {
        let d = &self.delta;
        let a = &self.attn;
        let m = &self.moe;
        let positive = [
            self.hidden,
            self.vocab,
            d.key_heads,
            d.key_dim,
            d.value_heads,
            d.value_dim,
            a.heads_q,
            a.heads_kv,
            a.head_dim,
        ];
        if self.kinds.is_empty() || positive.contains(&0) {
            return Err(invalid("a dimension is zero"));
        }
        let moe_dims = [m.experts, m.top_k, m.inter, m.shared_inter];
        match self.dense_inter {
            Some(0) => return Err(invalid("dense intermediate dimension is zero")),
            Some(_) if moe_dims != [0; 4] => {
                return Err(invalid("dense layers require zero MoE dimensions"));
            }
            None if moe_dims.contains(&0) => return Err(invalid("a dimension is zero")),
            _ => {}
        }
        if self.kinds.contains(&AttentionKind::Qsa) {
            return Err(invalid("the stepping stone has no QSA layer"));
        }
        if d.conv_kernel < 2 || !d.value_heads.is_multiple_of(d.key_heads) {
            return Err(invalid("DeltaNet kernel or head ratio is invalid"));
        }
        if !a.heads_q.is_multiple_of(a.heads_kv) {
            return Err(invalid("query heads must divide by key-value heads"));
        }
        if a.rotary_dim == 0 || !a.rotary_dim.is_multiple_of(2) || a.rotary_dim > a.head_dim {
            return Err(invalid("rotary dimension must be even and within the head"));
        }
        if !a.theta.is_finite() || a.theta <= 0.0 || !self.eps.is_finite() || self.eps < 0.0 {
            return Err(invalid("theta or epsilon is invalid"));
        }
        if m.top_k > m.experts {
            return Err(invalid("top-k exceeds the expert count"));
        }
        Ok(())
    }
}
