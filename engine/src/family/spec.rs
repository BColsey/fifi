use crate::EngError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Act {
    Silu,
    GeluApprox,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Router {
    SoftmaxTopK {
        renorm: bool,
    },
    /// Ungrouped sigmoid scores, normalized over the selected experts.
    SigmoidTopK,
    SigmoidGroup {
        groups: u32,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionKind {
    Full,
    /// Windowed attention over the last `FamilySpec::sliding_window` tokens. The
    /// layer stores its own K/V, like `Full`; only the reachable context differs.
    Sliding,
    DeltaNet,
    Qsa,
}

/// Gated delta recurrence dimensions read from the checkpoint configuration.
///
/// The fields mirror `linear_num_key_heads`, `linear_key_head_dim`,
/// `linear_num_value_heads`, `linear_value_head_dim`, and `linear_conv_kernel_dim`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LinearAttention {
    pub key_heads: u32,
    pub key_head_dim: u32,
    pub value_heads: u32,
    pub value_head_dim: u32,
    pub conv_kernel: u32,
}

impl LinearAttention {
    pub fn key_dim(&self) -> Result<u32, EngError> {
        checked_product(self.key_heads, self.key_head_dim, "key dimension")
    }

    pub fn value_dim(&self) -> Result<u32, EngError> {
        checked_product(self.value_heads, self.value_head_dim, "value dimension")
    }

    /// Channel count of the fused query, key, and value projection and its convolution.
    pub fn conv_dim(&self) -> Result<u32, EngError> {
        let keys = checked_product(self.key_dim()?, 2, "convolution dimension")?;
        keys.checked_add(self.value_dim()?)
            .ok_or_else(|| overflow("convolution dimension"))
    }

    pub(crate) fn validate(&self) -> Result<(), EngError> {
        for (field, value) in [
            ("key heads", self.key_heads),
            ("key head dimension", self.key_head_dim),
            ("value heads", self.value_heads),
            ("value head dimension", self.value_head_dim),
        ] {
            if value == 0 {
                return Err(EngError::Checkpoint(format!(
                    "linear attention {field} must exceed zero"
                )));
            }
        }
        // The convolution state keeps kernel - 1 positions. A kernel of 1 would make it empty.
        if self.conv_kernel < 2 {
            return Err(EngError::Checkpoint(format!(
                "linear attention convolution kernel {} must be at least 2",
                self.conv_kernel
            )));
        }
        if !self.value_heads.is_multiple_of(self.key_heads) {
            return Err(EngError::Checkpoint(format!(
                "linear attention value heads {} must divide by key heads {}",
                self.value_heads, self.key_heads
            )));
        }
        self.conv_dim().map(|_| ())
    }
}

fn checked_product(a: u32, b: u32, what: &str) -> Result<u32, EngError> {
    a.checked_mul(b).ok_or_else(|| overflow(what))
}

fn overflow(what: &str) -> EngError {
    EngError::Checkpoint(format!("linear attention {what} overflows u32"))
}

/// Hyper-connection widening of the residual stream.
///
/// The fields mirror `hc_count` and `hc_lowrank` in the checkpoint configuration.
/// The widened residual holds `branches` streams of `hidden` values each, so its
/// width is `branches * hidden`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HyperConnection {
    pub branches: u32,
    pub rank: u32,
}

impl HyperConnection {
    pub fn width(&self, hidden: u32) -> Result<u32, EngError> {
        self.branches
            .checked_mul(hidden)
            .ok_or_else(|| EngError::Checkpoint("hyper-connection width overflows u32".into()))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DenseMlp {
    pub intermediate: u32,
    pub context_limit: u32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FamilySpec {
    pub name: &'static str,
    pub layers: u32,
    pub hidden: u32,
    pub heads_q: u32,
    pub heads_kv: u32,
    pub head_dim: u32,
    /// Head dimension of `AttentionKind::Full` layers. Zero selects `head_dim`.
    pub global_head_dim: u32,
    pub rope_theta: f32,
    pub rms_norm_eps: f32,
    /// Accepted `model_type` values of the configuration, at the root and in `text_config`.
    pub model_types: &'static [&'static str],
    pub vocab: u32,
    /// `Some` selects dense feed-forward layers and requires all four MoE dimensions to be zero.
    /// `None` preserves the MoE layout and its existing dimensions.
    pub feed_forward: Option<DenseMlp>,
    pub experts: u32,
    pub top_k: u32,
    pub moe_inter: u32,
    pub shared_inter: u32,
    /// Affine bit width the checkpoint must declare. Zero, with `quant_group` zero, means the
    /// family fixes no width and takes it from the pack's `quantization` block.
    pub quant_bits: u32,
    /// Affine group size the checkpoint must declare. Zero, with `quant_bits` zero, means the
    /// pack declares it.
    pub quant_group: u32,
    pub act: Act,
    pub router: Router,
    pub mtp: bool,
    pub layers_attention: &'static [AttentionKind],
    /// Reachable context of `AttentionKind::Sliding` layers. Zero when the family
    /// has no sliding layers.
    pub sliding_window: u32,
    /// Count of trailing layers that store no K/V of their own and instead attend
    /// into the cache slot of the last physical layer of the same attention kind.
    /// Zero when every layer stores its own K/V.
    pub kv_shared_layers: u32,
    /// Required when any layer uses `AttentionKind::DeltaNet`.
    pub linear: Option<LinearAttention>,
    pub hyper: Option<HyperConnection>,
}

impl FamilySpec {
    /// Whether the family leaves the affine bit width and group size to the pack
    /// (`quant_bits` and `quant_group` both zero).
    pub fn reads_quantization_from_pack(&self) -> bool {
        self.quant_bits == 0 && self.quant_group == 0
    }

    /// Index of the first layer that stores no K/V of its own. Layers at or above
    /// it attend into the cache slot a physical source layer wrote. Saturating so
    /// an unvalidated descriptor (more shared layers than layers) cannot underflow.
    pub fn first_shared_layer(&self) -> u32 {
        self.layers.saturating_sub(self.kv_shared_layers)
    }

    /// The head dimension of a `Full` layer: `global_head_dim` when non-zero,
    /// otherwise `head_dim`. The single resolver of the zero-selects-head_dim rule.
    pub fn full_head_dim(&self) -> u32 {
        if self.global_head_dim != 0 {
            self.global_head_dim
        } else {
            self.head_dim
        }
    }

    /// The physical layer whose K/V a shared layer attends into: the last layer
    /// before `first_shared_layer` with the same attention kind. Safe on an
    /// unvalidated descriptor: every index is bounds-checked.
    pub fn shared_kv_source(&self, layer: u32) -> Result<u32, EngError> {
        let first_shared = self.first_shared_layer();
        if layer < first_shared || layer >= self.layers {
            return Err(EngError::Checkpoint(format!(
                "family {} layer {layer} is not a shared-K/V layer",
                self.name
            )));
        }
        let kind = *self.layers_attention.get(layer as usize).ok_or_else(|| {
            EngError::Checkpoint(format!(
                "family {} layer {layer} is outside the attention table",
                self.name
            ))
        })?;
        self.layers_attention
            .get(..first_shared as usize)
            .unwrap_or_default()
            .iter()
            .rposition(|candidate| *candidate == kind)
            .map(|index| index as u32)
            .ok_or_else(|| {
                EngError::Checkpoint(format!(
                    "family {} shared layer {layer} has no physical {kind:?} source",
                    self.name
                ))
            })
    }
}

const QWEN3_6_35B_A3B_LAYERS: [AttentionKind; 40] = [
    AttentionKind::DeltaNet,
    AttentionKind::DeltaNet,
    AttentionKind::DeltaNet,
    AttentionKind::Full,
    AttentionKind::DeltaNet,
    AttentionKind::DeltaNet,
    AttentionKind::DeltaNet,
    AttentionKind::Full,
    AttentionKind::DeltaNet,
    AttentionKind::DeltaNet,
    AttentionKind::DeltaNet,
    AttentionKind::Full,
    AttentionKind::DeltaNet,
    AttentionKind::DeltaNet,
    AttentionKind::DeltaNet,
    AttentionKind::Full,
    AttentionKind::DeltaNet,
    AttentionKind::DeltaNet,
    AttentionKind::DeltaNet,
    AttentionKind::Full,
    AttentionKind::DeltaNet,
    AttentionKind::DeltaNet,
    AttentionKind::DeltaNet,
    AttentionKind::Full,
    AttentionKind::DeltaNet,
    AttentionKind::DeltaNet,
    AttentionKind::DeltaNet,
    AttentionKind::Full,
    AttentionKind::DeltaNet,
    AttentionKind::DeltaNet,
    AttentionKind::DeltaNet,
    AttentionKind::Full,
    AttentionKind::DeltaNet,
    AttentionKind::DeltaNet,
    AttentionKind::DeltaNet,
    AttentionKind::Full,
    AttentionKind::DeltaNet,
    AttentionKind::DeltaNet,
    AttentionKind::DeltaNet,
    AttentionKind::Full,
];

pub const QWEN3_6_35B_A3B: FamilySpec = FamilySpec {
    name: "qwen3_6_35b_a3b",
    layers: 40,
    hidden: 2048,
    heads_q: 16,
    heads_kv: 2,
    head_dim: 256,
    global_head_dim: 0,
    rope_theta: 10_000_000.0,
    rms_norm_eps: 1e-6,
    model_types: &["qwen3_5_moe", "qwen3_5_moe_text"],
    vocab: 248_320,
    feed_forward: None,
    experts: 256,
    top_k: 8,
    moe_inter: 512,
    shared_inter: 512,
    quant_bits: 4,
    quant_group: 64,
    act: Act::Silu,
    router: Router::SoftmaxTopK { renorm: true },
    mtp: false,
    layers_attention: &QWEN3_6_35B_A3B_LAYERS,
    sliding_window: 0,
    kv_shared_layers: 0,
    linear: Some(LinearAttention {
        key_heads: 16,
        key_head_dim: 128,
        value_heads: 32,
        value_head_dim: 128,
        conv_kernel: 4,
    }),
    hyper: None,
};

const fn dense_27b_attention() -> [AttentionKind; 64] {
    let mut kinds = [AttentionKind::DeltaNet; 64];
    let mut index = 3;
    while index < kinds.len() {
        kinds[index] = AttentionKind::Full;
        index += 4;
    }
    kinds
}

const QWEN3_8_27B_DENSE_LAYERS: [AttentionKind; 64] = dense_27b_attention();

/// Portable descriptor for the Qwen3.8-27B checkpoint at any affine bit width and group size
/// the shared quantized matmul supports; the pack's `quantization` block declares them.
///
/// Attention uses output gating and untied embeddings, as checked by `check_config`.
/// The checkpoint also declares RoPE partial factor 0.25, `mrope_interleaved: true`,
/// and `mrope_section: [11, 11, 10]`. This descriptor does not implement multimodal RoPE.
pub const QWEN3_8_27B_DENSE: FamilySpec = FamilySpec {
    name: "qwen3_8_27b_dense",
    layers: 64,
    hidden: 5120,
    heads_q: 24,
    heads_kv: 4,
    head_dim: 256,
    global_head_dim: 0,
    rope_theta: 10_000_000.0,
    rms_norm_eps: 1e-6,
    model_types: &["qwen3_5", "qwen3_5_text"],
    vocab: 248_320,
    feed_forward: Some(DenseMlp {
        intermediate: 17_408,
        context_limit: 262_144,
    }),
    experts: 0,
    top_k: 0,
    moe_inter: 0,
    shared_inter: 0,
    quant_bits: 0,
    quant_group: 0,
    act: Act::Silu,
    // Dense layers do not use a router.
    router: Router::SoftmaxTopK { renorm: true },
    mtp: false,
    layers_attention: &QWEN3_8_27B_DENSE_LAYERS,
    sliding_window: 0,
    kv_shared_layers: 0,
    linear: Some(LinearAttention {
        key_heads: 16,
        key_head_dim: 128,
        value_heads: 48,
        value_head_dim: 128,
        conv_kernel: 4,
    }),
    hyper: None,
};

/// The twelve QSA (sparse full-attention) layers sit at indices 3, 7, ..., 47.
const QWEN3_8_FLASH_NEXT_LAYERS: [AttentionKind; 48] = [
    AttentionKind::DeltaNet,
    AttentionKind::DeltaNet,
    AttentionKind::DeltaNet,
    AttentionKind::Qsa,
    AttentionKind::DeltaNet,
    AttentionKind::DeltaNet,
    AttentionKind::DeltaNet,
    AttentionKind::Qsa,
    AttentionKind::DeltaNet,
    AttentionKind::DeltaNet,
    AttentionKind::DeltaNet,
    AttentionKind::Qsa,
    AttentionKind::DeltaNet,
    AttentionKind::DeltaNet,
    AttentionKind::DeltaNet,
    AttentionKind::Qsa,
    AttentionKind::DeltaNet,
    AttentionKind::DeltaNet,
    AttentionKind::DeltaNet,
    AttentionKind::Qsa,
    AttentionKind::DeltaNet,
    AttentionKind::DeltaNet,
    AttentionKind::DeltaNet,
    AttentionKind::Qsa,
    AttentionKind::DeltaNet,
    AttentionKind::DeltaNet,
    AttentionKind::DeltaNet,
    AttentionKind::Qsa,
    AttentionKind::DeltaNet,
    AttentionKind::DeltaNet,
    AttentionKind::DeltaNet,
    AttentionKind::Qsa,
    AttentionKind::DeltaNet,
    AttentionKind::DeltaNet,
    AttentionKind::DeltaNet,
    AttentionKind::Qsa,
    AttentionKind::DeltaNet,
    AttentionKind::DeltaNet,
    AttentionKind::DeltaNet,
    AttentionKind::Qsa,
    AttentionKind::DeltaNet,
    AttentionKind::DeltaNet,
    AttentionKind::DeltaNet,
    AttentionKind::Qsa,
    AttentionKind::DeltaNet,
    AttentionKind::DeltaNet,
    AttentionKind::DeltaNet,
    AttentionKind::Qsa,
];

/// The Flash-Next family: `Qwen/Qwen3.8-Flash-Next`, model type `qwen4_exp`.
///
/// The values come from the pinned artifact `config.json` (see
/// the Flash-Next executor design and the environment record).
/// The configuration names the root model type `qwen4_exp` and the `text_config`
/// model type `qwen4_exp_text`, so both belong to the family.
pub const QWEN3_8_FLASH_NEXT: FamilySpec = FamilySpec {
    name: "qwen3_8_flash_next",
    layers: 48,
    hidden: 2560,
    heads_q: 24,
    heads_kv: 2,
    head_dim: 256,
    global_head_dim: 0,
    rope_theta: 10_000_000.0,
    rms_norm_eps: 1e-6,
    model_types: &["qwen4_exp", "qwen4_exp_text"],
    vocab: 248_320,
    feed_forward: None,
    experts: 512,
    top_k: 10,
    moe_inter: 640,
    shared_inter: 640,
    quant_bits: 4,
    quant_group: 64,
    act: Act::Silu,
    router: Router::SoftmaxTopK { renorm: true },
    mtp: true,
    layers_attention: &QWEN3_8_FLASH_NEXT_LAYERS,
    sliding_window: 0,
    kv_shared_layers: 0,
    linear: Some(LinearAttention {
        key_heads: 16,
        key_head_dim: 128,
        value_heads: 48,
        value_head_dim: 128,
        conv_kernel: 4,
    }),
    hyper: Some(HyperConnection {
        branches: 4,
        rank: 320,
    }),
};

pub fn qwen3_8_flash_next() -> FamilySpec {
    QWEN3_8_FLASH_NEXT
}

const fn bolt_e4b_attention() -> [AttentionKind; 42] {
    let mut kinds = [AttentionKind::Sliding; 42];
    let mut index = 5;
    while index < kinds.len() {
        kinds[index] = AttentionKind::Full;
        index += 6;
    }
    kinds
}

const NEMOTRON_BOLT_E4B_LAYERS: [AttentionKind; 42] = bolt_e4b_attention();

/// A chat template that a spec names for a weights file whose header carries none, found by the
/// file's `general.architecture`. A family adds a row here, not code: the GGUF tokenizer reader
/// takes the template from the row and logs which spec supplied it.
pub struct TemplateFallback {
    /// The name of the spec that supplies the template.
    pub spec: &'static str,
    pub architecture: &'static str,
    pub template: &'static str,
}

/// The templates of the specs that name one.
pub const CHAT_TEMPLATE_FALLBACKS: &[TemplateFallback] = &[TemplateFallback {
    spec: NEMOTRON_BOLT_E4B.name,
    architecture: "gemma4",
    template: include_str!("../../testdata/gemma4/chat_template.jinja"),
}];

/// The integer keys a Bolt E4B weights file states in its header, for a directory with no
/// `config.json`: the geometry of the spec, under the file's key names.
pub const NEMOTRON_BOLT_E4B_GGUF_KEYS: &[(&str, u32)] = &[
    ("gemma4.block_count", NEMOTRON_BOLT_E4B.layers),
    ("gemma4.context_length", 65_536),
    ("gemma4.embedding_length", NEMOTRON_BOLT_E4B.hidden),
    ("gemma4.embedding_length_per_layer_input", 256),
    ("gemma4.feed_forward_length", 10_240),
    ("gemma4.attention.head_count", NEMOTRON_BOLT_E4B.heads_q),
    (
        "gemma4.attention.key_length",
        NEMOTRON_BOLT_E4B.global_head_dim,
    ),
    (
        "gemma4.attention.key_length_swa",
        NEMOTRON_BOLT_E4B.head_dim,
    ),
    (
        "gemma4.attention.value_length",
        NEMOTRON_BOLT_E4B.global_head_dim,
    ),
    (
        "gemma4.attention.value_length_swa",
        NEMOTRON_BOLT_E4B.head_dim,
    ),
    (
        "gemma4.attention.sliding_window",
        NEMOTRON_BOLT_E4B.sliding_window,
    ),
    (
        "gemma4.attention.shared_kv_layers",
        NEMOTRON_BOLT_E4B.kv_shared_layers,
    ),
];

/// The final logit soft cap key of the Bolt E4B header and its value.
pub const NEMOTRON_BOLT_E4B_GGUF_SOFTCAP: (&str, f32) = ("gemma4.final_logit_softcapping", 30.0);

/// The row for a `general.architecture`, if a spec names a template for it.
pub fn chat_template_fallback(architecture: &str) -> Option<&'static TemplateFallback> {
    CHAT_TEMPLATE_FALLBACKS
        .iter()
        .find(|row| row.architecture == architecture)
}

/// The Nemotron 3.5 Bolt E4B family:
/// `nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16`, model type `gemma4` (text
/// `gemma4_text`), served through the base Gemma 4 architecture.
///
/// The first 24 layers are physical attention layers that store their own K/V;
/// the last 18 (`kv_shared_layers`) have no K/V projections and attend into the
/// cache slot of the last physical layer of the same kind: layer 22 for sliding
/// layers, layer 23 for full-attention layers. Sliding layers use a 512-token
/// window and 256-dim heads; full-attention layers use 512-dim heads
/// (`global_head_dim`).
///
/// The descriptor does not carry these checkpoint properties: rope per kind
/// (sliding: full rotary, theta 10,000, carried by `rope_theta`; full attention:
/// proportional rotary, partial factor 0.25, theta 1,000,000), final logit
/// softcapping 30.0, a per-layer input projection (vocabulary 262,144 to hidden
/// 256), query/key/value RMS norms, attention without 1/sqrt(d) scaling, tied
/// embeddings, and `use_double_wide_mlp: false`. The quantization fields are
/// zero: the BF16 source carries no affine quantization.
pub const NEMOTRON_BOLT_E4B: FamilySpec = FamilySpec {
    name: "nemotron_bolt_e4b",
    layers: 42,
    hidden: 2560,
    heads_q: 8,
    heads_kv: 2,
    head_dim: 256,
    global_head_dim: 512,
    rope_theta: 10_000.0,
    rms_norm_eps: 1e-6,
    model_types: &["gemma4", "gemma4_text"],
    vocab: 262_144,
    feed_forward: Some(DenseMlp {
        intermediate: 10_240,
        context_limit: 65_536,
    }),
    experts: 0,
    top_k: 0,
    moe_inter: 0,
    shared_inter: 0,
    quant_bits: 0,
    quant_group: 0,
    act: Act::GeluApprox,
    // Dense layers do not use a router.
    router: Router::SoftmaxTopK { renorm: true },
    mtp: false,
    layers_attention: &NEMOTRON_BOLT_E4B_LAYERS,
    sliding_window: 512,
    kv_shared_layers: 18,
    linear: None,
    hyper: None,
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::family::keys::TierAssignment;

    #[test]
    fn dense_27b_descriptor_matches_checkpoint_dimensions() {
        let spec = QWEN3_8_27B_DENSE;
        TierAssignment::from_keys(&[], &spec).expect("valid dense family");
        assert_eq!((spec.layers, spec.hidden, spec.vocab), (64, 5120, 248_320));
        assert_eq!((spec.heads_q, spec.heads_kv, spec.head_dim), (24, 4, 256));
        assert_eq!(
            spec.feed_forward,
            Some(DenseMlp {
                intermediate: 17_408,
                context_limit: 262_144,
            })
        );
        assert!(spec.reads_quantization_from_pack());
        let full: Vec<_> = spec
            .layers_attention
            .iter()
            .enumerate()
            .filter(|(_, kind)| **kind == AttentionKind::Full)
            .map(|(index, _)| index)
            .collect();
        assert_eq!(full, (3..64).step_by(4).collect::<Vec<_>>());
        assert_eq!(spec.linear, QWEN3_8_FLASH_NEXT.linear);
        assert!(!spec.mtp);
        assert_eq!(spec.hyper, None);
    }

    #[test]
    fn q36_keeps_its_moe_descriptor() {
        let spec = QWEN3_6_35B_A3B;
        TierAssignment::from_keys(&[], &spec).expect("valid MoE family");
        assert_eq!(spec.feed_forward, None);
        assert_eq!((spec.experts, spec.top_k), (256, 8));
        assert_eq!((spec.moe_inter, spec.shared_inter), (512, 512));
    }

    #[test]
    fn flash_next_layer_attention_has_48_entries() {
        let spec = qwen3_8_flash_next();
        assert_eq!(spec.name, "qwen3_8_flash_next");
        assert_eq!(spec.layers, 48);
        assert_eq!(spec.layers_attention.len(), 48);
    }

    #[test]
    fn flash_next_places_twelve_qsa_layers_at_the_documented_indices() {
        let spec = qwen3_8_flash_next();
        let qsa: Vec<usize> = spec
            .layers_attention
            .iter()
            .enumerate()
            .filter(|(_, kind)| **kind == AttentionKind::Qsa)
            .map(|(index, _)| index)
            .collect();
        assert_eq!(qsa.len(), 12);
        assert_eq!(qsa, [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47]);
    }

    #[test]
    fn flash_next_has_thirty_six_delta_net_layers() {
        let spec = qwen3_8_flash_next();
        let delta = spec
            .layers_attention
            .iter()
            .filter(|kind| **kind == AttentionKind::DeltaNet)
            .count();
        assert_eq!(delta, 36);
    }

    #[test]
    fn flash_next_linear_attention_geometry_matches_the_pinned_dimensions() {
        let linear = qwen3_8_flash_next()
            .linear
            .expect("flash-next delta layers have linear attention dimensions");
        assert_eq!((linear.key_heads, linear.key_head_dim), (16, 128));
        assert_eq!((linear.value_heads, linear.value_head_dim), (48, 128));
        assert_eq!(linear.conv_kernel, 4);
        assert_eq!(linear.key_dim().expect("key width"), 2048);
        assert_eq!(linear.value_dim().expect("value width"), 6144);
        assert_eq!(linear.conv_dim().expect("fused width"), 10_240);
    }

    #[test]
    fn flash_next_expert_count_and_top_k_match() {
        let spec = qwen3_8_flash_next();
        assert_eq!(spec.experts, 512);
        assert_eq!(spec.top_k, 10);
        assert_eq!(spec.moe_inter, 640);
        assert_eq!(spec.shared_inter, 640);
    }

    #[test]
    fn flash_next_hyper_connection_width_is_four_times_hidden() {
        let spec = qwen3_8_flash_next();
        let hyper = spec.hyper.expect("flash-next widens the residual stream");
        assert_eq!(hyper.branches, 4);
        assert_eq!(hyper.rank, 320);
        let width = hyper.width(spec.hidden).expect("hyper width");
        assert_eq!(width, 4 * 2560);
        assert_eq!(width, 10_240);
    }

    #[test]
    fn flash_next_passes_family_validation() {
        TierAssignment::from_keys(&[], &qwen3_8_flash_next()).expect("valid family");
    }

    fn shared_test_spec(
        layers_attention: &'static [AttentionKind],
        sliding_window: u32,
        kv_shared_layers: u32,
    ) -> FamilySpec {
        FamilySpec {
            name: "shared_test",
            layers: layers_attention.len() as u32,
            hidden: 8,
            heads_q: 2,
            heads_kv: 1,
            head_dim: 4,
            global_head_dim: 0,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-6,
            model_types: &["shared_test"],
            vocab: 16,
            feed_forward: Some(DenseMlp {
                intermediate: 8,
                context_limit: 128,
            }),
            experts: 0,
            top_k: 0,
            moe_inter: 0,
            shared_inter: 0,
            quant_bits: 4,
            quant_group: 64,
            act: Act::Silu,
            router: Router::SoftmaxTopK { renorm: true },
            mtp: false,
            layers_attention,
            sliding_window,
            kv_shared_layers,
            linear: None,
            hyper: None,
        }
    }

    #[test]
    fn bolt_layer_attention_has_42_entries() {
        let spec = NEMOTRON_BOLT_E4B;
        assert_eq!(spec.layers, 42);
        assert_eq!(spec.layers_attention.len(), 42);
    }

    #[test]
    fn bolt_places_seven_full_layers_at_the_documented_indices() {
        let full: Vec<usize> = NEMOTRON_BOLT_E4B
            .layers_attention
            .iter()
            .enumerate()
            .filter(|(_, kind)| **kind == AttentionKind::Full)
            .map(|(index, _)| index)
            .collect();
        assert_eq!(full.len(), 7);
        assert_eq!(full, [5, 11, 17, 23, 29, 35, 41]);
    }

    #[test]
    fn bolt_shares_k_v_across_the_last_eighteen_layers() {
        let spec = NEMOTRON_BOLT_E4B;
        assert_eq!(spec.kv_shared_layers, 18);
        assert_eq!(spec.first_shared_layer(), 24);
        for layer in [24, 25, 28, 30, 40] {
            assert_eq!(spec.shared_kv_source(layer).expect("sliding source"), 22);
        }
        for layer in [29, 35, 41] {
            assert_eq!(spec.shared_kv_source(layer).expect("full source"), 23);
        }
        assert!(spec.shared_kv_source(23).is_err());
    }

    #[test]
    fn bolt_geometry_matches_the_pinned_dimensions() {
        let spec = NEMOTRON_BOLT_E4B;
        assert_eq!((spec.layers, spec.hidden, spec.vocab), (42, 2560, 262_144));
        assert_eq!((spec.heads_q, spec.heads_kv), (8, 2));
        assert_eq!(
            (spec.head_dim, spec.global_head_dim, spec.sliding_window),
            (256, 512, 512)
        );
        assert_eq!(
            spec.feed_forward,
            Some(DenseMlp {
                intermediate: 10_240,
                context_limit: 65_536,
            })
        );
        assert_eq!(spec.act, Act::GeluApprox);
        assert_eq!((spec.linear, spec.hyper, spec.mtp), (None, None, false));
    }

    #[test]
    fn bolt_passes_family_validation() {
        TierAssignment::from_keys(&[], &NEMOTRON_BOLT_E4B).expect("valid family");
    }

    const SHARED_SOURCE_LAYERS: [AttentionKind; 3] = [
        AttentionKind::Sliding,
        AttentionKind::Sliding,
        AttentionKind::Full,
    ];

    #[test]
    fn shared_k_v_without_a_physical_source_is_rejected() {
        // Layer 2 is a shared full-attention layer, but both physical layers are sliding.
        let spec = shared_test_spec(&SHARED_SOURCE_LAYERS, 64, 1);
        let error = TierAssignment::from_keys(&[], &spec).expect_err("no physical source");
        assert!(format!("{error:?}").contains("no physical Full source"));
    }

    #[test]
    fn sliding_layers_and_the_window_are_declared_together() {
        const SLIDING: [AttentionKind; 1] = [AttentionKind::Sliding];
        let without_window = shared_test_spec(&SLIDING, 0, 0);
        assert!(TierAssignment::from_keys(&[], &without_window).is_err());
        const FULL: [AttentionKind; 1] = [AttentionKind::Full];
        let unused_window = shared_test_spec(&FULL, 64, 0);
        assert!(TierAssignment::from_keys(&[], &unused_window).is_err());
    }

    #[test]
    fn shared_helpers_are_safe_on_an_unvalidated_descriptor() {
        // More shared layers than layers: first_shared_layer saturates to 0 and
        // shared_kv_source refuses without underflowing or indexing out of bounds.
        let spec = shared_test_spec(&SHARED_SOURCE_LAYERS, 64, 5);
        assert_eq!(spec.first_shared_layer(), 0);
        for layer in 0..spec.layers {
            assert!(spec.shared_kv_source(layer).is_err());
        }
        assert!(spec.shared_kv_source(spec.layers).is_err());
    }

    #[test]
    fn full_head_dim_resolves_the_zero_selects_head_dim_rule() {
        assert_eq!(NEMOTRON_BOLT_E4B.full_head_dim(), 512);
        let inherited = shared_test_spec(&SHARED_SOURCE_LAYERS, 64, 0);
        assert_eq!(inherited.global_head_dim, 0);
        assert_eq!(inherited.full_head_dim(), inherited.head_dim);
    }

    #[test]
    fn shared_k_v_boundaries_are_validated() {
        const THREE: [AttentionKind; 3] = [
            AttentionKind::Sliding,
            AttentionKind::Sliding,
            AttentionKind::Sliding,
        ];
        // Zero shared layers: every layer stores its own K/V.
        let none_shared = shared_test_spec(&THREE, 64, 0);
        TierAssignment::from_keys(&[], &none_shared).expect("zero shared is valid");
        // All-but-one shared: the single physical sliding layer serves the rest.
        let all_but_one = shared_test_spec(&THREE, 64, 2);
        TierAssignment::from_keys(&[], &all_but_one).expect("all-but-one shared is valid");
        assert_eq!(all_but_one.first_shared_layer(), 1);
        assert_eq!(all_but_one.shared_kv_source(2).expect("source"), 0);
        // Shared across every layer (and beyond) is refused: no physical source.
        for shared in [3, 5] {
            let over = shared_test_spec(&THREE, 64, shared);
            let error = TierAssignment::from_keys(&[], &over).expect_err("shared >= layers");
            assert!(format!("{error:?}").contains("shares K/V across"));
        }
        // All-but-one shared over a kind the physical layer cannot serve is refused.
        let mixed = shared_test_spec(&SHARED_SOURCE_LAYERS, 64, 2);
        let error = TierAssignment::from_keys(&[], &mixed).expect_err("no physical Full source");
        assert!(format!("{error:?}").contains("no physical Full source"));
    }

    #[test]
    fn shared_k_v_over_a_non_attention_layer_is_rejected() {
        // A shared Qsa layer stores no K/V, even with a same-kind physical source.
        const QSA: [AttentionKind; 2] = [AttentionKind::Qsa, AttentionKind::Qsa];
        let spec = shared_test_spec(&QSA, 0, 1);
        let error = TierAssignment::from_keys(&[], &spec).expect_err("Qsa shares no K/V");
        assert!(format!("{error:?}").contains("stores no K/V"));
        // DeltaNet state is not a shareable K/V cache; the kind check precedes the
        // linear-dimension check, so this refuses even without linear dimensions.
        const DELTA: [AttentionKind; 2] = [AttentionKind::DeltaNet, AttentionKind::DeltaNet];
        let spec = shared_test_spec(&DELTA, 0, 1);
        let error = TierAssignment::from_keys(&[], &spec).expect_err("DeltaNet shares no K/V");
        assert!(format!("{error:?}").contains("stores no K/V"));
    }

    #[test]
    fn global_head_dim_requires_a_full_attention_layer() {
        const SLIDING: [AttentionKind; 2] = [AttentionKind::Sliding, AttentionKind::Sliding];
        let mut spec = shared_test_spec(&SLIDING, 64, 0);
        spec.global_head_dim = 512;
        let error = TierAssignment::from_keys(&[], &spec).expect_err("no full layer");
        assert!(format!("{error:?}").contains("without a full-attention layer"));
        // A full-attention layer carries the global head dimension.
        const MIXED: [AttentionKind; 2] = [AttentionKind::Sliding, AttentionKind::Full];
        let mut spec = shared_test_spec(&MIXED, 64, 0);
        spec.global_head_dim = 512;
        TierAssignment::from_keys(&[], &spec).expect("full layer carries the global head dim");
    }
}
