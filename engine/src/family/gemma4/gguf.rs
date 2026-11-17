//! Checked openers for the Gemma 4 Q8_0 GGUF checkpoints.
//!
//! open_humanizer_q8 and open_bolt_q8 each open one single-file GGUF through
//! the shared reader, check the gemma4 architecture, the scalar metadata, the
//! per-layer attention arrays, and the exact tensor inventory. They never map
//! the file and claim no execution.

use std::path::Path;

use crate::{
    EngError,
    weights::{GgufLimits, GgufType, GgufWeights, MetaValue},
};

/// general.architecture of the llama.cpp conversion of Gemma 4.
const ARCHITECTURE: &str = "gemma4";
/// Reader limits: a 100-MB header cap and a 1-GiB tensor cap.
const LIMITS: GgufLimits = GgufLimits {
    max_header_bytes: 100_000_000,
    max_tensor_bytes: 1 << 30,
};
const LAYERS: usize = 48;
const HIDDEN: u64 = 3840;
const FEED_FORWARD: u64 = 15360;
const QUERY_HEADS: u64 = 16;
const VOCAB: u64 = 262_144;
const LOCAL_KV_HEADS: u64 = 8;
const LOCAL_HEAD: u64 = 256;
const GLOBAL_KV_HEADS: u64 = 1;
const GLOBAL_HEAD: u64 = 512;
const GGUF_I32: u32 = 5;
const GGUF_BOOL: u32 = 7;
/// The eoa/eoi ids both conversions name; each suppresses them on the text
/// path.
const SUPPRESSED_TOKENS: [i32; 2] = [258_883, 258_882];

const HUMANIZER_INT_SCALARS: [(&str, u64); 16] = [
    ("gemma4.block_count", LAYERS as u64),
    ("gemma4.context_length", 262_144),
    ("gemma4.embedding_length", HIDDEN),
    ("gemma4.feed_forward_length", FEED_FORWARD),
    ("gemma4.attention.head_count", QUERY_HEADS),
    ("gemma4.attention.key_length", GLOBAL_HEAD),
    ("gemma4.attention.key_length_swa", LOCAL_HEAD),
    ("gemma4.attention.value_length", GLOBAL_HEAD),
    ("gemma4.attention.value_length_swa", LOCAL_HEAD),
    ("gemma4.attention.sliding_window", 1024),
    ("gemma4.attention.shared_kv_layers", 0),
    ("gemma4.rope.dimension_count", 512),
    ("gemma4.rope.dimension_count_swa", 256),
    ("tokenizer.ggml.bos_token_id", 2),
    ("tokenizer.ggml.eos_token_id", 1),
    ("tokenizer.ggml.unknown_token_id", 3),
];

const FLOAT_SCALARS: [(&str, f64); 4] = [
    ("gemma4.rope.freq_base", 1_000_000.0),
    ("gemma4.rope.freq_base_swa", 10_000.0),
    ("gemma4.final_logit_softcapping", 30.0),
    ("gemma4.attention.layer_norm_rms_epsilon", 1.0e-6),
];

/// Opens the pinned Humanizer Q8_0 GGUF and checks its metadata and tensors.
///
/// Opener rejections are EngError::Checkpoint with the prefix gguf model: and
/// name the offending metadata key or tensor. Failures of GgufWeights::open
/// pass through unchanged, so a parse or header-hash failure keeps its own
/// gguf: prefix and a tensor over the limit stays EngError::MemoryPlan.
pub fn open_humanizer_q8(path: &Path, header_sha256: Option<&str>) -> Result<(), EngError> {
    open_humanizer_weights(path, header_sha256).map(|_| ())
}

/// Opens the pinned checkpoint and returns the checked reader, for the resident
/// loader. Applies exactly the checks of open_humanizer_q8.
pub(crate) fn open_humanizer_weights(
    path: &Path,
    header_sha256: Option<&str>,
) -> Result<GgufWeights, EngError> {
    let weights = GgufWeights::open(path, LIMITS, header_sha256)?;
    let header = weights.header();
    check_architecture(header)?;
    check_int_scalars(header, &HUMANIZER_INT_SCALARS)?;
    check_float_scalars(header)?;
    check_bool_scalar(header, "tokenizer.ggml.add_bos_token", true)?;
    let kv = read_i32_array(
        &weights,
        header,
        "gemma4.attention.head_count_kv",
        LAYERS,
        GGUF_I32,
    )?;
    let pattern = read_bool_array(
        &weights,
        header,
        "gemma4.attention.sliding_window_pattern",
        LAYERS,
        GGUF_BOOL,
    )?;
    let suppressed = read_i32_array(
        &weights,
        header,
        "tokenizer.ggml.suppress_tokens",
        SUPPRESSED_TOKENS.len(),
        GGUF_I32,
    )?;
    check_layer_pattern(&kv, &pattern)?;
    if suppressed != SUPPRESSED_TOKENS {
        return Err(reject(format!(
            "metadata key tokenizer.ggml.suppress_tokens is {suppressed:?}, expected {SUPPRESSED_TOKENS:?}"
        )));
    }
    check_inventory(header, expected_inventory(&kv))?;
    Ok(weights)
}

/// One expected tensor: storage type and dimensions in file order.
struct Expected {
    name: String,
    ty: GgufType,
    dims: Vec<u64>,
}

/// Builds the exact 667-tensor inventory driven by the checked KV-head array.
fn expected_inventory(kv: &[i32]) -> Vec<Expected> {
    let mut expected = Vec::with_capacity(667);
    expected.push(Expected {
        name: "token_embd.weight".to_owned(),
        ty: GgufType::Q80,
        dims: vec![HIDDEN, VOCAB],
    });
    expected.push(Expected {
        name: "output_norm.weight".to_owned(),
        ty: GgufType::F32,
        dims: vec![HIDDEN],
    });
    expected.push(Expected {
        name: "rope_freqs.weight".to_owned(),
        ty: GgufType::F32,
        dims: vec![256],
    });
    for (layer, &heads) in kv.iter().enumerate() {
        let local = heads == LOCAL_KV_HEADS as i32;
        let head = if local { LOCAL_HEAD } else { GLOBAL_HEAD };
        let kv_heads = if local {
            LOCAL_KV_HEADS
        } else {
            GLOBAL_KV_HEADS
        };
        let base = format!("blk.{layer}.");
        let norm = |suffix: &str, width: u64| Expected {
            name: format!("{base}{suffix}"),
            ty: GgufType::F32,
            dims: vec![width],
        };
        let matrix = |suffix: &str, rows: u64, cols: u64| Expected {
            name: format!("{base}{suffix}"),
            ty: GgufType::Q80,
            dims: vec![rows, cols],
        };
        expected.push(norm("attn_norm.weight", HIDDEN));
        expected.push(matrix("attn_q.weight", HIDDEN, QUERY_HEADS * head));
        expected.push(norm("attn_q_norm.weight", head));
        expected.push(matrix("attn_k.weight", HIDDEN, kv_heads * head));
        expected.push(norm("attn_k_norm.weight", head));
        if local {
            expected.push(matrix("attn_v.weight", HIDDEN, kv_heads * head));
        }
        expected.push(matrix("attn_output.weight", QUERY_HEADS * head, HIDDEN));
        expected.push(norm("post_attention_norm.weight", HIDDEN));
        expected.push(norm("ffn_norm.weight", HIDDEN));
        expected.push(matrix("ffn_gate.weight", HIDDEN, FEED_FORWARD));
        expected.push(matrix("ffn_up.weight", HIDDEN, FEED_FORWARD));
        expected.push(matrix("ffn_down.weight", FEED_FORWARD, HIDDEN));
        expected.push(norm("post_ffw_norm.weight", HIDDEN));
        expected.push(norm("layer_output_scale.weight", 1));
    }
    debug_assert_eq!(expected.len(), 667);
    expected
}

fn check_inventory(
    header: &crate::weights::GgufHeader,
    mut expected: Vec<Expected>,
) -> Result<(), EngError> {
    // The missing-tensor refusal below names the first absent entry in this
    // order; sorting keeps that name stable regardless of inventory build order.
    expected.sort_by(|left, right| left.name.cmp(&right.name));
    let table: std::collections::HashMap<&str, (&GgufType, &[u64])> = expected
        .iter()
        .map(|item| (item.name.as_str(), (&item.ty, item.dims.as_slice())))
        .collect();
    for tensor in header.tensors() {
        let Some((ty, dims)) = table.get(tensor.name()) else {
            return Err(reject(format!("unexpected tensor {}", tensor.name())));
        };
        if tensor.ggml_type() != **ty {
            return Err(reject(format!(
                "tensor {} has type {}, expected {}",
                tensor.name(),
                tensor.ggml_type().name(),
                ty.name()
            )));
        }
        if tensor.dims() != *dims {
            return Err(reject(format!(
                "tensor {} has dims {:?}, expected {:?}",
                tensor.name(),
                tensor.dims(),
                dims
            )));
        }
    }
    for item in &expected {
        if header.tensor(&item.name).is_none() {
            return Err(reject(format!("missing tensor {}", item.name)));
        }
    }
    Ok(())
}

/// Checks the five-local, one-full pattern against both per-layer arrays.
fn check_layer_pattern(kv: &[i32], pattern: &[u8]) -> Result<(), EngError> {
    for (layer, (kv_heads, sliding)) in kv.iter().zip(pattern).enumerate() {
        let local = layer % 6 != 5;
        let want_kv = if local {
            LOCAL_KV_HEADS as i32
        } else {
            GLOBAL_KV_HEADS as i32
        };
        let want_pattern = u8::from(local);
        if *kv_heads != want_kv {
            return Err(reject(format!(
                "metadata key gemma4.attention.head_count_kv is {kv_heads} at layer {layer}, expected {want_kv}"
            )));
        }
        if *sliding != want_pattern {
            return Err(reject(format!(
                "metadata key gemma4.attention.sliding_window_pattern is {sliding} at layer {layer}, expected {want_pattern}"
            )));
        }
    }
    Ok(())
}

fn check_architecture(header: &crate::weights::GgufHeader) -> Result<(), EngError> {
    match header.meta("general.architecture") {
        Some(MetaValue::String(found)) if found == ARCHITECTURE => Ok(()),
        Some(MetaValue::String(found)) => Err(reject(format!(
            "metadata key general.architecture is {found:?}, expected {ARCHITECTURE:?}"
        ))),
        Some(_) => Err(reject("metadata key general.architecture is not a string")),
        None => Err(reject("metadata key general.architecture is missing")),
    }
}

fn check_int_scalars(
    header: &crate::weights::GgufHeader,
    scalars: &[(&str, u64)],
) -> Result<(), EngError> {
    for (key, want) in scalars {
        let found = match header.meta(key) {
            Some(MetaValue::U32(value)) => u64::from(*value),
            Some(MetaValue::U64(value)) => *value,
            Some(_) => {
                return Err(reject(format!(
                    "metadata key {key} is not an unsigned integer"
                )));
            }
            None => return Err(reject(format!("metadata key {key} is missing"))),
        };
        if found != *want {
            return Err(reject(format!(
                "metadata key {key} is {found}, expected {want}"
            )));
        }
    }
    Ok(())
}

/// Checks each float scalar against the expected value in its stored precision.
fn check_float_scalars(header: &crate::weights::GgufHeader) -> Result<(), EngError> {
    for (key, want) in FLOAT_SCALARS {
        let (found, want) = match header.meta(key) {
            Some(MetaValue::F32(value)) => (f64::from(*value), f64::from(want as f32)),
            Some(MetaValue::F64(value)) => (*value, want),
            Some(_) => return Err(reject(format!("metadata key {key} is not a float"))),
            None => return Err(reject(format!("metadata key {key} is missing"))),
        };
        if found != want {
            return Err(reject(format!(
                "metadata key {key} is {found}, expected {want}"
            )));
        }
    }
    Ok(())
}

fn check_bool_scalar(
    header: &crate::weights::GgufHeader,
    key: &str,
    want: bool,
) -> Result<(), EngError> {
    match header.meta(key) {
        Some(MetaValue::Bool(found)) if *found == want => Ok(()),
        Some(MetaValue::Bool(found)) => Err(reject(format!(
            "metadata key {key} is {found}, expected {want}"
        ))),
        Some(_) => Err(reject(format!("metadata key {key} is not a bool"))),
        None => Err(reject(format!("metadata key {key} is missing"))),
    }
}

fn read_i32_array(
    weights: &GgufWeights,
    header: &crate::weights::GgufHeader,
    key: &str,
    len: usize,
    elem_type: u32,
) -> Result<Vec<i32>, EngError> {
    let bytes = read_array_bytes(weights, header, key, len, elem_type, len * 4)?;
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| i32::from_le_bytes(*chunk))
        .collect())
}

fn read_bool_array(
    weights: &GgufWeights,
    header: &crate::weights::GgufHeader,
    key: &str,
    len: usize,
    elem_type: u32,
) -> Result<Vec<u8>, EngError> {
    read_array_bytes(weights, header, key, len, elem_type, len)
}

fn read_array_bytes(
    weights: &GgufWeights,
    header: &crate::weights::GgufHeader,
    key: &str,
    len: usize,
    elem_type: u32,
    byte_len: usize,
) -> Result<Vec<u8>, EngError> {
    let meta = match header.meta(key) {
        Some(MetaValue::Array(meta)) => meta,
        Some(_) => return Err(reject(format!("metadata key {key} is not an array"))),
        None => return Err(reject(format!("metadata key {key} is missing"))),
    };
    if meta.elem_type != elem_type {
        return Err(reject(format!(
            "metadata key {key} has element type {}, expected {elem_type}",
            meta.elem_type
        )));
    }
    if meta.len != len as u64 {
        return Err(reject(format!(
            "metadata key {key} has {} elements, expected {len}",
            meta.len
        )));
    }
    let mut buf = vec![0; byte_len];
    weights.read_header_range(meta.byte_range.start, &mut buf)?;
    Ok(buf)
}

fn reject(message: impl Into<String>) -> EngError {
    EngError::Checkpoint(format!("gguf model: {}", message.into()))
}

/// Checked geometry of the Nemotron 3.5 Bolt E4B conversion: 42 layers of
/// which the last 18 share the K/V of the last physical layer of their kind
/// (source 22 sliding, 23 full), two KV heads everywhere, separate V
/// projections on the 24 physical layers.
const BOLT_LAYERS: usize = 42;
const BOLT_SHARED: usize = 18;
const BOLT_HIDDEN: u64 = 2560;
const BOLT_FEED_FORWARD: u64 = 10_240;
const BOLT_QUERY_HEADS: u64 = 8;
const BOLT_VOCAB: u64 = 262_144;
const BOLT_KV_HEADS: u64 = 2;
const BOLT_LOCAL_HEAD: u64 = 256;
const BOLT_GLOBAL_HEAD: u64 = 512;
/// Width of the per-layer input the E4B checkpoint feeds every decoder layer.
const BOLT_PER_LAYER_INPUT: u64 = 256;
const BOLT_PHYSICAL_LAYERS: usize = BOLT_LAYERS - BOLT_SHARED;

const BOLT_INT_SCALARS: [(&str, u64); 16] = [
    ("gemma4.block_count", BOLT_LAYERS as u64),
    ("gemma4.context_length", 65_536),
    ("gemma4.embedding_length", BOLT_HIDDEN),
    ("gemma4.feed_forward_length", BOLT_FEED_FORWARD),
    ("gemma4.attention.head_count", BOLT_QUERY_HEADS),
    ("gemma4.attention.key_length", BOLT_GLOBAL_HEAD),
    ("gemma4.attention.key_length_swa", BOLT_LOCAL_HEAD),
    ("gemma4.attention.value_length", BOLT_GLOBAL_HEAD),
    ("gemma4.attention.value_length_swa", BOLT_LOCAL_HEAD),
    ("gemma4.attention.sliding_window", 512),
    ("gemma4.attention.shared_kv_layers", BOLT_SHARED as u64),
    ("gemma4.rope.dimension_count", BOLT_GLOBAL_HEAD),
    ("gemma4.rope.dimension_count_swa", BOLT_LOCAL_HEAD),
    ("tokenizer.ggml.bos_token_id", 2),
    ("tokenizer.ggml.eos_token_id", 1),
    ("tokenizer.ggml.unknown_token_id", 3),
];

/// Opens the converted Bolt Q8_0 GGUF and checks its metadata and tensors.
/// Applies the checks of open_humanizer_q8 against the Bolt geometry: two KV
/// heads on the 24 physical layers and none on the 18 shared layers.
pub fn open_bolt_q8(path: &Path, header_sha256: Option<&str>) -> Result<(), EngError> {
    open_bolt_weights(path, header_sha256).map(|_| ())
}

/// Opens the converted Bolt checkpoint and returns the checked reader, for
/// the resident loader. Applies exactly the checks of open_bolt_q8.
pub(crate) fn open_bolt_weights(
    path: &Path,
    header_sha256: Option<&str>,
) -> Result<GgufWeights, EngError> {
    let weights = GgufWeights::open(path, LIMITS, header_sha256)?;
    let header = weights.header();
    check_architecture(header)?;
    check_int_scalars(header, &BOLT_INT_SCALARS)?;
    check_float_scalars(header)?;
    check_bool_scalar(header, "tokenizer.ggml.add_bos_token", true)?;
    let kv = read_i32_array(
        &weights,
        header,
        "gemma4.attention.head_count_kv",
        BOLT_LAYERS,
        GGUF_I32,
    )?;
    let pattern = read_bool_array(
        &weights,
        header,
        "gemma4.attention.sliding_window_pattern",
        BOLT_LAYERS,
        GGUF_BOOL,
    )?;
    let suppressed = read_i32_array(
        &weights,
        header,
        "tokenizer.ggml.suppress_tokens",
        SUPPRESSED_TOKENS.len(),
        GGUF_I32,
    )?;
    check_bolt_layer_pattern(&kv, &pattern)?;
    if suppressed != SUPPRESSED_TOKENS {
        return Err(reject(format!(
            "metadata key tokenizer.ggml.suppress_tokens is {suppressed:?}, expected {SUPPRESSED_TOKENS:?}"
        )));
    }
    check_inventory(header, bolt_expected_inventory())?;
    Ok(weights)
}

/// Builds the exact 707-tensor Bolt inventory: every layer carries its own
/// query, output, norms, feed-forward, per-layer input gate, projection and
/// table, and the layer output scale; the 24 physical layers additionally
/// carry K/V projections and the K norm; the 18 shared layers carry neither
/// K nor V. The model-level per-layer machinery - the 10752-wide projection,
/// its norm - sits at the top level.
fn bolt_expected_inventory() -> Vec<Expected> {
    let mut expected = Vec::with_capacity(707);
    expected.push(Expected {
        name: "token_embd.weight".to_owned(),
        ty: GgufType::Q80,
        dims: vec![BOLT_HIDDEN, BOLT_VOCAB],
    });
    expected.push(Expected {
        name: "output_norm.weight".to_owned(),
        ty: GgufType::F32,
        dims: vec![BOLT_HIDDEN],
    });
    expected.push(Expected {
        name: "rope_freqs.weight".to_owned(),
        ty: GgufType::F32,
        dims: vec![BOLT_GLOBAL_HEAD / 2],
    });
    expected.push(Expected {
        name: "per_layer_model_projection.weight".to_owned(),
        ty: GgufType::Q80,
        dims: vec![BOLT_HIDDEN, BOLT_PER_LAYER_INPUT * BOLT_LAYERS as u64],
    });
    expected.push(Expected {
        name: "per_layer_projection_norm.weight".to_owned(),
        ty: GgufType::F32,
        dims: vec![BOLT_PER_LAYER_INPUT],
    });
    for layer in 0..BOLT_LAYERS {
        let local = layer % 6 != 5;
        let head = if local {
            BOLT_LOCAL_HEAD
        } else {
            BOLT_GLOBAL_HEAD
        };
        let physical = layer < BOLT_PHYSICAL_LAYERS;
        let base = format!("blk.{layer}.");
        let norm = |suffix: &str, width: u64| Expected {
            name: format!("{base}{suffix}"),
            ty: GgufType::F32,
            dims: vec![width],
        };
        let matrix = |suffix: &str, rows: u64, cols: u64| Expected {
            name: format!("{base}{suffix}"),
            ty: GgufType::Q80,
            dims: vec![rows, cols],
        };
        expected.push(norm("attn_norm.weight", BOLT_HIDDEN));
        expected.push(matrix(
            "attn_q.weight",
            BOLT_HIDDEN,
            BOLT_QUERY_HEADS * head,
        ));
        expected.push(norm("attn_q_norm.weight", head));
        if physical {
            expected.push(matrix("attn_k.weight", BOLT_HIDDEN, BOLT_KV_HEADS * head));
            expected.push(norm("attn_k_norm.weight", head));
            expected.push(matrix("attn_v.weight", BOLT_HIDDEN, BOLT_KV_HEADS * head));
        }
        expected.push(matrix(
            "attn_output.weight",
            BOLT_QUERY_HEADS * head,
            BOLT_HIDDEN,
        ));
        expected.push(norm("post_attention_norm.weight", BOLT_HIDDEN));
        expected.push(norm("ffn_norm.weight", BOLT_HIDDEN));
        expected.push(matrix("ffn_gate.weight", BOLT_HIDDEN, BOLT_FEED_FORWARD));
        expected.push(matrix("ffn_up.weight", BOLT_HIDDEN, BOLT_FEED_FORWARD));
        expected.push(matrix("ffn_down.weight", BOLT_FEED_FORWARD, BOLT_HIDDEN));
        expected.push(norm("post_ffw_norm.weight", BOLT_HIDDEN));
        expected.push(matrix(
            "per_layer_gate.weight",
            BOLT_HIDDEN,
            BOLT_PER_LAYER_INPUT,
        ));
        expected.push(matrix(
            "per_layer_proj.weight",
            BOLT_PER_LAYER_INPUT,
            BOLT_HIDDEN,
        ));
        expected.push(norm("post_per_layer_input_norm.weight", BOLT_HIDDEN));
        expected.push(matrix(
            "per_layer_embd.weight",
            BOLT_PER_LAYER_INPUT,
            BOLT_VOCAB,
        ));
        expected.push(norm("layer_output_scale.weight", 1));
    }
    debug_assert_eq!(expected.len(), 707);
    expected
}

/// Checks the five-sliding, one-full pattern and the two-heads-on-physical,
/// none-on-shared KV array of the Bolt conversion.
fn check_bolt_layer_pattern(kv: &[i32], pattern: &[u8]) -> Result<(), EngError> {
    for (layer, (kv_heads, sliding)) in kv.iter().zip(pattern).enumerate() {
        let local = layer % 6 != 5;
        let physical = layer < BOLT_PHYSICAL_LAYERS;
        let want_kv = if physical { BOLT_KV_HEADS as i32 } else { 0 };
        let want_pattern = u8::from(local);
        if *kv_heads != want_kv {
            return Err(reject(format!(
                "metadata key gemma4.attention.head_count_kv is {kv_heads} at layer {layer}, expected {want_kv}"
            )));
        }
        if *sliding != want_pattern {
            return Err(reject(format!(
                "metadata key gemma4.attention.sliding_window_pattern is {sliding} at layer {layer}, expected {want_pattern}"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::family::spec::NEMOTRON_BOLT_E4B;

    /// The reader's geometry constants are the descriptor's, not a second
    /// source of truth: a descriptor change fails here until the reader agrees.
    #[test]
    fn bolt_geometry_constants_match_the_descriptor() {
        let spec = NEMOTRON_BOLT_E4B;
        assert_eq!(BOLT_LAYERS as u32, spec.layers);
        assert_eq!(BOLT_SHARED as u32, spec.kv_shared_layers);
        assert_eq!(BOLT_PHYSICAL_LAYERS as u32, spec.first_shared_layer());
        assert_eq!(BOLT_HIDDEN, u64::from(spec.hidden));
        assert_eq!(
            BOLT_FEED_FORWARD,
            u64::from(spec.feed_forward.expect("dense").intermediate)
        );
        assert_eq!(BOLT_QUERY_HEADS, u64::from(spec.heads_q));
        assert_eq!(BOLT_KV_HEADS, u64::from(spec.heads_kv));
        assert_eq!(BOLT_VOCAB, u64::from(spec.vocab));
        assert_eq!(BOLT_LOCAL_HEAD, u64::from(spec.head_dim));
        assert_eq!(BOLT_GLOBAL_HEAD, u64::from(spec.full_head_dim()));
    }
}
