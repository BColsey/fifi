use serde::Deserialize;
use serde_json::Value;

use crate::{EngError, checkpoint_json::UniqueValue};

const ARCHITECTURE: &str = "Gemma4UnifiedForConditionalGeneration";
const BOLT_ARCHITECTURE: &str = "Gemma4ForConditionalGeneration";

/// How many sessions the Bolt engine serves at once: an explicit count from 1 to 8, or `auto`,
/// which asks the sequence planner for the count the free device memory funds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(try_from = "RawMaxSeqs")]
pub(crate) enum MaxSeqs {
    Count(usize),
    Auto,
}

impl Default for MaxSeqs {
    /// One session: the engine serves one request at a time.
    fn default() -> Self {
        Self::Count(1)
    }
}

#[derive(Deserialize)]
#[serde(untagged)]
enum RawMaxSeqs {
    Count(u64),
    Word(String),
}

impl TryFrom<RawMaxSeqs> for MaxSeqs {
    type Error = String;

    fn try_from(raw: RawMaxSeqs) -> Result<Self, String> {
        match raw {
            RawMaxSeqs::Count(count) if (1..=8).contains(&count) => Ok(Self::Count(count as usize)),
            RawMaxSeqs::Count(count) => Err(format!("max_seqs {count} is outside 1 to 8")),
            RawMaxSeqs::Word(word) if word == "auto" => Ok(Self::Auto),
            RawMaxSeqs::Word(word) => {
                Err(format!("max_seqs accepts 1 to 8 or \"auto\", got {word:?}"))
            }
        }
    }
}

pub fn check_humanizer_config(json: &[u8]) -> Result<(), EngError> {
    let root = serde_json::from_slice::<UniqueValue>(json)
        .map_err(|error| EngError::Checkpoint(format!("configuration JSON: {error}")))?
        .0;
    if !root.is_object() {
        return Err(EngError::Checkpoint(
            "configuration must be a JSON object".into(),
        ));
    }
    expect_string(&root, "model_type", "model_type", "gemma4_unified")?;
    expect_architecture(&root, ARCHITECTURE)?;
    expect_bool(&root, "tie_word_embeddings", "tie_word_embeddings", true)?;
    let text = root
        .get("text_config")
        .ok_or_else(|| EngError::Checkpoint("configuration lacks text_config".into()))?;
    if !text.is_object() {
        return Err(EngError::Checkpoint(
            "configuration text_config must be a JSON object".into(),
        ));
    }
    expect_string(
        text,
        "model_type",
        "text_config.model_type",
        "gemma4_unified_text",
    )?;
    expect_bool(
        text,
        "tie_word_embeddings",
        "text_config.tie_word_embeddings",
        true,
    )?;
    check_layer_types(text, 48)?;
    check_rope_parameters(text)?;
    expect_string(
        text,
        "hidden_activation",
        "text_config.hidden_activation",
        "gelu_pytorch_tanh",
    )?;
    expect_f64(text, "rms_norm_eps", "text_config.rms_norm_eps", 1e-6)?;
    expect_f64(
        text,
        "final_logit_softcapping",
        "text_config.final_logit_softcapping",
        30.0,
    )?;
    expect_bool(text, "attention_bias", "text_config.attention_bias", false)?;
    expect_bool(
        text,
        "attention_k_eq_v",
        "text_config.attention_k_eq_v",
        true,
    )?;
    expect_bool(
        text,
        "enable_moe_block",
        "text_config.enable_moe_block",
        false,
    )?;
    expect_bool(text, "use_cache", "text_config.use_cache", true)?;
    expect_bool(
        text,
        "use_double_wide_mlp",
        "text_config.use_double_wide_mlp",
        false,
    )?;
    expect_string(
        text,
        "use_bidirectional_attention",
        "text_config.use_bidirectional_attention",
        "vision",
    )?;
    for name in ["num_experts", "top_k_experts", "moe_intermediate_size"] {
        expect_null(text, name)?;
    }
    for (name, expected) in [
        ("num_kv_shared_layers", 0),
        ("hidden_size_per_layer_input", 0),
    ] {
        expect_u32(text, name, expected)?;
    }
    for (name, expected) in [
        ("hidden_size", 3840),
        ("intermediate_size", 15360),
        ("num_hidden_layers", 48),
        ("num_attention_heads", 16),
        ("num_key_value_heads", 8),
        ("num_global_key_value_heads", 1),
        ("head_dim", 256),
        ("global_head_dim", 512),
        ("max_position_embeddings", 262144),
        ("sliding_window", 1024),
        ("vocab_size", 262144),
        ("vocab_size_per_layer_input", 262144),
        ("bos_token_id", 2),
        ("eos_token_id", 1),
        ("pad_token_id", 0),
    ] {
        expect_u32(text, name, expected)?;
    }
    Ok(())
}

/// The strict configuration check for the Nemotron 3.5 Bolt E4B checkpoint
/// (`nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16`): the base Gemma 4 architecture
/// with two KV heads and eighteen shared-K/V layers. The multimodal checkpoint
/// carries vision and audio sections this family does not serve; the check
/// reads only the root identity and the text stack.
pub fn check_bolt_config(json: &[u8]) -> Result<(), EngError> {
    let root = serde_json::from_slice::<UniqueValue>(json)
        .map_err(|error| EngError::Checkpoint(format!("configuration JSON: {error}")))?
        .0;
    if !root.is_object() {
        return Err(EngError::Checkpoint(
            "configuration must be a JSON object".into(),
        ));
    }
    expect_string(&root, "model_type", "model_type", "gemma4")?;
    expect_architecture(&root, BOLT_ARCHITECTURE)?;
    expect_bool(&root, "tie_word_embeddings", "tie_word_embeddings", true)?;
    let text = root
        .get("text_config")
        .ok_or_else(|| EngError::Checkpoint("configuration lacks text_config".into()))?;
    if !text.is_object() {
        return Err(EngError::Checkpoint(
            "configuration text_config must be a JSON object".into(),
        ));
    }
    expect_string(text, "model_type", "text_config.model_type", "gemma4_text")?;
    expect_bool(
        text,
        "tie_word_embeddings",
        "text_config.tie_word_embeddings",
        true,
    )?;
    check_layer_types(text, 42)?;
    check_rope_parameters(text)?;
    expect_string(
        text,
        "hidden_activation",
        "text_config.hidden_activation",
        "gelu_pytorch_tanh",
    )?;
    expect_f64(text, "rms_norm_eps", "text_config.rms_norm_eps", 1e-6)?;
    expect_f64(
        text,
        "final_logit_softcapping",
        "text_config.final_logit_softcapping",
        30.0,
    )?;
    expect_bool(text, "attention_bias", "text_config.attention_bias", false)?;
    expect_bool(
        text,
        "attention_k_eq_v",
        "text_config.attention_k_eq_v",
        false,
    )?;
    expect_bool(
        text,
        "enable_moe_block",
        "text_config.enable_moe_block",
        false,
    )?;
    expect_bool(text, "use_cache", "text_config.use_cache", true)?;
    expect_bool(
        text,
        "use_double_wide_mlp",
        "text_config.use_double_wide_mlp",
        false,
    )?;
    expect_null(text, "use_bidirectional_attention")?;
    for name in [
        "num_experts",
        "top_k_experts",
        "expert_intermediate_size",
        "num_global_key_value_heads",
    ] {
        expect_null(text, name)?;
    }
    for (name, expected) in [
        ("hidden_size", 2560),
        ("intermediate_size", 10_240),
        ("num_hidden_layers", 42),
        ("num_attention_heads", 8),
        ("num_key_value_heads", 2),
        ("num_kv_shared_layers", 18),
        ("hidden_size_per_layer_input", 256),
        ("head_dim", 256),
        ("global_head_dim", 512),
        ("max_position_embeddings", 65_536),
        ("sliding_window", 512),
        ("vocab_size", 262_144),
        ("vocab_size_per_layer_input", 262_144),
        ("bos_token_id", 2),
        ("eos_token_id", 1),
        ("pad_token_id", 0),
    ] {
        expect_u32(text, name, expected)?;
    }
    Ok(())
}

fn expect_architecture(root: &Value, architecture: &str) -> Result<(), EngError> {
    let name = "architectures";
    let value = required(root, name)?;
    let found = value
        .as_array()
        .ok_or_else(|| EngError::Checkpoint(format!("configuration {name} must be an array")))?;
    if found.len() == 1 && found[0].as_str() == Some(architecture) {
        return Ok(());
    }
    Err(EngError::Checkpoint(format!(
        "configuration {name} is {value}, expected [{architecture:?}]"
    )))
}

fn expect_string(object: &Value, key: &str, name: &str, expected: &str) -> Result<(), EngError> {
    let value = required(object, key)
        .map_err(|_| EngError::Checkpoint(format!("configuration lacks {name}")))?;
    let found = value
        .as_str()
        .ok_or_else(|| EngError::Checkpoint(format!("configuration {name} must be a string")))?;
    if found != expected {
        return Err(EngError::Checkpoint(format!(
            "configuration {name} is {found:?}, expected {expected:?}"
        )));
    }
    Ok(())
}

fn expect_bool(object: &Value, key: &str, name: &str, expected: bool) -> Result<(), EngError> {
    let value = required(object, key)
        .map_err(|_| EngError::Checkpoint(format!("configuration lacks {name}")))?;
    let found = value
        .as_bool()
        .ok_or_else(|| EngError::Checkpoint(format!("configuration {name} must be a boolean")))?;
    if found != expected {
        return Err(EngError::Checkpoint(format!(
            "configuration {name} is {found}, expected {expected}"
        )));
    }
    Ok(())
}

fn required<'a>(object: &'a Value, name: &str) -> Result<&'a Value, EngError> {
    object
        .get(name)
        .ok_or_else(|| EngError::Checkpoint(format!("configuration lacks {name}")))
}

fn expect_u32(object: &Value, name: &str, expected: u32) -> Result<(), EngError> {
    let path = format!("text_config.{name}");
    let value = required(object, name)
        .map_err(|_| EngError::Checkpoint(format!("configuration lacks {path}")))?;
    let found = value
        .as_u64()
        .and_then(|number| u32::try_from(number).ok())
        .ok_or_else(|| {
            EngError::Checkpoint(format!("configuration {path} must be a u32 integer"))
        })?;
    if found != expected {
        return Err(EngError::Checkpoint(format!(
            "configuration {path} is {found}, expected {expected}"
        )));
    }
    Ok(())
}

fn check_layer_types(text: &Value, layers: usize) -> Result<(), EngError> {
    let name = "text_config.layer_types";
    let value = required(text, "layer_types")
        .map_err(|_| EngError::Checkpoint(format!("configuration lacks {name}")))?;
    let kinds = value
        .as_array()
        .ok_or_else(|| EngError::Checkpoint(format!("configuration {name} must be an array")))?;
    if kinds.len() != layers {
        return Err(EngError::Checkpoint(format!(
            "configuration {name} has {} entries, expected {layers}",
            kinds.len()
        )));
    }
    for (index, layer) in kinds.iter().enumerate() {
        let expected = if index % 6 == 5 {
            "full_attention"
        } else {
            "sliding_attention"
        };
        if layer.as_str() != Some(expected) {
            return Err(EngError::Checkpoint(format!(
                "configuration {name}[{index}] is {layer}, expected {expected:?}"
            )));
        }
    }
    Ok(())
}

fn check_rope_parameters(text: &Value) -> Result<(), EngError> {
    let parameters = required(text, "rope_parameters").map_err(|_| {
        EngError::Checkpoint("configuration lacks text_config.rope_parameters".into())
    })?;
    if !parameters.is_object() {
        return Err(EngError::Checkpoint(
            "configuration text_config.rope_parameters must be an object".into(),
        ));
    }
    let full = required(parameters, "full_attention").map_err(|_| {
        EngError::Checkpoint(
            "configuration lacks text_config.rope_parameters.full_attention".into(),
        )
    })?;
    if !full.is_object() {
        return Err(EngError::Checkpoint(
            "configuration text_config.rope_parameters.full_attention must be an object".into(),
        ));
    }
    if let Some(factor) = full.get("factor")
        && factor.as_f64() != Some(1.0)
    {
        return Err(EngError::Checkpoint(
            "configuration text_config.rope_parameters.full_attention.factor must be the number 1.0".into(),
        ));
    }
    expect_string(
        full,
        "rope_type",
        "text_config.rope_parameters.full_attention.rope_type",
        "proportional",
    )?;
    expect_f64(
        full,
        "rope_theta",
        "text_config.rope_parameters.full_attention.rope_theta",
        1_000_000.0,
    )?;
    expect_f64(
        full,
        "partial_rotary_factor",
        "text_config.rope_parameters.full_attention.partial_rotary_factor",
        0.25,
    )?;

    let sliding = required(parameters, "sliding_attention").map_err(|_| {
        EngError::Checkpoint(
            "configuration lacks text_config.rope_parameters.sliding_attention".into(),
        )
    })?;
    if !sliding.is_object() {
        return Err(EngError::Checkpoint(
            "configuration text_config.rope_parameters.sliding_attention must be an object".into(),
        ));
    }
    expect_string(
        sliding,
        "rope_type",
        "text_config.rope_parameters.sliding_attention.rope_type",
        "default",
    )?;
    expect_f64(
        sliding,
        "rope_theta",
        "text_config.rope_parameters.sliding_attention.rope_theta",
        10_000.0,
    )
}

fn expect_f64(object: &Value, key: &str, path: &str, expected: f64) -> Result<(), EngError> {
    let value = required(object, key)
        .map_err(|_| EngError::Checkpoint(format!("configuration lacks {path}")))?;
    let found = value
        .as_f64()
        .ok_or_else(|| EngError::Checkpoint(format!("configuration {path} must be a number")))?;
    if found != expected {
        return Err(EngError::Checkpoint(format!(
            "configuration {path} is {found}, expected {expected}"
        )));
    }
    Ok(())
}

fn expect_null(object: &Value, name: &str) -> Result<(), EngError> {
    let path = format!("text_config.{name}");
    let value = required(object, name)
        .map_err(|_| EngError::Checkpoint(format!("configuration lacks {path}")))?;
    if value.is_null() {
        return Ok(());
    }
    Err(EngError::Checkpoint(format!(
        "configuration {path} is {value}, expected null"
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn architecture_refusal_names_found_and_expected() {
        let root = json!({"architectures": ["OtherArch"]});
        let Err(EngError::Checkpoint(message)) = expect_architecture(&root, "TestArch") else {
            panic!("expected a checkpoint refusal");
        };
        assert_eq!(
            message,
            "configuration architectures is [\"OtherArch\"], expected [\"TestArch\"]"
        );
    }

    #[test]
    fn layer_type_refusal_names_found_and_expected() {
        let text = json!({"layer_types": ["full_attention", "sliding_attention"]});
        let Err(EngError::Checkpoint(message)) = check_layer_types(&text, 2) else {
            panic!("expected a checkpoint refusal");
        };
        assert_eq!(
            message,
            "configuration text_config.layer_types[0] is \"full_attention\", \
             expected \"sliding_attention\""
        );
    }

    #[test]
    fn null_refusal_names_the_found_value() {
        let text = json!({"use_bidirectional_attention": true});
        let Err(EngError::Checkpoint(message)) = expect_null(&text, "use_bidirectional_attention")
        else {
            panic!("expected a checkpoint refusal");
        };
        assert_eq!(
            message,
            "configuration text_config.use_bidirectional_attention is true, expected null"
        );
    }
}
