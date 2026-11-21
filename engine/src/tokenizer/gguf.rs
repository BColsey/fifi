//! A tokenizer bundle read from the metadata of a GGUF file.
//!
//! A GGUF that carries `tokenizer.ggml.*` keys needs no tokenizer files beside it. This module
//! reads the vocabulary, the merges and the chat template from the header and writes the
//! `tokenizer.json`, `tokenizer_config.json` and chat template bytes that
//! [`TokenizerBundle::from_parts`] takes, so the bundle is the one every directory-based family
//! already gets. Only byte-level BPE vocabularies (`tokenizer.ggml.model` = `gpt2`) with a known
//! pre-tokenizer name are accepted; any other file is refused by name, never guessed.

use std::fs::File;
use std::io::BufReader;
use std::os::unix::fs::FileExt;
use std::path::Path;

use super::{TokenizerBundle, TokenizerParts};
use crate::EngError;
use crate::weights::{ArrayMeta, GgufHeader, MetaValue};

/// Header cap for the file scan. The Raon-1 header (vocabulary, merges, tensor table) is about
/// 12 MB.
const HEADER_CAP_BYTES: u64 = 256 << 20;

/// Token types of the GGUF vocabulary that become added tokens: control (3) and user defined (4).
const TYPE_CONTROL: i32 = 3;
const TYPE_USER_DEFINED: i32 = 4;

fn reject(message: impl Into<String>) -> EngError {
    EngError::Checkpoint(format!("gguf tokenizer: {}", message.into()))
}

/// The CJK and kana ranges that the Raon-1 pre-tokenizer keeps apart from other letters.
const RAON_CJK: &str = "\u{4e00}-\u{9fa5}\u{3040}-\u{309f}\u{30a0}-\u{30ff}";

/// The split pattern of a `tokenizer.ggml.pre` name, or `None` for a name this reader does not
/// know.
fn pre_tokenizer_pattern(pre: &str) -> Option<String> {
    match pre {
        "raon" => {
            let letter = format!("(?:(?:(?![{RAON_CJK}])\\p{{L}})\\p{{M}}*)+");
            Some(format!(
                " ?[{RAON_CJK}]+| ?{letter}(?: {letter})*|\\p{{N}}{{1,3}}(?=(?:\\p{{N}}{{3}})*(?!\\p{{N}}))| ?[^\\s\\p{{L}}\\p{{N}}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
            ))
        }
        _ => None,
    }
}

/// What the header of one GGUF file says about its tokenizer.
pub(crate) struct GgufTokenizer {
    tokens: Vec<String>,
    types: Vec<i32>,
    merges: Vec<String>,
    pre: String,
    nfc: bool,
    eos_id: u32,
    chat_template: String,
}

impl GgufTokenizer {
    /// Reads the tokenizer keys of the GGUF at `path`. Only the header is read.
    pub(crate) fn read(path: &Path) -> Result<Self, EngError> {
        let file = File::open(path)
            .map_err(|error| reject(format!("cannot open {}: {error}", path.display())))?;
        let len = file
            .metadata()
            .map_err(|error| reject(format!("cannot stat {}: {error}", path.display())))?
            .len();
        let header = GgufHeader::parse(&mut BufReader::new(&file), len, HEADER_CAP_BYTES)?;
        match header.meta("tokenizer.ggml.model") {
            Some(MetaValue::String(model)) if model == "gpt2" => {}
            Some(MetaValue::String(model)) => {
                return Err(reject(format!(
                    "tokenizer.ggml.model is {model}, only gpt2 (byte-level BPE) is read"
                )));
            }
            _ => return Err(reject("tokenizer.ggml.model is missing")),
        }
        let pre = match header.meta("tokenizer.ggml.pre") {
            Some(MetaValue::String(pre)) => pre.clone(),
            _ => return Err(reject("tokenizer.ggml.pre is missing")),
        };
        let tokens = string_array(&file, &header, "tokenizer.ggml.tokens")?;
        let merges = string_array(&file, &header, "tokenizer.ggml.merges")?;
        let types = match header.meta("tokenizer.ggml.token_type") {
            Some(MetaValue::Array(array)) => int_array(&file, array, tokens.len())?,
            _ => vec![1; tokens.len()],
        };
        let eos_id = match header.meta("tokenizer.ggml.eos_token_id") {
            Some(MetaValue::U32(id)) => *id,
            Some(MetaValue::I32(id)) if *id >= 0 => *id as u32,
            _ => return Err(reject("tokenizer.ggml.eos_token_id is missing")),
        };
        if eos_id as usize >= tokens.len() {
            return Err(reject(format!(
                "eos token id {eos_id} is outside the {} tokens",
                tokens.len()
            )));
        }
        let nfc = matches!(
            header.meta("tokenizer.ggml.normalizer.nfc"),
            Some(MetaValue::Bool(true))
        );
        let chat_template = match header.meta("tokenizer.chat_template") {
            Some(MetaValue::String(template)) => template.clone(),
            _ => return Err(reject("tokenizer.chat_template is missing")),
        };
        Ok(Self {
            tokens,
            types,
            merges,
            pre,
            nfc,
            eos_id,
            chat_template,
        })
    }

    /// The four tokenizer files' bytes. `tokenizer.json` lists the vocabulary in id order, so
    /// the bytes depend on the file's tokenizer keys alone.
    pub(crate) fn parts(&self) -> Result<TokenizerParts, EngError> {
        let pattern = pre_tokenizer_pattern(&self.pre).ok_or_else(|| {
            reject(format!(
                "tokenizer.ggml.pre {} has no known split pattern",
                self.pre
            ))
        })?;
        let mut vocab = serde_json::Map::with_capacity(self.tokens.len());
        let mut added = Vec::new();
        for (id, token) in self.tokens.iter().enumerate() {
            if vocab.insert(token.clone(), serde_json::json!(id)).is_some() {
                return Err(reject(format!("token {token:?} appears twice")));
            }
            let kind = self.types[id];
            if kind == TYPE_CONTROL || kind == TYPE_USER_DEFINED {
                added.push(serde_json::json!({
                    "id": id,
                    "content": token,
                    "single_word": false,
                    "lstrip": false,
                    "rstrip": false,
                    "normalized": false,
                    "special": kind == TYPE_CONTROL,
                }));
            }
        }
        let byte_level = serde_json::json!({
            "type": "ByteLevel",
            "add_prefix_space": false,
            "trim_offsets": false,
            "use_regex": false,
        });
        let tokenizer = serde_json::json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": added,
            "normalizer": if self.nfc { serde_json::json!({"type": "NFC"}) } else { serde_json::Value::Null },
            "pre_tokenizer": {
                "type": "Sequence",
                "pretokenizers": [
                    {
                        "type": "Split",
                        "pattern": {"Regex": pattern},
                        "behavior": "Isolated",
                        "invert": false,
                    },
                    byte_level,
                ],
            },
            "post_processor": null,
            "decoder": byte_level,
            "model": {
                "type": "BPE",
                "dropout": null,
                "unk_token": null,
                "continuing_subword_prefix": null,
                "end_of_word_suffix": null,
                "fuse_unk": false,
                "byte_fallback": false,
                "ignore_merges": false,
                "vocab": vocab,
                "merges": self.merges,
            },
        });
        let config = serde_json::json!({
            "eos_token": self.tokens[self.eos_id as usize],
            "add_bos_token": false,
        });
        Ok(TokenizerParts {
            tokenizer_json: serialize(&tokenizer)?,
            tokenizer_config_json: serialize(&config)?,
            chat_template: self.chat_template.clone().into_bytes(),
            generation_config_json: None,
        })
    }
}

fn serialize(value: &serde_json::Value) -> Result<Vec<u8>, EngError> {
    serde_json::to_vec(value).map_err(|error| reject(format!("cannot write JSON: {error}")))
}

/// The tokenizer bundle of the GGUF at `path`, built from its own header.
pub(crate) fn bundle_from_gguf(path: &Path) -> Result<TokenizerBundle, EngError> {
    let tokenizer = GgufTokenizer::read(path)?;
    let note = format!(
        "tokenizer: built from the metadata of {} (pre-tokenizer {}, {} tokens, {} merges)",
        path.display(),
        tokenizer.pre,
        tokenizer.tokens.len(),
        tokenizer.merges.len()
    );
    TokenizerBundle::from_parts(tokenizer.parts()?, None, vec![note])
}

/// Reads the whole data range of one metadata array.
fn array_bytes(file: &File, name: &str, array: &ArrayMeta) -> Result<Vec<u8>, EngError> {
    let len = usize::try_from(array.byte_range.end - array.byte_range.start)
        .map_err(|_| reject(format!("{name} is too large")))?;
    let mut bytes = vec![0u8; len];
    file.read_exact_at(&mut bytes, array.byte_range.start)
        .map_err(|error| reject(format!("cannot read {name}: {error}")))?;
    Ok(bytes)
}

fn string_array(file: &File, header: &GgufHeader, name: &str) -> Result<Vec<String>, EngError> {
    let Some(MetaValue::Array(array)) = header.meta(name) else {
        return Err(reject(format!("{name} is missing or not an array")));
    };
    if array.elem_type != 8 {
        return Err(reject(format!("{name} is not an array of strings")));
    }
    let bytes = array_bytes(file, name, array)?;
    let mut out = Vec::with_capacity(usize::try_from(array.len).unwrap_or(0));
    let mut at = 0usize;
    for _ in 0..array.len {
        let size = bytes
            .get(at..at + 8)
            .and_then(|raw| <[u8; 8]>::try_from(raw).ok())
            .map(u64::from_le_bytes)
            .and_then(|size| usize::try_from(size).ok())
            .ok_or_else(|| reject(format!("{name} is truncated")))?;
        at += 8;
        let text = bytes
            .get(at..at + size)
            .ok_or_else(|| reject(format!("{name} is truncated")))?;
        out.push(
            String::from_utf8(text.to_vec())
                .map_err(|_| reject(format!("{name} holds a string that is not UTF-8")))?,
        );
        at += size;
    }
    Ok(out)
}

fn int_array(file: &File, array: &ArrayMeta, expect: usize) -> Result<Vec<i32>, EngError> {
    let name = "tokenizer.ggml.token_type";
    if array.elem_type != 5 {
        return Err(reject(format!("{name} is not an array of i32")));
    }
    if array.len != expect as u64 {
        return Err(reject(format!(
            "{name} has {} entries for {expect} tokens",
            array.len
        )));
    }
    let bytes = array_bytes(file, name, array)?;
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|raw| i32::from_le_bytes(*raw))
        .collect())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::{GgufTokenizer, bundle_from_gguf};
    use crate::tokenizer::{PromptMessage, Role, Thinking};
    use std::path::PathBuf;

    fn string(text: &str) -> Vec<u8> {
        let mut out = (text.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(text.as_bytes());
        out
    }

    fn kv(key: &str, ty: u32, value: &[u8]) -> Vec<u8> {
        let mut out = string(key);
        out.extend_from_slice(&ty.to_le_bytes());
        out.extend_from_slice(value);
        out
    }

    fn kv_string(key: &str, value: &str) -> Vec<u8> {
        kv(key, 8, &string(value))
    }

    fn kv_array(key: &str, elem_type: u32, items: &[Vec<u8>]) -> Vec<u8> {
        let mut value = elem_type.to_le_bytes().to_vec();
        value.extend_from_slice(&(items.len() as u64).to_le_bytes());
        for item in items {
            value.extend_from_slice(item);
        }
        kv(key, 9, &value)
    }

    fn strings(items: &[&str]) -> Vec<Vec<u8>> {
        items.iter().map(|item| string(item)).collect()
    }

    const TEMPLATE: &str = "{%- for message in messages %}{{- '<|im_start|>' + message.role + '\\n' }}{%- generation %}{{- message.content }}{%- endgeneration %}{{- '<|im_end|>\\n' }}{%- endfor %}{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- endif %}";

    /// The bytes of a tokenizer-only GGUF of `architecture`: eight tokens, two merges, one control
    /// token pair, sampling defaults and no tensors.
    pub(crate) fn tiny_gguf(architecture: &str, model: &str, pre: &str) -> Vec<u8> {
        let tokens = [
            "a",
            "b",
            "\u{120}",
            "ab",
            "\u{120}ab",
            "<|im_end|>",
            "<|im_start|>",
            "\u{10a}",
        ];
        let types: Vec<Vec<u8>> = [1i32, 1, 1, 1, 1, 3, 3, 1]
            .iter()
            .map(|t| t.to_le_bytes().to_vec())
            .collect();
        let kvs = vec![
            kv_string("general.architecture", architecture),
            kv("general.sampling.temp", 6, &0.8f32.to_le_bytes()),
            kv("general.sampling.top_k", 4, &20u32.to_le_bytes()),
            kv("general.sampling.top_p", 6, &0.95f32.to_le_bytes()),
            kv_string("tokenizer.ggml.model", model),
            kv_string("tokenizer.ggml.pre", pre),
            kv_array("tokenizer.ggml.tokens", 8, &strings(&tokens)),
            kv_array("tokenizer.ggml.token_type", 5, &types),
            kv_array("tokenizer.ggml.merges", 8, &strings(&["a b", "\u{120} ab"])),
            kv("tokenizer.ggml.eos_token_id", 4, &5u32.to_le_bytes()),
            kv("tokenizer.ggml.normalizer.nfc", 7, &[1]),
            kv_string("tokenizer.chat_template", TEMPLATE),
        ];
        let mut file = b"GGUF".to_vec();
        file.extend_from_slice(&3u32.to_le_bytes());
        file.extend_from_slice(&0u64.to_le_bytes());
        file.extend_from_slice(&(kvs.len() as u64).to_le_bytes());
        for kv in &kvs {
            file.extend_from_slice(kv);
        }
        while !file.len().is_multiple_of(32) {
            file.push(0);
        }
        file
    }

    fn tiny_file(model: &str, pre: &str) -> PathBuf {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "gguf-tokenizer-{:x}-{}.gguf",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        std::fs::write(&path, tiny_gguf("tiny", model, pre)).expect("write the tokenizer file");
        path
    }

    #[test]
    fn a_tokenizer_only_gguf_builds_a_bundle_that_encodes_and_ends_on_its_eos() {
        let path = tiny_file("gpt2", "raon");
        let bundle = bundle_from_gguf(&path).expect("bundle");
        let tokenizer = bundle.tokenizer();
        assert_eq!(tokenizer.encode("ab ab").expect("encode"), [3, 4]);
        assert_eq!(tokenizer.encode("ab<|im_end|>").expect("encode"), [3, 5]);
        assert_eq!(bundle.eos_token_ids(), [5]);
        assert_eq!(bundle.vocab_size(), 8);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn the_chat_template_with_generation_tags_renders_as_plain_chatml() {
        let path = tiny_file("gpt2", "raon");
        let bundle = bundle_from_gguf(&path).expect("bundle");
        let message = |role, text: &str| PromptMessage {
            role,
            content: Some(text.to_string()),
            reasoning_content: None,
            name: None,
            tool_call_id: None,
            tool_calls: Vec::new(),
        };
        let rendered = bundle
            .render_prompt(
                &[message(Role::User, "ab"), message(Role::Assistant, "b")],
                &[],
                Thinking::Default,
            )
            .expect("render");
        assert_eq!(
            rendered,
            "<|im_start|>user\nab<|im_end|>\n<|im_start|>assistant\nb<|im_end|>\n<|im_start|>assistant\n"
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn the_same_file_gives_the_same_tokenizer_bytes() {
        let path = tiny_file("gpt2", "raon");
        let first = GgufTokenizer::read(&path)
            .expect("read")
            .parts()
            .expect("parts");
        let second = GgufTokenizer::read(&path)
            .expect("read")
            .parts()
            .expect("parts");
        assert_eq!(first.tokenizer_json, second.tokenizer_json);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn a_vocabulary_that_is_not_byte_level_bpe_is_refused_by_name() {
        let path = tiny_file("llama", "raon");
        let error = GgufTokenizer::read(&path).err().expect("refused");
        assert!(
            error.describe().contains("tokenizer.ggml.model is llama"),
            "{error:?}"
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn an_unknown_pre_tokenizer_name_is_refused_by_name() {
        let path = tiny_file("gpt2", "mystery");
        let tokenizer = GgufTokenizer::read(&path).expect("read");
        let error = tokenizer.parts().err().expect("refused");
        assert!(error.describe().contains("mystery"), "{error:?}");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn a_file_without_tokenizer_keys_is_refused() {
        let path =
            std::env::temp_dir().join(format!("gguf-no-tokenizer-{}.gguf", std::process::id()));
        let mut file = b"GGUF".to_vec();
        file.extend_from_slice(&3u32.to_le_bytes());
        file.extend_from_slice(&0u64.to_le_bytes());
        file.extend_from_slice(&0u64.to_le_bytes());
        file.resize(32, 0);
        std::fs::write(&path, file).expect("write");
        let error = GgufTokenizer::read(&path).err().expect("refused");
        assert!(
            error.describe().contains("tokenizer.ggml.model is missing"),
            "{error:?}"
        );
        let _ = std::fs::remove_file(path);
    }

    /// The published Raon-1 Q4_K_M file when RAON_Q4_KM points at it: the encodings of
    /// `testdata/raon1/tokenizer-cases.json` equal the ids a llama.cpp tokenizer produced for the
    /// same file (no BOS, special tokens parsed).
    #[test]
    fn the_published_file_encodes_the_reference_cases() {
        let Ok(path) = std::env::var("RAON_Q4_KM") else {
            eprintln!("skipped: RAON_Q4_KM is unset");
            return;
        };
        let bundle = bundle_from_gguf(std::path::Path::new(&path)).expect("bundle");
        let cases: Vec<serde_json::Value> =
            serde_json::from_str(include_str!("../../testdata/raon1/tokenizer-cases.json"))
                .expect("cases");
        let mut mismatches = Vec::new();
        for case in &cases {
            let text = case["text"].as_str().expect("text");
            let want: Vec<u32> = case["ids"]
                .as_array()
                .expect("ids")
                .iter()
                .map(|id| id.as_u64().expect("id") as u32)
                .collect();
            let got = bundle.tokenizer().encode(text).expect("encode");
            if got != want {
                mismatches.push(format!("{text:?}: want {want:?}, got {got:?}"));
            }
        }
        assert!(
            mismatches.is_empty(),
            "{} of {} cases differ:\n{}",
            mismatches.len(),
            cases.len(),
            mismatches.join("\n")
        );
        assert_eq!(bundle.eos_token_ids(), [259_202]);
        assert_eq!(bundle.vocab_size(), 259_840);
    }

    /// The published file's chat template against rendered reference texts, when RAON_Q4_KM
    /// names the file and RAON_TEMPLATE_CASES a JSON list of
    /// `{name, messages, tools, enable_thinking, expected}` cases produced by a Python Jinja
    /// renderer over the same template. The messages carry no reasoning content: the engine
    /// drops resent reasoning for this template, as the cases do.
    #[test]
    fn the_published_template_renders_the_reference_cases() {
        let (Ok(path), Ok(cases)) = (
            std::env::var("RAON_Q4_KM"),
            std::env::var("RAON_TEMPLATE_CASES"),
        ) else {
            eprintln!("skipped: RAON_Q4_KM or RAON_TEMPLATE_CASES is unset");
            return;
        };
        let bundle = bundle_from_gguf(std::path::Path::new(&path)).expect("bundle");
        let cases: Vec<serde_json::Value> =
            serde_json::from_slice(&std::fs::read(cases).expect("read cases")).expect("cases");
        let mut failures = Vec::new();
        for case in &cases {
            let messages: Vec<PromptMessage> = case["messages"]
                .as_array()
                .expect("messages")
                .iter()
                .map(|m| PromptMessage {
                    role: match m["role"].as_str().expect("role") {
                        "system" => Role::System,
                        "user" => Role::User,
                        "assistant" => Role::Assistant,
                        _ => Role::Tool,
                    },
                    content: m["content"].as_str().map(str::to_string),
                    reasoning_content: None,
                    name: None,
                    tool_call_id: None,
                    tool_calls: m["tool_calls"]
                        .as_array()
                        .map(|calls| {
                            calls
                                .iter()
                                .map(|call| crate::tokenizer::ToolCallSpec {
                                    id: call["id"].as_str().expect("id").to_string(),
                                    name: call["function"]["name"]
                                        .as_str()
                                        .expect("name")
                                        .to_string(),
                                    arguments: call["function"]["arguments"].to_string(),
                                })
                                .collect()
                        })
                        .unwrap_or_default(),
                })
                .collect();
            let tools: Vec<crate::tokenizer::ToolSpec> = case["tools"]
                .as_array()
                .expect("tools")
                .iter()
                .map(|tool| crate::tokenizer::ToolSpec(tool.clone()))
                .collect();
            let thinking = match case["enable_thinking"].as_bool() {
                None => Thinking::Default,
                Some(true) => Thinking::On,
                Some(false) => Thinking::Off,
            };
            let rendered = bundle.render_prompt(&messages, &tools, thinking);
            let name = case["name"].as_str().expect("name");
            let expected = case["expected"].as_str().expect("expected");
            match rendered {
                Ok(text) if text == expected => {
                    let ids = bundle.tokenizer().encode(&text).expect("encode");
                    eprintln!(
                        "case {name}: equal, {} tokens, tail {:?}",
                        ids.len(),
                        &ids[ids.len().saturating_sub(4)..]
                    );
                }
                Ok(text) => failures.push(format!("{name}:\n got {text:?}\nwant {expected:?}")),
                Err(error) => failures.push(format!("{name}: {}", error.describe())),
            }
        }
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }
}
