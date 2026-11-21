//! A tokenizer bundle read from the metadata of a GGUF file.
//!
//! A GGUF that carries `tokenizer.ggml.*` keys needs no tokenizer files beside it. This module
//! reads the vocabulary, the merges and the chat template from the header and writes the
//! `tokenizer.json`, `tokenizer_config.json` and chat template bytes that
//! [`TokenizerBundle::from_parts`] takes, so the bundle is the one every directory-based family
//! already gets. Only byte-level BPE vocabularies (`tokenizer.ggml.model` = `gpt2`) with a known
//! pre-tokenizer name are accepted; any other file is refused by name, never guessed.

use std::fs::File;
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

/// The texts of control tokens that end a turn or a sequence across the model families this
/// reader serves. Every control-type token with one of these texts is a stop id, so a raw
/// completion that emits `</s>` or `<|endoftext|>` ends instead of running to the token limit.
const END_MARKERS: [&str; 7] = [
    "</s>",
    "<|endoftext|>",
    "<|im_end|>",
    "<|eot_id|>",
    "<|end|>",
    "<end_of_turn>",
    "<|return|>",
];

/// The stop set of a vocabulary with no `generation_config.json`: the eos id first, then the eot
/// and eom ids when the file carries them, then every control-type token whose text is an end
/// marker, in id order, without repeats.
fn stop_ids(
    tokens: &[String],
    types: &[i32],
    eos: u32,
    eot: Option<u32>,
    eom: Option<u32>,
) -> Vec<u32> {
    let mut ids = vec![eos];
    for id in eot.into_iter().chain(eom) {
        if !ids.contains(&id) {
            ids.push(id);
        }
    }
    for (id, (token, kind)) in tokens.iter().zip(types).enumerate() {
        let id = id as u32;
        if *kind == TYPE_CONTROL && END_MARKERS.contains(&token.as_str()) && !ids.contains(&id) {
            ids.push(id);
        }
    }
    ids
}

fn reject(message: impl Into<String>) -> EngError {
    EngError::Checkpoint(format!("gguf tokenizer: {}", message.into()))
}

/// The CJK and kana ranges that the Raon-1 pre-tokenizer keeps apart from other letters.
const RAON_CJK: &str = "\u{4e00}-\u{9fa5}\u{3040}-\u{309f}\u{30a0}-\u{30ff}";

/// How a `tokenizer.ggml.pre` name splits text before the byte-level BPE merges.
struct PreTokenizer {
    /// The split regex; each match is one pre-token.
    pattern: String,
    /// Whether text is NFC-normalized first when the file carries no `normalizer.nfc` key.
    nfc: bool,
    /// Whether a pre-token that is a whole vocabulary entry is emitted without merging.
    ignore_merges: bool,
}

/// The pre-tokenizer of a `tokenizer.ggml.pre` name, or `None` for a name this reader does not
/// know. Each pattern describes the splitting rule of its family:
/// - `qwen2`: contractions, a letter run with at most one leading non-letter, single digits,
///   punctuation runs with trailing newlines, and whitespace runs that keep one space for the
///   next word.
/// - `qwen35`: the same, with combining marks counted as letters, and NFC normalization.
/// - `pixtral`: words split at case changes (an upper-case run then lower-case letters, or
///   lower-case letters), single digits, punctuation runs that may end in newlines or a slash,
///   and whitespace runs; whole vocabulary entries are emitted without merging.
/// - `raon`: CJK and kana runs apart, space-joined letter groups, digits in groups of three
///   counted from the right, punctuation runs, and whitespace runs.
fn pre_tokenizer(pre: &str) -> Option<PreTokenizer> {
    let (pattern, nfc, ignore_merges) = match pre {
        "raon" => {
            let letter = format!("(?:(?:(?![{RAON_CJK}])\\p{{L}})\\p{{M}}*)+");
            (
                format!(
                    " ?[{RAON_CJK}]+| ?{letter}(?: {letter})*|\\p{{N}}{{1,3}}(?=(?:\\p{{N}}{{3}})*(?!\\p{{N}}))| ?[^\\s\\p{{L}}\\p{{N}}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
                ),
                false,
                false,
            )
        }
        "qwen2" => (
            "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+".to_owned(),
            true,
            false,
        ),
        "qwen35" => (
            "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+".to_owned(),
            true,
            false,
        ),
        "pixtral" => (
            "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+".to_owned(),
            false,
            true,
        ),
        _ => return None,
    };
    Some(PreTokenizer {
        pattern,
        nfc,
        ignore_merges,
    })
}

/// What the header of one GGUF file says about its tokenizer.
pub(crate) struct GgufTokenizer {
    tokens: Vec<String>,
    types: Vec<i32>,
    merges: Vec<String>,
    pre: String,
    /// `normalizer.nfc` when the file states it; otherwise the pre-tokenizer's default.
    nfc: Option<bool>,
    eos_id: u32,
    bos_id: Option<u32>,
    pad_id: Option<u32>,
    unk_id: Option<u32>,
    eot_id: Option<u32>,
    eom_id: Option<u32>,
    /// The bytes of a `generation_config.json` beside the weights file, when there is one.
    generation_config: Option<Vec<u8>>,
    add_bos: bool,
    chat_template: String,
}

impl GgufTokenizer {
    /// Reads the tokenizer keys of the GGUF at `path`. Only the header is read.
    pub(crate) fn read(path: &Path) -> Result<Self, EngError> {
        let file = File::open(path)
            .map_err(|error| reject(format!("cannot open {}: {error}", path.display())))?;
        let header = crate::weights::shared_header(path, HEADER_CAP_BYTES)?;
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
        let nfc = match header.meta("tokenizer.ggml.normalizer.nfc") {
            Some(MetaValue::Bool(state)) => Some(*state),
            _ => None,
        };
        let optional_id = |key: &str| -> Option<u32> {
            let id = match header.meta(key) {
                Some(MetaValue::U32(id)) => *id,
                Some(MetaValue::I32(id)) if *id >= 0 => *id as u32,
                _ => return None,
            };
            ((id as usize) < tokens.len()).then_some(id)
        };
        let (bos_id, pad_id, unk_id) = (
            optional_id("tokenizer.ggml.bos_token_id"),
            optional_id("tokenizer.ggml.padding_token_id"),
            optional_id("tokenizer.ggml.unknown_token_id"),
        );
        let chat_template = match header.meta("tokenizer.chat_template") {
            Some(MetaValue::String(template)) => template.clone(),
            _ => return Err(reject("tokenizer.chat_template is missing")),
        };
        let (eot_id, eom_id) = (
            optional_id("tokenizer.ggml.eot_token_id"),
            optional_id("tokenizer.ggml.eom_token_id"),
        );
        Ok(Self {
            tokens,
            types,
            merges,
            pre,
            nfc,
            eos_id,
            bos_id,
            pad_id,
            unk_id,
            eot_id,
            eom_id,
            generation_config: path
                .parent()
                .and_then(|dir| std::fs::read(dir.join("generation_config.json")).ok()),
            add_bos: matches!(
                header.meta("tokenizer.ggml.add_bos_token"),
                Some(MetaValue::Bool(true))
            ),
            chat_template,
        })
    }

    /// The ids that end generation when no `generation_config.json` names them.
    pub(crate) fn stop_ids(&self) -> Vec<u32> {
        stop_ids(
            &self.tokens,
            &self.types,
            self.eos_id,
            self.eot_id,
            self.eom_id,
        )
    }

    /// The four tokenizer files' bytes. `tokenizer.json` lists the vocabulary in id order, so
    /// the bytes depend on the file's tokenizer keys alone.
    pub(crate) fn parts(&self) -> Result<TokenizerParts, EngError> {
        let pre = pre_tokenizer(&self.pre).ok_or_else(|| {
            reject(format!(
                "tokenizer.ggml.pre {} has no known split pattern (known: raon, qwen2, qwen35, pixtral)",
                self.pre
            ))
        })?;
        let pattern = pre.pattern;
        let nfc = self.nfc.unwrap_or(pre.nfc);
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
            "normalizer": if nfc { serde_json::json!({"type": "NFC"}) } else { serde_json::Value::Null },
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
                "ignore_merges": pre.ignore_merges,
                "vocab": vocab,
                "merges": self.merges,
            },
        });
        let mut config = serde_json::json!({
            "eos_token": self.tokens[self.eos_id as usize],
            "add_bos_token": self.add_bos,
        });
        for (key, id) in [
            ("bos_token", self.bos_id),
            ("pad_token", self.pad_id),
            ("unk_token", self.unk_id),
        ] {
            if let Some(id) = id {
                config[key] = serde_json::json!(self.tokens[id as usize]);
            }
        }
        Ok(TokenizerParts {
            tokenizer_json: serialize(&tokenizer)?,
            tokenizer_config_json: serialize(&config)?,
            chat_template: self.chat_template.clone().into_bytes(),
            generation_config_json: Some(match &self.generation_config {
                Some(bytes) => bytes.clone(),
                None => serialize(&serde_json::json!({ "eos_token_id": self.stop_ids() }))?,
            }),
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

/// The first GGUF file of `dir`, by name, whose header carries a vocabulary and a chat template:
/// the file a start builds its tokenizer from when no `tokenizer.json` sits beside the weights.
/// A draft model's file without a template is skipped; later shards of a split model hold no
/// tokenizer keys and are never first. `None` when no file qualifies.
pub(crate) fn vocabulary_gguf(dir: &Path) -> Result<Option<std::path::PathBuf>, EngError> {
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(_) => return Ok(None),
    };
    let mut files: Vec<std::path::PathBuf> = entries
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_file()
                && path
                    .extension()
                    .is_some_and(|extension| extension.eq_ignore_ascii_case("gguf"))
        })
        .collect();
    files.sort();
    for file in files {
        // A file that is not a readable GGUF is not a candidate; the start reports its own.
        let Ok(header) = crate::weights::shared_header(&file, HEADER_CAP_BYTES) else {
            continue;
        };
        if matches!(
            header.meta("tokenizer.ggml.tokens"),
            Some(MetaValue::Array(_))
        ) && matches!(
            header.meta("tokenizer.chat_template"),
            Some(MetaValue::String(_))
        ) {
            return Ok(Some(file));
        }
    }
    Ok(None)
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
    // Each string takes at least its 8-byte length, so a count above bytes / 8 is a lie.
    if array.len > (bytes.len() / 8) as u64 {
        return Err(reject(format!("{name} is truncated")));
    }
    let mut out = Vec::with_capacity(usize::try_from(array.len).unwrap_or(0));
    let mut at = 0usize;
    for _ in 0..array.len {
        let size = bytes
            .get(
                at..at
                    .checked_add(8)
                    .ok_or_else(|| reject(format!("{name} is truncated")))?,
            )
            .and_then(|raw| <[u8; 8]>::try_from(raw).ok())
            .map(u64::from_le_bytes)
            .and_then(|size| usize::try_from(size).ok())
            .ok_or_else(|| reject(format!("{name} is truncated")))?;
        at += 8;
        let text = bytes
            .get(
                at..at
                    .checked_add(size)
                    .ok_or_else(|| reject(format!("{name} is truncated")))?,
            )
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
        let path = tiny_file("unigram", "raon");
        let error = GgufTokenizer::read(&path).err().expect("refused");
        assert!(
            error.describe().contains("tokenizer.ggml.model is unigram"),
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
    /// `testdata/raon1/tokenizer-cases.json` equal the ids that an independent tokenizer
    /// implementation recorded for the same file (no BOS, special tokens parsed).
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

    /// The 42-token prompt that the host reference's frozen stage dumps were produced from: its
    /// text encodes back to the same ids.
    #[test]
    fn the_frozen_reference_prompt_round_trips() {
        let Ok(path) = std::env::var("RAON_Q4_KM") else {
            eprintln!("skipped: RAON_Q4_KM is unset");
            return;
        };
        let bundle = bundle_from_gguf(std::path::Path::new(&path)).expect("bundle");
        let ids: [u32; 42] = [
            166863, 54935, 7821, 44, 11456, 189096, 257706, 142907, 189286, 9305, 39154, 166294,
            107408, 51684, 46, 224037, 18624, 166295, 11234, 166471, 167757, 142907, 22233, 5407,
            5066, 46, 254733, 27335, 44, 4565, 46, 124319, 156431, 44, 235664, 73629, 177858, 9678,
            444, 2452, 44, 3248,
        ];
        let text = bundle.tokenizer().decode(&ids).expect("decode");
        eprintln!("frozen prompt text: {text:?}");
        assert_eq!(bundle.tokenizer().encode(&text).expect("encode"), ids);
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

    #[test]
    fn every_known_pre_tokenizer_builds_its_tokenizer_json() {
        for (pre, nfc, ignore_merges) in [
            ("raon", false, false),
            ("qwen2", true, false),
            ("qwen35", true, false),
            ("pixtral", false, true),
        ] {
            let path = tiny_file("gpt2", pre);
            let parts = GgufTokenizer::read(&path)
                .expect("read")
                .parts()
                .unwrap_or_else(|error| panic!("{pre}: {}", error.describe()));
            let json: serde_json::Value =
                serde_json::from_slice(&parts.tokenizer_json).expect("tokenizer json");
            assert_eq!(json["model"]["ignore_merges"], ignore_merges, "{pre}");
            // The file states normalizer.nfc, so the file wins over the pre-tokenizer default.
            assert_eq!(
                json["normalizer"]["type"], "NFC",
                "{pre} states nfc in the file"
            );
            let _ = nfc;
            let _ = std::fs::remove_file(path);
        }
    }

    #[test]
    fn an_unknown_pre_tokenizer_is_refused_by_name_with_the_known_names() {
        let path = tiny_file("gpt2", "unheard-of");
        let error = GgufTokenizer::read(&path)
            .expect("read")
            .parts()
            .err()
            .expect("refused")
            .describe();
        assert!(
            error.contains("unheard-of") && error.contains("pixtral"),
            "{error}"
        );
        let _ = std::fs::remove_file(path);
    }

    /// A directory of GGUF weights with no tokenizer.json starts from the vocabulary in the
    /// header: the shared bundle load finds the first GGUF that has a vocabulary and a template.
    #[test]
    fn a_directory_with_only_gguf_weights_loads_its_tokenizer_from_the_header() {
        use crate::tokenizer::{TokenizerBundle, TokenizerPin};
        let dir = tempfile::tempdir().expect("tempdir");
        // A file that is not a model is skipped; the model file is found by its keys.
        std::fs::write(dir.path().join("aaa-notes.gguf"), b"not a gguf").expect("junk");
        std::fs::write(
            dir.path().join("model.gguf"),
            tiny_gguf("tiny", "gpt2", "qwen35"),
        )
        .expect("model");
        let found = super::vocabulary_gguf(dir.path())
            .expect("scan")
            .expect("a file");
        assert_eq!(found, dir.path().join("model.gguf"));
        let pin = TokenizerPin {
            tokenizer_json: [0; 32],
            tokenizer_config_json: [0; 32],
            chat_template_jinja: [0; 32],
            generation_config_json: [0; 32],
            vocab_size: 0,
        };
        let bundle = TokenizerBundle::load(dir.path(), &pin).expect("bundle from the header");
        assert_eq!(bundle.eos_token_ids(), [5]);
        // With a tokenizer.json beside the weights the files are read, not the header.
        std::fs::write(dir.path().join("tokenizer.json"), b"{}").expect("json");
        assert!(TokenizerBundle::load(dir.path(), &pin).is_err());
        let empty = tempfile::tempdir().expect("tempdir");
        assert!(
            super::vocabulary_gguf(empty.path())
                .expect("scan")
                .is_none()
        );
    }

    /// The tokenizer built from a GGUF encodes like the `tokenizer.json` of the same model:
    /// `GGUF_TOKENIZER_FILE` names the GGUF, `GGUF_TOKENIZER_JSON` the reference tokenizer.json,
    /// and `GGUF_TOKENIZER_TEXTS` (a colon-separated list of files) the texts. Skipped when unset.
    #[test]
    fn the_gguf_tokenizer_matches_the_tokenizer_json_of_the_same_model() {
        let (Ok(gguf), Ok(json)) = (
            std::env::var("GGUF_TOKENIZER_FILE"),
            std::env::var("GGUF_TOKENIZER_JSON"),
        ) else {
            eprintln!("skipped: GGUF_TOKENIZER_FILE or GGUF_TOKENIZER_JSON is unset");
            return;
        };
        let bundle = bundle_from_gguf(std::path::Path::new(&gguf)).expect("bundle");
        let reference =
            crate::tokenizer::Tokenizer::from_bytes(&std::fs::read(&json).expect("json"))
                .expect("reference tokenizer");
        let mut texts: Vec<String> = [
            "Hello, world!",
            "def f(x):\n    return x ** 2  # square",
            "It's 2026-11-21, don't panic.",
            "  leading and trailing   spaces  ",
            "caf\u{e9} na\u{ef}ve \u{65e5}\u{672c}\u{8a9e} \u{1f600}",
            "123456789 3.14159 1,000,000",
            "<|im_start|>user\nhi<|im_end|>",
            "line one\r\nline two\n\n\nline four",
            "CamelCaseWord snake_case_word kebab-case-word",
            "e\u{301} (combining accent) and \u{fb01} ligature",
        ]
        .iter()
        .map(|text| (*text).to_owned())
        .collect();
        for file in std::env::var("GGUF_TOKENIZER_TEXTS")
            .unwrap_or_default()
            .split(':')
            .filter(|path| !path.is_empty())
        {
            texts.push(std::fs::read_to_string(file).expect("text file"));
        }
        let mut differing = Vec::new();
        for (index, text) in texts.iter().enumerate() {
            let got = bundle.tokenizer().encode(text).expect("encode");
            // `GGUF_TOKENIZER_DUMP` names a directory that receives one `<index>.json` of ids per
            // text, for comparison with another tokenizer implementation.
            if let Ok(dump) = std::env::var("GGUF_TOKENIZER_DUMP") {
                let ids = serde_json::to_vec(&got).expect("ids");
                std::fs::write(
                    std::path::Path::new(&dump).join(format!("{index}.json")),
                    ids,
                )
                .expect("dump ids");
            }
            let want = reference.encode(text).expect("reference encode");
            if got != want {
                let first = got.iter().zip(&want).position(|(a, b)| a != b);
                differing.push(format!(
                    "{:?}: lengths {} and {}, first difference at {first:?}",
                    text.chars().take(40).collect::<String>(),
                    got.len(),
                    want.len()
                ));
            }
        }
        assert!(
            differing.is_empty(),
            "{} of {} texts encode differently:\n{}",
            differing.len(),
            texts.len(),
            differing.join("\n")
        );
    }

    #[test]
    fn the_stop_rule_takes_eos_eot_eom_and_control_tokens_that_are_end_markers() {
        let tokens: Vec<String> = [
            "a",
            "</s>",
            "<|im_end|>",
            "<|endoftext|>",
            "<|im_start|>",
            "</s>x",
        ]
        .iter()
        .map(|t| (*t).to_owned())
        .collect();
        // </s> is a control token; the user-defined and normal tokens never count.
        let types = [1, 3, 3, 4, 3, 3];
        assert_eq!(super::stop_ids(&tokens, &types, 2, None, None), [2, 1]);
        assert_eq!(
            super::stop_ids(&tokens, &types, 2, Some(5), Some(5)),
            [2, 5, 1]
        );
        assert_eq!(super::stop_ids(&tokens, &[1; 6], 2, None, None), [2]);
    }

    /// For a checkpoint with a `generation_config.json`, the derived stop set equals its
    /// `eos_token_id` list: `GGUF_TOKENIZER_FILE` names the GGUF and `GGUF_GENERATION_CONFIG`
    /// the file. Skipped when unset.
    #[test]
    fn the_derived_stop_set_equals_the_eos_list_of_the_generation_config() {
        let (Ok(gguf), Ok(config)) = (
            std::env::var("GGUF_TOKENIZER_FILE"),
            std::env::var("GGUF_GENERATION_CONFIG"),
        ) else {
            eprintln!("skipped: GGUF_TOKENIZER_FILE or GGUF_GENERATION_CONFIG is unset");
            return;
        };
        let tokenizer = GgufTokenizer::read(std::path::Path::new(&gguf)).expect("read");
        let json: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&config).expect("config")).expect("json");
        let mut want: Vec<u32> = match &json["eos_token_id"] {
            serde_json::Value::Array(ids) => {
                ids.iter().map(|v| v.as_u64().expect("id") as u32).collect()
            }
            other => vec![other.as_u64().expect("id") as u32],
        };
        let mut got = tokenizer.stop_ids();
        want.sort_unstable();
        got.sort_unstable();
        assert_eq!(got, want);
    }

    /// The ids of an independent tokenizer (run on the same GGUF with no BOS token and no escape
    /// processing) over nine texts, kept in `testdata/gguf-tokenizer/oracle.json`; `env` names the GGUF.
    /// Also checks the derived stop ids against the end-marker ids of the checkpoint.
    fn check_oracle(model: &str, env: &str, stop: &[u32]) {
        let path = std::env::var(env).unwrap_or_else(|_| panic!("{env} names the GGUF file"));
        let data: serde_json::Value =
            serde_json::from_str(include_str!("../../testdata/gguf-tokenizer/oracle.json"))
                .expect("oracle data");
        let bundle = bundle_from_gguf(std::path::Path::new(&path)).expect("bundle");
        let texts = data["texts"].as_array().expect("texts");
        let ids = data["ids"][model].as_array().expect("ids of the model");
        assert_eq!(texts.len(), ids.len());
        for (case, want) in texts.iter().zip(ids) {
            let want: Vec<u32> = want
                .as_array()
                .expect("ids")
                .iter()
                .map(|id| id.as_u64().expect("id") as u32)
                .collect();
            let got = bundle
                .tokenizer()
                .encode(case["text"].as_str().expect("text"))
                .expect("encode");
            assert_eq!(got, want, "{model}: {}", case["name"]);
        }
        let mut stops = GgufTokenizer::read(std::path::Path::new(&path))
            .expect("read")
            .stop_ids();
        stops.sort_unstable();
        let mut stop = stop.to_vec();
        stop.sort_unstable();
        assert_eq!(stops, stop, "{model}: stop ids");
    }

    #[test]
    #[ignore = "needs GGUF_ORACLE_QWEN36, the Qwen3.6 GGUF"]
    fn the_qwen36_gguf_tokenizer_equals_the_recorded_ids_of_the_independent_tokenizer() {
        check_oracle("qwen36", "GGUF_ORACLE_QWEN36", &[248046, 248044]);
    }

    #[test]
    #[ignore = "needs GGUF_ORACLE_FLASH_NEXT, the first shard of the Flash-Next GGUF"]
    fn the_flash_next_gguf_tokenizer_equals_the_recorded_ids_of_the_independent_tokenizer() {
        check_oracle("flash_next", "GGUF_ORACLE_FLASH_NEXT", &[248046, 248044]);
    }

    #[test]
    #[ignore = "needs GGUF_ORACLE_NEMOTRON, the first shard of the Nemotron GGUF"]
    fn the_nemotron_gguf_tokenizer_equals_the_recorded_ids_of_the_independent_tokenizer() {
        check_oracle("nemotron", "GGUF_ORACLE_NEMOTRON", &[2, 11]);
    }
}
