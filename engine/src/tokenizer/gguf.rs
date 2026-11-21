//! A tokenizer bundle read from the metadata of a GGUF file.
//!
//! A GGUF that carries `tokenizer.ggml.*` keys needs no tokenizer files beside it. This module
//! reads the vocabulary, the merges and the chat template from the header and writes the
//! `tokenizer.json`, `tokenizer_config.json` and chat template bytes that
//! [`TokenizerBundle::from_parts`] takes, so the bundle is the one every directory-based family
//! already gets. Two vocabulary kinds are read: byte-level BPE (`tokenizer.ggml.model` = `gpt2`)
//! with a known pre-tokenizer name, and the space-marker BPE with byte fallback of the Gemma 4
//! family (`gemma4`). Any other file is refused by name, never guessed.

use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::Path;

use super::template::ChatTemplate;
use super::{
    PromptMessage, Role, Thinking, TokenizerBundle, TokenizerParts, ToolCallSpec, ToolSpec,
};
use crate::EngError;
use crate::weights::{ArrayMeta, GgufHeader, MetaValue};

/// Header cap for the file scan. The Raon-1 header (vocabulary, merges, tensor table) is about
/// 12 MB.
const HEADER_CAP_BYTES: u64 = 256 << 20;

/// The longest chat template a header may carry. Real templates are 10 to 30 KB.
const TEMPLATE_CAP_BYTES: usize = 256 << 10;

/// The two vocabulary kinds of `tokenizer.ggml.model` that this reader builds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    /// Byte-level BPE with a named pre-tokenizer.
    ByteLevel,
    /// Spaces become `▁`, the text splits after each one, unknown bytes fall back to byte tokens.
    SpaceMarker,
}

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
/// and eom ids when the file carries them, then the token the chat template ends an assistant
/// turn with, then the token it opens a tool response with (a model that calls a tool stops there
/// and does not write the response itself), then every control-type token whose text is an end
/// marker, in id order, without repeats.
fn stop_ids(
    tokens: &[String],
    types: &[i32],
    eos: u32,
    eot: Option<u32>,
    eom: Option<u32>,
    turn_end: Option<u32>,
    tool_response: Option<u32>,
) -> Vec<u32> {
    let mut ids = vec![eos];
    for id in eot
        .into_iter()
        .chain(eom)
        .chain(turn_end)
        .chain(tool_response)
    {
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

/// The control or user-defined token that the chat template writes right after an assistant
/// message: a fine-tune that ends its turns with a marker of its own stops on that marker even
/// when the header names another eos. `None` when the template does not render the probe or the
/// text after the message starts with no such token.
fn template_turn_end(tokens: &[String], types: &[i32], template: &str) -> Option<u32> {
    const PROBE: &str = "zq7xk";
    let message = |role, text: &str| PromptMessage {
        role,
        content: Some(text.to_owned()),
        reasoning_content: None,
        name: None,
        tool_call_id: None,
        tool_calls: Vec::new(),
    };
    let chat = ChatTemplate::probe(template).ok()?;
    let text = chat
        .render_without_generation_prompt(
            &[
                message(Role::User, "hello"),
                message(Role::Assistant, PROBE),
            ],
            &[],
            Thinking::Default,
        )
        .ok()?;
    let tail = &text[text.find(PROBE)? + PROBE.len()..];
    tokens
        .iter()
        .zip(types)
        .enumerate()
        .filter(|(_, (token, kind))| {
            (**kind == TYPE_CONTROL || **kind == TYPE_USER_DEFINED)
                && !token.is_empty()
                && tail.starts_with(token.as_str())
        })
        .max_by_key(|(_, (token, _))| token.len())
        .map(|(id, _)| id as u32)
}

/// Whether a token's text has the shape of a chat delimiter, `<|name>`, `<name|>` or `<|name|>`:
/// such a user-defined token marks structure and is special like a control token, so decoding
/// that skips special tokens leaves it out. Ordinary markup such as `<think>` has no bar and stays
/// plain text.
fn delimiter_shaped(token: &str) -> bool {
    token.len() >= 3
        && token.starts_with('<')
        && token.ends_with('>')
        && (token.starts_with("<|") || token.ends_with("|>"))
}

/// The delimiter token that the chat template writes between an assistant tool call and the
/// tool's answer: the point where a model that has called a tool must stop. Found by rendering a
/// tool call and its response and taking the last delimiter-shaped special token before the
/// response text that the plain generation prompt does not itself contain (a turn opener such as
/// `<|im_start|>` is in the generation prompt, so it is never taken). `None` when the template has
/// no tool support or writes no such token.
fn template_tool_response_opener(tokens: &[String], types: &[i32], template: &str) -> Option<u32> {
    const PROBE: &str = "zq7xk";
    let message = |role: Role, content: Option<&str>| {
        let tool = matches!(role, Role::Tool);
        PromptMessage {
            role,
            content: content.map(str::to_owned),
            reasoning_content: None,
            name: tool.then(|| "probe_tool".to_owned()),
            tool_call_id: tool.then(|| "call0".to_owned()),
            tool_calls: Vec::new(),
        }
    };
    let mut call = message(Role::Assistant, None);
    call.tool_calls.push(ToolCallSpec {
        id: "call0".to_owned(),
        name: "probe_tool".to_owned(),
        arguments: "{}".to_owned(),
    });
    let tools = [ToolSpec(serde_json::json!({
        "type": "function",
        "function": {
            "name": "probe_tool",
            "description": "probe",
            "parameters": {"type": "object", "properties": {}}
        }
    }))];
    let chat = ChatTemplate::probe(template).ok()?;
    let user = message(Role::User, Some("hello"));
    let asked = chat
        .render(std::slice::from_ref(&user), &tools, Thinking::Default)
        .ok()?;
    let text = chat
        .render_openai_calls(
            &[user, call, message(Role::Tool, Some(PROBE))],
            &tools,
            Thinking::Default,
        )
        .ok()?;
    let head = &text[..text.find(PROBE)?];
    tokens
        .iter()
        .zip(types)
        .enumerate()
        .filter(|(_, (token, kind))| {
            (**kind == TYPE_CONTROL || **kind == TYPE_USER_DEFINED)
                && delimiter_shaped(token)
                && !asked.contains(token.as_str())
        })
        .filter_map(|(id, (token, _))| head.rfind(token.as_str()).map(|at| (at, id as u32)))
        .max()
        .map(|(_, id)| id)
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
    kind: Kind,
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
    /// The token the chat template ends an assistant message with, see [`template_turn_end`].
    turn_end: Option<u32>,
    /// The token the chat template opens a tool response with, see [`template_tool_response_opener`].
    tool_response: Option<u32>,
    /// The line that says a spec's template stands in for a header that carries none.
    template_note: Option<String>,
    /// The bytes of a `generation_config.json` beside the weights file, when there is one.
    generation_config: Option<Vec<u8>>,
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
            self.turn_end,
            self.tool_response,
        )
    }

    /// The four tokenizer files' bytes. `tokenizer.json` lists the vocabulary in id order, so
    /// the bytes depend on the file's tokenizer keys alone.
    pub(crate) fn parts(&self) -> Result<TokenizerParts, EngError> {
        let tokenizer = match self.kind {
            Kind::ByteLevel => self.byte_level_json()?,
            Kind::SpaceMarker => self.space_marker_json()?,
        };
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

    /// The vocabulary map and the added-token list of the header's tokens.
    fn vocabulary(
        &self,
    ) -> Result<
        (
            serde_json::Map<String, serde_json::Value>,
            Vec<serde_json::Value>,
        ),
        EngError,
    > {
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
                    "special": kind == TYPE_CONTROL || delimiter_shaped(token),
                }));
            }
        }
        Ok((vocab, added))
    }

    /// The `tokenizer.json` of a byte-level BPE vocabulary.
    fn byte_level_json(&self) -> Result<serde_json::Value, EngError> {
        let pre = pre_tokenizer(&self.pre).ok_or_else(|| {
            reject(format!(
                "tokenizer.ggml.pre {} has no known split pattern (known: raon, qwen2, qwen35, pixtral)",
                self.pre
            ))
        })?;
        let pattern = pre.pattern;
        let nfc = self.nfc.unwrap_or(pre.nfc);
        let (vocab, added) = self.vocabulary()?;
        let byte_level = serde_json::json!({
            "type": "ByteLevel",
            "add_prefix_space": false,
            "trim_offsets": false,
            "use_regex": false,
        });
        Ok(serde_json::json!({
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
        }))
    }

    /// The `tokenizer.json` of a space-marker vocabulary: spaces become `▁`, the text splits
    /// after each marker, and a byte no token covers falls back to its `<0xNN>` token.
    fn space_marker_json(&self) -> Result<serde_json::Value, EngError> {
        let (vocab, added) = self.vocabulary()?;
        let unk = self
            .unk_id
            .map(|id| self.tokens[id as usize].clone())
            .ok_or_else(|| reject("tokenizer.ggml.unknown_token_id is missing"))?;
        Ok(serde_json::json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": added,
            "normalizer": {"type": "Replace", "pattern": {"String": " "}, "content": "\u{2581}"},
            "pre_tokenizer": {
                "type": "Split",
                "pattern": {"String": " "},
                "behavior": "MergedWithPrevious",
                "invert": false,
            },
            "post_processor": null,
            "decoder": {
                "type": "Sequence",
                "decoders": [
                    {"type": "Replace", "pattern": {"String": "\u{2581}"}, "content": " "},
                    {"type": "ByteFallback"},
                    {"type": "Fuse"},
                ],
            },
            "model": {
                "type": "BPE",
                "dropout": null,
                "unk_token": unk,
                "continuing_subword_prefix": null,
                "end_of_word_suffix": null,
                "fuse_unk": true,
                "byte_fallback": true,
                "ignore_merges": false,
                "vocab": vocab,
                "merges": self.merges,
            },
        }))
    }
}

fn serialize(value: &serde_json::Value) -> Result<Vec<u8>, EngError> {
    serde_json::to_vec(value).map_err(|error| reject(format!("cannot write JSON: {error}")))
}

/// The tokenizer bundle of the GGUF at `path`, built from its own header.
pub(crate) fn bundle_from_gguf(path: &Path) -> Result<TokenizerBundle, EngError> {
    bundle_of(GgufTokenizer::read(path)?, path)
}

/// The tokenizer bundle of one GGUF file named alone: its header and nothing beside it.
pub(crate) fn bundle_from_lone_gguf(path: &Path) -> Result<TokenizerBundle, EngError> {
    bundle_of(GgufTokenizer::read_alone(path)?, path)
}

fn bundle_of(tokenizer: GgufTokenizer, path: &Path) -> Result<TokenizerBundle, EngError> {
    let note = format!(
        "tokenizer: built from the metadata of {} (pre-tokenizer {}, {} tokens, {} merges)",
        path.display(),
        tokenizer.pre,
        tokenizer.tokens.len(),
        tokenizer.merges.len()
    );
    let mut notes = vec![note];
    notes.extend(tokenizer.template_note.clone());
    TokenizerBundle::from_parts(tokenizer.parts()?, None, notes)
}

/// The first GGUF file of `dir`, by name, whose header carries a vocabulary and a chat template (or
/// whose architecture has a spec template to stand in for it):
/// the file a start builds its tokenizer from when no `tokenizer.json` sits beside the weights.
/// A draft model's file without a template is skipped; later shards of a split model hold no
/// tokenizer keys and are never first. `None` when no file qualifies.
pub(crate) fn vocabulary_gguf(dir: &Path) -> Result<Option<std::path::PathBuf>, EngError> {
    // A path to one weights file is its own candidate.
    if crate::family::gguf_select::is_gguf_file(dir) {
        return Ok(has_vocabulary(dir).then(|| dir.to_path_buf()));
    }
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
        if has_vocabulary(&file) {
            return Ok(Some(file));
        }
    }
    Ok(None)
}

/// Whether the GGUF at `path` carries a vocabulary and a chat template (or an architecture whose
/// spec names one); an unreadable file does not.
fn has_vocabulary(path: &Path) -> bool {
    let Ok(header) = crate::weights::shared_header(path, HEADER_CAP_BYTES) else {
        return false;
    };
    let has_template = matches!(
        header.meta("tokenizer.chat_template"),
        Some(MetaValue::String(_))
    ) || matches!(
        header.meta("general.architecture"),
        Some(MetaValue::String(name))
            if crate::family::spec::chat_template_fallback(name).is_some()
    );
    has_template
        && matches!(
            header.meta("tokenizer.ggml.tokens"),
            Some(MetaValue::Array(_))
        )
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
    use super::{GgufTokenizer, bundle_from_gguf, vocabulary_gguf};
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

    #[cfg(feature = "native-cuda")]
    pub(crate) fn kv_u32_for_tests(key: &str, value: u32) -> Vec<u8> {
        kv(key, 4, &value.to_le_bytes())
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
        gguf_file(&kvs)
    }

    /// A GGUF with the given metadata entries and no tensors.
    pub(crate) fn gguf_file(kvs: &[Vec<u8>]) -> Vec<u8> {
        let mut file = b"GGUF".to_vec();
        file.extend_from_slice(&3u32.to_le_bytes());
        file.extend_from_slice(&0u64.to_le_bytes());
        file.extend_from_slice(&(kvs.len() as u64).to_le_bytes());
        for kv in kvs {
            file.extend_from_slice(kv);
        }
        while !file.len().is_multiple_of(32) {
            file.push(0);
        }
        file
    }

    /// The metadata of a small vocabulary file with the given tokens (all one type per entry of
    /// `types`), merges, template and eos id; `extra` adds more entries.
    fn vocabulary_kvs(
        model: &str,
        tokens: &[&str],
        types: &[i32],
        merges: &[&str],
        template: &str,
        eos: u32,
        extra: Vec<Vec<u8>>,
    ) -> Vec<Vec<u8>> {
        let types: Vec<Vec<u8>> = types.iter().map(|t| t.to_le_bytes().to_vec()).collect();
        let mut kvs = vec![
            kv_string("general.architecture", "tiny"),
            kv_string("tokenizer.ggml.model", model),
            kv_array("tokenizer.ggml.tokens", 8, &strings(tokens)),
            kv_array("tokenizer.ggml.token_type", 5, &types),
            kv_array("tokenizer.ggml.merges", 8, &strings(merges)),
            kv("tokenizer.ggml.eos_token_id", 4, &eos.to_le_bytes()),
            kv_string("tokenizer.chat_template", template),
        ];
        kvs.extend(extra);
        kvs
    }

    fn write_temp(bytes: &[u8]) -> PathBuf {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "gguf-tokenizer-extra-{:x}-{}.gguf",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        std::fs::write(&path, bytes).expect("write the file");
        path
    }

    #[test]
    fn a_space_marker_vocabulary_encodes_with_markers_and_byte_fallback() {
        let tokens = [
            "<pad>",
            "<eos>",
            "<bos>",
            "<unk>",
            "a",
            "b",
            "\u{2581}",
            "\u{2581}b",
            "<0x7A>",
        ];
        let types = [3, 3, 3, 3, 1, 1, 1, 1, 6];
        let kvs = vocabulary_kvs(
            "gemma4",
            &tokens,
            &types,
            &["\u{2581} b"],
            TEMPLATE,
            1,
            vec![
                kv("tokenizer.ggml.unknown_token_id", 4, &3u32.to_le_bytes()),
                kv("tokenizer.ggml.bos_token_id", 4, &2u32.to_le_bytes()),
            ],
        );
        let path = write_temp(&gguf_file(&kvs));
        let bundle = bundle_from_gguf(&path).expect("a gemma4 vocabulary builds");
        assert_eq!(bundle.tokenizer().encode("a b").expect("encode"), [4, 7]);
        assert_eq!(bundle.tokenizer().encode("z").expect("encode"), [8]);
        assert_eq!(bundle.tokenizer().decode(&[4, 7]).expect("decode"), "a b");
        let _ = std::fs::remove_file(path);
    }

    /// A Gemma-shaped header: delimiter tokens of user-defined type, a template that closes a
    /// turn with `<turn|>` and opens a tool response with `<|tool_response>`.
    fn delimiter_file() -> PathBuf {
        const GEMMA_LIKE: &str = "{%- for m in messages -%}\
            {%- if m.role == 'user' -%}<|turn>user\n{{ m.content }}<turn|>\n\
            {%- elif m.role == 'assistant' -%}<|turn>model\n\
            {%- if m.tool_calls -%}{%- for c in m.tool_calls -%}<|tool_call>call:{{ c.name }}{}<tool_call|><|tool_response>{%- endfor -%}\
            {%- else -%}{{ m.content }}<turn|>\n{%- endif -%}\
            {%- elif m.role == 'tool' -%}response:{{ m.content }}<tool_response|>{%- endif -%}\
            {%- endfor -%}{%- if add_generation_prompt -%}<|turn>model\n{%- endif -%}";
        let tokens = [
            "<pad>",
            "<eos>",
            "<bos>",
            "<unk>",
            "a",
            "b",
            "\u{2581}",
            "<0x7A>",
            "<|turn>",
            "<turn|>",
            "<|tool_call>",
            "<tool_call|>",
            "<|tool_response>",
            "<tool_response|>",
            "<|\"|>",
            "<|channel>",
            "<channel|>",
            "<think>",
        ];
        let types = [3, 3, 3, 3, 1, 1, 1, 6, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4];
        let kvs = vocabulary_kvs(
            "gemma4",
            &tokens,
            &types,
            &[],
            GEMMA_LIKE,
            1,
            vec![
                kv("tokenizer.ggml.bos_token_id", 4, &2u32.to_le_bytes()),
                kv("tokenizer.ggml.unknown_token_id", 4, &3u32.to_le_bytes()),
            ],
        );
        write_temp(&gguf_file(&kvs))
    }

    #[test]
    fn delimiter_shaped_user_defined_tokens_are_special_and_ordinary_markup_is_not() {
        let path = delimiter_file();
        let parts = GgufTokenizer::read(&path)
            .expect("read")
            .parts()
            .expect("parts");
        let json: serde_json::Value = serde_json::from_slice(&parts.tokenizer_json).expect("json");
        let special = |content: &str| {
            json["added_tokens"]
                .as_array()
                .expect("added tokens")
                .iter()
                .find(|token| token["content"] == content)
                .unwrap_or_else(|| panic!("{content} is an added token"))["special"]
                .as_bool()
                .expect("flag")
        };
        for token in [
            "<|tool_call>",
            "<tool_call|>",
            "<|tool_response>",
            "<tool_response|>",
            "<|\"|>",
            "<|channel>",
            "<channel|>",
        ] {
            assert!(special(token), "{token}");
        }
        assert!(!special("<think>"));
        // Decoding that skips special tokens leaves the delimiters out.
        let bundle = bundle_from_gguf(&path).expect("bundle");
        let ids: Vec<u32> = ["a", "<|channel>", "b", "<channel|>"]
            .iter()
            .map(|token| bundle.tokenizer().token_id(token).expect("id"))
            .collect();
        assert_eq!(bundle.tokenizer().decode(&ids).expect("decode"), "ab");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn the_stop_set_has_the_turn_end_and_the_tool_response_opener_but_no_turn_opener() {
        let path = delimiter_file();
        let tokenizer = GgufTokenizer::read(&path).expect("read");
        let id = |text: &str| {
            tokenizer
                .tokens
                .iter()
                .position(|token| token == text)
                .expect("token") as u32
        };
        assert_eq!(
            tokenizer.stop_ids(),
            [1, id("<turn|>"), id("<|tool_response>")]
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn a_lone_file_ignores_the_generation_config_beside_it_and_a_directory_reads_it() {
        let path = delimiter_file();
        let dir = tempfile::tempdir().expect("tempdir");
        let file = dir.path().join("m.gguf");
        std::fs::copy(&path, &file).expect("copy");
        std::fs::write(
            dir.path().join("generation_config.json"),
            br#"{"eos_token_id": [1, 5]}"#,
        )
        .expect("write");
        let ids = |bundle: &crate::tokenizer::TokenizerBundle| bundle.eos_token_ids().to_vec();
        let alone = super::bundle_from_lone_gguf(&file).expect("alone");
        let beside = bundle_from_gguf(&file).expect("beside");
        assert!(!ids(&alone).contains(&5), "{:?}", ids(&alone));
        assert!(ids(&beside).contains(&5), "{:?}", ids(&beside));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn two_vocabularies_in_one_directory_each_file_gets_its_own() {
        // No tokenizer.json and two files that both carry a vocabulary: a name-sorted scan of the
        // directory would give the first file's vocabulary to both.
        let first = delimiter_file();
        let second = space_marker_file_without_template("gemma4");
        let dir = tempfile::tempdir().expect("tempdir");
        let a = dir.path().join("a-first.gguf");
        let b = dir.path().join("b-second.gguf");
        std::fs::copy(&first, &a).expect("copy");
        std::fs::copy(&second, &b).expect("copy");
        let sizes = |path: &std::path::Path| {
            crate::tokenizer::gemma::load(path)
                .expect("a file is its own tokenizer")
                .vocab_size()
        };
        let (size_a, size_b) = (sizes(&a), sizes(&b));
        assert_ne!(size_a, size_b);
        assert_eq!(
            size_a,
            bundle_from_gguf(&first).expect("first").vocab_size()
        );
        assert_eq!(
            size_b,
            bundle_from_gguf(&second).expect("second").vocab_size()
        );
        // The preflight's resolver names the asked-for file too.
        assert_eq!(vocabulary_gguf(&b).expect("scan"), Some(b.clone()));
        assert_eq!(vocabulary_gguf(&a).expect("scan"), Some(a.clone()));
        // Naming the directory is the user's word for "whichever comes first".
        assert_eq!(vocabulary_gguf(dir.path()).expect("scan"), Some(a));
        let _ = std::fs::remove_file(first);
        let _ = std::fs::remove_file(second);
    }

    #[test]
    fn the_gemma_loader_reads_a_lone_gguf_and_pins_no_revision() {
        let tokens = [
            "<pad>",
            "<eos>",
            "<bos>",
            "<unk>",
            "a",
            "b",
            "\u{2581}",
            "\u{2581}b",
            "<0x7A>",
        ];
        let types = [3, 3, 3, 3, 1, 1, 1, 1, 6];
        let kvs = vocabulary_kvs(
            "gemma4",
            &tokens,
            &types,
            &["\u{2581} b"],
            TEMPLATE,
            1,
            vec![
                kv("tokenizer.ggml.unknown_token_id", 4, &3u32.to_le_bytes()),
                kv("tokenizer.ggml.bos_token_id", 4, &2u32.to_le_bytes()),
            ],
        );
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("model.gguf"), gguf_file(&kvs)).expect("write");
        let bundle =
            crate::tokenizer::gemma::load(dir.path()).expect("the header is the tokenizer");
        assert_eq!(bundle.tokenizer().encode("a b").expect("encode"), [4, 7]);
        assert!(
            bundle
                .load_notes()
                .iter()
                .all(|note| !note.contains("pinned")),
            "{:?}",
            bundle.load_notes()
        );
    }

    /// A space-marker vocabulary file of `architecture` whose header has no chat template.
    fn space_marker_file_without_template(architecture: &str) -> PathBuf {
        let tokens = [
            "<pad>",
            "<eos>",
            "<bos>",
            "<unk>",
            "a",
            "b",
            "\u{2581}",
            "\u{2581}b",
            "<0x7A>",
            "<|turn>",
            "<turn|>",
        ];
        let types = [3, 3, 3, 3, 1, 1, 1, 1, 6, 3, 3];
        let mut kvs = vocabulary_kvs(
            "gemma4",
            &tokens,
            &types,
            &["\u{2581} b"],
            "unused",
            1,
            vec![kv(
                "tokenizer.ggml.unknown_token_id",
                4,
                &3u32.to_le_bytes(),
            )],
        );
        kvs.retain(|entry| !entry.starts_with(&string("tokenizer.chat_template")));
        kvs[0] = kv_string("general.architecture", architecture);
        write_temp(&gguf_file(&kvs))
    }

    #[test]
    fn a_file_with_no_chat_template_takes_the_one_its_spec_names_and_says_so() {
        let path = space_marker_file_without_template("gemma4");
        let tokenizer = GgufTokenizer::read(&path).expect("the spec supplies the template");
        let note = tokenizer
            .template_note
            .clone()
            .expect("the fallback is logged");
        assert!(
            note.contains("carries none")
                && note.contains(crate::family::spec::NEMOTRON_BOLT_E4B.name),
            "{note}"
        );
        let parts = tokenizer.parts().expect("parts");
        assert_eq!(
            parts.chat_template,
            crate::family::spec::chat_template_fallback("gemma4")
                .expect("row")
                .template
                .as_bytes()
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn a_spec_template_that_does_not_fit_the_vocabulary_is_refused_not_applied() {
        // The vocabulary has no `<turn|>`: the Gemma template would end turns with a token the
        // file does not have, so no template is better than that one.
        let tokens = [
            "<pad>", "<eos>", "<bos>", "<unk>", "a", "b", "\u{2581}", "<0x7A>",
        ];
        let types = [3, 3, 3, 3, 1, 1, 1, 6];
        let mut kvs = vocabulary_kvs("gemma4", &tokens, &types, &[], "unused", 1, vec![]);
        kvs.retain(|entry| !entry.starts_with(&string("tokenizer.chat_template")));
        kvs[0] = kv_string("general.architecture", "gemma4");
        let path = write_temp(&gguf_file(&kvs));
        let error = GgufTokenizer::read(&path)
            .err()
            .expect("refused")
            .describe();
        assert!(error.contains("does not fit this vocabulary"), "{error}");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn a_directory_of_a_file_without_a_template_is_found_when_a_spec_names_one() {
        let dir = std::env::temp_dir().join(format!("gguf-vocab-scan-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("dir");
        let with_row = space_marker_file_without_template("gemma4");
        std::fs::copy(&with_row, dir.join("bolt.gguf")).expect("copy");
        assert_eq!(
            super::vocabulary_gguf(&dir).expect("scan"),
            Some(dir.join("bolt.gguf"))
        );
        std::fs::remove_file(dir.join("bolt.gguf")).expect("remove");
        let without_row = space_marker_file_without_template("someother");
        std::fs::copy(&without_row, dir.join("other.gguf")).expect("copy");
        assert_eq!(super::vocabulary_gguf(&dir).expect("scan"), None);
        let _ = std::fs::remove_file(with_row);
        let _ = std::fs::remove_file(without_row);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_file_with_no_chat_template_and_no_spec_row_is_refused_by_name() {
        let path = space_marker_file_without_template("someother");
        let error = GgufTokenizer::read(&path).err().expect("refused");
        assert!(
            error.describe().contains("chat_template is missing"),
            "{error:?}"
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn a_fine_tune_that_ends_its_turns_with_its_own_marker_stops_on_it() {
        // The template ends an assistant message with `<|turn_end|>` while the header's eos is
        // the end-of-text token: both stop the stream.
        let template = "{%- for message in messages %}{{- '<|turn_start|>' + message.role + '\n' + message.content + '<|turn_end|>\n' }}{%- endfor %}";
        let tokens = [
            "a",
            "b",
            "<|endoftext|>",
            "<|turn_start|>",
            "<|turn_end|>",
            "<|unrelated|>",
        ];
        let types = [1, 1, 3, 3, 3, 3];
        let kvs = vocabulary_kvs(
            "gpt2",
            &tokens,
            &types,
            &[],
            template,
            2,
            vec![kv_string("tokenizer.ggml.pre", "qwen2")],
        );
        let path = write_temp(&gguf_file(&kvs));
        let stops = GgufTokenizer::read(&path).expect("read").stop_ids();
        assert_eq!(stops, [2, 4]);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn a_template_over_the_size_cap_is_refused_by_name() {
        let template = format!("{}{{{{ 1 }}}}", " ".repeat(super::TEMPLATE_CAP_BYTES));
        let kvs = vocabulary_kvs(
            "gpt2",
            &["a", "<|im_end|>"],
            &[1, 3],
            &[],
            &template,
            1,
            vec![kv_string("tokenizer.ggml.pre", "qwen2")],
        );
        let path = write_temp(&gguf_file(&kvs));
        let error = GgufTokenizer::read(&path).err().expect("refused");
        assert!(error.describe().contains("byte cap"), "{error:?}");
        let _ = std::fs::remove_file(path);
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
        assert_eq!(
            super::stop_ids(&tokens, &types, 2, None, None, None, None),
            [2, 1]
        );
        assert_eq!(
            super::stop_ids(&tokens, &types, 2, Some(5), Some(5), None, None),
            [2, 5, 1]
        );
        assert_eq!(
            super::stop_ids(&tokens, &[1; 6], 2, None, None, None, None),
            [2]
        );
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

    /// The edge texts of `testdata/gguf-tokenizer/edge.json` (non-BMP, joined emoji, CJK,
    /// combining marks, digit runs, CRLF, whitespace runs, special-token text, punctuation,
    /// right-to-left scripts, case changes, empty and one-character input) against the ids of an
    /// independent tokenizer; `env` names a GGUF with the pre-tokenizer `kind`.
    fn check_edges(kind: &str, env: &str) {
        let path = std::env::var(env).unwrap_or_else(|_| panic!("{env} names the GGUF file"));
        let data: serde_json::Value =
            serde_json::from_str(include_str!("../../testdata/gguf-tokenizer/edge.json"))
                .expect("edge data");
        let bundle = bundle_from_gguf(std::path::Path::new(&path)).expect("bundle");
        let texts = data["texts"].as_array().expect("texts");
        let ids = data["ids"][kind]
            .as_array()
            .expect("ids of the pre-tokenizer");
        assert_eq!(texts.len(), ids.len());
        let mut differing = Vec::new();
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
            if got != want {
                differing.push(case["name"].as_str().unwrap_or("?").to_owned());
            }
        }
        assert!(differing.is_empty(), "{kind}: ids differ on {differing:?}");
    }

    #[test]
    #[ignore = "needs GGUF_EDGE_QWEN35, a GGUF with the qwen35 pre-tokenizer"]
    fn the_qwen35_pre_tokenizer_equals_the_recorded_ids_on_the_edge_texts() {
        check_edges("qwen35", "GGUF_EDGE_QWEN35");
    }

    #[test]
    #[ignore = "needs GGUF_EDGE_QWEN2, a GGUF with the qwen2 pre-tokenizer"]
    fn the_qwen2_pre_tokenizer_equals_the_recorded_ids_on_the_edge_texts() {
        check_edges("qwen2", "GGUF_EDGE_QWEN2");
    }

    #[test]
    #[ignore = "needs GGUF_EDGE_PIXTRAL, a GGUF with the pixtral pre-tokenizer"]
    fn the_pixtral_pre_tokenizer_equals_the_recorded_ids_on_the_edge_texts() {
        check_edges("pixtral", "GGUF_EDGE_PIXTRAL");
    }

    #[test]
    #[ignore = "needs GGUF_EDGE_RAON, a GGUF with the raon pre-tokenizer"]
    fn the_raon_pre_tokenizer_equals_the_recorded_ids_on_the_edge_texts() {
        check_edges("raon", "GGUF_EDGE_RAON");
    }

    /// Decoding with the special tokens skipped agrees with the reference `tokenizer.json` on a
    /// thinking turn and a tool-call turn: every delimiter the reference marks special is
    /// skipped here too. Skipped when `GGUF_TOKENIZER_FILE` or `GGUF_TOKENIZER_JSON` is unset.
    #[test]
    fn decoding_a_thinking_and_a_tool_call_turn_matches_the_tokenizer_json() {
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
        let ours = bundle.tokenizer();
        let id = |token: &str| ours.token_id(token).unwrap_or_else(|| panic!("{token}"));
        let mut turns: Vec<Vec<u32>> = Vec::new();
        for (opener, closer) in [
            ("<|channel>", "<channel|>"),
            ("<|tool_call>", "<tool_call|>"),
            ("<|tool_response>", "<tool_response|>"),
        ] {
            let mut ids = vec![id(opener)];
            ids.extend(
                ours.encode("thinking about it, then call:get_weather{city:")
                    .expect("encode"),
            );
            ids.push(id("<|\"|>"));
            ids.extend(ours.encode("Paris").expect("encode"));
            ids.push(id("<|\"|>"));
            ids.push(id(closer));
            ids.extend(ours.encode("The answer is 42.").expect("encode"));
            turns.push(ids);
        }
        for ids in &turns {
            assert_eq!(
                ours.decode(ids).expect("decode"),
                reference.decode(ids).expect("reference decode"),
                "{ids:?}"
            );
        }
        // Every id the reference skips, ours skips.
        for id in 0..ours.vocab_size() {
            assert_eq!(
                ours.decode(&[id]).expect("decode"),
                reference.decode(&[id]).expect("reference decode"),
                "token {id}"
            );
        }
    }

    /// A space-marker GGUF (`GGUF_TOKENIZER_FILE`) against the `tokenizer.json` of the same model
    /// (`GGUF_TOKENIZER_JSON`) over the edge texts and a few ordinary ones.
    #[test]
    #[ignore = "needs GGUF_TOKENIZER_FILE and GGUF_TOKENIZER_JSON of a gemma4 model"]
    fn the_space_marker_gguf_tokenizer_equals_the_tokenizer_json_on_the_edge_texts() {
        let gguf = std::env::var("GGUF_TOKENIZER_FILE").expect("GGUF_TOKENIZER_FILE");
        let json = std::env::var("GGUF_TOKENIZER_JSON").expect("GGUF_TOKENIZER_JSON");
        let bundle = bundle_from_gguf(std::path::Path::new(&gguf)).expect("bundle");
        let reference =
            crate::tokenizer::Tokenizer::from_bytes(&std::fs::read(&json).expect("json"))
                .expect("reference tokenizer");
        let data: serde_json::Value =
            serde_json::from_str(include_str!("../../testdata/gguf-tokenizer/edge.json"))
                .expect("edge data");
        let mut texts: Vec<String> = data["texts"]
            .as_array()
            .expect("texts")
            .iter()
            .map(|case| case["text"].as_str().expect("text").to_owned())
            .collect();
        texts.extend([
            "Hello, world!".to_owned(),
            "def f(x):\n    return x ** 2  # square".to_owned(),
            "  leading and trailing   spaces  ".to_owned(),
            "<start_of_turn>user\nhi<end_of_turn>".to_owned(),
        ]);
        let mut differing = Vec::new();
        for text in &texts {
            let got = bundle.tokenizer().encode(text).expect("encode");
            let want = reference.encode(text).expect("reference encode");
            if got != want {
                differing.push(text.chars().take(30).collect::<String>());
            }
        }
        assert!(differing.is_empty(), "ids differ on {differing:?}");
    }
}
