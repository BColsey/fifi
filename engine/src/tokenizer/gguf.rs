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

}
