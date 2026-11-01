//! Chat-template prefix stability on the pinned Flint template. Turn
//! N+1 reuses turn N's KV only if the rendered prompt of turn N+1 starts
//! with the exact bytes of turn N's rendered prompt, and the shared part
//! ends on a token boundary. The pinned template ends every generation
//! prompt with `<|assistant|>`, which the pinned tokenizer config
//! carries as exactly one added special token, so a byte prefix ending
//! there is also a token prefix: the always-run gate below pins that
//! token, and the real-tokenizer encoding test (ignored without
//! FLINT_MODEL_DIR) checks the ids directly. The model-level identity
//! gate stays with the device-side restore path.

use super::flint_tests;
use super::{ChatTemplate, PromptMessage, Role, Thinking, ToolCallSpec, ToolSpec};

/// The pinned publisher Flint template (sha 7940f4ff…, the bytes
/// `flint_tests` pins with `TokenizerPin::chat_template_jinja`), so the
/// always-run stability gate tests the real template, not a fixture.
const FLINT_TEMPLATE: &str = include_str!("../../testdata/flint/tokenizer/chat_template.jinja");
const TOKENIZER_CONFIG: &str = include_str!("../../testdata/flint/tokenizer/tokenizer_config.json");
const ASSISTANT_DELIMITER: &str = "<|assistant|>";

fn msg(role: Role, content: &str) -> PromptMessage {
    PromptMessage {
        role,
        content: Some(content.to_string()),
        reasoning_content: None,
        name: None,
        tool_call_id: None,
        tool_calls: Vec::new(),
    }
}

/// The developer row's content, pinned so a later test can assert its
/// bytes survive unchanged in every later render.
const DEVELOPER_ROW: &str =
    "bit-exact continuation text, with the from-scratch reference at the same rendered prompt";

/// A deterministic coding-agent conversation: system instructions, a
/// task with code, a reasoning + tool-call reply, the tool result, a
/// code answer, and follow-up turns. Every message shape the Flint
/// template renders appears at least once.
fn coding_transcript() -> Vec<PromptMessage> {
    vec![
        msg(
            Role::System,
            "You are a coding agent working in a Rust repository. Read the \
             code before changing it and keep diffs minimal.",
        ),
        msg(
            Role::User,
            "The decoder loops over every layer even when the prefix store \
             restored all tokens. Here is the loop:\n\nfor layer in \
             self.layers.iter() {\n    out = layer.step(out, &kv[layer]);\n}\n\
             \nCan we skip the layers the restored KV already covers?",
        ),
        PromptMessage {
            role: Role::Assistant,
            content: Some(
                "The restored KV covers the prefix, but each layer still needs \
                 its recurrent state advanced past those tokens before decode \
                 can continue. Let me check the session save shape."
                    .to_string(),
            ),
            reasoning_content: Some(
                "Prefix restore returns KV rows; the recurrent state is part of \
                 the checkpoint payload, not the KV rows. Parking must include \
                 it or the resumed run diverges."
                    .to_string(),
            ),
            name: None,
            tool_call_id: None,
            tool_calls: vec![ToolCallSpec {
                id: "call-7f3a".to_string(),
                name: "read_file".to_string(),
                arguments: r#"{"path": "engine/src/family/session.rs"}"#.to_string(),
            }],
        },
        PromptMessage {
            role: Role::Tool,
            content: Some(
                "pub(crate) struct SavedPrefix<S, P> { pub(crate) key: Vec<u32>, \
                 pub(crate) pending: u32, pub(crate) path: P, pub(crate) state: S }"
                    .to_string(),
            ),
            reasoning_content: None,
            name: None,
            tool_call_id: Some("call-7f3a".to_string()),
            tool_calls: Vec::new(),
        },
        msg(
            Role::Assistant,
            "The save already snapshots the per-layer state beside the KV rows \
             (SavedPrefix.state), so a parked conversation that serializes both \
             restores without re-running the prefix through any layer. I will \
             keep the charge accounting on the store side.",
        ),
        msg(
            Role::User,
            "Good. Now measure what a 128K conversation costs to park at f16 \
             before you wire it in — do not guess the budget.",
        ),
        msg(Role::Developer, DEVELOPER_ROW),
        msg(
            Role::User,
            "Also check that the q8 park restores into the q8 profile only.",
        ),
    ]
}

fn render(template: &ChatTemplate, msgs: &[PromptMessage], tools: &[ToolSpec]) -> String {
    template
        .render(msgs, tools, Thinking::Default)
        .expect("render")
}

/// The generation boundaries of the transcript: the message slices the
/// server actually renders and sends to the model (history plus the rows
/// that arrived since the last generation). Turn 1 renders [system,
/// task]; turn 2 renders through the tool result; turn 3 renders through
/// the developer row and the follow-up.
const GENERATION_BOUNDARIES: [usize; 3] = [2, 4, 7];

#[test]
fn flint_prompt_prefix_is_append_stable_across_turns() {
    let template = ChatTemplate::from_str(FLINT_TEMPLATE).expect("pinned template parses");
    let msgs = coding_transcript();
    let mut previous = render(&template, &msgs[..GENERATION_BOUNDARIES[0]], &[]);
    for &end in GENERATION_BOUNDARIES[1..].iter() {
        let current = render(&template, &msgs[..end], &[]);
        assert!(
            current.starts_with(&previous),
            "turn ending at message {end}: rendered prompt is not append-stable \
             at byte {} of the previous turn's prompt",
            previous.len()
        );
        assert!(
            previous.ends_with(ASSISTANT_DELIMITER),
            "previous turn's prompt (the slice ending before message {end}) does \
             not end at the assistant delimiter, so the shared bytes are not \
             provably a token prefix"
        );
        previous = current;
    }
}

#[test]
fn assistant_delimiter_is_one_added_special_token_in_the_pin() {
    // The token-prefix property rests on the delimiter being ONE added
    // (special) token in the pinned vocabulary: the AddedVocabulary
    // splits it out of raw text atomically, so no BPE merge crosses it
    // and a byte prefix ending at the delimiter ends at a token
    // boundary. This check reads tokenizer_config.json's
    // added_tokens_decoder as the always-run proxy; the vocabulary
    // itself is tokenizer.json, sha-pinned by flint_tests. A pin whose
    // added tokens do not carry the delimiter as exactly one special
    // token fails here.
    let config: serde_json::Value =
        serde_json::from_str(TOKENIZER_CONFIG).expect("pinned tokenizer config parses");
    let added = config["added_tokens_decoder"]
        .as_object()
        .expect("added_tokens_decoder is a map");
    let hits: Vec<_> = added
        .iter()
        .filter(|(_, token)| token["content"] == ASSISTANT_DELIMITER)
        .collect();
    assert_eq!(
        hits.len(),
        1,
        "exactly one added token must carry the delimiter"
    );
    let (_, token) = hits[0];
    assert_eq!(
        token["special"],
        serde_json::Value::Bool(true),
        "the delimiter must be a special token"
    );
}

#[test]
fn flint_render_is_deterministic_across_repeat_renders() {
    let template = ChatTemplate::from_str(FLINT_TEMPLATE).expect("pinned template parses");
    let msgs = coding_transcript();
    let a = render(&template, &msgs, &[]);
    let b = render(&template, &msgs, &[]);
    assert_eq!(a, b);
    assert!(a.contains("<tool_call>read_file"));
    assert!(a.contains("<think>"));
    assert!(a.contains("<tool_response>"));
}

#[test]
fn prefix_gate_detects_history_and_template_mutations() {
    // The gate must FAIL on a history edit, and must still fail when
    // the template itself changes; both are caught by the same prefix
    // assertion (what real drift detection is: see the note below).
    let template = ChatTemplate::from_str(FLINT_TEMPLATE).expect("pinned template parses");
    let msgs = coding_transcript();
    let turn3 = render(&template, &msgs[..3], &[]);
    // History edit: a rewritten user row rewrites bytes inside the prefix.
    let mut edited = msgs.clone();
    edited[1].content = Some("Actually, skip the loop question entirely.".to_string());
    let mutated = render(&template, &edited[..3], &[]);
    assert!(
        !mutated.starts_with(&turn3),
        "a history edit must break prefix stability"
    );
    // Template drift: renaming the delimiter changes the template, not
    // the history — the gate must still fail. Real drift detection is
    // the template hash pin on the key; this case only proves the
    // prefix gate itself can fail on template change.
    let drifted =
        ChatTemplate::from_str(&FLINT_TEMPLATE.replace(ASSISTANT_DELIMITER, "<|assistant_x|>"))
            .expect("mutated template still parses");
    let drifted_render = render(&drifted, &msgs[..3], &[]);
    assert!(
        !drifted_render.starts_with(&turn3),
        "template drift must break prefix stability"
    );
}

#[test]
fn tool_preamble_rows_append_stably_and_developer_row_survives() {
    // Real coding-agent turns carry tools: the template inserts the tool
    // preamble after the system message, so the preamble path needs its
    // own stability check. The developer row sits inside the third
    // slice; its bytes must still appear unchanged in that render.
    let template = ChatTemplate::from_str(FLINT_TEMPLATE).expect("pinned template parses");
    let msgs = coding_transcript();
    let tools = vec![ToolSpec(serde_json::json!({
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read one file from the repository",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"]
            }
        }
    }))];
    let first = render(&template, &msgs[..GENERATION_BOUNDARIES[0]], &tools);
    let second = render(&template, &msgs[..GENERATION_BOUNDARIES[1]], &tools);
    let third = render(&template, &msgs[..GENERATION_BOUNDARIES[2]], &tools);
    assert!(second.starts_with(&first));
    assert!(third.starts_with(&second));
    assert!(first.ends_with(ASSISTANT_DELIMITER));
    assert!(second.ends_with(ASSISTANT_DELIMITER));
    assert!(
        first.contains("read_file"),
        "the tool preamble renders the tool name"
    );
    assert!(
        third.contains(DEVELOPER_ROW),
        "the developer row bytes survive unchanged"
    );
}

#[ignore = "requires FLINT_MODEL_DIR with the pinned publisher tokenizer"]
#[test]
fn flint_prompt_prefix_is_a_token_prefix_with_the_real_tokenizer() {
    // The byte-prefix property with the real pinned tokenizer: the
    // encoded ids of every turn are a prefix of the next turn's ids, and
    // the delimiter encodes to exactly one added token id.
    let bundle = flint_tests::bundle();
    let template = ChatTemplate::from_str(FLINT_TEMPLATE).expect("pinned template parses");
    let msgs = coding_transcript();
    let previous = render(&template, &msgs[..GENERATION_BOUNDARIES[0]], &[]);
    let mut previous_ids = bundle.tokenizer().encode(&previous).expect("encode");
    for &end in GENERATION_BOUNDARIES[1..].iter() {
        let current = render(&template, &msgs[..end], &[]);
        let ids = bundle.tokenizer().encode(&current).expect("encode");
        assert!(
            ids.len() >= previous_ids.len() && ids[..previous_ids.len()] == previous_ids[..],
            "turn ending at message {end}: token ids are not append-stable"
        );
        previous_ids = ids;
    }
    let delimiter_ids = bundle
        .tokenizer()
        .encode(ASSISTANT_DELIMITER)
        .expect("encode");
    assert_eq!(delimiter_ids.len(), 1, "the delimiter is one token");
    assert!(bundle.tokenizer().is_added_token_id(delimiter_ids[0]));
}
