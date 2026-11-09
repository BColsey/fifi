//! The tokenizer routes: `POST /v1/token/encode`, `POST /v1/token/decode`, and
//! `POST /v1/apply-template`.
//!
//! They answer from the tokenizer bundle of the resolved engine and queue no job, so they work in
//! every engine state, as `count_tokens` does, and each served request writes one request-log
//! line, as `count_tokens` does. A chat request renders through the same bundle call that the
//! chat route uses, so its ids are the ids a completion would send to the engine.

use super::chat::read_body;
use super::error::ServerError;
use super::messages::{bad, optional_string, parse_messages, parse_tool_choice, parse_tools};
use super::reqlog::{RequestLog, default_clock};
use super::state::{AppState, EngineEntry};
use super::validate::{check_constrained_output, json_object, thinking};
use crate::EngError;
use crate::tokenizer::{PromptMessage, Thinking, ToolSpec};
use axum::Json;
use axum::body::Body;
use axum::extract::State;
use serde_json::{Map, Value, json};
use std::sync::Arc;

/// The fields that the chat template renders.
struct ChatPrompt {
    messages: Vec<PromptMessage>,
    tools: Vec<ToolSpec>,
    thinking: Thinking,
}

impl ChatPrompt {
    /// Reads the fields with the checks of the chat route, so a request that the chat route
    /// refuses is refused here too.
    fn parse(object: &Map<String, Value>) -> Result<Self, ServerError> {
        let messages = parse_messages(object.get("messages"))?;
        let mut tools = parse_tools(object.get("tools"))?;
        if parse_tool_choice(object.get("tool_choice"))? {
            tools.clear();
        }
        check_constrained_output(object)?;
        Ok(Self {
            messages,
            tools,
            thinking: thinking(object)?,
        })
    }
}

/// What `encode` tokenizes: the text as written, or the chat prompt rendered by the template.
enum Source {
    Text(String),
    Chat(ChatPrompt),
}

fn encode_source(object: &Map<String, Value>) -> Result<Source, ServerError> {
    match (object.get("text"), object.get("messages")) {
        (Some(_), Some(_)) => Err(bad("text and messages cannot both be set")),
        (Some(Value::String(text)), None) => Ok(Source::Text(text.clone())),
        (Some(_), None) => Err(bad("text must be a string")),
        (None, Some(_)) => Ok(Source::Chat(ChatPrompt::parse(object)?)),
        (None, None) => Err(bad("text or messages is required")),
    }
}

/// The engine that `x-session`, `x-engine`, or `model` selects.
fn entry_of<'a>(
    state: &'a AppState,
    object: &Map<String, Value>,
) -> Result<&'a EngineEntry, ServerError> {
    let session = optional_string(object, "x-session", "")?;
    let engine = optional_string(object, "x-engine", "")?;
    let model = optional_string(object, "model", "")?;
    state.resolve(session.as_deref(), engine.as_deref(), model.as_deref())
}

/// Writes the request-log line of a served request that ran no job. `tokens` is the token count
/// the request worked on, 0 when it tokenized nothing. Like `count_tokens`, the line has neither
/// finish reason nor timings, and a failed request writes none.
fn log_served(state: &AppState, entry: &EngineEntry, tokens: usize, endpoint: &'static str) {
    let tokens = u32::try_from(tokens).unwrap_or(u32::MAX);
    let mut log = RequestLog::new(
        state.request_log(),
        default_clock(),
        &entry.model,
        false,
        tokens,
        "POST",
        endpoint,
    );
    log.set_id(&state.next_completion_id());
    log.counted();
}

/// Runs tokenizer work off the async threads, as the chat route does.
pub(super) async fn blocking<T: Send + 'static>(
    work: impl FnOnce() -> Result<T, EngError> + Send + 'static,
) -> Result<T, ServerError> {
    let result = tokio::task::spawn_blocking(work).await.map_err(|error| {
        ServerError::Eng(EngError::Backend {
            operation: "tokenize",
            message: error.to_string(),
        })
    })?;
    result.map_err(ServerError::from)
}

/// `POST /v1/token/encode`: the ids and the count of `text` or of the rendered `messages`.
pub(crate) async fn encode(
    State(state): State<Arc<AppState>>,
    body: Body,
) -> Result<Json<Value>, ServerError> {
    let object = json_object(&read_body(body).await?)?;
    let source = encode_source(&object)?;
    let entry = entry_of(&state, &object)?;
    let bundle = Arc::clone(&entry.bundle);
    let ids = blocking(move || match source {
        Source::Text(text) => bundle.tokenizer().encode(&text),
        Source::Chat(chat) => bundle.encode_prompt(&chat.messages, &chat.tools, chat.thinking),
    })
    .await?;
    log_served(&state, entry, ids.len(), "/v1/token/encode");
    Ok(Json(json!({"tokens": ids, "length": ids.len()})))
}

fn token_array(object: &Map<String, Value>) -> Result<&Vec<Value>, ServerError> {
    object
        .get("tokens")
        .ok_or_else(|| bad("tokens is required"))?
        .as_array()
        .ok_or_else(|| bad("tokens must be an array"))
}

fn token_ids(items: &[Value]) -> Result<Vec<u32>, ServerError> {
    items
        .iter()
        .enumerate()
        .map(|(index, item)| {
            item.as_u64()
                .and_then(|id| u32::try_from(id).ok())
                .ok_or_else(|| bad(format!("tokens[{index}] must be a non-negative integer")))
        })
        .collect()
}

/// `POST /v1/token/decode`: the text of `tokens`. Special tokens are left out of the text. More
/// ids than the context limit of the engine are refused before they are converted, because no
/// prompt or completion of that engine is longer.
pub(crate) async fn decode(
    State(state): State<Arc<AppState>>,
    body: Body,
) -> Result<Json<Value>, ServerError> {
    let object = json_object(&read_body(body).await?)?;
    let items = token_array(&object)?;
    let entry = entry_of(&state, &object)?;
    if items.len() > entry.context_limit as usize {
        return Err(bad(format!(
            "tokens holds {} ids, more than the context limit of {} tokens",
            items.len(),
            entry.context_limit
        )));
    }
    let ids = token_ids(items)?;
    let bundle = Arc::clone(&entry.bundle);
    let limit = bundle.vocab_size();
    if let Some(index) = ids.iter().position(|&id| id >= limit) {
        return Err(bad(format!(
            "tokens[{index}] is not in the vocabulary of {limit} tokens"
        )));
    }
    let count = ids.len();
    let text = blocking(move || bundle.tokenizer().decode(&ids)).await?;
    log_served(&state, entry, count, "/v1/token/decode");
    Ok(Json(json!({"text": text})))
}

/// `POST /v1/apply-template`: the prompt text that the chat template makes of `messages`, with
/// the generation prompt. Nothing is generated.
pub(crate) async fn apply_template(
    State(state): State<Arc<AppState>>,
    body: Body,
) -> Result<Json<Value>, ServerError> {
    let object = json_object(&read_body(body).await?)?;
    let chat = ChatPrompt::parse(&object)?;
    let entry = entry_of(&state, &object)?;
    let bundle = Arc::clone(&entry.bundle);
    let prompt =
        blocking(move || bundle.render_prompt(&chat.messages, &chat.tools, chat.thinking)).await?;
    log_served(&state, entry, 0, "/v1/apply-template");
    Ok(Json(json!({"prompt": prompt})))
}

#[cfg(test)]
mod tests {
    use crate::sched::worker::{Engine, EngineState};
    use crate::server::contract::assert_schema;
    use crate::server::router::router;
    use crate::server::state::AppState;
    use crate::server::testkit;
    use axum::Router;
    use axum::http::StatusCode;
    use serde_json::{Value, json};
    use std::sync::Arc;

    const PROMPT_OF_HI: &str = "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n";

    fn app(engine: &Engine) -> Router {
        testkit::wait_for_state(&engine.handle(), EngineState::Ready);
        let entry = testkit::entry("e1", "model-a", engine, 4_096);
        router(Arc::new(AppState::new(testkit::API_KEY, vec![entry])))
    }

    /// A router whose one engine renders with the pinned Qwen template.
    fn pinned_app(engine: &Engine) -> Router {
        testkit::wait_for_state(&engine.handle(), EngineState::Ready);
        let mut entry = testkit::entry("e1", "model-a", engine, 4_096);
        entry.bundle = testkit::bundle_with_pinned_template();
        router(Arc::new(AppState::new(testkit::API_KEY, vec![entry])))
    }

    async fn post(app: &Router, uri: &str, body: &Value) -> (StatusCode, Value) {
        let (status, _headers, bytes) = testkit::send(app, testkit::post_json(uri, body)).await;
        (status, testkit::json_of(&bytes))
    }

    fn hi() -> Value {
        json!([{"role": "user", "content": "hi"}])
    }

    fn error_code(body: &Value) -> &str {
        body["error"]["code"].as_str().expect("error code")
    }

    /// A router whose one engine renders every chat to the empty string.
    fn empty_template_app(engine: &Engine) -> Router {
        testkit::wait_for_state(&engine.handle(), EngineState::Ready);
        let mut entry = testkit::entry("e1", "model-a", engine, 4_096);
        entry.bundle = testkit::bundle_with_template("{% if false %}x{% endif %}");
        router(Arc::new(AppState::new(testkit::API_KEY, vec![entry])))
    }

    #[tokio::test]
    async fn an_empty_render_returns_the_same_400_on_encode_and_apply_template() {
        let engine = testkit::engine(Vec::new());
        let app = empty_template_app(&engine);
        for uri in ["/v1/token/encode", "/v1/apply-template"] {
            let body = json!({"model": "model-a", "messages": hi()});
            let (status, reply) = post(&app, uri, &body).await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{uri} {reply}");
            assert_eq!(error_code(&reply), "bad_request", "{uri}");
            let message = reply["error"]["message"].as_str().expect("message");
            assert_eq!(message, "rendered prompt is empty", "{uri}");
        }
    }

    #[tokio::test]
    async fn a_body_over_16_mib_returns_400_on_every_route() {
        let engine = testkit::engine(Vec::new());
        let app = app(&engine);
        let cap = 16 * 1024 * 1024;
        for uri in ["/v1/token/encode", "/v1/token/decode", "/v1/apply-template"] {
            let head = r#"{"model":"model-a","pad":""#;
            let tail = r#""}"#;
            let fill = "x".repeat(cap - head.len() - tail.len() + 1);
            let body = format!("{head}{fill}{tail}").into_bytes();
            let request = testkit::post_bytes(uri, Some(testkit::API_KEY), body);
            let (status, _headers, bytes) = testkit::send(&app, request).await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{uri}");
            let reply = testkit::json_of(&bytes);
            let message = reply["error"]["message"].as_str().expect("message");
            assert!(message.contains("16 MiB"), "{uri}: {message}");
        }
    }

    #[tokio::test]
    async fn the_text_of_a_special_token_encodes_to_that_token() {
        let engine = testkit::engine(Vec::new());
        let body = json!({"model": "model-a", "text": "<|im_start|>hi<|im_end|>"});
        let (status, reply) = post(&app(&engine), "/v1/token/encode", &body).await;
        assert_eq!(status, StatusCode::OK, "{reply}");
        assert_eq!(reply["tokens"], json!([257, 104, 105, 256]));
    }

    #[tokio::test]
    async fn decode_refuses_more_ids_than_the_context_limit() {
        let engine = testkit::engine(Vec::new());
        let app = app(&engine);
        let at_limit = json!({"model": "model-a", "tokens": vec![104; 4_096]});
        let (status, reply) = post(&app, "/v1/token/decode", &at_limit).await;
        assert_eq!(status, StatusCode::OK, "{reply}");
        let over = json!({"model": "model-a", "tokens": vec![104; 4_097]});
        let (status, reply) = post(&app, "/v1/token/decode", &over).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(error_code(&reply), "bad_request");
        let message = reply["error"]["message"].as_str().expect("message");
        assert!(
            message.contains("4097") && message.contains("4096"),
            "{message}"
        );
    }

    #[tokio::test]
    async fn sampling_and_stream_fields_are_ignored() {
        let engine = testkit::engine(Vec::new());
        let body = json!({
            "model": "model-a", "messages": hi(),
            "temperature": 99, "top_p": 5, "top_k": -1, "max_tokens": 0, "stop": 5, "stream": "x"
        });
        let (status, reply) = post(&app(&engine), "/v1/apply-template", &body).await;
        assert_eq!(status, StatusCode::OK, "{reply}");
        assert_eq!(reply["prompt"], PROMPT_OF_HI);
    }

    #[tokio::test]
    async fn encode_returns_the_ids_and_the_count_of_a_text() {
        let engine = testkit::engine(Vec::new());
        let body = json!({"model": "model-a", "text": "hi!"});
        let (status, reply) = post(&app(&engine), "/v1/token/encode", &body).await;
        assert_eq!(status, StatusCode::OK, "{reply}");
        assert_eq!(reply, json!({"tokens": [104, 105, 33], "length": 3}));
    }

    #[tokio::test]
    async fn encode_of_messages_returns_the_ids_the_chat_route_sends_to_the_engine() {
        let (engine, seen) = testkit::probe_engine();
        let app = app(&engine);
        let chat = json!({"model": "model-a", "messages": hi(), "max_tokens": 1});
        let (status, _) = post(&app, "/v1/chat/completions", &chat).await;
        assert_eq!(status, StatusCode::OK);
        let body = json!({"model": "model-a", "messages": hi()});
        let (status, reply) = post(&app, "/v1/token/encode", &body).await;
        assert_eq!(status, StatusCode::OK, "{reply}");
        let sent = seen.lock().expect("prompt lock").clone();
        assert_eq!(reply["tokens"], json!(sent));
        assert_eq!(reply["length"], 21);
        let count = json!({"model": "model-a", "messages": hi()});
        let (_, counted) = post(&app, "/v1/messages/count_tokens", &count).await;
        assert_eq!(counted["input_tokens"], reply["length"]);
    }

    #[tokio::test]
    async fn decode_returns_the_text_of_the_ids_and_omits_special_tokens() {
        let engine = testkit::engine(Vec::new());
        let app = app(&engine);
        let body = json!({"model": "model-a", "tokens": [104, 105, 33]});
        let (status, reply) = post(&app, "/v1/token/decode", &body).await;
        assert_eq!(status, StatusCode::OK, "{reply}");
        assert_eq!(reply, json!({"text": "hi!"}));
        let special = json!({"model": "model-a", "tokens": [257, 104, 256]});
        let (_, reply) = post(&app, "/v1/token/decode", &special).await;
        assert_eq!(reply["text"], "h");
        let empty = json!({"model": "model-a", "tokens": []});
        let (_, reply) = post(&app, "/v1/token/decode", &empty).await;
        assert_eq!(reply["text"], "");
    }

    #[tokio::test]
    async fn decode_inverts_encode_for_plain_text() {
        let engine = testkit::engine(Vec::new());
        let app = app(&engine);
        let text = "Plain text, 123: ok.\nSecond line.";
        let (_, encoded) = post(
            &app,
            "/v1/token/encode",
            &json!({"model": "model-a", "text": text}),
        )
        .await;
        let decode = json!({"model": "model-a", "tokens": encoded["tokens"]});
        let (_, decoded) = post(&app, "/v1/token/decode", &decode).await;
        assert_eq!(decoded["text"], text);
    }

    #[tokio::test]
    async fn apply_template_returns_the_rendered_prompt() {
        let engine = testkit::engine(Vec::new());
        let body = json!({"model": "model-a", "messages": hi()});
        let (status, reply) = post(&app(&engine), "/v1/apply-template", &body).await;
        assert_eq!(status, StatusCode::OK, "{reply}");
        assert_eq!(reply, json!({"prompt": PROMPT_OF_HI}));
    }

    #[tokio::test]
    async fn encode_of_messages_encodes_the_applied_template() {
        let engine = testkit::engine(Vec::new());
        let app = app(&engine);
        let body = json!({"model": "model-a", "messages": hi()});
        let (_, applied) = post(&app, "/v1/apply-template", &body).await;
        let text = json!({"model": "model-a", "text": applied["prompt"]});
        let (_, from_text) = post(&app, "/v1/token/encode", &text).await;
        let (_, from_messages) = post(&app, "/v1/token/encode", &body).await;
        assert_eq!(from_text, from_messages);
    }

    #[tokio::test]
    async fn apply_template_follows_the_thinking_switches_of_the_chat_route() {
        let engine = testkit::engine(Vec::new());
        let app = app(&engine);
        let render = |extra: Value| {
            let mut body = json!({"model": "model-a", "messages": hi()});
            for (key, value) in extra.as_object().expect("object") {
                body[key] = value.clone();
            }
            body
        };
        let on = render(json!({"x-thinking": "on"}));
        let (_, reply) = post(&app, "/v1/apply-template", &on).await;
        assert_eq!(reply["prompt"], format!("{PROMPT_OF_HI}<think>\n"));
        let alias = render(json!({"chat_template_kwargs": {"enable_thinking": true}}));
        let (_, reply) = post(&app, "/v1/apply-template", &alias).await;
        assert_eq!(reply["prompt"], format!("{PROMPT_OF_HI}<think>\n"));
        let off = render(json!({"x-thinking": "off"}));
        let (_, reply) = post(&app, "/v1/apply-template", &off).await;
        assert_eq!(reply["prompt"], PROMPT_OF_HI);
    }

    #[tokio::test]
    async fn apply_template_renders_tools_unless_tool_choice_is_none() {
        let engine = testkit::engine(Vec::new());
        let app = pinned_app(&engine);
        let tools = json!([{"type": "function", "function": {"name": "add_numbers"}}]);
        let with = json!({"model": "model-a", "messages": hi(), "tools": tools});
        let (status, reply) = post(&app, "/v1/apply-template", &with).await;
        assert_eq!(status, StatusCode::OK, "{reply}");
        let prompt = reply["prompt"].as_str().expect("prompt");
        assert!(
            prompt.contains("# Tools") && prompt.contains("add_numbers"),
            "{prompt}"
        );
        let without = json!({
            "model": "model-a", "messages": hi(), "tools": tools, "tool_choice": "none"
        });
        let (_, reply) = post(&app, "/v1/apply-template", &without).await;
        let prompt = reply["prompt"].as_str().expect("prompt");
        assert!(!prompt.contains("add_numbers"), "{prompt}");
    }

    #[tokio::test]
    async fn x_engine_selects_the_engine_and_takes_precedence_over_model() {
        let engine = testkit::engine(Vec::new());
        let app = app(&engine);
        let body = json!({"model": "unknown", "x-engine": "e1", "text": "hi"});
        let (status, reply) = post(&app, "/v1/token/encode", &body).await;
        assert_eq!(status, StatusCode::OK, "{reply}");
        assert_eq!(reply["length"], 2);
    }

    #[tokio::test]
    async fn an_unknown_engine_model_or_session_returns_404_on_every_route() {
        let engine = testkit::engine(Vec::new());
        let app = app(&engine);
        let routes = [
            ("/v1/token/encode", json!({"text": "hi"})),
            ("/v1/token/decode", json!({"tokens": [104]})),
            ("/v1/apply-template", json!({"messages": hi()})),
        ];
        for (uri, base) in routes {
            for selector in [
                json!({"model": "other"}),
                json!({"x-engine": "e9"}),
                json!({"x-session": "s1"}),
            ] {
                let mut body = base.clone();
                for (key, value) in selector.as_object().expect("object") {
                    body[key] = value.clone();
                }
                let (status, reply) = post(&app, uri, &body).await;
                assert_eq!(status, StatusCode::NOT_FOUND, "{uri} {selector}");
                assert_eq!(error_code(&reply), "not_found", "{uri} {selector}");
                assert_schema("Error", &reply);
            }
        }
    }

    #[tokio::test]
    async fn a_request_without_an_engine_selector_returns_400() {
        let engine = testkit::engine(Vec::new());
        let (status, reply) = post(&app(&engine), "/v1/token/encode", &json!({"text": "hi"})).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(error_code(&reply), "bad_request");
    }

    #[tokio::test]
    async fn a_get_on_a_new_path_returns_the_json_404() {
        let engine = testkit::engine(Vec::new());
        let app = app(&engine);
        for uri in ["/v1/token/encode", "/v1/token/decode", "/v1/apply-template"] {
            let request = testkit::get(uri, Some(testkit::API_KEY));
            let (status, _headers, bytes) = testkit::send(&app, request).await;
            assert_eq!(status, StatusCode::NOT_FOUND, "{uri}");
            assert_eq!(error_code(&testkit::json_of(&bytes)), "not_found");
        }
    }

    #[tokio::test]
    async fn the_routes_need_the_bearer_key() {
        let engine = testkit::engine(Vec::new());
        let app = app(&engine);
        for uri in ["/v1/token/encode", "/v1/token/decode", "/v1/apply-template"] {
            for key in [None, Some("sk-wrong")] {
                let request = testkit::post_bytes(uri, key, b"{}".to_vec());
                let (status, _headers, bytes) = testkit::send(&app, request).await;
                assert_eq!(status, StatusCode::UNAUTHORIZED, "{uri}");
                let body = testkit::json_of(&bytes);
                assert_schema("Error", &body);
                assert_eq!(error_code(&body), "unauthorized");
            }
        }
    }

    #[tokio::test]
    async fn malformed_requests_return_400_that_names_the_field() {
        let engine = testkit::engine(Vec::new());
        let app = app(&engine);
        let cases = [
            (
                "/v1/token/encode",
                json!({"model": "model-a"}),
                "text or messages",
            ),
            (
                "/v1/token/encode",
                json!({"model": "model-a", "text": "a", "messages": hi()}),
                "cannot both be set",
            ),
            (
                "/v1/token/encode",
                json!({"model": "model-a", "text": 5}),
                "text must be a string",
            ),
            (
                "/v1/token/encode",
                json!({"model": "model-a", "messages": []}),
                "messages",
            ),
            (
                "/v1/token/encode",
                json!({"model": "model-a", "messages": hi(), "x-thinking": "maybe"}),
                "x-thinking",
            ),
            (
                "/v1/token/decode",
                json!({"model": "model-a"}),
                "tokens is required",
            ),
            (
                "/v1/token/decode",
                json!({"model": "model-a", "tokens": "ab"}),
                "must be an array",
            ),
            (
                "/v1/token/decode",
                json!({"model": "model-a", "tokens": [1, -2]}),
                "tokens[1]",
            ),
            (
                "/v1/token/decode",
                json!({"model": "model-a", "tokens": [1.5]}),
                "tokens[0]",
            ),
            (
                "/v1/token/decode",
                json!({"model": "model-a", "tokens": [4_294_967_296_u64]}),
                "tokens[0]",
            ),
            (
                "/v1/token/decode",
                json!({"model": "model-a", "tokens": [104, 258]}),
                "tokens[1]",
            ),
            (
                "/v1/apply-template",
                json!({"model": "model-a"}),
                "messages is required",
            ),
            (
                "/v1/apply-template",
                json!({"model": "model-a", "messages": hi(), "tool_choice": "required"}),
                "tool_choice",
            ),
            (
                "/v1/apply-template",
                json!({"model": "model-a", "messages": hi(), "response_format": {"type": "json_object"}}),
                "response_format",
            ),
            (
                "/v1/token/encode",
                json!({"model": "model-a", "messages": hi(), "response_format": {"type": "json_object"}}),
                "response_format",
            ),
            (
                "/v1/apply-template",
                json!({"model": "model-a", "messages": [{"role": "robot", "content": "x"}]}),
                "messages[0].role",
            ),
        ];
        for (uri, body, expected) in cases {
            let (status, reply) = post(&app, uri, &body).await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{uri} {body}");
            assert_eq!(error_code(&reply), "bad_request", "{uri} {body}");
            let message = reply["error"]["message"].as_str().expect("message");
            assert!(message.contains(expected), "{uri} {body}: {message}");
            assert_schema("Error", &reply);
        }
    }

    #[tokio::test]
    async fn a_body_that_is_not_a_json_object_returns_400() {
        let engine = testkit::engine(Vec::new());
        let app = app(&engine);
        for uri in ["/v1/token/encode", "/v1/token/decode", "/v1/apply-template"] {
            for raw in [&b"[1]"[..], &b"{\"text\":"[..]] {
                let request = testkit::post_bytes(uri, Some(testkit::API_KEY), raw.to_vec());
                let (status, _headers, bytes) = testkit::send(&app, request).await;
                assert_eq!(status, StatusCode::BAD_REQUEST, "{uri}");
                assert_eq!(error_code(&testkit::json_of(&bytes)), "bad_request");
            }
        }
    }

    #[tokio::test]
    async fn the_routes_answer_while_the_engine_is_loading_and_after_it_failed() {
        let (engine, gate) = testkit::gated_engine(Vec::new());
        let entry = testkit::entry("e1", "model-a", &engine, 4_096);
        let loading = router(Arc::new(AppState::new(testkit::API_KEY, vec![entry])));
        let body = json!({"model": "model-a", "text": "hi"});
        let (status, reply) = post(&loading, "/v1/token/encode", &body).await;
        assert_eq!(status, StatusCode::OK, "{reply}");
        drop(gate);
        let failed = testkit::failing_engine();
        let entry = testkit::entry("e1", "model-a", &failed, 4_096);
        testkit::wait_for_state(&entry.handle, EngineState::Error);
        let broken = router(Arc::new(AppState::new(testkit::API_KEY, vec![entry])));
        let (status, reply) = post(&broken, "/v1/token/encode", &body).await;
        assert_eq!(status, StatusCode::OK, "{reply}");
    }

    #[tokio::test]
    async fn the_routes_run_no_engine_job() {
        let probes = testkit::probes(std::time::Duration::ZERO);
        let app = app(&probes.engine);
        post(
            &app,
            "/v1/token/encode",
            &json!({"model": "model-a", "messages": hi()}),
        )
        .await;
        post(
            &app,
            "/v1/apply-template",
            &json!({"model": "model-a", "messages": hi()}),
        )
        .await;
        post(
            &app,
            "/v1/token/decode",
            &json!({"model": "model-a", "tokens": [104]}),
        )
        .await;
        assert!(probes.prompt.lock().expect("prompt lock").is_empty());
        assert_eq!(probes.decodes.load(std::sync::atomic::Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn the_existing_routes_still_answer() {
        let engine = testkit::engine(testkit::script("ok"));
        let app = app(&engine);
        let chat = json!({"model": "model-a", "messages": hi()});
        let (status, reply) = post(&app, "/v1/chat/completions", &chat).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(reply["choices"][0]["message"]["content"], "ok");
        let (status, _headers, _bytes) =
            testkit::send(&app, testkit::get("/v1/models", Some(testkit::API_KEY))).await;
        assert_eq!(status, StatusCode::OK);
    }
}
