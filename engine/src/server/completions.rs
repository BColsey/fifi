//! The raw completion handler: `POST /v1/completions`.
//!
//! The prompt is one string that the tokenizer encodes as written: no chat template, no start
//! token. The job, the stop strings, the usage, and the stream go through the same admission and
//! reply pump as a chat completion; only the request fields and the reply shape differ.

use super::chat::{Admission, ReplyMode, Submitted, admit, read_body};
use super::error::ServerError;
use super::messages::{bad, optional_string};
use super::reply::collect;
use super::sse;
use super::state::AppState;
use super::tokenize::blocking;
use super::validate::{
    RequestedSampling, flag, json_object, requested_max_tokens, requested_sampling, stops,
};
use super::wire::Completion;
use crate::family::sampler::SamplingParams;
use crate::tokenizer::{OutputAssembler, PromptBoundaries};
use axum::Json;
use axum::body::Body;
use axum::extract::State;
use axum::response::{IntoResponse, Response};
use serde_json::{Map, Value};
use std::sync::Arc;

/// Fields of the OpenAI completions API that the server does not implement and that would change
/// the reply or its shape: a request that sets one to a real value gets `400`.
///
/// Other fields of that API are not read and pass unchecked, exactly as on the chat route, whose
/// parser reads no `seed`, `presence_penalty`, `frequency_penalty`, `logit_bias`,
/// `stream_options` or `user` either: the contract permits extra request members. They have no
/// effect. In particular, a stream always ends with a usage chunk, whatever `stream_options` says.
const UNSUPPORTED: [&str; 3] = ["suffix", "logprobs", "best_of"];

/// A completion request that passed validation.
struct CompletionInput {
    model: Option<String>,
    engine: Option<String>,
    session: Option<String>,
    prompt: String,
    stops: Vec<String>,
    sampling: RequestedSampling,
    max_tokens: u32,
    stream: bool,
}

/// The route handler. Every failure before the reply is a JSON error.
pub(crate) async fn completions(State(state): State<Arc<AppState>>, body: Body) -> Response {
    match reply(&state, body).await {
        Ok(response) => response,
        Err(error) => error.into_response(),
    }
}

async fn reply(state: &AppState, body: Body) -> Result<Response, ServerError> {
    let input = parse_completion(&read_body(body).await?)?;
    let stream = input.stream;
    let mut submitted = submit(state, input).await?;
    if stream {
        let keep_alive = state.keep_alive();
        return Ok(sse::respond_text(
            submitted.head,
            submitted.running,
            keep_alive,
        ));
    }
    let (parts, finish, usage) = collect(&mut submitted.running).await?;
    let completion = Completion::reply(&submitted.head, parts.content, finish, usage);
    Ok(Json(completion).into_response())
}

fn parse_completion(body: &[u8]) -> Result<CompletionInput, ServerError> {
    let object = &json_object(body)?;
    let prompt = prompt_text(object)?;
    check_supported(object)?;
    Ok(CompletionInput {
        model: optional_string(object, "model", "")?,
        engine: optional_string(object, "x-engine", "")?,
        session: optional_string(object, "x-session", "")?,
        prompt,
        stops: completion_stops(object)?,
        sampling: requested_sampling(object)?,
        max_tokens: requested_max_tokens(object)?,
        stream: flag(object, "stream")?,
    })
}

/// Reads `prompt`: one non-empty string. A list of strings or of token ids is refused.
fn prompt_text(object: &Map<String, Value>) -> Result<String, ServerError> {
    match object.get("prompt") {
        None => Err(bad("prompt is required")),
        Some(Value::String(text)) if text.is_empty() => Err(bad("prompt must not be empty")),
        Some(Value::String(text)) => Ok(text.clone()),
        Some(Value::Array(_)) => Err(bad(
            "prompt must be a single string; a list of strings or of token ids is not supported",
        )),
        Some(_) => Err(bad("prompt must be a string")),
    }
}

/// Reads `stop` as the OpenAI API allows: one string or an array of strings.
fn completion_stops(object: &Map<String, Value>) -> Result<Vec<String>, ServerError> {
    match object.get("stop") {
        Some(Value::String(stop)) => Ok(vec![stop.clone()]),
        _ => stops(object),
    }
}

/// Refuses the fields that would change the reply and that the server does not implement.
fn check_supported(object: &Map<String, Value>) -> Result<(), ServerError> {
    if let Some(field) = UNSUPPORTED
        .into_iter()
        .find(|field| object.get(*field).is_some_and(|value| !value.is_null()))
    {
        return Err(bad(format!("{field} is not supported")));
    }
    if matches!(object.get("echo"), Some(Value::Bool(true))) {
        return Err(bad("echo is not supported"));
    }
    match object.get("n") {
        None | Some(Value::Null) => Ok(()),
        Some(value) if value.as_u64() == Some(1) => Ok(()),
        Some(_) => Err(bad("n must be 1: one choice is generated for each request")),
    }
}

/// Resolves the engine, tokenizes the prompt as written, and queues the job.
async fn submit(state: &AppState, input: CompletionInput) -> Result<Submitted, ServerError> {
    let entry = state.resolve(
        input.session.as_deref(),
        input.engine.as_deref(),
        input.model.as_deref(),
    )?;
    let sampling = SamplingParams::from_request(
        input.sampling.temperature,
        input.sampling.top_p,
        input.sampling.top_k,
        &entry.defaults,
    )?;
    let assembler = OutputAssembler::new(input.stops)?;
    let bundle = Arc::clone(&entry.bundle);
    let text = input.prompt;
    let prompt = blocking(move || bundle.tokenizer().encode(&text)).await?;
    // No boundary: a raw prompt has no message structure, so the engine neither saves nor
    // restores a prefix for it.
    let admission = Admission {
        prompt,
        boundaries: PromptBoundaries::default(),
        sampling,
        assembler,
        max_tokens: input.max_tokens,
        stream: input.stream,
        // The raw-completions route never arms capture; the field is a
        // chat extension.
        sample_trace: false,
        reply: ReplyMode::Raw,
        method: "POST",
        endpoint: "/v1/completions",
    };
    admit(state, entry, admission)
}

#[cfg(test)]
mod tests {
    use crate::EngError;
    use crate::family::sampler::SamplingParams;
    use crate::sched::job::{Job, QUEUE_CAPACITY};
    use crate::sched::worker::{Engine, EngineState};
    use crate::server::contract::assert_schema;
    use crate::server::router::router;
    use crate::server::state::AppState;
    use crate::server::testkit;
    use axum::Router;
    use axum::http::{StatusCode, header};
    use serde_json::{Value, json};
    use std::sync::Arc;

    fn app(engine: &Engine, context_limit: u32) -> Router {
        testkit::wait_for_state(&engine.handle(), EngineState::Ready);
        let entry = testkit::entry("e1", "model-a", engine, context_limit);
        router(Arc::new(AppState::new(testkit::API_KEY, vec![entry])))
    }

    fn request(extra: Value) -> Value {
        let mut body = json!({"model": "model-a", "prompt": "hi"});
        for (key, value) in extra.as_object().expect("object") {
            body[key] = value.clone();
        }
        body
    }

    async fn post(app: &Router, body: &Value) -> (StatusCode, Value) {
        let (status, _headers, bytes) =
            testkit::send(app, testkit::post_json("/v1/completions", body)).await;
        (status, testkit::json_of(&bytes))
    }

    fn error_code(body: &Value) -> &str {
        body["error"]["code"].as_str().expect("error code")
    }

    #[tokio::test]
    async fn a_completion_returns_the_text_and_the_usage() {
        let engine = testkit::engine(testkit::script("Hello!"));
        let (status, body) = post(&app(&engine, 4_096), &request(json!({}))).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["object"], "text_completion");
        assert_eq!(body["model"], "model-a");
        assert_eq!(body["engine"], "e1");
        assert!(
            body["id"]
                .as_str()
                .expect("id")
                .ends_with("0000000000000001")
        );
        assert!(body["created"].as_u64().expect("created") > 1_700_000_000);
        let choice = &body["choices"][0];
        assert_eq!(choice["index"], 0);
        assert_eq!(choice["text"], "Hello!");
        assert_eq!(choice["logprobs"], Value::Null);
        assert_eq!(choice["finish_reason"], "stop");
        // "hi" is two tokens, "Hello!" is six, and the completion count includes the EOS token.
        assert_eq!(body["usage"]["prompt_tokens"], 2);
        assert_eq!(body["usage"]["completion_tokens"], 7);
        assert_eq!(body["usage"]["total_tokens"], 9);
    }

    #[tokio::test]
    async fn the_engine_receives_the_prompt_without_a_template() {
        let (engine, seen) = testkit::probe_engine();
        let body = request(json!({"prompt": "Once upon\n", "max_tokens": 1}));
        let (status, reply) = post(&app(&engine, 4_096), &body).await;
        assert_eq!(status, StatusCode::OK, "{reply}");
        assert_eq!(
            *seen.lock().expect("prompt lock"),
            testkit::ids("Once upon\n")
        );
    }

    #[tokio::test]
    async fn the_sampling_fields_reach_the_executor() {
        let probes = testkit::probe_engine_with_sampling();
        let body = request(json!({"temperature": 0.5, "top_p": 0.9, "top_k": 7, "max_tokens": 1}));
        let (status, reply) = post(&app(&probes.engine, 4_096), &body).await;
        assert_eq!(status, StatusCode::OK, "{reply}");
        let observed = *probes.sampling.lock().expect("sampling lock");
        assert_eq!(
            observed,
            Some(SamplingParams::new(0.5, 0.9, 7).expect("params"))
        );
    }

    #[tokio::test]
    async fn max_tokens_and_the_context_limit_end_the_reply_with_length() {
        let engine = testkit::engine(testkit::script("abcdefgh"));
        let wide = app(&engine, 4_096);
        let (_, body) = post(&wide, &request(json!({"max_tokens": 3}))).await;
        assert_eq!(body["choices"][0]["text"], "abc");
        assert_eq!(body["choices"][0]["finish_reason"], "length");
        assert_eq!(body["usage"]["completion_tokens"], 3);
        let engine = testkit::engine(testkit::script("abcdefgh"));
        let (_, capped) = post(&app(&engine, 5), &request(json!({}))).await;
        assert_eq!(capped["choices"][0]["text"], "abc");
        assert_eq!(capped["choices"][0]["finish_reason"], "length");
    }

    #[tokio::test]
    async fn stop_takes_a_string_or_an_array_and_the_stop_text_stays_hidden() {
        for stop in [json!("END"), json!(["END"]), json!(["nope", "END"])] {
            let engine = testkit::engine(testkit::script("Hello END tail"));
            let (status, body) = post(&app(&engine, 4_096), &request(json!({"stop": stop}))).await;
            assert_eq!(status, StatusCode::OK, "{stop}");
            assert_eq!(body["choices"][0]["text"], "Hello ", "{stop}");
            assert_eq!(body["choices"][0]["finish_reason"], "stop", "{stop}");
        }
    }

    #[tokio::test]
    async fn the_reply_text_is_raw_so_think_and_tool_markers_stay() {
        let text = " <think>plan</think> <tool_call>\nx</tool_call>\n";
        let engine = testkit::engine(testkit::script(text));
        let (status, body) = post(&app(&engine, 4_096), &request(json!({}))).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["choices"][0]["text"], text);
        assert_eq!(body["choices"][0]["finish_reason"], "stop");
    }

    async fn stream(app: &Router, body: &Value) -> (String, Vec<testkit::SseEvent>) {
        let (status, headers, bytes) =
            testkit::send(app, testkit::post_json("/v1/completions", body)).await;
        assert_eq!(status, StatusCode::OK);
        let content_type = headers[header::CONTENT_TYPE].to_str().expect("type");
        (content_type.to_string(), testkit::sse_events(&bytes))
    }

    #[tokio::test]
    async fn a_stream_sends_text_chunks_then_the_usage_chunk_then_done() {
        let engine = testkit::engine(testkit::script("Hi!"));
        let (content_type, events) =
            stream(&app(&engine, 4_096), &request(json!({"stream": true}))).await;
        assert_eq!(content_type, "text/event-stream");
        assert_eq!(events.len(), 5, "{events:?}");
        assert_eq!(events[4].data, "[DONE]");
        let chunks: Vec<Value> = events[..4]
            .iter()
            .map(|event| {
                assert_eq!(event.name, None);
                serde_json::from_str(&event.data).expect("chunk json")
            })
            .collect();
        for chunk in &chunks {
            assert_eq!(chunk["object"], "text_completion");
            assert_eq!(chunk["model"], "model-a");
            assert_eq!(chunk["choices"][0]["index"], 0);
        }
        let texts: Vec<&Value> = chunks[..3]
            .iter()
            .map(|c| &c["choices"][0]["text"])
            .collect();
        assert_eq!(texts, [&json!("H"), &json!("i"), &json!("!")]);
        for early in &chunks[..3] {
            assert_eq!(early["choices"][0]["finish_reason"], Value::Null);
            assert!(early.get("usage").is_none(), "{early}");
        }
        let last = &chunks[3];
        assert_eq!(last["choices"][0]["text"], "");
        assert_eq!(last["choices"][0]["finish_reason"], "stop");
        assert_eq!(last["usage"]["prompt_tokens"], 2);
        assert_eq!(last["usage"]["completion_tokens"], 4);
        assert_eq!(last["usage"]["total_tokens"], 6);
    }

    #[tokio::test]
    async fn a_stream_has_no_role_chunk_and_ends_with_length_at_max_tokens() {
        let engine = testkit::engine(testkit::script("abcdef"));
        let body = request(json!({"stream": true, "max_tokens": 2}));
        let (_, events) = stream(&app(&engine, 4_096), &body).await;
        let chunks: Vec<Value> = events[..events.len() - 1]
            .iter()
            .map(|event| serde_json::from_str(&event.data).expect("chunk json"))
            .collect();
        let text: String = chunks
            .iter()
            .filter_map(|chunk| chunk["choices"][0]["text"].as_str())
            .collect();
        assert_eq!(text, "ab");
        assert!(
            chunks
                .iter()
                .all(|chunk| chunk["choices"][0].get("delta").is_none())
        );
        let last = chunks.last().expect("a last chunk");
        assert_eq!(last["choices"][0]["finish_reason"], "length");
        assert_eq!(events.last().expect("done").data, "[DONE]");
    }

    #[tokio::test]
    async fn a_stream_whose_engine_fails_after_admission_sends_an_error_event() {
        let engine = testkit::ending_engine(
            "ab",
            testkit::Ending::Error(EngError::Backend {
                operation: "qmm",
                message: "device lost".to_string(),
            }),
        );
        let (_, events) = stream(&app(&engine, 4_096), &request(json!({"stream": true}))).await;
        let last = events.last().expect("an event");
        assert_eq!(last.name.as_deref(), Some("error"), "{events:?}");
        assert!(!last.data.contains("device lost"), "{events:?}");
        assert!(events.iter().all(|event| event.data != "[DONE]"));
    }

    #[tokio::test]
    async fn the_routes_need_the_bearer_key() {
        let engine = testkit::engine(Vec::new());
        let app = app(&engine, 4_096);
        for key in [None, Some("sk-wrong")] {
            let request = testkit::post_bytes("/v1/completions", key, b"{}".to_vec());
            let (status, _headers, bytes) = testkit::send(&app, request).await;
            assert_eq!(status, StatusCode::UNAUTHORIZED);
            let body = testkit::json_of(&bytes);
            assert_schema("Error", &body);
            assert_eq!(error_code(&body), "unauthorized");
        }
    }

    #[tokio::test]
    async fn a_get_returns_the_json_404() {
        let engine = testkit::engine(Vec::new());
        let request = testkit::get("/v1/completions", Some(testkit::API_KEY));
        let (status, _headers, bytes) = testkit::send(&app(&engine, 4_096), request).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(error_code(&testkit::json_of(&bytes)), "not_found");
    }

    #[tokio::test]
    async fn routing_failures_return_the_status_of_the_chat_route() {
        let engine = testkit::engine(testkit::script("ok"));
        let app = app(&engine, 4_096);
        for extra in [
            json!({"model": "other"}),
            json!({"x-engine": "e9"}),
            json!({"x-session": "s1"}),
        ] {
            let (status, body) = post(&app, &request(extra.clone())).await;
            assert_eq!(status, StatusCode::NOT_FOUND, "{extra}");
            assert_eq!(error_code(&body), "not_found");
            assert_schema("Error", &body);
        }
        let unrouted = json!({"prompt": "hi"});
        let (status, body) = post(&app, &unrouted).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(error_code(&body), "bad_request");
        let both = request(json!({"model": "unknown", "x-engine": "e1"}));
        assert_eq!(post(&app, &both).await.0, StatusCode::OK);
    }

    #[tokio::test]
    async fn malformed_requests_return_400_that_names_the_field() {
        let engine = testkit::engine(Vec::new());
        let app = app(&engine, 4_096);
        let cases = [
            (json!({"model": "model-a"}), "prompt is required"),
            (request(json!({"prompt": ""})), "prompt must not be empty"),
            (request(json!({"prompt": 5})), "prompt must be a string"),
            (request(json!({"prompt": ["a", "b"]})), "single string"),
            (request(json!({"prompt": [1, 2, 3]})), "single string"),
            (request(json!({"max_tokens": 0})), "max_tokens"),
            (request(json!({"temperature": 3})), "temperature"),
            (request(json!({"top_p": 2})), "top_p"),
            (request(json!({"top_k": -1})), "top_k"),
            (request(json!({"stream": "yes"})), "stream"),
            (request(json!({"stop": 5})), "stop"),
            (request(json!({"stop": ["a", 3]})), "stop[1]"),
            (request(json!({"n": 2})), "n must be 1"),
            (request(json!({"echo": true})), "echo is not supported"),
            (request(json!({"logprobs": 5})), "logprobs is not supported"),
            (request(json!({"best_of": 2})), "best_of is not supported"),
            (request(json!({"suffix": "x"})), "suffix is not supported"),
        ];
        for (body, expected) in cases {
            let (status, reply) = post(&app, &body).await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
            assert_eq!(error_code(&reply), "bad_request", "{body}");
            let message = reply["error"]["message"].as_str().expect("message");
            assert!(message.contains(expected), "{body}: {message}");
            assert_schema("Error", &reply);
        }
    }

    #[tokio::test]
    async fn the_defaults_of_the_unsupported_fields_are_accepted() {
        let engine = testkit::engine(testkit::script("ok"));
        let body = request(json!({
            "n": 1, "echo": false, "logprobs": null, "best_of": null, "suffix": null,
            "x-unknown": 1
        }));
        let (status, reply) = post(&app(&engine, 4_096), &body).await;
        assert_eq!(status, StatusCode::OK, "{reply}");
    }

    #[tokio::test]
    async fn a_body_that_is_not_a_json_object_returns_400() {
        let engine = testkit::engine(Vec::new());
        let app = app(&engine, 4_096);
        for raw in [&b"[1]"[..], &b"{\"prompt\":"[..]] {
            let request =
                testkit::post_bytes("/v1/completions", Some(testkit::API_KEY), raw.to_vec());
            let (status, _headers, bytes) = testkit::send(&app, request).await;
            assert_eq!(status, StatusCode::BAD_REQUEST);
            assert_eq!(error_code(&testkit::json_of(&bytes)), "bad_request");
        }
    }

    #[tokio::test]
    async fn a_prompt_at_the_context_limit_is_refused_with_no_prefill() {
        let (engine, seen) = testkit::probe_engine();
        let app = app(&engine, 2);
        let (status, body) = post(&app, &request(json!({}))).await;
        assert_eq!(status, StatusCode::CONFLICT);
        assert_eq!(error_code(&body), "context_limit");
        let (status, headers, bytes) = testkit::send(
            &app,
            testkit::post_json("/v1/completions", &request(json!({"stream": true}))),
        )
        .await;
        assert_eq!(status, StatusCode::CONFLICT);
        assert_eq!(headers[header::CONTENT_TYPE], "application/json");
        assert_eq!(error_code(&testkit::json_of(&bytes)), "context_limit");
        assert!(seen.lock().expect("prompt lock").is_empty());
    }

    #[tokio::test]
    async fn engine_states_map_to_the_statuses_of_the_chat_route() {
        let failed = testkit::failing_engine();
        let entry = testkit::entry("e1", "model-a", &failed, 4_096);
        testkit::wait_for_state(&entry.handle, EngineState::Error);
        let broken = router(Arc::new(AppState::new(testkit::API_KEY, vec![entry])));
        let (status, body) = post(&broken, &request(json!({}))).await;
        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
        assert_schema("BackendError", &body);
        assert!(!body.to_string().contains("fixture load failure"), "{body}");
        let engine = testkit::engine(Vec::new());
        let app = app(&engine, 4_096);
        engine.shutdown();
        let (status, body) = post(&app, &request(json!({}))).await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(error_code(&body), "overloaded");
    }

    #[tokio::test]
    async fn the_fields_that_chat_ignores_are_ignored_here_too() {
        let engine = testkit::engine(testkit::script("ok"));
        let body = request(json!({
            "seed": 7, "presence_penalty": 1.5, "frequency_penalty": -1, "logit_bias": {"5": 10},
            "stream_options": {"include_usage": false}, "user": "u1"
        }));
        let (status, reply) = post(&app(&engine, 4_096), &body).await;
        assert_eq!(status, StatusCode::OK, "{reply}");
        assert_eq!(reply["choices"][0]["text"], "ok");
    }

    #[tokio::test]
    async fn a_full_engine_queue_returns_503_overloaded_for_both_reply_shapes() {
        let (engine, gate) = testkit::gated_engine(Vec::new());
        let handle = engine.handle();
        let mut held = Vec::new();
        for _ in 0..QUEUE_CAPACITY {
            let (job, receiver) = Job::new(vec![104], testkit::generation()).expect("job");
            handle.try_submit(job).expect("the queue has room");
            held.push(receiver);
        }
        let entry = testkit::entry("e1", "model-a", &engine, 4_096);
        let app = router(Arc::new(AppState::new(testkit::API_KEY, vec![entry])));
        for stream in [false, true] {
            let request =
                testkit::post_json("/v1/completions", &request(json!({"stream": stream})));
            let (status, headers, bytes) = testkit::send(&app, request).await;
            assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "stream {stream}");
            assert_eq!(headers[header::CONTENT_TYPE], "application/json");
            let body = testkit::json_of(&bytes);
            assert_eq!(error_code(&body), "overloaded");
            assert_schema("Error", &body);
        }
        drop(gate);
    }

    #[tokio::test]
    async fn the_chat_route_is_unchanged_beside_the_completions_route() {
        let engine = testkit::engine([testkit::script("Hi!"), testkit::script("ok")].concat());
        let app = app(&engine, 4_096);
        let chat = json!({
            "model": "model-a", "messages": [{"role": "user", "content": "hi"}], "stream": true
        });
        let (status, _headers, bytes) =
            testkit::send(&app, testkit::post_json("/v1/chat/completions", &chat)).await;
        assert_eq!(status, StatusCode::OK);
        let events = testkit::sse_events(&bytes);
        assert_eq!(events.len(), 6, "{events:?}");
        assert_eq!(events[5].data, "[DONE]");
        // The golden stream of the chat route: a role chunk, one chunk for each character, the
        // usage chunk, and [DONE]. `created` and the environment-dependent cache details of
        // `usage` are taken out before the comparison.
        let mut chunks: Vec<Value> = events[..5]
            .iter()
            .map(|event| {
                assert_eq!(event.name, None);
                serde_json::from_str(&event.data).expect("chunk json")
            })
            .collect();
        for chunk in &mut chunks {
            chunk.as_object_mut().expect("object").remove("created");
        }
        let last_usage = &mut chunks[4]["usage"];
        assert_eq!(last_usage["reused_tokens"], 0);
        *last_usage = json!({
            "prompt_tokens": last_usage["prompt_tokens"],
            "completion_tokens": last_usage["completion_tokens"],
            "total_tokens": last_usage["total_tokens"]
        });
        let chunk = |delta: Value, finish: Value| {
            let mut chunk = json!({
                "id": "chatcmpl-0000000000000001", "object": "chat.completion.chunk",
                "model": "model-a", "engine": "e1",
                "choices": [{"index": 0, "delta": delta}]
            });
            if !finish.is_null() {
                chunk["choices"][0]["finish_reason"] = finish;
            }
            chunk
        };
        let mut last = chunk(json!({}), json!("stop"));
        last["usage"] = json!({"prompt_tokens": 21, "completion_tokens": 4, "total_tokens": 25});
        let expected = vec![
            chunk(json!({"role": "assistant", "content": ""}), Value::Null),
            chunk(json!({"content": "H"}), Value::Null),
            chunk(json!({"content": "i"}), Value::Null),
            chunk(json!({"content": "!"}), Value::Null),
            last,
        ];
        assert_eq!(chunks, expected);
        let (status, body) = post(&app, &request(json!({}))).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["choices"][0]["text"], "ok");
    }
}
