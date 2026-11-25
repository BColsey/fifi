//! The HTTP smoke of the opt-in Flash-Next GGUF serving path: the real server, in process, one
//! engine over the pinned GGUF, a greedy three-token continuation checked against the host oracle.

use super::{config::EngineConfig, run, state::AppState, testkit};
use crate::{family::qwen::flash_next::settings::FlashEnv, sched::worker::Engine};
use serde_json::{Value, json};
use std::io::{Read, Write as _};
use std::time::{Duration, Instant};
use tokio::net::TcpListener;

const MODEL_DIR: &str = "FIFI_FLASH_GGUF_MODEL_DIR";
const TOKENIZER_DIR: &str = "FIFI_FLASH_GGUF_TOKENIZER_DIR";
const PROMPT: &str = "FIFI_FLASH_GGUF_SMOKE_PROMPT";
const EXPECTED: &str = "FIFI_FLASH_GGUF_SMOKE_EXPECTED";
const CONTEXT: &str = "FIFI_FLASH_GGUF_SMOKE_CONTEXT";
const SWITCH: &str = "INFERENCE_FLASH_NEXT_GGUF_SERVE";
const MODEL: &str = "flash-next-gguf";
/// The prompt ids of the host oracle (`full-host-oracle.json`: prompt `[9707, 11, 1879]`).
const PROMPT_IDS: [u32; 3] = [9707, 11, 1879];
/// The maximum wait for the engine to load (the 84 GB trunk uploads from the files in place).
const LOAD_WAIT: Duration = Duration::from_secs(3600);

fn required(name: &str) -> String {
    std::env::var(name)
        .ok()
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| panic!("{name} must name the smoke's input"))
}

fn context() -> u32 {
    std::env::var(CONTEXT)
        .ok()
        .as_deref()
        .unwrap_or("64")
        .parse()
        .unwrap_or_else(|_| panic!("{CONTEXT} must be a token count"))
}

fn expected() -> Vec<u32> {
    std::env::var(EXPECTED)
        .unwrap_or_else(|_| "5854,11,59710".to_string())
        .split(',')
        .map(|id| {
            id.trim()
                .parse()
                .expect("whole token ids in FIFI_FLASH_GGUF_SMOKE_EXPECTED")
        })
        .collect()
}

/// One HTTP exchange over a loopback socket, as the `run.rs` socket tests do: the request goes
/// out with `Connection: close` and the whole response comes back before the peer closes.
fn exchange(address: std::net::SocketAddr, request: String) -> String {
    let mut socket = std::net::TcpStream::connect(address).expect("connect");
    socket
        .set_read_timeout(Some(Duration::from_secs(600)))
        .expect("read timeout");
    socket.write_all(request.as_bytes()).expect("write");
    let mut answer = Vec::new();
    socket.read_to_end(&mut answer).expect("read");
    String::from_utf8_lossy(&answer).to_string()
}

async fn send(
    address: std::net::SocketAddr,
    method: &str,
    path: &str,
    body: Option<&Value>,
) -> String {
    let body = body.map(|value| value.to_string()).unwrap_or_default();
    let request = format!(
        "{method} {path} HTTP/1.1\r\nHost: test\r\nAuthorization: Bearer {}\r\nContent-Type: \
         application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        testkit::API_KEY,
        body.len()
    );
    tokio::task::spawn_blocking(move || exchange(address, request))
        .await
        .expect("client task")
}

/// The JSON body of a `200` reply; anything else fails with the whole response.
async fn json_call(address: std::net::SocketAddr, method: &str, path: &str, body: &Value) -> Value {
    let answer = send(address, method, path, Some(body)).await;
    assert!(
        answer.starts_with("HTTP/1.1 200"),
        "{method} {path}: {answer}"
    );
    let (_, body) = answer.split_once("\r\n\r\n").unwrap_or(("", &answer));
    serde_json::from_str(body).unwrap_or_else(|error| panic!("{method} {path}: not JSON ({error})"))
}

async fn encode(address: std::net::SocketAddr, text: &str) -> Vec<u32> {
    let reply = json_call(
        address,
        "POST",
        "/v1/token/encode",
        &json!({"model": MODEL, "text": text}),
    )
    .await;
    reply["tokens"]
        .as_array()
        .expect("the encode reply carries the tokens")
        .iter()
        .map(|id| id.as_u64().expect("a token id") as u32)
        .collect()
}

/// Waits for `/ready`, or fails with the engine's own cause when its load ends first.
async fn await_ready(address: std::net::SocketAddr, engine: &Engine) {
    let began = Instant::now();
    while !send(address, "GET", "/ready", None)
        .await
        .starts_with("HTTP/1.1 200")
    {
        if let Some(cause) = engine.handle().failure() {
            panic!("the Flash-Next GGUF engine failed to load: {cause}");
        }
        assert!(
            began.elapsed() < LOAD_WAIT,
            "the engine did not become ready"
        );
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
}

/// The opt-in serving path end to end: `INFERENCE_FLASH_NEXT_GGUF_SERVE=1`, the real server on a
/// loopback port, the pinned GGUF, and a greedy three-token continuation checked against the host
/// oracle's ids (`5854, 11, 59710`, the text `_len, eos`).
///
/// Route: `POST /v1/completions` is the one route whose prompt the tokenizer "encodes as written:
/// no chat template, no start token" (`completions.rs`), so the engine consumes exactly the bare
/// ids `9707, 11, 1879`. `/v1/chat/completions` and `/v1/messages` render the chat template, and
/// a route that prepended a start token would be wrong for this check in the same way: every added
/// id moves the first logits off the oracle's.
///
/// Ids: both lists are read through the engine's own `POST /v1/token/encode`. The prompt is
/// encoded before the request (a tokenizer mismatch fails there, by name); the completion's text
/// is encoded after it, which is exact for this continuation because its text `_len, eos`
/// re-encodes to the same three ids (59710 is the ordinary text token ` eos`, not a special
/// token). `logprobs` is not served on this route.
///
/// Inputs: `FIFI_FLASH_GGUF_MODEL_DIR` (the GGUF shards), `FIFI_FLASH_GGUF_TOKENIZER_DIR` (the
/// tokenizer files AND a `download-manifest.json` with the four pinned entries: the GGUF start
/// keeps the pack's manifest rule and refuses, with the lookup's own message, without one),
/// optional `FIFI_FLASH_GGUF_SMOKE_PROMPT` (default `.Q, input`), `FIFI_FLASH_GGUF_SMOKE_EXPECTED`
/// (default `5854,11,59710`) and `FIFI_FLASH_GGUF_SMOKE_CONTEXT` (default 64).
///
/// The switch is read from the process environment when the start captures `FlashEnv`. A test does
/// not write that environment, so the runner sets `INFERENCE_FLASH_NEXT_GGUF_SERVE=1`, and without
/// it this test fails by name rather than through a start refusal that would hide the cause.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires the pinned Flash-Next GGUF, its tokenizer files with a download manifest, and an admitted CUDA device"]
async fn flash_next_gguf_http_smoke_greedy_matches_host() {
    assert!(
        matches!(FlashEnv::capture().gguf_serve(), Ok(true)),
        "{SWITCH} must be set to 1 in the environment this test runs in"
    );
    let model_dir = required(MODEL_DIR);
    let tokenizer_dir = required(TOKENIZER_DIR);
    let prompt = std::env::var(PROMPT).unwrap_or(".Q, input".to_string());
    let expected = expected();
    let entry: EngineConfig = serde_json::from_value(json!({
        "id": "flash-gguf",
        "family": "qwen3_8_flash_next",
        "model": MODEL,
        "model_dir": model_dir,
        "tokenizer_dir": tokenizer_dir,
        "context_limit": null,
        "flash_next": {"cuda_decode": true, "context": context()}
    }))
    .expect("a valid engine entry");
    // A start refusal (a missing manifest, the switch off, a kernel gap) fails here with its text.
    let (engine, started) = run::start_one(&entry, Some(false))
        .unwrap_or_else(|error| panic!("the Flash-Next GGUF start refused: {}", error.0));
    let state = std::sync::Arc::new(AppState::new(testkit::API_KEY, vec![started]));
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let address = listener.local_addr().expect("address");
    let server = tokio::spawn(run::serve(listener, state, std::future::pending()));
    await_ready(address, &engine).await;

    let prompt_ids = encode(address, &prompt).await;
    let reply = json_call(
        address,
        "POST",
        "/v1/completions",
        &json!({"model": MODEL, "prompt": prompt, "max_tokens": 3, "temperature": 0.0}),
    )
    .await;
    let text = reply["choices"][0]["text"]
        .as_str()
        .unwrap_or("")
        .to_string();
    let generated = encode(address, &text).await;
    println!(
        "flash-next gguf http smoke: prompt {prompt:?} -> {prompt_ids:?}; generated {generated:?} text {text:?}"
    );
    server.abort();
    let _ = engine.shutdown();
    assert_eq!(
        prompt_ids, PROMPT_IDS,
        "the prompt must be the oracle's three ids"
    );
    assert_eq!(
        generated, expected,
        "the greedy continuation (text {text:?})"
    );
}
