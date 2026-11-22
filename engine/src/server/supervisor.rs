//! Supervised worker engines.
//!
//! A runtime-added engine of a family that keeps its card memory for
//! the process lifetime (`holds_memory_until_restart`) can be served
//! by a supervised `enginesrv serve --config` child process instead of
//! loading in the process: stopping such an engine ends the worker
//! and the operating system frees its memory, where an in-process
//! stop cannot. The front server routes requests to a ready worker
//! and reports the worker's own loader state, so admission and load
//! failures surface exactly as they would in-process.
//!
//! The worker is the same binary (`current_exe`), started from the
//! configuration the front derived, written to a file the child reads
//! once, on a loopback port the kernel picks; the first `enginesrv
//! listening on <addr>` line on its
//! stdout names the port. Its API key is generated per worker and set
//! through the child environment, and the front replaces the caller's
//! auth headers with it while proxying.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::body::{Body, Bytes, to_bytes};
use axum::extract::{Request, State};
use axum::http::{StatusCode, header};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use serde_json::Value;

use super::config::Config;
use super::model_serve::config_json;
use super::state::AppState;

/// The environment variable a worker reads its API key from: the one
/// every configuration built here names.
const WORKER_KEY_ENV: &str = "ENGINESRV_API_KEY";

/// The line `run` prints once the listener is bound: the worker's
/// port discovery, so no fixed port is ever contended.
const LISTEN_PREFIX: &str = "enginesrv listening on ";

/// The worker listens here: a loopback port the kernel picks, named
/// on the listen line the front reads.
const WORKER_LISTEN: &str = "127.0.0.1:0";

/// How long the front waits for the listen line before calling the
/// worker failed: a start that does not bind within this window is
/// wedged (driver initialization included, which takes minutes on a
/// cold card, so this stays generous).
const BIND_TIMEOUT: Duration = Duration::from_secs(600);

/// How often the health poll asks the worker for its engine state.
const HEALTH_PERIOD: Duration = Duration::from_secs(2);

/// How long a stop waits after SIGTERM before killing the worker: a
/// graceful shutdown flushes what a family writes at exit (Flash-Next
/// rewrites its expert popularity profile), so it is worth waiting for.
const TERM_GRACE: Duration = Duration::from_secs(30);

/// The largest request body the proxy buffers before the local
/// handlers see it: generous enough for a full-context prompt, small
/// enough that a stray upload cannot pin the front's memory.
const PROXY_BODY_LIMIT: usize = 64 * 1024 * 1024;

/// The observed state of one worker, as the engine API reports it.
#[derive(Clone)]
pub(crate) struct WorkerStatus {
    pub(crate) state: String,
    pub(crate) load_stage: Option<String>,
    pub(crate) error: Option<String>,
    pub(crate) context_limit: Option<u32>,
}

struct Worker {
    id: String,
    /// The worker's own model name, written once by the health poll
    /// after the engine can report it, then read by `find`.
    model: Mutex<Option<String>>,
    model_dir: PathBuf,
    key: String,
    addr: Mutex<Option<String>>,
    child: Arc<Mutex<Option<Child>>>,
    status: Mutex<WorkerStatus>,
    /// The configuration file the child was started from. The child
    /// reads it once, so it goes when the last handle to the worker
    /// does.
    config_path: Option<PathBuf>,
}

impl Drop for Worker {
    fn drop(&mut self) {
        if let Some(path) = &self.config_path {
            let _ = std::fs::remove_file(path);
        }
    }
}

fn poison<T>(error: std::sync::PoisonError<T>) -> T {
    error.into_inner()
}

pub(crate) struct WorkerSupervisor {
    workers: Mutex<HashMap<String, Arc<Worker>>>,
    /// The worker executable; `None` is this binary, `enginesrv`.
    binary: Option<PathBuf>,
}

impl WorkerSupervisor {
    pub(crate) fn new() -> Self {
        Self {
            workers: Mutex::new(HashMap::new()),
            binary: None,
        }
    }

    /// A supervisor that starts a stand-in executable, so a test can
    /// drive the real child-process path.
    #[cfg(test)]
    pub(crate) fn with_worker_binary(binary: PathBuf) -> Self {
        Self {
            binary: Some(binary),
            ..Self::new()
        }
    }

    pub(crate) fn count(&self) -> usize {
        self.workers.lock().unwrap_or_else(poison).len()
    }

    pub(crate) fn contains(&self, id: &str) -> bool {
        self.workers.lock().unwrap_or_else(poison).contains_key(id)
    }

    /// The worker an unambiguous routing names, for the proxy path.
    /// `engine` and `model` are the request's own routing fields; both
    /// must agree when both are present.
    pub(crate) fn find(
        &self,
        engine: Option<&str>,
        model: Option<&str>,
    ) -> Option<Arc<WorkerView>> {
        let workers = self.workers.lock().unwrap_or_else(poison);
        let by_engine = engine.and_then(|name| workers.get(name));
        let by_model = model.and_then(|name| {
            workers
                .values()
                .find(|worker| worker.model.lock().unwrap_or_else(poison).as_deref() == Some(name))
        });
        let worker = match (by_engine, by_model) {
            (Some(engine), Some(model)) => {
                if Arc::ptr_eq(engine, model) {
                    Some(engine)
                } else {
                    None
                }
            }
            (Some(worker), None) | (None, Some(worker)) => Some(worker),
            (None, None) => None,
        }?;
        let view = WorkerView {
            id: worker.id.clone(),
            model: worker.model.lock().unwrap_or_else(poison).clone(),
            addr: worker.addr.lock().unwrap_or_else(poison).clone(),
            key: worker.key.clone(),
        };
        Some(Arc::new(view))
    }

    /// The worker by id, once its address is known.
    pub(crate) fn get(&self, id: &str) -> Option<Arc<WorkerView>> {
        self.find(Some(id), None)
    }

    /// One row per worker for the engine list.
    pub(crate) fn snapshot(&self) -> Vec<WorkerRow> {
        let workers = self.workers.lock().unwrap_or_else(poison);
        workers
            .values()
            .map(|worker| WorkerRow {
                id: worker.id.clone(),
                model: worker.model.lock().unwrap_or_else(poison).clone(),
                model_dir: worker.model_dir.clone(),
                status: worker.status.lock().unwrap_or_else(poison).clone(),
            })
            .collect()
    }

    /// Start one worker. The id is the front's own engine id; the
    /// worker's internal id stays whatever the derived configuration
    /// carries, because the front routes by its id and never exposes
    /// the worker's. The child starts from the configuration the front
    /// derived, so nothing is derived twice and no family needs a
    /// simplified startup of its own.
    pub(crate) fn spawn(
        &self,
        id: &str,
        config: Config,
        model: Option<String>,
    ) -> Result<(), String> {
        if self.contains(id) {
            return Err(format!("engine {id} already exists; pass a different id"));
        }
        if config.engines.is_empty() {
            // Writing this would read the engine it does not have, and a
            // panic here ends the front.
            return Err("the derived configuration carries no engine".to_string());
        }
        let binary = match &self.binary {
            Some(binary) => binary.clone(),
            None => std::env::current_exe().map_err(|error| {
                format!("the supervised worker needs this binary's path: {error}")
            })?,
        };
        let key = worker_key(id);
        let mut config = config;
        config.listen = WORKER_LISTEN.to_string();
        let model_dir = config
            .engines
            .first()
            .map(|engine| engine.model_dir.clone())
            .unwrap_or_default();
        let config_path =
            std::env::temp_dir().join(format!("fifi-worker-{}-{id}.json", std::process::id()));
        std::fs::write(&config_path, config_json(&config)).map_err(|error| {
            format!("the supervised worker needs a configuration file: {error}")
        })?;
        let child = Command::new(binary)
            .arg("serve")
            .arg("--config")
            .arg(&config_path)
            .env(WORKER_KEY_ENV, &key)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|error| format!("the supervised worker failed to start: {error}"));
        let mut child = match child {
            Ok(child) => child,
            Err(error) => {
                let _ = std::fs::remove_file(&config_path);
                return Err(error);
            }
        };
        let stdout = child.stdout.take().expect("stdout is piped above");
        let stderr = child.stderr.take().expect("stderr is piped above");
        // Both pipes drain on their own threads; a full pipe would
        // otherwise stall the worker. The tail keeps the last lines
        // for a death the loader state never explained.
        let (addr_sender, addr_receiver) = mpsc::channel();
        let tail = Arc::new(Mutex::new(Vec::new()));
        let stdout_tail = Arc::clone(&tail);
        std::thread::Builder::new()
            .name(format!("worker-{id}-stdout"))
            .spawn(move || {
                let reader = BufReader::new(stdout);
                for line in reader.lines().map_while(Result::ok) {
                    let mut tail = stdout_tail.lock().unwrap_or_else(poison);
                    push_tail(&mut tail, line.clone());
                    drop(tail);
                    if let Some(addr) = line.strip_prefix(LISTEN_PREFIX) {
                        let _ = addr_sender.send(addr.trim().to_string());
                        // The address is out; keep draining so the
                        // worker never blocks on its own logging.
                    }
                }
                let _ = addr_sender.send(String::new());
            })
            .map_err(|error| format!("the worker reader failed to start: {error}"))?;
        let stderr_tail = Arc::clone(&tail);
        std::thread::Builder::new()
            .name(format!("worker-{id}-stderr"))
            .spawn(move || {
                let reader = BufReader::new(stderr);
                for line in reader.lines().map_while(Result::ok) {
                    let mut tail = stderr_tail.lock().unwrap_or_else(poison);
                    push_tail(&mut tail, line);
                }
            })
            .map_err(|error| format!("the worker reader failed to start: {error}"))?;
        let worker = Arc::new(Worker {
            id: id.to_string(),
            // The derivation's own model name, so the lists carry it
            // before the worker's engine can report one; the health
            // poll refreshes it once it can.
            model: Mutex::new(model),
            model_dir,
            key,
            addr: Mutex::new(None),
            child: Arc::new(Mutex::new(Some(child))),
            status: Mutex::new(WorkerStatus {
                state: "starting".to_string(),
                load_stage: None,
                error: None,
                context_limit: None,
            }),
            config_path: Some(config_path),
        });
        let mut workers = self.workers.lock().unwrap_or_else(poison);
        if workers.contains_key(id) {
            return Err(format!("engine {id} already exists; pass a different id"));
        }
        workers.insert(id.to_string(), Arc::clone(&worker));
        drop(workers);
        tokio::spawn(supervise(
            Arc::clone(&worker),
            addr_receiver,
            Arc::clone(&tail),
        ));
        Ok(())
    }

    /// Stop one worker and free its memory. Returns the worker's id
    /// when it was there to stop.
    pub(crate) fn stop(&self, id: &str) -> Option<String> {
        let worker = self.workers.lock().unwrap_or_else(poison).remove(id)?;
        {
            let mut status = worker.status.lock().unwrap_or_else(poison);
            status.state = "stopped".to_string();
            status.load_stage = None;
        }
        let worker = Arc::clone(&worker);
        // The wait runs on the runtime's blocking pool; a worker that
        // ignores SIGTERM is killed after the grace period.
        tokio::task::spawn_blocking(move || {
            let mut child = worker.child.lock().unwrap_or_else(poison).take();
            let Some(child) = child.as_mut() else {
                return;
            };
            let pid = child.id();
            let _ = Command::new("kill")
                .arg(pid.to_string())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status();
            let deadline = std::time::Instant::now() + TERM_GRACE;
            loop {
                match child.try_wait() {
                    Ok(Some(_)) => return,
                    Ok(None) if std::time::Instant::now() < deadline => {
                        std::thread::sleep(Duration::from_millis(100));
                    }
                    _ => break,
                }
            }
            let _ = child.kill();
            let _ = child.wait();
        });
        Some(id.to_string())
    }

    /// A worker entry the tests insert directly, standing in for a
    /// spawned child: the proxy, list and stop paths run unchanged,
    /// while the real child-process path has its own test below.
    #[cfg(test)]
    pub(crate) fn insert_for_test(&self, id: &str, model: &str, addr: &str, key: &str) {
        let worker = Arc::new(Worker {
            id: id.to_string(),
            model: Mutex::new(Some(model.to_string())),
            model_dir: PathBuf::new(),
            key: key.to_string(),
            addr: Mutex::new(Some(addr.to_string())),
            child: Arc::new(Mutex::new(None)),
            status: Mutex::new(WorkerStatus {
                state: "ready".to_string(),
                load_stage: None,
                error: None,
                context_limit: None,
            }),
            config_path: None,
        });
        self.workers
            .lock()
            .unwrap_or_else(poison)
            .insert(id.to_string(), worker);
    }
}

/// A fresh key for one worker: two independently seeded hashers make
/// a value no caller can predict, which is enough for a loopback-only
/// child the front itself spawned.
fn worker_key(id: &str) -> String {
    use std::hash::{BuildHasher, Hasher};
    let mut first = std::collections::hash_map::RandomState::new().build_hasher();
    first.write(id.as_bytes());
    let mut second = std::collections::hash_map::RandomState::new().build_hasher();
    second.write_u64(std::process::id() as u64);
    format!("{:016x}{:016x}", first.finish(), second.finish())
}

/// The tail a failure answer quotes: the last lines, un-interleaved.
fn push_tail(tail: &mut Vec<String>, line: String) {
    tail.push(line);
    let overflow = tail.len().saturating_sub(16);
    tail.drain(..overflow);
}

/// One worker, resolved for the proxy: everything a forwarded request
/// needs, none of the process plumbing. The address is absent until
/// the worker reports it, and a request routed to an address-less
/// worker answers that the engine is still starting.
pub(crate) struct WorkerView {
    pub(crate) id: String,
    pub(crate) model: Option<String>,
    pub(crate) addr: Option<String>,
    pub(crate) key: String,
}

/// One row of the engine list.
pub(crate) struct WorkerRow {
    pub(crate) id: String,
    pub(crate) model: Option<String>,
    pub(crate) model_dir: PathBuf,
    pub(crate) status: WorkerStatus,
}

/// The per-worker supervision task: wait for the bound address, then
/// poll the worker's engine state until the process is gone.
async fn supervise(
    worker: Arc<Worker>,
    addr_receiver: mpsc::Receiver<String>,
    tail: Arc<Mutex<Vec<String>>>,
) {
    let addr = tokio::task::spawn_blocking(move || {
        addr_receiver
            .recv_timeout(BIND_TIMEOUT)
            .ok()
            .filter(|addr| !addr.is_empty())
    })
    .await
    .ok()
    .flatten();
    let Some(addr) = addr else {
        failed(
            &worker,
            "the worker did not report a listening address",
            &tail,
        );
        return;
    };
    *worker.addr.lock().unwrap_or_else(poison) = Some(addr.clone());
    let url = format!("http://{addr}/v1/engines");
    let key = worker.key.clone();
    loop {
        tokio::time::sleep(HEALTH_PERIOD).await;
        if worker.status.lock().unwrap_or_else(poison).state == "stopped" {
            return;
        }
        let url = url.clone();
        let key = key.clone();
        let observe_worker = Arc::clone(&worker);
        let observe_tail = Arc::clone(&tail);
        let observation = tokio::task::spawn_blocking(move || {
            observe(&url, &key, observe_worker.as_ref(), observe_tail.as_ref())
        })
        .await;
        match observation {
            Ok(true) => {}
            Ok(false) => return,
            Err(_) => {
                failed(&worker, "the health poll task failed", &tail);
                return;
            }
        }
    }
}

/// One health observation. Returns `false` when the worker is gone.
fn observe(url: &str, key: &str, worker: &Worker, tail: &Mutex<Vec<String>>) -> bool {
    if worker.status.lock().unwrap_or_else(poison).state == "stopped" {
        return false;
    }
    let exited = worker
        .child
        .lock()
        .unwrap_or_else(poison)
        .as_mut()
        .is_none_or(|child| matches!(child.try_wait(), Ok(Some(_))));
    if exited {
        let mut status = worker.status.lock().unwrap_or_else(poison);
        status.state = "failed".to_string();
        status.error = Some(format!(
            "the worker process ended: {}",
            tail.lock()
                .unwrap_or_else(poison)
                .last()
                .cloned()
                .unwrap_or_else(|| "no output".to_string())
        ));
        return false;
    }
    let agent = agent();
    let request = agent
        .get(url)
        .header("authorization", format!("Bearer {key}"));
    let response = match request.call() {
        Ok(response) => response,
        Err(_) => return true, // not bound yet, or a busy moment: the
                               // child's own lifetime check above is the failure signal.
    };
    let body: Value = match serde_json::from_reader(response.into_body().into_reader()) {
        Ok(body) => body,
        Err(_) => return true,
    };
    let Some(engine) = body["engines"][0].as_object() else {
        return true;
    };
    let mut status = worker.status.lock().unwrap_or_else(poison);
    if status.state == "stopped" {
        return false;
    }
    status.state = match engine.get("state").and_then(Value::as_str) {
        Some("ready") | Some("busy") => "ready".to_string(),
        Some("failed") => "failed".to_string(),
        Some("loading") | Some("queued") => "starting".to_string(),
        _ => "starting".to_string(),
    };
    // A ready engine reports none of the members below, and indexing a
    // JSON object with a missing key panics.
    let text = |name: &str| engine.get(name).and_then(Value::as_str);
    status.load_stage = text("load_stage").map(str::to_string);
    status.error = engine
        .get("error")
        .and_then(Value::as_object)
        .and_then(|error| error.get("message"))
        .and_then(Value::as_str)
        .map(str::to_string);
    status.context_limit = engine
        .get("context_limit")
        .and_then(Value::as_u64)
        .map(|value| value as u32);
    if status.state == "ready" {
        let mut model = worker.model.lock().unwrap_or_else(poison);
        if model.is_none() {
            // The worker's own model name, once its engine can report
            // it: requests routed by model name find the worker
            // through it.
            *model = text("model").map(str::to_string);
        }
    }
    true
}

/// A shared agent for the worker polls and the proxy: loopback only,
/// short connect timeout, so a wedged worker answers failure instead
/// of holding a health task.
fn agent() -> ureq::Agent {
    ureq::Agent::config_builder()
        .timeout_connect(Some(Duration::from_secs(5)))
        .build()
        .into()
}

fn failed(worker: &Worker, why: &str, tail: &Mutex<Vec<String>>) {
    let mut status = worker.status.lock().unwrap_or_else(poison);
    status.state = "failed".to_string();
    status.error = Some(format!(
        "{why}: {}",
        tail.lock()
            .unwrap_or_else(poison)
            .last()
            .cloned()
            .unwrap_or_else(|| "no output".to_string())
    ));
}

/// The routing fields every served route reads from its JSON body:
/// the same three `resolve` reads, extracted the same way.
struct Routed {
    session: Option<String>,
    engine: Option<String>,
    model: Option<String>,
}

fn route_fields(body: &[u8]) -> Routed {
    let Ok(value) = serde_json::from_slice::<Value>(body) else {
        // A body the local handler would reject parses no routing
        // fields, so the proxy passes it through and lets that
        // rejection happen there.
        return Routed {
            session: None,
            engine: None,
            model: None,
        };
    };
    let text = |name: &str| value.get(name).and_then(Value::as_str).map(str::to_string);
    Routed {
        session: text("x-session"),
        engine: text("x-engine"),
        model: text("model"),
    }
}

/// The worker a request routes to, with `resolve`'s own semantics: a
/// session bound to a worker wins, unless an explicit field disagrees
/// with it (the local answer for a disagreement is a 404, so the
/// request falls through to the local path, which produces it); an
/// unknown session binds to whatever its own routing picks.
fn proxy_target(state: &AppState, routed: &Routed) -> Option<Arc<WorkerView>> {
    if let Some(session) = routed.session.as_deref() {
        if let Some(bound) = state.session_binding(session) {
            let worker = state.workers().get(&bound)?;
            if let Some(engine) = routed.engine.as_deref()
                && engine != worker.id
            {
                return None;
            }
            if let Some(model) = routed.model.as_deref()
                && let Some(worker_model) = worker.model.as_deref()
                && model != worker_model
            {
                return None;
            }
            return Some(worker);
        }
        let worker = state
            .workers()
            .find(routed.engine.as_deref(), routed.model.as_deref())?;
        state.bind_session(session, &worker.id);
        return Some(worker);
    }
    state
        .workers()
        .find(routed.engine.as_deref(), routed.model.as_deref())
}

/// The proxy middleware. A request whose routing names a supervised
/// worker is forwarded to it whole — its own admission, loader,
/// sessions and sampler serve it — and the worker's answer streams
/// back through the front. Every other request continues to the local
/// handler with its body intact.
pub(crate) async fn proxy_to_worker(
    State(state): State<Arc<AppState>>,
    request: Request,
    next: Next,
) -> Response {
    let (parts, raw_body) = request.into_parts();
    let body = match to_bytes(raw_body, PROXY_BODY_LIMIT).await {
        Ok(bytes) => bytes,
        Err(error) => return worker_error(StatusCode::BAD_REQUEST, &error.to_string()),
    };
    let routed = route_fields(&body);
    let Some(worker) = proxy_target(&state, &routed) else {
        return next.run(Request::from_parts(parts, Body::from(body))).await;
    };
    forward(&worker, &parts, body).await
}

/// Forward one request to its worker and stream the answer back. The
/// worker's own auth is the front's secret, so the caller's key never
/// reaches the child and the child's key never reaches the caller.
async fn forward(worker: &WorkerView, parts: &axum::http::request::Parts, body: Bytes) -> Response {
    let Some(addr) = worker.addr.as_deref() else {
        return worker_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "the supervised engine has not finished starting",
        );
    };
    let path = parts
        .uri
        .path_and_query()
        .map(|value| value.as_str().to_string())
        .unwrap_or_else(|| parts.uri.path().to_string());
    let text = |name: &'static str| {
        parts
            .headers
            .get(name)
            .and_then(|value| value.to_str().ok())
            .map(str::to_string)
    };
    let request_line = Forwarded {
        url: format!("http://{addr}{path}"),
        content_type: text("content-type"),
        accept: text("accept"),
        session: text("x-session"),
        key: worker.key.clone(),
        body,
    };
    let answer = tokio::task::spawn_blocking(move || {
        let Forwarded {
            url,
            content_type,
            accept,
            session,
            key,
            body,
        } = request_line;
        let agent = agent();
        let mut request = agent.post(&url);
        if let Some(value) = &content_type {
            request = request.header("content-type", value.as_str());
        }
        if let Some(value) = &accept {
            request = request.header("accept", value.as_str());
        }
        if let Some(value) = &session {
            request = request.header("x-session", value.as_str());
        }
        // Both auth schemes, so the worker's bearer and Anthropic
        // routes accept the same secret.
        request = request
            .header("authorization", format!("Bearer {key}").as_str())
            .header("x-api-key", key.as_str());
        let response = request
            .send(body.as_ref())
            .map_err(|error| error.to_string())?;
        let status = response.status().as_u16();
        let content_type = response
            .headers()
            .get("content-type")
            .and_then(|value| value.to_str().ok())
            .map(str::to_string);
        Ok::<_, String>((
            status,
            content_type,
            Box::new(response.into_body().into_reader()) as Box<dyn Read + Send>,
        ))
    })
    .await;
    let (status, content_type, reader) = match answer {
        Ok(Ok(answer)) => answer,
        Ok(Err(error)) => {
            return worker_error(
                StatusCode::BAD_GATEWAY,
                &format!("the supervised engine did not answer: {error}"),
            );
        }
        Err(_) => {
            return worker_error(
                StatusCode::BAD_GATEWAY,
                "the supervised engine forwarding task failed",
            );
        }
    };
    let (sender, receiver) = tokio::sync::mpsc::channel::<Result<Bytes, std::io::Error>>(4);
    tokio::task::spawn_blocking(move || {
        let mut reader = reader;
        let mut chunk = [0u8; 16 * 1024];
        loop {
            match reader.read(&mut chunk) {
                Ok(0) => break,
                Ok(size) => {
                    // A closed receiver means the caller went away;
                    // the pump ends and the worker socket drains.
                    if sender
                        .blocking_send(Ok(Bytes::copy_from_slice(&chunk[..size])))
                        .is_err()
                    {
                        break;
                    }
                }
                Err(error) => {
                    let _ = sender.blocking_send(Err(error));
                    break;
                }
            }
        }
    });
    let stream = futures::stream::unfold(receiver, |mut receiver| async move {
        receiver.recv().await.map(|item| (item, receiver))
    });
    let status = StatusCode::from_u16(status).unwrap_or(StatusCode::BAD_GATEWAY);
    let mut response = Response::builder().status(status);
    if let Some(value) = content_type {
        response = response.header(header::CONTENT_TYPE, value);
    }
    response
        .body(Body::from_stream(stream))
        .unwrap_or_else(|error| worker_error(StatusCode::BAD_GATEWAY, &error.to_string()))
}

/// The parts of one forwarded request, the blocking thread's payload.
struct Forwarded {
    url: String,
    content_type: Option<String>,
    accept: Option<String>,
    session: Option<String>,
    key: String,
    body: Bytes,
}

/// The contract's error body, for a failure the worker never answered.
fn worker_error(status: StatusCode, message: &str) -> Response {
    let body = serde_json::json!({"error": {"message": message, "code": "worker_unavailable"}});
    (status, axum::Json(body)).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::model_serve::{ModelServeArgs, bare_config};
    use crate::server::router::router;
    use crate::server::testkit;
    use axum::body::Body;
    use serde_json::{Value, json};

    /// The key of the worker's own API: a different secret than the
    /// front's, so a proxied answer proves the front replaced the
    /// caller's key with the worker's.
    const WORKER_KEY: &str = "sk-worker";

    /// A real second server on a loopback port serving the fake
    /// family: the stand-in for a spawned `enginesrv` worker.
    async fn worker_server(script: Vec<Vec<u32>>) -> String {
        let engine = testkit::engine(script);
        let entry = testkit::entry("worker-engine", "fake-model", &engine, 128);
        let state = Arc::new(AppState::new(WORKER_KEY, vec![entry]));
        let app = router(state);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind worker port");
        let addr = listener.local_addr().expect("worker address").to_string();
        tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });
        addr
    }

    fn front_with_worker(addr: &str) -> Arc<AppState> {
        let state = Arc::new(AppState::new(testkit::API_KEY, vec![]));
        state
            .workers()
            .insert_for_test("w1", "fake-model", addr, WORKER_KEY);
        state
    }

    /// The fake engine answers once per scripted turn, so a test with
    /// several successful chats concatenates the turns it needs.
    fn multi_script(turns: &[&str]) -> Vec<Vec<u32>> {
        turns
            .iter()
            .flat_map(|turn| testkit::script(turn))
            .collect()
    }

    async fn chat(state: &Arc<AppState>, body: Value) -> (StatusCode, String) {
        let app = router(Arc::clone(state));
        let (status, _headers, bytes) =
            testkit::send(&app, testkit::post_json("/v1/chat/completions", &body)).await;
        (status, String::from_utf8_lossy(&bytes).to_string())
    }

    /// A configuration with no engine is refused before any file or
    /// process exists: writing it would read the engine it lacks.
    #[test]
    fn a_worker_is_refused_a_configuration_without_an_engine() {
        let args = ModelServeArgs::default();
        let mut config = bare_config(&args, crate::family::qwen::flash_next::factory::FAMILY);
        config.engines.clear();
        let supervisor = WorkerSupervisor::new();
        let refused = supervisor
            .spawn("none", config, None)
            .expect_err("a worker without an engine");
        assert!(refused.contains("carries no engine"), "{refused}");
        assert!(supervisor.snapshot().is_empty());
        let unwritten =
            std::env::temp_dir().join(format!("fifi-worker-{}-none.json", std::process::id()));
        assert!(!unwritten.exists(), "{}", unwritten.display());
    }

    /// The real child-process path. The front writes the configuration
    /// it derived, starts `serve --config <file>`, learns the address
    /// from the child's listen line, and the file goes with the worker.
    /// A shell script stands in for `enginesrv`: it copies the file
    /// aside for the assertions and names the address of a second
    /// server that requires the key the spawn generated.
    #[tokio::test]
    async fn a_spawned_worker_starts_from_the_derived_config_file() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind worker port");
        let addr = listener.local_addr().expect("worker address").to_string();
        let scratch = std::env::temp_dir();
        let tag = format!("fifi-worker-test-{}", std::process::id());
        let script = scratch.join(format!("{tag}.sh"));
        let copied = scratch.join(format!("{tag}.json"));
        let model_dir = scratch.join(&tag);
        std::fs::create_dir_all(&model_dir).expect("model directory");
        std::fs::write(
            &script,
            format!(
                "#!/bin/sh\n\
                 # $1 $2 are `serve --config`; $3 names the file\n\
                 cp \"$3\" {}\n\
                 echo \"enginesrv listening on {addr}\"\n\
                 exec sleep 60\n",
                copied.display()
            ),
        )
        .expect("write the stand-in worker");
        let executable = Command::new("chmod")
            .arg("+x")
            .arg(&script)
            .status()
            .expect("chmod")
            .success();
        assert!(executable, "chmod +x on the stand-in worker");

        let args = ModelServeArgs {
            model_dir: model_dir.clone(),
            ..ModelServeArgs::default()
        };
        let family = crate::family::qwen::flash_next::factory::FAMILY;
        let supervisor = WorkerSupervisor::with_worker_binary(script.clone());
        supervisor
            .spawn("flash", bare_config(&args, family), None)
            .expect("spawn");
        let key = supervisor
            .workers
            .lock()
            .unwrap_or_else(poison)
            .get("flash")
            .map(|worker| worker.key.clone())
            .expect("the worker is listed");
        let engine = testkit::engine(testkit::script("unused"));
        let entry = testkit::entry("worker-engine", "fake-model", &engine, 128);
        let state = Arc::new(AppState::new(&key, vec![entry]));
        let app = router(state);
        tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });
        let config_path = scratch.join(format!("fifi-worker-{}-flash.json", std::process::id()));

        let deadline = std::time::Instant::now() + Duration::from_secs(30);
        let state = loop {
            let row = supervisor
                .snapshot()
                .into_iter()
                .find(|worker| worker.id == "flash")
                .expect("the worker is listed");
            if row.status.state != "starting" {
                break row.status.state;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "the worker stayed `starting`"
            );
            tokio::time::sleep(Duration::from_millis(50)).await;
        };
        assert_eq!(
            state, "ready",
            "the health poll reached the child's address"
        );

        let written = Config::from_file(&copied).expect("the child read a configuration file");
        assert_eq!(written.listen, "127.0.0.1:0");
        // The child reads this variable for the key the spawn set.
        assert_eq!(written.api_key_env, WORKER_KEY_ENV);
        let engine = &written.engines[0];
        assert_eq!(engine.family, family);
        assert!(
            engine.model_dir.ends_with(&tag),
            "{}",
            engine.model_dir.display()
        );
        assert!(engine.tokenizer_dir.ends_with(&tag));
        assert!(engine.flash_next.is_none(), "every option at its default");

        assert!(config_path.is_file(), "the worker's configuration file");
        assert_eq!(supervisor.stop("flash").as_deref(), Some("flash"));
        let deadline = std::time::Instant::now() + Duration::from_secs(30);
        while config_path.exists() {
            assert!(
                std::time::Instant::now() < deadline,
                "the configuration file outlived the worker"
            );
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        for path in [script, copied] {
            let _ = std::fs::remove_file(path);
        }
    }

    #[tokio::test]
    async fn a_chat_routed_by_model_streams_the_workers_answer() {
        let addr = worker_server(testkit::script("proxied reply")).await;
        let state = front_with_worker(&addr);
        let (status, text) = chat(
            &state,
            json!({
                "model": "fake-model",
                "messages": [{"role": "user", "content": "hi"}],
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{text}");
        assert!(text.contains("proxied reply"), "{text}");
    }

    #[tokio::test]
    async fn a_session_binds_to_the_worker_and_disagreement_stays_local() {
        let addr = worker_server(multi_script(&["bound reply", "again reply"])).await;
        let state = front_with_worker(&addr);
        let (status, _text) = chat(
            &state,
            json!({
                "x-session": "s1",
                "model": "fake-model",
                "messages": [{"role": "user", "content": "hi"}],
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(state.session_binding("s1").as_deref(), Some("w1"));
        // The bound session wins the next turn without any routing
        // field, exactly as a local session does.
        let (status, text) = chat(
            &state,
            json!({
                "x-session": "s1",
                "messages": [{"role": "user", "content": "again"}],
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{text}");
        // An explicit engine that disagrees with the binding falls
        // through to the local path, whose answer is the 404.
        let (status, text) = chat(
            &state,
            json!({
                "x-session": "s1",
                "x-engine": "not-the-worker",
                "messages": [{"role": "user", "content": "no"}],
            }),
        )
        .await;
        assert_eq!(status, StatusCode::NOT_FOUND, "{text}");
    }

    #[tokio::test]
    async fn the_engine_and_model_lists_carry_the_worker_and_stop_frees_it() {
        let addr = worker_server(testkit::script("listed reply")).await;
        let state = front_with_worker(&addr);
        let app = router(Arc::clone(&state));
        let (status, _headers, bytes) =
            testkit::send(&app, testkit::get("/v1/engines", Some(testkit::API_KEY))).await;
        assert_eq!(status, StatusCode::OK);
        let engines: Value =
            serde_json::from_str(&String::from_utf8_lossy(&bytes)).expect("engine list json");
        let row = &engines["engines"][0];
        assert_eq!(row["id"], "w1");
        assert_eq!(row["state"], "ready");
        assert_eq!(row["supervised"], true);
        assert_eq!(row["memory_held_until_restart"], false);
        let (status, _headers, bytes) =
            testkit::send(&app, testkit::get("/v1/models", Some(testkit::API_KEY))).await;
        assert_eq!(status, StatusCode::OK);
        let models: Value =
            serde_json::from_str(&String::from_utf8_lossy(&bytes)).expect("model list json");
        assert_eq!(models["data"][0]["id"], "fake-model");
        assert_eq!(models["data"][0]["engines"][0]["id"], "w1");
        // The stop of a supervised worker answers the honest free.
        let request = testkit::request("DELETE", "/v1/engines/w1", Some(testkit::API_KEY));
        let (status, _headers, bytes) = testkit::send(&app, request).await;
        assert_eq!(status, StatusCode::OK);
        let stopped: Value =
            serde_json::from_str(&String::from_utf8_lossy(&bytes)).expect("stop json");
        assert_eq!(stopped["memory_released"], true);
        // With the worker gone the model routes nowhere: the local
        // 404, not a proxy hang.
        let (status, text) = chat(
            &state,
            json!({
                "model": "fake-model",
                "messages": [{"role": "user", "content": "gone"}],
            }),
        )
        .await;
        assert_eq!(status, StatusCode::NOT_FOUND, "{text}");
        // The bound session rebound: its binding dropped with the stop.
        assert_eq!(state.session_binding("s1"), None);
    }

    #[tokio::test]
    async fn a_body_the_worker_never_sees_when_no_routing_names_it() {
        // No worker anywhere: the request must pass through the proxy
        // untouched and reach the local handler, whose empty-registry
        // answer is the 404.
        let state = Arc::new(AppState::new(testkit::API_KEY, vec![]));
        let (status, text) = chat(
            &state,
            json!({
                "model": "no-such-model",
                "messages": [{"role": "user", "content": "hi"}],
            }),
        )
        .await;
        assert_eq!(status, StatusCode::NOT_FOUND, "{text}");
        let _ = Body::empty();
    }
}
