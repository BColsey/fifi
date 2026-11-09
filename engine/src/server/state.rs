//! Routing rules: the scheduler design ("Request flow") and the server design
//! ("Session handling"). A named session (`x-session`) is an attribution
//! binding: the first request that carries it binds it to the engine that
//! serves it, and later requests with the same id return to that engine.
//! It carries no conversation state of its own; the chat history is the
//! client's. The registry is bounded and evicts the least recently used
//! binding, after which the id binds again on its next request.

use super::error::ServerError;
use super::reqlog::{RecentRing, Sink, default_sink};
use super::sampler_trace::{SamplerTraceRing, TraceMeta};
use super::wire::{EngineErrorInfo, EngineInfo, Model, ModelList, state_name};
use crate::EngError;
use crate::family::sampler::{CaptureHandle, SamplingDefaults};
use crate::sched::worker::{EngineHandle, EngineState, RetryClass};
use crate::tokenizer::TokenizerBundle;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// The quiet time after which a stream sends a keep-alive comment (the server design).
const KEEP_ALIVE: Duration = Duration::from_secs(15);

pub(crate) struct EngineEntry {
    /// The engine id of `x-engine`.
    pub(crate) id: String,
    pub(crate) model: String,
    pub(crate) handle: EngineHandle,
    pub(crate) bundle: Arc<TokenizerBundle>,
    /// The most tokens of prompt plus completion that the engine accepts. The family of the engine
    /// reports it when it starts. The chat handler reads no other source for it.
    pub(crate) context_limit: u32,
    /// True when the chat handler sends the prompt boundary of each job to the engine
    /// (`prefix_reuse` in the configuration file).
    pub(crate) prefix_reuse: bool,
    /// The per-engine warm-turn tokenization cache (one slot: this is a
    /// single-conversation-stream deployment; an interleaved conversation
    /// fails the render-prefix check, falls back to the derived path, and
    /// reseeds — always exact, never a cross-session reuse). None until
    /// the first eligible turn seeds it. Bounded by the longest turn:
    /// head render + head ids + the kept last-turn result and digest
    /// (~1.5 MB at 128K tokens), all reported by `accounted_bytes()`.
    pub(crate) warm: std::sync::Arc<std::sync::Mutex<Option<crate::tokenizer::WarmTurnCache>>>,
    /// The sampling defaults of this engine: the `temperature`, `top_p`, and `top_k` that its
    /// `model_dir/generation_config.json` sets, with the contract default for an absent file or
    /// key. The chat handler applies them to every field that a request omits.
    pub(crate) defaults: SamplingDefaults,
    /// The resolved per-key configuration rows retained at start for
    /// `GET /v1/system` (the `config effective` view over the settings the
    /// engine actually started with — never a second resolution). Empty
    /// for families whose per-key origins are not composed yet.
    pub(crate) effective: Vec<crate::capability::effective::EffectiveSetting>,
}

pub(crate) struct AppState {
    api_key: Vec<u8>,
    engines: Vec<EngineEntry>,
    started: u64,
    requests: AtomicU64,
    keep_alive: Duration,
    /// The ids of the engines whose failure text the log already holds.
    reported: Mutex<HashSet<String>>,
    log: Box<dyn Fn(&str) + Send + Sync>,
    request_log: Sink,
    /// The monitor's recent-request ring, fed by the served routes' log
    /// lines and read by `GET /v1/recent-requests`.
    recent: RecentRing,
    /// Whether the process permits sampler-trace capture: the config's
    /// `sample_trace` setting, checked before a request arms a sampler.
    sample_trace: bool,
    /// The sampler-trace ring, fed by submit when a request arms capture
    /// and read by `GET /v1/sampler-trace`.
    traces: SamplerTraceRing,
    /// The previous prompt ids per model, for the request log's LCP row.
    prompt_memory: Arc<Mutex<HashMap<String, Vec<u32>>>>,
    /// The built frontend directory served at `/`, when configured.
    web_dir: Option<std::path::PathBuf>,
    /// Named-session bindings: session id → engine id and last use
    /// (Unix milliseconds). Bounded by SESSION_CAP with LRU eviction.
    sessions: Mutex<HashMap<String, (String, u64)>>,
    /// The last GPU telemetry snapshot and when it was taken. The
    /// monitor polls every 5 s; the subprocess probe is cached so the
    /// poll cannot spawn it more than once per TELEMETRY_TTL.
    telemetry: Mutex<Option<(std::time::Instant, serde_json::Value)>>,
}

/// How long a GPU telemetry snapshot stays fresh.
pub(crate) const TELEMETRY_TTL: std::time::Duration = std::time::Duration::from_secs(2);

/// The most named sessions the registry holds at once.
const SESSION_CAP: usize = 256;

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| {
            u64::try_from(elapsed.as_millis()).unwrap_or(u64::MAX)
        })
}

pub(crate) fn unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_secs())
}

impl AppState {
    /// The caller has checked that the key is not empty.
    pub(crate) fn new(api_key: &str, engines: Vec<EngineEntry>) -> Self {
        Self {
            api_key: api_key.as_bytes().to_vec(),
            engines,
            started: unix_seconds(),
            requests: AtomicU64::new(0),
            keep_alive: KEEP_ALIVE,
            reported: Mutex::new(HashSet::new()),
            log: Box::new(|line| eprintln!("error: {line}")),
            request_log: default_sink(),
            recent: RecentRing::new(),
            sample_trace: false,
            traces: SamplerTraceRing::default(),
            prompt_memory: Arc::new(Mutex::new(HashMap::new())),
            web_dir: None,
            sessions: Mutex::new(HashMap::new()),
            telemetry: Mutex::new(None),
        }
    }

    /// Builder: run.rs applies the config's `sample_trace` setting.
    pub(crate) fn with_sample_trace(mut self, on: bool) -> Self {
        self.sample_trace = on;
        self
    }

    pub(crate) fn sample_trace_enabled(&self) -> bool {
        self.sample_trace
    }

    /// Registers one reply's capture buffer at submit: in-progress captures
    /// count against the ring's budgets from the first token on.
    pub(crate) fn register_trace(&self, id: &str, meta: TraceMeta, buffer: CaptureHandle) {
        self.traces.register(id, meta, buffer);
    }

    /// One reply's trace metadata and buffer, when the ring still holds it.
    pub(crate) fn sampler_trace(&self, id: &str) -> Option<(TraceMeta, CaptureHandle)> {
        self.traces.get(id)
    }

    pub(crate) fn with_web_dir(mut self, dir: std::path::PathBuf) -> Self {
        self.web_dir = Some(dir);
        self
    }

    pub(crate) fn web_dir(&self) -> Option<&std::path::Path> {
        self.web_dir.as_deref()
    }

    pub(crate) fn note_prompt(&self, model: &str, ids: &[u32]) -> u32 {
        let mut memory = self.prompt_memory.lock().unwrap_or_else(|e| e.into_inner());
        let lcp = memory
            .get(model)
            .map(|prev| prev.iter().zip(ids).take_while(|(a, b)| a == b).count())
            .unwrap_or(0) as u32;
        memory.insert(model.to_string(), ids.to_vec());
        lcp
    }

    pub(crate) fn request_log(&self) -> Sink {
        Arc::clone(&self.request_log)
    }

    #[cfg(test)]
    pub(crate) fn with_request_log(mut self, sink: impl std::io::Write + Send + 'static) -> Self {
        self.request_log = Arc::new(Mutex::new(sink));
        self
    }

    /// The monitor's recent-request ring: one per state, so every test
    /// server owns its own rows.
    pub(crate) fn recent_ring(&self) -> RecentRing {
        self.recent.clone()
    }

    #[cfg(test)]
    pub(crate) fn with_log(mut self, sink: impl Fn(&str) + Send + Sync + 'static) -> Self {
        self.log = Box::new(sink);
        self
    }

    /// Tests use this to avoid a wait of 15 seconds.
    #[cfg(test)]
    pub(crate) fn with_keep_alive(mut self, interval: Duration) -> Self {
        self.keep_alive = interval;
        self
    }

    pub(crate) fn keep_alive(&self) -> Duration {
        self.keep_alive
    }

    pub(crate) fn api_key(&self) -> &[u8] {
        &self.api_key
    }

    pub(crate) fn next_completion_id(&self) -> String {
        let number = self.requests.fetch_add(1, Ordering::Relaxed) + 1;
        format!("chatcmpl-{number:016x}")
    }

    pub(crate) fn resolve(
        &self,
        session: Option<&str>,
        engine: Option<&str>,
        model: Option<&str>,
    ) -> Result<&EngineEntry, ServerError> {
        if let Some(id) = session {
            return self.resolve_session(id, engine, model);
        }
        self.resolve_unbound(engine, model)
    }

    fn resolve_unbound(
        &self,
        engine: Option<&str>,
        model: Option<&str>,
    ) -> Result<&EngineEntry, ServerError> {
        if let Some(id) = engine {
            return self
                .engines
                .iter()
                .find(|entry| entry.id == id)
                .ok_or_else(|| ServerError::NotFound(format!("engine {id} does not exist")));
        }
        if let Some(name) = model {
            return self
                .engines
                .iter()
                .find(|entry| entry.model == name)
                .ok_or_else(|| ServerError::NotFound(format!("model {name} does not exist")));
        }
        Err(ServerError::Eng(EngError::BadRequest(
            "one of model, x-engine, or x-session is required".to_string(),
        )))
    }

    /// A known session returns to its bound engine; an explicit engine
    /// or model that disagrees with the binding is a 404, because the
    /// session does not exist there. An unknown session binds to the
    /// engine the request's own routing picks, and a session with no
    /// routing field at all stays a 404, as before sessions existed.
    fn resolve_session(
        &self,
        id: &str,
        engine: Option<&str>,
        model: Option<&str>,
    ) -> Result<&EngineEntry, ServerError> {
        let bound = {
            let mut sessions = self.sessions.lock().unwrap_or_else(PoisonError::into_inner);
            sessions.get_mut(id).map(|(engine_id, used)| {
                *used = now_ms();
                engine_id.clone()
            })
        };
        if let Some(engine_id) = bound {
            let entry = self
                .engines
                .iter()
                .find(|entry| entry.id == engine_id)
                .ok_or_else(|| ServerError::NotFound(format!("session {id} does not exist")))?;
            if let Some(wanted) = engine
                && wanted != entry.id
            {
                return Err(ServerError::NotFound(format!(
                    "session {id} does not exist on engine {wanted}"
                )));
            }
            if let Some(name) = model
                && name != entry.model
            {
                return Err(ServerError::NotFound(format!(
                    "session {id} does not exist for model {name}"
                )));
            }
            return Ok(entry);
        }
        if engine.is_none() && model.is_none() {
            return Err(ServerError::NotFound(format!(
                "session {id} does not exist"
            )));
        }
        let entry = self.resolve_unbound(engine, model)?;
        let mut sessions = self.sessions.lock().unwrap_or_else(PoisonError::into_inner);
        if sessions.len() >= SESSION_CAP
            && !sessions.contains_key(id)
            && let Some(oldest) = sessions
                .iter()
                .min_by_key(|(_, (_, used))| *used)
                .map(|(key, _)| key.clone())
        {
            sessions.remove(&oldest);
        }
        sessions.insert(id.to_string(), (entry.id.clone(), now_ms()));
        // The entry borrows self.engines; the registry holds only its id.
        Ok(entry)
    }

    /// A fresh-enough telemetry snapshot, when one is cached.
    pub(crate) fn telemetry_cached(&self) -> Option<serde_json::Value> {
        let guard = self
            .telemetry
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        match guard.as_ref() {
            Some((taken, value)) if taken.elapsed() < TELEMETRY_TTL => Some(value.clone()),
            _ => None,
        }
    }

    pub(crate) fn telemetry_store(&self, value: serde_json::Value) {
        *self
            .telemetry
            .lock()
            .unwrap_or_else(PoisonError::into_inner) = Some((std::time::Instant::now(), value));
    }

    /// The bound engine id of a session (test introspection).
    #[cfg(test)]
    pub(crate) fn session_engine(&self, id: &str) -> Option<String> {
        self.sessions
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get(id)
            .map(|(engine_id, _)| engine_id.clone())
    }

    pub(crate) fn is_ready(&self) -> bool {
        self.engines
            .iter()
            .any(|entry| matches!(entry.handle.state(), EngineState::Ready | EngineState::Busy))
    }

    /// The message of the `not_ready` answer: the state of every engine. It never shows the text
    /// of a failure, because the route is public. `new_failures` supplies that text for the log.
    pub(crate) fn not_ready_message(&self) -> String {
        if self.engines.is_empty() {
            return "no engine is configured".to_string();
        }
        let parts: Vec<String> = self
            .engines
            .iter()
            .map(|entry| format!("{} is {}", entry.id, state_name(entry.handle.state())))
            .collect();
        format!("no engine is ready: {}", parts.join("; "))
    }

    fn new_failures(&self) -> Vec<String> {
        let mut reported = self.reported.lock().unwrap_or_else(PoisonError::into_inner);
        self.engines
            .iter()
            .filter_map(|entry| {
                let cause = entry.handle.failure()?;
                reported
                    .insert(entry.id.clone())
                    .then(|| format!("engine {} failed: {cause}", entry.id))
            })
            .collect()
    }

    pub(crate) fn log_new_failures(&self) {
        for line in self.new_failures() {
            (self.log)(&line);
        }
    }

    /// Every engine entry once, in configuration order. For status
    /// surfaces that need fields the models list does not carry,
    /// such as the context limit.
    pub(crate) fn entries(&self) -> Vec<&EngineEntry> {
        self.engines.iter().collect()
    }

    pub(crate) fn models(&self) -> ModelList {
        let mut models: Vec<Model> = Vec::new();
        for entry in &self.engines {
            let info = EngineInfo {
                id: entry.id.clone(),
                model: entry.model.clone(),
                state: state_name(entry.handle.state()),
                load_stage: entry.handle.load_stage(),
                error: entry.handle.failure_info().map(|failure| EngineErrorInfo {
                    code: failure.code,
                    stage: failure.stage,
                    message: failure.message,
                    retry: match failure.retry {
                        RetryClass::AfterCorrection => "after_correction",
                        RetryClass::Unknown => "unknown",
                    },
                    suggestion: suggestion_for(failure.code),
                }),
            };
            match models.iter_mut().find(|model| model.id == entry.model) {
                Some(model) => model.engines.push(info),
                None => models.push(Model::new(entry.model.clone(), self.started, vec![info])),
            }
        }
        ModelList::new(models)
    }
}

/// The corrective action the recorded code justifies, and no more:
/// codes without an evidence-backed user action get none.
fn suggestion_for(code: &str) -> Option<&'static str> {
    match code {
        "memory_plan" => Some(
            "free memory or lower the context, then restart; `enginesrv check <model_dir>` estimates the largest fitting context",
        ),
        "checkpoint_failure" => Some(
            "verify the model and tokenizer files; `enginesrv check <model_dir>` runs the same file and pin checks",
        ),
        "bad_request" | "context_limit" => {
            Some("correct the engine entry in the configuration file, then restart")
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::{AppState, unix_seconds};
    use crate::EngError;
    use crate::sched::worker::EngineState;
    use crate::server::error::ServerError;
    use crate::server::testkit;

    fn two_engines() -> (AppState, Vec<crate::sched::worker::Engine>) {
        let first = testkit::engine(Vec::new());
        let second = testkit::engine(Vec::new());
        let entries = vec![
            testkit::entry("e1", "model-a", &first, 4_096),
            testkit::entry("e2", "model-b", &second, 4_096),
        ];
        (
            AppState::new(testkit::API_KEY, entries),
            vec![first, second],
        )
    }

    #[test]
    fn x_engine_takes_precedence_over_model() {
        let (state, _engines) = two_engines();
        let picked = state
            .resolve(None, Some("e2"), Some("model-a"))
            .expect("resolve");
        assert_eq!(picked.id, "e2");
    }

    #[test]
    fn a_model_name_picks_the_first_engine_that_holds_it() {
        let engines = [
            testkit::engine(Vec::new()),
            testkit::engine(Vec::new()),
            testkit::engine(Vec::new()),
        ];
        let entries = vec![
            testkit::entry("e1", "model-a", &engines[0], 4_096),
            testkit::entry("e2", "model-b", &engines[1], 4_096),
            testkit::entry("e3", "model-b", &engines[2], 4_096),
        ];
        let state = AppState::new(testkit::API_KEY, entries);
        let picked = state.resolve(None, None, Some("model-b")).expect("resolve");
        assert_eq!(picked.id, "e2");
    }

    #[test]
    fn unknown_engine_or_model_returns_404() {
        let (state, _engines) = two_engines();
        let engine = state.resolve(None, Some("e9"), None).err().expect("engine");
        assert_eq!(engine.status().as_u16(), 404);
        let model = state
            .resolve(None, None, Some("nope"))
            .err()
            .expect("model");
        assert_eq!(model.status().as_u16(), 404);
        let both = state
            .resolve(None, Some("e9"), Some("model-a"))
            .err()
            .expect("both");
        assert_eq!(
            both.status().as_u16(),
            404,
            "a bad x-engine does not fall back to model"
        );
    }

    #[test]
    fn x_session_missing_returns_404() {
        // A session id with no routing field has nothing to bind to.
        let (state, _engines) = two_engines();
        let error = state
            .resolve(Some("s1"), None, None)
            .err()
            .expect("session");
        assert_eq!(error.status().as_u16(), 404);
        assert_eq!(error.code(), "not_found");
    }

    #[test]
    fn a_session_binds_on_first_use_and_sticks_to_its_engine() {
        let (state, _engines) = two_engines();
        let first = state
            .resolve(Some("s1"), None, Some("model-b"))
            .expect("bind");
        assert_eq!(first.id, "e2");
        assert_eq!(state.session_engine("s1").as_deref(), Some("e2"));
        // Later requests return to the bound engine even when they
        // name the other engine's model only by session.
        let again = state.resolve(Some("s1"), None, None).expect("sticky");
        assert_eq!(again.id, "e2");
        // An explicit conflicting engine or model does not move it.
        assert_eq!(
            state
                .resolve(Some("s1"), Some("e1"), None)
                .err()
                .expect("conflict")
                .status()
                .as_u16(),
            404
        );
        assert_eq!(
            state
                .resolve(Some("s1"), None, Some("model-a"))
                .err()
                .expect("conflict")
                .status()
                .as_u16(),
            404
        );
    }

    #[test]
    fn a_request_without_any_routing_field_is_a_bad_request() {
        let (state, _engines) = two_engines();
        let error = state.resolve(None, None, None).err().expect("no route");
        assert!(matches!(error, ServerError::Eng(EngError::BadRequest(_))));
    }

    #[test]
    fn completion_ids_are_unique_and_fixed_width() {
        let (state, _engines) = two_engines();
        let first = state.next_completion_id();
        let second = state.next_completion_id();
        assert_eq!(first, "chatcmpl-0000000000000001");
        assert_eq!(second, "chatcmpl-0000000000000002");
    }

    #[test]
    fn the_state_is_ready_only_while_an_engine_can_take_a_job() {
        let (state, _engines) = two_engines();
        for entry in &state.engines {
            testkit::wait_for_state(&entry.handle, EngineState::Ready);
        }
        assert!(state.is_ready());
        let broken = testkit::failing_engine();
        let entries = vec![testkit::entry("e1", "model-a", &broken, 4_096)];
        let down = AppState::new(testkit::API_KEY, entries);
        testkit::wait_for_state(&down.engines[0].handle, EngineState::Error);
        assert!(!down.is_ready());
        let empty = AppState::new(testkit::API_KEY, Vec::new());
        assert!(!empty.is_ready());
    }

    #[test]
    fn the_not_ready_message_names_each_engine_and_its_state_and_hides_the_failure() {
        let broken = testkit::failing_engine();
        let (waiting, gate) = testkit::gated_engine(Vec::new());
        let entries = vec![
            testkit::entry("e1", "model-a", &broken, 4_096),
            testkit::entry("e2", "model-b", &waiting, 4_096),
        ];
        let state = AppState::new(testkit::API_KEY, entries);
        testkit::wait_for_state(&state.engines[0].handle, EngineState::Error);
        let message = state.not_ready_message();
        assert!(message.starts_with("no engine is ready: "), "{message}");
        assert!(message.contains("e1 is error"), "{message}");
        assert!(!message.contains("fixture load failure"), "{message}");
        assert!(message.contains("e2 is loading"), "{message}");
        drop(gate);
    }

    #[test]
    fn a_failure_text_goes_to_the_log_once_for_each_failed_engine() {
        let broken = testkit::failing_engine();
        let (waiting, gate) = testkit::gated_engine(Vec::new());
        let entries = vec![
            testkit::entry("e1", "model-a", &broken, 4_096),
            testkit::entry("e2", "model-b", &waiting, 4_096),
        ];
        let state = AppState::new(testkit::API_KEY, entries);
        testkit::wait_for_state(&state.engines[0].handle, EngineState::Error);
        let first = state.new_failures();
        assert_eq!(first.len(), 1, "{first:?}");
        assert!(first[0].starts_with("engine e1 failed: "), "{first:?}");
        assert!(first[0].contains("fixture load failure"), "{first:?}");
        assert!(
            state.new_failures().is_empty(),
            "the text was reported twice"
        );
        drop(gate);
    }

    #[test]
    fn the_not_ready_message_of_a_server_without_engines_says_so() {
        let state = AppState::new(testkit::API_KEY, Vec::new());
        assert_eq!(state.not_ready_message(), "no engine is configured");
    }

    #[test]
    fn the_keep_alive_interval_is_fifteen_seconds_unless_a_test_changes_it() {
        let state = AppState::new(testkit::API_KEY, Vec::new());
        assert_eq!(state.keep_alive(), std::time::Duration::from_secs(15));
        let quick = state.with_keep_alive(std::time::Duration::from_millis(5));
        assert_eq!(quick.keep_alive(), std::time::Duration::from_millis(5));
    }

    #[test]
    fn models_group_engines_by_model_name() {
        let first = testkit::engine(Vec::new());
        let second = testkit::engine(Vec::new());
        let third = testkit::engine(Vec::new());
        let entries = vec![
            testkit::entry("e1", "model-a", &first, 4_096),
            testkit::entry("e2", "model-b", &second, 4_096),
            testkit::entry("e3", "model-a", &third, 4_096),
        ];
        let state = AppState::new(testkit::API_KEY, entries);
        let list = serde_json::to_value(state.models()).expect("serialize");
        let ids: Vec<&str> = list["data"]
            .as_array()
            .expect("data")
            .iter()
            .map(|model| model["id"].as_str().expect("id"))
            .collect();
        assert_eq!(ids, ["model-a", "model-b"]);
        let engines_of_a: Vec<&str> = list["data"][0]["engines"]
            .as_array()
            .expect("engines")
            .iter()
            .map(|engine| engine["id"].as_str().expect("engine id"))
            .collect();
        assert_eq!(engines_of_a, ["e1", "e3"]);
    }

    #[test]
    fn the_model_creation_time_is_the_start_time() {
        let before = unix_seconds();
        let (state, _engines) = two_engines();
        let created =
            serde_json::to_value(state.models()).expect("serialize")["data"][0]["created"]
                .as_u64()
                .expect("created");
        assert!(created >= before && created <= unix_seconds(), "{created}");
    }
}
