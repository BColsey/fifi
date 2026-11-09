use super::fake::load_bundle;
use super::state::EngineEntry;
use crate::EngError;
use crate::exec_fake::FakeExecutor;
use crate::family::executor::{ExecutorOps, GenerationState};
use crate::family::sampler::SamplingParams;
use crate::sched::worker::{Engine, EngineHandle, EngineState, spawn_engine};
use crate::tokenizer::{DecodeBatch, DecodeFinish, IncrementalDecoder, TokenizerBundle, ToolSpec};
use axum::Router;
use axum::body::{Body, Bytes};
use axum::http::{HeaderMap, Request, StatusCode, header};
use serde_json::{Value, json};
use std::collections::VecDeque;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tower::ServiceExt;

pub(super) const API_KEY: &str = "sk-test";

/// The end-of-sequence id of the fixture tokenizer.
pub(super) const EOS: u32 = 256;

/// Loads the fixture tokenizer in `testdata/tokenizer`.
/// Its ids 0 to 255 are the byte values. Id 256 is `<|im_end|>` and id 257 is `<|im_start|>`.
pub(super) fn bundle() -> Arc<TokenizerBundle> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/server/testdata/tokenizer");
    Arc::new(load_bundle(&dir).expect("load fixture tokenizer"))
}

/// Loads the fixture tokenizer with the pinned Qwen chat template (`testdata/chat_template.jinja`)
/// in place of the toy template. The special ids stay as above: the pinned template writes
/// `<|im_start|>` and `<|im_end|>` as one id each and every other character as its byte value.
pub(super) fn bundle_with_pinned_template() -> Arc<TokenizerBundle> {
    bundle_with_template(include_str!("../../testdata/chat_template.jinja"))
}

/// A bundle on the fixture tokenizer whose template is the real Flint one:
/// its grammar never strips reasoning at the template, so it is the honest
/// probe for the serving-side rollback gate.
pub(super) fn bundle_with_flint_template() -> Arc<TokenizerBundle> {
    bundle_with_template(include_str!(
        "../../testdata/flint/tokenizer/chat_template.jinja"
    ))
}

pub(super) fn bundle_with_template(template_source: &str) -> Arc<TokenizerBundle> {
    let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/server/testdata/tokenizer");
    let dir = tempfile::tempdir().expect("tempdir");
    for name in [
        "tokenizer.json",
        "tokenizer_config.json",
        "generation_config.json",
    ] {
        std::fs::copy(source.join(name), dir.path().join(name)).expect("copy fixture file");
    }
    std::fs::write(dir.path().join("chat_template.jinja"), template_source)
        .expect("write the template");
    let mut bundle = load_bundle(dir.path()).expect("load the fixture bundle");
    // Env-proof: the flags read the environment at construction, and no
    // fixture may depend on the shell that runs the tests.
    bundle.force_flags_off();
    Arc::new(bundle)
}

/// The token ids of ASCII text under the fixture tokenizer.
pub(super) fn ids(text: &str) -> Vec<u32> {
    text.bytes().map(u32::from).collect()
}

pub(super) fn script(text: &str) -> Vec<Vec<u32>> {
    let mut batches: Vec<Vec<u32>> = ids(text).into_iter().map(|id| vec![id]).collect();
    batches.push(vec![EOS]);
    batches
}

pub(super) fn engine(script: Vec<Vec<u32>>) -> Engine {
    spawn_engine(512, move |_| {
        Ok((
            FakeExecutor::new(script, &[EOS], 100_000),
            Default::default(),
        ))
    })
    .expect("spawn engine")
}

/// The default limits are zero roots and one entry.
pub(super) fn engine_with_prefix_limits(
    script: Vec<Vec<u32>>,
    roots: usize,
    entries: usize,
) -> Engine {
    spawn_engine(512, move |_| {
        let mut executor = FakeExecutor::new(script, &[EOS], 100_000);
        executor.with_prefix_limits(roots, entries);
        Ok((executor, Default::default()))
    })
    .expect("spawn engine")
}

pub(super) fn failing_engine() -> Engine {
    spawn_engine::<FakeExecutor, _>(512, |_| {
        Err(EngError::Checkpoint("fixture load failure".to_string()))
    })
    .expect("spawn engine")
}

/// An engine whose load is refused with `error` on the engine thread, the way a real family's
/// `open` refuses at admission.
pub(super) fn load_refused_engine(error: EngError) -> Engine {
    spawn_engine::<FakeExecutor, _>(512, move |_| Err(error)).expect("spawn engine")
}

/// An engine that fails its load, with the engine's own error, once the returned sender fires or
/// is dropped: the failure arrives while the server is already listening and watching.
pub(super) fn gated_failing_engine() -> (Engine, std::sync::mpsc::Sender<()>) {
    let (gate, wait) = std::sync::mpsc::channel::<()>();
    let engine = spawn_engine::<FakeExecutor, _>(512, move |_| {
        let _ = wait.recv();
        Err(EngError::MemoryPlan(
            "cuda admission: need 10 bytes, device has 1 free".to_string(),
        ))
    })
    .expect("spawn engine");
    (engine, gate)
}

pub(super) fn gated_engine(script: Vec<Vec<u32>>) -> (Engine, std::sync::mpsc::Sender<()>) {
    let (gate, wait) = std::sync::mpsc::channel::<()>();
    let engine = spawn_engine(512, move |_| {
        let _ = wait.recv();
        Ok((
            FakeExecutor::new(script, &[EOS], 100_000),
            Default::default(),
        ))
    })
    .expect("spawn engine");
    (engine, gate)
}

pub(super) fn entry(id: &str, model: &str, engine: &Engine, context_limit: u32) -> EngineEntry {
    entry_with_defaults(
        id,
        model,
        engine,
        context_limit,
        crate::family::sampler::SamplingDefaults::CONTRACT,
    )
}

pub(super) fn entry_with_defaults(
    id: &str,
    model: &str,
    engine: &Engine,
    context_limit: u32,
    defaults: crate::family::sampler::SamplingDefaults,
) -> EngineEntry {
    entry_with(id, model, engine, context_limit, defaults, Vec::new())
}

/// An entry carrying retained effective-configuration rows, for the
/// `/v1/system` rendering tests.
pub(super) fn entry_with_effective(
    id: &str,
    model: &str,
    engine: &Engine,
    context_limit: u32,
    effective: Vec<crate::capability::effective::EffectiveSetting>,
) -> EngineEntry {
    entry_with(
        id,
        model,
        engine,
        context_limit,
        crate::family::sampler::SamplingDefaults::CONTRACT,
        effective,
    )
}

fn entry_with(
    id: &str,
    model: &str,
    engine: &Engine,
    context_limit: u32,
    defaults: crate::family::sampler::SamplingDefaults,
    effective: Vec<crate::capability::effective::EffectiveSetting>,
) -> EngineEntry {
    EngineEntry {
        id: id.to_string(),
        model: model.to_string(),
        handle: engine.handle(),
        bundle: bundle(),
        warm: std::sync::Arc::new(std::sync::Mutex::new(None)),
        context_limit,
        prefix_reuse: false,
        defaults,
        effective,
    }
}

pub(super) fn wait_for_state(handle: &EngineHandle, wanted: EngineState) {
    let deadline = Instant::now() + Duration::from_secs(5);
    while handle.state() != wanted {
        assert!(
            Instant::now() < deadline,
            "engine stayed in {:?}, wanted {wanted:?}",
            handle.state()
        );
        std::thread::sleep(Duration::from_millis(2));
    }
}

pub(super) fn get(uri: &str, key: Option<&str>) -> Request<Body> {
    request("GET", uri, key)
}

pub(super) fn request(method: &str, uri: &str, key: Option<&str>) -> Request<Body> {
    let mut builder = Request::builder().method(method).uri(uri);
    if let Some(key) = key {
        builder = builder.header(header::AUTHORIZATION, format!("Bearer {key}"));
    }
    builder.body(Body::empty()).expect("build request")
}

pub(super) async fn send(
    router: &Router,
    request: Request<Body>,
) -> (StatusCode, HeaderMap, Bytes) {
    let response = router
        .clone()
        .oneshot(request)
        .await
        .expect("route the request");
    let (parts, body) = response.into_parts();
    let bytes = axum::body::to_bytes(body, usize::MAX)
        .await
        .expect("read the body");
    (parts.status, parts.headers, bytes)
}

pub(super) fn json_of(bytes: &Bytes) -> Value {
    serde_json::from_slice(bytes).expect("json body")
}

pub(super) fn refusing_engine(error: EngError) -> Engine {
    spawn_engine(512, move |_| {
        Ok((RefusingExecutor(error), Default::default()))
    })
    .expect("spawn engine")
}

/// An executor whose every `prefill` call returns the same scripted error. The refusal is
/// persistent, so an eviction retry sees the same denial the first attempt saw.
struct RefusingExecutor(EngError);

impl ExecutorOps for RefusingExecutor {
    type Session = ();

    fn history<'s>(&self, _st: &'s ()) -> &'s [u32] {
        &[]
    }

    fn reset(&mut self, _st: &mut ()) -> Result<(), EngError> {
        Ok(())
    }

    fn prefill(&mut self, _st: &mut (), _toks: &[u32]) -> Result<(), EngError> {
        Err(self.0.clone())
    }

    fn decode(
        &mut self,
        _st: &mut (),
        _generation: &mut GenerationState,
    ) -> Result<DecodeBatch, EngError> {
        Err(EngError::BadRequest(
            "decode after a refused prefill".to_string(),
        ))
    }

    fn boundary(&mut self, _st: &mut ()) {}
}

/// A fresh generation state over the fixture tokenizer, with a greedy sampler.
pub(super) fn generation() -> GenerationState {
    GenerationState::new(IncrementalDecoder::new(bundle().tokenizer()))
}

pub(super) fn add_tool() -> ToolSpec {
    ToolSpec(json!({"type": "function", "function": {
        "name": "add",
        "parameters": {"type": "object", "properties": {
            "a": {"type": "integer"}, "b": {"type": "number"}}}}}))
}

pub(super) enum Ending {
    Error(EngError),
    Cancel,
}

struct Ends {
    ids: VecDeque<u32>,
    ending: Option<Ending>,
}

impl ExecutorOps for Ends {
    type Session = ();

    fn history<'s>(&self, _st: &'s ()) -> &'s [u32] {
        &[]
    }

    fn reset(&mut self, _st: &mut ()) -> Result<(), EngError> {
        Ok(())
    }

    fn prefill(&mut self, _st: &mut (), _toks: &[u32]) -> Result<(), EngError> {
        Ok(())
    }

    fn decode(
        &mut self,
        _st: &mut (),
        generation: &mut GenerationState,
    ) -> Result<DecodeBatch, EngError> {
        if let Some(id) = self.ids.pop_front() {
            return Ok(DecodeBatch {
                tokens: vec![id],
                text: generation.decoder.push(id)?,
                finish: None,
            });
        }
        match self.ending.take() {
            Some(Ending::Error(error)) => Err(error),
            _ => Ok(DecodeBatch {
                tokens: Vec::new(),
                text: String::new(),
                finish: Some(DecodeFinish::Cancelled),
            }),
        }
    }

    fn boundary(&mut self, _st: &mut ()) {}
}

pub(super) fn ending_engine(text: &str, ending: Ending) -> Engine {
    let ids: VecDeque<u32> = ids(text).into();
    spawn_engine(512, move |_| {
        Ok((
            Ends {
                ids,
                ending: Some(ending),
            },
            (),
        ))
    })
    .expect("spawn engine")
}

/// It never finishes on its own, so the output limit of the job or a stop string ends the reply.
struct Probe {
    prompt: Arc<Mutex<Vec<u32>>>,
    sampling: Arc<Mutex<Option<SamplingParams>>>,
    delay: Duration,
    decodes: Arc<AtomicUsize>,
}

impl ExecutorOps for Probe {
    type Session = Vec<u32>;

    fn history<'s>(&self, st: &'s Vec<u32>) -> &'s [u32] {
        st
    }

    fn reset(&mut self, st: &mut Vec<u32>) -> Result<(), EngError> {
        st.clear();
        Ok(())
    }

    fn prefill(&mut self, st: &mut Vec<u32>, toks: &[u32]) -> Result<(), EngError> {
        st.extend_from_slice(toks);
        self.prompt
            .lock()
            .expect("prompt lock")
            .extend_from_slice(toks);
        Ok(())
    }

    fn decode(
        &mut self,
        st: &mut Vec<u32>,
        generation: &mut GenerationState,
    ) -> Result<DecodeBatch, EngError> {
        std::thread::sleep(self.delay);
        self.decodes.fetch_add(1, Ordering::SeqCst);
        if generation.cancel.load(Ordering::SeqCst) {
            return Ok(DecodeBatch {
                tokens: Vec::new(),
                text: generation.decoder.finish()?,
                finish: Some(DecodeFinish::Cancelled),
            });
        }
        if generation.remaining_tokens() == Some(0) {
            return Ok(DecodeBatch {
                tokens: Vec::new(),
                text: generation.decoder.finish()?,
                finish: Some(DecodeFinish::Length),
            });
        }
        // The first decode records the settings the chat handler resolved for this job.
        let mut seen = self.sampling.lock().expect("sampling lock");
        if seen.is_none() {
            *seen = Some(generation.sampler.params());
        }
        drop(seen);
        let id = generation.sampler.sample(&[0.0, 1.0])?;
        st.push(id);
        generation.record_generated(1);
        let mut text = generation.decoder.push(id)?;
        let finish = (generation.remaining_tokens() == Some(0)).then_some(DecodeFinish::Length);
        if finish.is_some() {
            text.push_str(&generation.decoder.finish()?);
        }
        Ok(DecodeBatch {
            tokens: vec![id],
            text,
            finish,
        })
    }

    fn boundary(&mut self, _st: &mut Vec<u32>) {}
}

pub(super) struct Probes {
    pub(super) engine: Engine,
    pub(super) prompt: Arc<Mutex<Vec<u32>>>,
    pub(super) sampling: Arc<Mutex<Option<SamplingParams>>>,
    pub(super) decodes: Arc<AtomicUsize>,
}

pub(super) fn probes(delay: Duration) -> Probes {
    probes_behind(delay, None)
}

pub(super) fn gated_probes() -> (Probes, std::sync::mpsc::Sender<()>) {
    let (gate, wait) = std::sync::mpsc::channel::<()>();
    (probes_behind(Duration::ZERO, Some(wait)), gate)
}

fn probes_behind(delay: Duration, gate: Option<std::sync::mpsc::Receiver<()>>) -> Probes {
    let prompt = Arc::new(Mutex::new(Vec::new()));
    let sampling = Arc::new(Mutex::new(None));
    let decodes = Arc::new(AtomicUsize::new(0));
    let probe = Probe {
        prompt: Arc::clone(&prompt),
        sampling: Arc::clone(&sampling),
        delay,
        decodes: Arc::clone(&decodes),
    };
    let engine = spawn_engine(512, move |_| {
        if let Some(wait) = gate {
            let _ = wait.recv();
        }
        Ok((probe, Vec::new()))
    })
    .expect("spawn engine");
    Probes {
        engine,
        prompt,
        sampling,
        decodes,
    }
}

/// An engine that stays `Busy` until the returned job handle is cancelled or dropped: a probe
/// engine with a decode delay, given one job that has no output limit. Returns once the
/// engine is `Busy`.
pub(super) fn busy_engine() -> (Engine, crate::sched::job::JobHandle) {
    let probes = probes(Duration::from_millis(5));
    let (job, handle) = crate::sched::job::Job::new(vec![104], generation()).expect("job");
    probes.engine.handle().try_submit(job).expect("submit");
    wait_for_state(&probes.engine.handle(), EngineState::Busy);
    (probes.engine, handle)
}

pub(super) fn probe_engine() -> (Engine, Arc<Mutex<Vec<u32>>>) {
    let probes = probes(Duration::ZERO);
    (probes.engine, probes.prompt)
}

pub(super) fn probe_engine_with_sampling() -> Probes {
    probes(Duration::ZERO)
}

pub(super) fn wait_until(what: &str, condition: impl Fn() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(5);
    while !condition() {
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        std::thread::sleep(Duration::from_millis(2));
    }
}

pub(super) fn post_json(uri: &str, body: &Value) -> Request<Body> {
    post_bytes(uri, Some(API_KEY), body.to_string().into_bytes())
}

pub(super) fn post_bytes(uri: &str, key: Option<&str>, body: Vec<u8>) -> Request<Body> {
    let mut builder = Request::builder()
        .method("POST")
        .uri(uri)
        .header(header::CONTENT_TYPE, "application/json");
    if let Some(key) = key {
        builder = builder.header(header::AUTHORIZATION, format!("Bearer {key}"));
    }
    builder.body(Body::from(body)).expect("build request")
}

#[derive(Debug, PartialEq, Eq)]
pub(super) struct SseEvent {
    pub(super) name: Option<String>,
    pub(super) data: String,
}

/// Runs one child-process environment check (the decoder.rs re-exec
/// pattern): spawns this test binary with `--exact <selector>`, applies
/// the cache-usage environment (None removes the variable), marks the
/// child with `child_env`, and requires success, the given sentinel in
/// the output, and the runner's own `1 passed` line — so a child that
/// ran zero tests cannot pass. The CHILD env var selects the child body
/// inside the named test.
pub(super) fn child_env_check(
    selector: &str,
    child_env: (&str, &str),
    cache_usage: Option<&str>,
    sentinel: &str,
) {
    let (key, mode) = child_env;
    let mut command = std::process::Command::new(std::env::current_exe().unwrap());
    command
        .args(["--exact", selector, "--nocapture"])
        .env(key, mode);
    match cache_usage {
        Some(value) => {
            command.env(super::wire::CACHE_USAGE_VAR, value);
        }
        None => {
            command.env_remove(super::wire::CACHE_USAGE_VAR);
        }
    }
    let output = command.output().expect("child test run");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.status.success(), "{mode} child failed: {text}");
    assert!(text.contains(sentinel), "{mode}: sentinel missing: {text}");
    assert!(text.contains("1 passed"), "{mode} ran zero tests: {text}");
}

/// The OsStr twin for one non-Unicode cache-usage value: the same
/// re-exec pattern, with an environment value that is not valid UTF-8
/// (b"\xff\xfe") — the one setting a `&str` cannot express. Only a
/// Unix child can carry such a value, so the helper is Unix-only.
#[cfg(unix)]
pub(super) fn child_env_check_raw(
    selector: &str,
    child_env: (&str, &str),
    cache_usage: &std::ffi::OsStr,
    sentinel: &str,
) {
    let (key, mode) = child_env;
    let mut command = std::process::Command::new(std::env::current_exe().unwrap());
    command
        .args(["--exact", selector, "--nocapture"])
        .env(key, mode)
        .env(super::wire::CACHE_USAGE_VAR, cache_usage);
    let output = command.output().expect("child test run");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.status.success(), "{mode} child failed: {text}");
    assert!(text.contains(sentinel), "{mode}: sentinel missing: {text}");
    assert!(text.contains("1 passed"), "{mode} ran zero tests: {text}");
}

pub(super) fn sse_events(bytes: &Bytes) -> Vec<SseEvent> {
    let text = std::str::from_utf8(bytes).expect("utf-8 stream");
    text.split("\n\n")
        .filter(|block| !block.is_empty())
        .filter_map(|block| {
            let mut name = None;
            let mut data = Vec::new();
            for line in block.lines() {
                if let Some(value) = line.strip_prefix("event: ") {
                    name = Some(value.to_string());
                } else if let Some(value) = line.strip_prefix("data: ") {
                    data.push(value);
                } else if !line.starts_with(':') {
                    panic!("unexpected line in an event stream: {line:?}");
                }
            }
            (name.is_some() || !data.is_empty()).then(|| SseEvent {
                name,
                data: data.join("\n"),
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{
        EOS, SseEvent, bundle, gated_engine, generation, ids, probes, refusing_engine, script,
        sse_events, wait_for_state,
    };
    use crate::EngError;
    use crate::family::executor::GenerationState;
    use crate::family::sampler::{Sampler, SamplingParams};
    use crate::sched::job::{Job, JobEvent};
    use crate::sched::worker::EngineState;
    use crate::tokenizer::{DecodeFinish, IncrementalDecoder};
    use axum::body::Bytes;
    use std::time::Duration;

    #[test]
    fn the_fixture_tokenizer_encodes_ascii_as_byte_values() {
        let bundle = bundle();
        assert_eq!(
            bundle.tokenizer().encode("hi!").expect("encode"),
            ids("hi!")
        );
        assert_eq!(bundle.eos_token_id(), EOS);
        assert_eq!(bundle.vocab_size(), 258);
    }

    #[test]
    fn a_script_has_one_batch_per_byte_and_ends_with_eos() {
        assert_eq!(script("ab"), vec![vec![97], vec![98], vec![EOS]]);
    }

    #[test]
    fn a_gated_engine_loads_only_after_the_gate_drops() {
        let (engine, gate) = gated_engine(script("a"));
        let handle = engine.handle();
        std::thread::sleep(Duration::from_millis(30));
        assert_eq!(handle.state(), EngineState::Loading);
        drop(gate);
        wait_for_state(&handle, EngineState::Ready);
    }

    #[test]
    fn a_refusing_engine_fails_the_first_job_with_its_error() {
        let engine = refusing_engine(EngError::MemoryPlan("no room".to_string()));
        let (job, mut handle) = Job::new(vec![104], generation()).expect("job");
        engine.handle().try_submit(job).expect("submit");
        let started = handle.events.blocking_recv().expect("started");
        assert_eq!(started, JobEvent::Started);
        let event = handle.events.blocking_recv().expect("an event");
        assert_eq!(
            event,
            JobEvent::Error(EngError::MemoryPlan("no room".to_string()))
        );
    }

    #[test]
    fn the_probe_engine_records_the_prompt_and_samples_only_ids_zero_and_one() {
        let probes = probes(Duration::ZERO);
        let params = SamplingParams::new(1.0, 1.0, 0).expect("params");
        let decoder = IncrementalDecoder::new(bundle().tokenizer());
        let generation = GenerationState::with_sampler(decoder, Sampler::new(params, 7));
        let (job, mut handle) = Job::new(vec![104, 105], generation).expect("job");
        probes.engine.handle().try_submit(job).expect("submit");
        let mut ids = std::collections::BTreeSet::new();
        for _ in 0..40 {
            match handle.events.blocking_recv() {
                Some(JobEvent::Started) => continue,
                Some(JobEvent::Batch(batch)) => ids.extend(batch.tokens),
                _ => panic!("expected a batch"),
            }
        }
        handle.cancel();
        assert_eq!(ids, std::collections::BTreeSet::from([0, 1]));
        assert_eq!(*probes.prompt.lock().expect("prompt lock"), vec![104, 105]);
    }

    #[test]
    fn the_probe_engine_ends_a_job_at_its_output_limit() {
        let probes = probes(Duration::ZERO);
        let params = SamplingParams::new(1.0, 1.0, 0).expect("params");
        let decoder = IncrementalDecoder::new(bundle().tokenizer());
        let generation =
            GenerationState::with_sampler(decoder, Sampler::new(params, 7)).with_max_tokens(3);
        let (job, mut handle) = Job::new(vec![104], generation).expect("job");
        probes.engine.handle().try_submit(job).expect("submit");
        let mut finishes = Vec::new();
        while finishes.len() < 5 {
            match handle.events.blocking_recv() {
                Some(JobEvent::Started) => continue,
                Some(JobEvent::Batch(batch)) => {
                    finishes.push((batch.tokens.len(), batch.finish));
                    if batch.finish.is_some() {
                        break;
                    }
                }
                _ => break,
            }
        }
        assert_eq!(
            finishes,
            vec![(1, None), (1, None), (1, Some(DecodeFinish::Length))]
        );
    }

    #[test]
    fn sse_events_splits_named_and_plain_events_and_skips_comments() {
        let stream = Bytes::from_static(
            b": keep-alive\n\ndata: {\"a\":1}\n\nevent: error\ndata: {\"b\":2}\n\ndata: [DONE]\n\n",
        );
        assert_eq!(
            sse_events(&stream),
            vec![
                SseEvent {
                    name: None,
                    data: "{\"a\":1}".to_string()
                },
                SseEvent {
                    name: Some("error".to_string()),
                    data: "{\"b\":2}".to_string()
                },
                SseEvent {
                    name: None,
                    data: "[DONE]".to_string()
                },
            ]
        );
    }
}
