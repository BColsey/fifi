//! The Hugging Face hub bridge: search proxy, local-cache listing,
//! and download jobs that write the standard hub cache layout.
//!
//! The engine never grew an HTTP client because nothing it serves
//! needs one; the web GUI's model page is the first outbound
//! consumer, so the surface stays deliberately small: read-only
//! search and tree calls relayed behind the same bearer
//! authentication as every other `/v1` route, plus download jobs
//! that materialize `models--<org>--<repo>/{refs,blobs,snapshots}`
//! exactly the way `hf download` does — blob per content hash,
//! snapshot symlinks, `refs/<name>` pointing at the commit — so
//! every other hub-aware tool reads what the GUI downloaded.
//!
//! Downloads run on their own threads, not the tokio workers: a
//! multi-gigabyte stream must never stall request handling, and the
//! engine's own engine threads are plain OS threads for the same
//! reason.

use std::collections::HashMap;
use std::io::Read as _;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

use axum::Json;
use axum::extract::{Path as UrlPath, Query, State};
use axum::http::StatusCode;
use serde::Deserialize;
use sha2::Digest;

use super::error::ServerError;
use super::state::AppState;
use crate::EngError;

/// The hub this build talks to. `HF_ENDPOINT` is the standard mirror
/// override; the engine-specific name wins when both are set, so an
/// operator can point the engine at a mirror without dragging every
/// other hub tool on the host along.
fn endpoint_from_env() -> String {
    std::env::var("FIFI_HF_ENDPOINT")
        .or_else(|_| std::env::var("HF_ENDPOINT"))
        .unwrap_or_else(|_| "https://huggingface.co".to_string())
}

/// The token for gated repos, read the way every hub client does.
fn token_from_env() -> Option<String> {
    std::env::var("HF_TOKEN")
        .or_else(|_| std::env::var("HUGGING_FACE_HUB_TOKEN"))
        .ok()
        .filter(|value| !value.trim().is_empty())
}

/// The cache directory, resolved in the hub's own precedence order
/// (`HF_HUB_CACHE` over `HF_HOME/hub` over the default) with one
/// engine-local override (`FIFI_HF_HUB`) on top for deployments that
/// keep model storage off the home volume.
pub(crate) fn hub_dir_from_env() -> PathBuf {
    if let Ok(dir) = std::env::var("FIFI_HF_HUB")
        && !dir.trim().is_empty()
    {
        return PathBuf::from(dir);
    }
    if let Ok(dir) = std::env::var("HF_HUB_CACHE")
        && !dir.trim().is_empty()
    {
        return PathBuf::from(dir);
    }
    if let Ok(home) = std::env::var("HF_HOME")
        && !home.trim().is_empty()
    {
        return PathBuf::from(home).join("hub");
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    Path::new(&home)
        .join(".cache")
        .join("huggingface")
        .join("hub")
}

/// The cached snapshot of one repo's `main` ref, when the hub cache
/// holds one: the directory `serve --model org/repo` and
/// `POST /v1/engines` accept for a repo id. `None` when the repo is
/// not cached (or has no ref), so callers keep their own error text.
pub(crate) fn cached_main_snapshot(repo: &str) -> Option<PathBuf> {
    cached_main_snapshot_in(&hub_dir_from_env(), repo)
}

fn cached_main_snapshot_in(hub: &Path, repo: &str) -> Option<PathBuf> {
    if !valid_repo_id(repo) || !repo.contains('/') {
        return None;
    }
    let dir = repo_cache_dir(hub, repo);
    let commit = std::fs::read_to_string(dir.join("refs").join("main")).ok()?;
    let snapshot = dir.join("snapshots").join(commit.trim());
    snapshot.is_dir().then_some(snapshot)
}

/// A repo id is `org/name` (or a bare name for user-less repos) of
/// `[A-Za-z0-9._-]` segments. The rule exists because the id becomes
/// both a URL path and a cache directory name: everything the rule
/// rejects is a traversal or injection attempt, not a naming quirk.
pub(crate) fn valid_repo_id(repo: &str) -> bool {
    fn segment(text: &str) -> bool {
        !text.is_empty()
            && text.len() <= 64
            && text != "."
            && text != ".."
            && text
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'.' || b == b'_' || b == b'-')
    }
    let mut parts = repo.split('/');
    let Some(first) = parts.next() else {
        return false;
    };
    match parts.next() {
        None => segment(first),
        Some(second) => segment(first) && segment(second) && parts.next().is_none(),
    }
}

/// A revision we accept: one plain segment (`main`, a tag, a
/// branch, a 40-hex commit). It becomes both a URL segment and a
/// file name under `refs/`, so slashes and traversal are refused.
pub(crate) fn valid_revision(revision: &str) -> bool {
    !revision.is_empty()
        && revision.len() <= 64
        && revision != "."
        && revision != ".."
        && !revision.contains('/')
        && revision
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'.' || b == b'_' || b == b'-')
}

/// A hub-supplied file path or commit id that becomes a path under
/// the cache: one relative path of plain segments. This is the
/// filesystem half of `valid_repo_id` — the hub (or a configured
/// mirror) is external input, so absolute paths, backslashes, and
/// empty or dotted segments are injection attempts, not naming
/// quirks.
pub(crate) fn safe_rel_path(text: &str) -> bool {
    !text.is_empty()
        && text.len() <= 512
        && !text.starts_with('/')
        && !text.contains('\\')
        && text
            .split('/')
            .all(|part| !part.is_empty() && part != "." && part != "..")
}

/// The cache directory name of one repo, the hub's own encoding.
pub(crate) fn repo_cache_dir(hub_dir: &Path, repo: &str) -> PathBuf {
    hub_dir.join(format!("models--{}", repo.replace('/', "--")))
}

/// The symlink target for one snapshot file, relative to the
/// link's own directory: a file nested `n` directories deep needs
/// `n + 2` `..` levels to reach the repo root's `blobs/`.
fn blob_link_target(path: &str, blob: &str) -> PathBuf {
    let depth = path.matches('/').count();
    let mut target = PathBuf::from("../".repeat(depth + 2));
    target.push("blobs");
    target.push(blob);
    target
}

/// The state of one download job. `Downloading` carries the live
/// counters the GUI polls; the transitions are one-way.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum JobState {
    Resolving,
    Downloading {
        commit: String,
        files_total: u32,
        files_done: u32,
        bytes: u64,
        bytes_total: u64,
        current_file: String,
    },
    Done {
        commit: String,
        bytes: u64,
        path: PathBuf,
    },
    Cancelled,
    Failed(String),
}

impl JobState {
    fn is_active(&self) -> bool {
        matches!(self, JobState::Resolving | JobState::Downloading { .. })
    }
}

/// One download job: identity plus the shared cells its thread and
/// the status route read and write.
pub(crate) struct Job {
    pub(crate) id: u64,
    pub(crate) repo: String,
    pub(crate) revision: String,
    pub(crate) state: Mutex<JobState>,
    pub(crate) cancel: AtomicBool,
}

impl Job {
    fn status(&self) -> serde_json::Value {
        let state = self.state.lock().unwrap_or_else(|e| e.into_inner()).clone();
        let mut row = serde_json::json!({
            "id": self.id,
            "repo": self.repo,
            "revision": self.revision,
        });
        let cells = row.as_object_mut().expect("row is an object");
        match state {
            JobState::Resolving => {
                cells.insert("state".into(), "resolving".into());
            }
            JobState::Downloading {
                commit,
                files_total,
                files_done,
                bytes,
                bytes_total,
                current_file,
            } => {
                cells.insert("state".into(), "downloading".into());
                cells.insert("commit".into(), commit.into());
                cells.insert("files_total".into(), files_total.into());
                cells.insert("files_done".into(), files_done.into());
                cells.insert("bytes".into(), bytes.into());
                cells.insert("bytes_total".into(), bytes_total.into());
                cells.insert("current_file".into(), current_file.into());
                cells.insert("estimate".into(), resource_estimate(bytes_total));
            }
            JobState::Done {
                commit,
                bytes,
                path,
            } => {
                cells.insert("state".into(), "done".into());
                cells.insert("commit".into(), commit.into());
                cells.insert("bytes".into(), bytes.into());
                cells.insert("estimate".into(), resource_estimate(bytes));
                cells.insert("detail".into(), path.display().to_string().into());
            }
            JobState::Cancelled => {
                cells.insert("state".into(), "cancelled".into());
            }
            JobState::Failed(cause) => {
                cells.insert("state".into(), "failed".into());
                cells.insert("detail".into(), cause.into());
            }
        }
        row
    }
}

/// The resource ladder a pack of `weights_bytes` needs, stated the
/// way the engine chooses at serve time. The figures are the pack
/// itself: the GPU-resident tier must carry the weights on the card,
/// the host-streamed tier carries them in host memory (the card then
/// holds only a working set), and the disk-streamed tier leaves them
/// on disk. Fifi picks the smallest tier that fits, automatically,
/// at serve time — the runtime working set (KV cache, staging,
/// activations) is family- and context-dependent and is admitted
/// then, so a card sized exactly at `vram_bytes` still downshifts.
pub(crate) fn resource_estimate(weights_bytes: u64) -> serde_json::Value {
    serde_json::json!({
        "vram_bytes": weights_bytes,
        "ram_bytes": weights_bytes,
        "disk_bytes": weights_bytes,
        "mode": "auto: Fifi picks GPU-resident, host-streamed, or \
                 disk-streamed at serve time; the lower tiers hold only \
                 a working set on the GPU",
    })
}

/// The shared hub bridge. Held by `AppState` as an `Arc`; every
/// mutable piece is a lock or an atomic, so clones share one job
/// registry and one cache root.
pub(crate) struct HfHub {
    endpoint: String,
    token: Option<String>,
    hub_dir: PathBuf,
    api: ureq::Agent,
    downloader: ureq::Agent,
    next_id: AtomicU64,
    jobs: Mutex<Vec<Arc<Job>>>,
}

/// How many jobs may transfer at once. Two keeps a big pack and its
/// tokenizer moving in parallel without letting the page open a
/// dozen competing multi-gigabyte streams on one disk.
const MAX_ACTIVE_JOBS: usize = 2;

/// The hub API agents' request budget. Search and tree calls are
/// small JSON answers; a stuck connection must fail, not hang the
/// page's poll.
const API_TIMEOUT: Duration = Duration::from_secs(20);
/// Connecting (DNS + TCP + TLS) costs the same for a download as for
/// an API call; only the body may run for hours.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(20);

/// The directory the hub stores inside a repo that no client wants:
/// xet bookkeeping, not model data.
const SKIPPED_PREFIX: &str = ".cache/";

impl HfHub {
    pub(crate) fn from_env() -> Self {
        Self::new(endpoint_from_env(), hub_dir_from_env(), token_from_env())
    }

    pub(crate) fn new(endpoint: String, hub_dir: PathBuf, token: Option<String>) -> Self {
        let user_agent = concat!(
            "fifi-enginesrv/",
            env!("CARGO_PKG_VERSION"),
            " (hub bridge)"
        );
        let api: ureq::Agent = ureq::Agent::config_builder()
            .timeout_global(Some(API_TIMEOUT))
            .user_agent(user_agent)
            .build()
            .into();
        let downloader: ureq::Agent = ureq::Agent::config_builder()
            .timeout_connect(Some(CONNECT_TIMEOUT))
            .user_agent(user_agent)
            .build()
            .into();
        Self {
            endpoint,
            token,
            hub_dir,
            api,
            downloader,
            next_id: AtomicU64::new(1),
            jobs: Mutex::new(Vec::new()),
        }
    }

    /// One hub API call that returns JSON, with the token attached
    /// when one exists. The status code in the error keeps the hub's
    /// own refusal (rate limit, gated, unknown repo) visible in the
    /// GUI.
    fn api_get(&self, path: &str) -> Result<serde_json::Value, String> {
        let url = format!("{}{path}", self.endpoint);
        let request = self.authorized(self.api.get(&url));
        match request.call() {
            Ok(mut response) => {
                let mut text = String::new();
                let _ = response.body_mut().as_reader().read_to_string(&mut text);
                serde_json::from_str(&text)
                    .map_err(|error| format!("hub answer is not JSON: {error}"))
            }
            Err(ureq::Error::StatusCode(code)) => Err(format!("hub returned {code}")),
            Err(error) => Err(format!("hub unreachable: {error}")),
        }
    }

    fn authorized<B>(&self, request: ureq::RequestBuilder<B>) -> ureq::RequestBuilder<B> {
        match &self.token {
            Some(token) => request.header("authorization", &format!("Bearer {token}")),
            None => request,
        }
    }

    /// Search the hub's model list. The filters are the hub's own
    /// tags (`gguf`, `exl3`), passed through unchanged: the GUI
    /// decides what to offer, the engine only relays.
    pub(crate) fn search(
        &self,
        query: &str,
        gguf: bool,
        exl3: bool,
        limit: u32,
    ) -> Result<Vec<serde_json::Value>, String> {
        let mut path = format!("/api/models?search={}&limit={limit}", urlencode(query));
        if gguf {
            path.push_str("&filter=gguf");
        }
        if exl3 {
            path.push_str("&filter=exl3");
        }
        let found = self.api_get(&path)?;
        let rows = found
            .as_array()
            .ok_or_else(|| "hub answer is not a list".to_string())?;
        Ok(rows
            .iter()
            .map(|row| {
                serde_json::json!({
                    "id": row["id"],
                    "likes": row.get("likes").and_then(|v| v.as_u64()).unwrap_or(0),
                    "downloads": row.get("downloads").and_then(|v| v.as_u64()).unwrap_or(0),
                    "last_modified": row.get("lastModified").filter(|v| !v.is_null()),
                    "pipeline_tag": row.get("pipelineTag").filter(|v| !v.is_null()),
                    // The hub reports `false`, `"auto"`, or `"manual"`;
                    // only a non-false value blocks an anonymous pull.
                    "gated": matches!(
                        row.get("gated"),
                        Some(serde_json::Value::Bool(true)) | Some(serde_json::Value::String(_))
                    ),
                })
            })
            .collect())
    }

    /// The file list of one revision: the info call names the commit
    /// and the siblings, the recursive tree call adds sizes. Two
    /// calls because the hub offers no single endpoint with both.
    pub(crate) fn tree(
        &self,
        repo: &str,
        revision: &str,
    ) -> Result<(String, Vec<(String, u64)>), String> {
        let info = self.api_get(&format!(
            "/api/models/{}?revision={}",
            urlencode(repo),
            urlencode(revision)
        ))?;
        let commit = info
            .get("sha")
            .and_then(|v| v.as_str())
            .ok_or_else(|| "hub answer names no commit".to_string())?
            .to_string();
        let mut sizes: HashMap<String, u64> = HashMap::new();
        // Sizes are a display nicety, not a correctness input: a tree
        // call that fails just leaves the totals unknown (0).
        if let Ok(tree) = self.api_get(&format!(
            "/api/models/{}/tree/{}?recursive=true",
            urlencode(repo),
            urlencode(revision)
        )) && let Some(rows) = tree.as_array()
        {
            for row in rows {
                if row.get("type").and_then(|v| v.as_str()) != Some("file") {
                    continue;
                }
                let Some(path) = row.get("path").and_then(|v| v.as_str()) else {
                    continue;
                };
                let size = row
                    .get("lfs")
                    .and_then(|lfs| lfs.get("size"))
                    .and_then(|v| v.as_u64())
                    .or_else(|| row.get("size").and_then(|v| v.as_u64()))
                    .unwrap_or(0);
                sizes.insert(path.to_string(), size);
            }
        }
        let mut files: Vec<(String, u64)> = info
            .get("siblings")
            .and_then(|v| v.as_array())
            .map(|rows| {
                rows.iter()
                    .filter_map(|row| {
                        let name = row.get("rfilename").and_then(|v| v.as_str())?;
                        Some((name.to_string(), sizes.get(name).copied().unwrap_or(0)))
                    })
                    .collect()
            })
            .unwrap_or_default();
        files.sort_by(|a, b| a.0.cmp(&b.0));
        Ok((commit, files))
    }

    /// The local cache as the GUI shows it: every `models--*`
    /// directory with its refs, snapshot paths, blob bytes, and the
    /// families this build could actually serve from each snapshot.
    /// Reading `config.json`'s `model_type` against the detectable
    /// families is the same evidence `serve --model` uses, so the
    /// badge cannot disagree with the loader.
    pub(crate) fn cache(&self) -> serde_json::Value {
        let mut models: Vec<serde_json::Value> = Vec::new();
        let Ok(entries) = std::fs::read_dir(&self.hub_dir) else {
            return serde_json::json!({ "hub_dir": self.hub_dir.display().to_string(), "models": models });
        };
        let mut dirs: Vec<PathBuf> = entries
            .filter_map(|entry| entry.ok().map(|e| e.path()))
            .filter(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.starts_with("models--"))
            })
            .collect();
        dirs.sort();
        for dir in dirs {
            let name = dir
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or_default()
                .to_string();
            // The hub's own decoding: each `--` pair after the prefix
            // separates org from repo name.
            let repo = name["models--".len()..].replace("--", "/");
            let bytes = dir_bytes(&dir.join("blobs"));
            models.push(serde_json::json!({
                "repo": repo,
                "bytes": bytes,
                "estimate": resource_estimate(bytes),
                "refs": ref_list(&dir.join("refs")),
                "snapshots": snapshot_list(&dir),
            }));
        }
        serde_json::json!({
            "hub_dir": self.hub_dir.display().to_string(),
            "models": models,
        })
    }

    /// Starts one download job. The first network call happens on
    /// the job's own thread, so this returns as soon as the job is
    /// registered.
    pub(crate) fn start_download(
        self: &Arc<Self>,
        repo: &str,
        revision: &str,
    ) -> Result<Arc<Job>, String> {
        let mut jobs = self.jobs.lock().unwrap_or_else(|e| e.into_inner());
        let active = jobs
            .iter()
            .filter(|job| {
                job.state
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .is_active()
            })
            .count();
        if active >= MAX_ACTIVE_JOBS {
            return Err(format!(
                "{MAX_ACTIVE_JOBS} downloads are already running; wait or cancel one"
            ));
        }
        if jobs.iter().any(|job| {
            job.repo == repo
                && job.revision == revision
                && job
                    .state
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .is_active()
        }) {
            return Err(format!("{repo} is already downloading"));
        }
        // A finished job for the same repo+revision is replaced: a
        // retry after a failure must not show two rows for one pack.
        jobs.retain(|job| {
            let replaceable = job.repo == repo
                && job.revision == revision
                && !job
                    .state
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .is_active();
            !replaceable
        });
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let job = Arc::new(Job {
            id,
            repo: repo.to_string(),
            revision: revision.to_string(),
            state: Mutex::new(JobState::Resolving),
            cancel: AtomicBool::new(false),
        });
        jobs.push(Arc::clone(&job));
        drop(jobs);
        let hub = Arc::clone(self);
        let thread_job = Arc::clone(&job);
        let spawned = std::thread::Builder::new()
            .name(format!("hf-download-{id}"))
            .spawn(move || hub.run_job(&thread_job));
        if let Err(error) = spawned {
            // The job never ran: drop it from the registry so its
            // `Resolving` state cannot hold an active-job slot until
            // process restart.
            self.jobs
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .retain(|registered| registered.id != id);
            return Err(format!("cannot start the download thread: {error}"));
        }
        Ok(job)
    }

    /// The bytes the other running downloads still have to write:
    /// space this job must not count as free.
    fn committed_bytes(&self, except: u64) -> u64 {
        let jobs = self.jobs.lock().unwrap_or_else(|e| e.into_inner());
        jobs.iter()
            .filter(|job| job.id != except)
            .map(
                |job| match &*job.state.lock().unwrap_or_else(|e| e.into_inner()) {
                    JobState::Downloading {
                        bytes, bytes_total, ..
                    } => bytes_total.saturating_sub(*bytes),
                    _ => 0,
                },
            )
            .sum()
    }

    /// Marks a job cancelled. The thread notices at its next chunk
    /// boundary and cleans up its partial blob itself.
    pub(crate) fn cancel(&self, id: u64) -> bool {
        let jobs = self.jobs.lock().unwrap_or_else(|e| e.into_inner());
        match jobs.iter().find(|job| job.id == id) {
            Some(job) => {
                job.cancel.store(true, Ordering::SeqCst);
                true
            }
            None => false,
        }
    }

    /// Every job, newest last, including finished ones: the page
    /// keeps showing a failure until the user retries it.
    pub(crate) fn statuses(&self) -> Vec<serde_json::Value> {
        let jobs = self.jobs.lock().unwrap_or_else(|e| e.into_inner());
        jobs.iter().map(|job| job.status()).collect()
    }

    /// The job thread: resolve the revision, then move each file
    /// into the cache layout. Every failure lands in `Failed` with
    /// the hub's or the filesystem's own cause text.
    fn run_job(&self, job: &Job) {
        if let Err(cause) = self.run_job_inner(job) {
            let mut state = job.state.lock().unwrap_or_else(|e| e.into_inner());
            // A cancellation that loses the race to the error path
            // stays a cancellation, not a failure.
            if !matches!(*state, JobState::Cancelled) {
                *state = JobState::Failed(cause);
            }
        }
    }

    fn run_job_inner(&self, job: &Job) -> Result<(), String> {
        let (commit, files) = self.tree(&job.repo, &job.revision)?;
        if !safe_rel_path(&commit) {
            return Err(format!("hub names an unsafe commit: {commit}"));
        }
        if let Some(bad) = files
            .iter()
            .map(|(path, _)| path)
            .find(|path| !path.starts_with(SKIPPED_PREFIX) && !safe_rel_path(path))
        {
            return Err(format!("hub names an unsafe file path: {bad}"));
        }
        let wanted: Vec<&(String, u64)> = files
            .iter()
            .filter(|(path, _)| !path.starts_with(SKIPPED_PREFIX))
            .collect();
        let bytes_total: u64 = wanted.iter().map(|(_, size)| size).sum();
        let repo_dir = repo_cache_dir(&self.hub_dir, &job.repo);
        let snapshot_dir = repo_dir.join("snapshots").join(&commit);
        let blobs_dir = repo_dir.join("blobs");
        // Refuse before the first byte is written: the files this job
        // still has to fetch (their snapshot link is absent) against
        // the free space left after the other running downloads take
        // their remaining bytes.
        let needed: u64 = wanted
            .iter()
            .filter(|(path, _)| snapshot_dir.join(path).symlink_metadata().is_err())
            .map(|(_, size)| size)
            .sum();
        if let Some(free) = free_disk_bytes(&self.hub_dir) {
            disk_refusal(
                &job.repo,
                &self.hub_dir,
                needed,
                free,
                self.committed_bytes(job.id),
            )?;
        }
        std::fs::create_dir_all(&snapshot_dir)
            .and_then(|_| std::fs::create_dir_all(&blobs_dir))
            .map_err(|error| format!("cannot create the cache layout: {error}"))?;
        set_downloading(job, &commit, wanted.len() as u32, 0, 0, bytes_total, "");
        let mut bytes_done = 0_u64;
        for (index, (path, expected_size)) in wanted.iter().enumerate() {
            if cancelled(job) {
                return Ok(());
            }
            let link = snapshot_dir.join(path);
            if link.symlink_metadata().is_ok() {
                bytes_done += expected_size;
                set_downloading(
                    job,
                    &commit,
                    wanted.len() as u32,
                    index as u32 + 1,
                    bytes_done,
                    bytes_total,
                    path,
                );
                continue;
            }
            let blob = self.download_file(job, &blobs_dir, path, *expected_size)?;
            bytes_done = job_bytes(job).max(bytes_done);
            if let Some(parent) = link.parent() {
                std::fs::create_dir_all(parent)
                    .map_err(|error| format!("cannot create {}: {error}", parent.display()))?;
            }
            let _ = std::fs::remove_file(&link);
            let relative = blob_link_target(path, &blob);
            #[cfg(unix)]
            std::os::unix::fs::symlink(&relative, &link)
                .map_err(|error| format!("cannot link {}: {error}", link.display()))?;
            #[cfg(not(unix))]
            std::fs::copy(blobs_dir.join(&blob), &link)
                .map_err(|error| format!("cannot place {}: {error}", link.display()))?;
            set_downloading(
                job,
                &commit,
                wanted.len() as u32,
                index as u32 + 1,
                bytes_done,
                bytes_total,
                path,
            );
        }
        // A named ref (not a pinned commit) is recorded the way `hf`
        // records it, atomically, so a reader never sees a half line.
        if job.revision != commit {
            let refs = repo_dir.join("refs");
            std::fs::create_dir_all(&refs)
                .map_err(|error| format!("cannot create {}: {error}", refs.display()))?;
            let tmp = refs.join(format!(".job{}.tmp", job.id));
            std::fs::write(&tmp, &commit)
                .and_then(|_| std::fs::rename(&tmp, refs.join(&job.revision)))
                .map_err(|error| format!("cannot record the ref: {error}"))?;
        }
        let mut state = job.state.lock().unwrap_or_else(|e| e.into_inner());
        *state = JobState::Done {
            commit,
            bytes: job_bytes(job),
            path: snapshot_dir,
        };
        Ok(())
    }

    /// Streams one file into `blobs/` and returns the blob name. The
    /// etag of an LFS file is its sha256, which allows reusing an
    /// existing blob without a transfer; every other file is hashed
    /// while streaming, so the blob name is earned, not trusted.
    fn download_file(
        &self,
        job: &Job,
        blobs_dir: &Path,
        path: &str,
        expected_size: u64,
    ) -> Result<String, String> {
        let url = format!(
            "{}/{}/resolve/{}/{}",
            self.endpoint,
            job.repo,
            job.revision,
            urlencode_path(path)
        );
        let etag = self.head_etag(&url);
        if let Some(hash) = etag.as_deref().and_then(sha256_name)
            && blobs_dir.join(hash).is_file()
        {
            bump_bytes(job, expected_size, path);
            return Ok(hash.to_string());
        }
        let request = self.authorized(self.downloader.get(&url));
        let mut response = request.call().map_err(hub_error)?;
        let mut reader = response.body_mut().as_reader();
        let tmp = blobs_dir.join(format!(".{}.part", job.id));
        let mut file = std::fs::File::create(&tmp)
            .map_err(|error| format!("cannot create {}: {error}", tmp.display()))?;
        let mut hasher = sha2::Sha256::new();
        // One disk block per read: big enough that a 30 GB pack costs
        // tens of thousands of iterations, not millions.
        let mut buffer = [0_u8; 256 * 1024];
        let result = loop {
            if job.cancel.load(Ordering::SeqCst) {
                break Err("cancelled".to_string());
            }
            match reader.read(&mut buffer) {
                Ok(0) => break Ok(()),
                Ok(read) => {
                    hasher.update(&buffer[..read]);
                    if let Err(error) = file.write_all(&buffer[..read]) {
                        break Err(format!("cannot write {}: {error}", tmp.display()));
                    }
                    bump_bytes(job, read as u64, path);
                }
                Err(error) => break Err(format!("transfer of {path} broke: {error}")),
            }
        };
        match result {
            Ok(()) => {
                if let Err(error) = file.sync_all() {
                    let _ = std::fs::remove_file(&tmp);
                    return Err(format!("cannot flush {path}: {error}"));
                }
                drop(file);
                let name = format!("{:x}", hasher.finalize());
                let blob = blobs_dir.join(&name);
                std::fs::rename(&tmp, &blob)
                    .map_err(|error| format!("cannot place {}: {error}", blob.display()))?;
                Ok(name)
            }
            Err(cause) => {
                // The partial blob is not a checkpoint: a resumed run
                // restarts the file, the way `hf` does without
                // `hf_transfer` resume support.
                let _ = std::fs::remove_file(&tmp);
                if job.cancel.load(Ordering::SeqCst) {
                    let mut state = job.state.lock().unwrap_or_else(|e| e.into_inner());
                    *state = JobState::Cancelled;
                }
                Err(cause)
            }
        }
    }

    /// The etag of a resolve URL, following the hub's redirect to the
    /// CDN the way a browser does. `None` covers every case where
    /// the header is absent or not a content hash.
    fn head_etag(&self, url: &str) -> Option<String> {
        let request = self.authorized(self.api.head(url));
        let response = request.call().ok()?;
        response
            .headers()
            .get("etag")?
            .to_str()
            .ok()
            .map(|etag| etag.trim_matches('"').to_string())
    }
}

/// Space the downloads leave free on the cache volume, so the
/// machine keeps room for its own logs, caches and swap.
const DISK_SPARE_BYTES: u64 = 1 << 30;

/// Bytes an unprivileged writer can still allocate on the volume
/// that holds `path`, read from its nearest existing ancestor (the
/// cache directory is created by the first download). `None` when
/// the platform offers no answer; the writes then fail on their own.
fn free_disk_bytes(path: &Path) -> Option<u64> {
    #[cfg(unix)]
    for candidate in path.ancestors() {
        match rustix::fs::statvfs(candidate) {
            Ok(stat) => return Some(stat.f_bavail.saturating_mul(stat.f_frsize)),
            Err(rustix::io::Errno::NOENT | rustix::io::Errno::NOTDIR) => {}
            Err(_) => return None,
        }
    }
    None
}

/// The refusal for a download that cannot fit, naming the figures
/// the hub reported and the disk holds. `committed` is what other
/// running downloads still have to write. A hub that reported no
/// sizes (all zero) needs nothing here: the writes report their own
/// failure.
fn disk_refusal(
    repo: &str,
    hub_dir: &Path,
    needed: u64,
    free: u64,
    committed: u64,
) -> Result<(), String> {
    let available = free.saturating_sub(committed);
    if needed == 0 || needed.saturating_add(DISK_SPARE_BYTES) <= available {
        return Ok(());
    }
    let gib = |bytes: u64| bytes as f64 / (1u64 << 30) as f64;
    Err(format!(
        "{repo} needs {:.1} GiB but {} has {:.1} GiB free ({:.1} GiB of that is promised to \
         other running downloads, and {:.1} GiB is kept spare): nothing was written; free \
         space or point FIFI_HF_HUB at a larger volume",
        gib(needed),
        hub_dir.display(),
        gib(free),
        gib(committed),
        gib(DISK_SPARE_BYTES),
    ))
}

fn hub_error(error: ureq::Error) -> String {
    match error {
        ureq::Error::StatusCode(code) => format!("hub returned {code}"),
        other => format!("hub unreachable: {other}"),
    }
}

fn cancelled(job: &Job) -> bool {
    if !job.cancel.load(Ordering::SeqCst) {
        return false;
    }
    let mut state = job.state.lock().unwrap_or_else(|e| e.into_inner());
    *state = JobState::Cancelled;
    true
}

/// The live byte counter, read where the loop's own copy may have
/// missed the etag-reuse bump.
fn job_bytes(job: &Job) -> u64 {
    match &*job.state.lock().unwrap_or_else(|e| e.into_inner()) {
        JobState::Downloading { bytes, .. } | JobState::Done { bytes, .. } => *bytes,
        _ => 0,
    }
}

fn set_downloading(
    job: &Job,
    commit: &str,
    files_total: u32,
    files_done: u32,
    bytes: u64,
    bytes_total: u64,
    current_file: &str,
) {
    let mut state = job.state.lock().unwrap_or_else(|e| e.into_inner());
    if state.is_active() {
        *state = JobState::Downloading {
            commit: commit.to_string(),
            files_total,
            files_done,
            bytes,
            bytes_total,
            current_file: current_file.to_string(),
        };
    }
}

fn bump_bytes(job: &Job, add: u64, current_file: &str) {
    let mut state = job.state.lock().unwrap_or_else(|e| e.into_inner());
    if let JobState::Downloading {
        commit,
        files_total,
        files_done,
        bytes,
        bytes_total,
        ..
    } = &*state
    {
        *state = JobState::Downloading {
            commit: commit.clone(),
            files_total: *files_total,
            files_done: *files_done,
            bytes: *bytes + add,
            bytes_total: *bytes_total,
            current_file: current_file.to_string(),
        };
    }
}

/// A 64-hex etag is the hub's content hash; anything else (a weak
/// git hash, `W/"..."`) is not reusable as a blob name.
fn sha256_name(etag: &str) -> Option<&str> {
    let trimmed = etag.trim_matches('"');
    if trimmed.len() == 64 && trimmed.bytes().all(|b| b.is_ascii_hexdigit()) {
        Some(trimmed)
    } else {
        None
    }
}

/// Sum of a directory's regular file sizes; symlinks count their own
/// name, not their target, so a snapshot link never double-counts a
/// blob.
fn dir_bytes(dir: &Path) -> u64 {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return 0;
    };
    let mut total = 0_u64;
    for entry in entries.flatten() {
        let Ok(meta) = std::fs::symlink_metadata(entry.path()) else {
            continue;
        };
        if meta.is_dir() {
            total += dir_bytes(&entry.path());
        } else if meta.is_file() {
            total += meta.len();
        }
    }
    total
}

/// `refs/<name>` = commit, one row each.
fn ref_list(refs_dir: &Path) -> Vec<serde_json::Value> {
    let Ok(entries) = std::fs::read_dir(refs_dir) else {
        return Vec::new();
    };
    let mut rows: Vec<serde_json::Value> = entries
        .flatten()
        .filter_map(|entry| {
            let name = entry.file_name().into_string().ok()?;
            let commit = std::fs::read_to_string(entry.path()).ok()?;
            Some(serde_json::json!({
                "name": name,
                "commit": commit.trim(),
            }))
        })
        .collect();
    rows.sort_by(|a, b| a["name"].as_str().cmp(&b["name"].as_str()));
    rows
}

/// Every snapshot of one repo with the gguf files and the families
/// this build could serve from it. A snapshot that carries neither
/// config.json nor a gguf is not servable by this engine and is left
/// for the host's other tools.
fn snapshot_list(repo_dir: &Path) -> Vec<serde_json::Value> {
    let Ok(entries) = std::fs::read_dir(repo_dir.join("snapshots")) else {
        return Vec::new();
    };
    let mut rows: Vec<serde_json::Value> = entries
        .flatten()
        .filter(|entry| entry.path().is_dir())
        .filter_map(|entry| {
            let commit = entry.file_name().into_string().ok()?;
            let path = entry.path();
            let gguf = gguf_files(&path);
            let families = servable_families(&path);
            if families.is_empty() && gguf.is_empty() {
                return None;
            }
            Some(serde_json::json!({
                "commit": commit,
                "path": path.display().to_string(),
                "gguf": gguf,
                "families": families,
            }))
        })
        .collect();
    rows.sort_by(|a, b| a["commit"].as_str().cmp(&b["commit"].as_str()));
    rows
}

fn gguf_files(dir: &Path) -> Vec<String> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut names: Vec<String> = entries
        .flatten()
        .filter(|entry| {
            entry
                .path()
                .extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| e.eq_ignore_ascii_case("gguf"))
        })
        .filter_map(|entry| entry.file_name().into_string().ok())
        .collect();
    names.sort();
    names
}

/// The detectable families of this build, from `serve --model`'s own
/// table: family names paired with the `model_type` values that
/// identify them in a `config.json`.
fn servable_families(dir: &Path) -> Vec<String> {
    let Ok(text) = std::fs::read_to_string(dir.join("config.json")) else {
        return Vec::new();
    };
    let Ok(config) = serde_json::from_str::<serde_json::Value>(&text) else {
        return Vec::new();
    };
    let Some(model_type) = config.get("model_type").and_then(|v| v.as_str()) else {
        return Vec::new();
    };
    super::model_serve::DETECTABLE_FAMILIES
        .iter()
        .filter(|(_, types)| types.contains(&model_type))
        .map(|(family, _)| family.to_string())
        .collect()
}

/// Percent-encoding that keeps the query-side characters safe; the
/// search box is user text and must never reparse as a URL.
fn urlencode(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for byte in text.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(byte as char)
            }
            _ => out.push_str(&format!("%{byte:02X}")),
        }
    }
    out
}

/// Path segments of a repo file, each encoded, joined with `/`:
/// subdirectories stay readable to the hub while no segment can
/// escape the repo.
fn urlencode_path(path: &str) -> String {
    path.split('/').map(urlencode).collect::<Vec<_>>().join("/")
}

/// GET /v1/hf/search — the hub's model search, relayed.
pub(crate) async fn search(
    State(state): State<Arc<AppState>>,
    Query(params): Query<SearchParams>,
) -> Result<Json<serde_json::Value>, ServerError> {
    let query = params.q.trim().to_string();
    if query.is_empty() {
        return Err(EngError::BadRequest("q is required".into()).into());
    }
    let limit = params.limit.unwrap_or(20).min(50);
    let hub = Arc::clone(state.hf());
    let gguf = params.gguf;
    let exl3 = params.exl3;
    let rows = tokio::task::spawn_blocking(move || hub.search(&query, gguf, exl3, limit))
        .await
        .map_err(|error| EngError::BadRequest(format!("search task failed: {error}")))?
        .map_err(EngError::BadRequest)?;
    Ok(Json(serde_json::json!({ "results": rows })))
}

#[derive(Deserialize)]
pub(crate) struct SearchParams {
    q: String,
    #[serde(default)]
    gguf: bool,
    #[serde(default)]
    exl3: bool,
    limit: Option<u32>,
}

/// GET /v1/hf/cache — the local hub cache.
pub(crate) async fn cache(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let hub = Arc::clone(state.hf());
    let listing = tokio::task::spawn_blocking(move || hub.cache())
        .await
        .unwrap_or_else(
            |error| serde_json::json!({ "error": format!("cache scan failed: {error}") }),
        );
    Json(listing)
}

/// GET /v1/hf/tree — commit and files of one revision.
pub(crate) async fn tree(
    State(state): State<Arc<AppState>>,
    Query(params): Query<TreeParams>,
) -> Result<Json<serde_json::Value>, ServerError> {
    if !valid_repo_id(&params.repo) {
        return Err(EngError::BadRequest(format!(
            "{} is not a hub repo id (org/name)",
            params.repo
        ))
        .into());
    }
    let hub = Arc::clone(state.hf());
    let repo = params.repo;
    let revision = params.revision.unwrap_or_else(|| "main".to_string());
    let queried_repo = repo.clone();
    let queried_revision = revision.clone();
    let (commit, files) =
        tokio::task::spawn_blocking(move || hub.tree(&queried_repo, &queried_revision))
            .await
            .map_err(|error| EngError::BadRequest(format!("tree task failed: {error}")))?
            .map_err(EngError::BadRequest)?;
    let total: u64 = files.iter().map(|(_, size)| size).sum();
    Ok(Json(serde_json::json!({
        "repo": repo,
        "revision": revision,
        "commit": commit,
        "total_bytes": total,
        "files": files
            .into_iter()
            .map(|(path, size)| serde_json::json!({"path": path, "size": size}))
            .collect::<Vec<_>>(),
    })))
}

#[derive(Deserialize)]
pub(crate) struct TreeParams {
    repo: String,
    revision: Option<String>,
}

/// GET /v1/hf/downloads — every job's live status.
pub(crate) async fn downloads(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    Json(serde_json::json!({ "downloads": state.hf().statuses() }))
}

/// POST /v1/hf/downloads — start a job.
pub(crate) async fn start_download(
    State(state): State<Arc<AppState>>,
    Json(body): Json<DownloadRequest>,
) -> Result<(StatusCode, Json<serde_json::Value>), ServerError> {
    if !valid_repo_id(&body.repo) {
        return Err(
            EngError::BadRequest(format!("{} is not a hub repo id (org/name)", body.repo)).into(),
        );
    }
    let revision = body.revision.unwrap_or_else(|| "main".to_string());
    if !valid_revision(&revision) {
        return Err(EngError::BadRequest(format!(
            "{revision} is not a revision (one plain segment: main, a tag, a branch, a commit)"
        ))
        .into());
    }
    let job = state
        .hf()
        .start_download(&body.repo, &revision)
        .map_err(EngError::BadRequest)?;
    Ok((
        StatusCode::ACCEPTED,
        Json(serde_json::json!({
            "id": job.id,
            "repo": job.repo,
            "revision": job.revision,
        })),
    ))
}

#[derive(Deserialize)]
pub(crate) struct DownloadRequest {
    repo: String,
    revision: Option<String>,
}

/// DELETE /v1/hf/downloads/{id} — ask a job to stop.
pub(crate) async fn cancel_download(
    State(state): State<Arc<AppState>>,
    UrlPath(id): UrlPath<u64>,
) -> Result<Json<serde_json::Value>, ServerError> {
    match state.hf().cancel(id) {
        true => Ok(Json(serde_json::json!({ "id": id, "cancelling": true }))),
        false => Err(ServerError::NotFound(format!(
            "download {id} does not exist"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repo_ids_are_two_safe_segments() {
        for id in ["org/name", "openai-community/gpt2", "bare", "a.b_c-d/x"] {
            assert!(valid_repo_id(id), "{id}");
        }
        for id in [
            "",
            "/",
            "a/",
            "/a",
            "a/b/c",
            "a//b",
            "../..",
            "../etc",
            "a/../b",
            "..",
            ".",
            "a/..",
            "org/n\u{e9}me",
            "org/name/",
            &format!("org/{}", "x".repeat(65)),
        ] {
            assert!(!valid_repo_id(id), "{id} should be rejected");
        }
    }

    #[test]
    fn the_cache_dir_is_the_hub_layout() {
        let dir = repo_cache_dir(Path::new("/hub"), "org/name");
        assert_eq!(dir, Path::new("/hub/models--org--name"));
    }

    #[test]
    fn a_cached_main_ref_resolves_to_its_snapshot_dir() {
        let hub = tempfile::tempdir().expect("tempdir");
        let repo = repo_cache_dir(hub.path(), "org/name");
        let commit = "deadbeef";
        std::fs::create_dir_all(repo.join("snapshots").join(commit)).expect("snapshot dir");
        std::fs::create_dir_all(repo.join("refs")).expect("refs dir");
        std::fs::write(repo.join("refs").join("main"), format!("{commit}\n")).expect("ref");

        let resolved = cached_main_snapshot_in(hub.path(), "org/name");
        assert_eq!(resolved, Some(repo.join("snapshots").join(commit)));
    }

    #[test]
    fn snapshot_resolution_needs_a_ref_a_snapshot_and_a_safe_id() {
        let hub = tempfile::tempdir().expect("tempdir");
        assert_eq!(cached_main_snapshot_in(hub.path(), "org/absent"), None);
        assert_eq!(cached_main_snapshot_in(hub.path(), "../.."), None);
        assert_eq!(cached_main_snapshot_in(hub.path(), "bare"), None);

        // A ref that points at no snapshot directory resolves to nothing.
        let repo = repo_cache_dir(hub.path(), "org/dangling");
        std::fs::create_dir_all(repo.join("refs")).expect("refs dir");
        std::fs::write(repo.join("refs").join("main"), "missing\n").expect("ref");
        assert_eq!(cached_main_snapshot_in(hub.path(), "org/dangling"), None);
    }

    #[test]
    fn revisions_are_one_plain_segment() {
        for rev in ["main", "v1.0", "refs-pr-1", &"d".repeat(40)] {
            assert!(valid_revision(rev), "{rev}");
        }
        for rev in [
            "",
            ".",
            "..",
            "refs/pr/1",
            "../x",
            "a\nb",
            "a b",
            &"x".repeat(65),
        ] {
            assert!(!valid_revision(rev), "{rev:?} should be rejected");
        }
    }

    #[test]
    fn hub_paths_must_be_relative_and_plain() {
        for path in ["file.gguf", "gguf/file.gguf", "a/b/c.safetensors"] {
            assert!(safe_rel_path(path), "{path}");
        }
        for path in [
            "",
            "/etc/passwd",
            "../../home/x",
            "a/../b",
            "./x",
            "x/.",
            "..",
            "a//b",
            "a\\b",
            &"x/".repeat(300),
        ] {
            assert!(!safe_rel_path(path), "{path:?} should be rejected");
        }
    }

    #[test]
    fn blob_targets_reach_blobs_from_any_depth() {
        assert_eq!(
            blob_link_target("m.gguf", "cafe"),
            Path::new("../../blobs/cafe")
        );
        assert_eq!(
            blob_link_target("gguf/m.gguf", "cafe"),
            Path::new("../../../blobs/cafe")
        );
        assert_eq!(
            blob_link_target("a/b/c.safetensors", "cafe"),
            Path::new("../../../../blobs/cafe")
        );
    }

    #[test]
    fn resource_estimates_state_the_tier_ladder() {
        let est = resource_estimate(5_000_000_000);
        assert_eq!(est["vram_bytes"], 5_000_000_000_u64);
        assert_eq!(est["ram_bytes"], 5_000_000_000_u64);
        assert_eq!(est["disk_bytes"], 5_000_000_000_u64);
        assert!(est["mode"].as_str().expect("mode").contains("auto"));
    }

    #[test]
    fn nested_snapshot_links_resolve_to_the_blob() {
        let hub = tempfile::tempdir().expect("tempdir");
        let repo = repo_cache_dir(hub.path(), "org/name");
        let blob = "cafe1234";
        std::fs::create_dir_all(repo.join("blobs")).expect("blobs dir");
        std::fs::write(repo.join("blobs").join(blob), "payload").expect("blob");
        for path in ["model.gguf", "gguf/model.gguf", "a/b/c.safetensors"] {
            let link = repo.join("snapshots").join("c1").join(path);
            std::fs::create_dir_all(link.parent().expect("link parent")).expect("dirs");
            std::os::unix::fs::symlink(blob_link_target(path, blob), &link).expect("link");
            let hit = std::fs::read(&link).expect("link must resolve to the blob");
            assert_eq!(hit, b"payload", "{path} must reach the blob");
        }
    }

    #[test]
    fn a_download_that_exceeds_free_disk_is_refused_with_the_figures() {
        let hub = Path::new("/hub");
        let gib = 1u64 << 30;
        assert!(disk_refusal("o/r", hub, 10 * gib, 20 * gib, 0).is_ok());
        // Fits only without the spare: refused.
        let refusal = disk_refusal("o/r", hub, 10 * gib, 10 * gib, 0).expect_err("no spare left");
        assert!(refusal.contains("o/r needs 10.0 GiB"), "{refusal}");
        assert!(refusal.contains("/hub has 10.0 GiB free"), "{refusal}");
        assert!(refusal.contains("nothing was written"), "{refusal}");
        // Space promised to another running download is not free.
        assert!(disk_refusal("o/r", hub, 10 * gib, 20 * gib, 15 * gib).is_err());
        // A hub that reported no sizes cannot be judged here.
        assert!(disk_refusal("o/r", hub, 0, 0, 0).is_ok());
    }

    #[test]
    fn free_disk_is_read_from_the_nearest_existing_ancestor() {
        let dir = tempfile::tempdir().expect("tempdir");
        let missing = dir.path().join("not").join("created").join("yet");
        let free = free_disk_bytes(&missing).expect("a statvfs answer");
        assert!(free > 0);
    }

    /// A one-thread hub that answers every GET with the canned JSON
    /// of a repo whose single file reports `size` bytes.
    fn serve_repo_of_size(size: u64) -> String {
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
        let address = listener.local_addr().expect("address");
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { break };
                let mut buffer = [0_u8; 4096];
                let read = stream.read(&mut buffer).unwrap_or(0);
                let request = String::from_utf8_lossy(&buffer[..read]).to_string();
                let body = if request.contains("/tree/") {
                    format!(r#"[{{"type":"file","path":"m.gguf","size":{size}}}]"#)
                } else {
                    r#"{"sha":"c0ffee","siblings":[{"rfilename":"m.gguf"}]}"#.to_string()
                };
                let _ = write!(
                    stream,
                    "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\n\
                     content-length: {}\r\nconnection: close\r\n\r\n{body}",
                    body.len()
                );
            }
        });
        format!("http://{address}")
    }

    #[test]
    fn an_oversized_download_fails_before_anything_is_written() {
        let hub_dir = tempfile::tempdir().expect("tempdir");
        let endpoint = serve_repo_of_size(1 << 60);
        let hub = Arc::new(HfHub::new(endpoint, hub_dir.path().to_path_buf(), None));
        let job = hub.start_download("org/huge", "main").expect("job starts");
        let deadline = std::time::Instant::now() + Duration::from_secs(20);
        let state = loop {
            let state = job.state.lock().expect("state").clone();
            if !state.is_active() || std::time::Instant::now() > deadline {
                break state;
            }
            std::thread::sleep(Duration::from_millis(20));
        };
        match state {
            JobState::Failed(cause) => {
                assert!(cause.contains("org/huge needs"), "{cause}");
                assert!(cause.contains("nothing was written"), "{cause}");
            }
            other => panic!("expected a refusal, got {other:?}"),
        }
        assert_eq!(
            std::fs::read_dir(hub_dir.path()).expect("hub dir").count(),
            0,
            "a refused download must leave the cache untouched"
        );
    }
}
