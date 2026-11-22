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
    if !valid_commit(commit.trim()) {
        return None;
    }
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

/// A commit id as the hub states it: 40 hex digits. It names a
/// directory under `snapshots/` and a line in `refs/`.
pub(crate) fn valid_commit(commit: &str) -> bool {
    commit.len() == 40 && commit.bytes().all(|b| b.is_ascii_hexdigit())
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

/// One file of a revision as the hub lists it: the path, the byte
/// size, and the content hash when the hub states one (LFS files carry
/// their sha256 as the object id).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct HubFile {
    pub(crate) path: String,
    pub(crate) size: u64,
    pub(crate) sha256: Option<String>,
}

/// A revision resolved to its commit and complete file list.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Tree {
    pub(crate) commit: String,
    pub(crate) files: Vec<HubFile>,
}

/// The state of one download job. `Downloading` carries the live
/// counters the GUI polls; the transitions are one-way.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum JobState {
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
        matches!(self, JobState::Downloading { .. })
    }

    fn name(&self) -> &'static str {
        match self {
            JobState::Downloading { .. } => "downloading",
            JobState::Done { .. } => "done",
            JobState::Cancelled => "cancelled",
            JobState::Failed(_) => "failed",
        }
    }
}

/// One download job: identity plus the shared cells its thread and
/// the status route read and write.
pub(crate) struct Job {
    pub(crate) id: u64,
    pub(crate) repo: String,
    pub(crate) revision: String,
    pub(crate) commit: String,
    pub(crate) files: Vec<HubFile>,
    pub(crate) state: Mutex<JobState>,
    pub(crate) cancel: AtomicBool,
    /// The bytes this job still had to fetch when it was admitted: its
    /// claim on the disk, held until the job ends.
    reserved: u64,
    /// The bytes this job has written since admission.
    written: AtomicU64,
}

impl Job {
    fn status(&self) -> serde_json::Value {
        let state = self.state.lock().unwrap_or_else(|e| e.into_inner()).clone();
        let mut row = serde_json::json!({
            "id": self.id,
            "repo": self.repo,
            "revision": self.revision,
            "state": state.name(),
        });
        let cells = row.as_object_mut().expect("row is an object");
        match state {
            JobState::Downloading {
                commit,
                files_total,
                files_done,
                bytes,
                bytes_total,
                current_file,
            } => {
                cells.insert("commit".into(), commit.into());
                cells.insert("files_total".into(), files_total.into());
                cells.insert("files_done".into(), files_done.into());
                cells.insert("bytes".into(), bytes.into());
                cells.insert("bytes_total".into(), bytes_total.into());
                cells.insert("current_file".into(), current_file.into());
                cells.insert("estimate".into(), pack_estimate(&self.files));
            }
            JobState::Done {
                commit,
                bytes,
                path,
            } => {
                cells.insert("commit".into(), commit.into());
                cells.insert("bytes".into(), bytes.into());
                cells.insert("estimate".into(), pack_estimate(&self.files));
                cells.insert("detail".into(), path.display().to_string().into());
            }
            JobState::Cancelled => {}
            JobState::Failed(cause) => {
                cells.insert("detail".into(), cause.into());
            }
        }
        row
    }
}

/// The part of a split GGUF name that numbers the shard
/// (`-00001-of-00004`), so the shards of one quantization group.
fn gguf_group(path: &str) -> String {
    let stem = path.strip_suffix(".gguf").unwrap_or(path);
    match stem.rfind("-of-") {
        Some(at) if stem[at + 4..].bytes().all(|b| b.is_ascii_digit()) => {
            let head = &stem[..at];
            match head.rfind('-') {
                Some(dash) if head[dash + 1..].bytes().all(|b| b.is_ascii_digit()) => {
                    head[..dash].to_string()
                }
                _ => stem.to_string(),
            }
        }
        _ => stem.to_string(),
    }
}

/// The weight bytes one serve would load from a file listing: the
/// largest single quantization (a GGUF split set counts as one), the
/// order `gguf_select` serves them in, or the safetensors shards when
/// the listing carries no GGUF. The other quantizations of a repo are
/// not part of one engine's footprint.
fn pack_weight_bytes(files: &[HubFile]) -> (u64, &'static str) {
    let mut groups: HashMap<String, u64> = HashMap::new();
    for file in files.iter().filter(|f| f.path.ends_with(".gguf")) {
        *groups.entry(gguf_group(&file.path)).or_default() += file.size;
    }
    if let Some(largest) = groups.values().max() {
        return (*largest, "the largest GGUF quantization in the listing");
    }
    let safetensors: u64 = files
        .iter()
        .filter(|f| f.path.ends_with(".safetensors"))
        .map(|f| f.size)
        .sum();
    (safetensors, "the safetensors shards in the listing")
}

/// The ladder a listing's weights would need, before anything is
/// downloaded: only the weights one engine loads, one revision. The
/// KV cache and working set come from the planner once a snapshot is
/// on disk (`planned: true` rows of the cache listing).
pub(crate) fn pack_estimate(files: &[HubFile]) -> serde_json::Value {
    let (bytes, basis) = pack_weight_bytes(files);
    serde_json::json!({
        "vram_bytes": bytes,
        "ram_bytes": bytes,
        "disk_bytes": bytes,
        "planned": false,
        "basis": format!("weight bytes of {basis}; the cache and working set are planned once the snapshot is on disk"),
        "mode": ESTIMATE_MODE,
    })
}

const ESTIMATE_MODE: &str = "auto: Fifi picks GPU-resident, host-streamed, or disk-streamed at serve time; the lower tiers hold only a working set on the GPU";

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
    /// Serializes admission: the free-space check and the registration
    /// of the job that claims the space happen under this one lock, so
    /// two downloads cannot both pass a check one of them must fail.
    admission: Mutex<()>,
    chunk_bytes: u64,
    chunk_timeout: Duration,
    free_space: FreeSpaceFn,
    /// The server's event-ring tap, wired by `AppState::wire_hub_events`:
    /// download lifecycle lines for `GET /v1/logs`. `None` in tests that
    /// never wire it.
    event_logger: Mutex<Option<EventTap>>,
}

/// The signature of the server's event-ring tap.
type EventTap = Box<dyn Fn(&str) + Send + Sync>;

type FreeSpaceFn = Box<dyn Fn(&Path) -> Option<u64> + Send + Sync>;

/// How many jobs may transfer at once. Two keeps a big pack and its
/// tokenizer moving in parallel without letting the page open a
/// dozen competing multi-gigabyte streams on one disk.
const MAX_ACTIVE_JOBS: usize = 2;

/// Finished jobs the registry keeps for the page to show; older
/// ones are dropped when a new job registers.
const MAX_FINISHED_JOBS: usize = 20;

/// The hub API agents' request budget. Search and tree calls are
/// small JSON answers; a stuck connection must fail, not hang the
/// page's poll.
const API_TIMEOUT: Duration = Duration::from_secs(20);
/// Connecting (DNS + TCP + TLS) costs the same for a download as for
/// an API call.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(20);

/// One ranged request moves this many bytes. A stalled connection can
/// block a read for no longer than `CHUNK_TIMEOUT`, because the HTTP
/// client bounds a whole call, not each read; a file is a sequence of
/// bounded calls, so a stall, a cancel and a resume all act at a
/// chunk boundary.
const CHUNK_BYTES: u64 = 32 << 20;
/// The budget of one ranged request: a 32 MiB chunk at about 0.4 MB/s.
const CHUNK_TIMEOUT: Duration = Duration::from_secs(90);
/// Retries of one chunk after a transfer failure, from the same
/// offset, before the file attempt fails.
const CHUNK_RETRIES: usize = 2;
/// Whole-file attempts: a failed integrity check deletes the partial
/// file and fetches it again.
const FILE_ATTEMPTS: usize = 3;

/// The first read of a GGUF header: most headers (metadata and tensor table) are inside it; a
/// longer one is read again with four times the length, up to the cap.
const HEADER_PROBE_BYTES: u64 = 8 << 20;
const HEADER_CAP_BYTES: u64 = 256 << 20;

/// The directory the hub stores inside a repo that no client wants:
/// xet bookkeeping, not model data.
const SKIPPED_PREFIX: &str = ".cache/";

/// A marker file in the repo directory while a snapshot is being
/// filled: the snapshot is not servable until the job removes it.
fn incomplete_marker(repo_dir: &Path, commit: &str) -> PathBuf {
    repo_dir.join(format!(".fifi-incomplete-{commit}"))
}

fn unix_millis() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(1, |elapsed| u64::try_from(elapsed.as_millis()).unwrap_or(1))
}

/// Removes the temporary files an interrupted process left behind:
/// partial blobs, ref temporaries and placement links. None of them
/// is a checkpoint a later run can trust.
pub(crate) fn sweep_partials(hub_dir: &Path) {
    let Ok(repos) = std::fs::read_dir(hub_dir) else {
        return;
    };
    for repo in repos.flatten() {
        let dir = repo.path();
        if !repo.file_name().to_string_lossy().starts_with("models--") {
            continue;
        }
        for (sub, suffix) in [("blobs", ".part"), ("refs", ".tmp")] {
            if let Ok(entries) = std::fs::read_dir(dir.join(sub)) {
                for entry in entries.flatten() {
                    let name = entry.file_name().to_string_lossy().into_owned();
                    if name.starts_with('.') && name.ends_with(suffix) {
                        let _ = std::fs::remove_file(entry.path());
                    }
                }
            }
        }
        sweep_links(&dir.join("snapshots"));
    }
}

fn sweep_links(dir: &Path) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let Ok(meta) = std::fs::symlink_metadata(&path) else {
            continue;
        };
        if meta.is_dir() {
            sweep_links(&path);
        } else if entry.file_name().to_string_lossy().starts_with('.')
            && entry.file_name().to_string_lossy().ends_with(".lnk")
        {
            let _ = std::fs::remove_file(&path);
        }
    }
}

impl HfHub {
    pub(crate) fn from_env() -> Self {
        Self::new(endpoint_from_env(), hub_dir_from_env(), token_from_env())
    }

    /// Clears what an earlier process left half-written in this cache.
    /// The server calls it once at start, before any job exists.
    pub(crate) fn sweep_partials(&self) {
        sweep_partials(&self.hub_dir);
    }

    pub(crate) fn new(endpoint: String, hub_dir: PathBuf, token: Option<String>) -> Self {
        Self::with_limits(
            endpoint,
            hub_dir,
            token,
            CHUNK_BYTES,
            CHUNK_TIMEOUT,
            Box::new(free_disk_bytes),
        )
    }

    pub(crate) fn with_limits(
        endpoint: String,
        hub_dir: PathBuf,
        token: Option<String>,
        chunk_bytes: u64,
        chunk_timeout: Duration,
        free_space: FreeSpaceFn,
    ) -> Self {
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
            .timeout_global(Some(chunk_timeout))
            .user_agent(user_agent)
            .build()
            .into();
        Self {
            endpoint,
            token,
            hub_dir,
            api,
            downloader,
            // Job ids start at the process's start time in milliseconds:
            // a page that still shows an id from before a restart cannot
            // cancel a different job that took the same small number.
            next_id: AtomicU64::new(unix_millis()),
            jobs: Mutex::new(Vec::new()),
            admission: Mutex::new(()),
            chunk_bytes: chunk_bytes.max(1),
            chunk_timeout,
            free_space,
            event_logger: Mutex::new(None),
        }
    }

    /// Install the tap the server's event ring reads. The hub never
    /// holds the ring itself, only this closure.
    pub(crate) fn set_event_logger(&self, tap: impl Fn(&str) + Send + Sync + 'static) {
        *self.event_logger.lock().unwrap_or_else(|e| e.into_inner()) = Some(Box::new(tap));
    }
    pub(super) fn emit(&self, line: &str) {
        if let Some(tap) = self
            .event_logger
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
        {
            tap(line);
        }
    }

    /// The free bytes under the cache directory, as the downloads see them; `None` when unknown.
    pub(crate) fn free_bytes(&self) -> Option<u64> {
        (self.free_space)(&self.hub_dir)
    }

    /// The cache directory this bridge writes into.
    pub(crate) fn cache_dir(&self) -> &Path {
        &self.hub_dir
    }

    /// The cached snapshot of one repo's `main` ref in this bridge's cache.
    pub(crate) fn main_snapshot(&self, repo: &str) -> Option<PathBuf> {
        cached_main_snapshot_in(&self.hub_dir, repo)
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

    /// The commit and file list of one revision. The info call names
    /// the commit and the files; the recursive tree call adds sizes and
    /// content hashes. A tree answer that fails or does not list every
    /// file is an error: a size the hub did not state is never zero.
    pub(crate) fn tree(&self, repo: &str, revision: &str) -> Result<Tree, String> {
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
        if !valid_commit(&commit) {
            return Err("hub names a commit that is not 40 hex digits".to_string());
        }
        let names: Vec<String> = info
            .get("siblings")
            .and_then(|v| v.as_array())
            .map(|rows| {
                rows.iter()
                    .filter_map(|row| row.get("rfilename").and_then(|v| v.as_str()))
                    .filter(|name| !name.starts_with(SKIPPED_PREFIX))
                    .map(str::to_string)
                    .collect()
            })
            .unwrap_or_default();
        if names.is_empty() {
            return Err("the revision lists no files".to_string());
        }
        if let Some(bad) = names.iter().find(|name| !safe_rel_path(name)) {
            return Err(format!("hub names an unsafe file path: {bad}"));
        }
        let listing = self
            .api_get(&format!(
                "/api/models/{}/tree/{}?recursive=true",
                urlencode(repo),
                urlencode(revision)
            ))
            .map_err(|error| format!("the file sizes could not be read: {error}"))?;
        let rows = listing.as_array().ok_or_else(|| {
            "the file sizes could not be read: hub answer is not a list".to_string()
        })?;
        let mut stated: HashMap<&str, (u64, Option<String>)> = HashMap::new();
        for row in rows {
            if row.get("type").and_then(|v| v.as_str()) != Some("file") {
                continue;
            }
            let Some(path) = row.get("path").and_then(|v| v.as_str()) else {
                continue;
            };
            let lfs = row.get("lfs");
            let size = lfs
                .and_then(|lfs| lfs.get("size"))
                .and_then(|v| v.as_u64())
                .or_else(|| row.get("size").and_then(|v| v.as_u64()));
            let Some(size) = size else {
                continue;
            };
            let sha256 = lfs
                .and_then(|lfs| lfs.get("oid"))
                .and_then(|v| v.as_str())
                .and_then(sha256_name)
                .map(str::to_string);
            stated.insert(path, (size, sha256));
        }
        let mut files = Vec::with_capacity(names.len());
        for name in names {
            let Some((size, sha256)) = stated.get(name.as_str()).cloned() else {
                return Err(format!(
                    "the file sizes could not be read: the hub's listing states no size for {name}"
                ));
            };
            files.push(HubFile {
                path: name,
                size,
                sha256,
            });
        }
        files.sort_by(|a, b| a.path.cmp(&b.path));
        Ok(Tree { commit, files })
    }

    /// The local cache as the GUI shows it: every `models--*`
    /// directory with its refs, snapshot paths, blob bytes, and the
    /// family each complete snapshot would start as.
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
            models.push(serde_json::json!({
                "repo": repo,
                "bytes": dir_bytes(&dir.join("blobs")),
                "refs": ref_list(&dir.join("refs")),
                "snapshots": snapshot_list(&dir),
            }));
        }
        serde_json::json!({
            "hub_dir": self.hub_dir.display().to_string(),
            "models": models,
        })
    }

    /// Refuses a start that would run beside what already runs: too
    /// many active jobs, or the same repo and revision twice.
    fn check_slots(&self, repo: &str, revision: &str) -> Result<(), String> {
        let jobs = self.jobs.lock().unwrap_or_else(|e| e.into_inner());
        let active: Vec<&Arc<Job>> = jobs
            .iter()
            .filter(|job| {
                job.state
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .is_active()
            })
            .collect();
        if active.len() >= MAX_ACTIVE_JOBS {
            return Err(format!(
                "{MAX_ACTIVE_JOBS} downloads are already running; wait or cancel one"
            ));
        }
        if active
            .iter()
            .any(|job| job.repo == repo && job.revision == revision)
        {
            return Err(format!("{repo} is already downloading"));
        }
        Ok(())
    }

    /// Admits one download or refuses it before anything is written.
    /// The revision is resolved first; a failed listing refuses with its
    /// own error. The files still to fetch are checked against the free
    /// space minus what the running jobs have yet to write, and the job
    /// that passes registers its claim under the same lock.
    pub(crate) fn start_download(
        self: &Arc<Self>,
        repo: &str,
        revision: &str,
    ) -> Result<Arc<Job>, String> {
        self.start_download_selected(repo, revision, &|_| true)
    }

    /// Like [`Self::start_download`], for the files of the revision that `select` keeps: the files
    /// one serve loads, not the whole repo.
    pub(crate) fn start_download_selected(
        self: &Arc<Self>,
        repo: &str,
        revision: &str,
        select: &dyn Fn(&HubFile) -> bool,
    ) -> Result<Arc<Job>, String> {
        self.check_slots(repo, revision)?;
        let mut tree = self.tree(repo, revision)?;
        tree.files.retain(|file| select(file));
        if tree.files.is_empty() {
            return Err(format!("{repo} has no file that matches"));
        }
        let repo_dir = repo_cache_dir(&self.hub_dir, repo);
        let snapshot_dir = repo_dir.join("snapshots").join(&tree.commit);
        let blobs_dir = repo_dir.join("blobs");
        let needed: u64 = tree
            .files
            .iter()
            .filter(|file| !file_present(&snapshot_dir, &blobs_dir, file))
            .map(|file| file.size)
            .sum();
        let admission = self.admission.lock().unwrap_or_else(|e| e.into_inner());
        self.check_slots(repo, revision)?;
        if let Some(free) = (self.free_space)(&self.hub_dir) {
            disk_refusal(repo, &self.hub_dir, needed, free, self.committed_bytes())?;
        }
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let bytes_total: u64 = tree.files.iter().map(|file| file.size).sum();
        let job = Arc::new(Job {
            id,
            repo: repo.to_string(),
            revision: revision.to_string(),
            commit: tree.commit.clone(),
            state: Mutex::new(JobState::Downloading {
                commit: tree.commit.clone(),
                files_total: tree.files.len() as u32,
                files_done: 0,
                bytes: 0,
                bytes_total,
                current_file: String::new(),
            }),
            files: tree.files,
            cancel: AtomicBool::new(false),
            reserved: needed,
            written: AtomicU64::new(0),
        });
        {
            let mut jobs = self.jobs.lock().unwrap_or_else(|e| e.into_inner());
            // A finished job for the same repo+revision is replaced: a
            // retry after a failure must not show two rows for one pack.
            jobs.retain(|old| {
                let finished = !old
                    .state
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .is_active();
                !(finished && old.repo == repo && old.revision == revision)
            });
            jobs.push(Arc::clone(&job));
            prune_finished(&mut jobs);
        }
        drop(admission);
        let hub = Arc::clone(self);
        let thread_job = Arc::clone(&job);
        let spawned = std::thread::Builder::new()
            .name(format!("hf-download-{id}"))
            .spawn(move || hub.run_job(&thread_job));
        if let Err(error) = spawned {
            // The job never ran: drop it from the registry so its
            // claim cannot hold an active-job slot until restart.
            self.jobs
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .retain(|registered| registered.id != id);
            return Err(format!("cannot start the download thread: {error}"));
        }
        Ok(job)
    }

    /// The bytes the running jobs have yet to write: space a new job
    /// must not count as free.
    fn committed_bytes(&self) -> u64 {
        let jobs = self.jobs.lock().unwrap_or_else(|e| e.into_inner());
        jobs.iter()
            .filter(|job| {
                job.state
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .is_active()
            })
            .map(|job| {
                job.reserved
                    .saturating_sub(job.written.load(Ordering::Relaxed))
            })
            .sum()
    }

    /// The first bytes of one file of a commit that hold its complete GGUF header: a ranged read
    /// of growing length until the header parses. The caller plans against the real header
    /// without downloading the weights.
    pub(crate) fn gguf_header_bytes(
        &self,
        repo: &str,
        commit: &str,
        file: &HubFile,
    ) -> Result<Vec<u8>, String> {
        let url = format!(
            "{}/{}/resolve/{}/{}",
            self.endpoint,
            repo,
            commit,
            urlencode_path(&file.path)
        );
        let mut length = HEADER_PROBE_BYTES.min(file.size);
        loop {
            let bytes = chunk_request(
                &self.downloader,
                self.token.as_deref(),
                &url,
                0,
                length,
                length == file.size,
            )?;
            if crate::weights::GgufHeader::from_bytes(&bytes, file.size, HEADER_CAP_BYTES).is_ok() {
                return Ok(bytes);
            }
            if length >= file.size || length >= HEADER_CAP_BYTES {
                return Err(format!("{} has no readable GGUF header", file.path));
            }
            length = (length * 4).min(file.size).min(HEADER_CAP_BYTES);
        }
    }

    /// Runs one selected download to its end on the calling thread, handing the live state to
    /// `progress` about once a second. Returns the snapshot directory when every selected file
    /// is whole and linked; the cause when the job failed or was cancelled. An interrupted run
    /// leaves resumable partials that the next call continues.
    pub(crate) fn download_and_wait(
        self: &Arc<Self>,
        repo: &str,
        revision: &str,
        select: &dyn Fn(&HubFile) -> bool,
        progress: &mut dyn FnMut(&JobState),
    ) -> Result<PathBuf, String> {
        let job = self.start_download_selected(repo, revision, select)?;
        let mut last = std::time::Instant::now() - Duration::from_secs(2);
        loop {
            let state = job.state.lock().unwrap_or_else(|e| e.into_inner()).clone();
            match state {
                JobState::Downloading { .. } => {
                    if last.elapsed() >= Duration::from_secs(1) {
                        progress(&state);
                        last = std::time::Instant::now();
                    }
                    std::thread::sleep(Duration::from_millis(50));
                }
                JobState::Done { path, .. } => return Ok(path),
                JobState::Cancelled => return Err("the download was cancelled".to_string()),
                JobState::Failed(cause) => return Err(cause),
            }
        }
    }

    /// Asks a running job to stop; a finished job answers what it
    /// finished as. The thread notices at its next chunk wait.
    pub(crate) fn cancel(&self, id: u64) -> CancelOutcome {
        let jobs = self.jobs.lock().unwrap_or_else(|e| e.into_inner());
        let Some(job) = jobs.iter().find(|job| job.id == id) else {
            return CancelOutcome::Unknown;
        };
        let state = job.state.lock().unwrap_or_else(|e| e.into_inner());
        if state.is_active() {
            job.cancel.store(true, Ordering::SeqCst);
            CancelOutcome::Requested
        } else {
            CancelOutcome::Finished(state.name())
        }
    }

    /// Every job, newest last, including finished ones: the page
    /// keeps showing a failure until the user retries it.
    pub(crate) fn statuses(&self) -> Vec<serde_json::Value> {
        let jobs = self.jobs.lock().unwrap_or_else(|e| e.into_inner());
        jobs.iter().map(|job| job.status()).collect()
    }

    /// The job thread: move each file into the cache layout. Every
    /// failure lands in `Failed` with the hub's or the filesystem's own
    /// cause text.
    fn run_job(&self, job: &Job) {
        match self.run_job_inner(job) {
            Ok(()) => {}
            Err(cause) if cause == CANCELLED => {
                self.emit(&format!("download {} cancelled", job.repo));
                *job.state.lock().unwrap_or_else(|e| e.into_inner()) = JobState::Cancelled;
            }
            Err(cause) => {
                let mut state = job.state.lock().unwrap_or_else(|e| e.into_inner());
                // A cancellation that loses the race to the error path
                // stays a cancellation, not a failure.
                if state.is_active() {
                    self.emit(&format!("download {} failed", job.repo));
                    *state = JobState::Failed(cause);
                }
            }
        }
    }

    fn run_job_inner(&self, job: &Job) -> Result<(), String> {
        self.emit(&format!(
            "download {} started: {} files",
            job.repo,
            job.files.len()
        ));
        let repo_dir = repo_cache_dir(&self.hub_dir, &job.repo);
        let snapshot_dir = repo_dir.join("snapshots").join(&job.commit);
        let blobs_dir = repo_dir.join("blobs");
        std::fs::create_dir_all(&snapshot_dir)
            .and_then(|_| std::fs::create_dir_all(&blobs_dir))
            .map_err(|error| format!("cannot create the cache layout: {error}"))?;
        let marker = incomplete_marker(&repo_dir, &job.commit);
        std::fs::write(&marker, b"")
            .map_err(|error| format!("cannot mark the snapshot incomplete: {error}"))?;
        let mut bytes_done = 0_u64;
        for (index, file) in job.files.iter().enumerate() {
            if job.cancel.load(Ordering::SeqCst) {
                return Err(CANCELLED.to_string());
            }
            set_progress(job, index as u32, bytes_done, &file.path);
            self.ensure_file(job, file, &snapshot_dir, &blobs_dir, bytes_done)?;
            bytes_done += file.size;
            set_progress(job, index as u32 + 1, bytes_done, &file.path);
        }
        // A named ref (not a pinned commit) is recorded the way `hf`
        // records it, atomically, so a reader never sees a half line.
        if job.revision != job.commit {
            let refs = repo_dir.join("refs");
            std::fs::create_dir_all(&refs)
                .map_err(|error| format!("cannot create {}: {error}", refs.display()))?;
            let tmp = refs.join(format!(".job{}.tmp", job.id));
            std::fs::write(&tmp, &job.commit)
                .and_then(|_| std::fs::rename(&tmp, refs.join(&job.revision)))
                .map_err(|error| format!("cannot record the ref: {error}"))?;
        }
        let _ = std::fs::remove_file(&marker);
        self.emit(&format!(
            "download {} finished: {:.1} MiB",
            job.repo,
            bytes_done as f64 / (1024.0 * 1024.0)
        ));
        let mut state = job.state.lock().unwrap_or_else(|e| e.into_inner());
        *state = JobState::Done {
            commit: job.commit.clone(),
            bytes: bytes_done,
            path: snapshot_dir,
        };
        Ok(())
    }

    /// Makes one snapshot file exist, whole and verified: an existing
    /// link that resolves to a blob of the stated size is kept; a blob
    /// of the stated content hash (another revision of the repo, or an
    /// earlier run) is linked without a transfer; anything else is
    /// fetched, checked and then linked.
    fn ensure_file(
        &self,
        job: &Job,
        file: &HubFile,
        snapshot_dir: &Path,
        blobs_dir: &Path,
        bytes_before: u64,
    ) -> Result<(), String> {
        let link = snapshot_dir.join(&file.path);
        if link_resolves_to(&link, file.size) {
            return Ok(());
        }
        // A dangling or wrong-sized link is a leftover, not a file.
        if link.symlink_metadata().is_ok() {
            let _ = std::fs::remove_file(&link);
        }
        let blob = match file.sha256.as_deref() {
            Some(sha) if blob_has_size(&blobs_dir.join(sha), file.size) => sha.to_string(),
            _ => self.fetch_verified(job, file, blobs_dir, bytes_before)?,
        };
        place_link(&link, &file.path, &blob, job.id)
    }

    /// Fetches one file into `blobs/` and returns its blob name, the
    /// sha256 of the bytes. The size and (for an LFS file) the hub's
    /// hash are checked before the blob takes its name; a mismatch
    /// deletes the partial file and fetches again.
    fn fetch_verified(
        &self,
        job: &Job,
        file: &HubFile,
        blobs_dir: &Path,
        bytes_before: u64,
    ) -> Result<String, String> {
        // A file whose hash the hub states keeps its partial under that hash: a later run, in
        // this process or another, continues it. A file with no stated hash restarts.
        let resumable = file.sha256.is_some();
        let part = match &file.sha256 {
            Some(sha) => blobs_dir.join(format!("{sha}.incomplete")),
            None => blobs_dir.join(format!(".{}.part", job.id)),
        };
        let mut last = String::new();
        for _ in 0..FILE_ATTEMPTS {
            set_bytes(job, bytes_before, &file.path);
            match self.fetch_once(job, file, &part, bytes_before) {
                Ok((hash, length)) => {
                    let wrong_size = length != file.size;
                    let wrong_hash = file.sha256.as_deref().is_some_and(|sha| sha != hash);
                    if wrong_size || wrong_hash {
                        let _ = std::fs::remove_file(&part);
                        last = format!(
                            "{} failed its integrity check ({} bytes, hash {}; the hub states {} bytes{})",
                            file.path,
                            length,
                            &hash[..12],
                            file.size,
                            file.sha256
                                .as_deref()
                                .map(|sha| format!(", hash {}", &sha[..12]))
                                .unwrap_or_default()
                        );
                        continue;
                    }
                    let blob = blobs_dir.join(&hash);
                    std::fs::rename(&part, &blob)
                        .map_err(|error| format!("cannot place {}: {error}", blob.display()))?;
                    return Ok(hash);
                }
                Err(cause) => {
                    if !resumable {
                        let _ = std::fs::remove_file(&part);
                    }
                    if cause == CANCELLED {
                        return Err(cause);
                    }
                    last = cause;
                }
            }
        }
        Err(last)
    }

    /// One pass over a file: ranged chunks appended to `part`, hashed
    /// as they arrive. Returns the sha256 and the byte count.
    fn fetch_once(
        &self,
        job: &Job,
        file: &HubFile,
        part: &Path,
        bytes_before: u64,
    ) -> Result<(String, u64), String> {
        let url = format!(
            "{}/{}/resolve/{}/{}",
            self.endpoint,
            job.repo,
            job.commit,
            urlencode_path(&file.path)
        );
        let mut hasher = sha2::Sha256::new();
        let mut offset = 0_u64;
        // Bytes an earlier pass left in a resumable partial are hashed again and kept; a partial
        // longer than the file is not one of its prefixes and starts over.
        let kept = if file.sha256.is_some() {
            std::fs::metadata(part).map_or(0, |meta| meta.len())
        } else {
            0
        };
        let mut out = if kept > 0 && kept <= file.size {
            let mut earlier = std::fs::File::open(part)
                .map_err(|error| format!("cannot read {}: {error}", part.display()))?;
            let mut buffer = vec![0_u8; 1 << 20];
            loop {
                let read = earlier
                    .read(&mut buffer)
                    .map_err(|error| format!("cannot read {}: {error}", part.display()))?;
                if read == 0 {
                    break;
                }
                hasher.update(&buffer[..read]);
                offset += read as u64;
            }
            std::fs::OpenOptions::new()
                .append(true)
                .open(part)
                .map_err(|error| format!("cannot append to {}: {error}", part.display()))?
        } else {
            std::fs::File::create(part)
                .map_err(|error| format!("cannot create {}: {error}", part.display()))?
        };
        job.written.fetch_add(offset, Ordering::Relaxed);
        while offset < file.size {
            let length = self.chunk_bytes.min(file.size - offset);
            let mut tries = 0;
            let chunk = loop {
                if job.cancel.load(Ordering::SeqCst) {
                    return Err(CANCELLED.to_string());
                }
                match self.fetch_chunk(job, &url, offset, length, length == file.size) {
                    Ok(chunk) => break chunk,
                    Err(cause) if cause == CANCELLED => return Err(cause),
                    Err(cause) if tries < CHUNK_RETRIES => {
                        tries += 1;
                        let _ = cause;
                    }
                    Err(cause) => return Err(format!("{}: {cause}", file.path)),
                }
            };
            hasher.update(&chunk);
            out.write_all(&chunk)
                .map_err(|error| format!("cannot write {}: {error}", part.display()))?;
            offset += chunk.len() as u64;
            job.written.fetch_add(chunk.len() as u64, Ordering::Relaxed);
            set_bytes(job, bytes_before + offset, &file.path);
        }
        out.sync_all()
            .map_err(|error| format!("cannot flush {}: {error}", file.path))?;
        Ok((format!("{:x}", hasher.finalize()), offset))
    }

    /// One ranged request on its own thread, so a blocked read never
    /// holds the job: the job thread polls for the answer, for a
    /// cancel, and for the budget. The request thread ends by the
    /// client's own call budget at the latest.
    fn fetch_chunk(
        &self,
        job: &Job,
        url: &str,
        start: u64,
        length: u64,
        whole_file: bool,
    ) -> Result<Vec<u8>, String> {
        let (sender, receiver) = std::sync::mpsc::channel();
        let agent = self.downloader.clone();
        let token = self.token.clone();
        let target = url.to_string();
        std::thread::Builder::new()
            .name("hf-chunk".to_string())
            .spawn(move || {
                let _ = sender.send(chunk_request(
                    &agent,
                    token.as_deref(),
                    &target,
                    start,
                    length,
                    whole_file,
                ));
            })
            .map_err(|error| format!("cannot start a transfer thread: {error}"))?;
        let deadline = std::time::Instant::now() + self.chunk_timeout * 2 + Duration::from_secs(1);
        loop {
            if job.cancel.load(Ordering::SeqCst) {
                return Err(CANCELLED.to_string());
            }
            match receiver.recv_timeout(Duration::from_millis(50)) {
                Ok(result) => return result,
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                    if std::time::Instant::now() > deadline {
                        return Err("the hub stopped sending".to_string());
                    }
                }
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                    return Err("the transfer thread ended without an answer".to_string());
                }
            }
        }
    }
}

/// The cancel marker the job loop passes up as an error value.
const CANCELLED: &str = "cancelled";

/// The outcome of a cancel request.
#[derive(Debug, PartialEq)]
pub(crate) enum CancelOutcome {
    Unknown,
    Requested,
    /// The job had already ended; the state name it ended in.
    Finished(&'static str),
}

/// Drops the oldest finished jobs beyond the keep limit.
fn prune_finished(jobs: &mut Vec<Arc<Job>>) {
    let finished = jobs
        .iter()
        .filter(|job| {
            !job.state
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .is_active()
        })
        .count();
    let mut excess = finished.saturating_sub(MAX_FINISHED_JOBS);
    jobs.retain(|job| {
        let ended = !job
            .state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .is_active();
        if ended && excess > 0 {
            excess -= 1;
            false
        } else {
            true
        }
    });
}

/// One ranged GET, read whole: the body must be exactly the bytes
/// asked for. A server that ignores the range may answer 200 only to
/// a request for the entire file.
fn chunk_request(
    agent: &ureq::Agent,
    token: Option<&str>,
    url: &str,
    start: u64,
    length: u64,
    whole_file: bool,
) -> Result<Vec<u8>, String> {
    let mut request = agent
        .get(url)
        .header("range", &format!("bytes={start}-{}", start + length - 1));
    if let Some(token) = token {
        request = request.header("authorization", &format!("Bearer {token}"));
    }
    let mut response = request.call().map_err(hub_error)?;
    let status = response.status().as_u16();
    if status != 206 && !(status == 200 && whole_file) {
        return Err(format!("the hub answered {status} to a range request"));
    }
    let mut body = Vec::with_capacity(usize::try_from(length).unwrap_or(0));
    response
        .body_mut()
        .as_reader()
        .take(length + 1)
        .read_to_end(&mut body)
        .map_err(|error| format!("transfer broke: {error}"))?;
    if body.len() as u64 != length {
        return Err(format!(
            "the hub sent {} bytes of the {length} requested",
            body.len()
        ));
    }
    Ok(body)
}

/// True when `link` resolves to a regular file of exactly `size` bytes.
fn link_resolves_to(link: &Path, size: u64) -> bool {
    std::fs::metadata(link).is_ok_and(|meta| meta.is_file() && meta.len() == size)
}

fn blob_has_size(blob: &Path, size: u64) -> bool {
    std::fs::metadata(blob).is_ok_and(|meta| meta.is_file() && meta.len() == size)
}

/// True when the snapshot needs nothing fetched for this file: its
/// link resolves to a blob of the stated size, or the blob of the
/// stated hash is already in the cache.
fn file_present(snapshot_dir: &Path, blobs_dir: &Path, file: &HubFile) -> bool {
    link_resolves_to(&snapshot_dir.join(&file.path), file.size)
        || file
            .sha256
            .as_deref()
            .is_some_and(|sha| blob_has_size(&blobs_dir.join(sha), file.size))
}

/// Points `link` at the blob, atomically: the link is made under a
/// temporary name beside it and renamed over the final name, so a
/// reader sees the old state or the new one and two writers of the same
/// file cannot leave a half state.
fn place_link(link: &Path, path: &str, blob: &str, job_id: u64) -> Result<(), String> {
    let parent = link
        .parent()
        .ok_or_else(|| format!("{} has no directory", link.display()))?;
    std::fs::create_dir_all(parent)
        .map_err(|error| format!("cannot create {}: {error}", parent.display()))?;
    let name = link
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default();
    let temporary = parent.join(format!(".{name}.{job_id}.lnk"));
    let _ = std::fs::remove_file(&temporary);
    #[cfg(unix)]
    std::os::unix::fs::symlink(blob_link_target(path, blob), &temporary)
        .map_err(|error| format!("cannot link {}: {error}", link.display()))?;
    #[cfg(not(unix))]
    {
        let _ = path;
        let blobs = parent.join(blob_link_target(path, blob));
        std::fs::copy(blobs, &temporary)
            .map_err(|error| format!("cannot place {}: {error}", link.display()))?;
    }
    std::fs::rename(&temporary, link)
        .map_err(|error| format!("cannot place {}: {error}", link.display()))
}

/// Space the downloads leave free on the cache volume, so the
/// machine keeps room for its own logs, caches and swap.
const DISK_SPARE_BYTES: u64 = 1 << 30;

/// Bytes an unprivileged writer can still allocate on the volume
/// that holds `path`, read from its nearest existing ancestor (the
/// cache directory is created by the first download). `None` when
/// the platform offers no answer; the writes then fail on their own.
pub(crate) fn free_disk_bytes(path: &Path) -> Option<u64> {
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
/// running downloads still have to write.
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

/// Replaces the job's counters: files done, bytes done, current file.
fn set_progress(job: &Job, files_done: u32, bytes: u64, current_file: &str) {
    let mut state = job.state.lock().unwrap_or_else(|e| e.into_inner());
    if let JobState::Downloading {
        files_done: done,
        bytes: counted,
        current_file: current,
        ..
    } = &mut *state
    {
        *done = files_done;
        *counted = bytes;
        *current = current_file.to_string();
    }
}

fn set_bytes(job: &Job, bytes: u64, current_file: &str) {
    let mut state = job.state.lock().unwrap_or_else(|e| e.into_inner());
    if let JobState::Downloading {
        bytes: counted,
        current_file: current,
        ..
    } = &mut *state
    {
        *counted = bytes;
        *current = current_file.to_string();
    }
}

/// A 64-hex etag or object id is the hub's content hash; anything else
/// (a git blob hash, `W/"..."`) is not reusable as a blob name.
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

/// True when every link under `dir` resolves to a file.
fn links_resolve(dir: &Path) -> bool {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return false;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let Ok(link_meta) = std::fs::symlink_metadata(&path) else {
            return false;
        };
        let resolved = if link_meta.is_symlink() {
            match std::fs::metadata(&path) {
                Ok(meta) => meta,
                Err(_) => return false,
            }
        } else {
            link_meta
        };
        if resolved.is_dir() && !links_resolve(&path) {
            return false;
        }
    }
    true
}

/// Every snapshot of one repo: its GGUF files, whether it is
/// complete, and, when it is, the family `serve --model` would start
/// it as and the memory the planner estimates for it. A snapshot is
/// complete when no job is still filling it and every link resolves.
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
            let complete = !incomplete_marker(repo_dir, &commit).exists() && links_resolve(&path);
            let family = if complete {
                super::model_serve::servable_family(&path).ok()
            } else {
                None
            };
            let estimate = family
                .as_deref()
                .and_then(|family| super::model_serve::estimate_tiers(&path, family))
                .map(|tiers| {
                    serde_json::json!({
                        "vram_bytes": tiers.vram_bytes,
                        "ram_bytes": tiers.ram_bytes,
                        "disk_bytes": tiers.disk_bytes,
                        "planned": true,
                        "basis": tiers.basis,
                        "mode": ESTIMATE_MODE,
                    })
                });
            let mut row = serde_json::json!({
                "commit": commit,
                "path": path.display().to_string(),
                "gguf": gguf,
                "complete": complete,
                "servable": family.is_some(),
                "families": family.iter().collect::<Vec<_>>(),
            });
            if let Some(estimate) = estimate {
                row["estimate"] = estimate;
            }
            Some(row)
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
    if !valid_revision(&revision) {
        return Err(EngError::BadRequest(format!(
            "{revision} is not a revision (one plain segment: main, a tag, a branch, a commit)"
        ))
        .into());
    }
    let queried_repo = repo.clone();
    let queried_revision = revision.clone();
    let tree = tokio::task::spawn_blocking(move || hub.tree(&queried_repo, &queried_revision))
        .await
        .map_err(|_| EngError::BadRequest("the listing did not finish".into()))?
        .map_err(EngError::BadRequest)?;
    let total: u64 = tree.files.iter().map(|file| file.size).sum();
    Ok(Json(serde_json::json!({
        "repo": repo,
        "revision": revision,
        "commit": tree.commit,
        "total_bytes": total,
        "estimate": pack_estimate(&tree.files),
        "files": tree
            .files
            .iter()
            .map(|file| serde_json::json!({"path": file.path, "size": file.size}))
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
    let hub = Arc::clone(state.hf());
    let repo = body.repo.clone();
    let wanted = revision.clone();
    let job = tokio::task::spawn_blocking(move || hub.start_download(&repo, &wanted))
        .await
        .map_err(|_| EngError::BadRequest("the download did not start".into()))?
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
        CancelOutcome::Requested => Ok(Json(serde_json::json!({ "id": id, "cancelling": true }))),
        CancelOutcome::Finished(ended) => Ok(Json(serde_json::json!({
            "id": id,
            "cancelling": false,
            "state": ended,
        }))),
        CancelOutcome::Unknown => Err(ServerError::NotFound(format!(
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
        let commit = "0123456789abcdef0123456789abcdef01234567";
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
        std::fs::write(
            repo.join("refs").join("main"),
            "0123456789abcdef0123456789abcdef01234567\n",
        )
        .expect("ref");
        assert_eq!(cached_main_snapshot_in(hub.path(), "org/dangling"), None);
        // A ref that is not a commit id never names a directory.
        std::fs::write(repo.join("refs").join("main"), "../../x\n").expect("ref");
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
}
