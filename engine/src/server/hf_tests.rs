//! The hub bridge against a mock hub: a loopback HTTP server that
//! answers the info, listing and ranged-download calls of one repo,
//! with switches for the failures a real hub produces (a connection
//! that never answers, corrupt bytes, a short body, a failing listing).

use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::http::StatusCode;
use sha2::Digest;

use super::hf::{HfHub, JobState, free_disk_bytes, repo_cache_dir};
use super::router::router;
use super::state::AppState;
use super::testkit;

const COMMIT: &str = "0123456789abcdef0123456789abcdef01234567";

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", sha2::Sha256::digest(bytes))
}

#[derive(Clone)]
struct MockFile {
    path: String,
    content: Vec<u8>,
    /// What the hub actually sends, when it differs from `content`.
    served: Option<Vec<u8>>,
    /// Corrupt this many first full passes of the file.
    corrupt_passes: usize,
}

struct MockHub {
    endpoint: String,
    stall: Arc<AtomicBool>,
    list_fails: Arc<AtomicBool>,
    requests: Arc<Mutex<Vec<String>>>,
}

impl MockHub {
    fn start(files: Vec<MockFile>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let endpoint = format!("http://{}", listener.local_addr().expect("address"));
        let stall = Arc::new(AtomicBool::new(false));
        let list_fails = Arc::new(AtomicBool::new(false));
        let requests = Arc::new(Mutex::new(Vec::new()));
        let files = Arc::new(files);
        let passes: Arc<Mutex<HashMap<String, usize>>> = Arc::default();
        let (stall_in, fails_in, log_in) = (
            Arc::clone(&stall),
            Arc::clone(&list_fails),
            Arc::clone(&requests),
        );
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(stream) = stream else { break };
                let (files, stall, fails, log, passes) = (
                    Arc::clone(&files),
                    Arc::clone(&stall_in),
                    Arc::clone(&fails_in),
                    Arc::clone(&log_in),
                    Arc::clone(&passes),
                );
                std::thread::spawn(move || serve(stream, &files, &stall, &fails, &log, &passes));
            }
        });
        Self {
            endpoint,
            stall,
            list_fails,
            requests,
        }
    }

    fn resolve_requests(&self) -> Vec<String> {
        self.requests
            .lock()
            .expect("log")
            .iter()
            .filter(|line| line.contains("/resolve/"))
            .cloned()
            .collect()
    }
}

fn serve(
    mut stream: TcpStream,
    files: &[MockFile],
    stall: &AtomicBool,
    list_fails: &AtomicBool,
    log: &Mutex<Vec<String>>,
    passes: &Mutex<HashMap<String, usize>>,
) {
    let mut head = Vec::new();
    let mut byte = [0_u8; 1];
    while !head.ends_with(b"\r\n\r\n") {
        match stream.read(&mut byte) {
            Ok(1) => head.push(byte[0]),
            _ => return,
        }
    }
    let text = String::from_utf8_lossy(&head).to_string();
    let first = text.lines().next().unwrap_or_default().to_string();
    let path = first.split(' ').nth(1).unwrap_or_default().to_string();
    let range = text.lines().find_map(|line| {
        line.to_ascii_lowercase()
            .strip_prefix("range: bytes=")
            .map(str::to_string)
    });
    log.lock().expect("log").push(format!(
        "{path} range={}",
        range.clone().unwrap_or_default()
    ));
    let reply = |stream: &mut TcpStream, status: &str, extra: &str, body: &[u8], claimed: usize| {
        let _ = write!(
            stream,
            "HTTP/1.1 {status}\r\ncontent-type: application/json\r\ncontent-length: {claimed}\r\n{extra}connection: close\r\n\r\n"
        );
        let _ = stream.write_all(body);
    };
    // The real hub answers 400 to an encoded slash in the repo id.
    if path.contains("%2F") {
        return reply(&mut stream, "400 Bad Request", "", b"{}", 2);
    }
    if path.contains("/resolve/") {
        if stall.load(Ordering::SeqCst) {
            std::thread::sleep(Duration::from_secs(30));
            return;
        }
        let name = path
            .split("/resolve/")
            .nth(1)
            .and_then(|rest| rest.split_once('/'))
            .map(|(_, p)| p)
            .unwrap_or_default();
        let Some(file) = files.iter().find(|f| f.path == name) else {
            return reply(&mut stream, "404 Not Found", "", b"{}", 2);
        };
        let mut data = file.served.clone().unwrap_or_else(|| file.content.clone());
        let (start, end) = range
            .as_deref()
            .and_then(|r| r.split_once('-'))
            .and_then(|(a, b)| Some((a.parse::<usize>().ok()?, b.parse::<usize>().ok()?)))
            .unwrap_or((0, file.content.len().saturating_sub(1)));
        if start == 0 {
            let mut seen = passes.lock().expect("passes");
            let count = seen.entry(file.path.clone()).or_default();
            if *count < file.corrupt_passes && !data.is_empty() {
                data[0] ^= 0xff;
            }
            *count += 1;
        }
        let wanted = end + 1 - start;
        let slice = &data[start.min(data.len())..(end + 1).min(data.len())];
        reply(
            &mut stream,
            "206 Partial Content",
            &format!(
                "content-range: bytes {start}-{end}/{}\r\n",
                file.content.len()
            ),
            slice,
            wanted,
        );
    } else if path.contains("/tree/") {
        if list_fails.load(Ordering::SeqCst) {
            return reply(&mut stream, "500 Internal Server Error", "", b"{}", 2);
        }
        let rows: Vec<serde_json::Value> = files
            .iter()
            .map(|f| {
                let oid = sha256_hex(&f.content);
                serde_json::json!({
                    "type": "file",
                    "path": f.path,
                    "size": f.content.len(),
                    "lfs": {"oid": oid, "size": f.content.len()},
                })
            })
            .collect();
        let body = serde_json::to_vec(&rows).expect("json");
        reply(&mut stream, "200 OK", "", &body, body.len());
    } else if path.starts_with("/api/models/") {
        let siblings: Vec<serde_json::Value> = files
            .iter()
            .map(|f| serde_json::json!({"rfilename": f.path}))
            .collect();
        let body = serde_json::to_vec(&serde_json::json!({"sha": COMMIT, "siblings": siblings}))
            .expect("json");
        reply(&mut stream, "200 OK", "", &body, body.len());
    } else {
        reply(&mut stream, "404 Not Found", "", b"{}", 2);
    }
}

fn file(path: &str, content: &[u8]) -> MockFile {
    MockFile {
        path: path.to_string(),
        content: content.to_vec(),
        served: None,
        corrupt_passes: 0,
    }
}

fn payload(length: usize, seed: u8) -> Vec<u8> {
    (0..length)
        .map(|i| (i as u8).wrapping_mul(31).wrapping_add(seed))
        .collect()
}

fn hub_with(
    mock: &MockHub,
    dir: &Path,
    chunk: u64,
    timeout: Duration,
    free: Option<u64>,
) -> Arc<HfHub> {
    Arc::new(HfHub::with_limits(
        mock.endpoint.clone(),
        dir.to_path_buf(),
        None,
        chunk,
        timeout,
        Box::new(move |path| free.or_else(|| free_disk_bytes(path))),
    ))
}

fn wait_for<T>(mut probe: impl FnMut() -> Option<T>, seconds: u64) -> T {
    let deadline = Instant::now() + Duration::from_secs(seconds);
    loop {
        if let Some(found) = probe() {
            return found;
        }
        assert!(Instant::now() < deadline, "timed out after {seconds} s");
        std::thread::sleep(Duration::from_millis(20));
    }
}

fn finished(job: &super::hf::Job) -> JobState {
    wait_for(
        || {
            let state = job.state.lock().expect("state").clone();
            (!matches!(state, JobState::Downloading { .. })).then_some(state)
        },
        90,
    )
}

fn snapshot(dir: &Path, repo: &str) -> PathBuf {
    repo_cache_dir(dir, repo).join("snapshots").join(COMMIT)
}

#[test]
fn a_download_writes_the_hub_layout_with_verified_blobs_and_relative_links() {
    let dir = tempfile::tempdir().expect("hub dir");
    let big = payload(1000, 3);
    let nested = payload(37, 9);
    let mock = MockHub::start(vec![
        file("model.gguf", &big),
        file("sub/dir/extra.json", &nested),
        file("config.json", b"{\"model_type\":\"x\"}"),
    ]);
    // 64-byte chunks make the big file a 16-request resume-able sequence.
    let hub = hub_with(&mock, dir.path(), 64, Duration::from_secs(20), None);
    let job = hub.start_download("org/name", "main").expect("admitted");
    let state = finished(&job);
    assert!(matches!(state, JobState::Done { .. }), "{state:?}");

    let repo = repo_cache_dir(dir.path(), "org/name");
    let snap = snapshot(dir.path(), "org/name");
    assert_eq!(std::fs::read(snap.join("model.gguf")).expect("link"), big);
    assert_eq!(
        std::fs::read(snap.join("sub/dir/extra.json")).expect("nested"),
        nested
    );
    let target = std::fs::read_link(snap.join("sub/dir/extra.json")).expect("a symlink");
    assert_eq!(
        target,
        Path::new(&format!("../../../../blobs/{}", sha256_hex(&nested)))
    );
    assert!(repo.join("blobs").join(sha256_hex(&big)).is_file());
    assert_eq!(
        std::fs::read_to_string(repo.join("refs/main")).expect("ref"),
        COMMIT
    );
    assert!(!repo.join(format!(".fifi-incomplete-{COMMIT}")).exists());
    let ranges = mock.resolve_requests();
    assert!(
        ranges.iter().filter(|r| r.contains("model.gguf")).count() >= 16,
        "{ranges:?}"
    );
    let leftovers: Vec<_> = std::fs::read_dir(repo.join("blobs"))
        .expect("blobs")
        .flatten()
        .filter(|e| e.file_name().to_string_lossy().starts_with('.'))
        .collect();
    assert!(leftovers.is_empty(), "{leftovers:?}");

    // The cache listing shows the snapshot as complete.
    let listing = hub.cache();
    let row = &listing["models"][0]["snapshots"][0];
    assert_eq!(row["commit"], COMMIT);
    assert_eq!(row["complete"], true);
    assert_eq!(row["gguf"], serde_json::json!(["model.gguf"]));
}

#[test]
fn a_repeat_download_resumes_by_fetching_only_what_is_missing() {
    let dir = tempfile::tempdir().expect("hub dir");
    let first = payload(200, 1);
    let second = payload(300, 2);
    let mock = MockHub::start(vec![file("a.bin", &first), file("b.bin", &second)]);
    let hub = hub_with(&mock, dir.path(), 128, Duration::from_secs(20), None);
    let job = hub.start_download("org/name", "main").expect("admitted");
    assert!(matches!(finished(&job), JobState::Done { .. }));
    // An interrupted earlier run: one file's link and blob are gone.
    let snap = snapshot(dir.path(), "org/name");
    std::fs::remove_file(snap.join("b.bin")).expect("remove link");
    std::fs::remove_file(
        repo_cache_dir(dir.path(), "org/name")
            .join("blobs")
            .join(sha256_hex(&second)),
    )
    .expect("remove blob");
    mock.requests.lock().expect("log").clear();
    let job = hub.start_download("org/name", "main").expect("admitted");
    assert!(matches!(finished(&job), JobState::Done { .. }));
    let asked = mock.resolve_requests();
    assert!(asked.iter().all(|r| r.contains("b.bin")), "{asked:?}");
    assert!(!asked.is_empty());
    assert_eq!(std::fs::read(snap.join("b.bin")).expect("restored"), second);
}

#[test]
fn a_cancel_interrupts_a_blocked_transfer_and_frees_the_slot() {
    let dir = tempfile::tempdir().expect("hub dir");
    let mock = MockHub::start(vec![file("m.bin", &payload(500, 4))]);
    mock.stall.store(true, Ordering::SeqCst);
    let hub = hub_with(&mock, dir.path(), 128, Duration::from_secs(60), None);
    let job = hub.start_download("org/name", "main").expect("admitted");
    wait_for(|| (!mock.resolve_requests().is_empty()).then_some(()), 60);
    assert_eq!(hub.cancel(job.id), super::hf::CancelOutcome::Requested);
    let started = Instant::now();
    assert_eq!(finished(&job), JobState::Cancelled);
    assert!(
        started.elapsed() < Duration::from_secs(30),
        "cancel waited for the read"
    );
    assert_eq!(
        hub.cancel(job.id),
        super::hf::CancelOutcome::Finished("cancelled")
    );
    assert_eq!(
        hub.cancel(job.id + 999_999),
        super::hf::CancelOutcome::Unknown
    );
    // The slot is free again.
    mock.stall.store(false, Ordering::SeqCst);
    let again = hub
        .start_download("org/name", "main")
        .expect("slot is free");
    assert!(matches!(finished(&again), JobState::Done { .. }));
}

#[test]
fn a_stalled_hub_fails_the_job_and_frees_its_slot() {
    let dir = tempfile::tempdir().expect("hub dir");
    let mock = MockHub::start(vec![file("m.bin", &payload(500, 5))]);
    mock.stall.store(true, Ordering::SeqCst);
    let hub = hub_with(&mock, dir.path(), 128, Duration::from_millis(300), None);
    let job = hub.start_download("org/name", "main").expect("admitted");
    match finished(&job) {
        JobState::Failed(cause) => assert!(cause.contains("m.bin"), "{cause}"),
        other => panic!("expected a failure, got {other:?}"),
    }
    // Two stalled jobs on two repos never hold more than their time.
    let second = hub
        .start_download("org/other", "main")
        .expect("slot is free");
    assert!(matches!(finished(&second), JobState::Failed(_)));
    assert!(hub.statuses().iter().all(|row| row["state"] == "failed"));
}

#[test]
fn a_truncated_or_corrupt_file_is_deleted_and_never_linked() {
    let dir = tempfile::tempdir().expect("hub dir");
    let content = payload(300, 6);
    let mut short = file("short.bin", &content);
    short.served = Some(content[..150].to_vec());
    let mut corrupt = file("bad.bin", &content);
    corrupt.corrupt_passes = usize::MAX;
    let mock = MockHub::start(vec![short]);
    let hub = hub_with(&mock, dir.path(), 128, Duration::from_secs(20), None);
    let job = hub.start_download("org/short", "main").expect("admitted");
    match finished(&job) {
        JobState::Failed(cause) => assert!(cause.contains("short.bin"), "{cause}"),
        other => panic!("{other:?}"),
    }
    let snap = snapshot(dir.path(), "org/short");
    assert!(
        snap.join("short.bin").symlink_metadata().is_err(),
        "a short file must not be linked"
    );

    let mock = MockHub::start(vec![corrupt]);
    let hub = hub_with(&mock, dir.path(), 128, Duration::from_secs(20), None);
    let job = hub.start_download("org/bad", "main").expect("admitted");
    match finished(&job) {
        JobState::Failed(cause) => assert!(cause.contains("integrity check"), "{cause}"),
        other => panic!("{other:?}"),
    }
    let repo = repo_cache_dir(dir.path(), "org/bad");
    assert!(
        snapshot(dir.path(), "org/bad")
            .join("bad.bin")
            .symlink_metadata()
            .is_err()
    );
    let blobs: Vec<_> = std::fs::read_dir(repo.join("blobs"))
        .expect("blobs")
        .flatten()
        .collect();
    assert!(
        blobs.is_empty(),
        "no blob may survive a failed check: {blobs:?}"
    );
    // The failed snapshot is marked incomplete.
    assert!(repo.join(format!(".fifi-incomplete-{COMMIT}")).exists());
}

#[test]
fn one_corrupt_pass_is_retried_and_the_second_pass_verifies() {
    let dir = tempfile::tempdir().expect("hub dir");
    let content = payload(300, 7);
    let mut flaky = file("m.bin", &content);
    flaky.corrupt_passes = 1;
    let mock = MockHub::start(vec![flaky]);
    let hub = hub_with(&mock, dir.path(), 4096, Duration::from_secs(20), None);
    let job = hub.start_download("org/name", "main").expect("admitted");
    assert!(matches!(finished(&job), JobState::Done { .. }));
    assert_eq!(
        std::fs::read(snapshot(dir.path(), "org/name").join("m.bin")).expect("linked"),
        content
    );
}

#[test]
fn a_dangling_or_wrong_sized_link_is_replaced_not_trusted() {
    let dir = tempfile::tempdir().expect("hub dir");
    let content = payload(120, 8);
    let mock = MockHub::start(vec![file("m.bin", &content)]);
    let hub = hub_with(&mock, dir.path(), 4096, Duration::from_secs(20), None);
    let snap = snapshot(dir.path(), "org/name");
    std::fs::create_dir_all(&snap).expect("snapshot");
    std::os::unix::fs::symlink("../../blobs/gone", snap.join("m.bin")).expect("dangling");
    // The listing already calls the half-filled snapshot incomplete.
    assert_eq!(hub.cache()["models"][0]["snapshots"][0]["complete"], false);
    let job = hub.start_download("org/name", "main").expect("admitted");
    assert!(matches!(finished(&job), JobState::Done { .. }));
    assert_eq!(
        std::fs::read(snap.join("m.bin")).expect("replaced"),
        content
    );
    // A link to a blob of the wrong size is replaced too.
    let repo = repo_cache_dir(dir.path(), "org/name");
    std::fs::write(repo.join("blobs/wrong"), b"short").expect("wrong blob");
    std::fs::remove_file(snap.join("m.bin")).expect("remove");
    std::os::unix::fs::symlink("../../blobs/wrong", snap.join("m.bin")).expect("wrong link");
    let job = hub.start_download("org/name", "main").expect("admitted");
    assert!(matches!(finished(&job), JobState::Done { .. }));
    assert_eq!(
        std::fs::read(snap.join("m.bin")).expect("replaced"),
        content
    );
}

#[test]
fn a_failed_listing_refuses_the_download_with_its_error() {
    let dir = tempfile::tempdir().expect("hub dir");
    let mock = MockHub::start(vec![file("m.bin", &payload(10, 1))]);
    mock.list_fails.store(true, Ordering::SeqCst);
    let hub = hub_with(&mock, dir.path(), 4096, Duration::from_secs(20), None);
    let refusal = hub
        .start_download("org/name", "main")
        .err()
        .expect("refused");
    assert!(
        refusal.contains("file sizes could not be read"),
        "{refusal}"
    );
    assert!(
        !repo_cache_dir(dir.path(), "org/name").exists(),
        "nothing may be written"
    );
}

#[test]
fn two_downloads_started_together_on_a_disk_that_fits_one_admit_one() {
    let dir = tempfile::tempdir().expect("hub dir");
    let content = payload(4000, 2);
    let mock = MockHub::start(vec![file("m.bin", &content)]);
    // Jobs stay running (the hub stalls), so the first one's claim stands.
    mock.stall.store(true, Ordering::SeqCst);
    let spare = 1_u64 << 30;
    let hub = hub_with(
        &mock,
        dir.path(),
        4096,
        Duration::from_secs(60),
        Some(spare + 5000),
    );
    let results: Vec<Result<Arc<super::hf::Job>, String>> = std::thread::scope(|scope| {
        let handles: Vec<_> = ["org/one", "org/two"]
            .into_iter()
            .map(|repo| {
                let hub = Arc::clone(&hub);
                scope.spawn(move || hub.start_download(repo, "main"))
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("thread"))
            .collect()
    });
    let admitted = results.iter().filter(|r| r.is_ok()).count();
    assert_eq!(
        admitted,
        1,
        "{:?}",
        results.iter().map(|r| r.as_ref().err()).collect::<Vec<_>>()
    );
    let refusal = results
        .iter()
        .find_map(|r| r.as_ref().err())
        .expect("one refusal");
    assert!(
        refusal.contains("needs") && refusal.contains("nothing was written"),
        "{refusal}"
    );
    for job in results.iter().flatten() {
        hub.cancel(job.id);
        finished(job);
    }
}

#[test]
fn the_same_blob_is_shared_between_revisions_and_job_ids_do_not_restart_at_one() {
    let dir = tempfile::tempdir().expect("hub dir");
    let content = payload(200, 3);
    let mock = MockHub::start(vec![file("m.bin", &content)]);
    let hub = hub_with(&mock, dir.path(), 4096, Duration::from_secs(20), None);
    let first = hub.start_download("org/name", "main").expect("admitted");
    assert!(matches!(finished(&first), JobState::Done { .. }));
    assert!(
        first.id > 1_000_000_000_000,
        "ids start at the clock, not at 1: {}",
        first.id
    );
    mock.requests.lock().expect("log").clear();
    // The mock names the same commit for any revision; a second revision
    // that lists the same content reuses the blob.
    let snap = snapshot(dir.path(), "org/name");
    std::fs::remove_file(snap.join("m.bin")).expect("remove link");
    let second = hub.start_download("org/name", "v2").expect("admitted");
    assert!(matches!(finished(&second), JobState::Done { .. }));
    assert!(
        mock.resolve_requests().is_empty(),
        "the blob was already there"
    );
    assert_eq!(
        std::fs::read(snap.join("m.bin")).expect("relinked"),
        content
    );
}

#[test]
fn the_sweep_removes_what_a_dead_process_left_half_written() {
    let dir = tempfile::tempdir().expect("hub dir");
    let repo = repo_cache_dir(dir.path(), "org/name");
    std::fs::create_dir_all(repo.join("blobs")).expect("blobs");
    std::fs::create_dir_all(repo.join("refs")).expect("refs");
    std::fs::create_dir_all(repo.join("snapshots/c/sub")).expect("snapshot");
    std::fs::write(repo.join("blobs/.7.part"), b"x").expect("part");
    std::fs::write(repo.join("blobs/keep"), b"x").expect("blob");
    std::fs::write(repo.join("refs/.job7.tmp"), b"x").expect("tmp");
    std::fs::write(repo.join("refs/main"), COMMIT).expect("ref");
    std::os::unix::fs::symlink("x", repo.join("snapshots/c/sub/.f.7.lnk")).expect("tmp link");
    super::hf::sweep_partials(dir.path());
    assert!(!repo.join("blobs/.7.part").exists());
    assert!(!repo.join("refs/.job7.tmp").exists());
    assert!(
        repo.join("snapshots/c/sub/.f.7.lnk")
            .symlink_metadata()
            .is_err()
    );
    assert!(repo.join("blobs/keep").exists() && repo.join("refs/main").exists());
}

#[test]
fn the_listing_offers_serve_only_for_a_complete_snapshot_of_a_known_family() {
    let dir = tempfile::tempdir().expect("hub dir");
    let hub = HfHub::new(
        "http://127.0.0.1:9".to_string(),
        dir.path().to_path_buf(),
        None,
    );
    let repo = repo_cache_dir(dir.path(), "org/q36");
    let model_type = crate::family::QWEN3_6_35B_A3B.model_types[0];
    for (commit, marker) in [("a".repeat(40), false), ("b".repeat(40), true)] {
        let snap = repo.join("snapshots").join(&commit);
        std::fs::create_dir_all(&snap).expect("snapshot");
        std::fs::create_dir_all(repo.join("blobs")).expect("blobs");
        std::fs::write(
            snap.join("config.json"),
            format!("{{\"model_type\":\"{model_type}\"}}"),
        )
        .expect("config");
        std::fs::write(snap.join("model.safetensors"), vec![0_u8; 64]).expect("weights");
        if marker {
            std::fs::write(repo.join(format!(".fifi-incomplete-{commit}")), b"").expect("marker");
        }
    }
    let listing = hub.cache();
    let rows = listing["models"][0]["snapshots"]
        .as_array()
        .expect("snapshots");
    let complete = rows
        .iter()
        .find(|r| r["commit"] == "a".repeat(40))
        .expect("complete row");
    let partial = rows
        .iter()
        .find(|r| r["commit"] == "b".repeat(40))
        .expect("partial row");
    assert_eq!(complete["complete"], true);
    assert_eq!(complete["servable"], true);
    assert_eq!(
        complete["families"],
        serde_json::json!([crate::family::qwen::qwen36::factory::FAMILY])
    );
    assert_eq!(partial["complete"], false);
    assert_eq!(partial["servable"], false);
    assert_eq!(partial["families"], serde_json::json!([]));
}

fn derive_fake(
    args: &super::model_serve::ModelServeArgs,
) -> Result<super::config::Config, super::StartupError> {
    let tokenizer = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/server/testdata/tokenizer");
    Ok(serde_json::from_value(serde_json::json!({
        "listen": "127.0.0.1:1",
        "api_key_env": "K",
        "engines": [{
            "id": "derived",
            "model_dir": args.model_dir,
            "tokenizer_dir": tokenizer,
            "family": "fake",
        }],
    }))
    .expect("config"))
}

#[tokio::test]
async fn a_downloaded_model_is_served_by_repo_name_through_the_engine_route() {
    let dir = tempfile::tempdir().expect("hub dir");
    let mock = MockHub::start(vec![
        file("config.json", b"{}"),
        file("m.bin", &payload(100, 1)),
    ]);
    let hub = HfHub::with_limits(
        mock.endpoint.clone(),
        dir.path().to_path_buf(),
        None,
        4096,
        Duration::from_secs(20),
        Box::new(|_| None),
    );
    let state = Arc::new(
        AppState::new(testkit::API_KEY, Vec::new())
            .with_hf(hub)
            .with_derive(derive_fake),
    );
    let app = router(Arc::clone(&state));
    let post = |uri: &str, body: &str| {
        axum::http::Request::builder()
            .method("POST")
            .uri(uri)
            .header(
                axum::http::header::AUTHORIZATION,
                format!("Bearer {}", testkit::API_KEY),
            )
            .header(axum::http::header::CONTENT_TYPE, "application/json")
            .body(axum::body::Body::from(body.to_string()))
            .expect("request")
    };
    let (status, _headers, body) =
        testkit::send(&app, post("/v1/hf/downloads", r#"{"repo":"org/name"}"#)).await;
    assert_eq!(
        status,
        StatusCode::ACCEPTED,
        "{}",
        String::from_utf8_lossy(&body)
    );
    let mut last = serde_json::Value::Null;
    for _ in 0..3600 {
        let (_status, _headers, body) = testkit::send(
            &app,
            testkit::get("/v1/hf/downloads", Some(testkit::API_KEY)),
        )
        .await;
        last = testkit::json_of(&body);
        if last["downloads"][0]["state"] != "downloading" {
            break;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    assert_eq!(last["downloads"][0]["state"], "done", "{last}");
    let (_status, _headers, body) =
        testkit::send(&app, post("/v1/engines", r#"{"model_dir":"org/name"}"#)).await;
    let answer = testkit::json_of(&body);
    assert_eq!(answer["id"], "name", "{answer}");
    let expected = snapshot(dir.path(), "org/name").display().to_string();
    assert_eq!(answer["model_dir"], expected.as_str());
    let (_status, _headers, body) =
        testkit::send(&app, testkit::get("/v1/engines", Some(testkit::API_KEY))).await;
    let listed = testkit::json_of(&body);
    assert_eq!(listed["engines"][0]["runtime_added"], true);
    assert_eq!(listed["engines"][0]["model_dir"], expected.as_str());
    // A finished job answers a cancel with what it finished as.
    let id = testkit::json_of(
        &testkit::send(
            &app,
            testkit::get("/v1/hf/downloads", Some(testkit::API_KEY)),
        )
        .await
        .2,
    )["downloads"][0]["id"]
        .clone();
    let request = testkit::request(
        "DELETE",
        &format!("/v1/hf/downloads/{id}"),
        Some(testkit::API_KEY),
    );
    let (_status, _headers, body) = testkit::send(&app, request).await;
    let outcome = testkit::json_of(&body);
    assert_eq!(outcome["cancelling"], false);
    assert_eq!(outcome["state"], "done");
    let request = testkit::request("DELETE", "/v1/engines/name", Some(testkit::API_KEY));
    let (status, _headers, _body) = testkit::send(&app, request).await;
    assert_eq!(status, StatusCode::OK);
}

#[test]
fn a_pack_estimate_counts_one_quantization_of_one_revision() {
    use super::hf::{HubFile, pack_estimate};
    let files = |rows: &[(&str, u64)]| -> Vec<HubFile> {
        rows.iter()
            .map(|(path, size)| HubFile {
                path: path.to_string(),
                size: *size,
                sha256: None,
            })
            .collect()
    };
    let est = pack_estimate(&files(&[
        ("m-Q4_K_M.gguf", 4000),
        ("m-Q8_0.gguf", 8000),
        ("m-IQ2.gguf", 2000),
        ("config.json", 50),
    ]));
    assert_eq!(est["vram_bytes"], 8000);
    let split = pack_estimate(&files(&[
        ("big-Q4-00001-of-00002.gguf", 3000),
        ("big-Q4-00002-of-00002.gguf", 3000),
        ("big-Q2.gguf", 5000),
    ]));
    assert_eq!(split["vram_bytes"], 6000);
    let shards = pack_estimate(&files(&[("a.safetensors", 10), ("b.safetensors", 20)]));
    assert_eq!(shards["vram_bytes"], 30);
    assert_eq!(shards["planned"], false);
}

/// A GGUF file with no keys and no tensors, `length` bytes long: a header the planner can read.
fn gguf_file(length: usize, fill: u8) -> Vec<u8> {
    let mut bytes = b"GGUF".to_vec();
    bytes.extend_from_slice(&3u32.to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes());
    bytes.resize(length, fill);
    bytes
}

fn fetch_with(
    hub: &Arc<HfHub>,
    quant: Option<&str>,
    fit: &mut dyn FnMut(&Path) -> Result<super::fetch_name::Fit, super::config::StartupError>,
    accepts: &dyn Fn(&Path) -> Result<(), super::config::StartupError>,
) -> Result<PathBuf, super::config::StartupError> {
    super::fetch_name::resolve(
        hub,
        &super::fetch_name::Wanted {
            repo: "org/m".to_string(),
            quant: quant.map(str::to_string),
        },
        super::fetch_name::Planner { accepts, fit },
        &mut |_| {},
    )
}

#[test]
fn a_name_fetches_only_the_files_of_the_quantization_asked_for() {
    let q4 = gguf_file(300, 4);
    let q8 = gguf_file(600, 8);
    let mock = MockHub::start(vec![
        file("README.md", b"readme"),
        file("model-Q4_K_M.gguf", &q4),
        file("model-Q8_0.gguf", &q8),
        file("mmproj-M-F16.gguf", &gguf_file(90, 1)),
    ]);
    let dir = tempfile::tempdir().expect("tempdir");
    let hub = hub_with(&mock, dir.path(), 64, Duration::from_secs(5), None);
    let path = fetch_with(
        &hub,
        Some("q8_0"),
        &mut |_| Ok(super::fetch_name::Fit::Resident),
        &|_| Ok(()),
    )
    .expect("fetched");
    assert_eq!(path, snapshot(dir.path(), "org/m").join("model-Q8_0.gguf"));
    assert_eq!(std::fs::read(&path).expect("file"), q8);
    let blobs: Vec<String> = std::fs::read_dir(repo_cache_dir(dir.path(), "org/m").join("blobs"))
        .expect("blobs")
        .flatten()
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .collect();
    assert_eq!(
        blobs,
        vec![sha256_hex(&q8)],
        "only the chosen quantization is stored"
    );
}

#[test]
fn without_a_label_the_planner_chooses_and_a_planner_stub_has_the_real_header_and_size() {
    let q4 = gguf_file(300, 4);
    let q8 = gguf_file(600, 8);
    let mock = MockHub::start(vec![
        file("model-Q4_K_M.gguf", &q4),
        file("model-Q8_0.gguf", &q8),
    ]);
    let dir = tempfile::tempdir().expect("tempdir");
    let hub = hub_with(&mock, dir.path(), 64, Duration::from_secs(5), None);
    let mut seen = Vec::new();
    let path = fetch_with(
        &hub,
        None,
        &mut |stub| {
            let length = std::fs::metadata(stub).expect("stub").len();
            assert_eq!(&std::fs::read(stub).expect("stub")[..4], b"GGUF");
            seen.push(length);
            if length <= 300 {
                Ok(super::fetch_name::Fit::Resident)
            } else {
                Ok(super::fetch_name::Fit::NotResident)
            }
        },
        &|_| Ok(()),
    )
    .expect("fetched");
    // The planner saw the largest first, then the next, at each one's own size.
    assert_eq!(seen, vec![600, 300]);
    assert!(path.ends_with("model-Q4_K_M.gguf"), "{path:?}");
    // The stubs are gone once the choice is made.
    assert!(
        !dir.path()
            .join(format!(".fifi-plan-{}", std::process::id()))
            .exists()
    );
}

#[test]
fn a_repo_the_engine_does_not_serve_is_refused_before_any_weight_is_downloaded() {
    let mock = MockHub::start(vec![file("model-Q4_K_M.gguf", &gguf_file(300, 4))]);
    let dir = tempfile::tempdir().expect("tempdir");
    let hub = hub_with(&mock, dir.path(), 64, Duration::from_secs(5), None);
    let error = fetch_with(
        &hub,
        None,
        &mut |_| Ok(super::fetch_name::Fit::Resident),
        &|_| {
            Err(super::config::StartupError(
                "no family serves the GGUF architecture mamba9".into(),
            ))
        },
    )
    .expect_err("refused");
    assert!(error.0.contains("mamba9"), "{}", error.0);
    assert!(
        !repo_cache_dir(dir.path(), "org/m").join("blobs").exists(),
        "nothing was stored"
    );
}

#[test]
fn a_disk_too_small_for_the_smallest_quantization_refuses_before_any_file_is_read() {
    let mock = MockHub::start(vec![file("model-Q4_K_M.gguf", &gguf_file(300, 4))]);
    let dir = tempfile::tempdir().expect("tempdir");
    let hub = hub_with(&mock, dir.path(), 64, Duration::from_secs(5), Some(100));
    let error = fetch_with(
        &hub,
        None,
        &mut |_| Ok(super::fetch_name::Fit::Resident),
        &|_| Ok(()),
    )
    .expect_err("refused");
    assert!(error.0.contains("free in"), "{}", error.0);
    assert!(
        mock.resolve_requests().is_empty(),
        "{:?}",
        mock.resolve_requests()
    );
}

#[test]
fn a_repo_with_no_gguf_and_a_label_it_lacks_are_refused_naming_what_it_holds() {
    let mock = MockHub::start(vec![
        file("model.safetensors", b"x"),
        file("config.json", b"{}"),
    ]);
    let dir = tempfile::tempdir().expect("tempdir");
    let hub = hub_with(&mock, dir.path(), 64, Duration::from_secs(5), None);
    let error = fetch_with(
        &hub,
        None,
        &mut |_| Ok(super::fetch_name::Fit::Resident),
        &|_| Ok(()),
    )
    .expect_err("no gguf");
    assert!(error.0.contains("holds no complete GGUF"), "{}", error.0);
    let mock = MockHub::start(vec![file("model-Q4_K_M.gguf", &gguf_file(300, 4))]);
    let hub = hub_with(&mock, dir.path(), 64, Duration::from_secs(5), None);
    let error = fetch_with(
        &hub,
        Some("Q2_K"),
        &mut |_| Ok(super::fetch_name::Fit::Resident),
        &|_| Ok(()),
    )
    .expect_err("no such quantization");
    assert!(error.0.contains("Q4_K_M"), "{}", error.0);
}

#[test]
fn an_interrupted_download_resumes_from_its_partial_instead_of_starting_over() {
    let content = payload(200, 7);
    let mock = MockHub::start(vec![file("model-Q8_0.gguf", &content)]);
    let dir = tempfile::tempdir().expect("tempdir");
    // An earlier run died after 130 bytes: its partial is named by the file's hash.
    let blobs = repo_cache_dir(dir.path(), "org/m").join("blobs");
    std::fs::create_dir_all(&blobs).expect("blobs");
    std::fs::write(
        blobs.join(format!("{}.incomplete", sha256_hex(&content))),
        &content[..130],
    )
    .expect("partial");
    let hub = hub_with(&mock, dir.path(), 64, Duration::from_secs(5), None);
    let done = hub
        .download_and_wait("org/m", "main", &|_| true, &mut |_| {})
        .expect("resumed");
    assert_eq!(
        std::fs::read(done.join("model-Q8_0.gguf")).expect("file"),
        content
    );
    let ranges: Vec<String> = mock
        .resolve_requests()
        .iter()
        .filter_map(|line| line.split("range=").nth(1).map(str::to_string))
        .collect();
    assert_eq!(ranges, vec!["130-193".to_string(), "194-199".to_string()]);
    assert!(
        !blobs
            .join(format!("{}.incomplete", sha256_hex(&content)))
            .exists()
    );
}

fn all_resident() -> impl FnMut(&Path) -> Result<super::fetch_name::Fit, super::config::StartupError>
{
    |_| Ok(super::fetch_name::Fit::Resident)
}

#[test]
fn a_second_resolve_serves_the_cached_file_with_no_request() {
    let q4 = gguf_file(300, 4);
    let mock = MockHub::start(vec![file("model-Q4_K_M.gguf", &q4)]);
    let dir = tempfile::tempdir().expect("tempdir");
    let hub = hub_with(&mock, dir.path(), 64, Duration::from_secs(5), None);
    let first = fetch_with(&hub, None, &mut all_resident(), &|_| Ok(())).expect("fetched");
    let after_first = mock.requests.lock().expect("log").len();
    // A hub nobody answers for: the cache alone must serve.
    let dead = hub_with_endpoint("http://127.0.0.1:9", dir.path());
    let again = fetch_with(&dead, None, &mut all_resident(), &|_| Ok(())).expect("cached");
    assert_eq!(first, again);
    let named =
        fetch_with(&dead, Some("q4_k_m"), &mut all_resident(), &|_| Ok(())).expect("cached");
    assert_eq!(first, named);
    assert_eq!(mock.requests.lock().expect("log").len(), after_first);
}

fn hub_with_endpoint(endpoint: &str, dir: &Path) -> Arc<HfHub> {
    Arc::new(HfHub::with_limits(
        endpoint.to_string(),
        dir.to_path_buf(),
        None,
        64,
        Duration::from_secs(2),
        Box::new(free_disk_bytes),
    ))
}

#[test]
fn a_set_in_a_subdirectory_is_chosen_whole_and_a_set_missing_a_shard_is_never_offered() {
    let mock = MockHub::start(vec![
        file(
            "UD-Q4_K_XL/model-UD-Q4_K_XL-00001-of-00002.gguf",
            &gguf_file(200, 1),
        ),
        file(
            "UD-Q4_K_XL/model-UD-Q4_K_XL-00002-of-00002.gguf",
            &gguf_file(150, 2),
        ),
        // The larger set lacks its second shard: it is not a quantization of this repo.
        file("Q8_0/model-Q8_0-00001-of-00002.gguf", &gguf_file(900, 3)),
        file("model-Q2_K.gguf", &[]),
    ]);
    let dir = tempfile::tempdir().expect("tempdir");
    let hub = hub_with(&mock, dir.path(), 64, Duration::from_secs(5), None);
    let path = fetch_with(&hub, None, &mut all_resident(), &|_| Ok(())).expect("fetched");
    assert_eq!(
        path,
        snapshot(dir.path(), "org/m").join("UD-Q4_K_XL/model-UD-Q4_K_XL-00001-of-00002.gguf")
    );
    assert!(
        path.with_file_name("model-UD-Q4_K_XL-00002-of-00002.gguf")
            .is_file()
    );
    let error = fetch_with(&hub, Some("Q8_0"), &mut all_resident(), &|_| Ok(()))
        .expect_err("Q8_0 is incomplete in the listing");
    assert!(error.0.contains("UD-Q4_K_XL"), "{}", error.0);
    // Cached, the same set is found again with no label, through the same rule.
    let dead = hub_with_endpoint("http://127.0.0.1:9", dir.path());
    assert_eq!(
        fetch_with(&dead, None, &mut all_resident(), &|_| Ok(())).expect("cached"),
        path
    );
}

#[test]
fn a_planner_that_cannot_run_is_an_error_not_a_smaller_quantization() {
    let mock = MockHub::start(vec![
        file("model-Q4_K_M.gguf", &gguf_file(300, 4)),
        file("model-Q8_0.gguf", &gguf_file(600, 8)),
    ]);
    let dir = tempfile::tempdir().expect("tempdir");
    let hub = hub_with(&mock, dir.path(), 64, Duration::from_secs(5), None);
    let error = fetch_with(
        &hub,
        None,
        &mut |_| {
            Err(super::config::StartupError(
                "the plan could not be made".into(),
            ))
        },
        &|_| Ok(()),
    )
    .expect_err("planner error");
    assert!(
        error.0.contains("the plan could not be made"),
        "{}",
        error.0
    );
    assert!(!repo_cache_dir(dir.path(), "org/m").join("blobs").exists());
}

#[test]
fn two_processes_fetching_one_blob_write_it_once() {
    let content = payload(4000, 5);
    let mock = MockHub::start(vec![file("model-Q4_K_M.gguf", &content)]);
    let dir = tempfile::tempdir().expect("tempdir");
    // Two hub bridges over one cache stand for two processes: each holds its own lock handle.
    let run = |hub: Arc<HfHub>| {
        std::thread::spawn(move || hub.download_and_wait("org/m", "main", &|_| true, &mut |_| {}))
    };
    let first = run(hub_with(
        &mock,
        dir.path(),
        32,
        Duration::from_secs(5),
        None,
    ));
    let second = run(hub_with(
        &mock,
        dir.path(),
        32,
        Duration::from_secs(5),
        None,
    ));
    let (first, second) = (
        first.join().expect("thread"),
        second.join().expect("thread"),
    );
    for done in [first, second] {
        let done = done.expect("downloaded");
        assert_eq!(
            std::fs::read(done.join("model-Q4_K_M.gguf")).expect("file"),
            content
        );
    }
    let fetched: u64 = mock
        .resolve_requests()
        .iter()
        .filter_map(|line| line.split("range=").nth(1)?.split_once('-'))
        .map(|(start, end)| {
            end.parse::<u64>().expect("end") + 1 - start.parse::<u64>().expect("start")
        })
        .sum();
    assert_eq!(
        fetched,
        content.len() as u64,
        "the bytes crossed the wire once"
    );
}
