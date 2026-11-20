//! The on-disk tier of the Qwen3.6 prefix store: saved prefixes that outlive a restart and an
//! eviction from pinned host memory.
//!
//! The module is host-only and holds no CUDA type, so its tests run without the card. An entry is
//! a fixed list of byte objects (the DeltaNet state first, then one object per attention layer)
//! under the record envelope of `expert_bank::disk_store`: staged, fsynced and linked under a
//! content name, then published by renaming a manifest that carries the exact tokens, the
//! identity of the load and a CRC32C and SHA256 per object. A read verifies every byte before the
//! caller sees an `Ok`, so a truncated, flipped or foreign entry is a typed miss and never state.
//!
//! One process owns a cache directory (an advisory lock on `LOCK`), one writer thread streams
//! chunks to disk behind the request path, and one index in memory answers lookups. The byte quota
//! covers manifests and objects; the least recently used entry goes first.

use crate::family::expert_bank::{
    disk_store::{
        self, DiskMiss, DiskRoot, FixedState, H32, H64, Identity, Manifest, PageVersion,
        Provenance, RingOrigin, SaveLattice,
    },
    host_tier::{ODirectRing, SECTOR},
};
use sha2::{Digest, Sha256};
use std::{
    collections::HashSet,
    num::{NonZeroU64, NonZeroUsize},
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex, MutexGuard,
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
        mpsc::{self, Receiver, Sender, SyncSender, TrySendError},
    },
    thread::JoinHandle,
    time::{Duration, Instant},
};

/// Entries below this many committed positions are not written: a prefill of fewer tokens costs
/// about what reading the fixed DeltaNet state does.
pub(crate) const MIN_POSITIONS: usize = 1_024;
/// The largest share of the cache directory's free space the default quota takes, as a divisor.
pub(crate) const QUOTA_FREE_DIVISOR: u64 = 4;
/// The default quota never exceeds this many bytes.
pub(crate) const QUOTA_CAP_BYTES: u64 = 64 << 30;
/// Below this quota the default plan keeps the tier off: not even three 128K entries would fit.
pub(crate) const QUOTA_FLOOR_BYTES: u64 = 4 << 30;
/// The smallest quota an operator may set: one 32K entry and room to write the next.
pub(crate) const QUOTA_EXPLICIT_MIN_BYTES: u64 = 1 << 30;
/// One read or write chunk, in bytes. A multiple of the sector.
pub(crate) const CHUNK_BYTES: usize = 8 << 20;
/// Chunks queued between the engine thread and the writer.
const WRITER_QUEUE: usize = 4;

const SCOPE_SEED: &[u8] = b"fifi-kv-scope-v1\0";
const BILLING_SEED: &[u8] = b"fifi-kv-billing-q36\0";
const LOCK_FILE: &str = "LOCK";
const MARKER_FILE: &str = "CACHE.json";

fn sha256(parts: &[&[u8]]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    for part in parts {
        hasher.update((part.len() as u64).to_le_bytes());
        hasher.update(part);
    }
    hasher.finalize().into()
}

/// Which save point wrote an entry; the manifest stores it as `capture_kind`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Capture {
    Tip,
    Periodic,
}

impl Capture {
    fn name(self) -> &'static str {
        match self {
            Capture::Tip => "tip",
            Capture::Periodic => "periodic",
        }
    }

    fn of(name: &str) -> Self {
        if name == "tip" {
            Capture::Tip
        } else {
            Capture::Periodic
        }
    }
}

/// What a cache entry was built under. Two loads that differ in any field never share an entry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct LoadIdentity {
    /// The weights digest of the loader (`PrefixIdentity::weights`).
    pub(crate) weights: String,
    pub(crate) tokenizer: String,
    pub(crate) template: String,
    /// The numeric path of the prefill, as text (the KV format).
    pub(crate) numeric_path: String,
    /// The capacity-independent shape of an entry (layer counts, state sizes, KV geometry).
    pub(crate) layout: String,
}

impl LoadIdentity {
    /// False when any digest is missing: such a load must not read or write the cache.
    pub(crate) fn is_complete(&self) -> bool {
        [&self.weights, &self.tokenizer, &self.template]
            .iter()
            .all(|digest| !digest.is_empty())
            && !self.numeric_path.is_empty()
            && !self.layout.is_empty()
    }

    fn record(&self) -> Identity {
        let digest =
            |tag: &str, text: &str| H64::from_digest(sha256(&[tag.as_bytes(), text.as_bytes()]));
        Identity {
            model_digest: digest("model", &self.weights),
            weight_layout_digest: digest("layout", &self.layout),
            tokenizer_digest: digest("tokenizer", &self.tokenizer),
            template_digest: digest("template", &self.template),
            adapter_id: "qwen36.cuda".to_owned(),
            backend_id: "cuda.single-card".to_owned(),
            snapshot_schema: "qwen36.compact-state.v1".to_owned(),
            payload_abi: "delta-f32.kv-compact.v1".to_owned(),
            numeric_recipe_digest: digest("numeric", &self.numeric_path),
            position_policy_digest: digest("position", "absolute-rope-from-zero"),
            render_mode: "chat".to_owned(),
            kv_layout_digest: digest("kv-layout", &self.layout),
        }
    }
}

/// One entry to write: the exact tokens of the key and the sizes of its objects in order.
pub(crate) struct WriteSpec {
    /// The key: `positions + 1` tokens, the last one pending.
    pub(crate) tokens: Vec<u32>,
    pub(crate) capture: Capture,
    /// Byte length of every object, in the order the chunks arrive.
    pub(crate) object_bytes: Vec<u64>,
}

impl WriteSpec {
    fn positions(&self) -> u64 {
        self.tokens.len() as u64 - 1
    }

    fn payload_bytes(&self) -> u64 {
        self.object_bytes.iter().sum()
    }
}

/// Why a save was not started.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum WriteSkip {
    /// An entry with these exact tokens is already stored (its use order is refreshed).
    Present,
    /// The entry would not fit the quota even with every other entry gone.
    OverQuota { need: u64, quota: u64 },
    /// The entry is too short to repay a disk read.
    Short,
    /// The spec has no tokens or no objects.
    Invalid,
    /// The writer thread is gone.
    WriterDown,
}

impl std::fmt::Display for WriteSkip {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WriteSkip::Present => f.write_str("an entry with these tokens is stored"),
            WriteSkip::OverQuota { need, quota } => {
                write!(f, "the entry needs {need} B, the quota is {quota} B")
            }
            WriteSkip::Short => f.write_str("fewer positions than the minimum"),
            WriteSkip::Invalid => f.write_str("the entry has no tokens or objects"),
            WriteSkip::WriterDown => f.write_str("the writer thread stopped"),
        }
    }
}

/// What the writer thread reports.
#[derive(Debug)]
pub(crate) enum WriteEvent {
    Done {
        id: u64,
        positions: u64,
        bytes: u64,
        ms: f64,
    },
    Failed {
        id: u64,
        reason: String,
    },
}

/// The outcome of offering one chunk to the writer.
pub(crate) enum Push {
    Sent,
    /// The queue is full: the chunk comes back, try again later.
    Full(Vec<u8>),
    /// The writer stopped or the job was dropped: stop sending.
    Dead,
}

enum Command {
    Begin { id: u64, spec: WriteSpec, need: u64 },
    Data { id: u64, bytes: Vec<u8> },
    Abort { id: u64 },
    Stop,
}

/// Receives an entry's bytes as `(object, offset, bytes)` while it is read.
pub(crate) type ObjectSink<'a> = dyn FnMut(usize, u64, &[u8]) -> Result<(), DiskMiss> + 'a;

/// One entry of the in-memory index.
struct Record {
    sha: H64,
    namespace: [u8; 32],
    tokens: Arc<[u32]>,
    capture: Capture,
    /// Objects in order, with their sizes.
    objects: Vec<(H64, u64)>,
    manifest_bytes: u64,
    /// Use order: larger is more recent.
    last_use: u64,
}

struct State {
    records: Vec<Record>,
    clock: u64,
    /// Bytes promised to saves in flight.
    reserved: u64,
    /// The entry a read is streaming; eviction skips it.
    reading: Option<H64>,
    quota: u64,
    /// Entries removed to stay inside the quota since open.
    evicted: u64,
}

impl State {
    /// Bytes on disk: every manifest and each distinct object once.
    fn used(&self) -> u64 {
        let mut seen = HashSet::new();
        let mut bytes = 0u64;
        for record in &self.records {
            bytes += record.manifest_bytes;
            for &(id, size) in &record.objects {
                if seen.insert(id.0) {
                    bytes += size;
                }
            }
        }
        bytes
    }

    fn tick(&mut self) -> u64 {
        self.clock += 1;
        self.clock
    }
}

struct Shared {
    root: DiskRoot,
    namespace: [u8; 32],
    identity: Identity,
    state: Mutex<State>,
}

impl Shared {
    fn lock(&self) -> MutexGuard<'_, State> {
        self.state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
    }

    /// Removes one entry and every object no other entry names.
    fn remove_record(&self, state: &mut State, at: usize) {
        let record = state.records.remove(at);
        self.root.remove_entry(&record.sha);
        for (id, _) in &record.objects {
            let shared = state
                .records
                .iter()
                .any(|other| other.objects.iter().any(|(other_id, _)| other_id == id));
            if !shared {
                self.root.remove_object(id);
            }
        }
    }

    fn evict_for(&self, state: &mut State, need: u64) -> bool {
        loop {
            if state.used() + state.reserved + need <= state.quota {
                return true;
            }
            let victim = state
                .records
                .iter()
                .enumerate()
                .filter(|(_, record)| Some(record.sha) != state.reading)
                .min_by_key(|(_, record)| record.last_use)
                .map(|(at, _)| at);
            match victim {
                Some(at) => {
                    self.remove_record(state, at);
                    state.evicted += 1;
                }
                None => return false,
            }
        }
    }
}

/// What opening a directory found.
#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct OpenReport {
    /// Entries of this load.
    pub(crate) entries: usize,
    /// Entries of another load (kept, evicted by use order like any other).
    pub(crate) foreign: usize,
    /// Entries removed at open: unreadable, corrupt, or missing an object.
    pub(crate) dropped: usize,
    /// Object files removed because no entry named them.
    pub(crate) orphans: usize,
    pub(crate) used_bytes: u64,
    pub(crate) quota_bytes: u64,
}

/// A stored entry that can restore a prompt.
#[derive(Clone, Debug)]
pub(crate) struct Candidate {
    sha: H64,
    /// The key length: positions + 1, the tokens a restore covers.
    pub(crate) key_len: usize,
    pub(crate) capture: Capture,
}

/// What a verified read cost.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ReadStats {
    pub(crate) bytes: u64,
    pub(crate) ms: f64,
}

/// The cache of one directory.
pub(crate) struct KvDisk {
    shared: Arc<Shared>,
    commands: SyncSender<Command>,
    events: Receiver<WriteEvent>,
    writer: Option<JoinHandle<()>>,
    /// One read ring per reader thread.
    rings: Vec<ODirectRing>,
    next_id: u64,
    /// Bytes and milliseconds of the saves that completed, for the write rate.
    done_bytes: AtomicU64,
    done_ms: AtomicU64,
    // Held for the lifetime of the cache: the advisory lock names this process the owner.
    _lock: std::fs::File,
}

fn aligned_leak(bytes: usize) -> &'static mut [u8] {
    let slop = SECTOR as usize;
    let backing = Box::leak(vec![0u8; bytes + slop].into_boxed_slice());
    let base = backing.as_ptr() as usize;
    let skip = (slop - base % slop) % slop;
    &mut backing[skip..skip + bytes]
}

fn credential_scope() -> H32 {
    #[cfg(target_os = "linux")]
    let uid = disk_store::effective_uid();
    #[cfg(not(target_os = "linux"))]
    let uid = 0u32;
    let digest = sha256(&[SCOPE_SEED, &uid.to_le_bytes()]);
    let mut id = [0u8; 16];
    id.copy_from_slice(&digest[..16]);
    H32::from_digest(id)
}

fn billing_engine() -> H64 {
    H64::from_digest(sha256(&[BILLING_SEED]))
}

fn io_err(what: &str, error: impl std::fmt::Display) -> String {
    format!("{what}: {error}")
}

/// Creates `dir` (0700) when missing and takes the exclusive advisory lock on `dir/LOCK`.
fn lock_dir(dir: &Path) -> Result<std::fs::File, String> {
    use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt};
    std::fs::DirBuilder::new()
        .mode(0o700)
        .recursive(true)
        .create(dir)
        .map_err(|e| io_err("creating the cache directory", e))?;
    let file = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .mode(0o600)
        .open(dir.join(LOCK_FILE))
        .map_err(|e| io_err("opening the cache lock", e))?;
    rustix::fs::flock(&file, rustix::fs::FlockOperation::NonBlockingLockExclusive)
        .map_err(|_| "another process holds the cache directory".to_owned())?;
    Ok(file)
}

impl KvDisk {
    /// Opens (or creates) the cache under `dir` for this load. Entries of other loads stay and
    /// age out by use order; entries that fail to parse or lack an object are removed.
    pub(crate) fn open(
        dir: &Path,
        quota_bytes: u64,
        load: &LoadIdentity,
    ) -> Result<(Self, OpenReport), String> {
        if !load.is_complete() {
            return Err(
                "the load identity is incomplete: no cache entry is written or read".into(),
            );
        }
        let lock = lock_dir(dir)?;
        let dir =
            std::fs::canonicalize(dir).map_err(|e| io_err("resolving the cache directory", e))?;
        let root = DiskRoot::open_or_create(&dir, credential_scope(), billing_engine())
            .map_err(|e| format!("opening the cache root: {e}"))?;
        let identity = load.record();
        let namespace = identity.namespace();
        let mut report = OpenReport {
            quota_bytes,
            ..OpenReport::default()
        };
        let (entries, _unknown) = root.list_entries().map_err(|e| e.to_string())?;
        let (objects, _unknown) = root.list_objects().map_err(|e| e.to_string())?;
        let object_sizes: std::collections::HashMap<[u8; 32], u64> =
            objects.iter().map(|(id, _, size)| (id.0, *size)).collect();
        let mut ordered = entries;
        ordered.sort_by_key(|(_, modified, _)| *modified);
        let mut records = Vec::new();
        let mut incompatible = false;
        for (sha, _modified, file_bytes) in ordered {
            match root.read_entry(&sha) {
                Ok(manifest) => {
                    let sizes: Option<Vec<(H64, u64)>> = manifest
                        .dependencies
                        .iter()
                        .map(|dep| {
                            (object_sizes.get(&dep.id.0) == Some(&dep.file_bytes))
                                .then_some((dep.id, dep.file_bytes))
                        })
                        .collect();
                    let Some(objects) = sizes else {
                        root.remove_entry(&sha);
                        report.dropped += 1;
                        continue;
                    };
                    if manifest.namespace.0 == namespace {
                        report.entries += 1;
                    } else {
                        report.foreign += 1;
                    }
                    let clock = records.len() as u64 + 1;
                    records.push(Record {
                        sha,
                        namespace: manifest.namespace.0,
                        tokens: manifest.tokens.clone().into(),
                        capture: Capture::of(&manifest.provenance.capture_kind),
                        objects,
                        manifest_bytes: file_bytes,
                        last_use: clock,
                    });
                }
                Err(DiskMiss::Incompatible(_)) => {
                    // A manifest of another envelope version: its objects cannot be attributed.
                    incompatible = true;
                    report.foreign += 1;
                }
                Err(DiskMiss::Corrupt(_) | DiskMiss::Absent) => {
                    root.remove_entry(&sha);
                    report.dropped += 1;
                }
                Err(DiskMiss::Refused(_)) => {
                    // Unreadable now (permissions, scope): left alone, and its objects with it.
                    incompatible = true;
                    report.foreign += 1;
                }
            }
        }
        if !incompatible {
            let named: HashSet<[u8; 32]> = records
                .iter()
                .flat_map(|record| record.objects.iter().map(|(id, _)| id.0))
                .collect();
            for (id, _, _) in &objects {
                if !named.contains(&id.0) {
                    root.remove_object(id);
                    report.orphans += 1;
                }
            }
        }
        let clock = records.len() as u64;
        let state = State {
            records,
            clock,
            reserved: 0,
            reading: None,
            quota: quota_bytes,
            evicted: 0,
        };
        report.used_bytes = state.used();
        let shared = Arc::new(Shared {
            root,
            namespace,
            identity,
            state: Mutex::new(state),
        });
        let (commands, command_rx) = mpsc::sync_channel(WRITER_QUEUE + 2);
        let (event_tx, events) = mpsc::channel();
        let writer = {
            let shared = Arc::clone(&shared);
            std::thread::Builder::new()
                .name("kv-disk-writer".into())
                .spawn(move || writer_loop(&shared, &command_rx, &event_tx))
                .map_err(|e| io_err("starting the writer", e))?
        };
        let mut rings = Vec::with_capacity(READ_WORKERS);
        for _ in 0..READ_WORKERS {
            rings.push(
                ODirectRing::new(
                    NonZeroUsize::MIN,
                    NonZeroU64::new((CHUNK_BYTES as u64) + SECTOR).expect("a nonzero span"),
                    aligned_leak(CHUNK_BYTES + SECTOR as usize),
                )
                .map_err(|e| format!("the read ring: {e}"))?,
            );
        }
        Ok((
            Self {
                shared,
                commands,
                events,
                writer: Some(writer),
                rings,
                next_id: 0,
                done_bytes: AtomicU64::new(0),
                done_ms: AtomicU64::new(0),
                _lock: lock,
            },
            report,
        ))
    }

    pub(crate) fn quota_bytes(&self) -> u64 {
        self.shared.lock().quota
    }

    /// Entries removed to stay inside the quota since open.
    pub(crate) fn evicted(&self) -> u64 {
        self.shared.lock().evicted
    }

    /// Bytes the stored entries occupy.
    pub(crate) fn used_bytes(&self) -> u64 {
        self.shared.lock().used()
    }

    /// Whether an entry of this load holds exactly these tokens.
    pub(crate) fn has_exact(&self, tokens: &[u32]) -> bool {
        let namespace = self.shared.namespace;
        self.shared
            .lock()
            .records
            .iter()
            .any(|record| record.namespace == namespace && record.tokens.as_ref() == tokens)
    }

    /// Entries of this load.
    pub(crate) fn entry_count(&self) -> usize {
        let namespace = self.shared.namespace;
        self.shared
            .lock()
            .records
            .iter()
            .filter(|record| record.namespace == namespace)
            .count()
    }

    /// The longest stored key that is a strict prefix of `prompt`.
    pub(crate) fn lookup(&self, prompt: &[u32]) -> Option<Candidate> {
        let state = self.shared.lock();
        state
            .records
            .iter()
            .filter(|record| {
                record.namespace == self.shared.namespace
                    && record.tokens.len() < prompt.len()
                    && prompt.starts_with(&record.tokens)
            })
            .max_by_key(|record| record.tokens.len())
            .map(|record| Candidate {
                sha: record.sha,
                key_len: record.tokens.len(),
                capture: record.capture,
            })
    }

    /// Streams the entry into `sink(object, offset, bytes)` in object order, every byte verified
    /// against the manifest's CRC32C and SHA256 before the call returns `Ok`. The sink's bytes
    /// are provisional until then: a caller discards them on `Err`. An entry that fails its own
    /// integrity is removed; a transient refusal keeps it.
    pub(crate) fn read(
        &mut self,
        candidate: &Candidate,
        prompt: &[u32],
        object_bytes: &[u64],
        sink: &mut ObjectSink<'_>,
    ) -> Result<ReadStats, DiskMiss> {
        let began = Instant::now();
        {
            let mut state = self.shared.lock();
            state.reading = Some(candidate.sha);
        }
        let outcome = self.read_inner(candidate, prompt, object_bytes, sink);
        let mut state = self.shared.lock();
        state.reading = None;
        match &outcome {
            Ok(_) => {
                let stamp = state.tick();
                if let Some(record) = state
                    .records
                    .iter_mut()
                    .find(|record| record.sha == candidate.sha)
                {
                    record.last_use = stamp;
                }
                drop(state);
                self.shared.root.touch_entry(&candidate.sha);
            }
            Err(DiskMiss::Corrupt(_) | DiskMiss::Absent) => {
                if let Some(at) = state
                    .records
                    .iter()
                    .position(|record| record.sha == candidate.sha)
                {
                    self.shared.remove_record(&mut state, at);
                }
            }
            Err(_) => {}
        }
        outcome.map(|bytes| ReadStats {
            bytes,
            ms: began.elapsed().as_secs_f64() * 1000.0,
        })
    }

    fn read_inner(
        &mut self,
        candidate: &Candidate,
        prompt: &[u32],
        object_bytes: &[u64],
        sink: &mut ObjectSink<'_>,
    ) -> Result<u64, DiskMiss> {
        let manifest = self.shared.root.read_entry(&candidate.sha)?;
        if manifest.namespace.0 != self.shared.namespace {
            return Err(DiskMiss::Incompatible(
                "the entry belongs to another load".into(),
            ));
        }
        let key_len = candidate.key_len;
        if prompt.len() <= key_len || !manifest.tokens_equal(&prompt[..key_len]) {
            return Err(DiskMiss::Corrupt(
                "the stored tokens are not a prefix of the prompt".into(),
            ));
        }
        if manifest.dependencies.len() != object_bytes.len()
            || manifest
                .dependencies
                .iter()
                .zip(object_bytes)
                .any(|(dep, &bytes)| dep.file_bytes != bytes)
        {
            return Err(DiskMiss::Incompatible(
                "the entry's objects differ from this layout".into(),
            ));
        }
        let deps = &manifest.dependencies;
        let workers = self.rings.len().min(deps.len());
        let next = AtomicUsize::new(0);
        let stop = AtomicBool::new(false);
        let (tx, rx) = mpsc::sync_channel::<Piece>(2 * READ_WORKERS + 4);
        let root = &self.shared.root;
        let mut failure: Option<DiskMiss> = None;
        std::thread::scope(|scope| {
            for ring in self.rings.iter_mut().take(workers) {
                let tx = tx.clone();
                let (next, stop) = (&next, &stop);
                scope.spawn(move || read_objects(root, deps, ring, next, stop, &tx));
            }
            drop(tx);
            // The sink runs on this thread, in the order chunks arrive: out of order across
            // objects, in order within one.
            for piece in rx {
                match piece {
                    Piece::Chunk(object, offset, bytes) => {
                        if failure.is_none()
                            && let Err(miss) = sink(object, offset, &bytes)
                        {
                            failure = Some(miss);
                            stop.store(true, Ordering::Relaxed);
                        }
                    }
                    Piece::Done(_, Err(miss)) => {
                        failure.get_or_insert(miss);
                        stop.store(true, Ordering::Relaxed);
                    }
                    Piece::Done(_, Ok(())) => {}
                }
            }
        });
        match failure {
            Some(miss) => Err(miss),
            None => Ok(deps.iter().map(|dep| dep.file_bytes).sum()),
        }
    }

    /// Starts a save, or says why it is not worth one. The entry's bytes are reserved against the
    /// quota first, evicting the least recently used entries to make room.
    pub(crate) fn begin_write(&mut self, spec: WriteSpec) -> Result<WriteJob, WriteSkip> {
        if spec.tokens.len() < 2 || spec.object_bytes.is_empty() || spec.object_bytes.contains(&0) {
            return Err(WriteSkip::Invalid);
        }
        if (spec.positions() as usize) < MIN_POSITIONS {
            return Err(WriteSkip::Short);
        }
        let need = spec.payload_bytes() + 4_096 + 12 * spec.tokens.len() as u64;
        {
            let mut state = self.shared.lock();
            let namespace = self.shared.namespace;
            let stamp = state.tick();
            if let Some(record) = state.records.iter_mut().find(|record| {
                record.namespace == namespace && record.tokens.as_ref() == spec.tokens.as_slice()
            }) {
                record.last_use = stamp;
                let sha = record.sha;
                drop(state);
                self.shared.root.touch_entry(&sha);
                return Err(WriteSkip::Present);
            }
            if need > state.quota {
                return Err(WriteSkip::OverQuota {
                    need,
                    quota: state.quota,
                });
            }
            if !self.shared.evict_for(&mut state, need) {
                return Err(WriteSkip::OverQuota {
                    need,
                    quota: state.quota,
                });
            }
            state.reserved += need;
        }
        self.next_id += 1;
        let id = self.next_id;
        let positions = spec.positions();
        if self
            .commands
            .send(Command::Begin { id, spec, need })
            .is_err()
        {
            self.shared.lock().reserved -= need;
            return Err(WriteSkip::WriterDown);
        }
        Ok(WriteJob { id, positions })
    }

    /// Offers one chunk of the entry's bytes (objects back to back) to the writer.
    pub(crate) fn push(&self, job: &WriteJob, bytes: Vec<u8>) -> Push {
        match self.commands.try_send(Command::Data { id: job.id, bytes }) {
            Ok(()) => Push::Sent,
            Err(TrySendError::Full(Command::Data { bytes, .. })) => Push::Full(bytes),
            Err(_) => Push::Dead,
        }
    }

    /// Cancels a save: staged bytes are discarded and its reservation is released.
    pub(crate) fn abort(&self, job: &WriteJob) {
        let _ = self.commands.try_send(Command::Abort { id: job.id });
    }

    /// The writer's finished and failed saves since the last call.
    pub(crate) fn poll(&self) -> Vec<WriteEvent> {
        let events: Vec<WriteEvent> = self.events.try_iter().collect();
        for event in &events {
            if let WriteEvent::Done { bytes, ms, .. } = event {
                self.done_bytes.fetch_add(*bytes, Ordering::Relaxed);
                self.done_ms
                    .fetch_add(ms.round().max(1.0) as u64, Ordering::Relaxed);
            }
        }
        events
    }

    /// The measured write rate in bytes per second, from the saves that completed: their bytes
    /// over the time from their start to their report, syncs and publication included. `None`
    /// until 16 MiB have completed. Aborted saves do not count.
    pub(crate) fn write_rate(&self) -> Option<f64> {
        let bytes = self.done_bytes.load(Ordering::Relaxed);
        let ms = self.done_ms.load(Ordering::Relaxed);
        (bytes >= 16 << 20 && ms > 0).then(|| bytes as f64 / (ms as f64 / 1e3))
    }

    /// Whether no save is in flight.
    pub(crate) fn is_idle(&self) -> bool {
        self.shared.lock().reserved == 0
    }

    /// Waits until the writer has finished every save started so far, or `timeout` passes.
    pub(crate) fn wait_idle(&self, timeout: Duration) -> bool {
        let deadline = Instant::now() + timeout;
        loop {
            if self.shared.lock().reserved == 0 {
                return true;
            }
            if Instant::now() >= deadline {
                return false;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
    }
}

impl Drop for KvDisk {
    fn drop(&mut self) {
        let _ = self.commands.send(Command::Stop);
        if let Some(writer) = self.writer.take() {
            let _ = writer.join();
        }
    }
}

/// The reader threads that verify one entry's objects at once, each with its own ring.
const READ_WORKERS: usize = 4;

/// What a reader thread sends to the thread that owns the sink.
enum Piece {
    Chunk(usize, u64, Vec<u8>),
    Done(usize, Result<(), DiskMiss>),
}

/// Reads objects `next` hands out, one at a time, verifying each against its manifest entry and
/// sending its chunks on. Stops at once when `stop` is set.
fn read_objects(
    root: &DiskRoot,
    deps: &[disk_store::Dependency],
    ring: &mut ODirectRing,
    next: &AtomicUsize,
    stop: &AtomicBool,
    tx: &SyncSender<Piece>,
) {
    loop {
        let index = next.fetch_add(1, Ordering::Relaxed);
        if index >= deps.len() || stop.load(Ordering::Relaxed) {
            return;
        }
        let outcome = match ring.acquire() {
            None => Err(DiskMiss::Refused("the read ring is busy".into())),
            Some(slot) => {
                let outcome = disk_store::read_object_verified(
                    root,
                    &deps[index],
                    ring,
                    slot,
                    &mut |offset, bytes| {
                        if stop.load(Ordering::Relaxed) {
                            return Err(DiskMiss::Refused("the read was stopped".into()));
                        }
                        tx.send(Piece::Chunk(index, offset, bytes.to_vec()))
                            .map_err(|_| DiskMiss::Refused("the reader's consumer is gone".into()))
                    },
                );
                let _ = ring.abandon(slot);
                outcome
            }
        };
        if tx.send(Piece::Done(index, outcome)).is_err() {
            return;
        }
    }
}

/// A save in flight.
#[derive(Debug)]
pub(crate) struct WriteJob {
    id: u64,
    positions: u64,
}

impl WriteJob {
    pub(crate) fn id(&self) -> u64 {
        self.id
    }

    pub(crate) fn positions(&self) -> u64 {
        self.positions
    }
}

struct Current {
    id: u64,
    spec: WriteSpec,
    need: u64,
    installed: Vec<disk_store::Dependency>,
    writer: Option<disk_store::ObjectWriter>,
    written_in_object: u64,
    began: Instant,
}

fn writer_loop(shared: &Shared, commands: &Receiver<Command>, events: &Sender<WriteEvent>) {
    let mut current: Option<Current> = None;
    while let Ok(command) = commands.recv() {
        match command {
            Command::Stop => break,
            Command::Begin { id, spec, need } => {
                if let Some(old) = current.take() {
                    release(shared, old);
                }
                current = Some(Current {
                    id,
                    spec,
                    need,
                    installed: Vec::new(),
                    writer: None,
                    written_in_object: 0,
                    began: Instant::now(),
                });
            }
            Command::Abort { id } => {
                if current.as_ref().is_some_and(|c| c.id == id)
                    && let Some(old) = current.take()
                {
                    release(shared, old);
                }
            }
            Command::Data { id, bytes } => {
                let Some(mut job) = current.take_if(|job| job.id == id) else {
                    continue;
                };
                match feed(shared, &mut job, &bytes) {
                    Ok(None) => current = Some(job),
                    Ok(Some(done)) => {
                        let _ = events.send(done);
                        release(shared, job);
                    }
                    Err(reason) => {
                        let _ = events.send(WriteEvent::Failed { id, reason });
                        release(shared, job);
                    }
                }
            }
        }
    }
    if let Some(old) = current.take() {
        release(shared, old);
    }
}

/// Drops a job's reservation and any objects it installed that no entry names.
fn release(shared: &Shared, job: Current) {
    let mut state = shared.lock();
    state.reserved = state.reserved.saturating_sub(job.need);
    for dep in &job.installed {
        let named = state
            .records
            .iter()
            .any(|record| record.objects.iter().any(|(id, _)| *id == dep.id));
        if !named {
            shared.root.remove_object(&dep.id);
        }
    }
}

/// Appends `bytes` across object boundaries. Returns the completion event when the last object
/// has been installed and the manifest published.
fn feed(
    shared: &Shared,
    job: &mut Current,
    mut bytes: &[u8],
) -> Result<Option<WriteEvent>, String> {
    while !bytes.is_empty() {
        let index = job.installed.len();
        let Some(&size) = job.spec.object_bytes.get(index) else {
            return Err("more bytes than the entry holds".into());
        };
        if job.writer.is_none() {
            job.writer = Some(shared.root.begin_object().map_err(|e| e.to_string())?);
            job.written_in_object = 0;
        }
        let take = usize::try_from(size - job.written_in_object)
            .unwrap_or(usize::MAX)
            .min(bytes.len());
        let writer = job.writer.as_mut().expect("the object writer exists");
        writer.write(&bytes[..take]).map_err(|e| e.to_string())?;
        job.written_in_object += take as u64;
        bytes = &bytes[take..];
        if job.written_in_object == size {
            let writer = job.writer.take().expect("the object writer exists");
            job.installed
                .push(writer.finish(&shared.root).map_err(|e| e.to_string())?);
        }
    }
    if job.installed.len() < job.spec.object_bytes.len() {
        return Ok(None);
    }
    shared.root.sync_objects().map_err(|e| e.to_string())?;
    let manifest = build_manifest(shared, job)?;
    let manifest_bytes = manifest.canonical_bytes().len() as u64;
    let sha = shared
        .root
        .publish_manifest(&manifest)
        .map_err(|e| e.to_string())?;
    let objects: Vec<(H64, u64)> = job
        .installed
        .iter()
        .map(|dep| (dep.id, dep.file_bytes))
        .collect();
    let bytes_total = manifest_bytes + objects.iter().map(|(_, size)| size).sum::<u64>();
    let mut state = shared.lock();
    let stamp = state.tick();
    // A tip that extends an older tip of the same load replaces it, as a restored tip is
    // replaced in memory: the next turn of that conversation reads the longer one.
    if job.spec.capture == Capture::Tip {
        let stale: Vec<usize> = state
            .records
            .iter()
            .enumerate()
            .filter(|(_, record)| {
                record.namespace == shared.namespace
                    && record.capture == Capture::Tip
                    && record.tokens.len() < job.spec.tokens.len()
                    && job.spec.tokens.starts_with(&record.tokens)
                    && Some(record.sha) != state.reading
            })
            .map(|(at, _)| at)
            .collect();
        // Records are indexed before removal shifts them: remove from the back.
        state.records.push(Record {
            sha,
            namespace: shared.namespace,
            tokens: job.spec.tokens.clone().into(),
            capture: job.spec.capture,
            objects,
            manifest_bytes,
            last_use: stamp,
        });
        for at in stale.into_iter().rev() {
            shared.remove_record(&mut state, at);
        }
    } else {
        state.records.push(Record {
            sha,
            namespace: shared.namespace,
            tokens: job.spec.tokens.clone().into(),
            capture: job.spec.capture,
            objects,
            manifest_bytes,
            last_use: stamp,
        });
    }
    Ok(Some(WriteEvent::Done {
        id: job.id,
        positions: job.spec.positions(),
        bytes: bytes_total,
        ms: job.began.elapsed().as_secs_f64() * 1000.0,
    }))
}

fn build_manifest(shared: &Shared, job: &Current) -> Result<Manifest, String> {
    let spec = &job.spec;
    let p = spec.positions();
    let q = p + 1;
    let pending_token = *spec.tokens.last().ok_or("an entry without tokens")?;
    let unix_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(0));
    let mut manifest = Manifest {
        format: "fifi-kv-v3".to_owned(),
        format_version: 1,
        entry_id: disk_store::fresh_entry_id(),
        engine_id: billing_engine(),
        credential_scope: shared.root.credential_scope(),
        namespace: H64::from_digest(shared.namespace),
        identity: shared.identity.clone(),
        created_unix_ms: unix_ms,
        p,
        q,
        pending_token,
        tokens: spec.tokens.clone(),
        block_size: 256,
        block_hashes: Vec::new(),
        provenance: Provenance {
            capture_kind: spec.capture.name().to_owned(),
            numeric_recipe_digest: shared.identity.numeric_recipe_digest,
            actual_fed_cuts: vec![p],
            save_lattice: SaveLattice {
                origin_q: q,
                stride_q: 256,
                selected_q: vec![q],
                skipped_q: Vec::new(),
            },
        },
        dependencies: job.installed.clone(),
        page_versions: Vec::new(),
        fixed_state: Vec::new(),
    };
    for block in 0..q.div_ceil(256) as usize {
        let hash = manifest.block_hash(block);
        manifest.block_hashes.push(H64::from_digest(hash));
    }
    let delta = job.installed.first().ok_or("an entry without objects")?;
    manifest.fixed_state.push(FixedState {
        layer: 0,
        kind: "deltanet".to_owned(),
        dep: delta.id,
        byte_offset: 0,
        byte_length: delta.file_bytes,
        at_p: p,
        schema: "qwen36.delta-f32.v1".to_owned(),
    });
    for (layer, dep) in job.installed.iter().skip(1).enumerate() {
        manifest.page_versions.push(PageVersion {
            layer: u32::try_from(layer).map_err(|_| "too many layers".to_owned())?,
            dep: dep.id,
            byte_offset: 0,
            byte_length: dep.file_bytes,
            first_pos: 0,
            valid_rows: p,
            page_rows: 256,
            tail_rows: u32::try_from(1 + (p - 1) % 256).map_err(|_| "tail rows".to_owned())?,
            version: disk_store::fresh_entry_id(),
            ring_origin: RingOrigin::Null(()),
        });
    }
    Ok(manifest)
}

/// What a purge removed.
#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct PurgeReport {
    pub(crate) entries: usize,
    pub(crate) bytes: u64,
}

/// Empties a cache directory. It refuses a directory that holds anything besides the cache's own
/// files (no `CACHE.json` and not empty) and one another process holds open.
pub(crate) fn purge(dir: &Path) -> Result<PurgeReport, String> {
    if !dir.exists() {
        return Ok(PurgeReport::default());
    }
    let marker = dir.join(MARKER_FILE);
    let empty = std::fs::read_dir(dir)
        .map_err(|e| io_err("reading the cache directory", e))?
        .next()
        .is_none();
    if !marker.exists() && !dir.join(LOCK_FILE).exists() && !empty {
        return Err(format!(
            "{} holds no {MARKER_FILE}: it is not a cache directory and is left alone",
            dir.display()
        ));
    }
    let _lock = lock_dir(dir)?;
    let mut report = PurgeReport::default();
    for name in ["entries", "objects", "staging"] {
        let sub = dir.join(name);
        let Ok(listing) = std::fs::read_dir(&sub) else {
            continue;
        };
        for item in listing.flatten() {
            if let Ok(meta) = item.metadata()
                && meta.is_file()
            {
                report.bytes += meta.len();
                if name == "entries" {
                    report.entries += 1;
                }
            }
        }
        std::fs::remove_dir_all(&sub).map_err(|e| io_err("removing a cache directory", e))?;
    }
    if marker.exists() {
        std::fs::remove_file(&marker).map_err(|e| io_err("removing the cache marker", e))?;
    }
    // The lock is released when `_lock` drops; the file goes first, so the directory is empty.
    let _ = std::fs::remove_file(dir.join(LOCK_FILE));
    Ok(report)
}

/// The settings of the tier as the operator wrote them, before the host is probed.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Wanted {
    /// The directory, or `None` for the default location.
    pub(crate) dir: Option<PathBuf>,
    /// The quota in bytes, or `None` for the share of the free space.
    pub(crate) quota_bytes: Option<u64>,
}

/// A resolved directory and quota.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Resolved {
    pub(crate) dir: PathBuf,
    pub(crate) quota_bytes: u64,
}

/// The default directory under the user's cache directory.
pub(crate) fn default_dir(xdg_cache_home: Option<&str>, home: Option<&str>) -> Option<PathBuf> {
    let base = match (
        xdg_cache_home.filter(|v| !v.is_empty()),
        home.filter(|v| !v.is_empty()),
    ) {
        (Some(xdg), _) if Path::new(xdg).is_absolute() => PathBuf::from(xdg),
        (_, Some(home)) if Path::new(home).is_absolute() => Path::new(home).join(".cache"),
        _ => return None,
    };
    Some(base.join("fifi").join("kv"))
}

/// The default quota for `free_bytes` of free space: a quarter of it, at most 64 GiB, and `None`
/// (the tier stays off) when that is under 4 GiB.
pub(crate) fn default_quota(free_bytes: u64) -> Option<u64> {
    let quota = (free_bytes / QUOTA_FREE_DIVISOR).min(QUOTA_CAP_BYTES);
    (quota >= QUOTA_FLOOR_BYTES).then_some(quota)
}

/// The free bytes of the filesystem that holds `dir`, probed through the nearest existing
/// ancestor.
pub(crate) fn free_bytes(dir: &Path) -> Option<u64> {
    let mut probe = dir;
    while !probe.exists() {
        probe = probe.parent()?;
    }
    let stat = rustix::fs::statvfs(probe).ok()?;
    stat.f_bavail.checked_mul(stat.f_frsize)
}

/// Resolves the wanted directory and quota against the host. `Err` names why the tier stays off.
pub(crate) fn resolve(
    wanted: &Wanted,
    xdg_cache_home: Option<&str>,
    home: Option<&str>,
    free: impl Fn(&Path) -> Option<u64>,
) -> Result<Resolved, String> {
    let dir = match &wanted.dir {
        Some(dir) => dir.clone(),
        None => default_dir(xdg_cache_home, home)
            .ok_or("no cache directory: neither XDG_CACHE_HOME nor HOME is an absolute path")?,
    };
    let quota_bytes = match wanted.quota_bytes {
        Some(quota) => quota,
        None => {
            let free = free(&dir).ok_or("the free space of the cache directory is unknown")?;
            default_quota(free).ok_or_else(|| {
                format!(
                    "{free} B free is under the floor: a quarter of it is below {QUOTA_FLOOR_BYTES} B"
                )
            })?
        }
    };
    Ok(Resolved { dir, quota_bytes })
}

/// The number of entries a pinned store of `retained` slots may hold before the disk tier
/// starts to take its least recently used ones: all but one of the slots (a pool of one or two
/// uses them all), so the eviction a save causes never waits for a write.
pub(crate) fn watermark(retained: usize) -> usize {
    if retained <= 2 {
        retained.max(1)
    } else {
        retained - 1
    }
}

/// How many of the least recently used entries to write now, with `entries` stored: those the
/// next saves would push out once the count reaches the watermark.
pub(crate) fn spill_candidates(entries: usize, retained: usize) -> usize {
    (entries + 1)
        .saturating_sub(watermark(retained))
        .min(entries)
}

/// The write rate assumed before one is measured, in bytes per second: a busy shared disk,
/// measured at 25 to 40 MB/s for a verified write during a build.
pub(crate) const ASSUMED_WRITE_RATE: f64 = 20e6;

/// How long a graceful stop waits for `remaining` bytes of tip saves: the time they need at the
/// measured write rate (or the assumed one) three times over, which covers the copy out of
/// memory and the syncs the busy-time rate leaves out, and 10 s more, between 15 s and 300 s.
pub(crate) fn drain_bound(remaining: u64, rate: Option<f64>) -> Duration {
    let rate = rate
        .filter(|rate| rate.is_finite() && *rate > 0.0)
        .unwrap_or(ASSUMED_WRITE_RATE);
    let seconds = remaining as f64 / rate * 3.0 + 10.0;
    Duration::from_secs_f64(seconds.clamp(15.0, 300.0))
}

/// The environment names that override the `kv_cache` member.
pub(crate) const DIR_ENV: &str = "Q36_KV_CACHE_DIR";
pub(crate) const QUOTA_ENV: &str = "Q36_KV_CACHE_QUOTA_BYTES";
pub(crate) const SWITCH_ENV: &str = "Q36_KV_CACHE";

/// The raw values of the environment overrides.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct KvEnv {
    pub(crate) dir: Option<String>,
    pub(crate) quota: Option<String>,
    pub(crate) switch: Option<String>,
    pub(crate) xdg_cache_home: Option<String>,
    pub(crate) home: Option<String>,
}

impl KvEnv {
    pub(crate) fn from_process() -> Self {
        let get = |name: &str| std::env::var(name).ok();
        Self {
            dir: get(DIR_ENV),
            quota: get(QUOTA_ENV),
            switch: get(SWITCH_ENV),
            xdg_cache_home: get("XDG_CACHE_HOME"),
            home: get("HOME"),
        }
    }
}

/// The directory `--purge-kv-cache` empties for one engine: the environment's, else the config
/// member's, else the default location. `None` when none can be named.
pub(crate) fn purge_dir(
    options: Option<&crate::family::qwen36::settings::KvCacheOptions>,
    env: &KvEnv,
) -> Result<Option<PathBuf>, String> {
    if let Some(dir) = env.dir.as_deref() {
        return if Path::new(dir).is_absolute() {
            Ok(Some(PathBuf::from(dir)))
        } else {
            Err(format!("{DIR_ENV} must be an absolute path, got {dir:?}"))
        };
    }
    Ok(options
        .and_then(|options| options.dir.clone())
        .or_else(|| default_dir(env.xdg_cache_home.as_deref(), env.home.as_deref())))
}

/// What the tier will do at start.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Plan {
    /// Off, and why (one start line says it).
    Off(String),
    /// On at this directory and quota; `origin` says what asked for it.
    On {
        resolved: Resolved,
        origin: &'static str,
    },
}

/// Decides whether the tier runs and where, from the config member, the environment and the
/// build's default. The environment wins over the config. A malformed environment value is an
/// error (a typo must not silently turn the cache off or point it elsewhere).
pub(crate) fn plan(
    config: Option<&crate::family::qwen36::settings::KvCacheSettings>,
    env: &KvEnv,
    default_on: bool,
    free: impl Fn(&Path) -> Option<u64>,
) -> Result<Plan, String> {
    let switch = match env.switch.as_deref().map(str::trim) {
        None => None,
        Some("0" | "off" | "false") => Some(false),
        Some("1" | "on" | "true") => Some(true),
        Some(other) => {
            return Err(format!(
                "{SWITCH_ENV} must be 0, off, 1 or on, got {other:?}"
            ));
        }
    };
    let env_dir = match env.dir.as_deref() {
        None => None,
        Some(dir) if Path::new(dir).is_absolute() => Some(PathBuf::from(dir)),
        Some(dir) => return Err(format!("{DIR_ENV} must be an absolute path, got {dir:?}")),
    };
    let env_quota = match env.quota.as_deref() {
        None => None,
        Some(text) => match text.trim().parse::<u64>() {
            Ok(quota) if quota >= QUOTA_EXPLICIT_MIN_BYTES => Some(quota),
            _ => {
                return Err(format!(
                    "{QUOTA_ENV} must be a byte count of at least {QUOTA_EXPLICIT_MIN_BYTES}, got {text:?}"
                ));
            }
        },
    };
    let config_enabled = config.and_then(|c| c.enabled);
    let config_dir = config.and_then(|c| c.dir.clone());
    let (wanted, origin) = if switch == Some(false) {
        return Ok(Plan::Off(format!("{SWITCH_ENV} is off")));
    } else if switch == Some(true) || env_dir.is_some() {
        (true, "environment")
    } else if config_enabled == Some(false) {
        return Ok(Plan::Off("kv_cache.enabled is false".into()));
    } else if config_enabled == Some(true) || config_dir.is_some() {
        (true, "config")
    } else if default_on {
        (true, "default")
    } else {
        (false, "")
    };
    if !wanted {
        return Ok(Plan::Off(
            "not configured: set kv_cache.dir in the qwen36 object or Q36_KV_CACHE_DIR".into(),
        ));
    }
    let wanted = Wanted {
        dir: env_dir.or(config_dir),
        quota_bytes: env_quota.or(config.and_then(|c| c.quota_bytes)),
    };
    match resolve(
        &wanted,
        env.xdg_cache_home.as_deref(),
        env.home.as_deref(),
        free,
    ) {
        Ok(resolved) => Ok(Plan::On { resolved, origin }),
        Err(reason) => Ok(Plan::Off(reason)),
    }
}

#[cfg(test)]
#[path = "kv_cache_tests.rs"]
mod tests;
