//! Host side of the verify-window cost harness: statistics, identities and transfer accounting.
//!
//! The device harness (`qwen36/cuda/window_bench_device_tests.rs`) measures what the eager verify
//! window costs against captured single-row decode. Everything that does not need a device lives
//! here so that it runs and is tested on the host: medians, the SHA-256 identities of the
//! binary, the token ids and the pack, the byte and call arithmetic of the state copies and the
//! logits read-back, and the shape of the JSON record.
//!
//! **Why the bytes and calls are computed and not counted.** A counter in the serving path would
//! add code to a hot path to answer a measurement question. Every transfer of an eager window is
//! fixed by sizes the harness can read from the state layout and the logits width, so the
//! harness computes the totals from those and cross-checks them against the real exported state
//! (the exported planes must add up to the computed bytes). No counter exists in serving code. The
//! price: the call counts are those of the copy loops as read in `state/export.rs`, so a change
//! to those loops must change this arithmetic, and the cross-check catches a changed byte total
//! but not a changed call count. `export_calls_follow_the_copy_loops` pins the loop structure.
//!
//! **What the record can and cannot attribute.** The host wall times inside the window (export,
//! each row, import, each replayed row) come from clock reads at marks in the eager window,
//! compiled in test builds only. They are wall times between marks: a row's time holds its launch,
//! the wait for the device and the logits read-back, so it is not a GPU time. The GPU time of a
//! row is the step's in-graph event pair. Neither separates a row's read-back from its compute:
//! the tooling has no event between them, so that split is not in the record.

use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    fs, io,
    ops::Add,
    path::{Path, PathBuf},
};

/// The schema number of the JSON record.
pub(crate) const RECORD_SCHEMA: u64 = 1;
/// The largest file whose content is hashed into the pack manifest; larger files (the weight
/// shards) are identified by the content-addressed name of their cache blob and by their size.
const HASHED_FILE_LIMIT: u64 = 64 << 20;

/// The middle value of `values`: for an even count, the mean of the two middle ones. `None` for
/// an empty slice. NaN values are not accepted.
pub(crate) fn median(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let mid = sorted.len() / 2;
    Some(if sorted.len() % 2 == 1 {
        sorted[mid]
    } else {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    })
}

/// Count, median, minimum and maximum of a series, as a JSON object; `null` for no values.
pub(crate) fn summary(values: &[f64]) -> Value {
    match (
        median(values),
        values.iter().copied().reduce(f64::min),
        values.iter().copied().reduce(f64::max),
    ) {
        (Some(median), Some(min), Some(max)) => {
            json!({"n": values.len(), "median": median, "min": min, "max": max})
        }
        _ => Value::Null,
    }
}

pub(crate) fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// SHA-256 of a file, read in pieces.
pub(crate) fn sha256_file(path: &Path) -> io::Result<String> {
    use io::Read;
    let mut file = fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 1 << 20];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            return Ok(format!("{:x}", hasher.finalize()));
        }
        hasher.update(&buffer[..read]);
    }
}

/// SHA-256 of the little-endian u32 packing of `ids`: the identity of a token stream.
pub(crate) fn ids_sha256(ids: &[u32]) -> String {
    let bytes: Vec<u8> = ids.iter().flat_map(|id| id.to_le_bytes()).collect();
    sha256_hex(&bytes)
}

/// One file of a pack directory.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PackFile {
    pub(crate) name: String,
    pub(crate) bytes: u64,
    /// The file name the link points to, for a link (a cache blob named by its content hash).
    pub(crate) blob_name: Option<String>,
    /// SHA-256 of the content, for a file up to 64 MiB.
    pub(crate) content_sha256: Option<String>,
}

/// The identity of a pack directory: its files, with sizes and the strongest hash the harness
/// has cheaply, and one digest over all of them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PackManifest {
    pub(crate) dir: PathBuf,
    pub(crate) files: Vec<PackFile>,
    pub(crate) manifest_sha256: String,
}

impl PackManifest {
    /// Reads the directory. A shard larger than 64 MiB is not hashed here (the run plan verifies
    /// the shards against their blob names before the window); its entry keeps the blob name and
    /// the size, so a swapped shard changes the digest.
    pub(crate) fn read(dir: &Path) -> io::Result<Self> {
        let mut files = Vec::new();
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            let meta = fs::metadata(&path)?;
            if !meta.is_file() {
                continue;
            }
            let blob_name = fs::read_link(&path).ok().and_then(|target| {
                target
                    .file_name()
                    .map(|name| name.to_string_lossy().into_owned())
            });
            let content_sha256 = if meta.len() <= HASHED_FILE_LIMIT {
                Some(sha256_file(&path)?)
            } else {
                None
            };
            files.push(PackFile {
                name: entry.file_name().to_string_lossy().into_owned(),
                bytes: meta.len(),
                blob_name,
                content_sha256,
            });
        }
        files.sort_by(|a, b| a.name.cmp(&b.name));
        let canonical: String = files
            .iter()
            .map(|file| {
                let strongest = file
                    .content_sha256
                    .as_deref()
                    .or(file.blob_name.as_deref())
                    .unwrap_or("-");
                format!("{}\t{}\t{strongest}\n", file.name, file.bytes)
            })
            .collect();
        Ok(Self {
            dir: dir.to_path_buf(),
            files,
            manifest_sha256: sha256_hex(canonical.as_bytes()),
        })
    }

    pub(crate) fn to_json(&self) -> Value {
        json!({
            "dir": self.dir.display().to_string(),
            "manifest_sha256": self.manifest_sha256,
            "files": self.files.iter().map(|file| json!({
                "name": file.name,
                "bytes": file.bytes,
                "blob_name": file.blob_name,
                "content_sha256": file.content_sha256,
            })).collect::<Vec<_>>(),
        })
    }
}

/// Bytes and calls of the copies between host and device that one part of a window makes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Transfer {
    pub(crate) d2h_bytes: u64,
    pub(crate) d2h_calls: u64,
    pub(crate) h2d_bytes: u64,
    pub(crate) h2d_calls: u64,
}

impl Add for Transfer {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            d2h_bytes: self.d2h_bytes + other.d2h_bytes,
            d2h_calls: self.d2h_calls + other.d2h_calls,
            h2d_bytes: self.h2d_bytes + other.h2d_bytes,
            h2d_calls: self.h2d_calls + other.h2d_calls,
        }
    }
}

impl Transfer {
    pub(crate) fn to_json(self) -> Value {
        json!({
            "d2h_bytes": self.d2h_bytes, "d2h_calls": self.d2h_calls,
            "h2d_bytes": self.h2d_bytes, "h2d_calls": self.h2d_calls,
        })
    }
}

/// What the state copies move at one committed length, as the copy loops of `state/export.rs`
/// see it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct StateShape {
    /// DeltaNet layers; each copies its recurrent block and its convolution block.
    pub(crate) delta_layers: usize,
    pub(crate) recurrent_bytes: u64,
    pub(crate) conv_bytes: u64,
    /// For each attention layer, for each plane, the byte length of each copy run that is not
    /// empty (`prefix_ranges` of the layout at the committed length).
    pub(crate) kv_runs: Vec<Vec<Vec<u64>>>,
    /// The words block an import rewrites with zeros, in one copy.
    pub(crate) words_bytes: u64,
}

impl StateShape {
    pub(crate) fn delta_bytes(&self) -> u64 {
        self.delta_layers as u64 * (self.recurrent_bytes + self.conv_bytes)
    }

    pub(crate) fn kv_bytes(&self) -> u64 {
        self.kv_runs.iter().flatten().flatten().sum()
    }

    /// The bytes of an exported state: the DeltaNet blocks plus the KV planes.
    pub(crate) fn state_bytes(&self) -> u64 {
        self.delta_bytes() + self.kv_bytes()
    }

    fn copy_calls(&self) -> u64 {
        2 * self.delta_layers as u64
            + self
                .kv_runs
                .iter()
                .flatten()
                .map(|runs| runs.len() as u64)
                .sum::<u64>()
    }

    /// `export_state`: one device read per DeltaNet block and per non-empty KV run.
    pub(crate) fn export(&self) -> Transfer {
        Transfer {
            d2h_bytes: self.state_bytes(),
            d2h_calls: self.copy_calls(),
            ..Transfer::default()
        }
    }

    /// `import_state`: the same blocks and runs written back, plus the words block.
    pub(crate) fn import(&self) -> Transfer {
        Transfer {
            h2d_bytes: self.state_bytes() + self.words_bytes,
            h2d_calls: self.copy_calls() + 1,
            ..Transfer::default()
        }
    }
}

/// The small copies every step makes: the start block written (32 bytes) and the result pair read
/// back (8 bytes), one call each. Read from `write_start` and `read_back` in `step.rs`.
pub(crate) const START_BLOCK_BYTES: u64 = 32;
pub(crate) const RESULT_PAIR_BYTES: u64 = 8;

/// One step without logits: the start block and the result pair.
pub(crate) fn step_transfer() -> Transfer {
    Transfer {
        d2h_bytes: RESULT_PAIR_BYTES,
        d2h_calls: 1,
        h2d_bytes: START_BLOCK_BYTES,
        h2d_calls: 1,
    }
}

/// The extra read of one logits row (`head_rows` f32 values, one call).
pub(crate) fn logits_transfer(head_rows: u64) -> Transfer {
    Transfer {
        d2h_bytes: head_rows * 4,
        d2h_calls: 1,
        ..Transfer::default()
    }
}

/// The copies of one eager window of `rows` rows whose verdict keeps `kept` rows, by part.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct WindowTransfers {
    pub(crate) export: Transfer,
    /// The row steps, with their logits rows.
    pub(crate) rows: Transfer,
    pub(crate) import: Transfer,
    /// The kept rows fed again (steps without logits).
    pub(crate) replay: Transfer,
}

impl WindowTransfers {
    pub(crate) fn new(state: &StateShape, head_rows: u64, rows: usize, kept: usize) -> Self {
        let n = rows as u64;
        let per_row = step_transfer() + logits_transfer(head_rows);
        let rollback = kept < rows;
        let replayed = if rollback { kept as u64 } else { 0 };
        let scaled = |one: Transfer, times: u64| Transfer {
            d2h_bytes: one.d2h_bytes * times,
            d2h_calls: one.d2h_calls * times,
            h2d_bytes: one.h2d_bytes * times,
            h2d_calls: one.h2d_calls * times,
        };
        Self {
            export: state.export(),
            rows: scaled(per_row, n),
            import: if rollback {
                state.import()
            } else {
                Transfer::default()
            },
            replay: scaled(step_transfer(), replayed),
        }
    }

    pub(crate) fn total(&self) -> Transfer {
        self.export + self.rows + self.import + self.replay
    }

    pub(crate) fn to_json(self) -> Value {
        json!({
            "export": self.export.to_json(),
            "row_steps_and_logits": self.rows.to_json(),
            "import": self.import.to_json(),
            "replay": self.replay.to_json(),
            "total": self.total().to_json(),
            "basis": "computed from the state layout and the logits width, not counted at run time",
        })
    }
}

/// Fields every record must carry before it is written, as a path list. A missing or empty
/// identity makes the record unusable as evidence.
const REQUIRED_IDENTITY: [&str; 9] = [
    "commit",
    "binary_sha256",
    "pack.manifest_sha256",
    "kv_mode",
    "capture",
    "token_ids.text_sha256",
    "token_ids.tokenizer_sha256",
    "token_ids.ids_sha256",
    "token_ids.source",
];

/// Checks the identity block of a record: every required path exists and is not empty or null.
pub(crate) fn check_identity(record: &Value) -> Result<(), String> {
    let identity = &record["identity"];
    for path in REQUIRED_IDENTITY {
        let found = path
            .split('.')
            .try_fold(identity, |value, key| value.get(key));
        match found {
            None | Some(Value::Null) => return Err(format!("identity.{path} is missing")),
            Some(Value::String(text)) if text.is_empty() => {
                return Err(format!("identity.{path} is empty"));
            }
            Some(_) => {}
        }
    }
    Ok(())
}
