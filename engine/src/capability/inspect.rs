//! What is in a model directory, read without loading any weights.
//!
//! GGUF facts come from the shared `GgufHeader` parse, which stops at the end of
//! the tensor table and never reads payloads; safetensors directories are
//! inventoried by file size only (the loaders stay the authority on shard
//! contents). A directory that cannot be inventoried at all is an error;
//! a directory whose files merely look wrong is a report the preflight
//! verdict explains.

use serde::Serialize;
use std::path::{Path, PathBuf};

use crate::EngError;
use crate::weights::MetaValue;

/// Header cap for a scanned GGUF file. The real Qwen3.6 header (metadata
/// plus tensor table) is well under this; the cap exists so a hostile or
/// truncated file cannot make inspection read unboundedly.
const GGUF_HEADER_CAP_BYTES: u64 = 64 << 20;

/// The four files every engine entry reads from its `tokenizer_dir`
/// (`doc/reference/models.md`); whether their contents match a family's
/// pinned release is the tokenizer loader's check, not this inventory's.
pub const TOKENIZER_FILES: [&str; 4] = [
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "generation_config.json",
];

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FileFact {
    pub name: String,
    pub bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct GgufFact {
    pub file: String,
    pub file_bytes: u64,
    /// `general.architecture` from the header, when it is a string.
    pub architecture: Option<String>,
    pub tensor_count: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ModelInspection {
    /// The directory of the model's side files (`config.json`, tokenizer files): the model path,
    /// or the parent of the weights file it names.
    pub model_dir: PathBuf,
    /// What the model path names: the directory, or the one weights file (first shard of a
    /// split set) that was asked for.
    pub weights_path: PathBuf,
    pub files: Vec<FileFact>,
    /// Every `.gguf` file, parsed header-first. `None` inside the vector
    /// is impossible: an unparseable GGUF lands in `gguf_errors` instead,
    /// with the parser's own message.
    pub gguf_files: Vec<GgufFact>,
    pub gguf_errors: Vec<String>,
    pub safetensors_files: usize,
    pub safetensors_bytes: u64,
    pub has_config_json: bool,
    /// `model_type` of `config.json` (root or `text_config`), the value
    /// `FamilySpec::model_types` matches loaders against.
    pub config_model_types: Vec<String>,
    pub tokenizer_present: Vec<String>,
    pub tokenizer_missing: Vec<String>,
}

impl ModelInspection {
    /// Total checkpoint bytes on disk: GGUF payloads plus safetensors
    /// shards. This is the preflight's weight figure: what the loader
    /// uploads or maps, before any residency policy.
    pub fn weight_bytes(&self) -> u64 {
        let gguf: u64 = self.gguf_files.iter().map(|fact| fact.file_bytes).sum();
        gguf + self.safetensors_bytes
    }
}

fn checkpoint(message: String) -> EngError {
    EngError::Checkpoint(format!("model inspection: {message}"))
}

/// Inspect a model path: a directory of weights, or one GGUF file (a `.gguf` name, an Ollama
/// blob that only starts with the GGUF magic, or any shard of a split set). A file names its own
/// weights only: its siblings are not counted, and `config.json` and tokenizer files are looked
/// for beside it. Tokenizer presence is read from `tokenizer_dir` (pass the same path twice for
/// the usual one-directory layout).
pub fn inspect(model_path: &Path, tokenizer_dir: &Path) -> Result<ModelInspection, EngError> {
    crate::storage_guard::require_local(model_path, "model weights")?;
    let from_file = crate::family::gguf_select::is_gguf_file(model_path);
    if !from_file && !model_path.is_dir() {
        return Err(checkpoint(format!(
            "{} is not a directory or a GGUF file",
            model_path.display()
        )));
    }
    let (model_dir, weights_path) = if from_file {
        let first = crate::family::gguf_select::first_shard(model_path);
        (crate::family::gguf_select::directory_of(&first), first)
    } else {
        (model_path.to_path_buf(), model_path.to_path_buf())
    };
    let mut names: Vec<PathBuf> = Vec::new();
    if from_file {
        // The file, or every shard of its split set that is present.
        match crate::weights::shard_name_of(&weights_path) {
            Some(name) => {
                for number in 1..=name.count {
                    let shard = name.sibling(&weights_path, number);
                    if !shard.is_file() {
                        return Err(checkpoint(format!(
                            "shard {number} of {} of the split GGUF set {} is missing",
                            name.count, name.stem
                        )));
                    }
                    names.push(shard);
                }
            }
            None => names.push(weights_path.clone()),
        }
    } else {
        let entries = std::fs::read_dir(&model_dir)
            .map_err(|error| checkpoint(format!("cannot list {}: {error}", model_dir.display())))?;
        for entry in entries {
            let entry = entry.map_err(|error| {
                checkpoint(format!("cannot list {}: {error}", model_dir.display()))
            })?;
            names.push(entry.path());
        }
    }
    let mut found: Vec<(PathBuf, FileFact)> = Vec::new();
    for path in names {
        // `entry.metadata()` does not traverse symlinks, and every file
        // of a hub-cache snapshot is a symlink into `blobs/` — follow the
        // entry's path so hub layouts count as the files they point at.
        let metadata = match std::fs::metadata(&path) {
            Ok(metadata) => metadata,
            // A link whose blob is gone (a half-finished download) names
            // no file: skip it, the way a missing file is skipped.
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
            Err(error) => {
                return Err(checkpoint(format!(
                    "cannot stat {}: {error}",
                    path.display()
                )));
            }
        };
        if metadata.is_file() {
            let name = path
                .file_name()
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_default();
            found.push((
                path,
                FileFact {
                    name,
                    bytes: metadata.len(),
                },
            ));
        }
    }
    found.sort_by(|a, b| a.1.name.cmp(&b.1.name));
    let files: Vec<FileFact> = found.iter().map(|(_, fact)| fact.clone()).collect();

    let mut gguf_files = Vec::new();
    let mut gguf_errors = Vec::new();
    let mut safetensors_files = 0usize;
    let mut safetensors_bytes = 0u64;
    for (path, file) in &found {
        // A file asked for by path is a GGUF whatever its name (an Ollama blob has none).
        if from_file || file.name.ends_with(".gguf") {
            match gguf_fact(path, file.bytes) {
                Ok(mut fact) => {
                    fact.file.clone_from(&file.name);
                    gguf_files.push(fact);
                }
                Err(error) => {
                    gguf_errors.push(format!("{}: {}", file.name, super::describe_error(&error)))
                }
            }
        } else if file.name.ends_with(".safetensors") {
            safetensors_files += 1;
            safetensors_bytes += file.bytes;
        }
    }

    // A lone weights file is read for what its header says: a `config.json` beside it may belong
    // to another model in the directory.
    let config_path = model_dir.join("config.json");
    let has_config_json = !from_file && config_path.is_file();
    let config_model_types = if has_config_json {
        read_model_types(&config_path).unwrap_or_default()
    } else {
        Vec::new()
    };

    let mut tokenizer_present = Vec::new();
    let mut tokenizer_missing = Vec::new();
    for name in TOKENIZER_FILES {
        if tokenizer_dir.join(name).is_file() {
            tokenizer_present.push(name.to_string());
        } else {
            tokenizer_missing.push(name.to_string());
        }
    }

    Ok(ModelInspection {
        model_dir,
        weights_path,
        files,
        gguf_files,
        gguf_errors,
        safetensors_files,
        safetensors_bytes,
        has_config_json,
        config_model_types,
        tokenizer_present,
        tokenizer_missing,
    })
}

fn gguf_fact(path: &Path, file_bytes: u64) -> Result<GgufFact, EngError> {
    let header = crate::weights::shared_header(path, GGUF_HEADER_CAP_BYTES)?;
    let architecture = match header.meta("general.architecture") {
        Some(MetaValue::String(value)) => Some(value.clone()),
        _ => None,
    };
    Ok(GgufFact {
        file: String::new(),
        file_bytes,
        architecture,
        tensor_count: header.tensors().len(),
    })
}

/// The `model_type` strings of a `config.json`: root and `text_config`,
/// the two places `FamilySpec::check_config` accepts them. A config that
/// cannot be parsed yields none; the loader's own parse stays the
/// authority and reports the real error at load time.
fn read_model_types(path: &Path) -> Option<Vec<String>> {
    let text = std::fs::read(path).ok()?;
    let root: serde_json::Value = serde_json::from_slice(&text).ok()?;
    let mut types = Vec::new();
    for node in [&root, root.get("text_config").unwrap_or(&root)] {
        if let Some(value) = node.get("model_type").and_then(|v| v.as_str())
            && !types.iter().any(|seen| seen == value)
        {
            types.push(value.to_string());
        }
    }
    Some(types)
}

#[cfg(test)]
mod tests {
    use super::{TOKENIZER_FILES, inspect};
    use std::path::PathBuf;

    fn scratch(_name: &str) -> PathBuf {
        // Leaked on purpose: the report outlives this helper, and the
        // tempdir root is the test runner's to clean.
        tempfile::tempdir().unwrap().keep()
    }

    // The real entry point on a real network mount: without the guard the
    // error would be "not a directory". Skipped with a note on a host with no
    // network mount.
    #[cfg(target_os = "linux")]
    #[test]
    fn inspect_refuses_a_live_network_path() {
        let Some(path) = crate::storage_guard::seam::live_network_path() else {
            eprintln!("skipped: no network mount on this host");
            return;
        };
        let error = inspect(&path, &path).expect_err("refused");
        assert!(
            error.describe().contains("must be on local storage"),
            "{}",
            error.describe()
        );
    }

    // Fails if `inspect` stops guarding the model directory: an empty
    // directory would otherwise inspect without error.
    #[cfg(target_os = "linux")]
    #[test]
    fn a_model_directory_on_a_network_mount_is_refused() {
        let dir = scratch("network");
        let error = crate::storage_guard::seam::with_nfs_at(&dir, || {
            inspect(&dir, &dir).expect_err("refused")
        });
        assert!(
            error.describe().contains("must be on local storage"),
            "{}",
            error.describe()
        );
        assert!(inspect(&dir, &dir).is_ok());
    }

    #[test]
    fn a_missing_directory_is_an_error_not_a_report() {
        let dir = PathBuf::from("/nonexistent/fifi-model-dir");
        let error = crate::capability::describe_error(&inspect(&dir, &dir).unwrap_err());
        assert!(error.contains("not a directory"), "{error}");
    }

    #[test]
    fn inventory_counts_weights_and_tokenizer_files() {
        let dir = scratch("inventory");
        std::fs::write(dir.join("model-00001.safetensors"), vec![0u8; 100]).unwrap();
        std::fs::write(dir.join("model-00002.safetensors"), vec![0u8; 50]).unwrap();
        std::fs::write(dir.join("config.json"), r#"{"model_type":"qwen3_5_moe"}"#).unwrap();
        std::fs::write(dir.join("tokenizer.json"), "{}").unwrap();
        let report = inspect(&dir, &dir).unwrap();
        assert_eq!(report.safetensors_files, 2);
        assert_eq!(report.weight_bytes(), 150);
        assert_eq!(report.config_model_types, ["qwen3_5_moe"]);
        assert_eq!(report.tokenizer_present, ["tokenizer.json"]);
        assert_eq!(report.tokenizer_missing.len(), TOKENIZER_FILES.len() - 1);
    }

    #[test]
    fn a_hub_cache_snapshot_of_symlinks_counts_as_files() {
        // Every file of a hub-cache snapshot is a symlink into blobs/;
        // the inventory must count the targets, not skip the links.
        let dir = scratch("hubsnapshot");
        let blobs = dir.join("blobs");
        std::fs::create_dir_all(&blobs).unwrap();
        std::fs::write(blobs.join("weights"), vec![0u8; 80]).unwrap();
        #[cfg(unix)]
        std::os::unix::fs::symlink("blobs/weights", dir.join("model.safetensors")).unwrap();
        let report = inspect(&dir, &dir).unwrap();
        assert_eq!(report.safetensors_files, 1);
        assert_eq!(report.weight_bytes(), 80);
    }

    #[test]
    fn a_dangling_link_in_a_snapshot_is_skipped_not_fatal() {
        let dir = scratch("danglinglink");
        std::fs::write(dir.join("config.json"), "{}").unwrap();
        #[cfg(unix)]
        std::os::unix::fs::symlink("blobs/gone", dir.join("model-00002.safetensors")).unwrap();
        let report = inspect(&dir, &dir).expect("a dangling link must not fail the inventory");
        assert_eq!(report.safetensors_files, 0);
        assert!(
            report
                .files
                .iter()
                .all(|file| file.name != "model-00002.safetensors")
        );
    }

    #[test]
    fn a_truncated_gguf_is_reported_with_the_parsers_words() {
        let dir = scratch("badgguf");
        std::fs::write(dir.join("broken.gguf"), b"GGUF-not-a-header").unwrap();
        let report = inspect(&dir, &dir).unwrap();
        assert!(report.gguf_files.is_empty());
        assert_eq!(report.gguf_errors.len(), 1);
        assert!(
            report.gguf_errors[0].starts_with("broken.gguf: "),
            "{}",
            report.gguf_errors[0]
        );
    }

    #[test]
    fn nested_text_config_model_type_is_found_once() {
        let dir = scratch("nested");
        std::fs::write(
            dir.join("config.json"),
            r#"{"model_type":"outer","text_config":{"model_type":"outer"}}"#,
        )
        .unwrap();
        let report = inspect(&dir, &dir).unwrap();
        assert_eq!(report.config_model_types, ["outer"]);
    }

    /// A GGUF header with no keys and no tensors, padded to `len` bytes.
    fn gguf_bytes(len: usize) -> Vec<u8> {
        let mut bytes = b"GGUF".to_vec();
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.resize(len, 0);
        bytes
    }

    #[test]
    fn a_gguf_file_counts_only_itself_and_reads_its_side_files_beside_it() {
        let dir = scratch("one-file");
        std::fs::write(dir.join("a-q4.gguf"), gguf_bytes(100)).unwrap();
        std::fs::write(dir.join("a-q8.gguf"), gguf_bytes(300)).unwrap();
        std::fs::write(dir.join("config.json"), r#"{"model_type":"gemma4"}"#).unwrap();
        let file = dir.join("a-q4.gguf");
        let report = inspect(&file, &dir).unwrap();
        assert_eq!(report.gguf_files.len(), 1);
        assert_eq!(report.weight_bytes(), 100);
        assert_eq!(report.weights_path, file);
        assert_eq!(report.model_dir, dir);
        // A config.json beside a lone file is not read: it may belong to another model.
        assert!(report.config_model_types.is_empty());
        assert!(!report.has_config_json);
        // The directory form reads it and still counts both files.
        let whole = inspect(&dir, &dir).unwrap();
        assert_eq!(whole.gguf_files.len(), 2);
        assert_eq!(whole.config_model_types, ["gemma4"]);
    }

    #[test]
    fn an_ollama_blob_and_a_split_set_are_inspected_by_path() {
        let dir = scratch("blob-and-split");
        let blob = dir.join("sha256-feedbeef");
        std::fs::write(&blob, gguf_bytes(64)).unwrap();
        let report = inspect(&blob, &blob).unwrap();
        assert_eq!(report.gguf_files.len(), 1, "{:?}", report.gguf_errors);
        assert_eq!(report.gguf_files[0].file, "sha256-feedbeef");
        for number in 1..=2 {
            std::fs::write(
                dir.join(format!("s-{number:05}-of-00002.gguf")),
                gguf_bytes(200),
            )
            .unwrap();
        }
        // Any shard names the set, and the set is counted whole.
        let report = inspect(&dir.join("s-00002-of-00002.gguf"), &dir).unwrap();
        assert_eq!(report.gguf_files.len(), 2);
        assert_eq!(report.weight_bytes(), 400);
        assert_eq!(report.weights_path, dir.join("s-00001-of-00002.gguf"));
    }

    #[test]
    fn a_split_set_with_a_shard_missing_is_refused_naming_the_shard() {
        let dir = scratch("split-missing");
        std::fs::write(dir.join("s-00002-of-00003.gguf"), gguf_bytes(100)).unwrap();
        std::fs::write(dir.join("s-00003-of-00003.gguf"), gguf_bytes(100)).unwrap();
        let error = crate::capability::describe_error(
            &inspect(&dir.join("s-00002-of-00003.gguf"), &dir).unwrap_err(),
        );
        assert!(error.contains("shard 1 of 3"), "{error}");
        std::fs::write(dir.join("s-00001-of-00003.gguf"), gguf_bytes(100)).unwrap();
        std::fs::remove_file(dir.join("s-00003-of-00003.gguf")).unwrap();
        let error = crate::capability::describe_error(
            &inspect(&dir.join("s-00001-of-00003.gguf"), &dir).unwrap_err(),
        );
        assert!(error.contains("shard 3 of 3"), "{error}");
    }

    #[test]
    fn a_file_that_is_not_a_gguf_is_refused_naming_what_is_accepted() {
        let dir = scratch("not-gguf");
        let text = dir.join("notes.txt");
        std::fs::write(&text, b"hello").unwrap();
        let error = crate::capability::describe_error(&inspect(&text, &dir).unwrap_err());
        assert!(error.contains("not a directory or a GGUF file"), "{error}");
    }
}
