//! `serve --model <name>`: a model named instead of located.
//!
//! A name that is not a path is looked for in the local stores, in this order: the models
//! directory (`~/models`, or `FIFI_MODELS_DIR`), the Ollama store (`OLLAMA_MODELS`, or
//! `~/.ollama/models`; a tag is `name[:tag]`) and the hub cache. The first store with a match
//! decides: one match is the model, several are refused with their paths so the user names more
//! of it. A model in the Ollama store is a blob with no `.gguf` name: it is served by its path, and
//! the start knows it by the GGUF magic.

use std::path::{Path, PathBuf};

use super::config::StartupError;

/// The roots of the three stores.
pub(crate) struct Stores {
    pub(crate) models: PathBuf,
    pub(crate) ollama: PathBuf,
    pub(crate) hub: PathBuf,
}

impl Stores {
    pub(crate) fn from_env() -> Self {
        let home = PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| ".".to_string()));
        let from = |name: &str, default: PathBuf| match std::env::var(name) {
            Ok(value) if !value.trim().is_empty() => PathBuf::from(value),
            _ => default,
        };
        Self {
            models: from("FIFI_MODELS_DIR", home.join("models")),
            ollama: from("OLLAMA_MODELS", home.join(".ollama").join("models")),
            hub: super::hf::hub_dir_from_env(),
        }
    }
}

/// A model found in a store: the path a start takes (a model directory, or one weights file when
/// the name was a file's) and where it came from.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Found {
    pub(crate) dir: PathBuf,
    pub(crate) store: &'static str,
}

/// Whether `name` reads as a model name rather than a path or a hub id (`org/repo`).
pub(crate) fn is_plain_name(name: &str) -> bool {
    !name.is_empty() && !name.contains('/') && !name.starts_with('.') && !name.starts_with('~')
}

fn has_gguf(dir: &Path) -> bool {
    std::fs::read_dir(dir).is_ok_and(|entries| {
        entries.filter_map(Result::ok).any(|entry| {
            entry
                .path()
                .extension()
                .is_some_and(|extension| extension.eq_ignore_ascii_case("gguf"))
        })
    })
}

/// Which of `candidates` (name, directory) match `query`: the exact names when there are any,
/// otherwise the names that contain it.
fn best(query: &str, mut candidates: Vec<(String, PathBuf)>) -> Vec<PathBuf> {
    let query = query.to_lowercase();
    candidates.sort();
    let exact: Vec<PathBuf> = candidates
        .iter()
        .filter(|(name, _)| name.to_lowercase() == query)
        .map(|(_, dir)| dir.clone())
        .collect();
    if !exact.is_empty() {
        return exact;
    }
    candidates
        .into_iter()
        .filter(|(name, _)| name.to_lowercase().contains(&query))
        .map(|(_, dir)| dir)
        .collect()
}

/// The directories of the models directory whose name matches, and the weights files whose name
/// matches.
fn in_models(root: &Path, query: &str) -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    for entry in std::fs::read_dir(root).into_iter().flatten().flatten() {
        let dir = entry.path();
        if !dir.is_dir() || !has_gguf(&dir) {
            continue;
        }
        let name = entry.file_name().to_string_lossy().into_owned();
        candidates.push((name, dir.clone()));
        for file in std::fs::read_dir(&dir).into_iter().flatten().flatten() {
            let file_name = file.file_name().to_string_lossy().into_owned();
            // A weights file's name names that file, not the directory it sits in: the directory
            // may hold other quantizations. Later shards of a split set are not names of their own.
            let later_shard =
                crate::weights::shard_name_of(&file.path()).is_some_and(|shard| shard.number != 1);
            if file_name.to_lowercase().ends_with(".gguf") && !later_shard {
                candidates.push((file_name, file.path()));
            }
        }
    }
    let found = best(query, candidates);
    let dirs: Vec<PathBuf> = found.iter().filter(|path| path.is_dir()).cloned().collect();
    // A directory and a weights file in it that both match are one model: the directory.
    let mut paths: Vec<PathBuf> = found
        .into_iter()
        .filter(|path| {
            path.is_dir()
                || !path
                    .parent()
                    .is_some_and(|parent| dirs.iter().any(|dir| dir == parent))
        })
        .collect();
    paths.dedup();
    paths
}

/// The model layer of the Ollama manifest `name[:tag]`: its blob, which is a GGUF file with no
/// extension and is served by its path.
fn in_ollama(root: &Path, query: &str) -> Result<Vec<PathBuf>, StartupError> {
    let (name, tag) = match query.split_once(':') {
        Some((name, tag)) => (name, Some(tag)),
        None => (query, None),
    };
    let dir = root
        .join("manifests")
        .join("registry.ollama.ai")
        .join("library")
        .join(name);
    let tags: Vec<String> = std::fs::read_dir(&dir)
        .into_iter()
        .flatten()
        .flatten()
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .collect();
    let tag = match tag {
        Some(tag) if tags.iter().any(|found| found == tag) => tag.to_string(),
        Some(_) => return Ok(Vec::new()),
        None if tags.iter().any(|found| found == "latest") => "latest".to_string(),
        None if tags.len() == 1 => tags[0].clone(),
        None if tags.is_empty() => return Ok(Vec::new()),
        None => {
            let mut tags = tags;
            tags.sort();
            return Err(StartupError(format!(
                "serve --model: {name} has several Ollama tags ({}); name one, for example {name}:{}",
                tags.join(", "),
                tags[0]
            )));
        }
    };
    let manifest = std::fs::read(dir.join(&tag))
        .ok()
        .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok());
    let digest = manifest
        .as_ref()
        .and_then(|manifest| manifest["layers"].as_array())
        .and_then(|layers| {
            layers
                .iter()
                .find(|layer| layer["mediaType"] == "application/vnd.ollama.image.model")
        })
        .and_then(|layer| layer["digest"].as_str())
        .map(|digest| digest.replace(':', "-"));
    let Some(digest) = digest else {
        return Ok(Vec::new());
    };
    let blob = root.join("blobs").join(digest);
    if !blob.is_file() {
        return Ok(Vec::new());
    }
    Ok(vec![blob])
}

/// The snapshots of the hub cache whose repository name matches and that hold a GGUF.
fn in_hub(root: &Path, query: &str) -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    for entry in std::fs::read_dir(root).into_iter().flatten().flatten() {
        let folder = entry.file_name().to_string_lossy().into_owned();
        let Some(repo) = folder.strip_prefix("models--") else {
            continue;
        };
        let repo_name = repo.rsplit("--").next().unwrap_or(repo).to_string();
        let snapshots = entry.path().join("snapshots");
        let mut snapshot_dirs: Vec<PathBuf> = std::fs::read_dir(&snapshots)
            .into_iter()
            .flatten()
            .flatten()
            .map(|snapshot| snapshot.path())
            .filter(|path| has_gguf(path))
            .collect();
        snapshot_dirs.sort();
        if let Some(latest) = snapshot_dirs.pop() {
            candidates.push((repo_name, latest));
        }
    }
    best(query, candidates)
}

/// A model of a local store, for choosing one when none was named.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Local {
    /// The path a start takes: a model directory, or one weights file.
    pub(crate) path: PathBuf,
    pub(crate) store: &'static str,
    /// The name `--model` takes for it.
    pub(crate) name: String,
    pub(crate) modified: std::time::SystemTime,
}

fn modified(path: &Path) -> std::time::SystemTime {
    std::fs::metadata(path)
        .and_then(|meta| meta.modified())
        .unwrap_or(std::time::UNIX_EPOCH)
}

/// Every model of the three stores, the most recently written first (ties by path, so the order
/// is stable). A model directory counts by its newest weights file.
pub(crate) fn all(stores: &Stores) -> Vec<Local> {
    let newest = |path: &Path| -> std::time::SystemTime {
        if path.is_dir() {
            std::fs::read_dir(path)
                .into_iter()
                .flatten()
                .flatten()
                .map(|entry| modified(&entry.path()))
                .max()
                .unwrap_or_else(|| modified(path))
        } else {
            modified(path)
        }
    };
    let mut found = Vec::new();
    for path in in_models(&stores.models, "") {
        let name = path
            .file_name()
            .map_or_else(String::new, |n| n.to_string_lossy().into_owned());
        found.push(Local {
            modified: newest(&path),
            path,
            store: "the models directory",
            name,
        });
    }
    let library = stores
        .ollama
        .join("manifests")
        .join("registry.ollama.ai")
        .join("library");
    for model in std::fs::read_dir(&library).into_iter().flatten().flatten() {
        let name = model.file_name().to_string_lossy().into_owned();
        for tag in std::fs::read_dir(model.path())
            .into_iter()
            .flatten()
            .flatten()
        {
            let query = format!("{name}:{}", tag.file_name().to_string_lossy());
            if let Ok(blobs) = in_ollama(&stores.ollama, &query) {
                for path in blobs {
                    found.push(Local {
                        modified: modified(&path),
                        path,
                        store: "the Ollama store",
                        name: query.clone(),
                    });
                }
            }
        }
    }
    for path in in_hub(&stores.hub, "") {
        // `.../models--org--repo/snapshots/<commit>`
        let repo = path
            .ancestors()
            .nth(2)
            .and_then(Path::file_name)
            .map(|folder| {
                folder
                    .to_string_lossy()
                    .replacen("models--", "", 1)
                    .replace("--", "/")
            })
            .unwrap_or_default();
        found.push(Local {
            modified: newest(&path),
            path,
            store: "the hub cache",
            name: repo,
        });
    }
    found.sort_by(|a, b| {
        b.modified
            .cmp(&a.modified)
            .then_with(|| a.path.cmp(&b.path))
    });
    found
}

/// Looks `name` up in the stores. `Ok(None)` when nothing local matches.
pub(crate) fn resolve(name: &str, stores: &Stores) -> Result<Option<Found>, StartupError> {
    let lookups: [(&'static str, Result<Vec<PathBuf>, StartupError>); 3] = [
        ("the models directory", Ok(in_models(&stores.models, name))),
        ("the Ollama store", in_ollama(&stores.ollama, name)),
        ("the hub cache", Ok(in_hub(&stores.hub, name))),
    ];
    for (store, found) in lookups {
        let mut found = found?;
        match found.len() {
            0 => continue,
            1 => {
                return Ok(Some(Found {
                    dir: found.remove(0),
                    store,
                }));
            }
            _ => {
                let list: Vec<String> = found.iter().map(|dir| dir.display().to_string()).collect();
                return Err(StartupError(format!(
                    "serve --model: {name} matches several models in {store}: {}; name more of it or pass the directory",
                    list.join(", ")
                )));
            }
        }
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::{Stores, resolve};
    use std::path::Path;

    fn touch(path: &Path) {
        std::fs::create_dir_all(path.parent().expect("parent")).expect("dirs");
        std::fs::write(path, b"x").expect("write");
    }

    fn stores(root: &Path) -> Stores {
        Stores {
            models: root.join("models"),
            ollama: root.join("ollama"),
            hub: root.join("hub"),
        }
    }

    fn ollama_model(root: &Path, name: &str, tag: &str, digest: &str) {
        let manifest = format!(
            r#"{{"layers":[{{"mediaType":"application/vnd.ollama.image.projector","digest":"sha256:other"}},{{"mediaType":"application/vnd.ollama.image.model","digest":"sha256:{digest}"}}]}}"#
        );
        let path = root
            .join("ollama/manifests/registry.ollama.ai/library")
            .join(name)
            .join(tag);
        std::fs::create_dir_all(path.parent().expect("parent")).expect("dirs");
        std::fs::write(path, manifest).expect("manifest");
        touch(&root.join("ollama/blobs").join(format!("sha256-{digest}")));
    }

    #[test]
    fn a_name_finds_the_models_directory_by_directory_or_weights_file_name() {
        let root = tempfile::tempdir().expect("tempdir");
        touch(
            &root
                .path()
                .join("models/qwen3.6/Qwen3.6-35B-A3B-Q4_K_M.gguf"),
        );
        touch(&root.path().join("models/other/Other-7B.gguf"));
        let found = resolve("qwen3.6", &stores(root.path()))
            .expect("resolves")
            .expect("found");
        assert_eq!(found.dir, root.path().join("models/qwen3.6"));
        assert_eq!(found.store, "the models directory");
        let by_file = resolve("Other-7B", &stores(root.path()))
            .expect("resolves")
            .expect("found");
        assert_eq!(by_file.dir, root.path().join("models/other/Other-7B.gguf"));
    }

    #[test]
    fn an_ollama_tag_resolves_to_its_blob_and_the_only_tag_is_the_default() {
        let root = tempfile::tempdir().expect("tempdir");
        ollama_model(root.path(), "qwen3.6", "35b-a3b", "abc123");
        for name in ["qwen3.6", "qwen3.6:35b-a3b"] {
            let found = resolve(name, &stores(root.path()))
                .expect("resolves")
                .expect("found");
            assert_eq!(found.store, "the Ollama store");
            assert_eq!(found.dir, root.path().join("ollama/blobs/sha256-abc123"));
        }
        assert!(
            resolve("qwen3.6:other", &stores(root.path()))
                .expect("resolves")
                .is_none(),
            "an unknown tag is not a match"
        );
    }

    #[test]
    fn several_ollama_tags_without_one_named_are_refused_with_the_tags() {
        let root = tempfile::tempdir().expect("tempdir");
        ollama_model(root.path(), "qwen3", "8b", "aaa");
        ollama_model(root.path(), "qwen3", "8b-nothink", "bbb");
        let error = resolve("qwen3", &stores(root.path()))
            .expect_err("ambiguous")
            .0;
        assert!(
            error.contains("8b") && error.contains("8b-nothink"),
            "{error}"
        );
    }

    #[test]
    fn the_hub_cache_is_searched_by_repository_name_and_the_first_store_with_a_match_wins() {
        let root = tempfile::tempdir().expect("tempdir");
        touch(&root.path().join(
            "hub/models--unsloth--Qwen3.6-35B-A3B-GGUF/snapshots/abc/Qwen3.6-35B-A3B-Q4_K_M.gguf",
        ));
        touch(
            &root
                .path()
                .join("hub/models--org--no-weights/snapshots/abc/readme.md"),
        );
        let found = resolve("qwen3.6-35b-a3b-gguf", &stores(root.path()))
            .expect("resolves")
            .expect("found");
        assert_eq!(found.store, "the hub cache");
        touch(&root.path().join("models/qwen3.6-35b-a3b-gguf/m.gguf"));
        let found = resolve("qwen3.6-35b-a3b-gguf", &stores(root.path()))
            .expect("resolves")
            .expect("found");
        assert_eq!(found.store, "the models directory");
    }

    #[test]
    fn two_matches_in_one_store_are_refused_naming_both_and_no_match_is_none() {
        let root = tempfile::tempdir().expect("tempdir");
        touch(&root.path().join("models/qwen-a/a.gguf"));
        touch(&root.path().join("models/qwen-b/b.gguf"));
        let error = resolve("qwen", &stores(root.path()))
            .expect_err("ambiguous")
            .0;
        assert!(
            error.contains("qwen-a") && error.contains("qwen-b"),
            "{error}"
        );
        assert!(
            resolve("nothing-like-it", &stores(root.path()))
                .expect("resolves")
                .is_none()
        );
    }

    #[test]
    fn a_partial_name_that_matches_a_directory_and_its_one_file_is_one_model() {
        let root = tempfile::tempdir().expect("tempdir");
        touch(
            &root
                .path()
                .join("models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-Q4_K_M.gguf"),
        );
        let found = resolve("qwen3.6", &stores(root.path()))
            .expect("resolves")
            .expect("found");
        assert_eq!(found.dir, root.path().join("models/Qwen3.6-35B-A3B-GGUF"));
        // Two files of one directory that both match, and not the directory: still two models.
        touch(&root.path().join("models/pair/Model-7B-Q4_K_M.gguf"));
        touch(&root.path().join("models/pair/Model-7B-Q8_0.gguf"));
        let error = resolve("Model-7B", &stores(root.path())).expect_err("two files, no directory");
        assert!(error.0.contains("matches several models"), "{}", error.0);
    }

    #[test]
    fn every_store_is_listed_newest_first_with_the_name_to_pass() {
        let root = tempfile::tempdir().expect("tempdir");
        let root = root.path();
        touch(&root.join("models/a/a-Q4.gguf"));
        std::thread::sleep(std::time::Duration::from_millis(30));
        ollama_model(root, "qwen3", "8b", "abc");
        std::thread::sleep(std::time::Duration::from_millis(30));
        touch(&root.join("hub/models--org--repo/snapshots/c0ffee/m.gguf"));
        let listed: Vec<(String, &str)> = super::all(&stores(root))
            .into_iter()
            .map(|model| (model.name, model.store))
            .collect();
        assert_eq!(
            listed,
            [
                ("org/repo".to_string(), "the hub cache"),
                ("qwen3:8b".to_string(), "the Ollama store"),
                ("a".to_string(), "the models directory"),
            ]
        );
    }
}
