//! `serve --model <name>`: a model named instead of located.
//!
//! A name that is not a path is looked for in the local stores, in this order: the models
//! directory (`~/models`, or `FIFI_MODELS_DIR`), the Ollama store (`OLLAMA_MODELS`, or
//! `~/.ollama/models`; a tag is `name[:tag]`) and the hub cache. The first store with a match
//! decides: one match is the model, several are refused with their paths so the user names more
//! of it. A model in the Ollama store is a blob with no `.gguf` name, so the start reaches it
//! through a link under the engine's cache directory, never inside a store.

use std::path::{Path, PathBuf};

use super::config::StartupError;

/// The roots of the three stores.
pub(crate) struct Stores {
    pub(crate) models: PathBuf,
    pub(crate) ollama: PathBuf,
    pub(crate) hub: PathBuf,
    /// Where the links to Ollama blobs go.
    pub(crate) links: PathBuf,
}

impl Stores {
    pub(crate) fn from_env() -> Self {
        let home = PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| ".".to_string()));
        let from = |name: &str, default: PathBuf| match std::env::var(name) {
            Ok(value) if !value.trim().is_empty() => PathBuf::from(value),
            _ => default,
        };
        let cache = from("XDG_CACHE_HOME", home.join(".cache"));
        Self {
            models: from("FIFI_MODELS_DIR", home.join("models")),
            ollama: from("OLLAMA_MODELS", home.join(".ollama").join("models")),
            hub: super::hf::hub_dir_from_env(),
            links: cache.join("fifi").join("ollama"),
        }
    }
}

/// A model found in a store: the directory a start takes and where it came from.
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

/// The directories of the models directory whose name, or whose weights file's name, matches.
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
            if file_name.to_lowercase().ends_with(".gguf") {
                candidates.push((file_name, dir.clone()));
            }
        }
    }
    let mut dirs = best(query, candidates);
    dirs.dedup();
    dirs
}

/// The model layer of the Ollama manifest `name[:tag]`, as a link directory under `links`.
fn in_ollama(root: &Path, links: &Path, query: &str) -> Result<Vec<PathBuf>, StartupError> {
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
    let link_dir = links.join(format!("{name}-{tag}"));
    std::fs::create_dir_all(&link_dir)
        .map_err(|error| StartupError(format!("serve --model: {}: {error}", link_dir.display())))?;
    let link = link_dir.join("model.gguf");
    if std::fs::read_link(&link).ok().as_deref() != Some(blob.as_path()) {
        let _ = std::fs::remove_file(&link);
        std::os::unix::fs::symlink(&blob, &link)
            .map_err(|error| StartupError(format!("serve --model: {}: {error}", link.display())))?;
    }
    Ok(vec![link_dir])
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

/// Looks `name` up in the stores. `Ok(None)` when nothing local matches.
pub(crate) fn resolve(name: &str, stores: &Stores) -> Result<Option<Found>, StartupError> {
    let lookups: [(&'static str, Result<Vec<PathBuf>, StartupError>); 3] = [
        ("the models directory", Ok(in_models(&stores.models, name))),
        (
            "the Ollama store",
            in_ollama(&stores.ollama, &stores.links, name),
        ),
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
            links: root.join("links"),
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
        assert_eq!(by_file.dir, root.path().join("models/other"));
    }

    #[test]
    fn an_ollama_tag_resolves_to_a_link_named_for_gguf_and_the_only_tag_is_the_default() {
        let root = tempfile::tempdir().expect("tempdir");
        ollama_model(root.path(), "qwen3.6", "35b-a3b", "abc123");
        for name in ["qwen3.6", "qwen3.6:35b-a3b"] {
            let found = resolve(name, &stores(root.path()))
                .expect("resolves")
                .expect("found");
            assert_eq!(found.store, "the Ollama store");
            let link = found.dir.join("model.gguf");
            assert_eq!(
                std::fs::read_link(&link).expect("a link"),
                root.path().join("ollama/blobs/sha256-abc123")
            );
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
}
