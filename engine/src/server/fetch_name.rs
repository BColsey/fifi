//! `serve --model org/repo[:QUANT]` for a model that is not on the disk yet: the GGUF files of one
//! quantization are fetched into the hub cache, then served like any local path.
//!
//! The name is `org/repo`, optionally followed by `:` and a quantization label (`Q4_K_M`,
//! `UD-Q4_K_XL`, `Q8_0`, `F16`). A repo holds many quantizations of one model; only the files of
//! the chosen one are fetched. With no label the planner chooses: the largest quantization that it
//! plans fully resident on this host's card at the default context, else the smallest, which
//! serves on the lower rung. The choice, the reason and the `:QUANT` override are printed.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use super::config::StartupError;
use super::hf::{HfHub, HubFile, JobState, valid_repo_id};

/// A model name that asks for a fetch: the repo and the quantization label, when one was given.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Wanted {
    pub(crate) repo: String,
    pub(crate) quant: Option<String>,
}

/// Reads `org/repo` or `org/repo:QUANT`. `None` when the text is not a repo name at all (a path,
/// a plain name, an Ollama `name:tag`).
pub(crate) fn parse_wanted(text: &str) -> Option<Wanted> {
    let (repo, quant) = match text.split_once(':') {
        Some((repo, quant)) => (repo, Some(quant)),
        None => (text, None),
    };
    if !repo.contains('/') || !valid_repo_id(repo) {
        return None;
    }
    let quant = match quant {
        Some(label)
            if !label.is_empty()
                && label
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-' || b == b'.') =>
        {
            Some(label.to_string())
        }
        Some(_) => return None,
        None => None,
    };
    Some(Wanted {
        repo: repo.to_string(),
        quant,
    })
}

/// One quantization of a repo: its label and the files that make it up (all shards of a split
/// set), with the bytes they add up to.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Quant {
    pub(crate) label: String,
    pub(crate) files: Vec<HubFile>,
    pub(crate) bytes: u64,
}

impl Quant {
    /// The file a start names: the first shard, or the only file.
    pub(crate) fn first(&self) -> &HubFile {
        &self.files[0]
    }
}

fn is_quant_token(token: &str) -> bool {
    let upper = token.to_ascii_uppercase();
    if matches!(upper.as_str(), "BF16" | "F16" | "F32" | "MXFP4") {
        return true;
    }
    let digits = upper
        .strip_prefix("IQ")
        .or_else(|| upper.strip_prefix('Q'))
        .unwrap_or("");
    digits.bytes().next().is_some_and(|b| b.is_ascii_digit())
}

/// The quantization label in a GGUF path: the last name token that reads as one, with a `UD-`
/// prefix kept. A file whose own name carries none takes its directory's.
pub(crate) fn quant_label(path: &str) -> Option<String> {
    let from_name = |name: &str| -> Option<String> {
        let tokens: Vec<&str> = name.split(['-', '.']).collect();
        let at = tokens.iter().rposition(|token| is_quant_token(token))?;
        let label = tokens[at].to_string();
        Some(match at.checked_sub(1).map(|before| tokens[before]) {
            Some(before) if before.eq_ignore_ascii_case("UD") => format!("UD-{label}"),
            _ => label,
        })
    };
    let (dir, name) = match path.rsplit_once('/') {
        Some((dir, name)) => (dir.rsplit('/').next(), name),
        None => (None, path),
    };
    let stem = name.strip_suffix(".gguf")?;
    from_name(stem).or_else(|| dir.and_then(from_name))
}

/// A GGUF file that is not the model's own weights: a vision projector, or the multi-token
/// prediction head some repos ship beside the model (in the root or an `MTP/` directory), whose
/// header names another architecture.
fn is_auxiliary(lower_path: &str) -> bool {
    lower_path
        .split('/')
        .any(|part| part.starts_with("mmproj") || part.starts_with("mtp"))
}

/// The quantizations a listing holds: every complete GGUF set, projector files and empty files
/// left out, each with its label. A split set counts only when every shard of it is listed; two
/// sets with one label (two directories) keep the larger.
pub(crate) fn quants(files: &[HubFile]) -> Vec<Quant> {
    // (directory, stem, shard count; 0 for a single file)
    let mut sets: BTreeMap<(String, String, u32), Vec<(u32, HubFile)>> = BTreeMap::new();
    for file in files {
        let lower = file.path.to_ascii_lowercase();
        if !lower.ends_with(".gguf") || is_auxiliary(&lower) || file.size == 0 {
            continue;
        }
        let (dir, name) = file.path.rsplit_once('/').unwrap_or(("", &file.path));
        let (stem, count, number) = match crate::weights::ShardName::parse(name) {
            Some(shard) => (shard.stem, shard.count, shard.number),
            None => (name.to_string(), 0, 1),
        };
        sets.entry((dir.to_string(), stem, count))
            .or_default()
            .push((number, file.clone()));
    }
    let mut found: BTreeMap<String, Quant> = BTreeMap::new();
    for ((_, _, count), mut set) in sets {
        set.sort_by_key(|(number, _)| *number);
        let whole = if count == 0 {
            set.len() == 1
        } else {
            set.len() == count as usize
                && set
                    .iter()
                    .zip(1..)
                    .all(|((number, _), want)| *number == want)
        };
        if !whole {
            continue;
        }
        let set: Vec<HubFile> = set.into_iter().map(|(_, file)| file).collect();
        let Some(label) = quant_label(&set[0].path) else {
            continue;
        };
        let bytes = set.iter().map(|file| file.size).sum();
        let better = found.get(&label).is_none_or(|old| old.bytes < bytes);
        if better {
            found.insert(
                label.clone(),
                Quant {
                    label,
                    files: set,
                    bytes,
                },
            );
        }
    }
    found.into_values().collect()
}

/// How a quantization sits on this host, from the planner.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Fit {
    /// Planned fully resident on the card at the default context.
    Resident,
    /// Not resident; it serves on a lower rung, or the planner cannot say.
    NotResident,
}

/// The quantization to serve and why. With a label asked for, that one or an error naming what
/// the repo holds; without, the planner's choice.
pub(crate) fn choose(
    quants: &[Quant],
    wanted: Option<&str>,
    max_bytes: Option<u64>,
    mut fits: impl FnMut(&Quant) -> Result<Fit, String>,
) -> Result<(Quant, String), String> {
    if let Some(label) = wanted {
        return quants
            .iter()
            .find(|quant| quant.label.eq_ignore_ascii_case(label))
            .map(|quant| (quant.clone(), format!("{} was asked for", quant.label)))
            .ok_or_else(|| {
                format!(
                    "no quantization named {label}; the repo holds {}",
                    labels(quants)
                )
            });
    }
    // Only quantizations that fit the free disk are candidates.
    let mut ascending: Vec<&Quant> = quants
        .iter()
        .filter(|quant| max_bytes.is_none_or(|limit| quant.bytes <= limit))
        .collect();
    ascending.sort_by(|a, b| a.bytes.cmp(&b.bytes).then_with(|| a.label.cmp(&b.label)));
    let Some(smallest) = ascending.first() else {
        return Err(format!(
            "no quantization fits the free disk; the repo holds {}",
            labels(quants)
        ));
    };
    // A smaller file of the same model never plans worse, so the quantizations that fit
    // resident are a prefix of the sorted list: a binary search finds its end in a few plans
    // instead of one per quantization.
    let (mut low, mut high) = (0, ascending.len());
    while low < high {
        let middle = low + (high - low) / 2;
        if fits(ascending[middle])? == Fit::Resident {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    if low > 0 {
        return Ok((
            ascending[low - 1].clone(),
            format!(
                "the largest of {} that the planner fits fully resident on this card at the default context",
                ascending.len()
            ),
        ));
    }
    Ok((
        (*smallest).clone(),
        format!(
            "none of the {} fits fully resident at the default context, so the smallest, which serves on the lower rung",
            ascending.len()
        ),
    ))
}

fn labels(quants: &[Quant]) -> String {
    let mut names: Vec<&str> = quants.iter().map(|quant| quant.label.as_str()).collect();
    names.sort_unstable();
    names.join(", ")
}

fn mib(bytes: u64) -> u64 {
    bytes >> 20
}

/// What the choice asks of the rest of the server: the family check that refuses a model the
/// engine does not serve, and the planner's fit for a quantization. Both read a stub file that
/// has the real header and the quantization's size.
pub(crate) struct Planner<'a> {
    pub(crate) accepts: &'a dyn Fn(&Path) -> Result<(), StartupError>,
    pub(crate) fit: &'a mut dyn FnMut(&Path) -> Result<Fit, StartupError>,
}

/// The one quantization rule, for a cached snapshot and a listing alike: the family check on the
/// real header, then `choose` with the planner's fit read from stubs of each size.
fn pick(
    repo: &str,
    cache: &Path,
    held: &[Quant],
    wanted: Option<&str>,
    max_bytes: Option<u64>,
    header: &[u8],
    planner: Planner<'_>,
) -> Result<(Quant, String), StartupError> {
    let refuse = |message: String| StartupError(format!("serve --model {repo}: {message}"));
    let stubs = Stubs::new(cache, header)?;
    let size_of = |bytes: u64| bytes.max(header.len() as u64);
    let smallest = held
        .iter()
        .min_by_key(|quant| quant.bytes)
        .ok_or_else(|| refuse("the repo holds no GGUF file".to_string()))?;
    (planner.accepts)(&stubs.path(size_of(smallest.bytes), "check")?)
        .map_err(|error| refuse(error.0))?;
    let fit = planner.fit;
    let mut failure = None;
    let chosen = choose(held, wanted, max_bytes, |quant| {
        stubs
            .path(size_of(quant.bytes), &quant.label)
            .and_then(|stub| fit(&stub))
            .map_err(|error| {
                failure = Some(error.0.clone());
                error.0
            })
    });
    chosen.map_err(|message| refuse(failure.unwrap_or(message)))
}

fn announce(origin: &str, repo: &str, held: &[Quant], picked: &Quant, why: &str) -> String {
    format!(
        "{origin}: {repo}:{} ({} MiB, {}): {why}; name another with {repo}:<QUANT> (the repo holds {})",
        picked.label,
        mib(picked.bytes),
        picked
            .files
            .iter()
            .map(|file| file.path.as_str())
            .collect::<Vec<_>>()
            .join(", "),
        labels(held)
    )
}

/// The GGUF files of a cached snapshot, with the sizes of the blobs they link to.
fn cached_files(snapshot: &Path) -> Vec<HubFile> {
    fn walk(dir: &Path, base: &Path, out: &mut Vec<HubFile>) {
        for entry in std::fs::read_dir(dir).into_iter().flatten().flatten() {
            let path = entry.path();
            if path.is_dir() {
                walk(&path, base, out);
            } else if let (Ok(meta), Ok(relative)) =
                (std::fs::metadata(&path), path.strip_prefix(base))
            {
                out.push(HubFile {
                    path: relative.to_string_lossy().into_owned(),
                    size: meta.len(),
                    sha256: None,
                });
            }
        }
    }
    let mut files = Vec::new();
    walk(snapshot, snapshot, &mut files);
    files
}

/// The path to serve for `wanted`: a quantization of the cached snapshot when the cache holds a
/// complete one that fits the request, else one fetched into the hub cache first. The first shard
/// of a split set, inside the snapshot.
pub(crate) fn resolve(
    hub: &Arc<HfHub>,
    wanted: &Wanted,
    planner: Planner<'_>,
    say: &mut dyn FnMut(&str),
) -> Result<PathBuf, StartupError> {
    let repo = wanted.repo.as_str();
    if let Some(snapshot) = hub.main_snapshot(repo) {
        let held = quants(&cached_files(&snapshot));
        let usable = !held.is_empty()
            && wanted.quant.as_deref().is_none_or(|label| {
                held.iter()
                    .any(|quant| quant.label.eq_ignore_ascii_case(label))
            });
        if usable {
            let smallest = held
                .iter()
                .min_by_key(|quant| quant.bytes)
                .expect("a non-empty set");
            let first = snapshot.join(&smallest.first().path);
            let header = read_local_header(&first, smallest.first().size)
                .map_err(|error| StartupError(format!("serve --model {repo}: {error}")))?;
            let (picked, why) = pick(
                repo,
                hub.cache_dir(),
                &held,
                wanted.quant.as_deref(),
                None,
                &header,
                planner,
            )?;
            say(&announce("cache", repo, &held, &picked, &why));
            return Ok(snapshot.join(&picked.first().path));
        }
    }
    fetch(hub, wanted, planner, say)
}

fn read_local_header(path: &Path, size: u64) -> Result<Vec<u8>, String> {
    let name = path.display().to_string();
    super::hf::read_gguf_header(&name, size, &mut |length| {
        let mut bytes = Vec::new();
        std::io::Read::read_to_end(
            &mut std::io::Read::take(
                std::fs::File::open(path)
                    .map_err(|error| format!("cannot read {name}: {error}"))?,
                length,
            ),
            &mut bytes,
        )
        .map_err(|error| format!("cannot read {name}: {error}"))?;
        Ok(bytes)
    })
}

/// Makes the files of one quantization of `wanted.repo` exist in the hub cache and returns the
/// path to serve: the first shard, inside the snapshot.
fn fetch(
    hub: &Arc<HfHub>,
    wanted: &Wanted,
    planner: Planner<'_>,
    say: &mut dyn FnMut(&str),
) -> Result<PathBuf, StartupError> {
    let refuse =
        |message: String| StartupError(format!("serve --model {}: {message}", wanted.repo));
    let tree = hub.tree(&wanted.repo, "main").map_err(&refuse)?;
    let held = quants(&tree.files);
    if held.is_empty() {
        return Err(refuse(
            "the repo holds no complete GGUF file; name a repo of GGUF files".to_string(),
        ));
    }
    // Nothing is read from the repo's files while even the smallest quantization cannot fit.
    let smallest = held
        .iter()
        .min_by_key(|quant| quant.bytes)
        .expect("a non-empty set");
    if let Some(free) = hub.free_bytes()
        && smallest.bytes > free
    {
        return Err(refuse(format!(
            "the smallest quantization ({}, {} MiB) needs more than the {} MiB free in {}",
            smallest.label,
            mib(smallest.bytes),
            mib(free),
            hub.cache_dir().display()
        )));
    }
    // The real header of the smallest quantization stands for all of them.
    let header = hub
        .gguf_header_bytes(&wanted.repo, &tree.commit, smallest.first())
        .map_err(&refuse)?;
    let (picked, why) = pick(
        &wanted.repo,
        hub.cache_dir(),
        &held,
        wanted.quant.as_deref(),
        hub.free_bytes(),
        &header,
        planner,
    )?;
    say(&announce("fetch", &wanted.repo, &held, &picked, &why));
    let chosen: Vec<String> = picked.files.iter().map(|file| file.path.clone()).collect();
    let snapshot = hub
        .download_and_wait(
            &wanted.repo,
            "main",
            &|file| chosen.contains(&file.path),
            &mut |state| {
                if let JobState::Downloading {
                    bytes,
                    bytes_total,
                    current_file,
                    ..
                } = state
                {
                    say(&format!(
                        "fetch: {current_file}: {} of {} MiB",
                        mib(*bytes),
                        mib(*bytes_total)
                    ));
                }
            },
        )
        .map_err(&refuse)?;
    Ok(snapshot.join(&picked.first().path))
}

/// An exclusive advisory lock on a directory, taken without waiting.
fn try_lock(dir: &Path) -> Option<std::fs::File> {
    let file = std::fs::File::open(dir).ok()?;
    rustix::fs::flock(&file, rustix::fs::FlockOperation::NonBlockingLockExclusive)
        .ok()
        .map(|()| file)
}

/// Removes the planning stubs nobody is using: a `.fifi-plan-<pid>` directory is locked by the
/// process that plans in it for as long as it plans, so one whose lock can be taken belongs to a
/// process that is gone. The lock, unlike a process-table lookup, holds for processes in other
/// containers that share the cache.
fn sweep_stale_plans(cache: &Path) {
    for entry in std::fs::read_dir(cache).into_iter().flatten().flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        let owner = name
            .strip_prefix(".fifi-plan-")
            .and_then(|pid| pid.parse::<u32>().ok());
        if owner.is_none() || owner == Some(std::process::id()) {
            continue;
        }
        if let Some(_held) = try_lock(&entry.path()) {
            let _ = std::fs::remove_dir_all(entry.path());
        }
    }
}

/// Stub weight files under the hub cache: each holds the real GGUF header and has the length of
/// the quantization it stands for (a sparse file, no bytes written past the header).
struct Stubs {
    dir: PathBuf,
    header: Vec<u8>,
    /// Holds the planning lock on `dir` until the stubs are dropped.
    _lock: std::fs::File,
}

impl Stubs {
    fn new(cache: &Path, header: &[u8]) -> Result<Self, StartupError> {
        sweep_stale_plans(cache);
        let dir = cache.join(format!(".fifi-plan-{}", std::process::id()));
        std::fs::create_dir_all(&dir)
            .map_err(|error| StartupError(format!("cannot plan in {}: {error}", dir.display())))?;
        let lock = try_lock(&dir).ok_or_else(|| {
            StartupError(format!(
                "cannot lock the planning directory {}",
                dir.display()
            ))
        })?;
        Ok(Self {
            dir,
            header: header.to_vec(),
            _lock: lock,
        })
    }

    fn path(&self, length: u64, label: &str) -> Result<PathBuf, StartupError> {
        let dir = self.dir.join(label);
        let file = dir.join("model.gguf");
        let write = || -> std::io::Result<()> {
            std::fs::create_dir_all(&dir)?;
            let stub = std::fs::File::create(&file)?;
            std::io::Write::write_all(&mut &stub, &self.header)?;
            stub.set_len(length)
        };
        write()
            .map_err(|error| StartupError(format!("cannot plan in {}: {error}", file.display())))?;
        Ok(file)
    }
}

impl Drop for Stubs {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn file(path: &str, size: u64) -> HubFile {
        HubFile {
            path: path.to_string(),
            size,
            sha256: None,
        }
    }

    #[test]
    fn a_repo_name_with_or_without_a_quantization_is_read_and_anything_else_is_not() {
        assert_eq!(
            parse_wanted("unsloth/Qwen3.6-35B-A3B-GGUF"),
            Some(Wanted {
                repo: "unsloth/Qwen3.6-35B-A3B-GGUF".into(),
                quant: None
            })
        );
        assert_eq!(
            parse_wanted("unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_XL"),
            Some(Wanted {
                repo: "unsloth/Qwen3.6-35B-A3B-GGUF".into(),
                quant: Some("UD-Q4_K_XL".into())
            })
        );
        // A plain name, an Ollama tag, a path and a bad label are not fetch names.
        for text in [
            "qwen3.6",
            "qwen3.6:35b-a3b",
            "./a/b",
            "/a/b",
            "org/repo:",
            "org/repo:a b",
            "a/b/c",
        ] {
            assert_eq!(parse_wanted(text), None, "{text}");
        }
    }

    #[test]
    fn labels_are_read_from_names_shards_and_directories() {
        for (path, label) in [
            ("Qwen3.6-35B-A3B-Q4_K_M.gguf", "Q4_K_M"),
            ("Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf", "UD-Q4_K_XL"),
            ("model.Q8_0.gguf", "Q8_0"),
            ("model-BF16.gguf", "BF16"),
            (
                "UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00004.gguf",
                "UD-IQ3_XXS",
            ),
            ("IQ4_XS/weights-00001-of-00002.gguf", "IQ4_XS"),
        ] {
            assert_eq!(quant_label(path).as_deref(), Some(label), "{path}");
        }
        assert_eq!(quant_label("README.md"), None);
        assert_eq!(quant_label("tokenizer.gguf"), None);
    }

    #[test]
    fn a_listing_becomes_quantizations_with_shards_summed_and_projectors_left_out() {
        let held = quants(&[
            file("model-Q4_K_M.gguf", 400),
            file("model-Q8_0.gguf", 800),
            file("mmproj-M-F16.gguf", 90),
            file("MTP/mtp-M-F16.gguf", 80),
            file("mtp-M.gguf", 70),
            file("README.md", 1),
            file("Q6_K/model-Q6_K-00001-of-00002.gguf", 300),
            file("Q6_K/model-Q6_K-00002-of-00002.gguf", 250),
        ]);
        let summary: Vec<(&str, u64, usize)> = held
            .iter()
            .map(|quant| (quant.label.as_str(), quant.bytes, quant.files.len()))
            .collect();
        assert_eq!(
            summary,
            [("Q4_K_M", 400, 1), ("Q6_K", 550, 2), ("Q8_0", 800, 1)]
        );
        assert_eq!(held[1].first().path, "Q6_K/model-Q6_K-00001-of-00002.gguf");
    }

    #[test]
    fn the_planner_picks_the_largest_resident_quantization_else_the_smallest() {
        let held = quants(&[
            file("model-Q4_K_M.gguf", 400),
            file("model-Q6_K.gguf", 600),
            file("model-Q8_0.gguf", 800),
        ]);
        let resident_up_to = |limit: u64| {
            move |quant: &Quant| {
                Ok(if quant.bytes <= limit {
                    Fit::Resident
                } else {
                    Fit::NotResident
                })
            }
        };
        let (picked, why) = choose(&held, None, None, resident_up_to(650)).expect("choice");
        assert_eq!(picked.label, "Q6_K");
        assert!(why.contains("largest of 3"), "{why}");
        let (picked, why) = choose(&held, None, None, resident_up_to(100)).expect("choice");
        assert_eq!(picked.label, "Q4_K_M");
        assert!(why.contains("lower rung"), "{why}");
        // A label asked for wins over the planner, whatever its case.
        let (picked, _) = choose(&held, Some("q8_0"), None, resident_up_to(100)).expect("choice");
        assert_eq!(picked.label, "Q8_0");
        let error = choose(&held, Some("Q2_K"), None, resident_up_to(0)).unwrap_err();
        assert!(error.contains("Q4_K_M, Q6_K, Q8_0"), "{error}");
    }

    #[test]
    fn a_listing_offers_only_complete_non_empty_sets() {
        let held = quants(&[
            file("a-Q4_K_M.gguf", 0),
            file("b-Q6_K-00001-of-00003.gguf", 10),
            file("b-Q6_K-00003-of-00003.gguf", 10),
            file("c-Q8_0-00001-of-00002.gguf", 10),
            file("c-Q8_0-00002-of-00002.gguf", 10),
            // The same stem with another shard count is another set.
            file("c-Q8_0-00001-of-00003.gguf", 10),
        ]);
        let labels: Vec<&str> = held.iter().map(|quant| quant.label.as_str()).collect();
        assert_eq!(labels, ["Q8_0"]);
        assert_eq!(held[0].files.len(), 2);
    }

    #[test]
    fn the_planner_is_asked_about_few_quantizations_and_never_about_one_the_disk_cannot_hold() {
        let sizes = [300_u64, 400, 500, 600, 700, 800, 900, 1000];
        let files: Vec<HubFile> = sizes
            .iter()
            .map(|size| file(&format!("m-Q{}_K.gguf", size / 100), *size))
            .collect();
        let held = quants(&files);
        let mut asked = Vec::new();
        let (picked, _) = choose(&held, None, None, |quant| {
            asked.push(quant.bytes);
            Ok(if quant.bytes <= 650 {
                Fit::Resident
            } else {
                Fit::NotResident
            })
        })
        .expect("choice");
        assert_eq!(picked.bytes, 600);
        assert!(asked.len() <= 4, "{asked:?}");
        // With room for 550 bytes only, the larger ones are not asked about at all.
        let mut asked = Vec::new();
        let (picked, _) = choose(&held, None, Some(550), |quant| {
            asked.push(quant.bytes);
            Ok(Fit::Resident)
        })
        .expect("choice");
        assert_eq!(picked.bytes, 500);
        assert!(asked.iter().all(|bytes| *bytes <= 550), "{asked:?}");
        let error = choose(&held, None, Some(100), |_| Ok(Fit::Resident)).unwrap_err();
        assert!(error.contains("fits the free disk"), "{error}");
    }

    #[test]
    fn only_the_stubs_nobody_holds_are_swept() {
        let dir = tempfile::tempdir().expect("tempdir");
        let free = dir.path().join(".fifi-plan-4194999");
        let held = dir.path().join(".fifi-plan-4194998");
        let not_a_pid = dir.path().join(".fifi-plan-abc");
        let other = dir.path().join("models--a--b");
        for path in [&free, &held, &not_a_pid, &other] {
            std::fs::create_dir_all(path.join("x")).expect("dirs");
        }
        // Another process (here, another open file description) is planning in `held`.
        let _holder = try_lock(&held).expect("lock");
        let own = Stubs::new(dir.path(), b"GGUF").expect("stubs");
        assert!(!free.exists());
        assert!(held.exists() && not_a_pid.exists() && other.exists());
        assert!(own.dir.exists());
        // A second planner cannot take a held directory's lock.
        assert!(try_lock(&held).is_none());
    }
}
