//! Choosing the weights file of a model path that is a `.gguf` file or a directory holding
//! several of them.
//!
//! A model path names either one `.gguf` file or a directory. A directory with several files
//! (one model in several quantizations) is ranked by file size, largest first: for one model the
//! size orders the quantizations by bits per weight. The first file that the caller's memory
//! plan admits is served. The choice is deterministic: equal sizes order by path.

#![cfg_attr(
    all(not(test), not(feature = "native-cuda")),
    expect(
        dead_code,
        reason = "the host build only reads the directory of a model path; the CUDA loaders choose"
    )
)]

use crate::{EngError, weights::shard_name_of};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// One candidate weights file. A split set (`<name>-00001-of-0000N.gguf` and its siblings)
/// is one candidate: `path` is its first present shard and `bytes` the sum over the present
/// shards.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct GgufFile {
    pub(crate) path: PathBuf,
    pub(crate) bytes: u64,
}

impl GgufFile {
    fn name(&self) -> String {
        self.path.file_name().map_or_else(
            || self.path.display().to_string(),
            |name| name.to_string_lossy().into_owned(),
        )
    }

    fn mib(&self) -> u64 {
        self.bytes / (1 << 20)
    }
}

pub(crate) fn is_gguf(path: &Path) -> bool {
    path.extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"))
}

/// Whether `path` is a GGUF file: named `.gguf`, or holding the GGUF magic whatever its name
/// (an Ollama blob is `sha256-<hex>` with no extension).
pub(crate) fn is_gguf_file(path: &Path) -> bool {
    if !path.is_file() {
        return false;
    }
    if is_gguf(path)
        || path
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf_file"))
    {
        return true;
    }
    let mut magic = [0u8; 4];
    std::fs::File::open(path)
        .and_then(|mut file| std::io::Read::read_exact(&mut file, &mut magic))
        .is_ok()
        && &magic == b"GGUF"
}

/// The first shard of the split set that `path` belongs to, when it is a shard and the first
/// exists; any other path unchanged.
pub(crate) fn first_shard(path: &Path) -> PathBuf {
    match shard_name_of(path) {
        Some(name) if name.number != 1 => {
            let first = name.sibling(path, 1);
            if first.is_file() {
                first
            } else {
                path.to_path_buf()
            }
        }
        _ => path.to_path_buf(),
    }
}

/// The directory of a model path: the path itself, or the parent of a path to a GGUF file.
pub(crate) fn directory_of(model: &Path) -> PathBuf {
    if is_gguf_file(model) {
        match model.parent() {
            Some(parent) if !parent.as_os_str().is_empty() => return parent.to_path_buf(),
            _ => return PathBuf::from("."),
        }
    }
    model.to_path_buf()
}

/// The directory that holds the model's side files (configuration, tokenizer) and the weights
/// candidates ranked from the highest precision down. A path to one `.gguf` file gives that file
/// alone and its parent directory; a directory gives every `.gguf` file in it.
pub(crate) fn candidates(model: &Path) -> Result<(PathBuf, Vec<GgufFile>), EngError> {
    let read = |error: std::io::Error, what: &Path| {
        EngError::Checkpoint(format!("model path {}: {error}", what.display()))
    };
    if model.is_file() {
        if !is_gguf_file(model) {
            return Err(EngError::Checkpoint(format!(
                "model path {} is a file that does not start with the GGUF magic; name a GGUF \
                 file or the directory that holds it",
                model.display()
            )));
        }
        let first = first_shard(model);
        let model = first.as_path();
        let mut bytes = std::fs::metadata(model)
            .map_err(|error| read(error, model))?
            .len();
        if let Some(name) = shard_name_of(model) {
            bytes = (1..=name.count)
                .filter(|&number| number != name.number)
                .filter_map(|number| std::fs::metadata(name.sibling(model, number)).ok())
                .filter(std::fs::Metadata::is_file)
                .fold(bytes, |sum, meta| sum + meta.len());
        }
        let dir = match model.parent() {
            Some(parent) if !parent.as_os_str().is_empty() => parent.to_path_buf(),
            _ => PathBuf::from("."),
        };
        return Ok((
            dir,
            vec![GgufFile {
                path: model.to_path_buf(),
                bytes,
            }],
        ));
    }
    let mut files = Vec::new();
    // Shards of one set, keyed by (stem, count): the lowest shard number seen and the bytes.
    let mut sets: BTreeMap<(String, u32), (u32, GgufFile)> = BTreeMap::new();
    for entry in std::fs::read_dir(model).map_err(|error| read(error, model))? {
        let path = entry.map_err(|error| read(error, model))?.path();
        if is_gguf(&path) && path.is_file() {
            let bytes = std::fs::metadata(&path)
                .map_err(|error| read(error, &path))?
                .len();
            match shard_name_of(&path) {
                Some(name) => {
                    let slot = sets.entry((name.stem, name.count)).or_insert_with(|| {
                        (
                            name.number,
                            GgufFile {
                                path: path.clone(),
                                bytes: 0,
                            },
                        )
                    });
                    if name.number < slot.0 {
                        *slot = (
                            name.number,
                            GgufFile {
                                path,
                                bytes: slot.1.bytes,
                            },
                        );
                    }
                    slot.1.bytes += bytes;
                }
                None => files.push(GgufFile { path, bytes }),
            }
        }
    }
    files.extend(sets.into_values().map(|(_, file)| file));
    if files.is_empty() {
        return Err(EngError::Checkpoint(format!(
            "no .gguf weights file in {}",
            model.display()
        )));
    }
    files.sort_by(|a, b| b.bytes.cmp(&a.bytes).then_with(|| a.path.cmp(&b.path)));
    Ok((model.to_path_buf(), files))
}

/// The `general.architecture` string of one GGUF file, read from its header alone.
fn architecture_of(path: &Path) -> Result<Option<String>, EngError> {
    use crate::weights::{GgufHeader, MetaValue};
    let unreadable = |error: std::io::Error| {
        EngError::Checkpoint(format!("model file {}: {error}", path.display()))
    };
    let file = std::fs::File::open(path).map_err(unreadable)?;
    let len = file.metadata().map_err(unreadable)?.len();
    let header = GgufHeader::parse(&mut std::io::BufReader::new(file), len, 256 << 20)?;
    Ok(match header.meta("general.architecture") {
        Some(MetaValue::String(architecture)) => Some(architecture.clone()),
        _ => None,
    })
}

/// The weights files of `model` (a `.gguf` file or a directory) whose `general.architecture` is
/// `architecture`, ranked like [`candidates`]. A directory can hold files of other roles beside
/// the model (a draft model, say); they are skipped. An error names every architecture found when
/// none matches.
pub(crate) fn of_architecture(model: &Path, architecture: &str) -> Result<Vec<GgufFile>, EngError> {
    let (_, files) = candidates(model)?;
    let mut found = Vec::new();
    let mut kept = Vec::new();
    for file in files {
        match architecture_of(&file.path)? {
            Some(name) if name == architecture => kept.push(file),
            Some(name) => found.push(format!("{} is {name}", file.name())),
            None => found.push(format!("{} has no architecture", file.name())),
        }
    }
    if kept.is_empty() {
        return Err(EngError::Checkpoint(format!(
            "no .gguf file of architecture {architecture} in {}: {}",
            model.display(),
            found.join(", ")
        )));
    }
    Ok(kept)
}

/// The chosen file and the log line that names it and the other candidates.
#[derive(Debug)]
pub(crate) struct Choice {
    pub(crate) chosen: GgufFile,
    /// `None` when the path held one file, so there was nothing to choose.
    pub(crate) line: Option<String>,
}

/// Picks the first of `files` (ranked, highest precision first) that `admit` accepts. `admit`
/// returns the reason when the memory plan refuses a file. One file is returned without asking
/// `admit`: its own load reports the refusal in full. When every file is refused the error
/// carries each reason.
pub(crate) fn choose(
    files: &[GgufFile],
    admit: &dyn Fn(&GgufFile) -> Result<(), String>,
) -> Result<Choice, EngError> {
    if let [only] = files {
        return Ok(Choice {
            chosen: only.clone(),
            line: None,
        });
    }
    let mut refused = Vec::new();
    for (index, file) in files.iter().enumerate() {
        match admit(file) {
            Ok(()) => {
                let mut others = refused.clone();
                others.extend(
                    files[index + 1..]
                        .iter()
                        .map(|other| format!("{} ({} MiB)", other.name(), other.mib())),
                );
                let line = format!(
                    "gguf: serving {} ({} MiB), the highest-precision of {} files that the memory \
                     plan admits; other files: {}",
                    file.name(),
                    file.mib(),
                    files.len(),
                    others.join(", ")
                );
                return Ok(Choice {
                    chosen: file.clone(),
                    line: Some(line),
                });
            }
            Err(reason) => refused.push(format!(
                "{} ({} MiB) refused: {reason}",
                file.name(),
                file.mib()
            )),
        }
    }
    Err(EngError::MemoryPlan(format!(
        "no .gguf file of the model directory fits the memory plan: {}",
        refused.join("; ")
    )))
}

#[cfg(test)]
mod tests {
    use super::{GgufFile, candidates, choose, of_architecture};
    use std::path::Path;

    fn write(dir: &Path, name: &str, bytes: usize) {
        std::fs::write(dir.join(name), vec![0u8; bytes]).expect("write");
    }

    fn names(files: &[GgufFile]) -> Vec<String> {
        files.iter().map(GgufFile::name).collect()
    }

    #[test]
    fn a_directory_ranks_its_files_from_the_largest_down() {
        let dir = tempfile::tempdir().expect("tempdir");
        write(dir.path(), "m-q4_k_m.gguf", 50);
        write(dir.path(), "m-q8_0.gguf", 80);
        write(dir.path(), "m-q4_k.gguf", 50);
        write(dir.path(), "m-q4_k_pure.gguf", 40);
        write(dir.path(), "notes.txt", 500);
        let (found, files) = candidates(dir.path()).expect("candidates");
        assert_eq!(found, dir.path());
        assert_eq!(
            names(&files),
            [
                "m-q8_0.gguf",
                "m-q4_k.gguf",
                "m-q4_k_m.gguf",
                "m-q4_k_pure.gguf"
            ]
        );
    }

    #[test]
    fn the_shards_of_one_set_are_one_candidate_sized_by_the_sum() {
        let dir = tempfile::tempdir().expect("tempdir");
        write(dir.path(), "big-00002-of-00003.gguf", 30);
        write(dir.path(), "big-00001-of-00003.gguf", 5);
        write(dir.path(), "big-00003-of-00003.gguf", 40);
        write(dir.path(), "small.gguf", 60);
        // Another set of the same stem with a different count stays apart.
        write(dir.path(), "big-00001-of-00002.gguf", 7);
        let (_, files) = candidates(dir.path()).expect("candidates");
        assert_eq!(
            names(&files),
            [
                "big-00001-of-00003.gguf",
                "small.gguf",
                "big-00001-of-00002.gguf"
            ]
        );
        assert_eq!(files[0].bytes, 75);
        assert_eq!(files[2].bytes, 7);
    }

    #[test]
    fn an_incomplete_set_is_still_a_candidate_named_by_its_first_present_shard() {
        let dir = tempfile::tempdir().expect("tempdir");
        write(dir.path(), "m-00002-of-00004.gguf", 30);
        write(dir.path(), "m-00004-of-00004.gguf", 20);
        let (_, files) = candidates(dir.path()).expect("candidates");
        assert_eq!(names(&files), ["m-00002-of-00004.gguf"]);
        assert_eq!(files[0].bytes, 50);
    }

    #[test]
    fn a_path_to_one_shard_is_sized_by_every_shard_beside_it() {
        let dir = tempfile::tempdir().expect("tempdir");
        write(dir.path(), "m-00001-of-00002.gguf", 10);
        write(dir.path(), "m-00002-of-00002.gguf", 25);
        let (found, files) = candidates(&dir.path().join("m-00002-of-00002.gguf")).expect("shard");
        assert_eq!(found, dir.path());
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].bytes, 35);
    }

    #[test]
    fn a_path_to_one_gguf_file_names_that_file_and_its_directory() {
        let dir = tempfile::tempdir().expect("tempdir");
        write(dir.path(), "m-q8_0.gguf", 80);
        write(dir.path(), "m-q4_k.gguf", 50);
        let file = dir.path().join("m-q4_k.gguf");
        let (found, files) = candidates(&file).expect("candidates");
        assert_eq!(found, dir.path());
        assert_eq!(names(&files), ["m-q4_k.gguf"]);
        let choice = choose(&files, &|_| Err("never asked".into())).expect("one file");
        assert_eq!(choice.chosen.path, file);
        assert!(choice.line.is_none());
    }

    #[test]
    fn a_directory_without_a_gguf_file_and_a_file_of_another_kind_refuse() {
        let dir = tempfile::tempdir().expect("tempdir");
        write(dir.path(), "config.json", 2);
        let message = candidates(dir.path()).unwrap_err().describe();
        assert!(message.contains("no .gguf weights file"), "{message}");
        let message = candidates(&dir.path().join("config.json"))
            .unwrap_err()
            .describe();
        assert!(message.contains("GGUF magic"), "{message}");
    }

    #[test]
    fn the_highest_precision_file_that_the_plan_admits_wins_and_the_line_names_the_rest() {
        let dir = tempfile::tempdir().expect("tempdir");
        write(dir.path(), "m-q8_0.gguf", 80);
        write(dir.path(), "m-q4_k_m.gguf", 50);
        write(dir.path(), "m-q4_k_pure.gguf", 40);
        let (_, files) = candidates(dir.path()).expect("candidates");

        let roomy = choose(&files, &|_| Ok(())).expect("choice");
        assert_eq!(roomy.chosen.name(), "m-q8_0.gguf");
        let line = roomy.line.expect("line");
        assert!(line.contains("m-q4_k_m.gguf") && line.contains("m-q4_k_pure.gguf"));

        let tight = choose(&files, &|file| {
            if file.bytes > 60 {
                Err("needs 80 bytes".into())
            } else {
                Ok(())
            }
        })
        .expect("choice");
        assert_eq!(tight.chosen.name(), "m-q4_k_m.gguf");
        let line = tight.line.expect("line");
        assert!(line.contains("m-q8_0.gguf") && line.contains("refused: needs 80 bytes"));
        assert!(line.contains("m-q4_k_pure.gguf"));

        let none = choose(&files, &|_| Err("full".into()))
            .unwrap_err()
            .describe();
        assert!(
            none.contains("m-q8_0.gguf") && none.contains("m-q4_k_pure.gguf"),
            "{none}"
        );
    }

    #[test]
    fn the_raon_opener_refuses_a_split_set() {
        let dir = tempfile::tempdir().expect("tempdir");
        let first = crate::weights::write_two_shard_set(dir.path(), "raon");
        let error = crate::family::raon::raon1::open_raon_1_mini_preview(&first, None)
            .map(|_| ())
            .expect_err("a split set is refused");
        assert!(
            error.describe().contains("does not take a split GGUF set"),
            "{}",
            error.describe()
        );
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
    fn a_file_with_the_gguf_magic_is_a_model_whatever_it_is_named() {
        let dir = tempfile::tempdir().expect("tempdir");
        let blob = dir.path().join("sha256-0123abcd");
        std::fs::write(&blob, gguf_bytes(64)).expect("write");
        let other = dir.path().join("notes.txt");
        std::fs::write(&other, b"hello world").expect("write");
        assert!(super::is_gguf_file(&blob));
        assert!(!super::is_gguf_file(&other));
        assert!(!super::is_gguf_file(dir.path()));
        assert_eq!(super::directory_of(&blob), dir.path());
        let (found, files) = super::candidates(&blob).expect("a blob is a candidate");
        assert_eq!(found, dir.path());
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].path, blob);
        let error = super::candidates(&other)
            .expect_err("not a GGUF")
            .describe();
        assert!(error.contains("GGUF magic"), "{error}");
    }

    #[test]
    fn any_shard_of_a_split_set_names_the_whole_set_from_its_first_shard() {
        let dir = tempfile::tempdir().expect("tempdir");
        for number in 1..=3 {
            std::fs::write(
                dir.path().join(format!("m-{number:05}-of-00003.gguf")),
                gguf_bytes(100 * number),
            )
            .expect("write");
        }
        let third = dir.path().join("m-00003-of-00003.gguf");
        let first = dir.path().join("m-00001-of-00003.gguf");
        assert_eq!(super::first_shard(&third), first);
        let (_, files) = super::candidates(&third).expect("candidates");
        assert_eq!(
            files,
            vec![super::GgufFile {
                path: first,
                bytes: 600
            }]
        );
    }

    #[test]
    fn files_of_other_architectures_are_skipped_and_a_directory_without_a_match_names_them() {
        let dir = tempfile::tempdir().expect("tempdir");
        let tiny = |architecture: &str| {
            crate::tokenizer::gguf::tests::tiny_gguf(architecture, "gpt2", "raon")
        };
        std::fs::write(dir.path().join("model.gguf"), tiny("raon1")).expect("write");
        std::fs::write(dir.path().join("draft.gguf"), tiny("draft")).expect("write");
        let files = of_architecture(dir.path(), "raon1").expect("one match");
        assert_eq!(names(&files), ["model.gguf"]);
        let error = of_architecture(dir.path(), "llama").unwrap_err().describe();
        assert!(
            error.contains("model.gguf is raon1") && error.contains("draft.gguf is draft"),
            "{error}"
        );
    }
}
