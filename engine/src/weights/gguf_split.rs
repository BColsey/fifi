//! File names of a split GGUF set: `<stem>-00001-of-00004.gguf` through
//! `<stem>-00004-of-00004.gguf`, all in one directory.
//!
//! The split tools write five-digit, one-based shard numbers and the shard count in every
//! file name. Only shard 1 carries the model metadata; the other shards hold tensors and the
//! three `split.*` keys.

use std::path::{Path, PathBuf};

/// A file name that follows the split pattern.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ShardName {
    /// The name before `-NNNNN-of-NNNNN.gguf`.
    pub(crate) stem: String,
    /// One-based shard number.
    pub(crate) number: u32,
    pub(crate) count: u32,
    /// The file extension as spelled, `.gguf` or a case variant.
    extension: String,
}

impl ShardName {
    /// Reads `file_name` as a split shard name. `None` for any other name, including a
    /// shard number of zero or above the count.
    pub(crate) fn parse(file_name: &str) -> Option<Self> {
        let (body, extension) = strip_extension(file_name)?;
        let (rest, count) = body.rsplit_once("-of-")?;
        let (stem, number) = rest.rsplit_once('-')?;
        let digits = |text: &str| {
            (text.len() == 5 && text.bytes().all(|b| b.is_ascii_digit()))
                .then(|| text.parse::<u32>().ok())
                .flatten()
        };
        let (number, count) = (digits(number)?, digits(count)?);
        if stem.is_empty() || number == 0 || number > count {
            return None;
        }
        Some(Self {
            stem: stem.to_owned(),
            number,
            count,
            extension: extension.to_owned(),
        })
    }

    /// The file name of shard `number` (one-based) of the same set.
    pub(crate) fn file_name(&self, number: u32) -> String {
        format!(
            "{}-{number:05}-of-{:05}{}",
            self.stem, self.count, self.extension
        )
    }

    /// The path of shard `number` (one-based) next to `any_shard`.
    pub(crate) fn sibling(&self, any_shard: &Path, number: u32) -> PathBuf {
        any_shard.with_file_name(self.file_name(number))
    }
}

fn strip_extension(file_name: &str) -> Option<(&str, &str)> {
    let split = file_name.len().checked_sub(".gguf".len())?;
    let (body, extension) = file_name.split_at_checked(split)?;
    extension
        .eq_ignore_ascii_case(".gguf")
        .then_some((body, extension))
}

/// The shard name of `path`, when its file name follows the split pattern.
pub(crate) fn shard_name_of(path: &Path) -> Option<ShardName> {
    ShardName::parse(path.file_name()?.to_str()?)
}

/// Writes a two-shard set of tensor-less files (`<stem>-00001-of-00002.gguf` and `-00002-`)
/// into `dir` and returns shard 1's path, for tests of readers that must refuse a split set.
#[cfg(test)]
pub(crate) fn write_two_shard_set(dir: &Path, stem: &str) -> PathBuf {
    let string = |text: &str| {
        let mut out = (text.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(text.as_bytes());
        out
    };
    let kv = |key: &str, value: u16| {
        let mut out = string(key);
        out.extend_from_slice(&2u32.to_le_bytes());
        out.extend_from_slice(&value.to_le_bytes());
        out
    };
    let mut first = PathBuf::new();
    for number in 1..=2u16 {
        let mut bytes = b"GGUF".to_vec();
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&3u64.to_le_bytes());
        bytes.extend(kv("split.no", number - 1));
        bytes.extend(kv("split.count", 2));
        bytes.extend(kv("split.tensors.count", 0));
        let path = dir.join(format!("{stem}-{number:05}-of-00002.gguf"));
        std::fs::write(&path, bytes).expect("write shard");
        if number == 1 {
            first = path;
        }
    }
    first
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_shard_name_reads_back_its_parts() {
        let name = ShardName::parse("Model-Q3-00003-of-00004.gguf").expect("split name");
        assert_eq!(name.stem, "Model-Q3");
        assert_eq!((name.number, name.count), (3, 4));
        assert_eq!(name.file_name(1), "Model-Q3-00001-of-00004.gguf");
        let sibling = name.sibling(Path::new("/m/dir/a-00003-of-00004.gguf"), 2);
        assert_eq!(sibling, Path::new("/m/dir/Model-Q3-00002-of-00004.gguf"));
    }

    #[test]
    fn names_outside_the_pattern_are_not_shards() {
        for name in [
            "model.gguf",
            "model-1-of-4.gguf",
            "model-00000-of-00004.gguf",
            "model-00005-of-00004.gguf",
            "-00001-of-00004.gguf",
            "model-00001-of-00004.bin",
            "model-0000a-of-00004.gguf",
            "model-00001-of-004.gguf",
            "",
            ".gguf",
        ] {
            assert_eq!(ShardName::parse(name), None, "{name:?}");
        }
    }

    #[test]
    fn the_extension_is_case_insensitive_and_non_ascii_stems_parse() {
        let name = ShardName::parse("模型-00002-of-00002.GGUF").expect("split name");
        assert_eq!(
            (name.stem.as_str(), name.number, name.count),
            ("模型", 2, 2)
        );
        assert_eq!(name.file_name(1), "模型-00001-of-00002.GGUF");
    }
}
