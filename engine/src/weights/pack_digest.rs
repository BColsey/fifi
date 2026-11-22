//! The identity digest of a safetensors pack directory.

use crate::EngError;
use std::path::Path;

/// Whether the files whose names start with a dot count in a pack's digest.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Hidden {
    Counted,
    Skipped,
}

/// A SHA-256 over the loaded EXL3 pack's own metadata —
/// `config.json` and the weight index read verbatim, then every other
/// regular file of the pack by name and size in sorted order. The download
/// manifest is deliberately not an input: it names what was downloaded, not
/// what this loader opened, and two different packs can share one manifest.
/// Constant for one loaded pack, so folding it into a path leaves live
/// save/restore behaviour unchanged; a persisted store refuses entries
/// another load wrote. The weights half is layout plus sampled content —
/// file names and sizes, then the first and last 64 KiB of every weight
/// file — not a full digest.
///
/// `Hidden::Skipped` leaves out files whose names start with a dot (a Finder's `.DS_Store`
/// changes with a browse of the folder, and the identity must not); `Hidden::Counted` keeps the
/// digest of every pack already named by it.
pub(crate) fn pack_digest(pack: &Path, hidden: Hidden) -> Result<[u8; 32], EngError> {
    use sha2::{Digest, Sha256};
    let pack_error = |message: String| EngError::Checkpoint(format!("pack identity: {message}"));
    let mut hasher = Sha256::new();
    hasher.update(b"fifi-flash-pack-v1\0");
    for name in ["config.json", "model.safetensors.index.json"] {
        let bytes = std::fs::read(pack.join(name))
            .map_err(|error| pack_error(format!("{name}: {error}")))?;
        hasher.update((bytes.len() as u64).to_le_bytes());
        hasher.update(&bytes);
    }
    let mut rest: Vec<(String, u64)> = Vec::new();
    let entries = std::fs::read_dir(pack).map_err(|error| pack_error(error.to_string()))?;
    for entry in entries {
        let entry = entry.map_err(|error| pack_error(error.to_string()))?;
        let name = entry.file_name().to_string_lossy().into_owned();
        if name == "config.json" || name == "model.safetensors.index.json" {
            continue;
        }
        if hidden == Hidden::Skipped && name.starts_with('.') {
            continue;
        }
        // std::fs::metadata follows symlinks: a pack whose shards are
        // symlinks (the Hugging Face cache layout) still counts them.
        let meta = std::fs::metadata(entry.path())
            .map_err(|error| pack_error(format!("{name}: {error}")))?;
        if meta.is_file() {
            rest.push((name, meta.len()));
        }
    }
    rest.sort();
    if rest.is_empty() {
        return Err(pack_error(
            "no weight files beside the manifest: refusing an empty identity".into(),
        ));
    }
    for (name, len) in &rest {
        hasher.update((name.len() as u64).to_le_bytes());
        hasher.update(name.as_bytes());
        hasher.update(len.to_le_bytes());
        super::hash_sampled_edges(&mut hasher, &pack.join(name), 0, *len)
            .map_err(|error| pack_error(format!("{name}: {error}")))?;
    }
    Ok(hasher.finalize().into())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pack(with_hidden: bool) -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        let put = |name: &str, bytes: &[u8]| std::fs::write(dir.path().join(name), bytes).unwrap();
        put("config.json", br#"{"model_type":"toy"}"#);
        put("model.safetensors.index.json", br#"{"weight_map":{}}"#);
        put("model-00001-of-00002.safetensors", &[7u8; 300]);
        put("model-00002-of-00002.safetensors", &[9u8; 200]);
        put("tokenizer.json", b"{}");
        if with_hidden {
            put(".DS_Store", &[1u8; 6148]);
        }
        dir
    }

    fn hex(digest: [u8; 32]) -> String {
        digest.iter().map(|byte| format!("{byte:02x}")).collect()
    }

    #[test]
    fn a_pack_digest_has_fixed_values_and_a_hidden_file_moves_only_the_counted_one() {
        let plain = pack(false);
        let browsed = pack(true);
        let counted =
            |dir: &tempfile::TempDir| hex(pack_digest(dir.path(), Hidden::Counted).unwrap());
        let skipped =
            |dir: &tempfile::TempDir| hex(pack_digest(dir.path(), Hidden::Skipped).unwrap());
        assert_eq!(
            counted(&plain),
            GOLDEN_PLAIN,
            "the digest of a pack with no hidden file is the one every pack already has"
        );
        assert_eq!(
            skipped(&plain),
            GOLDEN_PLAIN,
            "no hidden file, no difference"
        );
        assert_ne!(
            counted(&browsed),
            GOLDEN_PLAIN,
            "a counted hidden file changes the digest"
        );
        assert_eq!(skipped(&browsed), GOLDEN_PLAIN, "a skipped one does not");
    }

    const GOLDEN_PLAIN: &str = "9b9f12c81f2d1f686b015237fb089e538b676c56dd3c9c95672db259b1109712";
}
