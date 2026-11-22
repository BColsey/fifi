//! The identity digest of a safetensors pack directory.

use crate::EngError;
use std::path::Path;

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
pub(crate) fn pack_digest(pack: &Path) -> Result<[u8; 32], EngError> {
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
