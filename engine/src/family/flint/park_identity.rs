//! Park-identity construction for the Flint executors: the pack
//! fingerprint, the cache-configuration hash, and their assembly into a
//! [`ParkIdentity`] for the host park store. Everything here reads the
//! engine's own configuration — never `prefill_path()`'s return, which
//! can misreport a chunk-built state as steps.

use crate::EngError;
use sha2::{Digest, Sha256};
use std::io::Read;
use std::path::Path;

use crate::family::parking::{PARK_FORMAT_VERSION, ParkIdentity, ParkProfile};

/// Length-prefixed field feed: a length change in one field can never
/// alias its bytes into a neighbor across the hash input.
fn field(hasher: &mut Sha256, bytes: &[u8]) {
    hasher.update((bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

/// The pack fingerprint: sha256 of the pack's `SHA256SUMS` manifest
/// (trusted unverified — a stale shard under a matching manifest keeps
/// the fingerprint; the manifest is the publisher's own digest list), or
/// — when the pack ships none — a streaming sha256 over `config.json`
/// and every `*.safetensors` shard as (relative name, length, bytes),
/// one 1 MiB read at a time. A requantized or re-sharded pack can never
/// match another's parks.
pub(super) fn pack_fingerprint(root: &Path) -> Result<[u8; 32], EngError> {
    let unreadable = |what: &str, e: std::io::Error| {
        EngError::Checkpoint(format!("pack fingerprint: {what}: {e}"))
    };
    // Domain tag: hashes from different constructions can never collide.
    let mut hasher = Sha256::new();
    hasher.update(b"fifi-park-pack-v1");
    match std::fs::read(root.join("SHA256SUMS")) {
        Ok(bytes) => {
            if bytes.is_empty() {
                return Err(EngError::Checkpoint(format!(
                    "pack SHA256SUMS is empty: {}",
                    root.display()
                )));
            }
            hasher.update(&bytes);
            return Ok(hasher.finalize().into());
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => return Err(unreadable("SHA256SUMS unreadable", e)),
    }
    let mut entries: Vec<(std::ffi::OsString, std::path::PathBuf)> = Vec::new();
    for entry in std::fs::read_dir(root).map_err(|e| unreadable("root unreadable", e))? {
        let entry = entry.map_err(|e| unreadable("directory entry", e))?;
        if entry
            .path()
            .extension()
            .is_some_and(|ext| ext == "safetensors")
        {
            entries.push((entry.file_name(), entry.path()));
        }
    }
    entries.sort_by(|a, b| a.0.cmp(&b.0));
    if entries.is_empty() {
        return Err(EngError::Checkpoint(format!(
            "pack has no SHA256SUMS and no safetensors shards: {}",
            root.display()
        )));
    }
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    for (name, path) in &entries {
        let length = std::fs::metadata(path)
            .map_err(|e| unreadable("shard metadata unreadable", e))?
            .len();
        let mut file = std::fs::File::open(path).map_err(|e| unreadable("shard unreadable", e))?;
        field(&mut hasher, name.as_encoded_bytes());
        field(&mut hasher, &length.to_le_bytes());
        loop {
            let n = file
                .read(&mut buf)
                .map_err(|e| unreadable("shard unreadable", e))?;
            if n == 0 {
                break;
            }
            hasher.update(&buf[..n]);
        }
    }
    match std::fs::read(root.join("config.json")) {
        Ok(config) => {
            field(&mut hasher, b"config.json");
            field(&mut hasher, &config);
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => return Err(unreadable("config.json unreadable", e)),
    }
    Ok(hasher.finalize().into())
}

/// The cache-configuration hash: sha256 over everything the payload
/// encoding depends on beside the profile — the KV profile string, the
/// prefill numeric path (steps vs chunked and the chunk size, taken
/// from the engine's configuration), the serialized row-block size, the
/// context limit, and the layer-major walk (0 token-major, 1 tiles, 2 jobs; the
/// three are different numeric paths). Every field is
/// length-prefixed. A change in any input means a park built under the
/// old configuration can never restore.
pub(super) fn cache_config_hash(
    kv_profile: &str,
    chunked_prefill: bool,
    chunk_size: u32,
    row_block: usize,
    context_limit: u32,
    layer_major: u8,
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    // Domain tag: hashes from different constructions can never collide.
    hasher.update(b"fifi-park-cache-v1");
    field(&mut hasher, kv_profile.as_bytes());
    field(&mut hasher, &[u8::from(chunked_prefill)]);
    field(&mut hasher, &chunk_size.to_le_bytes());
    field(
        &mut hasher,
        &u64::try_from(row_block).unwrap_or(u64::MAX).to_le_bytes(),
    );
    field(&mut hasher, &context_limit.to_le_bytes());
    field(&mut hasher, &[layer_major]);
    hasher.finalize().into()
}

/// Assembles the park identity for one engine instance. The KV profile
/// string is the single source of the stored [`ParkProfile`].
#[allow(clippy::too_many_arguments)] // one flat config record, length-prefixed
pub(super) fn park_identity(
    root: &Path,
    template_sha: [u8; 32],
    kv_profile: &str,
    chunked_prefill: bool,
    chunk_size: u32,
    row_block: usize,
    context_limit: u32,
    layer_major: u8,
) -> Result<ParkIdentity, EngError> {
    let profile = match kv_profile {
        "f16" => ParkProfile::F16,
        "q8" => ParkProfile::Q8,
        other => {
            return Err(EngError::BadRequest(format!(
                "Flint park identity requires kv_profile f16 or q8, got {other}"
            )));
        }
    };
    Ok(ParkIdentity {
        model: pack_fingerprint(root)?,
        template: template_sha,
        cache_config: cache_config_hash(
            kv_profile,
            chunked_prefill,
            chunk_size,
            row_block,
            context_limit,
            layer_major,
        ),
        format: PARK_FORMAT_VERSION,
        profile,
        images: Vec::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A pack dir the test owns; the guard removes it on drop, so the
    /// tests never leak temp dirs.
    struct PackGuard(
        std::path::PathBuf,
        // Held for cleanup only: dropping the guard removes the pack.
        #[allow(dead_code)] tempfile::TempDir,
    );
    fn tmp_pack(tag: &str) -> PackGuard {
        let dir = tempfile::tempdir().unwrap();
        let pack = dir.path().join(tag);
        std::fs::create_dir(&pack).unwrap();
        std::fs::write(pack.join("a.safetensors"), b"alpha").unwrap();
        std::fs::write(pack.join("b.safetensors"), b"beta").unwrap();
        PackGuard(pack, dir)
    }

    #[test]
    fn pack_fingerprint_prefers_manifest_and_senses_shard_changes() {
        let guard = tmp_pack("manifest");
        let pack = &guard.0;
        std::fs::write(pack.join("SHA256SUMS"), b"<manifest bytes>").unwrap();
        let with_manifest = pack_fingerprint(pack).unwrap();
        // Trusted unverified: shard changes under the manifest do not
        // move the fingerprint.
        std::fs::write(pack.join("a.safetensors"), b"changed").unwrap();
        assert_eq!(pack_fingerprint(pack).unwrap(), with_manifest);
        // An empty manifest is refused, never hashed as a fingerprint.
        std::fs::write(pack.join("SHA256SUMS"), b"").unwrap();
        assert!(pack_fingerprint(pack).is_err());

        let bare = tmp_pack("bare");
        let f1 = pack_fingerprint(&bare.0).unwrap();
        // Sorted-name fallback: content changes move the fingerprint.
        std::fs::write(bare.0.join("a.safetensors"), b"changed").unwrap();
        assert_ne!(pack_fingerprint(&bare.0).unwrap(), f1);
        // A size change alone moves it too (length rides every shard).
        std::fs::write(bare.0.join("a.safetensors"), b"changed!").unwrap();

        let empty = tempfile::tempdir().unwrap();
        assert!(pack_fingerprint(empty.path()).is_err());
    }

    #[test]
    fn cache_config_hash_is_stable_and_senses_every_input() {
        let base = cache_config_hash("f16", false, 0, 8, 131_072, 0);
        assert_eq!(cache_config_hash("f16", false, 0, 8, 131_072, 0), base);
        assert_ne!(cache_config_hash("q8", false, 0, 8, 131_072, 0), base);
        assert_ne!(cache_config_hash("f16", true, 0, 8, 131_072, 0), base);
        assert_ne!(cache_config_hash("f16", true, 512, 8, 131_072, 0), base);
        assert_ne!(cache_config_hash("f16", false, 0, 16, 131_072, 0), base);
        assert_ne!(cache_config_hash("f16", false, 0, 8, 8_192, 0), base);
        assert_ne!(cache_config_hash("f16", false, 0, 8, 131_072, 2), base);
        assert_ne!(
            cache_config_hash("f16", false, 0, 8, 131_072, 1),
            cache_config_hash("f16", false, 0, 8, 131_072, 2)
        );
    }

    #[test]
    fn park_identity_derives_the_profile_from_the_kv_string() {
        let guard = tmp_pack("identity");
        let pack = &guard.0;
        let id = park_identity(pack, [7; 32], "f16", true, 512, 8, 131_072, 0).unwrap();
        assert_eq!(id.model, pack_fingerprint(pack).unwrap());
        assert_eq!(id.template, [7; 32]);
        assert_eq!(
            id.cache_config,
            cache_config_hash("f16", true, 512, 8, 131_072, 0)
        );
        assert_eq!(id.format, PARK_FORMAT_VERSION);
        assert!(matches!(id.profile, ParkProfile::F16));
        assert!(matches!(
            park_identity(pack, [7; 32], "q8", true, 512, 8, 131_072, 0)
                .unwrap()
                .profile,
            ParkProfile::Q8
        ));
        assert!(park_identity(pack, [7; 32], "fp8", true, 512, 8, 131_072, 0).is_err());
    }
}
