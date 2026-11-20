//! Files of the Flash-Next artifact directory that the server loads.
//!
//! The artifact manifest (`download-manifest.json`) maps each file name to `{"sha256", "size"}`.
//! The manifest names the expected tokenizer files. `TokenizerBundle::load` compares each file
//! with the pin that `tokenizer_pin` builds and logs a warning for each one that differs.

use std::path::Path;

use crate::{
    EngError, family::session::PrefixIdentity, tokenizer::TokenizerPin, weights::sha256_hex,
};
use serde_json::Value;

fn manifest_error(message: String) -> EngError {
    EngError::Checkpoint(format!("artifact manifest: {message}"))
}

fn digest(manifest: &Value, name: &str) -> Result<[u8; 32], EngError> {
    let text = manifest
        .get(name)
        .and_then(|entry| entry.get("sha256"))
        .and_then(Value::as_str)
        .ok_or_else(|| manifest_error(format!("no sha256 for {name}")))?;
    if text.len() != 64 {
        return Err(manifest_error(format!("sha256 of {name} is not 64 digits")));
    }
    let mut out = [0u8; 32];
    for (slot, pair) in out.iter_mut().zip(text.as_bytes().as_chunks::<2>().0) {
        let digits = std::str::from_utf8(pair.as_slice())
            .ok()
            .and_then(|digits| u8::from_str_radix(digits, 16).ok())
            .ok_or_else(|| manifest_error(format!("sha256 of {name} is not hexadecimal")))?;
        *slot = digits;
    }
    Ok(out)
}

/// `vocab_size` is the vocabulary size of `tokenizer.json` with its added tokens
/// (248,077 for the pinned Flash-Next artifact).
pub(crate) fn tokenizer_pin(manifest: &[u8], vocab_size: u32) -> Result<TokenizerPin, EngError> {
    let value: Value =
        serde_json::from_slice(manifest).map_err(|error| manifest_error(error.to_string()))?;
    Ok(TokenizerPin {
        tokenizer_json: digest(&value, "tokenizer.json")?,
        tokenizer_config_json: digest(&value, "tokenizer_config.json")?,
        chat_template_jinja: digest(&value, "chat_template.jinja")?,
        generation_config_json: digest(&value, "generation_config.json")?,
        vocab_size,
    })
}

/// The load identity of the prefix store's path: the pinned `tokenizer.json`
/// digest and a SHA-256 over the loaded EXL3 pack's own metadata —
/// `config.json` and the weight index read verbatim, then every other
/// regular file of the pack by name and size in sorted order. The download
/// manifest is deliberately not an input: it names what was downloaded, not
/// what this loader opened, and two different packs can share one manifest.
/// Constant for one loaded pack, so folding it into a path leaves live
/// save/restore behaviour unchanged; a persisted store refuses entries
/// another load wrote. The weights half is layout plus sampled content —
/// file names and sizes, then the first and last 64 KiB of every weight
/// file — not a full digest.
pub(crate) fn pack_identity(pack: &Path, pin: &TokenizerPin) -> Result<PrefixIdentity, EngError> {
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
        crate::weights::hash_sampled_edges(&mut hasher, &pack.join(name), 0, *len)
            .map_err(|error| pack_error(format!("{name}: {error}")))?;
    }
    let weights: [u8; 32] = hasher.finalize().into();
    Ok(PrefixIdentity {
        weights: sha256_hex(&weights).into(),
        tokenizer: sha256_hex(&pin.tokenizer_json).into(),
    })
}

#[cfg(test)]
mod tests {
    use super::{pack_identity, tokenizer_pin};
    use crate::EngError;

    /// The four tokenizer entries of the Mac artifact manifest, with one unrelated entry.
    const MANIFEST: &str = r#"{
        "config.json": {"sha256": "319b334a1abb705acf06035aa0331bcb7c25976ff93a5035b543548738a10824", "size": 183155},
        "tokenizer.json": {"sha256": "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3", "size": 12809320},
        "tokenizer_config.json": {"sha256": "b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27", "size": 17928},
        "chat_template.jinja": {"sha256": "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041", "size": 8952},
        "generation_config.json": {"sha256": "e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e", "size": 202}
    }"#;

    #[test]
    fn the_pin_holds_the_manifest_hash_of_each_tokenizer_file() {
        let pin = tokenizer_pin(MANIFEST.as_bytes(), 248_077).unwrap();
        assert_eq!(pin.vocab_size, 248_077);
        assert_eq!(pin.tokenizer_json[..4], [0x09, 0x97, 0xf4, 0x10]);
        assert_eq!(pin.tokenizer_json[28..], [0x72, 0x29, 0xb9, 0xf3]);
        assert_eq!(pin.tokenizer_config_json[..4], [0xb1, 0x13, 0x49, 0xaa]);
        assert_eq!(pin.chat_template_jinja[..4], [0xc3, 0xcf, 0x9e, 0x34]);
        assert_eq!(pin.generation_config_json[..4], [0xe7, 0x0c, 0x13, 0x6c]);
        assert_eq!(pin.generation_config_json[28..], [0x70, 0xbe, 0x55, 0x0e]);
    }

    #[test]
    fn pack_identity_binds_the_loaded_pack_not_the_manifest() {
        // Two pack directories with identical metadata and identical
        // (name, size) lists share one identity; one byte of difference in
        // config.json — the kind of change a requant makes — splits them.
        let dir = tempfile::tempdir().unwrap();
        let a = dir.path().join("a");
        let b = dir.path().join("b");
        for pack in [&a, &b] {
            std::fs::create_dir(pack).unwrap();
            std::fs::write(pack.join("config.json"), b"{\"layers\": 2}").unwrap();
            std::fs::write(pack.join("model.safetensors.index.json"), b"{\"i\": 1}").unwrap();
            std::fs::write(pack.join("shard-0.bin"), [0u8; 64]).unwrap();
        }
        let pin = tokenizer_pin(MANIFEST.as_bytes(), 248_077).unwrap();
        let ia = pack_identity(&a, &pin).unwrap();
        let ib = pack_identity(&b, &pin).unwrap();
        assert_eq!(ia, ib, "identical packs must share one identity");
        std::fs::write(b.join("config.json"), b"{\"layers\": 3}").unwrap();
        let ic = pack_identity(&b, &pin).unwrap();
        assert_ne!(ia, ic, "a changed config must change the identity");
        // Same content, different size: the (name, size) list splits them.
        std::fs::write(b.join("shard-0.bin"), [0u8; 65]).unwrap();
        assert_ne!(ic.weights, pack_identity(&b, &pin).unwrap().weights);
        // Same size, one differing shard byte: the sampled edges split them
        // where (name, size) alone would collide.
        let mut same_size = [0u8; 64];
        same_size[63] = 1;
        std::fs::write(b.join("shard-0.bin"), same_size).unwrap();
        assert_ne!(ic.weights, pack_identity(&b, &pin).unwrap().weights);
        assert_eq!(
            ic.tokenizer, ia.tokenizer,
            "the tokenizer half comes from the pin, not the pack"
        );
    }

    #[test]
    fn a_pack_with_no_weight_files_refuses_an_identity() {
        let dir = tempfile::tempdir().unwrap();
        let pack = dir.path().join("empty");
        std::fs::create_dir(&pack).unwrap();
        std::fs::write(pack.join("config.json"), b"{}").unwrap();
        std::fs::write(pack.join("model.safetensors.index.json"), b"{}").unwrap();
        match pack_identity(&pack, &tokenizer_pin(MANIFEST.as_bytes(), 1).unwrap()) {
            Err(EngError::Checkpoint(message)) => {
                assert!(message.contains("no weight files"), "{message:?}")
            }
            other => panic!("expected a refusal, got {other:?}"),
        }
    }

    #[test]
    fn a_symlinked_shard_is_counted_like_a_real_one() {
        // The Hugging Face cache layout stores shards as symlinks; the
        // identity must follow them, not skip them.
        let dir = tempfile::tempdir().unwrap();
        let real = dir.path().join("shard-0.bin");
        std::fs::write(&real, [7u8; 64]).unwrap();
        let pack = dir.path().join("linked");
        std::fs::create_dir(&pack).unwrap();
        std::fs::write(pack.join("config.json"), b"{\"layers\": 2}").unwrap();
        std::fs::write(pack.join("model.safetensors.index.json"), b"{\"i\": 1}").unwrap();
        #[cfg(unix)]
        std::os::unix::fs::symlink(&real, pack.join("shard-0.bin")).unwrap();
        let pin = tokenizer_pin(MANIFEST.as_bytes(), 248_077).unwrap();
        let linked = pack_identity(&pack, &pin).unwrap();
        let plain = dir.path().join("plain");
        std::fs::create_dir(&plain).unwrap();
        std::fs::write(plain.join("config.json"), b"{\"layers\": 2}").unwrap();
        std::fs::write(plain.join("model.safetensors.index.json"), b"{\"i\": 1}").unwrap();
        std::fs::write(plain.join("shard-0.bin"), [7u8; 64]).unwrap();
        assert_eq!(
            linked,
            pack_identity(&plain, &pin).unwrap(),
            "a symlinked shard must contribute exactly what a real copy does"
        );
    }

    #[test]
    fn a_missing_pack_metadata_file_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        match pack_identity(dir.path(), &tokenizer_pin(MANIFEST.as_bytes(), 1).unwrap()) {
            Err(EngError::Checkpoint(message)) => assert!(
                message.starts_with("pack identity: config.json"),
                "{message:?}"
            ),
            other => panic!("expected a refusal, got {other:?}"),
        }
    }

    #[test]
    fn a_missing_tokenizer_entry_names_the_file() {
        let manifest = MANIFEST.replace("chat_template.jinja", "chat_template.other");
        let error = tokenizer_pin(manifest.as_bytes(), 1)
            .err()
            .expect("the manifest must be rejected");
        assert_eq!(
            error,
            EngError::Checkpoint("artifact manifest: no sha256 for chat_template.jinja".into())
        );
    }

    #[test]
    fn a_short_or_non_hex_hash_is_rejected() {
        let short = MANIFEST.replace("c3cf9e34abf4", "c3cf9e34ab");
        let error = tokenizer_pin(short.as_bytes(), 1)
            .err()
            .expect("the manifest must be rejected");
        assert_eq!(
            error,
            EngError::Checkpoint(
                "artifact manifest: sha256 of chat_template.jinja is not 64 digits".into()
            )
        );
        let bad = MANIFEST.replace("c3cf9e34abf4", "c3cf9e34abzz");
        let error = tokenizer_pin(bad.as_bytes(), 1)
            .err()
            .expect("the manifest must be rejected");
        assert_eq!(
            error,
            EngError::Checkpoint(
                "artifact manifest: sha256 of chat_template.jinja is not hexadecimal".into()
            )
        );
    }

    #[test]
    fn text_that_is_not_json_is_rejected() {
        let error = tokenizer_pin(b"not json", 1)
            .err()
            .expect("the manifest must be rejected");
        assert!(
            matches!(&error, EngError::Checkpoint(text) if text.starts_with("artifact manifest: ")),
            "{error:?}"
        );
    }
}
