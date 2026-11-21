//! The identity digest of a GGUF checkpoint, for the on-disk prefix cache.
//!
//! A cache entry holds the model's state after a prompt, so two checkpoints may share a cache
//! namespace only when they hold the same weights. Hashing the whole file at every start costs
//! tens of seconds on a large checkpoint, so the digest covers the header (which names every
//! tensor with its type, shape and offset), a sample of every tensor's bytes, the file length
//! and its modification time, and the full-file SHA-256 when the operator pinned one. A
//! fine-tune or merge changes values throughout a tensor, so the samples of every tensor tell
//! it from its base; a change confined to bytes the samples skip is not seen.

use super::GgufHeader;
use sha2::{Digest, Sha256};
use std::{io, os::unix::fs::FileExt, path::Path, time::UNIX_EPOCH};

/// Bytes read from the start, the middle and the end of every tensor. A tensor of at most three
/// samples is read whole.
pub const TENSOR_SAMPLE: u64 = 4096;

/// The identity digest of the checkpoint at `file`, whose header is `header`.
/// `pinned_sha256` is the operator's full-file digest when one is configured (the open has
/// already checked it against the file).
pub fn gguf_identity_digest(
    header: &GgufHeader,
    file: &Path,
    pinned_sha256: Option<&str>,
) -> Result<[u8; 32], io::Error> {
    let mut hasher = Sha256::new();
    hasher.update(b"fifi-gguf-identity-v2\0");
    hasher.update(header.header_sha256());
    match pinned_sha256 {
        Some(pinned) => {
            hasher.update([1u8]);
            hasher.update(pinned.to_ascii_lowercase().as_bytes());
        }
        None => hasher.update([0u8]),
    }
    let handle = std::fs::File::open(file)?;
    let meta = handle.metadata()?;
    hasher.update(meta.len().to_le_bytes());
    // The modification time breaks a tie between files of one length and sampled content; a
    // copy made without preserving it starts a new namespace, which costs only a cold prefill.
    let modified = meta
        .modified()
        .ok()
        .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
        .unwrap_or_default();
    hasher.update(modified.as_secs().to_le_bytes());
    hasher.update(modified.subsec_nanos().to_le_bytes());
    // A tensor of up to three samples is read whole, so the buffer holds three.
    let mut buf = vec![0u8; 3 * TENSOR_SAMPLE as usize];
    for tensor in header.tensors() {
        let (start, len) = (tensor.abs_offset(), tensor.byte_len());
        if start.checked_add(len).is_none_or(|end| end > meta.len()) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("tensor {} lies beyond the end of the file", tensor.name()),
            ));
        }
        hasher.update((tensor.name().len() as u64).to_le_bytes());
        hasher.update(tensor.name().as_bytes());
        hasher.update(start.to_le_bytes());
        hasher.update(len.to_le_bytes());
        let mut hash_at = |at: u64, take: u64| -> Result<(), io::Error> {
            let part = &mut buf[..take as usize];
            handle.read_exact_at(part, at)?;
            hasher.update(&*part);
            Ok(())
        };
        if len <= 3 * TENSOR_SAMPLE {
            hash_at(start, len)?;
        } else {
            hash_at(start, TENSOR_SAMPLE)?;
            hash_at(start + (len - TENSOR_SAMPLE) / 2, TENSOR_SAMPLE)?;
            hash_at(start + len - TENSOR_SAMPLE, TENSOR_SAMPLE)?;
        }
    }
    Ok(hasher.finalize().into())
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALIGN: usize = 32;

    fn string(text: &str) -> Vec<u8> {
        let mut out = (text.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(text.as_bytes());
        out
    }

    /// A GGUF v3 file with one F32 tensor per payload, no metadata, at aligned offsets.
    fn gguf(payloads: &[Vec<u8>]) -> Vec<u8> {
        let mut out = b"GGUF".to_vec();
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&(payloads.len() as u64).to_le_bytes());
        out.extend_from_slice(&0u64.to_le_bytes());
        let mut offset = 0usize;
        for (index, payload) in payloads.iter().enumerate() {
            out.extend_from_slice(&string(&format!("blk.{index}.weight")));
            out.extend_from_slice(&1u32.to_le_bytes());
            out.extend_from_slice(&(payload.len() as u64 / 4).to_le_bytes());
            out.extend_from_slice(&0u32.to_le_bytes());
            out.extend_from_slice(&(offset as u64).to_le_bytes());
            offset = (offset + payload.len()).div_ceil(ALIGN) * ALIGN;
        }
        for payload in payloads {
            out.resize(out.len().div_ceil(ALIGN) * ALIGN, 0);
            out.extend_from_slice(payload);
        }
        out
    }

    fn digest_of(
        dir: &tempfile::TempDir,
        name: &str,
        bytes: &[u8],
        pinned: Option<&str>,
        mtime: u64,
    ) -> [u8; 32] {
        let path = dir.path().join(name);
        std::fs::write(&path, bytes).unwrap();
        let file = std::fs::File::options().write(true).open(&path).unwrap();
        file.set_modified(UNIX_EPOCH + std::time::Duration::from_secs(mtime))
            .unwrap();
        let header = GgufHeader::from_bytes(bytes, bytes.len() as u64, 1 << 20).unwrap();
        gguf_identity_digest(&header, &path, pinned).unwrap()
    }

    fn model(middle_byte: u8) -> Vec<u8> {
        let tensor = |fill: u8| vec![fill; 16 * 1024];
        let mut middle = tensor(7);
        let at = (middle.len() - TENSOR_SAMPLE as usize) / 2 + 100;
        middle[at] = middle_byte;
        gguf(&[tensor(1), middle, tensor(3)])
    }

    #[test]
    fn two_files_differing_only_in_a_middle_tensor_get_different_digests() {
        let dir = tempfile::tempdir().unwrap();
        let base = digest_of(&dir, "base.gguf", &model(7), None, 10);
        let tuned = digest_of(&dir, "tuned.gguf", &model(8), None, 10);
        let same = digest_of(&dir, "same.gguf", &model(7), None, 10);
        assert_ne!(base, tuned);
        assert_eq!(base, same);
    }

    #[test]
    fn tensors_smaller_than_three_samples_are_read_whole() {
        let dir = tempfile::tempdir().unwrap();
        // 8 KiB and 12 KiB tensors (two and three samples) sit beside tiny ones, as norm weights do.
        let small = |fill: u8, len: usize| vec![fill; len];
        let base = gguf(&[
            small(1, 64),
            small(2, 8 * 1024),
            small(3, 12 * 1024),
            small(4, 4),
        ]);
        let mut changed = base.clone();
        let at = changed.len() - 4 - 100;
        changed[at] ^= 0x01;
        let a = digest_of(&dir, "a.gguf", &base, None, 10);
        let b = digest_of(&dir, "b.gguf", &changed, None, 10);
        assert_ne!(a, b, "a byte inside a whole-read tensor changes the digest");
    }

    #[test]
    fn the_pinned_digest_and_the_modification_time_enter_the_identity() {
        let dir = tempfile::tempdir().unwrap();
        let bytes = model(7);
        let plain = digest_of(&dir, "a.gguf", &bytes, None, 10);
        let pinned = digest_of(&dir, "b.gguf", &bytes, Some("ab"), 10);
        let other_pin = digest_of(&dir, "c.gguf", &bytes, Some("cd"), 10);
        let later = digest_of(&dir, "d.gguf", &bytes, None, 11);
        assert_ne!(plain, pinned);
        assert_ne!(pinned, other_pin);
        assert_ne!(plain, later);
    }

    #[test]
    fn a_tensor_beyond_the_end_of_the_file_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let bytes = model(7);
        let header = GgufHeader::from_bytes(&bytes, bytes.len() as u64, 1 << 20).unwrap();
        let path = dir.path().join("short.gguf");
        std::fs::write(&path, &bytes[..bytes.len() - 1024]).unwrap();
        let error = gguf_identity_digest(&header, &path, None).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
    }
}
