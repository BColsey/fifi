//! Per-tensor reader for a GGUF checkpoint of one file or one split set.
//!
//! `GgufWeights` is portable: it parses the header once, checks every tensor's
//! type up front, and then reads tensors on request. It never maps the file.
//! `read_tensor` reads one tensor at a time into a transient owned buffer, so
//! its peak host memory is the largest tensor, not the file. Device upload
//! belongs to the CUDA adapter, not to this module.
//!
//! `read_range` reads any byte range of one tensor into the caller's buffer,
//! so a caller that cannot hold a whole tensor streams it in chunks and holds
//! only one chunk at a time.
//!
//! A split set (`<stem>-00001-of-0000N.gguf` and its siblings) opens from any one of its
//! shard files. The reader finds the other shards by name in the same directory, checks
//! that they form one consistent set, and indexes every tensor of every shard under its
//! own name. `header()` is the joined view: the metadata of shard 1 and all tensors, each
//! tensor with its shard number and its offset inside that shard's file.

use std::{
    fs::File,
    io::{BufReader, ErrorKind},
    os::unix::fs::FileExt,
    path::{Path, PathBuf},
};

use super::gguf::{GgufHeader, GgufTensor, GgufType};
use super::gguf_split::shard_name_of;
use crate::EngError;

/// Tensor types the CUDA path loads: F32 and F16 hold norms and small tables, every quantized
/// type with a native kernel (`GgufType::has_native_kernel`) holds matrices. Which role takes
/// which type is the model open's rule.
fn supported_type(ty: GgufType) -> bool {
    matches!(ty, GgufType::F32 | GgufType::F16) || ty.has_native_kernel()
}

/// Size limits checked before anything large is allocated.
#[derive(Debug, Clone, Copy)]
pub(crate) struct GgufLimits {
    /// Upper bound on the header and tensor table, passed to the header parser.
    pub(crate) max_header_bytes: u64,
    /// Upper bound on one tensor's payload, and so on one read buffer.
    pub(crate) max_tensor_bytes: u64,
}

/// An open GGUF checkpoint (one file or a split set) whose header and tensor types have
/// been checked.
#[derive(Debug)]
pub(crate) struct GgufWeights {
    /// One handle per shard, in shard order; one entry for a single file.
    files: Vec<File>,
    paths: Vec<PathBuf>,
    /// The length of each file when it was opened.
    file_lens: Vec<u64>,
    header: GgufHeader,
    /// The tensors a family's inventory names as dense projections (see [`Self::mark_dense`]).
    #[cfg(feature = "native-cuda")]
    dense: std::collections::HashSet<String>,
}

impl GgufWeights {
    /// Opens `path` and checks the header, the recorded header hash, the split
    /// metadata, every tensor's type, and every tensor's size.
    ///
    /// The header parser already checks that every tensor range lies inside
    /// the file. A tensor of an unsupported type, a missing or inconsistent
    /// shard of a split set, or a hash mismatch fails with `EngError::Checkpoint`.
    /// A tensor above `limits.max_tensor_bytes` fails with `EngError::MemoryPlan`.
    ///
    /// `path` may be any shard of a split set; the set opens as a whole and
    /// `header_sha256` is compared with the header digest of shard 1.
    pub(crate) fn open(
        path: &Path,
        limits: GgufLimits,
        header_sha256: Option<&str>,
    ) -> Result<Self, EngError> {
        Self::open_checked(path, limits, header_sha256, true)
    }

    /// [`Self::open`] without the check of tensor types: every type the header reader can
    /// size is accepted. A loader that maps the tensors itself and refuses the types its
    /// kernels lack uses this.
    #[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
    pub(crate) fn open_any_type(
        path: &Path,
        limits: GgufLimits,
        header_sha256: Option<&str>,
    ) -> Result<Self, EngError> {
        Self::open_checked(path, limits, header_sha256, false)
    }

    fn open_checked(
        path: &Path,
        limits: GgufLimits,
        header_sha256: Option<&str>,
        check_types: bool,
    ) -> Result<Self, EngError> {
        let (first_file, first_len, first) = open_one(path, limits)?;
        let Some(split) = first.split().filter(|split| split.count > 1) else {
            if let Some(expected) = header_sha256 {
                first.check_header_sha256(expected)?;
            }
            check_tensors(&first, limits, check_types)?;
            return Ok(Self {
                files: vec![first_file],
                paths: vec![path.to_path_buf()],
                file_lens: vec![first_len],
                header: first,
                #[cfg(feature = "native-cuda")]
                dense: Default::default(),
            });
        };
        let name = shard_name_of(path).ok_or_else(|| {
            checkpoint(format!(
                "{} is shard {} of a split GGUF set of {}, but its file name does not follow \
                 the <name>-00001-of-0000N.gguf pattern, so the other shards cannot be found",
                path.display(),
                split.no + 1,
                split.count
            ))
        })?;
        if u64::from(name.number) != split.no + 1 || u64::from(name.count) != split.count {
            return Err(checkpoint(format!(
                "{}: the file name says shard {} of {}, the file's own split metadata says shard \
                 {} of {}",
                path.display(),
                name.number,
                name.count,
                split.no + 1,
                split.count
            )));
        }
        let mut opened = Vec::new();
        let mut paths = Vec::new();
        let mut handed = Some((first_file, first_len, first));
        for number in 1..=name.count {
            let shard_path = name.sibling(path, number);
            if number == name.number
                && let Some(one) = handed.take()
            {
                opened.push(one);
                paths.push(shard_path);
                continue;
            }
            if !shard_path.is_file() {
                return Err(checkpoint(format!(
                    "split GGUF set is incomplete: shard {number} of {} is missing ({})",
                    name.count,
                    shard_path.display()
                )));
            }
            let one = open_one(&shard_path, limits).map_err(|error| match error {
                EngError::Checkpoint(message) => EngError::Checkpoint(format!(
                    "split GGUF shard {number} of {} ({}): {message}",
                    name.count,
                    shard_path.display()
                )),
                other => other,
            })?;
            opened.push(one);
            paths.push(shard_path);
        }
        let mut files = Vec::new();
        let mut file_lens = Vec::new();
        let mut headers = Vec::new();
        for ((file, len, header), shard_path) in opened.into_iter().zip(&paths) {
            let shard = headers.len();
            if header
                .split()
                .is_none_or(|info| info.count != split.count || info.no != shard as u64)
            {
                return Err(checkpoint(format!(
                    "{} is not shard {} of {}: its split metadata names another position or \
                     another set",
                    shard_path.display(),
                    shard + 1,
                    split.count
                )));
            }
            files.push(file);
            file_lens.push(len);
            headers.push(header);
        }
        let header = GgufHeader::merge_split(headers)?;
        if let Some(expected) = header_sha256 {
            header.check_header_sha256(expected)?;
        }
        check_tensors(&header, limits, check_types)?;
        Ok(Self {
            files,
            paths,
            file_lens,
            header,
            #[cfg(feature = "native-cuda")]
            dense: Default::default(),
        })
    }

    /// Records the tensors the family's inventory names as dense projections. A native load
    /// encodes one of these that the file stores as F32, F16 or BF16 to Q8_0 blocks, the form its
    /// matvec and GEMM read; a tensor not named loads as the file has it. The model opener sets
    /// this from its plan, so every load of the model's weights, whole, partial or sliced, applies
    /// the same set and no caller can leave it out.
    #[cfg(feature = "native-cuda")]
    pub(crate) fn mark_dense<'n>(&mut self, names: impl IntoIterator<Item = &'n str>) {
        self.dense = names.into_iter().map(str::to_owned).collect();
    }

    /// Whether the family's inventory names `name` a dense projection.
    #[cfg(feature = "native-cuda")]
    pub(crate) fn is_dense(&self, name: &str) -> bool {
        self.dense.contains(name)
    }

    /// The shard files in order; one path for a single file.
    #[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
    pub(crate) fn paths(&self) -> &[PathBuf] {
        &self.paths
    }

    /// Whether the checkpoint is a split set of more than one file.
    pub(crate) fn is_split(&self) -> bool {
        self.files.len() > 1
    }

    /// Fails for a split set. A caller that addresses tensor bytes by file offset through
    /// its own handle on one path reads the wrong bytes of a set; it names itself in `what`.
    pub(crate) fn require_single_file(&self, what: &str) -> Result<(), EngError> {
        if self.is_split() {
            return Err(checkpoint(format!(
                "{what} reads tensor bytes by file offset and does not take a split GGUF set \
                 ({} shards starting at {}); merge the shards into one file",
                self.files.len(),
                self.paths[0].display()
            )));
        }
        Ok(())
    }

    pub(crate) fn header(&self) -> &GgufHeader {
        &self.header
    }

    /// The open file of shard 1 (the whole file outside a split set). The CUDA loader
    /// drops its page cache; the handle is read-only.
    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    pub(crate) fn file(&self) -> &File {
        &self.files[0]
    }

    /// Every open shard file in order. Read-only handles.
    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    pub(crate) fn files(&self) -> &[File] {
        &self.files
    }

    /// The open file that holds `tensor`.
    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    pub(crate) fn file_of(&self, tensor: &GgufTensor) -> &File {
        &self.files[tensor.shard()]
    }

    /// Reads one tensor's raw payload into a new buffer of exactly its length.
    ///
    /// The file length is checked against the parsed header before and after
    /// the read, so a file replaced or truncated after `open` fails with
    /// `EngError::Checkpoint` instead of returning mixed bytes.
    pub(crate) fn read_tensor(&self, name: &str) -> Result<(GgufType, Vec<u8>), EngError> {
        let tensor = self.tensor(name)?;
        let len = usize::try_from(tensor.byte_len()).map_err(|_| {
            EngError::MemoryPlan(format!("gguf: tensor {name} does not fit in memory"))
        })?;
        let mut buf = Vec::new();
        buf.try_reserve_exact(len).map_err(|e| {
            EngError::MemoryPlan(format!("gguf: tensor {name} needs {len} bytes: {e}"))
        })?;
        buf.resize(len, 0);
        self.read_range(name, 0, &mut buf)?;
        Ok((tensor.ggml_type(), buf))
    }

    /// Fills `buf` with the payload bytes of tensor `name` that start `offset`
    /// bytes into the tensor, so a caller can stream a tensor in chunks.
    ///
    /// The range must lie inside the tensor; one that does not (including an
    /// offset that overflows) fails with `EngError::BadRequest` before any
    /// read. An unknown name and a file length that differs from the parsed
    /// header, checked before and after the read as in [`Self::read_tensor`],
    /// fail with `EngError::Checkpoint`.
    pub(crate) fn read_range(
        &self,
        name: &str,
        offset: u64,
        buf: &mut [u8],
    ) -> Result<(), EngError> {
        let tensor = self.tensor(name)?;
        let len = buf.len() as u64;
        let end = offset.checked_add(len);
        if end.is_none_or(|end| end > tensor.byte_len()) {
            return Err(EngError::BadRequest(format!(
                "gguf: range of {len} bytes at offset {offset} lies outside tensor {name} of {} bytes",
                tensor.byte_len()
            )));
        }
        // The header parser checked that the whole tensor lies inside the
        // file, so this sum stays below the file length.
        let at = tensor.abs_offset().checked_add(offset).ok_or_else(|| {
            checkpoint(format!("tensor {name} offset {offset} overflows the file"))
        })?;
        self.check_len(tensor.shard())?;
        // A positioned read shares no file cursor, so concurrent callers on
        // one `GgufWeights` cannot receive each other's bytes.
        self.files[tensor.shard()]
            .read_exact_at(buf, at)
            .map_err(|e| read_error(name, &e))?;
        self.check_len(tensor.shard())
    }

    /// Fills each buffer from its shard file at the given absolute offset, as `reads` lists
    /// `(shard, file offset, buffer)`. Every range must lie inside its shard's length when
    /// opened (`EngError::BadRequest` before any read otherwise). Each distinct shard's length
    /// is checked once, after the reads, so a file changed since open fails the whole call.
    #[cfg_attr(not(test), expect(dead_code, reason = "no caller outside the tests"))]
    pub(crate) fn read_file_ranges(
        &self,
        reads: &mut [(usize, u64, &mut [u8])],
    ) -> Result<(), EngError> {
        for (shard, offset, buf) in reads.iter() {
            let len =
                self.file_lens.get(*shard).copied().ok_or_else(|| {
                    EngError::BadRequest(format!("gguf: shard {shard} is not open"))
                })?;
            let end = offset.checked_add(buf.len() as u64);
            if end.is_none_or(|end| end > len) {
                return Err(EngError::BadRequest(format!(
                    "gguf: {} bytes at offset {offset} lie outside shard {shard} of {len} bytes",
                    buf.len()
                )));
            }
        }
        for (shard, offset, buf) in reads.iter_mut() {
            self.files[*shard]
                .read_exact_at(buf, *offset)
                .map_err(|e| read_error(&format!("shard {shard}"), &e))?;
        }
        let mut checked: Vec<usize> = Vec::new();
        for (shard, _, _) in reads.iter() {
            if !checked.contains(shard) {
                self.check_len(*shard)?;
                checked.push(*shard);
            }
        }
        Ok(())
    }

    /// Fills buf with header bytes at file offset offset of shard 1 (the file itself
    /// outside a split set), where the metadata lives.
    ///
    /// The range must lie inside the parsed header extent: one that does not
    /// fails with EngError::BadRequest before any read. The file length is
    /// checked before and after the read as in read_range, so a file replaced
    /// after open fails with EngError::Checkpoint.
    pub(crate) fn read_header_range(&self, offset: u64, buf: &mut [u8]) -> Result<(), EngError> {
        let end = offset.checked_add(buf.len() as u64);
        if end.is_none_or(|end| end > self.header.data_start()) {
            return Err(EngError::BadRequest(format!(
                "gguf: header range of {} bytes at offset {offset} lies outside the header",
                buf.len()
            )));
        }
        self.check_len(0)?;
        self.files[0]
            .read_exact_at(buf, offset)
            .map_err(|e| read_error("metadata array", &e))?;
        self.check_len(0)
    }

    fn tensor(&self, name: &str) -> Result<&GgufTensor, EngError> {
        self.header
            .tensor(name)
            .ok_or_else(|| checkpoint(format!("tensor {name} is not in the file")))
    }

    fn check_len(&self, shard: usize) -> Result<(), EngError> {
        let now = current_len(&self.files[shard])?;
        if now != self.file_lens[shard] {
            return Err(checkpoint(format!(
                "file length changed from {} to {now} after open",
                self.file_lens[shard]
            )));
        }
        Ok(())
    }
}

/// Opens one file through the local-storage guard and parses its header.
fn open_one(path: &Path, limits: GgufLimits) -> Result<(File, u64, GgufHeader), EngError> {
    crate::storage_guard::require_local(path, "model weights")?;
    let file = File::open(path).map_err(|e| io_error(path, &e))?;
    let file_len = current_len(&file)?;
    let header = GgufHeader::parse(
        &mut BufReader::new(&file),
        file_len,
        limits.max_header_bytes,
    )?;
    Ok((file, file_len, header))
}

fn check_tensors(
    header: &GgufHeader,
    limits: GgufLimits,
    check_types: bool,
) -> Result<(), EngError> {
    for tensor in header.tensors() {
        check_tensor(tensor, limits.max_tensor_bytes, check_types)?;
    }
    Ok(())
}

fn check_tensor(
    tensor: &GgufTensor,
    max_tensor_bytes: u64,
    check_types: bool,
) -> Result<(), EngError> {
    let ty = tensor.ggml_type();
    // A BF16 norm or gate vector is widened to F32 at load, so it is a supported tensor.
    let widened = ty == GgufType::Bf16 && tensor.is_small_half_vector();
    if check_types && !supported_type(ty) && !widened {
        return Err(checkpoint(format!(
            "tensor {} has unsupported type {}",
            tensor.name(),
            ty.name()
        )));
    }
    if tensor.byte_len() > max_tensor_bytes {
        return Err(EngError::MemoryPlan(format!(
            "gguf: tensor {} has {} bytes, above the limit of {max_tensor_bytes}",
            tensor.name(),
            tensor.byte_len()
        )));
    }
    Ok(())
}

fn current_len(file: &File) -> Result<u64, EngError> {
    file.metadata()
        .map(|meta| meta.len())
        .map_err(|e| checkpoint(format!("cannot read file metadata: {e}")))
}

fn checkpoint(message: String) -> EngError {
    EngError::Checkpoint(format!("gguf: {message}"))
}

fn io_error(path: &Path, error: &std::io::Error) -> EngError {
    checkpoint(format!("cannot open {}: {error}", path.display()))
}

fn read_error(name: &str, error: &std::io::Error) -> EngError {
    if error.kind() == ErrorKind::UnexpectedEof {
        return checkpoint(format!("tensor {name} ends past the end of the file"));
    }
    checkpoint(format!("cannot read tensor {name}: {error}"))
}

#[cfg(test)]
mod tests {
    use std::{fs::OpenOptions, io::Write, path::PathBuf};

    use super::*;
    use crate::weights::{dequantize, sha256_hex};

    const LIMITS: GgufLimits = GgufLimits {
        max_header_bytes: 1 << 20,
        max_tensor_bytes: 1 << 20,
    };
    const ALIGN: usize = 32;

    // The real entry point on a real network mount: without the guard the
    // error would be the missing file. Skipped with a note on a host with no
    // network mount.
    #[cfg(target_os = "linux")]
    #[test]
    fn gguf_open_refuses_a_live_network_path() {
        let Some(path) = crate::storage_guard::seam::live_network_path() else {
            eprintln!("skipped: no network mount on this host");
            return;
        };
        let error = GgufWeights::open(&path, LIMITS, None).expect_err("refused");
        assert!(
            error.describe().contains("must be on local storage"),
            "{}",
            error.describe()
        );
    }

    // Fails if `GgufWeights::open` stops guarding the file: without the guard
    // the error would be the missing file, not the storage refusal.
    #[cfg(target_os = "linux")]
    #[test]
    fn a_gguf_file_on_a_network_mount_is_refused_before_it_is_opened() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("model.gguf");
        let error = crate::storage_guard::seam::with_nfs_at(dir.path(), || {
            GgufWeights::open(&path, LIMITS, None).expect_err("refused")
        });
        assert!(
            error.describe().contains("must be on local storage"),
            "{}",
            error.describe()
        );
    }

    fn string(text: &str) -> Vec<u8> {
        let mut out = (text.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(text.as_bytes());
        out
    }

    fn kv_u16(key: &str, value: u16) -> Vec<u8> {
        let mut out = string(key);
        out.extend_from_slice(&2u32.to_le_bytes());
        out.extend_from_slice(&value.to_le_bytes());
        out
    }

    /// One tensor: name, element count (one dimension), type, and payload.
    struct Entry<'a> {
        name: &'a str,
        elements: u64,
        ty: GgufType,
        payload: &'a [u8],
    }

    /// Builds a GGUF v3 file with one-dimensional tensors at aligned offsets.
    fn gguf(kvs: &[Vec<u8>], entries: &[Entry<'_>]) -> Vec<u8> {
        let mut out = b"GGUF".to_vec();
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&(entries.len() as u64).to_le_bytes());
        out.extend_from_slice(&(kvs.len() as u64).to_le_bytes());
        for kv in kvs {
            out.extend_from_slice(kv);
        }
        let mut offset = 0usize;
        for entry in entries {
            out.extend_from_slice(&string(entry.name));
            out.extend_from_slice(&1u32.to_le_bytes());
            out.extend_from_slice(&entry.elements.to_le_bytes());
            out.extend_from_slice(&entry.ty.id().to_le_bytes());
            out.extend_from_slice(&(offset as u64).to_le_bytes());
            offset = (offset + entry.payload.len()).div_ceil(ALIGN) * ALIGN;
        }
        for entry in entries {
            out.resize(out.len().div_ceil(ALIGN) * ALIGN, 0);
            out.extend_from_slice(entry.payload);
        }
        out
    }

    fn write(dir: &tempfile::TempDir, bytes: &[u8]) -> PathBuf {
        let path = dir.path().join("model.gguf");
        std::fs::write(&path, bytes).expect("write fixture");
        path
    }

    fn fixture(name: &str) -> Vec<u8> {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/gguf-dequant")
            .join(name);
        std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
    }

    fn elements(ty: GgufType, payload: &[u8]) -> u64 {
        payload.len() as u64 / ty.block_bytes() * ty.block_elems()
    }

    fn entry<'a>(name: &'a str, ty: GgufType, payload: &'a [u8]) -> Entry<'a> {
        Entry {
            name,
            elements: elements(ty, payload),
            ty,
            payload,
        }
    }

    fn expect_checkpoint<T: std::fmt::Debug>(result: Result<T, EngError>, expect: &str) {
        match result {
            Err(EngError::Checkpoint(message)) => {
                assert!(message.contains(expect), "{expect:?} not in {message:?}");
            }
            other => panic!("expected Checkpoint with {expect:?}, got {other:?}"),
        }
    }

    #[test]
    fn tensors_read_back_and_match_the_cpu_dequantizer_fixtures() {
        let cases = [
            ("f32", GgufType::F32),
            ("f16", GgufType::F16),
            ("q4_k", GgufType::Q4K),
            ("q6_k", GgufType::Q6K),
            ("q8_0", GgufType::Q80),
        ];
        let payloads: Vec<Vec<u8>> = cases
            .iter()
            .map(|(name, _)| fixture(&format!("{name}.blocks")))
            .collect();
        let entries: Vec<Entry<'_>> = cases
            .iter()
            .zip(&payloads)
            .map(|((name, ty), payload)| entry(name, *ty, payload))
            .collect();
        let dir = tempfile::tempdir().expect("tempdir");
        let weights =
            GgufWeights::open(&write(&dir, &gguf(&[], &entries)), LIMITS, None).expect("open");
        for ((name, ty), payload) in cases.iter().zip(&payloads) {
            let (read_ty, bytes) = weights.read_tensor(name).expect("read");
            assert_eq!(read_ty, *ty, "{name}");
            assert_eq!(&bytes, payload, "{name}");
            let values = dequantize(read_ty, &bytes).expect("dequantize");
            let want = fixture(&format!("{name}.f32"));
            let (want, rest) = want.as_chunks::<4>();
            assert!(rest.is_empty(), "{name} oracle length");
            assert_eq!(values.len(), want.len(), "{name} values");
            for (i, (got, want)) in values.iter().zip(want).enumerate() {
                let want = f32::from_le_bytes(*want);
                // x86 and arm64 default NaNs differ in sign; tests/gguf_dequant.rs
                // pins the exact bits per architecture.
                let same = got.to_bits() == want.to_bits() || (got.is_nan() && want.is_nan());
                assert!(same, "{name}[{i}]: {got:?} != {want:?}");
            }
        }
    }

    #[test]
    fn a_q5_0_tensor_opens_and_reads_back_its_blocks() {
        // Two blocks of 22 bytes with distinct bytes.
        let blocks: Vec<u8> = (0..44u8)
            .map(|i| i.wrapping_mul(5).wrapping_add(1))
            .collect();
        let dir = tempfile::tempdir().expect("tempdir");
        let path = write(&dir, &gguf(&[], &[entry("q5", GgufType::Q50, &blocks)]));
        let weights = GgufWeights::open(&path, LIMITS, None).expect("Q5_0 opens");
        let (ty, bytes) = weights.read_tensor("q5").expect("read");
        assert_eq!(ty, GgufType::Q50);
        assert_eq!(bytes, blocks);
        assert_eq!(dequantize(ty, &bytes).expect("decode").len(), 64);
    }

    #[test]
    fn unsupported_tensor_types_fail_the_open_naming_the_tensor() {
        let blocks = fixture("q5_k.blocks");
        let f32s = fixture("f32.blocks");
        let entries = [
            entry("norm", GgufType::F32, &f32s),
            entry("blk.0.ffn_down", GgufType::Q41, &blocks),
        ];
        let dir = tempfile::tempdir().expect("tempdir");
        let path = write(&dir, &gguf(&[], &entries));
        expect_checkpoint(
            GgufWeights::open(&path, LIMITS, None),
            "tensor blk.0.ffn_down has unsupported type Q4_1",
        );
    }

    #[test]
    fn a_split_file_with_an_unpatterned_name_names_the_pattern() {
        let kvs = [
            kv_u16("split.no", 0),
            kv_u16("split.count", 2),
            kv_u16("split.tensors.count", 2),
        ];
        let payload = [0u8; 8];
        let dir = tempfile::tempdir().expect("tempdir");
        let path = write(&dir, &gguf(&kvs, &[entry("a", GgufType::F32, &payload)]));
        expect_checkpoint(
            GgufWeights::open(&path, LIMITS, None),
            "does not follow the <name>-00001-of-0000N.gguf pattern",
        );
    }

    fn kv_str(key: &str, value: &str) -> Vec<u8> {
        let mut out = string(key);
        out.extend_from_slice(&8u32.to_le_bytes());
        out.extend_from_slice(&string(value));
        out
    }

    /// The tensors of a split fixture: shard `i` holds `names[i]`, each an F32
    /// tensor of 8 values whose bytes are the shard number and tensor position.
    struct Split {
        dir: tempfile::TempDir,
        names: Vec<Vec<&'static str>>,
    }

    impl Split {
        fn path(&self, number: usize) -> PathBuf {
            let count = self.names.len();
            self.dir
                .path()
                .join(format!("m-{number:05}-of-{count:05}.gguf"))
        }

        fn payload(shard: usize, at: usize) -> Vec<u8> {
            vec![(shard * 16 + at) as u8; 32]
        }

        /// Shard 1 carries the model keys; later shards carry only the split keys.
        fn shard_bytes(&self, shard: usize, total: usize, extra: &[Vec<u8>]) -> Vec<u8> {
            let mut kvs = Vec::new();
            if shard == 0 {
                kvs.push(kv_str("general.architecture", "test"));
            }
            kvs.push(kv_u16("split.no", shard as u16));
            kvs.push(kv_u16("split.count", self.names.len() as u16));
            kvs.push(kv_u16("split.tensors.count", total as u16));
            kvs.extend(extra.iter().cloned());
            let payloads: Vec<Vec<u8>> = (0..self.names[shard].len())
                .map(|at| Self::payload(shard, at))
                .collect();
            let entries: Vec<Entry<'_>> = self.names[shard]
                .iter()
                .zip(&payloads)
                .map(|(name, payload)| entry(name, GgufType::F32, payload))
                .collect();
            gguf(&kvs, &entries)
        }

        fn new(names: Vec<Vec<&'static str>>) -> Self {
            let split = Self {
                dir: tempfile::tempdir().expect("tempdir"),
                names,
            };
            let total = split.names.iter().map(Vec::len).sum();
            for shard in 0..split.names.len() {
                std::fs::write(split.path(shard + 1), split.shard_bytes(shard, total, &[]))
                    .expect("write shard");
            }
            split
        }
    }

    #[test]
    fn a_split_set_opens_from_any_shard_and_reads_each_tensor_from_its_file() {
        let split = Split::new(vec![vec![], vec!["a", "b"], vec!["c"]]);
        for from in 1..=3 {
            let weights = GgufWeights::open(&split.path(from), LIMITS, None).expect("open the set");
            assert!(weights.is_split(), "from shard {from}");
            assert_eq!(weights.paths().len(), 3);
            assert_eq!(weights.header().tensors().len(), 3);
            for (shard, at, name) in [(1, 0, "a"), (1, 1, "b"), (2, 0, "c")] {
                let tensor = weights.header().tensor(name).expect("indexed");
                assert_eq!(tensor.shard(), shard, "{name}");
                let (ty, bytes) = weights.read_tensor(name).expect("read");
                assert_eq!(ty, GgufType::F32);
                assert_eq!(bytes, Split::payload(shard, at), "{name} from shard {from}");
                let mut tail = [0u8; 4];
                weights.read_range(name, 28, &mut tail).expect("range");
                assert_eq!(tail, [Split::payload(shard, at)[0]; 4], "{name}");
            }
            // Shard 1 holds the model keys and the metadata arrays are read from it.
            assert!(weights.header().meta("general.architecture").is_some());
        }
    }

    #[test]
    fn a_missing_shard_is_named_with_its_number_and_path() {
        let split = Split::new(vec![vec![], vec!["a"], vec!["b"], vec!["c"]]);
        std::fs::remove_file(split.path(3)).expect("remove");
        let error = GgufWeights::open(&split.path(1), LIMITS, None).expect_err("incomplete");
        let text = error.describe();
        assert!(
            text.contains("shard 3 of 4 is missing") && text.contains("m-00003-of-00004.gguf"),
            "{text}"
        );
    }

    #[test]
    fn a_shard_with_another_position_in_its_metadata_is_refused() {
        let split = Split::new(vec![vec!["a"], vec!["b"], vec!["c"]]);
        // Shard 3's file claims to be shard 2.
        let total = 3;
        let mut wrong = split.shard_bytes(1, total, &[]);
        std::fs::write(split.path(3), std::mem::take(&mut wrong)).expect("overwrite");
        expect_checkpoint(
            GgufWeights::open(&split.path(1), LIMITS, None),
            "is not shard 3 of 3",
        );
    }

    #[test]
    fn a_file_name_that_disagrees_with_its_own_metadata_is_refused() {
        let split = Split::new(vec![vec!["a"], vec!["b"]]);
        // Copy shard 2's bytes under a name that says shard 1 of 2.
        let bytes = std::fs::read(split.path(2)).expect("read");
        std::fs::write(split.path(1), bytes).expect("overwrite");
        expect_checkpoint(
            GgufWeights::open(&split.path(1), LIMITS, None),
            "the file name says shard 1 of 2",
        );
    }

    #[test]
    fn shards_that_disagree_on_a_model_key_or_a_total_are_refused() {
        let split = Split::new(vec![vec!["a"], vec!["b"]]);
        let total = 2;
        // A later shard may repeat a model key only with shard 1's value.
        std::fs::write(
            split.path(2),
            split.shard_bytes(1, total, &[kv_str("general.architecture", "other")]),
        )
        .expect("overwrite");
        expect_checkpoint(
            GgufWeights::open(&split.path(1), LIMITS, None),
            "disagrees with shard 1 on metadata key general.architecture",
        );
        std::fs::write(
            split.path(2),
            split.shard_bytes(1, total, &[kv_str("general.name", "x")]),
        )
        .expect("overwrite");
        expect_checkpoint(
            GgufWeights::open(&split.path(1), LIMITS, None),
            "has metadata key general.name, which shard 1 lacks",
        );
        std::fs::write(
            split.path(2),
            split.shard_bytes(1, total, &[kv_str("general.architecture", "test")]),
        )
        .expect("overwrite");
        GgufWeights::open(&split.path(1), LIMITS, None).expect("a repeated equal key is fine");
        std::fs::write(split.path(2), split.shard_bytes(1, total + 1, &[])).expect("overwrite");
        expect_checkpoint(
            GgufWeights::open(&split.path(1), LIMITS, None),
            "split.tensors.count",
        );
    }

    #[test]
    fn a_tensor_name_in_two_shards_is_refused() {
        let split = Split::new(vec![vec!["a"], vec!["a"]]);
        expect_checkpoint(
            GgufWeights::open(&split.path(2), LIMITS, None),
            "tensor a appears in more than one shard",
        );
    }

    #[test]
    fn the_header_hash_of_a_set_is_the_hash_of_shard_one() {
        let split = Split::new(vec![vec![], vec!["a"]]);
        let first = GgufWeights::open(&split.path(1), LIMITS, None).expect("open");
        let recorded = sha256_hex(&first.header().header_sha256());
        GgufWeights::open(&split.path(2), LIMITS, Some(&recorded)).expect("shard 1 hash");
        expect_checkpoint(
            GgufWeights::open(&split.path(2), LIMITS, Some(&"0".repeat(64))),
            "header sha256",
        );
    }

    #[test]
    fn a_length_change_of_one_shard_fails_reads_of_its_tensors_only() {
        let split = Split::new(vec![vec!["a"], vec!["b"]]);
        let weights = GgufWeights::open(&split.path(1), LIMITS, None).expect("open");
        let mut file = OpenOptions::new()
            .append(true)
            .open(split.path(2))
            .expect("reopen");
        file.write_all(&[0]).expect("append");
        weights.read_tensor("a").expect("shard 1 is unchanged");
        expect_checkpoint(weights.read_tensor("b"), "file length changed");
    }

    #[test]
    fn require_single_file_refuses_a_set_and_names_the_caller() {
        let split = Split::new(vec![vec!["a"], vec!["b"]]);
        let weights = GgufWeights::open(&split.path(1), LIMITS, None).expect("open");
        expect_checkpoint(
            weights.require_single_file("the expert locator"),
            "the expert locator",
        );
        let (_dir, _path, whole, _first, _second) = range_fixture();
        whole.require_single_file("anything").expect("one file");
    }

    #[test]
    fn single_shard_split_metadata_is_accepted() {
        let kvs = [kv_u16("split.no", 0), kv_u16("split.count", 1)];
        let payload = [0u8; 8];
        let dir = tempfile::tempdir().expect("tempdir");
        let path = write(&dir, &gguf(&kvs, &[entry("a", GgufType::F32, &payload)]));
        GgufWeights::open(&path, LIMITS, None).expect("one shard is a whole file");
    }

    #[test]
    fn open_any_type_accepts_what_open_refuses_by_type() {
        let blocks = fixture("q5_k.blocks");
        let dir = tempfile::tempdir().expect("tempdir");
        let path = write(&dir, &gguf(&[], &[entry("w", GgufType::Q41, &blocks)]));
        expect_checkpoint(
            GgufWeights::open(&path, LIMITS, None),
            "unsupported type Q4_1",
        );
        GgufWeights::open_any_type(&path, LIMITS, None).expect("types are the caller's to judge");
    }

    #[test]
    fn tensors_above_the_size_limit_are_memory_plan_errors() {
        let payload = [0u8; 64];
        let dir = tempfile::tempdir().expect("tempdir");
        let path = write(&dir, &gguf(&[], &[entry("big", GgufType::F32, &payload)]));
        let limits = GgufLimits {
            max_tensor_bytes: 63,
            ..LIMITS
        };
        match GgufWeights::open(&path, limits, None) {
            Err(EngError::MemoryPlan(message)) => {
                assert!(message.contains("tensor big has 64 bytes"), "{message}");
            }
            other => panic!("expected MemoryPlan, got {other:?}"),
        }
        let limits = GgufLimits {
            max_tensor_bytes: 64,
            ..LIMITS
        };
        GgufWeights::open(&path, limits, None).expect("a tensor at the limit fits");
    }

    #[test]
    fn the_recorded_header_hash_is_checked() {
        let payload = [0u8; 8];
        let bytes = gguf(&[], &[entry("a", GgufType::F32, &payload)]);
        let dir = tempfile::tempdir().expect("tempdir");
        let path = write(&dir, &bytes);
        let weights = GgufWeights::open(&path, LIMITS, None).expect("open");
        let recorded = sha256_hex(&weights.header().header_sha256());
        GgufWeights::open(&path, LIMITS, Some(&recorded)).expect("matching hash");
        let wrong = "0".repeat(64);
        expect_checkpoint(
            GgufWeights::open(&path, LIMITS, Some(&wrong)),
            "header sha256",
        );
    }

    #[test]
    fn tensor_ranges_past_the_file_end_fail_the_open() {
        let payload = [0u8; 64];
        let mut bytes = gguf(&[], &[entry("a", GgufType::F32, &payload)]);
        bytes.truncate(bytes.len() - 1);
        let dir = tempfile::tempdir().expect("tempdir");
        let path = write(&dir, &bytes);
        assert!(matches!(
            GgufWeights::open(&path, LIMITS, None),
            Err(EngError::Checkpoint(_))
        ));
    }

    #[test]
    fn a_file_length_change_after_open_fails_the_read() {
        let payload = [1u8; 16];
        let dir = tempfile::tempdir().expect("tempdir");
        let path = write(&dir, &gguf(&[], &[entry("a", GgufType::F32, &payload)]));
        let weights = GgufWeights::open(&path, LIMITS, None).expect("open");
        let mut file = OpenOptions::new().append(true).open(&path).expect("reopen");
        file.write_all(&[0]).expect("append");
        expect_checkpoint(weights.read_tensor("a"), "file length changed");
    }

    #[test]
    fn a_truncation_after_open_fails_before_the_read() {
        let payload = [1u8; 16];
        let dir = tempfile::tempdir().expect("tempdir");
        let path = write(&dir, &gguf(&[], &[entry("a", GgufType::F32, &payload)]));
        let weights = GgufWeights::open(&path, LIMITS, None).expect("open");
        let len = std::fs::metadata(&path).expect("metadata").len();
        let file = OpenOptions::new().write(true).open(&path).expect("reopen");
        file.set_len(len - 8).expect("truncate");
        expect_checkpoint(weights.read_tensor("a"), "file length changed");
    }

    #[test]
    fn two_threads_read_their_own_tensors() {
        let first = fixture("q4_k.blocks");
        let second = fixture("q6_k.blocks");
        let entries = [
            entry("first", GgufType::Q4K, &first),
            entry("second", GgufType::Q6K, &second),
        ];
        let dir = tempfile::tempdir().expect("tempdir");
        let weights =
            GgufWeights::open(&write(&dir, &gguf(&[], &entries)), LIMITS, None).expect("open");
        let read_many = |name: &str, want: &[u8]| {
            for _ in 0..200 {
                let (_, bytes) = weights.read_tensor(name).expect("read");
                assert_eq!(bytes, want, "{name}");
            }
        };
        std::thread::scope(|scope| {
            scope.spawn(|| read_many("first", &first));
            scope.spawn(|| read_many("second", &second));
        });
    }

    /// Two F32 tensors with distinct byte patterns, so a range read at the wrong
    /// tensor offset or from the wrong tensor shows up as a byte mismatch.
    fn range_fixture() -> (tempfile::TempDir, PathBuf, GgufWeights, Vec<u8>, Vec<u8>) {
        let first: Vec<u8> = (0..64u8).collect();
        let second: Vec<u8> = (0..40u8).map(|byte| 200 - byte).collect();
        let entries = [
            entry("first", GgufType::F32, &first),
            entry("second", GgufType::F32, &second),
        ];
        let dir = tempfile::tempdir().expect("tempdir");
        let path = write(&dir, &gguf(&[], &entries));
        let weights = GgufWeights::open(&path, LIMITS, None).expect("open");
        (dir, path, weights, first, second)
    }

    fn expect_bad_request<T: std::fmt::Debug>(result: Result<T, EngError>, expect: &str) {
        match result {
            Err(EngError::BadRequest(message)) => {
                assert!(message.contains(expect), "{expect:?} not in {message:?}");
            }
            other => panic!("expected BadRequest with {expect:?}, got {other:?}"),
        }
    }

    #[test]
    fn read_range_returns_the_bytes_at_each_offset_of_each_tensor() {
        let (_dir, _path, weights, first, second) = range_fixture();
        for (name, payload) in [("first", &first), ("second", &second)] {
            for (offset, len) in [(0, payload.len()), (0, 1), (7, 13), (payload.len() - 4, 4)] {
                let mut buf = vec![0xAAu8; len];
                weights
                    .read_range(name, offset as u64, &mut buf)
                    .expect("read the range");
                assert_eq!(buf, payload[offset..offset + len], "{name} {offset}+{len}");
            }
        }
    }

    #[test]
    fn read_range_chunks_concatenate_to_the_whole_tensor() {
        let (_dir, _path, weights, first, _second) = range_fixture();
        let mut joined = Vec::new();
        for offset in (0..first.len()).step_by(24) {
            let mut buf = vec![0u8; 24.min(first.len() - offset)];
            weights
                .read_range("first", offset as u64, &mut buf)
                .expect("read a chunk");
            joined.extend_from_slice(&buf);
        }
        let (_, whole) = weights.read_tensor("first").expect("read_tensor");
        assert_eq!(joined, first);
        assert_eq!(whole, first);
    }

    #[test]
    fn read_range_accepts_an_empty_range_at_the_tensor_end() {
        let (_dir, _path, weights, first, _second) = range_fixture();
        weights
            .read_range("first", first.len() as u64, &mut [])
            .expect("an empty range at the end lies inside the tensor");
    }

    #[test]
    fn read_range_rejects_ranges_outside_the_tensor_before_reading() {
        let (_dir, _path, weights, first, second) = range_fixture();
        let mut buf = [0x55u8; 8];
        // One byte past the end of `second`.
        let past_end = second.len() as u64 - 7;
        expect_bad_request(
            weights.read_range("second", past_end, &mut buf),
            "range of 8 bytes at offset 33 lies outside tensor second of 40 bytes",
        );
        expect_bad_request(
            weights.read_range("first", first.len() as u64 + 1, &mut []),
            "outside tensor first",
        );
        expect_bad_request(
            weights.read_range("first", u64::MAX - 3, &mut buf),
            "outside tensor first",
        );
        // `first` is followed by `second` in the file, so only the tensor bound
        // rejects this range.
        let mut long = vec![0u8; first.len() + 1];
        expect_bad_request(
            weights.read_range("first", 0, &mut long),
            "outside tensor first",
        );
        assert_eq!(buf, [0x55; 8], "a rejected range wrote into the buffer");
        assert!(long.iter().all(|&byte| byte == 0));
    }

    #[test]
    fn read_range_names_a_missing_tensor() {
        let (_dir, _path, weights, _first, _second) = range_fixture();
        expect_checkpoint(
            weights.read_range("absent", 0, &mut [0u8; 4]),
            "tensor absent is not in the file",
        );
    }

    #[test]
    fn read_range_fails_when_the_file_length_changed_after_open() {
        let (_dir, path, weights, _first, _second) = range_fixture();
        let mut file = OpenOptions::new().append(true).open(&path).expect("reopen");
        file.write_all(&[0]).expect("append");
        expect_checkpoint(
            weights.read_range("first", 0, &mut [0u8; 4]),
            "file length changed",
        );
    }

    #[test]
    fn read_range_fails_before_reading_when_the_file_shrank() {
        let (_dir, path, weights, _first, _second) = range_fixture();
        let len = std::fs::metadata(&path).expect("metadata").len();
        let file = OpenOptions::new().write(true).open(&path).expect("reopen");
        file.set_len(len - 8).expect("truncate");
        // The range lies in the cut-off tail: without the length check before
        // the read, the failure would be a short read, not a length change.
        expect_checkpoint(
            weights.read_range("second", 32, &mut [0u8; 8]),
            "file length changed",
        );
    }

    #[test]
    fn missing_tensors_are_checkpoint_errors() {
        let payload = [0u8; 8];
        let dir = tempfile::tempdir().expect("tempdir");
        let path = write(&dir, &gguf(&[], &[entry("a", GgufType::F32, &payload)]));
        let weights = GgufWeights::open(&path, LIMITS, None).expect("open");
        expect_checkpoint(weights.read_tensor("b"), "tensor b is not in the file");
    }

    #[test]
    fn a_missing_file_is_a_checkpoint_error() {
        let dir = tempfile::tempdir().expect("tempdir");
        expect_checkpoint(
            GgufWeights::open(&dir.path().join("absent.gguf"), LIMITS, None),
            "cannot open",
        );
    }

    #[test]
    fn short_reads_report_the_tensor() {
        let error = std::io::Error::from(ErrorKind::UnexpectedEof);
        expect_checkpoint::<()>(Err(read_error("w", &error)), "tensor w ends past");
        let error = std::io::Error::other("disk");
        expect_checkpoint::<()>(Err(read_error("w", &error)), "cannot read tensor w");
    }

    /// Per-tensor type inventory of a real file, for the kernel-routing
    /// receipts: when FIFI_GGUF_INVENTORY points at a GGUF the header opens
    /// and the histogram prints (run with --nocapture). Skipped when unset.
    #[test]
    fn real_file_type_inventory() {
        let Ok(path) = std::env::var("FIFI_GGUF_INVENTORY") else {
            eprintln!("skipped: FIFI_GGUF_INVENTORY is unset");
            return;
        };
        let limits = GgufLimits {
            max_header_bytes: 100_000_000,
            max_tensor_bytes: 8 << 30,
        };
        let weights = GgufWeights::open_any_type(std::path::Path::new(&path), limits, None)
            .unwrap_or_else(|error| panic!("the file must open: {error:?}"));
        let mut rows: Vec<(String, usize, u64, &str)> = Vec::new();
        for tensor in weights.header().tensors() {
            let ty = tensor.ggml_type();
            let elements: u64 = tensor.dims().iter().product();
            let blocks: u64 = elements / ty.block_elems();
            let bytes = ty.block_bytes().checked_mul(blocks).unwrap_or(0);
            match rows.iter_mut().find(|r| r.0 == ty.name()) {
                Some(row) => {
                    row.1 += 1;
                    row.2 += bytes;
                }
                None => rows.push((ty.name().to_owned(), 1, bytes, tensor.name())),
            }
        }
        rows.sort();
        for (ty, count, bytes, example) in &rows {
            println!("{ty:8} {count:4} tensors {bytes:14} bytes  e.g. {example}");
        }
        assert!(!rows.is_empty());
    }
}
