//! The `ngram_embedding.safetensors` file of a Qwen3.8 Flash-Next EXL3 pack, opened through the
//! local-storage guard and read by byte range.
//!
//! The file is 30.40 GiB at 3.05 bpw and is read at random: 16 rows of 102 bytes per token. It
//! is outside the weight index, so [`Checkpoint::open`](super::Checkpoint) never sees it and this
//! is its only opener. [`NgramFile::open`] judges the path with the caller's
//! [`LocalGuard`](crate::storage_guard::LocalGuard) before it opens anything, reads the 8-byte
//! prefix and the JSON header (bounded), validates them against the model's layout through
//! [`NgramTable::from_header`], and keeps the file for positioned reads. A row read takes `&self`
//! and no lock; the table is immutable after the open.
//!
//! The caller decodes: [`NgramFile::read_row`] returns the packed row bytes, which
//! [`NgramDecoder::decode_row`](super::ngram::NgramDecoder::decode_row) turns into values.

// Staged: the Flash-Next EXL3 loader is the first production caller.
#![cfg_attr(not(test), allow(dead_code))]

use std::{
    fs::File,
    os::unix::fs::FileExt,
    path::{Path, PathBuf},
};

use super::ngram::{NgramAux, NgramLayout, NgramTable};
use crate::{EngError, storage_guard::LocalGuard};

/// Upper bound on the JSON header; the real one is 21,328 bytes.
const HEADER_LIMIT: u64 = 64 * 1024 * 1024;

/// A validated, locally stored n-gram table file.
#[derive(Debug)]
pub(crate) struct NgramFile {
    file: File,
    path: PathBuf,
    table: NgramTable,
}

fn io_error(path: &Path, error: &std::io::Error) -> EngError {
    EngError::Checkpoint(format!("EXL3 n-gram table {}: {error}", path.display()))
}

/// Judges the open descriptor, not the name: a file swapped for a link to a network mount between the
/// check of the path and the open is caught here, because `/proc/self/fd/<n>` resolves to the
/// opened file's real path. Linux only; elsewhere the check of the name stands.
#[cfg(target_os = "linux")]
fn check_open_file(guard: &LocalGuard, file: &File) -> Result<(), EngError> {
    use std::os::fd::AsRawFd;
    let fd_path = PathBuf::from(format!("/proc/self/fd/{}", file.as_raw_fd()));
    guard.check(&fd_path, "model weights")
}

#[cfg(not(target_os = "linux"))]
fn check_open_file(_guard: &LocalGuard, _file: &File) -> Result<(), EngError> {
    Ok(())
}

impl NgramFile {
    /// Refuses a path on a network-backed filesystem before opening it, then validates the header
    /// against `layout` and the length of the file. `key` is the tensor-name prefix of the table
    /// (`model.language_model.layers.<PLE layer>.ple.ple_embedding.ngram_embedding`).
    pub(crate) fn open(
        path: &Path,
        guard: &LocalGuard,
        key: &str,
        layout: NgramLayout,
    ) -> Result<Self, EngError> {
        Self::open_judging(path, guard, key, layout, &check_open_file)
    }

    /// [`Self::open`] with the judgement of the open descriptor passed in, so a test can show that the
    /// open runs it, once, after the file is opened and before any byte of it is read.
    fn open_judging(
        path: &Path,
        guard: &LocalGuard,
        key: &str,
        layout: NgramLayout,
        judge_open_file: &dyn Fn(&LocalGuard, &File) -> Result<(), EngError>,
    ) -> Result<Self, EngError> {
        guard.check(path, "model weights")?;
        // A FIFO or a device named like the table would block or hang the open.
        if !std::fs::metadata(path)
            .map_err(|error| io_error(path, &error))?
            .is_file()
        {
            return Err(EngError::Checkpoint(format!(
                "EXL3 n-gram table {} is not a regular file",
                path.display()
            )));
        }
        let file = File::open(path).map_err(|error| io_error(path, &error))?;
        judge_open_file(guard, &file)?;
        let metadata = file.metadata().map_err(|error| io_error(path, &error))?;
        if !metadata.is_file() {
            return Err(EngError::Checkpoint(format!(
                "EXL3 n-gram table {} is not a regular file",
                path.display()
            )));
        }
        let length = metadata.len();
        let mut prefix = [0u8; 8];
        file.read_exact_at(&mut prefix, 0)
            .map_err(|error| io_error(path, &error))?;
        let header_length = u64::from_le_bytes(prefix);
        if header_length > HEADER_LIMIT || header_length.saturating_add(8) > length {
            return Err(EngError::Checkpoint(format!(
                "EXL3 n-gram table {}: the header claims {header_length} bytes in a file of \
                 {length} (limit {HEADER_LIMIT})",
                path.display()
            )));
        }
        // At most HEADER_LIMIT, so it fits usize.
        let mut bytes = vec![0u8; 8 + header_length as usize];
        file.read_exact_at(&mut bytes, 0)
            .map_err(|error| io_error(path, &error))?;
        let table = NgramTable::from_header(&bytes, length, key, layout)?;
        Ok(Self {
            file,
            path: path.to_path_buf(),
            table,
        })
    }

    pub(crate) fn table(&self) -> &NgramTable {
        &self.table
    }

    /// The packed bytes of global row `row` into `out`, which must be `table().row_bytes()` long.
    pub(crate) fn read_row(&self, row: u64, out: &mut [u8]) -> Result<(), EngError> {
        let range = self.table.row_range(row)?;
        if out.len() != self.table.row_bytes() {
            return Err(EngError::BadRequest(format!(
                "an n-gram row is {} bytes, the buffer has {}",
                self.table.row_bytes(),
                out.len()
            )));
        }
        self.file
            .read_exact_at(out, range.start)
            .map_err(|error| io_error(&self.path, &error))
    }

    /// The bytes of one small auxiliary tensor (a few KiB), read once at load.
    pub(crate) fn read_aux(&self, which: NgramAux) -> Result<Vec<u8>, EngError> {
        let range = self.table.aux_range(which);
        let mut bytes = vec![0u8; (range.end - range.start) as usize];
        self.file
            .read_exact_at(&mut bytes, range.start)
            .map_err(|error| io_error(&self.path, &error))?;
        Ok(bytes)
    }
}

#[cfg(test)]
#[path = "ngram_file_tests.rs"]
pub(crate) mod tests;
