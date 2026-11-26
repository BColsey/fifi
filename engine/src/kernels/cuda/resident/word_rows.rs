//! Rows of 32-bit words zeroed or copied in one launch: [`CudaContext::zero_word_rows`] and
//! [`CudaContext::copy_word_rows`] queue one launch, one block per row, where a step would
//! otherwise queue a device memset or copy for each of many small bookkeeping rows (status
//! words, receipts). Words move as bits.
//!
//! Every refusal is a `BadRequest` before any native call, and a refused call changes nothing:
//! the backend is usable; the call names 1 to [`MAX_WORD_ROWS`] rows; every row is a positive
//! whole number of words at a 4-byte aligned offset; a copy's pair holds the same number of
//! words; no written row shares a byte with another row of the call (copy sources may share
//! bytes with each other).

#[cfg(test)]
mod tests;

use std::mem::size_of;

use crate::EngError;
use crate::kernels::cuda::{
    CudaContext, DeviceRange, DeviceStream,
    context::{Claim, require_apart, require_fit},
};

use super::super::native::resident::word_rows as native;
/// The most rows one launch takes.
pub(crate) const MAX_WORD_ROWS: usize = super::super::native::resident::word_rows::MAX_WORD_ROWS;

const WORD: usize = size_of::<u32>();

fn require_rows(count: usize) -> Result<(), EngError> {
    if (1..=MAX_WORD_ROWS).contains(&count) {
        return Ok(());
    }
    Err(EngError::BadRequest(format!(
        "the CUDA word rows take 1 to {MAX_WORD_ROWS} rows, got {count}"
    )))
}

fn require_words(name: &str, range: &DeviceRange<'_>) -> Result<(), EngError> {
    require_fit(name, range, WORD, WORD)?;
    if range.bytes().is_multiple_of(WORD) {
        return Ok(());
    }
    Err(EngError::BadRequest(format!(
        "the CUDA {name} holds {} bytes, not a whole number of words",
        range.bytes()
    )))
}

/// `BadRequest` when a written claim shares a byte with another written claim or a read one.
fn require_written_apart(written: &[Claim<'_>], read: &[Claim<'_>]) -> Result<(), EngError> {
    for (at, claim) in written.iter().enumerate() {
        for other in written[at + 1..].iter().chain(read) {
            require_apart(claim, other)?;
        }
    }
    Ok(())
}

impl CudaContext {
    /// Zero every row of `rows` in one launch on `stream`.
    pub(crate) fn zero_word_rows(
        &self,
        stream: &DeviceStream,
        rows: &[DeviceRange<'_>],
    ) -> Result<(), EngError> {
        require_rows(rows.len())?;
        for row in rows {
            require_words("word row", row)?;
        }
        let claims: Vec<Claim<'_>> = rows
            .iter()
            .map(|row| Claim::of("word row", row, row.bytes()))
            .collect();
        require_written_apart(&claims, &[])?;
        self.ensure_usable()?;
        let raw = rows
            .iter()
            .map(|row| self.raw_range(row))
            .collect::<Result<Vec<_>, EngError>>()?;
        self.launch_on(stream, |s| native::zero_word_rows(s, &raw))
    }

    /// Copy every `(source, destination)` pair of `pairs` in one launch on `stream`.
    pub(crate) fn copy_word_rows(
        &self,
        stream: &DeviceStream,
        pairs: &[(DeviceRange<'_>, DeviceRange<'_>)],
    ) -> Result<(), EngError> {
        require_rows(pairs.len())?;
        for (source, destination) in pairs {
            require_words("word row source", source)?;
            require_words("word row destination", destination)?;
            if source.bytes() != destination.bytes() {
                return Err(EngError::BadRequest(format!(
                    "the CUDA word row copy moves {} bytes into {}",
                    source.bytes(),
                    destination.bytes()
                )));
            }
        }
        let sources: Vec<Claim<'_>> = pairs
            .iter()
            .map(|(source, _)| Claim::of("word row source", source, source.bytes()))
            .collect();
        let destinations: Vec<Claim<'_>> = pairs
            .iter()
            .map(|(_, destination)| {
                Claim::of("word row destination", destination, destination.bytes())
            })
            .collect();
        require_written_apart(&destinations, &sources)?;
        self.ensure_usable()?;
        let raw = pairs
            .iter()
            .map(|(source, destination)| {
                Ok((self.raw_range(source)?, self.raw_range(destination)?))
            })
            .collect::<Result<Vec<_>, EngError>>()?;
        self.launch_on(stream, |s| native::copy_word_rows(s, &raw))
    }
}
