//! Safe wrappers of the row-batched argmax of a K-row verify window: [`CudaContext::argmax_rows`]
//! and [`CudaContext::window_status_fold`] (both kernels live in `native/resident/step_params.cu`).
//!
//! `argmax_rows` runs the greedy argmax of `argmax` (the largest value, the lowest index on a tie,
//! a NaN never wins) on each of `m` logits rows, with the same grid shape per row, so each row's
//! token is the token of the single-row kernel on that row. The result table holds one
//! `(token, bits)` pair per row, `bits` being that row's own mask 2 (a non-finite value in the row)
//! and nothing else: the single-row kernel stores `before | bits`, the return value of `atomicOr`
//! on the shared status word, which with K rows in flight depends on block scheduling, so a pair
//! could show another row's bits. No atomic runs in `argmax_rows` and it writes nothing into the
//! step parameters (not `status`, not `token_id`).
//!
//! `window_status_fold` runs after all rows. It ORs the pairs' bits and the EXL3 arithmetic flag
//! into `params.status` once, and stores the window word `w` (`status` after the OR) and a zero pad
//! word after the pairs. It only reads the pairs and clears nothing: `argmax_rows` stores every
//! pair whole in every window and the fold stores `w` whole, so the host reads the table
//! (`8 m + 8` bytes) after one synchronization with no preparation of the buffer. Router (mask 1),
//! invalid-state (mask 4) and arithmetic (mask 16) bits live in `w` only; the host refuses a
//! window with any non-zero pair bit or a non-zero `w` (`family::qwen36::krow::check_status`).
//!
//! `verify_greedy` runs after `argmax_rows` and compares the pair tokens with the window ids on the
//! device: the accepted run `a` (the leading rows `i < m - 1` whose token equals `ids[i + 1]`) and
//! the bonus (the token of row `a`) are stored after the window word and its pad, as words
//! `2 m + 2` and `2 m + 3`. The report of a window is therefore `8 m + 16` bytes, and the shortlist
//! records of the K-row shortlist (`resident::shortlist`) follow it at the next multiple of 16, so
//! the host reads all of it with one copy ([`WindowReport`]).
//!
//! Order of a K-row window's tail: `lm_head`, `argmax_rows`, `window_status_fold`, `verify_greedy`,
//! the K-row shortlist, the advance. The fold must be the last kernel that writes the status word;
//! the two after it write no status.

use crate::EngError;
use crate::kernels::cuda::context::{Claim, require_apart, require_fit};
use crate::kernels::cuda::native::resident::argmax_rows::{
    self as raw, ArgmaxRowsRaw, VerifyGreedyRaw, WindowFoldRaw,
};
use crate::kernels::cuda::resident::shortlist::RESULT_BYTES as SHORTLIST_RECORD_BYTES;
use crate::kernels::cuda::resident::step_params::{
    ARGMAX_PART_BYTES, MAX_LOGITS, RESULT_PAIR_BYTES, STEP_PARAMS_BYTES,
};
use crate::kernels::cuda::{CudaContext, DeviceRange, DeviceStream};

/// Most rows of one launch (a verify window has at most 8).
pub(crate) const MAX_ARGMAX_ROWS: usize = raw::MAX_ARGMAX_ROWS;

/// Bytes of the result table of an `m`-row window: `m` pairs, then the window word and a zero pad.
pub(crate) fn window_result_bytes(m: usize) -> usize {
    (m + 1) * RESULT_PAIR_BYTES
}

/// Bytes of the report of an `m`-row window: the table and the two `verify_greedy` words.
pub(crate) fn window_report_bytes(m: usize) -> usize {
    (m + 2) * RESULT_PAIR_BYTES
}

/// Byte offset, from the start of the table, of the first shortlist record of an `m`-row window:
/// the report rounded up to the 16 bytes a record needs.
pub(crate) fn window_shortlist_offset(m: usize) -> usize {
    window_report_bytes(m).next_multiple_of(16)
}

/// Bytes the host reads for an `m`-row window with the shortlist records.
pub(crate) fn window_full_report_bytes(m: usize) -> usize {
    window_shortlist_offset(m) + m * SHORTLIST_RECORD_BYTES
}

/// Bytes the host reads for an `m`-row window: the whole report with the shortlist records, or the
/// table and the verify words alone when the shortlist did not run.
pub(crate) fn window_read_bytes(m: usize, with_shortlist: bool) -> usize {
    if with_shortlist {
        window_full_report_bytes(m)
    } else {
        window_report_bytes(m)
    }
}

/// Bytes of the argmax scratch of `m` rows: one set of partials per row.
pub(crate) fn argmax_rows_part_bytes(m: usize) -> usize {
    m * ARGMAX_PART_BYTES
}

/// The arguments of one `argmax_rows` launch.
pub(crate) struct ArgmaxRowsCall<'b> {
    /// Rows of the window.
    pub(crate) m: usize,
    /// Values per row.
    pub(crate) n: usize,
    /// Floats between the starts of two logits rows (`>= n`).
    pub(crate) ld: usize,
    /// `(m - 1) * ld + n` floats, 4-byte aligned. Read only.
    pub(crate) logits: DeviceRange<'b>,
    /// Scratch of [`argmax_rows_part_bytes`], 4-byte aligned; needs no initialization.
    pub(crate) part: DeviceRange<'b>,
    /// The result table: at least `m` pairs (`8 m` bytes), 8-byte aligned. Pair `r` is
    /// `(token, bits)` of row `r`.
    pub(crate) result: DeviceRange<'b>,
}

/// The arguments of one `window_status_fold`.
pub(crate) struct WindowFoldCall<'b> {
    pub(crate) m: usize,
    /// The step parameters: `status` is ORed into.
    pub(crate) params: DeviceRange<'b>,
    /// The EXL3 sticky flag word; bit 0 is folded into mask 16 and the word is zeroed.
    pub(crate) flag: DeviceRange<'b>,
    /// The result table of [`window_result_bytes`] bytes, 8-byte aligned: the `m` pairs are read,
    /// `w` and the pad are written.
    pub(crate) result: DeviceRange<'b>,
}

/// The arguments of one `verify_greedy`.
pub(crate) struct VerifyGreedyCall<'b> {
    pub(crate) m: usize,
    /// The window ids: `m` int32 words, 4-byte aligned. Read only.
    pub(crate) ids: DeviceRange<'b>,
    /// The result table of [`window_report_bytes`] bytes, 8-byte aligned: the `m` pairs are read,
    /// the two words after the window word and its pad are written.
    pub(crate) result: DeviceRange<'b>,
}

/// One window's result table as the host reads it after the window's synchronization.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct WindowResult {
    /// The token of each row (-1 for a row with no comparable value).
    pub(crate) tokens: Vec<i32>,
    /// Each row's own status bits.
    pub(crate) bits: Vec<i32>,
    /// The window word.
    pub(crate) window: i32,
}

impl WindowResult {
    /// Decodes the `8 m + 8` bytes of an `m`-row table (little endian). `None` unless `bytes` is
    /// exactly that long and the pad word is zero.
    pub(crate) fn from_bytes(bytes: &[u8], m: usize) -> Option<Self> {
        if bytes.len() != window_result_bytes(m) {
            return None;
        }
        let words: Vec<i32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| i32::from_le_bytes(*word))
            .collect();
        (words[2 * m + 1] == 0).then(|| Self {
            tokens: (0..m).map(|row| words[2 * row]).collect(),
            bits: (0..m).map(|row| words[2 * row + 1]).collect(),
            window: words[2 * m],
        })
    }
}

/// One window's whole report as the host reads it with one copy after the window's
/// synchronization: the table, the two `verify_greedy` words and, when the K-row shortlist ran, one
/// record per row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct WindowReport {
    pub(crate) table: WindowResult,
    /// The accepted run the device compared: leading rows `i < m - 1` with `token[i] == ids[i + 1]`.
    pub(crate) accepted: i32,
    /// The token of row `accepted`.
    pub(crate) bonus: i32,
    /// The shortlist record of each row (empty when the shortlist did not run).
    pub(crate) shortlists: Vec<[u8; SHORTLIST_RECORD_BYTES]>,
}

impl WindowReport {
    /// Decodes the report of an `m`-row window: [`window_report_bytes`] bytes without the
    /// shortlist, [`window_full_report_bytes`] with it. `None` for any other length or a table whose
    /// pad word is not zero. The bytes between the report and the first record are padding and are
    /// not read.
    pub(crate) fn from_bytes(bytes: &[u8], m: usize, shortlist: bool) -> Option<Self> {
        if bytes.len() != window_read_bytes(m, shortlist) {
            return None;
        }
        let table = WindowResult::from_bytes(&bytes[..window_result_bytes(m)], m)?;
        let word = |at: usize| i32::from_le_bytes(bytes[at..at + 4].try_into().expect("a word"));
        let verify = window_result_bytes(m);
        let shortlists: Vec<[u8; SHORTLIST_RECORD_BYTES]> = if shortlist {
            bytes[window_shortlist_offset(m)..]
                .chunks_exact(SHORTLIST_RECORD_BYTES)
                .map(|record| record.try_into().expect("a record"))
                .collect()
        } else {
            Vec::new()
        };
        Some(Self {
            table,
            accepted: word(verify),
            bonus: word(verify + 4),
            shortlists,
        })
    }
}

fn check_m(m: usize) -> Result<std::ffi::c_int, EngError> {
    match std::ffi::c_int::try_from(m) {
        Ok(value) if (1..=MAX_ARGMAX_ROWS).contains(&m) => Ok(value),
        _ => Err(EngError::BadRequest(format!(
            "the CUDA argmax_rows takes 1 to {MAX_ARGMAX_ROWS} rows, got {m}"
        ))),
    }
}

impl CudaContext {
    /// Enqueues the row-batched greedy argmax (two kernels) on `stream`. Row `r` of `call.logits`
    /// gets its token in pair `r` of `call.result` and its own status bits next to it; nothing is
    /// written to the step parameters. The pairs are stored whole, so every launch rewrites every
    /// pair.
    pub(crate) fn argmax_rows(
        &self,
        stream: &DeviceStream,
        call: ArgmaxRowsCall<'_>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let m = check_m(call.m)?;
        let n = match std::ffi::c_int::try_from(call.n) {
            Ok(value) if (1..=MAX_LOGITS).contains(&call.n) => value,
            _ => {
                return Err(EngError::BadRequest(format!(
                    "the CUDA argmax_rows takes 1 to {MAX_LOGITS} values per row, got {}",
                    call.n
                )));
            }
        };
        if call.ld < call.n {
            return Err(EngError::BadRequest(format!(
                "the CUDA argmax_rows row stride {} is below the row length {}",
                call.ld, call.n
            )));
        }
        let overflow = || EngError::BadRequest("the CUDA argmax_rows sizes overflow".to_owned());
        let logits_bytes = (call.m - 1)
            .checked_mul(call.ld)
            .and_then(|floats| floats.checked_add(call.n))
            .and_then(|floats| floats.checked_mul(4))
            .ok_or_else(overflow)?;
        let part_bytes = argmax_rows_part_bytes(call.m);
        let result_bytes = call.m * RESULT_PAIR_BYTES;
        let logits = self.raw_range(&call.logits)?;
        let part = self.raw_range(&call.part)?;
        let result = self.raw_range(&call.result)?;
        require_fit("logits", &call.logits, logits_bytes, 4)?;
        require_fit("part", &call.part, part_bytes, 4)?;
        require_fit("result", &call.result, result_bytes, 8)?;
        let claims = [
            Claim::of("logits", &call.logits, logits_bytes),
            Claim::of("part", &call.part, part_bytes),
            Claim::of("result", &call.result, result_bytes),
        ];
        for (i, a) in claims.iter().enumerate() {
            for b in &claims[i + 1..] {
                require_apart(a, b)?;
            }
        }
        #[cfg(test)]
        self.note_row_kernel(crate::kernels::row_launches::RowKernel::ArgmaxRows);
        let raw_call = ArgmaxRowsRaw {
            logits,
            part,
            result,
            n,
            m,
            ld: call.ld,
        };
        self.launch_on(stream, move |raw_stream| {
            raw::argmax_rows_launch(raw_stream, &raw_call)
        })
    }

    /// Enqueues `window_status_fold` on `stream`: the last status writer of a K-row window. See
    /// the module doc for what it reads and writes.
    pub(crate) fn window_status_fold(
        &self,
        stream: &DeviceStream,
        call: WindowFoldCall<'_>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let m = check_m(call.m)?;
        let params = self.raw_range(&call.params)?;
        let flag = self.raw_range(&call.flag)?;
        let result = self.raw_range(&call.result)?;
        let table = window_result_bytes(call.m);
        require_fit("params", &call.params, STEP_PARAMS_BYTES, 4)?;
        require_fit("EXL3 flag", &call.flag, 4, 4)?;
        require_fit("result", &call.result, table, 8)?;
        let claims = [
            Claim::of("params", &call.params, STEP_PARAMS_BYTES),
            Claim::of("EXL3 flag", &call.flag, 4),
            Claim::of("result", &call.result, table),
        ];
        for (i, a) in claims.iter().enumerate() {
            for b in &claims[i + 1..] {
                require_apart(a, b)?;
            }
        }
        #[cfg(test)]
        self.note_row_kernel(crate::kernels::row_launches::RowKernel::StatusFold);
        let raw_call = WindowFoldRaw {
            params,
            flag,
            result,
            m,
        };
        self.launch_on(stream, move |raw_stream| {
            raw::window_status_fold_launch(raw_stream, &raw_call)
        })
    }

    /// Enqueues `verify_greedy` on `stream`: the device-side comparison of the pair tokens with
    /// the window ids. See the module doc for what it reads and writes.
    pub(crate) fn verify_greedy(
        &self,
        stream: &DeviceStream,
        call: VerifyGreedyCall<'_>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let m = check_m(call.m)?;
        let ids = self.raw_range(&call.ids)?;
        let result = self.raw_range(&call.result)?;
        let ids_bytes = call.m * size_of::<i32>();
        let table = window_report_bytes(call.m);
        require_fit("ids", &call.ids, ids_bytes, 4)?;
        require_fit("result", &call.result, table, 8)?;
        require_apart(
            &Claim::of("ids", &call.ids, ids_bytes),
            &Claim::of("result", &call.result, table),
        )?;
        #[cfg(test)]
        self.note_row_kernel(crate::kernels::row_launches::RowKernel::VerifyGreedy);
        let raw_call = VerifyGreedyRaw { ids, result, m };
        self.launch_on(stream, move |raw_stream| {
            raw::verify_greedy_launch(raw_stream, &raw_call)
        })
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod verify_tests;
