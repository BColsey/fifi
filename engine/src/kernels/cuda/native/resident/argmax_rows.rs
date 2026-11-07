//! Raw entries of the row-batched argmax of a K-row verify window: `argmax_rows`,
//! `window_status_fold` and `verify_greedy`, all in `native/resident/step_params.cu` (declared in
//! `step_params.h`).
//! Child module of `native/resident.rs`, so it inherits the `allow(unsafe_code)` of `native.rs`.
//!
//! Safety invariants this file relies on:
//! - Each entry is called only through `CudaContext::launch_on`, on the owning thread and under the
//!   dispatch guard. The entries only enqueue and return.
//! - Every range is a `RawRange`, proved inside a live allocation by `RawRange::new`. Sizes,
//!   alignment and disjointness (`super::Span`) are proved again here, before the call, on the
//!   extents the kernels touch; the C entries repeat the alignment checks on the addresses.
//! - A device allocation outlives the queued work because `inference_cuda_free` synchronizes the
//!   device before `cudaFree`.
//! - The C argument structs are live locals that the entries read only during the call. Their
//!   `repr(C)` mirrors are private here and pinned by `const` asserts to the `static_assert`s of
//!   `step_params.h`. No raw pointer and no unsafe function leaves this module.

use std::ffi::{c_int, c_void};
use std::mem::{offset_of, size_of};

use super::step_params::{ARGMAX_PART_BYTES, MAX_LOGITS, RESULT_PAIR_BYTES, STEP_PARAMS_BYTES};
use super::{DeviceFailure, RawRange, RawStream, STAGE_NONE, Span, checked, invalid_argument};

/// Most rows of one `argmax_rows` launch and of one `window_status_fold`.
pub(in crate::kernels::cuda) const MAX_ARGMAX_ROWS: usize = 16;

const WORD_ALIGN: usize = 4;
const PAIR_ALIGN: usize = 8;
const FLAG_BYTES: usize = 4;

/// Mirror of `inference_cuda_argmax_rows_args`.
#[repr(C)]
struct ArgmaxRowsArgsRaw {
    logits: *const c_void,
    part: *mut c_void,
    result: *mut c_void,
    ld: i64,
    n: c_int,
    m: c_int,
}

/// Mirror of `inference_cuda_verify_greedy_args`.
#[repr(C)]
struct VerifyGreedyArgsRaw {
    ids: *const c_void,
    result: *mut c_void,
    m: c_int,
}

/// Mirror of `inference_cuda_window_fold_args`.
#[repr(C)]
struct WindowFoldArgsRaw {
    params: *mut c_void,
    flag: *mut c_void,
    result: *mut c_void,
    m: c_int,
}

const _: () = assert!(
    size_of::<ArgmaxRowsArgsRaw>() == 40
        && offset_of!(ArgmaxRowsArgsRaw, logits) == 0
        && offset_of!(ArgmaxRowsArgsRaw, part) == 8
        && offset_of!(ArgmaxRowsArgsRaw, result) == 16
        && offset_of!(ArgmaxRowsArgsRaw, ld) == 24
        && offset_of!(ArgmaxRowsArgsRaw, n) == 32
        && offset_of!(ArgmaxRowsArgsRaw, m) == 36
);
const _: () = assert!(
    size_of::<WindowFoldArgsRaw>() == 32
        && offset_of!(WindowFoldArgsRaw, params) == 0
        && offset_of!(WindowFoldArgsRaw, flag) == 8
        && offset_of!(WindowFoldArgsRaw, result) == 16
        && offset_of!(WindowFoldArgsRaw, m) == 24
);
const _: () = assert!(
    size_of::<VerifyGreedyArgsRaw>() == 24
        && offset_of!(VerifyGreedyArgsRaw, ids) == 0
        && offset_of!(VerifyGreedyArgsRaw, result) == 8
        && offset_of!(VerifyGreedyArgsRaw, m) == 16
);
unsafe extern "C" {
    #[link_name = "inference_cuda_verify_greedy_launch"]
    fn verify_greedy_raw(
        args: *const VerifyGreedyArgsRaw,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
    #[link_name = "inference_cuda_argmax_rows_launch"]
    fn argmax_rows_raw(
        args: *const ArgmaxRowsArgsRaw,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
    #[link_name = "inference_cuda_window_status_fold_launch"]
    fn window_status_fold_raw(
        args: *const WindowFoldArgsRaw,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
}

/// What one `argmax_rows` launch reads and writes: `m` logits rows of `n` values at `ld` floats
/// apart, `m` sets of partials, and the `2 m` result words.
pub(in crate::kernels::cuda) struct ArgmaxRowsRaw<'a> {
    pub(in crate::kernels::cuda) logits: RawRange<'a>,
    pub(in crate::kernels::cuda) part: RawRange<'a>,
    pub(in crate::kernels::cuda) result: RawRange<'a>,
    pub(in crate::kernels::cuda) n: c_int,
    pub(in crate::kernels::cuda) m: c_int,
    pub(in crate::kernels::cuda) ld: usize,
}

fn rows_in_range(m: c_int) -> Option<usize> {
    usize::try_from(m)
        .ok()
        .filter(|m| (1..=MAX_ARGMAX_ROWS).contains(m))
}

/// The bytes of the `m` result pairs and of the pairs plus the window word and its pad.
fn pair_table_bytes(m: usize) -> usize {
    m * RESULT_PAIR_BYTES
}

fn check_argmax_rows(call: &ArgmaxRowsRaw<'_>) -> Result<(usize, usize), DeviceFailure> {
    let m = rows_in_range(call.m).ok_or_else(invalid_argument)?;
    let n = usize::try_from(call.n)
        .ok()
        .filter(|n| (1..=MAX_LOGITS).contains(n) && *n <= call.ld)
        .ok_or_else(invalid_argument)?;
    // The last row ends at (m - 1) * ld + n floats.
    let logits_bytes = (m - 1)
        .checked_mul(call.ld)
        .and_then(|floats| floats.checked_add(n))
        .and_then(|floats| floats.checked_mul(4))
        .ok_or_else(invalid_argument)?;
    let part_bytes = m * ARGMAX_PART_BYTES;
    let result_bytes = pair_table_bytes(m);
    let sized = fits(&call.logits, logits_bytes, WORD_ALIGN)
        && fits(&call.part, part_bytes, WORD_ALIGN)
        && fits(&call.result, result_bytes, PAIR_ALIGN);
    let logits = Span::of(&call.logits, logits_bytes);
    let part = Span::of(&call.part, part_bytes);
    let result = Span::of(&call.result, result_bytes);
    if !(sized && logits.apart(&part) && logits.apart(&result) && part.apart(&result)) {
        return Err(invalid_argument());
    }
    Ok((m, logits_bytes))
}

/// Queues the two kernels of the row-batched argmax.
pub(in crate::kernels::cuda) fn argmax_rows_launch(
    stream: &RawStream,
    call: &ArgmaxRowsRaw<'_>,
) -> Result<(), DeviceFailure> {
    check_argmax_rows(call)?;
    let args = ArgmaxRowsArgsRaw {
        logits: call.logits.address().cast_const(),
        part: call.part.address(),
        result: call.result.address(),
        ld: i64::try_from(call.ld).map_err(|_| invalid_argument())?,
        n: call.n,
        m: call.m,
    };
    let mut stage = STAGE_NONE;
    // SAFETY: `check_argmax_rows` proved every range inside its live allocation, large enough for
    // the call (`logits` for `(m - 1) * ld + n` floats, `part` for `m` sets of partials, `result`
    // for `m` pairs), aligned as the entry requires (4, and 8 for `result`), and that the three
    // extents share no byte, so the writes to `part` and `result` cannot alias the reads of
    // `logits`. The allocations outlive the queued work (`inference_cuda_free` synchronizes
    // first). `args` is a live local that the entry reads only during the call. `stream` is a live
    // handle for the call.
    let status = unsafe { argmax_rows_raw(&args, stream.0.as_ptr(), &mut stage) };
    checked(status, stage)
}

/// What one `window_status_fold` reads and writes: the status word of `params`, the EXL3 flag word,
/// and the result table (`m` pairs, then the window word and its pad).
pub(in crate::kernels::cuda) struct WindowFoldRaw<'a> {
    pub(in crate::kernels::cuda) params: RawRange<'a>,
    pub(in crate::kernels::cuda) flag: RawRange<'a>,
    pub(in crate::kernels::cuda) result: RawRange<'a>,
    pub(in crate::kernels::cuda) m: c_int,
}

/// Queues the window status fold.
pub(in crate::kernels::cuda) fn window_status_fold_launch(
    stream: &RawStream,
    call: &WindowFoldRaw<'_>,
) -> Result<(), DeviceFailure> {
    let m = rows_in_range(call.m).ok_or_else(invalid_argument)?;
    check_params(&call.params)?;
    // `m` pairs, the window word and its pad.
    let table_bytes = pair_table_bytes(m + 1);
    let params = Span::of(&call.params, STEP_PARAMS_BYTES);
    let flag = Span::of(&call.flag, FLAG_BYTES);
    let table = Span::of(&call.result, table_bytes);
    if !(fits(&call.flag, FLAG_BYTES, WORD_ALIGN)
        && fits(&call.result, table_bytes, PAIR_ALIGN)
        && params.apart(&flag)
        && params.apart(&table)
        && flag.apart(&table))
    {
        return Err(invalid_argument());
    }
    let args = WindowFoldArgsRaw {
        params: call.params.address(),
        flag: call.flag.address(),
        result: call.result.address(),
        m: call.m,
    };
    let mut stage = STAGE_NONE;
    // SAFETY: `check_params`, `fits` and the three `apart` tests proved `params` (32 bytes), `flag`
    // (4 bytes) and the result table (`m + 1` pairs) inside their live allocations, aligned as the
    // entry requires (4, 4 and 8), `m` in range, and the three extents disjoint, so the writes
    // cannot alias; the entry repeats the alignment checks on the addresses. The allocations
    // outlive the queued work (`inference_cuda_free` synchronizes first). `args` is a live local
    // that the entry reads only during the call. `stream` is a live handle for the call.
    let status = unsafe { window_status_fold_raw(&args, stream.0.as_ptr(), &mut stage) };
    checked(status, stage)
}

/// What one `verify_greedy` launch reads and writes: the `m` window ids, and the result table (the
/// `m` pairs are read; the two words after the window word and its pad are written).
pub(in crate::kernels::cuda) struct VerifyGreedyRaw<'a> {
    pub(in crate::kernels::cuda) ids: RawRange<'a>,
    pub(in crate::kernels::cuda) result: RawRange<'a>,
    pub(in crate::kernels::cuda) m: c_int,
}

/// Bytes of the table `verify_greedy` works on: `m` pairs, the window word and its pad, and the
/// two words it stores.
fn report_table_bytes(m: usize) -> usize {
    pair_table_bytes(m + 2)
}

/// Queues the device-side greedy comparison of a window.
pub(in crate::kernels::cuda) fn verify_greedy_launch(
    stream: &RawStream,
    call: &VerifyGreedyRaw<'_>,
) -> Result<(), DeviceFailure> {
    let m = rows_in_range(call.m).ok_or_else(invalid_argument)?;
    let ids_bytes = m * WORD_ALIGN;
    let table_bytes = report_table_bytes(m);
    let ids = Span::of(&call.ids, ids_bytes);
    let table = Span::of(&call.result, table_bytes);
    if !(fits(&call.ids, ids_bytes, WORD_ALIGN)
        && fits(&call.result, table_bytes, PAIR_ALIGN)
        && ids.apart(&table))
    {
        return Err(invalid_argument());
    }
    let args = VerifyGreedyArgsRaw {
        ids: call.ids.address().cast_const(),
        result: call.result.address(),
        m: call.m,
    };
    let mut stage = STAGE_NONE;
    // SAFETY: `rows_in_range`, `fits` and `apart` proved `m` in range, `ids` (`m` words) and the
    // result table (`m + 2` pairs) inside their live allocations, aligned as the entry requires (4
    // and 8), and the two extents disjoint, so the two stored words cannot alias the ids; the entry
    // repeats the alignment checks on the addresses. The allocations outlive the queued work
    // (`inference_cuda_free` synchronizes first). `args` is a live local that the entry reads only
    // during the call. `stream` is a live handle for the call.
    let status = unsafe { verify_greedy_raw(&args, stream.0.as_ptr(), &mut stage) };
    checked(status, stage)
}

fn fits(range: &RawRange<'_>, need: usize, align: usize) -> bool {
    range.bytes >= need && range.offset.is_multiple_of(align)
}

fn check_params(params: &RawRange<'_>) -> Result<(), DeviceFailure> {
    if fits(params, STEP_PARAMS_BYTES, WORD_ALIGN) {
        return Ok(());
    }
    Err(invalid_argument())
}
