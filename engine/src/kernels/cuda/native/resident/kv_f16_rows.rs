//! Raw FFI to inference_cuda_kv_f16_rows of native/resident/kv_f16_rows.cu
//! (declared in native/resident/kv_f16_rows.h). Child of native::resident, so it
//! inherits the allow(unsafe_code) of native.rs.
//!
//! Safety invariants this module relies on:
//! - A device range is proved in bounds by RawRange::new before it gets here. This
//!   module proves, again and on the sizes of the call, that x holds rows*columns
//!   f32 values and flag holds one u32, both 4-byte aligned, and that x and flag
//!   share no byte with each other or with the f16 row output y (when present).
//!   The C entry repeats these checks on the addresses.
//! - The launch entry only queues work on the stream it is given and returns the
//!   launch status; a fault in the kernel surfaces at the next synchronize.
//! - The allocations stay valid until the queued launch finishes because
//!   inference_cuda_free synchronizes the whole device before cudaFree.
//! - Every call runs under the process dispatch guard and on the thread that owns
//!   the context.
//! - No raw pointer and no unsafe function leaves this module.

use std::ffi::{c_int, c_void};
use std::mem::offset_of;
use std::ptr::null_mut;

use super::{DeviceFailure, RawRange, RawStream, STAGE_NONE, Span, checked, invalid_argument};

unsafe extern "C" {
    #[link_name = "inference_cuda_kv_f16_rows"]
    fn kv_f16_rows_raw(
        args: *const KvF16ArgsRepr,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
}

/// Mirror of inference_cuda_kv_f16_args (kv_f16_rows.h), field for field.
#[repr(C)]
struct KvF16ArgsRepr {
    x: *const f32,
    x_bytes: usize,
    y: *mut f32,
    y_bytes: usize,
    flag: *mut u32,
    flag_bytes: usize,
    rows: usize,
    columns: usize,
}

const _: () = assert!(
    size_of::<KvF16ArgsRepr>() == 64
        && offset_of!(KvF16ArgsRepr, x) == 0
        && offset_of!(KvF16ArgsRepr, x_bytes) == 8
        && offset_of!(KvF16ArgsRepr, y) == 16
        && offset_of!(KvF16ArgsRepr, y_bytes) == 24
        && offset_of!(KvF16ArgsRepr, flag) == 32
        && offset_of!(KvF16ArgsRepr, flag_bytes) == 40
        && offset_of!(KvF16ArgsRepr, rows) == 48
        && offset_of!(KvF16ArgsRepr, columns) == 56
);

pub(in crate::kernels::cuda) struct KvF16RowsRaw<'a> {
    pub(in crate::kernels::cuda) x: RawRange<'a>,
    /// f16-bit output range; `None` runs the guard without a written half-row.
    pub(in crate::kernels::cuda) y: Option<RawRange<'a>>,
    pub(in crate::kernels::cuda) flag: RawRange<'a>,
    pub(in crate::kernels::cuda) rows: usize,
    pub(in crate::kernels::cuda) columns: usize,
}

pub(in crate::kernels::cuda) fn kv_f16_rows_launch(
    stream: &RawStream,
    call: &KvF16RowsRaw<'_>,
) -> Result<(), DeviceFailure> {
    let values = call
        .rows
        .checked_mul(call.columns)
        .filter(|values| *values <= (1usize << 32))
        .ok_or_else(invalid_argument)?;
    let in_bytes = values.checked_mul(4).ok_or_else(invalid_argument)?;
    let out_bytes = values.checked_mul(2).ok_or_else(invalid_argument)?;
    let mut checks: Vec<(&RawRange<'_>, usize)> = vec![(&call.x, in_bytes)];
    if let Some(y) = &call.y {
        checks.push((y, out_bytes));
    }
    checks.push((&call.flag, 4));
    for (i, (range, need)) in checks.iter().enumerate() {
        if range.bytes < *need || !range.offset.is_multiple_of(4) {
            return Err(invalid_argument());
        }
        for (range_j, need_j) in checks.iter().take(i) {
            if !Span::of(range, *need).apart(&Span::of(range_j, *need_j)) {
                return Err(invalid_argument());
            }
        }
    }
    let args = KvF16ArgsRepr {
        x: call.x.address().cast::<f32>().cast_const(),
        x_bytes: call.x.bytes,
        y: call
            .y
            .as_ref()
            .map(|y| y.address().cast())
            .unwrap_or(null_mut()),
        y_bytes: call.y.as_ref().map(|y| y.bytes).unwrap_or(0),
        flag: call.flag.address().cast(),
        flag_bytes: call.flag.bytes,
        rows: call.rows,
        columns: call.columns,
    };
    let mut stage = STAGE_NONE;
    // SAFETY: every live range is sized, aligned and disjoint. Context owns the
    // dispatch/stream lease, and allocation drops drain outstanding device use.
    let status = unsafe { kv_f16_rows_raw(&args, stream.0.as_ptr(), &mut stage) };
    checked(status, stage)
}
