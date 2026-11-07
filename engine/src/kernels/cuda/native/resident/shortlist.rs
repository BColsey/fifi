//! Native device shortlist. RawRange proves live allocation ownership; sizes,alignment and
//! input/scratch disjointness are checked again here. The entry only enqueues on a live stream;
//! buffers outlive queued work because native free synchronizes. No pointer escapes this module.
use super::{DeviceFailure, RawRange, RawStream, STAGE_NONE, Span, checked, invalid_argument};
use std::ffi::{c_int, c_void};
use std::mem::{offset_of, size_of};

pub(in crate::kernels::cuda) const K: usize = 20;
pub(in crate::kernels::cuda) const RESULT_BYTES: usize = 176;
pub(in crate::kernels::cuda) const RESULT_OFFSET: usize = 2816;
pub(in crate::kernels::cuda) const SCRATCH_BYTES: usize = 2992;
pub(in crate::kernels::cuda) const MAX_LOGITS: usize = 1 << 30;
/// Most rows of one K-row launch.
pub(in crate::kernels::cuda) const MAX_ROWS: usize = 16;
/// Bytes of the partials of one row (16 tiles of one record each): the offset of the final record
/// of the single-row launch.
pub(in crate::kernels::cuda) const PARTS_BYTES: usize = RESULT_OFFSET;

#[repr(C)]
struct Args {
    logits: *const c_void,
    scratch: *mut c_void,
    n: c_int,
}
const _: () = assert!(size_of::<Args>() == 24 && offset_of!(Args, n) == 16);
/// Mirror of `inference_cuda_shortlist_rows_args`.
#[repr(C)]
struct RowsArgs {
    logits: *const c_void,
    scratch: *mut c_void,
    out: *mut c_void,
    ld: i64,
    scratch_stride: i64,
    n: c_int,
    m: c_int,
}
const _: () = assert!(
    size_of::<RowsArgs>() == 48
        && offset_of!(RowsArgs, logits) == 0
        && offset_of!(RowsArgs, scratch) == 8
        && offset_of!(RowsArgs, out) == 16
        && offset_of!(RowsArgs, ld) == 24
        && offset_of!(RowsArgs, scratch_stride) == 32
        && offset_of!(RowsArgs, n) == 40
        && offset_of!(RowsArgs, m) == 44
);
unsafe extern "C" {
    #[link_name = "inference_cuda_shortlist_launch"]
    fn launch_raw(args: *const Args, stream: *mut c_void, stage: *mut c_int) -> c_int;
    #[link_name = "inference_cuda_shortlist_rows_launch"]
    fn launch_rows_raw(args: *const RowsArgs, stream: *mut c_void, stage: *mut c_int) -> c_int;
}
pub(in crate::kernels::cuda) struct ShortlistRaw<'a> {
    pub(in crate::kernels::cuda) logits: RawRange<'a>,
    pub(in crate::kernels::cuda) scratch: RawRange<'a>,
    pub(in crate::kernels::cuda) n: usize,
}
pub(in crate::kernels::cuda) fn launch(
    stream: &RawStream,
    io: ShortlistRaw<'_>,
) -> Result<(), DeviceFailure> {
    let bytes = io.n.checked_mul(4).ok_or_else(invalid_argument)?;
    if !(1..=MAX_LOGITS).contains(&io.n)
        || io.logits.bytes < bytes
        || !io.logits.offset.is_multiple_of(4)
        || io.scratch.bytes < SCRATCH_BYTES
        || !io.scratch.offset.is_multiple_of(16)
        || !Span::of(&io.logits, bytes).apart(&Span::of(&io.scratch, SCRATCH_BYTES))
    {
        return Err(invalid_argument());
    }
    let args = Args {
        logits: io.logits.address().cast_const(),
        scratch: io.scratch.address(),
        n: io.n as c_int,
    };
    let mut stage = STAGE_NONE;
    // SAFETY: live ranges above hold the checked extents,alignments and disjoint spans.
    // The live stream/argument struct outlive this launch-only call;native free drains work.
    let status = unsafe { launch_raw(&raw const args, stream.as_ptr(), &mut stage) };
    checked(status, stage)
}

/// What one K-row launch reads and writes: `m` logits rows of `n` values `ld` floats apart, the
/// partials of each row (`PARTS_BYTES` at `scratch_stride` apart), and the `m` final records.
pub(in crate::kernels::cuda) struct ShortlistRowsRaw<'a> {
    pub(in crate::kernels::cuda) logits: RawRange<'a>,
    pub(in crate::kernels::cuda) scratch: RawRange<'a>,
    pub(in crate::kernels::cuda) out: RawRange<'a>,
    pub(in crate::kernels::cuda) n: usize,
    pub(in crate::kernels::cuda) m: usize,
    pub(in crate::kernels::cuda) ld: usize,
    pub(in crate::kernels::cuda) scratch_stride: usize,
}

/// Queues the two kernels of the K-row shortlist.
pub(in crate::kernels::cuda) fn launch_rows(
    stream: &RawStream,
    io: ShortlistRowsRaw<'_>,
) -> Result<(), DeviceFailure> {
    let logits_bytes = (io.m.checked_sub(1).ok_or_else(invalid_argument)?)
        .checked_mul(io.ld)
        .and_then(|floats| floats.checked_add(io.n))
        .and_then(|floats| floats.checked_mul(4))
        .ok_or_else(invalid_argument)?;
    let scratch_bytes =
        io.m.checked_mul(io.scratch_stride)
            .ok_or_else(invalid_argument)?;
    let out_bytes =
        io.m.checked_mul(RESULT_BYTES)
            .ok_or_else(invalid_argument)?;
    let logits = Span::of(&io.logits, logits_bytes);
    let scratch = Span::of(&io.scratch, scratch_bytes);
    let out = Span::of(&io.out, out_bytes);
    if !((1..=MAX_ROWS).contains(&io.m)
        && (1..=MAX_LOGITS).contains(&io.n)
        && io.ld >= io.n
        && io.scratch_stride >= PARTS_BYTES
        && io.scratch_stride.is_multiple_of(16)
        && io.logits.bytes >= logits_bytes
        && io.logits.offset.is_multiple_of(4)
        && io.scratch.bytes >= scratch_bytes
        && io.scratch.offset.is_multiple_of(16)
        && io.out.bytes >= out_bytes
        && io.out.offset.is_multiple_of(16)
        && logits.apart(&scratch)
        && logits.apart(&out)
        && scratch.apart(&out))
    {
        return Err(invalid_argument());
    }
    let args = RowsArgs {
        logits: io.logits.address().cast_const(),
        scratch: io.scratch.address(),
        out: io.out.address(),
        ld: i64::try_from(io.ld).map_err(|_| invalid_argument())?,
        scratch_stride: i64::try_from(io.scratch_stride).map_err(|_| invalid_argument())?,
        n: io.n as c_int,
        m: io.m as c_int,
    };
    let mut stage = STAGE_NONE;
    // SAFETY: the checks above proved `m` and `n` in range, `logits` large enough for the last row,
    // `scratch` for `m` strides and `out` for `m` records, aligned as the entry requires (4, 16 and
    // 16), and the three extents disjoint, so the writes to `scratch` and `out` cannot alias the
    // reads of `logits`. The live ranges outlive the queued work (native free synchronizes). `args`
    // is a live local that the entry reads only during the call; the stream is live for the call.
    let status = unsafe { launch_rows_raw(&raw const args, stream.as_ptr(), &mut stage) };
    checked(status, stage)
}
