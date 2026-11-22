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
/// Where the eight f64 mass partials of [`launch_mass`] sit in the shortlist scratch.
pub(in crate::kernels::cuda) const MASS_OFFSET: usize = 2992;
/// Blocks of the mass launch, one f64 partial each.
pub(in crate::kernels::cuda) const MASS_PARTS: usize = 8;
/// Scratch bytes the mass launch needs: the shortlist scratch plus the partials.
pub(in crate::kernels::cuda) const MASS_SCRATCH_BYTES: usize = 3056;
/// Most rows of one K-row launch.
pub(in crate::kernels::cuda) const MAX_ROWS: usize = 16;
/// The wide width a start can ask for (Bolt's top_k 64 default); [`K`] is the narrow one.
pub(in crate::kernels::cuda) const WIDE_K: usize = 64;
/// The tiles every width ranks over.
const TILES: usize = 16;

/// The byte layout of a single-row launch at one ranked width: `TILES` partial records, the final
/// record, then the eight f64 mass partials. Width 20 is the layout of the constants above.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Shape {
    pub(crate) k: usize,
    /// Bytes of one record: `k` (logit, id) pairs, the non-finite word and three pad words.
    pub(crate) part_bytes: usize,
    pub(crate) result_offset: usize,
    pub(crate) scratch_bytes: usize,
    pub(crate) mass_offset: usize,
    pub(crate) mass_scratch_bytes: usize,
}

impl Shape {
    /// The layout at `k`, or `None` for a width the kernels do not instantiate.
    pub(in crate::kernels::cuda) const fn of(k: usize) -> Option<Self> {
        if k != K && k != WIDE_K {
            return None;
        }
        let part_bytes = k * 8 + 16;
        Some(Self {
            k,
            part_bytes,
            result_offset: TILES * part_bytes,
            scratch_bytes: (TILES + 1) * part_bytes,
            mass_offset: (TILES + 1) * part_bytes,
            mass_scratch_bytes: (TILES + 1) * part_bytes + MASS_PARTS * 8,
        })
    }
}
const _: () = {
    let narrow = match Shape::of(K) {
        Some(shape) => shape,
        None => panic!("the narrow width has a layout"),
    };
    assert!(narrow.part_bytes == RESULT_BYTES && narrow.result_offset == RESULT_OFFSET);
    assert!(narrow.scratch_bytes == SCRATCH_BYTES && narrow.mass_offset == MASS_OFFSET);
    assert!(narrow.mass_scratch_bytes == MASS_SCRATCH_BYTES);
};
/// Bytes of the partials of one row (16 tiles of one record each): the offset of the final record
/// of the single-row launch.
pub(in crate::kernels::cuda) const PARTS_BYTES: usize = RESULT_OFFSET;

#[repr(C)]
struct Args {
    logits: *const c_void,
    scratch: *mut c_void,
    n: c_int,
    k: c_int,
}
const _: () =
    assert!(size_of::<Args>() == 24 && offset_of!(Args, n) == 16 && offset_of!(Args, k) == 20);
/// Mirror of `inference_cuda_shortlist_mass_args`.
#[repr(C)]
struct MassArgs {
    logits: *const c_void,
    scratch: *mut c_void,
    n: c_int,
    inv_temperature: f32,
    k: c_int,
    reserved: c_int,
}
const _: () = assert!(
    size_of::<MassArgs>() == 32
        && offset_of!(MassArgs, n) == 16
        && offset_of!(MassArgs, inv_temperature) == 20
        && offset_of!(MassArgs, k) == 24
);
const _: () = assert!(MASS_OFFSET + MASS_PARTS * 8 == MASS_SCRATCH_BYTES);
const _: () = assert!(SCRATCH_BYTES <= MASS_OFFSET);
// The mass launch runs in the scratch the decode step lends the argmax partials.
const _: () = assert!(MASS_SCRATCH_BYTES <= super::step_params::ARGMAX_PART_BYTES);
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
    k: c_int,
    reserved: c_int,
}
const _: () = assert!(
    size_of::<RowsArgs>() == 56
        && offset_of!(RowsArgs, logits) == 0
        && offset_of!(RowsArgs, scratch) == 8
        && offset_of!(RowsArgs, out) == 16
        && offset_of!(RowsArgs, ld) == 24
        && offset_of!(RowsArgs, scratch_stride) == 32
        && offset_of!(RowsArgs, n) == 40
        && offset_of!(RowsArgs, m) == 44
        && offset_of!(RowsArgs, k) == 48
);
unsafe extern "C" {
    #[link_name = "inference_cuda_shortlist_launch"]
    fn launch_raw(args: *const Args, stream: *mut c_void, stage: *mut c_int) -> c_int;
    #[link_name = "inference_cuda_shortlist_mass_launch"]
    fn launch_mass_raw(args: *const MassArgs, stream: *mut c_void, stage: *mut c_int) -> c_int;
    #[link_name = "inference_cuda_shortlist_rows_launch"]
    fn launch_rows_raw(args: *const RowsArgs, stream: *mut c_void, stage: *mut c_int) -> c_int;
}
pub(in crate::kernels::cuda) struct ShortlistRaw<'a> {
    pub(in crate::kernels::cuda) logits: RawRange<'a>,
    pub(in crate::kernels::cuda) scratch: RawRange<'a>,
    pub(in crate::kernels::cuda) n: usize,
    pub(in crate::kernels::cuda) k: usize,
}
pub(in crate::kernels::cuda) fn launch(
    stream: &RawStream,
    io: ShortlistRaw<'_>,
) -> Result<(), DeviceFailure> {
    let bytes = io.n.checked_mul(4).ok_or_else(invalid_argument)?;
    let shape = Shape::of(io.k).ok_or_else(invalid_argument)?;
    if !(1..=MAX_LOGITS).contains(&io.n)
        || io.logits.bytes < bytes
        || !io.logits.offset.is_multiple_of(4)
        || io.scratch.bytes < shape.scratch_bytes
        || !io.scratch.offset.is_multiple_of(16)
        || !Span::of(&io.logits, bytes).apart(&Span::of(&io.scratch, shape.scratch_bytes))
    {
        return Err(invalid_argument());
    }
    let args = Args {
        logits: io.logits.address().cast_const(),
        scratch: io.scratch.address(),
        n: io.n as c_int,
        k: io.k as c_int,
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
    pub(in crate::kernels::cuda) k: usize,
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
    let shape = Shape::of(io.k).ok_or_else(invalid_argument)?;
    let out_bytes =
        io.m.checked_mul(shape.part_bytes)
            .ok_or_else(invalid_argument)?;
    let logits = Span::of(&io.logits, logits_bytes);
    let scratch = Span::of(&io.scratch, scratch_bytes);
    let out = Span::of(&io.out, out_bytes);
    if !((1..=MAX_ROWS).contains(&io.m)
        && (1..=MAX_LOGITS).contains(&io.n)
        && io.ld >= io.n
        && io.scratch_stride >= shape.result_offset
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
        k: io.k as c_int,
        reserved: 0,
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

/// What the mass launch reads and writes: the logits row the shortlist ranked and the scratch that
/// holds its record and receives the partials.
pub(in crate::kernels::cuda) struct MassRaw<'a> {
    pub(in crate::kernels::cuda) logits: RawRange<'a>,
    pub(in crate::kernels::cuda) scratch: RawRange<'a>,
    pub(in crate::kernels::cuda) n: usize,
    pub(in crate::kernels::cuda) inv_temperature: f32,
    pub(in crate::kernels::cuda) k: usize,
}

/// Queues the softmax mass kernel after a shortlist launch on the same scratch and stream.
pub(in crate::kernels::cuda) fn launch_mass(
    stream: &RawStream,
    io: MassRaw<'_>,
) -> Result<(), DeviceFailure> {
    let bytes = io.n.checked_mul(4).ok_or_else(invalid_argument)?;
    let shape = Shape::of(io.k).ok_or_else(invalid_argument)?;
    if !(1..=MAX_LOGITS).contains(&io.n)
        || !io.inv_temperature.is_finite()
        || io.inv_temperature <= 0.0
        || io.logits.bytes < bytes
        || !io.logits.offset.is_multiple_of(4)
        || io.scratch.bytes < shape.mass_scratch_bytes
        || !io.scratch.offset.is_multiple_of(16)
        || !Span::of(&io.logits, bytes).apart(&Span::of(&io.scratch, shape.mass_scratch_bytes))
    {
        return Err(invalid_argument());
    }
    let args = MassArgs {
        logits: io.logits.address().cast_const(),
        scratch: io.scratch.address(),
        n: io.n as c_int,
        inv_temperature: io.inv_temperature,
        k: io.k as c_int,
        reserved: 0,
    };
    let mut stage = STAGE_NONE;
    // SAFETY: the checks above proved the logits extent, the scratch extent (the record the kernel
    // reads and the partials it writes), the alignments and the disjointness of the two spans.
    // The live ranges and the argument struct outlive this launch-only call; native free drains
    // the queued work.
    let status = unsafe { launch_mass_raw(&raw const args, stream.as_ptr(), &mut stage) };
    checked(status, stage)
}
