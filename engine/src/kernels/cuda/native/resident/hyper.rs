//! Raw FFI to `inference_cuda_hyper_read_launch` and `inference_cuda_hyper_apply_launch` of
//! `native/resident/hyper.cu` (declared in `native/resident/hyper.h`): the hyper-connection read
//! and apply of Qwen3.8 Flash-Next. Child of `native::resident`, so it inherits the
//! `allow(unsafe_code)` of `native.rs`.
//!
//! Safety invariants this file relies on:
//! - The entries are called only through `CudaContext::launch_on`, on the owning thread and under
//!   the dispatch guard. They only enqueue and return the launch status.
//! - Every range is a `RawRange`, proved inside a live allocation by `RawRange::new`. Sizes,
//!   alignment and disjointness (`super::Span`) are proved again here on the extents the kernels
//!   touch; the C entries repeat the checks on the addresses.
//! - `inject` and `gate` are both given or both absent; an absent one is passed as null.
//! - A device allocation outlives the queued work because `inference_cuda_free` synchronizes the
//!   device before `cudaFree`.
//! - The argument structs are live locals that the entries read only during the call. Their
//!   `repr(C)` mirrors are private here and pinned by `const` asserts to the field order of
//!   `hyper.h`. No raw pointer and no unsafe function leaves this module.

use std::ffi::{c_int, c_void};
use std::mem::{offset_of, size_of};
use std::ptr;

use super::{DeviceFailure, RawRange, RawStream, STAGE_NONE, Span, checked, invalid_argument};

/// Limits of `hyper.h`.
pub(in crate::kernels::cuda) const HYPER_MAX_M: usize = 32768;
pub(in crate::kernels::cuda) const HYPER_MAX_BRANCHES: usize = 16;
pub(in crate::kernels::cuda) const HYPER_MAX_HIDDEN: usize = 65536;
pub(in crate::kernels::cuda) const HYPER_MAX_WIDTH: usize = 1 << 18;
pub(in crate::kernels::cuda) const HYPER_MAX_RANK: usize = 4096;

const FLOAT: usize = 4;
const HALF: usize = 2;

/// Element type of every weight of one hyper-connection site.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum HyperDtype {
    F16,
    Bf16,
}

impl HyperDtype {
    fn code(self) -> c_int {
        match self {
            Self::F16 => 0,
            Self::Bf16 => 1,
        }
    }
}

/// Mirror of `inference_cuda_hyper_read_args`.
#[repr(C)]
struct HyperReadArgsRaw {
    x: *const c_void,
    norm_w: *const c_void,
    down: *const c_void,
    up: *const c_void,
    inject: *const c_void,
    normed: *mut c_void,
    low: *mut c_void,
    mix: *mut c_void,
    mixed: *mut c_void,
    gate: *mut c_void,
    m: c_int,
    branches: c_int,
    hidden: c_int,
    rank: c_int,
    weight_dtype: c_int,
    eps: f32,
}

const _: () = assert!(
    size_of::<HyperReadArgsRaw>() == 104
        && offset_of!(HyperReadArgsRaw, x) == 0
        && offset_of!(HyperReadArgsRaw, norm_w) == 8
        && offset_of!(HyperReadArgsRaw, down) == 16
        && offset_of!(HyperReadArgsRaw, up) == 24
        && offset_of!(HyperReadArgsRaw, inject) == 32
        && offset_of!(HyperReadArgsRaw, normed) == 40
        && offset_of!(HyperReadArgsRaw, low) == 48
        && offset_of!(HyperReadArgsRaw, mix) == 56
        && offset_of!(HyperReadArgsRaw, mixed) == 64
        && offset_of!(HyperReadArgsRaw, gate) == 72
        && offset_of!(HyperReadArgsRaw, m) == 80
        && offset_of!(HyperReadArgsRaw, branches) == 84
        && offset_of!(HyperReadArgsRaw, hidden) == 88
        && offset_of!(HyperReadArgsRaw, rank) == 92
        && offset_of!(HyperReadArgsRaw, weight_dtype) == 96
        && offset_of!(HyperReadArgsRaw, eps) == 100
);

/// Mirror of `inference_cuda_hyper_apply_args`.
#[repr(C)]
struct HyperApplyArgsRaw {
    x: *mut c_void,
    gate: *const c_void,
    block: *const c_void,
    m: c_int,
    branches: c_int,
    hidden: c_int,
}

const _: () = assert!(
    size_of::<HyperApplyArgsRaw>() == 40
        && offset_of!(HyperApplyArgsRaw, x) == 0
        && offset_of!(HyperApplyArgsRaw, gate) == 8
        && offset_of!(HyperApplyArgsRaw, block) == 16
        && offset_of!(HyperApplyArgsRaw, m) == 24
        && offset_of!(HyperApplyArgsRaw, branches) == 28
        && offset_of!(HyperApplyArgsRaw, hidden) == 32
);

unsafe extern "C" {
    #[link_name = "inference_cuda_hyper_read_launch"]
    fn hyper_read_raw(
        args: *const HyperReadArgsRaw,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;

    #[link_name = "inference_cuda_hyper_apply_launch"]
    fn hyper_apply_raw(
        args: *const HyperApplyArgsRaw,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
}

/// The geometry shared by read and apply.
#[derive(Debug, Clone, Copy)]
pub(in crate::kernels::cuda) struct HyperDims {
    pub(in crate::kernels::cuda) m: usize,
    pub(in crate::kernels::cuda) branches: usize,
    pub(in crate::kernels::cuda) hidden: usize,
}

/// What one read launch touches; see `hyper.h` for the layouts.
pub(in crate::kernels::cuda) struct HyperReadRaw<'a> {
    pub(in crate::kernels::cuda) dims: HyperDims,
    pub(in crate::kernels::cuda) rank: usize,
    pub(in crate::kernels::cuda) dtype: HyperDtype,
    pub(in crate::kernels::cuda) eps: f32,
    pub(in crate::kernels::cuda) x: RawRange<'a>,
    pub(in crate::kernels::cuda) norm_w: RawRange<'a>,
    pub(in crate::kernels::cuda) down: RawRange<'a>,
    pub(in crate::kernels::cuda) up: RawRange<'a>,
    pub(in crate::kernels::cuda) inject: Option<RawRange<'a>>,
    pub(in crate::kernels::cuda) normed: RawRange<'a>,
    pub(in crate::kernels::cuda) low: RawRange<'a>,
    pub(in crate::kernels::cuda) mix: RawRange<'a>,
    pub(in crate::kernels::cuda) mixed: RawRange<'a>,
    pub(in crate::kernels::cuda) gate: Option<RawRange<'a>>,
}

/// What one apply launch touches: `x` is read and written in place.
pub(in crate::kernels::cuda) struct HyperApplyRaw<'a> {
    pub(in crate::kernels::cuda) dims: HyperDims,
    pub(in crate::kernels::cuda) x: RawRange<'a>,
    pub(in crate::kernels::cuda) gate: RawRange<'a>,
    pub(in crate::kernels::cuda) block: RawRange<'a>,
}

/// Bytes each range of a read launch must hold.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct HyperReadBytes {
    pub(crate) x: usize,
    pub(crate) norm_w: usize,
    pub(crate) down: usize,
    pub(crate) up: usize,
    pub(crate) inject: usize,
    pub(crate) normed: usize,
    pub(crate) low: usize,
    pub(crate) mix: usize,
    pub(crate) mixed: usize,
    pub(crate) gate: usize,
}

/// `None` outside the limits of `hyper.h` or when a size overflows.
pub(in crate::kernels::cuda) fn read_need(dims: HyperDims, rank: usize) -> Option<HyperReadBytes> {
    let width = dims_width(dims)?;
    if !(1..=HYPER_MAX_RANK).contains(&rank) {
        return None;
    }
    let rows = |count: usize, per_row: usize, element: usize| {
        count.checked_mul(per_row)?.checked_mul(element)
    };
    Some(HyperReadBytes {
        x: rows(dims.m, width, FLOAT)?,
        norm_w: width.checked_mul(HALF)?,
        down: rows(rank, width, HALF)?,
        up: rows(width, rank, HALF)?,
        inject: rows(dims.branches, width, HALF)?,
        normed: rows(dims.m, width, FLOAT)?,
        low: rows(dims.m, rank, FLOAT)?,
        mix: rows(dims.m, width, FLOAT)?,
        mixed: rows(dims.m, dims.hidden, FLOAT)?,
        gate: rows(dims.m, dims.branches, FLOAT)?,
    })
}

/// `branches * hidden` when every dimension is inside the limits of `hyper.h`.
pub(in crate::kernels::cuda) fn dims_width(dims: HyperDims) -> Option<usize> {
    let inside = (1..=HYPER_MAX_M).contains(&dims.m)
        && (1..=HYPER_MAX_BRANCHES).contains(&dims.branches)
        && (1..=HYPER_MAX_HIDDEN).contains(&dims.hidden);
    let width = dims.branches.checked_mul(dims.hidden)?;
    (inside && width <= HYPER_MAX_WIDTH).then_some(width)
}

fn fits(range: &RawRange<'_>, need: usize, align: usize) -> bool {
    range.bytes() >= need && range.offset().is_multiple_of(align)
}

fn ints(dims: HyperDims) -> Option<(c_int, c_int, c_int)> {
    Some((
        c_int::try_from(dims.m).ok()?,
        c_int::try_from(dims.branches).ok()?,
        c_int::try_from(dims.hidden).ok()?,
    ))
}

/// Written ranges are apart from each other and from every read range; read ranges may share
/// bytes.
fn check_read(call: &HyperReadRaw<'_>) -> Result<HyperReadBytes, DeviceFailure> {
    let need = read_need(call.dims, call.rank).ok_or_else(invalid_argument)?;
    let paired = call.inject.is_some() == call.gate.is_some();
    let eps_ok = call.eps.is_finite() && call.eps >= 0.0;
    let optional_fits = |range: &Option<RawRange<'_>>, bytes: usize, align: usize| {
        range.as_ref().is_none_or(|range| fits(range, bytes, align))
    };
    let sized = fits(&call.x, need.x, FLOAT)
        && fits(&call.norm_w, need.norm_w, HALF)
        && fits(&call.down, need.down, HALF)
        && fits(&call.up, need.up, HALF)
        && optional_fits(&call.inject, need.inject, HALF)
        && fits(&call.normed, need.normed, FLOAT)
        && fits(&call.low, need.low, FLOAT)
        && fits(&call.mix, need.mix, FLOAT)
        && fits(&call.mixed, need.mixed, FLOAT)
        && optional_fits(&call.gate, need.gate, FLOAT);
    let mut written = vec![
        Span::of(&call.normed, need.normed),
        Span::of(&call.low, need.low),
        Span::of(&call.mix, need.mix),
        Span::of(&call.mixed, need.mixed),
    ];
    if let Some(gate) = &call.gate {
        written.push(Span::of(gate, need.gate));
    }
    let mut read = vec![
        Span::of(&call.x, need.x),
        Span::of(&call.norm_w, need.norm_w),
        Span::of(&call.down, need.down),
        Span::of(&call.up, need.up),
    ];
    if let Some(inject) = &call.inject {
        read.push(Span::of(inject, need.inject));
    }
    let apart = written.iter().enumerate().all(|(index, out)| {
        read.iter().all(|input| out.apart(input))
            && written[index + 1..].iter().all(|other| out.apart(other))
    });
    if !(paired && eps_ok && sized && apart) {
        return Err(invalid_argument());
    }
    Ok(need)
}

/// Queues the four read kernels on `stream` (norm, down and inject, up, combine).
pub(in crate::kernels::cuda) fn hyper_read_launch(
    stream: &RawStream,
    call: HyperReadRaw<'_>,
) -> Result<(), DeviceFailure> {
    check_read(&call)?;
    let (m, branches, hidden) = ints(call.dims).ok_or_else(invalid_argument)?;
    let rank = c_int::try_from(call.rank).map_err(|_| invalid_argument())?;
    let optional = |range: &Option<RawRange<'_>>| {
        range
            .as_ref()
            .map_or(ptr::null_mut(), |range| range.address())
    };
    let args = HyperReadArgsRaw {
        x: call.x.address().cast_const(),
        norm_w: call.norm_w.address().cast_const(),
        down: call.down.address().cast_const(),
        up: call.up.address().cast_const(),
        inject: optional(&call.inject).cast_const(),
        normed: call.normed.address(),
        low: call.low.address(),
        mix: call.mix.address(),
        mixed: call.mixed.address(),
        gate: optional(&call.gate),
        m,
        branches,
        hidden,
        rank,
        weight_dtype: call.dtype.code(),
        eps: call.eps,
    };
    let mut stage = STAGE_NONE;
    // SAFETY: `check_read` proved every range inside its live allocation (`RawRange::new`),
    // large enough for the call, aligned (f32 ranges to 4, weights to 2), every written range
    // apart from the other written ranges and from every read range, and `inject` and `gate` both
    // given or both null; the entry checks the addresses again. The allocations outlive the
    // queued work (`inference_cuda_free` synchronizes first). `args` is a live local that the
    // entry reads only during the call, `stream` is live, and `stage` is a writable c_int.
    let status = unsafe { hyper_read_raw(&raw const args, stream.0.as_ptr(), &mut stage) };
    checked(status, stage)
}

/// Queues the apply kernel on `stream`: `x[t][b][d] += gate[t][b] * block[t][d]`.
pub(in crate::kernels::cuda) fn hyper_apply_launch(
    stream: &RawStream,
    call: HyperApplyRaw<'_>,
) -> Result<(), DeviceFailure> {
    let dims = call.dims;
    let width = dims_width(dims).ok_or_else(invalid_argument)?;
    let sizes = (
        dims.m.checked_mul(width).and_then(|v| v.checked_mul(FLOAT)),
        dims.m
            .checked_mul(dims.branches)
            .and_then(|v| v.checked_mul(FLOAT)),
        dims.m
            .checked_mul(dims.hidden)
            .and_then(|v| v.checked_mul(FLOAT)),
    );
    let (Some(x_bytes), Some(gate_bytes), Some(block_bytes)) = sizes else {
        return Err(invalid_argument());
    };
    let x = Span::of(&call.x, x_bytes);
    let ok = fits(&call.x, x_bytes, FLOAT)
        && fits(&call.gate, gate_bytes, FLOAT)
        && fits(&call.block, block_bytes, FLOAT)
        && x.apart(&Span::of(&call.gate, gate_bytes))
        && x.apart(&Span::of(&call.block, block_bytes));
    if !ok {
        return Err(invalid_argument());
    }
    let (m, branches, hidden) = ints(dims).ok_or_else(invalid_argument)?;
    let args = HyperApplyArgsRaw {
        x: call.x.address(),
        gate: call.gate.address().cast_const(),
        block: call.block.address().cast_const(),
        m,
        branches,
        hidden,
    };
    let mut stage = STAGE_NONE;
    // SAFETY: the checks above proved the three ranges inside their live allocations, large
    // enough, 4-byte aligned, and `x`, which the kernel writes, apart from `gate` and `block`;
    // the entry checks the addresses again. The allocations outlive the queued work, `args` is a
    // live local read only during the call, `stream` is live and `stage` is writable.
    let status = unsafe { hyper_apply_raw(&raw const args, stream.0.as_ptr(), &mut stage) };
    checked(status, stage)
}
