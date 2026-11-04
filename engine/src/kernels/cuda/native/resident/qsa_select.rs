//! Raw FFI to `inference_cuda_qsa_select_launch` of `native/resident/qsa_select.cu` (declared in
//! `native/resident/qsa_select.h`): the block selection of Qwen3.8 Flash-Next QSA attention.
//! Child of `native::resident`, so it inherits the `allow(unsafe_code)` of `native.rs`.
//!
//! Safety invariants this file relies on:
//! - The entry is called only through `CudaContext::launch_on`, on the owning thread and under
//!   the dispatch guard. It only enqueues and returns the launch status.
//! - Every range is a `RawRange`, proved inside a live allocation by `RawRange::new`. Sizes,
//!   alignment and disjointness (`super::Span`) are proved again here on the extents the kernel
//!   touches; the C entry repeats the checks on the addresses.
//! - `pos` and `status` may be absent; an absent one is passed as null.
//! - A device allocation outlives the queued work because `inference_cuda_free` synchronizes the
//!   device before `cudaFree`.
//! - The argument struct is a live local that the entry reads only during the call. Its
//!   `repr(C)` mirror is private here and pinned by a `const` assert to the field order of
//!   `qsa_select.h`. No raw pointer and no unsafe function leaves this module.

use std::ffi::{c_int, c_void};
use std::mem::{offset_of, size_of};

use super::{DeviceFailure, RawRange, RawStream, STAGE_NONE, Span, checked, invalid_argument};
use crate::kernels::cuda::native::resident::qsa_index::{
    QSA_MAX_CAP_BLOCKS, QSA_MAX_RATIO, all_apart, fits, optional,
};

const WORD: usize = 4;

#[repr(C)]
struct QsaSelectArgsRaw {
    scores: *const c_void,
    blocks: *mut c_void,
    pos: *const c_void,
    status: *mut c_void,
    ratio: c_int,
    cap_blocks: c_int,
    budget_blocks: c_int,
    pos0: c_int,
}

const _: () = assert!(
    size_of::<QsaSelectArgsRaw>() == 48
        && offset_of!(QsaSelectArgsRaw, status) == 24
        && offset_of!(QsaSelectArgsRaw, ratio) == 32
        && offset_of!(QsaSelectArgsRaw, pos0) == 44
);

unsafe extern "C" {
    #[link_name = "inference_cuda_qsa_select_launch"]
    fn qsa_select_raw(
        args: *const QsaSelectArgsRaw,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
}

/// Blocks of `ratio` positions, a score array of `cap_blocks`, and at most `budget_blocks`
/// selected blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::kernels::cuda) struct QsaSelectDims {
    pub(in crate::kernels::cuda) ratio: usize,
    pub(in crate::kernels::cuda) cap_blocks: usize,
    pub(in crate::kernels::cuda) budget_blocks: usize,
}

impl QsaSelectDims {
    /// True when every dimension is inside the limits of `qsa_select.h`.
    pub(in crate::kernels::cuda) fn inside(self) -> bool {
        (1..=QSA_MAX_RATIO).contains(&self.ratio)
            && (1..=QSA_MAX_CAP_BLOCKS).contains(&self.cap_blocks)
            && (1..=self.cap_blocks).contains(&self.budget_blocks)
    }
}

pub(in crate::kernels::cuda) struct QsaSelectRaw<'a> {
    pub(in crate::kernels::cuda) dims: QsaSelectDims,
    pub(in crate::kernels::cuda) pos0: c_int,
    pub(in crate::kernels::cuda) scores: RawRange<'a>,
    pub(in crate::kernels::cuda) blocks: RawRange<'a>,
    pub(in crate::kernels::cuda) pos: Option<RawRange<'a>>,
    pub(in crate::kernels::cuda) status: Option<RawRange<'a>>,
}

fn check_select(call: &QsaSelectRaw<'_>) -> bool {
    let dims = call.dims;
    if !dims.inside() {
        return false;
    }
    let (scores, blocks) = (dims.cap_blocks * WORD, dims.budget_blocks * WORD);
    let word_fits = |range: &Option<RawRange<'_>>| range.as_ref().is_none_or(|r| fits(r, WORD));
    let sized = fits(&call.scores, scores)
        && fits(&call.blocks, blocks)
        && word_fits(&call.pos)
        && word_fits(&call.status);
    let mut written = vec![Span::of(&call.blocks, blocks)];
    written.extend(call.status.as_ref().map(|range| Span::of(range, WORD)));
    let mut read = vec![Span::of(&call.scores, scores)];
    read.extend(call.pos.as_ref().map(|range| Span::of(range, WORD)));
    sized && all_apart(&written, &read)
}

/// Queues the selection kernel on `stream`: the indices of the best blocks at the position, in
/// ascending order.
pub(in crate::kernels::cuda) fn qsa_select_launch(
    stream: &RawStream,
    call: QsaSelectRaw<'_>,
) -> Result<(), DeviceFailure> {
    if !check_select(&call) {
        return Err(invalid_argument());
    }
    let dims = call.dims;
    let as_int = |v: usize| c_int::try_from(v).map_err(|_| invalid_argument());
    let args = QsaSelectArgsRaw {
        scores: call.scores.address().cast_const(),
        blocks: call.blocks.address(),
        pos: optional(&call.pos).cast_const(),
        status: optional(&call.status),
        ratio: as_int(dims.ratio)?,
        cap_blocks: as_int(dims.cap_blocks)?,
        budget_blocks: as_int(dims.budget_blocks)?,
        pos0: call.pos0,
    };
    let mut stage = STAGE_NONE;
    // SAFETY: `check_select` proved every range inside its live allocation (`RawRange::new`),
    // large enough, 4-byte aligned, and `blocks` and `status`, which the kernel writes, apart
    // from each other and from `scores` and `pos`; the entry checks the addresses again. The
    // allocations outlive the queued work (`inference_cuda_free` synchronizes first). `args` is
    // a live local read only during the call, `stream` is live and `stage` is writable.
    let status = unsafe { qsa_select_raw(&raw const args, stream.0.as_ptr(), &mut stage) };
    checked(status, stage)
}
