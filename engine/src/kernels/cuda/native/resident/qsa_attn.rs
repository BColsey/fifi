//! Raw FFI to `inference_cuda_qsa_attn_launch` of `native/resident/qsa_attn.cu` (declared in
//! `native/resident/qsa_attn.h`): the sparse decode attention of Qwen3.8 Flash-Next QSA. Child of
//! `native::resident`, so it inherits the `allow(unsafe_code)` of `native.rs`.
//!
//! Safety invariants this file relies on:
//! - The entry is called only through `CudaContext::launch_on`, on the owning thread and under
//!   the dispatch guard. It only enqueues and returns the launch status.
//! - Every range is a `RawRange`, proved inside a live allocation by `RawRange::new`. Sizes,
//!   alignment and disjointness (`super::Span`) are proved again here on the extents the kernels
//!   touch; the C entry repeats the checks on the addresses.
//! - `pos` and `status` may be absent; an absent one is passed as null.
//! - A device allocation outlives the queued work because `inference_cuda_free` synchronizes the
//!   device before `cudaFree`.
//! - The argument struct is a live local that the entry reads only during the call. Its
//!   `repr(C)` mirror is private here and pinned by a `const` assert to the field order of
//!   `qsa_attn.h`. No raw pointer and no unsafe function leaves this module.

use std::ffi::{c_int, c_void};
use std::mem::{offset_of, size_of};

use super::{DeviceFailure, RawRange, RawStream, STAGE_NONE, Span, checked, invalid_argument};
use crate::kernels::cuda::native::resident::qsa_index::{QSA_MAX_M, all_apart, optional};

/// Limits of `qsa_attn.h`.
pub(in crate::kernels::cuda) const QSA_ATTN_HEAD: usize = 256;
pub(in crate::kernels::cuda) const QSA_ATTN_MAX_GROUP: usize = 32;
pub(in crate::kernels::cuda) const QSA_ATTN_MAX_KV_HEADS: usize = 64;
pub(in crate::kernels::cuda) const QSA_ATTN_MAX_ROWS: usize = (1 << 30) - 1;
pub(in crate::kernels::cuda) const QSA_ATTN_MAX_RATIO: usize = 64;
pub(in crate::kernels::cuda) const QSA_ATTN_MAX_BUDGET: usize = 1 << 24;
pub(in crate::kernels::cuda) const QSA_ATTN_MAX_SPLITS: usize = 128;
/// Floats per split of the scratch: a max, a sum and one partial value per dimension.
pub(in crate::kernels::cuda) const QSA_ATTN_PART_STRIDE: usize = QSA_ATTN_HEAD + 2;

const FLOAT: usize = 4;
const WIDE: usize = 16;

#[repr(C)]
struct QsaAttnArgsRaw {
    q: *const c_void,
    proj: *const c_void,
    k_cache: *const c_void,
    v_cache: *const c_void,
    chosen: *const c_void,
    pos: *const c_void,
    part: *mut c_void,
    out: *mut c_void,
    status: *mut c_void,
    ld: c_int,
    n_head: c_int,
    n_kv_head: c_int,
    rows: c_int,
    ratio: c_int,
    budget_blocks: c_int,
    max_splits: c_int,
    pos0: c_int,
    scale: f32,
    m: c_int,
}

const _: () = assert!(
    size_of::<QsaAttnArgsRaw>() == 112
        && offset_of!(QsaAttnArgsRaw, status) == 64
        && offset_of!(QsaAttnArgsRaw, ld) == 72
        && offset_of!(QsaAttnArgsRaw, pos0) == 100
        && offset_of!(QsaAttnArgsRaw, scale) == 104
        && offset_of!(QsaAttnArgsRaw, m) == 108
);

unsafe extern "C" {
    #[link_name = "inference_cuda_qsa_attn_launch"]
    fn qsa_attn_raw(
        args: *const QsaAttnArgsRaw,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
}

/// The geometry of one call: `m` tokens, `n_head` query heads over `n_kv_head` kv heads of 256, a
/// cache of `rows` positions, blocks of `ratio`, at most `budget_blocks` chosen blocks per token,
/// `max_splits` splits per kv head and token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::kernels::cuda) struct QsaAttnDims {
    pub(in crate::kernels::cuda) m: usize,
    pub(in crate::kernels::cuda) n_head: usize,
    pub(in crate::kernels::cuda) n_kv_head: usize,
    pub(in crate::kernels::cuda) rows: usize,
    pub(in crate::kernels::cuda) ratio: usize,
    pub(in crate::kernels::cuda) budget_blocks: usize,
    pub(in crate::kernels::cuda) max_splits: usize,
}

/// The bytes each range of a call touches. `gates` is the query and gate columns of one
/// projection row; `QsaAttnDims::proj_bytes` gives the extent of all `m` rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::kernels::cuda) struct QsaAttnBytes {
    pub(in crate::kernels::cuda) q: usize,
    pub(in crate::kernels::cuda) gates: usize,
    pub(in crate::kernels::cuda) cache: usize,
    pub(in crate::kernels::cuda) chosen: usize,
    pub(in crate::kernels::cuda) part: usize,
    pub(in crate::kernels::cuda) out: usize,
}

impl QsaAttnDims {
    /// True when every dimension is inside the limits of `qsa_attn.h`.
    pub(in crate::kernels::cuda) fn inside(self) -> bool {
        (1..=QSA_MAX_M).contains(&self.m)
            && (1..=QSA_ATTN_MAX_KV_HEADS).contains(&self.n_kv_head)
            && self.n_head >= self.n_kv_head
            && self.n_head.is_multiple_of(self.n_kv_head)
            && self.n_head / self.n_kv_head <= QSA_ATTN_MAX_GROUP
            && (1..=QSA_ATTN_MAX_ROWS).contains(&self.rows)
            && (1..=QSA_ATTN_MAX_RATIO).contains(&self.ratio)
            && (1..=QSA_ATTN_MAX_BUDGET).contains(&self.budget_blocks)
            && (1..=QSA_ATTN_MAX_SPLITS).contains(&self.max_splits)
    }

    /// The touched bytes, or `None` outside the limits. Inside them nothing overflows: the
    /// largest product is `2^30 * 64 * 256 * 2` bytes.
    pub(in crate::kernels::cuda) fn bytes(self) -> Option<QsaAttnBytes> {
        let rows_of = |per_token: usize| self.m * per_token * FLOAT;
        self.inside().then(|| QsaAttnBytes {
            q: rows_of(self.n_head * QSA_ATTN_HEAD),
            gates: self.n_head * 2 * QSA_ATTN_HEAD * FLOAT,
            cache: self.rows * self.n_kv_head * QSA_ATTN_HEAD * 2,
            chosen: rows_of(self.budget_blocks),
            part: rows_of(self.n_head * self.max_splits * QSA_ATTN_PART_STRIDE),
            out: rows_of(self.n_head * QSA_ATTN_HEAD),
        })
    }

    /// The extent of `m` projection rows `ld` floats apart, or `None` outside the limits, for an
    /// `ld` past `c_int`, or for a row narrower than the query and gate columns. `ld` is bounded
    /// first, so nothing overflows: the largest product is `2^15 * 2^31 * 4` bytes.
    pub(in crate::kernels::cuda) fn proj_bytes(self, ld: usize) -> Option<usize> {
        let gates = self.bytes()?.gates;
        c_int::try_from(ld).ok()?;
        (ld * FLOAT >= gates).then(|| (self.m - 1) * ld * FLOAT + gates)
    }

    fn ints(self) -> Option<[c_int; 7]> {
        Some([
            c_int::try_from(self.m).ok()?,
            c_int::try_from(self.n_head).ok()?,
            c_int::try_from(self.n_kv_head).ok()?,
            c_int::try_from(self.rows).ok()?,
            c_int::try_from(self.ratio).ok()?,
            c_int::try_from(self.budget_blocks).ok()?,
            c_int::try_from(self.max_splits).ok()?,
        ])
    }
}

pub(in crate::kernels::cuda) struct QsaAttnRaw<'a> {
    pub(in crate::kernels::cuda) dims: QsaAttnDims,
    pub(in crate::kernels::cuda) ld: usize,
    pub(in crate::kernels::cuda) scale: f32,
    pub(in crate::kernels::cuda) pos0: c_int,
    pub(in crate::kernels::cuda) q: RawRange<'a>,
    pub(in crate::kernels::cuda) proj: RawRange<'a>,
    pub(in crate::kernels::cuda) k_cache: RawRange<'a>,
    pub(in crate::kernels::cuda) v_cache: RawRange<'a>,
    pub(in crate::kernels::cuda) chosen: RawRange<'a>,
    pub(in crate::kernels::cuda) pos: Option<RawRange<'a>>,
    pub(in crate::kernels::cuda) part: RawRange<'a>,
    pub(in crate::kernels::cuda) out: RawRange<'a>,
    pub(in crate::kernels::cuda) status: Option<RawRange<'a>>,
}

fn fits(range: &RawRange<'_>, need: usize, align: usize) -> bool {
    range.bytes() >= need && range.offset().is_multiple_of(align)
}

fn check_attn(call: &QsaAttnRaw<'_>) -> Result<QsaAttnBytes, DeviceFailure> {
    let need = call.dims.bytes().ok_or_else(invalid_argument)?;
    let proj = call.dims.proj_bytes(call.ld).ok_or_else(invalid_argument)?;
    let word_fits = |r: &Option<RawRange<'_>>| r.as_ref().is_none_or(|r| fits(r, FLOAT, FLOAT));
    let sized = fits(&call.q, need.q, WIDE)
        && fits(&call.proj, proj, FLOAT)
        && fits(&call.k_cache, need.cache, WIDE)
        && fits(&call.v_cache, need.cache, WIDE)
        && fits(&call.chosen, need.chosen, FLOAT)
        && fits(&call.part, need.part, FLOAT)
        && fits(&call.out, need.out, FLOAT)
        && word_fits(&call.pos)
        && word_fits(&call.status);
    let mut written = vec![
        Span::of(&call.part, need.part),
        Span::of(&call.out, need.out),
    ];
    written.extend(call.status.as_ref().map(|range| Span::of(range, FLOAT)));
    let mut read = vec![
        Span::of(&call.q, need.q),
        Span::of(&call.proj, proj),
        Span::of(&call.k_cache, need.cache),
        Span::of(&call.v_cache, need.cache),
        Span::of(&call.chosen, need.chosen),
    ];
    read.extend(call.pos.as_ref().map(|range| Span::of(range, FLOAT)));
    if !(call.scale.is_finite() && sized && all_apart(&written, &read)) {
        return Err(invalid_argument());
    }
    Ok(need)
}

/// Queues the split and combine kernels on `stream`.
pub(in crate::kernels::cuda) fn qsa_attn_launch(
    stream: &RawStream,
    call: QsaAttnRaw<'_>,
) -> Result<(), DeviceFailure> {
    check_attn(&call)?;
    let [m, n_head, n_kv_head, rows, ratio, budget_blocks, max_splits] =
        call.dims.ints().ok_or_else(invalid_argument)?;
    let args = QsaAttnArgsRaw {
        q: call.q.address().cast_const(),
        proj: call.proj.address().cast_const(),
        k_cache: call.k_cache.address().cast_const(),
        v_cache: call.v_cache.address().cast_const(),
        chosen: call.chosen.address().cast_const(),
        pos: optional(&call.pos).cast_const(),
        part: call.part.address(),
        out: call.out.address(),
        status: optional(&call.status),
        ld: c_int::try_from(call.ld).map_err(|_| invalid_argument())?,
        n_head,
        n_kv_head,
        rows,
        ratio,
        budget_blocks,
        max_splits,
        pos0: call.pos0,
        scale: call.scale,
        m,
    };
    let mut stage = STAGE_NONE;
    // SAFETY: `check_attn` proved every range inside its live allocation (`RawRange::new`), large
    // enough for the call, aligned (16 bytes for q and the caches, 4 for the rest), and every
    // written range (part, out, status) apart from the other written ranges and from every read
    // range; the entry checks the addresses again. The allocations outlive the queued work
    // (`inference_cuda_free` synchronizes first). `args` is a live local that the entry reads
    // only during the call, `stream` is live, and `stage` is a writable c_int.
    let status = unsafe { qsa_attn_raw(&raw const args, stream.0.as_ptr(), &mut stage) };
    checked(status, stage)
}
