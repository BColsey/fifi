//! Raw FFI to `inference_cuda_qsa_index_launch` and `inference_cuda_qsa_score_launch` of
//! `native/resident/qsa_index.cu` (declared in `native/resident/qsa_index.h`): the block indexer
//! of Qwen3.8 Flash-Next QSA attention. Child of `native::resident`, so it inherits the
//! `allow(unsafe_code)` of `native.rs`.
//!
//! Safety invariants this file relies on:
//! - The entries are called only through `CudaContext::launch_on`, on the owning thread and under
//!   the dispatch guard. They only enqueue and return the launch status.
//! - Every range is a `RawRange`, proved inside a live allocation by `RawRange::new`. Sizes,
//!   alignment and disjointness (`super::Span`) are proved again here on the extents the kernels
//!   touch; the C entries repeat the checks on the addresses.
//! - `pos` and `status` may be absent; an absent one is passed as null.
//! - A device allocation outlives the queued work because `inference_cuda_free` synchronizes the
//!   device before `cudaFree`.
//! - The argument structs are live locals that the entries read only during the call. Their
//!   `repr(C)` mirrors are private here and pinned by `const` asserts to the field order of
//!   `qsa_index.h`. No raw pointer and no unsafe function leaves this module.

use std::ffi::{c_int, c_void};
use std::mem::{offset_of, size_of};
use std::ptr;

use super::{DeviceFailure, RawRange, RawStream, STAGE_NONE, Span, checked, invalid_argument};

/// Limits of `qsa_index.h`.
pub(in crate::kernels::cuda) const QSA_MAX_M: usize = 32768;
pub(in crate::kernels::cuda) const QSA_MAX_HEADS: usize = 64;
pub(in crate::kernels::cuda) const QSA_MAX_DIM: usize = 1024;
pub(in crate::kernels::cuda) const QSA_MAX_QUERY_FLOATS: usize = 8192;
pub(in crate::kernels::cuda) const QSA_MAX_RATIO: usize = 64;
pub(in crate::kernels::cuda) const QSA_MAX_CAP_BLOCKS: usize = 1 << 24;

const FLOAT: usize = 4;

#[repr(C)]
struct QsaIndexArgsRaw {
    proj: *const c_void,
    q_norm: *const c_void,
    k_norm: *const c_void,
    queries: *mut c_void,
    ring: *mut c_void,
    keys: *mut c_void,
    pos: *const c_void,
    status: *mut c_void,
    m: c_int,
    ld: c_int,
    col_off: c_int,
    heads: c_int,
    dim: c_int,
    rotary_dim: c_int,
    ratio: c_int,
    cap_blocks: c_int,
    pos0: c_int,
    theta: f32,
    eps: f32,
}

const _: () = assert!(
    size_of::<QsaIndexArgsRaw>() == 112
        && offset_of!(QsaIndexArgsRaw, status) == 56
        && offset_of!(QsaIndexArgsRaw, m) == 64
        && offset_of!(QsaIndexArgsRaw, pos0) == 96
        && offset_of!(QsaIndexArgsRaw, eps) == 104
);

#[repr(C)]
struct QsaScoreArgsRaw {
    queries: *const c_void,
    keys: *const c_void,
    scores: *mut c_void,
    pos: *const c_void,
    status: *mut c_void,
    heads: c_int,
    dim: c_int,
    ratio: c_int,
    cap_blocks: c_int,
    pos0: c_int,
}

const _: () = assert!(
    size_of::<QsaScoreArgsRaw>() == 64
        && offset_of!(QsaScoreArgsRaw, status) == 32
        && offset_of!(QsaScoreArgsRaw, heads) == 40
        && offset_of!(QsaScoreArgsRaw, pos0) == 56
);

unsafe extern "C" {
    #[link_name = "inference_cuda_qsa_index_launch"]
    fn qsa_index_raw(
        args: *const QsaIndexArgsRaw,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;

    #[link_name = "inference_cuda_qsa_score_launch"]
    fn qsa_score_raw(
        args: *const QsaScoreArgsRaw,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
}

/// The indexer geometry shared by both entries: `heads` query heads of `dim` values, blocks of
/// `ratio` positions, and a key cache of `cap_blocks` blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::kernels::cuda) struct QsaDims {
    pub(in crate::kernels::cuda) heads: usize,
    pub(in crate::kernels::cuda) dim: usize,
    pub(in crate::kernels::cuda) ratio: usize,
    pub(in crate::kernels::cuda) cap_blocks: usize,
}

impl QsaDims {
    /// True when every dimension is inside the limits of `qsa_index.h`.
    pub(in crate::kernels::cuda) fn inside(self) -> bool {
        (1..=QSA_MAX_HEADS).contains(&self.heads)
            && (32..=QSA_MAX_DIM).contains(&self.dim)
            && self.dim.is_multiple_of(32)
            && self.heads * self.dim <= QSA_MAX_QUERY_FLOATS
            && (1..=QSA_MAX_RATIO).contains(&self.ratio)
            && (1..=QSA_MAX_CAP_BLOCKS).contains(&self.cap_blocks)
    }

    /// The key cache `[cap_blocks][dim]` in bytes; cannot overflow inside the limits.
    pub(in crate::kernels::cuda) fn keys_bytes(self) -> usize {
        self.cap_blocks * self.dim * FLOAT
    }

    fn ints(self) -> Option<[c_int; 4]> {
        Some([
            c_int::try_from(self.heads).ok()?,
            c_int::try_from(self.dim).ok()?,
            c_int::try_from(self.ratio).ok()?,
            c_int::try_from(self.cap_blocks).ok()?,
        ])
    }
}

/// The row layout of the projection: `m` rows of `ld` floats, the indexer columns from
/// `col_off` on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::kernels::cuda) struct QsaRows {
    pub(in crate::kernels::cuda) m: usize,
    pub(in crate::kernels::cuda) ld: usize,
    pub(in crate::kernels::cuda) col_off: usize,
}

/// The bytes each range of an index call touches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::kernels::cuda) struct QsaIndexBytes {
    pub(in crate::kernels::cuda) proj: usize,
    pub(in crate::kernels::cuda) norm: usize,
    pub(in crate::kernels::cuda) queries: usize,
    pub(in crate::kernels::cuda) ring: usize,
    pub(in crate::kernels::cuda) keys: usize,
}

/// The touched bytes of an index call, or `None` when a dimension is outside the limits, the
/// row is too narrow for the indexer columns, or a size overflows.
pub(in crate::kernels::cuda) fn index_need(dims: QsaDims, rows: QsaRows) -> Option<QsaIndexBytes> {
    if !dims.inside() || !(1..=QSA_MAX_M).contains(&rows.m) {
        return None;
    }
    let row = (dims.heads + 1)
        .checked_mul(dims.dim)?
        .checked_add(rows.col_off)?;
    if rows.ld < row || c_int::try_from(rows.ld).is_err() {
        return None;
    }
    let proj_floats = (rows.m - 1).checked_mul(rows.ld)?.checked_add(row)?;
    Some(QsaIndexBytes {
        proj: proj_floats.checked_mul(FLOAT)?,
        norm: dims.dim * FLOAT,
        queries: rows.m * dims.heads * dims.dim * FLOAT,
        ring: dims.ratio * dims.dim * FLOAT,
        keys: dims.keys_bytes(),
    })
}

/// The RoPE and norm constants of the indexer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(in crate::kernels::cuda) struct QsaRope {
    pub(in crate::kernels::cuda) rotary_dim: usize,
    pub(in crate::kernels::cuda) theta: f32,
    pub(in crate::kernels::cuda) eps: f32,
}

impl QsaRope {
    /// The rules of `valid_index` in `qsa_index.cu` for these constants at head size `dim`.
    pub(in crate::kernels::cuda) fn valid_for(self, dim: usize) -> bool {
        (2..=dim).contains(&self.rotary_dim)
            && self.rotary_dim.is_multiple_of(2)
            && self.theta.is_finite()
            && self.theta > 0.0
            && self.eps.is_finite()
            && self.eps >= 0.0
    }
}

pub(in crate::kernels::cuda) struct QsaIndexRaw<'a> {
    pub(in crate::kernels::cuda) dims: QsaDims,
    pub(in crate::kernels::cuda) rows: QsaRows,
    pub(in crate::kernels::cuda) rope: QsaRope,
    pub(in crate::kernels::cuda) pos0: c_int,
    pub(in crate::kernels::cuda) proj: RawRange<'a>,
    pub(in crate::kernels::cuda) q_norm: RawRange<'a>,
    pub(in crate::kernels::cuda) k_norm: RawRange<'a>,
    pub(in crate::kernels::cuda) queries: RawRange<'a>,
    pub(in crate::kernels::cuda) ring: RawRange<'a>,
    pub(in crate::kernels::cuda) keys: RawRange<'a>,
    pub(in crate::kernels::cuda) pos: Option<RawRange<'a>>,
    pub(in crate::kernels::cuda) status: Option<RawRange<'a>>,
}

pub(in crate::kernels::cuda) struct QsaScoreRaw<'a> {
    pub(in crate::kernels::cuda) dims: QsaDims,
    pub(in crate::kernels::cuda) pos0: c_int,
    pub(in crate::kernels::cuda) queries: RawRange<'a>,
    pub(in crate::kernels::cuda) keys: RawRange<'a>,
    pub(in crate::kernels::cuda) scores: RawRange<'a>,
    pub(in crate::kernels::cuda) pos: Option<RawRange<'a>>,
    pub(in crate::kernels::cuda) status: Option<RawRange<'a>>,
}

fn fits(range: &RawRange<'_>, need: usize) -> bool {
    range.bytes() >= need && range.offset().is_multiple_of(FLOAT)
}

fn word_fits(range: &Option<RawRange<'_>>) -> bool {
    range.as_ref().is_none_or(|range| fits(range, FLOAT))
}

fn optional(range: &Option<RawRange<'_>>) -> *mut c_void {
    range
        .as_ref()
        .map_or(ptr::null_mut(), |range| range.address())
}

/// Every written span is apart from the other written spans and from every read span; read
/// spans may share bytes.
fn all_apart(written: &[Span<'_>], read: &[Span<'_>]) -> bool {
    written.iter().enumerate().all(|(index, out)| {
        read.iter().all(|input| out.apart(input))
            && written[index + 1..].iter().all(|other| out.apart(other))
    })
}

fn check_index(call: &QsaIndexRaw<'_>) -> Result<QsaIndexBytes, DeviceFailure> {
    let need = index_need(call.dims, call.rows).ok_or_else(invalid_argument)?;
    let sized = fits(&call.proj, need.proj)
        && fits(&call.q_norm, need.norm)
        && fits(&call.k_norm, need.norm)
        && fits(&call.queries, need.queries)
        && fits(&call.ring, need.ring)
        && fits(&call.keys, need.keys)
        && word_fits(&call.pos)
        && word_fits(&call.status);
    let mut written = vec![
        Span::of(&call.queries, need.queries),
        Span::of(&call.ring, need.ring),
        Span::of(&call.keys, need.keys),
    ];
    written.extend(call.status.as_ref().map(|range| Span::of(range, FLOAT)));
    let mut read = vec![
        Span::of(&call.proj, need.proj),
        Span::of(&call.q_norm, need.norm),
        Span::of(&call.k_norm, need.norm),
    ];
    read.extend(call.pos.as_ref().map(|range| Span::of(range, FLOAT)));
    if !(sized && call.rope.valid_for(call.dims.dim) && all_apart(&written, &read)) {
        return Err(invalid_argument());
    }
    Ok(need)
}

/// Queues the three index kernels on `stream`: queries, the keys of the blocks the chunk
/// completes, then the raw-key ring.
pub(in crate::kernels::cuda) fn qsa_index_launch(
    stream: &RawStream,
    call: QsaIndexRaw<'_>,
) -> Result<(), DeviceFailure> {
    check_index(&call)?;
    let [heads, dim, ratio, cap_blocks] = call.dims.ints().ok_or_else(invalid_argument)?;
    let rows = call.rows;
    let as_int = |v: usize| c_int::try_from(v).map_err(|_| invalid_argument());
    let args = QsaIndexArgsRaw {
        proj: call.proj.address().cast_const(),
        q_norm: call.q_norm.address().cast_const(),
        k_norm: call.k_norm.address().cast_const(),
        queries: call.queries.address(),
        ring: call.ring.address(),
        keys: call.keys.address(),
        pos: optional(&call.pos).cast_const(),
        status: optional(&call.status),
        m: as_int(rows.m)?,
        ld: as_int(rows.ld)?,
        col_off: as_int(rows.col_off)?,
        heads,
        dim,
        rotary_dim: as_int(call.rope.rotary_dim)?,
        ratio,
        cap_blocks,
        pos0: call.pos0,
        theta: call.rope.theta,
        eps: call.rope.eps,
    };
    let mut stage = STAGE_NONE;
    // SAFETY: `check_index` proved every range inside its live allocation (`RawRange::new`),
    // large enough for the call, 4-byte aligned, and every written range (queries, ring, keys,
    // status) apart from the other written ranges and from every read range; the entry checks
    // the addresses again. The allocations outlive the queued work (`inference_cuda_free`
    // synchronizes first). `args` is a live local that the entry reads only during the call,
    // `stream` is live, and `stage` is a writable c_int.
    let status = unsafe { qsa_index_raw(&raw const args, stream.0.as_ptr(), &mut stage) };
    checked(status, stage)
}

/// Queues the score kernel on `stream`: one score per block completed at the position.
pub(in crate::kernels::cuda) fn qsa_score_launch(
    stream: &RawStream,
    call: QsaScoreRaw<'_>,
) -> Result<(), DeviceFailure> {
    let dims = call.dims;
    if !dims.inside() {
        return Err(invalid_argument());
    }
    let (queries, keys, scores) = (
        dims.heads * dims.dim * FLOAT,
        dims.keys_bytes(),
        dims.cap_blocks * FLOAT,
    );
    let mut written = vec![Span::of(&call.scores, scores)];
    written.extend(call.status.as_ref().map(|range| Span::of(range, FLOAT)));
    let mut read = vec![Span::of(&call.queries, queries), Span::of(&call.keys, keys)];
    read.extend(call.pos.as_ref().map(|range| Span::of(range, FLOAT)));
    let ok = fits(&call.queries, queries)
        && fits(&call.keys, keys)
        && fits(&call.scores, scores)
        && word_fits(&call.pos)
        && word_fits(&call.status)
        && all_apart(&written, &read);
    if !ok {
        return Err(invalid_argument());
    }
    let [heads, dim, ratio, cap_blocks] = dims.ints().ok_or_else(invalid_argument)?;
    let args = QsaScoreArgsRaw {
        queries: call.queries.address().cast_const(),
        keys: call.keys.address().cast_const(),
        scores: call.scores.address(),
        pos: optional(&call.pos).cast_const(),
        status: optional(&call.status),
        heads,
        dim,
        ratio,
        cap_blocks,
        pos0: call.pos0,
    };
    let mut stage = STAGE_NONE;
    // SAFETY: the checks above proved the ranges inside their live allocations, large enough,
    // 4-byte aligned, and `scores` and `status`, which the kernel writes, apart from each other
    // and from every read range; the entry checks the addresses again. The allocations outlive
    // the queued work, `args` is a live local read only during the call, `stream` is live and
    // `stage` is writable.
    let status = unsafe { qsa_score_raw(&raw const args, stream.0.as_ptr(), &mut stage) };
    checked(status, stage)
}
