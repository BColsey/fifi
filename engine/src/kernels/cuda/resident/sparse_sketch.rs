//! Safe launch wrapper of the sparse-attention sketch append: projects the post-norm, post-rope
//! k rows of one step onto the fixed per-KV-head orthonormal basis and appends the fp16
//! projections to the sketch plane of the query-aware sparse KV path
//! (`sketch [n_kv_head][cells][rank]`, `cell = position`, value
//! `RNE_f16(sum basis[h][r][d] * k[t][h][d])` accumulated by a fixed f32 shuffle tree;
//! `basis` is f32 `[n_kv_head][rank][head_dim]`, output-major).
//!
//! [`CudaContext::sparse_sketch`] queues one kernel on a [`DeviceStream`] and returns;
//! [`CudaContext::sparse_sketch_chunk`] is the prefill entry with a host base position. Both
//! allocate nothing, copy nothing and never synchronize; a fault in the queued work surfaces at
//! the next synchronize of the stream and latches the backend there.
//!
//! The decode entry's `pos` word (`StepParams.pos`) is authoritative: token `t` goes to cell
//! `*pos + t`. A word outside `0..=cells - rows` is refused on the device exactly as `attn_prep`
//! and `kv8_append` refuse it: nothing is written and mask 4 is OR-ed into the status word. The
//! chunk entry's base position is a host value, checked here before any native call.
//!
//! Every argument is checked here, before any native call: the rules mirror the `valid_*`
//! functions of `sparse_sketch.cu`, so the native entries cannot return `INVALID_ARGUMENT`
//! (which would latch the backend). Local rules carry the tag `sparse_sketch: <rule>` or
//! `sparse_sketch_chunk: <rule>`; size, alignment and overlap are refused by `require_fit` and
//! `require_apart` with their own texts.
#![cfg_attr(
    not(test),
    expect(dead_code, reason = "staged for the resident decode step")
)]

use std::ffi::c_int;

use crate::EngError;
use crate::kernels::cuda::context::{
    Claim, CudaContext, DeviceRange, DeviceStream, require_apart, require_fit,
};
use crate::kernels::cuda::native::resident::sparse_sketch::{
    SparseSketchDataRaw, SparseSketchInts, SparseSketchWordsRaw, sparse_sketch_chunk_launch,
    sparse_sketch_launch,
};

/// The least head size (one warp holds the row).
pub(crate) const MIN_HEAD_DIM: usize = 32;
/// The most values per head (the register budget of the kernel: 32 lanes of 32).
pub(crate) const MAX_HEAD_DIM: usize = 1024;
pub(crate) const MAX_N_KV_HEAD: usize = 64;
/// The least projection outputs (a degenerate rank-1 sketch is refused).
pub(crate) const MIN_RANK: usize = 2;
pub(crate) const MAX_DECODE_ROWS: usize = 16;
/// The most rows of the chunk entry (the grid's y extent).
pub(crate) const MAX_CHUNK_ROWS: usize = 65_535;
pub(crate) const MAX_CELLS: usize = (1 << 30) - 1;

/// Geometry of one sketch append: `rows` tokens, `n_kv_head` kv heads, `head_dim` values per
/// head (a multiple of 32), `rank` projection outputs per head, and `cells` plane cells per
/// head (the KV capacity).
#[derive(Debug, Clone, Copy)]
pub(crate) struct SparseSketchShape {
    pub(crate) rows: usize,
    pub(crate) n_kv_head: usize,
    pub(crate) head_dim: usize,
    pub(crate) rank: usize,
    pub(crate) cells: usize,
}

impl SparseSketchShape {
    pub(crate) fn row_bytes(&self) -> usize {
        self.rows
            .saturating_mul(self.n_kv_head)
            .saturating_mul(self.head_dim)
            .saturating_mul(4)
    }

    pub(crate) fn basis_bytes(&self) -> usize {
        self.n_kv_head
            .saturating_mul(self.rank)
            .saturating_mul(self.head_dim)
            .saturating_mul(4)
    }

    pub(crate) fn sketch_bytes(&self) -> usize {
        self.n_kv_head
            .saturating_mul(self.cells)
            .saturating_mul(self.rank)
            .saturating_mul(2)
    }
}

/// The three data ranges of every sketch append. All are device ranges of this context.
///
/// - `k`: f32 `[rows][n_kv_head][head_dim]`, 4-byte aligned (k after the norm and the rope).
/// - `basis`: f32 `[n_kv_head][rank][head_dim]`, 4-byte aligned, output-major.
/// - `sketch`: fp16 `[n_kv_head][cells][rank]`, 4-byte aligned, `cell = position`.
pub(crate) struct SparseSketchData<'b> {
    pub(crate) k: DeviceRange<'b>,
    pub(crate) basis: DeviceRange<'b>,
    pub(crate) sketch: DeviceRange<'b>,
}

fn rule(tag: &str, detail: &str) -> EngError {
    EngError::BadRequest(format!("{tag}: {detail}"))
}

fn to_int(tag: &str, value: usize) -> Result<c_int, EngError> {
    c_int::try_from(value).map_err(|_| rule(tag, &format!("{tag} is {value}, above i32::MAX")))
}

/// Checks the geometry rules and converts the integers once, under `tag`, so that no conversion
/// can fail inside the launch closure.
fn check_shape(
    tag: &str,
    shape: &SparseSketchShape,
    max_rows: usize,
) -> Result<SparseSketchInts, EngError> {
    if !(1..=max_rows).contains(&shape.rows) {
        return Err(rule(
            tag,
            &format!("rows is {}, not 1 to {max_rows}", shape.rows),
        ));
    }
    if !(1..=MAX_N_KV_HEAD).contains(&shape.n_kv_head) {
        return Err(rule(
            tag,
            &format!("n_kv_head is {}, not 1 to {MAX_N_KV_HEAD}", shape.n_kv_head),
        ));
    }
    if !shape.head_dim.is_multiple_of(MIN_HEAD_DIM) {
        return Err(rule(
            tag,
            &format!(
                "head_dim is {}, not a multiple of {MIN_HEAD_DIM}",
                shape.head_dim
            ),
        ));
    }
    if !(MIN_HEAD_DIM..=MAX_HEAD_DIM).contains(&shape.head_dim) {
        return Err(rule(
            tag,
            &format!(
                "head_dim is {}, not {MIN_HEAD_DIM} to {MAX_HEAD_DIM}",
                shape.head_dim
            ),
        ));
    }
    if !(MIN_RANK..=shape.head_dim).contains(&shape.rank) {
        return Err(rule(
            tag,
            &format!(
                "rank is {}, not {MIN_RANK} to head_dim {}",
                shape.rank, shape.head_dim
            ),
        ));
    }
    if shape.cells < shape.rows || shape.cells > MAX_CELLS {
        return Err(rule(
            tag,
            &format!(
                "cells is {}, not {rows} to {MAX_CELLS}",
                shape.cells,
                rows = shape.rows
            ),
        ));
    }
    Ok(SparseSketchInts {
        rows: to_int(tag, shape.rows)?,
        n_kv_head: to_int(tag, shape.n_kv_head)?,
        head_dim: to_int(tag, shape.head_dim)?,
        rank: to_int(tag, shape.rank)?,
        cells: to_int(tag, shape.cells)?,
    })
}

/// Only valid after [`check_shape`], so the sizes cannot overflow.
fn check_fit(data: &SparseSketchData<'_>, shape: &SparseSketchShape) -> Result<(), EngError> {
    require_fit("k", &data.k, shape.row_bytes(), 4)?;
    require_fit("basis", &data.basis, shape.basis_bytes(), 4)?;
    require_fit("sketch", &data.sketch, shape.sketch_bytes(), 4)
}

/// Disjointness of the touched extents: the written sketch plane against the two read ranges;
/// the read ranges may overlap each other.
fn check_apart(data: &SparseSketchData<'_>, shape: &SparseSketchShape) -> Result<(), EngError> {
    let sketch = Claim::of("sketch", &data.sketch, shape.sketch_bytes());
    for read in [
        Claim::of("k", &data.k, shape.row_bytes()),
        Claim::of("basis", &data.basis, shape.basis_bytes()),
    ] {
        require_apart(&sketch, &read)?;
    }
    Ok(())
}

/// Only valid after [`check_shape`], [`check_fit`] and [`check_apart`].
fn raw_data<'a>(
    context: &CudaContext,
    data: &SparseSketchData<'a>,
    ints: SparseSketchInts,
) -> Result<SparseSketchDataRaw<'a>, EngError> {
    Ok(SparseSketchDataRaw {
        k: context.raw_range(&data.k)?,
        basis: context.raw_range(&data.basis)?,
        sketch: context.raw_range(&data.sketch)?,
        ints,
    })
}

impl CudaContext {
    /// Queues the sparse-attention sketch decode append of `data` on `stream`: `shape.rows`
    /// tokens (1 to 16) of k, projected onto the basis and stored at cells `*pos .. *pos + rows`
    /// of the sketch plane. Returns when the kernel is queued; the plane is valid after the
    /// stream drained.
    ///
    /// The device word `*pos` is authoritative. A word outside `0..=cells - rows` makes the
    /// launch write nothing and OR mask 4 into the status word (`StepParams.status`).
    ///
    /// Returns `BadRequest`, before any native call and with the backend unchanged, for a buffer
    /// or stream of another context, a freed buffer, a shape the kernel refuses, a range that is
    /// too short or misaligned, and a written range that overlaps another range of the call.
    pub(crate) fn sparse_sketch(
        &self,
        stream: &DeviceStream,
        shape: SparseSketchShape,
        data: SparseSketchData<'_>,
        pos: DeviceRange<'_>,
        status: DeviceRange<'_>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let ints = check_shape("sparse_sketch", &shape, MAX_DECODE_ROWS)?;
        check_fit(&data, &shape)?;
        check_apart(&data, &shape)?;
        require_fit("pos", &pos, 4, 4)?;
        require_fit("status", &status, 4, 4)?;
        let raw = raw_data(self, &data, ints)?;
        let words = SparseSketchWordsRaw {
            pos: self.raw_range(&pos)?,
            status: self.raw_range(&status)?,
        };
        // `status` is written (mask 4 on a refused launch), so it must be apart from every other
        // range of the call; `pos` is read only.
        let status_claim = Claim::of("status", &status, 4);
        for other in [
            Claim::of("k", &data.k, shape.row_bytes()),
            Claim::of("basis", &data.basis, shape.basis_bytes()),
            Claim::of("sketch", &data.sketch, shape.sketch_bytes()),
            Claim::of("pos", &pos, 4),
        ] {
            require_apart(&status_claim, &other)?;
        }
        self.launch_on(stream, move |raw_stream| {
            sparse_sketch_launch(raw_stream, raw, words)
        })
    }

    /// Queues the sparse-attention sketch chunk append of `data` on `stream`: `shape.rows`
    /// prefill tokens (1 to 65,535) of k, projected onto the basis and stored at cells
    /// `pos0 .. pos0 + rows` of the sketch plane. Returns when the kernel is queued.
    ///
    /// Returns `BadRequest`, before any native call and with the backend unchanged, for a
    /// `pos0` that does not leave `rows` cells, and every refusal of [`Self::sparse_sketch`]
    /// that does not concern the device words.
    pub(crate) fn sparse_sketch_chunk(
        &self,
        stream: &DeviceStream,
        shape: SparseSketchShape,
        data: SparseSketchData<'_>,
        pos0: usize,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let ints = check_shape("sparse_sketch_chunk", &shape, MAX_CHUNK_ROWS)?;
        if pos0
            .checked_add(shape.rows)
            .is_none_or(|end| end > shape.cells)
        {
            return Err(rule(
                "sparse_sketch_chunk",
                &format!(
                    "pos0 {pos0} + rows {} is above cells {}",
                    shape.rows, shape.cells
                ),
            ));
        }
        let pos0 = to_int("sparse_sketch_chunk", pos0)?;
        check_fit(&data, &shape)?;
        check_apart(&data, &shape)?;
        let raw = raw_data(self, &data, ints)?;
        self.launch_on(stream, move |raw_stream| {
            sparse_sketch_chunk_launch(raw_stream, raw, pos0)
        })
    }
}

#[cfg(all(test, feature = "native-cuda"))]
mod tests;
