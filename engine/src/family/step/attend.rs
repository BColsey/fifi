//! The gated full-attention mixer of a visit.
//!
//! Three projections write the query with its gate, the key and the value of every row into one
//! strided row. The prep norms every head, turns the rotary positions and appends the keys and
//! values to the layer's cache; the attention reads the cache and applies the gate; an output
//! projection returns to the hidden width. One row reads its position from the device words of
//! the step, so its launches are the same for every token; a chunk takes the position of its
//! first row as a launch argument.

use super::rows::{Out, Rows, StepIo, q8};
use crate::EngError;
use crate::kernels::cuda::prefill::attn_prefill::{
    AttnKvStorage, AttnPrefillCall, AttnPrefillShape, CPF_ATTN_Q_TILE, CpfAttnGeometry,
    CpfAttnSpan, CpfAttnSplit, cpf_attn_launch_plan,
};
use crate::kernels::cuda::prefill::attn_prep_chunk::{
    AttnPrepChunkCall, AttnPrepChunkShape, PrepChunkMode,
};
use crate::kernels::cuda::resident::attn_decode::AttnSplit;
use crate::kernels::cuda::resident::attn_decode_loop::{
    AttnDecodeLoopCall, AttnLoopShape, attn_loop_part_bytes,
};
use crate::kernels::cuda::resident::attn_prep::{
    AttnPrepCall, AttnPrepShape, RopeParams, SlotMode,
};
use crate::kernels::cuda::resident::quantize_q8_1::QuantizeCall;
use crate::kernels::cuda::{DeviceBuffer, DeviceRange, NativeWeight};

const F32: usize = size_of::<f32>();

fn floats(buffer: &DeviceBuffer, len: usize) -> Result<DeviceRange<'_>, EngError> {
    buffer.range(0, len * F32)
}

fn whole(buffer: &DeviceBuffer) -> Result<DeviceRange<'_>, EngError> {
    buffer.range(0, buffer.bytes())
}

/// The weights and the widths of one attention layer.
#[derive(Clone, Copy)]
pub(crate) struct Attend<'a> {
    pub(crate) q: &'a NativeWeight,
    pub(crate) k: &'a NativeWeight,
    pub(crate) v: &'a NativeWeight,
    pub(crate) o: &'a NativeWeight,
    pub(crate) q_norm: &'a DeviceBuffer,
    pub(crate) k_norm: &'a DeviceBuffer,
    pub(crate) heads: usize,
    pub(crate) kv_heads: usize,
    pub(crate) head_dim: usize,
    pub(crate) scale: f32,
    pub(crate) eps: f32,
    /// The rotary positions: the turned dimensions, the base and the inverse frequencies.
    pub(crate) rot_dims: usize,
    pub(crate) freq_base: f32,
    pub(crate) inv_freq: &'a DeviceBuffer,
    /// The most key splits of a one-row step.
    pub(crate) row_splits: usize,
}

/// The rows an attention visit works in.
#[derive(Clone, Copy)]
pub(crate) struct AttendRows<'a> {
    /// The normed input rows.
    pub(crate) input: Rows<'a>,
    /// The strided `[query and gate | key | value]` rows.
    pub(crate) proj: &'a DeviceBuffer,
    pub(crate) query: &'a DeviceBuffer,
    /// The attention output rows before the output projection, and their q8_1 rows.
    pub(crate) attended: Rows<'a>,
    pub(crate) part: &'a DeviceBuffer,
    /// The mixer output rows, hidden wide.
    pub(crate) out: &'a DeviceBuffer,
}

/// The key and value planes of one visit, `rows` rows deep.
#[derive(Clone, Copy)]
pub(crate) struct Cache<'a> {
    pub(crate) k: &'a DeviceBuffer,
    pub(crate) v: &'a DeviceBuffer,
    pub(crate) rows: usize,
}

/// Where the rows of a visit sit in the context.
#[derive(Clone, Copy)]
pub(crate) enum At<'a> {
    /// One row: the kernels read its position and the key count from the device words of the
    /// step and report into its status word.
    Row {
        pos: DeviceRange<'a>,
        kv_len: DeviceRange<'a>,
        status: DeviceRange<'a>,
    },
    /// A chunk whose first row sits at `p0`; `sm_count` sizes its key splits.
    Chunk { p0: usize, sm_count: usize },
}

/// One attention visit over `n` rows.
pub(crate) fn mix(
    io: &StepIo<'_>,
    layer: &Attend<'_>,
    rows: &AttendRows<'_>,
    cache: Cache<'_>,
    n: usize,
    at: At<'_>,
) -> Result<(), EngError> {
    let (hq, hkv, hd) = (layer.heads, layer.kv_heads, layer.head_dim);
    let (q_rows, kv_rows) = (layer.q.n_rows(), layer.k.n_rows());
    let (ld, hidden, width) = (q_rows + 2 * kv_rows, layer.o.n_rows(), hq * hd);
    let Some(attended_f32) = rows.attended.f32 else {
        return Err(EngError::BadRequest(format!(
            "{}: the attention needs f32 rows for its output",
            io.family
        )));
    };
    let proj = floats(rows.proj, n * ld)?;
    for (weight, col_off) in [(layer.q, 0), (layer.k, q_rows), (layer.v, q_rows + kv_rows)] {
        let out = Out {
            dst: proj,
            ld,
            col_off,
        };
        io.project_into(weight, rows.input, n, out)?;
    }
    let rope = RopeParams {
        eps: layer.eps,
        rope_base: layer.freq_base,
    };
    let (q_norm, k_norm) = (floats(layer.q_norm, hd)?, floats(layer.k_norm, hd)?);
    let inv_freq = Some(floats(layer.inv_freq, layer.rot_dims / 2)?);
    let (query, attended) = (
        floats(rows.query, n * width)?,
        floats(attended_f32, n * width)?,
    );
    let (k_cache, v_cache) = (whole(cache.k)?, whole(cache.v)?);
    match at {
        At::Row {
            pos,
            kv_len,
            status,
        } => {
            if n != 1 {
                return Err(EngError::BadRequest(format!(
                    "{}: {n} rows cannot read one position word",
                    io.family
                )));
            }
            io.ctx.attn_prep(
                io.stream,
                AttnPrepCall {
                    shape: AttnPrepShape {
                        m: 1,
                        n_head: hq,
                        n_kv_head: hkv,
                        head_dim: hd,
                        n_rot: layer.rot_dims,
                        has_gate: true,
                        slot: SlotMode::Position,
                        rows: cache.rows,
                        ld,
                    },
                    rope,
                    admit_pos: 0,
                    proj,
                    q_norm,
                    k_norm,
                    k_cache,
                    v_cache,
                    q: query,
                    pos,
                    status,
                    inv_freq,
                },
            )?;
            io.ctx.attn_decode_loop(
                io.stream,
                AttnDecodeLoopCall {
                    shape: AttnLoopShape {
                        m: 1,
                        n_head: hq,
                        n_kv_head: hkv,
                        cache_rows: cache.rows,
                        ld,
                    },
                    split: AttnSplit::STAGE1,
                    scale: layer.scale,
                    q: query,
                    proj: Some(proj),
                    k_cache,
                    v_cache,
                    kv_len,
                    part: rows
                        .part
                        .range(0, attn_loop_part_bytes(1, hq / hkv, layer.row_splits)?)?,
                    out: attended,
                },
            )?;
        }
        At::Chunk { p0, sm_count } => {
            io.ctx.attn_prep_chunk(
                io.stream,
                AttnPrepChunkCall {
                    shape: AttnPrepChunkShape {
                        m: n,
                        n_head: hq,
                        n_kv_head: hkv,
                        head_dim: hd,
                        n_rot: layer.rot_dims,
                        rows: cache.rows,
                        ld,
                        p0: p0 as i64,
                    },
                    rope,
                    proj,
                    q_norm,
                    k_norm,
                    q: query,
                    mode: PrepChunkMode::AppendFp16 { k_cache, v_cache },
                    status: None,
                    inv_freq,
                },
            )?;
            let plan = cpf_attn_launch_plan(
                CpfAttnGeometry {
                    m: n,
                    n_head: hq,
                    n_kv_head: hkv,
                    head_dim: hd,
                },
                p0 + n,
                0,
                sm_count,
            )?;
            io.ctx.attn_prefill(
                io.stream,
                AttnPrefillCall {
                    storage: AttnKvStorage::F16,
                    kv_views: None,
                    shape: AttnPrefillShape {
                        m: n,
                        n_head: hq,
                        n_kv_head: hkv,
                        head_dim: hd,
                        cache_rows: cache.rows,
                        ld,
                        p0: p0 as i64,
                    },
                    span: CpfAttnSpan::FULL,
                    split: CpfAttnSplit {
                        max_splits: plan.max_splits,
                        min_keys: plan.min_keys,
                    },
                    q_tile: CPF_ATTN_Q_TILE,
                    scale: layer.scale,
                    q: query,
                    proj: Some(proj),
                    sinks: None,
                    k_cache,
                    v_cache,
                    part: whole(rows.part)?,
                    out: attended,
                    status: None,
                },
            )?;
        }
    }
    if io.reads_q8(layer.o, rows.attended, n) {
        io.ctx.quantize_q8_1(
            io.stream,
            QuantizeCall {
                x: attended,
                y: rows.attended.q8.range(0, q8(width, n))?,
                k: width,
                ncols: n,
            },
        )?;
    }
    io.project_into(layer.o, rows.attended, n, Out::packed(rows.out, n, hidden)?)
}
