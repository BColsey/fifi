//! The gated delta-rule mixer of a visit.
//!
//! One fused projection writes the query, key, value and gate inputs of every row, a second the
//! decay and write strengths, both into one strided row. A short causal convolution over the
//! first columns carries its last inputs in a ring, the delta rule updates the layer's record
//! and reads it, and an output projection returns to the hidden width. One row runs the
//! one-step rule; a chunk runs the scan that walks its rows in order.

use super::rows::{Out, Rows, StepIo, q8};
use crate::EngError;
use crate::kernels::cuda::prefill::gdn_scan_seq::{
    GDN_SCAN_EPS, GdnScanSeqCall, ScanShape, ScanWeights,
};
use crate::kernels::cuda::resident::gdn_conv::{ConvCall, ConvShape};
use crate::kernels::cuda::resident::gdn_delta::{DeltaCall, DeltaShape, DeltaWeights, GDN_EPS};
use crate::kernels::cuda::resident::quantize_q8_1::QuantizeCall;
use crate::kernels::cuda::{DeviceBuffer, DeviceRange, NativeWeight};

const F32: usize = size_of::<f32>();

fn floats(buffer: &DeviceBuffer, len: usize) -> Result<DeviceRange<'_>, EngError> {
    buffer.range(0, len * F32)
}

/// The weights and the widths of one delta-rule layer.
#[derive(Clone, Copy)]
pub(crate) struct Delta<'a> {
    pub(crate) qkvz: &'a NativeWeight,
    pub(crate) beta_alpha: &'a NativeWeight,
    pub(crate) out: &'a NativeWeight,
    pub(crate) conv: &'a DeviceBuffer,
    pub(crate) a: &'a DeviceBuffer,
    pub(crate) dt_bias: &'a DeviceBuffer,
    pub(crate) norm: &'a DeviceBuffer,
    pub(crate) key_heads: usize,
    pub(crate) key_dim: usize,
    pub(crate) value_heads: usize,
    pub(crate) value_dim: usize,
    pub(crate) conv_kernel: usize,
    /// Columns of the fused projection the convolution covers.
    pub(crate) conv_dim: usize,
}

impl Delta<'_> {
    /// Floats of the convolution ring of one visit.
    pub(crate) fn ring_floats(&self) -> usize {
        self.conv_dim * (self.conv_kernel - 1)
    }

    /// Floats of the record of one visit.
    pub(crate) fn record_floats(&self) -> usize {
        self.value_heads * self.key_dim * self.value_dim
    }
}

/// The rows a delta-rule visit works in.
#[derive(Clone, Copy)]
pub(crate) struct DeltaRows<'a> {
    /// The normed input rows.
    pub(crate) input: Rows<'a>,
    /// The strided `[fused | strengths]` rows.
    pub(crate) proj: &'a DeviceBuffer,
    pub(crate) conv_out: &'a DeviceBuffer,
    /// What the rule reads from the record, and its q8_1 rows.
    pub(crate) read: Rows<'a>,
    /// The mixer output rows, hidden wide.
    pub(crate) out: &'a DeviceBuffer,
}

/// The carried state of one visit: its convolution ring and its record.
pub(crate) struct DeltaState<'a> {
    pub(crate) ring: DeviceRange<'a>,
    pub(crate) record: DeviceRange<'a>,
    /// Rows of the context the scan may reach.
    pub(crate) context: usize,
}

/// One delta-rule visit over `n` rows whose first row sits at `p0`.
pub(crate) fn mix(
    io: &StepIo<'_>,
    layer: &Delta<'_>,
    rows: &DeltaRows<'_>,
    state: DeltaState<'_>,
    n: usize,
    p0: usize,
) -> Result<(), EngError> {
    let (fused, strengths) = (layer.qkvz.n_rows(), layer.beta_alpha.n_rows());
    let (hidden, ld) = (layer.out.n_rows(), fused + strengths);
    let (vh, vd, conv_dim) = (layer.value_heads, layer.value_dim, layer.conv_dim);
    let read_floats = vh * vd;
    let Some(read_f32) = rows.read.f32 else {
        return Err(EngError::BadRequest(format!(
            "{}: the delta rule needs f32 rows for what it reads",
            io.family
        )));
    };
    let proj = floats(rows.proj, n * ld)?;
    for (weight, col_off) in [(layer.qkvz, 0), (layer.beta_alpha, fused)] {
        let out = Out {
            dst: proj,
            ld,
            col_off,
        };
        io.project_into(weight, rows.input, n, out)?;
    }
    io.ctx.gdn_conv(
        io.stream,
        ConvCall {
            shape: ConvShape {
                m: n,
                channels: conv_dim,
                ld,
            },
            // One row hands over the convolved columns alone.
            proj: floats(rows.proj, if n == 1 { conv_dim } else { n * ld })?,
            weights: floats(layer.conv, conv_dim * layer.conv_kernel)?,
            state: state.ring,
            out: floats(rows.conv_out, n * conv_dim)?,
        },
    )?;
    let (conv, read) = (
        floats(rows.conv_out, n * conv_dim)?,
        floats(read_f32, n * read_floats)?,
    );
    let (a, dt_bias, norm_w) = (
        floats(layer.a, vh)?,
        floats(layer.dt_bias, vh)?,
        floats(layer.norm, vd)?,
    );
    if n == 1 {
        io.ctx.gdn_delta(
            io.stream,
            DeltaCall {
                shape: DeltaShape {
                    m: 1,
                    n_k_heads: layer.key_heads,
                    n_v_heads: vh,
                    conv_ld: conv_dim,
                    ld,
                    z_off: conv_dim,
                    beta_off: fused,
                    alpha_off: fused + vh,
                },
                eps: GDN_EPS,
                conv,
                proj,
                weights: DeltaWeights { a, dt_bias, norm_w },
                state: state.record,
                out: read,
            },
        )?;
    } else {
        io.ctx.gdn_scan_seq(
            io.stream,
            GdnScanSeqCall {
                shape: ScanShape {
                    m: n,
                    p0: p0 as i64,
                    ctx_tokens: state.context,
                    n_k_heads: layer.key_heads,
                    n_v_heads: vh,
                    head_dim: vd,
                    conv_ld: conv_dim,
                    ld,
                    z_off: conv_dim,
                    beta_off: fused,
                    alpha_off: fused + vh,
                },
                eps: GDN_SCAN_EPS,
                conv,
                proj,
                weights: ScanWeights { a, dt_bias, norm_w },
                state: state.record,
                o_out: read,
                status: None,
            },
        )?;
    }
    if io.reads_q8(layer.out, rows.read, n) {
        io.ctx.quantize_q8_1(
            io.stream,
            QuantizeCall {
                x: read,
                y: rows.read.q8.range(0, q8(read_floats, n))?,
                k: read_floats,
                ncols: n,
            },
        )?;
    }
    io.project_into(layer.out, rows.read, n, Out::packed(rows.out, n, hidden)?)
}
