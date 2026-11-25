//! The prefill of the Loop trunk: the decode walk of the parent module over a chunk of
//! rows. Every launch is the rows twin of the decode launch over the same weights and the
//! same shared state (the per-visit GDN states and conv rings, the attention planes, the
//! YaRN table): the dense projections are the vendored MMQ over the quantized f32 rows, the
//! GDN visit is `gdn_conv` then the sequential scan (the kernel's own test equates a chunk
//! of it to that many decode steps), the attention visit is `attn_prep_chunk` then the
//! production prefill attention, and the mHC kernels take the row count.
//!
//! The scratch is the decode scratch sized for [`PREFILL_ROWS`] rows, allocated with the
//! step. A prompt walks every chunk through all the visits in order, so each chunk leaves
//! the shared state where the same tokens decoded one by one would; the head runs on the
//! last row.

use super::*;
use crate::kernels::cuda::prefill::attn_prefill::{
    AttnKvStorage, AttnPrefillCall, AttnPrefillShape, CPF_ATTN_PART_CAP, CPF_ATTN_Q_TILE,
    CpfAttnGeometry, CpfAttnSpan, CpfAttnSplit, cpf_attn_launch_plan,
};
use crate::kernels::cuda::prefill::attn_prep_chunk::{
    AttnPrepChunkCall, AttnPrepChunkShape, PrepChunkMode,
};
use crate::kernels::cuda::prefill::gdn_scan_seq::{
    GDN_SCAN_EPS, GdnScanSeqCall, ScanShape, ScanWeights,
};
use crate::kernels::cuda::prefill::gemm_dq::GemmDqTypedCall;
use crate::kernels::cuda::prefill::mmq_dense::{MmqDenseCall, MmqDenseInputCall};
use crate::kernels::cuda::prefill::mmq_quant::MmqPacking;
use crate::kernels::cuda::prefill::mmq_size::{fixup_floats_max, q8_1_bytes as mmq_q8_1_bytes};
use crate::kernels::cuda::prefill::swiglu_rows::{SwigluRowsCall, SwigluRowsShape};
use crate::kernels::cuda::resident::matvec_q::QuantType;
use crate::kernels::cuda::resident::pack_rows::{PackRowsCall, PackSegment};

/// Rows of one chunk. The mmq chunk cap is 4096; at 1024 the attention partials fit three
/// splits under the 64 MiB part cap, and `row_add` takes a whole chunk's residual add in one
/// launch.
pub(crate) const PREFILL_ROWS: usize = 1024;

/// `row_add`'s float limit per launch.
const ROW_ADD_MAX: usize = 4_194_304;

/// `pack_rows`' row limit per launch.
const PACK_ROWS: usize = 16;

/// `FIFI_SENSENOVA_PREFILL_GEMM=dq` puts every Q4_K/Q6_K projection on the dequantizing GEMM
/// (f32 activations, no q8_1 step): 4.8x slower than MMQ at 1024 rows (305 against 1490
/// tok/s on the 3090), and the accuracy reference until an f16 tensor-core GEMM serves these
/// blocks. Cached: every projection launch would otherwise take an env lock.
fn exact_gemm() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("FIFI_SENSENOVA_PREFILL_GEMM").is_ok_and(|v| v == "dq"))
}

/// The prefill scratch: every decode row buffer at [`PREFILL_ROWS`] rows, the MMQ q8_1
/// columns of the four projection widths, and the attention partials at the part cap.
pub(super) struct Rows {
    resid: [DeviceBuffer; 2],
    cur_ix: usize,
    /// The two mHC streams of every row, adjacent as `[a || b]`.
    streams: DeviceBuffer,
    /// The routed input rows of the current mHC visit.
    routed: DeviceBuffer,
    router_normed: DeviceBuffer,
    router_q8: DeviceBuffer,
    router_logits: DeviceBuffer,
    route_cell: DeviceBuffer,
    norm_out: DeviceBuffer,
    norm_q8: DeviceBuffer,
    /// The mixer output rows: the GDN out projection or the attention o projection.
    mix_out: DeviceBuffer,
    /// The attention context rows of the chunk, before the o projection.
    attn_out: DeviceBuffer,
    proj: DeviceBuffer,
    conv_out: DeviceBuffer,
    delta_out: DeviceBuffer,
    gu: DeviceBuffer,
    /// The activation rows; the decode FFN borrows the first row (the column-split SwiGLU
    /// writes it, the vendored quantize reads it).
    pub(super) act: DeviceBuffer,
    ffn_out: DeviceBuffer,
    q_out: DeviceBuffer,
    q8x: DeviceBuffer,
    q8d: DeviceBuffer,
    q8a: DeviceBuffer,
    q8f: DeviceBuffer,
    fixup: DeviceBuffer,
    part: DeviceBuffer,
    /// The SM count the attention launch plan fills.
    nsm: usize,
}

/// Where one dense projection writes: `m` rows into column `col_off` of `ld`-wide `dst` rows.
struct Out<'b> {
    m: usize,
    dst: DeviceRange<'b>,
    ld: usize,
    col_off: usize,
}

/// The `m` f32 rows of `x`, `stride` floats apart, and the buffer that holds their q8_1
/// columns once quantized.
#[derive(Clone)]
struct Columns<'b> {
    x: DeviceRange<'b>,
    stride: usize,
    m: usize,
    q8: &'b DeviceBuffer,
}

/// The resident q8_1 bytes of `rows` rows of `k` values.
fn q8_rows(k: usize, rows: usize) -> Result<usize, EngError> {
    q8_1_bytes(k, rows).ok_or_else(|| EngError::BadRequest("a q8 rows size overflows".to_owned()))
}

/// Allocates the scratch for a step of `shape`; `nsm` is the device's SM count.
pub(super) fn open_rows(
    ctx: &CudaContext,
    shape: &LoopShape,
    nsm: usize,
) -> Result<Rows, EngError> {
    let (h, ff) = (shape.hidden, shape.feed_forward);
    let value_width = shape.delta_value_heads * shape.delta_value_dim;
    let q_width = shape.heads_q * shape.head_dim;
    let proj_width = (qkvz_row(shape) + 2 * shape.delta_value_heads).max(attn_row(shape));
    if PREFILL_ROWS.checked_mul(h).is_none_or(|n| n > ROW_ADD_MAX) {
        return Err(EngError::BadRequest(
            "the Loop shape's hidden breaks row_add's one-launch limit over a prefill chunk"
                .to_owned(),
        ));
    }
    let zeros = |floats: usize| le_f32(&vec![0.0; floats]);
    let floats = |width: usize| ctx.upload(&zeros(PREFILL_ROWS * width));
    let mmq = |k: usize| -> Result<DeviceBuffer, EngError> {
        ctx.upload(&vec![0u8; mmq_q8_1_bytes(k, PREFILL_ROWS)?])
    };
    Ok(Rows {
        resid: [floats(h)?, floats(h)?],
        cur_ix: 0,
        streams: floats(2 * h)?,
        routed: floats(h)?,
        router_normed: floats(2 * h)?,
        router_q8: ctx.upload(&vec![0u8; q8_rows(2 * h, PREFILL_ROWS)?])?,
        router_logits: floats(8)?,
        route_cell: floats(6)?,
        norm_out: floats(h)?,
        norm_q8: ctx.upload(&vec![0u8; q8_rows(h, PREFILL_ROWS)?])?,
        mix_out: floats(h)?,
        attn_out: floats(q_width)?,
        proj: floats(proj_width)?,
        conv_out: floats(shape.conv_dim())?,
        delta_out: floats(value_width)?,
        gu: floats(2 * ff)?,
        act: floats(ff)?,
        ffn_out: floats(h)?,
        q_out: floats(q_width)?,
        q8x: mmq(h)?,
        q8d: mmq(value_width)?,
        q8a: mmq(q_width)?,
        q8f: mmq(ff)?,
        fixup: floats(fixup_floats_max(nsm)?)?,
        part: ctx.upload(&vec![0u8; CPF_ATTN_PART_CAP])?,
        nsm,
    })
}

impl LoopStep<'_> {
    /// The prompt: `embed_rows` holds `[m][hidden]`, token `t` at position `p0 + t`. Every
    /// chunk walks the 56 visits; the head runs on the last row, so `logits_row` is the next
    /// token's row, as after a decode step.
    pub(crate) fn prefill(&mut self, embed_rows: &[f32], p0: usize) -> Result<(), EngError> {
        let h = self.shape.hidden;
        if embed_rows.is_empty() || !embed_rows.len().is_multiple_of(h) {
            return Err(EngError::BadRequest(
                "the embed rows are empty or not whole hidden-wide rows".to_owned(),
            ));
        }
        let mut m = 0;
        for (chunk, rows) in embed_rows.chunks(PREFILL_ROWS * h).enumerate() {
            m = rows.len() / h;
            ensure_rows(p0 + chunk * PREFILL_ROWS, m, self.cache_rows)?;
            self.walk(&le_f32(rows), m, p0 + chunk * PREFILL_ROWS)?;
        }
        // The output norm and the head, on the last row.
        self.norm_and_q8(
            floats(&self.rows.resid[self.rows.cur_ix], (m - 1) * h, h)?,
            None,
            self.output_norm,
        )?;
        self.ctx.matvec_native(
            &self.stream,
            self.head,
            bytes_range(&self.norm_q8, q8_row(h)?)?,
            floats(&self.logits, 0, self.shape.vocab)?,
            1,
        )
    }

    /// One chunk: `m` rows whose first token sits at `p0`, through every visit in order.
    fn walk(&mut self, embed: &[u8], m: usize, p0: usize) -> Result<(), EngError> {
        let h = self.shape.hidden;
        self.rows.cur_ix = 0;
        // Behind the last chunk's kernels: a blocking copy would land at once, and the
        // chunk before would write its final residual over it.
        self.rows.resid[0].write_at_enqueue(&self.stream, 0, embed)?;
        for il in 0..TRUNK_LAYERS {
            let p = self.plan.physical(il);
            let mhc = self.plan.mhc_active(il);
            if mhc && p == self.plan.mhc_start {
                self.pack_streams(m)?;
            }
            if mhc {
                self.route_chunk(p, "hc_attn_fn", m)?;
                // The branch consumes the routed input, normed with this layer's attn_norm.
                let routed = floats(&self.rows.routed, 0, m * h)?;
                self.norm_chunk(routed, None, self.phys[p].attn_norm, m)?;
            } else {
                let cur = self.resid_rows(m)?;
                self.norm_chunk(cur, None, self.phys[p].attn_norm, m)?;
            }
            match &self.phys[p].mixer {
                Mixer::Delta { .. } => self.gdn_chunk(il, p, m, p0)?,
                Mixer::Attn { .. } => self.attn_chunk(il, p, m, p0)?,
            }
            if mhc {
                self.merge_chunk(m, false)?;
                self.route_chunk(p, "hc_ffn_fn", m)?;
                // The ffn norm reads the routed input the second router wrote.
                let routed = floats(&self.rows.routed, 0, m * h)?;
                self.norm_chunk(routed, None, self.phys[p].post_norm, m)?;
                self.ffn_chunk(p, m)?;
                self.merge_chunk(m, true)?;
                let (a, b) = self.stream_halves(m)?;
                self.ctx.mhc_average(
                    &self.stream,
                    MhcAverageCall {
                        hidden: h,
                        rows: m,
                        ld_ab: 2 * h,
                        a,
                        b,
                        out: self.resid_rows(m)?,
                    },
                )?;
            } else {
                // Fused residual add and post norm: the branch lands in the residual.
                let cur = self.resid_rows(m)?;
                self.norm_chunk(
                    cur,
                    Some(floats(&self.rows.mix_out, 0, m * h)?),
                    self.phys[p].post_norm,
                    m,
                )?;
                self.ffn_chunk(p, m)?;
                // row_add writes a distinct out: rotate the residual pair.
                let next = self.rows.cur_ix ^ 1;
                self.ctx.row_add(
                    &self.stream,
                    RowAddCall {
                        out: floats(&self.rows.resid[next], 0, m * h)?,
                        x: cur,
                        y: floats(&self.rows.ffn_out, 0, m * h)?,
                        n: m * h,
                        flag: None,
                        fused: false,
                    },
                )?;
                self.rows.cur_ix = next;
            }
        }
        Ok(())
    }

    /// The `m` rows of the current residual plane.
    fn resid_rows(&self, m: usize) -> Result<DeviceRange<'_>, EngError> {
        floats(&self.rows.resid[self.rows.cur_ix], 0, m * self.shape.hidden)
    }

    /// The a and b halves of the chunk's `[a || b]` stream rows.
    fn stream_halves(&self, m: usize) -> Result<(DeviceRange<'_>, DeviceRange<'_>), EngError> {
        let h = self.shape.hidden;
        let tail = (m - 1) * 2 * h + h;
        Ok((
            floats(&self.rows.streams, 0, tail)?,
            floats(&self.rows.streams, h, tail)?,
        ))
    }

    /// Both streams of the chunk start as the residual rows.
    fn pack_streams(&self, m: usize) -> Result<(), EngError> {
        let h = self.shape.hidden;
        for at in (0..m).step_by(PACK_ROWS) {
            let rows = (m - at).min(PACK_ROWS);
            let src = floats(&self.rows.resid[self.rows.cur_ix], at * h, rows * h)?;
            self.ctx.pack_rows(
                &self.stream,
                PackRowsCall {
                    m: rows,
                    ld: 2 * h,
                    dst: floats(&self.rows.streams, at * 2 * h, rows * 2 * h)?,
                    segments: vec![
                        PackSegment {
                            src,
                            n: h,
                            dst_off: 0,
                        },
                        PackSegment {
                            src,
                            n: h,
                            dst_off: h,
                        },
                    ],
                },
            )?;
        }
        Ok(())
    }

    /// One router pass over the chunk: the unit-RMS norm of every `[a || b]` row, the
    /// router rows, and the routing kernel per row (the bias folds into the logits there).
    fn route_chunk(&self, p: usize, part: &str, m: usize) -> Result<(), EngError> {
        let h = self.shape.hidden;
        let stem = format!("blk.{}", self.plan.block(p));
        let weight = side(self.weights, &format!("{stem}.{part}.weight"))?;
        let bias = side(self.weights, &format!("{stem}.{part}.bias"))?;
        self.ctx.add_rmsnorm_q8(
            &self.stream,
            AddNormQ8Call {
                n: 2 * h,
                eps: self.shape.eps,
                h: floats(&self.rows.streams, 0, m * 2 * h)?,
                delta: None,
                weight: floats(&self.ones_router, 0, 2 * h)?,
                norm_out: floats(&self.rows.router_normed, 0, m * 2 * h)?,
                q8: bytes_range(&self.rows.router_q8, q8_rows(2 * h, m)?)?,
                rows: m,
            },
        )?;
        self.ctx.router_logits_rows(
            &self.stream,
            RouterLogitsCall {
                shape: RouterShape {
                    m,
                    hidden: 2 * h,
                    n_expert: 8,
                    top_k: 1,
                },
                x: floats(&self.rows.router_normed, 0, m * 2 * h)?,
                w: floats(weight, 0, 8 * 2 * h)?,
                logits: floats(&self.rows.router_logits, 0, m * 8)?,
            },
        )?;
        let (a, b) = self.stream_halves(m)?;
        self.ctx.mhc_route(
            &self.stream,
            MhcRouteCall {
                hidden: h,
                rows: m,
                ld_ab: 2 * h,
                pre_scale: self.plan.mhc_pre_scale,
                iterations: self.plan.mhc_iterations,
                fixed_identity: self.plan.mhc_fixed_identity,
                logits: floats(&self.rows.router_logits, 0, m * 8)?,
                bias: Some(floats(bias, 0, 8)?),
                a,
                b,
                input: floats(&self.rows.routed, 0, m * h)?,
                route: floats(&self.rows.route_cell, 0, m * 6)?,
            },
        )
    }

    /// One merge of both streams with a mixer or FFN output row.
    fn merge_chunk(&self, m: usize, ffn: bool) -> Result<(), EngError> {
        let h = self.shape.hidden;
        let branch = if ffn {
            floats(&self.rows.ffn_out, 0, m * h)?
        } else {
            floats(&self.rows.mix_out, 0, m * h)?
        };
        let (a, b) = self.stream_halves(m)?;
        self.ctx.mhc_merge(
            &self.stream,
            MhcMergeCall {
                hidden: h,
                rows: m,
                ld_ab: 2 * h,
                a,
                b,
                branch,
                route: floats(&self.rows.route_cell, 0, m * 6)?,
            },
        )
    }

    /// The fused add (optional), RMSNorm and q8 quantize of the chunk rows into `norm_out`.
    fn norm_chunk(
        &self,
        row: DeviceRange<'_>,
        delta: Option<DeviceRange<'_>>,
        weight: &DeviceBuffer,
        m: usize,
    ) -> Result<(), EngError> {
        let h = self.shape.hidden;
        self.ctx.add_rmsnorm_q8(
            &self.stream,
            AddNormQ8Call {
                n: h,
                eps: self.shape.eps,
                h: row,
                delta,
                weight: floats(weight, 0, h)?,
                norm_out: floats(&self.rows.norm_out, 0, m * h)?,
                q8: bytes_range(&self.rows.norm_q8, q8_rows(h, m)?)?,
                rows: m,
            },
        )
    }

    /// The `m` f32 rows of `x` (`x_stride` floats apart) quantized for `weight`'s type.
    fn quantize<'b>(
        &self,
        weight: &NativeWeight,
        x: DeviceRange<'b>,
        x_stride: usize,
        m: usize,
        q8: &'b DeviceBuffer,
    ) -> Result<DeviceRange<'b>, EngError> {
        let y = bytes_range(q8, mmq_q8_1_bytes(weight.k(), m)?)?;
        self.ctx.mmq_dense_input(
            &self.stream,
            MmqDenseInputCall {
                packing: MmqPacking::CpuReference,
                weight,
                m,
                row_stride: x_stride,
                x,
                y,
            },
        )?;
        Ok(y)
    }

    /// One dense projection of the quantized rows.
    fn dense(
        &self,
        weight: &NativeWeight,
        q8_1: DeviceRange<'_>,
        out: Out<'_>,
    ) -> Result<(), EngError> {
        self.ctx.mmq_dense(
            &self.stream,
            MmqDenseCall {
                weight,
                w_seg: None,
                m: out.m,
                q8_1,
                dst: out.dst,
                ld: out.ld,
                col_off: out.col_off,
                fixup: bytes_range(&self.rows.fixup, self.rows.fixup.bytes())?,
            },
        )
    }

    /// `weight`'s dense projection of the columns; returns the type they now carry. A Q4_K or
    /// Q6_K weight rides the dequantizing GEMM on the f32 columns themselves: no q8_1 step.
    /// Every other type takes MMQ, the columns quantized into `cols.q8` unless `typed` says
    /// they already carry `weight`'s type. The q8_1 scale layout follows the weight type and a
    /// mismatch gives wrong values, not a fault: Loop's file keeps `v` in Q6_K on the
    /// attention blocks where `q` and `k` are Q4_K, and `ssm_beta_alpha` in Q4_K under a Q6_K
    /// fused projection.
    fn dense_typed(
        &self,
        weight: &NativeWeight,
        typed: Option<QuantType>,
        cols: Columns<'_>,
        out: Out<'_>,
    ) -> Result<QuantType, EngError> {
        let quant = weight.quant();
        if exact_gemm() && matches!(quant, QuantType::Q4K | QuantType::Q6K) {
            return self.gemm_dq(weight, cols, out).map(|()| quant);
        }
        let q8_1 = if typed == Some(quant) {
            bytes_range(cols.q8, mmq_q8_1_bytes(weight.k(), cols.m)?)?
        } else {
            self.quantize(weight, cols.x, cols.stride, cols.m, cols.q8)?
        };
        self.dense(weight, q8_1, out)?;
        Ok(quant)
    }

    /// The dequantizing GEMM: the f32 columns go in as they are, and the f32 output rows
    /// land at `col_off` of the strided destination.
    fn gemm_dq(
        &self,
        weight: &NativeWeight,
        cols: Columns<'_>,
        out: Out<'_>,
    ) -> Result<(), EngError> {
        let overflow = || EngError::MemoryPlan("the dense projection size overflows".to_owned());
        let (k, n) = (weight.k(), weight.n_rows());
        let row_bytes = weight.quant().weight_bytes(1, k).ok_or_else(overflow)?;
        let w = weight
            .buffer()
            .range(0, n.checked_mul(row_bytes).ok_or_else(overflow)?)?;
        let y_span = (out.m - 1)
            .checked_mul(out.ld)
            .and_then(|head| head.checked_add(n))
            .and_then(|elems| elems.checked_mul(4))
            .ok_or_else(overflow)?;
        let y = out
            .dst
            .buffer()
            .range(out.dst.offset() + out.col_off * 4, y_span)?;
        self.ctx.gemm_dq(
            &self.stream,
            weight.quant(),
            GemmDqTypedCall {
                w,
                x: cols.x,
                y,
                k,
                rows: n,
                ncols: out.m,
                x_col_stride: cols.stride,
                y_col_stride: out.ld,
                x_f32: true,
            },
        )
    }

    /// One GDN visit over the chunk: the fused projection, the causal conv, the scan of the
    /// layer's own recurrent state, the out projection.
    fn gdn_chunk(&self, il: usize, p: usize, m: usize, p0: usize) -> Result<(), EngError> {
        let s = &self.shape;
        let Mixer::Delta {
            qkvz,
            beta_alpha,
            out,
            conv,
            a,
            dt_bias,
            norm,
        } = &self.phys[p].mixer
        else {
            unreachable!("the delta view of a recurrent layer")
        };
        let (kh, kd, vh, vd, ck) = (
            s.delta_key_heads,
            s.delta_key_dim,
            s.delta_value_heads,
            s.delta_value_dim,
            s.conv_kernel,
        );
        let (h, conv_dim, qkvz_row_len) = (s.hidden, s.conv_dim(), qkvz_row(s));
        let ld = qkvz_row_len + 2 * vh;
        // One strided row holds [qkvz | beta_alpha]: the scan reads z, beta and alpha at
        // their offsets of the same row.
        let normed = Columns {
            x: floats(&self.rows.norm_out, 0, m * h)?,
            stride: h,
            m,
            q8: &self.rows.q8x,
        };
        let dst = floats(&self.rows.proj, 0, m * ld)?;
        let typed = self.dense_typed(
            qkvz,
            None,
            normed.clone(),
            Out {
                m,
                dst,
                ld,
                col_off: 0,
            },
        )?;
        self.dense_typed(
            beta_alpha,
            Some(typed),
            normed,
            Out {
                m,
                dst,
                ld,
                col_off: qkvz_row_len,
            },
        )?;
        let ring_floats = conv_dim * (ck - 1);
        self.ctx.gdn_conv(
            &self.stream,
            ConvCall {
                shape: ConvShape {
                    m,
                    channels: conv_dim,
                    ld,
                },
                proj: floats(&self.rows.proj, 0, m * ld)?,
                weights: floats(conv, 0, conv_dim * ck)?,
                state: floats(&self.ring, il * ring_floats, ring_floats)?,
                out: floats(&self.rows.conv_out, 0, m * conv_dim)?,
            },
        )?;
        let rec_floats = vh * kd * vd;
        self.ctx.gdn_scan_seq(
            &self.stream,
            GdnScanSeqCall {
                shape: ScanShape {
                    m,
                    p0: p0 as i64,
                    ctx_tokens: self.cache_rows,
                    n_k_heads: kh,
                    n_v_heads: vh,
                    head_dim: vd,
                    conv_ld: conv_dim,
                    ld,
                    z_off: conv_dim,
                    beta_off: qkvz_row_len,
                    alpha_off: qkvz_row_len + vh,
                },
                eps: GDN_SCAN_EPS,
                conv: floats(&self.rows.conv_out, 0, m * conv_dim)?,
                proj: floats(&self.rows.proj, 0, m * ld)?,
                weights: ScanWeights {
                    a: floats(a, 0, vh)?,
                    dt_bias: floats(dt_bias, 0, vh)?,
                    norm_w: floats(norm, 0, vd)?,
                },
                state: floats(&self.rec, il * rec_floats, rec_floats)?,
                o_out: floats(&self.rows.delta_out, 0, m * vh * vd)?,
                status: None,
            },
        )?;
        self.dense_typed(
            out,
            None,
            Columns {
                x: floats(&self.rows.delta_out, 0, m * vh * vd)?,
                stride: vh * vd,
                m,
                q8: &self.rows.q8d,
            },
            Out {
                m,
                dst: floats(&self.rows.mix_out, 0, m * h)?,
                ld: h,
                col_off: 0,
            },
        )?;
        Ok(())
    }

    /// One full-attention visit over the chunk: the projections, the prep (per-head norms,
    /// YaRN rope, cache append), the prefill attention with the in-kernel sigmoid gate, the
    /// o projection.
    fn attn_chunk(&self, il: usize, p: usize, m: usize, p0: usize) -> Result<(), EngError> {
        let s = &self.shape;
        let Mixer::Attn {
            q,
            k,
            v,
            o,
            q_norm,
            k_norm,
        } = &self.phys[p].mixer
        else {
            unreachable!("the attention view of an attention layer")
        };
        let (hq, hkv, hd, h) = (s.heads_q, s.heads_kv, s.head_dim, s.hidden);
        let (q_rows, kv_rows, row_len) = (hq * 2 * hd, hkv * hd, attn_row(s));
        let normed = Columns {
            x: floats(&self.rows.norm_out, 0, m * h)?,
            stride: h,
            m,
            q8: &self.rows.q8x,
        };
        let dst = floats(&self.rows.proj, 0, m * row_len)?;
        let mut typed = None;
        for (weight, col_off) in [(q, 0), (k, q_rows), (v, q_rows + kv_rows)] {
            typed = Some(self.dense_typed(
                weight,
                typed,
                normed.clone(),
                Out {
                    m,
                    dst,
                    ld: row_len,
                    col_off,
                },
            )?);
        }
        let slot = self
            .attn_slot
            .iter()
            .position(|&visit| visit == il)
            .ok_or_else(|| {
                EngError::BadRequest("the visit has no attention cache slot".to_owned())
            })?;
        let (k_cache, v_cache) = (&self.k_cache[slot], &self.v_cache[slot]);
        let (k_cache, v_cache) = (
            bytes_range(k_cache, k_cache.bytes())?,
            bytes_range(v_cache, v_cache.bytes())?,
        );
        self.ctx.attn_prep_chunk(
            &self.stream,
            AttnPrepChunkCall {
                shape: AttnPrepChunkShape {
                    m,
                    n_head: hq,
                    n_kv_head: hkv,
                    head_dim: hd,
                    n_rot: self.rope.dims,
                    rows: self.cache_rows,
                    ld: row_len,
                    p0: p0 as i64,
                },
                rope: RopeParams {
                    eps: s.eps,
                    rope_base: self.rope.freq_base,
                },
                proj: floats(&self.rows.proj, 0, m * row_len)?,
                q_norm: floats(q_norm, 0, hd)?,
                k_norm: floats(k_norm, 0, hd)?,
                q: floats(&self.rows.q_out, 0, m * hq * hd)?,
                mode: PrepChunkMode::AppendFp16 { k_cache, v_cache },
                status: None,
                inv_freq: Some(floats(&self.inv_freq, 0, self.rope.dims / 2)?),
            },
        )?;
        let plan = cpf_attn_launch_plan(
            CpfAttnGeometry {
                m,
                n_head: hq,
                n_kv_head: hkv,
                head_dim: hd,
            },
            p0 + m,
            0,
            self.rows.nsm,
        )?;
        self.ctx.attn_prefill(
            &self.stream,
            AttnPrefillCall {
                storage: AttnKvStorage::F16,
                kv_views: None,
                shape: AttnPrefillShape {
                    m,
                    n_head: hq,
                    n_kv_head: hkv,
                    head_dim: hd,
                    cache_rows: self.cache_rows,
                    ld: row_len,
                    p0: p0 as i64,
                },
                span: CpfAttnSpan::FULL,
                split: CpfAttnSplit {
                    max_splits: plan.max_splits,
                    min_keys: plan.min_keys,
                },
                q_tile: CPF_ATTN_Q_TILE,
                scale: ATTN_SCALE,
                q: floats(&self.rows.q_out, 0, m * hq * hd)?,
                proj: Some(floats(&self.rows.proj, 0, m * row_len)?),
                sinks: None,
                k_cache,
                v_cache,
                part: bytes_range(&self.rows.part, self.rows.part.bytes())?,
                out: floats(&self.rows.attn_out, 0, m * hq * hd)?,
                status: None,
            },
        )?;
        self.dense_typed(
            o,
            None,
            Columns {
                x: floats(&self.rows.attn_out, 0, m * hq * hd)?,
                stride: hq * hd,
                m,
                q8: &self.rows.q8a,
            },
            Out {
                m,
                dst: floats(&self.rows.mix_out, 0, m * h)?,
                ld: h,
                col_off: 0,
            },
        )?;
        Ok(())
    }

    /// The FFN of physical layer `p` over the chunk: the gate and up projections into one
    /// strided row, the SwiGLU, the down projection.
    fn ffn_chunk(&self, p: usize, m: usize) -> Result<(), EngError> {
        let phys = &self.phys[p];
        let (h, ff) = (self.shape.hidden, self.shape.feed_forward);
        let normed = Columns {
            x: floats(&self.rows.norm_out, 0, m * h)?,
            stride: h,
            m,
            q8: &self.rows.q8x,
        };
        let dst = floats(&self.rows.gu, 0, m * 2 * ff)?;
        let typed = self.dense_typed(
            phys.ffn_gate,
            None,
            normed.clone(),
            Out {
                m,
                dst,
                ld: 2 * ff,
                col_off: 0,
            },
        )?;
        self.dense_typed(
            phys.ffn_up,
            Some(typed),
            normed,
            Out {
                m,
                dst,
                ld: 2 * ff,
                col_off: ff,
            },
        )?;
        self.ctx.swiglu_rows(
            &self.stream,
            SwigluRowsCall {
                shape: SwigluRowsShape {
                    rows: m,
                    n_ff: ff,
                    up_off: ff,
                    ld_in: 2 * ff,
                    ld_out: ff,
                },
                gu: floats(&self.rows.gu, 0, m * 2 * ff)?,
                act: floats(&self.rows.act, 0, m * ff)?,
            },
        )?;
        self.dense_typed(
            phys.ffn_down,
            None,
            Columns {
                x: floats(&self.rows.act, 0, m * ff)?,
                stride: ff,
                m,
                q8: &self.rows.q8f,
            },
            Out {
                m,
                dst: floats(&self.rows.ffn_out, 0, m * h)?,
                ld: h,
                col_off: 0,
            },
        )?;
        Ok(())
    }
}
