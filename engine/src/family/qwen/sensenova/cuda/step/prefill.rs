//! The walk of the Loop trunk over a chunk of rows, and the mixers of a chunk of several
//! rows. Every launch of those is the rows twin of the one-row launch over the same weights and the
//! same shared state (the per-visit GDN states and conv rings, the attention planes, the
//! YaRN table): the dense projections are the vendored MMQ over the quantized f32 rows, the
//! GDN visit is `gdn_conv` then the sequential scan (the kernel's own test equates a chunk
//! of it to that many decode steps), the attention visit is `attn_prep_chunk` then the
//! production prefill attention, and the mHC kernels take the row count.
//!
//! The scratch holds [`PREFILL_ROWS`] rows, allocated with the step. A prompt walks every chunk through all the visits in order, so each chunk leaves
//! the shared state where the same tokens decoded one by one would; the head runs on the
//! last row.

use std::cell::Cell;

use super::*;
use crate::family::step::rows::{Norm, Out as StepOut, Rows as StepRows, StepIo};
use crate::family::step::walk::{self, Half, Visits};
use crate::family::step::{attend, delta, ffn, site};
use crate::kernels::cuda::prefill::attn_prefill::CPF_ATTN_PART_CAP;
use crate::kernels::cuda::prefill::mmq_size::{fixup_floats_max, q8_1_bytes as mmq_q8_1_bytes};

/// Rows of one chunk. The mmq chunk cap is 4096; at 1024 the attention partials fit three
/// splits under the 64 MiB part cap, and `row_add` takes a whole chunk's residual add in one
/// launch.
pub(crate) const PREFILL_ROWS: usize = 1024;

/// `row_add`'s float limit per launch.
const ROW_ADD_MAX: usize = 4_194_304;

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
    pub(super) resid: [DeviceBuffer; 2],
    pub(super) cur_ix: Cell<usize>,
    /// The two mHC streams of every row, adjacent as `[a || b]`.
    pub(super) streams: DeviceBuffer,
    /// The routed input rows of the current mHC visit.
    pub(super) routed: DeviceBuffer,
    pub(super) router_normed: DeviceBuffer,
    pub(super) router_q8: DeviceBuffer,
    pub(super) router_logits: DeviceBuffer,
    pub(super) route_cell: DeviceBuffer,
    pub(super) norm_out: DeviceBuffer,
    pub(super) norm_q8: DeviceBuffer,
    /// The mixer output rows: the GDN out projection or the attention o projection.
    pub(super) mix_out: DeviceBuffer,
    /// The attention context rows of the chunk, before the o projection.
    attn_out: DeviceBuffer,
    pub(super) proj: DeviceBuffer,
    pub(super) conv_out: DeviceBuffer,
    pub(super) delta_out: DeviceBuffer,
    pub(super) gu: DeviceBuffer,
    /// The activation rows; the decode FFN borrows the first row (the column-split SwiGLU
    /// writes it, the vendored quantize reads it).
    pub(super) act: DeviceBuffer,
    pub(super) ffn_out: DeviceBuffer,
    pub(super) q_out: DeviceBuffer,
    pub(super) q8d: DeviceBuffer,
    pub(super) q8a: DeviceBuffer,
    pub(super) q8f: DeviceBuffer,
    fixup: DeviceBuffer,
    pub(super) part: DeviceBuffer,
    /// The SM count the attention launch plan fills.
    nsm: usize,
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
        cur_ix: Cell::new(0),
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
        // A chunk may regrow the workspace the recorded step holds: the next steps record again.
        self.replay = Replay::default();
        let mut m = 0;
        for (chunk, rows) in embed_rows.chunks(PREFILL_ROWS * h).enumerate() {
            m = rows.len() / h;
            ensure_rows(p0 + chunk * PREFILL_ROWS, m, self.cache_rows)?;
            self.walk(&le_f32(rows), m, p0 + chunk * PREFILL_ROWS, None)?;
        }
        self.head(m - 1, false)
    }

    /// The output norm of residual row `row` and the head, into the logits row.
    pub(super) fn head(&self, row: usize, exact: bool) -> Result<(), EngError> {
        let h = self.shape.hidden;
        let last = floats(&self.rows.resid[self.rows.cur_ix.get()], row * h, h)?;
        self.norm_site(last, None, self.output_norm, 1)?;
        let x = StepRows::both(&self.rows.norm_q8, &self.rows.norm_out);
        self.io(exact).project(self.head, x, &self.logits, 1)
    }

    /// Row 0 of `buffer` joins the trace, once the stream drained.
    fn snap(
        &self,
        trace: &mut Option<&mut Vec<Vec<f32>>>,
        buffer: &DeviceBuffer,
    ) -> Result<(), EngError> {
        let Some(trace) = trace.as_deref_mut() else {
            return Ok(());
        };
        self.stream.synchronize()?;
        let mut bytes = vec![0u8; self.shape.hidden * F32];
        self.ctx.download(buffer, &mut bytes)?;
        let row = bytes.as_chunks::<4>().0.iter();
        trace.push(row.map(|c| f32::from_le_bytes(*c)).collect());
        Ok(())
    }

    /// One chunk: `m` rows whose first token sits at `p0`, through every visit in order. One
    /// row is a decode step: its mixers are the one-row kernels, which read the position from
    /// the step parameters. `trace` takes row 0 of every visit's mixer output and residual.
    pub(super) fn walk(
        &mut self,
        embed: &[u8],
        m: usize,
        p0: usize,
        mut trace: Option<&mut Vec<Vec<f32>>>,
    ) -> Result<(), EngError> {
        self.feed(embed, m, p0)?;
        self.visits(m, p0, trace.take())
    }

    /// The uploads of one chunk: its embedding rows into the residual and, for one row, the
    /// position into the step parameters.
    pub(super) fn feed(&self, embed: &[u8], m: usize, p0: usize) -> Result<(), EngError> {
        // Behind the last chunk's kernels: a blocking copy would land at once, and the
        // chunk before would write its final residual over it.
        self.rows.resid[0].write_at_enqueue(&self.stream, 0, embed)?;
        if m == 1 {
            let params = StepParams::new(0, p0 as i32)?;
            self.params
                .write_at_enqueue(&self.stream, 0, &params.to_bytes())?;
        }
        Ok(())
    }

    /// The visits of one fed chunk, in order. It queues launches only unless `trace` is set,
    /// and for one row the launches do not depend on `p0`. `trace` takes row 0 of every
    /// visit's mixer output and residual.
    pub(super) fn visits(
        &self,
        m: usize,
        p0: usize,
        mut trace: Option<&mut Vec<Vec<f32>>>,
    ) -> Result<(), EngError> {
        walk::visits(self, m, p0, |_, half| {
            let rows = match half {
                Half::Mixer => &self.rows.mix_out,
                Half::Forward => &self.rows.resid[self.rows.cur_ix.get()],
            };
            self.snap(&mut trace, rows)
        })
    }

    /// The `m` rows of the current residual plane.
    fn resid_rows(&self, m: usize) -> Result<DeviceRange<'_>, EngError> {
        floats(
            &self.rows.resid[self.rows.cur_ix.get()],
            0,
            m * self.shape.hidden,
        )
    }

    /// The routed site over the chunk's rows.
    pub(super) fn routed(&self) -> site::Routed<'_> {
        let rows = &self.rows;
        site::Routed {
            hidden: self.shape.hidden,
            eps: self.shape.eps,
            pre_scale: self.plan.mhc_pre_scale,
            iterations: self.plan.mhc_iterations,
            fixed_identity: self.plan.mhc_fixed_identity,
            ones: &self.ones_router,
            streams: &rows.streams,
            normed: &rows.router_normed,
            normed_q8: &rows.router_q8,
            logits: &rows.router_logits,
            input: &rows.routed,
            cell: &rows.route_cell,
        }
    }

    /// One router pass of physical layer `p` over `m` rows; `part` names the router.
    pub(super) fn route(&self, p: usize, part: &str, m: usize) -> Result<(), EngError> {
        let stem = format!("blk.{}", self.plan.block(p));
        let weight = side(self.weights, &format!("{stem}.{part}.weight"))?;
        let bias = side(self.weights, &format!("{stem}.{part}.bias"))?;
        self.routed().route(&self.io(false), weight, bias, m)
    }

    /// The fused add (optional), RMS norm and q8 quantize of `m` rows into the norm rows.
    pub(super) fn norm_site(
        &self,
        x: DeviceRange<'_>,
        delta: Option<DeviceRange<'_>>,
        weight: &DeviceBuffer,
        m: usize,
    ) -> Result<(), EngError> {
        let h = self.shape.hidden;
        let (out, q8) = (&self.rows.norm_out, &self.rows.norm_q8);
        let norm = Norm {
            d: h,
            eps: self.shape.eps,
            x,
            delta,
            weight: floats(weight, 0, h)?,
        };
        self.io(false).norm(
            m,
            norm,
            floats(out, 0, m * h)?,
            bytes_range(q8, q8_rows(h, m)?)?,
        )
    }

    /// The mixer of visit `il` over `m` rows whose first row sits at `p0`: the delta rule or
    /// the attention of physical layer `p`, reading the normed rows.
    fn mixer(&self, il: usize, p: usize, m: usize, p0: usize) -> Result<(), EngError> {
        let s = &self.shape;
        let io = self.io(self.exact(m));
        let rows = &self.rows;
        let input = StepRows::both(&rows.norm_q8, &rows.norm_out);
        match &self.phys[p].mixer {
            Mixer::Delta {
                qkvz,
                beta_alpha,
                out,
                conv,
                a,
                dt_bias,
                norm,
            } => {
                let layer = delta::Delta {
                    qkvz,
                    beta_alpha,
                    out,
                    conv,
                    a,
                    dt_bias,
                    norm,
                    key_heads: s.delta_key_heads,
                    key_dim: s.delta_key_dim,
                    value_heads: s.delta_value_heads,
                    value_dim: s.delta_value_dim,
                    conv_kernel: s.conv_kernel,
                    conv_dim: s.conv_dim(),
                };
                let (ring, record) = (layer.ring_floats(), layer.record_floats());
                delta::mix(
                    &io,
                    &layer,
                    &delta::DeltaRows {
                        input,
                        proj: &rows.proj,
                        conv_out: &rows.conv_out,
                        read: StepRows::both(&rows.q8d, &rows.delta_out),
                        out: &rows.mix_out,
                    },
                    delta::DeltaState {
                        ring: floats(&self.ring, il * ring, ring)?,
                        record: floats(&self.rec, il * record, record)?,
                        context: self.cache_rows,
                    },
                    m,
                    p0,
                )
            }
            Mixer::Attn {
                q,
                k,
                v,
                o,
                q_norm,
                k_norm,
            } => {
                let slot = self
                    .attn_slot
                    .iter()
                    .position(|&visit| visit == il)
                    .ok_or_else(|| {
                        EngError::BadRequest("the visit has no attention cache slot".to_owned())
                    })?;
                let layer = attend::Attend {
                    q,
                    k,
                    v,
                    o,
                    q_norm,
                    k_norm,
                    heads: s.heads_q,
                    kv_heads: s.heads_kv,
                    head_dim: s.head_dim,
                    scale: ATTN_SCALE,
                    eps: s.eps,
                    rot_dims: self.rope.dims,
                    freq_base: self.rope.freq_base,
                    inv_freq: &self.inv_freq,
                    row_splits: MAX_SPLITS,
                };
                let at = if m == 1 {
                    attend::At::Row {
                        pos: self.params_word(OFFSET_POS)?,
                        kv_len: self.params_word(OFFSET_KV_LEN)?,
                        status: self.params_word(OFFSET_STATUS)?,
                    }
                } else {
                    attend::At::Chunk {
                        p0,
                        sm_count: rows.nsm,
                    }
                };
                attend::mix(
                    &io,
                    &layer,
                    &attend::AttendRows {
                        input,
                        proj: &rows.proj,
                        query: &rows.q_out,
                        attended: StepRows::both(&rows.q8a, &rows.attn_out),
                        part: &rows.part,
                        out: &rows.mix_out,
                    },
                    attend::Cache {
                        k: &self.k_cache[slot],
                        v: &self.v_cache[slot],
                        rows: self.cache_rows,
                    },
                    m,
                    at,
                )
            }
        }
    }

    /// The FFN of physical layer `p` over the chunk: the gate and up projections into one
    /// strided row, the SwiGLU, the down projection.
    fn ffn(&self, p: usize, m: usize) -> Result<(), EngError> {
        let phys = &self.phys[p];
        let ff = self.shape.feed_forward;
        let gu = floats(&self.rows.gu, 0, m * 2 * ff)?;
        ffn::gated(
            &self.io(self.exact(m)),
            phys.ffn(),
            ffn::Join::Swiglu,
            StepRows::both(&self.rows.norm_q8, &self.rows.norm_out),
            m,
            ffn::Scratch {
                gate: StepOut {
                    dst: gu,
                    ld: 2 * ff,
                    col_off: 0,
                },
                up: StepOut {
                    dst: gu,
                    ld: 2 * ff,
                    col_off: ff,
                },
                // The matvec route reads these q8_1 rows and the batched route uses the same
                // buffer as its own scratch; a block takes one route or the other.
                act: StepRows::both(&self.rows.q8f, &self.rows.act),
            },
            &self.rows.ffn_out,
        )
    }

    /// Whether the projections of an `m`-row chunk take the decoding routes: each row count
    /// has its own switch.
    fn exact(&self, m: usize) -> bool {
        if m == 1 { self.dq } else { exact_gemm() }
    }

    /// What the shared projections of this step need. `exact` puts them on the decoding
    /// routes.
    pub(super) fn io(&self, exact: bool) -> StepIo<'_> {
        StepIo {
            family: "sensenova",
            ctx: &self.ctx,
            stream: &self.stream,
            weights: self.weights,
            mmq_q8: &self.rows.q8f,
            fixup: &self.rows.fixup,
            exact,
        }
    }
}

impl Visits for LoopStep<'_> {
    fn count(&self) -> usize {
        TRUNK_LAYERS
    }

    fn begin(&self, n: usize) -> Result<(), EngError> {
        self.rows.cur_ix.set(0);
        if n == 1 {
            self.ctx
                .step_begin(&self.stream, bytes_range(&self.params, STEP_PARAMS_BYTES)?)?;
        }
        Ok(())
    }

    /// A routed visit reads the input a router writes from both streams; a plain visit reads
    /// the residual, and its feed-forward read adds the mixer rows in the same launch.
    fn read(&self, visit: usize, half: Half, n: usize) -> Result<(), EngError> {
        let p = self.plan.physical(visit);
        let phys = &self.phys[p];
        let (part, norm) = match half {
            Half::Mixer => ("hc_attn_fn", phys.attn_norm),
            Half::Forward => ("hc_ffn_fn", phys.post_norm),
        };
        let h = self.shape.hidden;
        if !self.plan.mhc_active(visit) {
            let branch = match half {
                Half::Mixer => None,
                Half::Forward => Some(floats(&self.rows.mix_out, 0, n * h)?),
            };
            return self.norm_site(self.resid_rows(n)?, branch, norm, n);
        }
        if half == Half::Mixer && p == self.plan.mhc_start {
            let cur = &self.rows.resid[self.rows.cur_ix.get()];
            self.routed().fork(&self.io(false), cur, n)?;
        }
        self.route(p, part, n)?;
        self.norm_site(floats(&self.rows.routed, 0, n * h)?, None, norm, n)
    }

    fn branch(&self, visit: usize, half: Half, n: usize, p0: usize) -> Result<(), EngError> {
        let p = self.plan.physical(visit);
        match half {
            Half::Mixer => self.mixer(visit, p, n, p0),
            Half::Forward => self.ffn(p, n),
        }
    }

    /// A routed visit merges each branch into both streams and ends on their mean; a plain
    /// visit adds the feed-forward rows into the other residual plane.
    fn apply(&self, visit: usize, half: Half, n: usize) -> Result<(), EngError> {
        let io = self.io(false);
        if self.plan.mhc_active(visit) {
            return match half {
                Half::Mixer => self.routed().merge(&io, &self.rows.mix_out, n),
                Half::Forward => {
                    self.routed().merge(&io, &self.rows.ffn_out, n)?;
                    self.routed().mean(&io, self.resid_rows(n)?, n)
                }
            };
        }
        if half == Half::Mixer {
            return Ok(());
        }
        let len = n * self.shape.hidden;
        // The add writes a distinct out: rotate the residual pair.
        let next = self.rows.cur_ix.get() ^ 1;
        let out = floats(&self.rows.resid[next], 0, len)?;
        let branch = floats(&self.rows.ffn_out, 0, len)?;
        site::add(&io, out, self.resid_rows(n)?, branch, len)?;
        self.rows.cur_ix.set(next);
        Ok(())
    }
}
