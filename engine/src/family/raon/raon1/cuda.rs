//! The Raon-1 forward pass on one CUDA device.
//!
//! The stack is composed from the shared GGUF resident kernels (`matvec_q` and `matvec_expert`
//! over the file's own blocks, the F16 `dense_gemv`, `moe_combine`) and the row operations of
//! `kernels::cuda::resident::block_ops` (mixed residual, short convolution, delta-rule scan,
//! latent attention, SiTU, biased router). Every operation works on a chunk of token rows, so
//! prefill and decode share one path: a chunk of `n` tokens runs layer by layer, and decode is
//! the chunk of one. Quantized dense projections take up to 8 activation rows per launch and
//! larger chunks take the batched product; the routed experts run per token up to 8 rows and as
//! grouped products, one per expert stack, above that.
//!
//! The embedding table stays in the file: each token's row is dequantized on the host and
//! uploaded into the four residual streams. The latent keys of the seven attention layers are
//! kept as f16 rows of 576 values; the delta-rule state and the convolution history of the
//! other eighteen layers are f32.

use std::cell::RefCell;
use std::path::Path;

use super::reference::{Hyper, hyper};
use super::{RAON_1_MINI_PREVIEW, Raon1Model, Raon1Spec, RaonKind, open_raon_1_mini_preview};
use crate::EngError;
use crate::family::mem::DEVICE;
use crate::family::project::cuda::{
    BorrowedLoader, CudaWeights, LoadOptions, Reserve, WeightLayout, served_admission,
};
use crate::family::step::rows::{Out, Rows, StepIo, q8};
use crate::family::step::walk::{self, ChunkStep, Half, Visits};
use crate::family::step::{delta, experts, ffn, site};
use crate::kernels::cuda::prefill::mmq_size::q8_1_bytes as mmq_q8_bytes;
use crate::kernels::cuda::resident::block_ops::{BlockOp, MLA_GROUP, MlaShape};
use crate::kernels::cuda::resident::matvec_expert::ExpertMatvecCall;
use crate::kernels::cuda::resident::moe_combine::{CombineCall, CombineOut, CombineShape};
use crate::kernels::cuda::resident::router::{
    RouterLogitsCall, RouterShape, RouterSigmoidBias, RouterSigmoidCall,
};
use crate::kernels::cuda::{CudaContext, DeviceBuffer, DeviceRange, DeviceStream};
use crate::weights::dequantize;

/// Token rows per chunk: the scratch is sized for this many.
pub(crate) const CHUNK: usize = 256;

// The geometry of the checkpoint this device path serves, read from its spec (the opener pins
// the same fields against the GGUF header). The kernels of `block_ops` fix the head width, the
// latent rank and the rope width; `geometry_check` refuses a spec outside them by name.
const SPEC: Raon1Spec = RAON_1_MINI_PREVIEW;
const HIDDEN: usize = SPEC.hidden as usize;
const STREAMS: usize = SPEC.streams as usize;
const HEADS: usize = SPEC.heads as usize;
const HEAD_DIM: usize = SPEC.head_dim as usize;
const WIDTH: usize = HEADS * HEAD_DIM;
const RANK: usize = SPEC.kv_latent as usize;
const ROPE: usize = (SPEC.kv_a_width - SPEC.kv_latent) as usize;
const MLA_ROW: usize = RANK + ROPE;
const Q_NOPE: usize = SPEC.k_head as usize;
const Q_ROW: usize = Q_NOPE + ROPE;
const Q_LORA: usize = SPEC.q_lora as usize;
const EXPERTS: usize = SPEC.experts as usize;
const USED: usize = SPEC.experts_used as usize;
const MOE_INTER: usize = SPEC.moe_inter as usize;
const DENSE_INTER: usize = SPEC.dense_inter as usize;
const TIME_RANK: usize = SPEC.time_step_rank as usize;
const CONV_TAPS: usize = SPEC.conv_kernel as usize - 1;
// The row operations fix these widths: a spec outside them does not build.
const _: () = assert!(
    HEAD_DIM == 128
        && RANK == 512
        && ROPE == 64
        && Q_NOPE == 128
        && STREAMS == 4
        && HEADS.is_multiple_of(8)
);

/// The three convolutions of a delta-rule layer in the order of their columns.
const CONVS: [&str; 3] = ["ssm_conv1d_q", "ssm_conv1d_k", "ssm_conv1d_v"];
/// Bytes of KV cache per position: seven attention layers of 576 f16 values.
const KV_BYTES_PER_POSITION: u64 = SPEC.mla_layers.len() as u64 * MLA_ROW as u64 * 2;
/// Smallest context a start accepts.
const MIN_CONTEXT: usize = 2048;
/// The model's trained context.
const MAX_CONTEXT: usize = 262_144;

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn f32_values(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

fn plan_error(message: String) -> EngError {
    EngError::MemoryPlan(format!("raon1: {message}"))
}

/// The tensor names of one layer, built once.
struct LayerNames {
    kind: RaonKind,
    /// Index among the layers of the same kind.
    ordinal: usize,
    attn_norm: String,
    ffn_norm: String,
    hc_attn: SiteNames,
    hc_ffn: SiteNames,
    attn_output: String,
    attn_q: String,
    attn_k: String,
    attn_v: String,
    ssm_g: String,
    ssm_f_a: String,
    ssm_f_b: String,
    ssm_beta: String,
    ssm_rda: String,
    ssm_dt: String,
    ssm_a: String,
    ssm_norm: String,
    attn_gate: String,
    attn_q_a: String,
    attn_q_a_norm: String,
    attn_q_b: String,
    attn_kv_a: String,
    attn_kv_a_norm: String,
    ffn_gate: String,
    ffn_up: String,
    ffn_down: String,
    gate_inp: String,
    exp_bias: String,
    gate_exps: String,
    up_exps: String,
    down_exps: String,
    gate_shexp: String,
    up_shexp: String,
    down_shexp: String,
    moe: bool,
    layer: usize,
}

struct SiteNames {
    mix: String,
    base: String,
    scale: String,
}

impl LayerNames {
    fn new(spec: Raon1Spec, layer: usize) -> Self {
        let kind = if spec.mla_layers.contains(&layer) {
            RaonKind::Mla
        } else {
            RaonKind::Rda
        };
        let ordinal = (0..layer)
            .filter(|&l| spec.mla_layers.contains(&l) == (kind == RaonKind::Mla))
            .count();
        let n = |suffix: &str| format!("blk.{layer}.{suffix}");
        let site = |name: &str| SiteNames {
            mix: n(&format!("hc_{name}_fn.weight")),
            base: n(&format!("hc_{name}_base.weight")),
            scale: n(&format!("hc_{name}_scale.weight")),
        };
        Self {
            kind,
            ordinal,
            attn_norm: n("attn_norm.weight"),
            ffn_norm: n("ffn_norm.weight"),
            hc_attn: site("attn"),
            hc_ffn: site("ffn"),
            attn_output: n("attn_output.weight"),
            attn_q: n("attn_q.weight"),
            attn_k: n("attn_k.weight"),
            attn_v: n("attn_v.weight"),
            ssm_g: n("ssm_g.weight"),
            ssm_f_a: n("ssm_f_a.weight"),
            ssm_f_b: n("ssm_f_b.weight"),
            ssm_beta: n("ssm_beta.weight"),
            ssm_rda: n("ssm_rda.weight"),
            ssm_dt: n("ssm_dt.bias"),
            ssm_a: n("ssm_a"),
            ssm_norm: n("ssm_norm.weight"),
            attn_gate: n("attn_gate.weight"),
            attn_q_a: n("attn_q_a.weight"),
            attn_q_a_norm: n("attn_q_a_norm.weight"),
            attn_q_b: n("attn_q_b.weight"),
            attn_kv_a: n("attn_kv_a_mqa.weight"),
            attn_kv_a_norm: n("attn_kv_a_norm.weight"),
            ffn_gate: n("ffn_gate.weight"),
            ffn_up: n("ffn_up.weight"),
            ffn_down: n("ffn_down.weight"),
            gate_inp: n("ffn_gate_inp.weight"),
            exp_bias: n("exp_probs_b.bias"),
            gate_exps: n("ffn_gate_exps.weight"),
            up_exps: n("ffn_up_exps.weight"),
            down_exps: n("ffn_down_exps.weight"),
            gate_shexp: n("ffn_gate_shexp.weight"),
            up_shexp: n("ffn_up_shexp.weight"),
            down_shexp: n("ffn_down_shexp.weight"),
            moe: layer >= spec.leading_dense,
            layer,
        }
    }
}

/// The per-layer f32 factors of the absorbed attention, dequantized once at load.
struct MlaFactors {
    wk_b: DeviceBuffer,
    wv_b: DeviceBuffer,
}

/// The scratch of one chunk, allocated once at load.
struct Scratch {
    resid_a: DeviceBuffer,
    resid_b: DeviceBuffer,
    branch: DeviceBuffer,
    post: DeviceBuffer,
    comb: DeviceBuffer,
    cur: DeviceBuffer,
    q8_cur: DeviceBuffer,
    proj: DeviceBuffer,
    q8_wide: DeviceBuffer,
    proj_qkv: DeviceBuffer,
    scan_qkv: DeviceBuffer,
    scan_decay: DeviceBuffer,
    scan_proj: DeviceBuffer,
    g1: DeviceBuffer,
    g2: DeviceBuffer,
    fa: DeviceBuffer,
    beta: DeviceBuffer,
    delta: DeviceBuffer,
    scan_out: DeviceBuffer,
    qa: DeviceBuffer,
    q8_qa: DeviceBuffer,
    qb: DeviceBuffer,
    kv: DeviceBuffer,
    qeff: DeviceBuffer,
    part: DeviceBuffer,
    mix: DeviceBuffer,
    mla_out: DeviceBuffer,
    ffn_gate: DeviceBuffer,
    ffn_up: DeviceBuffer,
    q8_ffn: DeviceBuffer,
    ids: DeviceBuffer,
    weights: DeviceBuffer,
    gu: DeviceBuffer,
    q8_experts: DeviceBuffer,
    act_experts: DeviceBuffer,
    down: DeviceBuffer,
    shared: DeviceBuffer,
    head: DeviceBuffer,
    q8_head: DeviceBuffer,
    logits: DeviceBuffer,
    mla_gated: DeviceBuffer,
    act_ffn: DeviceBuffer,
    logits_route: DeviceBuffer,
    route_status: DeviceBuffer,
    mmq_q8: DeviceBuffer,
    mmq_q8_routed: DeviceBuffer,
    mmq_q8_act: DeviceBuffer,
    fixup: DeviceBuffer,
    ids_g: DeviceBuffer,
    ids_src1: DeviceBuffer,
    ids_dst: DeviceBuffer,
    expert_bounds: DeviceBuffer,
}

impl Scratch {
    fn ffn_scratch<'a>(
        &'a self,
        n: usize,
        width: usize,
        act: Rows<'a>,
    ) -> Result<ffn::Scratch<'a>, EngError> {
        ffn::Scratch::packed(&self.ffn_gate, &self.ffn_up, (n, width), act)
    }

    fn cur_rows(&self) -> Rows<'_> {
        Rows::both(&self.q8_cur, &self.cur)
    }

    /// The device bytes of the scratch of one chunk.
    fn bytes(vocab: usize, fixup: u64) -> u64 {
        let c = CHUNK;
        let f = |count: usize| (count * 4) as u64;
        let q = |width: usize, rows: usize| q8(width, rows) as u64;
        let part = (c * (HEADS / MLA_GROUP)).max(320) * MLA_GROUP * (RANK + 4);
        f(c * STREAMS * HIDDEN) * 2
            + f(c * HIDDEN) * 4
            + f(c * STREAMS * 17)
            + q(HIDDEN, c)
            + q(WIDTH, c)
            + f(c * WIDTH) * 9
            + f(c * TIME_RANK)
            + f(c * HEADS) * 2
            + f(c * Q_LORA) * 2
            + q(Q_LORA, c)
            + f(c * HEADS * Q_ROW)
            + f(c * MLA_ROW)
            + f(c * HEADS * MLA_ROW)
            + f(part)
            + f(c * HEADS * RANK)
            + f(c * DENSE_INTER) * 2
            + q(DENSE_INTER, c)
            + f(c * USED) * 2
            + f(c * USED * 2 * MOE_INTER)
            + q(MOE_INTER, c * USED)
            + f(c * USED * MOE_INTER)
            + f(c * USED * HIDDEN)
            + f(c * HIDDEN)
            + f(HIDDEN)
            + q(HIDDEN, 1)
            + f(vocab)
            + f(c * WIDTH)
            + f(c * DENSE_INTER)
            + mmq_q8_bytes(DENSE_INTER, c).expect("mmq q8 size") as u64
            + fixup
            + f(c * EXPERTS)
            + mmq_q8_bytes(HIDDEN, c * USED).expect("mmq q8 size") as u64
            + mmq_q8_bytes(MOE_INTER, c * USED).expect("mmq q8 size") as u64
            + f(c * USED) * 4
    }
}

pub(crate) struct Raon1Device {
    ctx: CudaContext,
    stream: DeviceStream,
    weights: CudaWeights,
    model: Raon1Model,
    spec: Raon1Spec,
    hy: Hyper,
    names: Vec<LayerNames>,
    mla: Vec<MlaFactors>,
    /// The taps of the `[q | k | v]` convolution of every delta-rule layer.
    conv: Vec<DeviceBuffer>,
    sc: Scratch,
    /// `[rda layers][heads][head_dim][head_dim]` f32 delta-rule state.
    state: DeviceBuffer,
    /// `[rda layers][taps][3 * width]` f32 convolution history.
    hist: DeviceBuffer,
    /// `[mla layers][cells][576]` f16 latent cache.
    cache: DeviceBuffer,
    cells: usize,
    vocab: usize,
    trace: RefCell<Option<Vec<Stage>>>,
}

/// One named stage of a traced feed.
pub(crate) type Stage = (String, Vec<f32>);

pub(crate) use crate::family::step::walk::HeadRows;

impl Raon1Device {
    /// Opens the GGUF at `path` strictly and loads it onto the device. `context_limit` caps the
    /// positions the cache holds; `None` takes what the free device memory admits, up to the
    /// model's trained context. `announce` receives the planned context before the weights upload.
    pub(crate) fn load(
        path: &Path,
        context_limit: Option<usize>,
        announce: &mut dyn FnMut(usize),
    ) -> Result<Self, EngError> {
        let model = open_raon_1_mini_preview(path, None)?;
        let spec = model.spec();
        let hy = hyper(&model, spec)?;
        if !hy.weights_norm {
            return Err(EngError::Checkpoint(
                "raon1: the shared router normalizes the selected weights; expert_weights_norm is false in this file"
                    .into(),
            ));
        }
        let vocab = spec.vocab as usize;
        let ctx = CudaContext::with_admission(&served_admission())?;
        let stream = ctx.stream()?;
        let names: Vec<String> = spec
            .inventory()
            .into_iter()
            .map(|expected| expected.name)
            .filter(|name| {
                name != "token_embd.weight"
                    && !name.ends_with("attn_k_b.weight")
                    && !name.ends_with("attn_v_b.weight")
                    && !CONVS
                        .iter()
                        .any(|conv| name.ends_with(&format!("{conv}.weight")))
            })
            .collect();
        let name_refs: Vec<&str> = names.iter().map(String::as_str).collect();
        let tensors = crate::family::project::cuda::native_load_bytes(model.weights(), &name_refs)?;
        let mla_factor_bytes = (spec.mla_layers.len() * 2 * HEADS * RANK * HEAD_DIM * 4) as u64;
        let fixup = ctx.mmq_fixup_bytes()? as u64;
        let fixed = tensors
            + mla_factor_bytes
            + Scratch::bytes(vocab, fixup)
            + state_bytes(spec)
            + DEVICE.raon_headroom;
        let free = u64::try_from(ctx.memory_info()?.free).unwrap_or(u64::MAX);
        let room = free.checked_sub(fixed).ok_or_else(|| {
            plan_error(format!(
                "the weights and scratch need {fixed} bytes with the headroom, the device has \
                 {free} free"
            ))
        })?;
        let affordable = usize::try_from(room / KV_BYTES_PER_POSITION)
            .unwrap_or(usize::MAX)
            .min(MAX_CONTEXT)
            / 256
            * 256;
        let cells = match context_limit {
            Some(wanted) if wanted > affordable => {
                return Err(plan_error(format!(
                    "a context of {wanted} tokens needs {} bytes of cache, the device has room \
                     for {affordable} tokens",
                    wanted as u64 * KV_BYTES_PER_POSITION
                )));
            }
            Some(wanted) => wanted,
            None => affordable,
        };
        if cells < MIN_CONTEXT {
            return Err(plan_error(format!(
                "the device has room for {affordable} tokens of context after the weights, below \
                 the minimum of {MIN_CONTEXT}"
            )));
        }
        announce(cells);
        let reserve = Reserve {
            kv_cache: cells as u64 * KV_BYTES_PER_POSITION,
            work: mla_factor_bytes + Scratch::bytes(vocab, fixup) + state_bytes(spec),
            headroom: DEVICE.raon_headroom,
        };
        let loader = BorrowedLoader::new(&ctx)?;
        let weights = loader.load_sliced(
            model.weights(),
            &name_refs,
            LoadOptions {
                reserve,
                drop_page_cache: None,
                layout: WeightLayout::Native,
            },
            &[],
            0,
        )?;
        let layer_names: Vec<LayerNames> =
            (0..spec.layers).map(|l| LayerNames::new(spec, l)).collect();
        let mut mla = Vec::new();
        for &layer in spec.mla_layers {
            let upload = |name: &str| -> Result<DeviceBuffer, EngError> {
                ctx.upload(&f32_bytes(&model.read_f32(&format!("blk.{layer}.{name}"))?))
            };
            mla.push(MlaFactors {
                wk_b: upload("attn_k_b.weight")?,
                wv_b: upload("attn_v_b.weight")?,
            });
        }
        let mut conv = Vec::new();
        for layer in (0..spec.layers).filter(|layer| !spec.mla_layers.contains(layer)) {
            let mut taps = Vec::with_capacity(3 * WIDTH * SPEC.conv_kernel as usize);
            for name in CONVS {
                taps.extend(model.read_f32(&format!("blk.{layer}.{name}.weight"))?);
            }
            conv.push(ctx.upload(&f32_bytes(&taps))?);
        }
        let sc = allocate_scratch(&ctx, vocab)?;
        let rda_layers = spec.layers - spec.mla_layers.len();
        let state = ctx.upload(&vec![0u8; rda_layers * HEADS * HEAD_DIM * HEAD_DIM * 4])?;
        let hist = ctx.upload(&vec![0u8; rda_layers * 3 * CONV_TAPS * WIDTH * 4])?;
        let cache = ctx.upload(&vec![0u8; spec.mla_layers.len() * cells * MLA_ROW * 2])?;
        Ok(Self {
            ctx,
            stream,
            weights,
            model,
            spec,
            hy,
            names: layer_names,
            mla,
            conv,
            sc,
            state,
            hist,
            cache,
            cells,
            vocab,
            trace: RefCell::new(None),
        })
    }

    /// Positions the latent cache holds.
    pub(crate) fn cells(&self) -> usize {
        self.cells
    }

    pub(crate) fn vocab(&self) -> usize {
        self.vocab
    }

    /// Starts a new sequence: clears the delta-rule state and the convolution history. The
    /// latent cache is position-addressed and needs no clearing.
    pub(crate) fn reset(&mut self) -> Result<(), EngError> {
        for buffer in [&self.state, &self.hist] {
            let count = buffer.bytes() / 4;
            self.io()
                .run(BlockOp::fill(count, 0.0, buffer.range(0, buffer.bytes())?)?)?;
        }
        self.stream.synchronize()
    }

    /// Records the named stages of the next feed as f32 vectors (residual, norm and gated mixer
    /// outputs per layer), for comparison with the host reference.
    #[cfg(test)]
    pub(crate) fn trace_next_feed(&self) {
        *self.trace.borrow_mut() = Some(Vec::new());
    }

    #[cfg(test)]
    pub(crate) fn take_trace(&self) -> Vec<Stage> {
        self.trace.borrow_mut().take().unwrap_or_default()
    }

    /// Feeds `tokens` at positions `pos0..`, in chunks. Returns the logits of the rows `head`
    /// names (one row of `vocab` values each).
    pub(crate) fn feed(
        &mut self,
        tokens: &[u32],
        pos0: usize,
        head: HeadRows,
    ) -> Result<Vec<Vec<f32>>, EngError> {
        crate::family::step::walk::feed(self, tokens, pos0, head)
    }

    /// What the projections of this step share.
    fn io(&self) -> StepIo<'_> {
        StepIo {
            family: "raon1",
            ctx: &self.ctx,
            stream: &self.stream,
            weights: &self.weights,
            mmq_q8: &self.sc.mmq_q8,
            fixup: &self.sc.fixup,
            exact: false,
        }
    }

    fn snap(&self, name: &str, buffer: &DeviceBuffer, values: usize) -> Result<(), EngError> {
        if self.trace.borrow().is_none() {
            return Ok(());
        }
        let mut bytes = vec![0u8; values * 4];
        buffer.read_at(&self.stream, 0, &mut bytes)?;
        if let Some(trace) = self.trace.borrow_mut().as_mut() {
            trace.push((name.to_owned(), f32_values(&bytes)));
        }
        Ok(())
    }

    /// One embedding row, dequantized on the host from the type the file stores it in.
    fn embedding_row(&self, token: u32) -> Result<Vec<f32>, EngError> {
        let name = "token_embd.weight";
        let ty = self
            .model
            .header()
            .tensor(name)
            .ok_or_else(|| EngError::Checkpoint(format!("raon1: tensor {name} is missing")))?
            .ggml_type();
        let row_bytes = (HIDDEN as u64 / ty.block_elems() * ty.block_bytes()) as usize;
        let mut bytes = vec![0u8; row_bytes];
        self.model
            .weights()
            .read_range(name, u64::from(token) * row_bytes as u64, &mut bytes)?;
        dequantize(ty, &bytes)
    }

    fn chunk(&mut self, tokens: &[u32], pos0: usize) -> Result<(), EngError> {
        let n = tokens.len();
        let mut resid = vec![0f32; n * STREAMS * HIDDEN];
        for (t, &token) in tokens.iter().enumerate() {
            let row = self.embedding_row(token)?;
            for s in 0..STREAMS {
                let at = (t * STREAMS + s) * HIDDEN;
                resid[at..at + HIDDEN].copy_from_slice(&row);
            }
        }
        self.sc
            .resid_a
            .write_at(&self.stream, 0, &f32_bytes(&resid))?;
        walk::visits(self, n, pos0, |_, _| Ok(()))
    }

    fn site(&self) -> site::Mixed<'_> {
        site::Mixed {
            hidden: HIDDEN,
            eps: self.hy.hc_eps,
            magnitude: self.hy.hc_magnitude,
            input: &self.sc.branch,
            post: &self.sc.post,
            comb: &self.sc.comb,
        }
    }

    /// The mixed-residual read of one site, then the layer norm into `cur` and `q8_cur`.
    fn site_read(
        &self,
        site: &SiteNames,
        norm: &str,
        n: usize,
        resid: &DeviceBuffer,
    ) -> Result<(), EngError> {
        let sc = &self.sc;
        let weights = site::MixWeights {
            mix: self.io().raw(&site.mix)?,
            base: self.io().raw(&site.base)?,
            scale: self.io().raw(&site.scale)?,
        };
        self.site().read(&self.io(), weights, resid, n)?;
        self.io().norm_rows(
            n,
            HIDDEN,
            self.hy.norm_eps,
            &sc.branch,
            self.io().raw(norm)?,
            &sc.cur,
            &sc.q8_cur,
        )
    }

    /// The stream update of one half: `out = comb * resid + post * proj`.
    fn site_apply(
        &self,
        n: usize,
        resid: &DeviceBuffer,
        out: &DeviceBuffer,
    ) -> Result<(), EngError> {
        self.site().apply(&self.io(), resid, out, &self.sc.proj, n)
    }

    fn rda_mixer(&self, names: &LayerNames, n: usize, pos0: usize) -> Result<(), EngError> {
        let (sc, io, rda) = (&self.sc, self.io(), names.ordinal);
        let layer = delta::Delta {
            rule: delta::Rule::Channels(delta::Channels {
                q: io.native(&names.attn_q)?,
                k: io.native(&names.attn_k)?,
                v: io.native(&names.attn_v)?,
                gate: io.native(&names.ssm_g)?,
                decay_a: io.raw(&names.ssm_f_a)?,
                decay_b: io.raw(&names.ssm_f_b)?,
                rank: TIME_RANK,
                beta: io.raw(&names.ssm_beta)?,
                delta: io.raw(&names.ssm_rda)?,
                lower_bound: self.hy.gate_lower_bound,
                norm_eps: self.hy.norm_eps,
                low: &sc.fa,
                decay_in: &sc.g1,
                decay: &sc.scan_decay,
                gate_in: &sc.g2,
                beta_rows: &sc.beta,
                delta_rows: &sc.delta,
                strengths: &sc.scan_proj,
            }),
            out: io.native(&names.attn_output)?,
            conv: StepIo::whole(&self.conv[rda])?,
            a: io.raw(&names.ssm_a)?,
            dt_bias: io.raw(&names.ssm_dt)?,
            norm: io.raw(&names.ssm_norm)?,
            key_heads: HEADS,
            key_dim: HEAD_DIM,
            value_heads: HEADS,
            value_dim: HEAD_DIM,
            conv_kernel: SPEC.conv_kernel as usize,
            conv_dim: 3 * WIDTH,
        };
        let (ring, record) = (layer.ring_floats() * 4, layer.record_floats() * 4);
        delta::mix(
            &io,
            &layer,
            &delta::DeltaRows {
                input: sc.cur_rows(),
                proj: &sc.proj_qkv,
                conv_out: &sc.scan_qkv,
                read: Rows::both(&sc.q8_wide, &sc.scan_out),
                out: &sc.proj,
            },
            delta::DeltaState {
                ring: self.hist.range(rda * ring, ring)?,
                record: self.state.range(rda * record, record)?,
                context: self.cells,
            },
            n,
            pos0,
        )
    }

    fn mla_mixer(&self, names: &LayerNames, n: usize, pos0: usize) -> Result<(), EngError> {
        let sc = &self.sc;
        let mla = names.ordinal;
        self.io().mv(&names.attn_q_a, sc.cur_rows(), &sc.qa, n)?;
        // The f32 rows go to `proj`, which the mixer has not written yet; only the q8_1 rows are read.
        self.io().norm_rows(
            n,
            Q_LORA,
            self.hy.latent_eps,
            &sc.qa,
            self.io().raw(&names.attn_q_a_norm)?,
            &sc.proj,
            &sc.q8_qa,
        )?;
        self.io()
            .mv(&names.attn_q_b, Rows::q8(&sc.q8_qa), &sc.qb, n)?;
        self.io().mv(&names.attn_kv_a, sc.cur_rows(), &sc.kv, n)?;
        let layer_cache = self.cells * MLA_ROW * 2;
        self.io().run(BlockOp::mla_kv_prep(
            n,
            RANK,
            ROPE,
            pos0,
            self.cells,
            self.hy.latent_eps,
            sc.kv.range(0, n * MLA_ROW * 4)?,
            self.io().raw(&names.attn_kv_a_norm)?,
            self.cache.range(mla * layer_cache, layer_cache)?,
        )?)?;
        let shape = MlaShape {
            n,
            heads: HEADS,
            nope: Q_NOPE,
            rope: ROPE,
            rank: RANK,
        };
        let factors = &self.mla[mla];
        self.io().run(BlockOp::mla_q_absorb(
            shape,
            sc.qb.range(0, n * HEADS * Q_ROW * 4)?,
            StepIo::whole(&factors.wk_b)?,
            sc.qeff.range(0, n * HEADS * MLA_ROW * 4)?,
        )?)?;
        let keys = pos0 + n;
        let splits = if n >= 64 {
            1
        } else {
            keys.div_ceil(1024)
                .clamp(1, 32)
                .min(320 / (n * HEADS / MLA_GROUP))
                .max(1)
        };
        let per_split = keys.div_ceil(splits).div_ceil(32) * 32;
        let part_bytes = n * (HEADS / MLA_GROUP) * splits * MLA_GROUP * (RANK + 4) * 4;
        self.io().run(BlockOp::mla_attn(
            shape,
            pos0,
            splits,
            per_split,
            1.0 / (self.spec.key_mla as f32).sqrt(),
            sc.qeff.range(0, n * HEADS * MLA_ROW * 4)?,
            self.cache.range(mla * layer_cache, layer_cache)?,
            sc.part.range(0, part_bytes)?,
        )?)?;
        self.io().run(BlockOp::mla_reduce(
            n,
            HEADS,
            RANK,
            splits,
            sc.part.range(0, part_bytes)?,
            sc.mix.range(0, n * HEADS * RANK * 4)?,
        )?)?;
        self.io().run(BlockOp::mla_project(
            n,
            HEADS,
            RANK,
            HEAD_DIM,
            sc.mix.range(0, n * HEADS * RANK * 4)?,
            StepIo::whole(&factors.wv_b)?,
            sc.mla_out.range(0, n * WIDTH * 4)?,
        )?)?;
        self.io().mv(&names.attn_gate, sc.cur_rows(), &sc.g2, n)?;
        if self.trace.borrow().is_some() {
            self.snap("mla_out", &sc.mla_out, n * WIDTH)?;
            self.snap("mla_gate", &sc.g2, n * WIDTH)?;
        }
        self.io().quantize(
            n,
            WIDTH,
            &sc.mla_out,
            Some(&sc.g2),
            Rows::both(&sc.q8_wide, &sc.mla_gated),
        )?;
        self.io().mv(
            &names.attn_output,
            Rows::both(&sc.q8_wide, &sc.mla_gated),
            &sc.proj,
            n,
        )
    }

    fn dense_block(&self, names: &LayerNames, n: usize) -> Result<(), EngError> {
        let sc = &self.sc;
        let io = self.io();
        ffn::gated(
            &io,
            ffn::Weights::named(&io, &names.ffn_gate, &names.ffn_up, &names.ffn_down)?,
            self.situ_join(),
            sc.cur_rows(),
            n,
            sc.ffn_scratch(n, DENSE_INTER, Rows::both(&sc.q8_ffn, &sc.act_ffn))?,
            &sc.proj,
        )
    }

    fn situ_join(&self) -> ffn::Join {
        ffn::Join::Situ {
            beta: self.hy.situ_beta,
            linear_beta: self.hy.situ_linear_beta,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn expert_mv(
        &self,
        name: &str,
        x_q8: DeviceRange<'_>,
        x_per_slot: bool,
        ids: DeviceRange<'_>,
        y: DeviceRange<'_>,
        stride: usize,
        rows: usize,
        k: usize,
    ) -> Result<(), EngError> {
        let weight = self.io().native(name)?;
        self.ctx.matvec_expert(
            &self.stream,
            ExpertMatvecCall {
                ty: weight.quant(),
                w_stack: StepIo::whole(weight.buffer())?,
                ids,
                x_q8,
                x_per_slot,
                y,
                y_slot_stride: stride,
                k,
                rows,
                n_expert: EXPERTS,
                n_used: USED,
                glu_gate: None,
            },
        )
    }

    fn picks(&self) -> experts::Picks<'_> {
        let sc = &self.sc;
        experts::Picks {
            experts: EXPERTS,
            used: USED,
            ids: &sc.ids,
            ids_g: &sc.ids_g,
            ids_src1: &sc.ids_src1,
            ids_dst: &sc.ids_dst,
            expert_bounds: &sc.expert_bounds,
        }
    }

    fn moe_block(&self, names: &LayerNames, n: usize) -> Result<(), EngError> {
        let sc = &self.sc;
        let shape = RouterShape {
            m: n,
            hidden: HIDDEN,
            n_expert: EXPERTS,
            top_k: USED,
        };
        self.ctx.router_logits_rows(
            &self.stream,
            RouterLogitsCall {
                shape,
                x: sc.cur.range(0, n * HIDDEN * 4)?,
                w: self.io().raw(&names.gate_inp)?,
                logits: sc.logits_route.range(0, n * EXPERTS * 4)?,
            },
        )?;
        self.ctx.router_sigmoid_select_rows(
            &self.stream,
            RouterSigmoidCall {
                shape,
                logits: sc.logits_route.range(0, n * EXPERTS * 4)?,
                ids: sc.ids.range(0, n * USED * 4)?,
                weights: sc.weights.range(0, n * USED * 4)?,
                status: StepIo::whole(&sc.route_status)?,
                bias: Some(RouterSigmoidBias::KernelRounded(
                    self.io().raw(&names.exp_bias)?,
                )),
            },
        )?;
        if self.trace.borrow().is_some() {
            let mut bytes = vec![0u8; n * USED * 4];
            sc.ids.read_at(&self.stream, 0, &mut bytes)?;
            let ids = bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| i32::from_le_bytes(*c) as f32)
                .collect();
            if let Some(trace) = self.trace.borrow_mut().as_mut() {
                trace.push((format!("moe_ids-{}", names.layer), ids));
            }
        }
        // The shared expert first: its output is the base the routed picks are summed onto.
        let io = self.io();
        ffn::gated(
            &io,
            ffn::Weights::named(&io, &names.gate_shexp, &names.up_shexp, &names.down_shexp)?,
            self.situ_join(),
            sc.cur_rows(),
            n,
            sc.ffn_scratch(n, MOE_INTER, Rows::q8(&sc.q8_ffn))?,
            &sc.shared,
        )?;
        let gu_row = USED * 2 * MOE_INTER * 4;
        let stride = 2 * MOE_INTER;
        let grouped = |name: &str| n > 8 && experts::serves(&io, name);
        let gate_up = grouped(&names.gate_exps) && grouped(&names.up_exps);
        let down = grouped(&names.down_exps);
        let ids = if gate_up || down {
            Some(self.picks().ids(&io, n)?)
        } else {
            None
        };
        if let Some(ids) = ids.as_ref().filter(|_| gate_up) {
            let x = experts::token_rows(
                ids,
                sc.cur.range(0, n * HIDDEN * 4)?,
                HIDDEN,
                sc.mmq_q8_routed.range(0, mmq_q8_bytes(HIDDEN, n * USED)?)?,
            );
            for (name, offset) in [(&names.gate_exps, 0), (&names.up_exps, MOE_INTER * 4)] {
                let dst = sc.gu.range(offset, n * gu_row - offset)?;
                experts::grouped(&io, name, ids, &x, stride, dst)?;
            }
        } else {
            let gu_span = ((USED - 1) * stride + MOE_INTER) * 4;
            let x_stride = q8(HIDDEN, 1);
            for t in 0..n {
                let ids = sc.ids.range(t * USED * 4, USED * 4)?;
                let x = sc.q8_cur.range(t * x_stride, x_stride)?;
                let base = t * gu_row;
                self.expert_mv(
                    &names.gate_exps,
                    x,
                    false,
                    ids,
                    sc.gu.range(base, gu_span)?,
                    stride,
                    MOE_INTER,
                    HIDDEN,
                )?;
                self.expert_mv(
                    &names.up_exps,
                    x,
                    false,
                    ids,
                    sc.gu.range(base + MOE_INTER * 4, gu_span)?,
                    stride,
                    MOE_INTER,
                    HIDDEN,
                )?;
            }
        }
        let gu = sc.gu.range(0, n * gu_row)?;
        let (ld, rows) = (stride, n * USED);
        let scratch = ffn::Scratch {
            gate: Out {
                dst: gu,
                ld,
                col_off: 0,
            },
            up: Out {
                dst: gu,
                ld,
                col_off: MOE_INTER,
            },
            act: if down {
                Rows::both(&sc.q8_experts, &sc.act_experts)
            } else {
                Rows::q8(&sc.q8_experts)
            },
        };
        ffn::joined(
            &self.io(),
            self.situ_join(),
            (rows, MOE_INTER),
            scratch,
            true,
        )?;
        let e_stride = q8(MOE_INTER, 1);
        if let Some(ids) = ids.as_ref().filter(|_| down) {
            let x = experts::slot_rows(
                ids,
                sc.act_experts.range(0, n * USED * MOE_INTER * 4)?,
                MOE_INTER,
                sc.mmq_q8_act.range(0, mmq_q8_bytes(MOE_INTER, n * USED)?)?,
            );
            let dst = sc.down.range(0, n * USED * HIDDEN * 4)?;
            experts::grouped(&io, &names.down_exps, ids, &x, HIDDEN, dst)?;
        } else {
            for t in 0..n {
                self.expert_mv(
                    &names.down_exps,
                    sc.q8_experts.range(t * USED * e_stride, USED * e_stride)?,
                    true,
                    sc.ids.range(t * USED * 4, USED * 4)?,
                    sc.down.range(t * USED * HIDDEN * 4, USED * HIDDEN * 4)?,
                    HIDDEN,
                    HIDDEN,
                    MOE_INTER,
                )?;
            }
        }
        self.ctx.moe_combine(
            &self.stream,
            CombineCall {
                shape: CombineShape {
                    m: n,
                    n: HIDDEN,
                    top_k: USED,
                    ld_down: HIDDEN,
                },
                res: sc.shared.range(0, n * HIDDEN * 4)?,
                down: sc.down.range(0, n * USED * HIDDEN * 4)?,
                weights: sc.weights.range(0, n * USED * 4)?,
                shared: None,
                out: CombineOut::To(sc.proj.range(0, n * HIDDEN * 4)?),
            },
        )
    }

    /// The output head of row `row` of the last chunk: stream mean, norm, vocabulary matvec.
    fn logits_row(&self, row: usize) -> Result<Vec<f32>, EngError> {
        let sc = &self.sc;
        self.io().run(BlockOp::hc_mean(
            row,
            HIDDEN,
            sc.resid_a.range(0, (row + 1) * STREAMS * HIDDEN * 4)?,
            sc.head.range(0, HIDDEN * 4)?,
        )?)?;
        self.io().norm_rows(
            1,
            HIDDEN,
            self.hy.norm_eps,
            &sc.head,
            self.io().raw("output_norm.weight")?,
            &sc.cur,
            &sc.q8_head,
        )?;
        self.io()
            .mv("output.weight", Rows::q8(&sc.q8_head), &sc.logits, 1)?;
        let mut bytes = vec![0u8; self.vocab * 4];
        sc.logits.read_at(&self.stream, 0, &mut bytes)?;
        Ok(f32_values(&bytes))
    }
}

impl Visits for Raon1Device {
    fn count(&self) -> usize {
        self.spec.layers
    }

    fn begin(&self, _n: usize) -> Result<(), EngError> {
        Ok(())
    }

    /// The mixer half reads the streams a feed or the last layer left in `resid_a`, the
    /// feed-forward half the streams the mixer half left in `resid_b`.
    fn read(&self, layer: usize, half: Half, n: usize) -> Result<(), EngError> {
        let (names, sc) = (&self.names[layer], &self.sc);
        let (site, norm, resid, stage) = match half {
            Half::Mixer => (&names.hc_attn, &names.attn_norm, &sc.resid_a, "attn_norm"),
            Half::Forward => (&names.hc_ffn, &names.ffn_norm, &sc.resid_b, "ffn_norm"),
        };
        self.site_read(site, norm, n, resid)?;
        if self.trace.borrow().is_some() {
            self.snap(&format!("{stage}-{layer}"), &sc.cur, n * HIDDEN)?;
        }
        Ok(())
    }

    fn branch(&self, layer: usize, half: Half, n: usize, pos0: usize) -> Result<(), EngError> {
        let names = &self.names[layer];
        match (half, names.kind) {
            (Half::Mixer, RaonKind::Rda) => self.rda_mixer(names, n, pos0),
            (Half::Mixer, RaonKind::Mla) => self.mla_mixer(names, n, pos0),
            (Half::Forward, _) if names.moe => self.moe_block(names, n),
            (Half::Forward, _) => self.dense_block(names, n),
        }
    }

    fn apply(&self, layer: usize, half: Half, n: usize) -> Result<(), EngError> {
        let sc = &self.sc;
        let (resid, out, stage) = match half {
            Half::Mixer => (&sc.resid_a, &sc.resid_b, "hc_attn_out"),
            Half::Forward => (&sc.resid_b, &sc.resid_a, "l_out"),
        };
        self.site_apply(n, resid, out)?;
        if self.trace.borrow().is_some() {
            self.snap(&format!("{stage}-{layer}"), out, n * STREAMS * HIDDEN)?;
        }
        Ok(())
    }
}

impl ChunkStep for Raon1Device {
    fn family(&self) -> &'static str {
        "raon1"
    }

    fn chunk_rows(&self) -> usize {
        CHUNK
    }

    fn cells(&self) -> usize {
        self.cells
    }

    fn vocab(&self) -> usize {
        self.vocab
    }

    fn run_chunk(&mut self, tokens: &[u32], pos0: usize) -> Result<(), EngError> {
        self.chunk(tokens, pos0)
    }

    fn head_row(&self, row: usize) -> Result<Vec<f32>, EngError> {
        self.logits_row(row)
    }

    fn drain(&self) -> Result<(), EngError> {
        self.stream.synchronize()
    }
}

fn state_bytes(spec: Raon1Spec) -> u64 {
    let rda = (spec.layers - spec.mla_layers.len()) as u64;
    // The record, the convolution history and the convolution taps of every delta-rule layer.
    rda * (HEADS * HEAD_DIM * HEAD_DIM * 4) as u64
        + rda * 3 * ((CONV_TAPS + SPEC.conv_kernel as usize) * WIDTH * 4) as u64
}

fn allocate_scratch(ctx: &CudaContext, vocab: usize) -> Result<Scratch, EngError> {
    let c = CHUNK;
    let f = |count: usize| ctx.allocate(count * 4);
    let q = |width: usize, rows: usize| ctx.allocate(q8(width, rows));
    let part = (c * (HEADS / MLA_GROUP)).max(320) * MLA_GROUP * (RANK + 4);
    Ok(Scratch {
        resid_a: f(c * STREAMS * HIDDEN)?,
        resid_b: f(c * STREAMS * HIDDEN)?,
        branch: f(c * HIDDEN)?,
        post: f(c * STREAMS)?,
        comb: f(c * 16)?,
        cur: f(c * HIDDEN)?,
        q8_cur: q(HIDDEN, c)?,
        proj: f(c * HIDDEN)?,
        q8_wide: q(WIDTH, c)?,
        proj_qkv: f(c * 3 * WIDTH)?,
        scan_qkv: f(c * 3 * WIDTH)?,
        scan_decay: f(c * WIDTH)?,
        scan_proj: f(c * (WIDTH + HEADS))?,
        g1: f(c * WIDTH)?,
        g2: f(c * WIDTH)?,
        fa: f(c * TIME_RANK)?,
        beta: f(c * HEADS)?,
        delta: f(c * HEADS)?,
        scan_out: f(c * WIDTH)?,
        qa: f(c * Q_LORA)?,
        q8_qa: q(Q_LORA, c)?,
        qb: f(c * HEADS * Q_ROW)?,
        kv: f(c * MLA_ROW)?,
        qeff: f(c * HEADS * MLA_ROW)?,
        part: f(part)?,
        mix: f(c * HEADS * RANK)?,
        mla_out: f(c * WIDTH)?,
        ffn_gate: f(c * DENSE_INTER)?,
        ffn_up: f(c * DENSE_INTER)?,
        q8_ffn: q(DENSE_INTER, c)?,
        ids: f(c * USED)?,
        weights: f(c * USED)?,
        gu: f(c * USED * 2 * MOE_INTER)?,
        q8_experts: q(MOE_INTER, c * USED)?,
        act_experts: f(c * USED * MOE_INTER)?,
        down: f(c * USED * HIDDEN)?,
        shared: f(c * HIDDEN)?,
        head: f(HIDDEN)?,
        q8_head: q(HIDDEN, 1)?,
        logits: f(vocab)?,
        mla_gated: f(c * WIDTH)?,
        act_ffn: f(c * DENSE_INTER)?,
        mmq_q8: ctx.allocate(mmq_q8_bytes(DENSE_INTER, c)?)?,
        logits_route: f(c * EXPERTS)?,
        route_status: ctx.upload(&[0u8; 4])?,
        mmq_q8_routed: ctx.allocate(mmq_q8_bytes(HIDDEN, c * USED)?)?,
        mmq_q8_act: ctx.allocate(mmq_q8_bytes(MOE_INTER, c * USED)?)?,
        fixup: ctx.allocate(ctx.mmq_fixup_bytes()?)?,
        ids_g: f(c * USED)?,
        ids_src1: f(c * USED)?,
        ids_dst: f(experts::Picks::dst_ints(USED, c)?)?,
        expert_bounds: f(EXPERTS + 1)?,
    })
}
