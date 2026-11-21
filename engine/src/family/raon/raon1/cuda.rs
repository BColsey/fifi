//! The Raon-1 forward pass on one CUDA device.
//!
//! The stack is composed from the shared GGUF resident kernels (`matvec_q` and `matvec_expert`
//! over the file's own blocks, the F16 `dense_gemv`, `moe_combine`) and the row operations of
//! `kernels::cuda::resident::block_ops` (mixed residual, short convolution, delta-rule scan,
//! latent attention, SiTU, biased router). Every operation works on a chunk of token rows, so
//! prefill and decode share one path: a chunk of `n` tokens runs layer by layer, and decode is
//! the chunk of one. Quantized dense projections take up to 8 activation rows per launch; the
//! routed experts run per token.
//!
//! The embedding table stays in the file: each token's row is dequantized on the host and
//! uploaded into the four residual streams. The latent keys of the seven attention layers are
//! kept as f16 rows of 576 values; the delta-rule state and the convolution history of the
//! other eighteen layers are f32.

use std::cell::RefCell;
use std::path::Path;

use super::reference::{Hyper, hyper};
use super::{Raon1Model, Raon1Spec, RaonKind, open_raon_1_mini_preview};
use crate::EngError;
use crate::family::project::cuda::{
    BorrowedLoader, CudaWeight, CudaWeights, LoadOptions, Reserve, WeightLayout, served_admission,
};
use crate::kernels::cuda::prefill::mmq_dense::{MmqDenseCall, MmqDenseInputCall};
use crate::kernels::cuda::prefill::mmq_mmid::{
    IdsGuardCall, MmidIdsOut, MmidLaunchCall, MmidQuantCall, MmidShape, MmidWeight,
};
use crate::kernels::cuda::prefill::mmq_quant::MmqPacking;
use crate::kernels::cuda::prefill::mmq_size::q8_1_bytes as mmq_q8_bytes;
use crate::kernels::cuda::prefill::mmq_size::{expert_bounds_ints, ids_dst_ints};
use crate::kernels::cuda::resident::block_ops::{
    BlockOp, MLA_GROUP, MlaShape, RdaInputs, RowOutputs,
};
use crate::kernels::cuda::resident::dense_gemv::{DenseDtype, DenseGemvCall, DenseGemvShape};
use crate::kernels::cuda::resident::gdn_conv::{ConvCall, ConvShape};
use crate::kernels::cuda::resident::matvec_expert::ExpertMatvecCall;
use crate::kernels::cuda::resident::matvec_q::MatvecCall;
use crate::kernels::cuda::resident::moe_combine::{CombineCall, CombineOut, CombineShape};
use crate::kernels::cuda::resident::quantize_q8_1::q8_1_bytes;
use crate::kernels::cuda::{CudaContext, DeviceBuffer, DeviceRange, DeviceStream, NativeWeight};
use crate::weights::{GgufType, dequantize};

/// Token rows per chunk: the scratch is sized for this many.
pub(crate) const CHUNK: usize = 256;

/// Upper bound of the stream-k fixup scratch of the batched GEMM, for the memory plan.
const MMQ_FIXUP_BOUND: usize = 16 << 20;

const HIDDEN: usize = 2560;
const STREAMS: usize = 4;
const HEADS: usize = 40;
const HEAD_DIM: usize = 128;
const WIDTH: usize = HEADS * HEAD_DIM;
const RANK: usize = 512;
const ROPE: usize = 64;
const MLA_ROW: usize = RANK + ROPE;
const Q_NOPE: usize = 128;
const Q_ROW: usize = Q_NOPE + ROPE;
const Q_LORA: usize = 576;
const EXPERTS: usize = 128;
const USED: usize = 6;
const MOE_INTER: usize = 1088;
const DENSE_INTER: usize = 12_032;
const TIME_RANK: usize = 128;
const CONV_TAPS: usize = 3;
/// Rows per thread of the multi-row convolution kernel; shorter chunks use the one-row kernel.
const CONV_ROWS_PER_THREAD: usize = 16;
/// Bytes of KV cache per position: seven attention layers of 576 f16 values.
const KV_BYTES_PER_POSITION: u64 = 7 * MLA_ROW as u64 * 2;
/// Device bytes kept free beside the weights, the scratch and the cache.
const HEADROOM: u64 = 768 << 20;
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

fn q8(width: usize, rows: usize) -> usize {
    q8_1_bytes(width, rows).expect("a q8_1 row size fits usize")
}

fn plan_error(message: String) -> EngError {
    EngError::MemoryPlan(format!("raon1: {message}"))
}

/// The activation rows a projection reads: q8_1 rows for the quantized matvec and, when the
/// producer wrote them, the f32 rows for the batched GEMM.
#[derive(Clone, Copy)]
struct Rows<'a> {
    q8: &'a DeviceBuffer,
    f32: Option<&'a DeviceBuffer>,
}

impl<'a> Rows<'a> {
    fn q8(q8: &'a DeviceBuffer) -> Self {
        Self { q8, f32: None }
    }

    fn both(q8: &'a DeviceBuffer, f32: &'a DeviceBuffer) -> Self {
        Self { q8, f32: Some(f32) }
    }
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
    conv: [String; 3],
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
            conv: [
                n("ssm_conv1d_q.weight"),
                n("ssm_conv1d_k.weight"),
                n("ssm_conv1d_v.weight"),
            ],
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
    proj_q: DeviceBuffer,
    proj_k: DeviceBuffer,
    proj_v: DeviceBuffer,
    conv_q: DeviceBuffer,
    conv_k: DeviceBuffer,
    conv_v: DeviceBuffer,
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
    down: DeviceBuffer,
    shared: DeviceBuffer,
    head: DeviceBuffer,
    q8_head: DeviceBuffer,
    logits: DeviceBuffer,
    mla_gated: DeviceBuffer,
    act_ffn: DeviceBuffer,
    mmq_q8: DeviceBuffer,
    mmq_q8_routed: DeviceBuffer,
    fixup: DeviceBuffer,
    ids_g: DeviceBuffer,
    ids_src1: DeviceBuffer,
    ids_dst: DeviceBuffer,
    expert_bounds: DeviceBuffer,
}

impl Scratch {
    fn cur_rows(&self) -> Rows<'_> {
        Rows::both(&self.q8_cur, &self.cur)
    }

    /// The device bytes of the scratch of one chunk.
    fn bytes(vocab: usize) -> u64 {
        let c = CHUNK;
        let f = |count: usize| (count * 4) as u64;
        let q = |width: usize, rows: usize| q8(width, rows) as u64;
        let part = (c * (HEADS / MLA_GROUP)).max(320) * MLA_GROUP * (RANK + 4);
        f(c * STREAMS * HIDDEN) * 2
            + f(c * HIDDEN) * 4
            + f(c * STREAMS * 17)
            + q(HIDDEN, c)
            + q(WIDTH, c)
            + f(c * WIDTH) * 12
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
            + f(c * USED * HIDDEN)
            + f(c * HIDDEN)
            + f(HIDDEN)
            + q(HIDDEN, 1)
            + f(vocab)
            + f(c * WIDTH)
            + f(c * DENSE_INTER)
            + mmq_q8_bytes(DENSE_INTER, c).expect("mmq q8 size") as u64
            + MMQ_FIXUP_BOUND as u64
            + mmq_q8_bytes(HIDDEN, c * USED).expect("mmq q8 size") as u64
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
    sc: Scratch,
    /// `[rda layers][heads][head_dim][head_dim]` f32 delta-rule state.
    state: DeviceBuffer,
    /// `[rda layers][3][3][width]` f32 convolution history.
    hist: DeviceBuffer,
    /// `[mla layers][cells][576]` f16 latent cache.
    cache: DeviceBuffer,
    cells: usize,
    vocab: usize,
    trace: RefCell<Option<Vec<Stage>>>,
}

/// One named stage of a traced feed.
pub(crate) type Stage = (String, Vec<f32>);

/// Which rows of a fed chunk run through the output head.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum HeadRows {
    None,
    Last,
    #[cfg(test)]
    All,
}

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
            })
            .collect();
        let name_refs: Vec<&str> = names.iter().map(String::as_str).collect();
        let tensors = crate::family::project::cuda::native_load_bytes(model.weights(), &name_refs)?;
        let mla_factor_bytes = (spec.mla_layers.len() * 2 * HEADS * RANK * HEAD_DIM * 4) as u64;
        let fixed =
            tensors + mla_factor_bytes + Scratch::bytes(vocab) + state_bytes(spec) + HEADROOM;
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
            work: mla_factor_bytes + Scratch::bytes(vocab) + state_bytes(spec),
            headroom: HEADROOM,
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
            self.run(BlockOp::fill(count, 0.0, buffer.range(0, buffer.bytes())?)?)?;
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
        if tokens.is_empty() {
            return Err(EngError::BadRequest("raon1: a feed needs a token".into()));
        }
        if pos0
            .checked_add(tokens.len())
            .is_none_or(|end| end > self.cells)
        {
            return Err(EngError::ContextLimit(format!(
                "raon1: {} tokens at position {pos0} exceed the {} cached positions",
                tokens.len(),
                self.cells
            )));
        }
        if let Some(&token) = tokens.iter().find(|&&t| t as usize >= self.vocab) {
            return Err(EngError::BadRequest(format!(
                "raon1: token {token} is outside the vocabulary of {}",
                self.vocab
            )));
        }
        let mut logits = Vec::new();
        let mut done = 0;
        while done < tokens.len() {
            let piece = &tokens[done..(done + CHUNK).min(tokens.len())];
            let last_piece = done + piece.len() == tokens.len();
            self.run_chunk(piece, pos0 + done)?;
            match head {
                HeadRows::None => {}
                HeadRows::Last if !last_piece => {}
                HeadRows::Last => logits.push(self.head_row(piece.len() - 1)?),
                #[cfg(test)]
                HeadRows::All => {
                    for row in 0..piece.len() {
                        logits.push(self.head_row(row)?);
                    }
                }
            }
            done += piece.len();
        }
        if head == HeadRows::None {
            self.stream.synchronize()?;
        }
        Ok(logits)
    }

    fn run(&self, op: BlockOp<'_>) -> Result<(), EngError> {
        self.ctx.block_op(&self.stream, op)
    }

    fn native(&self, name: &str) -> Result<&NativeWeight, EngError> {
        match self.weights.get(name) {
            Some(CudaWeight::Native { weight, .. }) => Ok(weight),
            _ => Err(EngError::Checkpoint(format!(
                "raon1: tensor {name} is not resident as native blocks"
            ))),
        }
    }

    fn raw(&self, name: &str) -> Result<DeviceRange<'_>, EngError> {
        match self.weights.get(name) {
            Some(CudaWeight::Raw { buffer, .. }) => buffer.range(0, buffer.bytes()),
            _ => Err(EngError::Checkpoint(format!(
                "raon1: tensor {name} is not resident as a raw tensor"
            ))),
        }
    }

    fn whole(buffer: &DeviceBuffer) -> Result<DeviceRange<'_>, EngError> {
        buffer.range(0, buffer.bytes())
    }

    /// `y[t] = W x[t]` for `n` rows. Up to 8 rows use the quantized matvec on the q8_1 rows;
    /// a larger chunk with f32 rows and a row length on the 256 grid uses the batched GEMM.
    fn mv(&self, name: &str, x: Rows<'_>, y: &DeviceBuffer, n: usize) -> Result<(), EngError> {
        let weight = self.native(name)?;
        let (k, rows) = (weight.k(), weight.n_rows());
        if n > 8
            && k.is_multiple_of(256)
            && let Some(f32_rows) = x.f32
        {
            let q8_range = self.sc.mmq_q8.range(0, mmq_q8_bytes(k, n)?)?;
            self.ctx.mmq_dense_input(
                &self.stream,
                MmqDenseInputCall {
                    packing: MmqPacking::CpuReference,
                    weight,
                    m: n,
                    row_stride: k,
                    x: f32_rows.range(0, n * k * 4)?,
                    y: q8_range,
                },
            )?;
            return self.ctx.mmq_dense(
                &self.stream,
                MmqDenseCall {
                    weight,
                    w_seg: None,
                    m: n,
                    q8_1: q8_range,
                    dst: y.range(0, n * rows * 4)?,
                    ld: rows,
                    col_off: 0,
                    fixup: Self::whole(&self.sc.fixup)?,
                },
            );
        }
        let w = Self::whole(weight.buffer())?;
        let x_stride = q8(k, 1);
        let mut first = 0;
        while first < n {
            let cols = (n - first).min(8);
            self.ctx.matvec_q(
                &self.stream,
                MatvecCall {
                    ty: weight.quant(),
                    w,
                    x_q8: x.q8.range(first * x_stride, q8(k, cols))?,
                    y: y.range(first * rows * 4, cols * rows * 4)?,
                    k,
                    rows,
                    ncols: cols,
                },
            )?;
            first += cols;
        }
        Ok(())
    }

    /// The F16 projection `y[t] = W x[t]` of `n` f32 rows.
    fn f16_mv(
        &self,
        name: &str,
        x: &DeviceBuffer,
        y: &DeviceBuffer,
        n: usize,
        rows: usize,
        cols: usize,
    ) -> Result<(), EngError> {
        self.ctx.dense_gemv(
            &self.stream,
            DenseGemvCall {
                shape: DenseGemvShape::packed(n, rows, cols),
                dtype: DenseDtype::F16,
                w: self.raw(name)?,
                x: x.range(0, n * cols * 4)?,
                y: y.range(0, n * rows * 4)?,
            },
        )
    }

    /// Quantizes `n` rows of `width` f32 values (x * sigmoid(gate) when a gate is given) into
    /// q8_1 rows, and also writes the f32 product when `f32_out` is given.
    fn quantize(
        &self,
        n: usize,
        width: usize,
        x: &DeviceBuffer,
        gate: Option<&DeviceBuffer>,
        out: Rows<'_>,
    ) -> Result<(), EngError> {
        self.run(BlockOp::gated_quant(
            n,
            width,
            x.range(0, n * width * 4)?,
            gate.map(|g| g.range(0, n * width * 4)).transpose()?,
            RowOutputs {
                q8: Some(out.q8.range(0, q8(width, n))?),
                f32: out.f32.map(|b| b.range(0, n * width * 4)).transpose()?,
            },
        )?)
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

    fn embedding_row(&self, token: u32) -> Result<Vec<f32>, EngError> {
        let row_bytes = HIDDEN / 256 * 144;
        let mut bytes = vec![0u8; row_bytes];
        self.model.weights().read_range(
            "token_embd.weight",
            u64::from(token) * row_bytes as u64,
            &mut bytes,
        )?;
        dequantize(GgufType::Q4K, &bytes)
    }

    fn run_chunk(&mut self, tokens: &[u32], pos0: usize) -> Result<(), EngError> {
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
        for layer in 0..self.spec.layers {
            self.attn_half(layer, n, pos0)?;
            self.ffn_half(layer, n)?;
        }
        Ok(())
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
        self.run(BlockOp::hc_read(
            n,
            HIDDEN,
            self.hy.hc_eps,
            self.hy.hc_magnitude,
            resid.range(0, n * STREAMS * HIDDEN * 4)?,
            self.raw(&site.mix)?,
            self.raw(&site.base)?,
            self.raw(&site.scale)?,
            sc.branch.range(0, n * HIDDEN * 4)?,
            sc.post.range(0, n * STREAMS * 4)?,
            sc.comb.range(0, n * 16 * 4)?,
        )?)?;
        self.run(BlockOp::rms_rows(
            n,
            HIDDEN,
            HIDDEN,
            self.hy.norm_eps,
            sc.branch.range(0, n * HIDDEN * 4)?,
            self.raw(norm)?,
            Some(sc.cur.range(0, n * HIDDEN * 4)?),
            Some(sc.q8_cur.range(0, q8(HIDDEN, n))?),
        )?)
    }

    /// The stream update of one half: `out = comb * resid + post * proj`.
    fn site_apply(
        &self,
        n: usize,
        resid: &DeviceBuffer,
        out: &DeviceBuffer,
    ) -> Result<(), EngError> {
        let sc = &self.sc;
        self.run(BlockOp::hc_apply(
            n,
            HIDDEN,
            resid.range(0, n * STREAMS * HIDDEN * 4)?,
            sc.proj.range(0, n * HIDDEN * 4)?,
            sc.post.range(0, n * STREAMS * 4)?,
            sc.comb.range(0, n * 16 * 4)?,
            out.range(0, n * STREAMS * HIDDEN * 4)?,
        )?)
    }

    fn attn_half(&self, layer: usize, n: usize, pos0: usize) -> Result<(), EngError> {
        let names = &self.names[layer];
        let sc = &self.sc;
        self.site_read(&names.hc_attn, &names.attn_norm, n, &sc.resid_a)?;
        if self.trace.borrow().is_some() {
            self.snap(&format!("attn_norm-{layer}"), &sc.cur, n * HIDDEN)?;
        }
        match names.kind {
            RaonKind::Rda => self.rda_mixer(names, n)?,
            RaonKind::Mla => self.mla_mixer(names, n, pos0)?,
        }
        self.site_apply(n, &sc.resid_a, &sc.resid_b)?;
        if self.trace.borrow().is_some() {
            self.snap(
                &format!("hc_attn_out-{layer}"),
                &sc.resid_b,
                n * STREAMS * HIDDEN,
            )?;
        }
        Ok(())
    }

    fn rda_mixer(&self, names: &LayerNames, n: usize) -> Result<(), EngError> {
        let sc = &self.sc;
        let rda = names.ordinal;
        self.mv(&names.attn_q, sc.cur_rows(), &sc.proj_q, n)?;
        self.mv(&names.attn_k, sc.cur_rows(), &sc.proj_k, n)?;
        self.mv(&names.attn_v, sc.cur_rows(), &sc.proj_v, n)?;
        self.mv(&names.ssm_g, sc.cur_rows(), &sc.g2, n)?;
        self.f16_mv(&names.ssm_f_a, &sc.cur, &sc.fa, n, TIME_RANK, HIDDEN)?;
        self.f16_mv(&names.ssm_f_b, &sc.fa, &sc.g1, n, WIDTH, TIME_RANK)?;
        self.f16_mv(&names.ssm_beta, &sc.cur, &sc.beta, n, HEADS, HIDDEN)?;
        self.f16_mv(&names.ssm_rda, &sc.cur, &sc.delta, n, HEADS, HIDDEN)?;
        let row = n * WIDTH * 4;
        let hist_at = |which: usize| (rda * 3 + which) * CONV_TAPS * WIDTH * 4;
        for (which, (input, output)) in [
            (&sc.proj_q, &sc.conv_q),
            (&sc.proj_k, &sc.conv_k),
            (&sc.proj_v, &sc.conv_v),
        ]
        .into_iter()
        .enumerate()
        {
            let call = ConvCall {
                shape: ConvShape {
                    m: n,
                    channels: WIDTH,
                    ld: WIDTH,
                },
                proj: input.range(0, row)?,
                weights: self.raw(&names.conv[which])?,
                state: self.hist.range(hist_at(which), CONV_TAPS * WIDTH * 4)?,
                out: output.range(0, row)?,
            };
            if n >= CONV_ROWS_PER_THREAD {
                self.ctx
                    .gdn_conv_rows(&self.stream, call, CONV_ROWS_PER_THREAD)?;
            } else {
                self.ctx.gdn_conv(&self.stream, call)?;
            }
        }
        let state_bytes = HEADS * HEAD_DIM * HEAD_DIM * 4;
        self.run(BlockOp::rda_scan(
            n,
            HEADS,
            HEAD_DIM,
            self.hy.gate_lower_bound,
            self.hy.norm_eps,
            RdaInputs {
                q: sc.conv_q.range(0, row)?,
                k: sc.conv_k.range(0, row)?,
                v: sc.conv_v.range(0, row)?,
                g1: sc.g1.range(0, row)?,
                dt_bias: self.raw(&names.ssm_dt)?,
                a: self.raw(&names.ssm_a)?,
                beta: sc.beta.range(0, n * HEADS * 4)?,
                delta: sc.delta.range(0, n * HEADS * 4)?,
                norm_w: self.raw(&names.ssm_norm)?,
                g2: sc.g2.range(0, row)?,
            },
            self.state.range(rda * state_bytes, state_bytes)?,
            sc.scan_out.range(0, row)?,
        )?)?;
        self.quantize(n, WIDTH, &sc.scan_out, None, Rows::q8(&sc.q8_wide))?;
        self.mv(
            &names.attn_output,
            Rows::both(&sc.q8_wide, &sc.scan_out),
            &sc.proj,
            n,
        )
    }

    fn mla_mixer(&self, names: &LayerNames, n: usize, pos0: usize) -> Result<(), EngError> {
        let sc = &self.sc;
        let mla = names.ordinal;
        self.mv(&names.attn_q_a, sc.cur_rows(), &sc.qa, n)?;
        self.run(BlockOp::rms_rows(
            n,
            Q_LORA,
            Q_LORA,
            self.hy.latent_eps,
            sc.qa.range(0, n * Q_LORA * 4)?,
            self.raw(&names.attn_q_a_norm)?,
            None,
            Some(sc.q8_qa.range(0, q8(Q_LORA, n))?),
        )?)?;
        self.mv(&names.attn_q_b, Rows::q8(&sc.q8_qa), &sc.qb, n)?;
        self.mv(&names.attn_kv_a, sc.cur_rows(), &sc.kv, n)?;
        let layer_cache = self.cells * MLA_ROW * 2;
        self.run(BlockOp::mla_kv_prep(
            n,
            RANK,
            ROPE,
            pos0,
            self.cells,
            self.hy.latent_eps,
            sc.kv.range(0, n * MLA_ROW * 4)?,
            self.raw(&names.attn_kv_a_norm)?,
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
        self.run(BlockOp::mla_q_absorb(
            shape,
            sc.qb.range(0, n * HEADS * Q_ROW * 4)?,
            Self::whole(&factors.wk_b)?,
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
        self.run(BlockOp::mla_attn(
            shape,
            pos0,
            splits,
            per_split,
            1.0 / (self.spec.key_mla as f32).sqrt(),
            sc.qeff.range(0, n * HEADS * MLA_ROW * 4)?,
            self.cache.range(mla * layer_cache, layer_cache)?,
            sc.part.range(0, part_bytes)?,
        )?)?;
        self.run(BlockOp::mla_reduce(
            n,
            HEADS,
            RANK,
            splits,
            sc.part.range(0, part_bytes)?,
            sc.mix.range(0, n * HEADS * RANK * 4)?,
        )?)?;
        self.run(BlockOp::mla_project(
            n,
            HEADS,
            RANK,
            HEAD_DIM,
            sc.mix.range(0, n * HEADS * RANK * 4)?,
            Self::whole(&factors.wv_b)?,
            sc.mla_out.range(0, n * WIDTH * 4)?,
        )?)?;
        self.mv(&names.attn_gate, sc.cur_rows(), &sc.g2, n)?;
        if self.trace.borrow().is_some() {
            self.snap("mla_out", &sc.mla_out, n * WIDTH)?;
            self.snap("mla_gate", &sc.g2, n * WIDTH)?;
        }
        self.quantize(
            n,
            WIDTH,
            &sc.mla_out,
            Some(&sc.g2),
            Rows::both(&sc.q8_wide, &sc.mla_gated),
        )?;
        self.mv(
            &names.attn_output,
            Rows::both(&sc.q8_wide, &sc.mla_gated),
            &sc.proj,
            n,
        )
    }

    fn ffn_half(&self, layer: usize, n: usize) -> Result<(), EngError> {
        let names = &self.names[layer];
        let sc = &self.sc;
        self.site_read(&names.hc_ffn, &names.ffn_norm, n, &sc.resid_b)?;
        if self.trace.borrow().is_some() {
            self.snap(&format!("ffn_norm-{layer}"), &sc.cur, n * HIDDEN)?;
        }
        if names.moe {
            self.moe_block(names, n)?;
        } else {
            self.dense_block(names, n)?;
        }
        self.site_apply(n, &sc.resid_b, &sc.resid_a)?;
        if self.trace.borrow().is_some() {
            self.snap(&format!("l_out-{layer}"), &sc.resid_a, n * STREAMS * HIDDEN)?;
        }
        Ok(())
    }

    fn situ(
        &self,
        rows: usize,
        width: usize,
        gate: &DeviceBuffer,
        up: &DeviceRange<'_>,
        ld: usize,
        out: Rows<'_>,
    ) -> Result<(), EngError> {
        let span = ((rows - 1) * ld + width) * 4;
        self.run(BlockOp::situ_quant(
            rows,
            width,
            ld,
            ld,
            (self.hy.situ_beta, self.hy.situ_linear_beta),
            gate.range(0, span)?,
            *up,
            RowOutputs {
                q8: Some(out.q8.range(0, q8(width, rows))?),
                f32: out.f32.map(|b| b.range(0, rows * width * 4)).transpose()?,
            },
        )?)
    }

    fn dense_block(&self, names: &LayerNames, n: usize) -> Result<(), EngError> {
        let sc = &self.sc;
        self.mv(&names.ffn_gate, sc.cur_rows(), &sc.ffn_gate, n)?;
        self.mv(&names.ffn_up, sc.cur_rows(), &sc.ffn_up, n)?;
        let up = sc.ffn_up.range(0, n * DENSE_INTER * 4)?;
        self.situ(
            n,
            DENSE_INTER,
            &sc.ffn_gate,
            &up,
            DENSE_INTER,
            Rows::both(&sc.q8_ffn, &sc.act_ffn),
        )?;
        self.mv(
            &names.ffn_down,
            Rows::both(&sc.q8_ffn, &sc.act_ffn),
            &sc.proj,
            n,
        )
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
        let weight = self.native(name)?;
        self.ctx.matvec_expert(
            &self.stream,
            ExpertMatvecCall {
                ty: weight.quant(),
                w_stack: Self::whole(weight.buffer())?,
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

    /// The gate and up projections of a chunk's routed picks as grouped GEMMs: tokens are
    /// regrouped by expert, each expert's weights are read once, and the rows land in `gu` at
    /// the slot rows the per-token launches write.
    fn grouped_gate_up(
        &self,
        names: &LayerNames,
        n: usize,
        gu_row: usize,
        stride: usize,
    ) -> Result<(), EngError> {
        let sc = &self.sc;
        let shape = MmidShape {
            n_experts: EXPERTS,
            m: n,
            top_k: USED,
        };
        let ids = self.ctx.mmq_build_ids(
            &self.stream,
            IdsGuardCall {
                shape,
                ids: sc.ids.range(0, n * USED * 4)?,
                ids_g: sc.ids_g.range(0, n * USED * 4)?,
                status: None,
            },
            MmidIdsOut {
                ids_src1: sc.ids_src1.range(0, n * USED * 4)?,
                ids_dst: sc.ids_dst.range(0, ids_dst_ints(n * USED, n)? * 4)?,
                expert_bounds: sc
                    .expert_bounds
                    .range(0, expert_bounds_ints(EXPERTS)? * 4)?,
            },
        )?;
        let gate = self.native(&names.gate_exps)?;
        let q8_rows = sc.mmq_q8_routed.range(0, mmq_q8_bytes(HIDDEN, n * USED)?)?;
        self.ctx.mmq_quantize_scatter_ids(
            &self.stream,
            &ids,
            MmidQuantCall {
                packing: MmqPacking::CpuReference,
                quant: gate.quant(),
                k: HIDDEN,
                row_stride: HIDDEN,
                x: sc.cur.range(0, n * HIDDEN * 4)?,
                y: q8_rows,
            },
        )?;
        for (weight, offset) in [(gate, 0), (self.native(&names.up_exps)?, MOE_INTER * 4)] {
            let quant = weight.quant();
            let stride_bytes = quant
                .weight_bytes(MOE_INTER, HIDDEN)
                .ok_or_else(|| EngError::BadRequest("raon1: the expert stride overflows".into()))?;
            self.ctx.mmq_mmid_launch(
                &self.stream,
                MmidLaunchCall {
                    shape,
                    weight: MmidWeight {
                        base: Self::whole(weight.buffer())?,
                        quant,
                        k: HIDDEN,
                        n: MOE_INTER,
                        stride_bytes,
                    },
                    ids: &ids,
                    y: q8_rows,
                    dst: sc.gu.range(offset, n * gu_row - offset)?,
                    ld: stride,
                    fixup: Self::whole(&sc.fixup)?,
                },
            )?;
        }
        Ok(())
    }

    fn moe_block(&self, names: &LayerNames, n: usize) -> Result<(), EngError> {
        let sc = &self.sc;
        self.run(BlockOp::moe_route(
            n,
            HIDDEN,
            EXPERTS,
            USED,
            self.hy.weights_norm,
            6.103_515_6e-5,
            sc.cur.range(0, n * HIDDEN * 4)?,
            self.raw(&names.gate_inp)?,
            self.raw(&names.exp_bias)?,
            sc.ids.range(0, n * USED * 4)?,
            sc.weights.range(0, n * USED * 4)?,
        )?)?;
        // The shared expert first: its output is the base the routed picks are summed onto.
        self.mv(&names.gate_shexp, sc.cur_rows(), &sc.ffn_gate, n)?;
        self.mv(&names.up_shexp, sc.cur_rows(), &sc.ffn_up, n)?;
        let up = sc.ffn_up.range(0, n * MOE_INTER * 4)?;
        self.situ(
            n,
            MOE_INTER,
            &sc.ffn_gate,
            &up,
            MOE_INTER,
            Rows::q8(&sc.q8_ffn),
        )?;
        self.mv(&names.down_shexp, Rows::q8(&sc.q8_ffn), &sc.shared, n)?;
        let gu_row = USED * 2 * MOE_INTER * 4;
        let stride = 2 * MOE_INTER;
        if n > 8 {
            self.grouped_gate_up(names, n, gu_row, stride)?;
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
        let up = sc.gu.range(MOE_INTER * 4, n * gu_row - MOE_INTER * 4)?;
        self.situ(
            n * USED,
            MOE_INTER,
            &sc.gu,
            &up,
            stride,
            Rows::q8(&sc.q8_experts),
        )?;
        let e_stride = q8(MOE_INTER, 1);
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
    fn head_row(&self, row: usize) -> Result<Vec<f32>, EngError> {
        let sc = &self.sc;
        self.run(BlockOp::hc_mean(
            row,
            HIDDEN,
            sc.resid_a.range(0, (row + 1) * STREAMS * HIDDEN * 4)?,
            sc.head.range(0, HIDDEN * 4)?,
        )?)?;
        self.run(BlockOp::rms_rows(
            1,
            HIDDEN,
            HIDDEN,
            self.hy.norm_eps,
            sc.head.range(0, HIDDEN * 4)?,
            self.raw("output_norm.weight")?,
            None,
            Some(sc.q8_head.range(0, q8(HIDDEN, 1))?),
        )?)?;
        self.mv("output.weight", Rows::q8(&sc.q8_head), &sc.logits, 1)?;
        let mut bytes = vec![0u8; self.vocab * 4];
        sc.logits.read_at(&self.stream, 0, &mut bytes)?;
        Ok(f32_values(&bytes))
    }
}

fn state_bytes(spec: Raon1Spec) -> u64 {
    let rda = (spec.layers - spec.mla_layers.len()) as u64;
    rda * (HEADS * HEAD_DIM * HEAD_DIM * 4) as u64 + rda * 3 * (CONV_TAPS * WIDTH * 4) as u64
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
        proj_q: f(c * WIDTH)?,
        proj_k: f(c * WIDTH)?,
        proj_v: f(c * WIDTH)?,
        conv_q: f(c * WIDTH)?,
        conv_k: f(c * WIDTH)?,
        conv_v: f(c * WIDTH)?,
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
        down: f(c * USED * HIDDEN)?,
        shared: f(c * HIDDEN)?,
        head: f(HIDDEN)?,
        q8_head: q(HIDDEN, 1)?,
        logits: f(vocab)?,
        mla_gated: f(c * WIDTH)?,
        act_ffn: f(c * DENSE_INTER)?,
        mmq_q8: ctx.allocate(mmq_q8_bytes(DENSE_INTER, c)?)?,
        mmq_q8_routed: ctx.allocate(mmq_q8_bytes(HIDDEN, c * USED)?)?,
        fixup: ctx.allocate(ctx.mmq_fixup_bytes()?)?,
        ids_g: f(c * USED)?,
        ids_src1: f(c * USED)?,
        ids_dst: f(ids_dst_ints(c * USED, c)?)?,
        expert_bounds: f(EXPERTS + 1)?,
    })
}
