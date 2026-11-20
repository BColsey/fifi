//! The resident Nemotron Bolt E4B decode step on one CUDA device.
//!
//! This module composes the checked resident operations over the converted Bolt
//! Q8_0 checkpoint. Each of the forty-two decoder layers runs three residual
//! blocks: attention over the KV8 planes, the dense GeLU-tanh feed-forward and
//! the per-layer-input block. The last eighteen layers hold no k/v projections
//! and no cache planes; their attention reads the cache of the last physical
//! layer of their own kind (layer 22 for sliding, 23 for full), so only the
//! twenty-four physical layers consume cache memory.
//!
//! The per-layer inputs are computed once per step from the scaled embedding:
//! the model projection times hidden^-0.5, one 256-wide RMSNorm slice per
//! layer, plus sixteen times the token's per-layer embedding row, all scaled by
//! 2^-0.5. The step ends with the final RMSNorm, the tied-embedding logit
//! projection and the logit softcap.
//!
//! Limits:
//! - The cache holds 32,768 cells per physical layer, half the model's 65,536
//!   maximum positions; widening is a constant change.
//! - Prefill runs bounded batches through the batched step.
//! - The per-head RMSNorm runs one batched launch across heads.
//! - Shared layers rotate one scratch k row, because the rope kernel requires
//!   at least one; the row is never read.
//! - A decode step's capped logits are read back to the host; host sampling
//!   draws from that row.

#![cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "native-cuda resident helpers beyond the serving path"
    )
)]

use super::bolt_graph::{
    DecodeGraphFacts, DecodeGraphMode, check_decode_graph, decode_graph_wanted, parse_switch,
};
use super::step::layer_buffers;
use super::{
    Kv, Matrix, PackedMatrix, Vector, epsilon, matrix, packed_matrix, scalar, vector,
    vector_filled, zero,
};
use crate::EngError;
use crate::family::gemma4::gguf::ple_rows_from_merged;
use crate::family::spec::{AttentionKind, NEMOTRON_BOLT_E4B};
use crate::kernels::cuda::DeviceBuffer;
use crate::kernels::cuda::DeviceRange;
use crate::kernels::cuda::resident::add_rmsnorm_q8::AddNormQ8Call;
use crate::kernels::cuda::resident::attn_decode_gemma::{
    AttnDecodeGemmaCall, AttnDecodeGemmaSplit,
};
use crate::kernels::cuda::resident::embed_q8::EmbedQ8Call;
use crate::kernels::cuda::resident::gelu_ffn::GeluFfnCall;
use crate::kernels::cuda::resident::gemma_matvec::GemmaMatvecCall;
use crate::kernels::cuda::resident::kv8_append::{Kv8CodeScale, Kv8Data, Kv8Shape};
use crate::kernels::cuda::resident::matvec_q::{MatvecCall, QuantType};
use crate::kernels::cuda::resident::quantize_q8_1::QuantizeCall;
use crate::kernels::cuda::resident::quantize_q8_1::q8_1_bytes;
use crate::kernels::cuda::resident::rope_gemma::RopeGemmaCall;
use crate::kernels::cuda::resident::row_add::RowAddCall;
use crate::kernels::cuda::resident::row_scale::RowScaleCall;
use crate::kernels::cuda::resident::softcap::SoftcapCall;
use crate::kernels::cuda::{CapturePlan, CudaContext, DeviceStream, GraphExec, is_capture_refusal};
use crate::weights::GgufType;
use std::sync::atomic::{AtomicU64, Ordering};

// Geometry is read from the family descriptor, the single source of truth;
// the executor pins only the cache cell count and the checkpoint properties
// the descriptor does not carry (PLE width, soft cap).
pub(crate) const LAYERS: usize = NEMOTRON_BOLT_E4B.layers as usize;
/// The physical k/v layers; the shared layers attend into their caches.
pub(crate) const PHYSICAL_LAYERS: usize =
    (NEMOTRON_BOLT_E4B.layers - NEMOTRON_BOLT_E4B.kv_shared_layers) as usize;
/// Values per hidden row.
pub(crate) const HIDDEN: usize = NEMOTRON_BOLT_E4B.hidden as usize;
/// Rows of the tied embedding and of the logit projection.
pub(crate) const VOCAB: usize = NEMOTRON_BOLT_E4B.vocab as usize;
pub(crate) const HEADS: usize = NEMOTRON_BOLT_E4B.heads_q as usize;
pub(crate) const KV_HEADS: usize = NEMOTRON_BOLT_E4B.heads_kv as usize;
pub(crate) const LOCAL_HEAD: usize = NEMOTRON_BOLT_E4B.head_dim as usize;
pub(crate) const GLOBAL_HEAD: usize = NEMOTRON_BOLT_E4B.global_head_dim as usize;
/// Width of one per-layer input and of the gated mixing inside every layer.
pub(crate) const PLE: usize = 256;
pub(crate) const FFN: usize = 10_240;
/// KV cells per physical layer: half the model's 65,536 maximum positions.
pub(crate) const CELLS: usize = 32_768;
pub(crate) const LOCAL_WINDOW: usize = NEMOTRON_BOLT_E4B.sliding_window as usize;
pub(crate) const SOFT_CAP: f32 = 30.0;
/// The identity and mixing scales applied to the per-layer inputs.
const PLE_IDENTITY_SCALE: f32 = 16.0;
const PLE_COMBINED_SCALE: f32 = std::f32::consts::SQRT_2.recip();

/// The device bytes the scratch can hold at once: every growable activation
/// vector and q8 buffer sized for `PREFILL_BATCH` rows (the maximum
/// `grow_scratch` allocates), plus the fixed ones/logits/head/qk buffers and
/// the three input cells. The starter admits this against free memory before
/// loading. `None` on overflow. Mirrors `grow_scratch` so the charge cannot
/// undercount the prefill working set. The attention split's partial buffer
/// is charged only when `attn_split_armed` (the split is the default since it
/// landed; the admission must refuse an armed load before allocating).
pub(crate) fn scratch_bytes(attn_split_armed: bool) -> Option<u64> {
    let rows = Bolt::PREFILL_BATCH as u64;
    let growable_f32 = [
        8 * HIDDEN,                 // h[4], nrm, nrm2, proj, down
        3 * HEADS * GLOBAL_HEAD,    // q_normed, q, ctx_out
        4 * KV_HEADS * GLOBAL_HEAD, // k_normed, v_normed, k, v
        3 * FFN,                    // gate, up, mixed
        2 * LAYERS * PLE,           // ple_all, ple_scaled
        3 * PLE,                    // ple_nrm, ple_tbl, ple_mixed
    ];
    let mut total = 0u64;
    for elements in growable_f32 {
        total = total.checked_add((elements as u64).checked_mul(rows)?.checked_mul(4)?)?;
    }
    // Fixed f32 buffers: ones, logits, logits_capped.
    total = total.checked_add(((GLOBAL_HEAD + 2 * VOCAB) as u64).checked_mul(4)?)?;
    // Growable q8 buffers at PREFILL_BATCH columns.
    for width in [HIDDEN, HEADS * GLOBAL_HEAD, FFN, PLE] {
        total = total.checked_add(q8_1_bytes(width, Bolt::PREFILL_BATCH)? as u64)?;
    }
    // Fixed q8 buffers: the head row and the qk dump.
    total = total.checked_add(q8_1_bytes(GLOBAL_HEAD, 1)? as u64)?;
    total = total.checked_add(q8_1_bytes(GLOBAL_HEAD, Bolt::PREFILL_BATCH * HEADS)? as u64)?;
    // ids, pos, kv_len input cells.
    total = total.checked_add(12)?;
    // The split partial buffer, present only when the attention split is armed.
    if attn_split_armed {
        total = total.checked_add(attn_part_bytes() as u64)?;
    }
    Some(total)
}

/// The parity dump directory, read from BOLT_TRACE_DIR once. The trace hook
/// runs per checkpoint per token; caching the environment read keeps the
/// off-path free of repeated allocations.
fn trace_dir() -> Option<&'static str> {
    static DIR: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    DIR.get_or_init(|| std::env::var("BOLT_TRACE_DIR").ok())
        .as_deref()
}

/// The environment variable that selects the split decode attention: one
/// part+combine launch pair per attention instead of the single-pass kernel.
/// The split changes the summation order, matches the f64 reference within
/// its tolerance, and is bit-identical only between its eager and graphed
/// runs (the KV8 combine lift is the bit-identical piece). Armed by default
/// since it qualified (11.7x on the full-attention call at 30K depth);
/// `0` or `off` is the rollback to the single-pass summation order.
pub(crate) const ATTN_SPLIT_VAR: &str = "FIFI_BOLT_ATTN_SPLIT";

/// Split policy: at most 32 splits over the visible range, at least 128 keys
/// per split, so a short context keeps one split and a long one fills the
/// device (8 q heads x 32 splits = 256 blocks on an 82-SM card).
pub(crate) const ATTN_SPLIT_MAX: usize = 32;
pub(crate) const ATTN_SPLIT_MIN_KEYS: usize = 128;

/// The split partial buffer of the resident's attention: stats (max, sum) plus
/// the accumulator vector, per (token, q head, split). Sized on the 512-dim
/// global head and the full prefill batch, because the eager step runs the
/// same kernels for m = 1..=PREFILL_BATCH; the 256-dim local layers need less.
pub(crate) const fn attn_part_bytes() -> usize {
    Bolt::PREFILL_BATCH * HEADS * ATTN_SPLIT_MAX * (2 + GLOBAL_HEAD) * 4
}

/// The armed state the environment text selects: unset arms the qualified
/// split (the default); `0`/`off` disarms it, `1`/`on` arms it, and anything
/// else is refused. Split from the env read so the default rule has a
/// portable test: edition 2024 makes `std::env::set_var` unsafe and the crate
/// denies `unsafe_code`, so a test cannot carry the variable into the read.
pub(crate) fn split_armed_from(raw: Option<&str>) -> Result<bool, EngError> {
    match raw {
        None => Ok(true),
        Some(v) => parse_switch(Some(v), &format!("bolt attention split: {ATTN_SPLIT_VAR}")),
    }
}

/// Whether the environment arms the attention split; a bad value fails the
/// load instead of arming or disarming by accident.
pub(crate) fn attn_split_wanted() -> Result<bool, EngError> {
    split_armed_from(std::env::var(ATTN_SPLIT_VAR).ok().as_deref())
}

pub(crate) struct BoltLayer {
    attn_norm: Vector,
    q_norm: Vector,
    k_norm: Option<Vector>,
    q: PackedMatrix,
    k: Option<PackedMatrix>,
    v: Option<PackedMatrix>,
    out: PackedMatrix,
    post_attn_norm: Vector,
    ffn_norm: Vector,
    ffn_gate: PackedMatrix,
    ffn_up: PackedMatrix,
    ffn_down: PackedMatrix,
    post_ffw_norm: Vector,
    ple_gate: PackedMatrix,
    ple_proj: PackedMatrix,
    post_ple_norm: Vector,
    table: Matrix,
    scale: f32,
    global: bool,
    head: usize,
    /// For a layer without its own cache planes, the physical layer whose planes it attends into.
    kv_source: Option<usize>,
}

impl BoltLayer {
    /// Whether the layer writes cache planes of its own.
    fn physical(&self) -> bool {
        self.kv_source.is_none()
    }
}

/// The device state of one sequence: its KV8 planes for the physical layers and the two words
/// the position and attention kernels read. A session owns one; the model owns none, so any
/// number of sequences can step through the same weights.
pub(crate) struct BoltSeq {
    id: u64,
    /// One entry per layer; `Some` for the physical layers.
    kv: Vec<Option<Kv>>,
    /// The position of the first row of the step (`StepParams.pos`).
    pos: DeviceBuffer,
    /// The key count the first row attends over (`pos + 1`).
    kv_len: DeviceBuffer,
}

/// The logits rows of a window: the raw head output and its softcap, one vocabulary row per
/// sequence. Allocated only for an engine that serves more than one sequence.
pub(crate) struct WindowBuffers {
    rows: usize,
    logits: Vector,
    capped: Vector,
}

/// What the rows of one step are.
enum Rows<'a> {
    /// `m` consecutive positions of one sequence from `pos`: a prompt chunk or one decode token.
    Chunk { seq: &'a BoltSeq, pos: usize },
    /// One token of each of several sequences, in window order.
    Window { seqs: &'a [&'a BoltSeq] },
}

impl Rows<'_> {
    /// The position label of the trace files.
    fn trace_pos(&self) -> usize {
        match self {
            Self::Chunk { pos, .. } => *pos,
            Self::Window { .. } => 0,
        }
    }
}

/// The columns of one head launch in a window. The vendored matvec keeps one summation order
/// for up to four columns and switches to another above that, so a window of more rows runs the
/// head in launches of at most this many columns and every row sees the order a lone row sees.
pub(crate) const HEAD_COLUMNS: usize = 4;

/// The bytes of one zero write when a sequence clears its planes.
const ZERO_CHUNK: usize = 8 << 20;

static NEXT_SEQ: AtomicU64 = AtomicU64::new(1);

/// The per-step buffers, sized once at load.
pub(crate) struct Scratch {
    pub(crate) h: [Vector; 4],
    pub(crate) nrm: Vector,
    pub(crate) nrm2: Vector,
    pub(crate) q_normed: Vector,
    pub(crate) k_normed: Vector,
    pub(crate) v_normed: Vector,
    pub(crate) ones: Vector,
    pub(crate) q: Vector,
    pub(crate) k: Vector,
    pub(crate) v: Vector,
    pub(crate) ctx_out: Vector,
    pub(crate) proj: Vector,
    pub(crate) gate: Vector,
    pub(crate) up: Vector,
    pub(crate) mixed: Vector,
    pub(crate) down: Vector,
    pub(crate) ple_all: Vector,
    pub(crate) ple_scaled: Vector,
    pub(crate) ple_nrm: Vector,
    pub(crate) ple_tbl: Vector,
    pub(crate) ple_mixed: Vector,
    /// Present only when the attention split is armed: the split partial
    /// buffer, sized by [attn_part_bytes].
    pub(crate) attn_part: Option<DeviceBuffer>,
    pub(crate) q8_hidden: DeviceBuffer,
    pub(crate) q8_ctx: DeviceBuffer,
    pub(crate) q8_mixed: DeviceBuffer,
    pub(crate) q8_ple_mixed: DeviceBuffer,
    pub(crate) q8_head: DeviceBuffer,
    /// The QK-norm dump: rows of quantized heads nobody reads back, sized for a
    /// full prefill batch of every q head at the wider head width.
    pub(crate) q8_qk: DeviceBuffer,
    pub(crate) ids: DeviceBuffer,
    /// The KV8 decode-append status word. The launch contract requires it;
    /// the caller's host range checks make the kernel's refuse path
    /// unreachable, so nothing reads it back.
    pub(crate) status: DeviceBuffer,
}

pub(crate) struct Bolt {
    /// The captured one-token decode body, armed by [`bolt_graph`]. Declared
    /// FIRST so it drops first: fields drop in declaration order, and the
    /// graph must be destroyed before any buffer whose address it baked (a
    /// pin test holds the order). The graph pins nothing (`CapturePlan::pins`
    /// is empty, the flash_next bench pattern): safety is by ownership
    /// instead - no buffer the body touches is reallocated (the step words
    /// are written in place, the scratch sits at its load-time capacity) and a
    /// scratch grow drops the graph before any replacement.
    graph: Option<GraphExec<'static>>,
    /// The sequence whose planes and words the captured body baked.
    graph_seq: u64,
    pub(crate) ctx: CudaContext,
    pub(crate) stream: DeviceStream,
    pub(crate) layers: Vec<BoltLayer>,
    pub(crate) embed: Matrix,
    pub(crate) ple_model: PackedMatrix,
    pub(crate) ple_model_norm: Vector,
    pub(crate) out_norm: Vector,
    pub(crate) rope_freqs: Option<Vector>,
    pub(crate) scratch: Scratch,
    pub(crate) logits: Vector,
    pub(crate) logits_capped: Vector,
    pub(crate) eps: f32,
    /// The sticky decode-graph mode: wanted from the environment, refused on
    /// the first reason, always counted. Every capture refusal pins the mode
    /// to eager; the eager body stays the reference.
    pub(crate) graph_mode: DecodeGraphMode,
    /// The logits rows of a window, present once `reserve_window` ran.
    window: Option<WindowBuffers>,
}

fn embed_scale() -> f32 {
    (HIDDEN as f32).sqrt()
}

/// The device demand of a Bolt load: the weights file bytes, the KV8 cache of
/// the physical layers at `CELLS` positions, the prefill scratch, and the
/// shared headroom reserve. Pure arithmetic, so the admission refusal is
/// testable on the host without a device.
pub(crate) fn bolt_demand(
    weights_bytes: u64,
    attn_split_armed: bool,
) -> Result<crate::kernels::cuda::plan::Demand, EngError> {
    let kv = super::super::budget::bolt_kv_plan(CELLS as u64)
        .ok_or_else(|| EngError::MemoryPlan("the Bolt KV budget overflows".into()))?;
    let kv_bytes = kv
        .positions
        .checked_mul(kv.bytes_per_position)
        .ok_or_else(|| EngError::MemoryPlan("the Bolt KV cache charge overflows".into()))?;
    let scratch = scratch_bytes(attn_split_armed)
        .ok_or_else(|| EngError::MemoryPlan("the Bolt scratch charge overflows".into()))?;
    Ok(crate::kernels::cuda::plan::Demand {
        tensors: weights_bytes,
        kv_cache: kv_bytes,
        work: scratch,
        headroom: super::super::budget::BOLT_HEADROOM_BYTES,
    })
}

pub(crate) fn load(path: &std::path::Path, header_sha256: Option<&str>) -> Result<Bolt, EngError> {
    let bytes = std::fs::metadata(path)
        .map_err(|error| EngError::MemoryPlan(format!("the Bolt weights file: {error}")))?
        .len();
    load_choosing(
        &[crate::family::gguf_select::GgufFile {
            path: path.to_path_buf(),
            bytes,
        }],
        header_sha256,
    )
}

/// Loads the first of `files` (highest precision first) whose device demand the free memory
/// admits, and prints which file it chose and why when there was a choice.
pub(crate) fn load_choosing(
    files: &[crate::family::gguf_select::GgufFile],
    header_sha256: Option<&str>,
) -> Result<Bolt, EngError> {
    let ctx = CudaContext::with_admission(&crate::family::project::cuda::served_admission())?;
    let stream = ctx.stream()?;
    // Admit the device budget before allocating: weights + KV cache + scratch
    // plus headroom against free device memory, through the shared plan::admit
    // the other resident executors use. An exact fit of the allocations alone
    // is refused; the shared error names the needed and free bytes.
    let free = ctx.memory_info()?.free as u64;
    let split = attn_split_wanted()?;
    let choice = crate::family::gguf_select::choose(files, &|file| {
        bolt_demand(file.bytes, split)
            .and_then(|demand| crate::kernels::cuda::plan::admit(&demand, free))
            .map(|_| ())
            .map_err(|error| error.describe())
    })?;
    if let Some(line) = &choice.line {
        eprintln!("{line}");
    }
    let path = choice.chosen.path.as_path();
    let (weights, spelling) = super::super::open_bolt_weights(path, header_sha256)?;
    let demand = bolt_demand(choice.chosen.bytes, split)?;
    crate::kernels::cuda::plan::admit(&demand, free)?;
    let mut layers = Vec::with_capacity(LAYERS);
    // The canonical spelling's merged per_layer_token_embd is read once and
    // sliced per layer below; the legacy spelling reads each layer's own
    // table instead.
    let merged = match spelling.ple_table(0) {
        Some(_) => None,
        None => {
            let name = spelling.per_layer_token_embd().ok_or_else(|| {
                EngError::Checkpoint(
                    "gguf model: the canonical spelling carries the merged table".to_string(),
                )
            })?;
            let (ty, bytes) = weights.read_tensor(name)?;
            if ty != GgufType::Q80 {
                return Err(EngError::Checkpoint(format!(
                    "gguf model: tensor {name} is {}, expected Q8_0",
                    ty.name()
                )));
            }
            Some(bytes)
        }
    };
    for layer in 0..LAYERS {
        let global = NEMOTRON_BOLT_E4B.layers_attention[layer] == AttentionKind::Full;
        let head = if global { GLOBAL_HEAD } else { LOCAL_HEAD };
        let physical = layer < PHYSICAL_LAYERS;
        // A shared layer attends into the cache of the last physical layer of its own kind.
        let kv_source = if physical {
            None
        } else {
            Some(NEMOTRON_BOLT_E4B.shared_kv_source(layer as u32)? as usize)
        };
        let base = format!("blk.{layer}.");
        layers.push(BoltLayer {
            attn_norm: vector(&weights, &ctx, &format!("{base}attn_norm.weight"), HIDDEN)?,
            q_norm: vector(&weights, &ctx, &format!("{base}attn_q_norm.weight"), head)?,
            k_norm: if physical {
                Some(vector(
                    &weights,
                    &ctx,
                    &format!("{base}attn_k_norm.weight"),
                    head,
                )?)
            } else {
                None
            },
            q: packed_matrix(
                &weights,
                &ctx,
                &format!("{base}attn_q.weight"),
                HIDDEN,
                HEADS * head,
            )?,
            k: if physical {
                Some(packed_matrix(
                    &weights,
                    &ctx,
                    &format!("{base}attn_k.weight"),
                    HIDDEN,
                    KV_HEADS * head,
                )?)
            } else {
                None
            },
            v: if physical {
                Some(packed_matrix(
                    &weights,
                    &ctx,
                    &format!("{base}attn_v.weight"),
                    HIDDEN,
                    KV_HEADS * head,
                )?)
            } else {
                None
            },
            out: packed_matrix(
                &weights,
                &ctx,
                &format!("{base}attn_output.weight"),
                HEADS * head,
                HIDDEN,
            )?,
            post_attn_norm: vector(
                &weights,
                &ctx,
                &format!("{base}post_attention_norm.weight"),
                HIDDEN,
            )?,
            ffn_norm: vector(&weights, &ctx, &format!("{base}ffn_norm.weight"), HIDDEN)?,
            ffn_gate: packed_matrix(
                &weights,
                &ctx,
                &format!("{base}ffn_gate.weight"),
                HIDDEN,
                FFN,
            )?,
            ffn_up: packed_matrix(&weights, &ctx, &format!("{base}ffn_up.weight"), HIDDEN, FFN)?,
            ffn_down: packed_matrix(
                &weights,
                &ctx,
                &format!("{base}ffn_down.weight"),
                FFN,
                HIDDEN,
            )?,
            post_ffw_norm: vector(
                &weights,
                &ctx,
                &format!("{base}post_ffw_norm.weight"),
                HIDDEN,
            )?,
            ple_gate: packed_matrix(
                &weights,
                &ctx,
                &format!("{base}{}", spelling.ple_gate()),
                HIDDEN,
                PLE,
            )?,
            ple_proj: packed_matrix(
                &weights,
                &ctx,
                &format!("{base}{}", spelling.ple_proj()),
                PLE,
                HIDDEN,
            )?,
            post_ple_norm: vector(
                &weights,
                &ctx,
                &format!("{base}{}", spelling.ple_post_norm()),
                HIDDEN,
            )?,
            table: match spelling.ple_table(layer) {
                Some(name) => matrix(&weights, &ctx, &name, PLE, VOCAB)?,
                None => {
                    let bytes = ple_rows_from_merged(
                        merged.as_deref().ok_or_else(|| {
                            EngError::Checkpoint(
                                "gguf model: the merged table was hoisted for the canonical spelling"
                                    .to_string(),
                            )
                        })?,
                        layer,
                        PLE,
                        LAYERS,
                        VOCAB,
                    );
                    Matrix {
                        buf: ctx.upload(&bytes)?,
                        rows: VOCAB,
                        k: PLE,
                    }
                }
            },
            scale: scalar(&weights, &format!("{base}layer_output_scale.weight"))?,
            global,
            head,
            kv_source,
        });
    }
    let embed = matrix(&weights, &ctx, "token_embd.weight", HIDDEN, VOCAB)?;
    let ple_model = packed_matrix(
        &weights,
        &ctx,
        spelling.ple_model_proj(),
        HIDDEN,
        LAYERS * PLE,
    )?;
    let ple_model_norm = vector(&weights, &ctx, spelling.ple_proj_norm(), PLE)?;
    let out_norm = vector(&weights, &ctx, "output_norm.weight", HIDDEN)?;
    let rope_freqs = Some(vector(&weights, &ctx, "rope_freqs.weight", 256)?);
    let eps = epsilon(&weights)?;
    let h = [
        vector_filled(&ctx, HIDDEN, 0.0)?,
        vector_filled(&ctx, HIDDEN, 0.0)?,
        vector_filled(&ctx, HIDDEN, 0.0)?,
        vector_filled(&ctx, HIDDEN, 0.0)?,
    ];
    // The attention split refuses a bad switch value before any allocation.
    let attn_split = attn_split_wanted()?;
    let scratch = Scratch {
        h,
        nrm: vector_filled(&ctx, HIDDEN, 0.0)?,
        nrm2: vector_filled(&ctx, HIDDEN, 0.0)?,
        q_normed: vector_filled(&ctx, HEADS * GLOBAL_HEAD, 0.0)?,
        k_normed: vector_filled(&ctx, KV_HEADS * GLOBAL_HEAD, 0.0)?,
        v_normed: vector_filled(&ctx, KV_HEADS * GLOBAL_HEAD, 0.0)?,
        ones: vector_filled(&ctx, GLOBAL_HEAD, 1.0)?,
        q: vector_filled(&ctx, HEADS * GLOBAL_HEAD, 0.0)?,
        k: vector_filled(&ctx, KV_HEADS * GLOBAL_HEAD, 0.0)?,
        v: vector_filled(&ctx, KV_HEADS * GLOBAL_HEAD, 0.0)?,
        ctx_out: vector_filled(&ctx, HEADS * GLOBAL_HEAD, 0.0)?,
        proj: vector_filled(&ctx, HIDDEN, 0.0)?,
        gate: vector_filled(&ctx, FFN, 0.0)?,
        up: vector_filled(&ctx, FFN, 0.0)?,
        mixed: vector_filled(&ctx, FFN, 0.0)?,
        down: vector_filled(&ctx, HIDDEN, 0.0)?,
        ple_all: vector_filled(&ctx, LAYERS * PLE, 0.0)?,
        ple_scaled: vector_filled(&ctx, LAYERS * PLE, 0.0)?,
        ple_nrm: vector_filled(&ctx, PLE, 0.0)?,
        ple_tbl: vector_filled(&ctx, PLE, 0.0)?,
        ple_mixed: vector_filled(&ctx, PLE, 0.0)?,
        q8_hidden: zero(&ctx, q8_1_bytes(HIDDEN, 1).expect("hidden q8 size"))?,
        q8_ctx: zero(
            &ctx,
            q8_1_bytes(HEADS * GLOBAL_HEAD, 1).expect("context q8 size"),
        )?,
        q8_mixed: zero(&ctx, q8_1_bytes(FFN, 1).expect("mixed q8 size"))?,
        q8_ple_mixed: zero(&ctx, q8_1_bytes(PLE, 1).expect("ple mixed q8 size"))?,
        q8_head: zero(&ctx, q8_1_bytes(GLOBAL_HEAD, 1).expect("head q8 size"))?,
        q8_qk: zero(
            &ctx,
            q8_1_bytes(GLOBAL_HEAD, Bolt::PREFILL_BATCH * HEADS).expect("qk dump size"),
        )?,
        ids: zero(&ctx, Bolt::PREFILL_BATCH * 4)?,
        status: zero(&ctx, 4)?,
        attn_part: if attn_split {
            Some(zero(&ctx, attn_part_bytes())?)
        } else {
            None
        },
    };
    let logits = vector_filled(&ctx, VOCAB, 0.0)?;
    let logits_capped = vector_filled(&ctx, VOCAB, 0.0)?;
    Ok(Bolt {
        ctx,
        stream,
        layers,
        embed,
        ple_model,
        ple_model_norm,
        out_norm,
        rope_freqs,
        scratch,
        logits,
        logits_capped,
        eps,
        graph_mode: DecodeGraphMode::new(decode_graph_wanted()?),
        graph: None,
        graph_seq: 0,
        window: None,
    })
}

impl Bolt {
    pub(crate) const PREFILL_BATCH: usize = 8;

    /// Allocates the device state of one more sequence: zeroed KV8 planes for the physical
    /// layers at `CELLS` positions and its two position words. The planes are read only below
    /// the position word, so a sequence that is reset needs no clearing.
    pub(crate) fn new_seq(&self) -> Result<BoltSeq, EngError> {
        let ctx = &self.ctx;
        // One zero chunk on the device, copied down every plane: the planes are cleared at
        // device speed and no host memory the size of a plane is touched.
        let zeros = zero(ctx, ZERO_CHUNK)?;
        let plane = |bytes: usize| -> Result<DeviceBuffer, EngError> {
            let buffer = ctx.allocate(bytes)?;
            let mut at = 0;
            while at < bytes {
                let len = ZERO_CHUNK.min(bytes - at);
                ctx.copy_device(&self.stream, zeros.range(0, len)?, buffer.range(at, len)?)?;
                at += len;
            }
            Ok(buffer)
        };
        let mut kv = Vec::with_capacity(LAYERS);
        for layer in &self.layers {
            let head = layer.head;
            kv.push(if layer.physical() {
                Some(Kv {
                    k_codes: plane(KV_HEADS * CELLS * head)?,
                    k_scales: plane(KV_HEADS * CELLS * (head / 32) * 2)?,
                    v_codes: plane(KV_HEADS * CELLS * head)?,
                    v_scales: plane(KV_HEADS * CELLS * (head / 32) * 2)?,
                    shape: Kv8Shape {
                        rows: 1,
                        n_kv_head: KV_HEADS,
                        head_dim: head,
                        cells: CELLS,
                        code_scale: Kv8CodeScale::UnroundedF32,
                    },
                })
            } else {
                None
            });
        }
        let seq = BoltSeq {
            id: NEXT_SEQ.fetch_add(1, Ordering::Relaxed),
            kv,
            pos: zero(ctx, 4)?,
            kv_len: zero(ctx, 4)?,
        };
        // The copies read `zeros`, which drops here.
        self.stream.synchronize()?;
        Ok(seq)
    }

    /// Allocates the logits rows that a window of up to `rows` sequences needs, once. A request
    /// for fewer rows than are held keeps the buffers.
    pub(crate) fn reserve_window(&mut self, rows: usize) -> Result<(), EngError> {
        if !(2..=Self::PREFILL_BATCH).contains(&rows) {
            return Err(EngError::BadRequest(format!(
                "a Bolt window holds 2 to {} rows, got {rows}",
                Self::PREFILL_BATCH
            )));
        }
        if self.window.as_ref().is_some_and(|held| held.rows >= rows) {
            return Ok(());
        }
        self.window = Some(WindowBuffers {
            rows,
            logits: vector_filled(&self.ctx, rows * VOCAB, 0.0)?,
            capped: vector_filled(&self.ctx, rows * VOCAB, 0.0)?,
        });
        Ok(())
    }

    /// Steps one token of each sequence together and returns the capped logits row of each, in
    /// order. The projections, norms, feed-forward and per-layer-input blocks run once over all
    /// rows; rotation, cache append and attention run once per sequence against its own planes
    /// and words, which is the launch a lone sequence makes. The head runs in launches of at
    /// most [`HEAD_COLUMNS`] columns. `positions[i]` is the cell the token of `seqs[i]` is
    /// written to.
    pub(crate) fn decode_rows(
        &mut self,
        seqs: &[&BoltSeq],
        tokens: &[u32],
        positions: &[usize],
    ) -> Result<Vec<Vec<f32>>, EngError> {
        let m = seqs.len();
        if m != tokens.len() || m != positions.len() {
            return Err(EngError::BadRequest(
                "a Bolt window needs one token and one position per sequence".to_owned(),
            ));
        }
        if !(2..=Self::PREFILL_BATCH).contains(&m) {
            return Err(EngError::BadRequest(format!(
                "a Bolt window holds 2 to {} rows, got {m}",
                Self::PREFILL_BATCH
            )));
        }
        if self.window.as_ref().is_none_or(|held| held.rows < m) {
            return Err(EngError::BadRequest(
                "the Bolt window buffers are not reserved for this many rows".to_owned(),
            ));
        }
        if positions.iter().any(|&pos| pos >= CELLS) {
            return Err(EngError::ContextLimit(
                "the Bolt decode window exceeds the context".to_owned(),
            ));
        }
        self.grow_scratch(m)?;
        let ids: Vec<u8> = tokens.iter().flat_map(|t| t.to_le_bytes()).collect();
        self.scratch.ids.write_at_enqueue(&self.stream, 0, &ids)?;
        for (seq, &pos) in seqs.iter().zip(positions) {
            seq.pos
                .write_at_enqueue(&self.stream, 0, &(pos as i32).to_le_bytes())?;
            seq.kv_len
                .write_at_enqueue(&self.stream, 0, &((pos as i32) + 1).to_le_bytes())?;
        }
        self.decode_body(&Rows::Window { seqs }, true, m)?;
        self.graph_mode.note_eager();
        let window = self.window.as_ref().expect("checked above");
        let mut bytes = vec![0u8; m * VOCAB * 4];
        window.capped.buf.read_at(&self.stream, 0, &mut bytes)?;
        Ok(bytes
            .as_chunks::<{ VOCAB * 4 }>()
            .0
            .iter()
            .map(|row| {
                row.as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| f32::from_le_bytes(*b))
                    .collect()
            })
            .collect())
    }

    /// Arms or disarms the decode graph for this model (the environment
    /// sets the load-time default; the tests flip it between runs).
    pub(crate) fn use_decode_graph(&mut self, on: bool) {
        self.graph_mode.set_on(on);
    }

    /// Arms or disarms the split decode attention for this model. The
    /// environment sets the load-time default; the qualification tests flip
    /// it between runs. Arming outside capture allocates the partial buffer
    /// on first use (the env-armed load already counted it in the device
    /// budget; this post-load path is test and qualification traffic). Any
    /// captured decode graph is dropped: it was captured against the previous
    /// split state, and its replay would read a freed partial buffer when the
    /// split is disarmed (or never split at all when armed after capture).
    pub(crate) fn use_attn_split(&mut self, on: bool) -> Result<(), EngError> {
        self.ctx.ensure_usable()?;
        self.ctx.ensure_uncaptured("attn split partial buffer")?;
        self.graph = None;
        if on {
            if self.scratch.attn_part.is_none() {
                self.scratch.attn_part = Some(zero(&self.ctx, attn_part_bytes())?);
            }
        } else {
            self.scratch.attn_part = None;
        }
        Ok(())
    }

    /// Whether a captured decode graph is held (the split invalidation pin).
    pub(crate) fn graph_captured(&self) -> bool {
        self.graph.is_some()
    }

    /// Whether the attention split is armed: the partial buffer exists. The
    /// disarm assert pins on this, not on `graph_captured`, which the disarm
    /// does not change (arming already dropped any graph).
    pub(crate) fn split_armed(&self) -> bool {
        self.scratch.attn_part.is_some()
    }

    /// The capped logits of the newest logits-bearing forward, for host
    /// sampling. The tie to `logits_capped` (not `logits`) matches the
    /// model's own output distribution.
    pub(crate) fn capped_logits(&self) -> Result<Vec<f32>, EngError> {
        let mut bytes = vec![0u8; VOCAB * 4];
        self.ctx.download(&self.logits_capped.buf, &mut bytes)?;
        Ok(bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect())
    }

    /// Debug parity hook: with BOLT_TRACE_DIR set, one raw f32 vector lands per
    /// checkpoint (embed, each layer's output) as p{pos}_{name}.f32. No-op
    /// without the variable; a parity comparison reads the dumps.
    fn trace(&self, pos: usize, name: &str, v: &Vector, bytes: usize) {
        let Some(dir) = trace_dir() else {
            return;
        };
        let mut raw = vec![0u8; bytes];
        if self.ctx.download(&v.buf, &mut raw).is_ok() {
            let _ = std::fs::write(
                std::path::Path::new(dir).join(format!("p{pos}_{name}.f32")),
                raw,
            );
        }
    }

    /// Runs a bounded prompt batch. Only its last row can produce logits.
    /// Scratch grows once; each batch synchronizes once at the end when no
    /// logits are drawn.
    pub(crate) fn prefill_batch(
        &mut self,
        seq: &BoltSeq,
        tokens: &[u32],
        pos: usize,
        logits: bool,
    ) -> Result<(), EngError> {
        if !(1..=Self::PREFILL_BATCH).contains(&tokens.len()) {
            return Err(EngError::BadRequest(
                "a Bolt prefill batch needs 1 to 8 tokens".to_owned(),
            ));
        }
        if pos.checked_add(tokens.len()).is_none_or(|end| end > CELLS) {
            return Err(EngError::ContextLimit(
                "the Bolt prefill batch exceeds the context".to_owned(),
            ));
        }
        self.grow_scratch(tokens.len())?;
        self.decode_with_logits(seq, tokens, pos, logits)?;
        if !logits {
            self.stream.synchronize()?;
        }
        Ok(())
    }

    fn grow_scratch(&mut self, rows: usize) -> Result<(), EngError> {
        // A buffer replacement invalidates a captured body (it baked the old
        // address), so the graph drops when a replacement actually happens -
        // not on every call, or no prefill would ever leave the graph armed.
        // (The load-time capacity already covers every PREFILL_BATCH chunk, so
        // nothing replaces in practice.)
        let replaced = self.grow_scratch_buffers(rows)?;
        if replaced {
            self.graph = None;
        }
        Ok(())
    }

    /// Grows every scratch buffer whose capacity is below `rows` up to the
    /// PREFILL_BATCH capacity, reporting whether any buffer was replaced.
    fn grow_scratch_buffers(&mut self, rows: usize) -> Result<bool, EngError> {
        let mut replaced = false;
        let sc = &mut self.scratch;
        for (v, width) in sc.h.iter_mut().map(|v| (v, HIDDEN)).chain([
            (&mut sc.nrm, HIDDEN),
            (&mut sc.nrm2, HIDDEN),
            (&mut sc.q_normed, HEADS * GLOBAL_HEAD),
            (&mut sc.k_normed, KV_HEADS * GLOBAL_HEAD),
            (&mut sc.v_normed, KV_HEADS * GLOBAL_HEAD),
            (&mut sc.q, HEADS * GLOBAL_HEAD),
            (&mut sc.k, KV_HEADS * GLOBAL_HEAD),
            (&mut sc.v, KV_HEADS * GLOBAL_HEAD),
            (&mut sc.ctx_out, HEADS * GLOBAL_HEAD),
            (&mut sc.proj, HIDDEN),
            (&mut sc.gate, FFN),
            (&mut sc.up, FFN),
            (&mut sc.mixed, FFN),
            (&mut sc.down, HIDDEN),
            (&mut sc.ple_all, LAYERS * PLE),
            (&mut sc.ple_scaled, LAYERS * PLE),
            (&mut sc.ple_nrm, PLE),
            (&mut sc.ple_tbl, PLE),
            (&mut sc.ple_mixed, PLE),
        ]) {
            if v.len < rows * width {
                let len = Self::PREFILL_BATCH * width;
                v.buf = self.ctx.allocate(len * 4)?;
                v.len = len;
                replaced = true;
            }
        }
        for (buffer, width) in [
            (&mut sc.q8_hidden, HIDDEN),
            (&mut sc.q8_ctx, HEADS * GLOBAL_HEAD),
            (&mut sc.q8_mixed, FFN),
            (&mut sc.q8_ple_mixed, PLE),
        ] {
            if buffer.bytes() < q8_1_bytes(width, rows).expect("bounded q8 rows") {
                *buffer = self
                    .ctx
                    .allocate(q8_1_bytes(width, Self::PREFILL_BATCH).expect("bounded q8 rows"))?;
                replaced = true;
            }
        }
        Ok(replaced)
    }

    /// Keeps the existing norm reduction and quantization order for every activation row.
    fn norm_rows<'b>(
        &self,
        stream: &DeviceStream,
        h: DeviceRange<'b>,
        weight: DeviceRange<'b>,
        norm_out: DeviceRange<'b>,
        q8: DeviceRange<'b>,
        rows: usize,
    ) -> Result<(), EngError> {
        self.ctx.add_rmsnorm_q8(
            stream,
            AddNormQ8Call {
                n: HIDDEN,
                eps: self.eps,
                h,
                delta: None,
                weight,
                norm_out,
                q8,
                rows,
            },
        )
    }

    /// Runs one decode step for the token at the position and leaves the capped
    /// logits in the device buffer. Synchronizes once at the end.
    pub(crate) fn decode(&mut self, seq: &BoltSeq, token: u32, pos: usize) -> Result<(), EngError> {
        self.decode_with_logits(seq, &[token], pos, true)
    }

    /// Runs one step without the output norm and the logits projection, for prompt tokens
    /// whose logits are never read. It queues the step and does not synchronize.
    /// The shared-KV layers are skipped: they write no cache cells, and their
    /// hidden state only feeds the final norm, so a token whose logits are
    /// never read does not need them.
    pub(crate) fn decode_hidden(
        &mut self,
        seq: &BoltSeq,
        token: u32,
        pos: usize,
    ) -> Result<(), EngError> {
        self.decode_with_logits(seq, &[token], pos, false)
    }

    fn decode_with_logits(
        &mut self,
        seq: &BoltSeq,
        tokens: &[u32],
        pos: usize,
        logits: bool,
    ) -> Result<(), EngError> {
        let m = tokens.len();
        if pos.checked_add(m).is_none_or(|end| end > CELLS) {
            return Err(EngError::ContextLimit(
                "the Bolt decode step exceeds the context".to_owned(),
            ));
        }
        let ids: Vec<u8> = tokens.iter().flat_map(|t| t.to_le_bytes()).collect();
        // The step words are written in place, never reallocated: a captured
        // decode graph bakes their addresses, and the eager path saves the
        // three per-step allocations the uploads made.
        self.scratch.ids.write_at_enqueue(&self.stream, 0, &ids)?;
        seq.pos
            .write_at_enqueue(&self.stream, 0, &(pos as i32).to_le_bytes())?;
        seq.kv_len
            .write_at_enqueue(&self.stream, 0, &((pos as i32) + 1).to_le_bytes())?;
        let rows = Rows::Chunk { seq, pos };
        let facts = DecodeGraphFacts {
            trace_armed: trace_dir().is_some(),
            multi_token: m > 1,
            hidden: !logits,
        };
        if check_decode_graph(facts).is_ok() {
            // The mode gates the replay too: a disarmed (or refused) mode must
            // not keep replaying a graph it captured earlier.
            if let Some(exec) = self
                .graph
                .as_ref()
                .filter(|_| self.graph_mode.active() && self.graph_seq == seq.id)
            {
                exec.launch(&self.stream)?;
                self.graph_mode.note_replay();
                return self.stream.synchronize();
            }
            if self.graph_mode.active() && self.graph_mode.warm() {
                match self.capture_decode_graph(seq, pos) {
                    Ok(()) => {
                        self.graph_mode.note_replay();
                        return self.stream.synchronize();
                    }
                    Err(failure) if is_capture_refusal(&failure) => {
                        // The body never ran; the sticky refusal makes every
                        // later step eager, and this one runs below.
                        self.graph_mode.refuse(&failure.describe());
                    }
                    Err(failure) => return Err(failure),
                }
            }
        }
        self.decode_body(&rows, logits, m)?;
        self.graph_mode.note_eager();
        if check_decode_graph(facts).is_ok() {
            // One clean eager step warms every kernel the capture records
            // (a first launch loads its module, which a capture forbids).
            self.graph_mode.note_warm();
        }
        self.stream.synchronize()
    }

    /// The launches of one decode step, from the embedding to the capped
    /// logits: everything a replay re-runs, and the body a capture records.
    /// No per-step scalar but the trace label's position reaches the kernels
    /// through arguments; the kernels that depend on the position or the KV
    /// length read the step words.
    fn decode_body(&self, rows: &Rows<'_>, logits: bool, m: usize) -> Result<(), EngError> {
        let pos = rows.trace_pos();
        let ctx = &self.ctx;
        let stream = &self.stream;
        let sc = &self.scratch;
        let h0 = &sc.h[0];
        let h1 = &sc.h[1];
        let h2 = &sc.h[2];
        let h3 = &sc.h[3];
        ctx.embed_q8(
            stream,
            EmbedQ8Call {
                out: h0.buf.range(0, h0.bytes())?,
                w: self.embed.buf.range(0, self.embed.bytes())?,
                ids: sc.ids.range(0, m * 4)?,
                m,
                n_rows: self.embed.rows,
                k: self.embed.k,
            },
        )?;
        ctx.row_scale(
            stream,
            RowScaleCall {
                out: h1.buf.range(0, h1.bytes())?,
                x: h0.buf.range(0, h0.bytes())?,
                scale: embed_scale(),
                m,
                n: HIDDEN,
            },
        )?;
        // K/V-only prefill: only the cache-writing prefix of layers runs when
        // no logits are read; the shared-KV layers after it neither write cache
        // cells nor feed anything but the final norm.
        let kv_prefix = if logits {
            LAYERS
        } else {
            self.layers
                .iter()
                .position(|layer| !layer.physical())
                .unwrap_or(LAYERS)
        };
        self.per_layer_inputs(h1, m, kv_prefix * PLE)?;
        self.trace(pos, "embed", h1, m * HIDDEN * 4);
        let h = [h0, h1, h2, h3];
        let mut current = 1usize;
        for index in 0..kv_prefix {
            let (input, r1, r2, out) = layer_buffers(current);
            self.decode_layer(
                index,
                h[input],
                h[r1],
                h[r2],
                h[out],
                rows,
                kv_prefix * PLE,
                m,
            )?;
            current = out;
        }
        if !logits {
            // Only the last prompt token contributes logits, so the output norm and the
            // tied-embedding projection are skipped for the rest.
            return Ok(());
        }
        let input = h[current];
        if let Rows::Window { .. } = rows {
            return self.window_head(input, m);
        }
        ctx.add_rmsnorm_q8(
            stream,
            AddNormQ8Call {
                n: HIDDEN,
                eps: self.eps,
                h: input.buf.range((m - 1) * HIDDEN * 4, HIDDEN * 4)?,
                delta: None,
                weight: self.out_norm.buf.range(0, self.out_norm.bytes())?,
                norm_out: sc.nrm.buf.range(0, sc.nrm.bytes())?,
                q8: sc.q8_hidden.range(0, sc.q8_hidden.bytes())?,
                rows: 1,
            },
        )?;
        ctx.matvec_q(
            stream,
            MatvecCall {
                ty: QuantType::Q80,
                w: self.embed.buf.range(0, self.embed.bytes())?,
                x_q8: sc.q8_hidden.range(0, sc.q8_hidden.bytes())?,
                y: self.logits.buf.range(0, self.logits.bytes())?,
                k: HIDDEN,
                rows: self.embed.rows,
                ncols: 1,
            },
        )?;
        ctx.softcap(
            stream,
            SoftcapCall {
                out: self
                    .logits_capped
                    .buf
                    .range(0, self.logits_capped.bytes())?,
                x: self.logits.buf.range(0, self.logits.bytes())?,
                cap: SOFT_CAP,
                n: VOCAB,
            },
        )?;
        self.trace(pos, "logits", &self.logits_capped, VOCAB * 4);
        Ok(())
    }

    /// Captures the one-token logits body and replays it once: a capture
    /// records the work without running it, so this step's token is produced
    /// by the first replay. A refusal propagates to the caller, which notes it
    /// sticky and runs the body eagerly.
    fn capture_decode_graph(&mut self, seq: &BoltSeq, pos: usize) -> Result<(), EngError> {
        let plan = CapturePlan {
            pins: &[],
            holds: Vec::new(),
        };
        let rows = Rows::Chunk { seq, pos };
        let (exec, ()) = self
            .ctx
            .capture(&self.stream, plan, || self.decode_body(&rows, true, 1))?;
        exec.launch(&self.stream)?;
        self.graph = Some(exec);
        self.graph_seq = seq.id;
        Ok(())
    }

    /// The head of a window: the output norm over every row, the tied-embedding projection in
    /// launches of at most [`HEAD_COLUMNS`] columns, and the softcap over all rows.
    fn window_head(&self, input: &Vector, m: usize) -> Result<(), EngError> {
        let ctx = &self.ctx;
        let stream = &self.stream;
        let sc = &self.scratch;
        let window = self
            .window
            .as_ref()
            .ok_or_else(|| EngError::BadRequest("the Bolt window buffers are absent".to_owned()))?;
        ctx.add_rmsnorm_q8(
            stream,
            AddNormQ8Call {
                n: HIDDEN,
                eps: self.eps,
                h: input.buf.range(0, m * HIDDEN * 4)?,
                delta: None,
                weight: self.out_norm.buf.range(0, self.out_norm.bytes())?,
                norm_out: sc.nrm.buf.range(0, m * HIDDEN * 4)?,
                q8: sc.q8_hidden.range(0, sc.q8_hidden.bytes())?,
                rows: m,
            },
        )?;
        let row_bytes = q8_1_bytes(HIDDEN, 1).expect("hidden q8 row size");
        for first in (0..m).step_by(HEAD_COLUMNS) {
            let columns = HEAD_COLUMNS.min(m - first);
            ctx.matvec_q(
                stream,
                MatvecCall {
                    ty: QuantType::Q80,
                    w: self.embed.buf.range(0, self.embed.bytes())?,
                    x_q8: sc.q8_hidden.range(first * row_bytes, columns * row_bytes)?,
                    y: window
                        .logits
                        .buf
                        .range(first * VOCAB * 4, columns * VOCAB * 4)?,
                    k: HIDDEN,
                    rows: self.embed.rows,
                    ncols: columns,
                },
            )?;
        }
        ctx.softcap(
            stream,
            SoftcapCall {
                out: window.capped.buf.range(0, m * VOCAB * 4)?,
                x: window.logits.buf.range(0, m * VOCAB * 4)?,
                cap: SOFT_CAP,
                n: m * VOCAB,
            },
        )
    }

    /// Projects the scaled embedding into the per-layer planes and scales by
    /// hidden^-0.5; the per-layer 256-wide norms and table sums run inside each
    /// layer, where the result is consumed.
    fn per_layer_inputs(&self, embed: &Vector, m: usize, ple_rows: usize) -> Result<(), EngError> {
        let ctx = &self.ctx;
        let stream = &self.stream;
        let sc = &self.scratch;
        ctx.quantize_q8_1(
            stream,
            QuantizeCall {
                x: embed.buf.range(0, m * HIDDEN * 4)?,
                y: sc.q8_hidden.range(0, sc.q8_hidden.bytes())?,
                k: HIDDEN,
                ncols: m,
            },
        )?;
        ctx.gemma_matvec(
            stream,
            GemmaMatvecCall {
                w_qs: self.ple_model.qs.range(0, self.ple_model.qs_bytes())?,
                w_scales: self
                    .ple_model
                    .scales
                    .range(0, self.ple_model.scales_bytes())?,
                x_q8: sc.q8_hidden.range(0, sc.q8_hidden.bytes())?,
                y: sc.ple_all.buf.range(0, m * ple_rows * 4)?,
                k: HIDDEN,
                rows: ple_rows,
                ncols: m,
            },
        )?;
        ctx.row_scale(
            stream,
            RowScaleCall {
                out: sc.ple_scaled.buf.range(0, sc.ple_scaled.bytes())?,
                x: sc.ple_all.buf.range(0, sc.ple_all.bytes())?,
                scale: (HIDDEN as f32).recip().sqrt(),
                m,
                n: ple_rows,
            },
        )
    }

    /// Rotation, cache append and attention of one layer. A chunk runs them once over its
    /// consecutive rows against its sequence; a window runs them once per row against that
    /// row's sequence, with the launch a lone row makes.
    fn rope_append_attend(&self, index: usize, rows: &Rows<'_>, m: usize) -> Result<(), EngError> {
        let layer = &self.layers[index];
        let ctx = &self.ctx;
        let stream = &self.stream;
        let sc = &self.scratch;
        let head = layer.head;
        let q_rows = HEADS * head;
        let kv_rows = KV_HEADS * head;
        let physical = layer.physical();
        let source = layer.kv_source.unwrap_or(index);
        let pos = rows.trace_pos();
        let attn_split = sc.attn_part.as_ref().map(|part| {
            let bytes = attn_part_bytes();
            AttnDecodeGemmaSplit {
                // Sized at load for the full prefill batch; the range cannot fail.
                part: part.range(0, bytes).expect("partial buffer"),
                max_splits: ATTN_SPLIT_MAX,
                min_keys: ATTN_SPLIT_MIN_KEYS,
            }
        });
        // `count` consecutive rows of `seq` starting at row `row` of the scratch, whose first
        // row sits at cell `first`.
        let group =
            |seq: &BoltSeq, row: usize, count: usize, first: usize| -> Result<(), EngError> {
                let own = seq.kv[source]
                    .as_ref()
                    .expect("the shared-KV source layer is physical");
                // A shared layer rotates its query only; the rope kernel requires at least one k
                // row, so it rotates one scratch row that no later launch reads.
                ctx.rope_gemma(
                    stream,
                    RopeGemmaCall {
                        q: sc
                            .q_normed
                            .buf
                            .range(row * q_rows * 4, count * q_rows * 4)?,
                        k: if physical {
                            sc.k_normed
                                .buf
                                .range(row * kv_rows * 4, count * kv_rows * 4)?
                        } else {
                            sc.k_normed.buf.range(0, kv_rows * 4)?
                        },
                        pos: seq.pos.range(0, 4)?,
                        factors: match (&self.rope_freqs, layer.global) {
                            (Some(freqs), true) => Some(freqs.buf.range(0, freqs.bytes())?),
                            _ => None,
                        },
                        rows_q: count,
                        rows_k: if physical { count } else { 1 },
                        n_q_head: HEADS,
                        n_kv_head: KV_HEADS,
                        head_dim: head,
                        global_mode: layer.global,
                    },
                )?;
                if physical {
                    let data = Kv8Data {
                        k: sc
                            .k_normed
                            .buf
                            .range(row * kv_rows * 4, count * kv_rows * 4)?,
                        v: sc
                            .v_normed
                            .buf
                            .range(row * kv_rows * 4, count * kv_rows * 4)?,
                        k_codes: own.k_codes.range(0, own.k_codes.bytes())?,
                        k_scales: own.k_scales.range(0, own.k_scales.bytes())?,
                        v_codes: own.v_codes.range(0, own.v_codes.bytes())?,
                        v_scales: own.v_scales.range(0, own.v_scales.bytes())?,
                    };
                    if count == 1 {
                        // The decode entry: the device pos word is authoritative, so a captured
                        // replay writes the cell of the step it replays instead of the
                        // capture-time cell. The status word satisfies the launch contract; the
                        // caller's host check that `pos + m` leaves the context makes the
                        // kernel's refuse path unreachable, so the word is never read back.
                        ctx.kv8_append(
                            stream,
                            Kv8Shape {
                                rows: 1,
                                ..own.shape
                            },
                            data,
                            seq.pos.range(0, 4)?,
                            sc.status.range(0, 4)?,
                        )?;
                    } else {
                        // The prefill chunk entry takes its position as a host scalar; it is never
                        // captured (a multi-token step refuses arming), so the baked-scalar hazard
                        // does not arise.
                        ctx.kv8_chunk(
                            stream,
                            Kv8Shape {
                                rows: count,
                                ..own.shape
                            },
                            data,
                            first,
                        )?;
                    }
                }
                self.trace(
                    pos,
                    &format!("L{index}_v"),
                    &sc.v_normed,
                    (row + count) * kv_rows * 4,
                );
                ctx.attn_decode_gemma(
                    stream,
                    AttnDecodeGemmaCall {
                        out: sc.ctx_out.buf.range(row * q_rows * 4, count * q_rows * 4)?,
                        status: None,
                        q: sc
                            .q_normed
                            .buf
                            .range(row * q_rows * 4, count * q_rows * 4)?,
                        k_codes: own.k_codes.range(0, own.k_codes.bytes())?,
                        k_scales: own.k_scales.range(0, own.k_scales.bytes())?,
                        v_codes: own.v_codes.range(0, own.v_codes.bytes())?,
                        v_scales: own.v_scales.range(0, own.v_scales.bytes())?,
                        kv_len: seq.kv_len.range(0, 4)?,
                        m: count,
                        n_q_head: HEADS,
                        n_kv_head: KV_HEADS,
                        head_dim: head,
                        cells: CELLS,
                        window: if layer.global { 0 } else { LOCAL_WINDOW },
                        scale: 1.0,
                        split: attn_split,
                    },
                )
            };
        match rows {
            Rows::Chunk { seq, pos } => group(seq, 0, m, *pos),
            Rows::Window { seqs } => seqs
                .iter()
                .enumerate()
                .try_for_each(|(row, seq)| group(seq, row, 1, 0)),
        }
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "the residual rotation keeps four separate buffers"
    )]
    fn decode_layer(
        &self,
        index: usize,
        input: &Vector,
        r1: &Vector,
        r2: &Vector,
        out: &Vector,
        rows: &Rows<'_>,
        ple_stride: usize,
        m: usize,
    ) -> Result<(), EngError> {
        let pos = rows.trace_pos();
        let layer = &self.layers[index];
        let ctx = &self.ctx;
        let stream = &self.stream;
        let sc = &self.scratch;
        let head = layer.head;
        let q_rows = HEADS * head;
        let kv_rows = KV_HEADS * head;
        let shared = !layer.physical();
        self.norm_rows(
            stream,
            input.buf.range(0, input.bytes())?,
            layer.attn_norm.buf.range(0, layer.attn_norm.bytes())?,
            sc.nrm.buf.range(0, sc.nrm.bytes())?,
            sc.q8_hidden.range(0, sc.q8_hidden.bytes())?,
            m,
        )?;
        ctx.gemma_matvec(
            stream,
            GemmaMatvecCall {
                w_qs: layer.q.qs.range(0, layer.q.qs_bytes())?,
                w_scales: layer.q.scales.range(0, layer.q.scales_bytes())?,
                x_q8: sc.q8_hidden.range(0, sc.q8_hidden.bytes())?,
                y: sc.q.buf.range(0, m * q_rows * 4)?,
                k: HIDDEN,
                rows: q_rows,
                ncols: m,
            },
        )?;
        if let (Some(k), Some(v)) = (&layer.k, &layer.v) {
            for (weight, y) in [(k, &sc.k), (v, &sc.v)] {
                ctx.gemma_matvec(
                    stream,
                    GemmaMatvecCall {
                        w_qs: weight.qs.range(0, weight.qs_bytes())?,
                        w_scales: weight.scales.range(0, weight.scales_bytes())?,
                        x_q8: sc.q8_hidden.range(0, sc.q8_hidden.bytes())?,
                        y: y.buf.range(0, m * kv_rows * 4)?,
                        k: HIDDEN,
                        rows: kv_rows,
                        ncols: m,
                    },
                )?;
            }
        }
        // The QK-norm of the reference graph: each head of q and of k is
        // RMS-normalized and scaled by its own weight before RoPE. Values get the
        // reference's unweighted per-head RMS norm before their cache write; the
        // ones vector makes the weighted kernel compute the unweighted norm. The
        // head rows are contiguous, so each loop is ONE batched launch; the q8
        // side of the kernel is a dump nobody reads.
        ctx.add_rmsnorm_q8(
            stream,
            AddNormQ8Call {
                n: head,
                eps: self.eps,
                h: sc.q.buf.range(0, m * q_rows * 4)?,
                delta: None,
                weight: layer.q_norm.buf.range(0, head * 4)?,
                norm_out: sc.q_normed.buf.range(0, m * q_rows * 4)?,
                q8: sc.q8_qk.range(0, sc.q8_qk.bytes())?,
                rows: m * HEADS,
            },
        )?;
        if !shared {
            ctx.add_rmsnorm_q8(
                stream,
                AddNormQ8Call {
                    n: head,
                    eps: self.eps,
                    h: sc.k.buf.range(0, m * kv_rows * 4)?,
                    delta: None,
                    weight: layer
                        .k_norm
                        .as_ref()
                        .expect("physical layers carry a k norm")
                        .buf
                        .range(0, head * 4)?,
                    norm_out: sc.k_normed.buf.range(0, m * kv_rows * 4)?,
                    q8: sc.q8_qk.range(0, sc.q8_qk.bytes())?,
                    rows: m * KV_HEADS,
                },
            )?;
            ctx.add_rmsnorm_q8(
                stream,
                AddNormQ8Call {
                    n: head,
                    eps: self.eps,
                    h: sc.v.buf.range(0, m * kv_rows * 4)?,
                    delta: None,
                    weight: sc.ones.buf.range(0, head * 4)?,
                    norm_out: sc.v_normed.buf.range(0, m * kv_rows * 4)?,
                    q8: sc.q8_qk.range(0, sc.q8_qk.bytes())?,
                    rows: m * KV_HEADS,
                },
            )?;
        }
        self.rope_append_attend(index, rows, m)?;
        self.trace(pos, &format!("L{index}_ctx"), &sc.ctx_out, m * q_rows * 4);
        ctx.quantize_q8_1(
            stream,
            QuantizeCall {
                x: sc.ctx_out.buf.range(0, m * q_rows * 4)?,
                y: sc.q8_ctx.range(0, sc.q8_ctx.bytes())?,
                k: q_rows,
                ncols: m,
            },
        )?;
        ctx.gemma_matvec(
            stream,
            GemmaMatvecCall {
                w_qs: layer.out.qs.range(0, layer.out.qs_bytes())?,
                w_scales: layer.out.scales.range(0, layer.out.scales_bytes())?,
                x_q8: sc.q8_ctx.range(0, sc.q8_ctx.bytes())?,
                y: sc.proj.buf.range(0, m * HIDDEN * 4)?,
                k: q_rows,
                rows: HIDDEN,
                ncols: m,
            },
        )?;
        self.trace(pos, &format!("L{index}_proj"), &sc.proj, m * HIDDEN * 4);
        self.norm_rows(
            stream,
            sc.proj.buf.range(0, m * HIDDEN * 4)?,
            layer
                .post_attn_norm
                .buf
                .range(0, layer.post_attn_norm.bytes())?,
            sc.nrm2.buf.range(0, sc.nrm2.bytes())?,
            sc.q8_hidden.range(0, sc.q8_hidden.bytes())?,
            m,
        )?;
        self.trace(pos, &format!("L{index}_nrm"), &sc.nrm2, m * HIDDEN * 4);
        ctx.row_add(
            stream,
            RowAddCall {
                out: r1.buf.range(0, r1.bytes())?,
                x: input.buf.range(0, input.bytes())?,
                y: sc.nrm2.buf.range(0, sc.nrm2.bytes())?,
                n: m * HIDDEN,
                flag: None,
                fused: false,
            },
        )?;
        self.trace(pos, &format!("L{index}_r1"), r1, m * HIDDEN * 4);
        self.norm_rows(
            stream,
            r1.buf.range(0, r1.bytes())?,
            layer.ffn_norm.buf.range(0, layer.ffn_norm.bytes())?,
            sc.nrm.buf.range(0, sc.nrm.bytes())?,
            sc.q8_hidden.range(0, sc.q8_hidden.bytes())?,
            m,
        )?;
        for (weight, y) in [(&layer.ffn_gate, &sc.gate), (&layer.ffn_up, &sc.up)] {
            ctx.gemma_matvec(
                stream,
                GemmaMatvecCall {
                    w_qs: weight.qs.range(0, weight.qs_bytes())?,
                    w_scales: weight.scales.range(0, weight.scales_bytes())?,
                    x_q8: sc.q8_hidden.range(0, sc.q8_hidden.bytes())?,
                    y: y.buf.range(0, m * FFN * 4)?,
                    k: HIDDEN,
                    rows: FFN,
                    ncols: m,
                },
            )?;
        }
        ctx.gelu_ffn(
            stream,
            GeluFfnCall {
                out: sc.mixed.buf.range(0, m * FFN * 4)?,
                gate: sc.gate.buf.range(0, m * FFN * 4)?,
                up: sc.up.buf.range(0, m * FFN * 4)?,
                m,
                n: FFN,
            },
        )?;
        ctx.quantize_q8_1(
            stream,
            QuantizeCall {
                x: sc.mixed.buf.range(0, m * FFN * 4)?,
                y: sc.q8_mixed.range(0, sc.q8_mixed.bytes())?,
                k: FFN,
                ncols: m,
            },
        )?;
        ctx.gemma_matvec(
            stream,
            GemmaMatvecCall {
                w_qs: layer.ffn_down.qs.range(0, layer.ffn_down.qs_bytes())?,
                w_scales: layer
                    .ffn_down
                    .scales
                    .range(0, layer.ffn_down.scales_bytes())?,
                x_q8: sc.q8_mixed.range(0, sc.q8_mixed.bytes())?,
                y: sc.down.buf.range(0, m * HIDDEN * 4)?,
                k: FFN,
                rows: HIDDEN,
                ncols: m,
            },
        )?;
        self.norm_rows(
            stream,
            sc.down.buf.range(0, m * HIDDEN * 4)?,
            layer
                .post_ffw_norm
                .buf
                .range(0, layer.post_ffw_norm.bytes())?,
            sc.nrm2.buf.range(0, sc.nrm2.bytes())?,
            sc.q8_hidden.range(0, sc.q8_hidden.bytes())?,
            m,
        )?;
        ctx.row_add(
            stream,
            RowAddCall {
                out: r2.buf.range(0, r2.bytes())?,
                x: r1.buf.range(0, r1.bytes())?,
                y: sc.nrm2.buf.range(0, sc.nrm2.bytes())?,
                n: m * HIDDEN,
                flag: None,
                fused: false,
            },
        )?;
        self.trace(pos, &format!("L{index}_r2"), r2, m * HIDDEN * 4);
        // The per-layer-input block: the gated mixing of the second residual with
        // this layer's token-identity input, projected back to the hidden width.
        for row in 0..m {
            // The skipped-shared prefill packs the per-layer planes at the prefix
            // stride, not the full 42-layer stride.
            let at = (row * ple_stride + index * PLE) * 4;
            ctx.add_rmsnorm_q8(
                stream,
                AddNormQ8Call {
                    n: PLE,
                    eps: self.eps,
                    h: sc.ple_scaled.buf.range(at, PLE * 4)?,
                    delta: None,
                    weight: self
                        .ple_model_norm
                        .buf
                        .range(0, self.ple_model_norm.bytes())?,
                    norm_out: sc.ple_nrm.buf.range(row * PLE * 4, PLE * 4)?,
                    q8: sc.q8_head.range(0, sc.q8_head.bytes())?,
                    rows: 1,
                },
            )?;
        }
        ctx.embed_q8(
            stream,
            EmbedQ8Call {
                out: sc.ple_tbl.buf.range(0, m * PLE * 4)?,
                w: layer.table.buf.range(0, layer.table.bytes())?,
                ids: sc.ids.range(0, m * 4)?,
                m,
                n_rows: layer.table.rows,
                k: layer.table.k,
            },
        )?;
        self.trace(pos, &format!("L{index}_plep"), &sc.ple_nrm, m * PLE * 4);
        self.trace(pos, &format!("L{index}_pletbl"), &sc.ple_tbl, m * PLE * 4);
        ctx.row_scale(
            stream,
            RowScaleCall {
                out: sc.ple_mixed.buf.range(0, m * PLE * 4)?,
                x: sc.ple_tbl.buf.range(0, m * PLE * 4)?,
                scale: PLE_IDENTITY_SCALE,
                m,
                n: PLE,
            },
        )?;
        ctx.row_add(
            stream,
            RowAddCall {
                out: sc.ple_tbl.buf.range(0, m * PLE * 4)?,
                x: sc.ple_nrm.buf.range(0, m * PLE * 4)?,
                y: sc.ple_mixed.buf.range(0, m * PLE * 4)?,
                n: m * PLE,
                flag: None,
                fused: false,
            },
        )?;
        ctx.row_scale(
            stream,
            RowScaleCall {
                out: sc.ple_mixed.buf.range(0, m * PLE * 4)?,
                x: sc.ple_tbl.buf.range(0, m * PLE * 4)?,
                scale: PLE_COMBINED_SCALE,
                m,
                n: PLE,
            },
        )?;
        ctx.quantize_q8_1(
            stream,
            QuantizeCall {
                x: r2.buf.range(0, m * HIDDEN * 4)?,
                y: sc.q8_hidden.range(0, sc.q8_hidden.bytes())?,
                k: HIDDEN,
                ncols: m,
            },
        )?;
        ctx.gemma_matvec(
            stream,
            GemmaMatvecCall {
                w_qs: layer.ple_gate.qs.range(0, layer.ple_gate.qs_bytes())?,
                w_scales: layer
                    .ple_gate
                    .scales
                    .range(0, layer.ple_gate.scales_bytes())?,
                x_q8: sc.q8_hidden.range(0, sc.q8_hidden.bytes())?,
                y: sc.ple_nrm.buf.range(0, m * PLE * 4)?,
                k: HIDDEN,
                rows: PLE,
                ncols: m,
            },
        )?;
        ctx.gelu_ffn(
            stream,
            GeluFfnCall {
                out: sc.ple_tbl.buf.range(0, m * PLE * 4)?,
                gate: sc.ple_nrm.buf.range(0, m * PLE * 4)?,
                up: sc.ple_mixed.buf.range(0, m * PLE * 4)?,
                m,
                n: PLE,
            },
        )?;
        self.trace(pos, &format!("L{index}_plegate"), &sc.ple_nrm, m * PLE * 4);
        self.trace(pos, &format!("L{index}_plemix"), &sc.ple_tbl, m * PLE * 4);
        ctx.quantize_q8_1(
            stream,
            QuantizeCall {
                x: sc.ple_tbl.buf.range(0, m * PLE * 4)?,
                y: sc.q8_ple_mixed.range(0, sc.q8_ple_mixed.bytes())?,
                k: PLE,
                ncols: m,
            },
        )?;
        ctx.gemma_matvec(
            stream,
            GemmaMatvecCall {
                w_qs: layer.ple_proj.qs.range(0, layer.ple_proj.qs_bytes())?,
                w_scales: layer
                    .ple_proj
                    .scales
                    .range(0, layer.ple_proj.scales_bytes())?,
                x_q8: sc.q8_ple_mixed.range(0, sc.q8_ple_mixed.bytes())?,
                y: sc.proj.buf.range(0, m * HIDDEN * 4)?,
                k: PLE,
                rows: HIDDEN,
                ncols: m,
            },
        )?;
        self.trace(pos, &format!("L{index}_pled"), &sc.proj, m * HIDDEN * 4);
        self.norm_rows(
            stream,
            sc.proj.buf.range(0, m * HIDDEN * 4)?,
            layer
                .post_ple_norm
                .buf
                .range(0, layer.post_ple_norm.bytes())?,
            sc.nrm2.buf.range(0, sc.nrm2.bytes())?,
            sc.q8_hidden.range(0, sc.q8_hidden.bytes())?,
            m,
        )?;
        ctx.row_add(
            stream,
            RowAddCall {
                out: input.buf.range(0, input.bytes())?,
                x: r2.buf.range(0, r2.bytes())?,
                y: sc.nrm2.buf.range(0, sc.nrm2.bytes())?,
                n: m * HIDDEN,
                flag: None,
                fused: false,
            },
        )?;
        // The scaled layer output lands in `out`; `input` held the dead pre-layer
        // residual, so the scale has a distinct source buffer.
        ctx.row_scale(
            stream,
            RowScaleCall {
                out: out.buf.range(0, out.bytes())?,
                x: input.buf.range(0, input.bytes())?,
                scale: layer.scale,
                m,
                n: HIDDEN,
            },
        )?;
        self.trace(pos, &format!("L{index}"), out, m * HIDDEN * 4);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_split_default_is_armed_and_zero_disarms() {
        // Unset arms the qualified split; `0`/`off` is the rollback to the
        // single-pass summation order; any other spelling refuses the load.
        assert!(split_armed_from(None).expect("unset arms"));
        assert!(split_armed_from(Some("1")).expect("1 arms"));
        assert!(!split_armed_from(Some("0")).expect("zero disarms"));
        assert!(!split_armed_from(Some("off")).expect("off disarms"));
        assert!(split_armed_from(Some("yes")).is_err());
    }

    #[test]
    fn scratch_bytes_charges_the_prefill_maximum() {
        // Growable f32 vectors at PREFILL_BATCH rows (89,856 elements x 8 x 4),
        // fixed f32 ones + two logits rows (524,800 x 4), growable q8 at eight
        // columns plus the fixed head/qk buffers, plus the 12 input-cell bytes;
        // the armed split adds its partial buffer on top.
        let disarmed = scratch_bytes(false).expect("the disarmed scratch computes");
        let armed = scratch_bytes(true).expect("the armed scratch computes");
        assert_eq!(disarmed, 5_168_716);
        assert_eq!(armed, 9_379_404);
        assert_eq!(armed - disarmed, attn_part_bytes() as u64);
    }

    #[test]
    fn the_bolt_demand_is_refused_without_the_headroom() {
        // A device holding exactly the allocations (weights + KV + scratch) with
        // no room for the headroom reserve must be refused by the shared admit;
        // the same demand fits once the reserve is included.
        let weights_bytes = 3_000_000_000u64;
        let demand = bolt_demand(weights_bytes, true).expect("the Bolt demand computes");
        assert_eq!(demand.tensors, weights_bytes);
        assert!(demand.kv_cache > 0 && demand.work > 0 && demand.headroom > 0);
        let allocations = demand.tensors + demand.kv_cache + demand.work;
        let error = crate::kernels::cuda::plan::admit(&demand, allocations)
            .expect_err("an exact fit of the allocations alone must be refused");
        let EngError::MemoryPlan(message) = error else {
            panic!("expected MemoryPlan, got {error:?}");
        };
        assert!(message.contains("headroom"), "{message}");
        assert!(
            message.ends_with(&format!("device has {allocations} free")),
            "{message}"
        );
        assert_eq!(
            crate::kernels::cuda::plan::admit(&demand, allocations + demand.headroom),
            Ok(allocations + demand.headroom)
        );
    }

    fn logits(model: &Bolt) -> Vec<f32> {
        let mut bytes = vec![0u8; VOCAB * 4];
        model
            .ctx
            .download(&model.logits_capped.buf, &mut bytes)
            .expect("logits");
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect()
    }

    fn assert_logits(got: &[f32], want: &[f32]) {
        let mut max_relative = 0.0f32;
        for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
            assert!(g.is_finite() && w.is_finite(), "non-finite logit {i}");
            let error = (g - w).abs();
            assert!(error <= 2e-5 * w.abs(), "logit {i}: {g} versus {w}");
            if w != 0.0 {
                max_relative = max_relative.max(error / w.abs());
            }
        }
        eprintln!("logits max relative error {max_relative}");
    }

    /// Every launch shape of the Bolt step, replayed with tiny buffers: the wrappers
    /// and native entries must admit the whole geometry before any model runs. This
    /// test exists because a validation reject surfaced only on the device path.
    #[test]
    #[ignore = "requires a CUDA device; use --release --test-threads=1"]
    fn bolt_step_shapes_are_admitted() {
        use crate::kernels::cuda::CudaContext;
        use crate::kernels::cuda::resident::attn_decode_gemma::AttnDecodeGemmaCall;
        use crate::kernels::cuda::resident::kv8_append::{Kv8Data, Kv8Shape};
        let ctx = CudaContext::with_admission(&crate::family::project::cuda::served_admission())
            .expect("context");
        let stream = ctx.stream().expect("stream");
        let zero = |n: usize| vec![0u8; n];
        let f32s = |n: usize| ctx.upload(&zero(n * 4)).expect("f32 upload");
        let u16s = |n: usize| ctx.upload(&zero(n * 2)).expect("u16 upload");
        let i8s = |n: usize| ctx.upload(&zero(n)).expect("i8 upload");
        let mut word_bytes = Vec::with_capacity(8);
        word_bytes.extend_from_slice(&0i32.to_le_bytes());
        word_bytes.extend_from_slice(&1i32.to_le_bytes());
        let words = ctx.upload(&word_bytes).expect("words");
        // Embedding gather and the hidden-path ops at full Bolt widths.
        ctx.embed_q8(
            &stream,
            EmbedQ8Call {
                out: f32s(2560).range(0, 2560 * 4).unwrap(),
                w: i8s(1024 * (2560 / 32) * 34)
                    .range(0, 1024 * (2560 / 32) * 34)
                    .unwrap(),
                ids: words.range(0, 4).unwrap(),
                m: 1,
                n_rows: 1024,
                k: 2560,
            },
        )
        .expect("embed 2560");
        for (m, n) in [(1, 2560), (1, 256 * 42), (1, 256), (1, 2560)] {
            ctx.row_scale(
                &stream,
                RowScaleCall {
                    out: f32s(n).range(0, n * 4).unwrap(),
                    x: f32s(n).range(0, n * 4).unwrap(),
                    scale: 0.5,
                    m,
                    n,
                },
            )
            .expect("row_scale {n}");
            ctx.row_add(
                &stream,
                RowAddCall {
                    out: f32s(n).range(0, n * 4).unwrap(),
                    x: f32s(n).range(0, n * 4).unwrap(),
                    y: f32s(n).range(0, n * 4).unwrap(),
                    n,
                    flag: None,
                    fused: false,
                },
            )
            .expect("row_add {n}");
        }
        // The quantize/matvec pairs at every k the step uses.
        for k in [2560, 2048, 1024, 256] {
            ctx.quantize_q8_1(
                &stream,
                QuantizeCall {
                    x: f32s(k).range(0, k * 4).unwrap(),
                    y: i8s(q8_1_bytes(k, 1).unwrap())
                        .range(0, q8_1_bytes(k, 1).unwrap())
                        .unwrap(),
                    k,
                    ncols: 1,
                },
            )
            .expect("quantize {k}");
            ctx.gemma_matvec(
                &stream,
                GemmaMatvecCall {
                    w_qs: i8s(256 * k).range(0, 256 * k).unwrap(),
                    w_scales: u16s(256 * (k / 32)).range(0, 256 * (k / 32) * 2).unwrap(),
                    x_q8: i8s(q8_1_bytes(k, 1).unwrap())
                        .range(0, q8_1_bytes(k, 1).unwrap())
                        .unwrap(),
                    y: f32s(256).range(0, 256 * 4).unwrap(),
                    k,
                    rows: 256,
                    ncols: 1,
                },
            )
            .expect("matvec k={k}");
        }
        ctx.gemma_matvec(
            &stream,
            GemmaMatvecCall {
                w_qs: i8s(10752 * 2560).range(0, 10752 * 2560).unwrap(),
                w_scales: u16s(10752 * 80).range(0, 10752 * 80 * 2).unwrap(),
                x_q8: i8s(q8_1_bytes(2560, 1).unwrap())
                    .range(0, q8_1_bytes(2560, 1).unwrap())
                    .unwrap(),
                y: f32s(10752).range(0, 10752 * 4).unwrap(),
                k: 2560,
                rows: 10752,
                ncols: 1,
            },
        )
        .expect("matvec ple projection");
        // The per-head norm at 256 and 512, including the interior-slice pattern.
        let ple = f32s(256 * 42);
        for n in [256usize, 512] {
            ctx.add_rmsnorm_q8(
                &stream,
                AddNormQ8Call {
                    n,
                    eps: 1e-6,
                    h: f32s(n).range(0, n * 4).unwrap(),
                    delta: None,
                    weight: f32s(n).range(0, n * 4).unwrap(),
                    norm_out: f32s(n).range(0, n * 4).unwrap(),
                    q8: i8s(q8_1_bytes(n, 1).unwrap())
                        .range(0, q8_1_bytes(n, 1).unwrap())
                        .unwrap(),
                    rows: 1,
                },
            )
            .expect("head norm {n}");
        }
        ctx.add_rmsnorm_q8(
            &stream,
            AddNormQ8Call {
                n: 256,
                eps: 1e-6,
                h: ple.range(256 * 4, 256 * 4).unwrap(),
                delta: None,
                weight: f32s(256).range(0, 256 * 4).unwrap(),
                norm_out: f32s(256).range(0, 256 * 4).unwrap(),
                q8: i8s(q8_1_bytes(256, 1).unwrap())
                    .range(0, q8_1_bytes(256, 1).unwrap())
                    .unwrap(),
                rows: 1,
            },
        )
        .expect("ple slice norm");
        // The PLE table gather and the gated mixing at 256.
        ctx.embed_q8(
            &stream,
            EmbedQ8Call {
                out: f32s(256).range(0, 256 * 4).unwrap(),
                w: i8s(1024 * (256 / 32) * 34)
                    .range(0, 1024 * (256 / 32) * 34)
                    .unwrap(),
                ids: words.range(0, 4).unwrap(),
                m: 1,
                n_rows: 1024,
                k: 256,
            },
        )
        .expect("ple table gather");
        ctx.gelu_ffn(
            &stream,
            GeluFfnCall {
                out: f32s(256).range(0, 256 * 4).unwrap(),
                gate: f32s(256).range(0, 256 * 4).unwrap(),
                up: f32s(256).range(0, 256 * 4).unwrap(),
                m: 1,
                n: 256,
            },
        )
        .expect("gelu 256");
        // The attention path at both head widths, 8 q heads, 2 kv heads.
        for head in [256usize, 512] {
            let shape = Kv8Shape {
                rows: 1,
                n_kv_head: 2,
                head_dim: head,
                cells: 32768,
                code_scale: crate::kernels::cuda::resident::kv8_append::Kv8CodeScale::UnroundedF32,
            };
            let codes = 2 * 32768 * head;
            let scales = 2 * 32768 * (head / 32);
            ctx.kv8_chunk(
                &stream,
                shape,
                Kv8Data {
                    k: f32s(2 * head).range(0, 2 * head * 4).unwrap(),
                    v: f32s(2 * head).range(0, 2 * head * 4).unwrap(),
                    k_codes: i8s(codes).range(0, codes).unwrap(),
                    k_scales: u16s(scales).range(0, scales * 2).unwrap(),
                    v_codes: i8s(codes).range(0, codes).unwrap(),
                    v_scales: u16s(scales).range(0, scales * 2).unwrap(),
                },
                0,
            )
            .expect("kv8 chunk {head}");
            ctx.rope_gemma(
                &stream,
                RopeGemmaCall {
                    q: f32s(8 * head).range(0, 8 * head * 4).unwrap(),
                    k: f32s(2 * head).range(0, 2 * head * 4).unwrap(),
                    pos: words.range(0, 4).unwrap(),
                    factors: None,
                    rows_q: 1,
                    rows_k: 1,
                    n_q_head: 8,
                    n_kv_head: 2,
                    head_dim: head,
                    global_mode: head == 512,
                },
            )
            .expect("rope {head}");
            ctx.attn_decode_gemma(
                &stream,
                AttnDecodeGemmaCall {
                    out: f32s(8 * head).range(0, 8 * head * 4).unwrap(),
                    status: None,
                    q: f32s(8 * head).range(0, 8 * head * 4).unwrap(),
                    k_codes: i8s(codes).range(0, codes).unwrap(),
                    k_scales: u16s(scales).range(0, scales * 2).unwrap(),
                    v_codes: i8s(codes).range(0, codes).unwrap(),
                    v_scales: u16s(scales).range(0, scales * 2).unwrap(),
                    kv_len: words.range(4, 4).unwrap(),
                    m: 1,
                    n_q_head: 8,
                    n_kv_head: 2,
                    head_dim: head,
                    cells: 32768,
                    window: if head == 512 { 0 } else { 512 },
                    scale: 1.0,
                    split: None,
                },
            )
            .expect("attn {head}");
            // The split path of the same geometry: the wrapper must admit the
            // part+combine plan at both head widths when the switch arms.
            let part_floats = 8 * super::ATTN_SPLIT_MAX * (2 + head);
            ctx.attn_decode_gemma(
                &stream,
                AttnDecodeGemmaCall {
                    out: f32s(8 * head).range(0, 8 * head * 4).unwrap(),
                    status: None,
                    q: f32s(8 * head).range(0, 8 * head * 4).unwrap(),
                    k_codes: i8s(codes).range(0, codes).unwrap(),
                    k_scales: u16s(scales).range(0, scales * 2).unwrap(),
                    v_codes: i8s(codes).range(0, codes).unwrap(),
                    v_scales: u16s(scales * 2).range(0, scales * 2).unwrap(),
                    kv_len: words.range(4, 4).unwrap(),
                    m: 1,
                    n_q_head: 8,
                    n_kv_head: 2,
                    head_dim: head,
                    cells: 32768,
                    window: if head == 512 { 0 } else { 512 },
                    scale: 1.0,
                    split: Some(AttnDecodeGemmaSplit {
                        part: ctx
                            .upload(&zero(part_floats * 4))
                            .expect("part upload")
                            .range(0, part_floats * 4)
                            .unwrap(),
                        max_splits: super::ATTN_SPLIT_MAX,
                        min_keys: super::ATTN_SPLIT_MIN_KEYS,
                    }),
                },
            )
            .expect("attn split {head}");
        }
        stream.synchronize().expect("sync");
    }

    /// Single-step parity vehicle: with BOLT_TRACE_DIR set, writes one f32
    /// checkpoint per layer (and the scaled embedding) for the first two
    /// positions, unmodified by any batched replay. The torch reference writes
    /// the same names; the first diverging layer names the wrong op.
    #[test]
    #[ignore = "requires the converted Bolt file and a device; use --release --test-threads=1"]
    fn trace_two_single_steps() {
        let path = std::env::var("BOLT_Q8_FILE").expect("converted model path");
        let mut model = super::load(std::path::Path::new(&path), None).expect("load");
        let seq = model.new_seq().expect("sequence");
        for (pos, token) in [2u32, 19].into_iter().enumerate() {
            if pos == 0 {
                model
                    .decode(&seq, token, pos)
                    .expect("first token with logits");
            } else {
                model
                    .decode_hidden(&seq, token, pos)
                    .expect("second token hidden");
            }
        }
    }

    /// Greedy-quality gate: the full reference prompt, logits traced at every step.
    #[test]
    #[ignore = "requires the converted Bolt file and a device; use --release --test-threads=1"]
    fn trace_full_prompt_logits() {
        let path = std::env::var("BOLT_Q8_FILE").expect("converted model path");
        let mut model = super::load(std::path::Path::new(&path), None).expect("load");
        let seq = model.new_seq().expect("sequence");
        for pos in 0..32u32 {
            let token = 2 + pos * 17;
            model.decode(&seq, token, pos as usize).expect("step");
        }
    }

    /// One model owns the device. Replaying from position zero overwrites every live cache cell.
    #[test]
    #[ignore = "requires the converted Bolt file and a device; use --release --test-threads=1"]
    fn batched_prefill_matches_single_steps() {
        let path = std::env::var("BOLT_Q8_FILE").expect("converted model path");
        let header = std::env::var("BOLT_Q8_HEADER_SHA256").ok();
        let mut model = super::load(std::path::Path::new(&path), header.as_deref()).expect("load");
        let seq = model.new_seq().expect("sequence");
        assert!(matches!(
            model.prefill_batch(&seq, &[], 0, false),
            Err(EngError::BadRequest(_))
        ));
        assert!(matches!(
            model.prefill_batch(&seq, &[2; 9], 0, false),
            Err(EngError::BadRequest(_))
        ));
        assert!(matches!(
            model.prefill_batch(&seq, &[2], CELLS, false),
            Err(EngError::ContextLimit(_))
        ));
        assert!(matches!(
            model.prefill_batch(&seq, &[2], usize::MAX, false),
            Err(EngError::ContextLimit(_))
        ));
        // Decode bounds its position like prefill: the last valid cell is one below the cap.
        assert!(matches!(
            model.decode(&seq, 2, CELLS),
            Err(EngError::ContextLimit(_))
        ));
        assert!(matches!(
            model.decode_hidden(&seq, 2, CELLS),
            Err(EngError::ContextLimit(_))
        ));
        assert!(matches!(
            model.decode(&seq, 2, usize::MAX),
            Err(EngError::ContextLimit(_))
        ));
        let prompt: Vec<u32> = (0..32).map(|i| 2 + i * 17).collect();
        let single_started = std::time::Instant::now();
        for (pos, &token) in prompt.iter().enumerate() {
            if pos + 1 == prompt.len() {
                model.decode(&seq, token, pos).expect("single last token");
            } else {
                model
                    .decode_hidden(&seq, token, pos)
                    .expect("single hidden token");
            }
        }
        let single_elapsed = single_started.elapsed();
        let expected = logits(&model);
        // Parity hook: write this run's single-step logits for the reference
        // comparison (scratch tooling reads it; no assertion depends on it).
        if let Ok(path) = std::env::var("BOLT_DUMP_LOGITS") {
            let bytes: Vec<u8> = expected.iter().flat_map(|v| v.to_le_bytes()).collect();
            std::fs::write(&path, bytes).expect("dump logits");
            eprintln!("logits dumped to {path}");
        }
        model
            .decode(&seq, 2, prompt.len())
            .expect("single continuation");
        let continued = logits(&model);
        eprintln!("timing 32 single steps {single_elapsed:?}");
        // Eight rows is the shipped cap; three also exercises a partial final batch.
        for width in [1, Bolt::PREFILL_BATCH, 3] {
            let started = std::time::Instant::now();
            let mut pos = 0;
            for batch in prompt.chunks(width) {
                model
                    .prefill_batch(&seq, batch, pos, pos + batch.len() == prompt.len())
                    .expect("batch");
                pos += batch.len();
            }
            let elapsed = started.elapsed();
            assert_logits(&logits(&model), &expected);
            model
                .decode(&seq, 2, prompt.len())
                .expect("batch continuation");
            assert_logits(&logits(&model), &continued);
            eprintln!(
                "timing 32 batched tokens m={width} {elapsed:?}; speedup {:.3}",
                single_elapsed.as_secs_f64() / elapsed.as_secs_f64()
            );
        }
    }

    /// The skipped-shared prefill must leave the same final logits as a
    /// full-layer prefill: a hidden batch runs only the cache-writing physical
    /// prefix, while a logits batch runs every layer. The shared layers write
    /// no cache cells and feed only the final norm, so both paths agree.
    /// Each width reloads a fresh (zeroed) cache and runs the skip path FIRST,
    /// so a missing skip-path cache write at any width corrupts its logits
    /// instead of being masked by an earlier full write; the full path then
    /// overwrites the same cells from position 0.
    #[test]
    #[ignore = "requires the converted Bolt file and a device; use --release --test-threads=1"]
    fn skip_prefill_matches_full_layer_prefill() {
        let path = std::env::var("BOLT_Q8_FILE").expect("converted model path");
        let prompt: Vec<u32> = (0..32).map(|i| 2 + i * 17).collect();
        let n = prompt.len();
        // Widths 8 (the shipped cap), 3 (a partial final batch) and 1.
        for width in [Bolt::PREFILL_BATCH, 3, 1] {
            // Fresh load per width: the cache starts zeroed, so a missing
            // skip-path write here is not masked by the last width's cells.
            let mut model = super::load(std::path::Path::new(&path), None).expect("load");
            let seq = model.new_seq().expect("sequence");
            // Skip-prefill: non-final batches hide logits and run only the
            // physical prefix; the final batch reads logits over the cache.
            let mut pos = 0;
            for batch in prompt.chunks(width) {
                model
                    .prefill_batch(&seq, batch, pos, pos + batch.len() == n)
                    .expect("skip batch");
                pos += batch.len();
            }
            let skip = logits(&model);
            model.decode(&seq, 2, n).expect("skip continuation");
            let skip_continued = logits(&model);
            // Full-layer prefill overwrites the same cells from position 0:
            // every batch reads logits, so every layer runs.
            let mut pos = 0;
            for batch in prompt.chunks(Bolt::PREFILL_BATCH) {
                model
                    .prefill_batch(&seq, batch, pos, true)
                    .expect("full batch");
                pos += batch.len();
            }
            assert_logits(&logits(&model), &skip);
            model.decode(&seq, 2, n).expect("full continuation");
            assert_logits(&logits(&model), &skip_continued);
        }
    }
}
