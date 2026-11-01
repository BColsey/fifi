//! The launches of one mixer block as DATA: an ordered list of [`BlockOp`] that `Mixers::run_block`
//! executes one by one. Pure host code.
//!
//! The list is the single source of what a block does: the order of the launches, every plan slot
//! each launch reads and writes, the weight roles it names, the `StepParams` words it takes and
//! the column of the shared projection row each projection writes. The executor takes ALL of
//! these from the op it runs (its wrapper arguments that are not slots come from the pure shape
//! builders of `mixer_shapes.rs`); `reads()` and `writes()` are derived from the same fields by a
//! `match`, so there is no second copy. The CPU tests check this list against the step plan
//! (lifetimes, write-before-read, phase order, column tiling), so they check what runs.

use super::{
    dims::KvMode,
    mixer_geometry::MixerGeometry,
    state::StepField,
    state_layout::Plane,
    step_plan::{Phase, Slot},
};
use crate::{
    EngError,
    family::{AttentionKind, gguf_model::Role, project::cuda::WeightLayout},
    kernels::cuda::resident::attn_decode::AttnSplit,
};

/// Rows of every block: the shared projection row exists only at one row (see `mixer_geometry.rs`).
pub(super) const BLOCK_ROWS: usize = 1;

const F32: usize = size_of::<f32>();

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum BlockOp {
    /// `resid += delta` (when given), then `xf = pack(RMSNorm(resid) * weight)` and, when asked,
    /// the f32 normalized row.
    NormPack {
        phase: Phase,
        weight: Role,
        delta: Option<Slot>,
        xf: Slot,
        norm_out: Option<Slot>,
    },
    /// The NATIVE twin of [`BlockOp::NormPack`]: the same fused add and norm through
    /// `add_rmsnorm_q8`, writing the padded q8_1 row `q8` (the matvec input) beside the f32
    /// normalized row `norm_out` — the producer always writes both, so every `NormQ8` names
    /// `NormRow` (the router's input at the epilogue, dead scratch at the prologue and the final
    /// norm, which is why the NATIVE plan holds `NormRow` live across the whole layer).
    NormQ8 {
        phase: Phase,
        weight: Role,
        delta: Option<Slot>,
        q8: Slot,
        norm_out: Slot,
    },
    /// One dense projection of the row packed in `xf` into columns `col..` of `y`'s row.
    Project {
        phase: Phase,
        role: Role,
        xf: Slot,
        y: Slot,
        col: usize,
    },
    /// The NATIVE twin of [`BlockOp::Project`]: one `matvec_q` launch against the native blocks of
    /// `role`, reading the padded q8_1 row `x_q8` and writing `rows` f32 values at column `col` of
    /// `y`'s row (one row only, the matvec's `y[j * rows + r]`).
    ProjectQ {
        phase: Phase,
        role: Role,
        x_q8: Slot,
        y: Slot,
        col: usize,
    },
    /// One EXL3 projection of the unquantized F32 row; scratch is shared in stream order. The
    /// sticky arithmetic flag is read once per token by the step, not after each projection.
    ProjectExl3 {
        phase: Phase,
        role: Role,
        x: Slot,
        y: Slot,
        col: usize,
    },
    /// One plain F32 matrix times the F32 row, for the DeltaNet gates an EXL3 pack leaves
    /// unquantized: the weight is the role's raw tensor, `rows` columns land at `col` of `y`.
    ProjectPlain {
        phase: Phase,
        role: Role,
        x: Slot,
        y: Slot,
        col: usize,
    },
    /// `pack` of the f32 row `x` (`k` values) into `xf`.
    Pack {
        phase: Phase,
        x: Slot,
        xf: Slot,
        k: usize,
    },
    /// The NATIVE twin of [`BlockOp::Pack`]: the q8_1 quantize of the f32 row `x` (`k` values) into
    /// the padded row `q8`.
    QuantizeQ8 {
        phase: Phase,
        x: Slot,
        q8: Slot,
        k: usize,
    },
    /// Causal conv: the qkv columns of `proj` -> `out`, with the conv weights `weights`.
    Conv {
        proj: Slot,
        out: Slot,
        weights: Role,
    },
    /// Gated delta rule: `conv` and the z, beta, alpha columns of `proj` -> `out`.
    Delta {
        conv: Slot,
        proj: Slot,
        out: Slot,
        a: Role,
        dt_bias: Role,
        norm: Role,
    },
    /// Attention prep: `proj` -> `q`, and the K and V rows into the cache at the `pos` word; a
    /// refused position ORs mask 4 into the `status` word.
    AttnPrep {
        proj: Slot,
        q: Slot,
        q_norm: Role,
        k_norm: Role,
        pos: StepField,
        status: StepField,
    },
    /// Decode attention: `q`, the gate columns of `proj` and the cache (up to the `kv_len` word) ->
    /// `part` partials -> `out`.
    AttnDecode {
        q: Slot,
        proj: Slot,
        part: Slot,
        out: Slot,
        kv_len: StepField,
    },
    /// The Int8G32 twin of [`BlockOp::AttnPrep`]: the same per-head RMSNorm and rope through the
    /// rows entry of `attn_prep`, but the post-rope k and the unchanged v go to the f32 rows slots
    /// `k_rows`/`v_rows` (indexed by the token) and no fp16 cache byte is written; a refused
    /// position ORs mask 4 into the `status` word.
    AttnPrepRows {
        proj: Slot,
        q: Slot,
        q_norm: Role,
        k_norm: Role,
        k_rows: Slot,
        v_rows: Slot,
        pos: StepField,
        status: StepField,
    },
    /// The KV8 append of the Int8G32 step: the f32 `k`/`v` rows of the prep op, quantized into the
    /// four code and scale `planes` of this layer's KV pool at cells `*pos ..`; a refused position
    /// ORs mask 4 into the `status` word.
    Kv8Append {
        k: Slot,
        v: Slot,
        k_codes: Plane,
        k_scales: Plane,
        v_codes: Plane,
        v_scales: Plane,
        pos: StepField,
        status: StepField,
    },
    /// The KV8 split-KV decode attention: `q`, the gate columns of `proj` and the code and scale
    /// `planes` (up to the `kv_len` word) -> `part` partials -> `out`, at the `split` the plan
    /// sized the partials with (Q8).
    AttnDecodeQ8 {
        q: Slot,
        proj: Slot,
        k_codes: Plane,
        k_scales: Plane,
        v_codes: Plane,
        v_scales: Plane,
        kv_len: StepField,
        part: Slot,
        out: Slot,
        split: AttnSplit,
    },
    /// The multi-device twin of [`BlockOp::AttnPrepRows`]: the same per-head
    /// RMSNorm and rope for THIS card's query-head run and `k_range` rows of
    /// the new K row - the range is absolute in the full row, because a
    /// card's run may start inside a head and a K row's RoPE phase comes
    /// from the range, never from the slice - with the unchanged V slice
    /// beside it, into the card's extents of the full replicated `k_rows`
    /// and `v_rows` slots. No cache byte is written: the append waits for
    /// the gathers.
    AttnPrepCard {
        proj: Slot,
        q: Slot,
        q_norm: Role,
        k_norm: Role,
        k_rows: Slot,
        v_rows: Slot,
        k_range: (usize, usize),
        pos: StepField,
        status: StepField,
    },
    /// The multi-device twin of the fused fp16 append inside
    /// [`BlockOp::AttnPrep`]: the gathered FULL f32 `k` and `v` rows written
    /// into this layer's fp16 cache `planes` at the `pos` word; a refused
    /// position ORs mask 4 into the `status` word.
    KvAppendF16 {
        k: Slot,
        v: Slot,
        k_plane: Plane,
        v_plane: Plane,
        pos: StepField,
        status: StepField,
    },
}

impl BlockOp {
    pub(super) fn phase(&self) -> Phase {
        match *self {
            Self::NormPack { phase, .. }
            | Self::NormQ8 { phase, .. }
            | Self::Project { phase, .. }
            | Self::ProjectQ { phase, .. }
            | Self::ProjectExl3 { phase, .. }
            | Self::ProjectPlain { phase, .. }
            | Self::Pack { phase, .. }
            | Self::QuantizeQ8 { phase, .. } => phase,
            Self::Conv { .. } => Phase::DeltaConv,
            Self::Delta { .. } => Phase::DeltaRule,
            Self::AttnPrep { .. } | Self::AttnPrepRows { .. } | Self::AttnPrepCard { .. } => {
                Phase::AttnPrep
            }
            Self::Kv8Append { .. } | Self::KvAppendF16 { .. } => Phase::Kv8Append,
            Self::AttnDecode { .. } | Self::AttnDecodeQ8 { .. } => Phase::AttnCore,
        }
    }

    /// The plan slots the launch reads (the residual stream is not a slot).
    pub(super) fn reads(&self) -> Vec<Slot> {
        match *self {
            Self::NormPack { delta, .. } | Self::NormQ8 { delta, .. } => {
                delta.into_iter().collect()
            }
            Self::Project { xf, .. } => vec![xf],
            Self::ProjectQ { x_q8, .. } => vec![x_q8],
            Self::ProjectExl3 { x, .. } => vec![x, Slot::Exl3Flag],
            Self::ProjectPlain { x, .. } => vec![x],
            Self::Pack { x, .. } | Self::QuantizeQ8 { x, .. } => vec![x],
            Self::Conv { proj, .. } => vec![proj],
            Self::Delta { conv, proj, .. } => vec![conv, proj],
            Self::AttnPrep { proj, .. }
            | Self::AttnPrepRows { proj, .. }
            | Self::AttnPrepCard { proj, .. } => {
                vec![proj]
            }
            Self::Kv8Append { k, v, .. } | Self::KvAppendF16 { k, v, .. } => vec![k, v],
            Self::AttnDecode { q, proj, .. } | Self::AttnDecodeQ8 { q, proj, .. } => {
                vec![q, proj]
            }
        }
    }

    pub(super) fn writes(&self) -> Vec<Slot> {
        match *self {
            Self::NormPack { xf, norm_out, .. } => {
                [Some(xf), norm_out].into_iter().flatten().collect()
            }
            Self::NormQ8 { q8, norm_out, .. } => vec![q8, norm_out],
            Self::Project { y, .. } | Self::ProjectQ { y, .. } => vec![y],
            Self::ProjectExl3 { y, .. } => {
                vec![y, Slot::Exl3Xh, Slot::Exl3Z, Slot::Exl3Flag]
            }
            Self::ProjectPlain { y, .. } => vec![y],
            Self::Pack { xf, .. } => vec![xf],
            Self::QuantizeQ8 { q8, .. } => vec![q8],
            Self::Conv { out, .. } | Self::Delta { out, .. } => vec![out],
            Self::AttnPrep { q, .. } => vec![q],
            Self::AttnPrepRows {
                q, k_rows, v_rows, ..
            } => vec![q, k_rows, v_rows],
            Self::AttnPrepCard {
                q, k_rows, v_rows, ..
            } => vec![q, k_rows, v_rows],
            // The quantized cells go to the state's KV planes, which are not plan slots.
            Self::Kv8Append { .. } | Self::KvAppendF16 { .. } => Vec::new(),
            Self::AttnDecode { part, out, .. } | Self::AttnDecodeQ8 { part, out, .. } => {
                vec![part, out]
            }
        }
    }
}

/// The byte offset of column `col` inside a fused f32 row, checked for every `ProjectQ` pair of the
/// geometry at block construction (a typed refusal here, never a launch-time wrapper error inside a
/// token): `col * 4` must not overflow and `rows` must fit the fused row's rest. There is no 4-byte
/// alignment arm: `col * 4` is a multiple of 4 for every integer `col`, so such an arm could never
/// fire; the wrapper's own alignment check on the (always 4-aligned, because 256-byte arena
/// offsets) slot remains.
pub(super) fn check_native_column(
    role: Role,
    col: usize,
    rows: usize,
    ld: usize,
) -> Result<(), EngError> {
    col.checked_mul(F32).ok_or_else(|| {
        EngError::Checkpoint(format!(
            "mixer ops: the column offset of {role:?} overflows a byte count"
        ))
    })?;
    if col.checked_add(rows).is_none_or(|end| end > ld) {
        return Err(EngError::Checkpoint(format!(
            "mixer ops: {role:?} writes {rows} columns at {col}, past the fused row of {ld}"
        )));
    }
    Ok(())
}

/// The projection twin of one fused-row launch (one role, one column): `Project` against the
/// fp16 pack `xf` or `ProjectQ` against the padded q8_1 row `x_q8`.
/// The input tuple holds the FP16, Q8 and original F32 views of the same activation.
fn project_op(
    phase: Phase,
    role: Role,
    (xf, x_q8, x_f32): (Slot, Slot, Slot),
    y: Slot,
    col: usize,
    layout: WeightLayout,
) -> BlockOp {
    match layout {
        WeightLayout::Tiled => BlockOp::Project {
            phase,
            role,
            xf,
            y,
            col,
        },
        WeightLayout::Native => BlockOp::ProjectQ {
            phase,
            role,
            x_q8,
            y,
            col,
        },
        // The pack keeps the 32-row DeltaNet gates as plain matrices.
        WeightLayout::Exl3 if matches!(role, Role::SsmBeta | Role::SsmAlpha) => {
            BlockOp::ProjectPlain {
                phase,
                role,
                x: x_f32,
                y,
                col,
            }
        }
        WeightLayout::Exl3 => BlockOp::ProjectExl3 {
            phase,
            role,
            x: x_f32,
            y,
            col,
        },
    }
}

/// The pack twin of one core-output launch: `Pack` into the fp16 `xf` or `QuantizeQ8` into the
/// padded q8_1 row `q8`.
fn pack_op(
    phase: Phase,
    x: Slot,
    xf: Slot,
    q8: Slot,
    k: usize,
    layout: WeightLayout,
) -> Option<BlockOp> {
    match layout {
        WeightLayout::Tiled => Some(BlockOp::Pack { phase, x, xf, k }),
        WeightLayout::Native => Some(BlockOp::QuantizeQ8 { phase, x, q8, k }),
        WeightLayout::Exl3 => None,
    }
}

fn prologue(layout: WeightLayout) -> BlockOp {
    match layout {
        WeightLayout::Tiled => BlockOp::NormPack {
            phase: Phase::MixPack,
            weight: Role::AttnNorm,
            delta: None,
            xf: Slot::XfMix,
            norm_out: None,
        },
        WeightLayout::Native => BlockOp::NormQ8 {
            phase: Phase::MixPack,
            weight: Role::AttnNorm,
            delta: None,
            q8: Slot::Q8Mix,
            norm_out: Slot::NormRow,
        },
        // The pack is discarded; EXL3 reads the producer's F32 `norm_out` only.
        WeightLayout::Exl3 => BlockOp::NormPack {
            phase: Phase::MixPack,
            weight: Role::AttnNorm,
            delta: None,
            xf: Slot::XfMix,
            norm_out: Some(Slot::NormRow),
        },
    }
}

/// The residual add of the mixer output and the norm for the MoE block: ends the mixer block.
fn epilogue(layout: WeightLayout) -> BlockOp {
    match layout {
        WeightLayout::Tiled => BlockOp::NormPack {
            phase: Phase::AddNorm,
            weight: Role::PostAttentionNorm,
            delta: Some(Slot::MixOut),
            xf: Slot::XfMoe,
            norm_out: Some(Slot::NormRow),
        },
        WeightLayout::Native => BlockOp::NormQ8 {
            phase: Phase::AddNorm,
            weight: Role::PostAttentionNorm,
            delta: Some(Slot::MixOut),
            q8: Slot::Q8Moe,
            norm_out: Slot::NormRow,
        },
        WeightLayout::Exl3 => BlockOp::NormPack {
            phase: Phase::AddNorm,
            weight: Role::PostAttentionNorm,
            delta: Some(Slot::MixOut),
            xf: Slot::XfMoe,
            norm_out: Some(Slot::NormRow),
        },
    }
}

fn delta_ops(g: &MixerGeometry, layout: WeightLayout, fuse_inproj: bool) -> Vec<BlockOp> {
    let d = &g.delta;
    let project = |role, col| {
        project_op(
            Phase::DeltaIn,
            role,
            (Slot::XfMix, Slot::Q8Mix, Slot::NormRow),
            Slot::DeltaProj,
            col,
            layout,
        )
    };
    // The four in-projections read the same normalized row; with the fusion flag one launch
    // against the row-concatenated weight writes all their columns (bitwise per segment).
    let mut ops = if fuse_inproj {
        vec![project(Role::AttnQkv, 0)]
    } else {
        vec![
            project(Role::AttnQkv, 0),
            project(Role::AttnGate, d.z_off),
            project(Role::SsmBeta, d.beta_off),
            project(Role::SsmAlpha, d.alpha_off),
        ]
    };
    ops.extend([
        BlockOp::Conv {
            proj: Slot::DeltaProj,
            out: Slot::DeltaConv,
            weights: Role::SsmConv1d,
        },
        BlockOp::Delta {
            conv: Slot::DeltaConv,
            proj: Slot::DeltaProj,
            out: Slot::DeltaCore,
            a: Role::SsmA,
            dt_bias: Role::SsmDtBias,
            norm: Role::SsmNorm,
        },
    ]);
    ops.extend(pack_op(
        Phase::DeltaPack,
        Slot::DeltaCore,
        Slot::XfDeltaOut,
        Slot::Q8DeltaOut,
        d.value_width,
        layout,
    ));
    ops.push(project_op(
        Phase::DeltaOut,
        Role::SsmOut,
        (Slot::XfDeltaOut, Slot::Q8DeltaOut, Slot::DeltaCore),
        Slot::MixOut,
        0,
        layout,
    ));
    ops
}

fn attn_ops(g: &MixerGeometry, layout: WeightLayout, kv_mode: KvMode) -> Vec<BlockOp> {
    let a = &g.attn;
    let project = |role, col| {
        project_op(
            Phase::AttnIn,
            role,
            (Slot::XfMix, Slot::Q8Mix, Slot::NormRow),
            Slot::AttnProj,
            col,
            layout,
        )
    };
    // The KV core of the mode: the fp16 pair writes the cache rows inside prep and decodes over the
    // fp16 planes; the Int8G32 list hands f32 rows from the rows entry of `attn_prep` to
    // `kv8_append`, which quantizes them into the code and scale planes the q8 decode then reads
    // (at the Q8 split, whose min_keys is a multiple of the 64-cell tile).
    let core = match kv_mode {
        KvMode::Fp16 => vec![
            BlockOp::AttnPrep {
                proj: Slot::AttnProj,
                q: Slot::AttnQ,
                q_norm: Role::AttnQNorm,
                k_norm: Role::AttnKNorm,
                pos: StepField::Pos,
                status: StepField::Status,
            },
            BlockOp::AttnDecode {
                q: Slot::AttnQ,
                proj: Slot::AttnProj,
                part: Slot::AttnPart,
                out: Slot::AttnCore,
                kv_len: StepField::KvLen,
            },
        ],
        KvMode::Int8G32 => vec![
            BlockOp::AttnPrepRows {
                proj: Slot::AttnProj,
                q: Slot::AttnQ,
                q_norm: Role::AttnQNorm,
                k_norm: Role::AttnKNorm,
                k_rows: Slot::KRows,
                v_rows: Slot::VRows,
                pos: StepField::Pos,
                status: StepField::Status,
            },
            BlockOp::Kv8Append {
                k: Slot::KRows,
                v: Slot::VRows,
                k_codes: Plane::K,
                k_scales: Plane::KScales,
                v_codes: Plane::V,
                v_scales: Plane::VScales,
                pos: StepField::Pos,
                status: StepField::Status,
            },
            BlockOp::AttnDecodeQ8 {
                q: Slot::AttnQ,
                proj: Slot::AttnProj,
                k_codes: Plane::K,
                k_scales: Plane::KScales,
                v_codes: Plane::V,
                v_scales: Plane::VScales,
                kv_len: StepField::KvLen,
                part: Slot::AttnPart,
                out: Slot::AttnCore,
                split: AttnSplit::Q8,
            },
        ],
    };
    let mut ops = vec![
        project(Role::AttnQ, 0),
        project(Role::AttnK, a.k_off),
        project(Role::AttnV, a.v_off),
    ];
    ops.extend(core);
    ops.extend(pack_op(
        Phase::AttnPack,
        Slot::AttnCore,
        Slot::XfAttnOut,
        Slot::Q8AttnOut,
        a.q_width,
        layout,
    ));
    ops.push(project_op(
        Phase::AttnOut,
        Role::AttnOutput,
        (Slot::XfAttnOut, Slot::Q8AttnOut, Slot::AttnCore),
        Slot::MixOut,
        0,
        layout,
    ));
    ops
}

/// Every fused-row column the `ProjectQ` launches of `kind` write, as (role, col, rows, ld): the
/// four (three) in-projections tile `DeltaProj`/`AttnProj` and the output projection writes
/// `MixOut` from column 0.
fn native_columns(
    g: &MixerGeometry,
    kind: AttentionKind,
    fuse_inproj: bool,
) -> Vec<(Role, usize, usize, usize)> {
    let (d, a, h) = (&g.delta, &g.attn, g.hidden);
    match kind {
        AttentionKind::DeltaNet if fuse_inproj => {
            vec![(Role::AttnQkv, 0, d.ld, d.ld), (Role::SsmOut, 0, h, h)]
        }
        AttentionKind::DeltaNet => vec![
            (Role::AttnQkv, 0, d.channels, d.ld),
            (Role::AttnGate, d.z_off, d.z_rows, d.ld),
            (Role::SsmBeta, d.beta_off, d.beta_rows, d.ld),
            (Role::SsmAlpha, d.alpha_off, d.alpha_rows, d.ld),
            (Role::SsmOut, 0, h, h),
        ],
        _ => vec![
            (Role::AttnQ, 0, a.q_rows, a.ld),
            (Role::AttnK, a.k_off, a.kv_rows, a.ld),
            (Role::AttnV, a.v_off, a.kv_rows, a.ld),
            (Role::AttnOutput, 0, h, h),
        ],
    }
}

/// The launches of one block for a layer of `kind`, in queue order. With `Tiled` the list is
/// byte-identical to the earlier Tiled lists (pinned by the literal fixture test); with `Native`
/// the `NormPack`/`Project`/`Pack` triples become `NormQ8`/`ProjectQ`/`QuantizeQ8` and the
/// fused-row column offsets are checked first (`check_native_column`), so a bad offset is a typed
/// refusal here at construction, never a launch-time wrapper error inside a token. Under `Int8G32`
/// (Native only, `check_layout_rows`) the attention core becomes the KV8 path (the rows entry, the
/// append, the q8 decode).
pub(super) fn block_ops(
    kind: AttentionKind,
    g: &MixerGeometry,
    layout: WeightLayout,
    kv_mode: KvMode,
) -> Result<Vec<BlockOp>, EngError> {
    // The loader fuses only Native weights, so the op list must make the SAME decision - fusion on
    // another layout would plan a wide qkv the weights cannot satisfy.
    let fuse_inproj =
        crate::family::project::cuda::gdn_fuse_inproj() && layout == WeightLayout::Native;
    block_ops_with(fuse_inproj, kind, g, layout, kv_mode)
}

/// The `block_ops` twin with an explicit fusion flag, so the CPU tests pin both arms without
/// touching the environment.
pub(super) fn block_ops_with(
    fuse_inproj: bool,
    kind: AttentionKind,
    g: &MixerGeometry,
    layout: WeightLayout,
    kv_mode: KvMode,
) -> Result<Vec<BlockOp>, EngError> {
    if layout == WeightLayout::Native {
        for (role, col, rows, ld) in native_columns(g, kind, fuse_inproj) {
            check_native_column(role, col, rows, ld)?;
        }
    }
    let mut ops = vec![prologue(layout)];
    match kind {
        AttentionKind::DeltaNet => ops.extend(delta_ops(g, layout, fuse_inproj)),
        AttentionKind::Full => {
            // The plan owns the refusal (`check_layout_rows`); the builder states it too: the
            // Int8G32 core's `KRows`/`VRows` slots exist in Native plans only, so a Tiled Int8G32
            // list could never be planned.
            if kv_mode == KvMode::Int8G32 && layout == WeightLayout::Tiled {
                return Err(EngError::BadRequest(
                    "step ops: the Int8G32 KV mode takes the native layout; Tiled is the fp16 \
                     control path"
                        .to_owned(),
                ));
            }
            ops.extend(attn_ops(g, layout, kv_mode));
        }
        other => {
            return Err(EngError::Checkpoint(format!(
                "block ops: attention kind {other:?} has no resident mixer"
            )));
        }
    }
    ops.push(epilogue(layout));
    Ok(ops)
}

/// One step of one card's mixer block: a local [`BlockOp`] launch, or one of
/// the two K/V row gathers of a full-attention layer, which sit between the
/// card's prep and the cache append and each consume one ring sequence (the
/// token plan of `multi.rs` counts one `KvGatherK` and one `KvGatherV` per
/// full-attention layer). The gathers are markers: the driver runs them
/// through the peer ring between the neighbouring local launches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum CardOp {
    Local(BlockOp),
    /// This card publishes its K row slice and every card's slice lands in
    /// its full `KRows` row: one `all_gather_row`, one sequence.
    GatherK,
    /// The V half, its own `all_gather_row` and sequence.
    GatherV,
    /// The sum of every card's `Slot::MixOut` partial at the mixer/MoE
    /// boundary, so the epilogue's fused residual add and norm see the
    /// full mixer output: one ring sum, one sequence. Sits between the
    /// block's out projection and the epilogue that consumes the row.
    MixerSum,
}

/// The attention core of one card: the card's projections at its geometry's
/// offsets, the card prep (absolute `k_range` rows into the full row slots),
/// the two gathers, the full-row append of the KV mode, and the decode over
/// the card's query-head run against the replicated cache.
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "staged for the multi-device step, its first production caller"
    )
)]
fn attn_ops_card(
    g: &MixerGeometry,
    k_range: (usize, usize),
    layout: WeightLayout,
    kv_mode: KvMode,
) -> Vec<CardOp> {
    let a = &g.attn;
    let project = |role, col| {
        CardOp::Local(project_op(
            Phase::AttnIn,
            role,
            (Slot::XfMix, Slot::Q8Mix, Slot::NormRow),
            Slot::AttnProj,
            col,
            layout,
        ))
    };
    let append = match kv_mode {
        KvMode::Fp16 => CardOp::Local(BlockOp::KvAppendF16 {
            k: Slot::KRows,
            v: Slot::VRows,
            k_plane: Plane::K,
            v_plane: Plane::V,
            pos: StepField::Pos,
            status: StepField::Status,
        }),
        KvMode::Int8G32 => CardOp::Local(BlockOp::Kv8Append {
            k: Slot::KRows,
            v: Slot::VRows,
            k_codes: Plane::K,
            k_scales: Plane::KScales,
            v_codes: Plane::V,
            v_scales: Plane::VScales,
            pos: StepField::Pos,
            status: StepField::Status,
        }),
    };
    let decode = match kv_mode {
        KvMode::Fp16 => CardOp::Local(BlockOp::AttnDecode {
            q: Slot::AttnQ,
            proj: Slot::AttnProj,
            part: Slot::AttnPart,
            out: Slot::AttnCore,
            kv_len: StepField::KvLen,
        }),
        KvMode::Int8G32 => CardOp::Local(BlockOp::AttnDecodeQ8 {
            q: Slot::AttnQ,
            proj: Slot::AttnProj,
            k_codes: Plane::K,
            k_scales: Plane::KScales,
            v_codes: Plane::V,
            v_scales: Plane::VScales,
            kv_len: StepField::KvLen,
            part: Slot::AttnPart,
            out: Slot::AttnCore,
            split: AttnSplit::Q8,
        }),
    };
    let mut ops = vec![
        project(Role::AttnQ, 0),
        project(Role::AttnK, a.k_off),
        project(Role::AttnV, a.v_off),
        CardOp::Local(BlockOp::AttnPrepCard {
            proj: Slot::AttnProj,
            q: Slot::AttnQ,
            q_norm: Role::AttnQNorm,
            k_norm: Role::AttnKNorm,
            k_rows: Slot::KRows,
            v_rows: Slot::VRows,
            k_range,
            pos: StepField::Pos,
            status: StepField::Status,
        }),
        CardOp::GatherK,
        CardOp::GatherV,
        append,
        decode,
    ];
    ops.extend(
        pack_op(
            Phase::AttnPack,
            Slot::AttnCore,
            Slot::XfAttnOut,
            Slot::Q8AttnOut,
            a.q_width,
            layout,
        )
        .into_iter()
        .map(CardOp::Local),
    );
    ops.push(CardOp::Local(project_op(
        Phase::AttnOut,
        Role::AttnOutput,
        (Slot::XfAttnOut, Slot::Q8AttnOut, Slot::AttnCore),
        Slot::MixOut,
        0,
        layout,
    )));
    ops
}

/// The mixer block of one card of a layer of `kind`, as pure planning: the
/// same prologue, epilogue and op vocabulary as [`block_ops_with`] over the
/// CARD's geometry (its query-head run, its K/V projection rows, its channel
/// segments), with the full-attention core split around the two K/V gathers.
/// A DeltaNet card block is exactly the single-device list over the card
/// geometry (every DeltaNet shape is a geometry field); a full-attention card
/// block is the only place the distributed step departs from the fused
/// single-device prep. Card lists are NATIVE-only: the card's fp16 append and
/// the `KRows`/`VRows` gather slots exist in native plans, and v1 serves
/// multi-device on native weights. The fused input projection is refused -
/// `check_fused_inproj` stops it at start, and the card path states it again
/// so no caller can fuse its way past the shard table's four separate
/// projections.
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "staged for the multi-device step, its first production caller"
    )
)]
pub(super) fn block_ops_card(
    kind: AttentionKind,
    g: &MixerGeometry,
    k_range: (usize, usize),
    layout: WeightLayout,
    kv_mode: KvMode,
) -> Result<Vec<CardOp>, EngError> {
    if layout != WeightLayout::Native {
        return Err(EngError::BadRequest(
            "card block ops: multi-device serving takes the native layout; the gather slots and \
             the card append exist in native plans"
                .to_owned(),
        ));
    }
    for (role, col, rows, ld) in native_columns(g, kind, false) {
        check_native_column(role, col, rows, ld)?;
    }
    let mut ops = vec![CardOp::Local(prologue(layout))];
    match kind {
        AttentionKind::DeltaNet => {
            ops.extend(delta_ops(g, layout, false).into_iter().map(CardOp::Local))
        }
        AttentionKind::Full => ops.extend(attn_ops_card(g, k_range, layout, kv_mode)),
        other => {
            return Err(EngError::Checkpoint(format!(
                "card block ops: attention kind {other:?} has no resident mixer"
            )));
        }
    }
    // The out projection above wrote this card's PARTIAL MixOut; the epilogue folds the row
    // into the residual, so the partials must sum first (the `MixerSum` of `multi.rs`'s
    // token plan).
    ops.push(CardOp::MixerSum);
    ops.push(CardOp::Local(epilogue(layout)));
    Ok(ops)
}
