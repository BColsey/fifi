//! The launches of one mixer block of a K-row verify window as DATA, and the layout of the small
//! device buffer the K-row path owns. Pure host code, like `mixer_ops.rs`: the device runner
//! (`cuda/krow_step.rs`) executes these lists, and the CPU tests check them against the K-row plan.
//!
//! A K-row block differs from the one-row block in where the projections write. The EXL3 rows
//! kernel writes each role's `K` output rows tight, `[K][n]` f32. The kernels downstream read ONE
//! row of `ld` floats with the roles at fixed columns (the delta rule reads z, beta and alpha of
//! one `proj` row; `attn_prep` and `attn_decode` read q, k and v of one row). So:
//!
//! * a DeltaNet layer projects qkv and z into tight regions of the K-row buffer; the conv kernel
//!   reads the tight qkv rows in place through its own row stride; beta and alpha go straight into
//!   the `[K][ld]` row of `DeltaProj` through the strided router launch; one pack kernel copies z
//!   into that row;
//! * an attention layer projects q, k and v into tight regions and one pack kernel interleaves
//!   them into the `[K][ld]` row of `AttnProj`.
//!
//! Every copy is a bit-exact word move, so each row's bytes equal the one-row projections writing
//! the same columns.

use super::{
    mixer_geometry::MixerGeometry,
    state_layout::align_up,
    step_plan::{MAX_STEP_ROWS, Phase, Slot},
};
use crate::{
    EngError,
    family::{AttentionKind, gguf_model::Role},
    kernels::cuda::resident::{argmax_rows::window_result_bytes, step_params::RESULT_PAIR_BYTES},
};

const F32: usize = size_of::<f32>();

/// A region of the K-row buffer that holds tight rows of an EXL3 projection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Tight {
    /// `[K][channels]`: the q, k and v columns of a DeltaNet layer (the conv input).
    Qkv,
    /// `[K][z_rows]`: the z columns of a DeltaNet layer.
    Z,
    /// `[K][q_rows]`: the q (and gate) columns of an attention layer.
    Q,
    /// `[K][kv_rows]`: the k columns of an attention layer.
    K,
    /// `[K][kv_rows]`: the v columns of an attention layer.
    V,
}

/// Where a block op reads or writes a row-major f32 block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Place {
    /// A slot of the K-row plan.
    Slot(Slot),
    /// A tight region of the K-row buffer.
    Tight(Tight),
}

/// One source of a pack launch: a tight region copied to column `col` of the destination rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct PackPart {
    pub(super) src: Tight,
    pub(super) n: usize,
    pub(super) col: usize,
}

/// One launch of a K-row mixer block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum RowsBlockOp {
    /// `resid += delta` (when given), then the normalized row into `norm_out` and the (unused)
    /// pack into `xf`, for the `K` rows.
    NormPack {
        phase: Phase,
        weight: Role,
        delta: Option<Slot>,
        xf: Slot,
        norm_out: Slot,
    },
    /// One EXL3 projection of the `K` rows of `x` into the tight rows of `y`.
    Project {
        phase: Phase,
        role: Role,
        x: Place,
        y: Place,
    },
    /// One plain F32 gate (beta or alpha) of the `K` rows of `x`, row `t`'s values written at
    /// column `col` of row `t` of the `ld`-wide rows of `y`.
    Gate {
        phase: Phase,
        role: Role,
        x: Slot,
        y: Slot,
        col: usize,
    },
    /// The pack of `parts` into the `ld`-wide rows of `dst`.
    Pack {
        phase: Phase,
        dst: Slot,
        ld: usize,
        parts: Vec<PackPart>,
    },
    /// The causal conv of the tight qkv rows into `out`; row `t < K - 1`'s state goes to rollback
    /// slot `t`.
    Conv {
        proj: Tight,
        out: Slot,
        weights: Role,
    },
    /// The gated delta rule of `conv` and the z, beta and alpha columns of `proj` into `out`; row
    /// `t < K - 1`'s state goes to rollback slot `t`.
    Delta {
        conv: Slot,
        proj: Slot,
        out: Slot,
        a: Role,
        dt_bias: Role,
        norm: Role,
    },
    /// Attention prep of the `K` rows: `proj` into `q`, and row `t`'s K and V into cache cell
    /// `pos + t`.
    AttnPrep {
        proj: Slot,
        q: Slot,
        q_norm: Role,
        k_norm: Role,
    },
    /// Decode attention of the `K` rows: row `t` sees the cells below `kv_len + t`.
    AttnDecode {
        q: Slot,
        proj: Slot,
        part: Slot,
        out: Slot,
    },
}

impl RowsBlockOp {
    #[cfg(test)]
    pub(super) fn phase(&self) -> Phase {
        match *self {
            Self::NormPack { phase, .. }
            | Self::Project { phase, .. }
            | Self::Gate { phase, .. }
            | Self::Pack { phase, .. } => phase,
            Self::Conv { .. } => Phase::DeltaConv,
            Self::Delta { .. } => Phase::DeltaRule,
            Self::AttnPrep { .. } => Phase::AttnPrep,
            Self::AttnDecode { .. } => Phase::AttnCore,
        }
    }

    /// What the launch reads (the residual stream and the weights are not places).
    pub(super) fn reads(&self) -> Vec<Place> {
        match self {
            Self::NormPack { delta, .. } => delta.iter().map(|slot| Place::Slot(*slot)).collect(),
            Self::Project { x, .. } => vec![*x, Place::Slot(Slot::Exl3Flag)],
            Self::Gate { x, .. } => vec![Place::Slot(*x)],
            Self::Pack { parts, .. } => parts.iter().map(|part| Place::Tight(part.src)).collect(),
            Self::Conv { proj, .. } => vec![Place::Tight(*proj)],
            Self::Delta { conv, proj, .. } => vec![Place::Slot(*conv), Place::Slot(*proj)],
            Self::AttnPrep { proj, .. } => vec![Place::Slot(*proj)],
            Self::AttnDecode { q, proj, .. } => vec![Place::Slot(*q), Place::Slot(*proj)],
        }
    }

    /// What the launch writes.
    pub(super) fn writes(&self) -> Vec<Place> {
        match self {
            Self::NormPack { xf, norm_out, .. } => {
                vec![Place::Slot(*xf), Place::Slot(*norm_out)]
            }
            Self::Project { y, .. } => vec![*y, Place::Slot(Slot::Exl3Flag)],
            Self::Gate { y, .. } => vec![Place::Slot(*y)],
            Self::Pack { dst, .. } => vec![Place::Slot(*dst)],
            Self::Conv { out, .. } | Self::Delta { out, .. } => vec![Place::Slot(*out)],
            Self::AttnPrep { q, .. } => vec![Place::Slot(*q)],
            Self::AttnDecode { part, out, .. } => vec![Place::Slot(*part), Place::Slot(*out)],
        }
    }
}

/// The op lists of both layer kinds.
pub(super) struct RowsKindOps {
    delta: Vec<RowsBlockOp>,
    attn: Vec<RowsBlockOp>,
}

impl RowsKindOps {
    pub(super) fn new(g: &MixerGeometry) -> Self {
        Self {
            delta: delta_rows_ops(g),
            attn: attn_rows_ops(g),
        }
    }

    pub(super) fn of(&self, kind: AttentionKind) -> Result<&[RowsBlockOp], EngError> {
        match kind {
            AttentionKind::DeltaNet => Ok(&self.delta),
            AttentionKind::Full => Ok(&self.attn),
            other => Err(EngError::Checkpoint(format!(
                "K-row mixer: attention kind {other:?} has no resident mixer"
            ))),
        }
    }

    /// Every op of both lists.
    pub(super) fn all(&self) -> impl Iterator<Item = &RowsBlockOp> {
        self.delta.iter().chain(self.attn.iter())
    }
}

fn prologue() -> RowsBlockOp {
    RowsBlockOp::NormPack {
        phase: Phase::MixPack,
        weight: Role::AttnNorm,
        delta: None,
        xf: Slot::XfMix,
        norm_out: Slot::NormRow,
    }
}

fn epilogue() -> RowsBlockOp {
    RowsBlockOp::NormPack {
        phase: Phase::AddNorm,
        weight: Role::PostAttentionNorm,
        delta: Some(Slot::MixOut),
        xf: Slot::XfMoe,
        norm_out: Slot::NormRow,
    }
}

fn project(phase: Phase, role: Role, y: Tight) -> RowsBlockOp {
    RowsBlockOp::Project {
        phase,
        role,
        x: Place::Slot(Slot::NormRow),
        y: Place::Tight(y),
    }
}

fn delta_rows_ops(g: &MixerGeometry) -> Vec<RowsBlockOp> {
    let d = &g.delta;
    vec![
        prologue(),
        project(Phase::DeltaIn, Role::AttnQkv, Tight::Qkv),
        project(Phase::DeltaIn, Role::AttnGate, Tight::Z),
        RowsBlockOp::Gate {
            phase: Phase::DeltaIn,
            role: Role::SsmBeta,
            x: Slot::NormRow,
            y: Slot::DeltaProj,
            col: d.beta_off,
        },
        RowsBlockOp::Gate {
            phase: Phase::DeltaIn,
            role: Role::SsmAlpha,
            x: Slot::NormRow,
            y: Slot::DeltaProj,
            col: d.alpha_off,
        },
        RowsBlockOp::Pack {
            phase: Phase::DeltaIn,
            dst: Slot::DeltaProj,
            ld: d.ld,
            parts: vec![PackPart {
                src: Tight::Z,
                n: d.z_rows,
                col: d.z_off,
            }],
        },
        RowsBlockOp::Conv {
            proj: Tight::Qkv,
            out: Slot::DeltaConv,
            weights: Role::SsmConv1d,
        },
        RowsBlockOp::Delta {
            conv: Slot::DeltaConv,
            proj: Slot::DeltaProj,
            out: Slot::DeltaCore,
            a: Role::SsmA,
            dt_bias: Role::SsmDtBias,
            norm: Role::SsmNorm,
        },
        RowsBlockOp::Project {
            phase: Phase::DeltaOut,
            role: Role::SsmOut,
            x: Place::Slot(Slot::DeltaCore),
            y: Place::Slot(Slot::MixOut),
        },
        epilogue(),
    ]
}

fn attn_rows_ops(g: &MixerGeometry) -> Vec<RowsBlockOp> {
    let a = &g.attn;
    vec![
        prologue(),
        project(Phase::AttnIn, Role::AttnQ, Tight::Q),
        project(Phase::AttnIn, Role::AttnK, Tight::K),
        project(Phase::AttnIn, Role::AttnV, Tight::V),
        RowsBlockOp::Pack {
            phase: Phase::AttnIn,
            dst: Slot::AttnProj,
            ld: a.ld,
            parts: vec![
                PackPart {
                    src: Tight::Q,
                    n: a.q_rows,
                    col: 0,
                },
                PackPart {
                    src: Tight::K,
                    n: a.kv_rows,
                    col: a.k_off,
                },
                PackPart {
                    src: Tight::V,
                    n: a.kv_rows,
                    col: a.v_off,
                },
            ],
        },
        RowsBlockOp::AttnPrep {
            proj: Slot::AttnProj,
            q: Slot::AttnQ,
            q_norm: Role::AttnQNorm,
            k_norm: Role::AttnKNorm,
        },
        RowsBlockOp::AttnDecode {
            q: Slot::AttnQ,
            proj: Slot::AttnProj,
            part: Slot::AttnPart,
            out: Slot::AttnCore,
        },
        RowsBlockOp::Project {
            phase: Phase::AttnOut,
            role: Role::AttnOutput,
            x: Place::Slot(Slot::AttnCore),
            y: Place::Slot(Slot::MixOut),
        },
        epilogue(),
    ]
}

/// Where each region of the K-row buffer starts and how many bytes a window of the longest length
/// uses; a window of `k` rows uses the first `k` rows of each. Every region starts at a multiple of
/// 256 bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Region {
    pub(super) offset: usize,
    /// Bytes of one row of the region.
    pub(super) row_bytes: usize,
}

impl Region {
    /// Bytes of `rows` rows.
    pub(super) fn bytes(&self, rows: usize) -> usize {
        rows * self.row_bytes
    }
}

/// The layout of the K-row buffer: the window ids, the result table and the tight regions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct KrowIoLayout {
    /// `i32 [K]`: the tokens of the window, read by the embedding.
    pub(super) ids: Region,
    /// The result table of the window: `K` pairs, the window word and a pad word
    /// ([`window_result_bytes`]). Its `row_bytes` is one pair.
    pub(super) result: Region,
    pub(super) qkv: Region,
    pub(super) z: Region,
    pub(super) q: Region,
    pub(super) k: Region,
    pub(super) v: Region,
    /// Bytes of the whole buffer.
    pub(super) total: usize,
}

impl KrowIoLayout {
    /// The layout for windows of up to `rows` rows (2 to 8) over the geometry `g`.
    pub(super) fn new(g: &MixerGeometry, rows: usize) -> Result<Self, EngError> {
        if !(2..=MAX_STEP_ROWS).contains(&rows) {
            return Err(EngError::BadRequest(format!(
                "K-row buffer: a window has 2 to {MAX_STEP_ROWS} rows, got {rows}"
            )));
        }
        let overflow = || EngError::MemoryPlan("K-row buffer: a size overflows".to_owned());
        let mut next = 0usize;
        let mut place = |row_bytes: usize, total: usize| -> Result<Region, EngError> {
            let region = Region {
                offset: next,
                row_bytes,
            };
            next = align_up(next.checked_add(total).ok_or_else(overflow)?).ok_or_else(overflow)?;
            Ok(region)
        };
        let tight = |width: usize| width.checked_mul(F32).ok_or_else(overflow);
        let ids = place(size_of::<i32>(), rows * size_of::<i32>())?;
        let result = place(RESULT_PAIR_BYTES, window_result_bytes(rows))?;
        let qkv_row = tight(g.delta.channels)?;
        let qkv = place(qkv_row, rows * qkv_row)?;
        let z_row = tight(g.delta.z_rows)?;
        let z = place(z_row, rows * z_row)?;
        let q_row = tight(g.attn.q_rows)?;
        let q = place(q_row, rows * q_row)?;
        let kv_row = tight(g.attn.kv_rows)?;
        let k = place(kv_row, rows * kv_row)?;
        let v = place(kv_row, rows * kv_row)?;
        Ok(Self {
            ids,
            result,
            qkv,
            z,
            q,
            k,
            v,
            total: next,
        })
    }

    /// The region of a tight place.
    pub(super) fn tight(&self, which: Tight) -> Region {
        match which {
            Tight::Qkv => self.qkv,
            Tight::Z => self.z,
            Tight::Q => self.q,
            Tight::K => self.k,
            Tight::V => self.v,
        }
    }
}

/// The one row count of the window tail: `argmax_rows` and `window_status_fold` each take their
/// own `m`, and a fold over more rows than the argmax wrote would read stale pairs. The step
/// builds both calls from one count and this check refuses a pair that disagrees.
///
/// It is a guard by construction, not a refusal before the launch: it runs when the argmax op
/// runs, after the layers were enqueued, and the two calls cannot disagree because they take the
/// same `k`. A mismatch would therefore fail the window after its launch (the window fails
/// closed). Building the tail calls before the first launch, so that a mismatch is a clean
/// refusal, is a later change.
pub(super) fn check_tail_rows(argmax_rows: usize, fold_rows: usize) -> Result<(), EngError> {
    if argmax_rows != fold_rows {
        return Err(EngError::BadRequest(format!(
            "K-row window tail: argmax_rows takes {argmax_rows} rows and window_status_fold {fold_rows}"
        )));
    }
    Ok(())
}

#[cfg(test)]
#[path = "krow_ops_tests.rs"]
mod tests;
