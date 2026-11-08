//! The two mixers of a Flash-Next layer on the device: Gated DeltaNet and QSA attention. Each
//! reads `Mixed(Attn)` (the hyper-connection's mixed row) and writes `BlockOut(Attn)`.
//!
//! The in-projections are EXL3 matrices written straight into columns of one projection row
//! (the plan's `DeltaProj` and `AttnProj`), so no copy kernel moves them. The DeltaNet `b` and
//! `a` gates are plain F32 matrices widened at upload; they take the router's strided dot (Qwen3.6's
//! path for the same gates) into the beta and alpha columns of the row.
//!
//! DeltaNet: `qkv | z | b | a` projections, `gdn_conv` over the ring, `gdn_scan_seq` in the
//! Flash-Next form (grouped key heads, `rsqrt(sum + eps)` L2, sigmoid gate) at one token, the
//! output projection.
//!
//! QSA: `q|gate`, `k`, `v` and the indexer projection, `attn_prep` (q and k norms, rope, the fp16
//! K/V append), the indexer (pooled keys, scores, selection of the best blocks), the gated sparse
//! attention over the selected blocks and the open tail, the output projection. Every QSA kernel
//! reads the position from the device word of the step parameters. The DeltaNet scan does not: it
//! takes its start position from the host launch, so only the QSA half of a layer is
//! position-independent.
#![cfg_attr(not(test), allow(dead_code, reason = "staged for the executor"))]

use super::{
    state::StateSlot,
    step::{FlashStep, refuse},
    step_plan::{Site, Slot},
};
use crate::{
    EngError,
    family::{AttentionKind, exl3_pack::layer_name},
    kernels::cuda::{
        DeviceRange,
        prefill::gdn_scan_seq::{
            GDN_SCAN_EPS, GdnForm, GdnScanSeqCall, ScanShape, ScanWeights, scan_bytes,
        },
        resident::{
            attn_prep::{AttnPrepCall, AttnPrepShape, RopeParams, SlotMode},
            gdn_conv::{ConvCall, ConvShape, conv_bytes},
            qsa_attn::{QsaAttnCall, QsaAttnShape},
            qsa_index::{QsaIndexCall, QsaIndexer, QsaPosition, QsaScoreCall, QsaSelectCall},
            router::RouterShape,
            router_rows::RouterLogitsStridedCall,
        },
    },
};

/// The first `bytes` bytes of `range`.
fn head<'r>(range: &DeviceRange<'r>, bytes: usize) -> Result<DeviceRange<'r>, EngError> {
    range.buffer().range(range.offset(), bytes)
}

impl FlashStep<'_> {
    /// The mixer of `layer` for the token at `pos`.
    pub(super) fn mixer(&self, layer: usize, pos: usize) -> Result<(), EngError> {
        match self.geometry.layers.get(layer) {
            Some(AttentionKind::DeltaNet) => self.delta_mixer(layer, pos),
            Some(AttentionKind::Qsa) => self.qsa_mixer(layer, pos),
            other => Err(refuse(format!(
                "layer {layer} has the mixer kind {other:?}"
            ))),
        }
    }

    fn delta_mixer(&self, layer: usize, pos: usize) -> Result<(), EngError> {
        let delta = self.geometry.delta;
        let hidden = self.geometry.hidden;
        let x = self.slot(Slot::Mixed(Site::Attn))?;
        let row = self.slot(Slot::DeltaProj)?;
        let at = |suffix: &str| layer_name(layer, suffix);
        let ld = delta.proj_ld();
        let z_off = delta.conv_dim;
        let beta_off = z_off + delta.value_width();
        let alpha_off = beta_off + delta.value_heads;

        // qkv (n = conv_dim) and z (n = value_width) differ in width, and the grouped launch
        // takes one width per group (exl3.cu group_args), so they stay two launches.
        self.project(
            &at("linear_attn.in_proj_qkv"),
            x,
            Self::columns(&row, 0, delta.conv_dim)?,
        )?;
        self.project(
            &at("linear_attn.in_proj_z"),
            x,
            Self::columns(&row, z_off, delta.value_width())?,
        )?;
        for (name, offset) in [
            ("linear_attn.in_proj_b.weight", beta_off),
            ("linear_attn.in_proj_a.weight", alpha_off),
        ] {
            self.ctx.router_logits_rows_strided(
                self.stream,
                RouterLogitsStridedCall {
                    shape: RouterShape {
                        m: 1,
                        hidden,
                        n_expert: delta.value_heads,
                        top_k: 1,
                    },
                    x,
                    w: self.float(&at(name))?,
                    logits: Self::columns(&row, offset, delta.value_heads)?,
                    logit_stride: ld,
                },
            )?;
        }

        let conv_shape = ConvShape {
            m: 1,
            channels: delta.conv_dim,
            ld,
        };
        let conv_need = conv_bytes(conv_shape)?;
        let ring = self.state_slot(StateSlot::DeltaConv(layer))?;
        let conv_out = self.slot(Slot::DeltaConv)?;
        let conv_weights = self.float(&at("linear_attn.conv1d.weight"))?;
        self.ctx.gdn_conv(
            self.stream,
            ConvCall {
                shape: conv_shape,
                proj: row.buffer().range(row.offset(), conv_need.proj)?,
                weights: conv_weights
                    .buffer()
                    .range(conv_weights.offset(), conv_need.weights)?,
                state: ring.buffer().range(ring.offset(), conv_need.state)?,
                out: conv_out.buffer().range(conv_out.offset(), conv_need.out)?,
            },
        )?;

        let shape = ScanShape {
            m: 1,
            p0: i64::try_from(pos).map_err(|_| refuse("position overflows".into()))?,
            ctx_tokens: self.state_plan.positions(),
            n_k_heads: delta.key_heads,
            n_v_heads: delta.value_heads,
            head_dim: delta.value_dim,
            conv_ld: delta.conv_dim,
            ld,
            z_off,
            beta_off,
            alpha_off,
        };
        let need = scan_bytes(shape)?;
        let rec = self.state_slot(StateSlot::DeltaRec(layer))?;
        let core = self.slot(Slot::DeltaCore)?;
        let (a, dt_bias, norm_w) = (
            self.float(&at("linear_attn.A_log"))?,
            self.float(&at("linear_attn.dt_bias"))?,
            self.float(&at("linear_attn.norm.weight"))?,
        );
        self.ctx.gdn_scan_seq_form(
            self.stream,
            GdnScanSeqCall {
                shape,
                eps: GDN_SCAN_EPS,
                conv: head(&conv_out, need.conv)?,
                proj: head(&row, need.proj)?,
                weights: ScanWeights {
                    a: head(&a, need.a)?,
                    dt_bias: head(&dt_bias, need.dt_bias)?,
                    norm_w: head(&norm_w, need.norm_w)?,
                },
                state: head(&rec, need.state)?,
                o_out: head(&core, need.o_out)?,
                status: None,
            },
            GdnForm::FLASH_NEXT,
        )?;

        self.project(
            &at("linear_attn.out_proj"),
            core,
            self.slot(Slot::BlockOut(Site::Attn))?,
        )
    }

    fn qsa_mixer(&self, layer: usize, pos: usize) -> Result<(), EngError> {
        let qsa = self.geometry.qsa;
        let dims = self.model.qsa.dims();
        let x = self.slot(Slot::Mixed(Site::Attn))?;
        let row = self.slot(Slot::AttnProj)?;
        let at = |suffix: &str| layer_name(layer, suffix);
        let (q_cols, kv_cols) = (2 * qsa.q_width(), qsa.kv_width());
        let k_off = q_cols;
        let v_off = k_off + kv_cols;
        let idx_off = v_off + kv_cols;
        let ld = qsa.proj_ld();
        if idx_off + qsa.idx_proj() != ld {
            return Err(refuse(format!(
                "the QSA projection row is {ld} wide, its columns end at {}",
                idx_off + qsa.idx_proj()
            )));
        }
        // q (2 * q_width) and index (idx_proj) have widths of their own; k and v share
        // kv_width, the one pair the grouped launch's single-width contract allows. One launch
        // instead of two also beats the 12-14 us small-call floor once, not twice.
        self.project(&at("self_attn.q_proj"), x, Self::columns(&row, 0, q_cols)?)?;
        self.project_group(
            &[
                (&at("self_attn.k_proj"), k_off, kv_cols),
                (&at("self_attn.v_proj"), v_off, kv_cols),
            ],
            x,
            &row,
        )?;
        self.project(
            &at("self_attn.indexer.index_qk_proj"),
            x,
            Self::columns(&row, idx_off, qsa.idx_proj())?,
        )?;

        let positions = self.state_plan.positions();
        // A context shorter than the selection budget completes fewer blocks than the budget, and
        // the selection kernel takes a budget of at most the cache's blocks: the same selection.
        let budget_blocks = qsa.budget_blocks().min(qsa.cap_blocks(positions));
        let k_cache = self.state_slot(StateSlot::QsaK(layer))?;
        let v_cache = self.state_slot(StateSlot::QsaV(layer))?;
        let q = self.slot(Slot::AttnQ)?;
        self.ctx.attn_prep(
            self.stream,
            AttnPrepCall {
                shape: AttnPrepShape {
                    m: 1,
                    n_head: qsa.heads_q,
                    n_kv_head: qsa.heads_kv,
                    head_dim: qsa.head_dim,
                    n_rot: dims.rotary_dim as usize,
                    has_gate: true,
                    slot: SlotMode::Position,
                    rows: positions,
                    ld,
                },
                rope: RopeParams {
                    eps: dims.eps,
                    rope_base: dims.theta,
                },
                admit_pos: pos,
                proj: row,
                q_norm: self.float(&at("self_attn.q_norm.weight"))?,
                k_norm: self.float(&at("self_attn.k_norm.weight"))?,
                k_cache,
                v_cache,
                q,
                pos: self.pos_word()?,
                status: self.status_word()?,
            },
        )?;

        let indexer = QsaIndexer {
            heads: qsa.idx_heads,
            dim: qsa.idx_dim,
            rotary_dim: dims.rotary_dim as usize,
            theta: dims.theta,
            eps: dims.eps,
            ratio: qsa.ratio,
            cap_blocks: qsa.cap_blocks(positions),
        };
        let position = || -> Result<QsaPosition<'_>, EngError> {
            Ok(QsaPosition::Device {
                word: self.pos_word()?,
                admit: pos,
            })
        };
        let queries = self.slot(Slot::IndexQueries)?;
        let keys = self.state_slot(StateSlot::QsaIndexKeys(layer))?;
        self.ctx.qsa_index(
            self.stream,
            QsaIndexCall {
                indexer,
                m: 1,
                ld,
                col_off: idx_off,
                position: position()?,
                proj: row,
                q_norm: self.float(&at("self_attn.indexer.q_layernorm.weight"))?,
                k_norm: self.float(&at("self_attn.indexer.k_layernorm.weight"))?,
                queries,
                ring: self.state_slot(StateSlot::QsaRawRing(layer))?,
                keys,
                status: Some(self.status_word()?),
            },
        )?;
        let scores = self.slot(Slot::Scores)?;
        self.ctx.qsa_score(
            self.stream,
            QsaScoreCall {
                indexer,
                m: 1,
                position: position()?,
                queries,
                keys,
                scores,
                status: Some(self.status_word()?),
            },
        )?;
        let chosen = self.slot(Slot::Chosen)?;
        self.ctx.qsa_select(
            self.stream,
            QsaSelectCall {
                indexer,
                m: 1,
                budget_blocks,
                position: position()?,
                scores,
                blocks: chosen,
                status: Some(self.status_word()?),
            },
        )?;
        let core = self.slot(Slot::AttnCore)?;
        self.ctx.qsa_attn(
            self.stream,
            QsaAttnCall {
                shape: QsaAttnShape {
                    m: 1,
                    n_head: qsa.heads_q,
                    n_kv_head: qsa.heads_kv,
                    rows: positions,
                    ratio: qsa.ratio,
                    budget_blocks,
                    max_splits: self.options.qsa_splits,
                },
                ld,
                scale: 1.0 / (qsa.head_dim as f32).sqrt(),
                position: position()?,
                q,
                proj: row,
                k_cache,
                v_cache,
                chosen,
                part: self.slot(Slot::AttnPart)?,
                out: core,
                status: Some(self.status_word()?),
            },
        )?;
        self.project(
            &at("self_attn.o_proj"),
            core,
            self.slot(Slot::BlockOut(Site::Attn))?,
        )
    }
}
