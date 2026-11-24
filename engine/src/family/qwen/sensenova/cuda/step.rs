//! The trunk decode executor of SenseNova-Loop on the device: one token through the 56
//! visits, every launch an existing resident kernel plus the two Loop kernel sets
//! (`attn_decode_loop`, `mhc_route`). The launch map and the buffer layout live in the
//! lane's `CUDA-PLAN.md` §B3-launch-map; this file is its code.
//!
//! The trunk is the CPU reference's (`super::super::step::token`), visit for visit: mHC
//! visits keep both streams, route twice (attn, ffn) through the router table, and end with
//! `cur = 0.5 * (a + b)`; a visit whose physical layer first enters the mHC range resets the
//! streams from `cur` (a loop revisit of that physical layer resets them again, as the
//! vendor graph does). Plain visits residual-add the branch (fused into the post norm) and
//! the FFN output (a two-buffer rotation, because `row_add` writes a distinct `out`).
//!
//! A loop revisit (the same physical weights) owns its own state, so the GDN recurrent
//! state, the conv rings and the attention caches are per-VISIT, sized once in [`open`].
//! The embedding row is gathered by the caller (the Flash-Next pattern: the table stays on
//! the host) and handed to [`LoopStep::decode`] with the position; the logits row stays on
//! the device.
//!
//! Every buffer is allocated once in [`LoopStep::open`]; a step enqueues its launch list and
//! returns without synchronizing (the caller drains via [`LoopStep::stream`]), so the same
//! list is what a later capture slice replays.

// Staged: the serve-path slice constructs LoopStep; until then the dead-code lint
// fires on every item here, and the expect flips to an error the day it is wired.
#![expect(dead_code, reason = "constructed by the serve wiring slice")]

use std::sync::Arc;

use crate::EngError;
use crate::family::project::cuda::{CudaWeight, CudaWeights};
use crate::kernels::cuda::prefill::swiglu_rows::{SwigluRowsCall, SwigluRowsShape};
use crate::kernels::cuda::resident::add_rmsnorm_q8::AddNormQ8Call;
use crate::kernels::cuda::resident::attn_decode::AttnSplit;
use crate::kernels::cuda::resident::attn_decode_loop::{
    AttnDecodeLoopCall, AttnLoopShape, attn_loop_part_bytes,
};
use crate::kernels::cuda::resident::attn_prep::{
    AttnPrepCall, AttnPrepShape, RopeParams, SlotMode,
};
use crate::kernels::cuda::resident::gdn_conv::{ConvCall, ConvShape};
use crate::kernels::cuda::resident::gdn_delta::{DeltaCall, DeltaShape, DeltaWeights, GDN_EPS};
use crate::kernels::cuda::resident::mhc_route::{MhcAverageCall, MhcMergeCall, MhcRouteCall};
use crate::kernels::cuda::resident::quantize_q8_1::q8_1_bytes;
use crate::kernels::cuda::resident::router::{RouterLogitsCall, RouterShape};
use crate::kernels::cuda::resident::row_add::RowAddCall;
use crate::kernels::cuda::resident::row_scale::RowScaleCall;
use crate::kernels::cuda::resident::step_params::{
    OFFSET_KV_LEN, OFFSET_POS, OFFSET_STATUS, STEP_PARAMS_BYTES, StepParams, ensure_rows,
};
use crate::kernels::cuda::{CudaContext, DeviceBuffer, DeviceRange, DeviceStream, NativeWeight};

use super::super::step::yarn_inv_freq;
use super::super::weights::LoopShape;
use super::super::{LoopPlan, PHYSICAL_LAYERS, RopePlan, TRUNK_LAYERS};

mod prefill;

const F32: usize = size_of::<f32>();
const GROUP: usize = 4;
const ATTN_SCALE: f32 = 0.0625; // 1/sqrt(256)
const MAX_SPLITS: usize = 64;

/// One physical layer's resolved device weights: quantized matrices stay
/// [`NativeWeight`] (the matvec kernels read the stored blocks), the small F32 side tensors
/// ride their raw [`DeviceBuffer`] (the loader copies them as read).
struct Phys<'a> {
    attn_norm: &'a DeviceBuffer,
    post_norm: &'a DeviceBuffer,
    ffn_gate: &'a NativeWeight,
    ffn_up: &'a NativeWeight,
    ffn_down: &'a NativeWeight,
    mixer: Mixer<'a>,
    /// The physical layer is in the mHC range (its routers resolve by part in `route`).
    mhc: bool,
}

enum Mixer<'a> {
    Delta {
        qkvz: &'a NativeWeight,
        beta_alpha: &'a NativeWeight,
        out: &'a NativeWeight,
        conv: &'a DeviceBuffer,
        a: &'a DeviceBuffer,
        dt_bias: &'a DeviceBuffer,
        norm: &'a DeviceBuffer,
    },
    Attn {
        q: &'a NativeWeight,
        k: &'a NativeWeight,
        v: &'a NativeWeight,
        o: &'a NativeWeight,
        q_norm: &'a DeviceBuffer,
        k_norm: &'a DeviceBuffer,
    },
}

/// A piece of the trunk's carried state, for the parity rigs.
#[cfg(all(test, feature = "native-cuda"))]
#[derive(Debug)]
pub(crate) enum Carried {
    Hidden,
    K(usize),
    V(usize),
    Rec(usize),
    Ring(usize),
}

/// The trunk executor: the resolved weights, the per-visit state and the scratch rows.
/// Built once per session; `decode` runs one token.
pub(crate) struct LoopStep<'a> {
    ctx: Arc<CudaContext>,
    stream: DeviceStream,
    weights: &'a CudaWeights,
    phys: Vec<Phys<'a>>,
    head: &'a NativeWeight,
    output_norm: &'a DeviceBuffer,
    shape: LoopShape,
    plan: LoopPlan,
    rope: RopePlan,
    cache_rows: usize,
    /// GDN state and conv rings, one slot per trunk visit (an attention visit's slot is
    /// unused; direct visit indexing keeps the mapping arithmetic-free).
    rec: DeviceBuffer,
    ring: DeviceBuffer,
    /// Attention cache planes, one pair per attention visit, in visit order.
    attn_slot: Vec<usize>,
    k_cache: Vec<DeviceBuffer>,
    v_cache: Vec<DeviceBuffer>,
    // scratch rows
    params: DeviceBuffer,
    resid: [DeviceBuffer; 2],
    cur_ix: usize,
    streams: DeviceBuffer,
    router_normed: DeviceBuffer,
    router_q8: DeviceBuffer,
    router_logits: DeviceBuffer,
    route_cell: DeviceBuffer,
    norm_out: DeviceBuffer,
    norm_q8: DeviceBuffer,
    branch: DeviceBuffer,
    out_q8: DeviceBuffer,
    proj: DeviceBuffer,
    conv_out: DeviceBuffer,
    delta_out: DeviceBuffer,
    gu: DeviceBuffer,
    gu_q8: DeviceBuffer,
    ffn_out: DeviceBuffer,
    q_out: DeviceBuffer,
    part: DeviceBuffer,
    logits: DeviceBuffer,
    ones_router: DeviceBuffer,
    inv_freq: DeviceBuffer,
    /// The prefill scratch, sized for `PREFILL_ROWS` rows.
    rows: prefill::Rows,
}

fn native<'a>(weights: &'a CudaWeights, name: &str) -> Result<&'a NativeWeight, EngError> {
    match weights.get(name) {
        Some(CudaWeight::Native { weight, .. }) => Ok(weight),
        _ => Err(EngError::Checkpoint(format!(
            "the device load is missing {name} in the Native layout ({} tensors held)",
            weights.len()
        ))),
    }
}

/// An F32 tensor the loader copied as read: the conv, `ssm_a`, the biases and every norm.
fn side<'a>(weights: &'a CudaWeights, name: &str) -> Result<&'a DeviceBuffer, EngError> {
    match weights.get(name) {
        Some(CudaWeight::Raw { buffer, .. }) => Ok(buffer),
        _ => Err(EngError::Checkpoint(format!(
            "the device load is missing {name} as a raw F32 tensor"
        ))),
    }
}

pub(super) fn floats(
    buffer: &DeviceBuffer,
    at: usize,
    len: usize,
) -> Result<DeviceRange<'_>, EngError> {
    buffer.range(at * F32, len * F32)
}

pub(super) fn bytes_range(buffer: &DeviceBuffer, len: usize) -> Result<DeviceRange<'_>, EngError> {
    buffer.range(0, len)
}

fn q8_row(k: usize) -> Result<usize, EngError> {
    q8_1_bytes(k, 1).ok_or_else(|| EngError::BadRequest("a q8 row size overflows".to_owned()))
}

pub(super) fn le_f32(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn qkvz_row(shape: &LoopShape) -> usize {
    2 * shape.delta_key_heads * shape.delta_key_dim
        + 2 * shape.delta_value_heads * shape.delta_value_dim
}

fn attn_row(shape: &LoopShape) -> usize {
    shape.heads_q * 2 * shape.head_dim + 2 * shape.heads_kv * shape.head_dim
}

impl<'a> LoopStep<'a> {
    /// Builds the executor over a completed device load. `cache_rows` is the session's
    /// attention cache depth; the GDN state and the conv rings follow the shape.
    pub(crate) fn open(
        ctx: Arc<CudaContext>,
        weights: &'a CudaWeights,
        shape: &LoopShape,
        plan: &LoopPlan,
        rope: &RopePlan,
        cache_rows: usize,
    ) -> Result<Self, EngError> {
        let h = shape.hidden;
        let visits = TRUNK_LAYERS;
        let rec_floats = shape.delta_value_heads * shape.delta_key_dim * shape.delta_value_dim;
        let ring_floats = shape.conv_dim() * (shape.conv_kernel - 1);
        let proj_floats = qkvz_row(shape) + 2 * shape.delta_value_heads.max(attn_row(shape));
        let cache_bytes = cache_rows * shape.heads_kv * shape.head_dim * 2;
        let attn_slot: Vec<usize> = (0..visits)
            .filter(|&il| !plan.recurrent[plan.block(plan.physical(il))])
            .collect();
        // One Phys per PHYSICAL layer (a loop revisit reads the same weights through its
        // own visit state); `recurrent` is block-indexed, so the mixer test goes through
        // `block(p)` exactly as the load template does.
        let phys = (0..PHYSICAL_LAYERS)
            .map(|p| {
                let stem = format!("blk.{}", plan.block(p));
                let mixer = if plan.recurrent[plan.block(p)] {
                    Mixer::Delta {
                        qkvz: native(weights, &format!("{stem}.attn_qkv_gate.weight"))?,
                        beta_alpha: native(weights, &format!("{stem}.ssm_beta_alpha.weight"))?,
                        out: native(weights, &format!("{stem}.ssm_out.weight"))?,
                        conv: side(weights, &format!("{stem}.ssm_conv1d.weight"))?,
                        a: side(weights, &format!("{stem}.ssm_a"))?,
                        dt_bias: side(weights, &format!("{stem}.ssm_dt.bias"))?,
                        norm: side(weights, &format!("{stem}.ssm_norm.weight"))?,
                    }
                } else {
                    Mixer::Attn {
                        q: native(weights, &format!("{stem}.attn_q.weight"))?,
                        k: native(weights, &format!("{stem}.attn_k.weight"))?,
                        v: native(weights, &format!("{stem}.attn_v.weight"))?,
                        o: native(weights, &format!("{stem}.attn_output.weight"))?,
                        q_norm: side(weights, &format!("{stem}.attn_q_norm.weight"))?,
                        k_norm: side(weights, &format!("{stem}.attn_k_norm.weight"))?,
                    }
                };
                Ok(Phys {
                    attn_norm: side(weights, &format!("{stem}.attn_norm.weight"))?,
                    post_norm: side(weights, &format!("{stem}.post_attention_norm.weight"))?,
                    ffn_gate: native(weights, &format!("{stem}.ffn_gate.weight"))?,
                    ffn_up: native(weights, &format!("{stem}.ffn_up.weight"))?,
                    ffn_down: native(weights, &format!("{stem}.ffn_down.weight"))?,
                    mixer,
                    mhc: (plan.mhc_start..plan.mhc_end).contains(&p),
                })
            })
            .collect::<Result<Vec<Phys<'a>>, EngError>>()?;
        let zeros = |floats: usize| le_f32(&vec![0.0; floats]);
        let step = Self {
            stream: ctx.stream()?,
            phys,
            head: native(weights, "output.weight")?,
            output_norm: side(weights, "output_norm.weight")?,
            rec: ctx.upload(&zeros(rec_floats * visits))?,
            ring: ctx.upload(&zeros(ring_floats * visits))?,
            k_cache: (0..attn_slot.len())
                .map(|_| ctx.upload(&vec![0u8; cache_bytes]))
                .collect::<Result<Vec<_>, _>>()?,
            v_cache: (0..attn_slot.len())
                .map(|_| ctx.upload(&vec![0u8; cache_bytes]))
                .collect::<Result<Vec<_>, _>>()?,
            params: ctx.upload(&vec![0u8; STEP_PARAMS_BYTES])?,
            resid: [ctx.upload(&zeros(h))?, ctx.upload(&zeros(h))?],
            cur_ix: 0,
            streams: ctx.upload(&zeros(2 * h))?,
            router_normed: ctx.upload(&zeros(2 * h))?,
            router_q8: ctx.upload(&vec![0u8; q8_row(2 * h)?])?,
            router_logits: ctx.upload(&zeros(8))?,
            route_cell: ctx.upload(&zeros(6))?,
            norm_out: ctx.upload(&zeros(h))?,
            norm_q8: ctx.upload(&vec![0u8; q8_row(h)?])?,
            branch: ctx.upload(&zeros(h))?,
            out_q8: ctx.upload(&vec![0u8; q8_row(h)?])?,
            proj: ctx.upload(&zeros(proj_floats))?,
            conv_out: ctx.upload(&zeros(shape.conv_dim()))?,
            delta_out: ctx.upload(&zeros(shape.delta_value_heads * shape.delta_value_dim))?,
            gu: ctx.upload(&zeros(2 * shape.feed_forward))?,
            gu_q8: ctx.upload(&vec![0u8; q8_row(shape.feed_forward)?])?,
            ffn_out: ctx.upload(&zeros(h))?,
            q_out: ctx.upload(&zeros(shape.heads_q * shape.head_dim))?,
            part: ctx.upload(&vec![0u8; attn_loop_part_bytes(1, GROUP, MAX_SPLITS)?])?,
            logits: ctx.upload(&zeros(shape.vocab))?,
            ones_router: ctx.upload(&le_f32(&vec![1.0; 2 * h]))?,
            inv_freq: ctx.upload(&le_f32(&yarn_inv_freq(rope)))?,
            rows: prefill::open_rows(&ctx, shape, ctx.sm_count() as usize)?,
            shape: shape.clone(),
            plan: plan.clone(),
            rope: rope.clone(),
            cache_rows,
            ctx,
            weights,
            attn_slot,
        };
        step.check_geometry()?;
        Ok(step)
    }

    /// The pinned-model geometry the kernels rely on; a different Loop shape needs this
    /// list revisited, so it fails closed here and not inside a launch.
    fn check_geometry(&self) -> Result<(), EngError> {
        let s = &self.shape;
        let refuse = |what: &str| {
            Err(EngError::BadRequest(format!(
                "the Loop shape is not the pinned geometry: {what}"
            )))
        };
        if s.head_dim != 256 {
            return refuse("head_dim is not 256");
        }
        if s.heads_q != GROUP * s.heads_kv {
            return refuse("the attention is not 4 query heads per kv head");
        }
        if s.delta_key_dim != 128 || s.delta_value_dim != 128 || s.conv_kernel != 4 {
            return refuse("the GDN geometry is not kd=vd=128, ck=4");
        }
        if !s.hidden.is_multiple_of(256) || !s.feed_forward.is_multiple_of(32) {
            return refuse("hidden or feed_forward breaks the norm or swiglu units");
        }
        Ok(())
    }

    /// One token: `embed_row` is the gathered embedding row of the drawn token at `pos`.
    /// Enqueues the trunk and the head; the caller drains and reads [`Self::logits_row`].
    pub(crate) fn decode(&mut self, embed_row: &[f32], pos: usize) -> Result<(), EngError> {
        self.decode_inner(embed_row, pos, None)
    }

    /// The parity rig's twin: also snapshots the residual row after every visit.
    pub(crate) fn decode_traced(
        &mut self,
        embed_row: &[f32],
        pos: usize,
        trace: &mut Vec<Vec<f32>>,
    ) -> Result<(), EngError> {
        self.decode_inner(embed_row, pos, Some(trace))
    }

    fn decode_inner(
        &mut self,
        embed_row: &[f32],
        pos: usize,
        mut trace: Option<&mut Vec<Vec<f32>>>,
    ) -> Result<(), EngError> {
        let h = self.shape.hidden;
        if embed_row.len() != h {
            return Err(EngError::BadRequest(
                "the embed row length is not hidden".to_owned(),
            ));
        }
        ensure_rows(pos, 1, self.cache_rows)?;
        let cur = self.cur_ix;
        // Behind the last step's kernels, which still read both.
        self.resid[cur].write_at_enqueue(&self.stream, 0, &le_f32(embed_row))?;
        let params = StepParams::new(0, pos as i32)?;
        self.params
            .write_at_enqueue(&self.stream, 0, &params.to_bytes())?;
        self.ctx
            .step_begin(&self.stream, bytes_range(&self.params, STEP_PARAMS_BYTES)?)?;

        for il in 0..TRUNK_LAYERS {
            let p = self.plan.physical(il);
            let mhc = self.plan.mhc_active(il);
            if mhc && p == self.plan.mhc_start {
                self.streams_from_cur()?;
            }
            if mhc {
                self.route(p, "hc_attn_fn")?;
                // The branch consumes the routed input the router just wrote, normed
                // with this layer's attn_norm (the CPU's rms_plain(r.input, attn_norm)).
                self.norm_and_q8(
                    floats(&self.router_normed, 0, h)?,
                    None,
                    self.phys[p].attn_norm,
                )?;
            } else {
                self.norm_and_q8(self.resid_range()?, None, self.phys[p].attn_norm)?;
            }
            self.branch(il)?;
            if let Some(trace) = trace.as_deref_mut() {
                self.stream.synchronize()?;
                let h = self.shape.hidden;
                let mut bytes = vec![0u8; self.branch.bytes()];
                self.ctx.download(&self.branch, &mut bytes)?;
                trace.push(
                    bytes
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .take(h)
                        .map(|c| f32::from_le_bytes(*c))
                        .collect(),
                );
            }
            if mhc {
                self.merge_streams(false)?;
                self.route(p, "hc_ffn_fn")?;
                // The ffn norm reads the routed input the second router wrote.
                self.norm_and_q8(
                    floats(&self.router_normed, 0, self.shape.hidden)?,
                    None,
                    self.phys[p].post_norm,
                )?;
                self.ffn(p)?;
                self.merge_streams(true)?;
                let h = self.shape.hidden;
                self.ctx.mhc_average(
                    &self.stream,
                    MhcAverageCall {
                        hidden: h,
                        rows: 1,
                        ld_ab: 2 * h,
                        a: floats(&self.streams, 0, h)?,
                        b: floats(&self.streams, h, h)?,
                        out: self.resid_range()?,
                    },
                )?;
            } else {
                // Fused residual add and post norm: the branch lands in the residual.
                self.norm_and_q8(
                    self.resid_range()?,
                    Some(floats(&self.branch, 0, h)?),
                    self.phys[p].post_norm,
                )?;
                self.ffn(p)?;
                // row_add writes a distinct out: rotate the residual pair. The read
                // plane is the CURRENT residual, not the step's starting plane.
                let next = self.cur_ix ^ 1;
                self.ctx.row_add(
                    &self.stream,
                    RowAddCall {
                        out: floats(&self.resid[next], 0, h)?,
                        x: floats(&self.resid[self.cur_ix], 0, h)?,
                        y: floats(&self.ffn_out, 0, h)?,
                        n: h,
                        flag: None,
                        fused: false,
                    },
                )?;
                self.cur_ix = next;
            }
            if let Some(trace) = trace.as_deref_mut() {
                self.stream.synchronize()?;
                let h = self.shape.hidden;
                let mut bytes = vec![0u8; self.resid[self.cur_ix].bytes()];
                self.ctx.download(&self.resid[self.cur_ix], &mut bytes)?;
                trace.push(
                    bytes
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .take(h)
                        .map(|c| f32::from_le_bytes(*c))
                        .collect(),
                );
            }
        }

        // The output norm and the head.
        let h = self.shape.hidden;
        self.norm_and_q8(self.resid_range()?, None, self.output_norm)?;
        self.ctx.matvec_native(
            &self.stream,
            self.head,
            bytes_range(&self.norm_q8, q8_row(h)?)?,
            floats(&self.logits, 0, self.shape.vocab)?,
            1,
        )?;
        Ok(())
    }

    /// The logits row (device floats); valid after the stream drained.
    pub(crate) fn logits_row(&self) -> &DeviceBuffer {
        &self.logits
    }

    /// The stored block and head weight bytes one decode step reads: every visit's dense
    /// weights (the loop layers once per revisit) and the head.
    #[cfg(all(test, feature = "native-cuda"))]
    pub(super) fn step_weight_bytes(&self) -> usize {
        let mixer_bytes = |mixer: &Mixer<'_>| match mixer {
            Mixer::Delta {
                qkvz,
                beta_alpha,
                out,
                ..
            } => qkvz.bytes() + beta_alpha.bytes() + out.bytes(),
            Mixer::Attn { q, k, v, o, .. } => q.bytes() + k.bytes() + v.bytes() + o.bytes(),
        };
        (0..TRUNK_LAYERS)
            .map(|il| {
                let phys = &self.phys[self.plan.physical(il)];
                phys.ffn_gate.bytes()
                    + phys.ffn_up.bytes()
                    + phys.ffn_down.bytes()
                    + mixer_bytes(&phys.mixer)
            })
            .sum::<usize>()
            + self.head.bytes()
    }

    /// The carried state for the parity rigs, as (piece, buffer, byte offset, byte length):
    /// the hidden row, every attention visit's K and V planes over the first `rows` rows
    /// (fp16), and every GDN visit's record and conv ring (f32).
    #[cfg(all(test, feature = "native-cuda"))]
    pub(crate) fn carried(
        &self,
        rows: usize,
    ) -> Result<Vec<(Carried, &DeviceBuffer, usize, usize)>, EngError> {
        let shape = &self.shape;
        let mut out = Vec::new();
        let mut slot = 0;
        for il in 0..TRUNK_LAYERS {
            if self.attn_slot.get(slot) == Some(&il) {
                let plane = rows * shape.heads_kv * shape.head_dim * 2;
                out.push((Carried::K(il), &self.k_cache[slot], 0, plane));
                out.push((Carried::V(il), &self.v_cache[slot], 0, plane));
                slot += 1;
            } else {
                let rec = shape.delta_value_heads * shape.delta_key_dim * shape.delta_value_dim;
                let ring = shape.conv_dim() * (shape.conv_kernel - 1);
                out.push((Carried::Rec(il), &self.rec, il * rec * 4, rec * 4));
                out.push((Carried::Ring(il), &self.ring, il * ring * 4, ring * 4));
            }
        }
        out.push((
            Carried::Hidden,
            &self.resid[self.cur_ix],
            0,
            shape.hidden * 4,
        ));
        Ok(out)
    }

    /// The executor's stream, for the caller's drain.
    pub(crate) fn stream(&self) -> &DeviceStream {
        &self.stream
    }

    fn resid_range(&self) -> Result<DeviceRange<'_>, EngError> {
        floats(&self.resid[self.cur_ix], 0, self.shape.hidden)
    }

    fn streams_from_cur(&mut self) -> Result<(), EngError> {
        let h = self.shape.hidden;
        let cur = self.resid_range()?;
        // Two copies of cur into the streams (row_scale with 1.0 is the copy).
        self.ctx.row_scale(
            &self.stream,
            RowScaleCall {
                out: floats(&self.streams, 0, h)?,
                x: cur.clone(),
                scale: 1.0,
                m: 1,
                n: h,
            },
        )?;
        self.ctx.row_scale(
            &self.stream,
            RowScaleCall {
                out: floats(&self.streams, h, h)?,
                x: cur,
                scale: 1.0,
                m: 1,
                n: h,
            },
        )
    }

    /// One router pass: the unit-RMS norm of `[a || b]` fused with its q8 quantize, the
    /// router matvec, and the routing kernel (the bias folds into the logits there).
    fn route(&mut self, p: usize, part: &str) -> Result<(), EngError> {
        let h = self.shape.hidden;
        let stem = format!("blk.{}", self.plan.block(p));
        let weight = side(self.weights, &format!("{stem}.{part}.weight"))?;
        let bias = side(self.weights, &format!("{stem}.{part}.bias"))?;
        self.ctx.add_rmsnorm_q8_wide(
            &self.stream,
            AddNormQ8Call {
                n: 2 * h,
                eps: self.shape.eps,
                h: floats(&self.streams, 0, 2 * h)?,
                delta: None,
                weight: floats(&self.ones_router, 0, 2 * h)?,
                norm_out: floats(&self.router_normed, 0, 2 * h)?,
                q8: bytes_range(&self.router_q8, q8_row(2 * h)?)?,
                rows: 1,
            },
        )?;
        // The router matrix is a raw F32 tensor: the split router dot reads it as stored,
        // no quantize and no new kernel (the bias folds into the logits inside mhc_route).
        self.ctx.router_logits_rows(
            &self.stream,
            RouterLogitsCall {
                shape: RouterShape {
                    m: 1,
                    hidden: 2 * h,
                    n_expert: 8,
                    top_k: 1,
                },
                x: floats(&self.router_normed, 0, 2 * h)?,
                w: floats(weight, 0, 8 * 2 * h)?,
                logits: floats(&self.router_logits, 0, 8)?,
            },
        )?;
        self.ctx.mhc_route(
            &self.stream,
            MhcRouteCall {
                hidden: h,
                rows: 1,
                ld_ab: 2 * h,
                pre_scale: self.plan.mhc_pre_scale,
                iterations: self.plan.mhc_iterations,
                fixed_identity: self.plan.mhc_fixed_identity,
                logits: floats(&self.router_logits, 0, 8)?,
                bias: Some(floats(bias, 0, 8)?),
                a: floats(&self.streams, 0, h)?,
                b: floats(&self.streams, h, h)?,
                // The routed input takes the router-normed row's first half (scratch the
                // second half was for); a dedicated buffer is not needed.
                input: floats(&self.router_normed, 0, h)?,
                route: floats(&self.route_cell, 0, 6)?,
            },
        )
    }

    /// One merge of both streams with a branch output: `ffn` picks the branch row
    /// (the mixer's `branch` or `ffn_out`).
    fn merge_streams(&mut self, ffn: bool) -> Result<(), EngError> {
        let h = self.shape.hidden;
        let branch = if ffn {
            floats(&self.ffn_out, 0, h)?
        } else {
            floats(&self.branch, 0, h)?
        };
        self.ctx.mhc_merge(
            &self.stream,
            MhcMergeCall {
                hidden: h,
                rows: 1,
                ld_ab: 2 * h,
                a: floats(&self.streams, 0, h)?,
                b: floats(&self.streams, h, h)?,
                branch,
                route: floats(&self.route_cell, 0, 6)?,
            },
        )
    }

    /// The fused add (optional), RMSNorm and q8 quantize of one row into `norm_q8`.
    fn norm_and_q8(
        &self,
        row: DeviceRange<'_>,
        delta: Option<DeviceRange<'_>>,
        weight: &DeviceBuffer,
    ) -> Result<(), EngError> {
        let h = self.shape.hidden;
        self.ctx.add_rmsnorm_q8_wide(
            &self.stream,
            AddNormQ8Call {
                n: h,
                eps: self.shape.eps,
                h: row,
                delta,
                weight: floats(weight, 0, h)?,
                norm_out: floats(&self.norm_out, 0, h)?,
                q8: bytes_range(&self.norm_q8, q8_row(h)?)?,
                rows: 1,
            },
        )
    }

    /// The mixer of visit `il` into `branch`.
    fn branch(&mut self, il: usize) -> Result<(), EngError> {
        let p = self.plan.physical(il);
        match &self.phys[p].mixer {
            Mixer::Delta { .. } => self.gdn(il, p),
            Mixer::Attn { .. } => self.attn(il, p),
        }
    }

    /// One GDN visit: the fused projection, the causal conv, the delta core, the out matvec.
    fn gdn(&mut self, il: usize, p: usize) -> Result<(), EngError> {
        let s = &self.shape;
        let (kh, kd, vh, vd, ck) = (
            s.delta_key_heads,
            s.delta_key_dim,
            s.delta_value_heads,
            s.delta_value_dim,
            s.conv_kernel,
        );
        let conv_dim = s.conv_dim();
        let qkvz_row_len = qkvz_row(s);
        let ba_len = 2 * vh;
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
        let h = s.hidden;
        // One strided row holds [qkvz | beta_alpha]: the delta kernel reads z, beta and
        // alpha at their offsets of the same row.
        self.ctx.matvec_native(
            &self.stream,
            qkvz,
            bytes_range(&self.norm_q8, q8_row(h)?)?,
            floats(&self.proj, 0, qkvz_row_len)?,
            1,
        )?;
        self.ctx.matvec_native(
            &self.stream,
            beta_alpha,
            bytes_range(&self.norm_q8, q8_row(h)?)?,
            floats(&self.proj, qkvz_row_len, ba_len)?,
            1,
        )?;
        let ring_floats = conv_dim * (ck - 1);
        let ring_at = il * ring_floats;
        self.ctx.gdn_conv(
            &self.stream,
            ConvCall {
                shape: ConvShape {
                    m: 1,
                    channels: conv_dim,
                    ld: qkvz_row_len + ba_len,
                },
                proj: floats(&self.proj, 0, conv_dim)?,
                weights: floats(conv, 0, conv_dim * ck)?,
                state: floats(&self.ring, ring_at, ring_floats)?,
                out: floats(&self.conv_out, 0, conv_dim)?,
            },
        )?;
        let rec_floats = vh * kd * vd;
        let eps = GDN_EPS;
        self.ctx.gdn_delta(
            &self.stream,
            DeltaCall {
                shape: DeltaShape {
                    m: 1,
                    n_k_heads: kh,
                    n_v_heads: vh,
                    conv_ld: conv_dim,
                    ld: qkvz_row_len + ba_len,
                    z_off: conv_dim,
                    beta_off: qkvz_row_len,
                    alpha_off: qkvz_row_len + vh,
                },
                eps,
                conv: floats(&self.conv_out, 0, conv_dim)?,
                proj: floats(&self.proj, 0, qkvz_row_len + ba_len)?,
                weights: DeltaWeights {
                    a: floats(a, 0, vh)?,
                    dt_bias: floats(dt_bias, 0, vh)?,
                    norm_w: floats(norm, 0, vd)?,
                },
                state: floats(&self.rec, il * rec_floats, rec_floats)?,
                out: floats(&self.delta_out, 0, vh * vd)?,
            },
        )?;
        let delta_floats = vh * vd;
        self.ctx.quantize_q8_1(
            &self.stream,
            crate::kernels::cuda::resident::quantize_q8_1::QuantizeCall {
                x: floats(&self.delta_out, 0, delta_floats)?,
                y: bytes_range(&self.out_q8, q8_row(delta_floats)?)?,
                k: delta_floats,
                ncols: 1,
            },
        )?;
        self.ctx.matvec_native(
            &self.stream,
            out,
            bytes_range(&self.out_q8, q8_row(delta_floats)?)?,
            floats(&self.branch, 0, h)?,
            1,
        )
    }

    /// One full-attention visit: the projections, the prep (norms, YaRN rope, cache
    /// append), the gated decode attention, the output matvec.
    fn attn(&mut self, il: usize, p: usize) -> Result<(), EngError> {
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
        let q_rows = hq * 2 * hd;
        let kv_rows = hkv * hd;
        let row_len = attn_row(s);
        self.ctx.matvec_native(
            &self.stream,
            q,
            bytes_range(&self.norm_q8, q8_row(h)?)?,
            floats(&self.proj, 0, q_rows)?,
            1,
        )?;
        self.ctx.matvec_native(
            &self.stream,
            k,
            bytes_range(&self.norm_q8, q8_row(h)?)?,
            floats(&self.proj, q_rows, kv_rows)?,
            1,
        )?;
        self.ctx.matvec_native(
            &self.stream,
            v,
            bytes_range(&self.norm_q8, q8_row(h)?)?,
            floats(&self.proj, q_rows + kv_rows, kv_rows)?,
            1,
        )?;
        let slot = self
            .attn_slot
            .iter()
            .position(|&visit| visit == il)
            .ok_or_else(|| {
                EngError::BadRequest("the visit has no attention cache slot".to_owned())
            })?;
        let (k_cache, v_cache) = (&self.k_cache[slot], &self.v_cache[slot]);
        self.ctx.attn_prep(
            &self.stream,
            AttnPrepCall {
                shape: AttnPrepShape {
                    m: 1,
                    n_head: hq,
                    n_kv_head: hkv,
                    head_dim: hd,
                    n_rot: self.rope.dims,
                    has_gate: true,
                    slot: SlotMode::Position,
                    rows: self.cache_rows,
                    ld: row_len,
                },
                rope: RopeParams {
                    eps: s.eps,
                    rope_base: self.rope.freq_base,
                },
                admit_pos: 0,
                proj: floats(&self.proj, 0, row_len)?,
                q_norm: floats(q_norm, 0, hd)?,
                k_norm: floats(k_norm, 0, hd)?,
                k_cache: bytes_range(k_cache, k_cache.bytes())?,
                v_cache: bytes_range(v_cache, v_cache.bytes())?,
                q: floats(&self.q_out, 0, hq * hd)?,
                pos: self.params_word(OFFSET_POS)?,
                status: self.params_word(OFFSET_STATUS)?,
                inv_freq: Some(floats(&self.inv_freq, 0, self.rope.dims / 2)?),
            },
        )?;
        self.ctx.attn_decode_loop(
            &self.stream,
            AttnDecodeLoopCall {
                shape: AttnLoopShape {
                    m: 1,
                    n_head: hq,
                    n_kv_head: hkv,
                    cache_rows: self.cache_rows,
                    ld: row_len,
                },
                split: AttnSplit::STAGE1,
                scale: ATTN_SCALE,
                q: floats(&self.q_out, 0, hq * hd)?,
                proj: Some(floats(&self.proj, 0, row_len)?),
                k_cache: bytes_range(k_cache, k_cache.bytes())?,
                v_cache: bytes_range(v_cache, v_cache.bytes())?,
                kv_len: self.params_word(OFFSET_KV_LEN)?,
                part: bytes_range(&self.part, self.part.bytes())?,
                out: floats(&self.branch, 0, hq * hd)?,
            },
        )?;
        let branch_floats = hq * hd;
        self.ctx.quantize_q8_1(
            &self.stream,
            crate::kernels::cuda::resident::quantize_q8_1::QuantizeCall {
                x: floats(&self.branch, 0, branch_floats)?,
                y: bytes_range(&self.out_q8, q8_row(branch_floats)?)?,
                k: branch_floats,
                ncols: 1,
            },
        )?;
        self.ctx.matvec_native(
            &self.stream,
            o,
            bytes_range(&self.out_q8, q8_row(branch_floats)?)?,
            floats(&self.branch, 0, h)?,
            1,
        )
    }

    /// The FFN of physical layer `p`: the gate and up matvecs into one strided row, the
    /// SwiGLU (the column-split rows kernel: `swiglu_q8` is one block for the whole row), its
    /// q8 quantize, the down matvec into `ffn_out`.
    fn ffn(&mut self, p: usize) -> Result<(), EngError> {
        let phys = &self.phys[p];
        let (h, ff) = (self.shape.hidden, self.shape.feed_forward);
        self.ctx.matvec_native(
            &self.stream,
            phys.ffn_gate,
            bytes_range(&self.norm_q8, q8_row(h)?)?,
            floats(&self.gu, 0, ff)?,
            1,
        )?;
        self.ctx.matvec_native(
            &self.stream,
            phys.ffn_up,
            bytes_range(&self.norm_q8, q8_row(h)?)?,
            floats(&self.gu, ff, ff)?,
            1,
        )?;
        self.ctx.swiglu_rows(
            &self.stream,
            SwigluRowsCall {
                shape: SwigluRowsShape {
                    rows: 1,
                    n_ff: ff,
                    up_off: ff,
                    ld_in: 2 * ff,
                    ld_out: ff,
                },
                gu: floats(&self.gu, 0, 2 * ff)?,
                act: floats(&self.rows.act, 0, ff)?,
            },
        )?;
        self.ctx.quantize_q8_1(
            &self.stream,
            crate::kernels::cuda::resident::quantize_q8_1::QuantizeCall {
                x: floats(&self.rows.act, 0, ff)?,
                y: bytes_range(&self.gu_q8, q8_row(ff)?)?,
                k: ff,
                ncols: 1,
            },
        )?;
        self.ctx.matvec_native(
            &self.stream,
            phys.ffn_down,
            bytes_range(&self.gu_q8, q8_row(ff)?)?,
            floats(&self.ffn_out, 0, h)?,
            1,
        )
    }

    fn params_word(&self, offset: usize) -> Result<DeviceRange<'_>, EngError> {
        self.params.range(offset, 4)
    }
}
