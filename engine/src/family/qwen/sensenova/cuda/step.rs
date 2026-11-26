//! The trunk executor of SenseNova-Loop on the device: a chunk of token rows through the 56
//! visits. The walk and the chunk mixers are in [`prefill`]; this module holds the step, its
//! state and the one-row mixers a decode step (a chunk of one row) takes.
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
//! Every buffer is allocated once in [`LoopStep::open`], and a decode step works in the first
//! row of the chunk's scratch; a step enqueues its launch list and
//! returns without synchronizing (the caller drains via [`LoopStep::stream`]), so the same
//! list is what a later capture slice replays.

use std::{iter::once, rc::Rc};

use crate::EngError;
use crate::family::project::cuda::CudaWeights;
use crate::family::step::ffn;
use crate::family::step::replay::Replay;
use crate::kernels::cuda::resident::attn_decode::AttnSplit;
use crate::kernels::cuda::resident::attn_decode_loop::{
    AttnDecodeLoopCall, AttnLoopShape, attn_loop_part_bytes,
};
use crate::kernels::cuda::resident::attn_prep::{
    AttnPrepCall, AttnPrepShape, RopeParams, SlotMode,
};
use crate::kernels::cuda::resident::gdn_conv::{ConvCall, ConvShape};
use crate::kernels::cuda::resident::gdn_delta::{DeltaCall, DeltaShape, DeltaWeights, GDN_EPS};
use crate::kernels::cuda::resident::quantize_q8_1::q8_1_bytes;
use crate::kernels::cuda::resident::step_params::{
    OFFSET_KV_LEN, OFFSET_POS, OFFSET_STATUS, STEP_PARAMS_BYTES, StepParams, ensure_rows,
};
use crate::kernels::cuda::{CudaContext, DeviceBuffer, DeviceRange, DeviceStream, NativeWeight};

use super::super::step::yarn_inv_freq;
use super::super::weights::LoopShape;
use super::super::{LoopPlan, PHYSICAL_LAYERS, RopePlan, TRUNK_LAYERS};

mod prefill;

pub(super) use prefill::PREFILL_ROWS;

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
}

impl<'a> Phys<'a> {
    fn ffn(&self) -> ffn::Weights<'a> {
        ffn::Weights {
            gate: self.ffn_gate,
            up: self.ffn_up,
            down: self.ffn_down,
        }
    }
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

/// A piece of the trunk's carried state, for the parity rigs; the visit index is what
/// names a row of their tables, through the derived `Debug` alone.
#[cfg(all(test, feature = "native-cuda"))]
#[expect(
    dead_code,
    reason = "read by the derived Debug, which dead code analysis ignores"
)]
#[derive(Debug)]
pub(crate) enum Carried {
    Hidden,
    K(usize),
    V(usize),
    Rec(usize),
    Ring(usize),
}

/// The trunk executor: the resolved weights, the per-visit state and the scratch rows.
/// Built once per engine; [`Self::reset`] starts a new session on the same buffers, and `decode`
/// runs one token.
pub(crate) struct LoopStep<'a> {
    /// The recorded decode step; first, so it drops before the buffers and the stream it uses.
    replay: Replay<usize>,
    ctx: Rc<CudaContext>,
    stream: DeviceStream,
    weights: &'a CudaWeights,
    phys: Vec<Phys<'a>>,
    head: &'a NativeWeight,
    output_norm: &'a DeviceBuffer,
    shape: LoopShape,
    plan: LoopPlan,
    rope: RopePlan,
    cache_rows: usize,
    /// The projections' route: the f32 decode matvec over f32 rows, or the matvec over q8_1.
    dq: bool,
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
    logits: DeviceBuffer,
    ones_router: DeviceBuffer,
    inv_freq: DeviceBuffer,
    /// The scratch rows of a chunk, sized for `PREFILL_ROWS` rows; a decode step works in
    /// the first row of each.
    rows: prefill::Rows,
}

fn native<'a>(weights: &'a CudaWeights, name: &str) -> Result<&'a NativeWeight, EngError> {
    crate::family::step::rows::native(weights, "sensenova", name)
}

/// An F32 tensor the loader copied as read: the conv, `ssm_a`, the biases and every norm.
fn side<'a>(weights: &'a CudaWeights, name: &str) -> Result<&'a DeviceBuffer, EngError> {
    crate::family::step::rows::raw(weights, "sensenova", name)
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

/// `FIFI_SENSENOVA_DECODE_GEMM=dq` puts the step's projections through the f32 decode matvec
/// (`CudaContext::matvec_dq_native`) over the f32 rows: every weight decoded per value, no
/// q8_1 step. The default keeps the matvec over the q8_1 rows. The route is opt-in and
/// default-off; a weight that is not Q4_K or Q6_K, or a `k` that is not a multiple of 256,
/// stops the step with the matvec's `BadRequest`, and nothing runs in its place. Read once:
/// every projection launch would otherwise take an env lock.
fn decode_dq() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("FIFI_SENSENOVA_DECODE_GEMM").is_ok_and(|v| v == "dq"))
}

/// One weight shape's sweep at one column: its description, then the microseconds of each group
/// on the matvec route and on the dq route.
#[cfg(all(test, feature = "native-cuda"))]
type ShapeSweep = (String, Vec<f64>, Vec<f64>);

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
        ctx: Rc<CudaContext>,
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
            params: ctx.upload(&[0u8; STEP_PARAMS_BYTES])?,
            logits: ctx.upload(&zeros(shape.vocab))?,
            ones_router: ctx.upload(&le_f32(&vec![1.0; 2 * h]))?,
            inv_freq: ctx.upload(&le_f32(&yarn_inv_freq(rope)))?,
            replay: Replay::default(),
            rows: prefill::open_rows(&ctx, shape, ctx.sm_count() as usize)?,
            shape: *shape,
            plan: plan.clone(),
            rope: rope.clone(),
            cache_rows,
            dq: decode_dq(),
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
    #[cfg(all(test, feature = "native-cuda"))]
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
        trace: Option<&mut Vec<Vec<f32>>>,
    ) -> Result<(), EngError> {
        let h = self.shape.hidden;
        if embed_row.len() != h {
            return Err(EngError::BadRequest(
                "the embed row length is not hidden".to_owned(),
            ));
        }
        ensure_rows(pos, 1, self.cache_rows)?;
        if trace.is_some() {
            self.walk(&le_f32(embed_row), 1, pos, trace)?;
            return self.head(0, self.dq);
        }
        self.feed(&le_f32(embed_row), 1, pos)?;
        let mut replay = std::mem::take(&mut self.replay);
        let left = replay.run(&self.ctx, &self.stream, || {
            self.visits(1, pos, None)?;
            self.head(0, self.dq)?;
            Ok(self.rows.cur_ix.get())
        });
        self.replay = replay;
        self.rows.cur_ix.set(left?);
        Ok(())
    }

    /// Whether the last decode step replayed the recorded step.
    #[cfg(test)]
    pub(crate) fn replayed(&self) -> bool {
        self.replay.recorded()
    }

    /// The logits row (device floats); valid after the stream drained.
    pub(crate) fn logits_row(&self) -> &DeviceBuffer {
        &self.logits
    }

    /// Clears the carried state for a new session on the same card: the GDN state, the conv
    /// rings and the attention planes return to the zeros of [`LoopStep::open`] (the planes too,
    /// so a session that ended on a non-finite value leaves nothing a masked read could meet),
    /// and the residual cursor returns to its start. Only the carried state is cleared: a step
    /// overwrites every scratch row before its first read. The fills are enqueued on the step's
    /// stream, behind a step that is still in flight.
    pub(crate) fn reset(&mut self) -> Result<(), EngError> {
        let carried = once(&self.rec)
            .chain(once(&self.ring))
            .chain(self.k_cache.iter())
            .chain(self.v_cache.iter());
        for buffer in carried {
            buffer.zero_at(&self.stream, 0, buffer.bytes())?;
        }
        self.rows.cur_ix.set(0);
        Ok(())
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

    /// Device microbenchmark, not a gate: every matvec the decode walk issues, at `ncols`
    /// activation columns, so the weights are read once for `ncols` tokens the way a batched
    /// step would read them. Returns the microseconds of one sweep for each of `groups`
    /// groups: a group queues `sweeps` sweeps (about 1,825 host launches for five) and
    /// synchronizes once, and its elapsed time is divided by `sweeps`. The first group warms
    /// the card up and is dropped. The inputs are zero and the outputs are never read. The
    /// figure leaves out the step's drain points and its norm and mHC glue kernels. It does
    /// include any device idle where the host launch stream falls behind inside a group,
    /// which is most likely for the small matrices. Above four columns `matvec_native` takes
    /// the vendored MMVQ geometry that the decode, at one column, never takes. With `dq`, every
    /// weight goes through the f32 decode matvec over one f32 row instead, whatever `ncols`
    /// says: the launch the step issues on its dq route.
    #[cfg(all(test, feature = "native-cuda"))]
    pub(super) fn sweep_micros(
        &self,
        ncols: usize,
        sweeps: usize,
        groups: usize,
        dq: bool,
    ) -> Result<Vec<f64>, EngError> {
        self.sweep_list_micros(&self.sweep_weights(), ncols, sweeps, groups, dq)
    }

    /// The same sweep for each distinct weight shape (k, rows, quant) at one column, both
    /// routes: the shape, then the microseconds of each group on the matvec route and on the
    /// dq route.
    #[cfg(all(test, feature = "native-cuda"))]
    pub(super) fn sweep_shape_micros(
        &self,
        sweeps: usize,
        groups: usize,
    ) -> Result<Vec<ShapeSweep>, EngError> {
        use crate::kernels::cuda::resident::matvec_q::QuantType;
        let mut shapes: Vec<(usize, usize, QuantType, Vec<&'a NativeWeight>)> = Vec::new();
        for w in self.sweep_weights() {
            let key = (w.k(), w.n_rows(), w.quant());
            match shapes.iter_mut().find(|s| (s.0, s.1, s.2) == key) {
                Some(s) => s.3.push(w),
                None => shapes.push((key.0, key.1, key.2, vec![w])),
            }
        }
        let mut out = Vec::with_capacity(shapes.len());
        for (k, rows, quant, weights) in shapes {
            let shape = format!("k {k} rows {rows} {quant:?}, {} weights", weights.len());
            let matvec = self.sweep_list_micros(&weights, 1, sweeps, groups, false)?;
            let dq = self.sweep_list_micros(&weights, 1, sweeps, groups, true)?;
            out.push((shape, matvec, dq));
        }
        Ok(out)
    }

    /// Every weight the decode walk's projections read, in walk order.
    #[cfg(all(test, feature = "native-cuda"))]
    fn sweep_weights(&self) -> Vec<&'a NativeWeight> {
        let mut weights = Vec::new();
        for il in 0..TRUNK_LAYERS {
            let phys = &self.phys[self.plan.physical(il)];
            match &phys.mixer {
                Mixer::Delta {
                    qkvz,
                    beta_alpha,
                    out,
                    ..
                } => weights.extend([*qkvz, *beta_alpha, *out]),
                Mixer::Attn { q, k, v, o, .. } => weights.extend([*q, *k, *v, *o]),
            }
            weights.extend([phys.ffn_gate, phys.ffn_up, phys.ffn_down]);
        }
        weights.push(self.head);
        weights
    }

    #[cfg(all(test, feature = "native-cuda"))]
    fn sweep_list_micros(
        &self,
        weights: &[&NativeWeight],
        ncols: usize,
        sweeps: usize,
        groups: usize,
        dq: bool,
    ) -> Result<Vec<f64>, EngError> {
        let (k_max, rows_max) = weights
            .iter()
            .fold((0, 0), |(k, n), w| (k.max(w.k()), n.max(w.n_rows())));
        let x = self.ctx.upload(&vec![0u8; ncols * q8_row(k_max)?])?;
        let x_f32 = self.ctx.upload(&le_f32(&vec![0.0; k_max]))?;
        let y = self.ctx.upload(&le_f32(&vec![0.0; ncols * rows_max]))?;
        let mut out = Vec::with_capacity(groups);
        for _ in 0..=groups {
            let t0 = std::time::Instant::now();
            for _ in 0..sweeps {
                for w in weights {
                    let y_w = y.range(0, ncols * w.n_rows() * F32)?;
                    if dq {
                        self.ctx.matvec_dq_native(
                            &self.stream,
                            w,
                            x_f32.range(0, w.k() * F32)?,
                            y_w,
                        )?;
                    } else {
                        let x_w = x.range(0, ncols * q8_row(w.k())?)?;
                        self.ctx.matvec_native(&self.stream, w, x_w, y_w, ncols)?;
                    }
                }
            }
            self.stream.synchronize()?;
            out.push(t0.elapsed().as_secs_f64() * 1e6 / sweeps as f64);
        }
        out.remove(0);
        Ok(out)
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
            &self.rows.resid[self.rows.cur_ix.get()],
            0,
            shape.hidden * 4,
        ));
        Ok(out)
    }

    /// The step's stream: a read enqueued on it lands behind the step's kernels.
    pub(crate) fn stream(&self) -> &DeviceStream {
        &self.stream
    }

    /// One projection of one row: through the matvec over the row's q8_1 bytes, or on the dq
    /// route through the f32 decode matvec over the f32 row.
    fn project(
        &self,
        weight: &NativeWeight,
        x_q8: DeviceRange<'_>,
        x_f32: DeviceRange<'_>,
        y: DeviceRange<'_>,
    ) -> Result<(), EngError> {
        if self.dq {
            return self.ctx.matvec_dq_native(&self.stream, weight, x_f32, y);
        }
        self.ctx.matvec_native(&self.stream, weight, x_q8, y, 1)
    }

    /// One GDN visit: the fused projection, the causal conv, the delta core, the out matvec.
    fn gdn(&self, il: usize, p: usize) -> Result<(), EngError> {
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
        self.project(
            qkvz,
            bytes_range(&self.rows.norm_q8, q8_row(h)?)?,
            floats(&self.rows.norm_out, 0, h)?,
            floats(&self.rows.proj, 0, qkvz_row_len)?,
        )?;
        self.project(
            beta_alpha,
            bytes_range(&self.rows.norm_q8, q8_row(h)?)?,
            floats(&self.rows.norm_out, 0, h)?,
            floats(&self.rows.proj, qkvz_row_len, ba_len)?,
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
                proj: floats(&self.rows.proj, 0, conv_dim)?,
                weights: floats(conv, 0, conv_dim * ck)?,
                state: floats(&self.ring, ring_at, ring_floats)?,
                out: floats(&self.rows.conv_out, 0, conv_dim)?,
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
                conv: floats(&self.rows.conv_out, 0, conv_dim)?,
                proj: floats(&self.rows.proj, 0, qkvz_row_len + ba_len)?,
                weights: DeltaWeights {
                    a: floats(a, 0, vh)?,
                    dt_bias: floats(dt_bias, 0, vh)?,
                    norm_w: floats(norm, 0, vd)?,
                },
                state: floats(&self.rec, il * rec_floats, rec_floats)?,
                out: floats(&self.rows.delta_out, 0, vh * vd)?,
            },
        )?;
        let delta_floats = vh * vd;
        if !self.dq {
            self.ctx.quantize_q8_1(
                &self.stream,
                crate::kernels::cuda::resident::quantize_q8_1::QuantizeCall {
                    x: floats(&self.rows.delta_out, 0, delta_floats)?,
                    y: bytes_range(&self.rows.q8d, q8_row(delta_floats)?)?,
                    k: delta_floats,
                    ncols: 1,
                },
            )?;
        }
        self.project(
            out,
            bytes_range(&self.rows.q8d, q8_row(delta_floats)?)?,
            floats(&self.rows.delta_out, 0, delta_floats)?,
            floats(&self.rows.mix_out, 0, h)?,
        )
    }

    /// One full-attention visit: the projections, the prep (norms, YaRN rope, cache
    /// append), the gated decode attention, the output matvec.
    fn attn(&self, il: usize, p: usize) -> Result<(), EngError> {
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
        self.project(
            q,
            bytes_range(&self.rows.norm_q8, q8_row(h)?)?,
            floats(&self.rows.norm_out, 0, h)?,
            floats(&self.rows.proj, 0, q_rows)?,
        )?;
        self.project(
            k,
            bytes_range(&self.rows.norm_q8, q8_row(h)?)?,
            floats(&self.rows.norm_out, 0, h)?,
            floats(&self.rows.proj, q_rows, kv_rows)?,
        )?;
        self.project(
            v,
            bytes_range(&self.rows.norm_q8, q8_row(h)?)?,
            floats(&self.rows.norm_out, 0, h)?,
            floats(&self.rows.proj, q_rows + kv_rows, kv_rows)?,
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
                proj: floats(&self.rows.proj, 0, row_len)?,
                q_norm: floats(q_norm, 0, hd)?,
                k_norm: floats(k_norm, 0, hd)?,
                k_cache: bytes_range(k_cache, k_cache.bytes())?,
                v_cache: bytes_range(v_cache, v_cache.bytes())?,
                q: floats(&self.rows.q_out, 0, hq * hd)?,
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
                q: floats(&self.rows.q_out, 0, hq * hd)?,
                proj: Some(floats(&self.rows.proj, 0, row_len)?),
                k_cache: bytes_range(k_cache, k_cache.bytes())?,
                v_cache: bytes_range(v_cache, v_cache.bytes())?,
                kv_len: self.params_word(OFFSET_KV_LEN)?,
                part: bytes_range(&self.rows.part, attn_loop_part_bytes(1, GROUP, MAX_SPLITS)?)?,
                // The dq route's matvec refuses an output over its own input; delta_out is
                // free on an attention visit.
                out: floats(
                    if self.dq {
                        &self.rows.delta_out
                    } else {
                        &self.rows.mix_out
                    },
                    0,
                    hq * hd,
                )?,
            },
        )?;
        let branch_floats = hq * hd;
        if !self.dq {
            self.ctx.quantize_q8_1(
                &self.stream,
                crate::kernels::cuda::resident::quantize_q8_1::QuantizeCall {
                    x: floats(&self.rows.mix_out, 0, branch_floats)?,
                    y: bytes_range(&self.rows.q8a, q8_row(branch_floats)?)?,
                    k: branch_floats,
                    ncols: 1,
                },
            )?;
        }
        self.project(
            o,
            bytes_range(&self.rows.q8a, q8_row(branch_floats)?)?,
            floats(&self.rows.delta_out, 0, branch_floats)?,
            floats(&self.rows.mix_out, 0, h)?,
        )
    }

    fn params_word(&self, offset: usize) -> Result<DeviceRange<'_>, EngError> {
        self.params.range(offset, 4)
    }
}
