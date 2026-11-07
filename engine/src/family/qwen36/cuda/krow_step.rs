//! The K-row verify window of the step (`qwen36.draft_verify = krow`): `K` rows (2 to 8) enqueued as
//! one launch list on the owned stream, one synchronization, one small read of the result table,
//! and the logits of the `K` rows. This body runs eagerly (no graph).
//!
//! The list is `step_ops::token_ops_rows`: the one-row body with every row-independent kernel
//! launched once for all rows (`m = K`), the routed experts as one chain of launches per row on
//! row-offset ranges, the DeltaNet kernels storing the state after rows `0..K-1` straight into the
//! rollback slots, `argmax_rows` and `window_status_fold` in place of the one-row argmax, and the
//! advance by `K`. Row `t` therefore sees the state after rows `0..t` and nothing else, and its
//! bytes equal the one-row step's.
//!
//! Arming. `enable_krow` runs once, after the device rollback slots exist and before the first
//! token. It checks the step against the one configuration the K-row path is built for, allocates
//! the small buffer (window ids, result table, tight projection rows) and keeps the op lists. The
//! activation arena of the window is `StepResources::krow` (built before the step).
//!
//! A window. Failure classes:
//!
//! * before anything is written or enqueued (the token check, the capacity check, the merged-block
//!   re-check, a bad write of the start block) the state decides what happened, as for one row;
//! * a failure of the first launch takes the step back (`cancel_enqueue`), as for one row;
//! * every other failure, and any non-zero status of a row or of the window, is a failure after
//!   the launch: the window fails closed whatever the host counters say.
//!
//! What a later change to read the logits on the device must keep: the logits stay in the arena
//! until `read_krow_rows` copies them (the only reader); the result table (pairs and window word)
//! is one read of `8 K + 8` bytes; the window ids are written once and never read by the host;
//! nothing runs on the host between the rows.

use super::super::{
    exl3_moe::{Exl3MoeBufs, Scratch, run_exl3_moe_rows},
    krow_ops::{KrowIoLayout, Place, RowsBlockOp, RowsKindOps, check_tail_rows},
    mixer::tail_extent,
    mixer_shapes::{attn_decode_shape, attn_prep_shape, delta_shape, plain_shape},
    mixer_weights::LayerWeights,
    moe::{Launch, MoeDims, RouteBufs},
    resources::{EmbedWeight, HeadWeight, f32_bytes},
    state::{StepField, rollback::SnapPart},
    state_layout::{Plane, Word},
    step_ops::{RowsOp, StepOp, final_resid, resid_in, resid_out, token_ops_rows},
    step_plan::Slot,
    verdict::{FailureAction, action_for},
};
use super::QwenResidentStep;
#[cfg(test)]
use crate::family::speculate::d1_oracle::Fault;
use crate::{
    EngError,
    family::{
        project::cuda::{WeightLayout, exl3::Exl3Io},
        qwen36::{
            krow::{RowStatus, RowsFacts, check_rows, rows_verdict},
            window_logic::{RowsFailure, RowsOut},
        },
    },
    kernels::cuda::{
        DeviceBuffer, DeviceRange,
        resident::{
            add_norm_pack::AddNormCall,
            argmax_rows::{ArgmaxRowsCall, WindowFoldCall, WindowResult, window_result_bytes},
            attn_decode::{AttnDecodeCall, AttnShape, AttnSpan, AttnSplit},
            attn_prep::{AttnPrepCall, AttnPrepShape, RopeParams},
            embed_bf16::EmbedBf16Call,
            gdn_conv::{ConvCall, ConvShape, SnapRows},
            gdn_delta::{DeltaCall, DeltaShape, DeltaWeights},
            pack_rows::{PackRowsCall, PackSegment},
            router::RouterShape,
            router_rows::RouterLogitsStridedCall,
            step_params::{RESULT_PAIR_BYTES, StepParams},
        },
    },
};

const F32: usize = size_of::<f32>();

fn refuse(detail: &str) -> EngError {
    EngError::BadRequest(format!("draft_verify krow: {detail}"))
}

fn not_armed() -> EngError {
    refuse("the K-row path is not armed on this step")
}

/// What the K-row path owns on a step; built by [`QwenResidentStep::enable_krow`].
pub(super) struct KrowRuntime {
    /// Window ids, result table and tight projection rows.
    io: DeviceBuffer,
    layout: KrowIoLayout,
    ops: RowsKindOps,
    /// The longest window the buffers and the slots serve.
    rows: usize,
    /// Host copy of the logits of the last window, `rows` rows of the head width.
    logits: Vec<u8>,
    #[cfg(test)]
    test: TestState,
}

/// Test builds only: the switches and counters of the D1 harness.
#[cfg(test)]
struct TestState {
    /// The K-row path is switched off: windows run the eager loop on this armed step.
    path_off: bool,
    /// K-row windows that completed.
    windows: u64,
    /// Return empty logits rows (the tap that a later device-side argmax keeps).
    skip_logits: bool,
    injection: Option<Injection>,
    /// The result table of the last window that got as far as the read, before any verdict.
    last_table: Option<WindowResult>,
}

/// Test builds only: a value made non-finite inside the body of the next K-row window.
#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Injection {
    /// Element 0 of logits row `j` after the head and before the argmax: no other row can see it.
    LogitsNan(usize),
    /// Element 0 of the embedding of row `j`: the NaN spreads to the rows after `j` only.
    HiddenNan(usize),
}

/// What the ops of one block share.
struct RowsRun<'r> {
    weights: &'r LayerWeights<'r>,
    ordinal: usize,
    resid: DeviceRange<'r>,
    k: usize,
    admit_pos: usize,
}

impl<'a> QwenResidentStep<'a> {
    /// Arms the K-row path. Call it after `request_krow_verify` and `enable_device_rollback`,
    /// before the first token. Refused, with the step untouched, unless: the arming refusals of
    /// `krow.rs` hold, the resources hold the K-row arena, the KV is the fp16 resident plane, the
    /// step is neither tiered nor in the per-op synchronization mode, and the embedding and the
    /// head are the pack's bf16 table and EXL3 matrix.
    pub(crate) fn enable_krow(&mut self) -> Result<(), EngError> {
        if !self.krow_wanted || self.krow.is_some() || self.started {
            return Err(refuse(
                "arm it once, after request_krow_verify and before the first token",
            ));
        }
        let rows = self
            .rollback
            .as_ref()
            .map(|slots| slots.plan().rows)
            .ok_or_else(|| refuse("needs the device rollback slots"))?;
        self.check_krow_arming(rows)?;
        self.check_krow_configuration()?;
        let res: &'a super::StepResources = self.res;
        let krow = res
            .krow
            .as_ref()
            .ok_or_else(|| refuse("the step resources hold no K-row arena"))?;
        let ops = RowsKindOps::new(&res.geometry);
        for op in ops.all() {
            for place in op.reads().into_iter().chain(op.writes()) {
                if let Place::Slot(slot) = place {
                    krow.plan
                        .index(slot)
                        .ok_or_else(|| refuse(&format!("the K-row plan has no slot {slot:?}")))?;
                }
            }
        }
        let layout = KrowIoLayout::new(&res.geometry, rows)?;
        let io = self.context.allocate(layout.total)?;
        let logits = vec![
            0;
            f32_bytes(res.dims.shape.vocab)?
                .checked_mul(rows)
                .ok_or_else(|| {
                    EngError::MemoryPlan("K-row window: the logits copy overflows".to_owned())
                })?
        ];
        self.krow = Some(KrowRuntime {
            io,
            layout,
            ops,
            rows,
            logits,
            #[cfg(test)]
            test: TestState {
                path_off: false,
                windows: 0,
                skip_logits: false,
                injection: None,
                last_table: None,
            },
        });
        Ok(())
    }

    /// The configuration the K-row body is built for.
    fn check_krow_configuration(&self) -> Result<(), EngError> {
        if self.state.layout().kv.mode != super::KvMode::Fp16 {
            return Err(refuse(
                "runs the fp16 KV cache; the int8 cache is not built",
            ));
        }
        if self.tiered_mode
            || self.sync_each
            || self.forked_experts
            || self.layout != WeightLayout::Exl3
        {
            return Err(refuse(
                "needs the plain EXL3 step: not tiered, not in per-op synchronization, no forked experts",
            ));
        }
        if !matches!(self.globals.embed, EmbedWeight::Bf16(_))
            || !matches!(self.globals.head, HeadWeight::Exl3(_))
        {
            return Err(refuse("needs the bf16 embedding table and the EXL3 head"));
        }
        Ok(())
    }

    /// Whether a window of `rows` rows runs the K-row body: armed, and a K-row length. (Test
    /// builds can switch the path off to run the eager loop on the same step.)
    pub(crate) fn krow_runs(&self, rows: usize) -> bool {
        let Some(runtime) = &self.krow else {
            return false;
        };
        #[cfg(test)]
        if runtime.test.path_off {
            return false;
        }
        crate::family::qwen36::krow::uses_krow(rows) && rows <= runtime.rows
    }

    fn krow_runtime(&self) -> Result<&KrowRuntime, EngError> {
        self.krow.as_ref().ok_or_else(not_armed)
    }

    /// The K-row arena's view of `slot`.
    fn kview(&self, slot: Slot) -> Result<DeviceRange<'a>, EngError> {
        let res: &'a super::StepResources = self.res;
        let krow = res.krow.as_ref().ok_or_else(not_armed)?;
        let index = krow
            .plan
            .index(slot)
            .ok_or_else(|| refuse(&format!("the K-row plan has no slot {slot:?}")))?;
        krow.arena.tensor(index)
    }

    /// The first `k` rows of a block op's place.
    fn place(&self, place: Place, k: usize) -> Result<DeviceRange<'_>, EngError> {
        match place {
            Place::Slot(slot) => self.kview(slot),
            Place::Tight(which) => {
                let runtime = self.krow_runtime()?;
                let region = runtime.layout.tight(which);
                runtime.io.range(region.offset, region.bytes(k))
            }
        }
    }

    /// Runs one K-row window. See the module doc for the failure classes.
    pub(crate) fn run_krow_window(
        &mut self,
        inputs: &[u32],
    ) -> Result<RowsOut<Vec<f32>>, RowsFailure> {
        let k = inputs.len();
        self.krow_prelaunch(inputs)
            .map_err(RowsFailure::StateDecides)?;
        #[cfg(test)]
        self.arm_injection();
        let admit_pos = self.krow_start(inputs).map_err(RowsFailure::StateDecides)?;
        let mut entered = 0;
        let ran = self.run_body_rows(k, admit_pos, &mut entered);
        #[cfg(test)]
        self.take_injection();
        if let Err(error) = ran {
            return Err(self.fail_rows(entered, error));
        }
        match self.krow_finish(k) {
            Ok(out) => Ok(out),
            Err(error) => {
                self.state.poison();
                Err(RowsFailure::AfterLaunch(error))
            }
        }
    }

    /// Everything that precedes the first write: the merged-block re-check, the row count, every
    /// token id, the KV capacity. A refusal leaves the state valid and unchanged.
    fn krow_prelaunch(&mut self, inputs: &[u32]) -> Result<(), EngError> {
        let longest = self.krow_runtime()?.rows;
        if inputs.len() > longest {
            return Err(refuse(&format!(
                "a window of {} rows, the K-row buffers serve {longest}",
                inputs.len()
            )));
        }
        let facts = RowsFacts {
            base: self.state.tokens(),
            capacity: self.kv_capacity_rows(),
            merged_block: self.exl3_merge.is_some(),
        };
        let verdict = self.widths.verdict;
        check_rows(
            inputs,
            |token| super::check_token_id(token, verdict),
            &facts,
        )?;
        #[cfg(test)]
        self.refusal_at(Fault::KRowRefusal)?;
        Ok(())
    }

    /// Writes the window ids and the start block and admits the rows; returns the host mirror of
    /// the device `pos` word. The ids stay on the device: the embedding reads them from there.
    fn krow_start(&mut self, inputs: &[u32]) -> Result<usize, EngError> {
        let too_big = |what| EngError::BadRequest(format!("step: the {what} does not fit an i32"));
        let mut ids = Vec::with_capacity(inputs.len() * size_of::<i32>());
        for &token in inputs {
            ids.extend(
                i32::try_from(token)
                    .map_err(|_| too_big("token"))?
                    .to_le_bytes(),
            );
        }
        let first = i32::try_from(inputs[0]).map_err(|_| too_big("token"))?;
        let pos = i32::try_from(self.state.tokens()).map_err(|_| too_big("position"))?;
        self.started = true;
        let Self {
            krow,
            stream,
            state,
            ..
        } = self;
        let runtime = krow.as_ref().ok_or_else(not_armed)?;
        runtime
            .io
            .write_at(stream, runtime.layout.ids.offset, &ids)?;
        #[cfg(test)]
        break_one_id(runtime, stream)?;
        // The start block carries the first token only because the block has the field; the
        // embedding reads the window ids.
        state.write_step_params(stream, &StepParams::new(first, pos)?)?;
        Ok(state.enqueue_rows(inputs.len())?.admit_pos)
    }

    /// The failure rule of a body that failed: a failure of its first op takes the step back; any
    /// later failure poisons and fails the window closed.
    fn fail_rows(&mut self, entered: usize, error: EngError) -> RowsFailure {
        match action_for(entered) {
            FailureAction::Cancel => match self.state.cancel_enqueue() {
                Ok(()) => RowsFailure::StateDecides(error),
                Err(cancel) => RowsFailure::AfterLaunch(cancel),
            },
            FailureAction::Poison => {
                self.state.poison();
                RowsFailure::AfterLaunch(error)
            }
        }
    }

    /// The body: it only enqueues launches of `token_ops_rows` on the owned stream. `entered`
    /// counts the ops that were entered, the failing one included.
    fn run_body_rows(
        &self,
        k: usize,
        admit_pos: usize,
        entered: &mut usize,
    ) -> Result<(), EngError> {
        let params = self.state.word(Word::StepParams)?;
        for op in token_ops_rows(self.res.table.len(), k)? {
            *entered += 1;
            self.run_op_rows(op, (k, admit_pos), params)?;
        }
        Ok(())
    }

    fn run_op_rows(
        &self,
        op: RowsOp,
        (k, admit_pos): (usize, usize),
        params: DeviceRange<'_>,
    ) -> Result<(), EngError> {
        let (ctx, s) = (self.context, &self.stream);
        match op {
            // `step_begin` clears the status word, as at the start of a one-row step: no bit of an
            // earlier window reaches this window's word.
            RowsOp::Step(StepOp::Begin) => {
                ctx.step_begin(s, params)?;
                ctx.exl3_flag_clear(s, self.kview(Slot::Exl3Flag)?)
            }
            RowsOp::Step(StepOp::Embed) => {
                self.embed_rows(k)?;
                #[cfg(test)]
                self.inject_hidden()?;
                Ok(())
            }
            RowsOp::Step(StepOp::Mixer(layer)) => self.mixer_rows(layer, k, admit_pos),
            RowsOp::Step(StepOp::Moe(layer)) => self.moe_rows(layer, k),
            RowsOp::Step(StepOp::FinalNorm) => self.final_norm_rows(k),
            RowsOp::Step(StepOp::LmHead) => {
                self.lm_head_rows(k)?;
                #[cfg(test)]
                self.inject_logits()?;
                Ok(())
            }
            RowsOp::ArgmaxRows => ctx.argmax_rows(s, self.tail_calls(k)?.0),
            RowsOp::StatusFold => ctx.window_status_fold(s, self.tail_calls(k)?.1),
            RowsOp::Step(StepOp::Advance) => ctx.step_advance(
                s,
                params,
                u32::try_from(k).map_err(|_| refuse("the window length does not fit u32"))?,
            ),
            RowsOp::Step(StepOp::Argmax) => Err(refuse("the K-row list has no one-row argmax")),
        }
    }

    /// The two calls of the window tail, built from the one row count `k`.
    fn tail_calls(&self, k: usize) -> Result<(ArgmaxRowsCall<'_>, WindowFoldCall<'_>), EngError> {
        let runtime = self.krow_runtime()?;
        let table = runtime.layout.result.offset;
        let argmax = ArgmaxRowsCall {
            m: k,
            n: self.widths.argmax_n,
            ld: self.res.dims.shape.vocab,
            logits: self.kview(Slot::Logits)?,
            part: self.kview(Slot::ArgmaxPart)?,
            result: runtime.io.range(table, k * RESULT_PAIR_BYTES)?,
        };
        let fold = WindowFoldCall {
            m: k,
            params: self.state.word(Word::StepParams)?,
            flag: self.kview(Slot::Exl3Flag)?,
            result: runtime.io.range(table, window_result_bytes(k))?,
        };
        check_tail_rows(argmax.m, fold.m)?;
        Ok((argmax, fold))
    }

    fn embed_rows(&self, k: usize) -> Result<(), EngError> {
        let runtime = self.krow_runtime()?;
        let EmbedWeight::Bf16(table) = self.globals.embed else {
            return Err(refuse("needs the bf16 embedding table"));
        };
        self.context.embed_bf16(
            &self.stream,
            EmbedBf16Call {
                table: table.range(0, table.bytes())?,
                ids: runtime
                    .io
                    .range(runtime.layout.ids.offset, runtime.layout.ids.bytes(k))?,
                x: self.kview(resid_in(0))?,
                status: self.state.step_status()?,
                m: k,
                n_rows: self.res.dims.shape.vocab,
                k: self.res.geometry.hidden,
            },
        )
    }

    fn mixer_rows(&self, layer: usize, k: usize, admit_pos: usize) -> Result<(), EngError> {
        self.state.check_block(&self.stream)?;
        let slot = self.res.table.slot(layer)?;
        let run = RowsRun {
            weights: self.mixers.weights().layer_weights(layer)?,
            ordinal: slot.ordinal,
            resid: self.kview(resid_in(layer))?,
            k,
            admit_pos,
        };
        for op in self.krow_runtime()?.ops.of(slot.kind)? {
            self.block_op_rows(&run, op)?;
        }
        Ok(())
    }

    /// The slots' snapshot ranges of layer `run.ordinal` for the part `part`.
    fn snap_of(&self, run: &RowsRun<'_>, part: SnapPart) -> Result<SnapRows<'_>, EngError> {
        self.rollback
            .as_ref()
            .ok_or_else(|| refuse("needs the device rollback slots"))?
            .snap_rows(&self.state, run.ordinal, part, run.k)
    }

    fn block_op_rows(&self, run: &RowsRun<'_>, op: &RowsBlockOp) -> Result<(), EngError> {
        let (ctx, s, g, k) = (self.context, &self.stream, &self.res.geometry, run.k);
        let capacity = self.state.layout().kv.capacity;
        match op {
            RowsBlockOp::NormPack {
                weight,
                delta,
                xf,
                norm_out,
                ..
            } => ctx.add_norm_pack(
                s,
                AddNormCall {
                    m: k,
                    n: g.hidden,
                    eps: g.eps,
                    h: run.resid,
                    delta: delta.map(|slot| self.kview(slot)).transpose()?,
                    weight: run.weights.raw(*weight)?,
                    xf: self.kview(*xf)?,
                    norm_out: Some(self.kview(*norm_out)?),
                },
            ),
            RowsBlockOp::Project { role, x, y, .. } => {
                let weight = run.weights.exl3(*role)?.view()?;
                let io = Exl3Io {
                    x: self.place(*x, k)?,
                    xh: self.kview(Slot::Exl3Xh)?,
                    z: self.kview(Slot::Exl3Z)?,
                    out: self.place(*y, k)?,
                    flag: self.kview(Slot::Exl3Flag)?,
                    rows: k,
                    fused: false,
                };
                ctx.exl3_linear(s, weight.linear_call(&io))
            }
            RowsBlockOp::Gate {
                role, x, y, col, ..
            } => {
                let shape = plain_shape(g, *role)?;
                let dst = self.kview(*y)?;
                let (offset, len) = tail_extent(dst.bytes(), *col)?;
                ctx.router_logits_rows_strided(
                    s,
                    RouterLogitsStridedCall {
                        shape: RouterShape { m: k, ..shape },
                        x: self.kview(*x)?,
                        w: run.weights.raw(*role)?,
                        logits: dst.buffer().range(dst.offset() + offset, len)?,
                        logit_stride: g.delta.ld,
                    },
                )
            }
            RowsBlockOp::Pack { dst, ld, parts, .. } => {
                let mut segments = Vec::with_capacity(parts.len());
                for part in parts {
                    segments.push(PackSegment {
                        src: self.place(Place::Tight(part.src), k)?,
                        n: part.n,
                        dst_off: part.col,
                    });
                }
                ctx.pack_rows(
                    s,
                    PackRowsCall {
                        m: k,
                        ld: *ld,
                        dst: self.kview(*dst)?,
                        segments,
                    },
                )
            }
            RowsBlockOp::Conv { proj, out, weights } => ctx.gdn_conv_snap(
                s,
                ConvCall {
                    shape: ConvShape {
                        m: k,
                        channels: g.delta.channels,
                        ld: g.delta.channels,
                    },
                    proj: self.place(Place::Tight(*proj), k)?,
                    weights: run.weights.raw(*weights)?,
                    state: self.state.delta_conv(0, run.ordinal)?,
                    out: self.kview(*out)?,
                },
                self.snap_of(run, SnapPart::Conv)?,
            ),
            RowsBlockOp::Delta {
                conv,
                proj,
                out,
                a,
                dt_bias,
                norm,
            } => ctx.gdn_delta_snap(
                s,
                DeltaCall {
                    shape: DeltaShape {
                        m: k,
                        ..delta_shape(g)
                    },
                    eps: g.eps,
                    conv: self.kview(*conv)?,
                    proj: self.kview(*proj)?,
                    weights: DeltaWeights {
                        a: run.weights.raw(*a)?,
                        dt_bias: run.weights.raw(*dt_bias)?,
                        norm_w: run.weights.raw(*norm)?,
                    },
                    state: self.state.delta_rec(0, run.ordinal)?,
                    out: self.kview(*out)?,
                },
                self.snap_of(run, SnapPart::Recurrent)?,
            ),
            RowsBlockOp::AttnPrep {
                proj,
                q,
                q_norm,
                k_norm,
            } => ctx.attn_prep(
                s,
                AttnPrepCall {
                    shape: AttnPrepShape {
                        m: k,
                        ..attn_prep_shape(g, capacity)
                    },
                    rope: RopeParams {
                        eps: g.eps,
                        rope_base: g.attn.rope_base,
                    },
                    admit_pos: run.admit_pos,
                    proj: self.kview(*proj)?,
                    q_norm: run.weights.raw(*q_norm)?,
                    k_norm: run.weights.raw(*k_norm)?,
                    k_cache: self.state.kv_plane(run.ordinal, Plane::K)?,
                    v_cache: self.state.kv_plane(run.ordinal, Plane::V)?,
                    q: self.kview(*q)?,
                    pos: self.state.step_field(StepField::Pos)?,
                    status: self.state.step_field(StepField::Status)?,
                },
            ),
            RowsBlockOp::AttnDecode { q, proj, part, out } => ctx.attn_decode(
                s,
                AttnDecodeCall {
                    shape: AttnShape {
                        m: k,
                        ..attn_decode_shape(g, capacity)
                    },
                    span: AttnSpan::FULL,
                    split: AttnSplit::STAGE1,
                    scale: g.attn.scale,
                    q: self.kview(*q)?,
                    proj: Some(self.kview(*proj)?),
                    k_cache: self.state.kv_plane(run.ordinal, Plane::K)?,
                    v_cache: self.state.kv_plane(run.ordinal, Plane::V)?,
                    kv_len: self.state.step_field(StepField::KvLen)?,
                    part: self.kview(*part)?,
                    out: self.kview(*out)?,
                },
            ),
        }
    }

    /// The MoE block of `layer` for the `k` rows, over the K-row arena.
    fn moe_rows(&self, layer: usize, k: usize) -> Result<(), EngError> {
        let weights = self
            .exl3_moe
            .get(layer)
            .ok_or_else(|| refuse(&format!("layer {layer} has no EXL3 MoE weights")))?;
        let launch = Launch {
            ctx: self.context,
            stream: &self.stream,
        };
        let parity = layer % 2;
        let bufs = Exl3MoeBufs {
            route: RouteBufs {
                resid_in: self.kview(resid_in(parity))?,
                resid_out: self.kview(resid_out(parity))?,
                norm_row: self.kview(Slot::NormRow)?,
                logits: self.kview(Slot::RouterLogits)?,
                ids: self.kview(Slot::TopkId)?,
                weights: self.kview(Slot::TopkWeight)?,
                shared_gate: self.kview(Slot::SharedGate)?,
                status: self.state.step_status()?,
            },
            expert_gu: self.kview(Slot::ExpertGu)?,
            expert_act: self.kview(Slot::ExpertAct)?,
            expert_down: self.kview(Slot::ExpertDown)?,
            shared_gu: self.kview(Slot::SharedGu)?,
            shared_act: self.kview(Slot::SharedAct)?,
            shared_out: self.kview(Slot::SharedOut)?,
            scratch: Scratch {
                xh: self.kview(Slot::Exl3Xh)?,
                z: self.kview(Slot::Exl3Z)?,
                flag: self.kview(Slot::Exl3Flag)?,
            },
        };
        let dims = MoeDims {
            rows: k,
            ..self.moe_dims
        };
        run_exl3_moe_rows(&launch, weights, &bufs, &dims)
    }

    fn final_norm_rows(&self, k: usize) -> Result<(), EngError> {
        let norm = self.globals.output_norm;
        self.context.add_norm_pack(
            &self.stream,
            AddNormCall {
                m: k,
                n: self.res.geometry.hidden,
                eps: self.res.geometry.eps,
                h: self.kview(final_resid(self.res.table.len()))?,
                delta: None,
                weight: norm.range(0, norm.bytes())?,
                xf: self.kview(Slot::XfFinal)?,
                norm_out: Some(self.kview(Slot::FinalNormRow)?),
            },
        )
    }

    fn lm_head_rows(&self, k: usize) -> Result<(), EngError> {
        let HeadWeight::Exl3(head) = self.globals.head else {
            return Err(refuse("needs the EXL3 head"));
        };
        let io = Exl3Io {
            x: self.kview(Slot::FinalNormRow)?,
            xh: self.kview(Slot::Exl3Xh)?,
            z: self.kview(Slot::Exl3Z)?,
            out: self.kview(Slot::Logits)?,
            flag: self.kview(Slot::Exl3Flag)?,
            rows: k,
            fused: false,
        };
        self.context
            .exl3_linear(&self.stream, head.view()?.linear_call(&io))
    }

    /// After the body: the one synchronization and read of the result table, the verdict on every
    /// row and on the window word, the logits rows, the commit of the rows, and last the mark of
    /// the slots the kernels stored. Any error is a failure after the launch (the caller poisons).
    fn krow_finish(&mut self, k: usize) -> Result<RowsOut<Vec<f32>>, EngError> {
        #[cfg(test)]
        self.fault_at(Fault::KRowLaunch)?;
        #[cfg(test)]
        self.fault_at(Fault::KRowReadback)?;
        let table = self.read_result_table(k)?;
        #[cfg(test)]
        let table = self.status_fault(table);
        #[cfg(test)]
        self.record_table(&table);
        let status = RowStatus {
            bits: table.bits.iter().map(|&bits| bits as u32).collect(),
            window: table.window as u32,
        };
        let argmax = rows_verdict(&table.tokens, &status, self.widths.verdict)?;
        let rows = self.read_krow_rows(k)?;
        self.state.commit_step(k, 0)?;
        let Self {
            rollback,
            stream,
            state,
            ..
        } = self;
        rollback
            .as_mut()
            .ok_or_else(|| refuse("needs the device rollback slots"))?
            .mark_row_slots(stream, state, k - 1)?;
        #[cfg(test)]
        if let Some(runtime) = self.krow.as_mut() {
            runtime.test.windows += 1;
        }
        Ok(RowsOut { argmax, rows })
    }

    /// The `8 K + 8` bytes of the result table: waits for the stream.
    fn read_result_table(&self, k: usize) -> Result<WindowResult, EngError> {
        let runtime = self.krow_runtime()?;
        let mut bytes = vec![0u8; window_result_bytes(k)];
        runtime
            .io
            .read_at(&self.stream, runtime.layout.result.offset, &mut bytes)?;
        WindowResult::from_bytes(&bytes, k)
            .ok_or_else(|| refuse("the result table does not decode (the pad word is not zero)"))
    }

    /// The logits of the `k` rows, one vector of the head width per row.
    fn read_krow_rows(&mut self, k: usize) -> Result<Vec<Vec<f32>>, EngError> {
        #[cfg(test)]
        if self
            .krow
            .as_ref()
            .is_some_and(|runtime| runtime.test.skip_logits)
        {
            return Ok(vec![Vec::new(); k]);
        }
        let view = self.kview(Slot::Logits)?;
        let row_bytes = self.res.dims.shape.vocab * F32;
        let Self { krow, stream, .. } = self;
        let runtime = krow.as_mut().ok_or_else(not_armed)?;
        let out = &mut runtime.logits[..k * row_bytes];
        view.buffer().read_at(stream, view.offset(), out)?;
        Ok(out
            .chunks_exact(row_bytes)
            .map(|row| {
                row.as_chunks::<F32>()
                    .0
                    .iter()
                    .map(|word| f32::from_le_bytes(*word))
                    .collect()
            })
            .collect())
    }
}

/// Test builds: the switches, counters and injections of the D1 harness.
#[cfg(test)]
impl QwenResidentStep<'_> {
    /// K-row windows that completed since the step was made (zero on a step that never armed it).
    pub(crate) fn krow_windows(&self) -> u64 {
        self.krow.as_ref().map_or(0, |runtime| runtime.test.windows)
    }

    /// Switches the K-row path off (`false`) or on: off, windows run the eager loop on the same
    /// armed step, so the two paths can be compared on one set of weights.
    pub(crate) fn use_krow(&mut self, on: bool) {
        if let Some(runtime) = self.krow.as_mut() {
            runtime.test.path_off = !on;
        }
    }

    /// Returns empty logits rows from K-row windows (`true`): the tap a later change that reads
    /// the logits on the device keeps.
    pub(crate) fn set_krow_skip_logits(&mut self, skip: bool) {
        if let Some(runtime) = self.krow.as_mut() {
            runtime.test.skip_logits = skip;
        }
    }

    /// Makes a value of the next K-row window non-finite inside its body.
    pub(crate) fn inject_krow(&mut self, injection: Injection) {
        if let Some(runtime) = self.krow.as_mut() {
            runtime.test.injection = Some(injection);
        }
    }

    /// The bytes of every armed rollback slot, as words.
    pub(crate) fn rollback_slot_words(&self) -> Result<Vec<Vec<u32>>, EngError> {
        self.rollback
            .as_ref()
            .ok_or_else(|| refuse("needs the device rollback slots"))?
            .slot_words(&self.stream)
    }

    /// The launches of the K-row kernel identities on this step's context.
    pub(crate) fn row_launches(&self) -> crate::kernels::row_launches::RowLaunches {
        self.context.row_launch_counts()
    }

    /// Turns the armed `KRowRowPair(j)` fault into the injection of the body.
    fn arm_injection(&mut self) {
        if let Some(Fault::KRowRowPair(row)) = self.window_fault {
            self.window_fault = None;
            self.inject_krow(Injection::LogitsNan(row));
        }
    }

    /// The injection is for one window: it is gone when the body has run.
    fn take_injection(&mut self) {
        if let Some(runtime) = self.krow.as_mut() {
            runtime.test.injection = None;
        }
    }

    fn injection(&self) -> Option<Injection> {
        self.krow
            .as_ref()
            .and_then(|runtime| runtime.test.injection)
    }

    /// Writes a quiet NaN into element 0 of the embedding of row `j` (after the embedding op).
    fn inject_hidden(&self) -> Result<(), EngError> {
        if let Some(Injection::HiddenNan(row)) = self.injection() {
            let view = self.kview(resid_in(0))?;
            let at = view.offset() + row * self.res.geometry.hidden * F32;
            view.buffer()
                .write_at(&self.stream, at, &f32::NAN.to_le_bytes())?;
        }
        Ok(())
    }

    /// Writes a quiet NaN into element 0 of logits row `j` (after the head, before the argmax).
    fn inject_logits(&self) -> Result<(), EngError> {
        if let Some(Injection::LogitsNan(row)) = self.injection() {
            let view = self.kview(Slot::Logits)?;
            let at = view.offset() + row * self.res.dims.shape.vocab * F32;
            view.buffer()
                .write_at(&self.stream, at, &f32::NAN.to_le_bytes())?;
        }
        Ok(())
    }

    fn record_table(&mut self, table: &WindowResult) {
        if let Some(runtime) = self.krow.as_mut() {
            runtime.test.last_table = Some(table.clone());
        }
    }

    /// The result table the last K-row window read back, before the verdict judged it.
    pub(crate) fn last_krow_table(&self) -> Option<WindowResult> {
        self.krow
            .as_ref()
            .and_then(|runtime| runtime.test.last_table.clone())
    }

    /// The armed `KRowStatus` fault: the window word shows a router fault no pair shows.
    fn status_fault(&mut self, mut table: WindowResult) -> WindowResult {
        if self.window_fault == Some(Fault::KRowStatus) {
            self.window_fault = None;
            table.window |= 1;
        }
        table
    }
}

/// Test builds: the deliberate-break control. `FIFI_KROW_BREAK=ids` makes row 1's token on the
/// device differ from the host's after the ids were written, inside the K-row path only, so a D1
/// run that still passes did not run the K-row path.
#[cfg(test)]
fn break_one_id(
    runtime: &KrowRuntime,
    stream: &crate::kernels::cuda::DeviceStream,
) -> Result<(), EngError> {
    if std::env::var("FIFI_KROW_BREAK").as_deref() != Ok("ids") {
        return Ok(());
    }
    let at = runtime.layout.ids.offset + runtime.layout.ids.row_bytes;
    let mut word = [0u8; 4];
    runtime.io.read_at(stream, at, &mut word)?;
    let other = (i32::from_le_bytes(word) + 1).to_le_bytes();
    runtime.io.write_at(stream, at, &other)
}
