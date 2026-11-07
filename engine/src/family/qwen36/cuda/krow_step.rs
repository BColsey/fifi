//! The K-row verify window of the step (`qwen36.draft_verify = krow`): `K` rows (2 to 8) enqueued as
//! one launch list on the owned stream, one synchronization, one small read of the result table,
//! and the logits of the `K` rows. The list runs eagerly or as one CUDA graph per window length
//! (`krow_graph`): the first window of a length is eager, the next captures the graph and launches
//! it, later windows replay it. The graph is the same launch list, so the bytes are the same.
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
//! The logits stay on the device. After the window the host makes ONE small read, the report
//! (`WindowReport`, `1.5 KB` at K = 8): the pairs, the window word, the accepted run and bonus
//! token that `verify_greedy` compared on the device, and one top-20 record per row from the
//! K-row shortlist. The host re-derives the run and the bonus from the pairs and the window ids and
//! fails the window closed on any difference, and checks every record against its pair
//! (`krow::check_verify`, `krow::check_shortlists`). The window returns no logits rows: a draw
//! gets what it needs from the report (greedy: the pair token; a seeded sampler that keeps 1 to 20
//! candidates: the row's record, by the same `Sampler::sample_shortlist` the one-row step uses),
//! and reads ONE full row from the arena (`read_krow_row`, 993,280 bytes) only for a sampler the
//! record cannot serve. The test-only tap `want_logits` makes the window read all K rows after the
//! verdict, as the checks of the D1 oracle need (`read_krow_rows`).
//!
//! What this body must keep: the logits stay in the arena until `draw` or the tap reads them, so
//! nothing may write the arena between the window and its verdict (`window_awaits` refuses every
//! step, export and window until then, and no kernel runs after the list); the window ids are
//! written once and never read by the host; nothing runs on the host between the rows.
//!
//! The row count of the window tail is checked when the argmax op runs (`tail_calls`), which is
//! after the layers were enqueued. It is a guard by construction: the tail calls are built from
//! the one `k`. Moving the construction before the first launch is a later change.

use super::super::{
    exl3_moe::{Exl3MoeBufs, Scratch, run_exl3_moe_rows},
    krow_ops::{KrowIoLayout, Place, RowsBlockOp, RowsKindOps, check_tail_rows},
    mixer::tail_extent,
    mixer_shapes::{attn_decode_shape, attn_prep_shape, delta_shape, plain_shape},
    mixer_weights::LayerWeights,
    moe::{Launch, MoeDims, RouteBufs},
    resources::{EmbedWeight, HeadWeight},
    state::{StepField, rollback::SnapPart},
    state_layout::{Plane, Word},
    step_ops::{RowsOp, StepOp, final_resid, resid_in, resid_out, token_ops_rows},
    step_plan::Slot,
    verdict::{FailureAction, action_for},
};
use super::{
    Mode, QwenResidentStep,
    krow_graph::{ENTERED_BY_GRAPH, Facts},
};
#[cfg(test)]
use crate::family::speculate::d1_oracle::Fault;
#[cfg(test)]
use crate::kernels::cuda::resident::argmax_rows::WindowResult;
use crate::{
    EngError,
    family::{
        project::cuda::{WeightLayout, exl3::Exl3Io},
        qwen36::{
            krow::{
                self, ConfigFacts, DeviceDraws, DrawFacts, DrawRoute, RowStatus, RowsFacts,
                check_rows, check_shortlists, check_verify, draw_route, rows_verdict,
            },
            window_logic::{self, RowsFailure, RowsOut, WindowSave, no_window},
        },
        sampler::Sampler,
        session::{TokenStepper, sample_logits},
    },
    kernels::cuda::{
        CapturePlan, DeviceBuffer, DeviceRange, is_capture_refusal,
        resident::{
            add_norm_pack::AddNormCall,
            argmax_rows::{
                ArgmaxRowsCall, VerifyGreedyCall, WindowFoldCall, WindowReport, window_read_bytes,
                window_report_bytes, window_result_bytes, window_shortlist_offset,
            },
            attn_decode::{AttnDecodeCall, AttnShape, AttnSpan, AttnSplit},
            attn_prep::{AttnPrepCall, AttnPrepShape, RopeParams},
            embed_bf16::EmbedBf16Call,
            gdn_conv::{ConvCall, ConvShape, SnapRows},
            gdn_delta::{DeltaCall, DeltaShape, DeltaWeights},
            pack_rows::{PackRowsCall, PackSegment},
            router::RouterShape,
            router_rows::RouterLogitsStridedCall,
            shortlist::{RESULT_BYTES as SHORTLIST_RECORD_BYTES, ShortlistRowsCall},
            step_params::{ARGMAX_PART_BYTES, RESULT_PAIR_BYTES, StepParams},
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
    /// Window ids, result table and tight projection rows. Under 1 MB, allocated at arming; it is
    /// not part of the admission figure (`decode_arena` counts the K-row arena only) and sits in
    /// the headroom.
    io: DeviceBuffer,
    layout: KrowIoLayout,
    ops: RowsKindOps,
    /// The longest window the buffers and the slots serve.
    rows: usize,
    /// Host copy of the logits rows the test tap reads, the head width each; grown on first use, so
    /// a serving window allocates none.
    logits: Vec<u8>,
    /// What the open window keeps for its draws; `None` when no K-row window is open.
    draws: Option<DeviceDraws>,
    #[cfg(test)]
    test: TestState,
}

/// Test builds only: the switches and counters of the D1 harness.
#[cfg(test)]
struct TestState {
    /// The K-row path is switched off: windows run the eager loop on this armed step.
    path_off: bool,
    /// K-row windows whose body was enqueued whole (a window that fails after its launch counts:
    /// its kernels ran, and the launch counters count them).
    windows: u64,
    /// The tap: every window also reads all K logits rows to the host after the verdict (what the
    /// D1 checks of the logits rows need). Off, a window reads the report and nothing else.
    want_logits: bool,
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

/// Test builds only: the residual rows after one op of a layer. `after_moe` false is the residual
/// after the mixer (the MoE block's input), true the layer's output.
#[cfg(test)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct LayerTap {
    pub(crate) layer: usize,
    pub(crate) after_moe: bool,
    /// The bytes of each row of the residual, `hidden` f32 values per row.
    pub(crate) rows: Vec<Vec<u8>>,
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
        // The pins of a K-row graph: the step's own pins (the weights and the one-row arena) and the
        // K-row arena. The K-row body does not touch the one-row arena; it is carried along because
        // the slice is the step's own and a pin is a lifetime borrow, not a check against the body
        // (`CapturePlan::pins`), so one more buffer in the list costs nothing. The small buffer, the
        // state and the slots are covered by drop order.
        let mut pinned: Vec<&'a DeviceBuffer> = self.pins.to_vec();
        pinned.push(krow.arena.buffer());
        self.krow_pins = Box::leak(pinned.into_boxed_slice());
        self.krow = Some(KrowRuntime {
            io,
            layout,
            ops,
            rows,
            logits: Vec::new(),
            draws: None,
            #[cfg(test)]
            test: TestState {
                path_off: false,
                windows: 0,
                want_logits: false,
                injection: None,
                last_table: None,
            },
        });
        Ok(())
    }

    /// The configuration the K-row body is built for (`krow::check_krow_configuration`).
    fn check_krow_configuration(&self) -> Result<(), EngError> {
        krow::check_krow_configuration(ConfigFacts {
            fp16_kv: self.state.layout().kv.mode == super::KvMode::Fp16,
            plain_step: !(self.tiered_mode
                || self.sync_each
                || self.forked_experts
                || self.layout != WeightLayout::Exl3),
            bf16_embedding: matches!(self.globals.embed, EmbedWeight::Bf16(_)),
            exl3_head: matches!(self.globals.head, HeadWeight::Exl3(_)),
        })
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
        krow::uses_krow(rows) && rows <= runtime.rows
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
        let base = self.state.tokens();
        // A new window replaces what the one before kept for its draws.
        if let Some(runtime) = self.krow.as_mut() {
            runtime.draws = None;
        }
        self.krow_prelaunch(inputs)
            .map_err(RowsFailure::StateDecides)?;
        #[cfg(test)]
        self.arm_injection();
        let admit_pos = self.krow_start(inputs).map_err(RowsFailure::StateDecides)?;
        let mut entered = 0;
        let ran = self.run_krow_body(k, admit_pos, &mut entered);
        #[cfg(test)]
        self.take_injection();
        #[cfg(test)]
        if ran.is_ok()
            && let Some(runtime) = self.krow.as_mut()
        {
            runtime.test.windows += 1;
        }
        if let Err(error) = ran {
            return Err(self.fail_rows(entered, error));
        }
        match self.krow_finish(inputs, base) {
            Ok(out) => Ok(out),
            Err(error) => {
                self.state.poison();
                Err(RowsFailure::AfterLaunch(error))
            }
        }
    }

    /// How the next window of `k` rows runs. Test builds run eagerly while an injection or the
    /// per-layer taps are pending: both read or write the device inside the body, which a capture
    /// forbids.
    fn krow_mode(&self, k: usize) -> Result<Mode, EngError> {
        #[cfg(test)]
        if self.injection().is_some() || self.taps.borrow().is_some() {
            return super::krow_graph::slot_of(k).map(|_| Mode::Eager);
        }
        self.krow_graphs.mode(k, self.capture_on)
    }

    /// The body of a window of `k` rows in the mode the graph table says: replay the graph of
    /// exactly `k` rows, capture it, or enqueue the list eagerly. An eager body that was enqueued
    /// whole has loaded every module it launches, which warms its length for the capture.
    fn run_krow_body(
        &mut self,
        k: usize,
        admit_pos: usize,
        entered: &mut usize,
    ) -> Result<(), EngError> {
        match self.krow_mode(k)? {
            Mode::Replay => self.replay_krow(k, entered, false),
            Mode::Capture => self.capture_krow(k, admit_pos, entered),
            Mode::Eager => self.run_body_rows(k, admit_pos, entered).map(|()| {
                self.krow_graphs.note_eager(k);
            }),
        }
    }

    /// Launches the graph of `k` rows. The graph is the whole body, so the failure rule sees a
    /// window that went past its first op: a failed launch poisons the state and fails the window
    /// closed, it does not take the step back. `captured_now` is true for the window that has just
    /// captured the graph.
    fn replay_krow(
        &mut self,
        k: usize,
        entered: &mut usize,
        captured_now: bool,
    ) -> Result<(), EngError> {
        *entered = ENTERED_BY_GRAPH;
        // The deliberate break of the replay: a device word changed after the capture, in a later
        // window than the one that captured.
        #[cfg(test)]
        if !captured_now && let Some(runtime) = self.krow.as_ref() {
            break_one_id(runtime, &self.stream, "ids-replay")?;
        }
        #[cfg(not(test))]
        let _ = captured_now;
        let Self {
            krow_graphs,
            stream,
            ..
        } = self;
        let graph = krow_graphs
            .graph(k)?
            .ok_or_else(|| refuse(&format!("there is no graph of {k} rows to launch")))?;
        graph.launch(stream)?;
        krow_graphs.note_replay(k);
        Ok(())
    }

    /// Captures the body of `k` rows, stores the graph under `k` and launches it (a capture
    /// records and does not run). A refused capture discards the attempt's op count, sets the
    /// sticky refusal and runs the body eagerly in the same window; any other error is the
    /// caller's to fail on.
    fn capture_krow(
        &mut self,
        k: usize,
        admit_pos: usize,
        entered: &mut usize,
    ) -> Result<(), EngError> {
        let res: &'a super::StepResources = self.res;
        let krow = res.krow.as_ref().ok_or_else(not_armed)?;
        let holds = vec![krow.arena.hold(), self.context.workspace_hold()?];
        #[cfg(test)]
        let free_before = self.context.memory_info().ok().map(|info| info.free);
        let attempt = self.context.capture(
            &self.stream,
            CapturePlan {
                pins: self.krow_pins,
                holds,
            },
            || self.run_body_rows(k, admit_pos, entered),
        );
        match super::fold_capture(
            attempt,
            is_capture_refusal,
            entered,
            self.krow_graphs.refused_mut(),
        )? {
            Some(graph) => {
                #[cfg(test)]
                let facts = Facts {
                    nodes: graph.nodes(),
                    vram_bytes: free_before
                        .zip(self.context.memory_info().ok().map(|info| info.free))
                        .map_or(0, |(before, after)| before.saturating_sub(after) as u64),
                };
                #[cfg(not(test))]
                let facts = Facts {};
                self.krow_graphs.store(k, graph, facts)?;
                self.replay_krow(k, entered, true)
            }
            None => {
                self.run_body_rows(k, admit_pos, entered)?;
                self.krow_graphs.note_eager(k);
                Ok(())
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
        break_one_id(runtime, stream, "ids")?;
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
            RowsOp::Step(StepOp::Mixer(layer)) => {
                self.mixer_rows(layer, k, admit_pos)?;
                #[cfg(test)]
                self.tap_after(StepOp::Mixer(layer), k, true)?;
                Ok(())
            }
            RowsOp::Step(StepOp::Moe(layer)) => {
                self.moe_rows(layer, k)?;
                #[cfg(test)]
                self.tap_after(StepOp::Moe(layer), k, true)?;
                Ok(())
            }
            RowsOp::Step(StepOp::FinalNorm) => self.final_norm_rows(k),
            RowsOp::Step(StepOp::LmHead) => {
                self.lm_head_rows(k)?;
                #[cfg(test)]
                self.inject_logits()?;
                Ok(())
            }
            RowsOp::ArgmaxRows => ctx.argmax_rows(s, self.tail_calls(k)?.0),
            RowsOp::StatusFold => ctx.window_status_fold(s, self.tail_calls(k)?.1),
            RowsOp::VerifyGreedy => ctx.verify_greedy(s, self.verify_call(k)?),
            // Switched off with the one-row shortlist: the window then reads no records.
            RowsOp::ShortlistRows if !self.shortlist_on => Ok(()),
            RowsOp::ShortlistRows => ctx.shortlist_rows(s, self.shortlist_call(k)?),
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

    /// The device comparison of the pair tokens with the window ids of a `k`-row window.
    fn verify_call(&self, k: usize) -> Result<VerifyGreedyCall<'_>, EngError> {
        let runtime = self.krow_runtime()?;
        Ok(VerifyGreedyCall {
            m: k,
            ids: runtime
                .io
                .range(runtime.layout.ids.offset, runtime.layout.ids.bytes(k))?,
            result: runtime
                .io
                .range(runtime.layout.result.offset, window_report_bytes(k))?,
        })
    }

    /// The K-row shortlist of a `k`-row window: the records go to the report, right after the
    /// verify words, and the partials use the argmax partials of the K-row plan (each row's set is
    /// `ARGMAX_PART_BYTES` apart and the argmax of the same window has finished with them).
    fn shortlist_call(&self, k: usize) -> Result<ShortlistRowsCall<'_>, EngError> {
        let runtime = self.krow_runtime()?;
        Ok(ShortlistRowsCall {
            m: k,
            n: self.widths.verdict,
            ld: self.res.dims.shape.vocab,
            logits: self.kview(Slot::Logits)?,
            scratch: self.kview(Slot::ArgmaxPart)?,
            scratch_stride: ARGMAX_PART_BYTES,
            out: runtime.io.range(
                runtime.layout.result.offset + window_shortlist_offset(k),
                k * SHORTLIST_RECORD_BYTES,
            )?,
        })
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

    /// After the body: the one synchronization and read of the report, the verdict on every row and
    /// on the window word, the device comparison and the records checked against the pairs, the
    /// logits rows when the test tap asks for them, the commit of the rows, and last the mark of
    /// the slots the kernels stored. What the draws need is kept for them. Any error is a failure
    /// after the launch (the caller poisons).
    fn krow_finish(&mut self, inputs: &[u32], base: usize) -> Result<RowsOut<Vec<f32>>, EngError> {
        let k = inputs.len();
        #[cfg(test)]
        self.fault_at(Fault::KRowLaunch)?;
        #[cfg(test)]
        self.fault_at(Fault::KRowReadback)?;
        let report = self.read_report(k)?;
        #[cfg(test)]
        let report = self.status_fault(report);
        #[cfg(test)]
        let report = self.verify_fault(report);
        #[cfg(test)]
        self.record_table(&report.table);
        let status = RowStatus {
            bits: report.table.bits.iter().map(|&bits| bits as u32).collect(),
            window: report.table.window as u32,
        };
        let argmax = rows_verdict(&report.table.tokens, &status, self.widths.verdict)?;
        let (accepted, bonus) =
            check_verify(&report.table.tokens, inputs, report.accepted, report.bonus)?;
        let shortlists = if self.shortlist_on {
            check_shortlists(
                &report.table.tokens,
                &report.table.bits,
                &report.shortlists,
                self.widths.verdict,
            )?
        } else {
            Vec::new()
        };
        let rows = if self.krow_wants_logits() {
            self.read_krow_rows(k)?
        } else {
            Vec::new()
        };
        self.state.commit_step(k, 0)?;
        let Self {
            rollback,
            stream,
            state,
            krow,
            ..
        } = self;
        rollback
            .as_mut()
            .ok_or_else(|| refuse("needs the device rollback slots"))?
            .mark_row_slots(stream, state, k - 1)?;
        if let Some(runtime) = krow.as_mut() {
            runtime.draws = Some(DeviceDraws {
                base,
                inputs: inputs.to_vec(),
                accepted,
                bonus,
                shortlists,
            });
        }
        Ok(RowsOut { argmax, rows })
    }

    /// The report of a `k`-row window, with one copy: waits for the stream. The records follow the
    /// table when the shortlist ran.
    fn read_report(&self, k: usize) -> Result<WindowReport, EngError> {
        let runtime = self.krow_runtime()?;
        let with_records = self.shortlist_on;
        let mut bytes = vec![0u8; window_read_bytes(k, with_records)];
        runtime
            .io
            .read_at(&self.stream, runtime.layout.result.offset, &mut bytes)?;
        WindowReport::from_bytes(&bytes, k, with_records)
            .ok_or_else(|| refuse("the report does not decode (the pad word is not zero)"))
    }

    /// Whether a window reads the logits rows of all its rows: only the test tap does.
    fn krow_wants_logits(&self) -> bool {
        #[cfg(test)]
        {
            self.krow
                .as_ref()
                .is_some_and(|runtime| runtime.test.want_logits)
        }
        #[cfg(not(test))]
        {
            false
        }
    }

    /// The logits of the `k` rows, one vector of the head width per row.
    fn read_krow_rows(&mut self, k: usize) -> Result<Vec<Vec<f32>>, EngError> {
        let view = self.kview(Slot::Logits)?;
        let row_bytes = self.res.dims.shape.vocab * F32;
        let Self { krow, stream, .. } = self;
        let runtime = krow.as_mut().ok_or_else(not_armed)?;
        if runtime.logits.len() < k * row_bytes {
            runtime.logits.resize(k * row_bytes, 0);
        }
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

impl QwenResidentStep<'_> {
    /// The draw of row `row` of an open K-row window whose logits stayed on the device, or `None`
    /// when the open window is not one (the caller then takes the default path). Called only for a
    /// sampler that is not greedy: a greedy draw is the pair token and needs nothing from here.
    ///
    /// The sampler draws from the row's top-20 record when the one-row step would (the record is
    /// the one-row step's own selection, so the token stream and the generator are the same);
    /// otherwise from the complete row: the test tap's copy when the tap read it, else one row read
    /// from the arena. A failed read is a device failure after the launch: the window fails closed.
    pub(crate) fn draw_from_krow_window(
        &mut self,
        row: usize,
        sampler: &mut Sampler,
        sampler_vocab: usize,
    ) -> Option<Result<u32, EngError>> {
        let window = self.window.as_ref()?;
        let route = draw_route(&DrawFacts {
            // Only a window that the K-row path ran has records: an eager window (the path switched
            // off, or one row) with the same base and inputs as an earlier K-row window is not
            // served theirs.
            krow_runs: self.krow_runs(window.inputs.len()),
            device_window: matches!(window.saved, WindowSave::Device),
            draws: self
                .krow
                .as_ref()
                .and_then(|runtime| runtime.draws.as_ref()),
            base: window.base,
            inputs: &window.inputs,
            row,
            tap_row: window.rows.get(row).is_some(),
            params: sampler.params(),
            sync_each: self.sync_each,
            vocab_matches: sampler_vocab == self.widths.verdict,
        });
        match route {
            DrawRoute::NotKrow => None,
            DrawRoute::OutOfWindow => Some(Err(no_window())),
            DrawRoute::Shortlist => {
                let candidates = self.krow.as_ref()?.draws.as_ref()?.shortlists.get(row)?;
                let drawn = sampler.sample_shortlist(candidates, sampler_vocab);
                // The report was judged entry by entry when it was read (`check_shortlists`), so an
                // error here is a record that changed since: the window fails closed.
                if drawn.is_err() {
                    window_logic::fail_window(self);
                }
                Some(drawn)
            }
            DrawRoute::TapRow => {
                let values = window.rows.get(row)?;
                Some(sample_logits(
                    Self::OPERATION,
                    values,
                    sampler,
                    sampler_vocab,
                ))
            }
            DrawRoute::DeviceRow => {
                #[cfg(test)]
                if let Err(error) = self.fault_at(Fault::KRowDraw) {
                    window_logic::fail_window(self);
                    return Some(Err(error));
                }
                match self.read_krow_row(row) {
                    Ok(values) => Some(sample_logits(
                        Self::OPERATION,
                        &values,
                        sampler,
                        sampler_vocab,
                    )),
                    Err(error) => {
                        window_logic::fail_window(self);
                        Some(Err(error))
                    }
                }
            }
        }
    }

    /// The complete logits row `row` of the last K-row window, read from the arena: one row of
    /// the head width, waits for the stream.
    fn read_krow_row(&self, row: usize) -> Result<Vec<f32>, EngError> {
        let view = self.kview(Slot::Logits)?;
        let row_bytes = self.res.dims.shape.vocab * F32;
        let mut bytes = vec![0u8; row_bytes];
        view.buffer()
            .read_at(&self.stream, view.offset() + row * row_bytes, &mut bytes)?;
        Ok(bytes
            .as_chunks::<F32>()
            .0
            .iter()
            .map(|word| f32::from_le_bytes(*word))
            .collect())
    }
}

/// Test builds: the switches, counters and injections of the D1 harness.
#[cfg(test)]
impl QwenResidentStep<'_> {
    /// K-row windows launched since the step was made (zero on a step that never armed it).
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

    /// The tap `want_logits`: with `true`, every K-row window also reads all K logits rows to the
    /// host after the verdict (the D1 checks of the rows need them); with `false`, the default, a
    /// window reads its report and nothing else, as in a serving build.
    pub(crate) fn set_krow_want_logits(&mut self, want: bool) {
        if let Some(runtime) = self.krow.as_mut() {
            runtime.test.want_logits = want;
        }
    }

    /// What the open K-row window keeps for its draws: the accepted run the device compared (and
    /// the host checked), the token after it, and the decoded records.
    pub(crate) fn krow_draws(&self) -> Option<DeviceDraws> {
        self.krow.as_ref().and_then(|runtime| runtime.draws.clone())
    }

    /// Switches the K-row graphs off (`false`) or on: off, windows run the eager K-row body on a
    /// step that captures, so the two bodies can be timed and compared on one set of weights.
    /// Graphs already captured are kept and used again once the graphs are on.
    pub(crate) fn use_krow_graphs(&mut self, on: bool) {
        self.krow_graphs.set_on(on);
    }

    /// Whether K-row windows may use graphs (the switch above; the step must also capture).
    pub(crate) fn krow_graphs_on(&self) -> bool {
        self.capture_on && self.krow_graphs.is_on()
    }

    /// The counters of the K-row graphs: per window length, eager windows, captures, windows that
    /// ran the graph, nodes and the device bytes of the capture.
    pub(crate) fn krow_graph_stats(&self) -> super::krow_graph::GraphStats {
        self.krow_graphs.stats().clone()
    }

    /// Whether a capture of a K-row graph was refused on this step.
    pub(crate) fn krow_graph_refused(&self) -> bool {
        self.krow_graphs.refused()
    }

    /// How a window of `k` rows would run next: `replay`, `capture` or `eager`.
    pub(crate) fn krow_next_mode(&self, k: usize) -> Result<&'static str, EngError> {
        Ok(match self.krow_mode(k)? {
            Mode::Replay => "replay",
            Mode::Capture => "capture",
            Mode::Eager => "eager",
        })
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

    /// The device-to-host reads this step's context has made so far: for each length in bytes, how
    /// many reads of that length. The window tests take the difference across a window.
    pub(crate) fn d2h_reads(&self) -> std::collections::BTreeMap<usize, usize> {
        self.context.buffer_requests().ranged_d2h
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

    /// Turns the per-layer taps on for the next eager body (one-row or K-row): after every mixer
    /// and every MoE block the residual rows are read back (a synchronization, so the body must
    /// not be captured).
    pub(crate) fn start_taps(&self) {
        *self.taps.borrow_mut() = Some(Vec::new());
    }

    /// The taps of the body that ran since [`Self::start_taps`], in launch order; turns them off.
    pub(crate) fn take_taps(&self) -> Vec<LayerTap> {
        self.taps.borrow_mut().take().unwrap_or_default()
    }

    /// Records the residual rows after `op` when taps are on. `krow` selects the K-row arena.
    pub(super) fn tap_after(&self, op: StepOp, rows: usize, krow: bool) -> Result<(), EngError> {
        if self.taps.borrow().is_none() {
            return Ok(());
        }
        let (layer, after_moe, slot) = match op {
            StepOp::Mixer(layer) => (layer, false, resid_in(layer)),
            StepOp::Moe(layer) => (layer, true, resid_out(layer)),
            _ => return Ok(()),
        };
        let view = if krow {
            self.kview(slot)?
        } else {
            self.view(slot)?
        };
        let row_bytes = self.res.geometry.hidden * F32;
        let mut bytes = vec![0u8; rows * row_bytes];
        view.buffer()
            .read_at(&self.stream, view.offset(), &mut bytes)?;
        let tap = LayerTap {
            layer,
            after_moe,
            rows: bytes.chunks_exact(row_bytes).map(<[u8]>::to_vec).collect(),
        };
        if let Some(taps) = self.taps.borrow_mut().as_mut() {
            taps.push(tap);
        }
        Ok(())
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

    /// The logits rows of the last K-row window read straight from the arena, also when the window
    /// failed after its launch (the production path reads them only after the verdict).
    pub(crate) fn last_krow_rows(&mut self, k: usize) -> Result<Vec<Vec<f32>>, EngError> {
        self.read_krow_rows(k)
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
    fn status_fault(&mut self, mut report: WindowReport) -> WindowReport {
        if self.window_fault == Some(Fault::KRowStatus) {
            self.window_fault = None;
            report.table.window |= 1;
        }
        report
    }

    /// The armed `KRowVerify` fault: the device comparison says one more accepted row than the
    /// pairs do (the run wraps to 0 for the longest run, so it always differs from the pairs').
    fn verify_fault(&mut self, mut report: WindowReport) -> WindowReport {
        if self.window_fault == Some(Fault::KRowVerify) {
            self.window_fault = None;
            let rows = report.table.tokens.len() as i32;
            report.accepted = (report.accepted + 1) % rows;
        }
        report
    }
}

/// Test builds: the deliberate-break control. `FIFI_KROW_BREAK=ids` makes row 1's token on the
/// device differ from the host's after the ids were written, inside the K-row path only, so a D1
/// run that still passes did not run the K-row path. `FIFI_KROW_BREAK=ids-replay` does it only
/// for a window that launches a graph captured by an earlier window, just before the launch: the
/// device word changes after the capture, so the break tests the replay and not the capture.
/// `variant` names the value of the variable the call applies to.
#[cfg(test)]
fn break_one_id(
    runtime: &KrowRuntime,
    stream: &crate::kernels::cuda::DeviceStream,
    variant: &str,
) -> Result<(), EngError> {
    if std::env::var("FIFI_KROW_BREAK").as_deref() != Ok(variant) {
        return Ok(());
    }
    let at = runtime.layout.ids.offset + runtime.layout.ids.row_bytes;
    let mut word = [0u8; 4];
    runtime.io.read_at(stream, at, &mut word)?;
    let other = (i32::from_le_bytes(word) + 1).to_le_bytes();
    runtime.io.write_at(stream, at, &other)
}
