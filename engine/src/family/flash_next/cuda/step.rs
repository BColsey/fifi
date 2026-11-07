//! The Flash-Next decode step on the device: one token, one row, queued on one owned stream.
//!
//! WHAT: [`FlashStep`] owns the two arenas the state and step plans describe (the persistent state and the
//! activations of one token), borrows the trunk weights the loader uploaded and the context and stream of
//! the caller, and runs the layers in the site order of the host reference
//! (`family/flash_next/step.rs`): per layer the attention hyper-connection read, the mixer
//! (Gated DeltaNet or QSA, `mixer.rs`), the apply, the MLP read, the MoE block and its apply; after
//! the last layer the final mixer, `lm_head` and the logits. The launch order is the phase order of
//! `step_plan::Phase`.
//!
//! WHY a struct over views: every tensor of the token lives in the step arena at an offset the
//! planner chose by lifetime, so tensors of different phases share addresses. The launches must
//! keep the phase order of `step_plan::Phase`.
//!
//! Reused: the kernel wrappers of the resident decode (hyper, `gdn_conv`, `gdn_scan_seq`, the
//! router's strided dot, `attn_prep`, the `qsa_*` kernels, `exl3_linear`), Qwen3.6's EXL3 MoE block
//! (`run_exl3_moe`, shared with this family) and `CudaWeights` keyed by pack name. New: the
//! op lists, the Flash sites and the glue between them.
//!
//! MoE: [`FlashStep::moe_block`] is the seam. It runs the layer's MoE through Qwen3.6's block with
//! the routed experts as an all-resident bank (`Exl3Bank`), which is what the reduced-layer parity
//! runs use; the split block puts the partial slot bank behind the same call.
//!
//! One row only: the step plan allows up to 8, the shared MoE block takes one.

// Staged: the executor is the first caller of the whole step.
#![cfg_attr(not(test), allow(dead_code, reason = "staged for the executor"))]

use crate::{
    EngError,
    family::{
        exl3_pack::layer_name,
        flash_next::{
            cuda::{
                moe_block::{SplitMoe, SplitMoeTrunk},
                state::{FlashGeometry, StatePlan, StateSlot, WORD_STEP_PARAMS},
                step_plan::{Site, Slot, StepOptions, StepPlan},
            },
            load::model_shape,
            moe::context_budget::KvFormat,
            step::ModelShape,
        },
        gguf_model::Role,
        project::cuda::{
            CudaWeight, CudaWeights,
            exl3::{Exl3Io, Exl3Weight},
        },
        qwen36::cuda::{
            exl3_moe::{Exl3MoeBufs, Exl3MoeWeights, Scratch, run_exl3_moe},
            moe::{Launch, MoeDims, RouteBufs},
        },
    },
    kernels::cuda::{
        CudaContext, DeviceArena, DeviceBuffer, DeviceRange, DeviceStream,
        resident::{
            hyper::{
                HyperApplyCall, HyperDtype, HyperReadCall, HyperScratch, HyperShape, HyperWeights,
            },
            step_params::{OFFSET_POS, OFFSET_STATUS},
        },
    },
    weights::GgufType,
};

pub(super) const F32: usize = size_of::<f32>();

pub(super) fn refuse(detail: String) -> EngError {
    EngError::BadRequest(format!("Flash-Next step: {detail}"))
}

/// The pack name of the hyper-connection mixer that feeds `lm_head`.
pub(super) const MIXER: &str = "model.language_model.hyper_connection_mixer";

/// The step of one token over a window of layers `first_layer..end_layer` (all 48 for a full
/// model, a few for a parity window).
pub(super) struct FlashStep<'a> {
    pub(super) ctx: &'a CudaContext,
    pub(super) stream: &'a DeviceStream,
    pub(super) weights: &'a CudaWeights,
    pub(super) geometry: FlashGeometry,
    pub(super) model: ModelShape,
    pub(super) state_plan: StatePlan,
    pub(super) step_plan: StepPlan,
    pub(super) options: StepOptions,
    state: DeviceArena,
    scratch: DeviceArena,
    /// f32 `[hidden]` zeros: the residual the shared MoE block adds to, since the hyper apply adds
    /// the block output to the stream instead.
    zeros: DeviceBuffer,
    pub(super) layers: std::ops::Range<usize>,
}

impl<'a> FlashStep<'a> {
    /// The step over `layers` of the model in `model_config`, with caches for `positions`
    /// positions. The state is zeroed: a new sequence.
    pub(super) fn new(
        ctx: &'a CudaContext,
        stream: &'a DeviceStream,
        weights: &'a CudaWeights,
        model_config: &[u8],
        positions: usize,
        layers: std::ops::Range<usize>,
    ) -> Result<Self, EngError> {
        let (model, kinds) = model_shape(model_config)?;
        let geometry = FlashGeometry::new(&model, &kinds)?;
        if layers.start >= layers.end || layers.end > geometry.layers.len() {
            return Err(refuse(format!(
                "layers {layers:?} are outside the model's {}",
                geometry.layers.len()
            )));
        }
        let state_plan = StatePlan::new(&geometry, positions, KvFormat::F16)?;
        let options = StepOptions::default();
        let step_plan = StepPlan::new(&geometry, 1, positions, &options)?;
        let state = ctx.arena(&state_plan.arena_tensors())?;
        let scratch = ctx.arena(&step_plan.arena_tensors())?;
        let zeros = ctx.upload(&vec![0u8; geometry.hidden * F32])?;
        let step = Self {
            ctx,
            stream,
            weights,
            geometry,
            model,
            state_plan,
            step_plan,
            options,
            state,
            scratch,
            zeros,
            layers,
        };
        step.reset()?;
        Ok(step)
    }

    /// Zeroes the persistent state (the recurrent states, the rings, the caches): a new sequence.
    pub(super) fn reset(&self) -> Result<(), EngError> {
        let zeros = vec![0u8; 1 << 24];
        let mut at = 0usize;
        let total = self.state.bytes();
        while at < total {
            let len = zeros.len().min(total - at);
            self.state
                .buffer()
                .write_at(self.stream, at, &zeros[..len])?;
            at += len;
        }
        Ok(())
    }

    /// The tensor of `slot` in the step arena.
    pub(super) fn slot(&self, slot: Slot) -> Result<DeviceRange<'_>, EngError> {
        let index = self
            .step_plan
            .index(slot)
            .ok_or_else(|| refuse(format!("the plan has no slot {slot:?}")))?;
        self.scratch.tensor(index)
    }

    /// The tensor of `slot` in the state arena.
    pub(super) fn state_slot(&self, slot: StateSlot) -> Result<DeviceRange<'_>, EngError> {
        let index = self
            .state_plan
            .index(slot)
            .ok_or_else(|| refuse(format!("the state has no slot {slot:?}")))?;
        self.state.tensor(index)
    }

    /// The `StepParams` block in the words tensor, and one of its words.
    pub(super) fn params(&self) -> Result<DeviceRange<'_>, EngError> {
        let words = self.state_slot(StateSlot::Words)?;
        words.buffer().range(
            words.offset() + WORD_STEP_PARAMS,
            crate::kernels::cuda::resident::step_params::STEP_PARAMS_BYTES,
        )
    }

    pub(super) fn pos_word(&self) -> Result<DeviceRange<'_>, EngError> {
        let params = self.params()?;
        params.buffer().range(params.offset() + OFFSET_POS, 4)
    }

    pub(super) fn status_word(&self) -> Result<DeviceRange<'_>, EngError> {
        let params = self.params()?;
        params.buffer().range(params.offset() + OFFSET_STATUS, 4)
    }

    /// The device buffer and offset the token start writes: the words block of the state arena.
    pub(super) fn words_buffer(&self) -> Result<(&DeviceBuffer, usize), EngError> {
        let words = self.state_slot(StateSlot::Words)?;
        Ok((self.state.buffer(), words.offset()))
    }

    pub(super) fn arena_buffers(&self) -> (&DeviceBuffer, &DeviceBuffer) {
        (self.state.buffer(), self.scratch.buffer())
    }

    /// The offset of `slot` in the step arena buffer.
    pub(super) fn slot_offset(&self, slot: Slot) -> Result<usize, EngError> {
        Ok(self.slot(slot)?.offset())
    }

    // ------------------------------------------------------------------ weights

    pub(super) fn exl3(&self, prefix: &str) -> Result<&Exl3Weight, EngError> {
        match self.weights.get(prefix) {
            Some(CudaWeight::Exl3 { weight, .. }) => Ok(weight),
            _ => Err(refuse(format!("{prefix} is not an uploaded EXL3 matrix"))),
        }
    }

    /// An F32 tensor uploaded as such (a vector widened at load).
    pub(super) fn float(&self, name: &str) -> Result<DeviceRange<'_>, EngError> {
        match self.weights.get(name) {
            Some(CudaWeight::Raw {
                ty: GgufType::F32,
                buffer,
                ..
            }) => buffer.range(0, buffer.bytes()),
            _ => Err(refuse(format!("{name} is not an uploaded F32 tensor"))),
        }
    }

    /// A tensor kept as stored (F16 or BF16) with its dtype.
    pub(super) fn half(&self, name: &str) -> Result<(DeviceRange<'_>, HyperDtype), EngError> {
        match self.weights.get(name) {
            Some(CudaWeight::Raw { ty, buffer, .. }) => {
                let dtype = match ty {
                    GgufType::F16 => HyperDtype::F16,
                    GgufType::Bf16 => HyperDtype::Bf16,
                    other => return Err(refuse(format!("{name} is stored as {other:?}"))),
                };
                Ok((buffer.range(0, buffer.bytes())?, dtype))
            }
            _ => Err(refuse(format!("{name} is not an uploaded half tensor"))),
        }
    }

    // ------------------------------------------------------------------ launches

    /// One EXL3 projection of the row `x` into `out`. The scratch is the plan's: rotated inputs,
    /// the split-K partial and the sticky arithmetic flag, which the step reads once per token.
    pub(super) fn project(
        &self,
        prefix: &str,
        x: DeviceRange<'_>,
        out: DeviceRange<'_>,
    ) -> Result<(), EngError> {
        let weight = self.exl3(prefix)?;
        let view = weight.view()?;
        let io = Exl3Io {
            x,
            xh: self.slot(Slot::Exl3Xh)?,
            z: self.slot(Slot::Exl3Z)?,
            out,
            flag: self.slot(Slot::Exl3Flag)?,
            rows: 1,
            fused: false,
        };
        self.ctx.exl3_linear(self.stream, view.linear_call(&io))
    }

    /// `width` floats of `row` from float column `col`.
    pub(super) fn columns<'r>(
        row: &DeviceRange<'r>,
        col: usize,
        width: usize,
    ) -> Result<DeviceRange<'r>, EngError> {
        row.buffer().range(row.offset() + col * F32, width * F32)
    }

    fn hyper_shape(&self) -> HyperShape {
        HyperShape {
            m: 1,
            branches: self.geometry.branches,
            hidden: self.geometry.hidden,
            rank: self.geometry.rank,
        }
    }

    /// The read of the hyper-connection site under `prefix`: the mixed row of `site` and, with an
    /// inject projection, its gate.
    pub(super) fn hyper_read(&self, site: Site, prefix: &str) -> Result<(), EngError> {
        let inject = site != Site::Final;
        let (norm, dtype) = self.half(&format!("{prefix}.hc_norm.weight"))?;
        let (down, _) = self.half(&format!("{prefix}.input_mix_weight_down.weight"))?;
        let (up, _) = self.half(&format!("{prefix}.input_mix_weight_up.weight"))?;
        let inject_weight = if inject {
            Some(
                self.half(&format!("{prefix}.block_inject_weight.weight"))?
                    .0,
            )
        } else {
            None
        };
        self.ctx.hyper_read(
            self.stream,
            HyperReadCall {
                shape: self.hyper_shape(),
                eps: self.model.hyper.eps(),
                x: self.slot(Slot::Stream)?,
                weights: HyperWeights {
                    dtype,
                    norm,
                    down,
                    up,
                    inject: inject_weight,
                },
                scratch: HyperScratch {
                    normed: self.slot(Slot::HyperNormed(site))?,
                    low: self.slot(Slot::HyperLow(site))?,
                    mix: self.slot(Slot::HyperMix(site))?,
                },
                mixed: self.slot(Slot::Mixed(site))?,
                gate: if inject {
                    Some(self.slot(Slot::Gate(site))?)
                } else {
                    None
                },
            },
        )
    }

    /// `Stream += gate * block` for the site.
    pub(super) fn hyper_apply(&self, site: Site) -> Result<(), EngError> {
        self.ctx.hyper_apply(
            self.stream,
            HyperApplyCall {
                shape: self.hyper_shape(),
                x: self.slot(Slot::Stream)?,
                gate: self.slot(Slot::Gate(site))?,
                block: self.slot(Slot::BlockOut(site))?,
            },
        )
    }

    // ------------------------------------------------------------------ the MoE seam

    pub(super) fn moe_dims(&self) -> MoeDims {
        MoeDims {
            rows: 1,
            hidden: self.geometry.hidden,
            n_expert: self.geometry.experts,
            top_k: self.geometry.top_k,
            inter: self.geometry.inter,
            shared_inter: self.geometry.shared_inter,
        }
    }

    /// The pack keys of the MoE tensors of `layer` in the uploaded set, by role. The routed banks
    /// are the three `Exl3Bank` entries named by [`bank_key`].
    pub(super) fn moe_roles(layer: usize) -> Vec<(Role, String)> {
        let at = |suffix: &str| layer_name(layer, suffix);
        vec![
            (Role::FfnGateInp, at("mlp.gate.weight")),
            (Role::FfnGateInpShexp, at("mlp.shared_expert_gate.weight")),
            (Role::FfnGateExps, bank_key(layer, "gate_proj")),
            (Role::FfnUpExps, bank_key(layer, "up_proj")),
            (Role::FfnDownExps, bank_key(layer, "down_proj")),
            (Role::FfnGateShexp, at("mlp.shared_expert.gate_proj")),
            (Role::FfnUpShexp, at("mlp.shared_expert.up_proj")),
            (Role::FfnDownShexp, at("mlp.shared_expert.down_proj")),
        ]
    }

    /// The buffers of the MoE block, views of the plan's slots: the block reads `Mixed(Mlp)` and
    /// writes `BlockOut(Mlp)` (no residual: the hyper apply adds it to the stream).
    pub(super) fn moe_bufs(&self) -> Result<Exl3MoeBufs<'_>, EngError> {
        Ok(Exl3MoeBufs {
            route: RouteBufs {
                resid_in: self.zeros.range(0, self.zeros.bytes())?,
                resid_out: self.slot(Slot::BlockOut(Site::Mlp))?,
                norm_row: self.slot(Slot::Mixed(Site::Mlp))?,
                logits: self.slot(Slot::RouterLogits)?,
                ids: self.slot(Slot::TopkId)?,
                weights: self.slot(Slot::TopkWeight)?,
                shared_gate: self.slot(Slot::SharedGate)?,
                status: self.status_word()?,
            },
            expert_gu: self.slot(Slot::ExpertGu)?,
            expert_act: self.slot(Slot::ExpertAct)?,
            expert_down: self.slot(Slot::ExpertDown)?,
            shared_gu: self.slot(Slot::SharedGu)?,
            shared_act: self.slot(Slot::SharedAct)?,
            shared_out: self.slot(Slot::SharedOut)?,
            scratch: Scratch {
                xh: self.slot(Slot::Exl3Xh)?,
                z: self.slot(Slot::Exl3Z)?,
                flag: self.slot(Slot::Exl3Flag)?,
            },
        })
    }

    /// The MoE block of `layer` at the phases `Route` to `Combine`, run by `moe`: the output lands
    /// in `BlockOut(Mlp)`.
    pub(super) fn moe_block(&self, layer: usize, moe: &mut dyn MoeBlock) -> Result<(), EngError> {
        let launch = Launch {
            ctx: self.ctx,
            stream: self.stream,
        };
        moe.run(&launch, layer, &self.moe_bufs()?, &self.moe_dims())
    }

    // ------------------------------------------------------------------ layers and the tail

    /// One layer at position `pos`, in the site order of the host reference.
    pub(super) fn run_layer(
        &self,
        layer: usize,
        pos: usize,
        moe: &mut dyn MoeBlock,
    ) -> Result<(), EngError> {
        self.hyper_read(Site::Attn, &layer_name(layer, "attn_hyper_connection"))?;
        self.mixer(layer, pos)?;
        self.hyper_apply(Site::Attn)?;
        self.hyper_read(Site::Mlp, &layer_name(layer, "mlp_hyper_connection"))?;
        self.moe_block(layer, moe)?;
        self.hyper_apply(Site::Mlp)
    }

    /// The final mixer and `lm_head`: the logits land in the `Logits` slot.
    pub(super) fn head(&self) -> Result<(), EngError> {
        self.hyper_read(Site::Final, MIXER)?;
        self.project(
            "lm_head",
            self.slot(Slot::Mixed(Site::Final))?,
            self.slot(Slot::Logits)?,
        )
    }

    /// Clears the sticky EXL3 flag and the status word: the first launch of a token.
    pub(super) fn begin_token(&self) -> Result<(), EngError> {
        let flag = self.slot(Slot::Exl3Flag)?;
        flag.buffer()
            .write_at(self.stream, flag.offset(), &[0u8; 4])?;
        let status = self.status_word()?;
        status
            .buffer()
            .write_at(self.stream, status.offset(), &[0u8; 4])
    }

    /// Queues the layers of the window for the token at `pos`.
    pub(super) fn run_layers(&self, pos: usize, moe: &mut dyn MoeBlock) -> Result<(), EngError> {
        for layer in self.layers.clone() {
            self.run_layer(layer, pos, moe)?;
        }
        Ok(())
    }

    /// `values` floats of `range` read back behind the queued work.
    pub(super) fn read_f32(
        &self,
        range: DeviceRange<'_>,
        values: usize,
    ) -> Result<Vec<f32>, EngError> {
        let mut bytes = vec![0u8; values * F32];
        range
            .buffer()
            .read_at(self.stream, range.offset(), &mut bytes)?;
        Ok(bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| f32::from_le_bytes(*word))
            .collect())
    }

    /// The status word and the EXL3 arithmetic flag, read back: `(status, flag)`.
    pub(super) fn health(&self) -> Result<(i32, u32), EngError> {
        let mut word = [0u8; 4];
        let status = self.status_word()?;
        status
            .buffer()
            .read_at(self.stream, status.offset(), &mut word)?;
        let status = i32::from_le_bytes(word);
        let flag = self.slot(Slot::Exl3Flag)?;
        flag.buffer()
            .read_at(self.stream, flag.offset(), &mut word)?;
        Ok((status, u32::from_le_bytes(word)))
    }
}

/// The key of the bank of every routed expert's `part` of `layer` in the uploaded set.
pub(super) fn bank_key(layer: usize, part: &str) -> String {
    format!("layer {layer} routed {part} bank")
}

/// The MoE block of a layer, as the step calls it between the MLP hyper read and its apply.
///
/// `run` queues the block for one row on `launch.stream`: it reads `bufs.route.norm_row` (the
/// hyper-mixed row, f32 `[hidden]`), writes the weighted picks plus the gated shared expert (no
/// residual) to `bufs.route.resid_out`, and returns once that output is complete and ordered on the
/// stream. `bufs.route.resid_in` is a zero row for the combine. It may wait on the stream and copy
/// (a partial expert bank does on a miss); the step does not rely on it being non-blocking.
pub(super) trait MoeBlock {
    fn run(
        &mut self,
        launch: &Launch<'_>,
        layer: usize,
        bufs: &Exl3MoeBufs<'_>,
        dims: &MoeDims,
    ) -> Result<(), EngError>;
}

/// The stand-in until the split block lands: Qwen3.6's EXL3 MoE block over routed experts that
/// are all resident, as three `Exl3Bank`s per layer in the uploaded set (keys from [`bank_key`]).
pub(super) struct AllResidentMoe<'w> {
    pub(super) weights: &'w CudaWeights,
}

impl MoeBlock for AllResidentMoe<'_> {
    fn run(
        &mut self,
        launch: &Launch<'_>,
        layer: usize,
        bufs: &Exl3MoeBufs<'_>,
        dims: &MoeDims,
    ) -> Result<(), EngError> {
        let roles = FlashStep::moe_roles(layer);
        let weights = Exl3MoeWeights::resolve_by(self.weights, layer, dims, |role| {
            roles
                .iter()
                .find(|(found, _)| *found == role)
                .map(|(_, name)| name.as_str())
                .ok_or_else(|| refuse(format!("layer {layer} has no key for {role:?}")))
        })?;
        run_exl3_moe(launch, &weights, bufs, dims, true)
    }
}

/// The trunk weights of `layer`'s MoE block that the split block needs, borrowed from the uploaded
/// set: the router, the shared-expert gate row and the three shared-expert matrices.
pub(super) fn split_trunk(
    weights: &CudaWeights,
    layer: usize,
) -> Result<SplitMoeTrunk<'_>, EngError> {
    let at = |suffix: &str| layer_name(layer, suffix);
    let raw = |name: String| match weights.get(&name) {
        Some(CudaWeight::Raw {
            ty: GgufType::F32,
            buffer,
            ..
        }) => Ok(buffer),
        _ => Err(refuse(format!("{name} is not an uploaded F32 tensor"))),
    };
    let matrix = |name: String| match weights.get(&name) {
        Some(CudaWeight::Exl3 { weight, .. }) => Ok(weight),
        _ => Err(refuse(format!("{name} is not an uploaded EXL3 matrix"))),
    };
    Ok(SplitMoeTrunk {
        router: raw(at("mlp.gate.weight"))?,
        router_shared: raw(at("mlp.shared_expert_gate.weight"))?,
        shared_gate: matrix(at("mlp.shared_expert.gate_proj"))?,
        shared_up: matrix(at("mlp.shared_expert.up_proj"))?,
        shared_down: matrix(at("mlp.shared_expert.down_proj"))?,
    })
}

/// The split block as the step's MoE block. `first_layer` is the model layer of the bank's row 0:
/// a bank over a window of layers is indexed from its own first row, a bank over the whole model
/// from 0.
pub(super) struct SplitMoeAt<'w> {
    pub(super) inner: SplitMoe<'w>,
    pub(super) first_layer: usize,
}

impl MoeBlock for SplitMoeAt<'_> {
    fn run(
        &mut self,
        launch: &Launch<'_>,
        layer: usize,
        bufs: &Exl3MoeBufs<'_>,
        dims: &MoeDims,
    ) -> Result<(), EngError> {
        let row = layer
            .checked_sub(self.first_layer)
            .ok_or_else(|| refuse(format!("layer {layer} is before the bank's first layer")))?;
        self.inner.run(launch, row, bufs, dims)
    }
}
