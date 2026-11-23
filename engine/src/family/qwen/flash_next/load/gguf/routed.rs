//! Staged GGUF decode-only routed experts with a persistent disk-backed bank.
//!
//! The caller runs the router and shared expert. This component reads actual device router IDs,
//! updates a per-layer LRU and native expert stacks, then writes routed plus gated shared output.
//! It is not connected to production startup, admission, prefill or batching.
#![cfg_attr(
    not(test),
    allow(dead_code, reason = "staged for GGUF decode integration")
)]

use std::ops::Range;

use super::{
    GgufModel,
    expert_layout::{
        ExpertDeviceStacks, ExpertLayout, ExpertProjection, ExpertProjectionInputs, ExpertStacks,
        PinnedDiskRefill,
    },
};
use crate::{
    EngError,
    family::{
        exl3::{launch::MoeDims, moe::Exl3MoeBufs},
        expert_bank::{BankGeometry, ExpertBankState, HostMode, PassCopies, ReplaceOnMissLru},
        recipe::Role,
    },
    kernels::cuda::{
        CudaContext, DeviceBuffer, DeviceRange, DeviceStream,
        prefill::swiglu_rows::{SwigluRowsCall, SwigluRowsShape},
        resident::{
            moe_combine::{CombineCall, CombineOut, CombineShape, SharedTerm},
            quantize_q8_1::{QuantizeCall, q8_1_bytes},
        },
    },
};

use crate::family::qwen::flash_next::step::GgufDecodePrecision;

#[cfg(test)]
use crate::family::expert_bank::BankStats;

fn bad(detail: &str) -> EngError {
    EngError::BadRequest(format!("Flash-Next GGUF routed decode: {detail}"))
}

fn all(buffer: &DeviceBuffer) -> Result<DeviceRange<'_>, EngError> {
    buffer.range(0, buffer.bytes())
}

#[cfg(test)]
fn down_probe_range(
    enabled: bool,
    selection: Option<&str>,
    count: usize,
) -> Result<Option<Range<usize>>, EngError> {
    if !enabled {
        return if selection.is_none() {
            Ok(None)
        } else {
            Err(bad(
                "expert down layer selection requires the precision probe",
            ))
        };
    }
    let range = match selection {
        None => 0..count,
        Some(value) => {
            let (start, end) = value
                .split_once(':')
                .ok_or_else(|| bad("expert down layer selection must be start:end"))?;
            let parse = |s: &str| {
                s.parse::<usize>()
                    .map_err(|_| bad("invalid expert down layer index"))
            };
            parse(start)?..parse(end)?
        }
    };
    if range.start >= range.end || range.end > count {
        return Err(bad("expert down layer selection is out of range"));
    }
    Ok(Some(range))
}

#[cfg(test)]
fn down_probe_from_env(count: usize) -> Result<Option<Range<usize>>, EngError> {
    let enabled = std::env::var("FIFI_FLASH_GGUF_TEST_IQ4_F16_DOWN").as_deref() == Ok("1");
    let selection = match std::env::var("FIFI_FLASH_GGUF_TEST_IQ4_F16_DOWN_LAYERS") {
        Ok(value) => Some(value),
        Err(std::env::VarError::NotPresent) => None,
        Err(_) => return Err(bad("expert down layer selection is not Unicode")),
    };
    down_probe_range(enabled, selection.as_deref(), count)
}

/// The model and stream must outlive the bank. All launches and refills use that one stream.
/// Discard the entire owner after any failed call: policy and device state may be out of sync.
pub(in super::super::super) struct PersistentRouted<'a> {
    ctx: &'a CudaContext,
    stream: &'a DeviceStream,
    layers: Range<usize>,
    dims: MoeDims,
    state: ExpertBankState<ReplaceOnMissLru>,
    banks: Vec<ExpertDeviceStacks<'a>>,
    refills: Vec<PinnedDiskRefill<'a>>,
    slots: DeviceBuffer,
    qx: DeviceBuffer,
    qa: DeviceBuffer,
    #[cfg(test)]
    f16_down_probe: Option<Range<usize>>,
    /// The arithmetic of the expert projections: the shipped `Q8` until
    /// [`Self::set_decode_precision`].
    precision: GgufDecodePrecision,
    poisoned: bool,
}

impl<'a> PersistentRouted<'a> {
    /// Each selected layer begins with experts `0..slots`, one per slot. The caller must admit
    /// the per-layer stacks, private scratch and the sum of per-layer pinned records before
    /// constructing. Each pinned record remains owned until this component is dropped.
    pub(in super::super::super) fn new(
        ctx: &'a CudaContext,
        stream: &'a DeviceStream,
        model: &'a GgufModel,
        layers: Range<usize>,
        slots: usize,
    ) -> Result<Self, EngError> {
        ctx.check_launch_stream(stream)?;
        let count = model.config().layers();
        if layers.start >= layers.end || layers.end > count {
            return Err(bad("invalid layer window"));
        }
        #[cfg(test)]
        let f16_down_probe = down_probe_from_env(count)?;
        let shape = model.config().model_shape()?;
        if !shape.moe.renorm() {
            return Err(bad("router weights must be renormalized"));
        }
        let dims = MoeDims {
            rows: 1,
            hidden: shape.hyper.hidden(),
            n_expert: shape.moe.experts(),
            top_k: shape.moe.top_k(),
            inter: shape.moe.inter(),
            shared_inter: shape.moe.shared_inter(),
        };
        let mut max_record = 0usize;
        for layer in layers.clone() {
            let layout = ExpertLayout::for_layer(model, layer)?;
            ExpertStacks::new(&layout, slots)?;
            max_record = max_record.max(layout.bytes);
        }
        // BankGeometry's record width is policy bookkeeping, not a GGUF allocation size:
        // different layers can have different encoded layouts and part alignment.
        let record = max_record
            .checked_add(3)
            .ok_or_else(|| bad("record size overflows"))?
            & !3;
        let geometry = BankGeometry::new(count, dims.n_expert, dims.top_k, slots, record)?;
        let qx_bytes = q8_1_bytes(dims.hidden, 1).ok_or_else(|| bad("input scratch overflows"))?;
        let qa_bytes = q8_1_bytes(dims.inter, dims.top_k)
            .ok_or_else(|| bad("activation scratch overflows"))?;
        let ids_bytes = dims
            .top_k
            .checked_mul(4)
            .ok_or_else(|| bad("slot scratch overflows"))?;
        let mut row = vec![-1; dims.n_expert];
        for (slot, place) in row.iter_mut().take(slots).enumerate() {
            *place = i32::try_from(slot).map_err(|_| bad("slot index overflows"))?;
        }
        let rows = vec![row.as_slice(); count];
        let counts = vec![vec![0; dims.n_expert]; count];
        let policy = ReplaceOnMissLru::from_fill(geometry, &rows, &counts, None)?;
        let state = ExpertBankState::with_policy(geometry, HostMode::Disk, policy)?;
        let initial: Vec<u32> = (0..slots).map(|v| v as u32).collect();
        let mut banks = Vec::with_capacity(layers.len());
        let mut refills = Vec::with_capacity(layers.len());
        for layer in layers.clone() {
            let refill = PinnedDiskRefill::new(ctx, stream, model, layer)?;
            banks.push(ExpertDeviceStacks::from_model(
                ctx, stream, model, layer, &initial,
            )?);
            refills.push(refill);
        }
        Ok(Self {
            ctx,
            stream,
            layers,
            dims,
            state,
            banks,
            refills,
            slots: ctx.allocate(ids_bytes)?,
            qx: ctx.allocate(qx_bytes)?,
            qa: ctx.allocate(qa_bytes)?,
            #[cfg(test)]
            f16_down_probe,
            precision: GgufDecodePrecision::default(),
            poisoned: false,
        })
    }

    /// Selects the arithmetic of the expert projections ([`GgufDecodePrecision`]). The q8_1
    /// scratch stays allocated either way.
    pub(in super::super::super) fn set_decode_precision(&mut self, precision: GgufDecodePrecision) {
        self.precision = precision;
    }

    /// Consumes the caller's GPU router output in rank order; the shared expert output and gate
    /// must already be queued on this stream. Returns the actual bank hit/miss plan.
    pub(in super::super::super) fn run(
        &mut self,
        layer: usize,
        bufs: &Exl3MoeBufs<'_>,
        dims: &MoeDims,
    ) -> Result<PassCopies, EngError> {
        if self.poisoned || self.state.is_poisoned() {
            return Err(bad("bank is poisoned"));
        }
        let result = self.run_inner(layer, bufs, dims);
        if result.is_err() {
            self.poisoned = true;
            self.state.poison();
        }
        result
    }

    fn run_inner(
        &mut self,
        layer: usize,
        bufs: &Exl3MoeBufs<'_>,
        dims: &MoeDims,
    ) -> Result<PassCopies, EngError> {
        if *dims != self.dims || !self.layers.contains(&layer) {
            return Err(bad("dimensions or layer do not match the owner"));
        }
        let picks = read_picks(self.stream, bufs.route.ids, dims)?;
        let index = layer - self.layers.start;
        let bank = &mut self.banks[index];
        let copies = self.refills[index].step(bank, &mut self.state, &picks)?;
        let row = self.state.slot_row(layer);
        let mut slot_bytes = Vec::with_capacity(dims.top_k * 4);
        #[cfg(test)]
        let mut host_slots = Vec::with_capacity(dims.top_k);
        for &id in &picks {
            let slot = row[id as usize];
            if slot < 0 {
                return Err(bad("a routed pick remains absent after refill"));
            }
            slot_bytes.extend_from_slice(&slot.to_ne_bytes());
            #[cfg(test)]
            host_slots.push(slot);
        }
        self.slots.write_at(self.stream, 0, &slot_bytes)?;
        launch_experts(
            self.ctx,
            self.stream,
            bank,
            &self.state,
            (&self.slots, &self.qx, &self.qa),
            bufs,
            dims,
            self.precision,
            #[cfg(test)]
            TestProbes {
                layer,
                slots: &host_slots,
                iq4_down: self
                    .f16_down_probe
                    .as_ref()
                    .is_some_and(|selected| selected.contains(&layer)),
            },
        )?;
        Ok(copies)
    }

    #[cfg(test)]
    pub(in super::super::super) fn stats(&self) -> &BankStats {
        self.state.stats()
    }

    #[cfg(test)]
    pub(in super::super::super) fn slot_row(&self, layer: usize) -> &[i32] {
        self.state.slot_row(layer)
    }
}

fn read_picks(
    stream: &DeviceStream,
    ids: DeviceRange<'_>,
    dims: &MoeDims,
) -> Result<Vec<i32>, EngError> {
    let bytes = dims
        .top_k
        .checked_mul(4)
        .ok_or_else(|| bad("pick size overflows"))?;
    if ids.bytes() < bytes {
        return Err(bad("router IDs range is too short"));
    }
    let mut raw = vec![0; bytes];
    ids.buffer().read_at(stream, ids.offset(), &mut raw)?;
    let mut picks = Vec::with_capacity(dims.top_k);
    for entry in raw.chunks_exact(4) {
        let id = i32::from_ne_bytes(entry.try_into().expect("four bytes"));
        if id < 0 || id as usize >= dims.n_expert || picks.contains(&id) {
            return Err(bad("router produced invalid or duplicate expert IDs"));
        }
        picks.push(id);
    }
    Ok(picks)
}

fn launch_experts(
    ctx: &CudaContext,
    stream: &DeviceStream,
    bank: &ExpertDeviceStacks<'_>,
    state: &ExpertBankState<ReplaceOnMissLru>,
    scratch: (&DeviceBuffer, &DeviceBuffer, &DeviceBuffer),
    bufs: &Exl3MoeBufs<'_>,
    dims: &MoeDims,
    precision: GgufDecodePrecision,
    #[cfg(test)] probes: TestProbes<'_>,
) -> Result<(), EngError> {
    let (ids, qx, qa) = scratch;
    let (gate, up, down) = (
        bank.projection(state, Role::ExpertGate)?,
        bank.projection(state, Role::ExpertUp)?,
        bank.projection(state, Role::ExpertDown)?,
    );
    if (gate.k, gate.rows) != (dims.hidden, dims.inter)
        || (up.k, up.rows) != (dims.hidden, dims.inter)
        || (down.k, down.rows) != (dims.inter, dims.hidden)
    {
        return Err(bad("expert projection shapes disagree with the model"));
    }
    if precision == GgufDecodePrecision::Exact {
        return launch_exact(ctx, stream, (gate, up, down), all(ids)?, bufs, dims);
    }
    #[cfg(test)]
    if super::native_f16::enabled() {
        return launch_unquantized(ctx, stream, (gate, up, down), probes.slots, bufs, dims);
    }
    ctx.quantize_q8_1(
        stream,
        QuantizeCall {
            x: bufs.route.norm_row,
            y: all(qx)?,
            k: dims.hidden,
            ncols: 1,
        },
    )?;
    let up_out = up_range(bufs, dims.inter)?;
    for (projection, output) in [(gate, bufs.expert_gu), (up, up_out)] {
        projection.launch(
            ctx,
            stream,
            ExpertProjectionInputs {
                ids: all(ids)?,
                x: all(qx)?,
                x_per_slot: false,
                output,
                output_stride: dims.inter * 2,
                used: dims.top_k,
            },
        )?;
    }
    swiglu(ctx, stream, bufs, dims)?;
    #[cfg(test)]
    let f16_down = if probes.iq4_down {
        use crate::kernels::cuda::resident::matvec_q::QuantType;
        match down.ty {
            QuantType::Iq4Nl => {
                let calls = super::native_f16::plan(
                    &down,
                    probes.slots,
                    bufs.expert_act,
                    bufs.expert_down,
                    true,
                    down.rows,
                )?;
                super::native_f16::launch(ctx, stream, down.ty, &calls)?;
                eprintln!(
                    "GGUF test-only expert down: layer={} IQ4_NL f16-intermediate",
                    probes.layer
                );
                true
            }
            QuantType::Q20 => {
                eprintln!(
                    "GGUF test-only expert down: layer={} Q2_0 native Q8 fallback",
                    probes.layer
                );
                false
            }
            _ => {
                return Err(bad(
                    "unsupported expert down type for mixed-precision probe",
                ));
            }
        }
    } else {
        false
    };
    #[cfg(not(test))]
    let f16_down = false;
    if !f16_down {
        ctx.quantize_q8_1_rows(
            stream,
            QuantizeCall {
                x: bufs.expert_act,
                y: all(qa)?,
                k: dims.inter,
                ncols: dims.top_k,
            },
        )?;
        down.launch(
            ctx,
            stream,
            ExpertProjectionInputs {
                ids: all(ids)?,
                x: all(qa)?,
                x_per_slot: true,
                output: bufs.expert_down,
                output_stride: dims.hidden,
                used: dims.top_k,
            },
        )?;
    }
    combine(ctx, stream, bufs, dims)
}

/// The `up` half of the gate/up rows, which follow the interleaved `[used][gate | up]` layout.
fn up_range<'a>(bufs: &'a Exl3MoeBufs<'a>, inter: usize) -> Result<DeviceRange<'a>, EngError> {
    let up_offset = inter
        .checked_mul(4)
        .ok_or_else(|| bad("up offset overflows"))?;
    if bufs.expert_gu.bytes() < up_offset {
        return Err(bad("expert gate/up range is too short"));
    }
    bufs.expert_gu.buffer().range(
        bufs.expert_gu.offset() + up_offset,
        bufs.expert_gu.bytes() - up_offset,
    )
}

/// The exact twin of the shipped launch ([`GgufDecodePrecision::Exact`]): the same launches,
/// buffer layout, SwiGLU and combine, with unquantized activations and exactly decoded weights.
fn launch_exact(
    ctx: &CudaContext,
    stream: &DeviceStream,
    (gate, up, down): (
        ExpertProjection<'_>,
        ExpertProjection<'_>,
        ExpertProjection<'_>,
    ),
    ids: DeviceRange<'_>,
    bufs: &Exl3MoeBufs<'_>,
    dims: &MoeDims,
) -> Result<(), EngError> {
    let up_out = up_range(bufs, dims.inter)?;
    let twice = dims
        .inter
        .checked_mul(2)
        .ok_or_else(|| bad("expert width overflows"))?;
    let inputs = |x, x_per_slot, output, output_stride| ExpertProjectionInputs {
        ids,
        x,
        x_per_slot,
        output,
        output_stride,
        used: dims.top_k,
    };
    gate.launch_exact(
        ctx,
        stream,
        inputs(bufs.route.norm_row, false, bufs.expert_gu, twice),
    )?;
    up.launch_exact(
        ctx,
        stream,
        inputs(bufs.route.norm_row, false, up_out, twice),
    )?;
    swiglu(ctx, stream, bufs, dims)?;
    down.launch_exact(
        ctx,
        stream,
        inputs(bufs.expert_act, true, bufs.expert_down, dims.hidden),
    )?;
    combine(ctx, stream, bufs, dims)
}

fn swiglu(
    ctx: &CudaContext,
    stream: &DeviceStream,
    bufs: &Exl3MoeBufs<'_>,
    dims: &MoeDims,
) -> Result<(), EngError> {
    ctx.swiglu_rows(
        stream,
        SwigluRowsCall {
            shape: SwigluRowsShape {
                rows: dims.top_k,
                n_ff: dims.inter,
                up_off: dims.inter,
                ld_in: dims.inter * 2,
                ld_out: dims.inter,
            },
            gu: bufs.expert_gu,
            act: bufs.expert_act,
        },
    )
}

fn combine(
    ctx: &CudaContext,
    stream: &DeviceStream,
    bufs: &Exl3MoeBufs<'_>,
    dims: &MoeDims,
) -> Result<(), EngError> {
    ctx.moe_combine(
        stream,
        CombineCall {
            shape: CombineShape {
                m: 1,
                n: dims.hidden,
                top_k: dims.top_k,
                ld_down: dims.hidden,
            },
            res: bufs.route.resid_in,
            down: bufs.expert_down,
            weights: bufs.route.weights,
            shared: Some(SharedTerm {
                gate: bufs.route.shared_gate,
                row: bufs.shared_out,
            }),
            out: CombineOut::To(bufs.route.resid_out),
        },
    )
}

/// Test-only selectors carried into the expert launch.
#[cfg(test)]
struct TestProbes<'s> {
    layer: usize,
    /// The bank slot holding each rank's expert.
    slots: &'s [i32],
    iq4_down: bool,
}

/// Test-only: gate, up and down of every routed expert through the dequantizing GEMM with
/// unquantized activations, then the usual SwiGLU and combine.
#[cfg(test)]
fn launch_unquantized(
    ctx: &CudaContext,
    stream: &DeviceStream,
    (gate, up, down): (
        ExpertProjection<'_>,
        ExpertProjection<'_>,
        ExpertProjection<'_>,
    ),
    slots: &[i32],
    bufs: &Exl3MoeBufs<'_>,
    dims: &MoeDims,
) -> Result<(), EngError> {
    let up_out = up_range(bufs, dims.inter)?;
    let twice = dims
        .inter
        .checked_mul(2)
        .ok_or_else(|| bad("expert width overflows"))?;
    for (projection, output) in [(gate, bufs.expert_gu), (up, up_out)] {
        let calls = super::native_f16::plan(
            &projection,
            slots,
            bufs.route.norm_row,
            output,
            false,
            twice,
        )?;
        super::native_f16::launch(ctx, stream, projection.ty, &calls)?;
    }
    swiglu(ctx, stream, bufs, dims)?;
    let calls = super::native_f16::plan(
        &down,
        slots,
        bufs.expert_act,
        bufs.expert_down,
        true,
        down.rows,
    )?;
    super::native_f16::launch(ctx, stream, down.ty, &calls)?;
    combine(ctx, stream, bufs, dims)
}

#[cfg(test)]
mod tests {
    use super::down_probe_range;

    #[test]
    fn down_probe_selection_is_explicit_and_model_bounded() {
        assert_eq!(down_probe_range(false, None, 48).unwrap(), None);
        assert_eq!(down_probe_range(true, None, 48).unwrap(), Some(0..48));
        assert_eq!(down_probe_range(true, Some("0:2"), 48).unwrap(), Some(0..2));
        assert_eq!(down_probe_range(true, Some("2:5"), 48).unwrap(), Some(2..5));
        assert!(down_probe_range(false, Some("0:2"), 48).is_err());
        assert!(down_probe_range(true, None, 0).is_err());
        for value in [
            "", ":", "0:0", "5:2", "0:49", "-1:2", "0:1:2", "a:2", " 0:2",
        ] {
            assert!(down_probe_range(true, Some(value), 48).is_err(), "{value}");
        }
    }
}
