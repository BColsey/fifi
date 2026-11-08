//! The tiered runtime of the resident step: what the step needs, beyond the plain weights, to run
//! the tiered MoE block of `moe/tiered.rs` in every layer.
//!
//! WHAT: [`TieredRuntime`] owns the per-layer expert banks, the scratch buffers and the host
//! (`TieredHost`: policy, residency ledger, CPU pool), and borrows each layer's router and shared
//! expert weights from the loaded model ([`LayerParts`]); the routed expert stacks are NOT loaded
//! in this mode (the bank holds the hot experts). [`TieredRuntime::run_layer`] runs one layer's
//! block, [`TieredRuntime::end_token`] runs the token boundary: policy P decides the swaps and the
//! bank slots are filled with blocking writes.

use std::ops::Range;
use std::sync::Arc;

use super::mixer_weights::DenseWeight;
use super::moe::tiered::{
    LayerPlan, RouteRead, TieredBlock, TieredLayerHost, TieredScratch, run_moe_block_tiered,
};
use super::moe::{Launch, MoeBuffers, MoeDims, MoeLayerWeights, StackWeights};
use super::run_report::HostSyncs;
use crate::EngError;
use crate::family::qwen36::tiered::cpu_pool::MappedRegion;
use crate::family::qwen36::tiered::host::{BankDevice, HostError, TieredHost};
use crate::family::qwen36::tiered::locator::ExpertRanges;
use crate::family::qwen36::tiered::report::TokenReport;
use crate::kernels::cuda::{
    CudaContext, DeviceBuffer, DeviceEvent, DeviceStream, ExpertBank, PinnedBuf,
};

/// The weights of one layer's MoE block that stay in the loaded model: the router, the shared
/// gate vector and the shared expert's three projections.
pub(super) struct LayerParts<'w> {
    pub(super) router: &'w DeviceBuffer,
    pub(super) router_shared: &'w DeviceBuffer,
    pub(super) shared_gate: DenseWeight<'w>,
    pub(super) shared_up: DenseWeight<'w>,
    pub(super) shared_down: DenseWeight<'w>,
}

fn host_error(error: HostError) -> EngError {
    EngError::Backend {
        operation: "tiered_host",
        message: error.to_string(),
    }
}

/// The block-side view of the host: converts the plain route into the host's slices and its
/// plan and errors back, and issues the router-guess prefetch fills of the advice that
/// arrived for this layer (advice exists to be early: it publishes at the token boundary
/// and serves a later token, so issuing it here cannot change this token's result).
struct HostAdapter<'h, 'd> {
    host: &'h mut TieredHost,
    fills: &'d mut dyn BankDevice,
}

impl TieredLayerHost for HostAdapter<'_, '_> {
    fn begin_layer(&mut self, layer: usize, route: &RouteRead) -> Result<LayerPlan, EngError> {
        let plan = self
            .host
            .begin_layer(
                layer,
                &route.ids,
                &route.weights,
                &route.hidden,
                &route.resid,
            )
            .map_err(host_error)?;
        if let Some((target, advice)) = self.host.take_arrived_advice() {
            self.host
                .prefetch(self.fills, target, &advice)
                .map_err(host_error)?;
        }
        Ok(LayerPlan {
            slot_ids: plan.slot_ids,
            device_weights: plan.device_weights,
        })
    }

    fn finish_layer(&mut self, layer: usize) -> Result<Option<Vec<f32>>, EngError> {
        self.host.finish_layer(layer).map_err(host_error)
    }
}

/// The device half of a bank change: blocking writes of one expert's three slices.
pub(super) struct BankFill<'d> {
    pub(super) banks: &'d [ExpertBank],
    pub(super) stream: &'d DeviceStream,
}

impl BankDevice for BankFill<'_> {
    fn fill_slot(
        &mut self,
        layer: usize,
        slot: usize,
        ranges: &ExpertRanges,
        region: &Arc<dyn MappedRegion>,
    ) -> Result<(), HostError> {
        let device = |what: String| HostError::Device(what);
        let bank = self
            .banks
            .get(layer)
            .ok_or_else(|| device(format!("no bank for layer {layer}")))?;
        let bytes = region.bytes();
        let cut = |range: &Range<usize>| {
            bytes
                .get(range.clone())
                .ok_or_else(|| device(format!("expert range {range:?} is outside the region")))
        };
        bank.fill(
            self.stream,
            slot,
            cut(&ranges.gate)?,
            cut(&ranges.up)?,
            cut(&ranges.down)?,
        )
        .map_err(|error| device(format!("{error:?}")))
    }
}

/// Whether tiered fills run through the async channel (`Q36_TIERED_ASYNC_FILLS=1`): the
/// default keeps the blocking pageable writes of `ExpertBank::fill`. Ships dark until the
/// channel is device-qualified; the guess prefetch's measured decode cost is the reason
/// it exists (MR #14: ~0.3 ms of blocking issuance per fill on the step thread).
const ASYNC_FILLS_VAR: &str = "Q36_TIERED_ASYNC_FILLS";
fn async_fills_enabled(value: Option<&std::ffi::OsStr>) -> bool {
    value == Some(std::ffi::OsStr::new("1"))
}

/// The owned half of the async fill channel: a private stream, one expert-sized pinned
/// staging buffer and a ring of events. Built once at attach; `MAX_FILL_OUTSTANDING`
/// bounds the queue, and a fill that arrives at a full ring first waits its oldest event
/// (one fill's PCIe latency, off the common path).
/// One staging slot of the fill ring: a pinned buffer and ITS event. A single shared
/// buffer would serialize every fill (a pinned buffer refuses a rewrite while a copy is
/// pending), so the ring holds one buffer per outstanding fill and the wait amortizes
/// `MAX_FILL_OUTSTANDING` fills deep.
struct FillSlot {
    pinned: PinnedBuf,
    event: DeviceEvent,
}

pub(super) struct FillParts {
    stream: DeviceStream,
    /// Slots with no copy in flight, ready for reuse.
    idle: std::collections::VecDeque<FillSlot>,
    /// Slots of the copies in flight, oldest first.
    outstanding: std::collections::VecDeque<FillSlot>,
}

/// ponytail: one staging buffer serializes consecutive fills at issue time; a second
/// buffer only if the layer window shows the serialization.
const MAX_FILL_OUTSTANDING: usize = 8;

impl FillParts {
    fn new(ctx: &CudaContext, stage_bytes: usize) -> Result<Self, EngError> {
        Ok(Self {
            stream: ctx.stream()?,
            idle: (0..MAX_FILL_OUTSTANDING)
                .map(|_| {
                    Ok(FillSlot {
                        pinned: ctx.pinned_buf(stage_bytes)?,
                        event: ctx.event()?,
                    })
                })
                .collect::<Result<_, EngError>>()?,
            outstanding: std::collections::VecDeque::new(),
        })
    }
}

/// The async fill device: stages the three slices of one expert in pinned memory and
/// queues them onto the channel's own stream, recording one event per fill. Returns when
/// the fill is QUEUED; [`BankDevice::wait_fills`] is the completion. The copies cannot
/// race the step stream: a filled slot belongs to a layer whose bank the step is not
/// reading (mid-token prefetch) or to the boundary (after the token committed), and
/// publication happens only after `wait_fills`.
struct FillChannel<'c, 'b, 'p> {
    ctx: &'c CudaContext,
    banks: &'b [ExpertBank],
    parts: &'p mut FillParts,
}

impl BankDevice for FillChannel<'_, '_, '_> {
    fn fill_slot(
        &mut self,
        layer: usize,
        slot_index: usize,
        ranges: &ExpertRanges,
        region: &Arc<dyn MappedRegion>,
    ) -> Result<(), HostError> {
        let device = |what: String| HostError::Device(what);
        // The bank reference borrows only self.banks; the fill ring and its staging
        // buffers are self.parts, and the fill mutates those below.
        let banks = self.banks;
        let bank = banks
            .get(layer)
            .ok_or_else(|| device(format!("no bank for layer {layer}")))?;
        let (gate_up, down) = bank.slice_bytes().map_err(|e| device(format!("{e:?}")))?;
        let bytes = region.bytes();
        let cut = |range: &Range<usize>| {
            bytes
                .get(range.clone())
                .ok_or_else(|| device(format!("expert range {range:?} is outside the region")))
        };
        let (gate, up, down_slice) = (cut(&ranges.gate)?, cut(&ranges.up)?, cut(&ranges.down)?);
        for (what, got, want) in [
            ("gate", gate.len(), gate_up),
            ("up", up.len(), gate_up),
            ("down", down_slice.len(), down),
        ] {
            if got != want {
                return Err(device(format!(
                    "the {what} slice has {got} bytes, a slot holds {want}"
                )));
            }
        }
        // An idle slot, or the oldest outstanding one waited (its pinned buffer may
        // not be rewritten while its copy is pending).
        let mut slot = match self.parts.idle.pop_front() {
            Some(slot) => slot,
            None => {
                let mut oldest = self
                    .parts
                    .outstanding
                    .pop_front()
                    .ok_or_else(|| device(String::from("the fill ring lost its slots")))?;
                oldest
                    .pinned
                    .wait(&oldest.event)
                    .map_err(|e| device(format!("{e:?}")))?;
                oldest
            }
        };
        let stream = &self.parts.stream;
        slot.pinned
            .write(0, gate)
            .map_err(|e| device(format!("{e:?}")))?;
        slot.pinned
            .write(gate_up, up)
            .map_err(|e| device(format!("{e:?}")))?;
        slot.pinned
            .write(2 * gate_up, down_slice)
            .map_err(|e| device(format!("{e:?}")))?;
        self.ctx
            .copy_from_pinned(
                stream,
                &mut slot.pinned,
                0,
                bank.gate()
                    .buffer()
                    .range(slot_index * gate_up, gate_up)
                    .map_err(|e| device(format!("{e:?}")))?,
            )
            .map_err(|e| device(format!("{e:?}")))?;
        self.ctx
            .copy_from_pinned(
                stream,
                &mut slot.pinned,
                gate_up,
                bank.up()
                    .buffer()
                    .range(slot_index * gate_up, gate_up)
                    .map_err(|e| device(format!("{e:?}")))?,
            )
            .map_err(|e| device(format!("{e:?}")))?;
        self.ctx
            .copy_from_pinned(
                stream,
                &mut slot.pinned,
                2 * gate_up,
                bank.down()
                    .buffer()
                    .range(slot_index * down, down)
                    .map_err(|e| device(format!("{e:?}")))?,
            )
            .map_err(|e| device(format!("{e:?}")))?;
        stream
            .record(&slot.event)
            .map_err(|e| device(format!("{e:?}")))?;
        self.parts.outstanding.push_back(slot);
        Ok(())
    }

    fn wait_fills(&mut self) -> Result<(), HostError> {
        let device = |what: String| HostError::Device(what);
        for mut slot in self.parts.outstanding.drain(..) {
            slot.pinned
                .wait(&slot.event)
                .map_err(|e| device(format!("{e:?}")))?;
            self.parts.idle.push_back(slot);
        }
        Ok(())
    }
}

/// The fill device of one call: the blocking pageable writes, or the async channel.
enum TieredFill<'c, 'b, 'p> {
    Blocking(BankFill<'b>),
    Async(FillChannel<'c, 'b, 'p>),
}

impl BankDevice for TieredFill<'_, '_, '_> {
    fn fill_slot(
        &mut self,
        layer: usize,
        slot: usize,
        ranges: &ExpertRanges,
        region: &Arc<dyn MappedRegion>,
    ) -> Result<(), HostError> {
        match self {
            TieredFill::Blocking(fill) => fill.fill_slot(layer, slot, ranges, region),
            TieredFill::Async(channel) => channel.fill_slot(layer, slot, ranges, region),
        }
    }

    fn wait_fills(&mut self) -> Result<(), HostError> {
        match self {
            TieredFill::Blocking(fill) => fill.wait_fills(),
            TieredFill::Async(channel) => channel.wait_fills(),
        }
    }
}

pub(super) struct TieredRuntime<'w> {
    parts: Vec<LayerParts<'w>>,
    banks: Vec<ExpertBank>,
    scratch: TieredScratch,
    host: TieredHost,
    /// The async fill channel, built at attach when `Q36_TIERED_ASYNC_FILLS=1`.
    fills: Option<FillParts>,
    /// The host waits of the token so far: the layers' and, after `end_token`, the bank fills'.
    syncs: HostSyncs,
}

impl<'w> TieredRuntime<'w> {
    pub(super) fn new(
        ctx: &CudaContext,
        parts: Vec<LayerParts<'w>>,
        banks: Vec<ExpertBank>,
        scratch: TieredScratch,
        host: TieredHost,
    ) -> Result<Self, EngError> {
        if parts.len() != banks.len() {
            return Err(EngError::BadRequest(format!(
                "tiered runtime: {} layers of weights but {} banks",
                parts.len(),
                banks.len()
            )));
        }
        let fills = if async_fills_enabled(std::env::var_os(ASYNC_FILLS_VAR).as_deref()) {
            let (gate_up, down) = banks
                .first()
                .ok_or_else(|| {
                    EngError::BadRequest(String::from(
                        "tiered runtime: no banks to size the fill channel",
                    ))
                })?
                .slice_bytes()?;
            Some(FillParts::new(ctx, 2 * gate_up + down)?)
        } else {
            None
        };
        Ok(Self {
            parts,
            banks,
            scratch,
            host,
            fills,
            syncs: HostSyncs::default(),
        })
    }

    /// Runs the MoE block of `layer` with the host decision between the router and the experts.
    pub(super) fn run_layer(
        &mut self,
        launch: &Launch<'_>,
        layer: usize,
        bufs: &MoeBuffers<'_>,
        dims: &MoeDims,
    ) -> Result<(), EngError> {
        let missing = || EngError::BadRequest(format!("tiered runtime: layer {layer} has no bank"));
        let (parts, bank) = (
            self.parts.get(layer).ok_or_else(missing)?,
            self.banks.get(layer).ok_or_else(missing)?,
        );
        let weights = MoeLayerWeights {
            router: parts.router,
            router_shared: parts.router_shared,
            gate: StackWeights::Native(bank.gate()),
            up: StackWeights::Native(bank.up()),
            down: StackWeights::Native(bank.down()),
            shared_gate: parts.shared_gate,
            shared_up: parts.shared_up,
            shared_down: parts.shared_down,
        };
        let mut block = TieredBlock {
            weights: &weights,
            bufs,
            scratch: &mut self.scratch,
            dims,
        };
        let blocking = BankFill {
            banks: &self.banks,
            stream: launch.stream,
        };
        let mut channel = self.fills.as_mut().map(|parts| FillChannel {
            ctx: launch.ctx,
            banks: &self.banks,
            parts,
        });
        let mut fills = match channel.take() {
            Some(channel) => TieredFill::Async(channel),
            None => TieredFill::Blocking(blocking),
        };
        let made = run_moe_block_tiered(
            launch,
            &mut HostAdapter {
                host: &mut self.host,
                fills: &mut fills,
            },
            layer,
            &mut block,
        )?;
        self.syncs.add(made);
        Ok(())
    }

    pub(super) fn preload(&mut self, stream: &DeviceStream) -> Result<usize, EngError> {
        let mut fills = BankFill {
            banks: &self.banks,
            stream,
        };
        self.host.preload_all(&mut fills).map_err(host_error)
    }

    pub(super) fn abort_token(&mut self) {
        self.host.abort_token();
        self.syncs = HostSyncs::default();
    }

    /// The host waits of the token that just ended (layers and fills), and zero for the next one.
    pub(super) fn take_syncs(&mut self) -> HostSyncs {
        std::mem::take(&mut self.syncs)
    }

    pub(super) fn end_token(
        &mut self,
        ctx: &CudaContext,
        stream: &DeviceStream,
    ) -> Result<TokenReport, EngError> {
        let report = match self.fills.as_mut() {
            Some(parts) => {
                let channel = FillChannel {
                    ctx,
                    banks: &self.banks,
                    parts,
                };
                self.host
                    .end_token(&mut TieredFill::Async(channel))
                    .map_err(host_error)?
            }
            None => {
                let fills = BankFill {
                    banks: &self.banks,
                    stream,
                };
                self.host
                    .end_token(&mut TieredFill::Blocking(fills))
                    .map_err(host_error)?
            }
        };
        // The blocking path counts its FILL_WRITES host writes per swap; the channel
        // queues them, so the boundary's drain waits instead (not attributed here).
        if self.fills.is_none() {
            self.syncs.writes += report.swaps * ExpertBank::FILL_WRITES;
        }
        Ok(report)
    }
}
