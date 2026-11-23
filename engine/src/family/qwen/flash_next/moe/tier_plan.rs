//! Placement of the routed experts of Qwen3.8 Flash-Next on one card: how many experts of every
//! layer stay in VRAM at a given context, which ones, and the slot table the device classifier
//! reads.
//!
//! WHAT: [`TierPlan::new`] settles the split before the first token. K/V and the indexer keys of
//! the context take VRAM first, the expert slots (the same count in every layer) take what is left,
//! and the experts outside the slots must fit the pinned host budget. [`TierPlan::hot_set`] then
//! picks the slot holders of every layer from a popularity profile and returns the slot table
//! ([`SlotTable`], one row of `-1` or a slot per layer) and the placement map
//! ([`expert_tier::Residency`](crate::expert_tier::Residency)). WHY: 24 GiB cannot hold 43.07 GiB of
//! EXL3 experts next to a long context, and the slot count shrinks as the context grows
//! (`docs/plans/flash-next-cuda.md`, memory budget).
//!
//! Reused unchanged from the Qwen3.6 tiered offload, which takes every number as an input:
//! `budget::plan_vram` (K/V first, bank the remainder, uniform slots),
//! `StaticProfile` (hottest experts per layer, ties to the lowest id) and `classify_reference`
//! (the oracle of `tiered_classify`). New here: the Flash-Next sizes, the cap of the slot count at the
//! expert count, the cold-expert host check, the placement map and the slot table. The expert
//! bank is a different object from Qwen3.6's Q4_K `ExpertBank`: a Flash-Next slot holds 1,881,600
//! bytes of EXL3 trellis and scales, and the bank itself (with the filler-expert rule for a missed
//! pick) is the CUDA step's job.
//!
//! The host check is the start's own: the store tier is chosen by the host plan's ladder over the
//! same host frame (`host_plan`), so the context choice and the start read the host one way.
//!
//! Pure host arithmetic: no device, no clock, no I/O.

// Staged: the Flash-Next CUDA step is the first caller.
#![cfg_attr(not(test), allow(dead_code))]

use crate::family::expert_bank::{BankGeometry, HostMode, bank_tier::TierChoice};
use crate::family::qwen::flash_next::host_plan::{HostFrame, HostPinPlan};
use crate::{
    EngError,
    expert_tier::{ExpertId, Placement, Residency},
    family::{
        device_budget::KvPlan,
        qwen::qwen36::tiered::{PolicyMode, TieredConfig, policy::StaticProfile},
        tier_budget::{BudgetError, ExpertSize, VramBudget, VramPlan, plan_vram},
        tiered_classify::{Classified, LengthMismatch, classify_reference},
    },
};

/// The sizes of the routed experts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in super::super) struct TierShape {
    /// Layers with routed experts (48).
    pub(in super::super) layers: usize,
    /// Routed experts per layer (512).
    pub(in super::super) experts: usize,
    /// Picks per token and layer (10).
    pub(in super::super) top_k: usize,
    /// Device bytes of one routed expert, identical for all of them (1,881,600 in the 3.05 pack).
    pub(in super::super) expert_bytes: u64,
}

impl TierShape {
    /// The shape from the header walk: every routed expert has the same size because every bank
    /// has one rate and the shapes are fixed (`PackReport::expert_bytes` is their sum).
    pub(in super::super) fn from_pack(
        layers: usize,
        experts: usize,
        top_k: usize,
        total_expert_bytes: u64,
    ) -> Result<Self, EngError> {
        let count = (layers as u64)
            .checked_mul(experts as u64)
            .filter(|&count| count > 0)
            .ok_or_else(|| refuse("the layer and expert counts must be nonzero".into()))?;
        if !total_expert_bytes.is_multiple_of(count) {
            return Err(refuse(format!(
                "{total_expert_bytes} expert bytes do not divide over {layers} x {experts} experts"
            )));
        }
        Ok(Self {
            layers,
            experts,
            top_k,
            expert_bytes: total_expert_bytes / count,
        })
    }

    /// Device bytes of one more slot in every layer.
    pub(in super::super) fn slot_cost(&self) -> u64 {
        self.layers as u64 * self.expert_bytes
    }
}

/// The indexer key bytes one position carries. Refused unless the key of `idx_dim` f32
/// values divides evenly over `ratio` positions.
pub(in super::super) fn indexer_key_bytes(idx_dim: u64, ratio: u64) -> Result<u64, EngError> {
    if ratio == 0 || !(idx_dim * 4).is_multiple_of(ratio) {
        return Err(refuse(format!(
            "the indexer key of {idx_dim} f32 values does not divide over {ratio} positions"
        )));
    }
    Ok(idx_dim * 4 / ratio)
}

/// K/V and indexer-key bytes of one context position: per QSA layer a K row and a V row of
/// `heads_kv * head_dim` values of `kv_elem_bytes`, and one f32 indexer key of `idx_dim` per `ratio`
/// positions. Int8 K/V (`kv_elem_bytes` 1) counts no scale bytes here; `KvGeometry::position_bytes`
/// (`context_budget.rs`) adds the f32 row scales and is what the step's budget uses.
/// Bytes one context position costs in the raw element size `kv_elem_bytes`. The per-layer
/// K/V payload is the shared arithmetic, mapped by element size; the indexer key and the
/// layer multiply are this family's geometry.
pub(in super::super) fn position_bytes(
    qsa_layers: u64,
    heads_kv: u64,
    head_dim: u64,
    kv_elem_bytes: u64,
    (idx_dim, ratio): (u64, u64),
) -> Result<u64, EngError> {
    let format = match kv_elem_bytes {
        1 => crate::family::kv_format::KvFormat::Int8 {
            group: 1,
            scale_bytes: 0,
        },
        2 => crate::family::kv_format::KvFormat::F16,
        4 => crate::family::kv_format::KvFormat::F32,
        other => {
            return Err(refuse(format!(
                "the K/V element size {other} bytes is neither 1, 2 nor 4"
            )));
        }
    };
    let kv =
        format.layer_position_bytes(crate::family::kv_format::KvRowShape { heads_kv, head_dim })?;
    indexer_key_bytes(idx_dim, ratio)?
        .checked_add(kv)
        .and_then(|per_position| per_position.checked_mul(qsa_layers))
        .ok_or_else(|| refuse("the bytes of one position overflow".into()))
}

/// What the plan reads. The VRAM and host fields are the caller's measurements.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in super::super) struct TierInputs {
    pub(in super::super) shape: TierShape,
    pub(in super::super) vram: VramBudget,
    /// Context positions and the K/V plus indexer bytes of one.
    pub(in super::super) kv: KvPlan,
    /// The host side: the machine and the process terms, the same frame the start's host plan
    /// is built from.
    pub(in super::super) host: HostFrame,
    /// The staging ring's backing bytes the disk tier would charge (the same
    /// infallible placeholder the start's two-phase choice charges first:
    /// the pooled depth's records in flight, from the one shared constant
    /// `rung_ring_placeholder_bytes`). Zero offers no disk tier, so a path
    /// without a ring source keeps today's pinned-store refusal.
    pub(in super::super) ring_backing_bytes: u64,
}

impl TierInputs {
    /// The inputs of a load, from what the machine reports: `free_vram` is the device's free memory
    /// measured after the context exists and before any weight is resident (the desktop and other
    /// processes are already excluded), `(mem_total, mem_available)` is the admission snapshot
    /// of the host — `MemTotal` and the cgroup-clamped budget (`HostFacts::host_memory`;
    /// `capability::probe` reports the same pair). The device reserves are the fixed admission ledger of Qwen3.6's
    /// `enginesrv check` (`qwen36::admission_sketch`): the measured runtime overhead floor for the
    /// kernel arena and the CUDA context, and the admission margin, the minimum spare and the
    /// uncertainty band as the margin (`desktop_reserve` of `VramBudget`, whose error text still says
    /// "desktop reserve"), plus the ledger's 64 MiB chunked-attention part charge, which stands in for
    /// Flash-Next's own engine state until the step's arena plan supplies the real term. The host
    /// side is answered by the host plan's own arithmetic over `host` (see [`TierPlan::new`]).
    /// The caller adds the live shared-prefill scratch
    /// through `with_prefill_arena` before selecting the expert slots; the legacy path adds zero.
    pub(in super::super) fn measured(
        shape: TierShape,
        (trunk_bytes, state_bytes): (u64, u64),
        kv: KvPlan,
        free_vram: u64,
        host: HostFrame,
        ring_backing_bytes: u64,
    ) -> Self {
        use crate::family::admission_sketch::{
            ADMISSION_UNCERTAINTY_BAND_BYTES, FIXED_ADMISSION_MARGIN_BYTES,
            PREFILL_PART_CHARGE_BYTES, PREFILL_SPARE_MIN_BYTES,
        };
        // The runtime floor of the device policy table, charged as one term: Qwen3.6's
        // measured value, borrowed until Flash-Next measures its own.
        let runtime_overhead_floor = crate::family::mem::DEVICE.runtime_overhead_floor;
        Self {
            shape,
            vram: VramBudget {
                usable: free_vram,
                non_expert: trunk_bytes.saturating_add(state_bytes),
                state: PREFILL_PART_CHARGE_BYTES as u64,
                arena: runtime_overhead_floor,
                desktop_reserve: FIXED_ADMISSION_MARGIN_BYTES as u64
                    + PREFILL_SPARE_MIN_BYTES as u64
                    + ADMISSION_UNCERTAINTY_BAND_BYTES,
            },
            kv,
            ring_backing_bytes,
            host,
        }
    }

    /// Adds the live shared-prefill arena without reclaiming the measured runtime reserves.
    /// Zero preserves the legacy inputs byte-for-byte.
    pub(in super::super) fn with_prefill_arena(mut self, bytes: u64) -> Result<Self, EngError> {
        self.vram.arena = self
            .vram
            .arena
            .checked_add(bytes)
            .ok_or_else(|| refuse("prefill arena bytes overflow".into()))?;
        Ok(self)
    }

    /// The cost model's card: an RTX 3090 (24 GiB) with a 62 GiB host and fixed reserves of 64 MiB of
    /// engine state, 512 MiB of kernel scratch and 1400 MiB for the desktop and the allocator,
    /// chosen so the slot counts match the residency cost model (142, 180 and 208 slots at 262,144,
    /// 131,072 and 32,768 positions). A test fixture: the load uses [`Self::measured`].
    #[cfg(test)]
    pub(in super::super) fn cost_model_3090(
        shape: TierShape,
        (trunk_bytes, state_bytes): (u64, u64),
        kv: KvPlan,
        ring_backing_bytes: u64,
    ) -> Self {
        use crate::family::qwen::flash_next::host_plan::HostMemory;
        const GIB: u64 = 1 << 30;
        const MIB: u64 = 1 << 20;
        Self {
            shape,
            vram: VramBudget {
                usable: 24 * GIB,
                non_expert: trunk_bytes + state_bytes,
                state: 64 * MIB,
                arena: 512 * MIB,
                desktop_reserve: 1400 * MIB,
            },
            kv,
            ring_backing_bytes,
            host: HostFrame::with_default_terms(HostMemory::from_pair(62 * GIB, 62 * GIB)),
        }
    }
}

/// The settled split.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in super::super) struct TierPlan {
    shape: TierShape,
    slots_per_layer: usize,
    vram: VramPlan,
    cold_bytes: u64,
    /// Host RAM left for the page cache once the process, the store and the OS floor are charged.
    host_spare: u64,
    /// The store tier the start's own ladder answers over the same host frame: the pinned
    /// store while it fits (the spare-place policy is the loader's, not the context budget's),
    /// else the disk tier the ring serves through.
    store_tier: TierChoice,
}

fn refuse(detail: String) -> EngError {
    EngError::MemoryPlan(format!("Flash-Next expert tier: {detail}"))
}

fn budget_error(error: BudgetError) -> EngError {
    refuse(format!("{error:?}"))
}

impl TierPlan {
    /// Refuses, with the budget that lost, a context whose K/V leaves fewer than `top_k` slots per
    /// layer, a K/V that does not fit the card (it never spills to the host), and a split whose
    /// cold experts exceed the host's page-locked budget.
    ///
    /// The host side is the start's own: the store tier is chosen by the same ladder
    /// (`HostPinPlan::choose_store_tier`) over the same frame (`inputs.host`: the machine, the OS
    /// floor, the process terms), at the slots the card leaves, so a context this plan admits is
    /// a host plan the start admits.
    pub(in super::super) fn new(inputs: &TierInputs) -> Result<Self, EngError> {
        let shape = inputs.shape;
        if shape.top_k == 0 || shape.top_k > shape.experts {
            return Err(refuse(format!(
                "top_k {} is outside 1..={}",
                shape.top_k, shape.experts
            )));
        }
        let size = ExpertSize {
            bytes: shape.expert_bytes,
            layers: shape.layers,
        };
        let vram =
            plan_vram(&inputs.vram, &inputs.kv, &size, shape.top_k, 0).map_err(budget_error)?;
        let slots_per_layer = vram.bank_slots_per_layer.min(shape.experts);
        let cold_bytes = (shape.experts - slots_per_layer) as u64 * shape.slot_cost();
        let record_bytes = usize::try_from(shape.expert_bytes)
            .map_err(|_| refuse("the expert record does not fit usize".into()))?;
        // The host arithmetic never reads `top_k`, so the bank's own slot minimum (one filler
        // slot beyond the picks) is the loader's to enforce, not this plan's.
        let geometry = BankGeometry {
            layers: shape.layers,
            experts: shape.experts,
            top_k: shape.top_k,
            slots: slots_per_layer,
            record_bytes,
        };
        let HostFrame { memory, terms } = inputs.host;
        let ring = inputs.ring_backing_bytes;
        let outside = shape.experts - slots_per_layer;
        let missed = |why: String| {
            refuse(format!(
                "{outside} experts outside the {slots_per_layer} slots of each layer need \
                 {cold_bytes} pinned host bytes, the host budget of {} bytes cannot carry them \
                 beside the process and the OS floor: {why}",
                memory.available
            ))
        };
        let tier = HostPinPlan::choose_store_tier(&geometry, &memory, &terms, ring)
            .map_err(|refused| missed(refused.describe()))?;
        let (store_tier, host_spare, cold_bytes) = match tier {
            TierChoice::Disk if ring == 0 => {
                // No ring is offered on this path: the pinned-store refusal stands.
                let pinned = HostPinPlan::new(&geometry, HostMode::Exclusive, &memory, &terms);
                return Err(missed(match pinned {
                    Err(refused) => refused.describe(),
                    Ok(_) => "the ladder offered the disk tier without a ring".into(),
                }));
            }
            TierChoice::Disk => {
                let plan = HostPinPlan::new_disk(&geometry, &memory, &terms, ring)?;
                (TierChoice::Disk, plan.page_cache_left, 0)
            }
            TierChoice::Inclusive => {
                let plan = HostPinPlan::new(&geometry, HostMode::Inclusive, &memory, &terms)?;
                (tier, plan.page_cache_left, cold_bytes)
            }
            TierChoice::Exclusive { .. } => {
                let plan = HostPinPlan::new(&geometry, HostMode::Exclusive, &memory, &terms)?;
                (tier, plan.page_cache_left, cold_bytes)
            }
            TierChoice::CardTiered { .. } | TierChoice::CardDisk { .. } => {
                return Err(refuse(
                    "the host ladder answered a card tier to a plan that passed no card figures"
                        .into(),
                ));
            }
        };
        Ok(Self {
            shape,
            slots_per_layer,
            vram,
            cold_bytes,
            host_spare,
            store_tier,
        })
    }

    /// Resident experts in every layer (at least `top_k`, at most the expert count).
    pub(in super::super) fn slots_per_layer(&self) -> usize {
        self.slots_per_layer
    }

    /// Device bytes of the slots of all layers.
    pub(in super::super) fn bank_bytes(&self) -> u64 {
        self.slots_per_layer as u64 * self.shape.slot_cost()
    }

    /// Pinned host bytes of the experts outside the slots.
    pub(in super::super) fn cold_bytes(&self) -> u64 {
        self.cold_bytes
    }

    /// Host bytes still free for the page-locked experts after the plan.
    pub(in super::super) fn host_spare_bytes(&self) -> u64 {
        self.host_spare
    }

    /// The store tier the one ladder answered for this plan (the disk tier
    /// when the ring serves; the spare-place policy stays the loader's).
    pub(in super::super) fn store_tier(&self) -> TierChoice {
        self.store_tier
    }

    /// Device bytes the card has left after K/V, the trunk and the slots (less than one more slot).
    pub(in super::super) fn vram_left(&self) -> u64 {
        self.vram.vram_left
    }

    pub(in super::super) fn shape(&self) -> TierShape {
        self.shape
    }

    /// The slot holders of every layer: the `slots_per_layer` experts with the highest counts
    /// (`counts[layer][expert]`), ties to the lowest id. Holders get slots `0..slots` in id order.
    pub(in super::super) fn hot_set(&self, counts: &[Vec<u64>]) -> Result<HotSet, EngError> {
        let shape = self.shape;
        let config = TieredConfig {
            layers: shape.layers,
            experts: shape.experts,
            top_k: shape.top_k,
            bank_slots_per_layer: self.slots_per_layer,
            mode: PolicyMode::Static,
            swaps_per_token: 0.0,
            decay: 1.0,
        };
        let profile = StaticProfile::from_counts(&config, counts).map_err(refuse)?;
        let experts = u32::try_from(shape.experts)
            .map_err(|_| refuse("the expert count does not fit u32".into()))?;
        let mut residency = Residency::default();
        let mut rows = Vec::with_capacity(shape.layers);
        for layer in 0..shape.layers {
            let held = profile.held(layer);
            let mut row = vec![-1i32; shape.experts];
            for (slot, &expert) in held.iter().enumerate() {
                row[usize::from(expert)] = slot as i32;
            }
            for (expert, &slot) in row.iter().enumerate() {
                let id = ExpertId {
                    layer: layer as u32,
                    expert: expert as u32,
                };
                let placement = if slot >= 0 {
                    Placement::Gpu
                } else {
                    Placement::Cpu
                };
                residency.set_placement(id, experts, placement)?;
            }
            rows.push(row);
        }
        Ok(HotSet {
            table: SlotTable {
                slots: self.slots_per_layer,
                rows,
            },
            residency,
        })
    }
}

/// Per layer, the slot of every expert or -1: the `slot_row` the device classifier reads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(in super::super) struct SlotTable {
    slots: usize,
    rows: Vec<Vec<i32>>,
}

impl SlotTable {
    pub(in super::super) fn slots(&self) -> usize {
        self.slots
    }

    /// The row of `layer`, `experts` entries.
    pub(in super::super) fn row(&self, layer: usize) -> &[i32] {
        self.rows.get(layer).map_or(&[], Vec::as_slice)
    }

    pub(in super::super) fn slot_of(&self, layer: usize, expert: usize) -> Option<usize> {
        let slot = *self.rows.get(layer)?.get(expert)?;
        usize::try_from(slot).ok()
    }

    /// What `tiered_classify` computes for `layer` from the router's picks: the host oracle.
    pub(in super::super) fn classify(
        &self,
        layer: usize,
        ids: &[i32],
        weights: &[f32],
    ) -> Result<Classified, LengthMismatch> {
        classify_reference(self.row(layer), self.slots, ids, weights)
    }
}

/// The slot holders of every layer as a slot table and as a placement map: GPU for a holder, CPU
/// (page-locked host) for every other expert.
#[derive(Debug)]
pub(in super::super) struct HotSet {
    pub(in super::super) table: SlotTable,
    pub(in super::super) residency: Residency,
}

/// Router pick counts per layer and expert: the popularity profile of [`TierPlan::hot_set`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub(in super::super) struct RouteCounts {
    counts: Vec<Vec<u64>>,
}

impl RouteCounts {
    pub(in super::super) fn new(layers: usize, experts: usize) -> Self {
        Self {
            counts: vec![vec![0; experts]; layers],
        }
    }

    /// Counts the picks of one token in `layer`; refuses an id outside the layer.
    pub(in super::super) fn record(&mut self, layer: usize, picks: &[u32]) -> Result<(), EngError> {
        let row = self
            .counts
            .get_mut(layer)
            .ok_or_else(|| refuse(format!("layer {layer} is outside the profile")))?;
        for &expert in picks {
            *row.get_mut(expert as usize)
                .ok_or_else(|| refuse(format!("expert {expert} is outside layer {layer}")))? += 1;
        }
        Ok(())
    }

    pub(in super::super) fn counts(&self) -> &[Vec<u64>] {
        &self.counts
    }
}

#[cfg(test)]
pub(in super::super) mod tests;
