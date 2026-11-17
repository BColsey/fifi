//! The chunk-facing contract over the bank: what a prefill chunk does to the expert bank,
//! formalized as one type so the chunked prefill paths of both families code against the same
//! shape. The served path of Flash-Next (`moe_block::run_rows`) still carries its own inline
//! copy of this sequence; the two can drift until the shared prefill pass routes `run_rows`
//! through the adapter. The adapter changes nothing that runs today.
//!
//! The sequence of one chunk, per `run_rows`:
//!
//! 1. The router runs over every row and its picks come back to the host once (the readback
//!    the chunk path already pays for its own classify work).
//! 2. [`AdmitPlan::of`] decides the chunk admit: the union of the chunk's picks, whether the
//!    union fits the bank's slots, and how many of the union are not resident.
//! 3. If the union fits and anything is missing, the caller drains its own stream, then
//!    the plan's union is admitted in one pass ([`PrefillAdapter::admit_planned`] over a
//!    plan [`PrefillAdapter::admit_plan`] computed, so a conditionally-draining caller
//!    does not pay the union twice). An admit that does not run returns `None`. An
//!    already-resident union needs only its slot mapping; an over-capacity union requires
//!    smaller sub-blocks or the legacy per-row step tail.
//! 4. Whenever the returned copies have ops, the caller uploads the bank's slot row again
//!    before the next classify.
//!
//! The ring path streams cold experts without admitting: `place_of`, `stream_place` and
//! `synchronize_store` are forwarded with their contracts, so a ring window never has to reach
//! around the adapter.

use super::moe_block::pick_union;

#[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
pub(super) fn drain_before_admit<T>(
    drain: impl FnOnce() -> Result<(), crate::EngError>,
    admit: impl FnOnce() -> Result<T, crate::EngError>,
) -> Result<T, crate::EngError> {
    drain()?;
    admit()
}

/// The chunk-admit decision of one layer: the union of the chunk's pick rows in first-seen
/// order, whether that union fits the bank's slots, and how many of the union are not
/// resident. An admit runs only when `fits && missing > 0`.
///
/// Ids must be inside `0..experts`: `pick_union` indexes its seen table by the id and panics
/// otherwise, and `slot_row` is indexed the same way. The served caller refuses faults (-1)
/// and out-of-range ids before this (`route_rows`), and [`PrefillAdapter::admit_plan`] refuses
/// them itself.
///
/// Pure host data: it is computed from a slot row, so the host tests pin it without a card.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::family::flash_next) struct AdmitPlan {
    /// The chunk's distinct picks, in first-seen order across the rows.
    pub(in crate::family::flash_next) union: Vec<u16>,
    /// Whether `union` fits the bank's slots; an over-capacity union is never admitted whole.
    pub(in crate::family::flash_next) fits: bool,
    /// How many experts of `union` the slot row does not hold.
    pub(in crate::family::flash_next) missing: usize,
}

impl AdmitPlan {
    /// The decision over one layer's slot row. `slot_row` is the bank's row of `layer`
    /// (`DeviceBank::slot_row`), `experts` and `slots` its geometry.
    pub(in crate::family::flash_next) fn of(
        picks_rows: &[&[i32]],
        experts: usize,
        slots: usize,
        slot_row: &[i32],
    ) -> Self {
        let union = pick_union(picks_rows, experts);
        let missing = union
            .iter()
            .filter(|&&id| slot_row[usize::from(id)] < 0)
            .count();
        Self {
            fits: union.len() <= slots,
            missing,
            union,
        }
    }

    /// Whether the chunk admit of this plan runs.
    pub(in crate::family::flash_next) fn admits(&self) -> bool {
        self.fits && self.missing > 0
    }
}

/// Maps picks against a frozen bank row, with cold picks in a disjoint logical member range.
/// Resolving does not admit: the ring owns those records until the queued reads finish.
#[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
pub(super) fn ring_slots(
    row: &[i32],
    bank_slots: usize,
    picks: &[usize],
    ring_slot: impl Fn(u16) -> Option<u16>,
) -> Result<Vec<usize>, crate::EngError> {
    picks
        .iter()
        .map(|&id| {
            let refuse = || {
                crate::EngError::BadRequest(
                    "prefill ring: a pick has no valid bank or ring slot".into(),
                )
            };
            let &slot = row.get(id).ok_or_else(refuse)?;
            if slot >= 0 {
                return usize::try_from(slot)
                    .ok()
                    .filter(|&s| s < bank_slots)
                    .ok_or_else(refuse);
            }
            let expert = u16::try_from(id).map_err(|_| refuse())?;
            let slot = ring_slot(expert).ok_or_else(refuse)?;
            bank_slots.checked_add(usize::from(slot)).ok_or_else(refuse)
        })
        .collect()
}

/// The device half of the contract: [`DeviceBank`](super::bank::device::DeviceBank) behind the
/// chunk-facing methods. Borrowing, additive: the served path keeps constructing the bank it
/// constructs today; a caller that wants the contract constructs this over it.
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
pub(in crate::family::flash_next) mod device {
    use super::super::bank::device::DeviceBank;
    use super::{AdmitPlan, drain_before_admit, ring_slots};
    use crate::family::expert_bank::{BankGeometry, BankStats, PassCopies};
    use crate::{
        EngError,
        family::{
            exl3::{
                prefill::window,
                prefill_moe::{PrefillStore, SlotResolver},
            },
            flash_next::cuda::{moe_block::SlotSpans, prefill_ring::device::PrefillRing},
            project::cuda::exl3::Exl3Members,
        },
        kernels::cuda::{CudaContext, DeviceBuffer, DeviceRange, DeviceStream},
        weights::exl3::Codebook,
    };

    /// Frozen-bank/ring views of one layer. The caller issues copies before the shared pass,
    /// and frees the layer only after all of its projections have been queued.
    pub(in crate::family::flash_next::cuda) struct RingStore<'a> {
        pub(in crate::family::flash_next::cuda) adapter: PrefillAdapter<'a>,
        pub(in crate::family::flash_next::cuda) ring: &'a PrefillRing,
        pub(in crate::family::flash_next::cuda) layer: usize,
        pub(in crate::family::flash_next::cuda) spans: &'a SlotSpans,
        pub(in crate::family::flash_next::cuda) ring_spans: SlotSpans,
        pub(in crate::family::flash_next::cuda) book: Codebook,
    }

    impl SlotResolver for RingStore<'_> {
        fn resolve_slots(
            &mut self,
            stream: &DeviceStream,
            picks: &[usize],
        ) -> Result<Vec<usize>, EngError> {
            self.adapter.record_route(picks);
            let slots = ring_slots(
                self.adapter.slot_row(self.layer)?,
                self.adapter.geometry().slots,
                picks,
                |expert| self.ring.slot_of(self.layer, expert),
            )?;
            self.ring.wait_layer(stream, self.layer)?;
            Ok(slots)
        }
    }

    impl PrefillStore for RingStore<'_> {
        fn members(&self) -> Result<[Exl3Members<'_>; 3], EngError> {
            let slots = self.adapter.slots(self.layer)?;
            Ok([
                self.spans.members_of(slots, 0, self.book)?,
                self.spans.members_of(slots, 1, self.book)?,
                self.spans.members_of(slots, 2, self.book)?,
            ])
        }

        fn ring_members(&self) -> Result<Option<[Exl3Members<'_>; 3]>, EngError> {
            let slots = self.ring.slots();
            Ok(Some([
                self.ring_spans.members_of(slots, 0, self.book)?,
                self.ring_spans.members_of(slots, 1, self.book)?,
                self.ring_spans.members_of(slots, 2, self.book)?,
            ]))
        }
    }

    /// One layer's borrowing store handle; no buffer or bank ownership is duplicated.
    pub(in crate::family::flash_next::cuda) struct LayerStore<'a> {
        pub(in crate::family::flash_next::cuda) adapter: PrefillAdapter<'a>,
        pub(in crate::family::flash_next::cuda) ctx: &'a CudaContext,
        pub(in crate::family::flash_next::cuda) layer: usize,
        pub(in crate::family::flash_next::cuda) spans: &'a SlotSpans,
        pub(in crate::family::flash_next::cuda) book: Codebook,
        pub(in crate::family::flash_next::cuda) slot_row: DeviceRange<'a>,
    }

    impl SlotResolver for LayerStore<'_> {
        fn resolve_slots(
            &mut self,
            stream: &DeviceStream,
            picks: &[usize],
        ) -> Result<Vec<usize>, EngError> {
            self.adapter.record_route(picks);
            let ids: Vec<i32> = picks
                .iter()
                .map(|&id| {
                    i32::try_from(id).map_err(|_| {
                        EngError::BadRequest("Flash-Next prefill store: a pick exceeds i32".into())
                    })
                })
                .collect::<Result<_, _>>()?;
            let plan = self.adapter.admit_plan(self.layer, &[&ids])?;
            if !plan.fits {
                return Err(EngError::BadRequest(
                    "Flash-Next prefill store: the sub-block exceeds the bank's slots".into(),
                ));
            }
            if plan.admits() {
                // Earlier sub-blocks can still read the slots this admit will overwrite.
                let copies = drain_before_admit(
                    || stream.synchronize(),
                    || self.adapter.admit_planned(self.ctx, self.layer, &plan),
                )?;
                if copies.as_ref().is_some_and(|pass| !pass.ops.is_empty()) {
                    let bytes: Vec<u8> = self
                        .adapter
                        .slot_row(self.layer)?
                        .iter()
                        .flat_map(|slot| slot.to_le_bytes())
                        .collect();
                    let destination = window(self.slot_row, 0, bytes.len())?;
                    destination
                        .buffer()
                        .write_at(stream, destination.offset(), &bytes)?;
                }
            }
            let row = self.adapter.slot_row(self.layer)?;
            picks
                .iter()
                .map(|&id| {
                    usize::try_from(row[id]).map_err(|_| {
                        EngError::BadRequest(
                            "Flash-Next prefill store: an admitted pick is absent".into(),
                        )
                    })
                })
                .collect()
        }
    }

    impl PrefillStore for LayerStore<'_> {
        fn members(&self) -> Result<[Exl3Members<'_>; 3], EngError> {
            let buffer = self.adapter.slots(self.layer)?;
            Ok([
                self.spans.members_of(buffer, 0, self.book)?,
                self.spans.members_of(buffer, 1, self.book)?,
                self.spans.members_of(buffer, 2, self.book)?,
            ])
        }
    }

    pub(in crate::family::flash_next) struct PrefillAdapter<'a> {
        bank: &'a mut DeviceBank,
    }

    impl<'a> PrefillAdapter<'a> {
        pub(in crate::family::flash_next) fn new(bank: &'a mut DeviceBank) -> Self {
            Self { bank }
        }

        /// The chunk-admit decision of `layer` for the chunk's pick rows. Faults (-1) and ids
        /// outside the layer are refused here, so a caller that cannot guarantee its router
        /// output still cannot panic the union.
        pub(in crate::family::flash_next) fn admit_plan(
            &self,
            layer: usize,
            picks_rows: &[&[i32]],
        ) -> Result<AdmitPlan, EngError> {
            let geometry = self.bank.geometry();
            for row in picks_rows {
                if let Some(&id) = row
                    .iter()
                    .find(|&&id| id < 0 || id >= geometry.experts as i32)
                {
                    return Err(EngError::BadRequest(format!(
                        "the chunk admit takes picks inside 0..{}, got {id}",
                        geometry.experts
                    )));
                }
            }
            Ok(AdmitPlan::of(
                picks_rows,
                geometry.experts,
                geometry.slots,
                self.bank.slot_row(layer)?,
            ))
        }

        /// Runs a plan `admit_plan` already computed, so a caller that drains its stream only
        /// when an admit will run does not pay the union twice. `None` when the plan does not
        /// admit. Copies on the bank's own stream, the caller drained first, re-upload
        /// [`Self::slot_row`] when ops are returned.
        pub(in crate::family::flash_next) fn admit_planned(
            &mut self,
            ctx: &CudaContext,
            layer: usize,
            plan: &AdmitPlan,
        ) -> Result<Option<PassCopies>, EngError> {
            if !plan.admits() {
                return Ok(None);
            }
            Ok(Some(self.bank.admit(ctx, layer, &plan.union)?))
        }

        /// The per-row fallback: one decode-shaped pass for the rows the union admit did not
        /// cover. Same contract as `DeviceBank::step` — copies on the bank's stream, the
        /// caller drains its own stream first.
        pub(in crate::family::flash_next) fn step(
            &mut self,
            ctx: &CudaContext,
            layer: usize,
            picks: &[i32],
        ) -> Result<PassCopies, EngError> {
            self.bank.step(ctx, layer, picks)
        }

        /// The classifier's slot row of `layer`; upload it after every pass that moved a slot.
        pub(in crate::family::flash_next) fn slot_row(
            &self,
            layer: usize,
        ) -> Result<&[i32], EngError> {
            self.bank.slot_row(layer)
        }

        /// The bank buffer of `layer`: slot `s` starts at `s * record_bytes`.
        pub(in crate::family::flash_next) fn slots(
            &self,
            layer: usize,
        ) -> Result<&DeviceBuffer, EngError> {
            self.bank.slots(layer)
        }

        /// The pinned host place of an expert's record, for the ring's stream plan (a
        /// non-resident expert without a place cannot stream).
        pub(in crate::family::flash_next) fn place_of(
            &self,
            layer: usize,
            expert: u16,
        ) -> Option<u32> {
            self.bank.place_of(layer, expert)
        }

        /// Queues one expert place's bytes from the pinned store into `dst` on the ring's copy
        /// stream. The host model is not changed: streaming is not an admit.
        pub(in crate::family::flash_next) fn stream_place(
            &mut self,
            ctx: &CudaContext,
            stream: &DeviceStream,
            layer: usize,
            place: u32,
            dst: DeviceRange<'_>,
        ) -> Result<(), EngError> {
            self.bank.stream_place(ctx, stream, layer, place, dst)
        }

        /// Waits the ring's copies still outstanding on `layer`'s pinned store and clears its
        /// pending record: the ring window hands the store back this way.
        pub(in crate::family::flash_next) fn synchronize_store(
            &mut self,
            layer: usize,
            stream: &DeviceStream,
        ) -> Result<(), EngError> {
            self.bank.synchronize_store(layer, stream)
        }

        /// The bank's own copy stream (the one `step` and `admit` run on).
        pub(in crate::family::flash_next) fn stream(&self) -> &DeviceStream {
            self.bank.stream()
        }

        /// The counters of the passes so far (hits, misses, write-backs, miss histogram).
        pub(in crate::family::flash_next) fn stats(&self) -> &BankStats {
            self.bank.stats()
        }

        /// Log-only: folds one resolve's picks into the bank's rows-per-expert route histogram.
        pub(in crate::family::flash_next) fn record_route(&mut self, picks: &[usize]) {
            self.bank.record_route(picks);
        }

        /// The sizes of the bank: layers, experts, top-k, slots per layer and the record size.
        pub(in crate::family::flash_next) fn geometry(&self) -> &BankGeometry {
            self.bank.geometry()
        }

        /// Whether a failed copy left the device out of step with the host model. While this
        /// holds, `admit_plan`, `admit_planned`, `step`, `slot_row`, `slots` and
        /// `stream_place` are refused; `place_of`, `synchronize_store`, `stream`, `stats`,
        /// `geometry` and this method still answer (the ring must be able to hand its stores
        /// back and the counters to be read out while a rebuild is decided).
        pub(in crate::family::flash_next) fn is_poisoned(&self) -> bool {
            self.bank.is_poisoned()
        }
    }
}
