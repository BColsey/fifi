//! The banked EXL3 expert load's Qwen3.6 half: the admission arithmetic that decides
//! banking and the slot count (pure, mirrored on `project::cuda`'s demand rule) and the
//! seam's table half ([`LayerTables`]). The store, the slot table and the resolver live in
//! [`crate::family::exl3::bank`]; every consumer imports them there.

use crate::{
    EngError,
    family::{
        exl3::bank::{LayerSlots, per_member_bytes},
        mem::HostMemory,
        project::cuda::Reserve,
        qwen36::settings::{BankMode, SlotRequest},
    },
};

/// The refusal helper of this module: every admission error names its numbers.
fn refuse(detail: String) -> EngError {
    EngError::MemoryPlan(format!("EXL3 expert bank: {detail}"))
}

fn gib(bytes: u64) -> String {
    format!("{:.2} GiB", bytes as f64 / (1 << 30) as f64)
}

pub(super) fn probe_host_store() -> Result<HostMemory, EngError> {
    crate::family::mem::HostFacts::read("EXL3 expert bank", "EXL3 expert bank")
        .map(|host| HostMemory::of(&host))
}

/// The resolved banked load: the slot count and the figures the start line reports.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct BankAdmission {
    pub(super) slots: usize,
    /// Dense bytes plus the slot banks' bytes: what the device holds after the load.
    pub(super) device_bytes: u64,
    /// The host store's bytes: every routed expert of every layer, in the bank layout.
    pub(super) host_bytes: u64,
}

/// Decides the load shape from the pack figures and the free memory. `Ok(None)` means stay
/// resident: the whole pack fits and the mode is `auto`. `dense` is the walk's device bytes
/// without the routed experts; `routed` is the routed experts' device bytes alone (the same
/// figure the host store holds, because a member's store bytes are its bank bytes).
///
/// The slot arithmetic mirrors `project::cuda::admit`'s demand rule: a load of `tensors`
/// bytes plus the reserve's `kv_cache + work + headroom` must fit `free`. Banking replaces
/// the routed share with `per_member * slots * layers`.
// Every figure is one named admission input; a config struct would only rename them.
#[allow(clippy::too_many_arguments)]
pub(super) fn decide(
    request: SlotRequest,
    mode: BankMode,
    dense: u64,
    routed: u64,
    layers: usize,
    n_expert: usize,
    free: u64,
    reserve: Reserve,
    host: impl FnOnce() -> Result<HostMemory, EngError>,
) -> Result<Option<BankAdmission>, EngError> {
    let hold = reserve
        .kv_cache
        .saturating_add(reserve.work)
        .saturating_add(reserve.headroom);
    let per_member = per_member_bytes(routed, layers, n_expert)?;
    let resident = dense
        .checked_add(routed)
        .ok_or_else(|| refuse("the resident figure overflows u64".into()))?;
    if mode == BankMode::Auto && resident.saturating_add(hold) <= free {
        return Ok(None);
    }
    admit_host(routed, host()?)?;
    // The load also leaves the resident flow's post-load floor free: the CUDA context and
    // workspaces eat bytes the figures above do not count, and the per-tensor admits after
    // the banks re-subtract the full hold (the embed table's admit is the last one).
    let budget = free
        .saturating_sub(hold)
        .saturating_sub(dense)
        .saturating_sub(
            u64::try_from(super::pinned::min_free_after_load()?)
                .map_err(|_| refuse("the free-after-load floor exceeds u64".into()))?,
        );
    let slots = match request {
        SlotRequest::Exactly(want) => {
            if want < 1 || want > n_expert {
                return Err(refuse(format!(
                    "{want} slots are outside 1 to {n_expert}, one banked expert to the experts \
                     of a layer",
                )));
            }
            let demand = per_member
                .checked_mul(want as u64)
                .and_then(|bytes| bytes.checked_mul(layers as u64))
                .ok_or_else(|| refuse("the slot demand overflows u64".into()))?;
            if demand > budget {
                return Err(refuse(format!(
                    "{} slots need {} of expert bytes, the budget after the dense weights and \
                     the reserve is {} (free {}, hold {}, dense {})",
                    want,
                    gib(demand),
                    gib(budget),
                    gib(free),
                    gib(hold),
                    gib(dense),
                )));
            }
            want
        }
        SlotRequest::Fit => {
            // One slot costs `per_member` per LAYER; the budget must cover every layer's bank.
            let slots = (budget / per_member.max(1) / layers as u64) as usize;
            if slots < 1 {
                return Err(refuse(format!(
                    "the budget after the dense weights and the reserve leaves {slots} slots, \
                     below one (budget {}, per member {})",
                    gib(budget),
                    gib(per_member),
                )));
            }
            slots.min(n_expert)
        }
    };
    let device_bytes = dense.saturating_add(
        per_member
            .saturating_mul(slots as u64)
            .saturating_mul(layers as u64),
    );
    Ok(Some(BankAdmission {
        slots,
        device_bytes,
        host_bytes: routed,
    }))
}

/// The pageable routed-expert store may spend what a payload may spend (`HostMemory::payload_spare`):
/// the budget above the one OS floor, and inside a binding cgroup the remainder less the
/// process's own overhead. The floor never keeps back more than the fifth of RAM this store kept
/// before there was one floor, so no host plans stricter than it did.
fn admit_host(routed: u64, host: HostMemory) -> Result<(), EngError> {
    let budget = host.payload_spare();
    if routed > budget {
        return Err(refuse(format!(
            "the expert store needs {} of host RAM, the budget is {} (available {}, kept back {})",
            gib(routed),
            gib(budget),
            gib(host.available),
            gib(host.available.saturating_sub(budget)),
        )));
    }
    Ok(())
}

/// The Q36 tables behind the shared seam's table half
/// ([`SlotTable`](crate::family::expert_bank::SlotTable)): one `LayerSlots` per layer, its
/// live `slot_of` row the tiered classifier reads. Only the table surface is shared — the
/// pass protocol stays Q36's own: `assign` and the refill loop are not routed through
/// `SlotPolicy`.
pub(super) struct LayerTables(pub(crate) Vec<LayerSlots>);

impl crate::family::expert_bank::SlotTable for LayerTables {
    fn slot_row(&self, layer: usize) -> &[i32] {
        self.0.get(layer).map_or(&[], |tables| tables.slot_row())
    }

    fn expert_in_slot(&self, layer: usize, slot: usize) -> Option<u16> {
        self.0.get(layer)?.expert_in_slot(slot)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Dense ≈ 6 GiB, routed ≈ 10.1 GiB: the q36 EXL3 pack's shape, with the routed figure
    /// built from an exactly-divisible per-member byte count (the guard rejects a figure
    /// that does not divide over the layers and experts).
    fn figures() -> (u64, u64) {
        let per_member = 900_000u64; // ≈ 0.86 MiB per expert per layer (the three parts)
        (6 << 30, 47 * 256 * per_member)
    }

    /// A 16 GiB host with 14 GiB available: the store fits above the 3.2 GiB OS floor, as it did
    /// above the 3.2 GiB margin before there was one floor.
    fn host() -> Result<HostMemory, EngError> {
        Ok(HostMemory::from_pair(16 << 30, 14 << 30))
    }

    fn reserve(hold: u64) -> Reserve {
        Reserve {
            kv_cache: hold,
            work: 0,
            headroom: 0,
        }
    }

    fn decide_auto(
        dense_routed: (u64, u64),
        free: u64,
        request: SlotRequest,
        host: impl FnOnce() -> Result<HostMemory, EngError>,
    ) -> Result<Option<BankAdmission>, EngError> {
        let (dense, routed) = dense_routed;
        decide(
            request,
            BankMode::Auto,
            dense,
            routed,
            47,
            256,
            free,
            reserve(1 << 30),
            host,
        )
    }

    #[test]
    fn a_pack_that_fits_stays_resident_in_auto_mode() {
        let (dense, routed) = figures();
        let free = dense + routed + (1 << 30) + (1 << 20);
        assert_eq!(
            decide_auto((dense, routed), free, SlotRequest::Fit, || panic!(
                "resident load queried host store budget"
            ))
            .unwrap(),
            None
        );
    }

    #[test]
    fn a_pack_that_does_not_fit_banks_the_fitting_slot_count() {
        let (dense, routed) = figures();
        // Free for 1 GiB of dense + hold + the post-load floor, plus 64 slots per layer.
        let per_member = routed / 47 / 256;
        let free = dense + (1 << 30) + (1 << 30) + per_member * 64 * 47;
        let got = decide_auto((dense, routed), free, SlotRequest::Fit, host)
            .unwrap()
            .expect("banks");
        assert_eq!(got.slots, 64);
        assert_eq!(got.host_bytes, routed);
        assert_eq!(got.device_bytes, dense + per_member * 64 * 47);
    }

    #[test]
    fn on_mode_banks_even_when_the_pack_fits() {
        let (dense, routed) = figures();
        let free = dense + routed + (1 << 30);
        let got = decide(
            SlotRequest::Exactly(32),
            BankMode::On,
            dense,
            routed,
            47,
            256,
            free,
            reserve(1 << 30),
            host,
        )
        .unwrap()
        .expect("banks");
        assert_eq!(got.slots, 32);
        assert_eq!(
            got.device_bytes,
            dense + (routed / 47 / 256) * 32 * 47,
            "linear in the slot count"
        );
    }

    #[test]
    fn an_exact_count_over_budget_is_refused_with_numbers() {
        let (dense, routed) = figures();
        let free = dense + (1 << 30) + (routed / 47 / 256) * 16;
        let error = format!(
            "{:?}",
            decide_auto((dense, routed), free, SlotRequest::Exactly(200), host).unwrap_err()
        );
        assert!(error.contains("200 slots"), "{error}");
        assert!(error.contains("budget"), "{error}");
    }

    #[test]
    #[ignore = "requires a Linux process with a 512 MiB memory cgroup"]
    fn process_memory_limit_512_mib() {
        let memory = probe_host_store().unwrap();
        assert!((1..=512 << 20).contains(&memory.available), "{memory:?}");
        let (dense, routed) = figures();
        let resident_free = dense + routed + (1 << 30);
        assert!(
            decide_auto(
                (dense, routed),
                resident_free,
                SlotRequest::Fit,
                probe_host_store
            )
            .unwrap()
            .is_none()
        );
        let err = decide_auto(
            (dense, routed),
            resident_free - 1,
            SlotRequest::Fit,
            probe_host_store,
        )
        .unwrap_err();
        assert!(err.describe().contains("expert store needs"), "{err:?}");
        eprintln!("Qwen EXL3 {memory:?}: resident admitted, pageable expert store refused");
    }

    /// The store this module admitted before there was one floor: the host budget less the larger
    /// of 2 GiB and a fifth of RAM. Kept here only as the bound the one floor must stay inside.
    fn admitted_before(routed: u64, total: u64, available: u64) -> bool {
        routed <= available.saturating_sub((2u64 << 30).max(total / 5))
    }

    /// No host plans stricter than before: on every (total, available) pair the old rule served,
    /// the store is admitted, including the 16 GiB host of the original fixture.
    #[test]
    fn no_host_that_served_the_store_before_refuses_it_now() {
        let (_, routed) = figures();
        assert!(admit_host(routed, HostMemory::from_pair(16 << 30, 14 << 30)).is_ok());
        for total_gib in 8..=128u64 {
            for available_gib_tenths in 0..=(total_gib * 10) {
                let total = total_gib << 30;
                let available = available_gib_tenths * (1 << 30) / 10;
                if admitted_before(routed, total, available) {
                    assert!(
                        admit_host(routed, HostMemory::from_pair(total, available)).is_ok(),
                        "{total_gib} GiB host with {available} available"
                    );
                }
            }
        }
    }

    /// Inside a binding cgroup the store keeps the process's overhead back, like every payload.
    #[test]
    fn a_cgroup_capped_host_keeps_the_process_overhead_back() {
        let (_, routed) = figures();
        let meminfo = "MemTotal: 65011712 kB\nMemAvailable: 52428800 kB\n";
        let cap = |remaining: u64| {
            let host =
                crate::family::mem::HostFacts::from_meminfo("test", meminfo, Some(remaining))
                    .expect("host");
            HostMemory::of(&host)
        };
        // 12 GiB left: 2 GiB of overhead leave 10 GiB, just under the 10.07 GiB store.
        assert!(admit_host(routed, cap(12 << 30)).is_err());
        assert!(admit_host(routed, cap((12 << 30) + (1 << 29))).is_ok());
    }

    #[test]
    fn a_host_without_room_is_refused_before_any_slot() {
        let (dense, routed) = figures();
        // On mode: banking is requested even though the pack would fit, and the host refuses.
        // Available 8 GiB less the 3.2 GiB OS floor leaves 4.8 GiB for ~10.1 GiB of experts.
        let error = format!(
            "{:?}",
            decide(
                SlotRequest::Fit,
                BankMode::On,
                dense,
                routed,
                47,
                256,
                1 << 40,
                reserve(1 << 30),
                || Ok(HostMemory::from_pair(16 << 30, 8 << 30)),
            )
            .unwrap_err()
        );
        assert!(error.contains("host RAM"), "{error}");
        // And a host without figures at all.
        let error = format!(
            "{:?}",
            decide(
                SlotRequest::Fit,
                BankMode::On,
                dense,
                routed,
                47,
                256,
                1 << 40,
                reserve(1 << 30),
                || Err(refuse("host figures unavailable".into())),
            )
            .unwrap_err()
        );
        assert!(error.contains("unavailable"), "{error}");
    }

    #[test]
    fn fit_never_returns_zero_or_above_the_expert_count() {
        let (dense, routed) = figures();
        // Barely past the dense figure + hold + the post-load floor: one slot fits.
        let per_member = routed / 47 / 256;
        let free = dense + (1 << 30) + (1 << 30) + per_member * 47;
        let got = decide_auto((dense, routed), free, SlotRequest::Fit, host)
            .unwrap()
            .expect("banks");
        assert_eq!(got.slots, 1);
        // And a huge free figure clamps at the full expert count (On mode: Auto would stay
        // resident with room to spare).
        let got = decide(
            SlotRequest::Fit,
            BankMode::On,
            dense,
            routed,
            47,
            256,
            1 << 44,
            reserve(1 << 30),
            host,
        )
        .unwrap()
        .expect("banks");
        assert_eq!(got.slots, 256);
    }
}

#[cfg(test)]
mod exl3_bank_pick_golden_host_tests;
#[cfg(test)]
mod policy_differential_host_tests;
