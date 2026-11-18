//! The admission arithmetic of the tiered offload.
//!
//! WHAT: two pure planners that decide, in bytes, how the machine is split at
//! load. [`plan_vram`] gives the KV cache of the configured context VRAM first
//! and the per-layer expert bank the rest, spilling KV positions to the host
//! only when they do not fit or when the minimum bank needs the room;
//! [`plan_host`] gives the host RAM to the OS page cache of the experts minus a
//! host KV overflow taken from that cache one-for-one. WHY (REV4 c.4, north
//! star = coding agents with long context): a 6 GiB card cannot hold the
//! model's experts, so the split must be settled before the first token, and a
//! context that cannot fit is refused with a typed error that names the budget
//! instead of dying mid-decode. Every number is an input measured by the
//! caller; no model constant lives here. All sizes are bytes (`u64`), every
//! operation is checked (an absurd input is refused as [`BudgetError`] — never
//! wrapped, never a panic), and the split is deterministic: no I/O, no clocks,
//! no randomness, no threads. The tiered load uses its VRAM plan.

/// What the card has and what is already spoken for (the caller measures these
/// at load; none of them is guessed here). All fields are bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VramBudget {
    pub usable: u64,
    pub non_expert: u64,
    /// Engine state resident in VRAM (samplers, logits, graphs).
    pub state: u64,
    /// The kernel scratch arena.
    pub arena: u64,
    pub desktop_reserve: u64,
}

use crate::family::device_budget::KvPlan;

/// The expert geometry: one expert size shared by every MoE layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExpertSize {
    pub bytes: u64,
    pub layers: usize,
}

/// Why a plan was refused. Every variant names the budget that lost, with all
/// numbers in bytes, so the caller can report the shortfall or retry with a
/// smaller context.
#[derive(Debug, PartialEq, Eq)]
pub enum BudgetError {
    Overcommitted {
        what: &'static str,
        needed: u64,
        available: u64,
    },
    /// The KV cache did not fit: it wants `kv_bytes`, VRAM grants
    /// `vram_for_kv` of them and the host is allowed `host_allowed`, so the
    /// remainder `kv_bytes - vram_for_kv` exceeds what the host may take.
    KvDoesNotFit {
        /// Total KV bytes the configured context wants.
        kv_bytes: u64,
        /// KV bytes VRAM grants after the minimum bank took its room.
        vram_for_kv: u64,
        host_allowed: u64,
    },
    /// A structurally invalid input (a zero size or dimension, or a sum that
    /// overflows `u64`); nothing was computed.
    BadInput(String),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VramPlan {
    pub kv_vram_positions: u64,
    /// KV positions spilled to host RAM. Invariant:
    /// `kv_vram_positions + kv_host_positions == KvPlan::positions`.
    pub kv_host_positions: u64,
    /// Bank slots per layer, uniform over the layers; at least the requested
    /// minimum whenever the plan is `Ok`.
    pub bank_slots_per_layer: usize,
    pub bank_bytes: u64,
    /// VRAM left after KV and the bank: the floor remainder, always less than
    /// one bank slot (`layers * expert bytes`).
    pub vram_left: u64,
}

/// WHY: REV4 c.4 orders the tiers exactly this way — KV is 8-bit on every
/// backend and takes VRAM first, the expert bank gets the remainder, and KV
/// spills to host RAM only when it does not fit; a decode that started without
/// its minimum bank would fall through to the CPU from the first token.
pub fn plan_vram(
    v: &VramBudget,
    kv: &KvPlan,
    e: &ExpertSize,
    min_bank_slots_per_layer: usize,
    host_kv_allowed: u64,
) -> Result<VramPlan, BudgetError> {
    if kv.bytes_per_position == 0 {
        return Err(bad_input("budget: kv bytes_per_position must be nonzero"));
    }
    if e.bytes == 0 {
        return Err(bad_input("budget: expert bytes must be nonzero"));
    }
    if e.layers == 0 {
        return Err(bad_input("budget: expert layers must be nonzero"));
    }
    let layers =
        u64::try_from(e.layers).map_err(|_| bad_input("budget: expert layers does not fit u64"))?;
    let slot_cost = layers
        .checked_mul(e.bytes)
        .ok_or_else(|| bad_input("budget: expert bytes * layers overflows u64"))?;
    let min_bank_bytes = u64::try_from(min_bank_slots_per_layer)
        .map_err(|_| bad_input("budget: min_bank_slots_per_layer does not fit u64"))?
        .checked_mul(slot_cost)
        .ok_or_else(|| bad_input("budget: minimum bank bytes overflow u64"))?;
    let commitments = v
        .non_expert
        .checked_add(v.state)
        .and_then(|sum| sum.checked_add(v.arena))
        .and_then(|sum| sum.checked_add(v.desktop_reserve))
        .ok_or_else(|| bad_input("budget: vram commitments overflow u64"))?;
    let free_vram = v
        .usable
        .checked_sub(commitments)
        .ok_or(BudgetError::Overcommitted {
            what: "vram (non-expert weights + state + arena + desktop reserve)",
            needed: commitments,
            available: v.usable,
        })?;
    if min_bank_bytes > free_vram {
        return Err(BudgetError::Overcommitted {
            what: "minimum expert bank",
            needed: min_bank_bytes,
            available: free_vram,
        });
    }

    // KV takes VRAM first but never the bytes the minimum bank owns; that is
    // the only lever that pushes KV positions to the host (REV4 c.4).
    let kv_bytes = kv
        .positions
        .checked_mul(kv.bytes_per_position)
        .ok_or_else(|| bad_input("budget: kv positions * bytes_per_position overflows u64"))?;
    let vram_for_kv = kv_bytes.min(free_vram - min_bank_bytes);
    let kv_vram_positions = vram_for_kv / kv.bytes_per_position;
    let kv_vram_bytes = kv_vram_positions
        .checked_mul(kv.bytes_per_position)
        .ok_or_else(|| bad_input("budget: vram kv bytes overflow u64"))?;
    let kv_host_positions = kv.positions - kv_vram_positions;
    let host_kv_bytes = kv_host_positions
        .checked_mul(kv.bytes_per_position)
        .ok_or_else(|| bad_input("budget: host kv bytes overflow u64"))?;
    if host_kv_bytes > host_kv_allowed {
        return Err(BudgetError::KvDoesNotFit {
            kv_bytes,
            vram_for_kv: kv_vram_bytes,
            host_allowed: host_kv_allowed,
        });
    }

    // The bank gets what is left, floored to whole slots, uniform over the
    // layers; `bank_free >= min_bank_bytes` holds by construction above.
    let bank_free = free_vram - kv_vram_bytes;
    let bank_slots = bank_free / slot_cost;
    let bank_bytes = bank_slots
        .checked_mul(slot_cost)
        .ok_or_else(|| bad_input("budget: bank bytes overflow u64"))?;
    let bank_slots_per_layer = usize::try_from(bank_slots)
        .map_err(|_| bad_input("budget: bank slots does not fit usize"))?;
    Ok(VramPlan {
        kv_vram_positions,
        kv_host_positions,
        bank_slots_per_layer,
        bank_bytes,
        vram_left: bank_free - bank_bytes,
    })
}

/// What the host has and what is already spoken for before the page cache (the
/// caller measures these at load; none of them is guessed here). All fields are
/// bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HostBudget {
    pub total: u64,
    /// RAM held by the OS, the loader and other processes; never returned.
    pub non_reclaimable: u64,
    /// The engine's own heap: activations, CPU jobs, bookkeeping.
    pub engine_heap: u64,
    /// Staging buffers for expert transfers to the GPU.
    pub staging: u64,
    /// Safety margin kept free so the plan does not ride the OOM line.
    pub margin: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HostPlan {
    /// Bytes left for the OS page cache of the experts.
    pub page_cache: u64,
    pub cached_experts: u64,
    /// The host KV bytes echoed back; each was taken from `page_cache`
    /// one-for-one (REV4 c.4).
    pub host_kv: u64,
}

/// WHY: the OS page cache is what makes the CPU tier fast, so it owns the
/// residual host RAM — but every host KV byte is taken from it one-for-one
/// (REV4 c.4), which this plan makes explicit instead of accidental.
pub fn plan_host(h: &HostBudget, host_kv: u64, expert_bytes: u64) -> Result<HostPlan, BudgetError> {
    if expert_bytes == 0 {
        return Err(bad_input("budget: expert bytes must be nonzero"));
    }
    let spent = h
        .non_reclaimable
        .checked_add(h.engine_heap)
        .and_then(|sum| sum.checked_add(h.staging))
        .and_then(|sum| sum.checked_add(h.margin))
        .and_then(|sum| sum.checked_add(host_kv))
        .ok_or_else(|| bad_input("budget: host commitments overflow u64"))?;
    let page_cache = h
        .total
        .checked_sub(spent)
        .ok_or(BudgetError::Overcommitted {
            what: "host RAM (non-reclaimable + engine heap + staging + margin + host KV)",
            needed: spent,
            available: h.total,
        })?;
    Ok(HostPlan {
        page_cache,
        cached_experts: page_cache / expert_bytes,
        host_kv,
    })
}

// ===== Store tiers =====

/// The tiers an expert store can be admitted to, in admission order. The
/// inclusive store pins every expert place, so a miss copies one way only;
/// the ring tier pins only some spare places and streams each miss through a
/// pinned ring. A host that cannot take either refuses — there is no third
/// downshift, so refusal is never silent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StoreTier {
    /// Every expert place stays pinned in host RAM.
    Inclusive,
    /// `places_per_layer` spare places stay pinned (zero is valid: every
    /// miss streams from disk, nothing is resident); each miss reads its
    /// expert through `ring_bytes` of pinned ring slots charged to the
    /// process budget, so the page cache holds no second copy of expert
    /// bytes.
    ExclusiveRing {
        places_per_layer: usize,
        ring_bytes: u64,
    },
}

/// What a store-tier admission knows. Every figure comes from the caller
/// (geometry, one host admission snapshot, the ring the exclusive tier
/// wants); none is guessed here. All fields are bytes unless named otherwise.
pub struct StoreTierBudget {
    pub layers: u64,
    pub record_bytes: u64,
    /// Places per layer the inclusive tier pins: one per expert.
    pub inclusive_places_per_layer: u64,
    /// The UPPER BOUND on resident places per layer (the bank's spare
    /// count). The planner itself picks the largest count that fits —
    /// zero is a valid outcome (all misses from disk) — so this field is
    /// a ceiling, never the answer.
    pub max_places_per_layer: u64,
    /// Pinned ring bytes the exclusive tier charges itself (slot count times
    /// slot bytes, sector-aligned; the caller computes the product).
    pub ring_bytes: u64,
    /// What the process may take now (the `MemAvailable`-style snapshot).
    pub available: u64,
    /// Every fixed term besides the store and ring: stream scratch, n-gram
    /// staging, engine heap, margin.
    pub fixed_terms: u64,
    /// The OS floor the plan must leave standing.
    pub os_floor: u64,
    /// The page-cache floor both tiers honour (the n-gram cache minimum).
    pub page_cache_min: u64,
}

/// The chosen tier and its exact byte accounting.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StoreTierPlan {
    pub tier: StoreTier,
    /// Pinned store bytes of the chosen tier.
    pub store_bytes: u64,
    /// Store, plus the ring on the exclusive tier, plus the fixed terms:
    /// what this process takes.
    pub process_bytes: u64,
    /// `available` minus `process_bytes` and the OS floor: the page cache
    /// left for everything else.
    pub page_cache_left: u64,
}

/// WHY the order is fixed: the inclusive store removes a write-back from
/// every miss, so it goes first when the host can pin every place; the
/// moment it cannot, the ring is required (some expert then lives only on
/// disk), so the ring tier is planned WITH its ring bytes, never without —
/// and at the largest resident count that fits, searched downward to zero,
/// because a smaller resident set is always better than a refusal. A host
/// that cannot even carry the ring with zero resident places cannot serve
/// the model at all, which is the only refusal left.
pub fn plan_store_tiers(b: &StoreTierBudget) -> Result<StoreTierPlan, BudgetError> {
    if b.layers == 0 {
        return Err(bad_input("store tiers: layers must be nonzero"));
    }
    if b.record_bytes == 0 {
        return Err(bad_input("store tiers: record bytes must be nonzero"));
    }
    if b.inclusive_places_per_layer == 0 {
        return Err(bad_input("store tiers: the inclusive pin must be nonzero"));
    }
    let store_bytes = |places: u64| {
        b.layers
            .checked_mul(places)
            .and_then(|p| p.checked_mul(b.record_bytes))
            .ok_or_else(|| bad_input("store tiers: store bytes overflow u64"))
    };
    let inclusive_store = store_bytes(b.inclusive_places_per_layer)?;
    let charge = |store: u64, ring: u64| -> Option<u64> {
        store
            .checked_add(ring)
            .and_then(|sum| sum.checked_add(b.fixed_terms))
            .and_then(|sum| sum.checked_add(b.os_floor))
    };
    // Both tiers must leave the page-cache floor standing, not merely fit.
    let fits = |needed: u64| needed <= b.available && b.available - needed >= b.page_cache_min;
    if let Some(plan) = charge(inclusive_store, 0).filter(|&needed| fits(needed)).map(|needed| {
        StoreTierPlan {
            tier: StoreTier::Inclusive,
            store_bytes: inclusive_store,
            process_bytes: needed - b.os_floor,
            page_cache_left: b.available - needed,
        }
    }) {
        return Ok(plan);
    }
    // The ring tier: the largest resident count that fits, downward to
    // zero (all misses from disk). The zero-place charge — ring + terms +
    // floor — is the last resort the refusal names when nothing fits.
    for places in (0..=b.max_places_per_layer).rev() {
        let store = store_bytes(places)?;
        let Some(needed) = charge(store, b.ring_bytes) else {
            return Err(bad_input("store tiers: ring-tier charge overflow u64"));
        };
        if needed <= b.available && b.available - needed >= b.page_cache_min {
            return Ok(StoreTierPlan {
                tier: StoreTier::ExclusiveRing {
                    places_per_layer: usize::try_from(places).map_err(|_| {
                        bad_input("store tiers: places per layer does not fit usize")
                    })?,
                    ring_bytes: b.ring_bytes,
                },
                store_bytes: store,
                process_bytes: needed - b.os_floor,
                page_cache_left: b.available - needed,
            });
        }
    }
    let ring_only = charge(0, b.ring_bytes)
        .ok_or_else(|| bad_input("store tiers: ring-only charge overflow u64"))?;
    Err(BudgetError::Overcommitted {
        what: "expert ring tier (the last tier: no resident count fits, not even zero places with the ring alone)",
        needed: ring_only,
        available: b.available,
    })
}

/// The steady-state decode rate a tier can hold, from its parts: the
/// resident figure (measured or the control number) and the direct-read
/// bandwidth the misses consume. Per-token time is the resident step plus
/// the miss bytes over the measured rate, so the exclusive tier's cost is
/// additive, never hidden.
///
/// `direct_bytes_per_s` is the O_DIRECT sequential rate measured once at
/// serve start (the same ring the misses use), logged with the line below.
/// The one serve log line that states the tier choice and its price:
/// the tier with its resident count and ring bytes, the exact-bytes figures
/// of the plan, the measured direct rate it was priced against, and the
/// expected tok/s beside the resident control figure.
fn bad_input(what: &str) -> BudgetError {
    BudgetError::BadInput(String::from(what))
}

#[cfg(test)]
mod tests {
    //! The 35B-A3B-shape numbers live here and only here; production code
    //! takes every number as an input. GiB = 2^30 bytes throughout.

    use super::{
        BudgetError, ExpertSize, HostBudget, KvPlan, VramBudget, bad_input, plan_host, plan_vram,
    };
    /// One GiB (2^30 bytes); 8-bit KV bytes of one position (10 attention
    /// layers x 1,088 bytes); one quantized expert; the MoE layer count; and the
    /// VRAM one slot per layer costs (`EXPERT_BYTES * LAYERS`).
    const GIB: u64 = 1 << 30;
    const BYTES_PER_POSITION: u64 = 10_880;
    const EXPERT_BYTES: u64 = 1_769_472;
    const LAYERS: usize = 40;
    const SLOT_COST: u64 = EXPERT_BYTES * 40;

    /// The 6 GiB card fixture: measured weights, state, arena and a reserve.
    fn vram(desktop_reserve: u64) -> VramBudget {
        VramBudget {
            usable: 6 * GIB,
            non_expert: 2_316_995_072,
            state: 62_914_560 + 2_949_120,
            arena: 107_374_182,
            desktop_reserve,
        }
    }

    fn vbudget(usable: u64, non_expert: u64) -> VramBudget {
        VramBudget {
            usable,
            non_expert,
            state: 0,
            arena: 0,
            desktop_reserve: 0,
        }
    }

    fn kv(positions: u64) -> KvPlan {
        kvz(positions, BYTES_PER_POSITION)
    }

    fn kvz(positions: u64, bpp: u64) -> KvPlan {
        KvPlan {
            positions,
            bytes_per_position: bpp,
        }
    }

    fn geom(bytes: u64, layers: usize) -> ExpertSize {
        ExpertSize { bytes, layers }
    }

    fn experts() -> ExpertSize {
        geom(EXPERT_BYTES, LAYERS)
    }

    /// The host RAM fixture: 14.76 GiB total with the fixed commitments.
    fn host() -> HostBudget {
        HostBudget {
            total: 15_847_931_904,
            non_reclaimable: 2 * GIB,
            engine_heap: GIB,
            staging: 214_748_365,
            margin: GIB / 2,
        }
    }

    #[test]
    fn the_32k_8bit_kv_leaves_about_39_slots_per_layer() {
        let plan = plan_vram(&vram(3 * GIB / 4), &kv(32_768), &experts(), 0, 0).unwrap();
        assert_eq!(
            (plan.kv_vram_positions, plan.kv_host_positions),
            (32_768, 0)
        );
        assert!((38..=40).contains(&plan.bank_slots_per_layer), "{plan:?}");
    }

    #[test]
    fn the_128k_leaves_about_24() {
        let plan = plan_vram(&vram(3 * GIB / 4), &kv(131_072), &experts(), 0, 0).unwrap();
        assert_eq!(
            (plan.kv_vram_positions, plan.kv_host_positions),
            (131_072, 0)
        );
        assert!((23..=25).contains(&plan.bank_slots_per_layer), "{plan:?}");
    }

    #[test]
    fn the_262k_in_vram_leaves_a_handful() {
        assert_eq!(262_144 * BYTES_PER_POSITION, 2_852_126_720);
        let plan = plan_vram(&vram(3 * GIB / 4), &kv(262_144), &experts(), 0, 0).unwrap();
        assert_eq!(
            (plan.kv_vram_positions, plan.kv_host_positions),
            (262_144, 0)
        );
        assert!((1..=6).contains(&plan.bank_slots_per_layer), "{plan:?}");
    }

    #[test]
    fn a_desktop_reserve_of_one_gib_at_262k_spills_kv_to_the_host_when_a_minimum_bank_is_required()
    {
        let plan = plan_vram(&vram(GIB), &kv(262_144), &experts(), 1, GIB).unwrap();
        assert_eq!(plan.kv_vram_positions + plan.kv_host_positions, 262_144);
        assert!(plan.kv_host_positions > 0, "{plan:?}");
        assert!(plan.bank_slots_per_layer >= 1, "{plan:?}");
        match plan_vram(&vram(GIB), &kv(262_144), &experts(), 1, 0) {
            Err(BudgetError::KvDoesNotFit {
                kv_bytes,
                vram_for_kv,
                host_allowed,
            }) => {
                assert_eq!((kv_bytes, host_allowed), (2_852_126_720, 0));
                assert!(vram_for_kv < kv_bytes);
            }
            other => panic!("expected KvDoesNotFit, got {other:?}"),
        }
    }

    #[test]
    fn kv_positions_split_exactly() {
        let geometry = experts();
        let mut fitted = 0;
        let mut spilled = 0;
        for reserve in [0, GIB / 2, GIB, 3 * GIB / 2] {
            let budget = vram(reserve);
            let fixed = budget.non_expert + budget.state + budget.arena + budget.desktop_reserve;
            for step in 0..64u64 {
                for min_slots in [0, 1, 4] {
                    for host_allowed in [0, 4 * GIB] {
                        let positions = step * 4_099;
                        let Ok(plan) =
                            plan_vram(&budget, &kv(positions), &geometry, min_slots, host_allowed)
                        else {
                            continue;
                        };
                        assert_eq!(plan.kv_vram_positions + plan.kv_host_positions, positions);
                        assert_eq!(
                            plan.bank_bytes,
                            u64::try_from(plan.bank_slots_per_layer).unwrap() * SLOT_COST
                        );
                        assert!(plan.bank_slots_per_layer >= min_slots);
                        assert!(plan.vram_left < SLOT_COST, "{plan:?}");
                        assert_eq!(
                            budget.usable,
                            fixed
                                + plan.kv_vram_positions * BYTES_PER_POSITION
                                + plan.bank_bytes
                                + plan.vram_left
                        );
                        assert!(plan.kv_host_positions * BYTES_PER_POSITION <= host_allowed);
                        if plan.kv_host_positions > 0 {
                            spilled += 1;
                        } else {
                            fitted += 1;
                        }
                    }
                }
            }
        }
        assert!(fitted > 0 && spilled > 0, "the sweep must cover both sides");
    }

    #[test]
    fn overflowing_inputs_are_refused_not_wrapped() {
        let max = u64::MAX;
        let e = experts();
        // A commitment sum that overflows u64 is BadInput, not a wrapped fit.
        let over = VramBudget {
            usable: max,
            non_expert: max / 2 + 1,
            state: max / 2 + 1,
            arena: 1,
            desktop_reserve: 1,
        };
        assert!(matches!(
            plan_vram(&over, &kv(1), &e, 0, 0),
            Err(BudgetError::BadInput(_))
        ));
        match plan_vram(&vbudget(GIB, 2 * GIB), &kv(1), &e, 0, 0) {
            Err(BudgetError::Overcommitted {
                needed, available, ..
            }) => {
                assert_eq!((needed, available), (2 * GIB, GIB));
            }
            other => panic!("expected Overcommitted, got {other:?}"),
        }
        let zero_card = vbudget(max, 0);
        assert!(matches!(
            plan_vram(&zero_card, &kvz(max, 2), &e, 0, 0),
            Err(BudgetError::BadInput(_))
        ));
        assert_eq!(
            plan_vram(&zero_card, &kvz(4, 0), &e, 0, 0),
            Err(bad_input("budget: kv bytes_per_position must be nonzero"))
        );
        assert_eq!(
            plan_vram(
                &zero_card,
                &kvz(4, BYTES_PER_POSITION),
                &geom(0, LAYERS),
                0,
                0
            ),
            Err(bad_input("budget: expert bytes must be nonzero"))
        );
        assert_eq!(
            plan_vram(&zero_card, &kvz(4, BYTES_PER_POSITION), &geom(8, 0), 0, 0),
            Err(bad_input("budget: expert layers must be nonzero"))
        );
        match plan_vram(&vbudget(2 * GIB, 0), &kv(0), &e, 41, max) {
            Err(BudgetError::Overcommitted {
                needed, available, ..
            }) => {
                assert_eq!((needed, available), (41 * SLOT_COST, 2 * GIB));
            }
            other => panic!("expected Overcommitted, got {other:?}"),
        }
        let host_over = HostBudget {
            total: max,
            non_reclaimable: max / 2 + 1,
            engine_heap: max / 2 + 1,
            staging: 0,
            margin: 0,
        };
        assert!(matches!(
            plan_host(&host_over, 0, EXPERT_BYTES),
            Err(BudgetError::BadInput(_))
        ));
        assert_eq!(
            plan_host(&host(), 0, 0),
            Err(bad_input("budget: expert bytes must be nonzero"))
        );
    }

    #[test]
    fn host_cache_loses_a_gib_per_host_kv_gib() {
        let base = plan_host(&host(), 0, EXPERT_BYTES).unwrap();
        // About 11.06 GiB of page cache before any host KV.
        assert!(
            (11 * GIB..11 * GIB + GIB / 8).contains(&base.page_cache),
            "{}",
            base.page_cache
        );
        assert_eq!(base.host_kv, 0);
        for extra in 0..=4u64 {
            let plan = plan_host(&host(), extra * GIB, EXPERT_BYTES).unwrap();
            assert_eq!(plan.host_kv, extra * GIB);
            assert_eq!(plan.page_cache, base.page_cache - extra * GIB);
        }
        let lost = base.cached_experts
            - plan_host(&host(), GIB, EXPERT_BYTES)
                .unwrap()
                .cached_experts;
        assert!((606..=607).contains(&lost), "{lost}"); // about 607 experts per GiB
    }

    #[test]
    fn host_plan_refuses_a_kv_larger_than_the_cache() {
        let budget = host();
        let cache = plan_host(&budget, 0, EXPERT_BYTES).unwrap().page_cache;
        match plan_host(&budget, cache + 1, EXPERT_BYTES) {
            Err(BudgetError::Overcommitted {
                what,
                needed,
                available,
            }) => {
                assert!(what.contains("host"), "{what}");
                assert_eq!(available, budget.total);
                assert!(needed > available);
            }
            other => panic!("expected Overcommitted, got {other:?}"),
        }
        let edge = plan_host(&budget, cache, EXPERT_BYTES).unwrap();
        assert_eq!(
            (edge.page_cache, edge.cached_experts, edge.host_kv),
            (0, 0, cache)
        );
    }
}

#[cfg(test)]
mod store_tier_tests {
    //! Exact-bytes boundaries for `plan_store_tiers`. Fixture: 3 layers, one
    //! 1000-byte record, 10 experts, a 4-place ceiling, a 2048-byte ring,
    //! 1000 bytes of fixed terms and a 500-byte OS floor. Every expected
    //! number below is computed by hand:
    //!
    //!   inclusive charge      = 3*10*1000          + 1000 + 500 = 31_500
    //!   ring-tier charge(p)   = 3*p*1000 + 2048    + 1000 + 500 = 3000*p + 3_548
    //!     p=4 -> 15_548   p=3 -> 12_548   p=2 -> 9_548   p=1 -> 6_548
    //!     p=0 (ring only) -> 3_548

    use super::{BudgetError, StoreTier, StoreTierBudget, plan_store_tiers};

    const INCLUSIVE_NEEDED: u64 = 31_500;
    const RING_TIER_P4: u64 = 15_548;
    const RING_TIER_P3: u64 = 12_548;
    const RING_ONLY: u64 = 3_548;

    fn budget(available: u64) -> StoreTierBudget {
        StoreTierBudget {
            layers: 3,
            record_bytes: 1_000,
            inclusive_places_per_layer: 10,
            max_places_per_layer: 4,
            ring_bytes: 2_048,
            available,
            fixed_terms: 1_000,
            os_floor: 500,
            page_cache_min: 0,
        }
    }

    #[test]
    fn the_exact_inclusive_boundary_is_inclusive() {
        let plan = plan_store_tiers(&budget(INCLUSIVE_NEEDED)).unwrap();
        assert_eq!(plan.tier, StoreTier::Inclusive);
        assert_eq!(plan.store_bytes, 30_000);
        assert_eq!(plan.process_bytes, 30_000 + 1_000);
        assert_eq!(plan.page_cache_left, 0);
    }

    #[test]
    fn one_byte_short_of_inclusive_picks_the_largest_ring_count() {
        // 31_499 < 31_500, so inclusive is out; the downward search starts
        // at 4 places (15_548 <= 31_499) and stops there.
        let plan = plan_store_tiers(&budget(INCLUSIVE_NEEDED - 1)).unwrap();
        assert_eq!(
            plan.tier,
            StoreTier::ExclusiveRing {
                places_per_layer: 4,
                ring_bytes: 2_048,
            }
        );
        assert_eq!(plan.store_bytes, 12_000);
        assert_eq!(plan.process_bytes, 12_000 + 2_048 + 1_000);
        // Hand-computed: 31_499 - 15_548 = 15_951.
        assert_eq!(plan.page_cache_left, 15_951);
    }

    #[test]
    fn one_byte_short_of_the_ceiling_drops_one_place_not_to_refusal() {
        // 15_547 < 15_548 (4 places), so the search drops to 3 places
        // (12_548 <= 15_547) — a smaller resident set, never a refusal.
        let plan = plan_store_tiers(&budget(RING_TIER_P4 - 1)).unwrap();
        assert_eq!(
            plan.tier,
            StoreTier::ExclusiveRing {
                places_per_layer: 3,
                ring_bytes: 2_048,
            }
        );
        // Hand-computed: 15_547 - 12_548 = 2_999.
        assert_eq!(plan.page_cache_left, (RING_TIER_P4 - 1) - RING_TIER_P3);
    }

    #[test]
    fn a_zero_ceiling_is_a_valid_all_from_disk_tier() {
        // Nothing resident: the ring alone (3_548) carries every miss.
        let mut b = budget(RING_ONLY);
        b.max_places_per_layer = 0;
        let plan = plan_store_tiers(&b).unwrap();
        assert_eq!(
            plan.tier,
            StoreTier::ExclusiveRing {
                places_per_layer: 0,
                ring_bytes: 2_048,
            }
        );
        assert_eq!(plan.store_bytes, 0);
        assert_eq!(plan.page_cache_left, 0);
    }

    #[test]
    fn the_refusal_names_the_ring_only_last_resort() {
        // 3_547 < 3_548: not even zero places with the ring fits.
        let err = plan_store_tiers(&budget(RING_ONLY - 1)).unwrap_err();
        match err {
            BudgetError::Overcommitted { what, needed, available } => {
                assert!(what.contains("not even zero"), "{what}");
                assert_eq!(needed, RING_ONLY);
                assert_eq!(available, RING_ONLY - 1);
            }
            other => panic!("expected Overcommitted, got {other:?}"),
        }
    }

    #[test]
    fn a_page_cache_floor_below_one_byte_refuses_inclusive_for_the_ring_tier() {
        // Available 32_499: inclusive needs 31_500 but leaves 999 < 1000,
        // so the floor pushes it out; 4 places leave
        // 32_499 - 15_548 = 16_951 >= 1000 (hand-computed).
        let mut b = budget(INCLUSIVE_NEEDED + 999);
        b.page_cache_min = 1_000;
        let plan = plan_store_tiers(&b).unwrap();
        assert_eq!(
            plan.tier,
            StoreTier::ExclusiveRing {
                places_per_layer: 4,
                ring_bytes: 2_048,
            }
        );
        assert_eq!(plan.page_cache_left, 16_951);
    }

    #[test]
    fn the_page_cache_floor_refuses_the_last_tier_too() {
        // Ring-only leaves 999 < 1000 even though it fits without the floor.
        let mut b = budget(RING_ONLY + 999);
        b.max_places_per_layer = 0;
        b.page_cache_min = 1_000;
        assert!(matches!(
            plan_store_tiers(&b),
            Err(BudgetError::Overcommitted { .. })
        ));
    }

    #[test]
    fn zero_geometry_is_bad_input() {
        let mut b = budget(RING_TIER_P4);
        b.record_bytes = 0;
        assert!(matches!(
            plan_store_tiers(&b),
            Err(BudgetError::BadInput(_))
        ));
    }

}
