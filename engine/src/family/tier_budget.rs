//! The admission arithmetic of the tiered offload.
//!
//! WHAT: a pure planner that decides, in bytes, how the card is split at
//! load. [`plan_vram`] gives the KV cache of the configured context VRAM first
//! and the per-layer expert bank the rest, spilling KV positions to the host
//! only when they do not fit or when the minimum bank needs the room. The host
//! side is planned from the shared host facts (`family::mem`) by each
//! family's host plan. WHY (REV4 c.4, north
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

fn bad_input(what: &str) -> BudgetError {
    BudgetError::BadInput(String::from(what))
}

#[cfg(test)]
mod tests {
    //! The 35B-A3B-shape numbers live here and only here; production code
    //! takes every number as an input. GiB = 2^30 bytes throughout.

    use super::{BudgetError, ExpertSize, KvPlan, VramBudget, bad_input, plan_vram};
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
    }
}
