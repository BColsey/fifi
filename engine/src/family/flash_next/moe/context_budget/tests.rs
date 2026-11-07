//! The context budget at the sizes of the 3.05 bpw pack on a 24 GiB card: the slot counts agree
//! with `TierPlan` at the cost model's contexts, the search finds the exact boundary of a slot
//! floor, and a refusal names the longest context that would work.

use super::*;
use crate::family::flash_next::moe::tier_plan::TierShape;

const MIB: u64 = 1 << 20;
const EXPERT_BYTES: u64 = 1_881_600;
const TRUNK_BYTES: u64 = 3_854_000_000;
const STATE_BYTES: u64 = 36 * 3 * MIB;

const FLASH_NEXT: KvGeometry = KvGeometry {
    cached_layers: 12,
    heads_kv: 2,
    head_dim: 256,
    idx_dim: 128,
    ratio: 4,
};

fn budget() -> ContextBudget {
    let shape = TierShape::from_pack(48, 512, 10, 24_576 * EXPERT_BYTES).unwrap();
    let placeholder = KvPlan {
        positions: 1,
        bytes_per_position: 1,
    };
    ContextBudget::new(
        TierInputs::cost_model_3090(shape, (TRUNK_BYTES, STATE_BYTES), placeholder),
        FLASH_NEXT,
    )
}

fn slots(positions: u64, format: KvFormat) -> usize {
    budget()
        .plan_at(positions, format)
        .unwrap()
        .slots_per_layer()
}

fn message(error: EngError) -> String {
    error.describe()
}

#[test]
fn a_position_costs_the_k_v_rows_the_indexer_keys_and_for_int8_the_row_scales() {
    // 12 layers x (2 K/V rows x 2 heads x 256 x 2 bytes + 128 f32 values per 4 positions).
    assert_eq!(FLASH_NEXT.position_bytes(KvFormat::F16).unwrap(), 26_112);
    // One-byte elements halve the rows (12,288), the indexer keys stay (1,536), and each of the
    // 12 x 2 x 2 cached rows carries an f32 scale (192).
    assert_eq!(FLASH_NEXT.position_bytes(KvFormat::Int8).unwrap(), 14_016);
    let mut broken = FLASH_NEXT;
    broken.ratio = 3;
    assert!(broken.position_bytes(KvFormat::F16).is_err());
}

#[test]
fn the_slot_counts_equal_the_tier_plan_at_the_cost_models_contexts() {
    assert_eq!(slots(262_144, KvFormat::F16), 142);
    assert_eq!(slots(131_072, KvFormat::F16), 180);
    assert_eq!(slots(32_768, KvFormat::F16), 208);
}

#[test]
fn the_hot_set_shrinks_as_the_context_grows_over_the_served_range() {
    let contexts: Vec<u64> = (1..=8).map(|n| n * 32_768).collect();
    let mut previous = usize::MAX;
    for (positions, plan) in budget().curve(KvFormat::F16, &contexts) {
        let plan = plan.unwrap_or_else(|e| panic!("{positions}: {}", message(e)));
        assert!(
            plan.slots_per_layer() < previous,
            "{positions}: {} slots after {previous}",
            plan.slots_per_layer()
        );
        previous = plan.slots_per_layer();
        // 128K to 256K keeps at least 142 of 512 experts per layer on the card, and the rest fits
        // the pinned host budget (31 GiB at the longest context).
        if positions >= 131_072 {
            assert!(previous >= 142, "{positions}: {previous} slots");
        }
        assert!(plan.cold_bytes() <= 40 * (1 << 30), "{positions}");
    }
}

#[test]
fn int8_k_v_gains_slots_but_fewer_than_unscaled_bytes_would_promise() {
    let f16 = slots(262_144, KvFormat::F16);
    let int8 = slots(262_144, KvFormat::Int8);
    // The unscaled count of the plan's own test is 178; the scales cost a few slots of it.
    assert!(f16 < int8 && int8 <= 178, "{f16} {int8}");
}

#[test]
fn the_longest_context_for_a_floor_is_the_exact_boundary() {
    let budget = budget();
    for floor in [143usize, 160, 180, 200] {
        let longest = budget
            .largest_context(KvFormat::F16, floor, 1 << 20)
            .unwrap_or_else(|| panic!("floor {floor}"));
        assert_eq!(longest % CONTEXT_GRANULE, 0);
        assert!(budget.meets(longest, KvFormat::F16, floor), "floor {floor}");
        assert!(
            !budget.meets(longest + CONTEXT_GRANULE, KvFormat::F16, floor),
            "floor {floor}: {longest} is not the boundary"
        );
    }
    // A higher floor allows a shorter context.
    let at = |floor| {
        budget
            .largest_context(KvFormat::F16, floor, 1 << 20)
            .unwrap()
    };
    assert!(at(143) > at(160) && at(160) > at(180) && at(180) > at(200));
    // The limit caps the answer.
    assert_eq!(
        budget.largest_context(KvFormat::F16, 100, 65_536),
        Some(65_536)
    );
    assert_eq!(budget.largest_context(KvFormat::F16, 100, 1_000), None);
    // 513 slots do not exist in a 512-expert layer.
    assert_eq!(budget.largest_context(KvFormat::F16, 513, 1 << 20), None);
}

#[test]
fn resolve_serves_what_fits_and_names_the_boundary_of_what_does_not() {
    let budget = budget();
    let served = budget.resolve(262_144, KvFormat::F16, 100).unwrap();
    assert_eq!(served.positions, 262_144);
    assert_eq!(served.plan.slots_per_layer(), 142);

    let text = message(budget.resolve(262_144, KvFormat::F16, 160).unwrap_err());
    let longest = budget.largest_context(KvFormat::F16, 160, 262_144).unwrap();
    assert!(text.contains("leave 142 slots per layer"), "{text}");
    assert!(
        text.contains(&format!("the longest context with 160 slots is {longest}")),
        "{text}"
    );

    // K/V that cannot fit the card at all: refused by the plan, with the boundary still named.
    let text = message(budget.resolve(2_000_000, KvFormat::F16, 100).unwrap_err());
    assert!(text.contains("KvDoesNotFit"), "{text}");
    assert!(
        text.contains("the longest context with 100 slots is"),
        "{text}"
    );

    let text = message(budget.resolve(0, KvFormat::F16, 1).unwrap_err());
    assert!(text.contains("zero positions"), "{text}");
}

#[test]
fn a_format_without_a_kernel_is_planned_but_not_served() {
    let text = message(budget().resolve(131_072, KvFormat::Int8, 100).unwrap_err());
    assert!(text.contains("no attention kernel yet"), "{text}");
    assert!(budget().plan_at(131_072, KvFormat::Int8).is_ok());
}

#[test]
fn another_model_needs_only_a_shape_and_a_geometry() {
    // 24 layers of 128 experts of 3 MiB, 6 cached attention layers of 4 key-value heads of 128, with an
    // indexer of 64 per 8 positions: nothing here is a Flash-Next number.
    let shape = TierShape::from_pack(24, 128, 8, 24 * 128 * 3 * MIB).unwrap();
    let placeholder = KvPlan {
        positions: 1,
        bytes_per_position: 1,
    };
    let other = ContextBudget::new(
        TierInputs::cost_model_3090(shape, (2_000_000_000, 64 * MIB), placeholder),
        KvGeometry {
            cached_layers: 6,
            heads_kv: 4,
            head_dim: 128,
            idx_dim: 64,
            ratio: 8,
        },
    );
    // 24 x 128 x 3 MiB = 9 GiB of experts: it all fits next to 131,072 positions of K/V.
    let plan = other.resolve(131_072, KvFormat::F16, 128).unwrap();
    assert_eq!(plan.plan.slots_per_layer(), 128);
    assert_eq!(plan.plan.cold_bytes(), 0);
}
