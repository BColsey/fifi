//! Unit tests of the pure part of the op executor. CPU only. The launch wiring against real
//! wrappers is covered by a scratch device run.
use super::common_tests::dims;
use super::dims::{KvMode, StepOptions};
use super::mixer::{CardKindOps, MixerViews, tail_extent};
use super::mixer_geometry::MixerGeometry;
use super::mixer_ops::CardOp;
use super::shard::ShardTable;
use super::step_plan::{KernelScratch, Slot, StepPlan};
use crate::family::project::cuda::WeightLayout;

#[test]
fn a_projection_writes_the_tail_of_its_slot_from_its_column() {
    // DeltaProj at one row: 12,352 floats; the z projection starts at column 8,192.
    assert_eq!(tail_extent(12352 * 4, 8192).expect("z"), (32768, 16640));
    assert_eq!(tail_extent(12352 * 4, 0).expect("qkv"), (0, 49408));
    assert_eq!(tail_extent(9216 * 4, 8704).expect("v"), (34816, 2048));
    assert!(
        tail_extent(12352 * 4, 12352).is_err(),
        "a column past the row"
    );
    assert!(tail_extent(16, usize::MAX).is_err(), "no overflow");
}

/// Every LOCAL op of a card's mixer lists names a slot the card plan holds, for both kinds,
/// both KV modes, and 1/2/3-card tables (the markers carry no slots - the ring moves ranges
/// the coordinator holds). Every card names the gathered rows under BOTH modes.
#[test]
fn card_ops_name_only_slots_the_card_plan_holds() {
    let d = dims();
    for cards in [1, 2, 3] {
        let table = ShardTable::new(&d.shape, cards, 32).expect("table");
        for kv_mode in [KvMode::Fp16, KvMode::Int8G32] {
            let options = StepOptions {
                kv_mode,
                context_tokens: 8,
                delta_slots: 1,
                forked_experts: false,
                tiered: None,
                layout: WeightLayout::Native,
            };
            let scratch = KernelScratch::for_dims(&d, 1, kv_mode).expect("scratch");
            // Same whole-group refusal as the slot test above.
            if matches!(kv_mode, KvMode::Int8G32) && cards == 3 {
                let g1 = MixerGeometry::for_card(&d, &table, 1).expect("card geometry");
                assert!(CardKindOps::new(&table, 1, &g1, kv_mode).is_err());
                continue;
            }
            for card in 0..cards {
                let plan =
                    StepPlan::for_card(&d, &table, card, &options, &scratch).expect("card plan");
                let g = MixerGeometry::for_card(&d, &table, card).expect("card geometry");
                let ops = CardKindOps::new(&table, card, &g, kv_mode).expect("card ops");
                let named = MixerViews::card_slot_indices(&plan, &ops).expect("slots");
                assert!(
                    named.iter().any(|(slot, _)| *slot == Slot::KRows),
                    "{cards}/{card} {kv_mode:?}: the card lists name the gathered K rows"
                );
                assert!(
                    named.iter().any(|(slot, _)| *slot == Slot::VRows),
                    "{cards}/{card} {kv_mode:?}: the card lists name the gathered V rows"
                );
            }
        }
    }
}

/// The per-card arena reuses a slot only across non-overlapping plan
/// lifetimes, so an exchange is safe exactly when it sits INSIDE its
/// slot's lifetime: the producing local op (before the marker) and the
/// consuming local op (after it) must both run at phases the plan keeps
/// the slot live for. A list that let the producer's phase fall before
/// `first` (or the consumer's past `last`) would let the planner alias
/// another tensor over the row a peer's publish still reads - the
/// "source overwritten before the publish" hazard. Markers carry no
/// slots; this test maps them to the rows the ring moves.
#[test]
fn exchange_markers_sit_inside_the_exchanged_slots_plan_lifetimes() {
    let d = dims();
    for cards in [1usize, 2, 3] {
        let table = ShardTable::new(&d.shape, cards, 32).expect("table");
        for kv_mode in [KvMode::Fp16, KvMode::Int8G32] {
            let options = StepOptions {
                kv_mode,
                context_tokens: 8,
                delta_slots: 1,
                forked_experts: false,
                tiered: None,
                layout: WeightLayout::Native,
            };
            let scratch = KernelScratch::for_dims(&d, 1, kv_mode).expect("scratch");
            // The q8 card decode serves whole 8-head groups, so the 3-card
            // split 8/4/4 plans fine but the card op list refuses the
            // partial groups before they reach the decode planning.
            if matches!(kv_mode, KvMode::Int8G32) && cards == 3 {
                let g1 = MixerGeometry::for_card(&d, &table, 1).expect("card geometry");
                assert!(CardKindOps::new(&table, 1, &g1, kv_mode).is_err());
                continue;
            }
            for card in 0..cards {
                let plan =
                    StepPlan::for_card(&d, &table, card, &options, &scratch).expect("card plan");
                let g = MixerGeometry::for_card(&d, &table, card).expect("card geometry");
                let ops = CardKindOps::new(&table, card, &g, kv_mode).expect("card ops");
                for kind in [
                    crate::family::AttentionKind::DeltaNet,
                    crate::family::AttentionKind::Full,
                ] {
                    let list = ops.of(kind).expect("kind list");
                    for (index, op) in list.iter().enumerate() {
                        let slot = match op {
                            CardOp::GatherK => Slot::KRows,
                            CardOp::GatherV => Slot::VRows,
                            CardOp::MixerSum => Slot::MixOut,
                            CardOp::Local(_) => continue,
                        };
                        let producer = list[..index].iter().rev().find_map(|op| match op {
                            CardOp::Local(block)
                                if block.writes().contains(&slot) =>
                            {
                                Some(block.phase() as usize)
                            }
                            _ => None,
                        }).unwrap_or_else(|| {
                            panic!("{cards}/{card} {kind:?} {kv_mode:?}: no local op writes {slot:?} before its exchange")
                        });
                        let consumer = list[index + 1..].iter().find_map(|op| match op {
                            CardOp::Local(block) if block.reads().contains(&slot) => {
                                Some(block.phase() as usize)
                            }
                            _ => None,
                        }).unwrap_or_else(|| {
                            panic!("{cards}/{card} {kind:?} {kv_mode:?}: no local op reads {slot:?} after its exchange")
                        });
                        let tensor = plan
                            .tensors()
                            .iter()
                            .find(|tensor| tensor.slot == slot)
                            .unwrap_or_else(|| panic!("the card plan has no {slot:?}"));
                        assert!(
                            tensor.first <= producer,
                            "{cards}/{card} {kind:?} {kv_mode:?}: {slot:?} live from phase {} but produced at {producer}",
                            tensor.first
                        );
                        assert!(
                            consumer <= tensor.last,
                            "{cards}/{card} {kind:?} {kv_mode:?}: {slot:?} dead after phase {} but consumed at {consumer}",
                            tensor.last
                        );
                    }
                }
            }
        }
    }
}
