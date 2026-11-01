//! Unit tests of the pure part of the op executor. CPU only. The launch wiring against real
//! wrappers is covered by a scratch device run.
use super::common_tests::dims;
use super::dims::{KvMode, StepOptions};
use super::mixer::{CardKindOps, MixerViews, tail_extent};
use super::mixer_geometry::MixerGeometry;
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
