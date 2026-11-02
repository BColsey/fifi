//! Host tests of the upload-plan half of the manifest: what one card
//! names, what it slices, and what it must not name at all.
use crate::family::gguf_model::Inventory;
use crate::family::qwen36::cuda::card_load::card_loads;
use crate::family::qwen36::cuda::card_weights::card_upload_plan;
use crate::family::qwen36::cuda::shard::{ShardLoad, ShardTable};
use crate::family::qwen36::host::HostShape;
use crate::family::spec::QWEN3_6_35B_A3B;
use crate::weights::GgufType;

fn shape() -> HostShape {
    HostShape::from_spec(&QWEN3_6_35B_A3B, 64).expect("real shape")
}

fn manifests() -> Vec<crate::family::qwen36::cuda::card_load::CardLoad> {
    let inventory = Inventory::for_spec(&QWEN3_6_35B_A3B).expect("inventory");
    let table = ShardTable::new(&shape(), 2, 256).expect("two cards");
    card_loads(&table, &inventory, GgufType::Q4K).expect("manifests")
}

#[test]
fn card_zero_names_the_embedding_and_card_one_does_not() {
    let loads = manifests();
    let plan0 = card_upload_plan(&loads[0], 0);
    let plan1 = card_upload_plan(&loads[1], 1);
    // Every manifest entry is named exactly once, except the
    // card-zero-only embedding which card 1 must not name at all.
    assert_eq!(plan0.names.len(), loads[0].entries.len());
    assert_eq!(plan1.names.len(), loads[1].entries.len() - 1);
    for entry in &loads[1].entries {
        let named = plan1.names.contains(&entry.name);
        assert_eq!(
            named,
            !matches!(entry.load, ShardLoad::CardZeroOnly),
            "{}: card 1 names exactly its non-embedding entries",
            entry.name
        );
    }
    assert_eq!(
        plan0.names.len(),
        plan0
            .names
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .len()
    );
}

#[test]
fn the_slices_are_the_manifests_slices_and_nothing_else() {
    let loads = manifests();
    for (card, load) in loads.iter().enumerate() {
        let plan = card_upload_plan(load, card);
        let sliced: Vec<&String> = plan.slices.iter().map(|(name, _)| name).collect();
        for entry in &load.entries {
            let in_slices = sliced.contains(&&entry.name);
            assert_eq!(
                in_slices,
                entry.slice.is_some(),
                "{}: card {card} slices exactly the manifest's slices",
                entry.name
            );
        }
        // Every sliced name is also a named tensor of the same call.
        for (name, _) in &plan.slices {
            assert!(plan.names.contains(name), "{name} is named");
        }
    }
}

#[test]
fn a_one_card_plan_slices_nothing() {
    let inventory = Inventory::for_spec(&QWEN3_6_35B_A3B).expect("inventory");
    let table = ShardTable::new(&shape(), 1, 256).expect("one card");
    let loads = card_loads(&table, &inventory, GgufType::Q4K).expect("manifests");
    let plan = card_upload_plan(&loads[0], 0);
    assert!(plan.slices.is_empty());
    assert_eq!(plan.names.len(), loads[0].entries.len());
}
