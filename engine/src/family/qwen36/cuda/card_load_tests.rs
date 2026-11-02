//! Host tests of the per-card upload manifest: the §A gates of the real
//! loader — completeness, exact byte tiling, replication rules, expert
//! accounting, and the one-card regression — all against the real
//! Qwen3.6-35B-A3B inventory the shard tests use.
use crate::family::gguf_model::{Inventory, Role};
use crate::family::qwen36::cuda::card_load::card_loads;
use crate::family::qwen36::cuda::shard::{ShardLoad, ShardTable};
use crate::family::qwen36::host::HostShape;
use crate::family::spec::QWEN3_6_35B_A3B;
use crate::kernels::cuda::plan::{assemble_slice, native_device_bytes, plan_slice_rows};
use crate::weights::GgufType;

fn shape() -> HostShape {
    HostShape::from_spec(&QWEN3_6_35B_A3B, 64).expect("real shape")
}

#[test]
fn two_card_manifests_cover_the_inventory_exactly() {
    let inventory = Inventory::for_spec(&QWEN3_6_35B_A3B).expect("inventory");
    let table = ShardTable::new(&shape(), 2, 256).expect("two cards at 256");
    let loads = card_loads(&table, &inventory, GgufType::Q4K).expect("manifests");
    assert_eq!(loads.len(), 2);
    let key_channels = (2 * shape().delta.key_heads * shape().delta.key_dim) as u64;
    for tensor in inventory.tensors() {
        let find = |card: usize| {
            loads[card]
                .entries
                .iter()
                .find(|entry| entry.name == tensor.name)
                .unwrap_or_else(|| panic!("{} missing from card {card}", tensor.name))
        };
        let (e0, e1) = (find(0), find(1));
        let whole = native_device_bytes(&tensor.name, e0.ty, &tensor.dims).expect("whole bytes");
        match (&e0.load, &e1.load) {
            (ShardLoad::Whole, ShardLoad::Whole) => {
                assert_eq!((e0.device_bytes, e1.device_bytes), (whole, whole));
            }
            (ShardLoad::CardZeroOnly, ShardLoad::CardZeroOnly) => {
                assert_eq!(tensor.role, Role::TokenEmbd);
                assert_eq!((e0.device_bytes, e1.device_bytes), (whole, whole));
                assert!(e0.slice.is_none() && e1.slice.is_none());
            }
            (ShardLoad::Sliced(_), ShardLoad::Sliced(_)) => {
                let (s0, s1) = (
                    e0.slice.as_ref().expect("card 0 slice"),
                    e1.slice.as_ref().expect("card 1 slice"),
                );
                assert_eq!(e0.device_bytes, s0.bytes);
                assert_eq!(e1.device_bytes, s1.bytes);
                // The pair uploads the whole tensor, except the fused
                // DeltaNet QKV/conv which over-cover by their replicated
                // key copy (see the module doc).
                let expected = match tensor.role {
                    Role::AttnQkv | Role::SsmConv1d => {
                        let key_rows: Vec<std::ops::Range<u64>> =
                            std::iter::once(0..key_channels).collect();
                        let key_copy = plan_slice_rows(
                            &tensor.name,
                            e0.ty,
                            &tensor.dims,
                            0..tensor.dims[0],
                            &key_rows,
                        )
                        .expect("key slice must plan")
                        .bytes;
                        whole + key_copy
                    }
                    _ => whole,
                };
                assert_eq!(
                    e0.device_bytes + e1.device_bytes,
                    expected,
                    "{}: the pair must cover the tensor exactly",
                    tensor.name
                );
            }
            other => panic!("{}: asymmetric loads {other:?}", tensor.name),
        }
    }
    // The totals are the entries; the embedding entry rides every
    // manifest but only card 0 uploads it.
    for (card, load) in loads.iter().enumerate() {
        let sum: u64 = load
            .entries
            .iter()
            .map(|entry| match (&entry.load, card) {
                (ShardLoad::CardZeroOnly, c) if c != 0 => 0,
                _ => entry.device_bytes,
            })
            .sum();
        assert_eq!(load.upload_bytes(), sum, "card {card} totals its entries");
    }
    assert_eq!(loads[1].solo_bytes, 0);
    let embed_entry = loads[0]
        .entries
        .iter()
        .find(|entry| matches!(entry.load, ShardLoad::CardZeroOnly))
        .expect("the embedding entry");
    // The embedding plans at the file quant when its role allows it, so
    // the manifest's own figure is the assertion, not a hardcoded type.
    assert_eq!(loads[0].solo_bytes, embed_entry.device_bytes);
}

#[test]
fn a_one_card_manifest_loads_whole_tensors_only() {
    let inventory = Inventory::for_spec(&QWEN3_6_35B_A3B).expect("inventory");
    let table = ShardTable::new(&shape(), 1, 256).expect("one card");
    let loads = card_loads(&table, &inventory, GgufType::Q4K).expect("manifests");
    let [load] = &loads[..] else {
        panic!("one card, one manifest");
    };
    assert_eq!(load.sliced_bytes, 0);
    let mut whole = 0u64;
    for entry in &load.entries {
        assert!(
            matches!(entry.load, ShardLoad::Whole | ShardLoad::CardZeroOnly),
            "{} must load whole on one card",
            entry.name
        );
        assert!(entry.slice.is_none());
        whole += entry.device_bytes;
    }
    // Card 0 is the only card, so it uploads everything: the one-device
    // regression invariant at manifest level.
    assert_eq!(load.upload_bytes(), whole);
}

#[test]
fn every_expert_plane_is_dealt() {
    let inventory = Inventory::for_spec(&QWEN3_6_35B_A3B).expect("inventory");
    let table = ShardTable::new(&shape(), 2, 256).expect("two cards");
    let loads = card_loads(&table, &inventory, GgufType::Q4K).expect("manifests");
    let layers = inventory
        .tensors()
        .iter()
        .filter(|tensor| tensor.role == Role::FfnGateInp)
        .count();
    assert!(layers > 0, "the spec has MoE layers");
    for (role, what) in [
        (Role::FfnGateExps, "gate"),
        (Role::FfnUpExps, "up"),
        (Role::FfnDownExps, "down"),
    ] {
        let entries: Vec<_> = loads[0]
            .entries
            .iter()
            .filter(|entry| inventory.get(&entry.name).is_some_and(|t| t.role == role))
            .collect();
        assert_eq!(entries.len(), layers, "{what}: one stack per MoE layer");
        for entry in entries {
            // The stack keeps all 256 expert planes; a slice cuts inside
            // every plane, so no card can hold a partial expert set.
            assert_eq!(entry.dims[2], 256, "{} holds 256 experts", entry.name);
            assert!(entry.slice.is_some(), "{} is dealt", entry.name);
        }
    }
}

#[test]
fn contiguous_row_slices_reassemble_the_source_bytes() {
    let inventory = Inventory::for_spec(&QWEN3_6_35B_A3B).expect("inventory");
    let table = ShardTable::new(&shape(), 2, 256).expect("two cards");
    let loads = card_loads(&table, &inventory, GgufType::Q4K).expect("manifests");
    // Contiguous single-range row slices: card 0's rows precede card 1's,
    // so the assembled halves concatenated in card order ARE the file's
    // bytes. (Multi-range and column slices are covered byte-exactly by
    // the pair-sum gate above and the GPU sliced-upload stages.)
    for role in [
        Role::AttnQ,
        Role::AttnK,
        Role::AttnV,
        Role::AttnGate,
        Role::FfnGateExps,
    ] {
        let tensors: Vec<_> = inventory
            .tensors()
            .iter()
            .filter(|tensor| tensor.role == role)
            .take(1)
            .collect();
        let Some(tensor) = tensors.first() else {
            panic!("the inventory holds a {role:?} tensor");
        };
        let entry = loads[0]
            .entries
            .iter()
            .find(|entry| entry.name == tensor.name)
            .expect("card 0 entry");
        let whole = native_device_bytes(&tensor.name, entry.ty, &tensor.dims).expect("whole bytes");
        let payload: Vec<u8> = (0..whole as usize)
            .map(|byte| (byte as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) as u8)
            .collect();
        let mut reassembled = Vec::with_capacity(payload.len());
        for load in &loads {
            let entry = load
                .entries
                .iter()
                .find(|entry| entry.name == tensor.name)
                .expect("card entry");
            let slice = entry.slice.as_ref().expect("sliced");
            reassembled.extend(assemble_slice(&payload, slice).expect("assemble"));
        }
        assert_eq!(
            reassembled, payload,
            "{}: the pair reassembles the source bytes",
            tensor.name
        );
    }
}
