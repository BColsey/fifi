//! The disk rung on the device: a Disk-mode bank over a synthetic pack file serves its
//! misses through the pooled O_DIRECT ring store, and the bytes that land in the
//! device slots are exactly the pack's records for that (layer, expert) - the same bytes
//! an Exclusive-mode bank holds for the same experts. Reloads from disk and from the
//! optional victim cache must both preserve the pack bytes.

use super::bank::{HostMemory, HostPinPlan, HostTerms, RecordSource, device::DeviceBank};
use super::bank_source::RungSource;
use super::bank_source_tests::synthetic;
use crate::family::expert_bank::{BankGeometry, ExpertBankState, HostMode, ReplaceOnMissLru};
use crate::kernels::cuda::device_present;

#[test]
#[ignore = "needs a GPU (under 1 GiB)"]
fn the_disk_rung_lands_the_packs_records_in_the_device_slots() {
    compare_disk_and_exclusive(None);
}

#[test]
#[ignore = "needs a GPU (under 1 GiB)"]
fn a_disk_victim_returns_from_ram_after_eviction() {
    compare_disk_and_exclusive(Some(false));
}

#[test]
#[ignore = "needs a GPU (under 1 GiB)"]
fn selective_writeback_saves_only_the_recently_reused_victim() {
    compare_disk_and_exclusive(Some(true));
}

fn compare_disk_and_exclusive(selective: Option<bool>) {
    assert!(device_present("Flash-Next disk rung"), "a GPU is required");

    // The synthetic pack mirrored onto a filesystem that serves O_DIRECT.
    let (base, filesystem) = crate::family::expert_bank::host_tier::tests::direct_test_dir();
    let dir = base.join(format!("rung-dev-{}", std::process::id()));
    eprintln!("rung device test: {dir:?} on {filesystem}");
    std::fs::create_dir_all(&dir).expect("the rung test dir");
    let (checkpoint, layout, _broken, image) =
        synthetic::checkpoint_with_image(&|_, _, _| synthetic::Tweak::Normal);
    std::fs::write(dir.join("m.safetensors"), &image).expect("the synthetic shard mirrors");
    // The ring source borrows its checkpoint for the rest of the process (the bank keeps it
    // as the retained source), so both live leaked.
    let checkpoint = Box::leak(Box::new(checkpoint));
    let buffered = synthetic::source(checkpoint, &layout);
    let record = layout.record_bytes();

    // 2 layers, 3 experts, two slots (one pick + the filler slot is the admission floor);
    // the fill preloads experts 0 and 1 (count ties make 1 the first victim), then picks
    // 1, 1, 2, 0, 1 hit 1 twice, evict 1, hit 0, and reload 1 after its eviction.
    let geometry = BankGeometry::new(synthetic::LAYERS, synthetic::EXPERTS, 1, 2, record).unwrap();
    let rung_bytes = RungSource::backing_bytes(checkpoint, &dir, 1).expect("the backing sizes");
    let backing: &'static mut [u8] = Box::leak(vec![0u8; rung_bytes].into_boxed_slice());
    let mode = RungSource::probe_read_mode(checkpoint, &dir).expect("the mode probes");
    let rung =
        RungSource::new(checkpoint, &dir, 1, mode, backing).expect("the rung source derives");
    // The rung source borrows its checkpoint for the rest of the process (the bank keeps it
    // as the retained reader), so it lives leaked.
    let retained: &'static RungSource = Box::leak(Box::new(rung));

    let row: Vec<i32> = vec![0, 1, -1];
    let rows: Vec<&[i32]> = vec![row.as_slice(); synthetic::LAYERS];
    let counts = vec![vec![0u64; synthetic::EXPERTS]; synthetic::LAYERS];
    let memory = HostMemory::from_pair(62 << 30, 55 << 30);
    let terms = HostTerms::DEFAULT;
    let disk_policy = ReplaceOnMissLru::from_fill(geometry, &rows, &counts, None).unwrap();
    let disk_state = ExpertBankState::with_policy(geometry, HostMode::Disk, disk_policy).unwrap();
    let excl_policy = ReplaceOnMissLru::from_fill(geometry, &rows, &counts, None).unwrap();
    let excl_state =
        ExpertBankState::with_policy(geometry, HostMode::Exclusive, excl_policy).unwrap();
    let disk_plan = HostPinPlan::new_disk(&geometry, &memory, &terms, rung_bytes as u64)
        .expect("the rung plan admits");
    let excl_plan =
        HostPinPlan::new(&geometry, HostMode::Exclusive, &memory, &terms).expect("the plan");

    let projector = crate::family::project::cuda::CudaProjector::new().expect("projector");
    let ctx = projector.context();
    let mut disk = DeviceBank::build(ctx, disk_state, &buffered, &disk_plan, Some(retained))
        .expect("the rung bank builds");
    if let Some(select) = selective {
        let layout = super::victim_cache::VictimLayout::new(record, 1, record).unwrap();
        disk.enable_victim_cache(ctx, &layout, select)
            .expect("cache allocation");
    }
    let mut excl = DeviceBank::build(ctx, excl_state, &buffered, &excl_plan, None)
        .expect("the exclusive bank builds");

    let mut expected = vec![0u8; record];
    for layer in 0..synthetic::LAYERS {
        // Expert 1 qualifies for selective write-back. Expert 2 has one use when evicted.
        for (position, expert) in [1u16, 1, 2, 0, 1].into_iter().enumerate() {
            buffered
                .record(layer, expert, &mut expected)
                .expect("the pack record");
            if selective.is_some() {
                disk.step_decode(ctx, layer, &[i32::from(expert)], position as u64, &[])
                    .expect("the cached rung step");
            } else {
                disk.step(ctx, layer, &[i32::from(expert)])
                    .expect("the rung step");
            }
            let _ = excl
                .step(ctx, layer, &[i32::from(expert)])
                .expect("the store step");
            let stream = disk.stream();
            let slot = disk.slot_row(layer).expect("the rung row")[usize::from(expert)];
            assert!(slot >= 0, "layer {layer} expert {expert} is resident");
            let mut from_rung = vec![0u8; record];
            disk.slots(layer)
                .expect("the rung slots")
                .read_at(stream, slot as usize * record, &mut from_rung)
                .expect("read the rung slot");
            assert_eq!(
                from_rung, expected,
                "layer {layer} expert {expert} matches the file"
            );
            let stream = excl.stream();
            let slot = excl.slot_row(layer).expect("the store row")[usize::from(expert)];
            let mut from_store = vec![0u8; record];
            excl.slots(layer)
                .expect("the store slots")
                .read_at(stream, slot as usize * record, &mut from_store)
                .expect("read the store slot");
            assert_eq!(
                from_rung, from_store,
                "layer {layer} expert {expert}: the rung matches the store"
            );
        }
    }
    let stats = disk.stats();
    assert_eq!(stats.misses, 4, "two exact misses per layer");
    assert_eq!(
        disk.victim_cache_counts(),
        selective.map(|select| (2, if select { 2 } else { 4 }))
    );
    assert_eq!(stats.demotions, 0, "the disk policy has no host place");
    std::fs::remove_dir_all(&dir).ok();
}
