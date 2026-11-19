//! The disk rung on the device: a Disk-mode bank over a synthetic pack file serves its
//! misses through the one-slot O_DIRECT staging ring, and the bytes that land in the
//! device slots are exactly the pack's records for that (layer, expert) - the same bytes
//! an Exclusive-mode bank holds for the same experts. An eviction and a second miss of
//! the same expert reads the file again (this tier never writes back) and lands the same
//! bytes again.

use super::bank::{HostMemory, HostPinPlan, HostTerms, RecordSource, device::DeviceBank};
use super::bank_source_tests::synthetic;
use super::ring_source::RingRecordSource;
use crate::family::expert_bank::{BankGeometry, ExpertBankState, HostMode, ReplaceOnMissLru};
use crate::kernels::cuda::device_present;

#[test]
#[ignore = "needs a GPU (under 1 GiB)"]
fn the_disk_rung_lands_the_packs_records_in_the_device_slots() {
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
    // 1, 2, 0, 1 hit, evict 1, hit, and re-miss 1 after its eviction: two ring reads per
    // layer, and expert 1 re-reads the file after its own eviction.
    let geometry = BankGeometry::new(synthetic::LAYERS, synthetic::EXPERTS, 1, 2, record).unwrap();
    let ring_slots = std::num::NonZeroUsize::MIN;
    let ring_bytes =
        RingRecordSource::backing_bytes(checkpoint, &dir, ring_slots).expect("the backing sizes");
    let backing: &'static mut [u8] = Box::leak(vec![0u8; ring_bytes].into_boxed_slice());
    let ring = RingRecordSource::new(checkpoint, &dir, ring_slots, backing)
        .expect("the ring source derives");
    let retained: &'static dyn RecordSource = Box::leak(Box::new(ring));

    let row: Vec<i32> = vec![0, 1, -1];
    let rows: Vec<&[i32]> = vec![row.as_slice(); synthetic::LAYERS];
    let counts = vec![vec![0u64; synthetic::EXPERTS]; synthetic::LAYERS];
    let memory = HostMemory {
        total: 62 << 30,
        available: 55 << 30,
    };
    let terms = HostTerms::DEFAULT;
    let disk_policy = ReplaceOnMissLru::from_fill(geometry, &rows, &counts, None).unwrap();
    let disk_state = ExpertBankState::with_policy(geometry, HostMode::Disk, disk_policy).unwrap();
    let excl_policy = ReplaceOnMissLru::from_fill(geometry, &rows, &counts, None).unwrap();
    let excl_state =
        ExpertBankState::with_policy(geometry, HostMode::Exclusive, excl_policy).unwrap();
    let disk_plan = HostPinPlan::new_disk(&geometry, &memory, &terms, ring_bytes as u64)
        .expect("the rung plan admits");
    let excl_plan =
        HostPinPlan::new(&geometry, HostMode::Exclusive, &memory, &terms).expect("the plan");

    let projector = crate::family::project::cuda::CudaProjector::new().expect("projector");
    let ctx = projector.context();
    let mut disk = DeviceBank::build(ctx, disk_state, &buffered, &disk_plan, Some(retained))
        .expect("the rung bank builds");
    let mut excl = DeviceBank::build(ctx, excl_state, &buffered, &excl_plan, None)
        .expect("the exclusive bank builds");

    let mut expected = vec![0u8; record];
    for layer in 0..synthetic::LAYERS {
        // Expert 1 is resident from the fill and hits; expert 2 misses and evicts 1 (the
        // tie victim); expert 0 hits; expert 1 misses again after its own eviction: two
        // ring reads, never a write-back.
        for expert in [1u16, 2, 0, 1] {
            buffered
                .record(layer, expert, &mut expected)
                .expect("the pack record");
            let _ = disk
                .step(ctx, layer, &[i32::from(expert)])
                .expect("the rung step");
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
                "layer {layer} expert {expert} from the file"
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
    assert_eq!(stats.misses, 4, "two ring reads per layer");
    assert_eq!(stats.demotions, 0, "the rung never writes back");
    std::fs::remove_dir_all(&dir).ok();
}
