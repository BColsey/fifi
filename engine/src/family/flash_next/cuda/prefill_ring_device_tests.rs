//! The prefill ring on the device: against a small frozen bank, the ring plans the non-resident
//! experts, streams them in behind a compute stream via events, and the bytes that land in the
//! ring slots are exactly the store's records. Three layers with a ring of two layers' worth of
//! entries exercises the wrap: layer 2 reuses layer 0's slots after `free_layer`.

use super::bank::{HostMemory, HostPinPlan, HostTerms, RecordSource, device::DeviceBank};
use super::prefill_ring::device::PrefillRing;
use crate::family::expert_bank::{BankGeometry, ExpertBankState, HostMode, ReplaceOnMissLru};
use crate::{EngError, kernels::cuda::device_present};

/// The record of an expert is a pattern of its layer and number, distinct per byte position
/// (the bank device tests' `Pattern`).
struct Pattern;

impl RecordSource for Pattern {
    fn record(&self, layer: usize, expert: u16, out: &mut [u8]) -> Result<(), EngError> {
        for (index, byte) in out.iter_mut().enumerate() {
            *byte = (index as u32 * 31 + u32::from(expert) * 7 + layer as u32 * 101) as u8;
        }
        Ok(())
    }
}

fn expected(layer: usize, expert: u16, record: usize) -> Vec<u8> {
    let mut out = vec![0u8; record];
    Pattern.record(layer, expert, &mut out).unwrap();
    out
}

/// 3 layers, 12 experts, 4 slots: experts 0..4 resident per layer, 8 streamed per layer, 24
/// entries through a ring of 16 (two layers' worth). Layers 0 and 1 take slots 0..16; layer 2
/// wraps onto 0..8 after layer 0 frees.
#[test]
#[ignore = "needs a GPU (under 1 GiB)"]
fn the_ring_streams_every_non_resident_expert_in_layer_order() {
    assert!(
        device_present("Flash-Next prefill ring"),
        "a GPU is required"
    );
    let record = 9216;
    let geometry = BankGeometry::new(3, 12, 2, 4, record).unwrap();
    let row: Vec<i32> = (0..12).map(|e| if e < 4 { e } else { -1 }).collect();
    let counts = vec![(0..12u64).collect::<Vec<_>>(); 3];
    let rows: Vec<&[i32]> = vec![row.as_slice(); 3];
    let policy = ReplaceOnMissLru::from_fill(geometry, &rows, &counts, None).unwrap();
    let state = ExpertBankState::with_policy(geometry, HostMode::Exclusive, policy).unwrap();

    let projector = crate::family::project::cuda::CudaProjector::new().expect("projector");
    let ctx = projector.context();
    let plan = HostPinPlan::new(
        &geometry,
        HostMode::Exclusive,
        &HostMemory {
            total: 62 << 30,
            available: 55 << 30,
        },
        &HostTerms::DEFAULT,
    )
    .expect("the host plan");
    let mut bank = DeviceBank::build(ctx, state, &Pattern, &plan, None).expect("build the bank");

    let compute = ctx.stream().expect("compute stream");
    let mut ring = PrefillRing::build(ctx, geometry, 16).expect("build the ring");
    ring.replan(&bank);
    assert_eq!(ring.slot_of(0, 4), Some(0));
    assert_eq!(ring.slot_of(0, 11), Some(7));
    assert_eq!(ring.slot_of(1, 4), Some(8));
    assert_eq!(ring.slot_of(2, 4), Some(0), "layer 2 wraps onto layer 0");
    assert_eq!(
        ring.slot_of(0, 3),
        None,
        "a resident expert is not streamed"
    );
    assert_eq!(ring.slot_of(3, 4), None, "past the last layer");

    // Two layers in flight covers every entry; the wrap onto layer 0's slots blocks until
    // layer 0 frees, which the driver cadence does between layers.
    ring.issue_until(ctx, &mut bank, 16).expect("issue");
    ring.wait_layer(&compute, 0).expect("wait layer 0");
    compute.synchronize().expect("sync");
    for expert in 4..12u16 {
        let slot = usize::from(ring.slot_of(0, expert).unwrap());
        let mut got = vec![0u8; record];
        ring.slots()
            .read_at(&compute, slot * record, &mut got)
            .expect("read the ring slot");
        assert_eq!(got, expected(0, expert, record), "layer 0 expert {expert}");
    }
    // The wrap: after layer 0 frees, layer 2's copies land in its slots, and layer 1's reads
    // (settled earlier) still see layer 1's records.
    ring.free_layer(&compute, 0).expect("free layer 0");
    ring.issue_until(ctx, &mut bank, 24)
        .expect("issue the wrap");
    ring.wait_layer(&compute, 1).expect("wait layer 1");
    compute.synchronize().expect("sync");
    for expert in 4..12u16 {
        let slot = usize::from(ring.slot_of(1, expert).unwrap());
        let mut got = vec![0u8; record];
        ring.slots()
            .read_at(&compute, slot * record, &mut got)
            .expect("read the ring slot");
        assert_eq!(got, expected(1, expert, record), "layer 1 expert {expert}");
    }
    ring.free_layer(&compute, 1).expect("free layer 1");
    ring.wait_layer(&compute, 2).expect("wait layer 2");
    compute.synchronize().expect("sync");
    for expert in 4..12u16 {
        let slot = usize::from(ring.slot_of(2, expert).unwrap());
        let mut got = vec![0u8; record];
        ring.slots()
            .read_at(&compute, slot * record, &mut got)
            .expect("read the ring slot");
        assert_eq!(got, expected(2, expert, record), "layer 2 expert {expert}");
    }
}

#[test]
#[ignore = "needs a GPU (under 1 GiB)"]
fn freeing_out_of_order_is_refused() {
    assert!(
        device_present("Flash-Next prefill ring"),
        "a GPU is required"
    );
    let record = 9216;
    let geometry = BankGeometry::new(2, 12, 2, 4, record).unwrap();
    let row: Vec<i32> = (0..12).map(|e| if e < 4 { e } else { -1 }).collect();
    let counts = vec![(0..12u64).collect::<Vec<_>>(); 2];
    let rows: Vec<&[i32]> = vec![row.as_slice(); 2];
    let policy = ReplaceOnMissLru::from_fill(geometry, &rows, &counts, None).unwrap();
    let state = ExpertBankState::with_policy(geometry, HostMode::Exclusive, policy).unwrap();
    let projector = crate::family::project::cuda::CudaProjector::new().expect("projector");
    let ctx = projector.context();
    let plan = HostPinPlan::new(
        &geometry,
        HostMode::Exclusive,
        &HostMemory {
            total: 62 << 30,
            available: 55 << 30,
        },
        &HostTerms::DEFAULT,
    )
    .expect("the host plan");
    let bank = DeviceBank::build(ctx, state, &Pattern, &plan, None).expect("build the bank");
    let compute = ctx.stream().expect("compute stream");
    let mut ring = PrefillRing::build(ctx, geometry, 16).expect("build the ring");
    ring.replan(&bank);
    assert!(ring.free_layer(&compute, 1).is_err());
}

#[test]
#[ignore = "needs a GPU (under 1 GiB)"]
fn selected_ring_copies_unique_misses_and_preserves_reads_across_wrap() {
    assert!(device_present("Flash-Next selected prefill ring"));
    let record = 9216;
    let geometry = BankGeometry::new(3, 12, 2, 4, record).unwrap();
    let row = [0, 1, 2, 3, -1, -1, -1, -1, -1, -1, -1, -1];
    let rows: Vec<&[i32]> = vec![row.as_slice(); geometry.layers];
    let counts = vec![vec![0; geometry.experts]; geometry.layers];
    let policy = ReplaceOnMissLru::from_fill(geometry, &rows, &counts, None).unwrap();
    let state = ExpertBankState::with_policy(geometry, HostMode::Exclusive, policy).unwrap();

    let projector = crate::family::project::cuda::CudaProjector::new().unwrap();
    let ctx = projector.context();
    let plan = HostPinPlan::new(
        &geometry,
        HostMode::Exclusive,
        &HostMemory {
            total: 62 << 30,
            available: 55 << 30,
        },
        &HostTerms::DEFAULT,
    )
    .unwrap();
    let mut bank = DeviceBank::build(ctx, state, &Pattern, &plan, None).unwrap();
    let compute = ctx.stream().unwrap();
    let captured = ctx.allocate(3 * record).unwrap();
    let mut ring = PrefillRing::build(ctx, geometry, 8).unwrap();
    ring.replan(&bank);
    ring.enable_selected();
    let before = bank.prefill_report().upload_bytes;

    let batches = [
        (0, &[4, 7, 4, 0][..], &[(4, 0), (7, 1)][..]),
        (1, &[0, 1][..], &[][..]),
        (2, &[4, 4][..], &[(4, 2)][..]),
    ];
    for (layer, picks, snapshots) in batches {
        ring.issue_selected(ctx, &mut bank, layer, picks).unwrap();
        ring.wait_layer(&compute, layer).unwrap();
        for &(expert, index) in snapshots {
            let slot = usize::from(ring.slot_of(layer, expert).unwrap());
            ctx.copy_device(
                &compute,
                ring.slots().range(slot * record, record).unwrap(),
                captured.range(index * record, record).unwrap(),
            )
            .unwrap();
        }
        ring.free_layer(&compute, layer).unwrap();
    }

    let mut got = vec![0; 3 * record];
    captured.read_at(&compute, 0, &mut got).unwrap();
    assert_eq!(
        bank.prefill_report().upload_bytes - before,
        (3 * record) as u64
    );
    for (bytes, (layer, expert)) in got.chunks_exact(record).zip([(0, 4), (0, 7), (2, 4)]) {
        assert_eq!(bytes, expected(layer, expert, record));
    }

    // A second issue for one layer in the same chunk is refused; a new chunk reopens it.
    ring.replan(&bank);
    ring.issue_selected(ctx, &mut bank, 0, &[4]).unwrap();
    let error = ring
        .issue_selected(ctx, &mut bank, 0, &[4, 7])
        .unwrap_err()
        .describe();
    assert!(error.contains("layer 0 twice"), "{error}");
}

#[test]
#[ignore = "needs a GPU (under 1 GiB)"]
fn lookahead_reuses_predictions_and_fills_misses_after_slot_conflicts() {
    assert!(device_present("Flash-Next prefill ring lookahead"));
    let record = 9216;
    let geometry = BankGeometry::new(2, 12, 2, 4, record).unwrap();
    let row = [0, 1, 2, 3, -1, -1, -1, -1, -1, -1, -1, -1];
    let rows: Vec<&[i32]> = vec![row.as_slice(); geometry.layers];
    let counts = vec![vec![0; geometry.experts]; geometry.layers];
    let policy = ReplaceOnMissLru::from_fill(geometry, &rows, &counts, None).unwrap();
    let state = ExpertBankState::with_policy(geometry, HostMode::Exclusive, policy).unwrap();

    let projector = crate::family::project::cuda::CudaProjector::new().unwrap();
    let ctx = projector.context();
    let plan = HostPinPlan::new(
        &geometry,
        HostMode::Exclusive,
        &HostMemory {
            total: 62 << 30,
            available: 55 << 30,
        },
        &HostTerms::DEFAULT,
    )
    .unwrap();
    let mut bank = DeviceBank::build(ctx, state, &Pattern, &plan, None).unwrap();
    let compute = ctx.stream().unwrap();
    let captured = ctx.allocate(4 * record).unwrap();
    let mut ring = PrefillRing::build(ctx, geometry, 12).unwrap();
    ring.replan(&bank);
    ring.enable_lookahead(geometry);
    let before = bank.prefill_report().upload_bytes;
    ring.issue_first(ctx, &mut bank).unwrap();
    for layer in 0..2 {
        ring.issue_selected(ctx, &mut bank, layer, &[4, 8]).unwrap();
        ring.wait_layer(&compute, layer).unwrap();
        ring.free_layer(&compute, layer).unwrap();
    }
    let uploads = |bank: &DeviceBank| (bank.prefill_report().upload_bytes - before) / record as u64;
    assert_eq!(uploads(&bank), 4);

    // A scheduler piece reopens the plan. Layer 0 preloads the two preceding picks,
    // including expert 4, whose slot conflicts with layer 1's predicted expert 8.
    ring.replan(&bank);
    assert_eq!(ring.slot_of(0, 4), ring.slot_of(1, 8));
    ring.issue_first(ctx, &mut bank).unwrap();
    assert_eq!(uploads(&bank), 6);
    let batches = [(0, &[8][..]), (1, &[4, 8, 10][..])];
    let mut index = 0;
    for (layer, picks) in batches {
        ring.issue_selected(ctx, &mut bank, layer, picks).unwrap();
        assert_eq!(uploads(&bank), if layer == 0 { 7 } else { 9 });
        ring.wait_layer(&compute, layer).unwrap();
        for &expert in picks {
            let slot = usize::from(ring.slot_of(layer, expert as u16).unwrap());
            ctx.copy_device(
                &compute,
                ring.slots().range(slot * record, record).unwrap(),
                captured.range(index * record, record).unwrap(),
            )
            .unwrap();
            index += 1;
        }
        ring.free_layer(&compute, layer).unwrap();
    }
    let mut got = vec![0; 4 * record];
    captured.read_at(&compute, 0, &mut got).unwrap();
    for (bytes, (layer, expert)) in got
        .chunks_exact(record)
        .zip([(0, 8), (1, 4), (1, 8), (1, 10)])
    {
        assert_eq!(bytes, expected(layer, expert, record));
    }

    ring.clear_history();
    ring.replan(&bank);
    ring.issue_first(ctx, &mut bank).unwrap();
    assert_eq!(uploads(&bank), 9);
}
