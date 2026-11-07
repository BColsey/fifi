//! Tests of the real-pack record source. The host test reads layer `FIFI_FLASH_NEXT_EXL3_MOE_LAYER`
//! (3 by default) of the pack in `FIFI_FLASH_NEXT_EXL3_PACK` and compares the records with the
//! bytes the shard headers locate; the device test builds a 16-slot bank over the same layer.

use std::{fs::File, os::unix::fs::FileExt, path::PathBuf};

use super::{bank::RecordSource, bank_source::PackRecordSource};
use crate::{
    EngError,
    family::{
        exl3_pack::{ExpertPart, Root, expert_name_in},
        flash_next::load::pack_files::FlashPackFiles,
    },
    tensor::Dtype,
    weights::ShardHeader,
};

const ROUTED: &str =
    include_str!("../../../../tests/fixtures/flash_next_routing/t0-decode-counts.tsv");

fn pack_dir() -> PathBuf {
    PathBuf::from(std::env::var_os("FIFI_FLASH_NEXT_EXL3_PACK").expect("FIFI_FLASH_NEXT_EXL3_PACK"))
}

fn moe_layer() -> usize {
    std::env::var("FIFI_FLASH_NEXT_EXL3_MOE_LAYER")
        .map_or(3, |value| value.parse().expect("a layer number"))
}

/// The bytes of tensor `key` straight from its shard: the index names the shard, the shard header
/// gives the offsets, and a positioned read takes them. Independent of the loader under test.
fn raw_tensor(dir: &std::path::Path, index: &serde_json::Value, key: &str) -> (Dtype, Vec<u8>) {
    let shard = index["weight_map"][key]
        .as_str()
        .unwrap_or_else(|| panic!("{key} is not in the index"));
    let file = File::open(dir.join(shard)).expect("open the shard");
    let len = file.metadata().expect("metadata").len();
    let mut reader = &file;
    let header = ShardHeader::read(&mut reader, len, 64 << 20).expect("the shard header");
    let entry = header
        .entry(key)
        .unwrap_or_else(|| panic!("{key} has no header entry"));
    let range = entry.offsets();
    let mut bytes = vec![0u8; (range.end - range.start) as usize];
    file.read_exact_at(&mut bytes, header.data_start() + range.start)
        .expect("read the tensor");
    (entry.dtype(), bytes)
}

/// F16 or F32 scale bytes as the F32 bytes a record holds.
fn widened(dtype: Dtype, bytes: &[u8]) -> Vec<u8> {
    match dtype {
        Dtype::F32 => bytes.to_vec(),
        Dtype::F16 => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .flat_map(|pair| half::f16::from_le_bytes(*pair).to_f32().to_le_bytes())
            .collect(),
        other => panic!("a scale stored as {other:?}"),
    }
}

#[test]
#[ignore = "reads experts of one layer of the real pack in FIFI_FLASH_NEXT_EXL3_PACK (about 8 MiB)"]
fn the_records_equal_the_pack_bytes_at_the_offsets_of_the_shard_headers() {
    let dir = pack_dir();
    let layer = moe_layer();
    let files = FlashPackFiles::open(&dir).expect("open the pack through the guard");
    let source = PackRecordSource::new(&files.checkpoint).expect("the record source");
    assert_eq!((source.layers(), source.experts()), (48, 512));
    assert_eq!(source.layout().record_bytes(), 1_881_600);
    let index: serde_json::Value = serde_json::from_slice(
        &std::fs::read(dir.join("model.safetensors.index.json")).expect("the index"),
    )
    .expect("the index json");
    for expert in [0usize, 1, 255, 256, 510, 511] {
        let record = source.read_record(layer, expert).expect("a record");
        assert_eq!(record.len(), 1_881_600);
        for (index_of_part, part) in ExpertPart::ALL.into_iter().enumerate() {
            let prefix = expert_name_in(Root::Layers, layer, expert, part);
            let layout = source.layout().part(index_of_part);
            let (_, trellis) = raw_tensor(&dir, &index, &format!("{prefix}.trellis"));
            assert_eq!(
                &record[layout.trellis.clone()],
                trellis.as_slice(),
                "expert {expert} {part:?}: trellis"
            );
            for (name, range) in [("suh", &layout.suh), ("svh", &layout.svh)] {
                let (dtype, stored) = raw_tensor(&dir, &index, &format!("{prefix}.{name}"));
                assert_eq!(
                    &record[range.clone()],
                    widened(dtype, &stored).as_slice(),
                    "expert {expert} {part:?}: {name} ({dtype:?} widened to F32)"
                );
            }
        }
        // The trait fills a caller's buffer with the same bytes.
        let mut out = vec![0u8; record.len()];
        source
            .record(layer, expert as u16, &mut out)
            .expect("the trait");
        assert_eq!(out, record);
    }
    assert!(source.read_record(48, 0).is_err(), "no layer 48");
    assert!(source.read_record(layer, 512).is_err(), "no expert 512");
    assert!(
        source.record(layer, 0, &mut [0u8; 16]).is_err(),
        "a short buffer"
    );
}

#[cfg(all(test, feature = "native-cuda", target_os = "linux"))]
mod device_tests {
    use super::*;
    use crate::{
        family::{
            flash_next::cuda::bank::{
                BankGeometry, ExpertBankState, HostMemory, HostMode, HostPinPlan, HostTerms,
                ReplaceOnMissLru, device::DeviceBank, parse_route_counts,
            },
            project::cuda::CudaProjector,
        },
        kernels::cuda::device_present,
    };

    /// A bank of one layer holds layer `layer` of the pack.
    struct OneLayer<'a, 'b> {
        inner: &'b PackRecordSource<'a>,
        layer: usize,
    }

    impl RecordSource for OneLayer<'_, '_> {
        fn record(&self, _layer: usize, expert: u16, out: &mut [u8]) -> Result<(), EngError> {
            self.inner.record(self.layer, expert, out)
        }
    }

    const SLOTS: usize = 16;

    /// One real layer, 16 slots filled from the popularity profile, 497 pinned places: about 31 MiB
    /// of VRAM and 0.9 GiB of pinned host memory.
    #[test]
    #[ignore = "needs a GPU (under 1 GiB), 0.9 GiB pinned host memory and FIFI_FLASH_NEXT_EXL3_PACK"]
    fn a_real_layer_fills_a_sixteen_slot_bank_and_a_miss_copies_and_writes_back() {
        assert!(
            device_present("Flash-Next real-layer bank"),
            "a GPU is required"
        );
        let layer = moe_layer();
        let files = FlashPackFiles::open(&pack_dir()).expect("open the pack");
        let source = PackRecordSource::new(&files.checkpoint).expect("the record source");
        let record = source.layout().record_bytes();
        let counts = parse_route_counts(ROUTED, 48, 512).expect("the profile")[layer].clone();
        // The static fill: the sixteen most picked experts of the layer, slots in id order.
        let mut ranked: Vec<usize> = (0..512).collect();
        ranked.sort_by_key(|&e| (std::cmp::Reverse(counts[e]), e));
        let mut held = ranked[..SLOTS].to_vec();
        held.sort_unstable();
        let mut row = vec![-1i32; 512];
        for (slot, &expert) in held.iter().enumerate() {
            row[expert] = slot as i32;
        }
        let geometry = BankGeometry::new(1, 512, 10, SLOTS, record).expect("geometry");
        let policy =
            ReplaceOnMissLru::from_fill(geometry, &[row.as_slice()], &[counts]).expect("fill");
        let state =
            ExpertBankState::with_policy(geometry, HostMode::Exclusive, policy).expect("state");
        let places = state.mode().places_per_layer(&geometry);
        assert_eq!(places, 512 - SLOTS + 1);

        let projector = CudaProjector::new().expect("projector");
        let ctx = projector.context();
        let pinned_before = ctx.live_owners().pinned;
        let started = std::time::Instant::now();
        // The host plan the build checks: this test pins 0.9 GiB, far under the 2 GiB bound, so the
        // plan is judged on a synthetic 62 GiB host and not on today's load from other lanes.
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
        let mut bank = DeviceBank::build(
            ctx,
            state,
            &OneLayer {
                inner: &source,
                layer,
            },
            &plan,
        )
        .expect("build");
        let pinned_built = ctx.live_owners().pinned;
        eprintln!(
            "built in {:.1} s; pinned owners {pinned_before} -> {pinned_built} bytes ({} places)",
            started.elapsed().as_secs_f64(),
            places
        );
        assert_eq!(pinned_built - pinned_before, places * record);
        assert!(places * record < 2 << 30, "under 2 GiB pinned");

        let read_slot = |bank: &DeviceBank, slot: usize| {
            let mut out = vec![0u8; record];
            bank.slots(0)
                .expect("layer 0")
                .read_at(bank.stream(), geometry.offset(slot), &mut out)
                .expect("read the slot");
            out
        };
        let pack_record = |expert: u16| {
            source
                .read_record(layer, usize::from(expert))
                .expect("record")
        };
        for slot in 0..SLOTS {
            let expert = bank.state().expert_in_slot(0, slot).expect("a filled slot");
            assert_eq!(
                read_slot(&bank, slot),
                pack_record(expert),
                "slot {slot} (expert {expert})"
            );
        }
        // Cold experts sit in the store at the places the bank assigned.
        let cold: Vec<u16> = (0..512u16).filter(|&e| row[usize::from(e)] < 0).collect();
        for &expert in [cold[0], cold[100], cold[495]].iter() {
            let place = bank
                .state()
                .places()
                .place_of(0, expert)
                .expect("a host place");
            let mut stored = vec![0u8; record];
            bank.read_place(0, place, &mut stored).expect("the place");
            assert_eq!(
                stored,
                pack_record(expert),
                "cold expert {expert} at place {place}"
            );
        }

        // One miss: nine resident picks and one cold expert.
        let resident: Vec<i32> = held.iter().take(9).map(|&e| e as i32).collect();
        let mut picks = resident.clone();
        picks.push(i32::from(cold[7]));
        let copies = bank.step(ctx, 0, &picks).expect("step");
        assert_eq!((copies.hits, copies.ops.len()), (9, 1));
        let op = copies.ops[0];
        assert_eq!(op.incoming, cold[7]);
        assert_eq!(
            read_slot(&bank, usize::from(op.slot)),
            pack_record(cold[7]),
            "the missed expert"
        );
        let demote = op.demote.expect("an exclusive store writes back");
        let mut written = vec![0u8; record];
        bank.read_place(0, demote.dst_place, &mut written)
            .expect("the place");
        assert_eq!(
            written,
            pack_record(demote.expert),
            "the evicted expert is in its new place"
        );
        assert_eq!(
            bank.state().slot_row(0)[usize::from(cold[7])],
            i32::from(op.slot)
        );
        assert_eq!(bank.state().slot_row(0)[usize::from(demote.expert)], -1);
        assert_ne!(op.slot, 0, "the filler slot is not refilled");

        // The same picks again hit; a second miss reuses the place the first copy freed.
        let again = bank.step(ctx, 0, &picks).expect("step");
        assert!(again.ops.is_empty());
        picks[9] = i32::from(cold[8]);
        let second = bank.step(ctx, 0, &picks).expect("step");
        assert_eq!(second.ops.len(), 1);
        assert_eq!(
            second.ops[0].demote.expect("write-back").dst_place,
            op.src_place
        );
        let op2 = second.ops[0];
        assert_eq!(
            read_slot(&bank, usize::from(op2.slot)),
            pack_record(op2.incoming)
        );
        let mut written = vec![0u8; record];
        bank.read_place(0, op2.demote.expect("write-back").dst_place, &mut written)
            .expect("the place");
        assert_eq!(written, pack_record(op2.demote.expect("write-back").expert));

        drop(bank);
        let pinned_after = ctx.live_owners().pinned;
        eprintln!(
            "pinned owners after the drop: {pinned_after} (before the build: {pinned_before})"
        );
        assert_eq!(pinned_after, pinned_before, "the pinned store was freed");
    }
}
