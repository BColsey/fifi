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
    let source = PackRecordSource::from_files(&files).expect("the record source");
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

// ---------------------------------------------------------------------------------------------
// A small synthetic pack: 2 layers of 3 experts, 128 x 128 matrices, one in-memory shard.
// ---------------------------------------------------------------------------------------------

mod synthetic {
    use std::{
        io::{Cursor, Read, Seek, SeekFrom},
        sync::{
            Arc,
            atomic::{AtomicBool, Ordering},
        },
    };

    use super::super::bank_source::PackRecordSource;
    use crate::{
        family::{
            exl3_pack::{ExpertPart, Root, expert_name_in},
            flash_next::moe::slot_bank::SlotBankLayout,
        },
        weights::{
            StreamedCheckpoint,
            exl3::{Checkpoint, Config},
        },
    };

    pub(super) const LAYERS: usize = 2;
    pub(super) const EXPERTS: usize = 3;
    pub(super) const WORDS: usize = 16;
    const WIDTH: usize = 128;

    /// What a matrix of the pack differs in from the normal one.
    #[derive(Clone, Copy, PartialEq, Eq)]
    pub(super) enum Tweak {
        Normal,
        /// Another rate: this many u16 words per tile.
        Words(usize),
        /// A wider input: this many rows.
        Rows(usize),
        /// Carries a `bias` tensor.
        Bias,
        /// Lacks its `svh` tensor.
        NoSvh,
    }

    /// A shard whose reads fail once `broken` is set (a file that shrinks or errors after open).
    struct Flaky {
        inner: Cursor<Vec<u8>>,
        broken: Arc<AtomicBool>,
    }

    impl Read for Flaky {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            if self.broken.load(Ordering::Acquire) {
                return Err(std::io::Error::other("the shard went away"));
            }
            self.inner.read(buf)
        }
    }

    impl Seek for Flaky {
        fn seek(&mut self, to: SeekFrom) -> std::io::Result<u64> {
            self.inner.seek(to)
        }
    }

    fn bytes(seed: usize, count: usize) -> Vec<u8> {
        (0..count)
            .map(|i| ((i * 31 + seed * 17 + (i >> 3)) & 0xff) as u8)
            .collect()
    }

    /// Finite F16 scales (0.5 to 2.0) as stored bytes.
    fn scales(seed: usize, count: usize) -> Vec<u8> {
        (0..count)
            .flat_map(|i| {
                half::f16::from_f32(((i * 7 + seed) % 97 + 32) as f32 / 64.0).to_le_bytes()
            })
            .collect()
    }

    type Entry = (String, &'static str, Vec<usize>, Vec<u8>);

    fn matrix_entries(prefix: &str, seed: usize, tweak: Tweak, entries: &mut Vec<Entry>) {
        let words = if let Tweak::Words(words) = tweak {
            words
        } else {
            WORDS
        };
        let rows = if let Tweak::Rows(rows) = tweak {
            rows
        } else {
            WIDTH
        };
        let trellis = bytes(seed, (rows / 16) * (WIDTH / 16) * words * 2);
        entries.push((
            format!("{prefix}.trellis"),
            "I16",
            vec![rows / 16, WIDTH / 16, words],
            trellis,
        ));
        entries.push((
            format!("{prefix}.suh"),
            "F16",
            vec![rows],
            scales(seed, rows),
        ));
        if tweak != Tweak::NoSvh {
            entries.push((
                format!("{prefix}.svh"),
                "F16",
                vec![WIDTH],
                scales(seed + 5, WIDTH),
            ));
        }
        entries.push((
            format!("{prefix}.mul1"),
            "I32",
            vec![],
            0x83dc_d12du32.to_le_bytes().to_vec(),
        ));
        if tweak == Tweak::Bias {
            entries.push((
                format!("{prefix}.bias"),
                "F16",
                vec![WIDTH],
                scales(seed + 9, WIDTH),
            ));
        }
    }

    /// Seed of the matrix of (layer, expert, part): every record differs.
    pub(super) fn seed(layer: usize, expert: usize, part: usize) -> usize {
        1 + layer * 100 + expert * 10 + part
    }

    /// The checkpoint of the pack, the layout of its normal matrices, and the switch that breaks
    /// the shard's reads. `tweak(layer, expert, part)` changes one matrix.
    pub(super) fn checkpoint(
        tweak: &dyn Fn(usize, usize, usize) -> Tweak,
    ) -> (Checkpoint, SlotBankLayout, Arc<AtomicBool>) {
        let mut entries: Vec<Entry> = Vec::new();
        for layer in 0..LAYERS {
            for expert in 0..EXPERTS {
                for (index, part) in ExpertPart::ALL.into_iter().enumerate() {
                    let prefix = expert_name_in(Root::Layers, layer, expert, part);
                    let seed = seed(layer, expert, index);
                    matrix_entries(&prefix, seed, tweak(layer, expert, index), &mut entries);
                }
            }
        }
        let mut header = serde_json::Map::new();
        let mut data = Vec::new();
        let mut weight_map = serde_json::Map::new();
        for (key, dtype, shape, payload) in &entries {
            let start = data.len();
            data.extend_from_slice(payload);
            header.insert(
                key.clone(),
                serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [start, data.len()]}),
            );
            weight_map.insert(key.clone(), serde_json::json!("m.safetensors"));
        }
        let mut json = serde_json::to_vec(&header).unwrap();
        while !json.len().is_multiple_of(8) {
            json.push(b' ');
        }
        let mut file = (json.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(&json);
        file.extend_from_slice(&data);
        let len = file.len() as u64;
        let index = serde_json::to_vec(&serde_json::json!({"weight_map": weight_map})).unwrap();
        let broken = Arc::new(AtomicBool::new(false));
        let shard = Flaky {
            inner: Cursor::new(file),
            broken: Arc::clone(&broken),
        };
        let source = StreamedCheckpoint::open(
            &index,
            vec![("m.safetensors".to_string(), Box::new(shard), len)],
            1 << 20,
        )
        .expect("the synthetic shard opens");
        let config = Config::from_json(
            br#"{"quantization_config":{"quant_method":"exl3","version":"1.4.9","codebook":"mul1"}}"#
                .as_slice(),
        )
        .unwrap();
        let checkpoint = Checkpoint {
            source,
            config,
            model_config: b"{}".to_vec(),
        };
        let layout = SlotBankLayout::new(WIDTH, WIDTH, [WORDS; 3]).unwrap();
        (checkpoint, layout, broken)
    }

    pub(super) fn source<'a>(
        checkpoint: &'a Checkpoint,
        layout: &SlotBankLayout,
    ) -> PackRecordSource<'a> {
        PackRecordSource::from_parts(checkpoint, layout.clone(), LAYERS, EXPERTS)
    }
}

use synthetic::{EXPERTS, LAYERS, Tweak};

fn normal(_: usize, _: usize, _: usize) -> Tweak {
    Tweak::Normal
}

fn text(error: EngError) -> String {
    error.describe()
}

#[test]
fn a_synthetic_pack_yields_the_record_of_each_expert_in_the_layout_order() {
    let (checkpoint, layout, _) = synthetic::checkpoint(&normal);
    let source = synthetic::source(&checkpoint, &layout);
    let mut seen = std::collections::BTreeSet::new();
    for layer in 0..LAYERS {
        for expert in 0..EXPERTS {
            let record = source.read_record(layer, expert).expect("a record");
            assert_eq!(record.len(), layout.record_bytes());
            // Each part's trellis is the stored bytes, so records of different experts differ and the
            // suh of part 0 is the widened F16 scale of that matrix.
            let part = layout.part(0);
            let suh = &record[part.suh.clone()];
            let first = f32::from_le_bytes(suh[..4].try_into().unwrap());
            let seed = synthetic::seed(layer, expert, 0);
            assert_eq!(
                first,
                half::f16::from_f32((seed % 97 + 32) as f32 / 64.0).to_f32()
            );
            assert!(
                seen.insert(record),
                "layer {layer} expert {expert} repeats a record"
            );
        }
    }
    assert_eq!(source.layout().record_bytes(), layout.record_bytes());
    let total = (LAYERS * EXPERTS * layout.record_bytes()) as u64;
    source
        .check_expert_bytes(total)
        .expect("the total of the walk");
    let error = text(source.check_expert_bytes(total + 1).unwrap_err());
    assert!(error.contains("header walk counts"), "{error}");
    assert!(source.check_expert_bytes(total - 1).is_err());
}

#[test]
fn a_wrong_rate_or_shape_is_refused_with_the_layer_and_the_expert() {
    let wrong = |layer: usize, expert: usize, part: usize, tweak: Tweak| {
        move |l: usize, e: usize, p: usize| {
            if (l, e, p) == (layer, expert, part) {
                tweak
            } else {
                Tweak::Normal
            }
        }
    };
    let (checkpoint, layout, _) = synthetic::checkpoint(&wrong(1, 2, 2, Tweak::Words(80)));
    let source = synthetic::source(&checkpoint, &layout);
    assert!(source.read_record(1, 1).is_ok(), "the neighbours are fine");
    let error = text(source.read_record(1, 2).unwrap_err());
    assert!(error.contains("layer 1 expert 2"), "{error}");
    assert!(error.contains("80 words"), "{error}");
    let (checkpoint, layout, _) = synthetic::checkpoint(&wrong(0, 1, 0, Tweak::Rows(256)));
    let source = synthetic::source(&checkpoint, &layout);
    let error = text(source.read_record(0, 1).unwrap_err());
    assert!(error.contains("layer 0 expert 1"), "{error}");
    assert!(error.contains("[256, 128]"), "{error}");
}

#[test]
fn a_matrix_with_a_bias_is_refused() {
    let tweak = |l: usize, e: usize, p: usize| {
        if (l, e, p) == (0, 2, 1) {
            Tweak::Bias
        } else {
            Tweak::Normal
        }
    };
    let (checkpoint, layout, _) = synthetic::checkpoint(&tweak);
    let source = synthetic::source(&checkpoint, &layout);
    assert!(source.read_record(0, 0).is_ok());
    let error = text(source.read_record(0, 2).unwrap_err());
    assert!(error.contains("layer 0 expert 2 Up has a bias"), "{error}");
}

#[test]
fn a_missing_tensor_is_refused_by_name() {
    let tweak = |l: usize, e: usize, p: usize| {
        if (l, e, p) == (1, 0, 0) {
            Tweak::NoSvh
        } else {
            Tweak::Normal
        }
    };
    let (checkpoint, layout, _) = synthetic::checkpoint(&tweak);
    let source = synthetic::source(&checkpoint, &layout);
    let error = text(source.read_record(1, 0).unwrap_err());
    let name = expert_name_in(Root::Layers, 1, 0, ExpertPart::Gate);
    assert!(error.contains("missing tensor"), "{error}");
    assert!(error.contains(&format!("{name}.svh")), "{error}");
    assert!(source.read_record(1, 1).is_ok());
}

#[test]
fn a_read_that_fails_after_the_open_is_refused_by_name() {
    let (checkpoint, layout, broken) = synthetic::checkpoint(&normal);
    let source = synthetic::source(&checkpoint, &layout);
    assert!(source.read_record(0, 0).is_ok());
    broken.store(true, std::sync::atomic::Ordering::Release);
    let error = text(source.read_record(0, 1).unwrap_err());
    let prefix = expert_name_in(Root::Layers, 0, 1, ExpertPart::Gate);
    assert!(error.contains(&prefix), "{error}");
    assert!(error.contains("the shard went away"), "{error}");
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
        let source = PackRecordSource::from_files(&files).expect("the record source");
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
        let policy = ReplaceOnMissLru::from_fill(geometry.clone(), &[row.as_slice()], &[counts])
            .expect("fill");
        let state = ExpertBankState::with_policy(geometry.clone(), HostMode::Exclusive, policy)
            .expect("state");
        let places = state.mode().places_per_layer(&geometry).expect("places");
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
