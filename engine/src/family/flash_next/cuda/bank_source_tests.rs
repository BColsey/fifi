//! Tests of the real-pack record source. The host test reads layer `FIFI_FLASH_NEXT_EXL3_MOE_LAYER`
//! (3 by default) of the pack in `FIFI_FLASH_NEXT_EXL3_PACK` and compares the records with the
//! bytes the shard headers locate; the device test builds a 16-slot bank over the same layer.

use std::{fs::File, os::unix::fs::FileExt, path::PathBuf};

use super::{
    bank::RecordSource,
    bank_source::{PackRecordSource, RungReadMode},
};
#[cfg(all(test, feature = "native-cuda", target_os = "linux"))]
use crate::weights::expert_store::ring::SpanRead;
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

pub(in crate::family::flash_next) mod synthetic {
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

    pub(in crate::family::flash_next) const LAYERS: usize = 2;
    pub(in crate::family::flash_next) const EXPERTS: usize = 3;
    pub(super) const WORDS: usize = 16;
    const WIDTH: usize = 128;

    /// What a matrix of the pack differs in from the normal one.
    #[derive(Clone, Copy, PartialEq, Eq)]
    pub(in crate::family::flash_next) enum Tweak {
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
    pub(in crate::family::flash_next) fn checkpoint(
        tweak: &dyn Fn(usize, usize, usize) -> Tweak,
    ) -> (Checkpoint, SlotBankLayout, Arc<AtomicBool>) {
        let (checkpoint, layout, broken, _image) = checkpoint_with_image(tweak);
        (checkpoint, layout, broken)
    }

    /// [`checkpoint`] plus the shard file's exact bytes, so a caller can
    /// mirror the pack onto disk for the O_DIRECT ring.
    pub(in crate::family::flash_next) fn checkpoint_with_image(
        tweak: &dyn Fn(usize, usize, usize) -> Tweak,
    ) -> (Checkpoint, SlotBankLayout, Arc<AtomicBool>, Vec<u8>) {
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
            inner: Cursor::new(file.clone()),
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
            // The real Flash-Next config: `PackRecordSource::new` parses it
            // for the layer and expert counts. Its counts only widen the
            // bounds; the tensors above answer every key the sources read.
            model_config: include_bytes!(
                "../../../../tests/fixtures/qwen3_8_flash_next_config.json"
            )
            .to_vec(),
        };
        let layout = SlotBankLayout::new(WIDTH, WIDTH, [WORDS; 3]).unwrap();
        (checkpoint, layout, broken, file)
    }

    pub(in crate::family::flash_next) fn source<'a>(
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

/// The decode miss counter driven exactly as production drives it: one
/// `observe` per layer per position (zero misses included — layer 0 of
/// the first step and the whole third layer of the second read nothing),
/// the next position closing the previous step, `line` closing the last.
/// Three layers, eight experts, pool depth four.
#[test]
fn the_decode_miss_counter_bins_the_production_call_pattern() {
    use super::bank_source::DecodeMisses;
    let misses = DecodeMisses::new(3, 8);
    // Step at pos 7: layers miss 0, 3, 3.
    misses.observe(7, 0, 0);
    misses.observe(7, 1, 3);
    misses.observe(7, 2, 3);
    // Step at pos 8: layers miss 2, 1, 0 — layer 2 serves from the bank.
    misses.observe(8, 0, 2);
    misses.observe(8, 1, 1);
    misses.observe(8, 2, 0);
    let line = misses.line(4).expect("the line");
    eprintln!("{line}");
    // Six layer observations: two 0s, one 1, one 2, two 3s — the lower
    // median of {0,0,1,2,3,3} is 1, and bin 0 carries the no-miss layers.
    assert!(
        line.contains("2 steps, misses per layer min 0 / median 1 / max 3"),
        "{line}"
    );
    // Step totals: 6 then 3.
    assert!(
        line.contains("per step min 3 / median 3 / max 6, pool depth 4"),
        "{line}"
    );
}

/// The same position run twice (a rollback, a retry) makes two steps:
/// the second run's layer 0 arrives at or before the first run's last
/// layer and closes the first run, so nothing merges.
#[test]
fn a_rerun_of_one_position_counts_as_two_steps() {
    use super::bank_source::DecodeMisses;
    let misses = DecodeMisses::new(3, 8);
    // First run of pos 4: layers 0, 1, 2 miss 1, 2, 3.
    misses.observe(4, 0, 1);
    misses.observe(4, 1, 2);
    misses.observe(4, 2, 3);
    // The rollback: pos 4 again, layers 0, 1, 2 miss 3, 0, 0.
    misses.observe(4, 0, 3);
    misses.observe(4, 1, 0);
    misses.observe(4, 2, 0);
    let line = misses.line(4).expect("the line");
    assert!(line.contains("2 steps"), "{line}");
    // Step totals 6 and 3, in their own bins.
    assert!(line.contains("per step min 3 / median 3 / max 6"), "{line}");
}

/// A step aborted by an error or a cancel drops whole: nothing it
/// observed enters the bins, and the next step starts clean.
#[test]
fn an_aborted_decode_step_drops_whole() {
    use super::bank_source::DecodeMisses;
    let misses = DecodeMisses::new(2, 8);
    misses.observe(1, 0, 4);
    // The step errors at layer 1; reset discards it.
    misses.discard_open();
    misses.observe(2, 0, 2);
    misses.observe(2, 1, 2);
    let line = misses.line(4).expect("the line");
    assert!(
        line.contains("1 step, misses per layer min 2 / median 2 / max 2"),
        "{line}"
    );
    assert!(line.contains("per step min 4 / median 4 / max 4"), "{line}");
}

/// A fresh rung has no line: decode never read through it.
#[test]
fn a_rung_decode_never_used_prints_no_line() {
    use super::bank_source::DecodeMisses;
    assert!(DecodeMisses::new(2, 8).line(4).is_none());
}

/// A cancelled turn keeps its fully observed last step: the cancel
/// arrived after the step's last layer, not during it, so the step
/// counts and the turn's line does not under-report.
#[test]
fn a_discarded_fully_observed_step_still_counts() {
    use super::bank_source::DecodeMisses;
    let misses = DecodeMisses::new(2, 8);
    misses.observe(1, 0, 4);
    misses.observe(1, 1, 1);
    // The turn is cancelled after the step's last layer: the step is
    // whole, so the discard keeps it.
    misses.discard_open();
    let line = misses.line(4).expect("the line");
    assert!(line.contains("1 step"), "{line}");
    // The kept step's layers missed 4 and 1.
    assert!(
        line.contains("misses per layer min 1 / median 1 / max 4"),
        "{line}"
    );
}

/// A device reset zeroes everything: a turn that ended without another
/// decode call — no line ever printed — leaves nothing for the next
/// turn's line.
#[test]
fn a_reset_zeroes_a_turn_that_never_printed() {
    use super::bank_source::DecodeMisses;
    let misses = DecodeMisses::new(2, 8);
    // One committed step (closed by the next position) plus one open.
    misses.observe(1, 0, 4);
    misses.observe(1, 1, 1);
    misses.observe(2, 0, 2);
    // The turn ends without a line; the reset clears bins and open step.
    misses.reset();
    assert!(misses.line(4).is_none());
    // The next turn reports only its own steps.
    misses.observe(3, 0, 1);
    misses.observe(3, 1, 1);
    let line = misses.line(4).expect("the line");
    assert!(line.contains("1 step"), "{line}");
    assert!(
        line.contains("misses per layer min 1 / median 1 / max 1"),
        "{line}"
    );
}

#[cfg(all(test, feature = "native-cuda", target_os = "linux"))]
mod device_tests {
    use super::*;
    use crate::family::expert_bank::ReplaceOnMissLru;
    use crate::{
        family::{
            expert_bank::{BankGeometry, ExpertBankState, HostMode},
            flash_next::cuda::bank::{
                HostMemory, HostPinPlan, HostTerms, device::DeviceBank, parse_route_counts,
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
        let policy = ReplaceOnMissLru::from_fill(geometry, &[row.as_slice()], &[counts], None)
            .expect("fill");
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
            None,
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

/// --- The rung source (the pooled O_DIRECT store) -------------------------
use super::bank_source::{RungSource, expert_spans, merge_abutting};
use crate::weights::exl3::Checkpoint;

fn opened_checkpoint() -> &'static Checkpoint {
    Box::leak(Box::new(
        Checkpoint::open(&pack_dir()).expect("the pack opens"),
    ))
}

#[test]
fn merge_abutting_coalesces_only_what_the_pack_wrote_contiguously() {
    let matrix =
        |shard: &str, file_offset: u64, len: u64, record_offset: u64| super::bank_source::Span {
            matrix: super::bank_source::MatrixSpan {
                shard: PathBuf::from(shard),
                file_offset,
                len,
                record_offset,
            },
            key: format!("t{record_offset}"),
            part: ExpertPart::ALL[0],
            dtype: Dtype::F32,
        };
    let spans = vec![
        matrix("a", 0, 4096, 0),
        // Same shard, abutting in file AND record order: one read.
        matrix("a", 4096, 4096, 4096),
        // A file gap splits the read even though the record tiles on.
        matrix("a", 16384, 4096, 8192),
        // A record gap splits the read even though the file tiles on.
        matrix("a", 20480, 4096, 20480),
        // Another shard is always a new read.
        matrix("b", 0, 4096, 24576),
    ];
    let shape = |read: &crate::weights::expert_store::ring::SpanRead| {
        (
            read.shard.display().to_string(),
            read.file_offset,
            read.len,
            read.record_offset,
        )
    };
    assert_eq!(
        merge_abutting(&spans).iter().map(shape).collect::<Vec<_>>(),
        vec![
            ("a".into(), 0, 8192, 0),
            ("a".into(), 16384, 4096, 8192),
            ("a".into(), 20480, 4096, 20480),
            ("b".into(), 0, 4096, 24576),
        ]
    );
}

#[test]
#[ignore = "reads the real pack in FIFI_FLASH_NEXT_EXL3_PACK through the pooled O_DIRECT store"]
fn the_rung_records_match_the_buffered_records_byte_for_byte() {
    let dir = pack_dir();
    let checkpoint = opened_checkpoint();
    let buffered = PackRecordSource::new(checkpoint).expect("the buffered source derives");
    let record_bytes = buffered.layout().record_bytes();
    let rung = || {
        let bytes = RungSource::backing_bytes(checkpoint, &dir, 2).expect("the backing sizes");
        let backing: &'static mut [u8] = Box::leak(vec![0u8; bytes].into_boxed_slice());
        let mode = RungSource::probe_read_mode(checkpoint, &dir).expect("the mode probes");
        RungSource::new(checkpoint, &dir, 2, mode, backing)
    };
    let rung = match rung() {
        Ok(rung) => rung,
        // A broken pack still refuses; a filesystem that refuses O_DIRECT
        // no longer does — the rung serves buffered there.
        Err(error) => {
            eprintln!("rung test: the store refuses this pack: {error:?}");
            return;
        }
    };
    for layer in [0usize, 3, 47] {
        for expert in [0usize, 1, 255, 256, 510, 511] {
            let want = buffered
                .read_record(layer, expert)
                .expect("the buffered record");
            let got = &rung
                .read_packed(layer, &[expert as u16])
                .expect("the rung record")[0];
            assert_eq!(got.len(), record_bytes);
            assert_eq!(got, &want, "layer {layer} expert {expert}");
        }
    }
    // One batch of mixed experts in one call: the step's miss shape.
    let misses: Vec<u16> = [0u16, 5, 17, 300, 511].to_vec();
    let batch = rung.read_packed(3, &misses).expect("the batch");
    for (expert, got) in misses.iter().zip(&batch) {
        assert_eq!(
            got,
            &buffered
                .read_record(3, usize::from(*expert))
                .expect("the buffered record"),
            "batch expert {expert}"
        );
    }
}

/// The bench's step: two seeded-random experts per layer, truncated to
/// 78 sites — the shape of one decode step's misses.
fn step_picks(layers: usize, experts: usize) -> Vec<Vec<u16>> {
    let mut seed = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let mut picks = Vec::with_capacity(layers);
    for _ in 0..layers {
        let mut layer_picks = Vec::new();
        for _ in 0..2 {
            layer_picks.push((next() % experts as u64) as u16);
        }
        picks.push(layer_picks);
    }
    picks.truncate(39);
    picks
}

/// Writes a sparse skeleton of the pack into `FIFI_SPAN_SINK` (a fresh
/// directory): every small file copied whole, and every shard a SPARSE
/// file carrying only its header windows and the bench picks' merged
/// extents at their true offsets. The parity test reads the real pack
/// (content); the bench reads the skeleton on a direct-capable
/// filesystem, where its reads are the production pattern, cold by
/// construction, without a 100 GiB copy.
#[test]
#[ignore = "writes the sparse bench skeleton of FIFI_FLASH_NEXT_EXL3_PACK into FIFI_SPAN_SINK"]
fn write_the_sparse_pack_skeleton() {
    use std::{collections::BTreeMap, os::unix::fs::FileExt};
    let dir = pack_dir();
    let sink = PathBuf::from(
        std::env::var_os("FIFI_SPAN_SINK").expect("FIFI_SPAN_SINK names the sink directory"),
    );
    std::fs::create_dir_all(&sink).expect("the sink directory");
    let checkpoint = opened_checkpoint();
    let buffered = PackRecordSource::new(checkpoint).expect("the buffered source derives");
    let picks = step_picks(buffered.layers(), buffered.experts());
    let mut extents: BTreeMap<std::ffi::OsString, Vec<(u64, u64)>> = BTreeMap::new();
    // The layout derivation reads layer 0's first expert; the skeleton
    // carries that record too.
    for (layer, expert) in
        std::iter::once((0usize, 0usize)).chain(picks.iter().enumerate().flat_map(
            |(layer, layer_picks)| layer_picks.iter().map(move |&e| (layer, usize::from(e))),
        ))
    {
        let (spans, _) = expert_spans(checkpoint, &dir, layer, expert).expect("the extents");
        for read in merge_abutting(&spans) {
            extents
                .entry(read.shard.file_name().expect("a file name").to_os_string())
                .or_default()
                .push((read.file_offset, read.len));
        }
    }
    for entry in std::fs::read_dir(&dir).expect("the pack directory") {
        let entry = entry.expect("an entry");
        let meta = entry.metadata().expect("the entry metadata");
        if !meta.is_file() {
            continue;
        }
        let name = entry.file_name();
        let source = entry.path();
        let target = sink.join(&name);
        if !name.to_string_lossy().ends_with(".safetensors") {
            std::fs::copy(&source, &target).expect("copy a small file");
            continue;
        }
        let file_len = meta.len();
        let input = std::fs::File::open(&source).expect("open the shard");
        let out = std::fs::File::create(&target).expect("create the shard skeleton");
        out.set_len(file_len).expect("the shard length");
        // Header windows (whichever end the format keeps) plus the picks'
        // extents, at their true offsets: sparse elsewhere.
        let mut ranges: Vec<(u64, u64)> = vec![(0, 16 << 20)];
        ranges.push((file_len.saturating_sub(16 << 20), 16 << 20));
        ranges.extend(
            extents
                .get(&name)
                .map_or([].as_slice(), |list| list.as_slice())
                .iter()
                .copied(),
        );
        let mut copied = 0u64;
        for (offset, len) in ranges {
            // A window can run past a small file's end; carry what is there.
            let len = len.min(file_len.saturating_sub(offset));
            let mut window = vec![0u8; len as usize];
            input
                .read_exact_at(&mut window, offset)
                .expect("a source window");
            out.write_all_at(&window, offset).expect("place the window");
            copied += len;
        }
        eprintln!(
            "skeleton: {} — {file_len} bytes long, {copied} real",
            name.to_string_lossy()
        );
    }
}

/// A test that runs in the battery and exercises
/// RungSource's own read path — the synthetic pack (real extents, real
/// bytes, no holes) mirrored onto disk, read through the rung's pooled
/// span store, compared with the buffered source record for record,
/// in BOTH shard-open modes. The probed mode is checked against an
/// independent fact: a raw O_DIRECT open of the mirrored shard made
/// here in the test, not the probe under test. The real pack needs an
/// env var and the container; this one runs wherever the battery runs.
#[test]
fn the_rung_reads_the_synthetic_pack_like_the_buffered_source() {
    let (checkpoint, _layout, _broken, image) =
        synthetic::checkpoint_with_image(&|_, _, _| synthetic::Tweak::Normal);
    let dir = tempfile::tempdir().expect("tempdir");
    let shard = dir.path().join("m.safetensors");
    std::fs::write(&shard, &image).expect("the shard mirrors");
    let checkpoint: &'static crate::weights::exl3::Checkpoint = Box::leak(Box::new(checkpoint));
    let buffered = PackRecordSource::new(checkpoint).expect("the buffered source derives");
    let slots = 2;
    let bytes = RungSource::backing_bytes(checkpoint, dir.path(), slots).expect("the backing size");
    let probed = RungSource::probe_read_mode(checkpoint, dir.path()).expect("the mode probes");
    // The independent fact this test owns: its own raw O_DIRECT open of
    // the same shard answers Direct exactly when the probe says so.
    let answers_direct = crate::family::expert_bank::host_tier::open_direct(&shard).is_ok();
    let probe_says_direct = matches!(probed, RungReadMode::Direct { .. });
    assert_eq!(
        probe_says_direct, answers_direct,
        "the probe's answer matches a raw O_DIRECT open of the same shard"
    );
    // Parity must hold in either mode: the probed one, and a forced
    // buffered leg — the mode a refusing filesystem lands in.
    let modes = [
        probed,
        RungReadMode::Buffered {
            refused: "the forced parity leg".into(),
            probes: 1,
        },
    ];
    for mode in modes {
        let backing: &'static mut [u8] = Box::leak(vec![0u8; bytes].into_boxed_slice());
        let rung = RungSource::new(checkpoint, dir.path(), slots, mode.clone(), backing)
            .expect("the rung builds");
        assert_eq!(
            rung.read_mode(),
            &mode,
            "the rung states its start-decided mode"
        );
        for layer in 0..synthetic::LAYERS {
            let misses: Vec<u16> = (0..synthetic::EXPERTS as u16).collect();
            let records = rung.read_packed(layer, &misses).expect("the batch reads");
            for (expert, record) in misses.iter().zip(&records) {
                let mut want = vec![0u8; record.len()];
                buffered
                    .record(layer, *expert, &mut want)
                    .expect("the buffered record");
                assert_eq!(record, &want, "layer {layer} expert {expert}");
            }
        }
    }
}

/// The probe's policy on an already-collected shard set: every shard is
/// examined before the mode is named, and a missing shard refuses the
/// start whatever else refused — here an unreadable shard (any error
/// that is not NotFound counts as a refusal) sits next to a missing
/// one, and the missing one wins.
#[test]
fn a_missing_shard_refuses_even_when_another_shard_refuses() {
    let dir = tempfile::tempdir().expect("tempdir");
    let refusing = dir.path().join("a.safetensors");
    std::fs::write(&refusing, b"shard").expect("write the shard");
    std::fs::set_permissions(
        &refusing,
        std::os::unix::fs::PermissionsExt::from_mode(0o000),
    )
    .expect("make the shard unreadable");
    let missing = dir.path().join("b.safetensors");
    let error = RungSource::probe_shards(&[refusing, missing])
        .expect_err("the missing shard refuses the start");
    assert!(
        format!("{error:?}").contains("missing"),
        "the refusal names the missing shard: {error:?}"
    );
}

/// A multi-span pack opens one file per DISTINCT shard at the probe,
/// never one per span: the synthetic pack's two layers of three experts
/// (many spans) name one shard, and the probe reports exactly that one
/// open; a hand-made set of three distinct shards reports three.
#[test]
fn the_probe_opens_one_file_per_distinct_shard() {
    let (checkpoint, _layout, _broken, image) =
        synthetic::checkpoint_with_image(&|_, _, _| synthetic::Tweak::Normal);
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::write(dir.path().join("m.safetensors"), &image).expect("the shard mirrors");
    let checkpoint: &'static crate::weights::exl3::Checkpoint = Box::leak(Box::new(checkpoint));
    let one = RungSource::probe_read_mode(checkpoint, dir.path()).expect("the mode probes");
    let other = tempfile::tempdir().expect("tempdir");
    let shards: Vec<std::path::PathBuf> = ["a", "b", "c"]
        .iter()
        .map(|name| other.path().join(format!("{name}.safetensors")))
        .collect();
    for shard in &shards {
        std::fs::write(shard, b"shard").expect("write the shard");
    }
    let three = RungSource::probe_shards(&shards).expect("three shards answer");
    let count = |mode: &RungReadMode| match mode {
        RungReadMode::Direct { probes, .. } | RungReadMode::Buffered { probes, .. } => *probes,
    };
    assert_eq!(count(&one), 1, "many spans, one shard: one open");
    assert_eq!(count(&three), 3, "three distinct shards: three opens");
}

/// The rung's read pattern measured over the pack's REAL extent lists:
/// one decode step = every layer's miss list read as one `read_packed`
/// batch (the exact call `run_copies` makes), cold by construction — the
/// store reads O_DIRECT, which on a direct-capable filesystem bypasses
/// the page cache. Depths 1 (one record at a time — what main served
/// before), 4 and 8 (the pool), plus the per-miss read shape after
/// coalescing. This is the bench the tier line's figure and
/// `RUNG_RING_SLOTS` cite. Native-view only: cold cells evict through the
/// tree's own fadvise, which lives behind the CUDA backend.
#[cfg(all(test, feature = "native-cuda", target_os = "linux"))]
#[test]
#[ignore = "bench: reads the real pack's real extent lists cold at depths 1/4/8"]
fn span_bench_depths_over_the_real_packs_extent_lists() {
    let dir = pack_dir();
    let checkpoint = opened_checkpoint();
    // The probe's own cost, measured on the same real pack before any
    // cell: distinct shards only, so a big pack's span count cannot
    // multiply it.
    let probe_at = std::time::Instant::now();
    let probed = RungSource::probe_read_mode(checkpoint, &dir).expect("the mode probes");
    let probe_elapsed = probe_at.elapsed();
    let probe_count = match &probed {
        RungReadMode::Direct { probes, .. } | RungReadMode::Buffered { probes, .. } => *probes,
    };
    eprintln!(
        "probe: {:?} in {probe_elapsed:?} ({probe_count} distinct shards)",
        probed
    );
    let buffered = PackRecordSource::new(checkpoint).expect("the buffered source derives");
    let (layers, experts) = (buffered.layers(), buffered.experts());
    let picks = step_picks(layers, experts);
    let mut raw_spans = Vec::new();
    let mut merged_spans = Vec::new();
    let mut merged_bytes = Vec::new();
    // The merged reads of every site the step reads, for the per-cell evict.
    let mut site_merged: Vec<Vec<SpanRead>> = Vec::new();
    for (layer, layer_picks) in picks.iter().enumerate() {
        for &expert in layer_picks {
            let (spans, raw) =
                expert_spans(checkpoint, &dir, layer, usize::from(expert)).expect("the extents");
            raw_spans.push(spans.len() as u64);
            let merged = merge_abutting(&spans);
            merged_bytes.push(merged.iter().map(|read| read.len).sum::<u64>());
            merged_spans.push(merged.len() as u64);
            assert_eq!(raw, merged_bytes.last().copied().unwrap_or(0));
            site_merged.push(merged);
        }
    }
    let median = |mut values: Vec<u64>| {
        values.sort_unstable();
        values[values.len() / 2]
    };
    eprintln!(
        "span bench: per miss, components {}..={} (median {}), reads after coalescing {}..={} \
         (median {}), bytes {}..={} (median {})",
        raw_spans.iter().copied().min().unwrap(),
        raw_spans.iter().copied().max().unwrap(),
        median(raw_spans.clone()),
        merged_spans.iter().copied().min().unwrap(),
        merged_spans.iter().copied().max().unwrap(),
        median(merged_spans.clone()),
        merged_bytes.iter().copied().min().unwrap(),
        merged_bytes.iter().copied().max().unwrap(),
        median(merged_bytes.clone()),
    );
    eprintln!(
        "span bench: coalescing turns {} component reads into {} reads across the step",
        raw_spans.iter().copied().sum::<u64>(),
        merged_spans.iter().copied().sum::<u64>(),
    );
    // Cache eviction for cold cells, through the tree's fadvise helper
    // (unsafe is forbidden crate-wide). On a copy-on-write filesystem
    // (btrfs) O_DIRECT is served through the page cache, so "cold" means
    // DONTNEED-evicted ranges re-read through the cache — not true DIO. On
    // a NOCOW/ext4 file the same evict gives the same cold state.
    let evict = |site_merged: &[Vec<SpanRead>]| {
        use std::collections::BTreeMap;
        let mut by_shard: BTreeMap<&std::path::Path, Vec<(u64, u64)>> = BTreeMap::new();
        for reads in site_merged {
            for read in reads {
                by_shard
                    .entry(read.shard.as_path())
                    .or_default()
                    .push((read.file_offset, read.len));
            }
        }
        for (shard, ranges) in by_shard {
            let file = File::open(shard).expect("the shard opens read-only");
            for (offset, len) in ranges {
                crate::kernels::cuda::drop_page_cache(&file, offset, len).unwrap_or_else(|error| {
                    panic!("the evict of {shard:?} at {offset}: {error:?}")
                });
            }
        }
    };
    eprintln!(
        "span bench: cells are DONTNEED-evicted then re-read cold; pack at {:?} is on btrfs, \
         so O_DIRECT rides the page cache (CoW): these are cold page-cache reads, not true DIO",
        dir
    );
    for depth in [1usize, 4, 8] {
        let bytes = RungSource::backing_bytes(checkpoint, &dir, depth).expect("the backing sizes");
        let backing: &'static mut [u8] = Box::leak(vec![0u8; bytes].into_boxed_slice());
        let mode = RungSource::probe_read_mode(checkpoint, &dir).expect("the mode probes");
        let rung = match RungSource::new(checkpoint, &dir, depth, mode, backing) {
            Ok(rung) => rung,
            Err(error) => {
                eprintln!("span bench: depth {depth} refuses this pack: {error:?}");
                return;
            }
        };
        // Phase split via the bench-only method: reads on the pool's worker
        // threads inside read_records, parse (F16 widening) and pack on this
        // caller thread after each chunk's leases return.
        let mut reads_ms = Vec::new();
        let mut parse_ms = Vec::new();
        let mut pack_ms = Vec::new();
        let mut total_ms = Vec::new();
        for _ in 0..5 {
            evict(&site_merged);
            let start = std::time::Instant::now();
            let (in_reads, in_parse, in_pack) = rung.phase_costs(&picks).expect("the phases");
            total_ms.push(start.elapsed().as_secs_f64() * 1000.0);
            reads_ms.push(in_reads);
            parse_ms.push(in_parse);
            pack_ms.push(in_pack);
        }
        let median = |values: &[f64]| {
            let mut sorted = values.to_vec();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            sorted[sorted.len() / 2]
        };
        eprintln!(
            "span bench: depth {depth}: per step reads {:.1} / parse {:.1} / pack {:.1} / \
             total {:.1} ms (median of 5; reads on the pool's threads, parse+pack on the caller)",
            median(&reads_ms),
            median(&parse_ms),
            median(&pack_ms),
            median(&total_ms),
        );
    }
}
