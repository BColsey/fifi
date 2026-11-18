//! The ring source against the buffered source: both answer the bank's
//! `RecordSource` contract from the same opened pack, so the record every
//! expert yields must be identical byte-for-byte. The corpus pack arrives
//! through `FIFI_FLASH_NEXT_EXL3_PACK` (the same env the buffered corpus
//! tests read); where its filesystem refuses O_DIRECT the ring source
//! refuses by contract, and the test reports the filesystem and skips.

use super::bank::RecordSource;
use super::bank_source::PackRecordSource;
use super::pack_reader::MatrixSpan;
use super::ring_source::{
    HostCopyFence, Placement, RingRecordSource, Span, UploadFence, parse_spans,
};
use crate::EngError;
use crate::tensor::Dtype;
use crate::weights::exl3::Checkpoint;
use std::num::NonZeroUsize;
use std::path::PathBuf;

fn pack_dir() -> PathBuf {
    PathBuf::from(std::env::var_os("FIFI_FLASH_NEXT_EXL3_PACK").expect("FIFI_FLASH_NEXT_EXL3_PACK"))
}

/// The ring source over a freshly leaked one-slot backing, sized by the
/// source itself.
fn ring_source<'a>(checkpoint: &'a Checkpoint, dir: &std::path::Path) -> RingRecordSource<'a> {
    let slots = NonZeroUsize::new(1).unwrap();
    let bytes = RingRecordSource::backing_bytes(checkpoint, dir, slots).expect("the backing sizes");
    let backing: &'static mut [u8] = Box::leak(vec![0u8; bytes].into_boxed_slice());
    RingRecordSource::new(checkpoint, dir, slots, backing).expect("the ring source derives")
}

fn filesystem_of(dir: &std::path::Path) -> String {
    std::fs::canonicalize(dir)
        .ok()
        .and_then(|canonical| canonical.parent().map(|parent| parent.to_path_buf()))
        .map_or_else(
            || "an uncanonicalizable path".into(),
            |parent| {
                std::fs::metadata(parent)
                    .map(|meta| format!("{:?}", meta.file_type()))
                    .unwrap_or_else(|error| format!("unreadable ({error})"))
            },
        )
}

#[test]
#[ignore = "reads every expert of the real pack in FIFI_FLASH_NEXT_EXL3_PACK through the O_DIRECT ring"]
fn the_ring_records_match_the_buffered_records_byte_for_byte() {
    let dir = pack_dir();
    eprintln!(
        "ring-source test: {} on {}",
        dir.display(),
        filesystem_of(&dir)
    );
    let checkpoint = Checkpoint::open(&dir).expect("the pack opens");
    let buffered = PackRecordSource::new(&checkpoint).expect("the buffered source derives");
    let slots = NonZeroUsize::new(1).unwrap();
    let bytes =
        RingRecordSource::backing_bytes(&checkpoint, &dir, slots).expect("the backing sizes");
    let backing: &'static mut [u8] = Box::leak(vec![0u8; bytes].into_boxed_slice());
    let ring = match RingRecordSource::new(&checkpoint, &dir, slots, backing) {
        Ok(ring) => ring,
        // A filesystem that refuses O_DIRECT refuses the ring by contract;
        // the parity claim is only testable where direct reads answer.
        Err(error) => {
            eprintln!("ring-source test: the ring refuses this pack: {error:?}");
            return;
        }
    };
    let record_bytes = buffered.layout().record_bytes();
    let (layers, experts) = (buffered.layers(), buffered.experts());
    assert_eq!(
        (ring.counts().0, ring.counts().1),
        (layers, experts),
        "both sources derive the same shape"
    );
    for layer in 0..layers {
        for expert in 0..experts {
            let mut want = vec![0u8; record_bytes];
            let mut got = vec![0u8; record_bytes];
            buffered
                .record(layer, expert as u16, &mut want)
                .unwrap_or_else(|error| panic!("buffered {layer}/{expert}: {error:?}"));
            ring.record(layer, expert as u16, &mut got)
                .unwrap_or_else(|error| panic!("ring {layer}/{expert}: {error:?}"));
            assert_eq!(want, got, "layer {layer} expert {expert} differs");
        }
    }
}

#[test]
#[ignore = "needs the real pack in FIFI_FLASH_NEXT_EXL3_PACK"]
fn an_out_of_range_expert_is_refused_by_both_sources() {
    let dir = pack_dir();
    let checkpoint = Checkpoint::open(&dir).expect("the pack opens");
    let buffered = PackRecordSource::new(&checkpoint).expect("the buffered source derives");
    let ring = ring_source(&checkpoint, &dir);
    let mut sink = [0u8; 1];
    assert!(
        ring.record(buffered.layers(), 0, &mut sink).is_err(),
        "a layer past the end is refused"
    );
    assert!(
        ring.record(0, buffered.experts() as u16, &mut sink)
            .is_err(),
        "an expert past the end is refused"
    );
}

/// A missing suh/svh span must refuse the parse, not fall through to a
/// half-built expert: the spans of a real expert, minus its suh component.
#[test]
#[ignore = "needs the real pack in FIFI_FLASH_NEXT_EXL3_PACK"]
fn a_missing_suh_span_refuses_the_parse() {
    let dir = pack_dir();
    let checkpoint = Checkpoint::open(&dir).expect("the pack opens");
    let (spans, raw_bytes) = super::ring_source::expert_spans(&checkpoint, &dir, 0, 0)
        .expect("layer 0 expert 0 derives");
    // The window parse_spans expects: each span's bytes staged at its
    // cursor offset — a plain buffered read gives the same bytes.
    let mut window = vec![0u8; raw_bytes as usize];
    let mut cursor = 0usize;
    for span in &spans {
        let file = std::fs::File::open(&span.matrix.shard).expect("the shard opens");
        use std::os::unix::fs::FileExt;
        file.read_exact_at(
            &mut window[cursor..cursor + span.matrix.len as usize],
            span.matrix.file_offset,
        )
        .expect("the buffered read");
        cursor += span.matrix.len as usize;
    }
    parse_spans(&window, &spans, &checkpoint, 0, 0)
        .expect("the control parse with every span succeeds");
    let without: Vec<Span> = spans
        .iter()
        .filter(|span| !span.key.contains("suh") && !span.key.contains("svh"))
        .cloned()
        .collect();
    assert!(
        without.len() < spans.len(),
        "the fixture must actually drop a suh/svh span"
    );
    let error = parse_spans(&window, &without, &checkpoint, 0, 0)
        .expect_err("a missing suh/svh span must refuse the parse");
    assert!(
        format!("{error:?}").to_lowercase().contains("suh")
            || format!("{error:?}").to_lowercase().contains("svh"),
        "the refusal names the missing component: {error:?}"
    );
}

/// Reader item (d): not synthetic lengths — the real span whose file
/// offset lands worst into its sector window, read through the O_DIRECT
/// ring, byte for byte against a plain buffered read.
#[test]
#[ignore = "reads one real expert of the pack in FIFI_FLASH_NEXT_EXL3_PACK"]
fn a_real_span_at_the_worst_slot_offset_reads_byte_for_byte() {
    use crate::family::expert_bank::host_tier::{ODirectRing, SECTOR};
    use crate::family::flash_next::cuda::pack_reader::{PackReader as RingReader, slot_span_bytes};
    use std::num::NonZeroU64;
    use std::sync::Arc;

    let dir = pack_dir();
    let checkpoint = Checkpoint::open(&dir).expect("the pack opens");
    let (spans, raw_bytes) = super::ring_source::expert_spans(&checkpoint, &dir, 0, 0)
        .expect("layer 0 expert 0 derives");
    let (worst_index, worst) = spans
        .iter()
        .enumerate()
        .max_by_key(|(_, span)| span.matrix.file_offset % SECTOR)
        .expect("the expert has spans");
    let misalignment = worst.matrix.file_offset % SECTOR;
    let matrix_spans: Vec<_> = spans.iter().map(|span| span.matrix.clone()).collect();
    let span_total = slot_span_bytes(raw_bytes, &matrix_spans).expect("the slot sizes");
    let leaked: &'static mut [u8] =
        Box::leak(vec![0u8; span_total as usize + SECTOR as usize].into_boxed_slice());
    let base = leaked.as_ptr() as usize;
    let skip = (SECTOR as usize - (base % SECTOR as usize)) % SECTOR as usize;
    let mem = &mut leaked[skip..skip + span_total as usize];
    let mut ring = ODirectRing::new(
        NonZeroUsize::new(1).unwrap(),
        NonZeroU64::new(span_total).unwrap(),
        mem,
    )
    .expect("the ring wraps the aligned window");
    let slot = ring.acquire().expect("the single slot is free");
    let mut reader = RingReader::new();
    reader
        .read_expert(&mut ring, &matrix_spans, raw_bytes, slot)
        .expect("the O_DIRECT read of the real expert");
    ring.begin_copy(slot, Arc::new(HostCopyFence))
        .expect("the copy arm takes");
    let window = ring
        .slot_window(slot, 0, raw_bytes)
        .expect("the read-only window");
    let start: usize = spans[..worst_index]
        .iter()
        .map(|span| span.matrix.len as usize)
        .sum();
    let len = worst.matrix.len as usize;
    let staged = &window[start..start + len];
    let mut plain = vec![0u8; len];
    let file = std::fs::File::open(&worst.matrix.shard).expect("the shard opens");
    use std::os::unix::fs::FileExt;
    file.read_exact_at(&mut plain, worst.matrix.file_offset)
        .expect("the buffered read of the same span");
    assert_eq!(
        staged, plain,
        "the span at file offset {} ({} bytes into its sector window) differs",
        worst.matrix.file_offset, misalignment
    );
    eprintln!(
        "worst-offset span: {} bytes at file offset {} = {} into the sector window",
        len, worst.matrix.file_offset, misalignment
    );
}

/// The upload lane's safety pin: the slot's fence stays incomplete until
/// the fill thread's own event query marks it, so a release from any other
/// thread cannot free memory a DMA may still be reading.
#[test]
fn a_slot_whose_fence_is_incomplete_refuses_release() {
    use crate::family::expert_bank::host_tier::{ODirectRing, SECTOR};
    use std::num::NonZeroU64;
    use std::sync::Arc;

    let (slots, span) = (
        NonZeroUsize::new(1).unwrap(),
        NonZeroU64::new(SECTOR).unwrap(),
    );
    let leaked: &'static mut [u8] = Box::leak(vec![0u8; SECTOR as usize * 2].into_boxed_slice());
    let base = leaked.as_ptr() as usize;
    let off = (SECTOR as usize - (base % SECTOR as usize)) % SECTOR as usize;
    let mem = &mut leaked[off..off + SECTOR as usize];
    let mut ring = ODirectRing::new(slots, span, mem).expect("the ring wraps the aligned window");
    let slot = ring.acquire().expect("the single slot is free");
    let fence = Arc::new(UploadFence::new());
    ring.begin_copy(slot, fence.clone())
        .expect("the copy arm takes");
    assert!(
        ring.release(slot).is_err(),
        "an incomplete fence (no fill-thread query) must pin the slot"
    );
    fence.mark_complete();
    ring.release(slot)
        .expect("release follows the marked fence");
    assert!(
        ring.acquire().is_some(),
        "the released slot returns to the free pool"
    );
}

/// The direct lane's parity, on the card: upload one expert's record via
/// `upload` (asserting the lane ENGAGED — a silent fallback fails here,
/// which is what let the F16-length bug hide), read the device bytes
/// back, and compare with the buffered record.
/// The real pack carries a fourth span per part (`mul1`) that the row
/// drops; placements must pair to spans by index, never by zip position,
/// and their ranges must tile the row exactly. This is the regression
/// test for the out-of-bounds upload the first direct-lane run wrote
/// (1228800 bytes at offset 1241600 of a 1881600-byte buffer).
#[test]
fn span_placements_pair_skipped_spans_by_index() {
    use super::ring_source::{Span, span_placements};
    use crate::family::exl3_pack::ExpertPart;
    use crate::family::flash_next::cuda::pack_reader::MatrixSpan;
    use crate::family::flash_next::moe::slot_bank::SlotBankLayout;
    use crate::tensor::Dtype;

    let layout = SlotBankLayout::new(2560, 640, [48, 48, 48]).unwrap();
    let shapes = [[2560usize, 640], [2560, 640], [640, 2560]];
    let mut spans = Vec::new();
    let mut cursor = 0u64;
    for (part, [k, n]) in ExpertPart::ALL.into_iter().zip(shapes) {
        for (suffix, dtype, bytes) in [
            ("trellis", Dtype::I16, (k / 16) * (n / 16) * 48 * 2),
            ("suh", Dtype::F16, k * 2),
            ("svh", Dtype::F16, n * 2),
            ("mul1", Dtype::I32, 4),
        ] {
            spans.push(Span {
                matrix: MatrixSpan {
                    shard: std::path::PathBuf::new(),
                    file_offset: 0,
                    len: bytes as u64,
                    record_offset: cursor,
                },
                key: format!("model.test.experts.0.part.{suffix}"),
                part,
                dtype,
            });
            cursor += bytes as u64;
        }
    }
    let placements = span_placements(&spans, &layout)
        .unwrap()
        .expect("every span of this pack places");
    assert_eq!(placements.len(), 9, "mul1 places nothing");
    let mut at = 0;
    for placement in &placements {
        assert_eq!(placement.range.start, at, "ranges tile the row in order");
        at = placement.range.end;
    }
    assert_eq!(at, layout.record_bytes());
    let named: Vec<usize> = placements.iter().map(|p| p.span).collect();
    assert_eq!(named, vec![0, 1, 2, 4, 5, 6, 8, 9, 10]);
    for placement in &placements {
        let suffix = spans[placement.span].key.rsplit('.').next().unwrap();
        assert_ne!(suffix, "mul1");
    }
}

/// N3 regression, on the card: layer 0's last upload parks the only
/// slot; layer 1's first record() must wait it out (not refuse), and the
/// next upload must still engage. Every answer is checked against the
/// buffered source.
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
#[test]
#[ignore = "needs a GPU (under 1 GiB) and FIFI_FLASH_NEXT_EXL3_PACK"]
fn record_waits_out_a_parked_upload_before_refusing() {
    use crate::family::project::cuda::CudaProjector;
    use crate::kernels::cuda::device_present;

    assert!(
        device_present("Flash-Next upload/record interleave"),
        "a GPU is required"
    );
    let dir = pack_dir();
    let checkpoint = Checkpoint::open(&dir).expect("the pack opens");
    let buffered = PackRecordSource::new(&checkpoint).expect("the buffered source derives");
    let ring = ring_source(&checkpoint, &dir);
    let record = buffered.layout().record_bytes();
    let projector = CudaProjector::new().expect("projector");
    let ctx = projector.context();
    let stream = ctx.stream().expect("stream");
    let dst = ctx.allocate(record).expect("the device buffer");

    let mut want = vec![0u8; record];
    let read = || {
        let mut out = vec![0u8; record];
        ring.record(1, 0, &mut out)
            .expect("record after a parked upload");
        out
    };

    // upload (layer 0) parks the slot; record (layer 1) must drain it.
    assert!(
        ring.upload(0, 0, ctx, &stream, &dst, 0)
            .expect("the first upload"),
        "the direct lane must engage"
    );
    assert_eq!(read(), {
        buffered
            .record(1, 0, &mut want)
            .expect("the buffered record");
        want.clone()
    });
    // and the slot serves an upload again.
    ring.drain(&stream).expect("drain");
    assert!(
        ring.upload(1, 1, ctx, &stream, &dst, 0)
            .expect("the second upload"),
        "the direct lane must engage again"
    );
    ring.drain(&stream).expect("drain");
    let mut got = vec![0u8; record];
    dst.read_at(&stream, 0, &mut got).expect("read back");
    buffered
        .record(1, 1, &mut want)
        .expect("the second buffered record");
    assert_eq!(want, got, "the second upload's bytes differ");
}

#[cfg(all(feature = "native-cuda", target_os = "linux"))]
#[test]
#[ignore = "needs a GPU (under 1 GiB) and FIFI_FLASH_NEXT_EXL3_PACK"]
fn a_direct_upload_lands_the_buffered_record_bytes_on_the_device() {
    use crate::family::project::cuda::CudaProjector;
    use crate::kernels::cuda::device_present;

    assert!(
        device_present("Flash-Next direct-upload parity"),
        "a GPU is required"
    );
    let dir = pack_dir();
    let checkpoint = Checkpoint::open(&dir).expect("the pack opens");
    let buffered = PackRecordSource::new(&checkpoint).expect("the buffered source derives");
    let ring = ring_source(&checkpoint, &dir);
    let record = buffered.layout().record_bytes();
    let projector = CudaProjector::new().expect("projector");
    let ctx = projector.context();
    let stream = ctx.stream().expect("stream");
    let dst = ctx.allocate(record).expect("the device buffer");
    let engaged = ring
        .upload(0, 0, ctx, &stream, &dst, 0)
        .expect("the upload call");
    assert!(
        engaged,
        "the direct lane must engage for this pack — a silent fallback is a failure"
    );
    ring.drain(&stream).expect("drain");
    let mut got = vec![0u8; record];
    dst.read_at(&stream, 0, &mut got).expect("read back");
    let mut want = vec![0u8; record];
    buffered
        .record(0, 0, &mut want)
        .expect("the buffered record");
    assert_eq!(
        want, got,
        "the uploaded bytes differ from the buffered record"
    );
}

/// Reader item (2): the REAL pack's span table, not a synthetic one —
/// every placement fits inside the record row, and the placements tile
/// the row exactly, for every expert of every layer.
#[test]
#[ignore = "needs the real pack in FIFI_FLASH_NEXT_EXL3_PACK"]
fn the_real_packs_placements_fit_and_tile_the_record() {
    use super::ring_source::span_placements;

    let dir = pack_dir();
    let checkpoint = Checkpoint::open(&dir).expect("the pack opens");
    let buffered = PackRecordSource::new(&checkpoint).expect("the buffered source derives");
    let layout = buffered.layout();
    let record = layout.record_bytes();
    let (layers, experts) = (buffered.layers(), buffered.experts());
    let mut placed = 0usize;
    for layer in 0..layers {
        for expert in 0..experts {
            let (spans, _) = super::ring_source::expert_spans(&checkpoint, &dir, layer, expert)
                .unwrap_or_else(|error| panic!("spans {layer}/{expert}: {error:?}"));
            let placements = span_placements(&spans, layout)
                .unwrap_or_else(|error| panic!("placements {layer}/{expert}: {error:?}"))
                .unwrap_or_else(|| panic!("layer {layer} expert {expert} places nothing"));
            let mut at = 0usize;
            for placement in &placements {
                assert!(
                    placement.range.end <= record,
                    "layer {layer} expert {expert}: a placement overflows the record row"
                );
                assert_eq!(
                    placement.range.start, at,
                    "layer {layer} expert {expert}: the placements tile in order"
                );
                at = placement.range.end;
            }
            assert_eq!(
                at, record,
                "layer {layer} expert {expert}: the placements tile the whole record"
            );
            placed += placements.len();
        }
    }
    eprintln!("placement coverage: {placed} placements across {layers}x{experts} experts");
}

/// N3's host half, in every battery: the ring's ONE slot sits parked (a
/// fence-only upload — no card, no event) when the next record arrives;
/// `record` must drain it instead of refusing. The pre-fix `record`
/// refused here with "every ring slot is outstanding".
#[cfg(all(target_os = "linux", not(feature = "native-cuda")))]
#[test]
fn record_drains_a_parked_upload_instead_of_refusing() {
    use super::bank_source_tests::synthetic;

    let (base, filesystem) = crate::family::expert_bank::host_tier::tests::direct_test_dir();
    let dir = base.join(format!("ring-n3-{}", std::process::id()));
    eprintln!("ring N3 test: {dir:?} on {filesystem}");
    std::fs::create_dir_all(&dir).expect("the ring test dir");
    let (checkpoint, _layout, _broken, image) =
        synthetic::checkpoint_with_image(&|_, _, _| synthetic::Tweak::Normal);
    std::fs::write(dir.join("m.safetensors"), &image).expect("the synthetic shard mirrors");

    let slots = NonZeroUsize::MIN;
    let bytes =
        RingRecordSource::backing_bytes(&checkpoint, &dir, slots).expect("the backing sizes");
    let backing: &'static mut [u8] = Box::leak(vec![0u8; bytes].into_boxed_slice());
    let ring = RingRecordSource::new(&checkpoint, &dir, slots, backing).expect("the ring source");
    let buffered = PackRecordSource::new(&checkpoint).expect("the buffered source");
    let record_bytes = ring.layout_ref().record_bytes();
    let mut expected = vec![0u8; record_bytes];
    buffered
        .record(0, 1, &mut expected)
        .expect("the buffered record");

    ring.park_for_test().expect("the fake parked upload");
    let mut got = vec![0u8; record_bytes];
    ring.record(0, 1, &mut got)
        .expect("record waits out the parked upload");
    assert_eq!(got, expected);
    std::fs::remove_dir_all(&dir).ok();
}

/// One span of a hand-built span table: the placement walk only reads
/// `part`, the key suffix, `dtype` and `matrix.len`, so a fake extent is
/// enough to drive the refusals.
fn fake_span(
    part: crate::family::exl3_pack::ExpertPart,
    suffix: &str,
    dtype: Dtype,
    len: usize,
) -> Span {
    Span {
        matrix: MatrixSpan {
            shard: PathBuf::from("m.safetensors"),
            file_offset: 0,
            len: len as u64,
            record_offset: 0,
        },
        key: format!("layers.0.experts.0.{part:?}.{suffix}"),
        part,
        dtype,
    }
}

fn err_text(result: Result<Option<Vec<Placement>>, EngError>) -> String {
    match result {
        Err(error) => error.describe(),
        Ok(Some(_)) => panic!("a malformed table placed"),
        Ok(None) => panic!("a malformed table stepped aside"),
    }
}

/// The placement walk refuses malformed span tables BY NAME: a gap, an
/// overlap, a missing suh, a missing svh and an overrun never build a
/// placement set, while the clean F16 table and a full-F32 table place
/// every component (mul1 stays skipped).
#[test]
fn span_placements_refuses_malformed_tables_by_name() {
    use super::bank_source_tests::synthetic;
    use super::ring_source::span_placements;
    use crate::family::exl3_pack::ExpertPart;

    let (_, layout, _) = synthetic::checkpoint(&|_, _, _| synthetic::Tweak::Normal);
    let trellis = |index: usize| layout.part(index).trellis.len();
    let scale = |index: usize, suh: bool| {
        if suh {
            layout.part(index).suh.len()
        } else {
            layout.part(index).svh.len()
        }
    };
    let full = |f32_scales: bool| {
        let mut spans = Vec::new();
        let parts = [ExpertPart::Gate, ExpertPart::Up, ExpertPart::Down];
        for (index, part) in parts.into_iter().enumerate() {
            spans.push(fake_span(part, "trellis", Dtype::F16, trellis(index)));
            let (len, dtype) = if f32_scales {
                (scale(index, true), Dtype::F32)
            } else {
                (scale(index, true) / 2, Dtype::F16)
            };
            spans.push(fake_span(part, "suh", dtype, len));
            let (len, dtype) = if f32_scales {
                (scale(index, false), Dtype::F32)
            } else {
                (scale(index, false) / 2, Dtype::F16)
            };
            spans.push(fake_span(part, "svh", dtype, len));
        }
        spans.push(fake_span(ExpertPart::Gate, "mul1", Dtype::I32, 4));
        spans
    };

    // The clean tables place every component and skip mul1.
    for f32_scales in [false, true] {
        let placed = span_placements(&full(f32_scales), &layout)
            .expect("the clean table places")
            .expect("without stepping aside");
        assert_eq!(placed.len(), 9, "mul1 places nothing");
    }

    // A gap: the gate trellis missing leaves the row's head uncovered.
    let mut gap = full(false);
    gap.remove(0);
    let text = err_text(span_placements(&gap, &layout));
    assert!(text.contains("tile"), "gap: {text}");

    // A missing suh (part 1) and a missing svh (part 2): each leaves its
    // region uncovered.
    let mut no_suh = full(false);
    no_suh.remove(4);
    let text = err_text(span_placements(&no_suh, &layout));
    assert!(
        text.contains("tile") || text.contains("uncovered"),
        "suh: {text}"
    );
    let mut no_svh = full(false);
    no_svh.remove(8);
    let text = err_text(span_placements(&no_svh, &layout));
    assert!(
        text.contains("tile") || text.contains("uncovered"),
        "svh: {text}"
    );

    // An overlap: the gate trellis twice claims the same region.
    let mut overlap = full(false);
    overlap.push(fake_span(
        ExpertPart::Gate,
        "trellis",
        Dtype::F16,
        trellis(0),
    ));
    let text = err_text(span_placements(&overlap, &layout));
    assert!(text.contains("tile"), "overlap: {text}");

    // An overrun (and its mirror, a short span): named, with the key and
    // the two disagreeing lengths.
    for (name, delta) in [("overrun", 8isize), ("short", -8isize)] {
        let mut spans = full(false);
        spans[0] = fake_span(
            ExpertPart::Gate,
            "trellis",
            Dtype::F16,
            (trellis(0) as isize + delta) as usize,
        );
        let text = err_text(span_placements(&spans, &layout));
        assert!(
            text.contains("carries") && text.contains("region"),
            "{name}: {text}"
        );
    }
}
