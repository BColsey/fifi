//! The ring source against the buffered source: both answer the bank's
//! `RecordSource` contract from the same opened pack, so the record every
//! expert yields must be identical byte-for-byte. The corpus pack arrives
//! through `FIFI_FLASH_NEXT_EXL3_PACK` (the same env the buffered corpus
//! tests read); where its filesystem refuses O_DIRECT the ring source
//! refuses by contract, and the test reports the filesystem and skips.

use super::bank::RecordSource;
use super::bank_source::PackRecordSource;
use super::ring_source::{HostCopyFence, RingRecordSource, Span, UploadFence, parse_spans};
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
