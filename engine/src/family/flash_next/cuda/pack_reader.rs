//! Reading one expert's record from the pack itself, through the ring.
//!
//! The bank assembles a record from three checkpoint matrices, so no
//! contiguous expert extent exists in any file — and the pack is NOT
//! duplicated on disk (no sidecar): this reader re-opens the shard a
//! matrix lives in for direct reads and stages each matrix's ALIGNED SPAN
//! (offset rounded down, end rounded up) into a ring slot, then places the
//! exact bytes where the record layout wants them inside the slot. Three
//! reads per expert; the slot is sized for the worst-case record plus the
//! worst-case span sum.
//!
//! Where O_DIRECT is refused — a filesystem that will not answer it, named
//! with the path — the reader refuses; the caller's plan falls to the next
//! valid tier or stops. There is no buffered fallback and no page-cache
//! copy of expert bytes anywhere on this path.

use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

use crate::EngError;
use crate::family::expert_bank::host_tier::{
    ODirectRing, Slot, align_up_sector, aligned_span, open_direct,
};

/// One matrix of one expert: the shard file it lives in, the exact extent
/// inside that file, and where the bytes land in the packed record.
#[derive(Clone, Debug)]
pub(crate) struct MatrixSpan {
    pub(crate) shard: PathBuf,
    pub(crate) file_offset: u64,
    pub(crate) len: u64,
    pub(crate) record_offset: u64,
}

/// The worst-case slot span the reader needs: the record area, padded to
/// the sector so the staging area behind it stays sector-aligned, plus
/// every span's worst-case aligned read. A span's aligned file range
/// (`align_up(offset+len) - align_down(offset)`) can reach `len +
/// 2·SECTOR − 2` — head and tail slack each up to a sector — and the
/// staging walk charges that total per SPAN, so the sizing must too:
/// charging per part undercounts a part split into several components.
pub(crate) fn slot_span_bytes(record_bytes: u64, spans: &[MatrixSpan]) -> Result<u64, EngError> {
    if spans.is_empty() {
        return Err(EngError::BadRequest(
            "pack reader: an expert record needs at least one span".into(),
        ));
    }
    let mut total = align_up_sector(record_bytes);
    for span in spans {
        let worst = align_up_sector(span.len.saturating_add(2 * 4096));
        total = total
            .checked_add(worst)
            .ok_or_else(|| EngError::BadRequest("pack reader: slot span overflows".into()))?;
    }
    Ok(total)
}

/// The pack reader: shard files re-opened for direct reads on first use and
/// kept open for the engine's lifetime. Nothing here knows record layouts;
/// the caller computes each expert's [`MatrixSpan`]s (from the checkpoint's
/// extents and the layout's part offsets) and sizes its slots with
/// [`slot_span_bytes`].
pub(crate) struct PackReader {
    shards: HashMap<PathBuf, File>,
}

impl PackReader {
    pub(crate) fn new() -> Self {
        Self {
            shards: HashMap::new(),
        }
    }

    /// One expert's record assembled into `slot`: every span staged and
    /// placed in order. Returns the exact bytes read from disk, for the
    /// serve-start rate measurement. Refuses (never falls back) where a
    /// shard will not answer O_DIRECT.
    pub(crate) fn read_expert(
        &mut self,
        ring: &mut ODirectRing,
        spans: &[MatrixSpan],
        record_bytes: u64,
        slot: Slot,
    ) -> Result<u64, EngError> {
        let outcome = self.read_expert_staged(ring, spans, record_bytes, slot);
        if outcome.is_err() {
            // The read failed: the slot goes back to the free list on every
            // error path — the reader never leaves a dead slot behind.
            let _ = ring.abandon(slot);
        }
        outcome
    }

    fn read_expert_staged(
        &mut self,
        ring: &mut ODirectRing,
        spans: &[MatrixSpan],
        record_bytes: u64,
        slot: Slot,
    ) -> Result<u64, EngError> {
        // The spans must tile the record exactly: sorted by record offset
        // they start at 0, abut with neither gap nor overlap, and end at
        // record_bytes. A wrong mapping is refused before any disk I/O.
        let mut coverage: Vec<&MatrixSpan> = spans.iter().collect();
        coverage.sort_by_key(|span| span.record_offset);
        let mut cursor = 0u64;
        for span in &coverage {
            if span.record_offset < cursor {
                return Err(EngError::BadRequest(format!(
                    "pack reader: spans at +{} overlap the record by {} bytes",
                    span.record_offset,
                    cursor - span.record_offset
                )));
            }
            if span.record_offset > cursor {
                return Err(EngError::BadRequest(format!(
                    "pack reader: spans leave a {}-byte gap at +{} of the record",
                    span.record_offset - cursor,
                    cursor
                )));
            }
            cursor = cursor
                .checked_add(span.len)
                .ok_or_else(|| EngError::BadRequest("pack reader: the record overflows".into()))?;
        }
        if cursor != record_bytes {
            return Err(EngError::BadRequest(format!(
                "pack reader: the spans cover {cursor} of {record_bytes} record bytes"
            )));
        }
        let mut stage = align_up_sector(record_bytes);
        let mut exact = 0u64;
        for span in spans {
            let file = self.shard_for(&span.shard)?;
            let in_span = ring
                .stage_span(file, span.file_offset, span.len, slot, stage)
                .map_err(|error| EngError::Backend {
                    operation: "pack_read_span",
                    message: format!("{}: {error}", span.shard.display()),
                })?;
            ring.place(slot, stage + in_span, span.record_offset, span.len)
                .map_err(|error| EngError::BadRequest(format!("pack reader: {error}")))?;
            let (_, total) = aligned_span(span.file_offset, span.len);
            stage = stage.checked_add(total).ok_or_else(|| {
                EngError::BadRequest("pack reader: stage offset overflows".into())
            })?;
            exact += span.len;
        }
        Ok(exact)
    }

    /// Opens one shard for direct reads, once; the refusal already names the
    /// path and the filesystem (open_direct's probe message carries both).
    fn shard_for(&mut self, path: &Path) -> Result<&File, EngError> {
        if !self.shards.contains_key(path) {
            let file = open_direct(path).map_err(|error| EngError::Backend {
                operation: "pack_reader_open",
                message: error.to_string(),
            })?;
            self.shards.insert(path.to_path_buf(), file);
        }
        Ok(&self.shards[path])
    }
}

/// Slots for one worst-case expert, the tests' sizing helper.
#[cfg(test)]
pub(crate) fn test_ring(
    slots: std::num::NonZeroUsize,
    slot_span: std::num::NonZeroU64,
) -> Result<ODirectRing, String> {
    use crate::family::expert_bank::host_tier::SECTOR;
    let bytes = slots.get() as u64 * slot_span.get();
    // A Vec's pointer is only 16-byte aligned; production ring memory is
    // page-aligned. Carve a sector-aligned sub-slice out of an oversized
    // leaked buffer so the ring's alignment refusal stays silent here.
    let slop = SECTOR as usize - 1;
    let buf = Box::leak(vec![0u8; bytes as usize + slop].into_boxed_slice());
    let base = buf.as_ptr() as usize;
    let off = (SECTOR as usize - (base % SECTOR as usize)) % SECTOR as usize;
    ODirectRing::new(slots, slot_span, &mut buf[off..off + bytes as usize])
}

#[cfg(test)]
mod tests {
    //! A synthetic three-matrix pack on the local NVMe scratch (never /tmp:
    //! tmpfs refuses O_DIRECT): the tensors sit at deliberately unaligned
    //! file offsets, the record places them out of order, and the read must
    //! come back byte-exact through the ring. Where the scratch is absent
    //! or O_DIRECT is refused the exact-byte test skips with a printed
    //! note; the sizing math and refusal paths still run.

    use super::{MatrixSpan, PackReader, slot_span_bytes, test_ring};
    use crate::family::expert_bank::host_tier::open_direct;
    use std::io::Write;
    use std::num::{NonZeroU64, NonZeroUsize};
    use std::path::PathBuf;

    const PART_A: &[u8] = &[0xA5; 5000];
    const PART_B: &[u8] = &[0x5B; 3000];
    const PART_C: &[u8] = &[0xC7; 7_532];

    /// Writes a safetensors shard with the three parts at hand-picked
    /// unaligned offsets and returns each part's (file offset, len).
    fn write_pack(dir: &std::path::Path) -> (PathBuf, Vec<(u64, u64)>) {
        let header = serde_json::json!({
            "a": {"dtype": "U8", "shape": [PART_A.len()], "data_offsets": [7, 7 + PART_A.len()]},
            "b": {"dtype": "U8", "shape": [PART_B.len()], "data_offsets": [7 + PART_A.len() + 13, 7 + PART_A.len() + 13 + PART_B.len()]},
            "c": {"dtype": "U8", "shape": [PART_C.len()], "data_offsets": [7 + PART_A.len() + 13 + PART_B.len() + 29, 7 + PART_A.len() + 13 + PART_B.len() + 29 + PART_C.len()]},
        });
        let header = serde_json::to_vec(&header).unwrap();
        let path = dir.join("pack-reader-test.safetensors");
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&header).unwrap();
        file.write_all(&[0u8; 7]).unwrap();
        file.write_all(PART_A).unwrap();
        file.write_all(&[0u8; 13]).unwrap();
        file.write_all(PART_B).unwrap();
        file.write_all(&[0u8; 29]).unwrap();
        file.write_all(PART_C).unwrap();
        file.flush().unwrap();
        let a = 8 + header.len() as u64 + 7;
        let b = a + PART_A.len() as u64 + 13;
        let c = b + PART_B.len() as u64 + 29;
        (
            path,
            vec![
                (a, PART_A.len() as u64),
                (b, PART_B.len() as u64),
                (c, PART_C.len() as u64),
            ],
        )
    }

    /// The direct-read tests need a filesystem that answers O_DIRECT.
    /// `FIFI_TEST_DIRECT_DIR` wins; otherwise `CARGO_TARGET_TMPDIR` (a
    /// real filesystem under the build dir) or the temp dir. No host path
    /// is hard-coded: tmpfs refuses O_DIRECT and the tests skip there
    /// with a printed note — including when the dir is not writable.
    fn scratch() -> Option<PathBuf> {
        let dir = std::env::var_os("FIFI_TEST_DIRECT_DIR")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("CARGO_TARGET_TMPDIR").map(PathBuf::from))
            .unwrap_or_else(std::env::temp_dir);
        eprintln!("pack-reader test: {dir:?} on {}", filesystem_of(&dir));
        let probe = dir.join(".o_direct_probe");
        match std::fs::write(&probe, [0u8; 4096]) {
            Ok(()) => {
                let _ = std::fs::remove_file(&probe);
                Some(dir)
            }
            Err(e) => {
                eprintln!(
                    "pack-reader test: {dir:?} is not writable ({e}); the exact-byte read skips"
                );
                None
            }
        }
    }

    /// The filesystem type the direct-read test runs on, from mountinfo
    /// (longest mount-point prefix wins), so the log says what answered
    /// O_DIRECT — tmpfs accepts it from 6.6 on, which proves no bypass.
    fn filesystem_of(path: &std::path::Path) -> String {
        let mut best = (0usize, "unknown".to_string());
        for line in std::fs::read_to_string("/proc/self/mountinfo")
            .unwrap_or_default()
            .lines()
        {
            let Some((left, right)) = line.split_once(" - ") else {
                continue;
            };
            let mut fields = left.split(' ');
            let _ = (fields.next(), fields.next(), fields.next(), fields.next());
            let (Some(mount_point), Some(fstype)) = (fields.next(), right.split(' ').next()) else {
                continue;
            };
            if path.starts_with(mount_point) && mount_point.len() >= best.0 {
                best = (mount_point.len(), fstype.to_string());
            }
        }
        best.1
    }

    #[test]
    fn the_slot_span_covers_the_record_and_every_worst_case_span() {
        // record 5000 -> 8192; spans 5000/3000/7532 -> worst aligned reads
        // align_up(len + 2·4096) = 16384/12288/16384; total 53_248 —
        // computed by hand.
        let spans: Vec<MatrixSpan> = [5000u64, 3000, 7532]
            .into_iter()
            .enumerate()
            .map(|(i, len)| MatrixSpan {
                shard: PathBuf::new(),
                file_offset: 0,
                len,
                record_offset: i as u64,
            })
            .collect();
        let got = slot_span_bytes(5000, &spans).unwrap();
        assert_eq!(got, 8192 + 16384 + 12288 + 16384);
        assert!(matches!(
            slot_span_bytes(5000, &[]),
            Err(crate::EngError::BadRequest(_))
        ));
    }

    #[test]
    fn a_three_matrix_read_places_each_exact_bytes_where_the_record_wants() {
        let Some(dir) = scratch() else { return };
        let (path, extents) = write_pack(&dir);
        if open_direct(&path).is_err() {
            eprintln!(
                "pack-reader test: {path:?} does not answer O_DIRECT here; \
                 the exact-byte read skips"
            );
            let _ = std::fs::remove_file(&path);
            return;
        }
        let record_bytes = (PART_A.len() + PART_B.len() + PART_C.len()) as u64;
        // The record wants C first, then A, then B — placement is the
        // layout's business, not file order.
        let spans = vec![
            MatrixSpan {
                shard: path.clone(),
                file_offset: extents[2].0,
                len: extents[2].1,
                record_offset: 0,
            },
            MatrixSpan {
                shard: path.clone(),
                file_offset: extents[0].0,
                len: extents[0].1,
                record_offset: PART_C.len() as u64,
            },
            MatrixSpan {
                shard: path.clone(),
                file_offset: extents[1].0,
                len: extents[1].1,
                record_offset: (PART_C.len() + PART_A.len()) as u64,
            },
        ];
        let span = slot_span_bytes(record_bytes, &spans).unwrap();
        let mut ring = test_ring(
            NonZeroUsize::new(1).unwrap(),
            NonZeroU64::new(span).unwrap(),
        )
        .unwrap();
        let slot = ring.acquire().unwrap();
        let mut reader = PackReader::new();
        let exact = reader
            .read_expert(&mut ring, &spans, record_bytes, slot)
            .unwrap();
        assert_eq!(exact, record_bytes);
        let record = ring.slot_range(slot, 0, record_bytes).unwrap();
        let want: Vec<u8> = [PART_C, PART_A, PART_B].concat();
        assert_eq!(
            record,
            &want[..],
            "byte-exact through the ring, placed as the layout wants"
        );
        let _ = std::fs::remove_file(&path);
    }

    /// Dummy spans of the given lengths: sizing reads only `len`.
    fn len_spans(lens: &[u64]) -> Vec<MatrixSpan> {
        lens.iter()
            .enumerate()
            .map(|(i, &len)| MatrixSpan {
                shard: PathBuf::new(),
                file_offset: 0,
                len,
                record_offset: i as u64,
            })
            .collect()
    }

    /// Coverage refuses a mapping with a hole BEFORE any disk I/O: the
    /// spans need not point at a real file for the refusal to fire.
    #[test]
    fn a_gap_in_the_span_coverage_is_refused_before_any_read() {
        let span = slot_span_bytes(8192, &len_spans(&[4096, 4096])).unwrap();
        let mut ring = test_ring(
            NonZeroUsize::new(1).unwrap(),
            NonZeroU64::new(span).unwrap(),
        )
        .unwrap();
        let slot = ring.acquire().unwrap();
        let nowhere = std::path::PathBuf::from("/nonexistent-pack/shard.bin");
        let spans = vec![
            MatrixSpan {
                shard: nowhere.clone(),
                file_offset: 0,
                len: 4096,
                record_offset: 0,
            },
            MatrixSpan {
                shard: nowhere,
                file_offset: 4096,
                len: 4096,
                record_offset: 8192,
            },
        ];
        let err = PackReader::new()
            .read_expert(&mut ring, &spans, 8192, slot)
            .unwrap_err();
        let message = format!("{err:?}");
        assert!(message.contains("gap"), "{message}");
    }

    #[test]
    fn an_overlap_in_the_span_coverage_is_refused_before_any_read() {
        let span = slot_span_bytes(8192, &len_spans(&[6144, 6144])).unwrap();
        let mut ring = test_ring(
            NonZeroUsize::new(1).unwrap(),
            NonZeroU64::new(span).unwrap(),
        )
        .unwrap();
        let slot = ring.acquire().unwrap();
        let nowhere = std::path::PathBuf::from("/nonexistent-pack/shard.bin");
        let spans = vec![
            MatrixSpan {
                shard: nowhere.clone(),
                file_offset: 0,
                len: 6144,
                record_offset: 0,
            },
            MatrixSpan {
                shard: nowhere,
                file_offset: 6144,
                len: 6144,
                record_offset: 2048,
            },
        ];
        let err = PackReader::new()
            .read_expert(&mut ring, &spans, 8192, slot)
            .unwrap_err();
        let message = format!("{err:?}");
        assert!(message.contains("overlap"), "{message}");
    }

    /// Every error path abandons the slot: a failed read never leaves a
    /// dead slot behind, so the ring recycles it for the next expert.
    #[test]
    fn a_failed_read_returns_its_slot_to_the_free_list() {
        let span = slot_span_bytes(8192, &len_spans(&[8192])).unwrap();
        let mut ring = test_ring(
            NonZeroUsize::new(1).unwrap(),
            NonZeroU64::new(span).unwrap(),
        )
        .unwrap();
        let slot = ring.acquire().unwrap();
        let spans = vec![MatrixSpan {
            shard: std::path::PathBuf::from("/nonexistent-pack/shard.bin"),
            file_offset: 0,
            len: 8192,
            record_offset: 0,
        }];
        assert!(
            PackReader::new()
                .read_expert(&mut ring, &spans, 8192, slot)
                .is_err()
        );
        assert_eq!(
            ring.acquire(),
            Some(slot),
            "the failed read abandoned its slot back to the free list"
        );
    }

    #[test]
    fn the_slot_span_bounds_the_worst_aligned_file_range() {
        use super::{align_up_sector, aligned_span};
        use crate::family::expert_bank::host_tier::SECTOR;
        // The charge is align_up(len + 2*SECTOR) per span; the exact
        // staging cost at the WORST file offset (head and tail slack each
        // a sector minus one) must never exceed it.
        for &len in &[
            1u64,
            100,
            SECTOR - 1,
            SECTOR,
            SECTOR + 1,
            3 * SECTOR + 5,
            618_496,
        ] {
            for residue in [0u64, 1, SECTOR / 2, SECTOR - 1] {
                let offset = 97 * SECTOR + residue;
                let (_, exact) = aligned_span(offset, len);
                let charge = slot_span_bytes(0, &len_spans(&[len])).unwrap() - align_up_sector(0);
                assert!(
                    exact <= charge,
                    "len {len} offset {offset}: exact {exact} exceeds the charge {charge}"
                );
            }
        }
    }
}
