//! Parallel direct reads of one contiguous file extent into a sector-aligned buffer.
//!
//! A layer's expert stacks are each one contiguous byte range of a model file, 100 MiB to 1.3 GiB
//! long. One buffered `pread` of such a range runs at the page cache's single-thread rate (1.9
//! GB/s measured on a 990 PRO). [`read_extent`] splits the range into sector-aligned chunks and
//! reads them with several threads at once, each thread keeping one read in flight, straight into
//! the destination slice (pinned host memory in the card transport), bypassing the page cache.
//!
//! Alignment. A direct read needs a sector-aligned memory address, file offset and length. The
//! extent is widened to the aligned span `[start, start + span)` ([`aligned_span`]); the extent's
//! own bytes sit at `offset - start` inside the destination. The caller sizes the destination for
//! the whole span and places it so that this position is where it wants the bytes.
//!
//! Plan. The thread count and chunk size depend on the drive: [`probe_plan`] times each candidate
//! on the first bytes of a real extent and keeps the fastest.

use std::{fs::File, os::unix::fs::FileExt, time::Instant};

use super::host_tier::{SECTOR, aligned_span};
use crate::EngError;

fn read_error(detail: String) -> EngError {
    EngError::Checkpoint(format!("expert extent read: {detail}"))
}

/// How an extent is split across threads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ExtentPlan {
    /// Reads in flight: one per thread.
    pub(crate) threads: usize,
    /// Bytes of one read, a multiple of the sector.
    pub(crate) chunk: usize,
}

impl ExtentPlan {
    /// The plan used until a probe has measured the drive.
    pub(crate) const DEFAULT: Self = Self {
        threads: 8,
        chunk: 2 << 20,
    };

    /// The candidates [`probe_plan`] times: a queue depth and a read size each.
    pub(crate) const CANDIDATES: [Self; 8] = [
        Self {
            threads: 2,
            chunk: 4 << 20,
        },
        Self {
            threads: 4,
            chunk: 1 << 20,
        },
        Self {
            threads: 4,
            chunk: 4 << 20,
        },
        Self {
            threads: 8,
            chunk: 1 << 20,
        },
        Self {
            threads: 8,
            chunk: 2 << 20,
        },
        Self {
            threads: 8,
            chunk: 8 << 20,
        },
        Self {
            threads: 16,
            chunk: 1 << 20,
        },
        Self {
            threads: 16,
            chunk: 4 << 20,
        },
    ];

    fn check(self) -> Result<(), EngError> {
        if self.threads == 0 || self.chunk == 0 || !(self.chunk as u64).is_multiple_of(SECTOR) {
            return Err(read_error(format!(
                "{} threads of {}-byte reads: the chunk must be a positive multiple of {SECTOR}",
                self.threads, self.chunk
            )));
        }
        Ok(())
    }
}

/// What a read did.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ExtentStats {
    /// Bytes the drive was asked for: the aligned span.
    pub(crate) bytes: u64,
    pub(crate) seconds: f64,
}

impl ExtentStats {
    pub(crate) fn gbps(&self) -> f64 {
        if self.seconds > 0.0 {
            self.bytes as f64 / self.seconds / 1e9
        } else {
            0.0
        }
    }
}

/// The destination bytes a read of `bytes` at `offset` needs, and where the extent's first byte
/// sits in them: `(span, delta)`.
pub(crate) fn span_of(offset: u64, bytes: usize) -> (usize, usize) {
    let (start, span) = aligned_span(offset, bytes as u64);
    (span as usize, (offset - start) as usize)
}

/// Reads `bytes` at `offset` of `file` (opened for direct reads) into the front of `dst`. `dst`
/// must start at a sector-aligned address and hold at least [`span_of`]'s span; the extent's
/// bytes end up `delta` bytes into it. A read that runs past the end of the file leaves the rest
/// of its chunk untouched.
pub(crate) fn read_extent(
    file: &File,
    offset: u64,
    bytes: usize,
    dst: &mut [u8],
    plan: ExtentPlan,
) -> Result<ExtentStats, EngError> {
    plan.check()?;
    let (start, span) = aligned_span(offset, bytes as u64);
    let span = span as usize;
    if dst.len() < span {
        return Err(read_error(format!(
            "the destination holds {} bytes, the aligned span of the extent is {span}",
            dst.len()
        )));
    }
    if !(dst.as_ptr() as usize).is_multiple_of(SECTOR as usize) {
        return Err(read_error(
            "the destination is not sector aligned, which a direct read needs".into(),
        ));
    }
    let started = Instant::now();
    let threads = plan.threads.min(span.div_ceil(plan.chunk)).max(1);
    let mut lanes: Vec<Vec<(u64, &mut [u8])>> = (0..threads).map(|_| Vec::new()).collect();
    for (index, part) in dst[..span].chunks_mut(plan.chunk).enumerate() {
        lanes[index % threads].push((start + (index * plan.chunk) as u64, part));
    }
    let outcome: Result<(), String> = std::thread::scope(|scope| {
        let handles: Vec<_> = lanes
            .into_iter()
            .map(|lane| {
                scope.spawn(move || -> Result<(), String> {
                    for (at, part) in lane {
                        let mut done = 0usize;
                        while done < part.len() {
                            match file.read_at(&mut part[done..], at + done as u64) {
                                Ok(0) => break,
                                Ok(n) => done += n,
                                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {}
                                Err(error) => {
                                    return Err(format!("read at {}: {error}", at + done as u64));
                                }
                            }
                        }
                    }
                    Ok(())
                })
            })
            .collect();
        let mut result = Ok(());
        for handle in handles {
            let joined = handle
                .join()
                .unwrap_or_else(|_| Err("a reader thread panicked".to_string()));
            if result.is_ok() {
                result = joined;
            }
        }
        result
    });
    outcome.map_err(read_error)?;
    Ok(ExtentStats {
        bytes: span as u64,
        seconds: started.elapsed().as_secs_f64(),
    })
}

/// Times every candidate plan on the first `sample` bytes of the extent at `offset` (capped at
/// `bytes`) and returns the fastest with every candidate's stats. `scratch` must be sector aligned
/// and hold the sample's aligned span; the file is opened for direct reads, so the page cache
/// does not flatter a repeat.
pub(crate) fn probe_plan(
    file: &File,
    offset: u64,
    bytes: usize,
    sample: usize,
    scratch: &mut [u8],
) -> Result<(ExtentPlan, Vec<(ExtentPlan, ExtentStats)>), EngError> {
    let take = sample.min(bytes);
    let mut seen = Vec::with_capacity(ExtentPlan::CANDIDATES.len());
    for plan in ExtentPlan::CANDIDATES {
        let stats = read_extent(file, offset, take, scratch, plan)?;
        seen.push((plan, stats));
    }
    let best = seen
        .iter()
        .max_by(|a, b| a.1.gbps().total_cmp(&b.1.gbps()))
        .map_or(ExtentPlan::DEFAULT, |entry| entry.0);
    Ok((best, seen))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::family::expert_bank::host_tier::{open_direct, tests::direct_test_dir};
    use std::io::Write;

    /// A buffer whose first byte is sector aligned, over a larger allocation.
    fn aligned(len: usize) -> (Vec<u8>, usize) {
        let buffer = vec![0u8; len + SECTOR as usize];
        let shift = SECTOR as usize - (buffer.as_ptr() as usize % SECTOR as usize);
        let shift = shift % SECTOR as usize;
        (buffer, shift)
    }

    fn pattern(index: usize) -> u8 {
        (index.wrapping_mul(31).wrapping_add(index >> 9)) as u8
    }

    fn scratch_file(len: usize) -> Option<(std::path::PathBuf, File)> {
        let (dir, _) = direct_test_dir();
        let path = dir.join(format!("extent-read-{}-{len}.bin", std::process::id()));
        let mut out = std::fs::File::create(&path).ok()?;
        let bytes: Vec<u8> = (0..len).map(pattern).collect();
        out.write_all(&bytes).ok()?;
        out.sync_all().ok()?;
        match open_direct(&path) {
            Ok(file) => Some((path, file)),
            Err(error) => {
                eprintln!("extent read test: {error}; skipped");
                let _ = std::fs::remove_file(&path);
                None
            }
        }
    }

    #[test]
    fn every_plan_reads_the_extent_byte_exact_at_any_offset() {
        let len = 5 * (1 << 20) + 1234;
        let Some((path, file)) = scratch_file(len) else {
            return;
        };
        for (offset, bytes) in [
            (0usize, len),
            (100, 3 << 20),
            (4096, 1 << 20),
            (4097, 777_777),
        ] {
            for plan in [
                ExtentPlan {
                    threads: 1,
                    chunk: 4096,
                },
                ExtentPlan {
                    threads: 3,
                    chunk: 64 << 10,
                },
                ExtentPlan::DEFAULT,
                ExtentPlan {
                    threads: 16,
                    chunk: 1 << 20,
                },
            ] {
                let (span, delta) = span_of(offset as u64, bytes);
                let (mut buffer, shift) = aligned(span);
                let dst = &mut buffer[shift..shift + span];
                let stats = read_extent(&file, offset as u64, bytes, dst, plan).expect("read");
                assert!(stats.bytes as usize >= bytes);
                for (i, byte) in dst[delta..delta + bytes].iter().enumerate() {
                    assert_eq!(*byte, pattern(offset + i), "{plan:?} at {offset}+{i}");
                }
            }
        }
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn a_read_refuses_what_a_direct_read_cannot_do() {
        let len = 1 << 20;
        let Some((path, file)) = scratch_file(len) else {
            return;
        };
        let (mut buffer, shift) = aligned(2 << 20);
        let plan = ExtentPlan::DEFAULT;
        // Too small a destination.
        let dst = &mut buffer[shift..shift + 4096];
        assert!(read_extent(&file, 0, 1 << 16, dst, plan).is_err());
        // A destination off the sector.
        let dst = &mut buffer[shift + 1..shift + 1 + (1 << 20)];
        assert!(read_extent(&file, 0, 1 << 16, dst, plan).is_err());
        // A chunk that is not a sector multiple.
        let dst = &mut buffer[shift..shift + (1 << 20)];
        let bad = ExtentPlan {
            threads: 2,
            chunk: 1000,
        };
        assert!(read_extent(&file, 0, 1 << 16, dst, bad).is_err());
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn the_probe_keeps_the_fastest_candidate() {
        let len = 8 << 20;
        let Some((path, file)) = scratch_file(len) else {
            return;
        };
        let (mut buffer, shift) = aligned(len + SECTOR as usize);
        let scratch = &mut buffer[shift..shift + len + SECTOR as usize];
        let (best, seen) = probe_plan(&file, 0, len, 4 << 20, scratch).expect("probe");
        assert_eq!(seen.len(), ExtentPlan::CANDIDATES.len());
        let top = seen.iter().map(|(_, s)| s.gbps()).fold(0.0, f64::max);
        let chosen = seen.iter().find(|(p, _)| *p == best).expect("chosen").1;
        assert!((chosen.gbps() - top).abs() < 1e-9);
        let _ = std::fs::remove_file(path);
    }
}
