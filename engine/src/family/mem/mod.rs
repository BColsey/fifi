//! The memory facts every plan reads, and the numbers every host plan shares.
//!
//! One [`Facts`] is read per start and handed to every stage of it: the host (`MemTotal`,
//! `MemAvailable` and the enclosing cgroup's remaining bytes), the card (free and total), and the
//! free bytes of each directory a plan writes to. A plan takes its figures from the facts it is
//! given and never re-reads the machine, so two stages of one start cannot read the host in two
//! frames.
//!
//! The policy numbers of the host side live here too, once: [`os_floor_of`] is the bytes the OS
//! keeps out of every plan, and [`HOST_MARGIN_BYTES`] is the slack a plan that pins a store keeps
//! beyond it. A family supplies the byte costs of its own objects, never a floor or a margin.

// The facts are read by the CUDA-build starts; the host build compiles them without a reader.
#![allow(dead_code, reason = "the CUDA-build starts read the facts")]

// The host reader parses `/proc`, so it exists where a Linux host does (and in the test build);
// the numbers below are plain arithmetic and exist everywhere.
#[cfg(any(test, target_os = "linux"))]
mod host;

#[cfg(any(test, target_os = "linux"))]
#[allow(unused_imports, reason = "the name a caller matches a provenance on")]
pub(crate) use host::{HostFacts, Provenance};

use std::path::{Path, PathBuf};

#[cfg(any(test, target_os = "linux"))]
use crate::EngError;

/// One gibibyte.
pub(crate) const GIB: u64 = 1 << 30;

/// The smallest OS floor, bytes.
const MIN_OS_FLOOR_BYTES: u64 = 8 * GIB;

/// The slack a plan that pins a host store keeps free beyond the OS floor, so the plan does not
/// ride the line (the expert caches kept 2 GiB before there was one rule).
pub(crate) const HOST_MARGIN_BYTES: u64 = 2 * GIB;

/// The flat headroom an admission of one host object (a mapped expert set, a saved prefix, the
/// tiered path's scratch and heap) keeps free beside it. These admissions keep no OS floor yet;
/// moving them under [`os_floor_of`] changes what they admit, so it is its own change.
pub(crate) const HOST_HEADROOM_BYTES: u64 = GIB;

/// The bytes the OS keeps out of every host plan: a tenth of physical memory, never under 8 GiB.
/// The one floor: the host guard, every family's host plan and the facts' `spare` read it here.
pub(crate) fn os_floor_of(total: u64) -> u64 {
    total.div_ceil(10).max(MIN_OS_FLOOR_BYTES)
}

/// The card's memory as read at one moment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CardFacts {
    pub(crate) free: u64,
    pub(crate) total: u64,
}

/// The free bytes of the filesystem that holds a directory.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct DiskFacts {
    pub(crate) dir: PathBuf,
    /// `None` when the filesystem could not be read.
    pub(crate) free: Option<u64>,
}

/// The free bytes of the filesystem that holds `dir`, probed through the nearest existing
/// ancestor: a cache directory that does not exist yet lives on its parent's filesystem.
pub(crate) fn free_bytes(dir: &Path) -> Option<u64> {
    let mut probe = dir;
    while !probe.exists() {
        probe = probe.parent()?;
    }
    let stat = rustix::fs::statvfs(probe).ok()?;
    stat.f_bavail.checked_mul(stat.f_frsize)
}

/// The facts of one start. Read once, then lent to every stage.
#[cfg(any(test, target_os = "linux"))]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Facts {
    pub(crate) host: HostFacts,
    pub(crate) card: Option<CardFacts>,
    pub(crate) disks: Vec<DiskFacts>,
}

#[cfg(any(test, target_os = "linux"))]
impl Facts {
    /// The facts with the host read from the machine and no card or disk yet.
    pub(crate) fn read(label: &'static str, root_label: &'static str) -> Result<Self, EngError> {
        Ok(Self::with_host(HostFacts::read(label, root_label)?))
    }

    /// The facts around a host that is already read (or built by a test).
    pub(crate) fn with_host(host: HostFacts) -> Self {
        Self {
            host,
            card: None,
            disks: Vec::new(),
        }
    }

    /// Adds the card's free and total bytes.
    pub(crate) fn with_card(mut self, free: u64, total: u64) -> Self {
        self.card = Some(CardFacts { free, total });
        self
    }

    /// Adds the free bytes of the filesystem that holds `dir`.
    pub(crate) fn with_disk(mut self, dir: &Path) -> Self {
        self.disks.push(DiskFacts {
            dir: dir.to_path_buf(),
            free: free_bytes(dir),
        });
        self
    }

    /// The free bytes read for `dir`, if it was added and readable.
    pub(crate) fn disk_free(&self, dir: &Path) -> Option<u64> {
        self.disks
            .iter()
            .find(|disk| disk.dir == dir)
            .and_then(|disk| disk.free)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SPELLING_OF_THE_FLOOR_RULE: &str = concat!(".div_ceil(", "10).max(");

    #[test]
    fn the_floor_is_a_tenth_of_the_total_and_never_under_eight_gib() {
        for (total_gib, floor_gib) in [
            (8u64, 8u64),
            (16, 8),
            (32, 8),
            (62, 8),
            (64, 8),
            (80, 8),
            (128, 13),
        ] {
            let floor = os_floor_of(total_gib * GIB);
            assert_eq!(floor.div_ceil(GIB), floor_gib, "{total_gib} GiB host");
        }
        assert_eq!(os_floor_of(0), 8 * GIB);
        assert_eq!(os_floor_of(u64::MAX), u64::MAX.div_ceil(10));
    }

    #[test]
    fn facts_floor_and_spare_follow_the_one_rule() {
        let text = "MemTotal: 65011712 kB\nMemAvailable: 33554432 kB\n";
        let host = HostFacts::from_meminfo("test", text, Some(20 * GIB)).unwrap();
        let facts = Facts::with_host(host).with_card(20 * GIB, 24 * GIB);
        assert_eq!(facts.host.os_floor(), os_floor_of(facts.host.total()));
        assert_eq!(facts.host.budget(), 20 * GIB, "the cgroup clamps the host");
        assert_eq!(
            facts.host.spare(),
            20 * GIB,
            "the cgroup binds, so no floor inside it"
        );
        assert_eq!(
            facts.card,
            Some(CardFacts {
                free: 20 * GIB,
                total: 24 * GIB
            })
        );
    }

    #[test]
    fn a_disk_fact_is_read_through_the_nearest_existing_ancestor() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("not").join("yet");
        let text = "MemTotal: 65011712 kB\nMemAvailable: 33554432 kB\n";
        let host = HostFacts::from_meminfo("test", text, None).unwrap();
        let facts = Facts::with_host(host)
            .with_disk(dir.path())
            .with_disk(&missing);
        let there = facts.disk_free(dir.path()).expect("the directory exists");
        assert!(there > 0);
        // The two reads are two moments of one filesystem: equal up to what other writers do
        // between them.
        let through_ancestor = facts.disk_free(&missing).expect("the ancestor exists");
        assert!(
            there.abs_diff(through_ancestor) < GIB,
            "{there} {through_ancestor}"
        );
        assert_eq!(facts.disk_free(Path::new("/never/added")), None);
    }

    fn rust_files(dir: &Path, found: &mut Vec<PathBuf>) {
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                rust_files(&path, found);
            } else if path.extension().is_some_and(|extension| extension == "rs") {
                found.push(path);
            }
        }
    }

    /// The floor rule is spelled in exactly one place. A second copy is how a plan and the guard
    /// that checks it came to disagree about the same RAM.
    #[test]
    fn the_floor_rule_is_written_once_in_the_engine() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let mut files = Vec::new();
        rust_files(&root, &mut files);
        let mut spellings = Vec::new();
        for file in files {
            let text = std::fs::read_to_string(&file).unwrap();
            let count = text.matches(SPELLING_OF_THE_FLOOR_RULE).count();
            if count > 0 {
                spellings.push((file.strip_prefix(&root).unwrap().to_path_buf(), count));
            }
        }
        assert_eq!(
            spellings,
            vec![(PathBuf::from("family/mem/mod.rs"), 1)],
            "{spellings:?}"
        );
    }
}
