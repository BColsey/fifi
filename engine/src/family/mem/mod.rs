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

mod device;
// The host facts exist on every target: `/proc` and the cgroup tree on Linux, the unified memory
// of `unified` on a Mac.
mod host;
pub(crate) mod layer_rung;
#[cfg(test)]
mod layer_rung_real_tests;
mod unified;

// The Mac half of the facts: unified-memory fractions, Metal's working set and the expert bytes
// of a checkpoint. Every item keeps the gating it has in `unified`.
#[allow(
    unused_imports,
    reason = "read by the MLX starts and the tests; a given build reads a subset"
)]
pub(crate) use unified::*;

pub(crate) use device::DEVICE;

/// The host memory a stored prefix may still take, read now: the bytes the OS can hand out
/// without paging on a Mac (free, inactive and speculative pages), the host budget elsewhere.
/// `None` when the host cannot be read.
pub(crate) fn tier_available_bytes(label: &'static str) -> Option<u64> {
    #[cfg(target_os = "macos")]
    {
        let _ = label;
        unified::available_memory_bytes()
    }
    #[cfg(not(target_os = "macos"))]
    {
        HostFacts::read(label, label).ok().map(|host| host.budget())
    }
}

/// The bytes of snapshots a host store may hold when no planner funds a figure: the machine's
/// memory less the OS floor the plans keep, less the `resident` bytes the engine already holds,
/// and no more than the enclosing cgroup leaves a payload (the room the save admission
/// [`HostFacts::admit_payload`] admits against), so a store never funds an entry that admission
/// then refuses. `None` when the machine's memory is unknown.
pub(crate) fn host_store_budget(resident: u64) -> Option<u64> {
    let total = physical_memory_bytes()?;
    let cgroup = HostFacts::read("kv disk", "kv disk")
        .ok()
        .and_then(|host| host.cgroup_payload_room());
    Some(host_store_budget_of(total, resident, cgroup))
}

/// [`host_store_budget`] of a machine with `total` bytes, `resident` bytes held and `cgroup`
/// bytes of room left by its cgroup (`None` when no limit binds).
pub(crate) fn host_store_budget_of(total: u64, resident: u64, cgroup: Option<u64>) -> u64 {
    let host = total
        .saturating_sub(os_floor_of(total))
        .saturating_sub(resident);
    cgroup.map_or(host, |room| host.min(room))
}

/// The bytes this process holds resident now (its resident set, which counts pinned buffers),
/// or zero when they cannot be read.
pub(crate) fn process_resident_bytes() -> u64 {
    HostFacts::read("kv disk", "kv disk")
        .map(|host| host.process_resident())
        .unwrap_or(0)
}

#[allow(unused_imports, reason = "the name a caller matches a provenance on")]
pub(crate) use host::{HostFacts, Provenance};

use std::path::{Path, PathBuf};

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

/// The bytes the OS keeps out of every host plan: a tenth of physical memory, never under 8 GiB,
/// and never more than a quarter of it, so a small host keeps room for a store (an 8 GiB host
/// keeps 2 GiB, a 16 GiB host 4 GiB, a 32 GiB host 8 GiB, a 62 GiB host 8 GiB).
/// The one floor: the host guard, every family's host plan and the facts' `spare` read it here.
pub(crate) fn os_floor_of(total: u64) -> u64 {
    total
        .div_ceil(10)
        .max(MIN_OS_FLOOR_BYTES)
        .min(total / MAX_OS_FLOOR_SHARE)
}

/// The floor is at most this fraction (one over) of physical memory: a quarter.
const MAX_OS_FLOOR_SHARE: u64 = 4;

/// The machine as a host plan reads it: `MemTotal`, the budget (`available`: `MemAvailable`
/// clamped by every cgroup ancestor's remaining bytes) and the bytes of the budget kept back.
///
/// `floor` is what a plan that carries its own process terms (heap, margin, staging) keeps back:
/// the OS floor while the host is the binding limit, nothing while a cgroup is (the cgroup
/// already fences the process off from the host). `payload_floor` is what a plan that charges only
/// a payload keeps back: the same, plus the process's own overhead when a cgroup binds. Both are
/// read from one [`HostFacts`], so a plan and the guard that checks it agree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct HostMemory {
    pub(crate) total: u64,
    pub(crate) available: u64,
    floor: u64,
    payload_floor: u64,
}

impl HostMemory {
    /// The memory of one start's host facts: the figures the guard admits against.
    pub(crate) fn of(host: &HostFacts) -> Self {
        Self {
            total: host.total(),
            available: host.budget(),
            floor: host.kept_back(),
            payload_floor: host.payload_kept_back(),
        }
    }

    /// A memory with a floor read elsewhere (the report path carries it from the host facts).
    pub(crate) fn with_floor(total: u64, available: u64, floor: u64) -> Self {
        Self {
            total,
            available,
            floor,
            payload_floor: floor,
        }
    }

    /// A memory from a probed pair `(MemTotal, budget)` with no cgroup knowledge: the host is
    /// taken as the binding limit and the whole OS floor is kept back.
    pub(crate) fn from_pair(total: u64, available: u64) -> Self {
        Self::with_floor(total, available, os_floor_of(total))
    }

    /// The bytes of the budget a plan with its own process terms keeps back.
    pub(crate) fn os_floor(&self) -> u64 {
        self.floor
    }

    /// The bytes a payload-only admission may spend: the budget less what it keeps back.
    pub(crate) fn payload_spare(&self) -> u64 {
        self.available.saturating_sub(self.payload_floor)
    }
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
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Facts {
    pub(crate) host: HostFacts,
    pub(crate) card: Option<CardFacts>,
    pub(crate) disks: Vec<DiskFacts>,
}

impl Facts {
    /// The facts with the host read from the machine and no card or disk yet.
    pub(crate) fn read(label: &'static str, root_label: &'static str) -> Result<Self, EngError> {
        Ok(Self::with_host(HostFacts::read(label, root_label)?))
    }

    /// The facts of one start's first moment: the host and the card read together, so a plan
    /// that sizes the card and the host reads one snapshot of both. `card` returns the card's
    /// `(free, total)` bytes (the one CUDA query stays in the caller; this module holds no CUDA
    /// type).
    pub(crate) fn read_card(
        (label, root_label): (&'static str, &'static str),
        card: Result<(u64, u64), EngError>,
    ) -> Result<Self, EngError> {
        let host = HostFacts::read(label, root_label)?;
        let (free, total) = card?;
        Ok(Self::with_host(host).with_card(free, total))
    }

    /// The card's free bytes, 0 when no card was read.
    pub(crate) fn card_free(&self) -> u64 {
        self.card.map_or(0, |card| card.free)
    }

    /// The card's total bytes, 0 when no card was read.
    pub(crate) fn card_total(&self) -> u64 {
        self.card.map_or(0, |card| card.total)
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
    fn the_floor_is_a_tenth_never_under_eight_gib_and_never_over_a_quarter() {
        // (total GiB, floor bytes): a quarter below 32 GiB, 8 GiB from 32 to 80 GiB, a tenth above.
        for (total_gib, floor) in [
            (8u64, 2 * GIB),
            (16, 4 * GIB),
            (24, 6 * GIB),
            (32, 8 * GIB),
            (62, 8 * GIB),
            (80, 8 * GIB),
            (128, (128 * GIB).div_ceil(10)),
        ] {
            assert_eq!(os_floor_of(total_gib * GIB), floor, "{total_gib} GiB host");
        }
        assert_eq!(os_floor_of(0), 0);
        assert_eq!(os_floor_of(u64::MAX), u64::MAX.div_ceil(10));
    }

    #[test]
    fn a_cgroup_limit_bounds_the_host_store_budget_to_what_the_save_admission_admits() {
        let text = "MemTotal: 67108864 kB\nMemAvailable: 62914560 kB\n";
        let total = 64 * GIB;
        let floor = os_floor_of(total);

        // No cgroup limit: the machine's figure, as before.
        let free = HostFacts::from_meminfo("test", text, None).unwrap();
        assert_eq!(free.cgroup_payload_room(), None);
        let unlimited = host_store_budget_of(total, GIB, free.cgroup_payload_room());
        assert_eq!(unlimited, total - floor - GIB);

        // A fake 12 GiB cgroup remainder: the budget is the cgroup's room, not the host's.
        let fenced = HostFacts::from_meminfo("test", text, Some(12 * GIB)).unwrap();
        let room = fenced.cgroup_payload_room().expect("a limit binds");
        assert_eq!(room, 12 * GIB - fenced.process_overhead());
        let budget = host_store_budget_of(total, GIB, Some(room));
        assert_eq!(budget, room);
        assert!(
            budget < unlimited,
            "the host figure over-counts under a cgroup"
        );

        // The save admission takes exactly what the budget funds and refuses one byte more.
        assert!(fenced.admit_payload(budget, "a stored entry").is_ok());
        assert!(fenced.admit_payload(budget + 1, "a stored entry").is_err());
        assert!(fenced.admit_payload(unlimited, "a stored entry").is_err());

        // A cgroup roomier than the host leaves the host figure binding.
        let roomy = HostFacts::from_meminfo("test", text, Some(200 * GIB)).unwrap();
        assert_eq!(
            host_store_budget_of(total, GIB, roomy.cgroup_payload_room()),
            unlimited
        );

        // A cgroup nearly full leaves nothing, never an underflow.
        let full = HostFacts::from_meminfo("test", text, Some(GIB)).unwrap();
        assert_eq!(
            host_store_budget_of(total, 0, full.cgroup_payload_room()),
            0
        );
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
            // Whitespace-blind: a formatter may split the chained calls over lines.
            let text: String = std::fs::read_to_string(&file)
                .unwrap()
                .split_whitespace()
                .collect();
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

    /// One reader of the host: nothing outside `family/mem` opens `/proc/meminfo` or a cgroup
    /// file in production code (tests and benches read their own fixtures). A second reader is a
    /// second rule for an unverifiable tree and a second floor for the same RAM.
    #[test]
    fn nothing_outside_the_facts_reads_meminfo_or_the_cgroup_tree() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let mut files = Vec::new();
        rust_files(&root, &mut files);
        let needles = [
            "\"/proc/meminfo\"",
            "\"/sys/fs/cgroup\"",
            "\"memory.max\"",
            "\"memory.current\"",
            "\"cgroup.procs\"",
        ];
        let mut found = Vec::new();
        for file in files {
            let relative = file.strip_prefix(&root).unwrap().to_path_buf();
            let name = relative.to_string_lossy().to_string();
            if relative.starts_with("family/mem") || name.contains("test") || name.contains("bench")
            {
                continue;
            }
            let text = std::fs::read_to_string(&file).unwrap();
            for (number, line) in text.lines().enumerate() {
                if !line.trim_start().starts_with("//") && needles.iter().any(|n| line.contains(n))
                {
                    found.push(format!("{name}:{}: {}", number + 1, line.trim()));
                }
            }
        }
        assert!(found.is_empty(), "{found:#?}");
    }
}
