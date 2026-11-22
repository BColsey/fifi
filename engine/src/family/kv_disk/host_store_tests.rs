//! The host-store adapter with a family of plain byte vectors: a real cache in a scratch
//! directory, a real prefix store, no device.
use super::*;
use crate::family::kv_disk::{FamilyContract, ObjectRole, Run};
use crate::family::session::StoreLimits;

const CELL: usize = 3_000;

#[derive(Clone)]
struct Snap {
    positions: usize,
    data: Rc<Vec<u8>>,
}

/// An entry of `positions` is `positions * 3 * CELL / 3` bytes cut into three objects.
struct Fake;

impl Fake {
    fn bytes_of(positions: usize) -> usize {
        positions * CELL
    }
}

thread_local! {
    /// How many times a spill asked the family to ready a snapshot for a write.
    static PREPARES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

impl HostFamily for Fake {
    fn prepare(&self, _snapshot: &Snap) -> Result<Option<Rc<dyn SlotBytes>>, EngError> {
        PREPARES.with(|count| count.set(count.get() + 1));
        Ok(None)
    }

    type Snapshot = Snap;
    type Path = u8;
    type Landing = Vec<u8>;

    fn view(&self, snapshot: &Snap) -> Option<SnapshotView> {
        Some(SnapshotView {
            positions: snapshot.positions,
            bytes: snapshot.data.len() as u64,
            slot: Rc::clone(&snapshot.data) as Rc<dyn SlotBytes>,
        })
    }
    fn layout(&self, positions: usize) -> Result<SlabLayout, EngError> {
        if positions > 4_000 {
            return Err(EngError::BadRequest("longer than the capacity".into()));
        }
        let third = Self::bytes_of(positions) / 3;
        Ok(SlabLayout {
            objects: (0..3)
                .map(|at| {
                    vec![Run {
                        slab: at * third,
                        bytes: third,
                    }]
                })
                .collect(),
            roles: vec![
                ObjectRole::FixedState { kind: "deltanet" },
                ObjectRole::Pages,
                ObjectRole::Pages,
            ],
        })
    }
    fn begin_landing(
        &mut self,
        _tier: &mut DiskTier,
        _positions: usize,
        layout: &SlabLayout,
    ) -> Result<Vec<u8>, EngError> {
        vec_landing("fake", layout, None)
    }
    fn place(&self, landing: &mut Vec<u8>, slab: usize, part: &[u8]) -> Result<(), EngError> {
        place_in_vec(landing, slab, part)
    }
    fn finish_landing(&mut self, landing: Vec<u8>, positions: usize) -> Result<Snap, EngError> {
        Ok(Snap {
            positions,
            data: Rc::new(landing),
        })
    }
}

fn load() -> LoadIdentity {
    LoadIdentity {
        weights: "w".into(),
        tokenizer: "t".into(),
        template: "p".into(),
        numeric_path: "steps".into(),
        layout: "l".into(),
        family: FamilyContract {
            adapter: "test",
            backend: "host",
            snapshot_schema: "test.state.v1",
            payload_abi: "test.payload.v1",
            position_policy: "absolute",
            render_mode: "chat",
        },
    }
}

fn key(len: usize, seed: u32) -> Vec<u32> {
    (0..len as u32)
        .map(|at| at.wrapping_mul(31).wrapping_add(seed))
        .collect()
}

fn snap(positions: usize, seed: u8) -> Snap {
    Snap {
        positions,
        data: Rc::new(
            (0..Fake::bytes_of(positions))
                .map(|at| (at as u8).wrapping_mul(7).wrapping_add(seed))
                .collect(),
        ),
    }
}

fn store(entries: usize) -> PrefixStore<Snap, u8> {
    PrefixStore::with_limits(StoreLimits { roots: 0, entries })
}

/// A stop that waits as long as the disk needs: the tests run beside others that write gigabytes.
const PATIENT: fn(u64, Option<f64>) -> std::time::Duration =
    |_, _| std::time::Duration::from_secs(300);

fn open(dir: &std::path::Path, retained: usize) -> HostStoreDisk {
    let resolved = Resolved {
        dir: dir.join("cache"),
        quota_bytes: 1 << 30,
        host_bytes: None,
    };
    HostStoreDisk::open("fake", &resolved, "test", &load(), retained).expect("the tier opens")
}

fn pump_until_idle(disk: &mut HostStoreDisk) {
    let began = Instant::now();
    while disk.pending() {
        disk.pump(&Fake);
        assert!(began.elapsed().as_secs() < 240, "the writer stalls");
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
}

fn put(store: &mut PrefixStore<Snap, u8>, positions: usize, seed: u32, kind: SaveKind) -> Vec<u32> {
    let key = key(positions + 1, seed);
    let pending = *key.last().unwrap();
    assert!(
        store
            .save_loaded(&key, pending, 0u8, kind, snap(positions, seed as u8))
            .unwrap()
    );
    key
}

#[test]
fn nothing_is_written_while_the_host_has_memory_and_a_stop_writes_the_tips() {
    let dir = tempfile::tempdir().unwrap();
    let mut disk = open(dir.path(), 2);
    let mut store = store(2);
    put(&mut store, 1_500, 1, SaveKind::Tip);
    put(&mut store, 1_600, 2, SaveKind::Periodic);
    // At the watermark with memory to spare: no write.
    disk.spill_over_watermark(&Fake, &store, false);
    assert!(!disk.pending());
    assert_eq!(disk.tier.cache.entry_count(), 0);
    // A graceful stop writes the live tip only.
    disk.drain_within(&Fake, &store, &PATIENT);
    assert_eq!(disk.tier.cache.entry_count(), 1);
    assert_eq!(disk.tier.counters.stop.entries, 1);
}

#[test]
fn a_short_host_spills_the_oldest_entries_past_the_watermark() {
    let dir = tempfile::tempdir().unwrap();
    let mut disk = open(dir.path(), 2);
    let mut store = store(2);
    let first = put(&mut store, 1_500, 1, SaveKind::Periodic);
    put(&mut store, 1_600, 2, SaveKind::Periodic);
    disk.spill_over_watermark(&Fake, &store, true);
    pump_until_idle(&mut disk);
    assert_eq!(
        disk.tier.counters.watermark.entries, 1,
        "one entry is pushed out"
    );
    assert!(
        disk.tier.cache.has_exact(&first),
        "the least recently used one"
    );
}

#[test]
fn entries_over_the_byte_bound_go_to_disk_oldest_first() {
    // Sizes oldest first: the bound keeps what fits of the newest, and writes the rest.
    assert_eq!(over_budget_picks(&[10, 10, 10], 30), Vec::<usize>::new());
    assert_eq!(over_budget_picks(&[10, 10, 10], 25), vec![0]);
    assert_eq!(over_budget_picks(&[10, 10, 10], 10), vec![0, 1]);
    assert_eq!(over_budget_picks(&[10, 10, 10], 0), vec![0, 1, 2]);
    assert_eq!(over_budget_picks(&[], 0), Vec::<usize>::new());

    let dir = tempfile::tempdir().unwrap();
    let mut disk = open(dir.path(), 8);
    let mut store = store(8);
    let first = put(&mut store, 1_500, 1, SaveKind::Periodic);
    let second = put(&mut store, 1_600, 2, SaveKind::Periodic);
    let third = put(&mut store, 1_700, 3, SaveKind::Periodic);
    // No bound, and a host with room: the count watermark is far off, nothing is written.
    disk.spill_over_budget(&Fake, &store);
    assert!(!disk.pending());
    // A bound that holds the newest two: the oldest goes.
    disk.host_bytes = Some((Fake::bytes_of(1_600) + Fake::bytes_of(1_700)) as u64);
    disk.spill_over_budget(&Fake, &store);
    pump_until_idle(&mut disk);
    assert!(disk.tier.cache.has_exact(&first));
    assert!(!disk.tier.cache.has_exact(&second));
    assert!(!disk.tier.cache.has_exact(&third));
    // A bound below the only entry: it is written.
    let dir = tempfile::tempdir().unwrap();
    let mut disk = open(dir.path(), 8);
    let mut alone = self::store(8);
    let only = put(&mut alone, 1_700, 3, SaveKind::Periodic);
    disk.host_bytes = Some(10);
    disk.spill_over_budget(&Fake, &alone);
    pump_until_idle(&mut disk);
    assert!(disk.tier.cache.has_exact(&only));
}

#[test]
fn a_restart_restores_what_a_stop_wrote_byte_for_byte() {
    let dir = tempfile::tempdir().unwrap();
    let written = {
        let mut disk = open(dir.path(), 2);
        let mut store = store(2);
        let tip = put(&mut store, 1_500, 3, SaveKind::Tip);
        disk.drain_within(&Fake, &store, &PATIENT);
        (tip, snap(1_500, 3))
    };
    // A new process: an empty store and the same directory.
    let mut disk = open(dir.path(), 2);
    let mut store = store(2);
    let mut prompt = written.0.clone();
    prompt.extend(key(50, 9));
    disk.promote(&mut Fake, &mut store, &0u8, &prompt);
    assert_eq!(disk.tier.counters.restored, 1);
    let restored = store.snapshots().next().expect("the entry is in the store");
    assert_eq!(restored.positions, 1_500);
    assert_eq!(
        *restored.data, *written.1.data,
        "the bytes are the ones saved"
    );
    let tier = restore_tier(Some(&mut disk), written.0.len(), None);
    assert_eq!(tier.source, PrefixSource::Disk);
    assert_eq!(tier.disk_bytes, Fake::bytes_of(1_500) as u64);
    // A prompt the store already covers is not read again.
    disk.promote(&mut Fake, &mut store, &0u8, &prompt);
    assert_eq!(disk.tier.counters.restored, 1);
}

#[test]
fn a_damaged_entry_is_a_miss_that_leaves_the_store_empty() {
    let dir = tempfile::tempdir().unwrap();
    let tip = {
        let mut disk = open(dir.path(), 2);
        let mut store = store(2);
        let tip = put(&mut store, 1_500, 4, SaveKind::Tip);
        disk.drain_within(&Fake, &store, &PATIENT);
        tip
    };
    // Flip a byte in the second object of the stored entry.
    let objects = dir.path().join("cache").join("objects");
    let mut files: Vec<_> = std::fs::read_dir(&objects)
        .unwrap()
        .flatten()
        .map(|item| item.path())
        .collect();
    files.sort();
    let victim = files.last().unwrap();
    let mut bytes = std::fs::read(victim).unwrap();
    let at = bytes.len() / 2;
    bytes[at] ^= 0x40;
    std::fs::write(victim, bytes).unwrap();
    let mut disk = open(dir.path(), 2);
    let mut store = store(2);
    let mut prompt = tip.clone();
    prompt.extend(key(50, 9));
    disk.promote(&mut Fake, &mut store, &0u8, &prompt);
    assert_eq!(disk.tier.counters.restored, 0);
    assert!(store.snapshots().next().is_none());
    assert_eq!(
        disk.tier.cache.entry_count(),
        0,
        "the damaged entry is removed"
    );
}

#[test]
fn a_prompt_the_load_cannot_hold_is_a_logged_miss() {
    let dir = tempfile::tempdir().unwrap();
    let mut disk = open(dir.path(), 2);
    let mut store = store(2);
    let tip = put(&mut store, 4_500, 5, SaveKind::Tip);
    // The family refuses to lay out an entry past its capacity: nothing is queued or written.
    disk.drain_within(&Fake, &store, &PATIENT);
    assert_eq!(disk.tier.cache.entry_count(), 0);
    let mut prompt = tip;
    prompt.extend(key(10, 1));
    let mut other = store_empty();
    disk.promote(&mut Fake, &mut other, &0u8, &prompt);
    assert_eq!(disk.tier.counters.restored, 0);
}

fn store_empty() -> PrefixStore<Snap, u8> {
    store(2)
}

#[test]
fn a_short_snapshot_or_a_key_of_the_wrong_length_is_not_offered() {
    let short = snap(10, 1);
    assert!(held_of(&Fake, &key(11, 1), SaveKind::Tip, &short, Reason::Stop).is_none());
    let long = snap(1_500, 1);
    assert!(held_of(&Fake, &key(1_500, 1), SaveKind::Tip, &long, Reason::Stop).is_none());
    let held = held_of(&Fake, &key(1_501, 1), SaveKind::Tip, &long, Reason::Stop).unwrap();
    assert_eq!((held.positions, held.capture), (1_500, Capture::Tip));
    assert_eq!(held.bytes, Fake::bytes_of(1_500) as u64);
}

#[test]
fn the_budget_bounds_what_the_system_has_free() {
    assert_eq!(bounded_by_cap(Some(10), 100, 70), Some(10));
    assert_eq!(bounded_by_cap(Some(50), 100, 70), Some(30));
    assert_eq!(bounded_by_cap(Some(50), 100, 100), Some(0));
    assert_eq!(
        bounded_by_cap(Some(50), 100, 130),
        Some(0),
        "an engine over its cap leaves nothing"
    );
    assert_eq!(bounded_by_cap(None, 100, 70), None);
}

#[test]
fn an_entry_already_on_disk_is_not_prepared_again_by_the_next_save() {
    let dir = tempfile::tempdir().unwrap();
    let mut disk = open(dir.path(), 8);
    let mut held = store(8);
    let only = put(&mut held, 1_700, 3, SaveKind::Periodic);
    disk.host_bytes = Some(10);
    PREPARES.with(|count| count.set(0));
    disk.spill_over_budget(&Fake, &held);
    assert_eq!(
        PREPARES.with(|count| count.get()),
        1,
        "the first save prepares it"
    );
    // While it is queued, a second save asks again and prepares nothing.
    disk.spill_over_budget(&Fake, &held);
    assert_eq!(PREPARES.with(|count| count.get()), 1, "queued");
    pump_until_idle(&mut disk);
    assert!(disk.tier.cache.has_exact(&only));
    // Once it is on disk, every later save leaves it alone.
    disk.spill_over_budget(&Fake, &held);
    disk.spill_over_budget(&Fake, &held);
    assert_eq!(PREPARES.with(|count| count.get()), 1, "on disk");
}

#[test]
fn byte_bound_spills_leave_the_pinned_roots_for_last_and_say_why() {
    let dir = tempfile::tempdir().unwrap();
    let mut disk = open(dir.path(), 8);
    let mut held: PrefixStore<Snap, u8> = PrefixStore::with_limits(StoreLimits {
        roots: 2,
        entries: 8,
    });
    let root = put(&mut held, 1_600, 1, SaveKind::Root);
    let oldest = put(&mut held, 1_500, 2, SaveKind::Periodic);
    put(&mut held, 1_700, 3, SaveKind::Periodic);
    // One byte under what the store holds: one entry must go, and it is the oldest unpinned one.
    let total: usize = [1_600, 1_500, 1_700].map(Fake::bytes_of).iter().sum();
    disk.host_bytes = Some(total as u64 - 1);
    disk.spill_over_budget(&Fake, &held);
    pump_until_idle(&mut disk);
    assert!(
        disk.tier.cache.has_exact(&oldest),
        "the oldest unpinned entry"
    );
    assert!(!disk.tier.cache.has_exact(&root), "the root stays for last");
    assert_eq!(disk.tier.counters.budget.entries, 1);
    assert_eq!(disk.tier.counters.watermark.entries, 0);
    assert!(
        disk.tier
            .counters
            .summary()
            .contains("over the byte budget"),
        "the log line names the reason"
    );
}

#[test]
fn an_opened_tier_takes_the_bound_the_plan_resolved() {
    for bound in [Some(1_234u64), None] {
        let dir = tempfile::tempdir().unwrap();
        let resolved = Resolved {
            dir: dir.path().join("cache"),
            quota_bytes: 1 << 30,
            host_bytes: bound,
        };
        let disk = HostStoreDisk::open("fake", &resolved, "test", &load(), 2).unwrap();
        assert_eq!(disk.host_bytes, bound);
    }
}

#[test]
fn the_default_budget_leaves_out_what_the_engine_holds() {
    let none = default_host_budget(0).expect("the machine's memory is known");
    let held = default_host_budget(1 << 30).expect("the machine's memory is known");
    assert_eq!(none - held, 1 << 30);
    assert_eq!(default_host_budget(u64::MAX), Some(0), "never below zero");
}

/// A vector-landed family with no planner budget, and one with one.
struct Planned(Option<u64>);

impl VecFamily for Planned {
    type Snapshot = Snap;
    type Path = u8;
    const LABEL: &'static str = "planned";

    fn longest_entry(&self) -> usize {
        4_000
    }
    fn part_bytes(&self, positions: usize) -> Result<Vec<usize>, EngError> {
        Ok(vec![Fake::bytes_of(positions)])
    }
    fn view_of(&self, _snapshot: &Snap) -> Option<SnapshotView> {
        None
    }
    fn finish(&mut self, _landing: Vec<u8>, positions: usize) -> Result<Snap, EngError> {
        Ok(snap(positions, 0))
    }
    fn budget_bytes(&self) -> Option<u64> {
        self.0
            .or_else(|| default_host_budget(HostFamily::resident_bytes(self)))
    }
    fn resident(&self) -> u64 {
        4_096
    }
}

#[test]
fn a_family_budget_reaches_the_tier_through_the_shared_trait() {
    assert_eq!(HostFamily::host_budget(&Planned(Some(77))), Some(77));
    assert_eq!(
        HostFamily::host_budget(&Planned(None)),
        default_host_budget(4_096),
        "without a planner budget the machine's default, less what the family holds"
    );
}
