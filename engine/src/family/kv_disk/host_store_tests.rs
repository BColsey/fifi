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

impl HostFamily for Fake {
    type Snapshot = Snap;
    type Path = u8;

    fn positions(&self, snapshot: &Snap) -> usize {
        snapshot.positions
    }
    fn host_bytes(&self, snapshot: &Snap) -> u64 {
        snapshot.data.len() as u64
    }
    fn slot(&self, snapshot: &Snap) -> Rc<dyn SlotBytes> {
        Rc::clone(&snapshot.data) as Rc<dyn SlotBytes>
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
    fn rebuild(&self, positions: usize, slab: Vec<u8>) -> Result<Snap, EngError> {
        Ok(Snap {
            positions,
            data: Rc::new(slab),
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
    disk.promote(&Fake, &mut store, &0u8, &prompt);
    assert_eq!(disk.tier.counters.restored, 1);
    let restored = store.snapshots().next().expect("the entry is in the store");
    assert_eq!(restored.positions, 1_500);
    assert_eq!(
        *restored.data, *written.1.data,
        "the bytes are the ones saved"
    );
    let tier = restore_tier(Some(&mut disk), written.0.len());
    assert_eq!(tier.source, PrefixSource::Disk);
    assert_eq!(tier.disk_bytes, Fake::bytes_of(1_500) as u64);
    // A prompt the store already covers is not read again.
    disk.promote(&Fake, &mut store, &0u8, &prompt);
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
    disk.promote(&Fake, &mut store, &0u8, &prompt);
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
    disk.promote(&Fake, &mut other, &0u8, &prompt);
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
