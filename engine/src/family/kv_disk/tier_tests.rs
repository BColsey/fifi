//! The disk tier's logic, without any family: a slot is memory, the cache is a real one in a
//! scratch directory, and an entry is three small objects.
use super::super::FamilyContract;
use super::super::{LoadIdentity, ObjectRole, spill_candidates};
use super::*;

const OBJECT: usize = 8192;

struct Memory(Vec<u8>);

impl SlotBytes for Memory {
    fn read(&self, at: usize, out: &mut [u8]) -> Result<(), EngError> {
        let end = at + out.len();
        if end > self.0.len() {
            return Err(EngError::BadRequest("read past the slot".into()));
        }
        out.copy_from_slice(&self.0[at..end]);
        Ok(())
    }
}

fn shape_of(_positions: usize) -> Result<SlabLayout, EngError> {
    Ok(SlabLayout {
        roles: vec![
            ObjectRole::FixedState { kind: "deltanet" },
            ObjectRole::Pages,
            ObjectRole::Pages,
        ],
        objects: (0..3)
            .map(|at| {
                vec![Run {
                    slab: at * OBJECT,
                    bytes: OBJECT,
                }]
            })
            .collect(),
    })
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

fn tier(dir: &std::path::Path) -> DiskTier {
    let (cache, _) = KvDisk::open(&dir.join("cache"), 1 << 30, &load()).unwrap();
    DiskTier::new("test engine", cache, 2)
}

fn key(len: usize, seed: u32) -> Vec<u32> {
    (0..len as u32)
        .map(|at| at.wrapping_mul(31).wrapping_add(seed))
        .collect()
}

fn held(key: Vec<u32>, capture: Capture, reason: Reason, seed: u8) -> Held {
    let bytes: Vec<u8> = (0..3 * OBJECT)
        .map(|at| {
            (at as u8)
                .wrapping_mul(13)
                .wrapping_add(seed)
                .wrapping_add(((at / OBJECT) as u8).wrapping_mul(41))
        })
        .collect();
    Held {
        reason,
        slot: Rc::new(Memory(bytes)),
        positions: key.len() - 1,
        key,
        capture,
        bytes: (3 * OBJECT) as u64,
    }
}

fn pump_until_idle(tier: &mut DiskTier) {
    let began = Instant::now();
    while tier.pending() {
        tier.pump(&shape_of, 1 << 30);
        assert!(
            began.elapsed() < Duration::from_secs(240),
            "the writer stalls"
        );
        std::thread::sleep(Duration::from_millis(2));
    }
}

#[test]
fn a_spill_queue_drops_a_dead_tip_and_a_periodic_entry_before_a_live_tip() {
    let dir = tempfile::tempdir().unwrap();
    let mut tier = tier(dir.path());
    let base = key(1_100, 1);
    let mut longer = base.clone();
    longer.extend(key(300, 9));
    tier.enqueue(held(key(1_100, 5), Capture::Periodic, Reason::Watermark, 1));
    tier.enqueue(held(base.clone(), Capture::Tip, Reason::Watermark, 2));
    // The same offer twice changes nothing.
    tier.enqueue(held(base.clone(), Capture::Tip, Reason::Watermark, 2));
    assert_eq!(tier.queued.len(), 2);
    // A longer tip of the conversation kills the shorter one waiting.
    tier.enqueue(held(longer.clone(), Capture::Tip, Reason::Watermark, 3));
    assert_eq!(tier.queued.len(), 2);
    assert!(tier.queued.iter().all(|held| held.key != base));
    // A full queue gives up its periodic entry for a tip, and counts it.
    tier.enqueue(held(key(1_100, 7), Capture::Tip, Reason::Watermark, 4));
    assert_eq!(
        tier.queued
            .iter()
            .filter(|held| held.capture == Capture::Tip)
            .count(),
        2
    );
    assert_eq!(tier.counters.dropped, 1);
    // A periodic offer finds only tips waiting and is itself dropped.
    tier.enqueue(held(key(1_100, 8), Capture::Periodic, Reason::Watermark, 5));
    assert_eq!(tier.counters.dropped, 2);
    assert_eq!(tier.queued.len(), 2);
}

#[test]
fn an_entry_already_on_disk_is_not_offered_again() {
    let dir = tempfile::tempdir().unwrap();
    let mut tier = tier(dir.path());
    let stored = key(1_100, 3);
    tier.enqueue(held(stored.clone(), Capture::Tip, Reason::Watermark, 1));
    pump_until_idle(&mut tier);
    assert_eq!(tier.counters.watermark.entries, 1);
    tier.enqueue(held(stored, Capture::Tip, Reason::Watermark, 1));
    assert!(tier.queued.is_empty());
}

#[test]
fn tips_are_copied_before_periodic_entries() {
    let dir = tempfile::tempdir().unwrap();
    let mut tier = tier(dir.path());
    tier.enqueue(held(key(1_100, 1), Capture::Periodic, Reason::Watermark, 1));
    tier.enqueue(held(key(1_100, 2), Capture::Tip, Reason::Watermark, 2));
    tier.pump(&shape_of, 1);
    let copying = tier.copying.as_ref().expect("a copy has begun");
    assert_eq!(copying.held.capture, Capture::Tip);
    pump_until_idle(&mut tier);
    assert_eq!(tier.counters.watermark.entries, 2);
}

#[test]
fn the_watermark_decides_how_many_entries_leave_the_disk_alone() {
    // Below the watermark nothing is offered; the tier holds no entry and writes none.
    for stored in 0..watermark(2) {
        assert_eq!(spill_candidates(stored, 2), 0, "{stored} entries");
    }
    assert!(spill_candidates(watermark(2), 2) >= 1);
}

#[test]
fn a_drain_writes_only_the_tips_and_reports_them_as_a_stop() {
    let dir = tempfile::tempdir().unwrap();
    let mut tier = tier(dir.path());
    tier.enqueue(held(key(1_100, 1), Capture::Periodic, Reason::Watermark, 1));
    let tips = vec![
        held(key(1_200, 2), Capture::Tip, Reason::Stop, 2),
        held(key(1_300, 3), Capture::Tip, Reason::Stop, 3),
    ];
    let hurry = AtomicBool::new(false);
    tier.drain(tips, 4, &shape_of, &hurry, &|_, _| Duration::from_secs(240));
    assert!(!tier.pending());
    assert_eq!(tier.counters.stop.entries, 2);
    assert_eq!(tier.counters.watermark.entries, 0);
    assert_eq!(tier.cache.entry_count(), 2);
}

#[test]
fn a_drain_gives_up_at_its_bound_and_at_a_second_signal() {
    let dir = tempfile::tempdir().unwrap();
    let mut tier = tier(dir.path());
    let hurry = AtomicBool::new(false);
    let tips = vec![held(key(1_200, 2), Capture::Tip, Reason::Stop, 2)];
    tier.drain(tips, 0, &shape_of, &hurry, &|_, _| Duration::ZERO);
    assert!(tier.pending(), "a zero bound writes nothing");
    assert_eq!(tier.counters.stop.entries, 0);

    let mut tier2 = self::tier(&dir.path().join("second"));
    let hurried = AtomicBool::new(true);
    let began = Instant::now();
    let tips = vec![held(key(1_200, 4), Capture::Tip, Reason::Stop, 4)];
    tier2.drain(tips, 0, &shape_of, &hurried, &|_, _| {
        Duration::from_secs(300)
    });
    assert!(
        began.elapsed() < Duration::from_secs(5),
        "a hurry ends a 300 s wait at once"
    );
    assert_eq!(tier2.counters.stop.entries, 0);
}

#[test]
fn slots_are_freed_for_a_save_only_when_none_is_free() {
    let dir = tempfile::tempdir().unwrap();
    let mut tier = tier(dir.path());
    tier.enqueue(held(key(1_100, 1), Capture::Periodic, Reason::Watermark, 1));
    tier.enqueue(held(key(1_100, 2), Capture::Tip, Reason::Watermark, 2));
    assert_eq!(tier.drop_spills_for_slots(1, "a save"), 0);
    assert_eq!(tier.queued.len(), 2);
    assert_eq!(tier.drop_spills_for_slots(0, "a restore"), 2);
    assert!(tier.queued.is_empty());
    assert_eq!(tier.counters.dropped, 2);
}

#[test]
fn repeated_write_failures_pause_the_tier_and_it_resumes_later() {
    let dir = tempfile::tempdir().unwrap();
    let mut tier = tier(dir.path());
    let now = Instant::now();
    tier.enqueue(held(key(1_100, 1), Capture::Tip, Reason::Watermark, 1));
    for _ in 0..WRITE_FAILURES_BEFORE_PAUSE {
        tier.note_failure(now);
    }
    assert!(tier.queued.is_empty(), "the waiting spill is dropped");
    assert!(tier.paused(now));
    tier.enqueue(held(key(1_100, 2), Capture::Tip, Reason::Watermark, 2));
    assert!(tier.queued.is_empty(), "a paused tier takes nothing");
    assert!(!tier.paused(now + WRITE_PAUSE + Duration::from_secs(1)));
    assert_eq!(tier.failures, 0);
    tier.enqueue(held(key(1_100, 3), Capture::Tip, Reason::Watermark, 3));
    assert_eq!(tier.queued.len(), 1);
}

#[test]
fn a_damaged_entry_is_a_removed_miss_and_the_next_lookup_finds_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let mut tier = tier(dir.path());
    let stored = key(1_100, 3);
    tier.enqueue(held(stored.clone(), Capture::Tip, Reason::Watermark, 6));
    pump_until_idle(&mut tier);
    let mut prompt = stored.clone();
    prompt.extend(key(40, 77));
    // A clean read hands back the bytes the slot held.
    let candidate = tier.cache.lookup(&prompt).expect("the entry is stored");
    let shape = shape_of(1_099).unwrap();
    let mut placed = vec![0u8; 3 * OBJECT];
    let stats = tier
        .read_into(&candidate, &prompt, &shape, &mut |slab, part| {
            placed[slab..slab + part.len()].copy_from_slice(part);
            Ok(())
        })
        .unwrap();
    assert_eq!(stats.bytes, (3 * OBJECT) as u64);
    let expected = held(stored.clone(), Capture::Tip, Reason::Watermark, 6);
    let mut original = vec![0u8; 3 * OBJECT];
    expected.slot.read(0, &mut original).unwrap();
    assert_eq!(placed, original);
    // A flipped byte is a miss that removes the entry; the prompt is then a cold prefill.
    let path = std::fs::read_dir(dir.path().join("cache").join("objects"))
        .unwrap()
        .next()
        .expect("an object is stored")
        .unwrap()
        .path();
    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .unwrap();
    use std::os::unix::fs::FileExt;
    let mut byte = [0u8];
    file.read_exact_at(&mut byte, 100).unwrap();
    file.write_all_at(&[byte[0] ^ 0x10], 100).unwrap();
    drop(file);
    let candidate = tier.cache.lookup(&prompt).expect("still indexed");
    let removed = tier
        .read_into(&candidate, &prompt, &shape, &mut |_, _| Ok(()))
        .unwrap_err();
    assert!(removed);
    assert_eq!(tier.counters.missed, 1);
    assert!(tier.cache.lookup(&prompt).is_none());
}

#[test]
fn the_start_line_names_the_directory_the_privacy_and_the_wait() {
    let resolved = Resolved {
        dir: "/var/cache/fifi/kv".into(),
        quota_bytes: 64 << 30,
        host_bytes: None,
    };
    let report = OpenReport::default();
    let line = start_line("test engine", &resolved, "default", &report, 4);
    for needle in [
        "/var/cache/fifi/kv",
        "exact token lists",
        "kv_cache.enabled false",
        "FIFI_KV_CACHE=0",
        "a second stop signal ends the wait",
        "up to 300 s",
    ] {
        assert!(line.contains(needle), "{needle}: {line}");
    }
}

#[test]
fn spill_picks_offer_the_oldest_entries_and_skip_a_tip_that_a_longer_tip_extends() {
    let base = key(1_100, 1);
    let mut longer = base.clone();
    longer.extend(key(300, 9));
    let other = key(1_100, 5);
    // Oldest first: a tip, an unrelated block entry, the same conversation's longer tip.
    let entries = [
        Stored {
            key: &base,
            tip: true,
        },
        Stored {
            key: &other,
            tip: false,
        },
        Stored {
            key: &longer,
            tip: true,
        },
    ];
    // Two to offer: the first is dead (its conversation continued), so only the block entry.
    assert_eq!(spill_picks(&entries, 2), vec![1]);
    assert_eq!(spill_picks(&entries, 3), vec![1, 2]);
    assert_eq!(spill_picks(&entries, 0), Vec::<usize>::new());
    // A block entry is never dead, whatever extends it.
    let block_then_tip = [
        Stored {
            key: &base,
            tip: false,
        },
        Stored {
            key: &longer,
            tip: true,
        },
    ];
    assert_eq!(spill_picks(&block_then_tip, 2), vec![0, 1]);
}

#[test]
fn stop_picks_are_the_live_tips_newest_first() {
    let base = key(1_100, 1);
    let mut longer = base.clone();
    longer.extend(key(300, 9));
    let other = key(1_200, 5);
    let block = key(1_100, 7);
    let entries = [
        Stored {
            key: &base,
            tip: true,
        },
        Stored {
            key: &block,
            tip: false,
        },
        Stored {
            key: &other,
            tip: true,
        },
        Stored {
            key: &longer,
            tip: true,
        },
    ];
    // The dead tip and the block entry stay in memory; the newest tip goes first.
    assert_eq!(stop_picks(&entries), vec![3, 2]);
    assert_eq!(stop_picks(&[]), Vec::<usize>::new());
}

#[test]
fn memory_is_short_only_when_the_host_cannot_fund_another_entry_above_the_reserve() {
    let gib = 1u64 << 30;
    assert!(!host_memory_is_short(Some(40 * gib), gib, 2 * gib));
    assert!(!host_memory_is_short(Some(3 * gib), gib, 2 * gib));
    assert!(host_memory_is_short(Some(3 * gib - 1), gib, 2 * gib));
    // A host that cannot be read counts as short: no entry is lost for lack of a look.
    assert!(host_memory_is_short(None, gib, 2 * gib));
}

#[test]
fn a_restore_retries_only_while_the_entry_it_tried_was_damaged() {
    let mut tries = 0;
    assert_eq!(
        retry_while_removed(3, || {
            tries += 1;
            tries < 3
        }),
        3
    );
    // The first lookup that finds nothing or restores ends it.
    assert_eq!(retry_while_removed(3, || false), 1);
    // Three damaged entries in a row end at the limit, not after it.
    let mut damaged = 0;
    assert_eq!(
        retry_while_removed(3, || {
            damaged += 1;
            true
        }),
        3
    );
    assert_eq!(damaged, 3);
}

#[test]
fn a_stop_longer_than_a_service_managers_timeout_says_so() {
    assert_eq!(stop_timeout_note(Duration::from_secs(9)), "");
    let docker = stop_timeout_note(Duration::from_secs(30));
    assert!(
        docker.contains("Docker") && !docker.contains("systemd"),
        "{docker}"
    );
    let both = stop_timeout_note(Duration::from_secs(200));
    assert!(
        both.contains("Docker") && both.contains("systemd"),
        "{both}"
    );
}

#[test]
fn the_drain_names_its_rate_and_where_it_came_from() {
    // Captured stderr is not available to a unit test; the rate the line is built from is: a
    // fresh cache has only the probe's.
    let dir = tempfile::tempdir().unwrap();
    let tier = tier(dir.path());
    let (rate, source) = tier.cache.write_rate();
    assert_eq!(source, crate::family::kv_disk::RateSource::Probe);
    assert!(rate > 0.0 && source.describe().contains("probe"));
}
