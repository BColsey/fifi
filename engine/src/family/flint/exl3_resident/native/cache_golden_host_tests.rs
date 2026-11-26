//! The pick-replay golden of the Flint resident cache's row: one `LayerCache` over a tiny
//! host store, replayed through the full pass protocol (`prepare`, `copy_ready`, `slot`,
//! `release`) on a deterministic pick stream with duplicates, plus the arms the prose tests
//! describe but a digest pins forever — the touch order, `swap_in` with its refusals, the
//! refusal atomicity (`an Err changes nothing`), and the poison/reset cycle. The row is the
//! shared `LayerLru` (state: expert->slot, resident slots, recency LRU-first); the digests
//! move only when a decision moves.
//!
//! Host-only, modelless, GPUless. `FIFI_GOLDEN_UPDATE=1` rewrites this family's golden
//! `testdata/pick-goldens.json` (own file, same mechanism as the Flash-Next pick
//! golden); without it the compile-time pin is compared.

use super::*;
use sha2::{Digest, Sha256};
use std::fmt::Write;

const SLOTS: usize = 8;
const STRIDE: usize = 6;
const CALLS: usize = 120;
const PICKS: usize = 5;

fn digest(text: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    format!("{:x}", hasher.finalize())
}

/// One fresh cache: 256 experts, `SLOTS` slots of `STRIDE` bytes, layer 0, stream 1,
/// published at identity.
fn cache() -> LayerCache {
    let store = HostStore::build(HostLayout::new(&[STRIDE], SLOTS).unwrap(), |_, _, range| {
        Ok(range.bytes)
    })
    .unwrap();
    let mut cache = LayerCache::new(&store, 0, 1).unwrap();
    cache.publish_initial(1, |_| Ok(())).unwrap();
    cache
}

/// The row as two lines: resident experts in slot order, then the recency order (LRU first).
fn row_text(cache: &LayerCache) -> String {
    let (slot_of, resident, recency) = cache.row.state();
    let experts: Vec<i32> = (0..EXPERTS)
        .map(|e| slot_of[e])
        .filter(|s| *s >= 0)
        .collect();
    format!("res={resident:?} recency={recency:?} experts={experts:?}")
}

/// One call's picks: `PICKS` distinct in-range experts (the stream arm stays valid;
/// refusals get their own arm).
fn call_picks(seed: u64, call: usize) -> Vec<i32> {
    let mut state = seed ^ (call as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    let mut picks = Vec::with_capacity(PICKS);
    while picks.len() < PICKS {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let expert = (state >> 33) as usize % EXPERTS;
        if !picks.contains(&(expert as i32)) {
            picks.push(expert as i32);
        }
    }
    picks
}

/// Drives the full pass protocol for one call and renders it.
fn run_call(cache: &mut LayerCache, call: usize, picks: &[i32], text: &mut String) {
    let key = cache.prepare(picks, usize::MAX).unwrap();
    let distinct = picks
        .iter()
        .collect::<std::collections::BTreeSet<_>>()
        .len();
    let _ = writeln!(
        text,
        "call {call} gen={} picks={picks:?} distinct={distinct}",
        key.generation
    );
    let mut copies = Vec::new();
    let ready = cache
        .copy_ready(key, 1, |misses| {
            copies = misses.to_vec();
            Ok(())
        })
        .unwrap();
    for miss in &copies {
        let _ = writeln!(
            text,
            "copy {call} expert={} slot={} src={} dst={}",
            miss.expert, miss.slot, miss.source.offset, miss.destination.offset
        );
    }
    let first = picks[0] as u16;
    let _ = writeln!(
        text,
        "slot {call} expert={first} slot={}",
        cache.slot(ready, first).unwrap()
    );
    cache.release(ready, 1, || Ok(())).unwrap();
    let _ = writeln!(text, "row {call} {}", row_text(cache));
}

/// The stream arm: `CALLS` protocol-complete calls on one cache.
fn stream_text(seed: u64) -> String {
    let mut cache = cache();
    let mut text = String::new();
    let _ = writeln!(text, "row start {}", row_text(&cache));
    for call in 0..CALLS {
        let picks = call_picks(seed, call);
        run_call(&mut cache, call, &picks, &mut text);
    }
    text
}

/// Duplicates keep every raw occurrence in the order; the recency a pass leaves is the
/// last-occurrence order. A follow-up miss pins that: expert 5 (last-touched) survives,
/// the first-touched pair evicts.
fn duplicates_text() -> String {
    let mut cache = cache();
    let mut text = String::new();
    run_call(&mut cache, 0, &[5, 5, 3, 5], &mut text);
    run_call(&mut cache, 1, &[100, 3], &mut text);
    run_call(&mut cache, 2, &[101], &mut text);
    text
}

/// `touch` re-orders the next evictions without any copy; the missed pass after it evicts
/// the untouched oldest first.
fn touch_text() -> String {
    let mut cache = cache();
    let mut text = String::new();
    cache.touch(&[2, 0, 1]).unwrap();
    let _ = writeln!(text, "touched [2, 0, 1] {}", row_text(&cache));
    run_call(&mut cache, 0, &[100], &mut text);
    run_call(&mut cache, 1, &[101], &mut text);
    run_call(&mut cache, 2, &[102], &mut text);
    text
}

/// `swap_in` and its refusals: each refusal leaves the row byte-identical.
fn swap_text() -> String {
    let mut cache = cache();
    let mut text = String::new();
    for (incoming, outgoing) in [(9u16, 1u16), (10, 0), (11, 2)] {
        let copy = cache.swap_in(incoming, outgoing).unwrap();
        let _ = writeln!(
            text,
            "swap in={incoming} out={outgoing} slot={} src={} dst={}",
            copy.slot, copy.source.offset, copy.destination.offset
        );
        let _ = writeln!(text, "row {}", row_text(&cache));
    }
    let before = row_text(&cache);
    for (incoming, outgoing, why) in [
        (9u16, 2u16, "incoming already resident"),
        (12, 1, "outgoing holds no slot"),
        (256, 2, "incoming out of range"),
    ] {
        let error = cache.swap_in(incoming, outgoing).unwrap_err();
        let unchanged = row_text(&cache) == before;
        let _ = writeln!(
            text,
            "swap-refused in={incoming} out={outgoing} why={why} unchanged={unchanged} err={error:?}"
        );
    }
    text
}

/// The refusal arm: unknown ids, out-of-range, over-capacity union, the byte bound, a
/// foreign stream, a wrong-phase call — each atomic (row unchanged).
fn refusals_text() -> String {
    let mut cache = cache();
    let mut text = String::new();
    let cases: [(&str, Vec<i32>, usize); 4] = [
        ("unknown id", vec![-1], usize::MAX),
        ("out of range", vec![EXPERTS as i32], usize::MAX),
        (
            "union over capacity",
            (0..=SLOTS).map(|e| e as i32).collect(),
            usize::MAX,
        ),
        ("byte bound", vec![100], STRIDE - 1),
    ];
    for (why, picks, bound) in cases {
        let before = row_text(&cache);
        let error = cache.prepare(&picks, bound).unwrap_err();
        let unchanged = row_text(&cache) == before;
        let _ = writeln!(
            text,
            "refused why={why} unchanged={unchanged} err={error:?}"
        );
    }
    // A prepared pass blocks re-preparation and swaps; release clears it.
    let key = cache.prepare(&[0], usize::MAX).unwrap();
    let error = cache.prepare(&[1], usize::MAX).unwrap_err();
    let _ = writeln!(text, "refused why=phase err={error:?}");
    let wrong_stream = cache.copy_ready(key, 2, |_| Ok(()));
    let _ = writeln!(
        text,
        "wrong-stream copy_ready err={:?}",
        wrong_stream.unwrap_err()
    );
    let ready = cache.copy_ready(key, 1, |_| Ok(())).unwrap();
    let outside = cache.slot(ready, 1);
    let _ = writeln!(text, "outside-pass slot err={:?}", outside.unwrap_err());
    cache.release(ready, 1, || Ok(())).unwrap();
    let ok = cache.prepare(&[1], usize::MAX).is_ok();
    let _ = writeln!(text, "after-release reprepare ok={ok}");
    text
}

/// Poison mid-pass refuses everything; `reset_after_drain` returns the identity row with
/// the generation advanced, and the cache serves passes again.
fn poison_text() -> String {
    let mut cache = cache();
    let mut text = String::new();
    let _key = cache.prepare(&[100], usize::MAX).unwrap();
    cache.poison();
    let refused = cache.prepare(&[1], usize::MAX).unwrap_err();
    let _ = writeln!(text, "poisoned prepare err={refused:?}");
    let _ = writeln!(text, "poisoned row {}", row_text(&cache));
    cache.reset_after_drain(1, || Ok(())).unwrap();
    let (slot_of, _, _) = cache.row.state();
    let _ = writeln!(
        text,
        "reset identity={}",
        (0..SLOTS).all(|s| slot_of[s] == s as i32)
    );
    // The reset lands on the Initial phase: the row republishes before it serves again.
    cache.publish_initial(1, |_| Ok(())).unwrap();
    run_call(&mut cache, 0, &[100, 7], &mut text);
    text
}

/// The digests of this family's arms.
pub(super) fn sections() -> serde_json::Map<String, serde_json::Value> {
    let mut map = serde_json::Map::new();
    for (name, text) in [
        ("stream", stream_text(0xfeed_0011)),
        ("duplicates", duplicates_text()),
        ("touch", touch_text()),
        ("swap", swap_text()),
        ("refusals", refusals_text()),
        ("poison-reset", poison_text()),
    ] {
        map.insert(name.into(), digest(&text).into());
    }
    map
}

/// Same capture mechanism as the Flash-Next pick golden (see that file's doc): a
/// `create_new` guard beside the golden, atomic replace, sorted keys.
fn with_golden_lock<R>(path: &std::path::Path, update: impl FnOnce() -> R) -> R {
    let lock = path.with_extension("lock");
    let began = std::time::Instant::now();
    while std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&lock)
        .is_err()
    {
        if began.elapsed() > std::time::Duration::from_secs(60) {
            panic!(
                "the pick-goldens update lock {} is held or stale; delete it if no capture runs",
                lock.display()
            );
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    let result = update();
    let _ = std::fs::remove_file(&lock);
    result
}

#[test]
fn the_pick_replay_holds_its_golden_digests() {
    const GOLDENS_TEXT: &str = include_str!("../../testdata/pick-goldens.json");
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src/family/flint/testdata/pick-goldens.json");
    let sections = sections();
    if std::env::var_os("FIFI_GOLDEN_UPDATE").is_some_and(|value| !value.is_empty()) {
        with_golden_lock(&path, || {
            let text = std::fs::read_to_string(&path)
                .unwrap_or_else(|error| panic!("{}: {error}", path.display()));
            let mut goldens: serde_json::Value =
                serde_json::from_str(&text).expect("the committed pick goldens on disk");
            let mut arms: std::collections::BTreeMap<&str, serde_json::Value> =
                std::collections::BTreeMap::new();
            for (name, sha) in &sections {
                arms.insert(name.as_str(), sha.clone());
            }
            goldens["arms"] = serde_json::Value::Object(
                arms.into_iter().map(|(k, v)| (k.to_string(), v)).collect(),
            );
            let tmp = path.with_extension("json.tmp");
            let mut text = serde_json::to_string_pretty(&goldens).expect("serialize");
            text.push('\n');
            std::fs::write(&tmp, text).expect("write the update beside the golden");
            std::fs::rename(&tmp, &path).expect("the atomic replace");
        });
        println!("UPDATED flint row goldens in {}", path.display());
        return;
    }
    let pinned: serde_json::Value =
        serde_json::from_str(GOLDENS_TEXT).expect("the committed pick goldens");
    let pinned_arms = pinned["arms"]
        .as_object()
        .unwrap_or_else(|| panic!("no flint row arms; capture with FIFI_GOLDEN_UPDATE=1"));
    let mut drifted: Vec<String> = Vec::new();
    for (name, sha) in &sections {
        let pinned_sha = pinned_arms.get(name).and_then(|v| v.as_str()).unwrap_or("");
        if pinned_sha != sha.as_str().unwrap_or("") {
            drifted.push(format!("{name}: pinned {pinned_sha:?}, computed {sha}"));
        }
    }
    for name in pinned_arms.keys() {
        if !sections.contains_key(name) {
            drifted.push(format!("{name}: pinned but no longer computed — stale arm"));
        }
    }
    assert!(
        drifted.is_empty(),
        "the Flint resident row moved; sections:\n{}",
        drifted.join("\n")
    );
    println!("GOLDEN MATCH flint row");
}
