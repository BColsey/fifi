use super::*;
use std::os::unix::fs::{FileExt, OpenOptionsExt};

fn load(weights: &str) -> LoadIdentity {
    LoadIdentity {
        weights: weights.to_owned(),
        tokenizer: "tok".to_owned(),
        template: "tpl".to_owned(),
        numeric_path: "steps-fp16".to_owned(),
        layout: "delta-2x-kv-3".to_owned(),
        family: FamilyContract {
            adapter: "test.adapter",
            backend: "test.backend",
            snapshot_schema: "test.state.v1",
            payload_abi: "test.payload.v1",
            position_policy: "absolute",
        },
    }
}

#[test]
fn the_numeric_recipe_version_and_every_digest_enter_the_namespace() {
    let base = load("w").record().namespace();
    assert_eq!(load("w").record().namespace(), base);
    assert_ne!(load("w2").record().namespace(), base);
    let mut other = load("w");
    other.numeric_path = "steps-int8".into();
    assert_ne!(other.record().namespace(), base);
    // A bump of the version (simulated by the text it adds to the numeric path) changes it too.
    let recipe = format!("recipe-v{NUMERIC_RECIPE_VERSION}; steps-fp16");
    let bumped = format!("recipe-v{}; steps-fp16", NUMERIC_RECIPE_VERSION + 1);
    let digest = |text: &str| H64::from_digest(sha256(&[b"numeric", text.as_bytes()]));
    assert_ne!(digest(&recipe), digest(&bumped));
    assert_eq!(load("w").record().numeric_recipe_digest, digest(&recipe));
}

#[test]
fn a_directory_that_serves_direct_io_opens_and_a_missing_one_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    match probe_direct_io(dir.path()) {
        Ok(()) => {}
        Err(reason) => {
            eprintln!("direct I/O refused on the test directory ({reason}); the probe test skips");
            return;
        }
    }
    assert!(
        std::fs::read_dir(dir.path()).unwrap().next().is_none(),
        "the probe leaves no file"
    );
    let error = probe_direct_io(&dir.path().join("absent")).unwrap_err();
    assert!(error.contains("does not support direct I/O"), "{error}");
}

#[test]
fn a_cancel_survives_a_full_command_queue() {
    let dir = tempfile::tempdir().unwrap();
    let (mut cache, _) = KvDisk::open(&dir.path().join("c"), 2 << 30, &load("w")).unwrap();
    let spec = WriteSpec {
        tokens: tokens(2_049, 1),
        capture: Capture::Tip,
        object_bytes: vec![1 << 20, 1 << 20],
    };
    let job = cache.begin_write(spec).unwrap();
    // The queue is not what carries the cancel: the shared state records it.
    cache.abort(&job);
    assert!(cache.wait_idle(Duration::from_secs(120)));
    assert!(cache.shared.lock().aborted.is_empty());
    assert_eq!(cache.entry_count(), 0);
}

fn tokens(len: usize, seed: u32) -> Vec<u32> {
    (0..len as u32)
        .map(|i| i.wrapping_mul(31).wrapping_add(seed))
        .collect()
}

fn objects(seed: u8) -> Vec<Vec<u8>> {
    [10_000usize, 20_000, 30_000]
        .iter()
        .enumerate()
        .map(|(index, &len)| {
            (0..len)
                .map(|at| (at as u8).wrapping_mul(7).wrapping_add(seed + index as u8))
                .collect()
        })
        .collect()
}

fn store(
    cache: &mut KvDisk,
    key: &[u32],
    capture: Capture,
    parts: &[Vec<u8>],
) -> Result<(), WriteSkip> {
    let spec = WriteSpec {
        tokens: key.to_vec(),
        capture,
        object_bytes: parts.iter().map(|part| part.len() as u64).collect(),
    };
    let job = cache.begin_write(spec)?;
    let flat: Vec<u8> = parts.concat();
    for chunk in flat.chunks(7_001) {
        let mut chunk = chunk.to_vec();
        loop {
            match cache.push(&job, chunk) {
                Push::Sent => break,
                Push::Full(back) => {
                    chunk = back;
                    std::thread::sleep(Duration::from_millis(1));
                }
                Push::Dead => panic!("the writer stopped"),
            }
        }
    }
    assert!(
        cache.wait_idle(Duration::from_secs(120)),
        "the writer finished"
    );
    Ok(())
}

fn fetch(
    cache: &mut KvDisk,
    prompt: &[u32],
    shape: &[u64],
) -> Option<Result<Vec<Vec<u8>>, DiskMiss>> {
    let candidate = cache.lookup(prompt)?;
    let mut got: Vec<Vec<u8>> = shape.iter().map(|&len| vec![0; len as usize]).collect();
    let read = cache.read(&candidate, prompt, shape, &mut |object, offset, bytes| {
        got[object][offset as usize..offset as usize + bytes.len()].copy_from_slice(bytes);
        Ok(())
    });
    Some(read.map(|_| got))
}

fn shape(parts: &[Vec<u8>]) -> Vec<u64> {
    parts.iter().map(|part| part.len() as u64).collect()
}

fn extended(key: &[u32], extra: usize) -> Vec<u32> {
    let mut prompt = key.to_vec();
    prompt.extend((0..extra as u32).map(|i| 9_000_000 + i));
    prompt
}

#[test]
fn an_entry_written_behind_survives_a_reopen_and_reads_back_exact() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("cache");
    let key = tokens(1_500, 5);
    let parts = objects(3);
    {
        let (mut cache, report) = KvDisk::open(&root, 1 << 30, &load("w1")).unwrap();
        assert_eq!(report.entries, 0);
        store(&mut cache, &key, Capture::Tip, &parts).unwrap();
        assert!(matches!(
            cache.poll().as_slice(),
            [WriteEvent::Done {
                positions: 1_499,
                ..
            }]
        ));
    }
    let (mut cache, report) = KvDisk::open(&root, 1 << 30, &load("w1")).unwrap();
    assert_eq!(
        (
            report.entries,
            report.foreign,
            report.dropped,
            report.orphans
        ),
        (1, 0, 0, 0)
    );
    assert!(report.used_bytes > 60_000);
    let prompt = extended(&key, 40);
    let candidate = cache.lookup(&prompt).expect("the stored key is a prefix");
    assert_eq!(candidate.key_len, key.len());
    assert_eq!(candidate.capture, Capture::Tip);
    let got = fetch(&mut cache, &prompt, &shape(&parts)).unwrap().unwrap();
    assert_eq!(got, parts);
    // The key must be a strict prefix: the same tokens alone, or a different tail, never match.
    assert!(cache.lookup(&key).is_none());
    let mut diverged = prompt.clone();
    diverged[10] ^= 1;
    assert!(cache.lookup(&diverged).is_none());
}

#[test]
fn a_truncated_object_is_a_miss_that_removes_the_entry() {
    let dir = tempfile::tempdir().unwrap();
    let (mut cache, _) = KvDisk::open(&dir.path().join("c"), 1 << 30, &load("w1")).unwrap();
    let key = tokens(1_500, 7);
    let parts = objects(1);
    store(&mut cache, &key, Capture::Tip, &parts).unwrap();
    let object = cache.shared.lock().records[0].objects[1].0;
    let path = cache.shared.root.object_path(&object);
    std::fs::OpenOptions::new()
        .write(true)
        .open(&path)
        .unwrap()
        .set_len(19_999)
        .unwrap();
    let prompt = extended(&key, 5);
    let miss = fetch(&mut cache, &prompt, &shape(&parts))
        .unwrap()
        .unwrap_err();
    assert!(matches!(miss, DiskMiss::Corrupt(_)), "{miss}");
    assert!(
        cache.lookup(&prompt).is_none(),
        "a corrupt entry is dropped"
    );
    assert_eq!(cache.entry_count(), 0);
}

#[test]
fn a_flipped_byte_is_a_miss_that_removes_the_entry() {
    let dir = tempfile::tempdir().unwrap();
    let (mut cache, _) = KvDisk::open(&dir.path().join("c"), 1 << 30, &load("w1")).unwrap();
    let key = tokens(1_500, 9);
    let parts = objects(2);
    store(&mut cache, &key, Capture::Tip, &parts).unwrap();
    let object = cache.shared.lock().records[0].objects[2].0;
    let path = cache.shared.root.object_path(&object);
    let file = std::fs::OpenOptions::new()
        .write(true)
        .read(true)
        .open(&path)
        .unwrap();
    let mut byte = [0u8];
    file.read_exact_at(&mut byte, 12_345).unwrap();
    file.write_all_at(&[byte[0] ^ 0x40], 12_345).unwrap();
    drop(file);
    let prompt = extended(&key, 5);
    let miss = fetch(&mut cache, &prompt, &shape(&parts))
        .unwrap()
        .unwrap_err();
    assert!(matches!(miss, DiskMiss::Corrupt(_)), "{miss}");
    assert!(cache.lookup(&prompt).is_none());
}

#[test]
fn an_entry_of_another_load_is_never_offered_and_is_kept() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("c");
    let key = tokens(1_500, 11);
    let parts = objects(4);
    {
        let (mut cache, _) = KvDisk::open(&root, 1 << 30, &load("weights-a")).unwrap();
        store(&mut cache, &key, Capture::Tip, &parts).unwrap();
    }
    let prompt = extended(&key, 5);
    let (cache, report) = KvDisk::open(&root, 1 << 30, &load("weights-b")).unwrap();
    assert_eq!((report.entries, report.foreign), (0, 1));
    assert!(
        cache.lookup(&prompt).is_none(),
        "another load's entry is a miss"
    );
    drop(cache);
    for changed in [
        LoadIdentity {
            tokenizer: "tok2".into(),
            ..load("weights-a")
        },
        LoadIdentity {
            template: "tpl2".into(),
            ..load("weights-a")
        },
        LoadIdentity {
            numeric_path: "steps-q8".into(),
            ..load("weights-a")
        },
        LoadIdentity {
            layout: "other-geometry".into(),
            ..load("weights-a")
        },
    ] {
        let (cache, _) = KvDisk::open(&root, 1 << 30, &changed).unwrap();
        assert!(cache.lookup(&prompt).is_none(), "{changed:?}");
    }
    let (cache, report) = KvDisk::open(&root, 1 << 30, &load("weights-a")).unwrap();
    assert_eq!(report.entries, 1);
    assert!(cache.lookup(&prompt).is_some());
}

#[test]
fn an_incomplete_identity_cannot_open_the_cache() {
    let dir = tempfile::tempdir().unwrap();
    for broken in [
        LoadIdentity {
            weights: String::new(),
            ..load("w")
        },
        LoadIdentity {
            tokenizer: String::new(),
            ..load("w")
        },
        LoadIdentity {
            template: String::new(),
            ..load("w")
        },
    ] {
        assert!(KvDisk::open(&dir.path().join("c"), 1 << 30, &broken).is_err());
    }
}

#[test]
fn the_oldest_entry_goes_first_when_the_quota_is_reached() {
    let dir = tempfile::tempdir().unwrap();
    // One entry takes about 72 KB: the quota holds two.
    let (mut cache, _) = KvDisk::open(&dir.path().join("c"), 170_000, &load("w")).unwrap();
    let (a, b, c) = (tokens(1_500, 1), tokens(1_500, 2), tokens(1_500, 3));
    let (parts_a, parts_b, parts_c) = (objects(10), objects(20), objects(30));
    store(&mut cache, &a, Capture::Periodic, &parts_a).unwrap();
    store(&mut cache, &b, Capture::Periodic, &parts_b).unwrap();
    assert_eq!(cache.entry_count(), 2);
    // Reading `a` makes `b` the least recently used.
    let prompt_a = extended(&a, 3);
    fetch(&mut cache, &prompt_a, &shape(&parts_a))
        .unwrap()
        .unwrap();
    store(&mut cache, &c, Capture::Periodic, &parts_c).unwrap();
    assert_eq!(cache.entry_count(), 2);
    assert!(cache.lookup(&prompt_a).is_some());
    assert!(
        cache.lookup(&extended(&b, 3)).is_none(),
        "the least recently used entry went"
    );
    assert!(cache.lookup(&extended(&c, 3)).is_some());
    assert!(cache.used_bytes() <= 170_000);
    assert_eq!(cache.evicted(), 1);
    let (objects_on_disk, _) = cache.shared.root.list_objects().unwrap();
    assert_eq!(
        objects_on_disk.len(),
        6,
        "the evicted entry's objects were removed"
    );
}

#[test]
fn an_entry_larger_than_the_quota_is_skipped() {
    let dir = tempfile::tempdir().unwrap();
    let (mut cache, _) = KvDisk::open(&dir.path().join("c"), 10_000, &load("w")).unwrap();
    let skip = store(&mut cache, &tokens(1_500, 1), Capture::Tip, &objects(0)).unwrap_err();
    assert!(matches!(skip, WriteSkip::OverQuota { .. }), "{skip}");
    let short = store(&mut cache, &tokens(500, 1), Capture::Tip, &objects(0)).unwrap_err();
    assert_eq!(short, WriteSkip::Short);
}

#[test]
fn the_same_key_is_stored_once_and_a_longer_tip_replaces_its_older_tip() {
    let dir = tempfile::tempdir().unwrap();
    let (mut cache, _) = KvDisk::open(&dir.path().join("c"), 1 << 30, &load("w")).unwrap();
    let parts = objects(5);
    let short = tokens(1_500, 4);
    store(&mut cache, &short, Capture::Tip, &parts).unwrap();
    assert_eq!(
        store(&mut cache, &short, Capture::Tip, &parts),
        Err(WriteSkip::Present)
    );
    let mut longer = short.clone();
    longer.extend(tokens(300, 99));
    store(&mut cache, &longer, Capture::Tip, &parts).unwrap();
    assert_eq!(
        cache.entry_count(),
        1,
        "the older tip of the conversation was replaced"
    );
    assert_eq!(
        cache.lookup(&extended(&longer, 2)).unwrap().key_len,
        longer.len()
    );
    // A periodic entry is never replaced this way.
    let mut longest = longer.clone();
    longest.extend(tokens(300, 98));
    store(&mut cache, &longer, Capture::Periodic, &parts).unwrap_err();
    store(&mut cache, &longest, Capture::Periodic, &parts).unwrap();
    assert_eq!(cache.entry_count(), 2);
}

#[test]
fn the_longest_stored_prefix_is_chosen() {
    let dir = tempfile::tempdir().unwrap();
    let (mut cache, _) = KvDisk::open(&dir.path().join("c"), 1 << 30, &load("w")).unwrap();
    let parts = objects(6);
    let short = tokens(1_500, 4);
    let mut long = short.clone();
    long.extend(tokens(500, 7));
    store(&mut cache, &short, Capture::Periodic, &parts).unwrap();
    store(&mut cache, &long, Capture::Periodic, &parts).unwrap();
    assert_eq!(
        cache.lookup(&extended(&long, 9)).unwrap().key_len,
        long.len()
    );
    assert_eq!(
        cache.lookup(&extended(&short, 9)).unwrap().key_len,
        short.len()
    );
    // The long key equals the prompt's head but not a strict prefix of a prompt of its own length.
    assert_eq!(cache.lookup(&long).unwrap().key_len, short.len());
}

#[test]
fn a_second_process_cannot_take_an_open_directory() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("c");
    let (cache, _) = KvDisk::open(&root, 1 << 30, &load("w")).unwrap();
    let error = KvDisk::open(&root, 1 << 30, &load("w"))
        .err()
        .expect("locked");
    assert!(error.contains("another process"), "{error}");
    assert!(purge(&root).is_err(), "a purge refuses an open directory");
    drop(cache);
    assert!(KvDisk::open(&root, 1 << 30, &load("w")).is_ok());
}

#[test]
fn a_purge_empties_the_directory_and_refuses_a_directory_that_is_not_a_cache() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("c");
    {
        let (mut cache, _) = KvDisk::open(&root, 1 << 30, &load("w")).unwrap();
        store(&mut cache, &tokens(1_500, 1), Capture::Tip, &objects(0)).unwrap();
    }
    let report = purge(&root).unwrap();
    assert_eq!(report.entries, 1);
    assert!(report.bytes > 60_000);
    assert_eq!(std::fs::read_dir(&root).unwrap().count(), 0);
    assert!(purge(&root).unwrap().bytes == 0);
    let foreign = dir.path().join("documents");
    std::fs::create_dir(&foreign).unwrap();
    std::fs::write(foreign.join("thesis.txt"), b"keep").unwrap();
    assert!(purge(&foreign).is_err());
    assert!(foreign.join("thesis.txt").exists());
    assert_eq!(
        purge(&dir.path().join("absent")).unwrap(),
        PurgeReport::default()
    );
}

#[test]
fn a_manifest_that_lost_an_object_is_dropped_and_stray_objects_are_removed_at_open() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("c");
    let parts = objects(8);
    let (gone, stray) = {
        let (mut cache, _) = KvDisk::open(&root, 1 << 30, &load("w")).unwrap();
        store(&mut cache, &tokens(1_500, 1), Capture::Tip, &parts).unwrap();
        store(
            &mut cache,
            &tokens(1_500, 2),
            Capture::Periodic,
            &objects(9),
        )
        .unwrap();
        let state = cache.shared.lock();
        (state.records[0].objects[0].0, state.records[1].sha)
    };
    let root_dir = std::fs::canonicalize(&root).unwrap();
    std::fs::remove_file(root_dir.join("objects").join(format!("{gone}.bin"))).unwrap();
    // An object nobody names.
    std::fs::write(
        root_dir
            .join("objects")
            .join(format!("{}.bin", "ab".repeat(32))),
        b"x",
    )
    .unwrap();
    let _ = stray;
    let (cache, report) = KvDisk::open(&root, 1 << 30, &load("w")).unwrap();
    assert_eq!(report.dropped, 1);
    assert!(report.orphans >= 1);
    assert_eq!(cache.entry_count(), 1);
}

#[test]
fn the_default_directory_and_quota_follow_the_host() {
    assert_eq!(
        default_dir(Some("/x/cache"), Some("/home/u")),
        Some(PathBuf::from("/x/cache/fifi/kv"))
    );
    assert_eq!(
        default_dir(None, Some("/home/u")),
        Some(PathBuf::from("/home/u/.cache/fifi/kv"))
    );
    assert_eq!(default_dir(Some("relative"), Some("also/relative")), None);
    assert_eq!(default_dir(Some(""), None), None);
    let gib = 1u64 << 30;
    assert_eq!(default_quota(80 * gib), Some(20 * gib));
    assert_eq!(default_quota(16 * gib), Some(4 * gib));
    assert_eq!(default_quota(16 * gib - 4), None);
    assert_eq!(default_quota(10_000 * gib), Some(64 * gib));
    let wanted = Wanted::default();
    let resolved = resolve(&wanted, None, Some("/home/u"), |_| Some(40 * gib)).unwrap();
    assert_eq!(resolved.dir, PathBuf::from("/home/u/.cache/fifi/kv"));
    assert_eq!(resolved.quota_bytes, 10 * gib);
    assert!(resolve(&wanted, None, Some("/home/u"), |_| Some(8 * gib)).is_err());
    assert!(resolve(&wanted, None, None, |_| Some(40 * gib)).is_err());
    let exact = Wanted {
        dir: Some("/d".into()),
        quota_bytes: Some(3 * gib),
    };
    assert_eq!(
        resolve(&exact, None, None, |_| None).unwrap(),
        Resolved {
            dir: "/d".into(),
            quota_bytes: 3 * gib
        }
    );
}

fn env(dir: Option<&str>, quota: Option<&str>, switch: Option<&str>) -> KvEnv {
    KvEnv {
        dir: dir.map(str::to_owned),
        quota: quota.map(str::to_owned),
        switch: switch.map(str::to_owned),
        xdg_cache_home: None,
        home: Some("/home/u".to_owned()),
    }
}

#[test]
fn the_plan_follows_the_environment_then_the_config_then_the_build_default() {
    use crate::family::kv_disk::KvCacheSettings;
    let gib = 1u64 << 30;
    let free = |_: &Path| Some(100 * gib);
    let off = |plan: Result<Plan, String>| matches!(plan, Ok(Plan::Off(_)));
    // Nothing set: the build default decides.
    assert!(off(plan(None, &env(None, None, None), false, free)));
    let on = plan(None, &env(None, None, None), true, free).unwrap();
    assert!(matches!(
        on,
        Plan::On {
            origin: "default",
            ..
        }
    ));
    // A directory in the config or the environment turns it on.
    let named = KvCacheSettings {
        dir: Some("/c".into()),
        ..KvCacheSettings::default()
    };
    let Plan::On { resolved, origin } =
        plan(Some(&named), &env(None, None, None), false, free).unwrap()
    else {
        panic!("on")
    };
    assert_eq!(
        (resolved.dir, resolved.quota_bytes, origin),
        ("/c".into(), 25 * gib, "config")
    );
    let Plan::On { resolved, origin } = plan(
        Some(&named),
        &env(Some("/e"), Some("2147483648"), None),
        false,
        free,
    )
    .unwrap() else {
        panic!("on")
    };
    assert_eq!(
        (resolved.dir, resolved.quota_bytes, origin),
        ("/e".into(), 2 * gib, "environment")
    );
    // Off wins: the switch over everything, enabled=false over a config directory.
    assert!(off(plan(
        Some(&named),
        &env(Some("/e"), None, Some("off")),
        true,
        free
    )));
    let disabled = KvCacheSettings {
        enabled: Some(false),
        dir: Some("/c".into()),
        quota_bytes: None,
    };
    assert!(off(plan(
        Some(&disabled),
        &env(None, None, None),
        true,
        free
    )));
    // enabled=false is not undone by a directory in the environment; only the switch undoes it.
    assert!(off(plan(
        Some(&disabled),
        &env(Some("/e"), None, None),
        true,
        free
    )));
    let Plan::On { origin, .. } = plan(
        Some(&disabled),
        &env(Some("/e"), None, Some("1")),
        true,
        free,
    )
    .unwrap() else {
        panic!("on")
    };
    assert_eq!(origin, "environment");
    // enabled=true with no directory takes the default location.
    let enabled = KvCacheSettings {
        enabled: Some(true),
        ..KvCacheSettings::default()
    };
    let Plan::On { resolved, .. } =
        plan(Some(&enabled), &env(None, None, None), false, free).unwrap()
    else {
        panic!("on")
    };
    assert_eq!(resolved.dir, PathBuf::from("/home/u/.cache/fifi/kv"));
    // Too little free space keeps it off with the reason; a typo is an error.
    let tight = |_: &Path| Some(8 * gib);
    assert!(off(plan(
        Some(&enabled),
        &env(None, None, None),
        false,
        tight
    )));
    for bad in [
        env(Some("relative"), None, None),
        env(Some("/e"), Some("12"), None),
        env(Some("/e"), Some("lots"), None),
        env(None, None, Some("maybe")),
    ] {
        assert!(plan(None, &bad, false, free).is_err(), "{bad:?}");
    }
}

/// Prints the cost of each stage of a verified read of a 256 MiB object: the bare read, the
/// checksum, the digest and the whole read through the cache. Run on purpose with --ignored.
#[test]
#[ignore = "a measurement, not a check"]
fn read_stage_costs_probe() {
    let dir = tempfile::tempdir().unwrap();
    let (mut cache, _) = KvDisk::open(&dir.path().join("c"), 8 << 30, &load("w")).unwrap();
    let size = 256usize << 20;
    let part: Vec<u8> = (0..size)
        .map(|i| (i as u32).wrapping_mul(2_654_435_761) as u8)
        .collect();
    let parts = vec![vec![1u8; 4096], part];
    let key = tokens(1_500, 1);
    store(&mut cache, &key, Capture::Tip, &parts).unwrap();
    let t = Instant::now();
    let crc = disk_store::crc32c(&parts[1]);
    eprintln!(
        "crc32c   {:.0} MB/s ({crc:08x})",
        size as f64 / 1e6 / t.elapsed().as_secs_f64()
    );
    let t = Instant::now();
    let digest = Sha256::digest(&parts[1]);
    eprintln!(
        "sha256   {:.0} MB/s ({:02x})",
        size as f64 / 1e6 / t.elapsed().as_secs_f64(),
        digest[0]
    );
    // The bare O_DIRECT loop, then the verified read with a sink that drops the bytes.
    let (dep, big) = {
        let state = cache.shared.lock();
        let id = state.records[0].objects[1].0;
        (id, state.records[0].objects[1].1)
    };
    let manifest = {
        let sha = cache.shared.lock().records[0].sha;
        cache.shared.root.read_entry(&sha).unwrap()
    };
    let dependency = manifest
        .dependencies
        .iter()
        .find(|d| d.id == dep)
        .unwrap()
        .clone();
    let path = cache.shared.root.object_path(&dep);
    for round in 0..3 {
        let file = std::fs::OpenOptions::new()
            .read(true)
            .custom_flags(rustix::fs::OFlags::DIRECT.bits() as i32)
            .open(&path)
            .unwrap();
        let slot = cache.rings[0].acquire().unwrap();
        let t = Instant::now();
        let mut at = 0u64;
        while at < big {
            let want = (big - at).min(CHUNK_BYTES as u64);
            cache.rings[0].stage_span(&file, at, want, slot, 0).unwrap();
            at += want;
        }
        eprintln!(
            "bare O_DIRECT {round}: {:.0} MB/s",
            big as f64 / 1e6 / t.elapsed().as_secs_f64()
        );
        let t = Instant::now();
        disk_store::read_object_verified(
            &cache.shared.root,
            &dependency,
            &mut cache.rings[0],
            slot,
            &mut |_, _| Ok(()),
        )
        .unwrap();
        eprintln!(
            "verified, no sink {round}: {:.0} MB/s",
            big as f64 / 1e6 / t.elapsed().as_secs_f64()
        );
        let _ = cache.rings[0].abandon(slot);
    }
    let prompt = extended(&key, 3);
    for round in 0..3 {
        let t = Instant::now();
        let stats = fetch(&mut cache, &prompt, &shape(&parts)).unwrap().unwrap();
        let _ = stats;
        eprintln!(
            "verified read {round}: {:.0} MB/s",
            (size + 4096) as f64 / 1e6 / t.elapsed().as_secs_f64()
        );
    }
}

#[test]
fn the_drain_bound_follows_the_remaining_bytes_and_the_write_rate() {
    let secs = |bytes: u64, rate: Option<f64>| drain_bound(bytes, rate).as_secs_f64();
    // 1.46 GB at the assumed 20 MB/s: 73 s, three times over, and 10 s.
    assert!((secs(1_460_000_000, None) - 229.0).abs() < 0.1);
    // At a measured 400 MB/s the same bytes need 3.65 s: 20.95 s.
    assert!((secs(1_460_000_000, Some(400e6)) - 20.95).abs() < 0.01);
    // A small save still waits the 15 s floor.
    assert_eq!(secs(1_000_000, Some(400e6)), 15.0);
    // A slow disk is capped.
    assert_eq!(secs(10_000_000_000, Some(10e6)), 300.0);
    // An unusable rate falls back to the assumed one.
    assert_eq!(secs(1_460_000_000, Some(0.0)), secs(1_460_000_000, None));
    assert_eq!(
        secs(1_460_000_000, Some(f64::NAN)),
        secs(1_460_000_000, None)
    );
}

#[test]
fn the_writer_reports_a_measured_rate_after_a_save() {
    let dir = tempfile::tempdir().unwrap();
    let (mut cache, _) = KvDisk::open(&dir.path().join("c"), 1 << 30, &load("w")).unwrap();
    assert_eq!(cache.write_rate(), None);
    let part = vec![7u8; 17 << 20];
    store(
        &mut cache,
        &tokens(1_500, 1),
        Capture::Tip,
        &[vec![1u8; 4096], part],
    )
    .unwrap();
    assert_eq!(cache.write_rate(), None, "a rate is read from the reports");
    assert_eq!(cache.poll().len(), 1);
    let rate = cache.write_rate().expect("a rate after 17 MiB");
    assert!(rate > 1e4 && rate.is_finite(), "{rate}");
}

#[test]
fn the_disk_is_used_only_from_the_watermark_up() {
    // Eight retained entries: nothing below seven stored, then the oldest, then two.
    assert_eq!(watermark(8), 7);
    let spills: Vec<usize> = (0..=8).map(|n| spill_candidates(n, 8)).collect();
    assert_eq!(spills, [0, 0, 0, 0, 0, 0, 0, 1, 2]);
    // A pool of one spills its only entry; a pool of two waits until both slots are used.
    assert_eq!(watermark(1), 1);
    assert_eq!((spill_candidates(0, 1), spill_candidates(1, 1)), (0, 1));
    assert_eq!(watermark(2), 2);
    assert_eq!((spill_candidates(1, 2), spill_candidates(2, 2)), (0, 1));
    // Three retained: the watermark is two entries.
    assert_eq!((spill_candidates(1, 3), spill_candidates(2, 3)), (0, 1));
    assert_eq!(spill_candidates(3, 3), 2);
}

#[test]
fn an_exact_key_is_found_in_the_index_only_for_this_load() {
    let dir = tempfile::tempdir().unwrap();
    let (mut cache, _) = KvDisk::open(&dir.path().join("c"), 1 << 30, &load("w")).unwrap();
    let key = tokens(1_500, 3);
    assert!(!cache.has_exact(&key));
    store(&mut cache, &key, Capture::Tip, &objects(2)).unwrap();
    assert!(cache.has_exact(&key));
    assert!(!cache.has_exact(&key[..1_499]));
}
