//! The sequence, the position, the poison rule and the chunk accounting of the chunk driver, on
//! fakes.

use super::ChunkDriver;
use crate::family::flash_next::cuda::ple_feed::PleKeyValue;
use crate::{
    EngError,
    family::{flash_next::cuda::driver::fakes, session::TokenStepper},
};
use fakes::{FakeChunkDevice, FakeFeeds, LOGITS, SAMPLER_VOCAB, STREAM, shape};

fn chunk_driver() -> ChunkDriver<FakeChunkDevice, FakeFeeds> {
    ChunkDriver::new(FakeChunkDevice::default(), FakeFeeds::default(), shape()).unwrap()
}

#[test]
fn shared_prefill_opens_once_for_two_chunks_and_closes_after_sampling() {
    let device = FakeChunkDevice {
        track_windows: true,
        ..Default::default()
    };
    let mut driver = ChunkDriver::new(device, FakeFeeds::default(), shape()).unwrap();
    driver.chunk_cap = 2;
    driver.feed_rows(&[1, 2, 3]).unwrap();
    assert_eq!(driver.fed(), 3);
    assert_eq!(
        driver.device().log,
        [
            "begin_prefill",
            "chunk_params 1 0 2",
            "chunk_stream",
            "chunk_ple",
            "begin",
            "layers 0..2 @0 m2",
            "read_stream",
            "write_back",
            "layers 2..4 @0 m2",
            "health",
            "chunk_params 3 2 1",
            "chunk_stream",
            "chunk_ple",
            "begin",
            "layers 0..2 @2 m1",
            "read_stream",
            "write_back",
            "layers 2..4 @2 m1",
            "health",
        ]
    );
    let prefed_calls = driver.device().log.len();
    assert_eq!(driver.logits(4).unwrap().len(), LOGITS);
    assert_eq!(driver.fed(), 4);
    let log = &driver.device().log;
    assert_eq!(
        &log[prefed_calls..prefed_calls + 2],
        ["begin_prefill_logits", "chunk_params 4 3 1"].map(String::from)
    );
    assert_eq!(
        &log[log.len() - 4..],
        ["head", "read_logits", "health", "end_prefill"].map(String::from)
    );
    assert_eq!(
        log.iter().filter(|call| *call == "begin_prefill").count(),
        1
    );
}

fn shared_driver(feeds: FakeFeeds) -> ChunkDriver<FakeChunkDevice, FakeFeeds> {
    let device = FakeChunkDevice {
        track_windows: true,
        ..Default::default()
    };
    let mut driver = ChunkDriver::new(device, feeds, shape()).unwrap();
    driver.chunk_cap = 2;
    driver
}

#[test]
fn shared_prefill_preflight_refusals_do_not_open_or_poison() {
    use crate::family::flash_next::cuda::driver::ResumesState;
    for token in [SAMPLER_VOCAB as u32, u32::MAX] {
        let mut driver = shared_driver(FakeFeeds::default());
        assert!(driver.feed_rows(&[1, 2, token]).is_err());
        assert!(driver.device().log.is_empty());
        assert_eq!(driver.resume_generation(), 0);
        driver.feed_rows(&[1]).unwrap();
    }
    for feeds in [
        FakeFeeds {
            fail_begin: Some(2),
            ..Default::default()
        },
        FakeFeeds {
            start_width: Some(STREAM + 1),
            ..Default::default()
        },
    ] {
        let mut driver = shared_driver(feeds);
        assert!(driver.feed_rows(&[1, 2]).is_err());
        assert!(driver.device().log.is_empty());
        assert_eq!(driver.fed(), 0);
        assert_eq!(driver.resume_generation(), 0);
    }
    let mut driver = shared_driver(FakeFeeds::default());
    assert!(driver.feed_rows(&[1; 7]).is_err());
    // A chunk cap above the rows ceiling (the supported width, raisable by the measurement
    // variable) refuses preflight without opening or poisoning the device.
    driver.chunk_cap = crate::family::project::exl3_jobs_plan::exl3_rows_ceiling().unwrap() + 1;
    assert!(driver.feed_rows(&[1]).is_err());
    assert!(driver.device().log.is_empty());
    driver.feed_rows(&[]).unwrap();
}

#[test]
fn shared_prefill_device_failures_poison_before_commit_and_reset_recovers() {
    use crate::family::flash_next::cuda::driver::ResumesState;
    for failure in [
        "begin_prefill",
        "chunk_params",
        "chunk_stream",
        "chunk_ple",
        "begin",
        "layers",
        "read_stream",
        "write_back",
        "health",
    ] {
        let mut driver = shared_driver(FakeFeeds::default());
        driver.reset().unwrap();
        driver.device_mut().fail = Some(failure);
        assert!(driver.feed_rows(&[1, 2]).is_err(), "{failure}");
        assert_eq!(driver.fed(), 0, "{failure}");
        assert_eq!(driver.resume_generation(), 1, "{failure}");
        assert!(driver.feed_rows(&[1]).is_err(), "{failure}");
        driver.reset().unwrap();
        driver.feed_rows(&[1, 2]).unwrap();
        assert_eq!(driver.resume_generation(), 2);
        assert_eq!(driver.fed(), 2);
        driver.logits(3).unwrap();
    }
}

#[test]
fn shared_prefill_sampling_failures_preserve_commit_and_close_rules() {
    for failure in [
        "begin_prefill_logits",
        "head",
        "read_logits",
        "health",
        "end_prefill",
    ] {
        let mut driver = shared_driver(FakeFeeds::default());
        driver.feed_rows(&[1, 2]).unwrap();
        driver.device_mut().fail = Some(failure);
        assert!(driver.logits(3).is_err(), "{failure}");
        assert_eq!(driver.fed(), if failure == "end_prefill" { 3 } else { 2 });
        assert!(driver.feed(1).is_err());
        driver.reset().unwrap();
        driver.feed_rows(&[1]).unwrap();
        driver.logits(2).unwrap();
    }
}

#[test]
fn shared_prefill_reset_and_resume_keep_the_family_generation_rules() {
    use crate::family::flash_next::cuda::driver::ResumesState;
    let mut driver = shared_driver(FakeFeeds::default());
    driver.reset().unwrap();
    driver.feed_rows(&[1, 2]).unwrap();
    driver.logits(3).unwrap();
    let generation = driver.resume_generation();
    let snapshot = driver.save_resume().unwrap();
    driver.reset().unwrap();
    driver.restore_resume(&snapshot).unwrap();
    assert_eq!(driver.fed(), 3);
    assert_eq!(driver.resume_generation(), generation);
    driver.feed_rows(&[4]).unwrap();
    assert!(
        driver
            .device()
            .log
            .contains(&"chunk_params 4 3 1".to_owned())
    );
    driver.reset().unwrap();
    driver.feed_rows(&[1]).unwrap();
    assert_ne!(driver.resume_generation(), generation);
}

#[test]
fn shared_prefill_refuses_a_partial_device_ple_pair_list() {
    let mut driver = shared_driver(FakeFeeds::default());
    driver.ple_device = true;
    driver.device_mut().ple_pairs = Some(vec![PleKeyValue {
        keys: vec![1.0; STREAM],
        value: vec![1.0],
    }]);
    assert!(driver.feed_rows(&[1, 2]).is_err());
    assert_eq!(driver.fed(), 0);
    assert!(driver.feed(1).is_err());
}

#[test]
fn default_chunk_refuses_partial_device_ple_pairs() {
    let mut driver = chunk_driver();
    driver.chunk_cap = 2;
    driver.ple_device = true;
    driver.device_mut().ple_pairs = Some(vec![PleKeyValue {
        keys: vec![1.0; STREAM],
        value: vec![1.0],
    }]);
    assert!(driver.feed_rows(&[1, 2]).is_err());
    assert_eq!(driver.fed(), 0);
    assert!(driver.feed(1).is_err());
}

#[test]
fn shared_prefill_switch_precedes_cuda_initialization() {
    let source = include_str!("../../start.rs");
    let build = &source[source.find("fn build_driver(").expect("driver builder")..];
    let check = build
        .find("prof::check_switch_spelling()?")
        .expect("switch check");
    let initialize = build
        .find("CudaProjector::new()")
        .expect("CUDA initialization");
    assert!(check < initialize);
}

/// The route counters are process globals: every test that reads them serialises here, so an
/// interleaved `run_chunk` of another test cannot add to a read.
static ROUTE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[test]
fn a_fed_chunk_runs_the_start_the_layers_around_one_ple_roundtrip_and_no_head() {
    let mut driver = chunk_driver();
    driver.run_chunk(&[3, 4], false).unwrap();
    assert_eq!(
        driver.device().log,
        [
            "chunk_params 3 0 2",
            "chunk_stream",
            "chunk_ple",
            "begin",
            "layers 0..2 @0 m2",
            "read_stream",
            "write_back",
            "layers 2..4 @0 m2",
            "health",
        ]
    );
    assert_eq!(driver.fed(), 2);
    // Each row's update saw that row's stream after layers 0 and 1: every layer adds
    // layer + 1 + pos + row, so row 0 (token 3) is 3+1+2=6 and row 1 (token 4) is 4+2+3=9.
    assert_eq!(
        driver.feeds().updates,
        [(3, vec![6.0; STREAM]), (4, vec![9.0; STREAM]),]
    );
}

#[test]
fn the_logits_chunk_runs_the_head_over_the_last_row() {
    let mut driver = chunk_driver();
    let row = driver.run_chunk(&[3, 4], true).unwrap().unwrap();
    let log = &driver.device().log;
    assert_eq!(
        &log[log.len() - 3..],
        ["head", "read_logits", "health"].map(String::from)
    );
    assert_eq!(row.len(), LOGITS);
    // The last row (token 4, row 1) after every layer, plus its 0.5 update: the layers add
    // 2+3 before the PLE roundtrip and 4+5 after, so 4 + 5 + 9 + 0.5 = 18.5 at every value.
    for (i, value) in row.iter().enumerate() {
        assert_eq!(*value, 18.5 * i as f32);
    }
}

#[test]
fn the_device_ple_route_projects_on_the_device_and_hands_the_pairs_over() {
    let mut driver = chunk_driver();
    driver.ple_device = true;
    driver.device_mut().ple_pairs = Some(
        (0..2)
            .map(|r| PleKeyValue {
                keys: vec![r as f32; STREAM],
                value: vec![r as f32; 1],
            })
            .collect(),
    );
    driver.run_chunk(&[3, 4], false).unwrap();
    assert!(
        driver
            .device()
            .log
            .iter()
            .any(|call| call.starts_with("chunk_ple_project"))
    );
    // The pairs reach the step in row order.
    assert_eq!(driver.feeds().pairs, [0.0, 1.0]);
    // The plain route never projects on the device.
    let mut plain = chunk_driver();
    plain.run_chunk(&[3, 4], false).unwrap();
    assert!(
        !plain
            .device()
            .log
            .iter()
            .any(|call| call.starts_with("chunk_ple_project"))
    );
    assert!(plain.feeds().pairs.is_empty());
}

#[test]
fn the_device_route_without_a_projection_is_refused_and_poisons() {
    let mut driver = chunk_driver();
    driver.ple_device = true;
    // The fake has no pairs, so `ple_project_chunk` fails; the driver must poison, not limp on.
    assert!(driver.run_chunk(&[3], false).is_err());
    assert!(driver.feed(4).is_err());
}

#[test]
fn the_position_advances_by_the_chunk_and_reaches_the_layers() {
    let mut driver = chunk_driver();
    driver.run_chunk(&[1, 2, 3], false).unwrap();
    driver.run_chunk(&[4], true).unwrap();
    let log = &driver.device().log;
    assert!(log.contains(&"layers 0..2 @3 m1".to_string()));
    assert_eq!(driver.fed(), 4);
}

#[test]
fn a_chunk_that_would_overrun_the_caches_is_refused_without_writes() {
    let mut driver = chunk_driver();
    driver.run_chunk(&[1, 2, 3, 4], false).unwrap();
    // 4 + 3 > POSITIONS (6): refused before any device write, nothing poisoned.
    let error = driver.run_chunk(&[5, 6, 7], false).unwrap_err();
    assert!(matches!(error, EngError::ContextLimit(_)));
    assert_eq!(driver.fed(), 4);
    assert_eq!(driver.device().log.last().unwrap(), "health");
    driver.run_chunk(&[5, 4], false).unwrap();
    assert_eq!(driver.fed(), 6);
}

#[test]
fn an_empty_chunk_and_an_out_of_range_token_change_nothing() {
    let mut driver = chunk_driver();
    assert!(driver.run_chunk(&[], false).is_err());
    assert!(driver.run_chunk(&[3, SAMPLER_VOCAB as u32], false).is_err());
    assert!(driver.device().log.is_empty());
    assert_eq!(driver.fed(), 0);
}

#[test]
fn the_generation_advances_at_the_attempt_of_a_fresh_sequence() {
    // The serve driver is the chunk driver, so the generation guard is proven here too:
    // the bump lands BEFORE the queue call, so a FAILED first chunk (device work may still
    // have written rows) already invalidates older snapshots, and every fresh sequence
    // after a reset bumps again — the generation survives resets by design.
    use crate::family::flash_next::cuda::driver::ResumesState;

    let device = FakeChunkDevice {
        fail: Some("layers"),
        ..FakeChunkDevice::default()
    };
    let mut driver = ChunkDriver::new(device, FakeFeeds::default(), shape()).unwrap();
    assert_eq!(driver.resume_generation(), 0);
    assert!(driver.run_chunk(&[3, 4], false).is_err());
    // The attempt itself advanced the generation although the chunk never landed.
    assert_eq!(driver.resume_generation(), 1);
    assert_eq!(driver.fed(), 0);

    driver.reset().unwrap();
    driver.run_chunk(&[2, 3], false).unwrap();
    // A fresh sequence after the reset advanced it again.
    assert_eq!(driver.resume_generation(), 2);
    assert_eq!(driver.fed(), 2);
}

#[test]
fn a_failure_after_the_first_write_poisons_until_reset() {
    let device = FakeChunkDevice {
        fail: Some("layers"),
        ..FakeChunkDevice::default()
    };
    let mut driver = ChunkDriver::new(device, FakeFeeds::default(), shape()).unwrap();
    assert!(driver.run_chunk(&[3, 4], false).is_err());
    // The chunk did not land: the position stayed, and the poison refuses later chunks.
    assert_eq!(driver.fed(), 0);
    assert!(driver.run_chunk(&[3], false).is_err());
    driver.reset().unwrap();
    driver.run_chunk(&[3, 4], false).unwrap();
    assert_eq!(driver.fed(), 2);
}

#[test]
fn a_failed_host_begin_leaves_the_state_unpoisoned() {
    let feeds = FakeFeeds {
        fail_begin: Some(4),
        ..FakeFeeds::default()
    };
    let mut driver = ChunkDriver::new(FakeChunkDevice::default(), feeds, shape()).unwrap();
    assert!(driver.run_chunk(&[3, 4], false).is_err());
    assert_eq!(driver.fed(), 0);
    // A host-read failure is not a device failure: the next chunk still runs.
    driver.run_chunk(&[3], false).unwrap();
    assert_eq!(driver.fed(), 1);
}

#[test]
fn a_ple_update_of_the_wrong_width_is_refused_and_poisons() {
    let feeds = FakeFeeds {
        update_width: Some(STREAM + 1),
        ..FakeFeeds::default()
    };
    let mut driver = ChunkDriver::new(FakeChunkDevice::default(), feeds, shape()).unwrap();
    assert!(driver.run_chunk(&[3, 4], false).is_err());
    assert_eq!(driver.fed(), 0);
    assert!(driver.run_chunk(&[3], false).is_err());
}

#[test]
fn the_shape_refuses_a_ple_layer_at_or_past_the_layers() {
    let mut bad = shape();
    bad.ple_layer = bad.layers;
    assert!(ChunkDriver::new(FakeChunkDevice::default(), FakeFeeds::default(), bad).is_err());
}

#[test]
fn feed_rows_splits_the_prompt_into_chunks_of_the_cap() {
    let mut driver = chunk_driver();
    driver.chunk_cap = 3;
    driver.feed_rows(&[1, 2, 3, 4, 5]).unwrap();
    let params: Vec<_> = driver
        .device()
        .log
        .iter()
        .filter(|line| line.starts_with("chunk_params"))
        .cloned()
        .collect();
    assert_eq!(
        params,
        ["chunk_params 1 0 3", "chunk_params 4 3 2"].map(String::from)
    );
    assert_eq!(driver.fed(), 5);
}

#[test]
fn feed_rows_with_the_default_cap_feeds_one_token_at_a_time() {
    let mut driver = chunk_driver();
    driver.feed_rows(&[1, 2]).unwrap();
    let params: Vec<_> = driver
        .device()
        .log
        .iter()
        .filter(|line| line.starts_with("chunk_params"))
        .cloned()
        .collect();
    assert_eq!(
        params,
        ["chunk_params 1 0 1", "chunk_params 2 1 1"].map(String::from)
    );
}

#[test]
fn the_switch_spelling_is_strict() {
    use super::prof::parse_switch;
    assert_eq!(parse_switch(None), Ok(false));
    for off in ["0", "false", "off"] {
        assert_eq!(parse_switch(Some(off)), Ok(false), "off spelling {off:?}");
    }
    for on in ["1", "true", "on"] {
        assert_eq!(parse_switch(Some(on)), Ok(true), "on spelling {on:?}");
    }
    for bad in ["", "2", "yes", "TRUE", "On", " off"] {
        assert!(parse_switch(Some(bad)).is_err(), "refuse spelling {bad:?}");
    }
}

#[test]
fn a_boundary_drains_before_it_charges_the_slot_and_only_under_both_switches() {
    // The order the drain mode pins: the drain synchronises the stream the phase just
    // queued, so its wait lands inside the slot being closed, and neither half runs
    // without the profile. The OnceLock arming itself cannot be re-set within a process,
    // so this holds the boundary given the two flags, which is where the order lives.
    let mut dev = FakeChunkDevice::default();
    let mut t = std::time::Instant::now();
    let mut charged: Vec<usize> = Vec::new();
    super::advance_boundary(&mut dev, 34, &mut t, true, true, |slot, _| {
        charged.push(slot)
    })
    .unwrap();
    assert_eq!(dev.log.last().map(String::as_str), Some("drain"));
    assert_eq!(charged, [34]);

    // Profile off: no drain and no charge.
    let mut quiet = FakeChunkDevice::default();
    let mut t = std::time::Instant::now();
    let mut none: Vec<usize> = Vec::new();
    super::advance_boundary(&mut quiet, 1, &mut t, false, false, |slot, _| {
        none.push(slot)
    })
    .unwrap();
    assert!(quiet.log.is_empty());
    assert!(none.is_empty());

    // A failing drain surfaces the error and charges nothing.
    let mut failing = FakeChunkDevice {
        fail: Some("drain"),
        ..Default::default()
    };
    let mut t = std::time::Instant::now();
    let error = super::advance_boundary(&mut failing, 2, &mut t, true, true, |slot, _| {
        none.push(slot)
    })
    .unwrap_err();
    assert!(matches!(error, EngError::Backend { .. }));
    assert!(none.is_empty());
}

#[test]
fn the_host_batch_route_projects_the_chunk_and_hands_each_row_its_pair() {
    use crate::family::flash_next::cuda::ple_feed::{HostMatrix, HostProjection};

    // Key matrix value i is i % PLE_EMBED, so row 0 holds 0,1,2 and its dot with the fake's
    // [t; PLE_EMBED] embedding is 3t: token 3 projects 9, token 4 projects 12. The value matrix
    // is constant 0.25 and never reaches `pairs` (only the first key is recorded).
    let key = (0..6 * fakes::PLE_EMBED)
        .map(|i| (i % fakes::PLE_EMBED) as f32)
        .collect::<Vec<_>>();
    let value = vec![0.25; 3 * fakes::PLE_EMBED];
    let projection = HostProjection::new(
        HostMatrix::new(6, fakes::PLE_EMBED, key).unwrap(),
        HostMatrix::new(3, fakes::PLE_EMBED, value).unwrap(),
        2,
    );
    let mut driver = chunk_driver();
    driver.ple_host_batch = Some(Box::new(projection));
    let _guard = ROUTE_LOCK.lock().unwrap();
    super::prof::reset_route_counters();
    driver.run_chunk(&[3, 4], false).unwrap();
    // The fake's `begin` embeds each token as [token; PLE_EMBED]: key row 0 holds 0..3
    // (sum 3), so token t's first key is 3t — the pairs were projected and set in order.
    assert_eq!(driver.feeds().pairs, [9.0, 12.0]);
    // A two-row chunk on a two-thread projection takes the product route.
    assert_eq!(super::prof::route_counters(), (1, 0));
    // Everything else ran exactly as the per-row route: same log, same updates.
    let mut plain = chunk_driver();
    plain.run_chunk(&[3, 4], false).unwrap();
    assert_eq!(driver.device().log, plain.device().log);
    assert_eq!(driver.feeds().updates, plain.feeds().updates);
}

#[test]
fn a_one_row_chunk_takes_the_per_row_projections_not_the_product() {
    use crate::family::flash_next::cuda::ple_feed::{HostMatrix, HostProjection};

    let _guard = ROUTE_LOCK.lock().unwrap();

    let key = (0..6 * fakes::PLE_EMBED)
        .map(|i| (i % fakes::PLE_EMBED) as f32)
        .collect::<Vec<_>>();
    let value = vec![0.25; 3 * fakes::PLE_EMBED];
    // Two projection threads: a one-row chunk is narrower than them, so it must keep the
    // per-row projections — a one-row product would sweep the weights on a single thread.
    let projection = HostProjection::new(
        HostMatrix::new(6, fakes::PLE_EMBED, key).unwrap(),
        HostMatrix::new(3, fakes::PLE_EMBED, value).unwrap(),
        2,
    );
    let mut driver = chunk_driver();
    driver.ple_host_batch = Some(Box::new(projection));
    super::prof::reset_route_counters();
    driver.run_chunk(&[3], false).unwrap();
    assert_eq!(
        super::prof::route_counters(),
        (0, 1),
        "a one-row chunk must not run the chunk product"
    );
    // The narrow chunk still gets its projected pair, same as the product route would give.
    assert_eq!(driver.feeds().pairs, [9.0]);
    // The wide chunk on the same driver flips to the product: one product, no extra rows.
    driver.run_chunk(&[3, 4], false).unwrap();
    assert_eq!(super::prof::route_counters(), (1, 1));
}
