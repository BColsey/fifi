//! The turn line through the driver chain the served start builds: the
//! same `ResumesState` forwarding `StepDriver` uses, over the driver
//! types themselves (`StepDriver` is concrete over the card devices; its
//! two arms are these drivers). The executor's print is two lines on a
//! card-only type — the card start is not host-buildable — so the glue
//! tested here ends at the driver's `rung_miss_line`, the exact call the
//! executor makes at the turn boundary.

use super::bank_source::DecodeMisses;
use super::driver::ResumesState;
use super::driver::TokenDriver;
use super::driver::chunk::ChunkDriver;
use super::driver::fakes::{self, FakeChunkDevice, FakeDevice, FakeFeeds};
use crate::family::session::TokenStepper;

/// A turn of `steps` decode tokens through a token driver whose device
/// observes the card pattern (layer `l` misses `l`, so layer 0 is a real
/// zero) over `LAYERS` layers.
fn turn_of_steps(steps: u64) -> (TokenDriver<FakeDevice, FakeFeeds>, String) {
    let misses = DecodeMisses::new(fakes::LAYERS, 8);
    let device = FakeDevice {
        decode_misses: Some(misses),
        ..Default::default()
    };
    let mut driver = TokenDriver::new(device, FakeFeeds::default(), fakes::shape()).unwrap();
    for token in 1..=steps as u32 {
        driver.feed(token).unwrap();
    }
    let line = driver.rung_miss_line().unwrap_or_default();
    (driver, line)
}

#[test]
fn a_turn_of_steps_returns_its_line_through_the_token_driver() {
    let (_driver, line) = turn_of_steps(3);
    // 3 steps; every layer of every step read 0+1+2+3 = 6 records, the
    // per-layer bins span 0..=3 (three observations each, lower median 1).
    assert!(line.contains("3 steps"), "{line}");
    assert!(
        line.contains("misses per layer min 0 / median 1 / max 3"),
        "{line}"
    );
    assert!(line.contains("per step min 6 / median 6 / max 6"), "{line}");
    assert!(line.contains("pool depth 4"), "{line}");
}

#[test]
fn the_second_turn_reports_only_its_own_steps() {
    let (mut driver, first) = turn_of_steps(3);
    assert!(first.contains("3 steps"), "{first}");
    for token in 3..=5u32 {
        driver.feed(token).unwrap();
    }
    let second = driver.rung_miss_line().expect("the second turn's line");
    assert!(second.contains("3 steps"), "{second}");
    // The take zeroed the bins: the second line carries the same figures,
    // not six steps' worth.
    assert!(
        second.contains("per step min 6 / median 6 / max 6"),
        "{second}"
    );
}

#[test]
fn a_reset_between_turns_loses_nothing_that_completed() {
    let (mut driver, first) = turn_of_steps(2);
    assert!(first.contains("2 steps"), "{first}");
    // The worker's between-job reset: the driver reset finds nothing open
    // (the line already took the bins), so the completed turn survived it.
    driver.reset().unwrap();
    driver.feed(3).unwrap();
    let after = driver.rung_miss_line().expect("the next turn's line");
    assert!(after.contains("1 step"), "{after}");
    assert!(
        after.contains("per step min 6 / median 6 / max 6"),
        "{after}"
    );
}

#[test]
fn a_turn_that_never_touched_the_rung_has_no_line() {
    let mut driver =
        TokenDriver::new(FakeDevice::default(), FakeFeeds::default(), fakes::shape()).unwrap();
    driver.feed(1).unwrap();
    assert!(driver.rung_miss_line().is_none());
}

#[test]
fn the_chunk_driver_answers_the_same_turn_line() {
    let misses = DecodeMisses::new(fakes::LAYERS, 8);
    let device = FakeChunkDevice {
        decode_misses: Some(misses),
        ..Default::default()
    };
    let mut driver = ChunkDriver::new(device, FakeFeeds::default(), fakes::shape()).unwrap();
    for token in 1..=2u32 {
        driver.feed(token).unwrap();
    }
    let line = driver.rung_miss_line().expect("the chunk driver's line");
    assert!(line.contains("2 steps"), "{line}");
    assert!(line.contains("pool depth 4"), "{line}");
    // The take: a second turn stands alone here too.
    for token in 3..=3u32 {
        driver.feed(token).unwrap();
    }
    let second = driver.rung_miss_line().expect("the second line");
    assert!(second.contains("1 step"), "{second}");
}
