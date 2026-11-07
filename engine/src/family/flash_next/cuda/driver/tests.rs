//! The sequence, the position, the poison rule and the greedy draw of the token driver, on fakes.

use super::{
    TokenDriver,
    fakes::{FakeDevice, FakeFeeds, LOGITS, POSITIONS, SAMPLER_VOCAB, STREAM, shape},
};
use crate::{
    EngError,
    family::{
        sampler::Sampler,
        session::{TokenStepper, sample_logits},
    },
};

fn driver() -> TokenDriver<FakeDevice, FakeFeeds> {
    TokenDriver::new(FakeDevice::default(), FakeFeeds::default(), shape()).unwrap()
}

#[test]
fn a_fed_token_runs_the_start_the_layers_around_the_ple_update_and_no_head() {
    let mut driver = driver();
    driver.feed(3).unwrap();
    assert_eq!(
        driver.device().log,
        [
            "params 3 0",
            "stream",
            "ple_embedding",
            "begin",
            "layers 0..2 @0",
            "read_stream",
            "write_back",
            "layers 2..4 @0",
            "health",
        ]
    );
    assert_eq!(driver.fed(), 1);
    // The update saw the stream after layers 0 and 1: 3 + 1 + 2.
    assert_eq!(driver.feeds().updates, [(3, vec![6.0; STREAM])]);
}

#[test]
fn the_logits_token_runs_the_head_and_the_row_follows_the_update() {
    let mut driver = driver();
    let row = driver.logits(3).unwrap();
    let log = &driver.device().log;
    assert_eq!(
        &log[log.len() - 3..],
        ["head", "read_logits", "health"].map(String::from)
    );
    // 3 + (1 + 2) + 0.5 + (3 + 4) = 13.5 on every value; value i is 13.5 * i.
    assert_eq!(row.len(), LOGITS);
    for (i, value) in row.iter().enumerate() {
        assert_eq!(*value, 13.5 * i as f32);
    }
}

#[test]
fn the_position_advances_per_token_and_reaches_the_device_and_the_layers() {
    let mut driver = driver();
    driver.feed(1).unwrap();
    driver.feed(2).unwrap();
    let log = &driver.device().log;
    assert!(log.contains(&"params 2 1".to_string()), "{log:?}");
    assert!(log.contains(&"layers 2..4 @1".to_string()), "{log:?}");
    assert_eq!(driver.fed(), 2);
}

#[test]
fn a_token_outside_the_sampler_vocabulary_is_refused_before_any_device_write() {
    let mut driver = driver();
    let error = driver.feed(SAMPLER_VOCAB as u32).unwrap_err();
    assert!(matches!(error, EngError::BadRequest(_)), "{error:?}");
    assert!(driver.device().log.is_empty());
    assert_eq!(driver.fed(), 0);
    // Not poisoned: a valid token goes through.
    driver.feed(1).unwrap();
    assert_eq!(driver.fed(), 1);
}

#[test]
fn a_device_failure_poisons_the_driver_until_reset() {
    let mut driver = driver();
    driver.feed(1).unwrap();
    driver.device.fail = Some("layers");
    assert!(matches!(
        driver.feed(2).unwrap_err(),
        EngError::Backend { .. }
    ));
    assert_eq!(driver.fed(), 1, "a failed token is not fed");
    let again = driver.feed(2).unwrap_err();
    assert!(
        matches!(&again, EngError::BadRequest(text) if text.contains("needs a reset")),
        "{again:?}"
    );
    driver.reset().unwrap();
    assert_eq!((driver.device().resets, driver.feeds().resets), (1, 1));
    assert_eq!(driver.fed(), 0);
    driver.feed(2).unwrap();
    assert_eq!(driver.fed(), 1);
    let log = &driver.device().log;
    assert!(log.contains(&"params 2 0".to_string()), "{log:?}");
}

/// Every failure after the first write to the card poisons the driver, whichever call it is in.
#[test]
fn a_failure_at_any_device_call_after_the_first_write_poisons() {
    for call in [
        "params",
        "stream",
        "begin",
        "layers",
        "read_stream",
        "write_back",
        "health",
    ] {
        let mut driver = driver();
        driver.device.fail = Some(call);
        assert!(driver.feed(1).is_err(), "{call}");
        assert_eq!(driver.fed(), 0, "{call}");
        let again = driver.feed(1).unwrap_err();
        assert!(
            matches!(&again, EngError::BadRequest(text) if text.contains("needs a reset")),
            "{call}: {again:?}"
        );
        driver.reset().unwrap();
        driver.feed(1).unwrap();
    }
}

#[test]
fn a_failed_ple_update_poisons_and_keeps_the_position() {
    let mut driver = driver();
    driver.feeds.fail_update = true;
    assert!(matches!(
        driver.feed(1).unwrap_err(),
        EngError::Checkpoint(_)
    ));
    assert_eq!(driver.fed(), 0);
    assert!(driver.feed(1).is_err());
    driver.reset().unwrap();
    driver.feeds.fail_update = false;
    driver.feed(1).unwrap();
}

#[test]
fn the_phases_of_a_token_are_timed_and_a_fed_token_has_no_head_time() {
    let mut driver = driver();
    driver.feed(1).unwrap();
    assert_eq!(driver.last_times().head, std::time::Duration::ZERO);
    assert_eq!(driver.bank_misses(), None);
}

#[test]
fn a_failed_health_check_poisons_and_does_not_advance() {
    let mut driver = driver();
    driver.device.fail = Some("health");
    driver.logits(1).unwrap_err();
    assert_eq!(driver.fed(), 0);
    assert!(driver.feed(1).is_err());
}

#[test]
fn a_host_read_failure_leaves_the_driver_usable() {
    let mut driver = driver();
    driver.feeds.fail_begin = Some(4);
    assert!(matches!(
        driver.feed(4).unwrap_err(),
        EngError::Checkpoint(_)
    ));
    assert!(driver.device().log.is_empty());
    driver.feed(1).unwrap();
}

#[test]
fn a_ple_update_of_the_wrong_width_is_refused_and_poisons() {
    let mut driver = driver();
    driver.feeds.update_width = Some(STREAM - 1);
    assert!(driver.feed(1).is_err());
    assert_eq!(driver.fed(), 0);
    assert!(driver.feed(1).is_err());
}

#[test]
fn the_caches_bound_the_tokens_and_the_refusal_does_not_poison() {
    let mut driver = driver();
    for _ in 0..POSITIONS {
        driver.feed(1).unwrap();
    }
    assert!(matches!(
        driver.feed(1).unwrap_err(),
        EngError::ContextLimit(_)
    ));
    assert_eq!(driver.fed(), POSITIONS);
    driver.reset().unwrap();
    driver.feed(1).unwrap();
}

#[test]
fn greedy_draws_within_the_sampler_vocabulary_not_from_the_padded_head() {
    let mut driver = driver();
    let mut sampler = Sampler::greedy();
    let token = driver.next_token(1, &mut sampler, SAMPLER_VOCAB).unwrap();
    // The row is increasing, so the head's own argmax is its last id, above the vocabulary.
    assert_eq!(token as usize, SAMPLER_VOCAB - 1);
    driver.reset().unwrap();
    let row = driver.logits(1).unwrap();
    assert_eq!(row.len(), LOGITS);
    assert_eq!(
        sample_logits("t", &row, &mut Sampler::greedy(), SAMPLER_VOCAB).unwrap(),
        token
    );
    let narrow = sample_logits("t", &row[..2], &mut Sampler::greedy(), SAMPLER_VOCAB);
    assert!(matches!(narrow, Err(EngError::Backend { .. })));
}
