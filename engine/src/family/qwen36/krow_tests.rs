use super::*;
use std::ffi::OsString;

fn os(text: &str) -> OsString {
    OsString::from(text)
}

fn message(result: Result<(), EngError>) -> String {
    format!("{:?}", result.expect_err("refused"))
}

fn facts() -> KrowFacts<'static> {
    KrowFacts {
        rollback_armed: true,
        exl3: true,
        rows: 8,
        single_gather: None,
        shared_in_routed: false,
    }
}

#[test]
fn the_variable_name_is_the_one_the_kernel_reads() {
    assert_eq!(
        SINGLE_GATHER_NAME,
        include_str!("../../kernels/cuda/resident/exl3.rs")
            .lines()
            .find(|line| line.contains("pub(crate) const SINGLE_GATHER_VAR"))
            .and_then(|line| line.split('"').nth(1))
            .expect("the kernel module names the variable")
    );
}

#[test]
fn the_qualified_configuration_arms() {
    check_krow_arming(facts()).expect("armed");
    check_krow_env(None, None).expect("both unset");
    check_krow_env(None, Some(&os("0"))).expect("the merged block is off at 0");
}

/// Unset is the default of the merged block and yields to krow at `request_krow_verify`; only an
/// explicit `1` is an environment-level refusal.
#[test]
fn only_an_explicit_one_refuses_from_the_environment() {
    check_krow_env(None, None).expect("unset yields, it does not refuse");
    check_krow_env(None, Some(&os(" 0 "))).expect("0 is off");
    let text = message(check_krow_env(None, Some(&os("1"))));
    assert!(text.contains("FIFI_EXL3_SHARED_IN_ROUTED=1"), "{text}");
    assert!(text.contains("unset it"), "{text}");
}

#[test]
fn any_value_of_the_single_row_switch_refuses_and_the_message_says_unset_it() {
    // Deliberate: an explicit wide1 is the served kernel, and it is refused
    // too, so that exactly one configuration is qualified.
    for value in ["off", "1", "wide1", "wide2", "wide4", "", "bogus"] {
        let value = os(value);
        let text = message(check_krow_env(Some(&value), None));
        assert!(
            text.contains("FIFI_EXL3_SINGLE_GATHER"),
            "{value:?}: {text}"
        );
        assert!(text.contains("unset it"), "{value:?}: {text}");
        let text = message(check_krow_arming(KrowFacts {
            single_gather: Some(&value),
            ..facts()
        }));
        assert!(text.contains("unset it"), "{value:?}: {text}");
    }
}

#[test]
fn the_merged_moe_block_refuses_from_the_environment_value_and_from_the_step_field_alike() {
    let by_env = message(check_krow_env(None, Some(&os("1"))));
    let by_field = message(check_krow_arming(KrowFacts {
        shared_in_routed: true,
        ..facts()
    }));
    let by_hook = message(check_krow_switches(None, true));
    assert!(by_env.contains("FIFI_EXL3_SHARED_IN_ROUTED=1"), "{by_env}");
    assert!(by_env.contains("unset it"), "{by_env}");
    assert_eq!(by_env, by_field, "one refusal for every source");
    assert_eq!(by_env, by_hook, "the test hook meets the same refusal");
}

#[test]
fn a_bad_value_of_the_merged_block_switch_is_the_variables_own_refusal() {
    let text = message(check_krow_env(None, Some(&os("2"))));
    assert!(text.contains("must be unset"), "{text}");
}

#[test]
fn arming_refuses_without_the_rollback_slots_the_exl3_pack_or_a_window_of_two_to_eight_rows() {
    let text = message(check_krow_arming(KrowFacts {
        rollback_armed: false,
        ..facts()
    }));
    assert!(text.contains("device rollback"), "{text}");
    let text = message(check_krow_arming(KrowFacts {
        exl3: false,
        ..facts()
    }));
    assert!(text.contains("EXL3"), "{text}");
    for rows in [0, 1, 9, 64] {
        let text = message(check_krow_arming(KrowFacts { rows, ..facts() }));
        assert!(text.contains("2 to 8 rows"), "{rows}: {text}");
    }
    for rows in 2..=8 {
        check_krow_arming(KrowFacts { rows, ..facts() }).expect("2 to 8");
    }
}

#[test]
fn the_window_cap_is_the_rollback_cap() {
    assert_eq!(MAX_ROWS, 8);
    assert!(uses_krow(2) && uses_krow(8));
    assert!(!uses_krow(1) && !uses_krow(9));
}

fn vocab_check(token: u32) -> Result<(), EngError> {
    if token >= 100 {
        return Err(EngError::BadRequest(format!(
            "token id {token} is out of range"
        )));
    }
    Ok(())
}

fn rows_facts() -> RowsFacts {
    RowsFacts {
        base: 10,
        capacity: 4096,
        merged_block: false,
    }
}

#[test]
fn every_id_of_the_window_is_checked_before_the_window_may_launch() {
    check_rows(&[1, 2, 3], vocab_check, &rows_facts()).expect("clean window");
    for at in 0..4 {
        let mut ids = [1, 2, 3, 4];
        ids[at] = 100;
        let text = message(check_rows(&ids, vocab_check, &rows_facts()));
        assert!(text.contains("token id 100"), "id at {at}: {text}");
    }
}

#[test]
fn the_window_must_fit_the_kv_and_have_two_to_eight_rows() {
    let tight = RowsFacts {
        base: 4094,
        ..rows_facts()
    };
    check_rows(&[1, 2], vocab_check, &tight).expect("fits exactly");
    let error = check_rows(&[1, 2, 3], vocab_check, &tight).expect_err("one cell short");
    assert!(matches!(error, EngError::ContextLimit(_)), "{error:?}");
    for ids in [&[][..], &[1][..], &[1; 9][..]] {
        assert!(
            check_rows(ids, vocab_check, &rows_facts()).is_err(),
            "{} ids",
            ids.len()
        );
    }
}

#[test]
fn the_merged_block_is_read_again_before_every_window() {
    let merged = RowsFacts {
        merged_block: true,
        ..rows_facts()
    };
    let text = message(check_rows(&[1, 2], vocab_check, &merged));
    assert!(text.contains("unset it"), "{text}");
}

#[test]
fn status_is_clean_only_when_every_pair_and_the_window_word_are_zero() {
    let clean = RowStatus {
        bits: vec![0; 4],
        window: 0,
    };
    check_status(&clean).expect("clean");
    // A bit in any pair, first or last or between.
    for row in 0..4 {
        let mut bits = vec![0; 4];
        bits[row] = 2;
        let text = message(check_status(&RowStatus { bits, window: 0 }));
        assert!(text.contains(&format!("row {row}")), "{text}");
    }
    // Router fault, invalid state and arithmetic live in the window word only.
    for word in [1, 4, 16, 2] {
        let text = message(check_status(&RowStatus {
            bits: vec![0; 4],
            window: word,
        }));
        assert!(text.contains("window word"), "{word}: {text}");
    }
}

fn clean(rows: usize) -> RowStatus {
    RowStatus {
        bits: vec![0; rows],
        window: 0,
    }
}

#[test]
fn the_tokens_of_a_clean_window_are_the_verdict_and_each_must_be_in_the_sampler_vocabulary() {
    assert_eq!(
        rows_verdict(&[3, 0, 7], &clean(3), 8).expect("clean"),
        vec![3, 0, 7]
    );
    // A row with no comparable value (-1), an id at the vocabulary width and a huge id are
    // refused, and the message names the row.
    for (tokens, row) in [
        (vec![1, -1, 2], 1),
        (vec![1, 2, 8], 2),
        (vec![i32::MAX, 0, 0], 0),
    ] {
        let text = message(rows_verdict(&tokens, &clean(3), 8).map(|_| ()));
        assert!(text.contains(&format!("row {row}")), "{tokens:?}: {text}");
    }
}

#[test]
fn a_status_bit_anywhere_fails_the_window_before_any_token_is_looked_at() {
    // The tokens are valid; the pairs or the word are not. The status judgement comes first, so
    // the error is the status's, whichever row or word carries the bit.
    let tokens = [1, 2, 3, 4];
    for row in 0..4 {
        let mut bits = vec![0; 4];
        bits[row] = 2;
        let text = message(rows_verdict(&tokens, &RowStatus { bits, window: 0 }, 8).map(|_| ()));
        assert!(text.contains(&format!("row {row} reports")), "{text}");
    }
    for word in [1, 2, 4, 16] {
        let status = RowStatus {
            bits: vec![0; 4],
            window: word,
        };
        let text = message(rows_verdict(&tokens, &status, 8).map(|_| ()));
        assert!(text.contains("window word"), "{word}: {text}");
    }
}

#[test]
fn a_table_that_disagrees_with_its_status_is_refused() {
    let text = message(rows_verdict(&[1, 2, 3], &clean(2), 8).map(|_| ()));
    assert!(text.contains("3 tokens and 2 status words"), "{text}");
}

fn config() -> ConfigFacts {
    ConfigFacts {
        fp16_kv: true,
        plain_step: true,
        bf16_embedding: true,
        exl3_head: true,
    }
}

#[test]
fn the_one_configuration_the_body_is_written_for_arms_and_each_departure_is_refused_by_name() {
    check_krow_configuration(config()).expect("the plain EXL3 step on the fp16 cache");
    for (facts, needle) in [
        (
            ConfigFacts {
                fp16_kv: false,
                ..config()
            },
            "fp16 KV",
        ),
        (
            ConfigFacts {
                plain_step: false,
                ..config()
            },
            "plain EXL3 step",
        ),
        (
            ConfigFacts {
                bf16_embedding: false,
                ..config()
            },
            "bf16 embedding",
        ),
        (
            ConfigFacts {
                exl3_head: false,
                ..config()
            },
            "EXL3 head",
        ),
    ] {
        let text = message(check_krow_configuration(facts));
        assert!(text.contains(needle), "{facts:?}: {text}");
    }
}
