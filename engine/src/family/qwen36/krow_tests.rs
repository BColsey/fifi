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
