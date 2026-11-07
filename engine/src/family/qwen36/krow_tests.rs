use super::*;
use crate::family::sampler::Sampler;
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

// ---- the device verify, judged on the host ----

fn ids_of(tokens: &[i32], rejected_at: Option<usize>) -> Vec<u32> {
    // Window ids whose drafts are the tokens of the rows before them, except one that differs.
    let mut ids: Vec<u32> = std::iter::once(5)
        .chain(tokens.iter().map(|&token| token as u32))
        .take(tokens.len())
        .collect();
    if let Some(draft) = rejected_at {
        ids[draft + 1] += 1;
    }
    ids
}

#[test]
fn the_accepted_run_is_the_leading_rows_whose_token_is_the_next_window_id() {
    for rows in 1..=8usize {
        let tokens: Vec<i32> = (0..rows as i32).map(|i| 20 + i).collect();
        // All accepted: run K - 1, the bonus is the last row's token.
        assert_eq!(
            greedy_run(&tokens, &ids_of(&tokens, None)),
            (rows - 1, tokens[rows - 1]),
            "{rows} rows"
        );
        // The first mismatch ends the run, whatever follows it.
        for draft in 0..rows.saturating_sub(1) {
            assert_eq!(
                greedy_run(&tokens, &ids_of(&tokens, Some(draft))),
                (draft, tokens[draft]),
                "{rows} rows, mismatch at draft {draft}"
            );
        }
    }
    // Reject-first and keep-middle by name, at 8 rows.
    let tokens = [11, 12, 13, 14, 15, 16, 17, 18];
    assert_eq!(
        greedy_run(&tokens, &[0, 99, 12, 13, 14, 15, 16, 17]),
        (0, 11)
    );
    assert_eq!(
        greedy_run(&tokens, &[0, 11, 12, 13, 99, 15, 16, 17]),
        (3, 14)
    );
    // A row without a token never matches an id, also an id that holds the same bits (as the
    // kernel: the comparison needs a token of at least 0).
    assert_eq!(greedy_run(&[4, -1, 6], &[0, 4, 0]), (1, -1));
    assert_eq!(greedy_run(&[4, -1, 6], &[0, 4, u32::MAX]), (1, -1));
}

#[test]
fn the_device_verify_is_enforced_against_the_pairs_and_any_difference_is_an_error() {
    let tokens = [30, 31, 32, 33];
    let ids = ids_of(&tokens, Some(2));
    // The sound device answer.
    assert_eq!(check_verify(&tokens, &ids, 2, 32).unwrap(), (2, 32));
    // Off by one either way, a bonus from the wrong row, a bonus from the next row.
    for (accepted, bonus) in [(3, 32), (1, 32), (2, 31), (2, 33), (2, 0), (0, 30)] {
        let text = format!("{:?}", check_verify(&tokens, &ids, accepted, bonus));
        assert!(
            text.contains("the pairs say run 2 and bonus 32"),
            "{accepted} {bonus}: {text}"
        );
    }
    // A run outside the window (negative, or as long as the window) is refused by itself.
    for accepted in [-1, 4, 100] {
        assert!(
            check_verify(&tokens, &ids, accepted, 33).is_err(),
            "{accepted}"
        );
    }
    // Rows and ids of different counts, or no rows, are refused.
    assert!(check_verify(&tokens, &ids[..3], 2, 32).is_err());
    assert!(check_verify(&[], &[], 0, 0).is_err());
    // A device that agrees with its pairs on a token the host cannot use is refused.
    assert!(check_verify(&[7, -1], &[0, 7], 1, -1).is_err());
    // A device that let a row without a token extend the run (the bits of -1 equal that id) is not
    // the host's rule: the run stops at that row.
    assert!(check_verify(&[4, -1, 6], &[0, 4, u32::MAX], 2, 6).is_err());
}

// ---- the shortlist records ----

/// A record as the device writes it: 20 entries of (f32 logit bits, u32 id), the non-finite word,
/// three zero pad words. Entries past `entries.len()` are the empty candidate.
fn record(entries: &[(u32, f32)], non_finite: u32) -> [u8; SHORTLIST_RECORD_BYTES] {
    let mut bytes = [0u8; SHORTLIST_RECORD_BYTES];
    for at in 0..SHORTLIST_K {
        let (id, logit) = entries
            .get(at)
            .copied()
            .unwrap_or((u32::MAX, f32::NEG_INFINITY));
        bytes[at * 8..at * 8 + 4].copy_from_slice(&logit.to_le_bytes());
        bytes[at * 8 + 4..at * 8 + 8].copy_from_slice(&id.to_le_bytes());
    }
    bytes[160..164].copy_from_slice(&non_finite.to_le_bytes());
    bytes
}

/// The top-20 of a row by the device's order: larger logit first, then the lower id.
fn ranked_top(row: &[f32], keep: usize) -> Vec<(u32, f32)> {
    let mut all: Vec<(u32, f32)> = row
        .iter()
        .copied()
        .enumerate()
        .map(|(i, v)| (i as u32, v))
        .collect();
    all.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
    all.truncate(keep);
    all
}

#[test]
fn a_record_decodes_to_its_ranked_candidates_and_the_non_finite_word_is_the_one_row_error() {
    let row: Vec<f32> = (0..64)
        .map(|i| ((i * 37 + 11) % 61) as f32 / 8.0 - 3.0)
        .collect();
    let ranked = ranked_top(&row, SHORTLIST_K);
    let bytes = record(&ranked, 0);
    let decoded = decode_shortlist(&bytes, 64).unwrap();
    assert_eq!(decoded.len(), 20);
    for (got, want) in decoded.iter().zip(&ranked) {
        assert_eq!((got.0, got.1.to_bits()), (want.0, want.1.to_bits()));
    }
    // A vocabulary below 20 reads that many entries.
    let few = ranked_top(&row[..7], 7);
    assert_eq!(decode_shortlist(&record(&few, 0), 7).unwrap().len(), 7);
    // The non-finite word is the error of the one-row step.
    let error = decode_shortlist(&record(&ranked, 1), 64).expect_err("non-finite");
    assert!(format!("{error:?}").contains("device shortlist found non-finite logits"));
    // Only whole records decode.
    assert!(decode_shortlist(&bytes[..175], 64).is_err());
}

#[test]
fn a_shortlist_that_disagrees_with_its_pair_is_an_error_and_a_bad_row_is_not_read() {
    let rows: Vec<Vec<f32>> = (0..3)
        .map(|r| {
            (0..40)
                .map(|i| ((i * 7 + r * 13) % 29) as f32 - 9.0)
                .collect()
        })
        .collect();
    let tokens: Vec<i32> = rows
        .iter()
        .map(|row| ranked_top(row, 1)[0].0 as i32)
        .collect();
    let records: Vec<[u8; SHORTLIST_RECORD_BYTES]> = rows
        .iter()
        .map(|row| record(&ranked_top(row, 20), 0))
        .collect();
    let bits = [0, 0, 0];
    let ok = check_shortlists(&tokens, &bits, &records, 40).unwrap();
    assert_eq!(ok.len(), 3);
    assert_eq!(ok[1], ranked_top(&rows[1], 20));
    // The record of another row (the K-row grid indexing the wrong row) starts at another id.
    let mut swapped = records.clone();
    swapped.swap(0, 1);
    let text = format!("{:?}", check_shortlists(&tokens, &bits, &swapped, 40));
    assert!(text.contains("row 0: the shortlist starts at"), "{text}");
    // A non-finite word where the argmax saw nothing wrong is an error ...
    let mut flagged = records.clone();
    flagged[2] = record(&ranked_top(&rows[2], 20), 1);
    let text = format!("{:?}", check_shortlists(&tokens, &bits, &flagged, 40));
    assert!(
        text.contains("row 2: the shortlist saw a non-finite value"),
        "{text}"
    );
    // ... and where the argmax did see it, the row's record is not read (the window fails on its bit).
    let flagged_bits = [0, 0, 2];
    let read = check_shortlists(&tokens, &flagged_bits, &flagged, 40).unwrap();
    assert!(read[2].is_empty());
    // One record per row.
    assert!(check_shortlists(&tokens, &bits, &records[..2], 40).is_err());
}

/// The ways entry `at` of a record can be corrupt that a later entry-by-entry check must find and
/// the first entry and the non-finite word cannot show.
fn corruptions(
    good: &[(u32, f32)],
    at: usize,
    vocab: usize,
) -> Vec<(&'static str, Vec<(u32, f32)>)> {
    let mut out = Vec::new();
    let mut repeated = good.to_vec();
    repeated[at].0 = good[at - 4].0;
    out.push(("an id used twice", repeated));
    let mut outside = good.to_vec();
    outside[at].0 = vocab as u32;
    out.push(("an id outside the vocabulary", outside));
    for (name, value) in [("a NaN", f32::NAN), ("an infinity", f32::INFINITY)] {
        let mut not_finite = good.to_vec();
        not_finite[at].1 = value;
        out.push((name, not_finite));
    }
    let mut swapped = good.to_vec();
    swapped.swap(at, at + 1);
    out.push(("two entries out of order", swapped));
    let mut raised = good.to_vec();
    raised[at].1 = good[at - 1].1 + 1.0;
    out.push(("an entry ranked above the one before it", raised));
    out
}

#[test]
fn a_corrupt_later_entry_fails_the_check_that_reads_the_report_with_the_predicate_of_the_sampler() {
    use crate::family::sampler::check_shortlist;
    let vocab = 40;
    let row: Vec<f32> = (0..vocab)
        .map(|i| ((i * 7 + 3) % 29) as f32 - 9.0)
        .collect();
    let good = ranked_top(&row, SHORTLIST_K);
    let token = good[0].0 as i32;
    let params = SamplingParams::new(0.9, 0.95, 20).unwrap();
    // The sound record passes both the report check and the sampler.
    let sound = [record(&good, 0)];
    check_shortlists(&[token], &[0], &sound, vocab).expect("a sound record");
    check_shortlist(&good, vocab).expect("the shared predicate");
    Sampler::new(params, 1)
        .sample_shortlist(&good, vocab)
        .expect("the sampler");
    for (what, bad) in corruptions(&good, 7, vocab) {
        // Entry 0 is the argmax token and the non-finite word is clear: only the entry-by-entry
        // check can find it.
        let rec = [record(&bad, 0)];
        let text = format!("{:?}", check_shortlists(&[token], &[0], &rec, vocab));
        assert!(
            text.starts_with("Err") && text.contains("row 0"),
            "{what}: {text}"
        );
        // And it is the predicate the sampler applies: the two agree on every corruption.
        assert!(check_shortlist(&bad, vocab).is_err(), "{what}");
        assert!(
            Sampler::new(params, 1)
                .sample_shortlist(&bad, vocab)
                .is_err(),
            "{what}"
        );
    }
    // A record of fewer entries than the vocabulary allows is the wrong length for both.
    assert!(check_shortlist(&good[..19], vocab).is_err());
    assert!(
        Sampler::new(params, 1)
            .sample_shortlist(&good[..19], vocab)
            .is_err()
    );
}

// ---- which draws need a full row ----

#[test]
fn only_a_record_and_a_sampler_that_keeps_one_to_twenty_candidates_draw_from_the_shortlist() {
    let keeps = |top_k: u32| SamplingParams::new(0.8, 0.95, top_k).unwrap();
    for top_k in [1, 5, 20] {
        assert_eq!(
            draw_source(keeps(top_k), false, true, true),
            DrawSource::Shortlist
        );
    }
    for top_k in [0, 21, 64] {
        assert_eq!(
            draw_source(keeps(top_k), false, true, true),
            DrawSource::Row,
            "{top_k}"
        );
    }
    assert_eq!(
        draw_source(keeps(20), true, true, true),
        DrawSource::Row,
        "sync each"
    );
    assert_eq!(
        draw_source(keeps(20), false, false, true),
        DrawSource::Row,
        "vocabulary"
    );
    assert_eq!(
        draw_source(keeps(20), false, true, false),
        DrawSource::Row,
        "no record"
    );
    assert_eq!(
        draw_source(SamplingParams::greedy(), false, true, true),
        DrawSource::Row
    );
}

#[test]
fn the_draws_belong_to_the_window_that_began_at_their_base_over_their_ids() {
    let draws = DeviceDraws {
        base: 12,
        inputs: vec![1, 2, 3],
        accepted: 1,
        bonus: 2,
        shortlists: Vec::new(),
    };
    assert!(draws.belongs_to(12, &[1, 2, 3]));
    assert!(!draws.belongs_to(13, &[1, 2, 3]));
    assert!(!draws.belongs_to(12, &[1, 2, 4]));
    assert!(!draws.belongs_to(12, &[1, 2]));
}

// ---- the route of a draw ----

fn draws_of(base: usize, inputs: &[u32], rows_with_record: usize) -> DeviceDraws {
    DeviceDraws {
        base,
        inputs: inputs.to_vec(),
        accepted: 0,
        bonus: 0,
        shortlists: (0..inputs.len())
            .map(|row| {
                if row < rows_with_record {
                    vec![(row as u32, 1.0)]
                } else {
                    Vec::new()
                }
            })
            .collect(),
    }
}

fn route_facts<'a>(draws: Option<&'a DeviceDraws>, inputs: &'a [u32]) -> DrawFacts<'a> {
    DrawFacts {
        krow_runs: true,
        device_window: true,
        draws,
        base: 12,
        inputs,
        row: 0,
        tap_row: false,
        params: SamplingParams::new(0.8, 0.95, 20).unwrap(),
        sync_each: false,
        vocab_matches: true,
    }
}

#[test]
fn a_draw_goes_to_the_record_the_tap_copy_or_one_device_row() {
    let inputs = [1u32, 2, 3];
    let draws = draws_of(12, &inputs, 3);
    let facts = route_facts(Some(&draws), &inputs);
    assert_eq!(draw_route(&facts), DrawRoute::Shortlist);
    // A sampler that keeps 1 to 20 candidates is served by the record, tap or not.
    for top_k in [1, 5, 20] {
        let params = SamplingParams::new(0.8, 0.95, top_k).unwrap();
        for tap_row in [false, true] {
            let served = DrawFacts {
                params,
                tap_row,
                ..facts
            };
            assert_eq!(
                draw_route(&served),
                DrawRoute::Shortlist,
                "{top_k} {tap_row}"
            );
        }
    }
    // A sampler the record cannot serve reads the tap's copy when there is one, else one row.
    for top_k in [0, 21, 64] {
        let params = SamplingParams::new(0.8, 0.95, top_k).unwrap();
        assert_eq!(
            draw_route(&DrawFacts { params, ..facts }),
            DrawRoute::DeviceRow
        );
        let tapped = DrawFacts {
            params,
            tap_row: true,
            ..facts
        };
        assert_eq!(draw_route(&tapped), DrawRoute::TapRow);
    }
    // The other reasons the record does not serve it.
    assert_eq!(
        draw_route(&DrawFacts {
            sync_each: true,
            ..facts
        }),
        DrawRoute::DeviceRow
    );
    assert_eq!(
        draw_route(&DrawFacts {
            vocab_matches: false,
            ..facts
        }),
        DrawRoute::DeviceRow
    );
    assert_eq!(
        draw_route(&DrawFacts {
            vocab_matches: false,
            tap_row: true,
            ..facts
        }),
        DrawRoute::TapRow
    );
    // A row without a record (the shortlist is off, or the window has none for it).
    let none = draws_of(12, &inputs, 0);
    assert_eq!(
        draw_route(&route_facts(Some(&none), &inputs)),
        DrawRoute::DeviceRow
    );
    let first_only = draws_of(12, &inputs, 1);
    let second = DrawFacts {
        row: 1,
        ..route_facts(Some(&first_only), &inputs)
    };
    assert_eq!(draw_route(&second), DrawRoute::DeviceRow);
    assert_eq!(
        draw_route(&DrawFacts {
            row: 0,
            ..route_facts(Some(&first_only), &inputs)
        }),
        DrawRoute::Shortlist
    );
}

#[test]
fn a_row_past_the_window_is_refused_and_a_window_the_k_row_path_did_not_run_takes_the_default_path()
{
    let inputs = [1u32, 2, 3];
    let draws = draws_of(12, &inputs, 3);
    let facts = route_facts(Some(&draws), &inputs);
    for row in [3, 4, 100] {
        assert_eq!(
            draw_route(&DrawFacts { row, ..facts }),
            DrawRoute::OutOfWindow,
            "{row}"
        );
    }
    // The default path: the path is off for this window (an eager window with the same base and
    // inputs as the window the records belong to), the state is not in the slots, there are no
    // records, or the records are another window's.
    for (what, other) in [
        (
            "the path does not run it",
            DrawFacts {
                krow_runs: false,
                ..facts
            },
        ),
        (
            "host rollback",
            DrawFacts {
                device_window: false,
                ..facts
            },
        ),
        (
            "no records",
            DrawFacts {
                draws: None,
                ..facts
            },
        ),
        ("another base", DrawFacts { base: 13, ..facts }),
        (
            "other inputs",
            DrawFacts {
                inputs: &[1, 2, 4],
                ..facts
            },
        ),
        (
            "a shorter window",
            DrawFacts {
                inputs: &[1, 2],
                ..facts
            },
        ),
    ] {
        assert_eq!(draw_route(&other), DrawRoute::NotKrow, "{what}");
        // And a row past the window of such a window is the default path's business too.
        assert_eq!(
            draw_route(&DrawFacts { row: 50, ..other }),
            DrawRoute::NotKrow,
            "{what}"
        );
    }
}

// ---- a seeded draw from the K shortlists is the one-row draw ----

/// For fixed rows and seeds, drawing row after row from the top-20 records gives the tokens and
/// leaves the generator where `Sampler::sample` on the full rows gives and leaves it, including a
/// window that stops early (rows after the run are never drawn). Several settings; top-k from 1
/// to 20, the temperature and the nucleus cut vary.
#[test]
fn drawing_rows_from_their_records_is_drawing_them_from_the_full_rows() {
    let vocab = 300;
    let rows: Vec<Vec<f32>> = (0..8)
        .map(|r| {
            (0..vocab)
                .map(|i| (((i * 31 + r * 17) % 97) as f32) / 7.0 - 6.0)
                .collect()
        })
        .collect();
    let records: Vec<Vec<(u32, f32)>> = rows.iter().map(|row| ranked_top(row, 20)).collect();
    for (temperature, top_p, top_k) in [
        (1.0, 1.0, 20),
        (0.8, 0.95, 20),
        (1.3, 0.9, 5),
        (0.7, 0.0, 1),
    ] {
        let params = SamplingParams::new(temperature, top_p, top_k).unwrap();
        for seed in [0u64, 7, 12345, u64::MAX] {
            for drawn in 1..=rows.len() {
                let mut from_records = Sampler::new(params, seed);
                let mut from_rows = Sampler::new(params, seed);
                for row in 0..drawn {
                    assert_eq!(
                        from_records.sample_shortlist(&records[row], vocab).unwrap(),
                        from_rows.sample(&rows[row]).unwrap(),
                        "params {params:?} seed {seed} row {row}"
                    );
                }
                assert_eq!(
                    from_records.rng_state(),
                    from_rows.rng_state(),
                    "the generator after {drawn} rows"
                );
            }
        }
    }
}
