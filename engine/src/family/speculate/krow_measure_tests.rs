//! Host tests of the K-row measurement arithmetic (`krow_measure`) and of the guards the device
//! tests carry. No device is needed.

use super::krow_measure::{
    Inputs, alpha, first_word_difference, mean, percentile, rotated, union_sizes, weighted_alpha,
};

#[test]
fn alpha_is_the_marginal_cost_over_the_solo_cost() {
    // t(1) = 10, t(2) = 12.8: one more row costs 0.28 of the first.
    assert!((alpha(10.0, 12.8, 2).unwrap() - 0.28).abs() < 1e-12);
    // t(4) = 10 + 3 * 3.3: alpha 0.33.
    assert!((alpha(10.0, 19.9, 4).unwrap() - 0.33).abs() < 1e-12);
    // No marginal cost is 0 and a fully per-row launch is 1.
    assert_eq!(alpha(10.0, 10.0, 8), Some(0.0));
    assert_eq!(alpha(10.0, 80.0, 8), Some(1.0));
    // R = 1, an empty solo time and non-finite readings have no alpha.
    assert_eq!(alpha(10.0, 10.0, 1), None);
    assert_eq!(alpha(0.0, 5.0, 2), None);
    assert_eq!(alpha(f64::NAN, 5.0, 2), None);
    assert_eq!(alpha(10.0, f64::INFINITY, 2), None);
}

#[test]
fn the_weighted_alpha_weighs_each_role_by_its_solo_time() {
    // A heavy role at 0.1 and a light one at 0.5: (0.1*90 + 0.5*10) / 100 = 0.14, not the mean 0.3.
    let roles = [(90.0, 90.0 + 9.0), (10.0, 10.0 + 5.0)];
    assert!((weighted_alpha(&roles, 2).unwrap() - 0.14).abs() < 1e-12);
    // Three rows: the marginal time is per extra row.
    let roles = [(100.0, 100.0 + 2.0 * 30.0)];
    assert!((weighted_alpha(&roles, 3).unwrap() - 0.3).abs() < 1e-12);
    assert_eq!(weighted_alpha(&[], 2), None);
    assert_eq!(weighted_alpha(&[(1.0, 2.0)], 1), None);
}

#[test]
fn the_first_difference_names_row_column_and_index() {
    let words =
        |values: &[u32]| -> Vec<u8> { values.iter().flat_map(|v| v.to_le_bytes()).collect() };
    let left = words(&[1, 2, 3, 4, 5, 6]);
    assert_eq!(first_word_difference(&left, &left, 3), None);
    let mut right = left.clone();
    right[4 * 4] ^= 1; // word 4, which is row 1, column 1 of rows of three
    let found = first_word_difference(&left, &right, 3).unwrap();
    assert_eq!((found.row, found.col, found.index), (1, 1, 4));
    assert_eq!((found.left_bits, found.right_bits), (5, 4));
    // The first of two differences is the one reported.
    right[4] ^= 0x80;
    assert_eq!(first_word_difference(&left, &right, 3).unwrap().index, 1);
    // A signed zero is a difference: the comparison is on bits.
    let plus = words(&[0.0f32.to_bits()]);
    let minus = words(&[(-0.0f32).to_bits()]);
    assert_eq!(first_word_difference(&plus, &minus, 1).unwrap().index, 0);
    // A length difference is never hidden.
    assert!(first_word_difference(&left, &left[..8], 3).is_some());
}

#[test]
fn the_union_counts_distinct_ids_over_sliding_windows() {
    let picks = vec![vec![1, 2, 3], vec![3, 4, 5], vec![1, 2, 6], vec![1, 2, 3]];
    assert_eq!(union_sizes(&picks, 1).unwrap(), vec![3, 3, 3, 3]);
    assert_eq!(union_sizes(&picks, 2).unwrap(), vec![5, 6, 4]);
    assert_eq!(union_sizes(&picks, 4).unwrap(), vec![6]);
    // A window longer than the stream has no start.
    assert!(union_sizes(&picks, 5).unwrap().is_empty());
    assert!(union_sizes(&picks, 0).is_err());
    // An invalid pick (-1 marks a faulty router row) refuses the whole layer.
    assert!(union_sizes(&[vec![1, -1]], 1).is_err());
}

#[test]
fn percentiles_are_nearest_rank() {
    let values = [5.0, 1.0, 4.0, 2.0, 3.0, 10.0, 9.0, 8.0, 7.0, 6.0];
    assert_eq!(percentile(&values, 90.0), Some(9.0));
    assert_eq!(percentile(&values, 100.0), Some(10.0));
    assert_eq!(percentile(&values, 0.0), Some(1.0));
    assert_eq!(percentile(&[], 90.0), None);
    assert_eq!(percentile(&values, 101.0), None);
    assert_eq!(mean(&[1.0, 2.0, 6.0]), Some(3.0));
    assert_eq!(mean(&[]), None);
}

#[test]
fn rotation_starts_each_round_at_a_different_item() {
    let items = [0, 1, 2, 3];
    assert_eq!(rotated(&items, 0), vec![0, 1, 2, 3]);
    assert_eq!(rotated(&items, 1), vec![1, 2, 3, 0]);
    assert_eq!(rotated(&items, 6), vec![2, 3, 0, 1]);
    assert!(rotated::<u8>(&[], 3).is_empty());
}

#[test]
fn the_inputs_are_deterministic_and_unit_scale() {
    let a = Inputs::new(1).values(20_000);
    assert_eq!(a, Inputs::new(1).values(20_000));
    assert_ne!(a, Inputs::new(2).values(20_000));
    let n = a.len() as f64;
    let m = a.iter().map(|v| f64::from(*v)).sum::<f64>() / n;
    let var = a.iter().map(|v| (f64::from(*v) - m).powi(2)).sum::<f64>() / n;
    assert!(m.abs() < 0.05, "mean {m}");
    assert!((var - 1.0).abs() < 0.1, "variance {var}");
    assert!(a.iter().all(|v| v.is_finite()));
}

/// The device tests must refuse the non-default paths before they touch the device: the switch
/// guard is the first statement of every measurement test, and the identity records the effective
/// value of the shared-expert switch.
#[test]
fn every_measurement_test_refuses_the_switches_first() {
    for (name, source) in [
        (
            "krow_alpha_device_tests.rs",
            include_str!("../qwen36/cuda/krow_alpha_device_tests.rs"),
        ),
        (
            "krow_step_device_tests.rs",
            include_str!("../qwen36/cuda/krow_step_device_tests.rs"),
        ),
    ] {
        let mut tests = 0;
        for (at, _) in source.match_indices("#[ignore") {
            let rest = &source[at..];
            let body = &rest[rest.find("fn ").unwrap()..];
            let open = body.find('{').unwrap();
            let first = body[open + 1..].trim_start();
            assert!(
                first.starts_with("refuse_switches();"),
                "{name}: a test does not refuse first"
            );
            tests += 1;
        }
        assert!(tests >= 1, "{name}");
    }
    let alpha = include_str!("../qwen36/cuda/krow_alpha_device_tests.rs");
    assert!(alpha.contains("FIFI_EXL3_SHARED_IN_ROUTED") && alpha.contains("\"effective\""));
    assert!(
        alpha.contains("FIFI_EXL3_CHECKPOINT"),
        "the one switch the guard allows"
    );
}
