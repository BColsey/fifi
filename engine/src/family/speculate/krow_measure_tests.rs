//! Host tests of the K-row measurement arithmetic (`krow_measure`) and of the guards the device
//! tests carry. No device is needed.

use super::krow_measure::{
    ALLOWED_VARS, Inputs, ROWS, Reading, SEP_CODE_OFFSET, UNION_CAVEAT, alpha, check_default_path,
    decode_receipt_marker, depth_json, first_word_difference, mean, percentile,
    receipt_marker_bytes, refused_switches, rotated, union_json, union_sizes, weighted_alpha,
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
    assert!(
        alpha.contains("check_default_path("),
        "the guard reads the shared check"
    );
    assert!(alpha.contains("FIFI_EXL3_SHARED_IN_ROUTED") && alpha.contains("\"effective\""));
    assert!(
        alpha.contains("FIFI_EXL3_CHECKPOINT"),
        "the one switch the guard allows"
    );
}

#[test]
fn the_rows_include_three_and_six() {
    assert_eq!(ROWS, [1, 2, 3, 4, 6, 8]);
}

#[test]
fn every_switch_that_changes_the_step_is_refused() {
    let names = |list: &[&str]| list.iter().map(|s| (*s).to_owned()).collect::<Vec<_>>();
    // The ones the review named, and the single-row and shared-expert switches.
    let bad = [
        "INFERENCE_CUDA_GDN_FUSE_INPROJ",
        "INFERENCE_QWEN36_SPARSE_ATTN",
        "Q36_NO_ACTMUL",
        "Q36_CUDA_DEVICE_SHORTLIST",
        "INFERENCE_CUDA_KV8_ATTN_R1",
        "INFERENCE_CUDA_SYNC_EACH",
        "Q36_NO_MOECOMBINE",
        "Q36_NO_GDNPREP",
        "Q36_FLASH_DELTA",
        "FIFI_EXL3_SINGLE_GATHER",
        "FIFI_EXL3_SHARED_IN_ROUTED",
        "FIFI_EXL3_DP4A",
        "FIFI_EXL3_INDEXED_PAIRS",
        "FIFI_EXL3_LEGACY",
        "FIFI_GRAPH_BENCH_PHASES",
        "FIFI_DELTA_TRACE",
    ];
    assert_eq!(refused_switches(names(&bad)), names(&bad));
    // What the runner sets stays allowed, with the measurement's own variables.
    let fine = [
        "FIFI_EXL3_CHECKPOINT",
        "INFERENCE_CUDA_REQUIRE_DEVICE",
        "FIFI_SPEC_TEXT",
        "FIFI_SPEC_TOKENIZER",
        "FIFI_KROW_OUT",
        "FIFI_KROW_ROUNDS",
        "PATH",
        "LD_LIBRARY_PATH",
        "TMPDIR",
    ];
    assert!(refused_switches(names(&fine)).is_empty());
    assert_eq!(ALLOWED_VARS.len(), 2);
}

#[test]
fn the_receipt_markers_round_trip() {
    for op in 0..20 {
        for rows in ROWS {
            let bytes = receipt_marker_bytes(op, rows);
            assert_eq!(decode_receipt_marker(bytes), Some((op, rows)));
            assert!(bytes < (1 << 20));
        }
    }
    // Sizes that are not markers: the 128 MiB flush, a bare power of two, a row count of zero.
    assert_eq!(decode_receipt_marker(128 << 20), None);
    assert_eq!(decode_receipt_marker(4096), None);
    assert_eq!(decode_receipt_marker(4096 + 7), None);
    assert_eq!(decode_receipt_marker(100), None);
}

#[test]
fn the_depth_record_scales_each_marked_layer_by_its_run_size() {
    use crate::family::AttentionKind::{DeltaNet, Full};
    // Marked layers 1 (GDN, run of 3), 3 (attention, run of 1), 5 (GDN, run of 3), 7 (attention).
    let plan = [
        (1, 3, DeltaNet),
        (3, 1, Full),
        (5, 3, DeltaNet),
        (7, 1, Full),
    ];
    // GDN layers cost 0.2 and 0.4 ms, attention layers 0.3 and 0.5 ms; embed 0.01, tail 0.8.
    let group = [0.2, 0.3, 0.4, 0.5];
    // The whole token the marks would read if the unmarked layers cost what their group does.
    let whole = 0.01 + 0.8 + 3.0 * 0.2 + 0.3 + 3.0 * 0.4 + 0.5;
    let one = |wall_ms: f64| Reading {
        wall_ms,
        whole_ms: whole,
        embed_ms: 0.01,
        tail_ms: 0.8,
        groups: group.to_vec(),
    };
    let readings = vec![one(9.0), one(9.5)];
    let json = depth_json(8192, &readings, &plan);
    let gdn = json["gdn_layers_ms_scaled_to_30 (mixer + moe)"]["median"]
        .as_f64()
        .unwrap();
    let full = json["full_attention_layers_ms_scaled_to_10 (mixer + moe)"]["median"]
        .as_f64()
        .unwrap();
    assert!((gdn - (3.0 * 0.2 + 3.0 * 0.4)).abs() < 1e-12, "{gdn}");
    assert!((full - (0.3 + 0.5)).abs() < 1e-12, "{full}");
    // The scaled sections plus embed and tail make the whole token.
    let ratio = json["marked_sum_over_whole_token"]["median"]
        .as_f64()
        .unwrap();
    assert!((ratio - 1.0).abs() < 1e-12, "{ratio}");
    let shares = &json["median_shares_of_gpu_event"];
    let total: f64 = ["embed", "tail", "gdn_layers", "full_attention_layers"]
        .iter()
        .map(|k| shares[k].as_f64().unwrap())
        .sum();
    assert!((total - 1.0).abs() < 1e-12);
    assert_eq!(json["groups"].as_array().unwrap().len(), 4);
    assert_eq!(json["groups"][1]["kind"], "Full");
    assert_eq!(json["tokens_timed"], 2);
}

#[test]
fn the_union_record_pools_windows_and_layers() {
    // Two layers, four tokens, two picks each (top_k = 2 keeps the arithmetic readable).
    let layer0 = vec![vec![1, 2], vec![3, 4], vec![1, 2], vec![5, 6]];
    let layer1 = vec![vec![7, 8], vec![7, 8], vec![7, 9], vec![7, 8]];
    let json = union_json(&[layer0, layer1]);
    // K = 1: every window is top_k wide.
    assert_eq!(json["1"]["mean"], 2.0);
    assert_eq!(json["1"]["windows_total"], 8);
    // K = 2: layer 0 windows {1,2,3,4}=4, {3,4,1,2}=4, {1,2,5,6}=4; layer 1: {7,8}=2, {7,8,9}=3, {7,9,8}=3.
    let k2 = &json["2"];
    assert_eq!(k2["windows_total"], 6);
    assert!((k2["mean"].as_f64().unwrap() - (12.0 + 8.0) / 6.0).abs() < 1e-12);
    assert_eq!(k2["p90"], 4.0);
    assert_eq!(k2["per_layer"][0]["mean"], 4.0);
    assert!((k2["per_layer"][1]["mean"].as_f64().unwrap() - 8.0 / 3.0).abs() < 1e-12);
    assert!((k2["mean_over_k_times_top_k"].as_f64().unwrap() - (20.0 / 6.0) / 4.0).abs() < 1e-12);
    // K = 4 has one window per layer; K = 5 and up have none (mean is null).
    assert_eq!(json["4"]["windows_total"], 2);
    assert!(json["5"]["mean"].is_null());
    for k in 1..=8 {
        assert!(json[k.to_string()].is_object());
    }
}

#[test]
fn the_union_caveat_names_what_the_windows_are() {
    for word in ["true-continuation", "depth 1K to 4K", "EAGER"] {
        assert!(UNION_CAVEAT.contains(word), "{word}");
    }
}

#[test]
fn each_refused_switch_alone_stops_the_start() {
    for name in [
        "INFERENCE_CUDA_GDN_FUSE_INPROJ",
        "INFERENCE_QWEN36_SPARSE_ATTN",
        "Q36_NO_ACTMUL",
        "Q36_CUDA_DEVICE_SHORTLIST",
        "INFERENCE_CUDA_KV8_ATTN_R1",
        "INFERENCE_CUDA_SYNC_EACH",
        "FIFI_EXL3_SINGLE_GATHER",
        "FIFI_EXL3_SHARED_IN_ROUTED",
        "FIFI_EXL3_DP4A",
        "FIFI_EXL3_INDEXED_PAIRS",
        "FIFI_GRAPH_BENCH_PHASES",
        "FIFI_DELTA_TRACE",
    ] {
        let env = ["FIFI_EXL3_CHECKPOINT", "FIFI_KROW_OUT", name].map(str::to_owned);
        let error = check_default_path(env, Ok(false)).expect_err(name);
        assert!(error.contains(name), "{name}: {error}");
    }
    let clean = [
        "FIFI_EXL3_CHECKPOINT",
        "INFERENCE_CUDA_REQUIRE_DEVICE",
        "FIFI_KROW_OUT",
    ]
    .map(str::to_owned);
    assert_eq!(check_default_path(clean.clone(), Ok(false)), Ok(()));
}

#[test]
fn the_shared_in_routed_reading_stops_the_start_whatever_the_names() {
    let clean = ["FIFI_EXL3_CHECKPOINT".to_owned()];
    let on = check_default_path(clean.clone(), Ok(true)).unwrap_err();
    assert!(on.contains("SHARED_IN_ROUTED is on"), "{on}");
    let bad =
        check_default_path(clean.clone(), Err("must be unset, \"0\" or \"1\"".into())).unwrap_err();
    assert!(bad.contains("FIFI_EXL3_SHARED_IN_ROUTED"), "{bad}");
    // Both reasons are named together.
    let both = check_default_path(["Q36_NO_ACTMUL".to_owned()], Ok(true)).unwrap_err();
    assert!(
        both.contains("Q36_NO_ACTMUL") && both.contains("is on"),
        "{both}"
    );
}

#[test]
fn a_separate_launch_code_stays_inside_the_marker_range() {
    for rows in ROWS {
        let code = SEP_CODE_OFFSET + rows;
        assert!(rows < SEP_CODE_OFFSET);
        assert_eq!(
            decode_receipt_marker(receipt_marker_bytes(3, code)),
            Some((3, code))
        );
    }
}
