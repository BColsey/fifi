//! What the K-row device tests (in `d1_window_device_tests.rs`) share: the refusal of the kernel
//! switches the K-row path is not qualified for (the production `check_krow_env`, called with the
//! values of the environment), the check on the counters of a run, the comparison of per-layer
//! taps, and the decoding of the result table by the production `WindowResult::from_bytes`. All of
//! it is host logic and runs without a device (these tests are compiled with the native backend
//! and run in the CPU-only native gate); the device tests that use it start with
//! [`refuse_switches`], so a run with a refused switch set stops before it loads the pack.

use std::ffi::OsString;

use super::{
    d1_window_device_tests::{require_krow_enqueued, require_krow_idle},
    step::LayerTap,
};
use crate::{
    family::qwen36::{
        exl3_merge_plan::SHARED_IN_ROUTED_VAR,
        krow::{RowStatus, SINGLE_GATHER_NAME, check_krow_env, rows_verdict},
    },
    kernels::{
        cuda::resident::argmax_rows::{WindowResult, window_result_bytes},
        row_launches::RowLaunches,
    },
};

/// The refusal of a switch that selects another kernel or another MoE body than the one the K-row
/// path is qualified on, given a way to read the environment: the production rule
/// (`krow::check_krow_env`, the one the start runs) over the two values. The switches that leave
/// the K-row bytes and counters unchanged (`FIFI_EXL3_DP4A`, `FIFI_EXL3_INDEXED_PAIRS`,
/// `FIFI_EXL3_LEGACY`, `FIFI_D1_CAPTURE`) are run on purpose and are not refused. Unset and `0` of
/// the merged-block switch are not refused: the default block yields to the K-row path.
pub(super) fn switch_refusal(get: impl Fn(&str) -> Option<OsString>) -> Result<(), String> {
    check_krow_env(
        get(SINGLE_GATHER_NAME).as_deref(),
        get(SHARED_IN_ROUTED_VAR).as_deref(),
    )
    .map_err(|error| format!("{error:?}"))
}

/// [`switch_refusal`] over the process environment.
pub(super) fn refuse_switches() -> Result<(), String> {
    switch_refusal(|name| std::env::var_os(name))
}

/// Check B: the first difference between the per-layer taps of a K-row window and those of the
/// one-row steps of the same rows, in launch order and, within a tap, row by row. `window` holds the
/// taps of the K-row body (every tap has one residual row per window row); `steps[t]` holds the taps
/// of the one-row step of row `t` (one residual row each). `None` when every tap of every row is
/// equal to the byte. The message names the layer, the block after which the residual was read
/// and the row, so a failure of the end-to-end checks can be bisected to a layer.
pub(super) fn tap_difference(window: &[LayerTap], steps: &[Vec<LayerTap>]) -> Option<String> {
    for (row, step) in steps.iter().enumerate() {
        if step.len() != window.len() {
            return Some(format!(
                "the one-row step of row {row} has {} taps, the window {}",
                step.len(),
                window.len()
            ));
        }
    }
    for (at, tap) in window.iter().enumerate() {
        let block = if tap.after_moe { "MoE" } else { "mixer" };
        for (row, step) in steps.iter().enumerate() {
            let one = &step[at];
            if (one.layer, one.after_moe) != (tap.layer, tap.after_moe) {
                return Some(format!(
                    "tap {at}: the window read layer {} after the {block} block, the one-row step of row {row} layer {}",
                    tap.layer, one.layer
                ));
            }
            let (Some(wide), Some(single)) = (tap.rows.get(row), one.rows.first()) else {
                return Some(format!(
                    "layer {} after the {block} block: row {row} was not read",
                    tap.layer
                ));
            };
            if wide.len() != single.len() {
                return Some(format!(
                    "layer {} after the {block} block, row {row}: {} bytes against {}",
                    tap.layer,
                    wide.len(),
                    single.len()
                ));
            }
            if let Some(byte) = wide.iter().zip(single).position(|(a, b)| a != b) {
                return Some(format!(
                    "layer {} after the {block} block, row {row}, byte {byte}",
                    tap.layer
                ));
            }
        }
    }
    None
}

fn env<'a>(pairs: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<OsString> + 'a {
    move |name| {
        pairs
            .iter()
            .find(|(key, _)| *key == name)
            .map(|(_, value)| OsString::from(value))
    }
}

#[test]
fn an_unset_environment_and_the_switches_that_leave_the_bytes_alone_are_not_refused() {
    switch_refusal(env(&[])).expect("nothing set");
    for allowed in [
        ("FIFI_EXL3_DP4A", "1"),
        ("FIFI_EXL3_INDEXED_PAIRS", "20"),
        ("FIFI_EXL3_LEGACY", "1"),
        ("FIFI_D1_CAPTURE", "0"),
        (SHARED_IN_ROUTED_VAR, "0"),
    ] {
        switch_refusal(env(&[allowed])).unwrap_or_else(|error| panic!("{allowed:?}: {error}"));
    }
}

#[test]
fn every_value_of_the_single_row_switch_and_the_merged_block_are_refused_and_say_unset() {
    for value in ["off", "1", "wide1", "wide2", "wide4", ""] {
        let error =
            switch_refusal(env(&[(SINGLE_GATHER_NAME, value)])).expect_err("a refused switch");
        assert!(error.contains("unset it"), "{value}: {error}");
    }
    let error = switch_refusal(env(&[(SHARED_IN_ROUTED_VAR, "1")])).expect_err("refused");
    assert!(error.contains("unset it"), "{error}");
}

fn counts(argmax: u64, fold: u64, pack: u64, conv: u64, delta: u64, router: u64) -> RowLaunches {
    RowLaunches {
        argmax_rows: argmax,
        status_fold: fold,
        pack_rows: pack,
        conv_snap: conv,
        delta_snap: delta,
        router_strided: router,
        ..RowLaunches::default()
    }
}

#[test]
fn a_run_that_enqueued_the_k_row_kernels_once_per_window_passes_the_counter_check() {
    require_krow_enqueued(&counts(3, 3, 120, 90, 90, 180), 3).expect("a K-row run");
}

#[test]
fn a_run_with_no_window_or_a_missing_kernel_family_fails_the_counter_check() {
    assert!(require_krow_enqueued(&RowLaunches::default(), 0).is_err());
    // An eager run claiming K-row windows: windows counted, nothing enqueued.
    assert!(require_krow_enqueued(&RowLaunches::default(), 3).is_err());
    // One family missing, one at a time.
    for missing in 0..6 {
        let mut all = [3, 3, 120, 90, 90, 180];
        all[missing] = 0;
        let run = counts(all[0], all[1], all[2], all[3], all[4], all[5]);
        assert!(
            require_krow_enqueued(&run, 3).is_err(),
            "family {missing} missing"
        );
    }
    // The argmax and the fold count exactly once per window: two folds in a window are wrong.
    assert!(require_krow_enqueued(&counts(3, 6, 120, 90, 90, 180), 3).is_err());
}

fn tap(layer: usize, after_moe: bool, rows: &[&[u8]]) -> LayerTap {
    LayerTap {
        layer,
        after_moe,
        rows: rows.iter().map(|row| row.to_vec()).collect(),
    }
}

/// A window of three rows over three layers and the three one-row steps that equal it.
fn equal_taps() -> (Vec<LayerTap>, Vec<Vec<LayerTap>>) {
    let rows: [[u8; 4]; 3] = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]];
    let mut window = Vec::new();
    for layer in 0..3 {
        for after_moe in [false, true] {
            let shifted: Vec<[u8; 4]> = rows
                .iter()
                .map(|row| row.map(|b| b + layer as u8))
                .collect();
            let refs: Vec<&[u8]> = shifted.iter().map(|row| &row[..]).collect();
            window.push(tap(layer, after_moe, &refs));
        }
    }
    let steps = (0..3)
        .map(|row| {
            window
                .iter()
                .map(|w| tap(w.layer, w.after_moe, &[&w.rows[row][..]]))
                .collect()
        })
        .collect();
    (window, steps)
}

#[test]
fn equal_taps_have_no_difference() {
    let (window, steps) = equal_taps();
    assert_eq!(tap_difference(&window, &steps), None);
}

#[test]
fn the_first_differing_tap_is_named_by_layer_block_and_row() {
    let (mut window, steps) = equal_taps();
    // Two differences: the first in launch order is reported, not the first in row order.
    window[5].rows[0][2] ^= 1;
    window[3].rows[2][1] ^= 1;
    let text = tap_difference(&window, &steps).expect("a difference");
    assert!(
        text.contains("layer 1 after the MoE block, row 2, byte 1"),
        "{text}"
    );
    let (mut window, steps) = equal_taps();
    window[0].rows[1][0] ^= 0x80;
    let text = tap_difference(&window, &steps).expect("a difference");
    assert!(
        text.contains("layer 0 after the mixer block, row 1"),
        "{text}"
    );
}

#[test]
fn a_missing_tap_a_reordered_tap_and_a_short_row_are_differences() {
    let (window, mut steps) = equal_taps();
    steps[1].pop();
    assert!(
        tap_difference(&window, &steps)
            .expect("count")
            .contains("row 1 has")
    );
    let (mut window, steps) = equal_taps();
    window.swap(0, 1);
    assert!(
        tap_difference(&window, &steps)
            .expect("order")
            .contains("tap 0")
    );
    let (mut window, steps) = equal_taps();
    window[2].rows.pop();
    assert!(
        tap_difference(&window, &steps)
            .expect("rows")
            .contains("row 2 was not read")
    );
}

#[test]
fn the_eager_control_wants_every_real_enqueue_counter_at_zero() {
    require_krow_idle(&RowLaunches::default()).expect("an eager run");
    // The echo counters of the projection table do not count; each real-enqueue counter does.
    require_krow_idle(&RowLaunches {
        rows4: 7,
        small_wide1: 100,
        ..RowLaunches::default()
    })
    .expect("projection echoes are not asked");
    for run in [
        counts(1, 0, 0, 0, 0, 0),
        counts(0, 1, 0, 0, 0, 0),
        counts(0, 0, 1, 0, 0, 0),
        counts(0, 0, 0, 1, 0, 0),
        counts(0, 0, 0, 0, 1, 0),
        counts(0, 0, 0, 0, 0, 1),
    ] {
        assert!(require_krow_idle(&run).is_err(), "{}", run.describe());
    }
}

// ---- the result table, decoded by the production code ----

/// The `8 K + 8` bytes a K-row window leaves: K pairs, the window word and a zero pad word, each a
/// little-endian i32. This is the input the device writes; the decoding under test is production's.
fn table_bytes(tokens: &[i32], bits: &[i32], window: i32) -> Vec<u8> {
    let mut words = Vec::new();
    for (token, own) in tokens.iter().zip(bits) {
        words.extend([*token, *own]);
    }
    words.extend([window, 0]);
    words.iter().flat_map(|word| word.to_le_bytes()).collect()
}

/// A table of the longest window with a bad pair (row 6) and a window word, overwritten at its
/// start by a clean 3-row table: the bytes after the new table are the old window's.
fn short_after_long() -> Vec<u8> {
    let mut buffer = table_bytes(&[5; 8], &[0, 0, 0, 0, 0, 0, 2, 0], 4);
    let short = table_bytes(&[1, 2, 3], &[0, 0, 0], 0);
    buffer[..short.len()].copy_from_slice(&short);
    buffer
}

fn status_of(table: &WindowResult) -> RowStatus {
    RowStatus {
        bits: table.bits.iter().map(|&bits| bits as u32).collect(),
        window: table.window as u32,
    }
}

#[test]
fn a_short_window_after_a_long_one_on_the_same_table_reads_none_of_the_long_one() {
    // The device writes the new pairs, the new window word and the pad over the table; the bytes
    // after them are the old window's. The host reads exactly `8 K + 8` bytes and decodes them
    // with the production decoder.
    let buffer = short_after_long();
    let short = WindowResult::from_bytes(&buffer[..window_result_bytes(3)], 3).expect("decodes");
    assert_eq!(short.tokens, vec![1, 2, 3]);
    assert_eq!(short.bits, vec![0, 0, 0]);
    assert_eq!(short.window, 0);
    rows_verdict(&short.tokens, &status_of(&short), 8).expect("a clean window after a bad one");
    // The control: the long table itself fails, so the clean result is not vacuous.
    let long = table_bytes(&[5; 8], &[0, 0, 0, 0, 0, 0, 2, 0], 4);
    let table = WindowResult::from_bytes(&long, 8).expect("decodes");
    assert!(rows_verdict(&table.tokens, &status_of(&table), 8).is_err());
}

#[test]
fn a_host_that_decodes_more_rows_than_the_window_ran_sees_the_stale_bits() {
    // The mutant of the read length: the 3-row window read as the 8-row table of before. Row 6
    // shows the old bit 2 and the window word the old 4, and the verdict refuses it.
    let buffer = short_after_long();
    let wide = WindowResult::from_bytes(&buffer[..window_result_bytes(8)], 8).expect("decodes");
    assert_eq!(wide.bits[6], 2);
    assert_eq!(wide.window, 4);
    assert!(rows_verdict(&wide.tokens, &status_of(&wide), 8).is_err());
    // The decoder takes exactly the bytes of the rows it is told: any other length is refused.
    assert!(WindowResult::from_bytes(&buffer[..window_result_bytes(3)], 8).is_none());
    assert!(WindowResult::from_bytes(&buffer[..window_result_bytes(8)], 3).is_none());
    assert!(WindowResult::from_bytes(&buffer[..window_result_bytes(3) - 1], 3).is_none());
}

#[test]
fn a_table_with_a_non_zero_pad_word_does_not_decode() {
    let mut bytes = table_bytes(&[1, 2], &[0, 0], 0);
    let at = bytes.len() - 4;
    bytes[at] = 1;
    assert!(WindowResult::from_bytes(&bytes, 2).is_none());
}
