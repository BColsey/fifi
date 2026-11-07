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
    step::{GraphStats, LayerTap},
};
use crate::{
    family::qwen36::{
        exl3_merge_plan::SHARED_IN_ROUTED_VAR,
        krow::{RowStatus, SINGLE_GATHER_NAME, check_krow_env, rows_verdict},
    },
    kernels::{
        cuda::resident::{
            argmax_rows::{
                WindowReport, WindowResult, window_full_report_bytes, window_read_bytes,
                window_report_bytes, window_result_bytes, window_shortlist_offset,
            },
            shortlist::RESULT_BYTES as RECORD_BYTES,
        },
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

/// The windows that enqueued the K-row body whole: the eager windows and the capturing windows.
/// The kernel counters count at the enqueue (or at the capture), never at a replay, so this is the
/// number they must be compared with.
pub(super) fn graph_enqueued_windows(stats: &GraphStats) -> u64 {
    stats.eager_total() + stats.capture_total()
}

/// Check H for the graphs. `windows` are the K-row windows whose body completed; each of them ran
/// the eager body or launched a graph (the window that captured launches it once). With the graphs
/// off nothing may be captured or launched. A length is captured at most once, and a captured
/// length launched at least once (its capturing window).
pub(super) fn require_graph_counts(
    windows: u64,
    stats: &GraphStats,
    graphs_on: bool,
) -> Result<(), String> {
    if !graphs_on {
        if stats.capture_total() != 0 || stats.replay_total() != 0 {
            return Err(format!(
                "graphs are off and {} were captured, {} launched",
                stats.capture_total(),
                stats.replay_total()
            ));
        }
        if windows != stats.eager_total() {
            return Err(format!(
                "{windows} K-row windows, {} eager",
                stats.eager_total()
            ));
        }
        return Ok(());
    }
    let ran = stats.eager_total() + stats.replay_total();
    if windows != ran {
        return Err(format!(
            "{windows} K-row windows, {} eager and {} launched a graph",
            stats.eager_total(),
            stats.replay_total()
        ));
    }
    for rows in 0..stats.captures.len() {
        if stats.captures[rows] > 1 {
            return Err(format!(
                "the graph of {rows} rows was captured {} times",
                stats.captures[rows]
            ));
        }
        if stats.captures[rows] > stats.replays[rows] {
            return Err(format!(
                "the graph of {rows} rows was captured and never launched"
            ));
        }
    }
    Ok(())
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
    // The device verify and the shortlist run once per window, like the argmax.
    RowLaunches {
        argmax_rows: argmax,
        verify_greedy: argmax,
        shortlist_rows: argmax,
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

/// The device verify and the K-row shortlist run exactly once per window. (The shortlist count is
/// asked while the one-row shortlist is on, the default; a run with `Q36_CUDA_DEVICE_SHORTLIST=0`
/// asks it at zero instead.)
#[test]
fn the_device_verify_and_the_shortlist_are_each_counted_once_per_window() {
    let ok = counts(3, 3, 120, 90, 90, 180);
    require_krow_enqueued(&ok, 3).expect("one each per window");
    let mut missing = ok;
    missing.verify_greedy = 0;
    assert!(require_krow_enqueued(&missing, 3).is_err(), "no verify");
    let mut twice = ok;
    twice.verify_greedy = 6;
    assert!(require_krow_enqueued(&twice, 3).is_err(), "two verifies");
    let mut missing = ok;
    missing.shortlist_rows = 0;
    assert!(require_krow_enqueued(&missing, 3).is_err(), "no shortlist");
}

#[test]
fn the_eager_control_wants_the_device_verify_and_the_shortlist_at_zero() {
    for run in [
        RowLaunches {
            verify_greedy: 1,
            ..RowLaunches::default()
        },
        RowLaunches {
            shortlist_rows: 1,
            ..RowLaunches::default()
        },
    ] {
        assert!(require_krow_idle(&run).is_err(), "{}", run.describe());
    }
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

// ---- the device-to-host reads of a window ----

/// Reads shorter than this are the scalar health words of a step; every longer read is accounted
/// for by name.
const SCALAR_READ_LIMIT: usize = 64;

/// How many more reads of each length there are in `after` than in `before` (two snapshots of
/// `QwenResidentStep::d2h_reads`, which maps a read length in bytes to the number of reads of that
/// length). Only lengths that grew appear.
pub(super) fn d2h_growth(
    before: &std::collections::BTreeMap<usize, usize>,
    after: &std::collections::BTreeMap<usize, usize>,
) -> std::collections::BTreeMap<usize, usize> {
    after
        .iter()
        .filter_map(|(&len, &count)| {
            let earlier = before.get(&len).copied().unwrap_or(0);
            (count > earlier).then_some((len, count - earlier))
        })
        .collect()
}

/// The reads a K-row window may make, against the reads it made (`grown`, from [`d2h_growth`]):
/// exactly one read of `report` bytes, exactly the `extra` reads (length, count) the case asks for,
/// and no other read of [`SCALAR_READ_LIMIT`] bytes or more. A second read of the report, a read of
/// a row nobody asked for, or a missing report is an error.
pub(super) fn check_window_reads(
    grown: &std::collections::BTreeMap<usize, usize>,
    report: usize,
    extra: &[(usize, usize)],
) -> Result<(), String> {
    let mut expected = std::collections::BTreeMap::<usize, usize>::new();
    *expected.entry(report).or_default() += 1;
    for &(len, count) in extra {
        *expected.entry(len).or_default() += count;
    }
    let lengths: std::collections::BTreeSet<usize> = grown
        .keys()
        .chain(expected.keys())
        .copied()
        .filter(|len| *len >= SCALAR_READ_LIMIT)
        .collect();
    for len in lengths {
        let got = grown.get(&len).copied().unwrap_or(0);
        let want = expected.get(&len).copied().unwrap_or(0);
        if got != want {
            return Err(format!("{got} reads of {len} bytes, expected {want}"));
        }
    }
    Ok(())
}

#[test]
fn the_reads_of_a_window_are_the_reads_that_grew_between_two_snapshots() {
    use std::collections::BTreeMap;
    let before: BTreeMap<usize, usize> = [(4, 3), (1488, 1)].into();
    assert!(d2h_growth(&before, &before).is_empty());
    // One more report, two more scalars, and a logits row that was not read before.
    let after: BTreeMap<usize, usize> = [(4, 5), (1488, 2), (993_280, 1)].into();
    let grown = d2h_growth(&before, &after);
    assert_eq!(grown, [(4, 2), (1488, 1), (993_280, 1)].into());
    assert!(d2h_growth(&BTreeMap::new(), &BTreeMap::new()).is_empty());
}

#[test]
fn a_window_reads_its_report_once_and_what_the_case_asks_for_and_nothing_else() {
    use std::collections::BTreeMap;
    let row = 993_280;
    let reads = |pairs: &[(usize, usize)]| -> BTreeMap<usize, usize> { pairs.iter().copied().collect() };
    // The report alone, with scalar health words around it.
    check_window_reads(&reads(&[(4, 41), (1488, 1)]), 1488, &[]).expect("the report only");
    // A sampler the record cannot serve: one row per drawn row.
    check_window_reads(&reads(&[(1488, 1), (row, 3)]), 1488, &[(row, 3)]).expect("three rows");
    // The tap: all the rows in one copy.
    check_window_reads(&reads(&[(1488, 1), (8 * row, 1)]), 1488, &[(8 * row, 1)]).expect("the tap");
    // A second read of the report passes a byte bound but not this check.
    let twice = check_window_reads(&reads(&[(1488, 2)]), 1488, &[]).expect_err("two reports");
    assert!(twice.contains("2 reads of 1488"), "{twice}");
    // No report at all, a row nobody asked for, a missing row, a read of the wrong size.
    assert!(check_window_reads(&reads(&[(4, 1)]), 1488, &[]).is_err());
    assert!(check_window_reads(&reads(&[(1488, 1), (row, 1)]), 1488, &[]).is_err());
    assert!(check_window_reads(&reads(&[(1488, 1), (row, 2)]), 1488, &[(row, 3)]).is_err());
    assert!(check_window_reads(&reads(&[(1488, 1), (2 * row, 1)]), 1488, &[(row, 2)]).is_err());
    // A report of another size (the K - 1 mutant, or the shortlist on when it is off) fails.
    assert!(check_window_reads(&reads(&[(1312, 1)]), 1488, &[]).is_err());
}

// ---- the whole report of a window, decoded by the production code ----

/// The bytes a K-row window leaves for its one read: the table, the two words of the device verify,
/// padding to the next multiple of 16 (the device leaves those bytes alone: they hold garbage), then
/// one record per row.
fn report_bytes(
    tokens: &[i32],
    bits: &[i32],
    (window, accepted, bonus): (i32, i32, i32),
    records: &[[u8; RECORD_BYTES]],
) -> Vec<u8> {
    let mut bytes = table_bytes(tokens, bits, window);
    bytes.extend(accepted.to_le_bytes());
    bytes.extend(bonus.to_le_bytes());
    bytes.resize(window_shortlist_offset(tokens.len()), 0xA5);
    for record in records {
        bytes.extend(record);
    }
    bytes
}

#[test]
fn the_report_sizes_are_the_table_two_words_a_pad_to_16_and_a_record_per_row() {
    for m in 1..=16 {
        assert_eq!(window_report_bytes(m), window_result_bytes(m) + 8, "{m}");
        assert_eq!(window_report_bytes(m), 8 * m + 16);
        assert_eq!(
            window_shortlist_offset(m) % 16,
            0,
            "a record is 16-byte aligned"
        );
        assert!(window_shortlist_offset(m) - window_report_bytes(m) < 16);
        assert_eq!(
            window_full_report_bytes(m),
            window_shortlist_offset(m) + m * RECORD_BYTES
        );
        // What the host reads: the records when the shortlist ran, the table alone when it did not.
        assert_eq!(window_read_bytes(m, true), window_full_report_bytes(m));
        assert_eq!(window_read_bytes(m, false), window_report_bytes(m));
    }
    // The widest window of the K-row path reads about a kilobyte and a half, not 7.9 MB.
    assert_eq!(window_full_report_bytes(8), 1488);
    assert!(window_full_report_bytes(8) < 2048);
    assert_eq!(
        RECORD_BYTES,
        crate::family::qwen36::krow::SHORTLIST_RECORD_BYTES
    );
}

#[test]
fn the_report_decodes_the_pairs_the_window_word_the_verify_words_and_the_records() {
    for m in 2..=8usize {
        let tokens: Vec<i32> = (0..m as i32).map(|i| 10 + i).collect();
        let bits = vec![0; m];
        let records: Vec<[u8; RECORD_BYTES]> =
            (0..m).map(|i| [i as u8 + 1; RECORD_BYTES]).collect();
        let bytes = report_bytes(&tokens, &bits, (0, 1, 11), &records);
        assert_eq!(bytes.len(), window_full_report_bytes(m));
        let report = WindowReport::from_bytes(&bytes, m, true).expect("decodes");
        assert_eq!(report.table.tokens, tokens);
        assert_eq!((report.accepted, report.bonus), (1, 11));
        assert_eq!(report.shortlists, records, "{m} rows");
        // Without the shortlist the report stops after the verify words.
        let short = &bytes[..window_report_bytes(m)];
        let report = WindowReport::from_bytes(short, m, false).expect("decodes");
        assert!(report.shortlists.is_empty());
        assert_eq!((report.accepted, report.bonus), (1, 11));
    }
}

/// The mutant of the read length: a report sized for one row fewer than the window ran.
#[test]
fn a_report_read_for_one_row_fewer_does_not_decode() {
    for m in 2..=8usize {
        let tokens = vec![1; m];
        let bytes = report_bytes(&tokens, &vec![0; m], (0, 0, 1), &vec![[0; RECORD_BYTES]; m]);
        let short = &bytes[..window_full_report_bytes(m - 1)];
        assert!(WindowReport::from_bytes(short, m, true).is_none(), "{m}");
        assert!(WindowReport::from_bytes(&bytes[..bytes.len() - 1], m, true).is_none());
        // The records are not optional when the shortlist ran.
        assert!(WindowReport::from_bytes(&bytes[..window_report_bytes(m)], m, true).is_none());
    }
}

#[test]
fn the_padding_before_the_first_record_is_not_read() {
    let m = 3;
    let mut bytes = report_bytes(&[1, 2, 3], &[0; 3], (0, 2, 3), &[[7; RECORD_BYTES]; 3]);
    for byte in &mut bytes[window_report_bytes(m)..window_shortlist_offset(m)] {
        *byte ^= 0xFF;
    }
    let report = WindowReport::from_bytes(&bytes, m, true).expect("decodes");
    assert_eq!(report.shortlists[0], [7; RECORD_BYTES]);
    // A non-zero pad word of the table does not decode, as for the table alone.
    let mut dirty = bytes.clone();
    dirty[window_result_bytes(m) - 4] = 1;
    assert!(WindowReport::from_bytes(&dirty, m, true).is_none());
}

fn stats(
    eager: &[(usize, u64)],
    captures: &[(usize, u64)],
    replays: &[(usize, u64)],
) -> GraphStats {
    let mut stats = GraphStats::default();
    for &(rows, n) in eager {
        stats.eager[rows] = n;
    }
    for &(rows, n) in captures {
        stats.captures[rows] = n;
    }
    for &(rows, n) in replays {
        stats.replays[rows] = n;
    }
    stats
}

#[test]
fn the_graph_counters_of_a_run_with_graphs_add_up_to_the_windows() {
    // 4 rows: one eager, one capture (which launches), five replays; 2 rows: one eager only.
    let run = stats(&[(4, 1), (2, 1)], &[(4, 1)], &[(4, 6)]);
    require_graph_counts(8, &run, true).expect("windows are eager plus launches");
    assert_eq!(
        graph_enqueued_windows(&run),
        3,
        "two eager and one capture enqueued the body"
    );
    assert!(
        require_graph_counts(9, &run, true).is_err(),
        "a window nothing accounts for"
    );
    assert!(
        require_graph_counts(7, &run, true).is_err(),
        "a counted window that did not run"
    );
}

#[test]
fn a_run_with_the_graphs_off_must_not_have_captured_or_launched_one() {
    let eager_only = stats(&[(3, 4)], &[], &[]);
    require_graph_counts(4, &eager_only, false).expect("an eager run");
    let leaked = stats(&[(3, 3)], &[(3, 1)], &[(3, 1)]);
    assert!(require_graph_counts(4, &leaked, false).is_err());
    assert!(require_graph_counts(5, &eager_only, false).is_err());
}

#[test]
fn a_length_captured_twice_or_never_launched_fails_the_graph_counter_check() {
    let twice = stats(&[(5, 1)], &[(5, 2)], &[(5, 3)]);
    assert!(require_graph_counts(4, &twice, true).is_err());
    let never = stats(&[(5, 1)], &[(5, 1)], &[]);
    assert!(require_graph_counts(1, &never, true).is_err());
}
