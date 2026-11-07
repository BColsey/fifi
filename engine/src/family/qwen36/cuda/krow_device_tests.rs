//! What the K-row device tests (in `d1_window_device_tests.rs`) share: the refusal of the kernel
//! switches the K-row path is not qualified for, and the check on the counters of a run. Both are
//! host logic and run without a device; the device tests that use them start with
//! [`refuse_switches`], so a run with a refused switch set stops before it loads the pack.

use std::ffi::OsString;

use super::d1_window_device_tests::require_krow_enqueued;
use crate::{
    family::qwen36::{exl3_merge_plan::SHARED_IN_ROUTED_VAR, krow::SINGLE_GATHER_NAME},
    kernels::row_launches::RowLaunches,
};

/// The refusal of a switch that selects another kernel or another MoE body than the one the K-row
/// path is qualified on, given a way to read the environment. The switches that leave the K-row
/// bytes and counters unchanged (`FIFI_EXL3_DP4A`, `FIFI_EXL3_INDEXED_PAIRS`, `FIFI_EXL3_LEGACY`,
/// `FIFI_D1_CAPTURE`) are run on purpose and are not refused.
pub(super) fn switch_refusal(get: impl Fn(&str) -> Option<OsString>) -> Result<(), String> {
    if let Some(value) = get(SINGLE_GATHER_NAME) {
        return Err(format!(
            "{SINGLE_GATHER_NAME} is set to {value:?}: unset it, the K-row path is qualified on the served wide1 kernel only"
        ));
    }
    if get(SHARED_IN_ROUTED_VAR).is_some_and(|value| value.to_str().map(str::trim) == Some("1")) {
        return Err(format!(
            "{SHARED_IN_ROUTED_VAR}=1 is set: the K-row path takes the reference MoE block, unset it"
        ));
    }
    Ok(())
}

/// [`switch_refusal`] over the process environment.
pub(super) fn refuse_switches() -> Result<(), String> {
    switch_refusal(std::env::var_os)
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

fn counts(
    argmax: u64,
    fold: u64,
    pack: u64,
    conv: u64,
    delta: u64,
    router: u64,
) -> RowLaunches {
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
