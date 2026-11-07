use super::*;

/// The thresholds of `rows_kernel_for` in `exl3.cu`: one row is the one-row body, 2 rows the
/// `<2>` instantiation, 3 and 4 the `<4>` one (one masked row at 3), 5 to 8 the `<8>` one.
/// Mutating a threshold fails here on the host; a device test compares the table with the native
/// answer.
#[test]
fn the_row_counts_select_the_kernel_of_exl3_cu() {
    let table = [
        (1, RowsKernel::SmallWide1),
        (2, RowsKernel::Rows2),
        (3, RowsKernel::Rows4),
        (4, RowsKernel::Rows4),
        (5, RowsKernel::Rows8),
        (6, RowsKernel::Rows8),
        (7, RowsKernel::Rows8),
        (8, RowsKernel::Rows8),
    ];
    for (rows, kernel) in table {
        assert_eq!(rows_kernel(rows), Some(kernel), "{rows} rows");
    }
    assert_eq!(rows_kernel(0), None);
    assert_eq!(rows_kernel(GROUPED_MAX_ROWS + 1), None);
}

#[test]
fn the_native_codes_decode_to_the_same_kernels() {
    for rows in 1..=GROUPED_MAX_ROWS {
        let code = match rows_kernel(rows).unwrap() {
            RowsKernel::SmallWide1 => 1,
            RowsKernel::Rows2 => 2,
            RowsKernel::Rows4 => 4,
            RowsKernel::Rows8 => 8,
        };
        assert_eq!(rows_kernel_from_native(code), rows_kernel(rows));
    }
    assert_eq!(rows_kernel_from_native(0), None);
    assert_eq!(rows_kernel_from_native(3), None);
}

fn full_run(window: usize, windows: u64) -> RowLaunches {
    let mut counts = RowLaunches::default();
    counts.record_projection(expected_rows_kernel(window).unwrap());
    for _ in 0..windows {
        for kernel in [
            RowKernel::ArgmaxRows,
            RowKernel::StatusFold,
            RowKernel::PackRows,
            RowKernel::ConvSnap,
            RowKernel::DeltaSnap,
        ] {
            counts.record(kernel);
        }
    }
    counts
}

#[test]
fn a_k_row_run_shows_its_rows_kernel_and_its_per_window_kernels() {
    for window in 2..=GROUPED_MAX_ROWS {
        assert_eq!(full_run(window, 3).require_krow_ran(window, 3), Ok(()));
    }
}

/// The negative controls: a run that fell through to K one-row launches has the rows kernels at
/// zero, and a run that lost any per-window kernel is refused by name.
#[test]
fn a_run_without_the_k_row_kernels_is_refused() {
    let mut fell_through = RowLaunches::default();
    fell_through.record_projection(RowsKernel::SmallWide1);
    assert!(fell_through.require_krow_ran(4, 1).is_err());

    for dropped in 0..5 {
        let mut counts = full_run(4, 2);
        match dropped {
            0 => counts.argmax_rows = 1,
            1 => counts.status_fold = 1,
            2 => counts.pack_rows = 0,
            3 => counts.conv_snap = 1,
            _ => counts.delta_snap = 0,
        }
        let refusal = counts.require_krow_ran(4, 2).unwrap_err();
        assert!(refusal.contains("did not run"), "{refusal}");
    }
    let mut no_rows_kernel = full_run(4, 2);
    no_rows_kernel.rows4 = 0;
    assert!(
        no_rows_kernel
            .require_krow_ran(4, 2)
            .unwrap_err()
            .contains("Rows4")
    );
}

#[test]
fn a_window_that_ran_another_rows_kernel_is_refused() {
    let mut counts = full_run(4, 1);
    counts.record_projection(RowsKernel::Rows8);
    let refusal = counts.require_krow_ran(4, 1).unwrap_err();
    assert!(refusal.contains("Rows8"), "{refusal}");
}

#[test]
fn only_two_to_eight_rows_are_a_k_row_window() {
    assert!(full_run(2, 1).require_krow_ran(1, 1).is_err());
    assert!(full_run(8, 1).require_krow_ran(9, 1).is_err());
}

#[test]
fn the_eager_control_shows_every_k_row_counter_at_zero() {
    let mut counts = RowLaunches::default();
    counts.record_projection(RowsKernel::SmallWide1);
    assert_eq!(counts.require_krow_idle(), Ok(()));
    for kernel in [
        RowKernel::ArgmaxRows,
        RowKernel::StatusFold,
        RowKernel::PackRows,
        RowKernel::ConvSnap,
        RowKernel::DeltaSnap,
        RowKernel::RouterStrided,
    ] {
        let mut used = RowLaunches::default();
        used.record(kernel);
        assert!(used.require_krow_idle().is_err(), "{kernel:?}");
    }
    let mut rows = RowLaunches::default();
    rows.record_projection(RowsKernel::Rows2);
    assert!(rows.require_krow_idle().is_err());
}

#[test]
fn the_line_names_every_counter() {
    let line = full_run(2, 1).describe();
    for name in [
        "small_wide1",
        "rows2",
        "rows4",
        "rows8",
        "argmax_rows",
        "status_fold",
        "pack_rows",
        "conv_snap",
        "delta_snap",
        "router_strided",
    ] {
        assert!(line.contains(name), "{name} in {line}");
    }
}
