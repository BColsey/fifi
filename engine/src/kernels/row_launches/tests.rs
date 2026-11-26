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

#[test]
fn the_line_names_every_counter() {
    let line = RowLaunches::default().describe();
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
        "verify_greedy",
        "shortlist_rows",
    ] {
        assert!(line.contains(name), "{name} in {line}");
    }
}
