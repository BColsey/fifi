use super::*;

/// The served start registers `wide1`, and a harness must run the same kernel. Mutating the
/// constant, the legacy name or the precedence fails here, on the host.
#[test]
fn the_served_single_row_kernel_is_wide1() {
    assert_eq!(SERVED_SINGLE_ROW, SingleRow::Wide(1));
    assert_eq!(single_row_name(SERVED_SINGLE_ROW), "wide1");
}

#[test]
fn an_unset_variable_takes_the_registered_default_and_an_explicit_value_wins() {
    assert_eq!(effective_single_row(None, None), SingleRow::Legacy);
    assert_eq!(
        effective_single_row(None, Some(SERVED_SINGLE_ROW)),
        SingleRow::Wide(1)
    );
    assert_eq!(
        effective_single_row(Some(SingleRow::Legacy), Some(SERVED_SINGLE_ROW)),
        SingleRow::Legacy,
        "an explicit `off` beats the default"
    );
    assert_eq!(
        effective_single_row(Some(SingleRow::Gather), Some(SERVED_SINGLE_ROW)),
        SingleRow::Gather
    );
    assert_eq!(
        effective_single_row(Some(SingleRow::Wide(4)), None),
        SingleRow::Wide(4)
    );
}

#[test]
fn the_record_names_never_leave_the_kernel_unsaid() {
    assert_eq!(single_row_name(SingleRow::Legacy), "legacy");
    assert_eq!(single_row_name(SingleRow::Gather), "gather");
    assert_eq!(single_row_name(SingleRow::Wide(1)), "wide1");
    assert_eq!(single_row_name(SingleRow::Wide(2)), "wide2");
    assert_eq!(single_row_name(SingleRow::Wide(4)), "wide4");
    assert_eq!(single_row_name(SingleRow::Wide(3)), "wide");
}

#[test]
fn the_load_line_names_only_an_armed_single_row_kernel() {
    assert_eq!(single_row_label(SingleRow::Legacy), None);
    assert_eq!(single_row_label(SingleRow::Gather), Some("gather"));
    for (cols, name) in [(1, "wide1"), (2, "wide2"), (4, "wide4")] {
        assert_eq!(single_row_label(SingleRow::Wide(cols)), Some(name));
    }
}

#[test]
fn a_harness_on_the_shuffle_kernel_is_refused_unless_the_variable_names_it() {
    assert!(require_served_kernel("wide1", false).is_ok());
    let error = require_served_kernel("legacy", false).expect_err("the unregistered default");
    assert!(error.contains("SERVED_SINGLE_ROW"), "{error}");
    assert!(require_served_kernel("gather", false).is_err());
    // An explicit variable is the operator's choice, whatever it names.
    assert!(require_served_kernel("legacy", true).is_ok());
    assert!(require_served_kernel("gather", true).is_ok());
}
