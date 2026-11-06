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

fn counts(mutate: impl FnOnce(&mut SingleRowLaunches)) -> SingleRowLaunches {
    let mut counts = SingleRowLaunches::default();
    mutate(&mut counts);
    counts
}

#[test]
fn a_launch_is_counted_under_its_own_variant_and_other_shapes_apart() {
    let c = counts(|c| {
        c.record(SingleRow::Wide(1));
        c.record(SingleRow::Wide(1));
        c.record(SingleRow::Wide(2));
        c.record(SingleRow::Wide(4));
        c.record(SingleRow::Gather);
        c.record(SingleRow::Legacy);
        c.record_other_shape();
        c.record_indexed();
    });
    assert_eq!(
        (
            c.wide1,
            c.wide2,
            c.wide4,
            c.gather,
            c.legacy,
            c.other_shapes,
            c.indexed
        ),
        (2, 1, 1, 1, 1, 1, 1)
    );
    assert_eq!(c.of("wide1"), Some(2));
    assert_eq!(c.of("nope"), None);
    assert!(c.describe().contains("wide1=2"), "{}", c.describe());
}

/// The negative control of the execution path: a run that was selected on `wide1` but whose
/// routed calls fell back to the shuffle kernel is rejected, and so is one that never launched the
/// kernel at all. A correct output does not excuse either.
#[test]
fn the_execution_path_guard_rejects_a_fallback_and_a_kernel_that_never_ran() {
    let served = counts(|c| (0..10).for_each(|_| c.record(SingleRow::Wide(1))));
    assert!(require_kernel_ran(&served, "wide1").is_ok());
    let with_other_shapes = counts(|c| {
        c.record(SingleRow::Wide(1));
        c.record_other_shape();
    });
    assert!(
        require_kernel_ran(&with_other_shapes, "wide1").is_ok(),
        "calls outside the routed shape are not a fallback"
    );
    let fallback = counts(|c| {
        c.record(SingleRow::Wide(1));
        c.record(SingleRow::Legacy);
    });
    let error = require_kernel_ran(&fallback, "wide1").expect_err("a fallback");
    assert!(error.contains("fallback"), "{error}");
    // Indexed expert launches run the wide kernel whatever the switch says: they never count as the
    // selected kernel having run.
    let only_indexed = counts(|c| c.record_indexed());
    assert!(require_kernel_ran(&only_indexed, "wide1").is_err());
    let never = counts(|c| c.record(SingleRow::Gather));
    assert!(require_kernel_ran(&never, "wide1").is_err());
    assert!(require_kernel_ran(&SingleRowLaunches::default(), "wide1").is_err());
    // The shuffle kernel is accepted only when it is what was expected.
    let legacy = counts(|c| c.record(SingleRow::Legacy));
    assert!(require_kernel_ran(&legacy, "wide1").is_err());
    assert!(require_kernel_ran(&legacy, "legacy").is_ok());
    assert!(require_kernel_ran(&served, "nonsense").is_err());
}
