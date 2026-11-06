//! Which kernel the single-row EXL3 projections run on, as pure code: the choice, its precedence and
//! its name in records. The device side (`kernels/cuda/resident/exl3.rs`) re-exports it and does the
//! launches; this file is compiled and tested on the host, so a change to the served default, to the
//! precedence of an explicit variable or to a record name fails a host test, not only a native one.
#![cfg_attr(not(test), allow(dead_code))]

/// The kernel a single-row call runs on. The switch governs every grouped launch the Qwen3.6 EXL3
/// step makes as well: see `grouped_kernel_for` in `exl3.rs`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SingleRow {
    Legacy,
    Gather,
    Wide(u32),
}

/// The kernel the Qwen3.6 EXL3 start registers on its context (`family/qwen36/cuda/start.rs`), and
/// that any harness which claims to measure or qualify the served step must register too. An
/// explicit `FIFI_EXL3_SINGLE_GATHER` still wins over it.
pub(crate) const SERVED_SINGLE_ROW: SingleRow = SingleRow::Wide(1);

/// The name of a kernel in a record: unlike [`single_row_label`] the shuffle kernel has one, so a
/// record never leaves the kernel unsaid.
pub(crate) fn single_row_name(kernel: SingleRow) -> &'static str {
    single_row_label(kernel).unwrap_or("legacy")
}

/// The check a harness runs on the kernel its context reports: the served `wide1`, unless the
/// operator named a kernel in `FIFI_EXL3_SINGLE_GATHER` (`explicit_variable` is true then). A
/// harness that forgot to register the served default reports `legacy` and is refused here, so a
/// result never speaks for a kernel the served engine does not run.
pub(crate) fn require_served_kernel(
    effective_name: &str,
    explicit_variable: bool,
) -> Result<(), String> {
    let served = single_row_name(SERVED_SINGLE_ROW);
    if explicit_variable || effective_name == served {
        return Ok(());
    }
    Err(format!(
        "the single-row kernel is {effective_name:?}, the served kernel is {served:?}: register \
         SERVED_SINGLE_ROW on the context before the load, or name a kernel in FIFI_EXL3_SINGLE_GATHER"
    ))
}

/// The kernel an unset variable leaves to the context: an explicit value of the variable always
/// wins, then the default the session that owns the context registered on it
/// (`CudaContext::set_single_row_default`), then the shuffle kernel.
pub(crate) fn effective_single_row(
    explicit: Option<SingleRow>,
    context_default: Option<SingleRow>,
) -> SingleRow {
    explicit.or(context_default).unwrap_or(SingleRow::Legacy)
}

/// The load-line name of the kernel a switch value selects, `None` for the shuffle kernel.
pub(crate) fn single_row_label(kernel: SingleRow) -> Option<&'static str> {
    match kernel {
        SingleRow::Legacy => None,
        SingleRow::Gather => Some("gather"),
        SingleRow::Wide(1) => Some("wide1"),
        SingleRow::Wide(2) => Some("wide2"),
        SingleRow::Wide(4) => Some("wide4"),
        SingleRow::Wide(_) => Some("wide"),
    }
}

#[cfg(test)]
mod tests;
