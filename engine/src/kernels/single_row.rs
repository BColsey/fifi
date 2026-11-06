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

/// How many launches of each single-row kernel one context has enqueued (or captured into a graph),
/// per variant, counted where the dispatch decides. A replay of a captured graph does not count
/// again: its launches were counted when they were captured, and a graph that was captured on the
/// wrong kernel keeps that kernel in its counts for good. Only calls of the routed shape (rows 1 to
/// 8 at split-K 2) are counted per variant; the others always run the shuffle kernel and are
/// counted apart, so they never read as a fallback. Test builds only: the serving path counts
/// nothing.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct SingleRowLaunches {
    pub(crate) legacy: u64,
    pub(crate) gather: u64,
    pub(crate) wide1: u64,
    pub(crate) wide2: u64,
    pub(crate) wide4: u64,
    /// Calls outside the routed shape: not a fallback.
    pub(crate) other_shapes: u64,
    /// Id-indexed expert launches of a captured step: they run the wide kernel whatever the switch
    /// says, so they say nothing about which kernel was selected and are counted apart.
    pub(crate) indexed: u64,
}

impl SingleRowLaunches {
    /// Counts one launch of the routed shape on `kernel`.
    pub(crate) fn record(&mut self, kernel: SingleRow) {
        let slot = match kernel {
            SingleRow::Legacy => &mut self.legacy,
            SingleRow::Gather => &mut self.gather,
            SingleRow::Wide(1) => &mut self.wide1,
            SingleRow::Wide(2) => &mut self.wide2,
            SingleRow::Wide(_) => &mut self.wide4,
        };
        *slot += 1;
    }

    /// Counts one call outside the routed shape.
    pub(crate) fn record_other_shape(&mut self) {
        self.other_shapes += 1;
    }

    /// Counts one id-indexed expert launch.
    pub(crate) fn record_indexed(&mut self) {
        self.indexed += 1;
    }

    /// The launches of the variant `name` (`legacy`, `gather`, `wide1`, `wide2`, `wide4`).
    pub(crate) fn of(&self, name: &str) -> Option<u64> {
        Some(match name {
            "legacy" => self.legacy,
            "gather" => self.gather,
            "wide1" => self.wide1,
            "wide2" => self.wide2,
            "wide4" => self.wide4,
            _ => return None,
        })
    }

    /// One line for D1_CONFIG and the record.
    pub(crate) fn describe(&self) -> String {
        format!(
            "legacy={} gather={} wide1={} wide2={} wide4={} other_shapes={} indexed={}",
            self.legacy,
            self.gather,
            self.wide1,
            self.wide2,
            self.wide4,
            self.other_shapes,
            self.indexed
        )
    }
}

/// The execution-path check, separate from the selection check: a correct output can come from a
/// fallback, so a receipt must also show that the intended kernel ran. `expected` is the kernel
/// name the context reports (`wide1` when served). It must have launched at least once, and no
/// routed call may have run any other variant (the shuffle kernel, the gather kernel or another
/// wide kernel: a silent fallback), unless the shuffle kernel is what was expected.
pub(crate) fn require_kernel_ran(counts: &SingleRowLaunches, expected: &str) -> Result<(), String> {
    let ran = counts
        .of(expected)
        .ok_or_else(|| format!("unknown kernel name {expected:?}"))?;
    if ran == 0 {
        return Err(format!(
            "the {expected} kernel never launched ({}): the output does not come from it",
            counts.describe()
        ));
    }
    // Any other variant on the routed shape is a fallback: the shuffle kernel, the gather kernel
    // (what a wide setting degrades to when a width is not a multiple of its tile span) or another
    // wide variant. The shuffle kernel as the selected one is accepted with the others unchecked:
    // a legacy setting routes its grouped launches to the gather kernel by design.
    if expected != "legacy" {
        for other in ["legacy", "gather", "wide1", "wide2", "wide4"] {
            let count = counts.of(other).unwrap_or(0);
            if other != expected && count > 0 {
                return Err(format!(
                    "{count} routed calls ran the {other} kernel although {expected} was selected ({}): \
                     a fallback path was exercised",
                    counts.describe()
                ));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
