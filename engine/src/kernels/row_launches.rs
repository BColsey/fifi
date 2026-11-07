//! Which kernel a K-row verify window really ran, as pure code: the launch counters of the test
//! builds, per kernel identity, and the check that a run used the kernels it claims.
//!
//! A bitwise-equal output cannot tell the K-row path from K single-row launches, so a negative
//! control needs a count of what ran. [`crate::kernels::single_row::SingleRowLaunches`] counts the
//! `wide1` calls of the single-row path and cannot tell `grouped_project_small_wide<1>` from
//! `grouped_project_rows<R>`: it counts both as `wide1`. This file adds the identities the K-row
//! path adds. Counted where the dispatch decides, at enqueue or capture; a replay of a captured
//! graph does not count again. Test builds only: the serving path counts nothing.
//!
//! The thresholds of [`rows_kernel`] are the ones of `rows_kernel_for` in
//! `native/resident/exl3.cu`, which makes the launch and answers the native
//! `inference_cuda_exl3_rows_kernel` that the device counters call; a host test pins the table and
//! a device test compares it with the native answer for every row count.
#![cfg_attr(not(test), allow(dead_code))]

/// The most rows of one group of a rows launch (`grouped_max_rows` in `exl3.cu`).
pub(crate) const GROUPED_MAX_ROWS: usize = 8;

/// The projection kernel of a wide EXL3 launch whose largest group has a given row count.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RowsKernel {
    /// `grouped_project_small_wide<1>`: every group has one row (the single-row step's launch).
    SmallWide1,
    /// `grouped_project_rows<2>`.
    Rows2,
    /// `grouped_project_rows<4>`.
    Rows4,
    /// `grouped_project_rows<8>`.
    Rows8,
}

/// The kernel for a largest group of `max_rows` rows, `None` outside `1..=GROUPED_MAX_ROWS`.
pub(crate) fn rows_kernel(max_rows: usize) -> Option<RowsKernel> {
    match max_rows {
        1 => Some(RowsKernel::SmallWide1),
        2 => Some(RowsKernel::Rows2),
        3..=4 => Some(RowsKernel::Rows4),
        5..=GROUPED_MAX_ROWS => Some(RowsKernel::Rows8),
        _ => None,
    }
}

/// The kernel a `window`-row dense projection of the K-row path runs: its group has `window` rows.
pub(crate) fn expected_rows_kernel(window: usize) -> Option<RowsKernel> {
    rows_kernel(window)
}

/// The decoded value of the native answer (`inference_cuda_exl3_rows_kernel`: 0, 1, 2, 4 or 8).
pub(crate) fn rows_kernel_from_native(code: i32) -> Option<RowsKernel> {
    match code {
        1 => Some(RowsKernel::SmallWide1),
        2 => Some(RowsKernel::Rows2),
        4 => Some(RowsKernel::Rows4),
        8 => Some(RowsKernel::Rows8),
        _ => None,
    }
}

/// The other launches the K-row path adds, by identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RowKernel {
    ArgmaxRows,
    StatusFold,
    PackRows,
    /// `conv_silu_k<true>`: the conv with strided snapshots.
    ConvSnap,
    /// `gated_delta_k<true>`: the delta with strided snapshots.
    DeltaSnap,
    /// `router_logits_k` with a logit stride other than `n_expert` (beta and alpha written into the
    /// `[K][ld]` row).
    RouterStrided,
}

/// How many launches of each kernel identity one context has enqueued since it was made.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct RowLaunches {
    pub(crate) small_wide1: u64,
    pub(crate) rows2: u64,
    pub(crate) rows4: u64,
    pub(crate) rows8: u64,
    pub(crate) argmax_rows: u64,
    pub(crate) status_fold: u64,
    pub(crate) pack_rows: u64,
    pub(crate) conv_snap: u64,
    pub(crate) delta_snap: u64,
    pub(crate) router_strided: u64,
}

impl RowLaunches {
    /// Counts one wide projection launch of the rows kernel `kernel`.
    pub(crate) fn record_projection(&mut self, kernel: RowsKernel) {
        let slot = match kernel {
            RowsKernel::SmallWide1 => &mut self.small_wide1,
            RowsKernel::Rows2 => &mut self.rows2,
            RowsKernel::Rows4 => &mut self.rows4,
            RowsKernel::Rows8 => &mut self.rows8,
        };
        *slot += 1;
    }

    /// Counts one launch of `kernel`.
    pub(crate) fn record(&mut self, kernel: RowKernel) {
        let slot = match kernel {
            RowKernel::ArgmaxRows => &mut self.argmax_rows,
            RowKernel::StatusFold => &mut self.status_fold,
            RowKernel::PackRows => &mut self.pack_rows,
            RowKernel::ConvSnap => &mut self.conv_snap,
            RowKernel::DeltaSnap => &mut self.delta_snap,
            RowKernel::RouterStrided => &mut self.router_strided,
        };
        *slot += 1;
    }

    /// The launches of the projection kernel `kernel`.
    pub(crate) fn projection(&self, kernel: RowsKernel) -> u64 {
        match kernel {
            RowsKernel::SmallWide1 => self.small_wide1,
            RowsKernel::Rows2 => self.rows2,
            RowsKernel::Rows4 => self.rows4,
            RowsKernel::Rows8 => self.rows8,
        }
    }

    /// Launches of `grouped_project_rows<R>` of any R: the rows kernels, not the one-row body.
    pub(crate) fn rows_kernels(&self) -> u64 {
        self.rows2 + self.rows4 + self.rows8
    }

    /// One line for a record.
    pub(crate) fn describe(&self) -> String {
        format!(
            "small_wide1={} rows2={} rows4={} rows8={} argmax_rows={} status_fold={} pack_rows={} \
             conv_snap={} delta_snap={} router_strided={}",
            self.small_wide1,
            self.rows2,
            self.rows4,
            self.rows8,
            self.argmax_rows,
            self.status_fold,
            self.pack_rows,
            self.conv_snap,
            self.delta_snap,
            self.router_strided
        )
    }

    /// The check a K-row run makes on the counters it took over `windows` windows of `window`
    /// rows: the rows kernel the window size selects launched, the argmax, the status fold, the
    /// pack and the snapshot kernels ran once per window at least, and no window ran a rows
    /// kernel of another size. A run that cost the same K single-row launches has these at zero.
    pub(crate) fn require_krow_ran(&self, window: usize, windows: u64) -> Result<(), String> {
        let expected = expected_rows_kernel(window)
            .filter(|kernel| *kernel != RowsKernel::SmallWide1)
            .ok_or_else(|| {
                format!("a K-row window has 2 to {GROUPED_MAX_ROWS} rows, got {window}")
            })?;
        let mut missing = Vec::new();
        if self.projection(expected) == 0 {
            missing.push(format!("{expected:?}"));
        }
        for (name, count) in [
            ("argmax_rows", self.argmax_rows),
            ("status_fold", self.status_fold),
            ("pack_rows", self.pack_rows),
            ("conv_snap", self.conv_snap),
            ("delta_snap", self.delta_snap),
        ] {
            if count < windows {
                missing.push(format!("{name} (ran {count} times in {windows} windows)"));
            }
        }
        if !missing.is_empty() {
            return Err(format!(
                "the K-row path did not run: {} ({})",
                missing.join(", "),
                self.describe()
            ));
        }
        for other in [RowsKernel::Rows2, RowsKernel::Rows4, RowsKernel::Rows8] {
            if other != expected && self.projection(other) > 0 && window > 1 {
                return Err(format!(
                    "a {window}-row window ran {other:?} although {expected:?} was expected ({})",
                    self.describe()
                ));
            }
        }
        Ok(())
    }

    /// The check of the eager control: a run that must not have used the K-row path shows every
    /// counter of it at zero (the one-row `small_wide1` launches are the eager path's own).
    pub(crate) fn require_krow_idle(&self) -> Result<(), String> {
        let used = self.rows_kernels()
            + self.argmax_rows
            + self.status_fold
            + self.pack_rows
            + self.conv_snap
            + self.delta_snap
            + self.router_strided;
        if used == 0 {
            Ok(())
        } else {
            Err(format!(
                "the K-row path ran in an eager run ({})",
                self.describe()
            ))
        }
    }
}

#[cfg(test)]
mod tests;
