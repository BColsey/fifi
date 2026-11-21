//! The one entry the frozen EXL3 expert store reads the host through. That file is not edited
//! (the format takes no new work), so it keeps its call; the read itself is the shared reader of
//! [`crate::family::mem`], with the same lenient rule as every other caller. No other caller uses
//! this module.

use crate::EngError;

pub(crate) struct HostAdmission;

impl HostAdmission {
    /// `(MemTotal, budget)` of this host: the cgroup-clamped budget, or the bare figure when the
    /// tree cannot be verified.
    pub(crate) fn probe_or_bare(
        label: &'static str,
        root_label: &'static str,
    ) -> Result<(u64, u64), EngError> {
        crate::family::mem::HostFacts::read(label, root_label).map(|host| host.host_memory())
    }
}
