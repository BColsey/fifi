//! The host side of a resident decode step, with no model in it: the failure rule of a step, how a
//! token is read back, the report of a run, and the types a step shares with its callers (the
//! cache form, the chunk-gate verdict). A child supplies its kernels and its state; what the host
//! does around a graph launch is here, once.

pub(crate) use crate::family::step_dims::KvMode;

/// The typed verdict of the chunk gate: whether `feed_rows` may run the chunk path for this
/// session. Every refusal keeps the per-token prompt path; the run report records the reason.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ChunkVerdict {
    /// The session may use the chunk path. `m_sess` is the ADMITTED `M_sess`, one of
    /// `M_LADDER` (`PrefillArena::m_sess`), never the raw cap: the arena admission narrows the
    /// cap of `set_m_sess_cap` to the largest ladder size whose device sum fits the budget.
    Ready {
        /// The first reader is the chunk list of the prefill driver.
        m_sess: usize,
    },
    /// The session runs the Tiled layout: the chunk path is defined for Native sessions only.
    TiledLayout,
    /// The `Int8G32` session has no explicit chunk-prefill opt-in and keeps per-token prefill.
    Kv8,
    /// The session is tiered: `chunk_ready` sends tiered sessions to the per-token loop; no
    /// batched tiered prefill exists yet.
    Tiered,
    /// An MMQ weight of the chunk is Q6_K: no Q6_K chunk row exists, so the session keeps the
    /// per-token path.
    Q6KWeight,
    /// The session admitted no prefill arena (no `M` of the ladder fits the budget, or
    /// `admit_prefill` was not called); the session keeps the per-token path.
    NoArena,
}

pub(crate) mod readback;
#[cfg(test)]
mod readback_tests;
pub(crate) mod run_report;
#[cfg(test)]
pub(crate) mod run_report_tests;
pub(crate) mod verdict;
#[cfg(test)]
mod verdict_tests;
