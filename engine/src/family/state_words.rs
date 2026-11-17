//! The persistent words of the resident step, shared by the families.
//!
//! `Word` names the three small regions that live in their own block (not in a
//! step arena) because their address must be the same for every captured graph
//! of every row count and for the prefill path: `StepParams` (32 bytes, the
//! step parameters), `ResultPair` (8 bytes, `{token, status}`, copied by the
//! host after a step) and `PrefillStatus` (4 bytes, zeroed by the host at chunk
//! open). Every family state layout reads the same ABI from here; the offsets
//! and sizes stay the contract the kernels, the attention path and
//! `export_state` / `import_state` share.

use crate::kernels::cuda::resident::step_params::{RESULT_PAIR_BYTES, STEP_PARAMS_BYTES};

/// Offsets inside a state pool are multiples of this many bytes (the `cudaMalloc` alignment, as
/// in the arena planner); one slot per word, so a later word never lands inside an earlier
/// word's alignment padding.
pub(in crate::family) const STATE_ALIGN: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Word {
    StepParams,
    /// The result pair `{token, status}`; the host copies these 8 bytes after a step.
    ResultPair,
    /// The prefill status word; the host zeroes it at chunk open
    /// (`ResidentState::clear_prefill_status`), not the first kernel of the chunk.
    PrefillStatus,
}

// The merged ABI: the pair must be 8-aligned in its slot and every word must fit one slot.
const _: () = assert!(
    Word::StepParams.bytes() == 32 && Word::ResultPair.bytes() == 8,
    "the words block is laid out for StepParams of 32 bytes and a result pair of 8 bytes"
);
const _: () = assert!(
    Word::StepParams.bytes() <= STATE_ALIGN
        && Word::ResultPair.bytes() <= STATE_ALIGN
        && Word::PrefillStatus.bytes() <= STATE_ALIGN
        && Word::ResultPair.offset().is_multiple_of(8)
);

impl Word {
    /// Bytes the kernels use: `STEP_PARAMS_BYTES` (32) and `RESULT_PAIR_BYTES` (8), taken from the
    /// kernel wrapper, and one i32 for the prefill status.
    pub(crate) const fn bytes(self) -> usize {
        match self {
            Self::StepParams => STEP_PARAMS_BYTES,
            Self::ResultPair => RESULT_PAIR_BYTES,
            Self::PrefillStatus => 4,
        }
    }

    /// Byte offset in the words block: one `STATE_ALIGN` slot per word, in declaration order.
    pub(crate) const fn offset(self) -> usize {
        match self {
            Self::StepParams => 0,
            Self::ResultPair => STATE_ALIGN,
            Self::PrefillStatus => 2 * STATE_ALIGN,
        }
    }
}
