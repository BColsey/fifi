//! The EXL3 MoE block shared by the families.
//!
//! Promoted from `qwen36::cuda`: the routed expert block, its
//! merged-pass host path, and the weight source that lays experts out as a
//! bank. Consumers import these modules directly; `qwen36::cuda` keeps only
//! what its own code uses.

pub(in crate::family) mod bank;
pub(in crate::family) mod launch;
pub(in crate::family) mod layer_table;
pub(in crate::family) mod moe;
pub(in crate::family) mod prefill;
pub(in crate::family) mod prefill_moe;
pub(in crate::family) mod shapes;
pub(in crate::family) mod source;
