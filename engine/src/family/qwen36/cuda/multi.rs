//! The multi-device decode driver's pure groundwork: the ring operations
//! one token performs, in order, and how ring status words fold into the
//! step's poison rule. The executor that runs this plan against per-device
//! step shards is the remaining Slice 3 work; this module is its contract,
//! stated where it can be tested without a device.
//!
//! Where each operation sits in the existing step, by the slots it moves:
//! - `EmbedBroadcast`: card 0 embeds the token; the embedded row (the
//!   `Embed` op's output in the residual slot) is broadcast before layer 0.
//! - `KvGather` (full-attention layers only): inside the mixer, after the
//!   K/V projections and before attention reads the cache — each card's new
//!   K/V slice is replicated into every card's cache row for this position.
//!   Only the new row moves; history never does.
//! - `MixerSum`: at the mixer/MoE boundary, the exchange sums
//!   `Slot::MixOut` — the out-projection partial the next norm consumes as
//!   its fused `delta` — so every card's norm adds the full mixer output.
//! - `RouteBroadcast`: after the MoE router on card 0, its picks ride one
//!   broadcast; every card then computes the same experts.
//! - `MoeSum`: inside the MoE block before `Combine`, over the expert and
//!   shared down-projection partials, so `Combine` folds full values into
//!   the replicated residual on every card.
//!
//! Every operation consumes exactly one ring sequence (the parity chain
//! of `kernels::cuda::peer_ring` covers mixed operation kinds), and all of
//! them queue on the one per-device step stream, which is what the slot
//! reuse argument requires.

#![cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "staged for the two-device executor, which runs this plan per token"
    )
)]

use crate::family::AttentionKind;
use crate::kernels::cuda::peer_ring::{STATUS_OVERFLOW, STATUS_TIMEOUT};

/// One ring operation of a token, in execution order. `layer` is the
/// absolute layer index the operation belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TokenOp {
    EmbedBroadcast,
    KvGather { layer: usize },
    MixerSum { layer: usize },
    RouteBroadcast { layer: usize },
    MoeSum { layer: usize },
}

/// The full ring sequence of one decoded token over `layers` (the step
/// table's attention kinds, in layer order): one embedding broadcast,
/// then per layer the mixer sum and the MoE sum with a route broadcast
/// between them, and a K/V gather inside each full-attention layer,
/// queued before that layer's mixer sum.
pub(crate) fn exchange_plan(layers: &[AttentionKind]) -> Vec<TokenOp> {
    let mut ops = Vec::with_capacity(1 + layers.len() * 3 + 10);
    ops.push(TokenOp::EmbedBroadcast);
    for (layer, &kind) in layers.iter().enumerate() {
        if kind == AttentionKind::Full {
            ops.push(TokenOp::KvGather { layer });
        }
        ops.push(TokenOp::MixerSum { layer });
        ops.push(TokenOp::RouteBroadcast { layer });
        ops.push(TokenOp::MoeSum { layer });
    }
    ops
}

/// The step-status bit a ring timeout folds to. Ring bits 1 and 2 already
/// mean other faults in the step status word, so ring outcomes fold to
/// fresh bits; the step's rule that any nonzero status poisons the state
/// (`state.rs`) makes every fold result a poisoned token, which is the
/// enforcement: a timed-out or overflowed exchange can never leave a
/// usable row behind it.
pub(crate) const STEP_RING_TIMEOUT: u32 = 16;
/// The step-status bit a ring overflow folds to.
pub(crate) const STEP_RING_OVERFLOW: u32 = 32;
/// The step-status bit any other (future) ring status bit folds to.
pub(crate) const STEP_RING_UNKNOWN: u32 = 64;

/// Folds every device's ring status word of one token into one step
/// status word. Zero in, zero out; any timeout, overflow, or unknown ring
/// bit on any device sets its step bit, so the existing readback poisons
/// the state before the token's row can be used.
pub(crate) fn fold_ring_status(words: &[u32]) -> u32 {
    let mut folded = 0u32;
    for &word in words {
        if word & STATUS_TIMEOUT != 0 {
            folded |= STEP_RING_TIMEOUT;
        }
        if word & STATUS_OVERFLOW != 0 {
            folded |= STEP_RING_OVERFLOW;
        }
        if word & !(STATUS_TIMEOUT | STATUS_OVERFLOW) != 0 {
            folded |= STEP_RING_UNKNOWN;
        }
    }
    folded
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::family::spec::QWEN3_6_35B_A3B;

    #[test]
    fn the_exchange_plan_counts_one_sequence_per_operation() {
        let layers: Vec<AttentionKind> = QWEN3_6_35B_A3B.layers_attention.to_vec();
        let ops = exchange_plan(&layers);
        // 1 embed + 40 layers * (mixer + route + moe) + 10 full-layer gathers.
        assert_eq!(ops.len(), 131);
        assert_eq!(ops[0], TokenOp::EmbedBroadcast);
        // Layer order is absolute and each layer's operations stay in order:
        // a full layer's gather precedes its mixer sum.
        let mut last_layer = 0usize;
        for op in &ops[1..] {
            let layer = match *op {
                TokenOp::KvGather { layer }
                | TokenOp::MixerSum { layer }
                | TokenOp::RouteBroadcast { layer }
                | TokenOp::MoeSum { layer } => layer,
                TokenOp::EmbedBroadcast => unreachable!("one embed broadcast"),
            };
            assert!(layer >= last_layer, "ops leave layer order at {op:?}");
            last_layer = layer;
        }
        let gathers = ops
            .iter()
            .filter(|op| matches!(op, TokenOp::KvGather { .. }))
            .count();
        assert_eq!(gathers, 10);
        // First full layer of the 3+1 pattern is absolute layer 3.
        assert_eq!(
            ops.iter().find(|op| matches!(op, TokenOp::KvGather { .. })),
            Some(&TokenOp::KvGather { layer: 3 })
        );
    }

    #[test]
    fn ring_status_folds_to_poison_bits_and_clean_words_stay_clean() {
        assert_eq!(fold_ring_status(&[]), 0);
        assert_eq!(fold_ring_status(&[0, 0]), 0);
        assert_eq!(fold_ring_status(&[STATUS_TIMEOUT, 0]), STEP_RING_TIMEOUT);
        assert_eq!(fold_ring_status(&[0, STATUS_OVERFLOW]), STEP_RING_OVERFLOW);
        assert_eq!(
            fold_ring_status(&[STATUS_TIMEOUT | STATUS_OVERFLOW]),
            STEP_RING_TIMEOUT | STEP_RING_OVERFLOW
        );
        assert_eq!(fold_ring_status(&[0x8000]), STEP_RING_UNKNOWN);
        // Every nonzero fold is a poison under the step's any-nonzero rule,
        // and ring bits never alias the step's own 1/2/4/8 meanings.
        for word in [1u32, 2, 3, 0xFFFF] {
            let folded = fold_ring_status(&[word]);
            assert_ne!(folded, 0);
            assert_eq!(folded & 0b1111, 0, "ring bits must not alias step bits");
        }
    }
}
