//! Host geometry of the CUDA path: the [`HostShape`] the shared device planning
//! (state layout, kernel scratch, step plan) consumes, built from the model's own
//! [`LoopShape`] and [`LoopPlan`] instead of a `FamilySpec` — the trunk's 56 visits
//! over 32 physical weight sets have no spec representation.
//!
//! Pure host arithmetic, portable across builds.

use super::super::weights::LoopShape;
use super::super::{LoopPlan, PHYSICAL_LAYERS, RopePlan};
use crate::EngError;
use crate::family::AttentionKind;
use crate::family::host_shape::{AttnDims, DeltaDims, HostShape, MoeDims};

/// The attention kind of every PHYSICAL layer, in physical order. A revisit carries
/// the kind of the weight set it shares; the plan constructor already rejects a visit
/// whose kind disagrees with its physical layer.
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "only this module's tests read it today")
)]
pub(crate) fn physical_kinds(plan: &LoopPlan) -> Vec<AttentionKind> {
    (0..PHYSICAL_LAYERS)
        .map(|p| {
            if plan.recurrent[plan.block(p)] {
                AttentionKind::DeltaNet
            } else {
                AttentionKind::Full
            }
        })
        .collect()
}

/// The shared host shape of one resident step: dense FFN (`dense_inter` set, the MoE
/// dims zeroed), DeltaNet and attention dims from the model, rotary width and base
/// from the rope plan. `check()` inside validates the kernel contract (head ratio,
/// conv kernel, nonzero dims).
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "only this module's tests read it today")
)]
pub(crate) fn host_shape(
    shape: &LoopShape,
    kinds: Vec<AttentionKind>,
    rope: &RopePlan,
) -> Result<HostShape, EngError> {
    let host = HostShape {
        hidden: shape.hidden,
        vocab: shape.vocab,
        kinds,
        eps: shape.eps,
        delta: DeltaDims {
            key_heads: shape.delta_key_heads,
            key_dim: shape.delta_key_dim,
            value_heads: shape.delta_value_heads,
            value_dim: shape.delta_value_dim,
            conv_kernel: shape.conv_kernel,
        },
        attn: AttnDims {
            heads_q: shape.heads_q,
            heads_kv: shape.heads_kv,
            head_dim: shape.head_dim,
            rotary_dim: rope.dims,
            theta: rope.freq_base,
        },
        dense_inter: Some(shape.feed_forward),
        moe: MoeDims {
            experts: 0,
            top_k: 0,
            inter: 0,
            shared_inter: 0,
            renorm: false,
        },
    };
    host.check()?;
    Ok(host)
}

/// The YaRN descriptor the rope generalization needs beyond `theta` + `rotary_dim`:
/// the shared kernels bake plain NTK-style rotary; SenseNova scales by `factor` from
/// `original_context` (see CUDA-PLAN §math, divergence 3).
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "only this module's tests read it today")
)]
pub(crate) struct YarnPlan {
    pub(crate) factor: f32,
    pub(crate) original_context: usize,
}

impl YarnPlan {
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "only this module's tests read it today")
    )]
    pub(crate) fn of(rope: &RopePlan) -> Self {
        Self {
            factor: rope.factor,
            original_context: rope.original_context,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pinned_shape() -> LoopShape {
        LoopShape {
            hidden: 4096,
            feed_forward: 12_288,
            heads_q: 16,
            heads_kv: 4,
            head_dim: 256,
            delta_key_heads: 16,
            delta_key_dim: 128,
            delta_value_heads: 32,
            delta_value_dim: 128,
            conv_kernel: 4,
            vocab: 248_320,
            context: 524_288,
            eps: 1e-6,
        }
    }

    /// A loop of width 8 revisited 4 times: 24 revisits over 32 physical layers,
    /// shaped like the pinned model's plan but with arbitrary attention placement.
    fn synthetic_plan() -> LoopPlan {
        let mut recurrent = vec![true; 56];
        // An attention layer inside the loop range, with every revisit of the same
        // weight set marked consistently (the plan constructor rejects otherwise), and
        // a tail attention layer.
        for il in [7, 15, 23, 31, 40] {
            recurrent[il] = false;
        }
        LoopPlan {
            loop_start: 4,
            loop_end: 12,
            loop_count: 4,
            mhc_start: 0,
            mhc_end: 0,
            mhc_iterations: 1,
            mhc_pre_scale: 1.0,
            mhc_fixed_identity: false,
            recurrent,
        }
    }

    #[test]
    fn physical_kinds_follow_the_loop_mapping() {
        let plan = synthetic_plan();
        // Logical 16 revisits physical 8 inside the loop; the tail visit 40 is its own
        // physical layer 16 (40 - 24 revisits).
        assert_eq!(plan.physical(16), 8);
        assert_eq!(plan.physical(40), 16);
        let kinds = physical_kinds(&plan);
        assert_eq!(kinds.len(), 32);
        // Physical 7 is the looped attention layer; physical 16 is the tail attention
        // layer (block(16) = 40). Both revisits of 7 and the tail visit keep the kind.
        assert_eq!(kinds[7], AttentionKind::Full);
        assert_eq!(kinds[16], AttentionKind::Full);
        assert_eq!(kinds[15], AttentionKind::DeltaNet);
        assert_eq!(
            kinds.iter().filter(|k| **k == AttentionKind::Full).count(),
            2
        );
        // Every revisit of an attention layer maps back to its attention physical.
        for il in 0..56 {
            let kind = if plan.recurrent[il] {
                AttentionKind::DeltaNet
            } else {
                AttentionKind::Full
            };
            assert_eq!(kinds[plan.physical(il)], kind, "visit {il}");
        }
    }

    #[test]
    fn host_shape_matches_the_pinned_model() {
        let plan = synthetic_plan();
        let rope = RopePlan {
            dims: 64,
            sections: [32, 0, 0, 0],
            freq_base: 1e7,
            factor: 2.0,
            original_context: 262_144,
        };
        let host = host_shape(&pinned_shape(), physical_kinds(&plan), &rope)
            .expect("the pinned geometry passes the shared checks");
        assert_eq!(host.hidden, 4096);
        assert_eq!(host.vocab, 248_320);
        assert_eq!(host.kinds.len(), 32);
        assert_eq!(host.delta.key_heads, 16);
        assert_eq!(host.delta.value_heads, 32);
        assert_eq!(host.delta.conv_kernel, 4);
        assert_eq!(host.attn.head_dim, 256);
        assert_eq!(host.attn.rotary_dim, 64);
        assert_eq!(host.attn.theta, 1e7);
        assert_eq!(host.dense_inter, Some(12_288));
        assert_eq!(host.moe.experts, 0);
        let yarn = YarnPlan::of(&rope);
        assert_eq!(
            yarn,
            YarnPlan {
                factor: 2.0,
                original_context: 262_144
            }
        );
    }
}
