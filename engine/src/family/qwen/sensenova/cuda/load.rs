//! The device load of SenseNova-Loop: the trunk name template over the GGUF, uploaded
//! whole through the shared loader in Native layout — Q4_K_M stays 5.5 GB of blocks the
//! MMVQ matvec reads as stored, the small F32 tensors (conv, `ssm_a`, `dt.bias`, every
//! norm, the mHC bias) land as raw F32 copies.
//!
//! No [`Inventory`](crate::family::gguf_model::Inventory): the CPU reference owns the same
//! name template (`weights.rs`), and this list is its device twin. `load_sliced` refuses a
//! named tensor the file lacks, which is the whole validation; a file tensor the list does
//! not name is simply never uploaded.
//!
//! The embedding stays off the device (the Flash-Next pattern): one 8 KiB F16 row per
//! token is gathered on the host, and the card spends its bytes on `output.weight`
//! (≈0.5 GB Q4_K_M) which every step needs.

use super::super::weights::LoopShape;
use super::super::{LoopPlan, PHYSICAL_LAYERS};

/// Every tensor the device holds, in load order: the root head, then each physical
/// layer's common tensors, its mixer tensors, its mHC routers when in range.
pub(crate) fn tensor_names(plan: &LoopPlan) -> Vec<String> {
    let mut names = vec!["output.weight".to_owned()];
    for p in 0..PHYSICAL_LAYERS {
        let stem = format!("blk.{}", plan.block(p));
        names.extend(common(&stem));
        if plan.recurrent[plan.block(p)] {
            names.extend(delta(&stem));
        } else {
            names.extend(attn(&stem));
        }
        if (plan.mhc_start..plan.mhc_end).contains(&p) {
            for part in ["hc_attn_fn", "hc_ffn_fn"] {
                names.push(format!("{stem}.{part}.weight"));
                names.push(format!("{stem}.{part}.bias"));
            }
        }
    }
    names.push("output_norm.weight".to_owned());
    names
}

fn common(stem: &str) -> impl Iterator<Item = String> + use<'_> {
    [
        "attn_norm.weight",
        "post_attention_norm.weight",
        "ffn_gate.weight",
        "ffn_up.weight",
        "ffn_down.weight",
    ]
    .iter()
    .map(move |tail| format!("{stem}.{tail}"))
}

fn delta(stem: &str) -> impl Iterator<Item = String> + use<'_> {
    [
        "attn_qkv_gate.weight",
        "ssm_beta_alpha.weight",
        "ssm_out.weight",
        "ssm_conv1d.weight",
        "ssm_a",
        "ssm_dt.bias",
        "ssm_norm.weight",
    ]
    .iter()
    .map(move |tail| format!("{stem}.{tail}"))
}

fn attn(stem: &str) -> impl Iterator<Item = String> + use<'_> {
    [
        "attn_q.weight",
        "attn_k.weight",
        "attn_v.weight",
        "attn_output.weight",
        "attn_q_norm.weight",
        "attn_k_norm.weight",
    ]
    .iter()
    .map(move |tail| format!("{stem}.{tail}"))
}

/// Uploads every tensor of the template in one all-or-nothing Native load. Admission
/// and reserve are the caller's `options`; the layout must be Native (the resident
/// kernels read stored blocks).
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
pub(crate) fn load(
    loader: &crate::family::project::cuda::BorrowedLoader<'_>,
    weights: &crate::weights::GgufWeights,
    plan: &LoopPlan,
    options: crate::family::project::cuda::LoadOptions<'_>,
) -> Result<crate::family::project::cuda::CudaWeights, crate::EngError> {
    let names = tensor_names(plan);
    let refs: Vec<&str> = names.iter().map(String::as_str).collect();
    loader.load_sliced(weights, &refs, options, &[], 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pinned_plan() -> LoopPlan {
        let mut recurrent = vec![true; 56];
        for il in [7, 15, 23, 31, 40] {
            recurrent[il] = false;
        }
        LoopPlan {
            loop_start: 4,
            loop_end: 12,
            loop_count: 4,
            mhc_start: 8,
            mhc_end: 24,
            mhc_iterations: 1,
            mhc_pre_scale: 1.0,
            mhc_fixed_identity: false,
            recurrent,
        }
    }

    #[test]
    fn template_covers_every_physical_layer_once() {
        let shape = LoopShape {
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
        };
        let plan = pinned_plan();
        let names = tensor_names(&plan);
        // No tensor twice (a revisited block's tensors appear under one stem only).
        let mut sorted = names.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), names.len());
        // Head and final norm.
        assert!(names.first().is_some_and(|n| n == "output.weight"));
        assert!(names.last().is_some_and(|n| n == "output_norm.weight"));
        // A looped delta layer keeps its own stem and tensors.
        assert!(names.contains(&"blk.5.attn_qkv_gate.weight".to_owned()));
        // The loop's revisits introduce no second stem: physical layers run over blocks
        // {0..3} head, {4..11} loop, {36..55} tail — nothing at or above 56.
        assert!(names.contains(&"blk.36.attn_qkv_gate.weight".to_owned()));
        assert!(!names.contains(&"blk.56.attn_qkv_gate.weight".to_owned()));
        // The attention layer inside the loop has attention tensors, no GDN ones.
        assert!(names.contains(&"blk.7.attn_q.weight".to_owned()));
        assert!(!names.contains(&"blk.7.ssm_out.weight".to_owned()));
        // Tail physical 16 is the tail attention visit 40's weight set: stem blk.40.
        assert!(names.contains(&"blk.40.attn_q.weight".to_owned()));
        // mHC routers on physical layers 8..24, both parts, weight and bias.
        assert!(names.contains(&"blk.8.hc_attn_fn.weight".to_owned()));
        assert!(names.contains(&"blk.47.hc_ffn_fn.bias".to_owned()));
        assert!(!names.contains(&"blk.7.hc_attn_fn.weight".to_owned()));
        assert!(!names.contains(&"blk.48.hc_ffn_fn.weight".to_owned()));
        // 32 layers: 30 delta (7 tensors) + 2 attn (6 tensors) + 5 common each, 16
        // mhc layers x 4 tensors, head + final norm.
        assert_eq!(names.len(), 2 + 32 * 5 + 30 * 7 + 2 * 6 + 16 * 4);
    }
}
