#![allow(unsafe_code)]
//! AArch64 integer row dot for GGUF Q8_0 weights and 32-value activation groups.

use std::arch::aarch64::{vaddvq_s32, vdotq_s32, vdupq_n_s32, vld1q_s8};

use super::{GROUP, Q8gRow};

#[derive(Clone, Copy)]
pub(crate) struct Q8DotProduct(());

impl Q8DotProduct {
    pub(crate) fn detect() -> Option<Self> {
        std::arch::is_aarch64_feature_detected!("dotprod").then_some(Self(()))
    }

    pub(crate) fn row(self, weights: &[u8], input: &Q8gRow) -> f32 {
        // SAFETY: Only detect can construct this capability.
        unsafe { row_dot(weights, input) }
    }
}

#[target_feature(enable = "dotprod")]
fn row_dot(row: &[u8], input: &Q8gRow) -> f32 {
    let (blocks, tail) = row.as_chunks::<{ 2 + GROUP }>();
    let (values, input_tail) = input.q.as_chunks::<GROUP>();
    debug_assert!(tail.is_empty() && input_tail.is_empty());
    debug_assert_eq!(blocks.len(), values.len());
    debug_assert_eq!(blocks.len(), input.d.len());
    let mut result = -0.0_f32;
    for ((block, activation), &scale) in blocks.iter().zip(values).zip(&input.d) {
        let weights = block[2..].as_ptr().cast::<i8>();
        let activation = activation.as_ptr();
        // This function enables dotprod and vld1q_s8 supports unaligned reads.
        // SAFETY: Both 32-byte slices contain their two full vector loads.
        let product = unsafe {
            let low = vdotq_s32(vdupq_n_s32(0), vld1q_s8(weights), vld1q_s8(activation));
            vdotq_s32(low, vld1q_s8(weights.add(16)), vld1q_s8(activation.add(16)))
        };
        let weight_scale = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
        result += (weight_scale * scale) * vaddvq_s32(product) as f32;
    }
    result
}
