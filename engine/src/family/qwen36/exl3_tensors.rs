//! Conversions between tensors of a stock EXL3 pack and the F32 vectors the CUDA step reads, and
//! the head order of the DeltaNet adapter.
//!
//! The pack keeps the HF layout: the value heads of a DeltaNet layer follow their key head
//! (`h` uses key head `h / rep`), and `A_log` is stored as the log. The step's kernels read the
//! tiled layout of the GGUF files (`h` uses key head `h % key_heads`) and the decay as
//! `-exp(A_log)`. The loader moves whole value heads into the tiled order and applies the decay
//! once, so no kernel changes.

use half::{bf16, f16};

use crate::{
    EngError,
    tensor::{Dtype, Tensor},
};

/// Exact source vectors, not reshapes of another tensor with the same byte count.
pub(super) fn norm_layout(dtype: Dtype, shape: &[u32], values: usize) -> Result<usize, EngError> {
    if values == 0 || shape.len() != 1 || shape[0] as usize != values {
        return Err(EngError::Checkpoint(format!(
            "EXL3 attention norm: shape {shape:?}, expected [{values}]"
        )));
    }
    if !matches!(dtype, Dtype::F16 | Dtype::Bf16 | Dtype::F32) {
        return Err(EngError::Checkpoint(format!(
            "EXL3 attention norm: unsupported dtype {dtype:?}"
        )));
    }
    values
        .checked_mul(size_of::<f32>())
        .ok_or_else(|| EngError::MemoryPlan("EXL3 norm staging byte count overflow".into()))
}

/// Convert actual storage to F32, then add one in F32, never in F16 or BF16.
pub(super) fn effective_norm_bytes(tensor: &Tensor, values: usize) -> Result<Vec<u8>, EngError> {
    let output_len = norm_layout(tensor.dtype(), tensor.shape(), values)?;
    let bytes = tensor
        .host_bytes()
        .ok_or_else(|| EngError::Checkpoint("EXL3 attention norm requires host storage".into()))?;
    let width = if tensor.dtype() == Dtype::F32 { 4 } else { 2 };
    if bytes.len() != output_len / (4 / width) {
        return Err(EngError::Checkpoint(
            "EXL3 attention norm has invalid storage length".into(),
        ));
    }
    let mut output = Vec::new();
    output
        .try_reserve_exact(output_len)
        .map_err(|error| EngError::MemoryPlan(format!("EXL3 norm staging allocation: {error}")))?;
    for word in bytes.chunks_exact(width) {
        let value = match tensor.dtype() {
            Dtype::F16 => f16::from_bits(u16::from_le_bytes([word[0], word[1]])).to_f32(),
            Dtype::Bf16 => bf16::from_bits(u16::from_le_bytes([word[0], word[1]])).to_f32(),
            Dtype::F32 => f32::from_le_bytes([word[0], word[1], word[2], word[3]]),
            _ => unreachable!("norm_layout checked the dtype"),
        };
        let effective = value + 1.0f32;
        if !value.is_finite() || !effective.is_finite() {
            return Err(EngError::Checkpoint(
                "EXL3 attention norm contains nonfinite values".into(),
            ));
        }
        output.extend_from_slice(&effective.to_le_bytes());
    }
    Ok(output)
}

pub(super) fn widen_f32(tensor: &Tensor, values: usize, what: &str) -> Result<Vec<f32>, EngError> {
    let elements = tensor.shape().iter().try_fold(1usize, |count, &dim| {
        count.checked_mul(usize::try_from(dim).ok()?)
    });
    if elements != Some(values) {
        return Err(EngError::Checkpoint(format!(
            "EXL3 tensor {what}: shape {:?}, expected {values} values",
            tensor.shape()
        )));
    }
    let bytes = tensor
        .host_bytes()
        .ok_or_else(|| EngError::Checkpoint(format!("EXL3 tensor {what} requires host storage")))?;
    let widened: Vec<f32> = match tensor.dtype() {
        Dtype::F16 => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|w| f16::from_bits(u16::from_le_bytes(*w)).to_f32())
            .collect(),
        Dtype::Bf16 => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|w| bf16::from_bits(u16::from_le_bytes(*w)).to_f32())
            .collect(),
        Dtype::F32 => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|w| f32::from_le_bytes(*w))
            .collect(),
        other => {
            return Err(EngError::Checkpoint(format!(
                "EXL3 tensor {what}: dtype {other:?} is not F16, BF16 or F32"
            )));
        }
    };
    if widened.len() != values || widened.iter().any(|value| !value.is_finite()) {
        return Err(EngError::Checkpoint(format!(
            "EXL3 tensor {what} has a bad length or a nonfinite value"
        )));
    }
    Ok(widened)
}

/// Little-endian bytes of F32 values, the form the device copy takes.
pub(super) fn f32_le_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

/// A log whose `-exp(A_log)` overflows F32 is refused rather than loaded as an infinite decay.
pub(super) fn decay_from_log(log: &[f32]) -> Result<Vec<f32>, EngError> {
    let decay: Vec<f32> = log.iter().map(|a| -a.exp()).collect();
    if decay.iter().all(|a| a.is_finite()) {
        return Ok(decay);
    }
    Err(EngError::Checkpoint(
        "EXL3 DeltaNet: -exp(A_log) overflows F32".into(),
    ))
}

/// For each value head `t` of the kernel's tiled order, the head it takes from the pack's
/// grouped order.
pub(super) fn tiled_head_order(
    key_heads: usize,
    value_heads: usize,
) -> Result<Vec<usize>, EngError> {
    if key_heads == 0 || value_heads == 0 || !value_heads.is_multiple_of(key_heads) {
        return Err(EngError::Checkpoint(format!(
            "DeltaNet heads: {value_heads} value heads do not group under {key_heads} key heads"
        )));
    }
    let rep = value_heads / key_heads;
    Ok((0..value_heads)
        .map(|tiled| (tiled % key_heads) * rep + tiled / key_heads)
        .collect())
}

pub(super) fn permute_blocks<T: Copy>(
    values: &[T],
    block: usize,
    order: &[usize],
) -> Result<Vec<T>, EngError> {
    if block == 0 || values.len() != block * order.len() {
        return Err(EngError::Checkpoint(format!(
            "EXL3 head order: {} values do not form {} blocks of {block}",
            values.len(),
            order.len()
        )));
    }
    crate::weights::exl3::check_order(order, order.len())?;
    Ok(order
        .iter()
        .flat_map(|&old| values[old * block..(old + 1) * block].iter().copied())
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_decay_is_minus_exp_of_the_log_and_an_overflowing_log_is_refused() {
        let decay = decay_from_log(&[0.0, 1.0, -2.0]).unwrap();
        assert_eq!(decay[0], -1.0);
        assert!((decay[1] + std::f32::consts::E).abs() < 1e-6);
        assert!((decay[2] + (-2.0f32).exp()).abs() < 1e-7);
        // exp(89) exceeds f32::MAX; one bad head refuses the whole vector.
        for bad in [89.0, f32::INFINITY, f32::NAN] {
            let error = decay_from_log(&[0.0, bad]).unwrap_err();
            assert!(
                matches!(&error, EngError::Checkpoint(m) if m.contains("overflows F32")),
                "{bad}: {error:?}"
            );
        }
    }

    #[test]
    fn tiled_order_regroups_value_heads_under_their_key_head() {
        // Two key heads, four value heads: grouped [g0v0, g0v1, g1v0, g1v1] becomes tiled
        // [g0v0, g1v0, g0v1, g1v1].
        assert_eq!(tiled_head_order(2, 4).unwrap(), [0, 2, 1, 3]);
        // The model's 16 key and 32 value heads.
        let order = tiled_head_order(16, 32).unwrap();
        assert_eq!(order[..4], [0, 2, 4, 6]);
        assert_eq!(order[16..20], [1, 3, 5, 7]);
        let mut sorted = order.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..32).collect::<Vec<_>>());
        // Every tiled head `t` reads the key head the grouped head it holds belongs to.
        for (tiled, &grouped) in order.iter().enumerate() {
            assert_eq!(tiled % 16, grouped / 2);
        }
        assert_eq!(tiled_head_order(4, 4).unwrap(), [0, 1, 2, 3]);
    }

    #[test]
    fn tiled_order_refuses_heads_that_do_not_group() {
        for (keys, values) in [(0, 4), (4, 0), (3, 4), (4, 2)] {
            assert!(matches!(
                tiled_head_order(keys, values),
                Err(EngError::Checkpoint(_))
            ));
        }
    }

    #[test]
    fn blocks_move_whole_and_a_bad_shape_or_order_is_refused() {
        let values = [10, 11, 20, 21, 30, 31];
        assert_eq!(
            permute_blocks(&values, 2, &[2, 0, 1]).unwrap(),
            [30, 31, 10, 11, 20, 21]
        );
        assert!(permute_blocks(&values, 4, &[0, 1]).is_err());
        assert!(permute_blocks(&values, 2, &[0, 1]).is_err());
        assert!(permute_blocks(&values, 2, &[0, 0, 1]).is_err());
        assert!(permute_blocks(&values, 0, &[0, 1, 2]).is_err());
    }

    #[test]
    fn widening_keeps_each_stored_value_and_refuses_bad_input() {
        let half = Tensor::try_from_host(
            Dtype::F16,
            vec![2, 2],
            [0x3c00u16, 0xc000, 0x0000, 0x3800]
                .into_iter()
                .flat_map(u16::to_le_bytes)
                .collect(),
        )
        .unwrap();
        assert_eq!(
            widen_f32(&half, 4, "t").unwrap(),
            [1.0, -2.0, 0.0, 0.5],
            "F16"
        );
        let brain = Tensor::try_from_host(
            Dtype::Bf16,
            vec![2],
            [0x3f80u16, 0xbf80]
                .into_iter()
                .flat_map(u16::to_le_bytes)
                .collect(),
        )
        .unwrap();
        assert_eq!(widen_f32(&brain, 2, "t").unwrap(), [1.0, -1.0], "BF16");
        assert!(widen_f32(&brain, 3, "t").is_err());
        let bad =
            Tensor::try_from_host(Dtype::F16, vec![1], 0x7c00u16.to_le_bytes().to_vec()).unwrap();
        assert!(widen_f32(&bad, 1, "t").is_err());
        let integer = Tensor::try_from_host(Dtype::I16, vec![1], vec![0; 2]).unwrap();
        assert!(widen_f32(&integer, 1, "t").is_err());
        assert_eq!(
            f32_le_bytes(&[1.0, -2.0]),
            [0, 0, 0x80, 0x3f, 0, 0, 0, 0xc0]
        );
    }
}
