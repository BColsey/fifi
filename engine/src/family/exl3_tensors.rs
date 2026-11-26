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
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
pub(in crate::family) fn norm_layout(
    dtype: Dtype,
    shape: &[u32],
    values: usize,
) -> Result<usize, EngError> {
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
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
pub(in crate::family) fn effective_norm_bytes(
    tensor: &Tensor,
    values: usize,
) -> Result<Vec<u8>, EngError> {
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

pub(in crate::family) fn widen_f32(
    tensor: &Tensor,
    values: usize,
    what: &str,
) -> Result<Vec<f32>, EngError> {
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
pub(in crate::family) fn f32_le_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

/// A log whose `-exp(A_log)` overflows F32 is refused rather than loaded as an infinite decay.
pub(in crate::family) fn decay_from_log(log: &[f32]) -> Result<Vec<f32>, EngError> {
    let decay: Vec<f32> = log.iter().map(|a| -a.exp()).collect();
    if decay.iter().all(|a| a.is_finite()) {
        return Ok(decay);
    }
    Err(EngError::Checkpoint(
        "EXL3 DeltaNet: -exp(A_log) overflows F32".into(),
    ))
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
