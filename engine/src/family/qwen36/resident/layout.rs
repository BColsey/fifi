//! Portable helpers of the Q36 resident loader: tensor layout changes and the expert page plan.
//!
//! Nothing here calls the native adapter, so the tests run on Linux.

use crate::{
    EngError,
    family::{
        flash_next::stored::{RoleRange, StackLayout, widen_bf16_in_place},
        qwen36::host_cpu::dequantize_matrix,
    },
    kernels::{PackedWeight, QuantizationMode},
    tensor::{Dtype, Tensor},
};

fn mismatch(what: &str) -> EngError {
    EngError::BadRequest(format!("{what} length does not match its shape"))
}

/// Splits a row-major `[rows, kernel]` convolution into its first `kernel - 1` taps (`[rows, kernel - 1]`,
/// oldest tap first) and its last column (`[rows]`, the tap of the newest value).
pub(super) fn split_conv(
    conv: &[f32],
    rows: usize,
    kernel: usize,
) -> Result<(Vec<f32>, Vec<f32>), EngError> {
    if kernel < 2 || Some(conv.len()) != rows.checked_mul(kernel) {
        return Err(mismatch("convolution"));
    }
    let (mut taps, mut last) = (
        Vec::with_capacity(conv.len() - rows),
        Vec::with_capacity(rows),
    );
    for row in conv.chunks_exact(kernel) {
        if let Some((tail, head)) = row.split_last() {
            taps.extend_from_slice(head);
            last.push(*tail);
        }
    }
    Ok((taps, last))
}

pub(super) fn transpose(values: &[f32], rows: usize, cols: usize) -> Result<Vec<f32>, EngError> {
    if Some(values.len()) != rows.checked_mul(cols) {
        return Err(mismatch("matrix"));
    }
    (0..cols)
        .flat_map(|c| (0..rows).map(move |r| r * cols + c))
        .map(|i| values.get(i).copied())
        .collect::<Option<Vec<f32>>>()
        .ok_or_else(|| mismatch("matrix"))
}

/// Returns `-exp(A_log)` for each value head. The decay of a step is `exp(neg_exp * softplus(a + dt_bias))`.
pub(super) fn neg_exp(a_log: &[f32]) -> Vec<f32> {
    a_log.iter().map(|v| -(v.exp())).collect()
}

/// Dequantizes the router `[experts, hidden]` and returns it transposed as `[hidden, experts]`,
/// the right operand of `mm(x, router_t)`.
pub(super) fn router_transposed(
    router: &PackedWeight,
    experts: usize,
    hidden: usize,
) -> Result<Vec<f32>, EngError> {
    transpose(&dequantize_matrix(router)?, experts, hidden)
}

/// Dequantizes the one-row shared expert gate `[1, hidden]`. The values are also the `[hidden, 1]`
/// right operand of `mm(x, gate)`.
pub(super) fn scalar_gate(gate: &PackedWeight, hidden: usize) -> Result<Vec<f32>, EngError> {
    let values = dequantize_matrix(gate)?;
    if values.len() != hidden {
        return Err(mismatch("shared expert gate"));
    }
    Ok(values)
}

/// Expands every BF16 value of `tensor` to F32. BF16 is the high half of an F32, so the expansion is
/// exact. Any other dtype returns `BadRequest`.
fn widen_tensor(tensor: &Tensor) -> Result<Tensor, EngError> {
    if tensor.dtype() != Dtype::Bf16 {
        return Err(EngError::BadRequest(
            "F32 scale storage needs BF16 scales and biases".into(),
        ));
    }
    let bytes = tensor
        .host_bytes()
        .ok_or_else(|| EngError::BadRequest("scales and biases need host bytes".into()))?;
    let wide = bytes
        .as_chunks::<2>()
        .0
        .iter()
        .flat_map(|&[low, high]| [0, 0, low, high])
        .collect();
    Tensor::try_from_host(Dtype::F32, tensor.shape().to_vec(), wide)
}

/// Returns `weight` with its scales and biases stored as F32. The payload tensor is shared.
///
/// This is the default storage: MLX casts BF16 scales and
/// biases to the F32 of the activation on every call of `qmm` and `gather_qmm`, and an F32
/// operand skips the cast. The values are the same, so the products do not change. Only affine
/// weights with BF16 scales and biases are accepted; others return `BadRequest`.
pub(super) fn with_f32_scales(weight: &PackedWeight) -> Result<PackedWeight, EngError> {
    let spec = weight.quantization();
    if spec.mode != QuantizationMode::Affine {
        return Err(EngError::BadRequest(
            "F32 scale storage needs an affine weight".into(),
        ));
    }
    let scales = widen_tensor(weight.scales())?;
    let biases = weight.biases().map(widen_tensor).transpose()?;
    PackedWeight::try_new(
        weight.payload().clone(),
        scales,
        biases,
        weight.logical_shape().to_vec(),
        spec,
    )
}

/// The wide form of one expert stack and the layout that reads it.
///
/// `wide` is the layout of the finished pages: the payload as stored, the scales and biases as F32.
/// `read` places each BF16 role in the upper half of its F32 role, so that `read_stack_into` fills
/// it and `widen_stack_scales` expands it in place.
pub(super) fn widen_layout(
    narrow: &StackLayout,
    page: usize,
) -> Result<(StackLayout, StackLayout), EngError> {
    let overflow = || EngError::BadRequest("widened stack layout overflows".into());
    let biases = narrow.biases.as_ref().ok_or_else(|| {
        EngError::BadRequest("F32 scale storage needs affine expert stacks".into())
    })?;
    let count = usize::try_from(narrow.count).map_err(|_| overflow())?;
    let mut cursor = narrow.scales.offset;
    let mut place = |role: &RoleRange| -> Result<(RoleRange, RoleRange), EngError> {
        if role.dtype != Dtype::Bf16 {
            return Err(EngError::BadRequest(
                "F32 scale storage needs BF16 scales and biases".into(),
            ));
        }
        let narrow_total = role.slice_len.checked_mul(count).ok_or_else(overflow)?;
        let wide_total = narrow_total.checked_mul(2).ok_or_else(overflow)?;
        let wide = RoleRange {
            offset: cursor,
            slice_len: role.slice_len.checked_mul(2).ok_or_else(overflow)?,
            dtype: Dtype::F32,
            shape: role.shape.clone(),
        };
        let read = RoleRange {
            offset: cursor.checked_add(narrow_total).ok_or_else(overflow)?,
            ..role.clone()
        };
        cursor = cursor
            .checked_add(wide_total)
            .and_then(|end| end.checked_next_multiple_of(page))
            .ok_or_else(overflow)?;
        Ok((wide, read))
    };
    let (wide_scales, read_scales) = place(&narrow.scales)?;
    let (wide_biases, read_biases) = place(biases)?;
    let wide = StackLayout {
        scales: wide_scales,
        biases: Some(wide_biases),
        end: cursor,
        ..narrow.clone()
    };
    let read = StackLayout {
        scales: read_scales,
        biases: Some(read_biases),
        end: cursor,
        ..narrow.clone()
    };
    Ok((wide, read))
}

/// Expands the BF16 scales and biases that `read_stack_into` wrote at the read layout of `wide`.
/// `wide` is the first layout that `ExpertPlan::new_f32_scales` returns for the stack.
pub(super) fn widen_stack_scales(bytes: &mut [u8], wide: &StackLayout) -> Result<(), EngError> {
    for role in [Some(&wide.scales), wide.biases.as_ref()]
        .into_iter()
        .flatten()
    {
        let len = usize::try_from(wide.count)
            .ok()
            .and_then(|count| role.slice_len.checked_mul(count))
            .ok_or_else(|| EngError::BadRequest("widened stack role overflows".into()))?;
        let region = role
            .offset
            .checked_add(len)
            .and_then(|end| bytes.get_mut(role.offset..end))
            .ok_or_else(|| EngError::BadRequest("widened stack role passes the end".into()))?;
        widen_bf16_in_place(region)?;
    }
    Ok(())
}
