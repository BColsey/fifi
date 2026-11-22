//! Predecode the small immutable weights read at every layer visit.

use super::{GgufTextWeights, Tensor, VectorOrder, bad};
use crate::{EngError, weights::dequantize_into};

const MAX_BYTES: usize = 16 << 20;

fn selected(name: &str) -> bool {
    let Some(name) = name.strip_prefix("model.llm.") else {
        return false;
    };
    if name.starts_with("attn_res.")
        || matches!(
            name,
            "middle_cycle_cross_pass_queries"
                | "middle_cycle_cross_pass_kv_queries"
                | "norm.weight"
                | "embed_norm.weight"
        )
    {
        return true;
    }
    let Some((_, suffix)) = name
        .strip_prefix("layers.")
        .and_then(|name| name.split_once('.'))
    else {
        return false;
    };
    matches!(
        suffix,
        "attn_sconv.weight"
            | "mlp_sconv.weight"
            | "attn.k_sconv.weight"
            | "attn.v_sconv.weight"
            | "attn.k_sconv.static_w"
            | "attn.v_sconv.static_w"
            | "attn.q_l2_clip.scale"
            | "attn.rel_l2_clip.scale"
            | "mlp_norm.weight"
            | "attn_norm.weight"
            | "q_input_norm.weight"
            | "k_input_norm.weight"
            | "v_input_norm.weight"
            | "mlp.gate.bias"
    )
}

pub(super) fn populate(weights: &mut GgufTextWeights) -> Result<(usize, usize), EngError> {
    let mut count = 0;
    let mut bytes = 0;
    for (name, tensor) in &mut weights.tensors {
        if !selected(name) || tensor.planes != 1 {
            continue;
        }
        let Some(size) = tensor
            .rows
            .checked_mul(tensor.columns)
            .and_then(|n| n.checked_mul(size_of::<f32>()))
        else {
            continue;
        };
        if size > MAX_BYTES - bytes {
            continue;
        }
        let source = weights
            .mapping
            .as_ref()
            .get(tensor.bytes.clone())
            .ok_or_else(|| bad(format!("GGUF tensor {name} escapes the mapping")))?;
        tensor.decoded = Some(decode_vec(tensor, source)?.into_boxed_slice());
        bytes += size;
        count += 1;
    }
    Ok((count, bytes))
}

pub(super) fn decode_vec(tensor: &Tensor, bytes: &[u8]) -> Result<Vec<f32>, EngError> {
    let (cols, rows) = (tensor.columns, tensor.rows);
    let mut values = vec![0.0; cols * rows];
    dequantize_into(tensor.kind, bytes, &mut values)?;
    let transpose = |flip| {
        let values = &values;
        (0..cols)
            .flat_map(move |col| {
                let col = if flip { cols - 1 - col } else { col };
                (0..rows).map(move |row| values[row * cols + col])
            })
            .collect()
    };
    match tensor.vector_order {
        VectorOrder::Stored => Ok(values),
        VectorOrder::Transpose => Ok(transpose(false)),
        VectorOrder::ConvTaps => Ok(transpose(true)),
    }
}

#[cfg(test)]
mod tests;
