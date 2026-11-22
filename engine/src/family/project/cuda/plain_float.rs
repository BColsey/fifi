//! A dense role the file stores as F32, F16 or BF16, loaded as Q8_0 blocks.
//!
//! The decode matvec, the prefill GEMMs and the shard planner read the file's own quantized
//! blocks of every dense role; a plain float matrix (the DeltaNet gates of an Unsloth dynamic
//! file are F32) gets there by one host encode at load, the same Q8_0 form any Q8_0 file holds.
use super::{CudaWeight, LoadKnobs};
use crate::{
    EngError,
    kernels::cuda::{QuantTensor, plan},
    weights::{GgufTensor, GgufType, GgufWeights, plain_float_as_q8_0},
};

/// Uploads `tensor` (a plain float matrix of a dense role) as Q8_0 blocks in the native layout,
/// whole or as the `slice` this device holds.
pub(super) fn upload_as_q8_0(
    knobs: &LoadKnobs<'_>,
    weights: &GgufWeights,
    tensor: &GgufTensor,
    slice: Option<&plan::TensorSlice>,
) -> Result<CudaWeight, EngError> {
    let name = tensor.name();
    let (_, bytes) = weights.read_tensor(name)?;
    let (dims, bytes) = match slice {
        Some(slice) => (slice.dims.clone(), plan::assemble_slice(&bytes, slice)?),
        None => (tensor.dims().to_vec(), bytes),
    };
    let shape_error = || EngError::Checkpoint(format!("tensor {name} has an unusable shape"));
    let (&k, rest) = dims.split_first().ok_or_else(shape_error)?;
    let n_rows = rest
        .iter()
        .try_fold(1u64, |rows, &dim| rows.checked_mul(dim))
        .ok_or_else(shape_error)?;
    let (n_rows, k) = (
        usize::try_from(n_rows).map_err(|_| shape_error())?,
        usize::try_from(k).map_err(|_| shape_error())?,
    );
    let encoded = plain_float_as_q8_0(tensor.ggml_type(), &bytes)?;
    let quant = QuantTensor {
        name,
        ty: GgufType::Q80,
        n_rows,
        k,
        source_len: encoded.len(),
    };
    let weight = knobs
        .context
        .upload_native(quant, knobs.chunk_limit, &mut |at, buf| {
            let end = at + buf.len();
            buf.copy_from_slice(encoded.get(at..end).ok_or_else(|| {
                EngError::BadRequest(format!("tensor {name}: the encoded read ran past its end"))
            })?);
            Ok(())
        })?;
    Ok(CudaWeight::Native {
        name: name.to_owned(),
        weight,
    })
}
