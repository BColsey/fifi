//! A dense role the file stores as F32, F16 or BF16, loaded as Q8_0 blocks.
//!
//! The decode matvec, the prefill GEMMs and the shard planner read the file's own quantized
//! blocks of every dense role; a plain float matrix (the DeltaNet gates of an Unsloth dynamic
//! file are F32) gets there by a host encode at load, the same Q8_0 form any Q8_0 file holds.
//! The encode runs one run of whole rows at a time inside the upload's own chunking, so the
//! source rows, their f32 values and the encoded rows exist together only for one chunk.
use super::{CudaWeight, LoadKnobs, SliceReader};
use crate::{
    EngError,
    kernels::cuda::{QuantTensor, plan},
    weights::{GgufTensor, GgufType, GgufWeights, encode_plain_float_rows},
};

/// Uploads `tensor` (a plain float matrix of a dense role) as Q8_0 blocks in the native layout,
/// whole or as the `slice` this device holds. The device bytes are
/// [`plan::q8_0_encoded_bytes`], the figure admission counted.
pub(super) fn upload_as_q8_0(
    knobs: &LoadKnobs<'_>,
    weights: &GgufWeights,
    tensor: &GgufTensor,
    slice: Option<&plan::TensorSlice>,
) -> Result<CudaWeight, EngError> {
    let name = tensor.name();
    let ty = tensor.ggml_type();
    let dims = slice.map_or_else(|| tensor.dims().to_vec(), |slice| slice.dims.clone());
    let shape_error = || EngError::Checkpoint(format!("tensor {name} has an unusable shape"));
    let (&k, rest) = dims.split_first().ok_or_else(shape_error)?;
    let n_rows = rest
        .iter()
        .try_fold(1u64, |rows, &dim| rows.checked_mul(dim))
        .ok_or_else(shape_error)?;
    let encoded_len = usize::try_from(plan::q8_0_encoded_bytes(name, &dims)?)
        .map_err(|_| EngError::MemoryPlan(format!("tensor {name} does not fit")))?;
    let (n_rows, k) = (
        usize::try_from(n_rows).map_err(|_| shape_error())?,
        usize::try_from(k).map_err(|_| shape_error())?,
    );
    let element = if ty == GgufType::F32 { 4 } else { 2 };
    let (source_row, encoded_row) = (k * element, k / 32 * 34);
    let reader = slice.map(|slice| SliceReader::new(&slice.segments));
    let mut fill = |at: usize, out: &mut [u8]| -> Result<(), EngError> {
        if !at.is_multiple_of(encoded_row) || !out.len().is_multiple_of(encoded_row) {
            return Err(EngError::BadRequest(format!(
                "tensor {name}: the encoded read {at}+{} does not start and end on rows of {encoded_row}",
                out.len()
            )));
        }
        let rows = out.len() / encoded_row;
        let mut source = vec![0u8; rows * source_row];
        let first = (at / encoded_row * source_row) as u64;
        match &reader {
            Some(reader) => reader.fill(first, &mut source, &mut |offset, part| {
                weights.read_range(name, offset, part)
            })?,
            None => weights.read_range(name, first, &mut source)?,
        }
        encode_plain_float_rows(ty, &source, out)
    };
    let quant = QuantTensor {
        name,
        ty: GgufType::Q80,
        n_rows,
        k,
        source_len: encoded_len,
    };
    let weight = knobs
        .context
        .upload_native(quant, knobs.chunk_limit, &mut fill)?;
    Ok(CudaWeight::Native {
        name: name.to_owned(),
        weight,
    })
}
