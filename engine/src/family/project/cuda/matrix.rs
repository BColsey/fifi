//! One matrix of a checkpoint uploaded in the file's own blocks, whatever type it is stored in as
//! long as a native kernel reads it: the entry a family uses for the tensors it holds outside the
//! planned load (an embedding that is also the tied head, a per-layer table).
use crate::{
    EngError,
    kernels::cuda::{CHUNK_LIMIT, CudaContext, NativeWeight, QuantTensor},
    weights::{GgufType, GgufWeights},
};

/// Uploads the tensor `name`, which must be `[k, rows]` in file order.
pub(crate) fn upload_matrix(
    weights: &GgufWeights,
    ctx: &CudaContext,
    name: &str,
    k: usize,
    rows: usize,
) -> Result<NativeWeight, EngError> {
    let tensor = weights
        .header()
        .tensor(name)
        .ok_or_else(|| EngError::Checkpoint(format!("gguf model: tensor {name} is missing")))?;
    if tensor.dims() != [k as u64, rows as u64] {
        return Err(EngError::Checkpoint(format!(
            "gguf model: tensor {name} has dims {:?}, expected [{k}, {rows}]",
            tensor.dims()
        )));
    }
    let source_len = usize::try_from(tensor.byte_len())
        .map_err(|_| EngError::MemoryPlan(format!("gguf model: tensor {name} does not fit")))?;
    let quant = QuantTensor {
        name,
        ty: tensor.ggml_type(),
        n_rows: rows,
        k,
        source_len,
    };
    ctx.upload_native(quant, CHUNK_LIMIT, &mut |at, buf| {
        weights.read_range(name, at as u64, buf)
    })
}

/// Uploads a matrix whose blocks are already in host memory, `rows` rows of `k` values of `ty`.
pub(crate) fn upload_matrix_bytes(
    ctx: &CudaContext,
    name: &str,
    ty: GgufType,
    k: usize,
    rows: usize,
    bytes: &[u8],
) -> Result<NativeWeight, EngError> {
    let quant = QuantTensor {
        name,
        ty,
        n_rows: rows,
        k,
        source_len: bytes.len(),
    };
    ctx.upload_native(quant, CHUNK_LIMIT, &mut |at, buf| {
        let end = at + buf.len();
        buf.copy_from_slice(bytes.get(at..end).ok_or_else(|| {
            EngError::BadRequest(format!("tensor {name}: the read ran past its blocks"))
        })?);
        Ok(())
    })
}
