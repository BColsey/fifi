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
    let reader = slice.map(|slice| SliceReader::new(&slice.segments));
    let mut read_source = |at: u64, out: &mut [u8]| -> Result<(), EngError> {
        match &reader {
            Some(reader) => reader.fill(at, out, &mut |offset, part| {
                weights.read_range(name, offset, part)
            }),
            None => weights.read_range(name, at, out),
        }
    };
    let mut fill = |at: usize, out: &mut [u8]| -> Result<(), EngError> {
        encode_rows_at(name, ty, k, at, out, &mut read_source)
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

/// The reader of source bytes at a byte offset that the row encode pulls from.
type SourceReader<'a> = dyn FnMut(u64, &mut [u8]) -> Result<(), EngError> + 'a;

/// Fills `out`, whole Q8_0 rows of `k` values starting at the encoded byte `at`, from the float
/// rows of type `ty` that `read_source(byte offset, buffer)` returns. The upload asks for one
/// chunk of rows at a time, so a matrix never exists in its source, f32 and encoded forms at once.
fn encode_rows_at(
    name: &str,
    ty: GgufType,
    k: usize,
    at: usize,
    out: &mut [u8],
    read_source: &mut SourceReader<'_>,
) -> Result<(), EngError> {
    let element = if ty == GgufType::F32 { 4 } else { 2 };
    let (source_row, encoded_row) = (k * element, k / 32 * 34);
    if !at.is_multiple_of(encoded_row) || !out.len().is_multiple_of(encoded_row) {
        return Err(EngError::BadRequest(format!(
            "tensor {name}: the encoded read {at}+{} does not start and end on rows of {encoded_row}",
            out.len()
        )));
    }
    let rows = out.len() / encoded_row;
    let mut source = vec![0u8; rows * source_row];
    read_source((at / encoded_row * source_row) as u64, &mut source)?;
    encode_plain_float_rows(ty, &source, out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weights::encode_q8_0;

    const K: usize = 64;
    const ROWS: usize = 7;

    /// A float matrix of `ty`: the bytes the file stores and the f32 values those bytes mean.
    fn matrix(ty: GgufType) -> (Vec<u8>, Vec<f32>) {
        let values: Vec<f32> = (0..ROWS * K)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.173)
            .collect();
        let mut bytes = Vec::new();
        let mut stored = Vec::new();
        for v in values {
            match ty {
                GgufType::F32 => {
                    bytes.extend(v.to_le_bytes());
                    stored.push(v);
                }
                GgufType::F16 => {
                    let h = half::f16::from_f32(v);
                    bytes.extend(h.to_bits().to_le_bytes());
                    stored.push(h.to_f32());
                }
                _ => {
                    let h = half::bf16::from_f32(v);
                    bytes.extend(h.to_bits().to_le_bytes());
                    stored.push(h.to_f32());
                }
            }
        }
        (bytes, stored)
    }

    /// Assembles the whole encoded matrix `rows_per_chunk` rows at a time, as the upload does.
    fn assemble(
        ty: GgufType,
        k: usize,
        rows: usize,
        rows_per_chunk: usize,
        source: &[u8],
    ) -> Vec<u8> {
        let encoded_row = k / 32 * 34;
        let mut out = vec![0u8; rows * encoded_row];
        let mut read = |at: u64, buf: &mut [u8]| -> Result<(), EngError> {
            let at = at as usize;
            buf.copy_from_slice(&source[at..at + buf.len()]);
            Ok(())
        };
        for first in (0..rows).step_by(rows_per_chunk) {
            let n = rows_per_chunk.min(rows - first);
            let chunk = &mut out[first * encoded_row..(first + n) * encoded_row];
            encode_rows_at("t", ty, k, first * encoded_row, chunk, &mut read).expect("chunk");
        }
        out
    }

    #[test]
    fn every_chunk_size_assembles_the_same_blocks_as_one_encode_of_the_whole_matrix() {
        for ty in [GgufType::F32, GgufType::F16, GgufType::Bf16] {
            let (bytes, stored) = matrix(ty);
            let whole = encode_q8_0(&stored).expect("whole");
            assert_eq!(
                whole.len() as u64,
                plan::q8_0_encoded_bytes("t", &[K as u64, ROWS as u64]).unwrap(),
                "{ty:?}: the encoded size is the planned size"
            );
            for rows_per_chunk in [1, 2, 3, ROWS] {
                assert_eq!(
                    assemble(ty, K, ROWS, rows_per_chunk, &bytes),
                    whole,
                    "{ty:?} in chunks of {rows_per_chunk} rows"
                );
            }
        }
    }

    #[test]
    fn a_read_that_does_not_sit_on_whole_rows_is_refused() {
        let mut read = |_: u64, _: &mut [u8]| -> Result<(), EngError> { Ok(()) };
        let mut out = vec![0u8; 34];
        assert!(encode_rows_at("t", GgufType::F32, K, 0, &mut out, &mut read).is_err());
        let mut out = vec![0u8; 2 * 34];
        assert!(encode_rows_at("t", GgufType::F32, K, 34, &mut out, &mut read).is_err());
    }

    /// The sliced upload reads the assembled slice through the segments of its plan: columns and
    /// rows cut out of the file's matrix encode to the blocks of that sub-matrix, and the bytes
    /// are the planned count for the slice's own dims.
    #[test]
    fn a_sliced_matrix_encodes_to_the_blocks_of_its_cut_and_the_planned_bytes() {
        for ty in [GgufType::F32, GgufType::F16, GgufType::Bf16] {
            let (bytes, stored) = matrix(ty);
            let slice =
                plan::plan_slice_rows("t", ty, &[K as u64, ROWS as u64], 32..64, &[1..3, 4..7])
                    .expect("slice");
            assert_eq!(slice.dims, vec![32, 5]);
            let reader = SliceReader::new(&slice.segments);
            let mut read = |at: u64, buf: &mut [u8]| {
                reader.fill(at, buf, &mut |offset, part| {
                    let offset = offset as usize;
                    part.copy_from_slice(&bytes[offset..offset + part.len()]);
                    Ok(())
                })
            };
            let encoded_row = 34;
            let mut out = vec![0u8; 5 * encoded_row];
            for (row, chunk) in out.chunks_mut(2 * encoded_row).enumerate() {
                encode_rows_at("t", ty, 32, row * 2 * encoded_row, chunk, &mut read)
                    .expect("sliced chunk");
            }
            let cut: Vec<f32> = [1usize, 2, 4, 5, 6]
                .iter()
                .flat_map(|&r| stored[r * K + 32..r * K + 64].to_vec())
                .collect();
            assert_eq!(out, encode_q8_0(&cut).expect("cut"), "{ty:?}");
            assert_eq!(
                out.len() as u64,
                plan::q8_0_encoded_bytes("t", &slice.dims).unwrap(),
                "{ty:?}"
            );
        }
    }
}
