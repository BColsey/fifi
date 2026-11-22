//! Host projections from checked GGUF tensor ranges. Weight chunks stay packed,
//! and scratch storage is reused across projections.

use std::sync::Arc;

use crate::{
    EngError,
    family::math::dot,
    weights::{GgufHeader, GgufType, GgufWeights, dequantize_into},
};

use super::Projector;

const CHUNK_BYTES: usize = 16 << 20;

/// One stored matrix: `rows x cols` in projection orientation
/// (`y = x . W^T`), still quantized in the file.
#[derive(Debug, Clone)]
pub(crate) struct GgufMatrix {
    name: String,
    ty: GgufType,
    rows: usize,
    cols: usize,
    row_bytes: usize,
}

impl GgufMatrix {
    pub(crate) fn new(
        header: &GgufHeader,
        name: &str,
        rows: usize,
        cols: usize,
    ) -> Result<Self, EngError> {
        let tensor = header
            .tensor(name)
            .ok_or_else(|| bad(format!("tensor {name} is missing")))?;
        let dims = tensor.dims();
        if dims != [cols as u64, rows as u64] || rows == 0 || cols == 0 {
            return Err(bad(format!(
                "tensor {name} must be {rows}x{cols}, found {dims:?}"
            )));
        }
        let ty = tensor.ggml_type();
        // Scaled NVFP4 needs a separate tensor scale, which this adapter cannot apply.
        if ty == GgufType::Nvfp4 || dequantize_into(ty, &[], &mut []).is_err() {
            return Err(bad(format!(
                "tensor {name}: unsupported host type {}",
                ty.name()
            )));
        }
        let block = ty.block_elems() as usize;
        if !cols.is_multiple_of(block) {
            return Err(bad(format!("tensor {name} has ragged rows")));
        }
        let row_bytes = (cols / block)
            .checked_mul(ty.block_bytes() as usize)
            .ok_or_else(|| bad(format!("tensor {name} row size overflows")))?;
        if row_bytes > CHUNK_BYTES
            || (row_bytes as u64).checked_mul(rows as u64) != Some(tensor.byte_len())
        {
            return Err(bad(format!("tensor {name} has an invalid host row size")));
        }
        Ok(Self {
            name: name.into(),
            ty,
            rows,
            cols,
            row_bytes,
        })
    }
}

/// Portable projector for dense GGUF matrices and embedding rows.
pub(crate) struct GgufProjector {
    weights: Arc<GgufWeights>,
    raw: Vec<u8>,
    row: Vec<f32>,
}

impl GgufProjector {
    pub(crate) fn new(weights: Arc<GgufWeights>) -> Self {
        Self {
            weights,
            raw: Vec::new(),
            row: Vec::new(),
        }
    }

    fn read_rows(
        &mut self,
        weight: &GgufMatrix,
        first: usize,
        count: usize,
    ) -> Result<(), EngError> {
        self.raw.resize(count * weight.row_bytes, 0);
        self.weights.read_range(
            &weight.name,
            first as u64 * weight.row_bytes as u64,
            &mut self.raw,
        )
    }
}

impl Projector for GgufProjector {
    type Weight = GgufMatrix;

    fn linear(&mut self, weight: &GgufMatrix, x: &[f32]) -> Result<Vec<f32>, EngError> {
        if x.len() != weight.cols {
            return Err(EngError::BadRequest(format!(
                "gguf projection: {} takes {} inputs, got {}",
                weight.name,
                weight.cols,
                x.len()
            )));
        }
        let mut result = Vec::new();
        result
            .try_reserve_exact(weight.rows)
            .map_err(|error| EngError::MemoryPlan(error.to_string()))?;
        result.resize(weight.rows, 0.0);
        let chunk_rows = CHUNK_BYTES / weight.row_bytes;
        for (chunk, out) in result.chunks_mut(chunk_rows).enumerate() {
            self.read_rows(weight, chunk * chunk_rows, out.len())?;
            if weight.ty == GgufType::Q4K {
                crate::kernels::expert_cpu::q4k_matvec(&self.raw, out.len(), weight.cols, x, out)
                    .map_err(|error| bad(format!("tensor {}: {error:?}", weight.name)))?;
                continue;
            }
            self.row.resize(weight.cols, 0.0);
            for (bytes, value) in self.raw.chunks_exact(weight.row_bytes).zip(out) {
                dequantize_into(weight.ty, bytes, &mut self.row)?;
                *value = dot(x, &self.row)?;
            }
        }
        Ok(result)
    }

    fn expert_linear(
        &mut self,
        _stack: &GgufMatrix,
        _expert: u32,
        _x: &[f32],
    ) -> Result<Vec<f32>, EngError> {
        Err(EngError::BadRequest(
            "dense GGUF projections do not have experts".into(),
        ))
    }

    fn rows(&mut self, table: &GgufMatrix, wanted: &[u32]) -> Result<Vec<f32>, EngError> {
        let mut out = Vec::new();
        let values = wanted
            .len()
            .checked_mul(table.cols)
            .ok_or_else(|| EngError::MemoryPlan("gguf embedding size overflows".into()))?;
        out.try_reserve_exact(values)
            .map_err(|error| EngError::MemoryPlan(error.to_string()))?;
        for &id in wanted {
            let row = id as usize;
            if row >= table.rows {
                return Err(EngError::BadRequest(format!(
                    "gguf embedding: token {id} is out of range"
                )));
            }
            self.read_rows(table, row, 1)?;
            let start = out.len();
            out.resize(start + table.cols, 0.0);
            dequantize_into(table.ty, &self.raw, &mut out[start..])?;
        }
        Ok(out)
    }
}

fn bad(message: String) -> EngError {
    EngError::Checkpoint(format!("gguf projection: {message}"))
}
