//! Batch all three stages of one resident controller without changing dot products.

use super::{
    ControllerWeights, DenseGemvCall, DenseGemvShape, DenseStep, Slot, bad, dense_gemv_bytes,
};
use crate::{EngError, kernels::cuda::CudaContext};

impl DenseStep {
    pub(in crate::family::inkling2::cuda_projection) fn controller_batch_capacity(
        &self,
        [rank, hidden]: [usize; 2],
    ) -> Result<usize, EngError> {
        let words = self.arena.tensor(Slot::Output as usize)?.bytes() / size_of::<f32>();
        let stride = hidden
            .checked_mul(4)
            .and_then(|words| words.checked_add(rank))
            .ok_or_else(|| bad("CUDA batched controller stride overflows"))?;
        Ok(words
            .checked_div(stride)
            .unwrap_or(0)
            .min(DenseGemvShape::MAX_ACTIVATION_ROWS))
    }

    pub(in crate::family::inkling2::cuda_projection) fn project_controller_batch(
        &self,
        context: &CudaContext,
        inputs: &[&[f32]],
        weights: ControllerWeights<'_>,
        shape @ [rank, hidden]: [usize; 2],
    ) -> Result<Option<Vec<[Vec<f32>; 2]>>, EngError> {
        let capacity = self.controller_batch_capacity(shape)?;
        if inputs.len() < 2 || capacity < 2 {
            return Ok(None);
        }
        if inputs.iter().any(|input| input.len() != 2 * hidden)
            || weights.map(|(weight, _)| weight.shape)
                != [[rank, 2 * hidden], [hidden, rank], [hidden, rank]]
        {
            return Err(bad("CUDA batched controller shape differs"));
        }
        let mut rows = Vec::with_capacity(inputs.len());
        for tile in inputs.chunks(capacity) {
            let values = self.project_controller_tile(context, tile, weights, shape)?;
            rows.extend(values.chunks_exact(2 * hidden).map(|row| {
                let (gain, forget) = row.split_at(hidden);
                [gain.to_vec(), forget.to_vec()]
            }));
        }
        Ok(Some(rows))
    }

    fn project_controller_tile(
        &self,
        context: &CudaContext,
        inputs: &[&[f32]],
        weights: ControllerWeights<'_>,
        [rank, hidden]: [usize; 2],
    ) -> Result<Vec<f32>, EngError> {
        let m = inputs.len();
        // [gate rows | low-rank rows | interleaved (gain, forget) rows].
        let gate_words = m * 2 * hidden;
        let output_offset = gate_words + m * rank;
        let input = self.range(Slot::Output, 0, gate_words)?;
        let output = self.range(Slot::Output, output_offset, gate_words)?;
        let call = |index: usize, x, y, ld_y| {
            let (weight, dtype) = weights[index];
            let [rows, cols] = weight.shape;
            let mut shape = DenseGemvShape::packed(m, rows, cols);
            shape.ld_y = ld_y;
            let need = dense_gemv_bytes(shape, dtype)?;
            Ok::<_, EngError>(DenseGemvCall {
                shape,
                dtype,
                w: weight.codes.range(0, need.w)?,
                x: self.range(Slot::Output, x, need.x / size_of::<f32>())?,
                y: self.range(Slot::Output, y, need.y / size_of::<f32>())?,
            })
        };
        let calls = [
            call(0, 0, gate_words, rank)?,
            call(1, gate_words, output_offset, 2 * hidden)?,
            call(2, gate_words, output_offset + hidden, 2 * hidden)?,
        ];
        self.run_rows(context, &self.stream, inputs, [input, output], || {
            context.dense_gemv_group(&self.stream, &calls)
        })
    }
}
