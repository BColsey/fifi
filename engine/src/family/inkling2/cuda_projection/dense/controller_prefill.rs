//! Batch all three stages of one resident controller without changing dot products.

use super::{
    ControllerWeights, DenseGemvCall, DenseGemvShape, DenseStep, Slot, bad, dense_gemv_bytes,
};
use crate::{
    EngError,
    kernels::cuda::{CudaContext, DeviceRange},
};

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
        let reduced = self.range(Slot::Output, gate_words, m * rank)?;
        let output = self.range(Slot::Output, output_offset, gate_words)?;
        let calls = batch_controller_calls(weights, m, [input, reduced, output])?;
        self.run_rows(context, &self.stream, inputs, [input, output], || {
            context.dense_gemv_group(&self.stream, &calls)
        })
    }
}

/// Both controller paths use the same batched GEMVs and interleaved gain/forget layout.
pub(super) fn batch_controller_calls<'a>(
    weights: ControllerWeights<'a>,
    rows: usize,
    [gate, reduced, coefficients]: [DeviceRange<'a>; 3],
) -> Result<[DenseGemvCall<'a>; 3], EngError> {
    let hidden = weights[1].0.shape[0];
    let shapes = controller_shapes(weights, rows);
    let call = |index: usize, input: DeviceRange<'a>, (output, at): (DeviceRange<'a>, usize)| {
        let (weight, dtype) = weights[index];
        let shape = shapes[index];
        let need = dense_gemv_bytes(shape, dtype)?;
        Ok::<_, EngError>(DenseGemvCall {
            shape,
            dtype,
            w: weight.codes.range(0, need.w)?,
            x: input.slice(0, need.x)?,
            y: output.slice(at, need.y)?,
        })
    };
    Ok([
        call(0, gate, (reduced, 0))?,
        call(1, reduced, (coefficients, 0))?,
        call(2, reduced, (coefficients, hidden * size_of::<f32>()))?,
    ])
}

pub(super) fn admits_controller_weights(weights: ControllerWeights<'_>, rows: usize) -> bool {
    controller_shapes(weights, rows)
        .into_iter()
        .zip(weights)
        .all(|(shape, (weight, dtype))| {
            dense_gemv_bytes(shape, dtype).is_ok_and(|need| weight.codes.bytes() >= need.w)
        })
}

fn controller_shapes(weights: ControllerWeights<'_>, rows: usize) -> [DenseGemvShape; 3] {
    let mut shapes = weights.map(|(weight, _)| {
        let [n, k] = weight.shape;
        DenseGemvShape::packed(rows, n, k)
    });
    let stride = 2 * weights[1].0.shape[0];
    shapes[1].ld_y = stride;
    shapes[2].ld_y = stride;
    shapes
}
