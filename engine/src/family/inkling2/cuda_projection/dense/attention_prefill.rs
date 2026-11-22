//! Tile independent attention projections over the idle logits workspace.

use super::{
    DenseGemvCall, DenseGemvShape, DenseStep, ResidentProjection, Slot, bad, dense_gemv_bytes,
};
use crate::{EngError, kernels::cuda::CudaContext};

struct Layout<const N: usize> {
    input: [usize; 3],
    output: [usize; N],
}

impl<const N: usize> Layout<N> {
    fn new(inputs: [&[f32]; 3], weights: [Option<ResidentProjection<'_>>; N]) -> Self {
        Self {
            input: inputs.map(<[f32]>::len),
            output: weights.map(|projection| projection.map_or(0, |p| p.weight.shape[0])),
        }
    }
}

impl DenseStep {
    pub(in crate::family::inkling2::cuda_projection) fn project_group_rows<const N: usize>(
        &self,
        context: &CudaContext,
        inputs: &[[&[f32]; 3]],
        weights: [Option<ResidentProjection<'_>>; N],
    ) -> Result<Option<Vec<[Vec<f32>; N]>>, EngError> {
        let Some(first) = inputs.first().filter(|_| inputs.len() > 1) else {
            return Ok(None);
        };
        let layout = Layout::new(*first, weights);
        let ld_x: usize = layout.input.iter().sum();
        let ld_y: usize = layout.output.iter().sum();
        let capacity = self.arena.tensor(Slot::Output as usize)?.bytes() / size_of::<f32>();
        let tile = capacity
            .checked_div(ld_x + ld_y)
            .unwrap_or(0)
            .min(DenseGemvShape::MAX_ACTIVATION_ROWS);
        if tile < 2 || ld_y == 0 {
            return Ok(None);
        }
        if inputs
            .iter()
            .any(|row| row.map(<[f32]>::len) != layout.input)
            || weights
                .iter()
                .flatten()
                .any(|p| layout.input.get(p.input) != Some(&p.weight.shape[1]))
        {
            return Err(bad("batched CUDA attention projection input shape differs"));
        }

        let mut rows = Vec::with_capacity(inputs.len());
        for chunk in inputs.chunks(tile) {
            let values = self.project_group_tile(context, chunk, weights, &layout)?;
            for row in values.chunks_exact(ld_y) {
                let mut offset = 0;
                rows.push(layout.output.map(|width| {
                    let result = row[offset..offset + width].to_vec();
                    offset += width;
                    result
                }));
            }
        }
        Ok(Some(rows))
    }

    fn project_group_tile<const N: usize>(
        &self,
        context: &CudaContext,
        inputs: &[[&[f32]; 3]],
        weights: [Option<ResidentProjection<'_>>; N],
        layout: &Layout<N>,
    ) -> Result<Vec<f32>, EngError> {
        let m = inputs.len();
        let ld_x: usize = layout.input.iter().sum();
        let ld_y: usize = layout.output.iter().sum();
        let input_words = m * ld_x;
        let input = self.range(Slot::Output, 0, input_words)?;
        let output = self.range(Slot::Output, input_words, m * ld_y)?;
        let mut output_offset = input_words;
        let mut calls = Vec::with_capacity(N);
        for p in weights.into_iter().flatten() {
            let [rows, cols] = p.weight.shape;
            let shape = DenseGemvShape {
                m,
                rows,
                cols,
                ld_x,
                ld_y,
            };
            let need = dense_gemv_bytes(shape, p.dtype)?;
            let input_offset = layout.input[..p.input].iter().sum();
            calls.push(DenseGemvCall {
                shape,
                dtype: p.dtype,
                w: p.weight.codes.range(0, need.w)?,
                x: self.range(Slot::Output, input_offset, need.x / size_of::<f32>())?,
                y: self.range(Slot::Output, output_offset, need.y / size_of::<f32>())?,
            });
            output_offset += rows;
        }
        let input_rows: Vec<_> = inputs.iter().flatten().copied().collect();
        self.run_rows(context, &self.stream, &input_rows, [input, output], || {
            context.dense_gemv_group(&self.stream, &calls)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        family::inkling2::cuda_projection::{Resident, dense::DensePlan, upload_tensor},
        kernels::cuda::{device_present, resident::dense_gemv::DenseDtype},
        weights::GgufType,
    };

    #[test]
    fn attention_projection_tiles_match_scalar_groups() {
        if !device_present("Inkling attention projection tiles") {
            return;
        }
        let context = CudaContext::new().unwrap();
        // Allow exactly three packed token rows in the logits slot.
        let plan = DensePlan::new([3 * (3 * 32 + 17), 32], 32, 4).unwrap();
        let step = DenseStep::new(&context, &plan).unwrap();
        let make = |data: Vec<u8>, rows| Resident {
            codes: upload_tensor(&context, &data).unwrap(),
            shape: [rows, 32],
        };
        let f32 = make(
            (0..7 * 32)
                .flat_map(|i| (((i % 13) as f32 - 7.0) * 0.125).to_ne_bytes())
                .collect(),
            7,
        );
        let bf16 = make(
            (0..5 * 32)
                .flat_map(|i| {
                    half::bf16::from_f32((i % 19) as f32 * -0.03125)
                        .to_bits()
                        .to_ne_bytes()
                })
                .collect(),
            5,
        );
        let q8 = make(
            (0..5)
                .flat_map(|row| {
                    [0, 0x30]
                        .into_iter()
                        .chain((0..32).map(move |i| (i + row) * 3))
                })
                .collect(),
            5,
        );
        let weights = [
            Some(ResidentProjection {
                weight: &f32,
                dtype: DenseDtype::F32,
                input: 2,
            }),
            None,
            Some(ResidentProjection {
                weight: &bf16,
                dtype: DenseDtype::Bf16,
                input: 0,
            }),
            Some(ResidentProjection {
                weight: &q8,
                dtype: DenseDtype::PackedGguf(GgufType::Q80),
                input: 0,
            }),
        ];
        let values: Vec<_> = (0..7)
            .map(|r| {
                std::array::from_fn(|p| {
                    std::array::from_fn::<_, 32, _>(|c| {
                        ((r * 17 + p * 23 + c * 7) % 53) as f32 * 0.125
                    })
                })
            })
            .collect();
        let input: Vec<_> = values
            .iter()
            .map(|row| row.each_ref().map(|v| v.as_slice()))
            .collect();
        let scalar: Vec<_> = input
            .iter()
            .map(|&row| step.project_group(&context, row, weights).unwrap())
            .collect();
        let tiled = step
            .project_group_rows(&context, &input, weights)
            .unwrap()
            .unwrap();
        assert_eq!(tiled, scalar);
        assert!(
            step.project_group_rows(&context, &input[..1], weights)
                .unwrap()
                .is_none()
        );
    }
}
