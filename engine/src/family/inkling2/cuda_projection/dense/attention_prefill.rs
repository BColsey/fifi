//! Tile independent attention projections over the idle logits workspace.

use super::{
    DenseGemvCall, DenseGemvShape, DenseStep, ResidentProjection, Slot, bad, dense_gemv_bytes,
    float_bytes,
};
use crate::{
    EngError,
    family::inkling2::cuda_projection::q8_row_reuse,
    kernels::cuda::{CudaContext, DeviceBuffer, DeviceRange, DeviceStream},
};

fn finish_borrowed<T>(
    stream: &DeviceStream,
    work: impl FnOnce() -> Result<T, EngError>,
) -> Result<T, EngError> {
    match work() {
        Ok(value) => Ok(value),
        Err(error) => {
            // A launch can fail after earlier kernels were queued against borrowed staging.
            // A failed fence is more important than the original error: reuse is unsafe.
            stream.synchronize()?;
            Err(error)
        }
    }
}

pub(in crate::family::inkling2) struct PrefillProjection<'a> {
    pub(in crate::family::inkling2::cuda_projection) context: &'a CudaContext,
    pub(in crate::family::inkling2::cuda_projection) step: &'a DenseStep,
    pub(in crate::family::inkling2::cuda_projection) weights: [Option<ResidentProjection<'a>>; 8],
    pub(in crate::family::inkling2::cuda_projection) scratch: Option<&'a DeviceBuffer>,
}

impl PrefillProjection<'_> {
    pub(in crate::family::inkling2) fn capacity(
        &self,
        input: [&[f32]; 3],
        keep: &[usize],
    ) -> Result<usize, EngError> {
        self.step
            .projection_tile_capacity(&Layout::new(input, self.weights, keep), self.scratch)
    }

    #[allow(clippy::type_complexity)]
    pub(in crate::family::inkling2) fn tile(
        &self,
        inputs: &[[&[f32]; 3]],
        keep: &[usize],
    ) -> Result<(Vec<[Vec<f32>; 8]>, DeviceRange<'_>), EngError> {
        let first = inputs
            .first()
            .ok_or_else(|| bad("empty Inkling prefill projection tile"))?;
        let layout = Layout::new(*first, self.weights, keep);
        let (values, side) = self.step.project_group_tile(
            self.context,
            inputs,
            self.weights,
            &layout,
            self.scratch,
            true,
        )?;
        Ok((
            layout.host_rows(&values).collect(),
            side.ok_or_else(|| bad("missing Inkling attention sidecar"))?,
        ))
    }
}

struct Layout<const N: usize> {
    input: [usize; 3],
    host: [usize; N],
    side: [usize; N],
}

impl<const N: usize> Layout<N> {
    fn new(
        inputs: [&[f32]; 3],
        weights: [Option<ResidentProjection<'_>>; N],
        keep: &[usize],
    ) -> Self {
        let mut host = weights.map(|projection| projection.map_or(0, |p| p.weight.shape[0]));
        let mut side = [0; N];
        for &i in keep {
            side[i] = std::mem::take(&mut host[i]);
        }
        Self {
            input: inputs.map(<[f32]>::len),
            host,
            side,
        }
    }

    fn host_rows<'a>(&'a self, values: &'a [f32]) -> impl Iterator<Item = [Vec<f32>; N]> + 'a {
        values.chunks_exact(self.host.iter().sum()).map(|row| {
            let mut offset = 0;
            self.host.map(|width| {
                let result = row[offset..offset + width].to_vec();
                offset += width;
                result
            })
        })
    }
}

impl DenseStep {
    pub(in crate::family::inkling2::cuda_projection) fn project_group_rows<const N: usize>(
        &self,
        context: &CudaContext,
        inputs: &[[&[f32]; 3]],
        weights: [Option<ResidentProjection<'_>>; N],
        allow_q8_row_reuse: bool,
    ) -> Result<Option<Vec<[Vec<f32>; N]>>, EngError> {
        let Some(first) = inputs.first().filter(|_| inputs.len() > 1) else {
            return Ok(None);
        };
        let layout = Layout::new(*first, weights, &[]);
        let tile = self.projection_tile_capacity(&layout, None)?;
        if tile < 2 || layout.host.iter().all(|&n| n == 0) {
            return Ok(None);
        }
        let mut rows = Vec::with_capacity(inputs.len());
        for chunk in inputs.chunks(tile) {
            let (values, _) = self.project_group_tile(
                context,
                chunk,
                weights,
                &layout,
                None,
                allow_q8_row_reuse,
            )?;
            rows.extend(layout.host_rows(&values));
        }
        Ok(Some(rows))
    }

    fn projection_tile_capacity<const N: usize>(
        &self,
        layout: &Layout<N>,
        scratch: Option<&DeviceBuffer>,
    ) -> Result<usize, EngError> {
        let capacity = match scratch {
            Some(buffer) => buffer.bytes(),
            None => self.arena.tensor(Slot::Output as usize)?.bytes(),
        } / size_of::<f32>();
        let per_row = layout
            .input
            .iter()
            .chain(&layout.host)
            .chain(&layout.side)
            .sum();
        Ok(capacity
            .checked_div(per_row)
            .unwrap_or(0)
            .min(DenseGemvShape::MAX_ACTIVATION_ROWS))
    }

    fn project_group_tile<'a, const N: usize>(
        &'a self,
        context: &CudaContext,
        inputs: &[[&[f32]; 3]],
        weights: [Option<ResidentProjection<'_>>; N],
        layout: &Layout<N>,
        scratch: Option<&'a DeviceBuffer>,
        allow_q8_row_reuse: bool,
    ) -> Result<(Vec<f32>, Option<DeviceRange<'a>>), EngError> {
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
        let m = inputs.len();
        let ld_x: usize = layout.input.iter().sum();
        let ld_host: usize = layout.host.iter().sum();
        let ld_side: usize = layout.side.iter().sum();
        let input_words = m * ld_x;
        let side_at = input_words + m * ld_host;
        let range = |offset, len| match scratch {
            Some(buffer) => buffer.range(float_bytes(offset)?, float_bytes(len)?),
            None => self.range(Slot::Output, offset, len),
        };
        let input = range(0, input_words)?;
        let output = range(input_words, m * ld_host)?;
        let side = (ld_side > 0)
            .then(|| range(side_at, m * ld_side))
            .transpose()?;
        let mut calls = Vec::with_capacity(N);
        for (i, p) in weights
            .into_iter()
            .enumerate()
            .filter_map(|(i, p)| p.map(|p| (i, p)))
        {
            let [rows, cols] = p.weight.shape;
            let (base, widths, ld_y) = if layout.side[i] > 0 {
                (side_at, &layout.side, ld_side)
            } else {
                (input_words, &layout.host, ld_host)
            };
            let output_offset = base + widths[..i].iter().sum::<usize>();
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
                x: range(input_offset, need.x / size_of::<f32>())?,
                y: range(output_offset, need.y / size_of::<f32>())?,
            });
        }
        let project = || {
            if allow_q8_row_reuse {
                let stage = if layout.side.iter().any(|&n| n > 0) {
                    "qkv_sidecar"
                } else {
                    "qkv"
                };
                q8_row_reuse::project_group(context, &self.stream, &calls, stage)
            } else {
                context.dense_gemv_group(&self.stream, &calls)
            }
        };
        let input_rows: Vec<_> = inputs.iter().flatten().copied().collect();
        let values = if let Some(buffer) = scratch {
            let mut encoded = Vec::with_capacity(input.bytes());
            for row in &input_rows {
                if !row.iter().all(|v| v.is_finite()) {
                    return Err(bad("CUDA projection requires finite input"));
                }
                encoded.extend(row.iter().flat_map(|v| v.to_ne_bytes()));
            }
            buffer.write_at(&self.stream, input.offset(), &encoded)?;
            let mut bytes = vec![[0; 4]; output.bytes() / 4];
            finish_borrowed(&self.stream, || {
                project()?;
                buffer.read_at(&self.stream, output.offset(), bytes.as_flattened_mut())
            })?;
            let values: Vec<_> = bytes.into_iter().map(f32::from_ne_bytes).collect();
            if !values.iter().all(|v| v.is_finite()) {
                return Err(bad("CUDA projection produced a nonfinite output"));
            }
            values
        } else {
            self.run_rows(context, &self.stream, &input_rows, [input, output], project)?
        };
        // run_rows fenced the producer stream; attention may consume this range on its own stream.
        Ok((values, side))
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
    fn borrowed_stage_fences_work_on_error_before_cold_reuse() {
        if !device_present("Inkling borrowed staging fence") {
            return;
        }
        let context = CudaContext::new().unwrap();
        let stream = context.stream().unwrap();
        let event = context.event().unwrap();
        let stage = context.allocate(4096).unwrap();
        let failed: Result<(), EngError> = finish_borrowed(&stream, || {
            stage.write_at_enqueue(&stream, 0, &[7; 4096])?;
            stream.record(&event)?;
            Err(bad("injected after accepted work"))
        });
        assert!(
            matches!(failed, Err(EngError::Checkpoint(ref s)) if s.contains("injected after accepted work"))
        );
        assert!(
            event.query().unwrap(),
            "failure returned before accepted work completed"
        );
        let cold = context.stream().unwrap();
        stage.write_at(&cold, 0, &[9; 4096]).unwrap();
        let mut output = vec![0; 4096];
        stage.read_at(&cold, 0, &mut output).unwrap();
        assert!(output.iter().all(|&b| b == 9));
    }

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
            .project_group_rows(&context, &input, weights, true)
            .unwrap()
            .unwrap();
        assert_eq!(tiled, scalar);
        let scratch = context.allocate(7 * (3 * 32 + 17) * 4).unwrap();
        let layout = Layout::new(input[0], weights, &[]);
        assert_eq!(
            step.projection_tile_capacity(&layout, Some(&scratch))
                .unwrap(),
            7
        );
        let (values, side) = step
            .project_group_tile(&context, &input, weights, &layout, Some(&scratch), true)
            .unwrap();
        assert!(side.is_none());
        assert_eq!(layout.host_rows(&values).collect::<Vec<_>>(), scalar);
        // The cold expert path overwrites the stage with new weights after attention.
        let cold: Vec<_> = (0..7 * 32)
            .flat_map(|i| (((i % 17) as f32 - 8.0) * 0.25).to_ne_bytes())
            .collect();
        context.upload_at(&scratch, 0, &cold).unwrap();
        let cold_resident = make(cold, 7);
        assert_eq!(
            step.project_buffer(&context, &scratch, [7, 32], input[0][0])
                .unwrap(),
            step.project(&context, &cold_resident, input[0][0]).unwrap()
        );
        assert!(
            step.project_group_rows(&context, &input[..1], weights, true)
                .unwrap()
                .is_none()
        );
    }
}
