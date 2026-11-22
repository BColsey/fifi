//! Launch dense projections on one stream. Controller reductions live in the
//! same fixed arena as the input and output rows.

use std::{cell::RefCell, mem::size_of};

use crate::{
    EngError,
    kernels::cuda::{
        ArenaTensor, CudaContext, DeviceArena, DeviceBuffer, DeviceRange, DeviceStream, PinnedBuf,
        plan_arena,
        resident::{
            dense_gemv::{DenseDtype, DenseGemvCall, DenseGemvShape, dense_gemv_bytes},
            matvec_q::{MatvecCall, QuantType, ROWS_MULTIPLE},
            quantize_q8_1::{QuantizeCall, q8_1_bytes},
            router::{MAX_ROWS as ROUTER_MAX_ROWS, RouterLogitsCall, RouterShape},
        },
    },
    weights::GgufType,
};

use super::{Resident, bad};

mod attention_prefill;
mod draft_head;

pub(super) use draft_head::DraftHead;

#[repr(usize)]
enum Slot {
    Input,
    Reduced,
    Output,
}

pub(super) struct DensePlan {
    tensors: [ArenaTensor; 3],
    pub(super) device_bytes: usize,
}

impl DensePlan {
    pub(super) fn new(
        [rows, columns]: [usize; 2],
        hidden: usize,
        rank: usize,
    ) -> Result<Self, EngError> {
        // The controller reuses the input slot for neither its reduction nor its outputs.
        let controller_width = hidden
            .checked_mul(2)
            .ok_or_else(|| bad("CUDA controller output width overflows"))?;
        let extents = [
            float_bytes(columns)?,
            float_bytes(rank)?,
            float_bytes(rows.max(controller_width))?,
        ];
        let tensors = extents.map(|bytes| ArenaTensor {
            bytes,
            first: 0,
            last: 0,
        });
        let device_bytes = plan_arena(&tensors)?.total;
        Ok(Self {
            tensors,
            device_bytes,
        })
    }
}

#[derive(Clone, Copy)]
pub(super) struct ResidentProjection<'a> {
    pub(super) weight: &'a Resident,
    pub(super) dtype: DenseDtype,
    pub(super) input: usize,
}

pub(super) type ControllerWeights<'a> = [(&'a Resident, DenseDtype); 3];

pub(super) struct ControllerRow<'a> {
    pub(super) weights: ControllerWeights<'a>,
    pub(super) input: &'a [f32],
}

pub(super) struct ProjectionBatch<'a> {
    pub(super) weight: DeviceRange<'a>,
    pub(super) activations: usize,
    pub(super) scale: f32,
}

pub(super) struct DenseStep {
    stream: DeviceStream,
    arena: DeviceArena,
    transfer: RefCell<PinnedBuf>,
    q8_activations: bool,
}

impl DenseStep {
    pub(super) fn new(context: &CudaContext, plan: &DensePlan) -> Result<Self, EngError> {
        Ok(Self {
            stream: context.stream()?,
            arena: context.arena(&plan.tensors)?,
            q8_activations: false,
            transfer: RefCell::new(
                context.pinned_buf(
                    plan.tensors[Slot::Input as usize]
                        .bytes
                        .max(plan.tensors[Slot::Output as usize].bytes),
                )?,
            ),
        })
    }

    pub(super) fn with_q8_activations(mut self, enabled: bool) -> Self {
        self.q8_activations = enabled;
        self
    }

    pub(super) fn project(
        &self,
        context: &CudaContext,
        weight: &Resident,
        x: &[f32],
    ) -> Result<Vec<f32>, EngError> {
        self.project_buffer(context, &weight.codes, weight.shape, x)
    }

    pub(super) fn project_buffer(
        &self,
        context: &CudaContext,
        codes: &DeviceBuffer,
        shape: [usize; 2],
        x: &[f32],
    ) -> Result<Vec<f32>, EngError> {
        let output = self.range(Slot::Output, 0, shape[0])?;
        self.run(context, &self.stream, x, output, |input| {
            self.launch(context, (codes, shape, DenseDtype::Bf16), input, output)
        })
    }

    pub(super) fn project_gguf(
        &self,
        context: &CudaContext,
        stream: Option<&DeviceStream>,
        (dtype, weight, [rows, cols], scale): (DenseDtype, DeviceRange<'_>, [usize; 2], f32),
        x: &[f32],
    ) -> Result<Vec<f32>, EngError> {
        if cols != x.len() {
            return Err(bad("streamed expert projection input width differs"));
        }
        if scale == 0.0 {
            return Ok(vec![0.0; rows]);
        }
        let stream = stream.unwrap_or(&self.stream);
        let output = self.range(Slot::Output, 0, rows)?;
        let values = self.run(context, stream, x, output, |input| {
            self.enqueue_gguf_projection(
                context,
                stream,
                DenseGemvCall {
                    shape: DenseGemvShape::packed(1, rows, cols),
                    dtype,
                    w: weight,
                    x: input,
                    y: output,
                },
            )
        })?;
        scale_projection(values, scale)
    }

    fn enqueue_gguf_projection(
        &self,
        context: &CudaContext,
        stream: &DeviceStream,
        call: DenseGemvCall<'_>,
    ) -> Result<(), EngError> {
        let [rows, cols] = [call.shape.rows, call.shape.cols];
        if let Some(x_q8) = self.q8_scratch(call.dtype, call.w, [rows, cols]) {
            context.quantize_q8_1(
                stream,
                QuantizeCall {
                    x: call.x,
                    y: x_q8,
                    k: cols,
                    ncols: 1,
                },
            )?;
            context.matvec_q(
                stream,
                MatvecCall {
                    ty: QuantType::Q80,
                    w: call.w,
                    x_q8,
                    y: call.y,
                    k: cols,
                    rows,
                    ncols: 1,
                },
            )
        } else {
            context.dense_gemv(stream, call)
        }
    }

    pub(super) fn fits_gguf_rows(
        &self,
        dtype: DenseDtype,
        [rows, cols]: [usize; 2],
        count: usize,
    ) -> Result<bool, EngError> {
        // Preserve the original single-row Q8_1 activation path when requested.
        Ok(count >= 2 && count <= self.gguf_rows_capacity(dtype, [rows, cols])?)
    }

    pub(super) fn gguf_rows_capacity(
        &self,
        dtype: DenseDtype,
        [rows, cols]: [usize; 2],
    ) -> Result<usize, EngError> {
        if rows == 0 || cols == 0 || self.uses_q8_activations(dtype) {
            return Ok(0);
        }
        let need = dense_gemv_bytes(DenseGemvShape::packed(1, rows, cols), dtype)?;
        let bytes = need
            .x
            .checked_add(need.y)
            .ok_or_else(|| bad("GGUF expert batch size overflows"))?;
        let arena = self.arena.tensor(Slot::Output as usize)?.bytes();
        Ok((arena / bytes).min(DenseGemvShape::MAX_ACTIVATION_ROWS))
    }

    pub(super) fn project_gguf_rows(
        &self,
        context: &CudaContext,
        stream: Option<&DeviceStream>,
        (dtype, weight, [rows, cols], scale): (DenseDtype, DeviceRange<'_>, [usize; 2], f32),
        inputs: &[&[f32]],
    ) -> Result<Vec<f32>, EngError> {
        if inputs.iter().any(|row| row.len() != cols) {
            return Err(bad("streamed expert projection input width differs"));
        }
        if !self.fits_gguf_rows(dtype, [rows, cols], inputs.len())? {
            return Err(bad(
                "streamed expert batch exceeds the CUDA projection arena",
            ));
        }
        if scale == 0.0 {
            return Ok(vec![0.0; inputs.len() * rows]);
        }

        // During expert computation, both batched matrices fit in the idle logits slot.
        let input_elements = inputs.len() * cols;
        let input = self.range(Slot::Output, 0, input_elements)?;
        let output = self.range(Slot::Output, input_elements, inputs.len() * rows)?;
        let stream = stream.unwrap_or(&self.stream);
        let values = self.run_rows(context, stream, inputs, [input, output], || {
            context.dense_gemv(
                stream,
                DenseGemvCall {
                    shape: DenseGemvShape::packed(inputs.len(), rows, cols),
                    dtype,
                    w: weight,
                    x: input,
                    y: output,
                },
            )
        })?;
        scale_projection(values, scale)
    }

    /// Group resident projections of the same geometry into one upload and readback.
    pub(super) fn project_gguf_batches(
        &self,
        context: &CudaContext,
        (dtype, [rows, cols]): (DenseDtype, [usize; 2]),
        batches: &[ProjectionBatch<'_>],
        inputs: &[&[f32]],
    ) -> Result<Vec<f32>, EngError> {
        let count = inputs.len();
        if count == 0
            || count > self.gguf_rows_capacity(dtype, [rows, cols])?
            || inputs.iter().any(|row| row.len() != cols)
        {
            return Err(bad("bad resident GGUF projection tile"));
        }
        let output_start = count * cols;
        let input = self.range(Slot::Output, 0, output_start)?;
        let output = self.range(Slot::Output, output_start, count * rows)?;
        let mut calls = Vec::with_capacity(batches.len());
        let mut at = 0;
        for batch in batches {
            let n = batch.activations;
            if n == 0 || n > count - at {
                return Err(bad("resident GGUF projection groups exceed the tile"));
            }
            calls.push(DenseGemvCall {
                shape: DenseGemvShape::packed(n, rows, cols),
                dtype,
                w: batch.weight,
                x: self.range(Slot::Output, at * cols, n * cols)?,
                y: self.range(Slot::Output, output_start + at * rows, n * rows)?,
            });
            at += n;
        }
        if at != count {
            return Err(bad("resident GGUF projection groups do not cover the tile"));
        }
        let mut values = self.run_rows(context, &self.stream, inputs, [input, output], || {
            context.dense_gemv_group(&self.stream, &calls)
        })?;
        let mut at = 0;
        for batch in batches {
            let end = at + batch.activations * rows;
            scale_values(&mut values[at..end], batch.scale)?;
            at = end;
        }
        Ok(values)
    }

    fn q8_scratch(
        &self,
        dtype: DenseDtype,
        weight: DeviceRange<'_>,
        [rows, cols]: [usize; 2],
    ) -> Option<DeviceRange<'_>> {
        if !self.uses_q8_activations(dtype)
            || !weight.offset().is_multiple_of(4)
            || !cols.is_multiple_of(32)
            || !rows.is_multiple_of(ROWS_MULTIPLE)
        {
            return None;
        }
        let bytes = q8_1_bytes(cols, 1)?;
        // The head or controller sets the size of the input arena. Shorter
        // projections use the idle tail without changing the VRAM plan.
        self.range(Slot::Input, cols, bytes / size_of::<f32>()).ok()
    }

    pub(super) fn uses_q8_activations(&self, dtype: DenseDtype) -> bool {
        self.q8_activations && dtype == DenseDtype::PackedGguf(GgufType::Q80)
    }

    pub(super) fn project_group<const N: usize>(
        &self,
        context: &CudaContext,
        inputs: [&[f32]; 3],
        weights: [Option<ResidentProjection<'_>>; N],
    ) -> Result<[Vec<f32>; N], EngError> {
        let input_offsets = [0, inputs[0].len(), inputs[0].len() + inputs[1].len()];
        let input_rows = input_offsets[2] + inputs[2].len();
        // The large logits slot is idle during attention. Its non-overlapping
        // input and result ranges keep the group's staging within the same arena.
        let input = self.range(Slot::Output, 0, input_rows)?;
        let mut rows = 0;
        let mut calls = Vec::with_capacity(N);
        for projection in weights.iter().flatten() {
            let ResidentProjection {
                weight,
                dtype,
                input,
            } = *projection;
            let [out_rows, cols] = weight.shape;
            let x = self.range(Slot::Output, input_offsets[input], cols)?;
            let y = self.range(Slot::Output, input_rows + rows, out_rows)?;
            calls.push(self.call((&weight.codes, weight.shape, dtype), x, y)?);
            rows += out_rows;
        }
        let output = self.range(Slot::Output, input_rows, rows)?;
        let mut values = self
            .run_rows(context, &self.stream, &inputs, [input, output], || {
                context.dense_gemv_group(&self.stream, &calls)
            })?
            .into_iter();
        Ok(weights.map(|weight| {
            let rows = weight.map_or(0, |projection| projection.weight.shape[0]);
            values.by_ref().take(rows).collect()
        }))
    }

    pub(super) fn project_router(
        &self,
        context: &CudaContext,
        weight: &Resident,
        x: &[f32],
    ) -> Result<Vec<f32>, EngError> {
        let output = self.range(Slot::Output, 0, weight.shape[0])?;
        self.run(context, &self.stream, x, output, |input| {
            self.launch_router(context, weight, 1, [input, output])
        })
    }

    pub(super) fn project_router_rows(
        &self,
        context: &CudaContext,
        weight: &Resident,
        inputs: &[&[f32]],
    ) -> Result<Vec<f32>, EngError> {
        let [rows, cols] = weight.shape;
        if inputs.iter().any(|input| input.len() != cols) {
            return Err(bad("CUDA router input width differs"));
        }
        let slot = self.arena.tensor(Slot::Output as usize)?;
        // The logits slot is idle while routing. Its capacity bounds both
        // matrices and the existing pinned transfer buffer.
        let capacity = (slot.bytes() / float_bytes(rows + cols)?).clamp(1, ROUTER_MAX_ROWS);
        let mut logits = Vec::with_capacity(inputs.len() * rows);
        for tile in inputs.chunks(capacity) {
            if let [only] = tile {
                logits.extend(self.project_router(context, weight, only)?);
                continue;
            }
            let input_elements = tile.len() * cols;
            let input = self.range(Slot::Output, 0, input_elements)?;
            let output = self.range(Slot::Output, input_elements, tile.len() * rows)?;
            logits.extend(
                self.run_rows(context, &self.stream, tile, [input, output], || {
                    self.launch_router(context, weight, tile.len(), [input, output])
                })?,
            );
        }
        Ok(logits)
    }

    fn launch_router(
        &self,
        context: &CudaContext,
        weight: &Resident,
        count: usize,
        [input, output]: [DeviceRange<'_>; 2],
    ) -> Result<(), EngError> {
        let [rows, cols] = weight.shape;
        context.router_logits_rows(
            &self.stream,
            RouterLogitsCall {
                shape: RouterShape {
                    m: count,
                    hidden: cols,
                    n_expert: rows,
                    top_k: 1,
                },
                x: input,
                w: weight.codes.range(0, weight.codes.bytes())?,
                logits: output,
            },
        )
    }

    pub(super) fn controller(
        &self,
        context: &CudaContext,
        [(down, down_type), (gain, gain_type), (forget, forget_type)]: ControllerWeights<'_>,
        x: &[f32],
    ) -> Result<[Vec<f32>; 2], EngError> {
        let rank = down.shape[0];
        let width = gain.shape[0];
        let reduced = self.range(Slot::Reduced, 0, rank)?;
        let all = self.range(Slot::Output, 0, width + forget.shape[0])?;
        let first = self.range(Slot::Output, 0, width)?;
        let second = self.range(Slot::Output, width, forget.shape[0])?;
        let mut values = self.run(context, &self.stream, x, all, |input| {
            self.launch(
                context,
                (&down.codes, down.shape, down_type),
                input,
                reduced,
            )?;
            self.launch(
                context,
                (&gain.codes, gain.shape, gain_type),
                reduced,
                first,
            )?;
            self.launch(
                context,
                (&forget.codes, forget.shape, forget_type),
                reduced,
                second,
            )
        })?;
        let forget = values.split_off(width);
        Ok([values, forget])
    }

    pub(super) fn controller_capacity(&self, hidden: usize) -> Result<usize, EngError> {
        let output = self.arena.tensor(Slot::Output as usize)?;
        let per_row = float_bytes(
            hidden
                .checked_mul(4)
                .ok_or_else(|| bad("CUDA grouped controller row size overflows"))?,
        )?;
        Ok(output.bytes().checked_div(per_row).unwrap_or(0))
    }

    pub(super) fn controller_rows(
        &self,
        context: &CudaContext,
        rows: &[ControllerRow<'_>],
        [rank, hidden]: [usize; 2],
    ) -> Result<Vec<[Vec<f32>; 2]>, EngError> {
        let capacity = self.controller_capacity(hidden)?;
        if capacity < 2 || rows.iter().any(|row| row.input.len() != 2 * hidden) {
            return Err(bad(
                "CUDA grouped controller rows exceed their workspace or shape",
            ));
        }
        let reduced = self.range(Slot::Reduced, 0, rank)?;
        let mut results = Vec::with_capacity(rows.len());
        for tile in rows.chunks(capacity) {
            let words = tile.len() * 2 * hidden;
            let input = self.range(Slot::Output, 0, words)?;
            let output = self.range(Slot::Output, words, words)?;
            let mut calls = Vec::with_capacity(tile.len() * 3);
            let mut inputs = Vec::with_capacity(tile.len());
            for (index, row) in tile.iter().enumerate() {
                let offset = index * 2 * hidden;
                let gate = self.range(Slot::Output, offset, 2 * hidden)?;
                let gain = self.range(Slot::Output, words + offset, hidden)?;
                let forget = self.range(Slot::Output, words + offset + hidden, hidden)?;
                // Each triple finishes reading the shared reduction before the next down GEMV.
                for ((weight, dtype), (x, y)) in row.weights.into_iter().zip([
                    (gate, reduced),
                    (reduced, gain),
                    (reduced, forget),
                ]) {
                    calls.push(self.call((&weight.codes, weight.shape, dtype), x, y)?);
                }
                inputs.push(row.input);
            }
            let values = self.run_rows(context, &self.stream, &inputs, [input, output], || {
                context.dense_gemv_group(&self.stream, &calls)
            })?;
            results.extend(values.chunks_exact(2 * hidden).map(|row| {
                let (gain, forget) = row.split_at(hidden);
                [gain.to_vec(), forget.to_vec()]
            }));
        }
        Ok(results)
    }

    fn launch(
        &self,
        context: &CudaContext,
        weight: (&DeviceBuffer, [usize; 2], DenseDtype),
        x: DeviceRange<'_>,
        y: DeviceRange<'_>,
    ) -> Result<(), EngError> {
        context.dense_gemv(&self.stream, self.call(weight, x, y)?)
    }

    fn call<'a>(
        &self,
        (codes, [rows, cols], dtype): (&'a DeviceBuffer, [usize; 2], DenseDtype),
        x: DeviceRange<'a>,
        y: DeviceRange<'a>,
    ) -> Result<DenseGemvCall<'a>, EngError> {
        let shape = DenseGemvShape::packed(1, rows, cols);
        let need = dense_gemv_bytes(shape, dtype)?;
        Ok(DenseGemvCall {
            shape,
            dtype,
            w: codes.range(0, need.w)?,
            x,
            y,
        })
    }

    pub(super) fn run_layer(
        &self,
        context: &CudaContext,
        stream: Option<&DeviceStream>,
        x: &[f32],
        enqueue: impl FnOnce(&DeviceStream, DeviceRange<'_>, DeviceRange<'_>) -> Result<(), EngError>,
    ) -> Result<Vec<f32>, EngError> {
        let stream = stream.unwrap_or(&self.stream);
        let output = self.range(Slot::Output, 0, x.len())?;
        self.run(context, stream, x, output, |input| {
            enqueue(stream, input, output)
        })
    }

    pub(super) fn run_routed_layer(
        &self,
        context: &CudaContext,
        x: &[f32],
        bias: &[f32],
        enqueue: impl FnOnce(&DeviceStream, [DeviceRange<'_>; 4]) -> Result<(), EngError>,
    ) -> Result<Vec<f32>, EngError> {
        let full_input = self.range(Slot::Input, 0, x.len() + bias.len() + 1)?;
        let input = self.range(Slot::Input, 0, x.len())?;
        let bias_range = self.range(Slot::Input, x.len(), bias.len())?;
        let status = self.range(Slot::Input, x.len() + bias.len(), 1)?;
        let output = self.range(Slot::Output, 0, x.len())?;
        self.run_rows(
            context,
            &self.stream,
            &[x, bias, &[0.0]],
            [full_input, output],
            || enqueue(&self.stream, [input, bias_range, status, output]),
        )
    }

    fn run(
        &self,
        context: &CudaContext,
        stream: &DeviceStream,
        x: &[f32],
        output: DeviceRange<'_>,
        enqueue: impl FnOnce(DeviceRange<'_>) -> Result<(), EngError>,
    ) -> Result<Vec<f32>, EngError> {
        let input = self.range(Slot::Input, 0, x.len())?;
        self.run_rows(context, stream, &[x], [input, output], || enqueue(input))
    }

    fn run_rows(
        &self,
        context: &CudaContext,
        stream: &DeviceStream,
        inputs: &[&[f32]],
        [input, output]: [DeviceRange<'_>; 2],
        enqueue: impl FnOnce() -> Result<(), EngError>,
    ) -> Result<Vec<f32>, EngError> {
        let mut bytes = Vec::with_capacity(input.bytes() / size_of::<f32>());
        for row in inputs {
            if !row.iter().all(|value| value.is_finite()) {
                return Err(bad("CUDA projection requires finite input"));
            }
            bytes.extend(row.iter().map(|&value| value.to_ne_bytes()));
        }
        let mut transfer = self.transfer.borrow_mut();
        transfer.write(0, bytes.as_flattened())?;
        context.copy_from_pinned(stream, &mut transfer, 0, input)?;
        let result =
            enqueue().and_then(|()| context.copy_to_pinned(stream, output, &mut transfer, 0));
        // Even a refused launch may follow an accepted input copy.
        let waited = transfer.synchronize(stream);
        result?;
        waited?;

        let mut bytes = vec![[0; size_of::<f32>()]; output.bytes() / size_of::<f32>()];
        transfer.read(0, bytes.as_flattened_mut())?;
        let values: Vec<_> = bytes.into_iter().map(f32::from_ne_bytes).collect();
        if !values.iter().all(|value| value.is_finite()) {
            return Err(bad("CUDA projection produced a nonfinite output"));
        }
        Ok(values)
    }

    fn range(&self, slot: Slot, skip: usize, len: usize) -> Result<DeviceRange<'_>, EngError> {
        let slot = self.arena.tensor(slot as usize)?;
        let skip = float_bytes(skip)?;
        let bytes = float_bytes(len)?;
        if skip.checked_add(bytes).is_none_or(|end| end > slot.bytes()) {
            return Err(bad("CUDA BF16 projection exceeds its arena slot"));
        }
        slot.buffer().range(slot.offset() + skip, bytes)
    }
}

fn scale_projection(mut values: Vec<f32>, scale: f32) -> Result<Vec<f32>, EngError> {
    scale_values(&mut values, scale)?;
    Ok(values)
}

fn scale_values(values: &mut [f32], scale: f32) -> Result<(), EngError> {
    if scale != 1.0 {
        for value in values {
            *value *= scale;
            if !value.is_finite() {
                return Err(bad("GGUF expert scale overflowed"));
            }
        }
    }
    Ok(())
}

fn float_bytes(len: usize) -> Result<usize, EngError> {
    len.checked_mul(size_of::<f32>())
        .ok_or_else(|| bad("CUDA BF16 projection byte count overflows"))
}

#[cfg(test)]
mod controller_tests {
    use super::*;
    use crate::{family::inkling2::cuda_projection::upload_tensor, kernels::cuda::device_present};

    #[test]
    fn grouped_controller_tiles_match_scalar_bits() {
        if !device_present("Inkling grouped controller tiles") {
            return;
        }
        let context = CudaContext::new().unwrap();
        let [rank, hidden] = [32, 32];
        let plan = DensePlan::new([8 * hidden, 2 * hidden], hidden, rank).unwrap();
        let step = DenseStep::new(&context, &plan).unwrap();
        assert_eq!(step.controller_capacity(hidden).unwrap(), 2);
        let bank = |seed: usize| {
            [
                ([rank, 2 * hidden], seed),
                ([hidden, rank], seed + 1),
                ([hidden, rank], seed + 2),
            ]
            .map(|(shape, seed)| Resident {
                codes: upload_tensor(
                    &context,
                    &(0..shape[0] * shape[1])
                        .flat_map(|i| {
                            ((i.wrapping_mul(seed) % 73) as f32 * 0.003 - 0.06).to_ne_bytes()
                        })
                        .collect::<Vec<_>>(),
                )
                .unwrap(),
                shape,
            })
        };
        let banks = [bank(19), bank(23)];
        let inputs: Vec<Vec<f32>> = (0..5)
            .map(|row| {
                (0..2 * hidden)
                    .map(|i| ((row * 13 + i * 5) % 113) as f32 * -0.012 + 0.3)
                    .collect()
            })
            .collect();
        let rows: Vec<_> = inputs
            .iter()
            .enumerate()
            .map(|(index, input)| ControllerRow {
                weights: banks[index % 2]
                    .each_ref()
                    .map(|weight| (weight, DenseDtype::F32)),
                input,
            })
            .collect();
        let expected: Vec<_> = rows
            .iter()
            .map(|row| step.controller(&context, row.weights, row.input).unwrap())
            .collect();
        assert!(expected.iter().all(|[gain, forget]| gain != forget));
        let actual = step
            .controller_rows(&context, &rows, [rank, hidden])
            .unwrap();
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().flatten().zip(expected.iter().flatten()) {
            assert!(
                actual
                    .iter()
                    .map(|v| v.to_bits())
                    .eq(expected.iter().map(|v| v.to_bits()))
            );
        }
    }
}
