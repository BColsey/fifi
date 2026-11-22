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
            router::{RouterLogitsCall, RouterShape},
        },
    },
    weights::GgufType,
};

use super::{Resident, bad};

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
        let q8 = self.q8_scratch(dtype, weight, [rows, cols]);
        let mut values = self.run(context, stream, x, output, |input| {
            if let Some(x_q8) = q8 {
                context.quantize_q8_1(
                    stream,
                    QuantizeCall {
                        x: input,
                        y: x_q8,
                        k: cols,
                        ncols: 1,
                    },
                )?;
                return context.matvec_q(
                    stream,
                    MatvecCall {
                        ty: QuantType::Q80,
                        w: weight,
                        x_q8,
                        y: output,
                        k: cols,
                        rows,
                        ncols: 1,
                    },
                );
            }
            context.dense_gemv(
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
        if scale != 1.0 {
            for value in &mut values {
                *value *= scale;
                if !value.is_finite() {
                    return Err(bad("GGUF expert scale overflowed"));
                }
            }
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
        let [rows, cols] = weight.shape;
        let output = self.range(Slot::Output, 0, rows)?;
        self.run(context, &self.stream, x, output, |input| {
            context.router_logits_rows(
                &self.stream,
                RouterLogitsCall {
                    shape: RouterShape {
                        m: 1,
                        hidden: cols,
                        n_expert: rows,
                        top_k: 1,
                    },
                    x: input,
                    w: weight.codes.range(0, weight.codes.bytes())?,
                    logits: output,
                },
            )
        })
    }

    pub(super) fn controller(
        &self,
        context: &CudaContext,
        [(down, down_type), (gain, gain_type), (forget, forget_type)]: [(&Resident, DenseDtype); 3],
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

fn float_bytes(len: usize) -> Result<usize, EngError> {
    len.checked_mul(size_of::<f32>())
        .ok_or_else(|| bad("CUDA BF16 projection byte count overflows"))
}
