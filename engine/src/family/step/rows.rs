//! The activation rows of a chunk and the projections over them.
//!
//! A projection reads its input as q8_1 rows (the quantized matvec) and, when the producer wrote
//! them, as f32 rows (the batched matrix product). [`StepIo`] carries what every projection of a
//! step needs: the context, the stream, the resident weights by name and the scratch of the
//! batched product. The kernel follows the type of the weight tensor.

use crate::EngError;
use crate::family::project::cuda::{CudaWeight, CudaWeights};
use crate::kernels::cuda::prefill::mmq_dense::{self, MmqDenseCall, MmqDenseInputCall};
use crate::kernels::cuda::prefill::mmq_quant::MmqPacking;
use crate::kernels::cuda::prefill::mmq_size::q8_1_bytes as mmq_q8_bytes;
use crate::kernels::cuda::resident::add_rmsnorm_q8::AddNormQ8Call;
use crate::kernels::cuda::resident::block_ops::{BlockOp, RowOutputs};
use crate::kernels::cuda::resident::dense_gemv::{DenseDtype, DenseGemvCall, DenseGemvShape};
use crate::kernels::cuda::resident::quantize_q8_1::q8_1_bytes;
use crate::kernels::cuda::{CudaContext, DeviceBuffer, DeviceRange, DeviceStream, NativeWeight};

/// Bytes of `rows` q8_1 rows of `width` values.
pub(crate) fn q8(width: usize, rows: usize) -> usize {
    q8_1_bytes(width, rows).expect("a q8_1 row size fits usize")
}

/// The tensor `name` of a device load, resident in the blocks of its file type. `family` names
/// the model in the refusal.
pub(crate) fn native<'a>(
    weights: &'a CudaWeights,
    family: &str,
    name: &str,
) -> Result<&'a NativeWeight, EngError> {
    match weights.get(name) {
        Some(CudaWeight::Native { weight, .. }) => Ok(weight),
        _ => Err(EngError::Checkpoint(format!(
            "{family}: tensor {name} is not resident as native blocks"
        ))),
    }
}

/// The tensor `name` of a device load, resident as its raw bytes.
pub(crate) fn raw<'a>(
    weights: &'a CudaWeights,
    family: &str,
    name: &str,
) -> Result<&'a DeviceBuffer, EngError> {
    match weights.get(name) {
        Some(CudaWeight::Raw { buffer, .. }) => Ok(buffer),
        _ => Err(EngError::Checkpoint(format!(
            "{family}: tensor {name} is not resident as a raw tensor"
        ))),
    }
}

/// The activation rows a projection reads: q8_1 rows for the quantized matvec and, when the
/// producer wrote them, the f32 rows for the batched matrix product.
#[derive(Clone, Copy)]
pub(crate) struct Rows<'a> {
    pub(crate) q8: &'a DeviceBuffer,
    pub(crate) f32: Option<&'a DeviceBuffer>,
}

impl<'a> Rows<'a> {
    pub(crate) fn q8(q8: &'a DeviceBuffer) -> Self {
        Self { q8, f32: None }
    }

    pub(crate) fn both(q8: &'a DeviceBuffer, f32: &'a DeviceBuffer) -> Self {
        Self { q8, f32: Some(f32) }
    }
}

/// What the projections of one step share. `family` names the model in refusals.
#[derive(Clone, Copy)]
pub(crate) struct StepIo<'a> {
    pub(crate) family: &'static str,
    pub(crate) ctx: &'a CudaContext,
    pub(crate) stream: &'a DeviceStream,
    pub(crate) weights: &'a CudaWeights,
    /// Scratch of the batched product: the q8_1 rows of its input.
    pub(crate) mmq_q8: &'a DeviceBuffer,
    /// Scratch of the batched product: its fix-up words.
    pub(crate) fixup: &'a DeviceBuffer,
}

impl<'a> StepIo<'a> {
    pub(crate) fn run(&self, op: BlockOp<'_>) -> Result<(), EngError> {
        self.ctx.block_op(self.stream, op)
    }

    /// The tensor `name`, resident in the blocks of its file type.
    pub(crate) fn native(&self, name: &str) -> Result<&'a NativeWeight, EngError> {
        native(self.weights, self.family, name)
    }

    /// The tensor `name`, resident as its raw bytes.
    pub(crate) fn raw(&self, name: &str) -> Result<DeviceRange<'a>, EngError> {
        Self::whole(raw(self.weights, self.family, name)?)
    }

    pub(crate) fn whole<'b>(buffer: &'b DeviceBuffer) -> Result<DeviceRange<'b>, EngError> {
        buffer.range(0, buffer.bytes())
    }

    /// RMS norm of `n` rows of `d` values scaled by `weight`: the f32 rows into `y` and the
    /// q8_1 rows into `q8_out`, through the shared fused norm and quantize kernel.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn norm_rows(
        &self,
        n: usize,
        d: usize,
        eps: f32,
        x: &DeviceBuffer,
        weight: DeviceRange<'_>,
        y: &DeviceBuffer,
        q8_out: &DeviceBuffer,
    ) -> Result<(), EngError> {
        self.ctx.add_rmsnorm_q8(
            self.stream,
            AddNormQ8Call {
                n: d,
                eps,
                h: x.range(0, n * d * 4)?,
                delta: None,
                weight,
                norm_out: y.range(0, n * d * 4)?,
                q8: q8_out.range(0, q8(d, n))?,
                rows: n,
            },
        )
    }

    /// `y[t] = W x[t]` for `n` rows of the tensor `name`: [`Self::project`] on the resident
    /// weight.
    pub(crate) fn mv(
        &self,
        name: &str,
        x: Rows<'_>,
        y: &DeviceBuffer,
        n: usize,
    ) -> Result<(), EngError> {
        self.project(self.native(name)?, x, y, n)
    }

    /// `y[t] = W x[t]` for `n` rows. Up to 8 rows use the quantized matvec on the q8_1 rows;
    /// a larger chunk with f32 rows and a row length on the 256 grid uses the batched product.
    pub(crate) fn project(
        &self,
        weight: &NativeWeight,
        x: Rows<'_>,
        y: &DeviceBuffer,
        n: usize,
    ) -> Result<(), EngError> {
        let (k, rows) = (weight.k(), weight.n_rows());
        if n > 8
            && k.is_multiple_of(256)
            && mmq_dense::takes(weight.quant())
            && let Some(f32_rows) = x.f32
        {
            let q8_range = self.mmq_q8.range(0, mmq_q8_bytes(k, n)?)?;
            self.ctx.mmq_dense_input(
                self.stream,
                MmqDenseInputCall {
                    packing: MmqPacking::CpuReference,
                    weight,
                    m: n,
                    row_stride: k,
                    x: f32_rows.range(0, n * k * 4)?,
                    y: q8_range,
                },
            )?;
            return self.ctx.mmq_dense(
                self.stream,
                MmqDenseCall {
                    weight,
                    w_seg: None,
                    m: n,
                    q8_1: q8_range,
                    dst: y.range(0, n * rows * 4)?,
                    ld: rows,
                    col_off: 0,
                    fixup: Self::whole(self.fixup)?,
                },
            );
        }
        self.ctx.matvec_native(
            self.stream,
            weight,
            x.q8.range(0, q8(k, n))?,
            y.range(0, n * rows * 4)?,
            n,
        )
    }

    /// The F16 projection `y[t] = W x[t]` of `n` f32 rows.
    pub(crate) fn f16_mv(
        &self,
        name: &str,
        x: &DeviceBuffer,
        y: &DeviceBuffer,
        n: usize,
        rows: usize,
        cols: usize,
    ) -> Result<(), EngError> {
        self.ctx.dense_gemv(
            self.stream,
            DenseGemvCall {
                shape: DenseGemvShape::packed(n, rows, cols),
                dtype: DenseDtype::F16,
                w: self.raw(name)?,
                x: x.range(0, n * cols * 4)?,
                y: y.range(0, n * rows * 4)?,
            },
        )
    }

    /// Quantizes `n` rows of `width` f32 values (x * sigmoid(gate) when a gate is given) into
    /// q8_1 rows, and also writes the f32 product when `out` has f32 rows.
    pub(crate) fn quantize(
        &self,
        n: usize,
        width: usize,
        x: &DeviceBuffer,
        gate: Option<&DeviceBuffer>,
        out: Rows<'_>,
    ) -> Result<(), EngError> {
        self.run(BlockOp::gated_quant(
            n,
            width,
            x.range(0, n * width * 4)?,
            gate.map(|g| g.range(0, n * width * 4)).transpose()?,
            RowOutputs {
                q8: Some(out.q8.range(0, q8(width, n))?),
                f32: out.f32.map(|b| b.range(0, n * width * 4)).transpose()?,
            },
        )?)
    }
}
