//! The pack-reading source shared by the families: one reader over a stock EXL3
//! checkpoint that admits against the load's reserve, reads stored-component
//! matrices, and uploads norms, plain vectors and tables as the device weights
//! the steps look up. The Qwen3.6 mixers' geometry-checked loaders stay in
//! `qwen36::cuda::exl3_source` and read through this reader; Flash-Next reads
//! its trunk through the pack's recorded header walk directly.
//!
//! `DenseSpec` and its shape refusal live here because every family's dense
//! EXL3 upload checks the same role-rows-k contract.

use crate::family::exl3_pack::Root;
use crate::family::exl3_tensors::{effective_norm_bytes, f32_le_bytes, norm_layout, widen_f32};
use crate::{
    EngError,
    family::{
        gguf_model::Role,
        project::cuda::{CudaProjector, CudaWeight, Reserve},
    },
    tensor::Tensor,
    weights::{
        GgufType, WeightSource,
        exl3::{Axis, COMPONENT_SUFFIXES, Checkpoint, Matrix},
    },
};

/// A dense (tiled) mixer weight: its role, output rows and input length.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::family) struct DenseSpec {
    pub(in crate::family) role: Role,
    pub(in crate::family) rows: usize,
    pub(in crate::family) k: usize,
}

/// Refuses a dense weight whose shape is not the one the launch needs. `experts` is the stack
/// depth: a dense mixer weight has none.
pub(in crate::family) fn check_dense(
    name: &str,
    have: (usize, usize, Option<usize>),
    want: &DenseSpec,
) -> Result<(), EngError> {
    let (rows, k, experts) = have;
    if rows != want.rows || k != want.k || experts.is_some() {
        return Err(EngError::Checkpoint(format!(
            "mixer weights: {name} is {rows} rows of {k} (stack {experts:?}), the step needs {} rows of {} and no stack",
            want.rows, want.k
        )));
    }
    Ok(())
}

/// Largest stored matrix read at once: the head's trellis is the biggest, at 318 MB.
const MATRIX_READ_LIMIT: u64 = 1 << 30;

pub(in crate::family) struct PackReader<'a> {
    pub(in crate::family) projector: &'a CudaProjector,
    pub(in crate::family) checkpoint: &'a Checkpoint,
    reserve: Reserve,
    root: Root,
}

impl<'a> PackReader<'a> {
    pub(in crate::family) fn new(
        projector: &'a CudaProjector,
        checkpoint: &'a Checkpoint,
        reserve: Reserve,
    ) -> Self {
        Self {
            projector,
            checkpoint,
            reserve,
            root: Root::Layers,
        }
    }

    /// Reads the layers from `root` instead of the model's own: the MTP head's block.
    pub(in crate::family) fn under(mut self, root: Root) -> Self {
        self.root = root;
        self
    }

    /// Where the layers of this pack are named.
    pub(in crate::family) fn root(&self) -> Root {
        self.root
    }

    /// Admits `bytes` of weights against the free device memory and the reserve of the load.
    pub(in crate::family) fn admit(&self, bytes: u64) -> Result<(), EngError> {
        self.projector.admit(bytes, self.reserve)
    }

    /// The stored components of the matrix under `prefix`. Unsupported old representations are
    /// selected too, so `Matrix::load` can refuse them.
    pub(in crate::family) fn matrix(&self, prefix: &str) -> Result<Matrix, EngError> {
        let names = COMPONENT_SUFFIXES.map(|suffix| format!("{prefix}.{suffix}"));
        let keys = names.each_ref().map(String::as_str);
        let selected = self
            .checkpoint
            .source
            .read_selected(&keys, MATRIX_READ_LIMIT)?;
        Matrix::load(&selected, prefix, &self.checkpoint.config)
    }

    fn plain(&self, name: &str, shape: &[u32]) -> Result<Tensor, EngError> {
        let missing = || EngError::Checkpoint(format!("EXL3 source: missing tensor {name}"));
        let (dtype, stored) = self.checkpoint.source.stored(name).ok_or_else(missing)?;
        if stored != shape {
            return Err(EngError::Checkpoint(format!(
                "EXL3 source: {name} is stored as {stored:?}, expected {shape:?}"
            )));
        }
        let values = shape.iter().map(|&dim| u64::from(dim)).product::<u64>();
        let budget = values.saturating_mul(dtype.byte_width());
        let selected = self.checkpoint.source.read_selected(&[name], budget)?;
        selected.tensor(name).ok_or_else(missing)
    }

    pub(in crate::family) fn exl3(
        &self,
        name: String,
        prefix: &str,
        want: &DenseSpec,
        reorder: Option<(Axis, &[usize])>,
    ) -> Result<CudaWeight, EngError> {
        let matrix = self.matrix(prefix)?;
        let [k, n] = matrix.shape();
        check_dense(prefix, (n, k, None), want)?;
        let matrix = match reorder {
            Some((axis, order)) => matrix.reorder_groups(axis, order)?,
            None => matrix,
        };
        self.upload(name, &matrix)
    }

    /// Uploads `matrix` as it is stored (no geometry check, no reorder): the caller has checked
    /// its shape, as the Flash-Next trunk does through the pack's header walk.
    pub(in crate::family) fn upload(
        &self,
        name: String,
        matrix: &Matrix,
    ) -> Result<CudaWeight, EngError> {
        let weight = self.projector.upload_exl3(matrix, self.reserve)?;
        Ok(CudaWeight::Exl3 { name, weight })
    }

    pub(in crate::family) fn bank(
        &self,
        name: String,
        count: usize,
        member: impl FnMut(usize) -> Result<Matrix, EngError>,
    ) -> Result<CudaWeight, EngError> {
        let bank = self
            .projector
            .upload_exl3_bank(count, self.reserve, member)?;
        Ok(CudaWeight::Exl3Bank { name, bank })
    }

    fn raw_bytes(&self, name: String, bytes: &[u8]) -> Result<CudaWeight, EngError> {
        let budget = u64::try_from(bytes.len())
            .map_err(|_| EngError::MemoryPlan("EXL3 raw upload byte count overflow".into()))?;
        self.projector.admit(budget, self.reserve)?;
        let buffer = self.projector.context().upload(bytes)?;
        Ok(CudaWeight::Raw {
            name,
            ty: GgufType::F32,
            buffer,
        })
    }

    /// Uploads the zero-centered norm `source` as `name` with one added in F32.
    pub(in crate::family) fn norm(
        &self,
        name: String,
        source: &str,
        values: usize,
    ) -> Result<CudaWeight, EngError> {
        let (dtype, shape) = self.checkpoint.source.stored(source).ok_or_else(|| {
            EngError::Checkpoint(format!("EXL3 attention source: missing tensor {source}"))
        })?;
        // Validate the exact source shape and type before reading even a tiny norm.
        let output_len = norm_layout(dtype, &shape, values)?;
        let budget = u64::try_from(output_len)
            .map_err(|_| EngError::MemoryPlan("EXL3 norm read budget overflow".into()))?;
        let selected = self.checkpoint.source.read_selected(&[source], budget)?;
        let tensor = selected.tensor(source).ok_or_else(|| {
            EngError::Checkpoint(format!("EXL3 attention source: missing tensor {source}"))
        })?;
        self.raw_bytes(name, &effective_norm_bytes(&tensor, values)?)
    }

    pub(in crate::family) fn vector(
        &self,
        name: String,
        source: &str,
        shape: &[u32],
        transform: impl FnOnce(Vec<f32>) -> Result<Vec<f32>, EngError>,
    ) -> Result<CudaWeight, EngError> {
        let tensor = self.plain(source, shape)?;
        let values = shape.iter().map(|&dim| dim as usize).product();
        let values = transform(widen_f32(&tensor, values, source)?)?;
        self.raw_bytes(name, &f32_le_bytes(&values))
    }

    pub(in crate::family) fn table(
        &self,
        name: String,
        source: &str,
        shape: &[u32],
    ) -> Result<CudaWeight, EngError> {
        let tensor = self.plain(source, shape)?;
        let ty = match tensor.dtype() {
            crate::tensor::Dtype::Bf16 => GgufType::Bf16,
            crate::tensor::Dtype::F16 => GgufType::F16,
            other => {
                return Err(EngError::Checkpoint(format!(
                    "EXL3 source: table {source} has dtype {other:?}"
                )));
            }
        };
        let bytes = tensor
            .host_bytes()
            .ok_or_else(|| EngError::Checkpoint(format!("EXL3 table {source} needs host bytes")))?;
        let budget = u64::try_from(bytes.len())
            .map_err(|_| EngError::MemoryPlan("EXL3 table byte count overflow".into()))?;
        self.projector.admit(budget, self.reserve)?;
        let buffer = self.projector.context().upload(bytes)?;
        Ok(CudaWeight::Raw { name, ty, buffer })
    }
}
