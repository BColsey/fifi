//! Import one selected expert set from bounded host pages and project in its stored format.

use super::{InklingExpertPlan, InklingExpertProjection};
use crate::{
    EngError,
    kernels::{
        BoundedIndices, HostPages, MlxKernels, PackedPages, PageRange, QuantizationMode,
        QuantizationSpec, ResidentPackedWeight,
    },
    tensor::{Dtype, ExecutionTensor},
};

/// The selected experts for both projections of one physical layer.
pub struct MlxExpertStage {
    gate_up: Projection,
    down: Projection,
}

impl MlxExpertStage {
    /// Reads directly into reusable pages and imports both packed projections together.
    /// Create the plan with `HostPages::page_size()` alignment. Use F32 inputs for
    /// scaled NVFP4 projections. The caller maps routes with the plan's compact slots.
    /// Keep the page pins until all consuming graphs have finished. Drop the stage,
    /// release the pins, and synchronize before reusing the pages.
    pub fn read(
        kernels: &mut MlxKernels,
        plan: &InklingExpertPlan<'_>,
        pages: &mut HostPages,
    ) -> Result<Self, EngError> {
        kernels.check_ready()?;
        let page_size = HostPages::page_size();
        if page_size == 0 || !plan.alignment().is_multiple_of(page_size) {
            return Err(EngError::BadRequest(
                "Inkling MLX expert staging requires page-aligned tensor offsets".into(),
            ));
        }
        let projections = [plan.gate_up(), plan.down()];
        let stages = projections.map(packed_pages);
        let bytes = pages.try_write().ok_or_else(|| EngError::Backend {
            operation: "Inkling expert staging",
            message: "host pages are still in use by a previous stage".into(),
        })?;
        plan.read_into(bytes)?;
        let mut scale_ranges = Vec::new();
        for projection in projections {
            if let Some(range) = normalize_multipliers(bytes, projection)? {
                scale_ranges.push(range);
            }
        }
        let (weights, multipliers) = kernels.import_packed_pages(pages, &stages, &scale_ranges)?;
        let [gate_up, down] = weights.try_into().map_err(|_| EngError::Backend {
            operation: "Inkling expert staging",
            message: "native import returned a different packed tensor count".into(),
        })?;
        // The two projections of a layer share their format.
        let mut multipliers = multipliers.into_iter();
        Ok(Self {
            gate_up: Projection {
                weight: gate_up,
                multipliers: multipliers.next(),
            },
            down: Projection {
                weight: down,
                multipliers: multipliers.next(),
            },
        })
    }

    pub fn project_gate_up(
        &self,
        kernels: &mut MlxKernels,
        input: &ExecutionTensor,
        ids: &BoundedIndices,
    ) -> Result<ExecutionTensor, EngError> {
        self.gate_up.run(kernels, input, ids)
    }

    pub fn project_down(
        &self,
        kernels: &mut MlxKernels,
        input: &ExecutionTensor,
        ids: &BoundedIndices,
    ) -> Result<ExecutionTensor, EngError> {
        self.down.run(kernels, input, ids)
    }
}

struct Projection {
    weight: ResidentPackedWeight,
    multipliers: Option<ExecutionTensor>,
}

impl Projection {
    fn run(
        &self,
        kernels: &mut MlxKernels,
        input: &ExecutionTensor,
        ids: &BoundedIndices,
    ) -> Result<ExecutionTensor, EngError> {
        match &self.multipliers {
            Some(scales) => kernels.gather_nvfp4(input, &self.weight, ids, scales),
            None => kernels.gather_qmm_bounded(input, &self.weight, ids),
        }
    }
}

fn packed_pages(projection: &InklingExpertProjection) -> PackedPages {
    let [experts, rows, columns] = projection.shape.map(|dim| dim as u32);
    let (mode, bits, group_size) = if projection.amax.is_some() {
        (QuantizationMode::Nvfp4, 4, 16)
    } else {
        (QuantizationMode::Mxfp8, 8, 32)
    };
    PackedPages {
        payload: PageRange {
            offset: projection.weight.start,
            dtype: Dtype::U32,
            shape: vec![experts, rows, columns * bits / 32],
        },
        scales: PageRange {
            offset: projection.scale.start,
            dtype: Dtype::U8,
            shape: vec![experts, rows, columns / group_size],
        },
        biases: None,
        logical_shape: vec![experts, rows, columns],
        quantization: QuantizationSpec {
            mode,
            bits,
            group_size,
        },
    }
}

fn normalize_multipliers(
    bytes: &mut [u8],
    projection: &InklingExpertProjection,
) -> Result<Option<PageRange>, EngError> {
    let (Some(amax), Some(denominator)) = (&projection.amax, &projection.denominator) else {
        return Ok(None);
    };
    let invalid = || {
        EngError::Checkpoint(
            "Inkling NVFP4 outer scales must be finite and nonnegative with a positive denominator"
                .into(),
        )
    };
    let denominator = f32::from_le_bytes(bytes[denominator.clone()].try_into().unwrap());
    if !denominator.is_finite() || denominator <= 0.0 {
        return Err(invalid());
    }
    // The amax slot already has the required alignment and shape. Reuse it as
    // the native multiplier tensor without making another page or read.
    for encoded in bytes[amax.clone()].as_chunks_mut::<4>().0 {
        let amax = f32::from_le_bytes(*encoded);
        let scale = amax / denominator;
        if amax < 0.0 || !scale.is_finite() {
            return Err(invalid());
        }
        *encoded = scale.to_le_bytes();
    }
    Ok(Some(PageRange {
        offset: amax.start,
        dtype: Dtype::F32,
        shape: vec![projection.shape[0] as u32],
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{kernels::PackedWeight, tensor::Tensor};

    #[test]
    fn scales_reordered_nvfp4_routes_on_metal() {
        let _lock = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let mut codes = vec![0x22; 8 * 32];
        codes.extend(vec![0x44; 8 * 32]);
        let weight = PackedWeight::try_new(
            Tensor::try_from_host(Dtype::U32, vec![2, 8, 8], codes).unwrap(),
            Tensor::try_from_host(Dtype::U8, vec![2, 8, 4], vec![0x38; 64]).unwrap(),
            None,
            vec![2, 8, 64],
            QuantizationSpec {
                mode: QuantizationMode::Nvfp4,
                bits: 4,
                group_size: 16,
            },
        )
        .unwrap();
        let weight = kernels.import_packed(&weight).unwrap();
        let input = Tensor::try_from_host(
            Dtype::F32,
            vec![1, 1, 64],
            [1.0_f32; 64]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect(),
        )
        .unwrap();
        let ids = Tensor::try_from_host(
            Dtype::U32,
            vec![3],
            [1_u32, 0, 1]
                .into_iter()
                .flat_map(u32::to_le_bytes)
                .collect(),
        )
        .unwrap();
        let multipliers = Tensor::try_from_host(
            Dtype::F32,
            vec![2],
            [0.25_f32, 2.0]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect(),
        )
        .unwrap();
        let input = kernels.import_tensor(&input).unwrap();
        let ids = kernels.import_tensor(&ids).unwrap();
        let ids = kernels.bound_indices(&ids, 2).unwrap();
        let multipliers = kernels.import_tensor(&multipliers).unwrap();
        let y = kernels
            .gather_nvfp4(&input, &weight, &ids, &multipliers)
            .unwrap();
        let y = kernels.readback(&y).unwrap();
        assert_eq!(y.shape(), [3, 1, 8]);
        let (bytes, tail) = y.host_bytes().unwrap().as_chunks::<4>();
        assert!(tail.is_empty());
        let values: Vec<_> = bytes.iter().copied().map(f32::from_le_bytes).collect();
        assert_eq!(values, [[256.0; 8], [16.0; 8], [256.0; 8]].concat());
    }
}
