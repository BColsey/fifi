//! Take BF16 matrices from the GGUF mapping and reuse Inkling's CUDA projection
//! kernels. Cold experts reuse fixed staging buffers.

use std::{cell::RefCell, collections::BTreeMap};

use super::{DEVICE_HEADROOM, DensePlan, DenseStep, DenseUpload, Resident, bad, upload_tensor};
use crate::{
    EngError,
    kernels::cuda::{
        CudaContext, DeviceBuffer,
        plan::{Demand, admit},
    },
    weights::{GgufType, repack_nvfp4_into},
};

pub(in crate::family::inkling2) struct GgufCuda {
    step: DenseStep,
    dense: BTreeMap<String, Resident>,
    expert_codes: DeviceBuffer,
    expert_scales: DeviceBuffer,
    packed: RefCell<Packed>,
    context: CudaContext,
}

struct Packed {
    codes: Vec<u8>,
    scales: Vec<u8>,
}

pub(in crate::family::inkling2) struct ExpertView<'a> {
    pub kind: GgufType,
    pub shape: [usize; 2],
    pub bytes: &'a [u8],
    pub factor: f32,
}

impl GgufCuda {
    pub(in crate::family::inkling2) fn load(
        dense: Vec<DenseUpload<'_>>,
        spec: &super::TextSpec,
    ) -> Result<Self, EngError> {
        let hidden = spec.hidden_size as usize;
        let ff = spec.intermediate_size as usize;
        let rows = hidden.max(2 * ff);
        let shape = dense.iter().fold([rows, hidden.max(ff)], |[r, c], item| {
            [r.max(item.shape[0]), c.max(item.shape[1])]
        });
        let elements = rows * hidden.max(ff);
        let stage = [2 * elements, elements / 16];
        let tensors: u64 = dense.iter().map(|item| item.codes.len() as u64).sum();
        let plan = DensePlan::new(shape, hidden, spec.core.gate_rank as usize)?;
        let (admission, _) = CudaContext::fixed_projection_plan(shape)?;
        let context = CudaContext::with_admission(&admission)?;
        let demand = Demand {
            tensors,
            work: (stage[0] + stage[1] + plan.device_bytes) as u64,
            kv_cache: 0,
            headroom: DEVICE_HEADROOM,
        };
        admit(&demand, context.memory_info()?.free as u64)?;
        let step = DenseStep::new(&context, &plan)?;
        let expert_codes = context.allocate(stage[0])?;
        let expert_scales = context.allocate(stage[1])?;
        let dense: BTreeMap<_, _> = dense
            .into_iter()
            .map(|item| {
                let codes = upload_tensor(&context, item.codes)?;
                Ok((
                    item.name.into(),
                    Resident {
                        codes,
                        shape: item.shape,
                    },
                ))
            })
            .collect::<Result<_, EngError>>()?;
        eprintln!(
            "Inkling GGUF CUDA projections {}",
            serde_json::json!({"resident_projection_count": dense.len(), "resident_bf16_bytes": tensors,
                "packed_expert_staging_bytes": stage[0] + stage[1], "dense_stream_arena_bytes": plan.device_bytes})
        );
        Ok(Self {
            step,
            dense,
            expert_codes,
            expert_scales,
            packed: RefCell::new(Packed {
                codes: vec![0; elements / 2],
                scales: vec![0; elements / 16],
            }),
            context,
        })
    }

    pub(in crate::family::inkling2) fn project(
        &self,
        name: &str,
        rows: usize,
        x: &[f32],
    ) -> Option<Result<Vec<f32>, EngError>> {
        let resident = self.dense.get(name)?;
        if resident.shape != [rows, x.len()] {
            return Some(Err(bad(format!(
                "bad GGUF CUDA projection shape for {name}"
            ))));
        }
        Some(self.step.project(&self.context, resident, x))
    }

    pub(in crate::family::inkling2) fn expert(
        &self,
        expert: &ExpertView<'_>,
        x: &[f32],
    ) -> Option<Result<Vec<f32>, EngError>> {
        match expert.kind {
            GgufType::Bf16 => Some(self.bf16_expert(expert, x)),
            GgufType::Nvfp4 => {
                // The GGUF unsigned scale 0xff has no E4M3FN representation.
                if expert
                    .bytes
                    .as_chunks::<36>()
                    .0
                    .iter()
                    .any(|b| b[..4].contains(&0xff))
                {
                    return None;
                }
                Some(self.nvfp4_expert(expert, x))
            }
            _ => None,
        }
    }

    fn bf16_expert(&self, expert: &ExpertView<'_>, x: &[f32]) -> Result<Vec<f32>, EngError> {
        self.context
            .upload_at(&self.expert_codes, 0, expert.bytes)?;
        let mut output =
            self.step
                .project_buffer(&self.context, &self.expert_codes, expert.shape, x)?;
        if expert.factor != 1.0 {
            for value in &mut output {
                *value *= expert.factor;
            }
        }
        Ok(output)
    }

    fn nvfp4_expert(&self, expert: &ExpertView<'_>, x: &[f32]) -> Result<Vec<f32>, EngError> {
        if expert.factor == 0.0 {
            return Ok(vec![0.0; expert.shape[0]]);
        }
        let n = expert.shape[0] * expert.shape[1];
        let Packed { codes, scales } = &mut *self.packed.borrow_mut();
        let (codes, scales) = (&mut codes[..n / 2], &mut scales[..n / 16]);
        repack_nvfp4_into(expert.bytes, codes, scales)?;
        self.context.upload_at(&self.expert_codes, 0, codes)?;
        self.context.upload_at(&self.expert_scales, 0, scales)?;
        self.context.nvfp4_matvec_range(
            self.expert_codes.range(0, codes.len())?,
            self.expert_scales.range(0, scales.len())?,
            expert.factor,
            expert.shape,
            x,
        )
    }
}
