//! CUDA projections for the short-context text backend.
//!
//! BF16 projections reside on the device. The selected experts reuse four
//! packed staging pairs. Routing, attention, and each visit's state stay on
//! the host so the projection changes can be compared with the CPU oracle.

use std::collections::BTreeMap;

use super::{
    TextSpec, bad,
    cpu_weights::{ExpertMatrix, ExpertPacking, Weights},
};
use crate::{
    EngError,
    kernels::cuda::{
        Admission, CHUNK_LIMIT, CudaContext, DeviceBuffer,
        plan::{Demand, admit},
    },
};

const DEVICE_HEADROOM: u64 = 256 * 1024 * 1024;

pub(super) struct DenseUpload<'a> {
    pub(super) name: &'a str,
    pub(super) codes: &'a [u8],
    pub(super) shape: [usize; 2],
}

#[derive(Clone, Copy)]
struct Stage {
    packing: ExpertPacking,
    shape: [usize; 2],
    lengths: [usize; 2],
}

struct Plan<'a> {
    dense: Vec<DenseUpload<'a>>,
    stages: Vec<Stage>,
    demand: Demand,
    admission: Admission,
    workspace_bytes: u64,
}

impl<'a> Plan<'a> {
    fn new(weights: &'a Weights, spec: &TextSpec) -> Result<Self, EngError> {
        let dense = weights.dense_uploads(spec)?;
        let stages = stages(spec)?;
        let overflow = || EngError::MemoryPlan("Inkling CUDA projection plan overflows".into());
        let tensors = dense
            .iter()
            .try_fold(0u64, |sum, upload| {
                sum.checked_add(upload.codes.len() as u64)
            })
            .ok_or_else(overflow)?;
        let work = stages
            .iter()
            .flat_map(|stage| stage.lengths)
            .try_fold(0u64, |sum, bytes| sum.checked_add(bytes as u64))
            .ok_or_else(overflow)?;
        let shape = dense
            .iter()
            .map(|upload| upload.shape)
            .chain(stages.iter().map(|stage| stage.shape))
            .fold([0, 0], |[rows, cols], [r, c]| [rows.max(r), cols.max(c)]);
        let (admission, workspace_bytes) = CudaContext::fixed_projection_plan(shape)?;
        let demand = Demand {
            tensors,
            work,
            kv_cache: 0,
            headroom: DEVICE_HEADROOM,
        };
        Ok(Self {
            dense,
            stages,
            demand,
            admission,
            workspace_bytes,
        })
    }

    fn receipt(&self) -> Result<serde_json::Value, EngError> {
        let with_workspace = Demand {
            work: self
                .demand
                .work
                .checked_add(self.workspace_bytes)
                .ok_or_else(|| {
                    EngError::MemoryPlan("Inkling CUDA workspace byte count overflows".into())
                })?,
            ..self.demand
        };
        let budget = admit(&with_workspace, u64::MAX)?;
        Ok(serde_json::json!({
            "resident_projection_count": self.dense.len(),
            "resident_bf16_bytes": self.demand.tensors,
            "packed_expert_staging_bytes": self.demand.work,
            "workspace_bound_bytes": self.workspace_bytes,
            "device_kv_bytes": self.demand.kv_cache,
            "headroom_bytes": self.demand.headroom,
            "device_budget_bytes": budget,
        }))
    }
}

fn stages(spec: &TextSpec) -> Result<Vec<Stage>, EngError> {
    let hidden = spec.hidden_size as usize;
    let intermediate = spec.intermediate_size as usize;
    let gate_up = intermediate
        .checked_mul(2)
        .ok_or_else(|| bad("expert rows overflow"))?;
    [ExpertPacking::Nvfp4, ExpertPacking::Mxfp8]
        .into_iter()
        .flat_map(|packing| {
            [[gate_up, hidden], [hidden, intermediate]]
                .into_iter()
                .map(move |shape| (packing, shape))
        })
        .map(|(packing, shape)| {
            let lengths = packing
                .lengths(shape)
                .ok_or_else(|| bad("invalid CUDA expert shape"))?;
            Ok(Stage {
                packing,
                shape,
                lengths,
            })
        })
        .collect()
}

struct Resident {
    codes: DeviceBuffer,
    shape: [usize; 2],
}

struct Staging {
    layout: Stage,
    codes: DeviceBuffer,
    scales: DeviceBuffer,
}

pub(super) struct CudaProjection {
    dense: BTreeMap<String, Resident>,
    stages: Vec<Staging>,
    context: CudaContext,
}

impl CudaProjection {
    pub(super) fn load(weights: &Weights, spec: &TextSpec) -> Result<Self, EngError> {
        let plan = Plan::new(weights, spec)?;
        eprintln!("Inkling CUDA projections {}", plan.receipt()?);
        let context = CudaContext::with_admission(&plan.admission)?;
        // The fixed workspace is already charged to cudaMemGetInfo.
        admit(&plan.demand, context.memory_info()?.free as u64)?;
        let dense = plan
            .dense
            .into_iter()
            .map(|item| {
                let codes = context.allocate(item.codes.len())?;
                for (index, chunk) in item.codes.chunks(CHUNK_LIMIT).enumerate() {
                    context.upload_at(&codes, index * CHUNK_LIMIT, chunk)?;
                }
                Ok((
                    item.name.into(),
                    Resident {
                        codes,
                        shape: item.shape,
                    },
                ))
            })
            .collect::<Result<_, EngError>>()?;
        let stages = plan
            .stages
            .into_iter()
            .map(|layout| {
                Ok(Staging {
                    layout,
                    codes: context.allocate(layout.lengths[0])?,
                    scales: context.allocate(layout.lengths[1])?,
                })
            })
            .collect::<Result<_, EngError>>()?;
        Ok(Self {
            dense,
            stages,
            context,
        })
    }

    pub(super) fn project(
        &self,
        name: &str,
        shape: [usize; 2],
        x: &[f32],
    ) -> Result<Vec<f32>, EngError> {
        let projection = self
            .dense
            .get(name)
            .filter(|projection| projection.shape == shape)
            .ok_or_else(|| {
                bad(format!(
                    "no resident BF16 projection with matching shape for {name}"
                ))
            })?;
        self.context.bf16_matvec(&projection.codes, shape, x)
    }

    pub(super) fn expert(
        &self,
        weights: &ExpertMatrix<'_>,
        x: &[f32],
    ) -> Result<Vec<f32>, EngError> {
        let stage = self
            .stages
            .iter()
            .find(|stage| {
                stage.layout.shape == weights.shape && stage.layout.packing == weights.packing
            })
            .ok_or_else(|| bad("no matching CUDA expert staging pair"))?;
        self.context.upload_at(&stage.codes, 0, weights.codes)?;
        self.context.upload_at(&stage.scales, 0, weights.scales)?;
        match weights.packing {
            ExpertPacking::Nvfp4 => self.context.nvfp4_matvec(
                &stage.codes,
                &stage.scales,
                weights.outer,
                weights.shape,
                x,
            ),
            ExpertPacking::Mxfp8 => {
                self.context
                    .mxfp8_matvec(&stage.codes, &stage.scales, weights.shape, x)
            }
        }
    }
}

#[test]
#[ignore = "requires the native Mini checkpoint, runs without a CUDA device"]
fn native_cuda_projection_plan() {
    use std::{env, path::Path};

    let dir = env::var_os("FIFI_INKLING_MODEL_DIR").expect("set FIFI_INKLING_MODEL_DIR");
    let path = Path::new(&dir);
    let spec = TextSpec::from_config(&std::fs::read(path.join("config.json")).unwrap()).unwrap();
    let weights = Weights::open(path, &spec).unwrap();
    let plan = Plan::new(&weights, &spec).unwrap();
    assert!(
        plan.dense
            .iter()
            .any(|tensor| tensor.name == "model.llm.unembed.weight")
    );
    assert!(
        !plan
            .dense
            .iter()
            .any(|tensor| tensor.name == "model.llm.embed.weight")
    );
    for layer in 0..spec.stored_layers() {
        if spec.local_layer_ids.binary_search(&layer).is_ok() {
            continue;
        }
        for plane in ["k", "v"] {
            let name = format!("model.llm.layers.{layer}.attn.{plane}_sconv.weight_proj");
            assert!(
                plan.dense.iter().any(|tensor| tensor.name == name),
                "missing dynamic convolution projection {name}"
            );
        }
    }
    assert_eq!(plan.stages.len(), 4);
    assert!(plan.demand.tensors < 3 * 1024 * 1024 * 1024);
    assert!(plan.demand.work < 16 * 1024 * 1024);
    println!("{}", plan.receipt().unwrap());
}

#[test]
#[ignore = "requires a leased CUDA device"]
fn packed_mxfp8_cuda_matches_cpu() {
    use super::cpu_weights::mxfp8_dot;

    let context = CudaContext::new().unwrap();
    let mut codes = [0x38; 128];
    codes[32..64].fill(0xb8);
    let scales = [127, 126, 0, 127];
    let mut x = [1.0; 64];
    x[32] = 0.5;
    let device_codes = context.upload(&codes).unwrap();
    let device_scales = context.upload(&scales).unwrap();
    let actual = context
        .mxfp8_matvec(&device_codes, &device_scales, [2, 64], &x)
        .unwrap();
    for (row, got) in actual.into_iter().enumerate() {
        let want = mxfp8_dot(&codes[row * 64..][..64], &scales[row * 2..][..2], &x).unwrap();
        assert_eq!(got.to_bits(), want.to_bits());
    }
    let subnormal = context.upload(&[0, 0]).unwrap();
    let want = mxfp8_dot(&codes[..64], &[0, 0], &x).unwrap();
    let subnormal_codes = context.upload(&codes[..64]).unwrap();
    let got = context
        .mxfp8_matvec(&subnormal_codes, &subnormal, [1, 64], &x)
        .unwrap();
    assert_eq!(got[0].to_bits(), want.to_bits());
    context.upload_at(&device_scales, 0, &[255]).unwrap();
    assert!(matches!(
        context.mxfp8_matvec(&device_codes, &device_scales, [2, 64], &x),
        Err(EngError::BadRequest(_))
    ));
    context.upload_at(&device_scales, 0, &scales).unwrap();
    context.upload_at(&device_codes, 1, &[0x7f]).unwrap();
    assert!(matches!(
        context.mxfp8_matvec(&device_codes, &device_scales, [2, 64], &x),
        Err(EngError::BadRequest(_))
    ));
    assert!(matches!(
        context.mxfp8_matvec(&device_codes, &device_scales, [2, 63], &x),
        Err(EngError::BadRequest(_))
    ));
}
