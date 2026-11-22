//! Project directly from mapped GGUF matrices. Cold experts reuse fixed staging buffers.

use std::{cell::RefCell, collections::BTreeMap};

use super::{
    DEVICE_HEADROOM, DensePlan, DenseStep, DenseUpload, Resident, bad,
    dense::{ControllerRow, ControllerWeights, DraftHead, ResidentProjection},
    upload_tensor,
};
use crate::family::inkling2::cpu_weights::Projection;
use crate::{
    EngError,
    expert_tier::{ExpertId, ExpertPayloadSource, Placement, VRAM_HEADROOM_BYTES},
    family::mapped_file::MappedFile,
    kernels::cuda::{
        CudaContext, DeviceBuffer, DeviceRange, DeviceStream,
        plan::{Demand, admit},
        resident::dense_gemv::{
            DenseDtype, GgufMoeCall, GgufMoePicks, GgufMoeShape, GgufMoeWeights,
        },
    },
    sched::job::PrefillBankReport,
    weights::{GgufType, repack_nvfp4_into},
};

pub(in crate::family::inkling2) mod attention;
mod cache;
mod device_prefill;
mod residency;
mod resident_prefill;
pub(in crate::family::inkling2) mod router;
pub(in crate::family::inkling2) mod streaming;

pub(in crate::family::inkling2) use residency::{ExpertLayer, ExpertStack};
use residency::{PackedBudget, ResidentExperts};

pub(in crate::family::inkling2) struct GgufCuda {
    step: DenseStep,
    draft_head: Option<DraftHead>,
    dense: Projections,
    packed_gguf: bool,
    group_projections: bool,
    group_attention: bool,
    batch_router: bool,
    resident_attention: Option<attention::ResidentAttention>,
    moe_scratch: Option<ResidentMoe>,
    route_scratch: Option<router::RouteScratch>,
    batch_streamed: bool,
    prefill_resident: bool,
    device_prefill: Option<RefCell<device_prefill::Workspace>>,
    resident: ResidentExperts,
    cache: RefCell<BTreeMap<u32, cache::LayerCache>>,
    prefill_report: Option<RefCell<PrefillBankReport>>,
    stream_banks: Option<RefCell<crate::expert_tier::CudaStreamBanks>>,
    expert_codes: DeviceBuffer,
    expert_scales: DeviceBuffer,
    packed: RefCell<Packed>,
    context: CudaContext,
}

pub(in crate::family::inkling2) struct GgufDenseUpload<'a> {
    pub matrix: DenseUpload<'a>,
    pub dtype: DenseDtype,
}

type Projections = BTreeMap<String, (Resident, DenseDtype)>;

type ControllerOutput = Option<[Vec<f32>; 2]>;

pub(in crate::family::inkling2) use super::dense::ControlledAttentionInput;

pub(in crate::family::inkling2) struct BatchController<'a> {
    cuda: &'a GgufCuda,
    weights: ControllerWeights<'a>,
    shape: [usize; 2],
    capacity: usize,
}

impl BatchController<'_> {
    pub(in crate::family::inkling2) fn project(
        &self,
        inputs: &[&[f32]],
        role: &str,
    ) -> Result<Vec<[Vec<f32>; 2]>, EngError> {
        let values = self
            .cuda
            .step
            .project_controller_batch(&self.cuda.context, inputs, self.weights, self.shape)?
            .ok_or_else(|| bad("CUDA controller batch was not admitted"))?;
        let receipt = || {
            serde_json::json!({
                "role": role, "rows": inputs.len(), "tiles": inputs.len().div_ceil(self.capacity),
            })
        };
        static USED: std::sync::Once = std::sync::Once::new();
        USED.call_once(|| eprintln!("Inkling CUDA used multirow controller {}", receipt()));
        if self.cuda.prefill_report.is_some() {
            eprintln!("Inkling CUDA multirow controller tile report {}", receipt());
        }
        Ok(values)
    }
}

pub(in crate::family::inkling2) struct AttentionControllers<'a> {
    cuda: &'a GgufCuda,
    weights: [Option<ControllerWeights<'a>>; 3],
    shape: [usize; 2],
}

impl AttentionControllers<'_> {
    pub(in crate::family::inkling2) fn chained_projections<const N: usize>(
        &self,
        inputs: [ControlledAttentionInput<'_>; 3],
        requests: [Option<Projection<'_>>; N],
        eps: f32,
    ) -> Result<Option<[Vec<f32>; N]>, EngError> {
        let [Some(q), Some(k), Some(v)] = self.weights else {
            return Ok(None);
        };
        let Some(weights) = self.cuda.attention_group([self.shape[1]; 3], requests) else {
            return Ok(None);
        };
        self.cuda
            .step
            .controlled_attention(&self.cuda.context, inputs, [q, k, v], weights?, eps)
    }

    pub(in crate::family::inkling2) fn project(
        &self,
        inputs: &[[&[f32]; 3]],
        multirow: bool,
    ) -> Result<Vec<[ControllerOutput; 3]>, EngError> {
        if multirow && inputs.len() > 1 {
            let capacity = self.cuda.step.controller_batch_capacity(self.shape)?;
            if capacity >= 2 {
                let mut result = vec![[None, None, None]; inputs.len()];
                for (index, weights) in self.weights.into_iter().enumerate() {
                    let Some(weights) = weights else { continue };
                    let controller = BatchController {
                        cuda: self.cuda,
                        weights,
                        shape: self.shape,
                        capacity,
                    };
                    let role = ["q", "k", "v"][index];
                    let gates: Vec<_> = inputs.iter().map(|row| row[index]).collect();
                    for (row, projection) in
                        result.iter_mut().zip(controller.project(&gates, role)?)
                    {
                        row[index] = Some(projection);
                    }
                }
                return Ok(result);
            }
        }
        let rows: Vec<_> = inputs
            .iter()
            .flat_map(|row| self.weights.into_iter().zip(row))
            .filter_map(|(weights, &input)| weights.map(|weights| ControllerRow { weights, input }))
            .collect();
        let values = self
            .cuda
            .step
            .controller_rows(&self.cuda.context, &rows, self.shape)?;
        if self.cuda.prefill_report.is_some() {
            eprintln!(
                "Inkling CUDA grouped controllers {}",
                serde_json::json!({"attention_rows": inputs.len(), "controller_rows": rows.len()})
            );
        }
        let mut values = values.into_iter();
        Ok(inputs
            .iter()
            .map(|_| {
                self.weights.map(|weights| {
                    weights.map(|_| {
                        values
                            .next()
                            .expect("each resident controller produced a row")
                    })
                })
            })
            .collect())
    }
}

fn projection_bytes(matrices: &[GgufDenseUpload<'_>]) -> u64 {
    matrices
        .iter()
        .map(|item| item.matrix.codes.len() as u64)
        .sum()
}

fn upload_projections(
    context: &CudaContext,
    matrices: &[GgufDenseUpload<'_>],
) -> Result<Projections, EngError> {
    matrices
        .iter()
        .map(|GgufDenseUpload { matrix, dtype }| {
            let codes = upload_tensor(context, matrix.codes)?;
            let weight = Resident {
                codes,
                shape: matrix.shape,
            };
            Ok((matrix.name.to_owned(), (weight, *dtype)))
        })
        .collect()
}

fn upload_optional_projections(
    context: &CudaContext,
    matrices: &[GgufDenseUpload<'_>],
    dense: &mut Projections,
    headroom: u64,
) -> Result<(), EngError> {
    // A controller needs all three matrices resident to use one upload and readback.
    fn group_name(name: &str) -> &str {
        name.rsplit_once('.').map_or(name, |(stem, _)| stem)
    }
    for group in matrices.chunk_by(|a, b| group_name(a.matrix.name) == group_name(b.matrix.name)) {
        let free = (context.memory_info()?.free as u64).saturating_sub(headroom);
        if projection_bytes(group) > free {
            continue;
        }
        let uploaded = upload_projections(context, group)?;
        if context.memory_info()?.free as u64 >= headroom {
            dense.extend(uploaded);
        }
    }
    Ok(())
}

pub(in crate::family::inkling2) struct Options {
    pub direct: bool,
    pub packed: bool,
    pub group_projections: bool,
    pub stream: bool,
    pub budget: u64,
    pub cache_reserve: Option<u64>,
    pub resident_attention: Option<attention::Plan>,
    pub allow_device_routing: bool,
}

struct ResidentMoe {
    shape: GgufMoeShape,
    scratch: DeviceBuffer,
}

struct Packed {
    codes: Vec<u8>,
    scales: Vec<u8>,
}

pub(in crate::family::inkling2) struct ExpertView<'a> {
    pub id: ExpertId,
    pub part: usize,
    pub kind: GgufType,
    pub shape: [usize; 2],
    pub bytes: &'a [u8],
    pub factor: f32,
}

impl GgufCuda {
    pub(in crate::family::inkling2) fn context(&self) -> &CudaContext {
        &self.context
    }

    pub(in crate::family::inkling2) fn resident_attention(
        &self,
    ) -> Option<(&CudaContext, &attention::ResidentAttention)> {
        self.resident_attention
            .as_ref()
            .map(|attn| (&self.context, attn))
    }

    pub(in crate::family::inkling2) fn relative_attention_projection(
        &self,
        name: &str,
        shape: [usize; 2],
    ) -> Result<Option<attention::OutputProjection<'_>>, EngError> {
        let Some(attention) = &self.resident_attention else {
            return Ok(None);
        };
        if attention.projected_rows() != Some(shape[0]) {
            return Ok(None);
        }
        let Some((weight, dtype)) = self.dense.get(name) else {
            return Ok(None);
        };
        if weight.shape != shape || self.step.uses_q8_activations(*dtype) {
            return Ok(None);
        }
        Ok(Some(attention::OutputProjection {
            weight: weight.codes.range(0, weight.codes.bytes())?,
            dtype: *dtype,
            rows: shape[0],
        }))
    }

    pub(in crate::family::inkling2) fn load(
        dense: Vec<GgufDenseUpload<'_>>,
        spec: &super::TextSpec,
        layers: Vec<ExpertLayer<'_>>,
        file: &MappedFile,
        Options {
            direct,
            packed,
            group_projections,
            stream,
            budget,
            cache_reserve,
            resident_attention,
            allow_device_routing,
        }: Options,
    ) -> Result<Self, EngError> {
        let hidden = spec.hidden_size as usize;
        let ff = spec.intermediate_size as usize;
        let rows = hidden.max(2 * ff);
        let shape = dense.iter().fold([rows, hidden.max(ff)], |[r, c], item| {
            [r.max(item.matrix.shape[0]), c.max(item.matrix.shape[1])]
        });
        let elements = rows * hidden.max(ff);
        let stage = [2 * elements, elements / 16];
        let group_attention =
            std::env::var("FIFI_INKLING_CUDA_GROUP_ATTENTION").as_deref() == Ok("1");
        let [rows, columns] = shape;
        let attention_rows = dense
            .iter()
            .filter(|item| item.matrix.name.contains(".attn."))
            .map(|item| item.matrix.shape[0])
            .max()
            .unwrap_or(0);
        let packed_outputs = if group_attention {
            rows.max(3 * hidden + 8 * attention_rows)
        } else {
            rows
        };
        let batch_experts =
            packed && std::env::var("FIFI_INKLING_CUDA_BATCH_EXPERTS").as_deref() == Ok("1");
        let device_route = allow_device_routing
            && group_projections
            && batch_experts
            && std::env::var("FIFI_INKLING_CUDA_DEVICE_ROUTING").as_deref() == Ok("1");
        let route_shape = device_route.then(|| router::route_shape(spec));
        let route_plan = route_shape
            .map(|shape| router::RoutePlan::new(shape, &layers))
            .transpose()?;
        let route_bytes = route_plan.as_ref().map_or(0, router::RoutePlan::bytes);
        let route_input = route_shape.map_or(0, |shape| hidden + shape.n_expert + 1);
        let plan = DensePlan::new(
            [packed_outputs, columns.max(route_input)],
            hidden,
            spec.core.gate_rank as usize,
        )?;
        let (required, mut optional): (Vec<_>, Vec<_>) = dense
            .into_iter()
            .partition(|item| item.dtype != DenseDtype::F32);
        let tensors = projection_bytes(&required);
        let (admission, _) = CudaContext::fixed_projection_plan(shape)?;
        let context = CudaContext::with_admission(&admission)?;
        let prefill_resident =
            packed && std::env::var("FIFI_INKLING_CUDA_RESIDENT_PREFILL").as_deref() == Ok("1");
        let device_prefill = packed
            && std::env::var("FIFI_INKLING_CUDA_DEVICE_EXPERT_PREFILL").as_deref() == Ok("1");
        let stream_plan = stream.then(|| streaming::plan(hidden, ff)).transpose()?;
        let mut stream_bytes = stream_plan.map_or(0, |plan| plan.device_bytes());
        let moe_shape = GgufMoeShape {
            hidden,
            middle: ff,
            experts: spec.n_routed_experts as usize,
        };
        let moe_bytes = if batch_experts {
            moe_shape
                .scratch_bytes(spec.num_experts_per_tok as usize)
                .ok_or_else(|| bad("CUDA GGUF expert group exceeds its limits"))?
        } else {
            0
        };
        let demand = Demand {
            tensors,
            work: (stage[0] + stage[1] + plan.device_bytes + moe_bytes + route_bytes) as u64
                + stream_bytes,
            kv_cache: 0,
            headroom: DEVICE_HEADROOM,
        };
        admit(&demand, context.memory_info()?.free as u64)?;
        let q8_activations =
            std::env::var("FIFI_INKLING_CUDA_Q8_ACTIVATIONS").as_deref() == Ok("1");
        let step = DenseStep::new(&context, &plan)?.with_q8_activations(q8_activations);
        let expert_codes = context.allocate(stage[0])?;
        let expert_scales = context.allocate(stage[1])?;
        let mut dense = upload_projections(&context, &required)?;
        let mut stream_banks = stream_plan
            .map(|plan| crate::expert_tier::CudaStreamBanks::new(&context, plan))
            .transpose()?;
        let moe_scratch = (moe_bytes > 0)
            .then(|| {
                context.allocate(moe_bytes).map(|scratch| ResidentMoe {
                    shape: moe_shape,
                    scratch,
                })
            })
            .transpose()?;
        let route_scratch = route_plan.map(|plan| plan.upload(&context)).transpose()?;
        let experts = spec.n_routed_experts as u32;
        let full_expert_fit = packed && residency::packed_model_fits(&context, &layers, budget)?;
        let mut batch_streamed = false;
        // Keep the exact resident placement when the model fits with the original banks.
        // A streamed group needs one larger slot; the second still serves serial prefill.
        if batch_experts
            && !full_expert_fit
            && std::env::var("FIFI_INKLING_CUDA_BATCH_STREAMED_EXPERTS").as_deref() == Ok("1")
            && let (Some(banks), Some(original)) = (&mut stream_banks, stream_plan)
        {
            let plan = streaming::group_plan(
                &layers,
                experts as usize,
                spec.num_experts_per_tok as usize,
                original,
            )?;
            let free = (context.memory_info()?.free as u64).saturating_sub(VRAM_HEADROOM_BYTES);
            if plan.bank_bytes() <= free {
                banks.replace_first(&context, plan)?;
                stream_bytes = original.bank_bytes() + plan.bank_bytes();
                batch_streamed = true;
            }
        }
        optional.sort_by_key(|item| !item.matrix.name.ends_with(".mlp.gate.weight"));
        if !full_expert_fit {
            upload_optional_projections(&context, &optional, &mut dense, VRAM_HEADROOM_BYTES)?;
        }
        let resident = if packed {
            ResidentExperts::load_packed(
                &context,
                experts,
                layers,
                file,
                PackedBudget {
                    total: budget,
                    cache: cache_reserve,
                },
            )?
        } else {
            ResidentExperts::load_nvfp4(&context, experts, layers, file, direct)?
        };
        let resident_attention = match resident_attention {
            Some(plan) if resident.placement().full_layers().len() == spec.stored_layers() => {
                let charge_reserve = std::env::var("FIFI_INKLING_CUDA_RESIDENT_ATTENTION_RESERVE")
                    .as_deref()
                    == Ok("1");
                plan.load(&context, charge_reserve)?
            }
            Some(_) => {
                eprintln!("Inkling CUDA relative attention requires resident expert weights");
                None
            }
            None => None,
        };
        // Admit optional dense projections after persistent state and resident experts.
        if full_expert_fit {
            let headroom = resident_attention.as_ref().map_or(
                VRAM_HEADROOM_BYTES,
                attention::ResidentAttention::projection_headroom,
            );
            upload_optional_projections(&context, &optional, &mut dense, headroom)?;
        }
        let sum_bytes = |kind: Option<DenseDtype>| {
            dense
                .values()
                .filter(|(_, dtype)| kind.is_none_or(|kind| kind == *dtype))
                .map(|(weight, _)| weight.codes.bytes() as u64)
                .sum::<u64>()
        };
        if let Some(scratch) = &route_scratch {
            let layers: Vec<_> = resident
                .placement()
                .full_layers()
                .into_iter()
                .filter(|&layer| {
                    scratch.has_layer(layer)
                        && dense
                            .get(&format!("model.llm.layers.{layer}.mlp.gate.weight"))
                            .is_some_and(|(_, dtype)| *dtype == DenseDtype::F32)
                })
                .collect();
            let visits = spec
                .visits()
                .iter()
                .filter(|visit| layers.contains(&(visit.weight_layer as u32)))
                .count();
            eprintln!(
                "Inkling GGUF CUDA device routing {}",
                serde_json::json!({"layers": layers, "visits": visits})
            );
        }
        eprintln!(
            "Inkling GGUF CUDA projections {}",
            serde_json::json!({"resident_projection_count": dense.len(), "resident_bf16_bytes": sum_bytes(Some(DenseDtype::Bf16)), "resident_f32_bytes": sum_bytes(Some(DenseDtype::F32)), "resident_dense_bytes": sum_bytes(None), "packed_gguf": packed, "q8_1_projection_activations": q8_activations, "batched_resident_experts": moe_scratch.is_some(), "batched_streamed_experts": batch_streamed, "batched_resident_prefill": prefill_resident,
                "direct_resident_nvfp4": direct || packed, "cold_stream_device_bytes": stream_bytes, "packed_expert_staging_bytes": stage[0] + stage[1], "dense_stream_arena_bytes": plan.device_bytes,
                "resident_expert_bytes": resident.bytes(),
                "resident_expert_layers": resident.placement().full_layers(),
                "resident_expert_count": resident.placement().resident_experts()})
        );
        let profile = stream && std::env::var("FIFI_INKLING_CUDA_PROFILE").as_deref() == Ok("1");
        if profile {
            let admitted: Vec<_> = optional
                .iter()
                .map(|item| item.matrix.name)
                .filter(|name| dense.contains_key(*name))
                .collect();
            eprintln!(
                "Inkling GGUF CUDA optional projections {}",
                serde_json::json!(admitted)
            );
        }
        Ok(Self {
            step,
            draft_head: None,
            dense,
            packed_gguf: packed,
            group_projections,
            group_attention,
            batch_router: std::env::var("FIFI_INKLING_CUDA_BATCH_ROUTER").as_deref() == Ok("1"),
            resident_attention,
            moe_scratch,
            route_scratch,
            batch_streamed,
            prefill_resident,
            device_prefill: device_prefill.then(RefCell::default),
            resident,
            cache: RefCell::default(),
            prefill_report: profile.then(RefCell::default),
            stream_banks: stream_banks.map(RefCell::new),
            expert_codes,
            expert_scales,
            packed: RefCell::new(Packed {
                codes: vec![0; elements / 2],
                scales: vec![0; elements / 16],
            }),
            context,
        })
    }

    pub(in crate::family::inkling2) fn enable_draft_head(
        &mut self,
        cap: f32,
    ) -> Result<(), EngError> {
        let (weight, dtype) = self.draft_head_weight()?;
        let head = DraftHead::new(&self.context, weight.shape, *dtype, cap)?;
        eprintln!("Inkling CUDA DSpark corrected head enabled");
        self.draft_head = Some(head);
        Ok(())
    }

    pub(in crate::family::inkling2) fn draft_greedy(
        &self,
        rows: [&[f32]; 2],
        vocab: usize,
        cap: f32,
    ) -> Result<Option<u32>, EngError> {
        let Some(head) = &self.draft_head else {
            return Ok(None);
        };
        let (weight, dtype) = self.draft_head_weight()?;
        if weight.shape[0] != vocab {
            return Err(bad("CUDA draft head vocabulary changed after admission"));
        }
        head.choose(&self.step, &self.context, (weight, *dtype), rows, cap)
            .map(Some)
    }

    fn draft_head_weight(&self) -> Result<&(Resident, DenseDtype), EngError> {
        self.dense
            .get(crate::family::inkling2::OUTPUT_HEAD)
            .ok_or_else(|| bad("CUDA draft head requires a resident GGUF vocabulary projection"))
    }

    pub(in crate::family::inkling2) fn project(
        &self,
        name: &str,
        rows: usize,
        x: &[f32],
    ) -> Option<Result<Vec<f32>, EngError>> {
        let (resident, dtype) = self.dense.get(name)?;
        if resident.shape != [rows, x.len()] {
            return Some(Err(bad(format!(
                "bad GGUF CUDA projection shape for {name}"
            ))));
        }
        Some(
            if *dtype == DenseDtype::F32 && name.ends_with(".mlp.gate.weight") {
                self.step.project_router(&self.context, resident, x)
            } else if *dtype == DenseDtype::Bf16 {
                self.step.project(&self.context, resident, x)
            } else {
                resident
                    .codes
                    .range(0, resident.codes.bytes())
                    .and_then(|raw| {
                        self.step.project_gguf(
                            &self.context,
                            None,
                            (*dtype, raw, resident.shape, 1.0),
                            x,
                        )
                    })
            },
        )
    }

    pub(in crate::family::inkling2) fn batch_prefill_router(&self, layer: u32, name: &str) -> bool {
        // The cold path interleaves route observation with prediction for the
        // next token. Keep its existing ordering.
        self.batch_router
            && !self.resident.placement().has_cold(layer)
            && self
                .dense
                .get(name)
                .is_some_and(|(_, dtype)| *dtype == DenseDtype::F32)
    }

    pub(in crate::family::inkling2) fn prefill_router_logits(
        &self,
        name: &str,
        inputs: &[&[f32]],
    ) -> Result<Vec<f32>, EngError> {
        let (weight, _) = self
            .dense
            .get(name)
            .ok_or_else(|| bad("missing resident prefill router projection"))?;
        self.step.project_router_rows(&self.context, weight, inputs)
    }

    pub(in crate::family::inkling2) fn attention_projections<const N: usize>(
        &self,
        inputs: [&[f32]; 3],
        requests: [Option<Projection<'_>>; N],
    ) -> Option<Result<[Vec<f32>; N], EngError>> {
        let weights = self.attention_group(inputs.map(<[f32]>::len), requests)?;
        Some(weights.and_then(|weights| self.step.project_group(&self.context, inputs, weights)))
    }

    pub(in crate::family::inkling2) fn attention_projections_rows<const N: usize>(
        &self,
        inputs: &[[&[f32]; 3]],
        requests: [Option<Projection<'_>>; N],
    ) -> Result<Option<Vec<[Vec<f32>; N]>>, EngError> {
        let Some(first) = inputs.first() else {
            return Ok(None);
        };
        let Some(weights) = self.attention_group(first.map(<[f32]>::len), requests) else {
            return Ok(None);
        };
        let projected = self
            .step
            .project_group_rows(&self.context, inputs, weights?)?;
        if projected.is_some() && self.prefill_report.is_some() {
            eprintln!(
                "Inkling CUDA batched attention projections: {} rows",
                inputs.len()
            );
        }
        Ok(projected)
    }

    fn attention_group<const N: usize>(
        &self,
        widths: [usize; 3],
        requests: [Option<Projection<'_>>; N],
    ) -> Option<Result<[Option<ResidentProjection<'_>>; N], EngError>> {
        if !self.group_attention {
            return None;
        }
        let mut weights = [None; N];
        for (request, out) in requests.into_iter().zip(&mut weights) {
            let Some(p) = request else { continue };
            let (weight, dtype) = self.dense.get(p.name)?;
            if weight.shape != [p.rows, widths[p.input]] {
                return Some(Err(bad(format!(
                    "bad GGUF CUDA projection shape for {}",
                    p.name
                ))));
            }
            *out = Some(ResidentProjection {
                weight,
                dtype: *dtype,
                input: p.input,
            });
        }
        Some(Ok(weights))
    }

    pub(in crate::family::inkling2) fn controller(
        &self,
        names: [&str; 3],
        x: &[f32],
        shape: [usize; 2],
    ) -> Option<Result<[Vec<f32>; 2], EngError>> {
        Some(
            self.controller_weights(names, x.len(), shape)?
                .and_then(|weights| self.step.controller(&self.context, weights, x)),
        )
    }

    fn controller_weights(
        &self,
        [down, gain, forget]: [&str; 3],
        input_width: usize,
        [rank, hidden]: [usize; 2],
    ) -> Option<Result<ControllerWeights<'_>, EngError>> {
        if !self.group_projections {
            return None;
        }
        let shapes = [[rank, input_width], [hidden, rank], [hidden, rank]];
        let weights = [
            self.dense.get(down)?,
            self.dense.get(gain)?,
            self.dense.get(forget)?,
        ];
        if weights
            .iter()
            .zip(shapes)
            .any(|((w, _), shape)| w.shape != shape)
        {
            return Some(Err(bad("bad GGUF CUDA controller projection shape")));
        }
        Some(Ok(weights.map(|(w, dtype)| (w, *dtype))))
    }

    pub(in crate::family::inkling2) fn batch_controller(
        &self,
        names: [&str; 3],
        shape: [usize; 2],
        rows: usize,
    ) -> Result<Option<BatchController<'_>>, EngError> {
        let capacity = self.step.controller_batch_capacity(shape)?;
        if rows < 2 || capacity < 2 {
            return Ok(None);
        }
        let Some(weights) = self
            .controller_weights(names, 2 * shape[1], shape)
            .transpose()?
        else {
            return Ok(None);
        };
        Ok(Some(BatchController {
            cuda: self,
            weights,
            shape,
            capacity,
        }))
    }

    pub(in crate::family::inkling2) fn attention_controllers(
        &self,
        names: [[&str; 3]; 3],
        shape: [usize; 2],
        rows: usize,
    ) -> Result<Option<AttentionControllers<'_>>, EngError> {
        let mut weights = [None; 3];
        for (weight, names) in weights.iter_mut().zip(names) {
            *weight = self
                .controller_weights(names, 2 * shape[1], shape)
                .transpose()?;
        }
        if weights.iter().flatten().count().saturating_mul(rows) < 2
            || self.step.controller_capacity(shape[1])? < 2
        {
            return Ok(None);
        }
        Ok(Some(AttentionControllers {
            cuda: self,
            weights,
            shape,
        }))
    }

    pub(in crate::family::inkling2) fn expert_dtype(&self, kind: GgufType) -> Option<DenseDtype> {
        if self.packed_gguf || matches!(kind, GgufType::Bf16 | GgufType::Nvfp4) {
            DenseDtype::from_gguf(kind)
        } else {
            None
        }
    }

    pub(in crate::family::inkling2) fn expert(
        &self,
        expert: &ExpertView<'_>,
        x: &[f32],
    ) -> Option<Result<Vec<f32>, EngError>> {
        if self.packed_gguf {
            let dtype = self.expert_dtype(expert.kind)?;
            return (self.resident.placement().placement(expert.id) == Placement::Gpu)
                .then(|| self.resident_packed_expert(expert, dtype, x));
        }
        match expert.kind {
            GgufType::Bf16 => Some(self.bf16_expert(expert, x)),
            GgufType::Nvfp4 => {
                if self.resident.placement().placement(expert.id) != Placement::Gpu
                    && !can_repack(expert.bytes)
                {
                    return None;
                }
                Some(self.nvfp4_expert(expert, x))
            }
            _ => None,
        }
    }

    pub(in crate::family::inkling2) fn batches_expert_layer(
        &self,
        layer: u32,
        dtypes: [DenseDtype; 2],
    ) -> bool {
        self.moe_scratch.is_some()
            && (!self.resident.placement().has_cold(layer)
                || (self.batch_streamed
                    && self.streams_layer(layer)
                    && !dtypes
                        .into_iter()
                        .any(|dtype| self.step.uses_q8_activations(dtype))))
    }

    pub(in crate::family::inkling2) fn expert_group(
        &self,
        source: &mut impl ExpertPayloadSource,
        group: streaming::ExpertGroup<'_>,
    ) -> Result<Vec<f32>, EngError> {
        if self.streams_layer(group.route.layer) {
            return self.stream_expert_group(source, &group);
        }
        let stacks = self.resident.original_layer(group.route.layer)?;
        self.run_expert_group(None, &group, GgufMoeWeights::Stacks(stacks))
    }

    fn run_expert_group(
        &self,
        stream: Option<&DeviceStream>,
        group: &streaming::ExpertGroup<'_>,
        weights: GgufMoeWeights<'_>,
    ) -> Result<Vec<f32>, EngError> {
        let moe = self
            .moe_scratch
            .as_ref()
            .ok_or_else(|| bad("no CUDA GGUF expert scratch"))?;
        if group.route.input.len() != moe.shape.hidden {
            return Err(bad("CUDA GGUF expert group input width differs"));
        }
        let scratch = moe.scratch.range(0, moe.scratch.bytes())?;
        self.step
            .run_layer(&self.context, stream, group.route.input, |stream, x, y| {
                self.context.gguf_moe(
                    stream,
                    GgufMoeCall {
                        shape: moe.shape,
                        dtypes: group.dtypes,
                        caps: group.route.caps,
                        picks: GgufMoePicks::Host(group.picks),
                        weights,
                        x,
                        scratch,
                        y,
                    },
                )
            })
    }

    fn resident_packed_expert(
        &self,
        expert: &ExpertView<'_>,
        dtype: DenseDtype,
        x: &[f32],
    ) -> Result<Vec<f32>, EngError> {
        let Some(residency::Weight::Original(raw)) =
            self.resident
                .ranges(expert.id, expert.part, [expert.bytes.len(), 0])?
        else {
            return Err(bad("resident GGUF expert has no original packed weight"));
        };
        self.step.project_gguf(
            &self.context,
            None,
            (dtype, raw, expert.shape, expert.factor),
            x,
        )
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
        if let Some(buffers) = self
            .resident
            .ranges(expert.id, expert.part, [n / 2, n / 16])?
        {
            return match buffers {
                residency::Weight::Original(raw) => self.step.project_gguf(
                    &self.context,
                    None,
                    (DenseDtype::GgufNvfp4, raw, expert.shape, expert.factor),
                    x,
                ),
                residency::Weight::Converted(pair) => self.project_nvfp4(pair, expert, x),
            };
        }
        let Packed { codes, scales } = &mut *self.packed.borrow_mut();
        let (codes, scales) = (&mut codes[..n / 2], &mut scales[..n / 16]);
        repack_nvfp4_into(expert.bytes, codes, scales)?;
        self.context.upload_at(&self.expert_codes, 0, codes)?;
        self.context.upload_at(&self.expert_scales, 0, scales)?;
        self.project_nvfp4(
            [
                self.expert_codes.range(0, codes.len())?,
                self.expert_scales.range(0, scales.len())?,
            ],
            expert,
            x,
        )
    }

    fn project_nvfp4(
        &self,
        [codes, scales]: [DeviceRange<'_>; 2],
        expert: &ExpertView<'_>,
        x: &[f32],
    ) -> Result<Vec<f32>, EngError> {
        self.context
            .nvfp4_matvec_range(codes, scales, expert.factor, expert.shape, x)
    }
}

pub(in crate::family::inkling2) fn can_repack(bytes: &[u8]) -> bool {
    let (blocks, tail) = bytes.as_chunks::<36>();
    // The GGUF unsigned scale 0xff has no E4M3FN representation.
    tail.is_empty() && blocks.iter().all(|b| !b[..4].contains(&0xff))
}
