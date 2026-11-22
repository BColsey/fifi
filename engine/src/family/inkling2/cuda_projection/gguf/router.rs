use std::collections::BTreeMap;

use super::{ExpertLayer, GgufCuda, bad};
use crate::{
    EngError,
    family::inkling2::{
        TextSpec, cpu_forward::ROUTE_NORMALIZATION_FLOOR, gguf_weights::expert_scale,
    },
    kernels::cuda::{
        CudaContext, DeviceBuffer, DeviceRange, DeviceStream,
        resident::{
            dense_gemv::{DenseDtype, GgufMoeCall, GgufMoePicks, GgufMoeWeights},
            router::{
                RouterLogitsCall, RouterShape, RouterSigmoidBias, RouterSigmoidCall,
                router_packed_logits_bytes,
            },
        },
    },
};

pub(in crate::family::inkling2) struct RouteInput<'a> {
    pub(in crate::family::inkling2) layer: u32,
    pub(in crate::family::inkling2) input: &'a [f32],
    pub(in crate::family::inkling2) caps: [f32; 2],
    pub(in crate::family::inkling2) scale: f32,
}

pub(super) fn route_shape(spec: &TextSpec) -> RouterShape {
    RouterShape {
        m: 1,
        hidden: spec.hidden_size as usize,
        n_expert: spec.n_routed_experts as usize,
        top_k: spec.num_experts_per_tok as usize,
    }
}

pub(super) struct RoutePlan {
    shape: RouterShape,
    values: Vec<u8>,
    scales: BTreeMap<u32, usize>,
}

impl RoutePlan {
    pub(super) fn new(shape: RouterShape, layers: &[ExpertLayer<'_>]) -> Result<Self, EngError> {
        // The allocation starts with the mutable logits and picks, followed by read-only
        // gate and down scale rows for each eligible physical layer.
        let mut values = vec![0; RouteScratch::bytes(shape)?];
        let mut scales = BTreeMap::new();
        for layer in layers {
            let tensors = layer.parts.each_ref().map(|part| part.tensor_scales);
            if let Some(row) = scale_rows(tensors, shape.n_expert) {
                scales.insert(layer.layer, values.len());
                values.extend_from_slice(&row);
            }
        }
        Ok(Self {
            shape,
            values,
            scales,
        })
    }

    pub(super) fn bytes(&self) -> usize {
        self.values.len()
    }

    pub(super) fn upload(self, context: &CudaContext) -> Result<RouteScratch, EngError> {
        RouteScratch::new(self.shape, context.upload(&self.values)?, self.scales)
    }
}

fn scale_rows(tensors: [Option<&[u8]>; 2], experts: usize) -> Option<Vec<u8>> {
    let bytes = experts * size_of::<f32>();
    let mut row = Vec::with_capacity(2 * bytes);
    for tensor in tensors {
        if tensor.is_some_and(|raw| raw.len() != bytes) {
            return None;
        }
        for expert in 0..experts {
            let scale = expert_scale(tensor, expert).ok()?;
            // A zero factor takes the host single-expert path, which skips reading that matrix.
            if scale == 0.0 {
                return None;
            }
            row.extend_from_slice(&scale.to_ne_bytes());
        }
    }
    Some(row)
}

pub(super) struct RouteScratch {
    shape: RouterShape,
    scratch: DeviceBuffer,
    logits_bytes: usize,
    scales: BTreeMap<u32, usize>,
}

struct RouterInputs<'a> {
    x: DeviceRange<'a>,
    w: DeviceRange<'a>,
    bias: DeviceRange<'a>,
    status: DeviceRange<'a>,
    scale: f32,
}

impl RouteScratch {
    pub(super) fn bytes(shape: RouterShape) -> Result<usize, EngError> {
        Ok(router_packed_logits_bytes(shape)? + 2 * shape.top_k * size_of::<f32>())
    }

    fn new(
        shape: RouterShape,
        scratch: DeviceBuffer,
        scales: BTreeMap<u32, usize>,
    ) -> Result<Self, EngError> {
        Ok(Self {
            shape,
            scratch,
            logits_bytes: router_packed_logits_bytes(shape)?,
            scales,
        })
    }

    pub(super) fn has_layer(&self, layer: u32) -> bool {
        self.scales.contains_key(&layer)
    }

    fn scale_ranges(&self, layer: u32) -> Result<[DeviceRange<'_>; 2], EngError> {
        let offset = self
            .scales
            .get(&layer)
            .ok_or_else(|| bad("no CUDA expert scales"))?;
        let bytes = self.shape.n_expert * size_of::<f32>();
        Ok([
            self.scratch.range(*offset, bytes)?,
            self.scratch.range(offset + bytes, bytes)?,
        ])
    }

    fn ranges(&self) -> Result<[DeviceRange<'_>; 3], EngError> {
        let picks = self.shape.top_k * size_of::<f32>();
        Ok([
            self.scratch.range(0, self.logits_bytes)?,
            self.scratch.range(self.logits_bytes, picks)?,
            self.scratch.range(self.logits_bytes + picks, picks)?,
        ])
    }

    fn enqueue(
        &self,
        context: &CudaContext,
        stream: &DeviceStream,
        input: RouterInputs<'_>,
    ) -> Result<[DeviceRange<'_>; 2], EngError> {
        let [logits, ids, weights] = self.ranges()?;
        let shape = self.shape;
        context.router_logits_rows(
            stream,
            RouterLogitsCall {
                shape,
                x: input.x,
                w: input.w,
                logits,
            },
        )?;
        context.router_sigmoid_select_rows(
            stream,
            RouterSigmoidCall {
                shape,
                bias: Some(RouterSigmoidBias {
                    values: input.bias,
                    floor: ROUTE_NORMALIZATION_FLOOR,
                    scale: input.scale,
                }),
                logits,
                ids,
                weights,
                status: input.status,
            },
        )?;
        Ok([ids, weights])
    }
}

impl GgufCuda {
    pub(in crate::family::inkling2) fn has_resident_route(&self, layer: u32) -> bool {
        self.route_scratch
            .as_ref()
            .is_some_and(|scratch| scratch.has_layer(layer))
            && !self.resident.placement().has_cold(layer)
    }

    pub(in crate::family::inkling2) fn routed_expert_group(
        &self,
        prefix: &str,
        route: RouteInput<'_>,
        dtypes: [DenseDtype; 2],
        bias: impl FnOnce(usize) -> Result<Vec<f32>, EngError>,
    ) -> Result<Option<Vec<f32>>, EngError> {
        let Some((router, DenseDtype::F32)) = self.dense.get(&format!("{prefix}.gate.weight"))
        else {
            return Ok(None);
        };
        let work = self
            .route_scratch
            .as_ref()
            .ok_or_else(|| bad("no CUDA route scratch"))?;
        let moe = self
            .moe_scratch
            .as_ref()
            .ok_or_else(|| bad("no CUDA GGUF expert scratch"))?;
        let shape = work.shape;
        let bias = bias(shape.n_expert)?;
        if router.shape != [shape.n_expert, shape.hidden]
            || route.input.len() != shape.hidden
            || bias.len() != shape.n_expert
        {
            return Err(bad("CUDA GGUF routing shape differs"));
        }
        let w = router.codes.range(0, router.codes.bytes())?;
        let stacks = self.resident.original_layer(route.layer)?;
        let scales = work.scale_ranges(route.layer)?;
        let scratch = moe.scratch.range(0, moe.scratch.bytes())?;
        self.step
            .run_routed_layer(
                &self.context,
                route.input,
                &bias,
                |stream, [x, bias, status, y]| {
                    let [ids, weights] = work.enqueue(
                        &self.context,
                        stream,
                        RouterInputs {
                            x,
                            w,
                            bias,
                            status,
                            scale: route.scale,
                        },
                    )?;
                    self.context.gguf_moe(
                        stream,
                        GgufMoeCall {
                            shape: moe.shape,
                            dtypes,
                            caps: route.caps,
                            picks: GgufMoePicks::Device {
                                count: shape.top_k,
                                ids,
                                weights,
                                scales,
                            },
                            weights: GgufMoeWeights::Stacks(stacks),
                            x,
                            scratch,
                            y,
                        },
                    )
                },
            )
            .map(Some)
    }
}

#[cfg(test)]
mod tests;
