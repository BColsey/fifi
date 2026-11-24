//! Default-off canary: compose a whole device MLP input path at the first visit.
//! The next visit still owns a host Stream, so the raw residual is materialized
//! once at that boundary. The mixed/normalized MLP input is never downloaded.

use super::{
    CpuForward, EngError, ExpertPrefillRows, GgufTextWeights, LayerVisit, PromptRow, VisitState,
    bad,
};
use crate::{
    family::inkling2::cuda_projection::gguf::attention::prefill,
    kernels::cuda::{
        CudaContext, DeviceBuffer, DeviceRange, DeviceStream, PinnedBuf,
        resident::{inkling_local_prep::DepthShape, row_add::RowAddCall},
    },
};
use std::sync::{Once, OnceLock};

const ENABLED: &str = "FIFI_INKLING_CUDA_PREFILL_RESIDENT_FIRST_VISIT";
const EXPERT_CONV: &str = "FIFI_INKLING_CUDA_PREFILL_RESIDENT_EXPERT_CONV";

struct Scratch {
    shape: DepthShape,
    stream: DeviceStream,
    transfer: PinnedBuf,
    device: DeviceBuffer,
}

impl Scratch {
    fn allocate(ctx: &CudaContext, shape: DepthShape) -> Result<Self, EngError> {
        let [p, q, c, o] = shape
            .words()
            .ok_or_else(|| bad("unsupported Inkling first-visit depth shape"))?;
        // Put compact scores/health last: an odd row count must not shift the
        // router input off its required 16-byte alignment.
        let device = ctx.allocate(4 * (p + 2 * q + c + 3 * o))?;
        Ok(Self {
            shape,
            stream: ctx.stream()?,
            transfer: ctx.pinned_buf(4 * c)?,
            device,
        })
    }

    fn prepare(
        &self,
        rows: &[PromptRow<'_>],
        query: &[f32],
        gamma: &[f32],
    ) -> Result<(), EngError> {
        let [p, q, _, o] = self.shape.words().unwrap();
        let mut sealed = Vec::with_capacity(o);
        for row in rows {
            let mut normalized = row.stream.current().to_vec();
            super::super::rms_norm(&mut normalized, super::super::MIX_EPS);
            sealed.extend(normalized.into_iter().map(f32::to_ne_bytes));
        }
        self.device
            .write_at(&self.stream, 0, sealed.as_flattened())?;
        let query: Vec<_> = query.iter().map(|x| x.to_ne_bytes()).collect();
        let gamma: Vec<_> = gamma.iter().map(|x| x.to_ne_bytes()).collect();
        self.device
            .write_at(&self.stream, p * 4, query.as_flattened())?;
        self.device
            .write_at(&self.stream, (p + q) * 4, gamma.as_flattened())
    }

    fn depth_and_norm(
        &mut self,
        ctx: &CudaContext,
        convolved: DeviceRange<'_>,
        eps: f32,
    ) -> Result<DeviceRange<'_>, EngError> {
        let [p, q, c, o] = self.shape.words().unwrap();
        let range = |at, len| self.device.range(at * 4, len * 4);
        let shape = [self.shape.rows, self.shape.width];
        let raw_at = p + 2 * q;
        let compact = range(raw_at + 3 * o, c)?;
        let mixed = range(raw_at + o, o)?;
        let expert_input = range(raw_at + 2 * o, o)?;
        ctx.inkling_depth_current(
            &self.stream,
            [range(0, o)?, convolved, range(raw_at, o)?, range(o, o)?],
            shape,
            true,
        )?;
        ctx.inkling_depth_mix(
            &self.stream,
            &mut self.transfer,
            [range(0, p)?, range(p, q)?, compact, mixed],
            self.shape,
        )?;
        let health = compact.slice(0, 4 * self.shape.rows)?;
        ctx.inkling_weighted_norm(
            &self.stream,
            &mut self.transfer,
            [mixed, range(p + q, q)?, expert_input, health],
            shape,
            eps,
        )?;
        Ok(expert_input)
    }

    fn status(&self) -> Result<DeviceRange<'_>, EngError> {
        let [p, q, _, o] = self.shape.words().unwrap();
        self.device
            .range((p + 2 * q + 3 * o) * 4, self.shape.rows * 4)
    }

    fn materialize(
        &self,
        update: Option<(&CudaContext, DeviceRange<'_>)>,
    ) -> Result<Vec<u8>, EngError> {
        let [p, q, _, o] = self.shape.words().unwrap();
        let raw = self.device.range((p + 2 * q) * 4, o * 4)?;
        let result = if let Some((ctx, convolved)) = update {
            // The mix is dead after MLP norm; reuse that plane for the residual sum.
            let out = self.device.range(raw.offset() + raw.bytes(), raw.bytes())?;
            ctx.row_add(
                &self.stream,
                RowAddCall {
                    out,
                    x: raw,
                    y: convolved,
                    n: o,
                    flag: None,
                    fused: false,
                },
            )?;
            out
        } else {
            raw
        };
        let mut host = vec![0; result.bytes()];
        self.device
            .read_at(&self.stream, result.offset(), &mut host)?;
        Ok(host)
    }
}

impl CpuForward {
    pub(super) fn try_resident_first_visit(
        &self,
        visit: LayerVisit,
        state: &mut VisitState,
        rows: &mut [PromptRow<'_>],
        weights: &GgufTextWeights,
        (layer, prefix): (&str, &str),
        inputs: &[[Vec<f32>; 3]],
    ) -> Result<bool, EngError> {
        static ENABLE: OnceLock<bool> = OnceLock::new();
        if !*ENABLE.get_or_init(|| std::env::var(ENABLED).as_deref() == Ok("1"))
            || visit.state_slot != 0
            || visit.weight_layer != 0
            || rows
                .iter()
                .any(|row| row.bank.is_some() || row.stream.block_count() != 0)
            || !self.math.rms_norm_eps.is_finite()
            || self.math.rms_norm_eps <= 0.0
        {
            return Ok(false);
        }
        let width = self.hidden();
        let experts = self.spec.n_routed_experts as usize;
        let router = format!("{prefix}.gate.weight");
        let names = [
            format!("{prefix}.experts.w13_weight"),
            format!("{prefix}.experts.w2_weight"),
        ];
        let names = names.each_ref().map(String::as_str);
        let caps = [self.math.mlp_gate_softcap, self.math.mlp_up_softcap];
        if !width.is_multiple_of(4)
            || weights.resident_prefill_router_shape(0, &router) != Some([experts, width])
        {
            return Ok(false);
        }
        let Some((ctx, attention)) = weights.resident_attention() else {
            return Ok(false);
        };
        let Some(bridge) = self.prefill_bridge(layer, visit, prefill::resident_query_enabled())?
        else {
            return Ok(false);
        };
        let shape = DepthShape {
            rows: rows.len(),
            width,
            depth: 2,
        };
        let Some([_, _, _, plane]) = shape.words() else {
            return Ok(false);
        };
        static DEVICE_CONV: OnceLock<bool> = OnceLock::new();
        let device_conv =
            *DEVICE_CONV.get_or_init(|| std::env::var(EXPERT_CONV).as_deref() == Ok("1"));
        if device_conv && !plane.is_multiple_of(32) {
            return Ok(false);
        }
        let mut scratch = match Scratch::allocate(ctx, shape) {
            Ok(scratch) => scratch,
            Err(EngError::MemoryPlan(_)) => return Ok(false),
            Err(error) => return Err(error),
        };
        let Some(mut conv) = bridge.admit_conv(self, state, rows.len())? else {
            return Ok(false);
        };
        let mlp_conv = if device_conv {
            let Some(conv) = bridge.admit_mlp_conv(self, state, rows.len())? else {
                return Ok(false);
            };
            Some(conv)
        } else {
            None
        };
        if !weights.admit_resident_prefill_experts(
            names,
            0,
            [
                rows.len(),
                self.spec.num_experts_per_tok as usize,
                width,
                experts,
            ],
            caps,
        )? {
            return Ok(false);
        }
        let query = self
            .weights
            .vec("model.llm.attn_res.mlp_queries.0", width)?;
        let gamma = self
            .weights
            .vec(&format!("{layer}.mlp_norm.weight"), width)?;
        let bias = self.weights.vec(&format!("{prefix}.gate.bias"), experts)?;
        scratch.prepare(rows, &query, &gamma)?;
        let Some(host_output) =
            self.attention_prefill_bridge(layer, visit, state, inputs, Some(&mut conv))?
        else {
            return Ok(false);
        };
        // No fallback here: the attention and K/V state have advanced.
        if !host_output.is_empty() {
            return Err(bad("unexpected first-visit attention readback"));
        }
        let activations =
            scratch.depth_and_norm(ctx, conv.convolved_plane()?, self.math.rms_norm_eps)?;
        let routes = {
            let _span = self.phase_span(super::super::Work::Router);
            let logits = weights.resident_prefill_router_logits(&router, activations)?;
            self.route_prefill_logits(0, &logits, &bias, rows.len())?
        };
        let ids: Vec<_> = routes
            .iter()
            .map(|route| route.selected.as_slice())
            .collect();
        let outputs = {
            let _span = self.phase_span(super::super::Work::Experts);
            weights
                .prefill_device_experts(
                    names,
                    0,
                    (
                        ExpertPrefillRows::Resident {
                            ids: &ids,
                            hidden: width,
                            activations,
                        },
                        mlp_conv
                            .as_ref()
                            .map(|conv| conv.expert_destination())
                            .transpose()?,
                    ),
                    caps,
                    |row, id| routes[row].weight(id, self.math.route_scale),
                )?
                .ok_or_else(|| {
                    bad("previously admitted Inkling device experts declined the first visit")
                })?
        };
        bridge.publish_conv_tail(state, conv)?;
        let raw = if let Some(mut conv) = mlp_conv {
            attention.finish_prefill_expert_conv(ctx, &mut conv, scratch.status()?)?;
            let raw = scratch.materialize(Some((ctx, conv.convolved_plane()?)))?;
            bridge.publish_mlp_conv_tail(state, conv)?;
            static USED_CONV: Once = Once::new();
            USED_CONV
                .call_once(|| eprintln!("Inkling CUDA used resident expert output convolution"));
            raw
        } else {
            scratch.materialize(None)?
        };
        let mut outputs = outputs.into_iter();
        for (row, raw_row) in rows.iter_mut().zip(raw.chunks_exact(4 * width)) {
            row.taps.capture(visit.state_slot, row.stream.current());
            row.stream.seal(
                raw_row
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|x| f32::from_ne_bytes(*x))
                    .collect(),
            );
            if !device_conv {
                let output = outputs
                    .next()
                    .ok_or_else(|| bad("missing CUDA expert output row"))?;
                self.finish_layer(layer, state, &mut row.stream, output)?;
            }
        }
        static USED: Once = Once::new();
        USED.call_once(|| eprintln!("Inkling CUDA used resident prefill first visit"));
        if std::env::var("FIFI_INKLING_GGUF_PHASE_PROFILE").as_deref() == Ok("1") {
            eprintln!(
                "Inkling CUDA resident first visit {}",
                serde_json::json!({
                    "rows": rows.len(), "mlp_input_h2d_bytes_avoided": raw.len(),
                    "raw_handoff_d2h_bytes": raw.len(), "normalized_embedding_h2d_bytes": raw.len(),
                })
            );
        }
        Ok(true)
    }
}
