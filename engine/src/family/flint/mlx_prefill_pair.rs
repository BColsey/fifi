//! Two-piece disk prefill with one expert stage per layer.

use super::{super::weights::FlintCheckpoint, *};
use crate::family::prefill::mlx::{local_expert_ids, sorted_expert_ids, submit_shared_expert};

impl Model {
    pub(in crate::family::flint) fn prefill_pair(
        &mut self,
        state: &mut State,
        tokens: &[u32],
        split: usize,
    ) -> Result<()> {
        let Some((first, second)) = tokens.split_at_checked(split) else {
            return Err(EngError::BadRequest(
                "Flint prefill pair split is outside the input".into(),
            ));
        };
        if state.pos != 0
            || !state.valid
            || !(2..=1024).contains(&first.len())
            || !(2..=1024).contains(&second.len())
            || tokens.iter().any(|&id| id >= 150_144)
        {
            return Err(EngError::BadRequest(
                "Flint prefill pair has invalid state or rows".into(),
            ));
        }
        let rows = [first.len() as u32, second.len() as u32];
        ChunkCachePlan::new(0, rows[0], rows[1], context_limit())?;
        #[cfg(feature = "native-fault-qualification")]
        let timing = super::super::mlx_expert_timing::ForwardTiming::begin(&self.kernels, 0);
        // Once writes begin, an execution failure requires the normal session reset.
        state.valid = false;
        self.forward_pair(state, [first, second], rows)?;
        state.pos = rows.iter().sum();
        state.valid = true;
        #[cfg(feature = "native-fault-qualification")]
        if let Some(timing) = timing {
            timing.finish_group(&self.kernels, state.pos, state.full_capacity, Some(rows));
        }
        Ok(())
    }

    fn forward_pair(
        &mut self,
        state: &mut State,
        tokens: [&[u32]; 2],
        rows: [u32; 2],
    ) -> Result<()> {
        let k = &mut self.kernels;
        state.ensure_full_capacity(k, rows.iter().sum())?;
        let [first, second] = tokens;
        let mut first = k.import_tensor(&super::super::precision::embedding(
            &self.source.embedding_rows(first)?,
        )?)?;
        let mut second = k.import_tensor(&super::super::precision::embedding(
            &self.source.embedding_rows(second)?,
        )?)?;
        let mut first_health = Vec::new();
        let mut second_health = Vec::new();
        for (index, (layer, plane)) in self.layers.iter().zip(&mut state.planes).enumerate() {
            let plan = ChunkCachePlan::new(index, 0, rows[0], state.full_capacity)?;
            let first_input = FfnInput::prepare(k, layer, &first, plane, &plan)?;
            let plan = ChunkCachePlan::new(index, rows[0], rows[1], state.full_capacity)?;
            let second_input = FfnInput::prepare(k, layer, &second, plane, &plan)?;
            if let Ffn::Moe {
                router,
                routed: Routed::Disk { source, stem, .. },
                shared,
            } = &layer.ffn
            {
                let a = RoutedInput::prepare(k, first_input, router)?;
                let b = RoutedInput::prepare(k, second_input, router)?;
                let (a_out, b_out) = shared_stage(k, source, stem, shared, &a, &b)?;
                first_health.push(a.route.health);
                second_health.push(b.route.health);
                first = a_out;
                second = b_out;
            } else {
                first = first_input.finish(k, &layer.ffn, &mut first_health)?;
                second = second_input.finish(k, &layer.ffn, &mut second_health)?;
            }
        }
        k.evaluate(&[&first, &second])?;
        mlx_route::read_step(k, None, &first_health)?;
        mlx_route::read_step(k, None, &second_health)?;
        Ok(())
    }
}

struct FfnInput {
    normalized: ExecutionTensor,
    residual: ExecutionTensor,
}

impl FfnInput {
    fn prepare(
        k: &mut MlxKernels,
        layer: &Layer,
        x: &ExecutionTensor,
        plane: &mut Option<Plane>,
        plan: &ChunkCachePlan,
    ) -> Result<Self> {
        let normalized = k.rms_norm(x, &layer.input_norm, 1e-6)?;
        let attended = attention(k, layer, &normalized, plane, plan)?;
        let residual = k.binary(BinaryOp::Add, x, &attended)?;
        let normalized = k.rms_norm(&residual, &layer.post_norm, 1e-6)?;
        Ok(Self {
            normalized,
            residual,
        })
    }

    fn finish(
        self,
        k: &mut MlxKernels,
        weights: &Ffn,
        health: &mut Vec<ExecutionTensor>,
    ) -> Result<ExecutionTensor> {
        let (output, route_health) = ffn(k, weights, &self.normalized)?;
        health.extend(route_health);
        k.binary(BinaryOp::Add, &self.residual, &output)
    }
}

struct RoutedInput {
    ffn: FfnInput,
    route: mlx_route::Route,
    expert_ids: Vec<u32>,
}

impl RoutedInput {
    fn prepare(k: &mut MlxKernels, ffn: FfnInput, router: &ExecutionTensor) -> Result<Self> {
        let logits = k.mm(&ffn.normalized, router)?;
        let route = mlx_route::select(k, &logits)?;
        let expert_ids = host_expert_ids(&k.readback(route.ids.ids())?)?;
        Ok(Self {
            ffn,
            route,
            expert_ids,
        })
    }

    fn finish(
        &self,
        k: &mut MlxKernels,
        weights: &SwiGlu,
        ids: &ExecutionTensor,
        experts: u32,
        shared: &ExecutionTensor,
    ) -> Result<ExecutionTensor> {
        let ids = k.bound_indices(ids, experts)?;
        let routed = grouped_experts(k, weights, &self.ffn.normalized, &ids)?;
        let output = precision_native::combine(k, &routed, &self.route.weights, shared)?;
        let output = k.binary(BinaryOp::Add, &self.ffn.residual, &output)?;
        k.evaluate(&[&output])?;
        Ok(output)
    }
}

fn shared_stage(
    k: &mut MlxKernels,
    source: &FlintCheckpoint,
    stem: &str,
    shared: &SwiGlu,
    first: &RoutedInput,
    second: &RoutedInput,
) -> Result<(ExecutionTensor, ExecutionTensor)> {
    let selected = [&first.expert_ids[..], &second.expert_ids[..]].concat();
    let experts = sorted_expert_ids(&selected);
    let local = local_expert_ids(&selected, &experts)?;
    let total = local.len() as u32;
    let split = first.expert_ids.len() as u32;
    let local = Tensor::try_from_host(
        Dtype::U32,
        vec![total],
        local.into_iter().flat_map(u32::to_le_bytes).collect(),
    )?;
    let first_shared = shared.apply(k, &first.ffn.normalized)?;
    let second_shared = shared.apply(k, &second.ffn.normalized)?;
    let first_shared = submit_shared_expert(k, first_shared)?;
    let second_shared = submit_shared_expert(k, second_shared)?;
    let rows = first.ffn.normalized.shape()[0] + second.ffn.normalized.shape()[0];
    let (weights, local, mut pages) = SwiGlu::load_expert_pages(
        k,
        source,
        stem,
        &experts,
        &local,
        expert_read_runs().coalesce(rows),
    )?;
    let first_ids = k.slice(&local, &[0], &[split])?;
    let first = first.finish(k, &weights, &first_ids, experts.len() as u32, &first_shared)?;
    let second_ids = k.slice(&local, &[split], &[total])?;
    let second = second.finish(
        k,
        &weights,
        &second_ids,
        experts.len() as u32,
        &second_shared,
    )?;
    drop((first_ids, second_ids, weights, local));
    pages.release();
    k.evaluate(&[&second])?;
    if pages.in_use() {
        return Err(EngError::Backend {
            operation: "flint prefill pair",
            message: "shared expert pages did not retire".into(),
        });
    }
    #[cfg(feature = "native-fault-qualification")]
    trace_expert_accesses(stem, rows, &experts);
    Ok((first, second))
}
