//! Overlap the upload of the next routed expert with projection on the current bank.
use super::{GgufCuda, bad};
use crate::{
    EngError,
    expert_tier::{CudaBankLease, ExpertId, ExpertPayloadSource, Placement, StreamBankPlan},
    kernels::cuda::resident::dense_gemv::DenseDtype,
};

pub(in crate::family::inkling2) struct RoutedExperts<'a> {
    pub ids: &'a [ExpertId],
    pub dtypes: [DenseDtype; 2],
    pub input: &'a [f32],
}

pub(super) fn plan(hidden: usize, ff: usize) -> Result<Option<StreamBankPlan>, EngError> {
    if std::env::var("FIFI_INKLING_CUDA_STREAM_EXPERTS").as_deref() != Ok("1") {
        return Ok(None);
    }
    // BF16 gate/up and down are the largest pair. Part starts are 256-byte aligned.
    let matrix = hidden
        .checked_mul(ff)
        .and_then(|n| n.checked_mul(2))
        .ok_or_else(|| bad("GGUF stream matrix size overflows"))? as u64;
    let bank = matrix
        .checked_mul(3)
        .and_then(|n| n.checked_add(255))
        .ok_or_else(|| bad("GGUF stream bank size overflows"))?;
    // The transport validates the largest borrowed part and metadata per slot.
    let scratch = matrix
        .checked_mul(4)
        .and_then(|n| n.checked_add(4096))
        .ok_or_else(|| bad("GGUF stream scratch size overflows"))?;
    StreamBankPlan::new(bank, 0, scratch).map(Some)
}

impl GgufCuda {
    pub(in crate::family::inkling2) fn streams_layer(&self, layer: u32) -> bool {
        self.stream_banks.is_some() && self.resident.placement().has_cold(layer)
    }

    pub(in crate::family::inkling2) fn stream_experts(
        &self,
        source: &mut impl ExpertPayloadSource,
        route: RoutedExperts<'_>,
        activate: impl Fn(f32, f32) -> f32,
        mut combine: impl FnMut(usize, &[f32]),
    ) -> Result<bool, EngError> {
        let Some(banks) = &self.stream_banks else {
            return Ok(false);
        };
        if route
            .ids
            .iter()
            .any(|&id| self.resident.placement().placement(id) != Placement::Nvme)
        {
            return Ok(false);
        }
        let mut banks = banks.borrow_mut();
        let (first, next) = banks.slots_mut();
        for pair in route.ids.chunks(2) {
            let a = first
                .stage_demand(&self.context, source, &pair[..1])?
                .ok_or_else(|| bad("missing first GGUF stream bank"))?;
            let b = next.stage_demand(&self.context, source, &pair[1..])?;
            let mut consume = |lease, id: ExpertId| -> Result<(), EngError> {
                let output = self.streamed_pair(lease, id, &route, &activate)?;
                combine(id.expert as usize, &output);
                Ok(())
            };
            consume(a, pair[0])?;
            if let Some(b) = b {
                consume(b, pair[1])?;
            }
        }
        Ok(true)
    }

    fn streamed_pair(
        &self,
        lease: CudaBankLease<'_>,
        id: ExpertId,
        route: &RoutedExperts<'_>,
        activate: &impl Fn(f32, f32) -> f32,
    ) -> Result<Vec<f32>, EngError> {
        let project = |part: usize, x: &[f32]| {
            let view = lease.part(id, part)?;
            if view.tensor_scale == 0.0 {
                return Ok(vec![0.0; view.shape[0] as usize]);
            }
            let mut values = self.step.project_streamed(
                &self.context,
                lease.stream(),
                route.dtypes[part],
                view.data,
                view.shape.map(|v| v as usize),
                x,
            )?;
            if view.tensor_scale != 1.0 {
                for value in &mut values {
                    *value *= view.tensor_scale;
                    if !value.is_finite() {
                        return Err(bad("streamed GGUF expert scale overflowed"));
                    }
                }
            }
            Ok::<_, EngError>(values)
        };
        let gates = project(0, route.input)?;
        // The GGUF stores complete gate rows followed by complete up rows.
        let (gate, up) = gates.split_at(gates.len() / 2);
        let mid: Vec<_> = gate.iter().zip(up).map(|(&g, &u)| activate(g, u)).collect();
        let output = project(1, &mid)?;
        lease.finish()?;
        Ok(output)
    }
}
