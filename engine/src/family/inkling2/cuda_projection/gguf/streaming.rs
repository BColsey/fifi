//! Overlap the upload of the next routed expert with projection on the current bank.
use super::{GgufCuda, bad, cache::LayerCache};
use crate::{
    EngError,
    expert_tier::{CudaBankLease, ExpertId, ExpertPayloadSource, Placement, StreamBankPlan},
    kernels::cuda::{DeviceRange, DeviceStream, resident::dense_gemv::DenseDtype},
};

pub(super) struct Matrix<'a> {
    pub(super) data: DeviceRange<'a>,
    pub(super) shape: [usize; 2],
    pub(super) scale: f32,
}

pub(in crate::family::inkling2) struct RoutedExperts<'a> {
    pub ids: &'a [ExpertId],
    pub dtypes: [DenseDtype; 2],
    pub input: &'a [f32],
}

pub(super) fn plan(hidden: usize, ff: usize) -> Result<StreamBankPlan, EngError> {
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
    StreamBankPlan::new(bank, 0, scratch)
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
        let mut caches = self.cache.borrow_mut();
        let mut cache = route.ids.first().and_then(|id| caches.get_mut(&id.layer));
        if let Some(c) = &mut cache {
            c.observe(route.ids)?;
        }
        let mut hits = 0;
        for pair in route.ids.chunks(2) {
            let hit = |id: &ExpertId| cache.as_ref().and_then(|c| c.hit(id.expert));
            let slots = [hit(&pair[0]), pair.get(1).and_then(hit)];
            let a = if slots[0].is_some() {
                None
            } else {
                first.stage_demand(&self.context, source, &pair[..1])?
            };
            let b = if slots[1].is_some() || pair.len() == 1 {
                None
            } else {
                next.stage_demand(&self.context, source, &pair[1..])?
            };
            for ((&id, lease), slot) in pair.iter().zip([a, b]).zip(slots) {
                let output = if let Some(lease) = lease {
                    self.staged_pair(lease, id, &route, &activate, cache.as_deref_mut())?
                } else {
                    let (c, slot) = cache
                        .as_ref()
                        .zip(slot)
                        .ok_or_else(|| bad("missing GGUF expert bank"))?;
                    hits += 1;
                    self.project_pair(&route, &activate, None, |i| c.part(slot, i))?
                };
                combine(id.expert as usize, &output);
            }
        }
        if let (Some(c), Some(id)) = (cache, route.ids.first()) {
            c.record(id.layer, route.ids.len(), hits);
        }
        Ok(true)
    }

    fn staged_pair(
        &self,
        lease: CudaBankLease<'_>,
        id: ExpertId,
        route: &RoutedExperts<'_>,
        activate: &impl Fn(f32, f32) -> f32,
        cache: Option<&mut LayerCache>,
    ) -> Result<Vec<f32>, EngError> {
        let output = self.project_pair(route, activate, Some(lease.stream()), |index| {
            let part = lease.part(id, index)?;
            Ok(Matrix {
                data: part.data,
                shape: part.shape.map(|d| d as usize),
                scale: part.tensor_scale,
            })
        })?;
        match cache {
            Some(layer) => layer.finish(&self.context, lease, id)?,
            None => lease.finish()?,
        }
        Ok(output)
    }

    fn project_pair<'a>(
        &self,
        route: &RoutedExperts<'_>,
        activate: &impl Fn(f32, f32) -> f32,
        stream: Option<&DeviceStream>,
        matrix: impl Fn(usize) -> Result<Matrix<'a>, EngError>,
    ) -> Result<Vec<f32>, EngError> {
        let project = |part: usize, x: &[f32]| {
            let view = matrix(part)?;
            self.step.project_gguf(
                &self.context,
                stream,
                (route.dtypes[part], view.data, view.shape, view.scale),
                x,
            )
        };
        let gates = project(0, route.input)?;
        // The GGUF stores complete gate rows followed by complete up rows.
        let (gate, up) = gates.split_at(gates.len() / 2);
        let mid: Vec<_> = gate.iter().zip(up).map(|(&g, &u)| activate(g, u)).collect();
        project(1, &mid)
    }
}
