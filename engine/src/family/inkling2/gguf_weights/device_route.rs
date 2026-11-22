use super::GgufTextWeights;
use crate::{
    EngError,
    family::inkling2::{
        cuda_projection::gguf::router::RouteInput,
        phase_profile::{PhaseProfile, Work},
    },
};

impl GgufTextWeights {
    pub(in crate::family::inkling2) fn device_routed_experts(
        &self,
        prefix: &str,
        route: RouteInput<'_>,
        profile: Option<&PhaseProfile>,
    ) -> Result<Option<Vec<f32>>, EngError> {
        let Some(cuda) = self.cuda.as_ref().filter(|_| self.ram.is_none()) else {
            return Ok(None);
        };
        if !cuda.has_resident_route(route.layer) {
            return Ok(None);
        }
        let gate = format!("{prefix}.experts.w13_weight");
        let down = format!("{prefix}.experts.w2_weight");
        let dtype = |name| {
            let (tensor, _) = self.tensor(name)?;
            Ok::<_, EngError>(
                cuda.expert_dtype(tensor.kind)
                    .filter(|dtype| dtype.is_packed_gguf()),
            )
        };
        let [Some(gate_dtype), Some(down_dtype)] = [dtype(&gate)?, dtype(&down)?] else {
            return Ok(None);
        };
        // This path measures the router and experts as one resident backend call.
        let _span = profile.and_then(|profile| profile.span(Work::Experts));
        cuda.routed_expert_group(prefix, route, [gate_dtype, down_dtype], |experts| {
            self.vec(&format!("{prefix}.gate.bias"), experts)
        })
    }
}
