//! Missing controller matrices for the token chain only. Preserve the normal
//! placement and its cold staging for prefill and host fallback. Check the
//! device-prefill growth floor and cap the incremental charge before upload.
use super::*;
impl GgufCuda {
    pub(in crate::family::inkling2) fn token_growth_reserve(
        &self,
        spec: &crate::family::inkling2::TextSpec,
        context: usize,
    ) -> Option<usize> {
        controller_residency::expert_workspace(spec, context, self.device_prefill.is_some())
            .and_then(controller_residency::workspace_floor)
            .and_then(|b| b.try_into().ok())
    }
}

pub(super) fn load(
    ctx: &CudaContext,
    optional: &[GgufDenseUpload<'_>],
    ordinary: &Projections,
    workspace: Option<u64>,
) -> Result<Projections, EngError> {
    let Some(floor) = workspace.and_then(controller_residency::workspace_floor) else {
        return Ok(Projections::new());
    };
    let mut out = Projections::new();
    let missing: Vec<_> = optional
        .iter()
        .filter(|w| {
            w.dtype == DenseDtype::F32
                && w.matrix.name.contains(".controllers.")
                && !ordinary.contains_key(w.matrix.name)
        })
        .collect();
    let bytes = missing
        .iter()
        .map(|w| w.matrix.codes.len() as u64)
        .sum::<u64>();
    if missing.is_empty()
        || bytes > 64 << 20
        || (ctx.memory_info()?.free as u64).saturating_sub(bytes) < floor
    {
        return Ok(out);
    }
    for w in missing {
        match upload_projections(ctx, std::slice::from_ref(w)) {
            Ok(item) => out.extend(item),
            Err(EngError::MemoryPlan(_)) => return Ok(Projections::new()),
            Err(e) => return Err(e),
        }
    }
    if (ctx.memory_info()?.free as u64) < floor {
        return Ok(Projections::new());
    }
    eprintln!(
        "Inkling CUDA token-only controllers: {} matrices, {bytes} payload bytes",
        out.len()
    );
    Ok(out)
}
