//! Group prompt rows over packed experts whose weights are already on the GPU.
use std::collections::BTreeMap;

use super::{
    GgufCuda, bad,
    residency::Weight,
    streaming::{ExpertRows, Matrix, PrefillRoute, activate_rows, group_routes},
};
use crate::{
    EngError,
    expert_tier::{ExpertId, ExpertPayloadSource},
    family::inkling2::cuda_projection::dense::ProjectionBatch,
    kernels::cuda::resident::dense_gemv::DenseDtype,
};

type RouteSlot = (ExpertId, (usize, usize));

struct ResidentPrefill<'a, 'b> {
    matrices: BTreeMap<ExpertId, [Matrix<'a>; 2]>,
    geometry: [(DenseDtype, [usize; 2]); 2],
    routes: &'b [PrefillRoute<'b>],
}

impl GgufCuda {
    pub(in crate::family::inkling2) fn prefills_resident_layer(&self, layer: u32) -> bool {
        self.prefill_resident && !self.resident.placement().has_cold(layer)
    }

    pub(in crate::family::inkling2) fn prefill_resident_experts(
        &self,
        source: &mut impl ExpertPayloadSource,
        layer: u32,
        routes: &[PrefillRoute<'_>],
        dtypes: [DenseDtype; 2],
        activate: impl Fn(f32, f32) -> f32,
    ) -> Result<Option<ExpertRows>, EngError> {
        let groups = group_routes(layer, routes);
        let Some((&first, _)) = groups.first_key_value() else {
            return Ok(None);
        };
        let views = self.resident_prefill_matrices(source, first)?;
        let shapes = views.each_ref().map(|matrix| matrix.shape);
        let tile_rows = self
            .step
            .gguf_rows_capacity(dtypes[0], shapes[0])?
            .min(self.step.gguf_rows_capacity(dtypes[1], shapes[1])?);
        if tile_rows == 0 {
            return Ok(None);
        }
        let mut matrices = BTreeMap::from([(first, views)]);
        for &id in groups.keys().skip(1) {
            matrices.insert(id, self.resident_prefill_matrices(source, id)?);
        }
        // A zero factor skips projection entirely in the per-row path.
        if matrices
            .values()
            .flatten()
            .any(|matrix| matrix.scale == 0.0)
        {
            return Ok(None);
        }
        let prefill = ResidentPrefill {
            matrices,
            geometry: [(dtypes[0], shapes[0]), (dtypes[1], shapes[1])],
            routes,
        };
        let slots = groups
            .into_iter()
            .flat_map(|(id, lanes)| lanes.into_iter().map(move |lane| (id, lane)))
            .collect::<Vec<_>>();
        let mut rows = routes
            .iter()
            .map(|route| vec![Vec::new(); route.ids.len()])
            .collect::<Vec<_>>();
        // The host keeps routed row/rank order even when one expert spans several tiles.
        for tile in slots.chunks(tile_rows) {
            let batch = self.project_resident_tile(&prefill, tile, &activate)?;
            for (&(_, (row, rank)), output) in tile.iter().zip(batch.chunks_exact(shapes[1][0])) {
                rows[row][rank] = output.to_vec();
            }
        }
        Ok(Some(rows))
    }

    pub(super) fn resident_prefill_matrices<'a>(
        &'a self,
        source: &mut impl ExpertPayloadSource,
        id: ExpertId,
    ) -> Result<[Matrix<'a>; 2], EngError> {
        let mut part = |index| {
            source.with_payload(id, index, |view| {
                let Some(Weight::Original(data)) =
                    self.resident.ranges(id, index, [view.data.len(), 0])?
                else {
                    return Err(bad("resident GGUF prefill has no original packed expert"));
                };
                Ok(Matrix {
                    data,
                    shape: view.shape.map(|dim| dim as usize),
                    scale: view.tensor_scale,
                })
            })
        };
        Ok([part(0)?, part(1)?])
    }

    fn project_resident_tile(
        &self,
        prefill: &ResidentPrefill<'_, '_>,
        tile: &[RouteSlot],
        activate: &impl Fn(f32, f32) -> f32,
    ) -> Result<Vec<f32>, EngError> {
        let mut batches = [Vec::new(), Vec::new()];
        for group in tile.chunk_by(|a, b| a.0 == b.0) {
            for (batch, matrix) in batches.iter_mut().zip(&prefill.matrices[&group[0].0]) {
                batch.push(ProjectionBatch {
                    weight: matrix.data,
                    activations: group.len(),
                    scale: matrix.scale,
                });
            }
        }
        let inputs = tile
            .iter()
            .map(|&(_, (row, _))| prefill.routes[row].input)
            .collect::<Vec<_>>();
        let middle = prefill.geometry[1].1[1];
        if middle.checked_mul(2) != Some(prefill.geometry[0].1[0]) {
            return Err(bad(
                "resident GGUF batch has incompatible gate and down widths",
            ));
        }
        let gates = self.step.project_gguf_batches(
            &self.context,
            prefill.geometry[0],
            &batches[0],
            &inputs,
        )?;
        let mid = activate_rows(&gates, middle, activate);
        let activations = mid.chunks_exact(middle).collect::<Vec<_>>();
        self.step.project_gguf_batches(
            &self.context,
            prefill.geometry[1],
            &batches[1],
            &activations,
        )
    }
}
