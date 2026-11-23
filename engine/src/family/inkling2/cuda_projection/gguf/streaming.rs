//! Overlap the upload of the next routed expert with projection on the current bank.
use std::collections::BTreeMap;

use super::{ExpertLayer, GgufCuda, bad, cache::LayerCache};
use crate::{
    EngError,
    expert_tier::{
        CudaBankLease, CudaStreamBanks, ExpertId, ExpertPayloadSource, Placement, StreamBankPlan,
    },
    kernels::cuda::{
        DeviceRange, DeviceStream,
        resident::dense_gemv::{DenseDtype, GgufMoePick, GgufMoeWeights},
    },
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

pub(in crate::family::inkling2) struct ResidentRoute<'a> {
    pub layer: u32,
    pub ids: &'a [usize],
    pub input: &'a [f32],
    pub caps: [f32; 2],
}

pub(in crate::family::inkling2) struct ExpertGroup<'a> {
    pub(in crate::family::inkling2) route: ResidentRoute<'a>,
    pub(in crate::family::inkling2) dtypes: [DenseDtype; 2],
    pub(in crate::family::inkling2) picks: &'a [GgufMoePick],
}

pub(in crate::family::inkling2) struct PrefillRoute<'a> {
    pub ids: &'a [usize],
    pub input: &'a [f32],
}

#[derive(Default)]
struct StreamTransfers {
    hits: usize,
    uploads: usize,
    bytes: u64,
}

struct ExpertMatrices<'a> {
    stream: Option<&'a DeviceStream>,
    parts: [Matrix<'a>; 2],
}

impl<'a> ExpertMatrices<'a> {
    fn staged(lease: &'a CudaBankLease<'_>, id: ExpertId) -> Result<Self, EngError> {
        let part = |index| -> Result<Matrix<'a>, EngError> {
            let p = lease.part(id, index)?;
            Ok(Matrix {
                data: p.data,
                shape: p.shape.map(|d| d as usize),
                scale: p.tensor_scale,
            })
        };
        Ok(Self {
            stream: Some(lease.stream()),
            parts: [part(0)?, part(1)?],
        })
    }
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

pub(super) fn group_plan(
    layers: &[ExpertLayer<'_>],
    experts: usize,
    count: usize,
    serial: StreamBankPlan,
) -> Result<StreamBankPlan, EngError> {
    let bytes = layers
        .iter()
        .try_fold(serial.bank_bytes(), |largest, layer| {
            let spans = layer.parts.each_ref().map(|part| part.bytes.len());
            let group = aligned_group_bytes(spans, experts, count)
                .ok_or_else(|| bad("invalid GGUF expert group extent"))?;
            Ok::<_, EngError>(largest.max(group as u64))
        })?;
    // Each routed group visits one layer. GGUF scale metadata is scalar, so the
    // transport stores only the aligned gate/up and down matrix payloads.
    let scratch = serial.host_bytes() - 2 * serial.bank_bytes();
    StreamBankPlan::new(bytes, 0, scratch)
}

fn aligned_group_bytes(stack_bytes: [usize; 2], experts: usize, count: usize) -> Option<usize> {
    let per_expert = stack_bytes.into_iter().try_fold(0usize, |sum, bytes| {
        let stride = bytes.checked_div(experts)?;
        if stride == 0 || stride.checked_mul(experts)? != bytes {
            return None;
        }
        sum.checked_add(stride.checked_next_multiple_of(CudaStreamBanks::PAYLOAD_ALIGNMENT)?)
    })?;
    per_expert.checked_mul(count)
}

impl GgufCuda {
    pub(in crate::family::inkling2) fn streams_layer(&self, layer: u32) -> bool {
        self.stream_banks.is_some() && self.resident.placement().has_cold(layer)
    }

    pub(super) fn stream_expert_group(
        &self,
        source: &mut impl ExpertPayloadSource,
        group: &ExpertGroup<'_>,
    ) -> Result<Vec<f32>, EngError> {
        let layer = group.route.layer;
        let ids: Vec<_> = group
            .picks
            .iter()
            .map(|pick| ExpertId {
                layer,
                expert: pick.expert,
            })
            .collect();
        let mut caches = self.cache.borrow_mut();
        let mut cache = caches.get_mut(&layer);
        if let Some(c) = cache.as_deref_mut() {
            c.observe(&ids)?;
        }
        let slots: Vec<_> = ids
            .iter()
            .map(|id| cache.as_ref().and_then(|c| c.hit(id.expert)))
            .collect();
        let misses: Vec<_> = ids
            .iter()
            .zip(&slots)
            .filter_map(|(&id, slot)| slot.is_none().then_some(id))
            .collect();
        let mut banks = self
            .stream_banks
            .as_ref()
            .ok_or_else(|| bad("no GGUF stream banks"))?
            .borrow_mut();
        let (bank, _) = banks.slots_mut();
        let lease = if misses.is_empty() {
            None
        } else {
            bank.stage_demand(&self.context, source, &misses)?
        };
        let stream = lease.as_ref().map(CudaBankLease::stream);
        let matrices = gather_matrices(&ids, &slots, cache.as_deref(), lease.as_ref())?;
        let output = self.run_expert_group(stream, group, GgufMoeWeights::Separate(&matrices))?;
        let transfers = StreamTransfers {
            hits: ids.len() - misses.len(),
            uploads: misses.len(),
            bytes: lease.as_ref().map_or(0, |bank| bank.bytes() as u64),
        };
        if let Some(lease) = lease {
            if let Some(c) = cache.as_deref_mut() {
                c.finish_many(&self.context, lease, &misses)?;
            } else {
                lease.finish()?;
            }
        }
        if let Some(c) = cache {
            c.record(layer, ids.len(), transfers.hits);
        }
        self.record_transfer(&transfers, std::iter::repeat_n(1, ids.len()));
        Ok(output)
    }

    pub(in crate::family::inkling2) fn stream_experts(
        &self,
        source: &mut impl ExpertPayloadSource,
        route: RoutedExperts<'_>,
        activate: impl Fn(f32, f32) -> f32,
        mut combine: impl FnMut(usize, &[f32]),
    ) -> Result<bool, EngError> {
        let Some(layer) = route.ids.first().map(|id| id.layer) else {
            return Ok(false);
        };
        if !self.can_stream(route.ids) {
            return Ok(false);
        }
        let mut caches = self.cache.borrow_mut();
        let mut cache = caches.get_mut(&layer);
        if let Some(c) = cache.as_deref_mut() {
            c.observe(route.ids)?;
        }
        let transfers =
            self.visit_experts(source, route.ids, cache.as_deref_mut(), |id, matrices| {
                let out = self.project_pair(&matrices, route.dtypes, route.input, &activate)?;
                combine(id.expert as usize, &out);
                Ok(())
            })?;
        if let Some(c) = cache {
            c.record(layer, route.ids.len(), transfers.hits);
        }
        self.record_transfer(&transfers, std::iter::repeat_n(1, route.ids.len()));
        Ok(true)
    }

    pub(in crate::family::inkling2) fn has_stream_banks(&self) -> bool {
        self.stream_banks.is_some()
    }

    fn can_stream(&self, ids: &[ExpertId]) -> bool {
        self.has_stream_banks()
            && ids
                .iter()
                .all(|&id| self.resident.placement().placement(id) == Placement::Nvme)
    }

    pub(in crate::family::inkling2) fn prefill_experts(
        &self,
        source: &mut impl ExpertPayloadSource,
        layer: u32,
        routes: &[PrefillRoute<'_>],
        dtypes: [DenseDtype; 2],
        activate: impl Fn(f32, f32) -> f32,
    ) -> Result<Option<ExpertRows>, EngError> {
        let groups = group_routes(layer, routes);
        let ids = groups.keys().copied().collect::<Vec<_>>();
        if !self.can_stream(&ids) {
            return Ok(None);
        }
        let mut cache = self.cache.borrow_mut();
        let mut cache = cache.get_mut(&layer);
        if let Some(c) = cache.as_deref_mut() {
            for route in routes {
                let ids = route
                    .ids
                    .iter()
                    .map(|&expert| ExpertId {
                        layer,
                        expert: expert as u32,
                    })
                    .collect::<Vec<_>>();
                c.observe(&ids)?;
            }
        }
        // Store each result at its routing rank so each token adds contributions
        // in the original order, irrespective of the order we stage experts.
        let mut rows: Vec<_> = routes
            .iter()
            .map(|route| vec![Vec::new(); route.ids.len()])
            .collect();
        let transfers = self.visit_experts(source, &ids, cache, |id, matrices| {
            let [gate, down] = &matrices.parts;
            let tile = self
                .step
                .gguf_rows_capacity(dtypes[0], gate.shape)?
                .min(self.step.gguf_rows_capacity(dtypes[1], down.shape)?)
                .max(1);
            for lanes in groups[&id].chunks(tile) {
                if lanes.len() == 1 {
                    let (row, rank) = lanes[0];
                    rows[row][rank] =
                        self.project_pair(&matrices, dtypes, routes[row].input, &activate)?;
                } else {
                    let batch =
                        self.project_prefill_rows(&matrices, dtypes, lanes, routes, &activate)?;
                    for (&(row, rank), output) in
                        lanes.iter().zip(batch.chunks_exact(down.shape[0]))
                    {
                        rows[row][rank] = output.to_vec();
                    }
                }
            }
            Ok(())
        })?;
        self.record_transfer(&transfers, groups.values().map(Vec::len));
        Ok(Some(rows))
    }

    pub(in crate::family::inkling2) fn prefill_bank_report(
        &self,
    ) -> Option<crate::sched::job::PrefillBankReport> {
        self.prefill_report
            .as_ref()
            .map(|report| report.borrow().clone())
    }

    fn record_transfer(
        &self,
        transfers: &StreamTransfers,
        routed_rows: impl Iterator<Item = usize>,
    ) {
        let Some(report) = &self.prefill_report else {
            return;
        };
        let mut report = report.borrow_mut();
        report.passes += 1;
        report.misses += transfers.uploads as u64;
        report.upload_bytes += transfers.bytes;
        if transfers.uploads > 0 {
            note_bucket(&mut report.miss_histogram, transfers.uploads);
        }
        for rows in routed_rows {
            note_bucket(&mut report.route_histogram, rows);
        }
    }

    fn visit_experts(
        &self,
        source: &mut impl ExpertPayloadSource,
        ids: &[ExpertId],
        mut cache: Option<&mut LayerCache>,
        mut execute: impl FnMut(ExpertId, ExpertMatrices<'_>) -> Result<(), EngError>,
    ) -> Result<StreamTransfers, EngError> {
        let mut banks = self
            .stream_banks
            .as_ref()
            .ok_or_else(|| bad("no GGUF stream banks"))?
            .borrow_mut();
        let (first, next) = banks.slots_mut();
        let mut transfers = StreamTransfers::default();
        for pair in ids.chunks(2) {
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
                if let Some(lease) = lease {
                    execute(id, ExpertMatrices::staged(&lease, id)?)?;
                    transfers.uploads += 1;
                    transfers.bytes += lease.bytes() as u64;
                    match cache.as_deref_mut() {
                        Some(c) => c.finish(&self.context, lease, id)?,
                        None => lease.finish()?,
                    }
                } else {
                    let (c, slot) = cache
                        .as_ref()
                        .zip(slot)
                        .ok_or_else(|| bad("missing GGUF expert bank"))?;
                    execute(
                        id,
                        ExpertMatrices {
                            stream: None,
                            parts: [c.part(slot, 0)?, c.part(slot, 1)?],
                        },
                    )?;
                    transfers.hits += 1;
                }
            }
        }
        Ok(transfers)
    }

    fn project_prefill_rows(
        &self,
        matrices: &ExpertMatrices<'_>,
        dtypes: [DenseDtype; 2],
        lanes: &[(usize, usize)],
        routes: &[PrefillRoute<'_>],
        activate: &impl Fn(f32, f32) -> f32,
    ) -> Result<Vec<f32>, EngError> {
        let [gate, down] = &matrices.parts;
        let middle = down.shape[1];
        if gate.shape[0] != 2 * middle {
            return Err(bad(
                "streamed expert batch has incompatible gate and down widths",
            ));
        }

        let project = |part: usize, input: &[&[f32]]| {
            let view = &matrices.parts[part];
            self.step.project_gguf_rows(
                &self.context,
                matrices.stream,
                (dtypes[part], view.data, view.shape, view.scale),
                input,
            )
        };
        let inputs: Vec<_> = lanes.iter().map(|&(row, _)| routes[row].input).collect();
        let gates = project(0, &inputs)?;
        let mid = activate_rows(&gates, middle, activate);
        let activations = mid.chunks_exact(middle).collect::<Vec<_>>();
        project(1, &activations)
    }

    fn project_pair(
        &self,
        weights: &ExpertMatrices<'_>,
        dtypes: [DenseDtype; 2],
        input: &[f32],
        activate: &impl Fn(f32, f32) -> f32,
    ) -> Result<Vec<f32>, EngError> {
        let project = |part: usize, x: &[f32]| {
            let view = &weights.parts[part];
            self.step.project_gguf(
                &self.context,
                weights.stream,
                (dtypes[part], view.data, view.shape, view.scale),
                x,
            )
        };
        let gates = project(0, input)?;
        // The GGUF stores complete gate rows followed by complete up rows.
        let (gate, up) = gates.split_at(gates.len() / 2);
        let mid: Vec<_> = gate.iter().zip(up).map(|(&g, &u)| activate(g, u)).collect();
        project(1, &mid)
    }
}

fn gather_matrices<'a>(
    ids: &[ExpertId],
    slots: &[Option<usize>],
    cache: Option<&'a LayerCache>,
    lease: Option<&'a CudaBankLease<'_>>,
) -> Result<Vec<[DeviceRange<'a>; 2]>, EngError> {
    ids.iter()
        .zip(slots)
        .map(|(&id, &slot)| {
            if let Some(slot) = slot {
                let c = cache.ok_or_else(|| bad("missing GGUF expert cache"))?;
                Ok([c.part(slot, 0)?.data, c.part(slot, 1)?.data])
            } else {
                let bank = lease.ok_or_else(|| bad("missing GGUF expert staging bank"))?;
                Ok([bank.part(id, 0)?.data, bank.part(id, 1)?.data])
            }
        })
        .collect()
}

fn note_bucket(histogram: &mut Vec<u64>, count: usize) {
    let bucket = count.next_power_of_two().ilog2().min(12) as usize;
    if histogram.len() <= bucket {
        histogram.resize(bucket + 1, 0);
    }
    histogram[bucket] += 1;
}

pub(super) fn activate_rows(
    gates: &[f32],
    middle: usize,
    activate: &impl Fn(f32, f32) -> f32,
) -> Vec<f32> {
    let mut result = Vec::with_capacity(gates.len() / 2);
    for combined in gates.chunks_exact(2 * middle) {
        let (gate, up) = combined.split_at(middle);
        result.extend(gate.iter().zip(up).map(|(&g, &u)| activate(g, u)));
    }
    result
}

pub(in crate::family::inkling2) type ExpertRows = Vec<Vec<Vec<f32>>>;

pub(super) fn group_routes(
    layer: u32,
    routes: &[PrefillRoute<'_>],
) -> BTreeMap<ExpertId, Vec<(usize, usize)>> {
    group_route_ids(layer, routes.iter().map(|route| route.ids))
}

pub(super) fn group_route_ids<'a>(
    layer: u32,
    ids: impl IntoIterator<Item = &'a [usize]>,
) -> BTreeMap<ExpertId, Vec<(usize, usize)>> {
    let mut groups: BTreeMap<_, Vec<_>> = BTreeMap::new();
    for (row, ids) in ids.into_iter().enumerate() {
        for (rank, &expert) in ids.iter().enumerate() {
            groups
                .entry(ExpertId {
                    layer,
                    expert: expert as u32,
                })
                .or_default()
                .push((row, rank));
        }
    }
    groups
}

#[test]
fn packed_group_reserves_aligned_matrix_parts_and_checks_extents() {
    assert_eq!(aligned_group_bytes([2 * 257, 2 * 255], 2, 8), Some(8 * 768));
    assert_eq!(aligned_group_bytes([515, 510], 2, 8), None);
    assert_eq!(aligned_group_bytes([512, 512], 0, 8), None);
    assert_eq!(aligned_group_bytes([512, 512], 2, usize::MAX), None);
}

#[test]
fn grouped_routes_keep_the_rank_of_each_expert_for_each_row() {
    let a = PrefillRoute {
        ids: &[7, 4],
        input: &[],
    };
    let b = PrefillRoute {
        ids: &[5, 7],
        input: &[],
    };
    let groups = group_routes(3, &[a, b]);
    assert_eq!(groups.len(), 3);
    assert_eq!(
        groups[&ExpertId {
            layer: 3,
            expert: 7
        }],
        [(0, 0), (1, 1)]
    );
}
