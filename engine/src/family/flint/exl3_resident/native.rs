//! Borrowed native views. Only the loader may construct the fully initialized owner.

mod host;
mod load;

use super::packing;

pub(crate) use host::BankStats;

#[cfg(test)]
pub(crate) use load::load_flint_fixture;
pub(crate) use load::load_flint_resident;

use super::plan::{ABSENT_MEMBER, check_member_offsets};
use super::{FfnRoles, ResidentPlan};
use crate::{
    EngError,
    family::project::cuda::Exl3MatrixView,
    kernels::cuda::{CudaContext, DeviceBuffer, DeviceRange},
    weights::exl3::Codebook,
};

pub(crate) struct FlintExl3Resident {
    buffer: DeviceBuffer,
    host: Option<host::HostOwner>,
    /// Per layer, what the id-indexed launch reads (`None`: the layer declined); worked out on
    /// first use, resident loads only.
    pick_tables: Vec<std::cell::OnceCell<Option<PickEntry>>>,
    matrices: Vec<PackedMatrix>,
    plain: Vec<PackedRange>,
    roles: ResidentPlan,
    model_config: Vec<u8>,
}

/// A layer's id-addressed experts: `spans` is, per role (gate, up, down) and part (trellis, suh,
/// svh), the device range from expert 0 to the end of the last expert, and `table` the 32-bit
/// byte offset of each expert from expert 0.
struct PickEntry {
    members: u32,
    spans: [(usize, usize); 9],
    table: DeviceBuffer,
}

type PickLayout = (Vec<u32>, [(usize, usize); 9]);

pub(super) use super::packing::PackedRange;

#[derive(Clone, Copy)]
struct PackedMatrix {
    trellis: PackedRange,
    suh: PackedRange,
    svh: PackedRange,
    bias: Option<PackedRange>,
    shape: [usize; 2],
    bits_twice: u8,
    book: Codebook,
}

pub(crate) struct FlintExpertBank<'a> {
    owner: &'a FlintExl3Resident,
    indices: &'a [[usize; 3]],
    layer: usize,
    prepared: Option<&'a [Option<usize>; 256]>,
}

#[derive(Clone, Copy)]
pub(crate) struct FlintExpertView<'a> {
    pub(crate) gate: Exl3MatrixView<'a>,
    pub(crate) up: Exl3MatrixView<'a>,
    pub(crate) down: Exl3MatrixView<'a>,
}

pub(crate) enum FlintFfnView<'a> {
    Dense(FlintExpertView<'a>),
    Moe {
        router: DeviceRange<'a>,
        experts: FlintExpertBank<'a>,
        shared: FlintExpertView<'a>,
        routing_weight_multiplier: f32,
    },
}

/// One projection role (gate, up or down) of every expert of a layer, as ranges that start at
/// expert 0 and run to the end of the last expert.
pub(crate) struct PickRole<'a> {
    pub(crate) trellis: DeviceRange<'a>,
    pub(crate) suh: DeviceRange<'a>,
    pub(crate) svh: DeviceRange<'a>,
}

/// A layer's experts addressed by id on the device: the three roles and the table of each
/// expert's byte offset from expert 0 (the same offset moves the trellis and both scale vectors).
pub(crate) struct PickBank<'a> {
    pub(crate) members: u32,
    pub(crate) table: DeviceRange<'a>,
    pub(crate) roles: [PickRole<'a>; 3],
}

pub(crate) struct FlintLayerView<'a> {
    pub(crate) q: Exl3MatrixView<'a>,
    pub(crate) k: Exl3MatrixView<'a>,
    pub(crate) v: Exl3MatrixView<'a>,
    pub(crate) o: Exl3MatrixView<'a>,
    pub(crate) input_norm: DeviceRange<'a>,
    pub(crate) post_attention_norm: DeviceRange<'a>,
    pub(crate) q_norm: DeviceRange<'a>,
    pub(crate) k_norm: DeviceRange<'a>,
    pub(crate) sinks: DeviceRange<'a>,
    pub(crate) ffn: FlintFfnView<'a>,
}

impl FlintExl3Resident {
    pub(crate) fn model_config(&self) -> &[u8] {
        &self.model_config
    }

    pub(crate) fn layer_count(&self) -> usize {
        self.roles.layers.len()
    }

    pub(crate) fn device_weight_bytes(&self) -> u64 {
        self.buffer.bytes() as u64
    }

    /// Copies made into the bank so far, or `None` when every expert is resident.
    pub(crate) fn bank_stats(&self) -> Option<BankStats> {
        self.host.as_ref().map(|host| host.stats.get())
    }

    /// Device bytes of the pack with every expert resident, for the advice of a refused load.
    pub(crate) fn resident_weight_bytes(path: &std::path::Path) -> Result<u64, EngError> {
        load::resident_weight_bytes(path)
    }

    /// Whether `FIFI_FLINT_CUDA_EXPERT_CACHE` selects the host buffer for every load. An
    /// unlisted word counts as not selected: the load itself refuses it.
    pub(crate) fn cache_forced_on() -> bool {
        load::cache_forced_on()
    }

    /// Whether `FIFI_FLINT_CUDA_EXPERT_CACHE=0` forces every expert onto the GPU.
    pub(crate) fn cache_forced_off() -> bool {
        load::cache_forced_off()
    }

    /// Why the host cache is in use (`auto`, `forced` or `override`), or `None` when every expert
    /// is resident.
    pub(crate) fn bank_reason(&self) -> Option<&'static str> {
        self.host.as_ref().map(|host| host.reason.as_str())
    }

    /// Routed experts per layer on the device when the host cache holds the rest of the 256.
    pub(crate) fn banked_experts(&self) -> Option<usize> {
        self.host.as_ref().map(|host| host.slots)
    }

    pub(crate) fn embedding(&self) -> Result<DeviceRange<'_>, EngError> {
        self.plain(self.roles.embedding)
    }

    pub(crate) fn final_norm(&self) -> Result<DeviceRange<'_>, EngError> {
        self.plain(self.roles.final_norm)
    }

    pub(crate) fn head(&self) -> Result<Exl3MatrixView<'_>, EngError> {
        self.matrix(self.roles.head)
    }

    pub(crate) fn layer(&self, layer: usize) -> Result<FlintLayerView<'_>, EngError> {
        let roles = self.roles.layers.get(layer).ok_or_else(|| {
            EngError::BadRequest(format!("Flint EXL3 layer {layer} is out of range"))
        })?;
        let ffn = match &roles.ffn {
            FfnRoles::Dense(indices) => FlintFfnView::Dense(self.expert(*indices)?),
            FfnRoles::Moe {
                router,
                experts,
                shared,
                routing_weight_multiplier,
            } => FlintFfnView::Moe {
                router: self.plain(*router)?,
                experts: FlintExpertBank {
                    owner: self,
                    indices: experts,
                    layer,
                    prepared: None,
                },
                shared: self.expert(*shared)?,
                routing_weight_multiplier: *routing_weight_multiplier,
            },
        };
        Ok(FlintLayerView {
            q: self.matrix(roles.qkvo[0])?,
            k: self.matrix(roles.qkvo[1])?,
            v: self.matrix(roles.qkvo[2])?,
            o: self.matrix(roles.qkvo[3])?,
            input_norm: self.plain(roles.norms[0])?,
            post_attention_norm: self.plain(roles.norms[1])?,
            q_norm: self.plain(roles.norms[2])?,
            k_norm: self.plain(roles.norms[3])?,
            sinks: self.plain(roles.sinks)?,
            ffn,
        })
    }

    fn range(&self, range: PackedRange) -> Result<DeviceRange<'_>, EngError> {
        self.buffer.range(range.offset, range.bytes)
    }

    fn plain(&self, index: usize) -> Result<DeviceRange<'_>, EngError> {
        let range = self.plain.get(index).ok_or_else(|| {
            EngError::BadRequest(format!(
                "Flint EXL3 plain descriptor {index} is out of range"
            ))
        })?;
        self.range(*range)
    }

    fn matrix(&self, index: usize) -> Result<Exl3MatrixView<'_>, EngError> {
        let matrix = self.matrices.get(index).ok_or_else(|| {
            EngError::BadRequest(format!(
                "Flint EXL3 matrix descriptor {index} is out of range"
            ))
        })?;
        Ok(Exl3MatrixView {
            trellis: self.range(matrix.trellis)?,
            suh: self.range(matrix.suh)?,
            svh: self.range(matrix.svh)?,
            bias: matrix.bias.map(|range| self.range(range)).transpose()?,
            shape: matrix.shape,
            bits_twice: matrix.bits_twice,
            book: matrix.book,
        })
    }

    fn expert(&self, indices: [usize; 3]) -> Result<FlintExpertView<'_>, EngError> {
        Ok(FlintExpertView {
            gate: self.matrix(indices[0])?,
            up: self.matrix(indices[1])?,
            down: self.matrix(indices[2])?,
        })
    }
}

impl FlintExpertBank<'_> {
    pub(crate) fn len(&self) -> usize {
        self.indices.len()
    }

    /// Experts that can be staged together: the resident count of the host cache,
    /// or the whole bank when every expert is resident for good.
    pub(crate) fn round_capacity(&self) -> usize {
        self.owner
            .host
            .as_ref()
            .map_or(self.indices.len(), |host| host.slots)
    }

    /// The layer's experts as an id-addressed bank, or `None` when the load keeps experts in the
    /// host store (their device places change) or the experts are not laid out at one common
    /// byte offset per expert for all three roles. The table holds 32-bit offsets, so a layer
    /// whose experts span 4 GiB or more declines too. The layout is worked out and the table
    /// uploaded on first use; every decline names its reason once on stderr.
    pub(crate) fn pick_bank(&self, ctx: &CudaContext) -> Result<Option<PickBank<'_>>, EngError> {
        let owner = self.owner;
        if owner.host.is_some() {
            static ANNOUNCED: std::sync::Once = std::sync::Once::new();
            ANNOUNCED.call_once(|| {
                eprintln!(
                    "Flint device picks: declined, the expert bank keeps experts in the host store"
                );
            });
            return Ok(None);
        }
        let Some(cell) = owner.pick_tables.get(self.layer) else {
            return Ok(None);
        };
        if cell.get().is_none() {
            let _ = cell.set(self.build_pick_entry(ctx)?);
        }
        let Some(Some(entry)) = cell.get() else {
            return Ok(None);
        };
        let range = |(offset, bytes): (usize, usize)| owner.buffer.range(offset, bytes);
        let role = |r: usize| -> Result<PickRole<'_>, EngError> {
            Ok(PickRole {
                trellis: range(entry.spans[r * 3])?,
                suh: range(entry.spans[r * 3 + 1])?,
                svh: range(entry.spans[r * 3 + 2])?,
            })
        };
        Ok(Some(PickBank {
            members: entry.members,
            table: entry.table.range(0, entry.members as usize * 4)?,
            roles: [role(0)?, role(1)?, role(2)?],
        }))
    }

    fn build_pick_entry(&self, ctx: &CudaContext) -> Result<Option<PickEntry>, EngError> {
        let (offsets, spans) = match self.pick_layout() {
            Ok(layout) => layout,
            Err(reason) => {
                eprintln!(
                    "Flint device picks: layer {} declined, {reason}",
                    self.layer
                );
                return Ok(None);
            }
        };
        for role in 0..3 {
            let m = &self.owner.matrices[self.indices[0][role]];
            let need = [m.trellis.bytes, m.shape[0], m.shape[1]];
            let bank = [
                spans[role * 3].1,
                spans[role * 3 + 1].1,
                spans[role * 3 + 2].1,
            ];
            check_member_offsets(&offsets, need, bank)?;
        }
        let members = u32::try_from(offsets.len())
            .map_err(|_| EngError::BadRequest("Flint expert count overflows".into()))?;
        let bytes: Vec<u8> = offsets.iter().flat_map(|o| o.to_le_bytes()).collect();
        Ok(Some(PickEntry {
            members,
            spans,
            table: ctx.upload(&bytes)?,
        }))
    }

    /// Each expert's byte offset from expert 0 and, per role and part (trellis, suh, svh), the
    /// range from expert 0 to the end of the last expert; or why the layout has no such offset.
    fn pick_layout(&self) -> Result<PickLayout, &'static str> {
        if self.indices.is_empty() {
            return Err("the layer has no experts");
        }
        let matrix = |expert: usize, role: usize| &self.owner.matrices[self.indices[expert][role]];
        let parts = |m: &PackedMatrix| [m.trellis, m.suh, m.svh];
        let mut offsets = vec![0u32; self.indices.len()];
        let mut spans = [(0usize, 0usize); 9];
        for role in 0..3 {
            let first = matrix(0, role);
            for (expert, offset) in offsets.iter_mut().enumerate() {
                let current = matrix(expert, role);
                if current.shape != first.shape
                    || current.bits_twice != first.bits_twice
                    || current.book != first.book
                {
                    return Err("the experts differ in shape, rate or codebook");
                }
                let mut deltas = parts(current)
                    .into_iter()
                    .zip(parts(first))
                    .map(|(c, f)| c.offset.checked_sub(f.offset));
                let delta = deltas
                    .next()
                    .flatten()
                    .and_then(|d| u32::try_from(d).ok())
                    .filter(|&d| d != ABSENT_MEMBER)
                    .ok_or("the experts span 4 GiB or more, or are not in id order")?;
                if deltas.any(|d| d != Some(delta as usize)) {
                    return Err("the experts do not share one offset for trellis and scales");
                }
                if role == 0 {
                    *offset = delta;
                } else if *offset != delta {
                    return Err("the three roles do not share one offset per expert");
                }
                for (part, range) in parts(current).into_iter().enumerate() {
                    let end = range.offset + range.bytes - parts(first)[part].offset;
                    spans[role * 3 + part].1 = spans[role * 3 + part].1.max(end);
                }
            }
            for (part, range) in parts(first).into_iter().enumerate() {
                spans[role * 3 + part].0 = range.offset;
            }
        }
        Ok((offsets, spans))
    }

    pub(crate) fn shape(&self, expert: usize) -> Result<(usize, usize), EngError> {
        let indices = self.indices.get(expert).ok_or_else(|| {
            EngError::BadRequest(format!("Flint EXL3 expert {expert} is out of range"))
        })?;
        let shape = |index: usize| {
            self.owner
                .matrices
                .get(index)
                .map(|matrix| matrix.shape)
                .ok_or_else(|| {
                    EngError::BadRequest(
                        "Flint CUDA FFN: expert matrix index is out of range".into(),
                    )
                })
        };
        let gate = shape(indices[0])?;
        let up = shape(indices[1])?;
        let down = shape(indices[2])?;
        let [hidden, intermediate] = gate;
        if up != gate || down != [intermediate, hidden] {
            return Err(EngError::BadRequest(
                "Flint CUDA FFN: EXL3 gate/up/down matrix shapes disagree".into(),
            ));
        }
        Ok((hidden, intermediate))
    }

    pub(crate) fn get(&self, expert: usize) -> Result<FlintExpertView<'_>, EngError> {
        let indices = self.indices.get(expert).ok_or_else(|| {
            EngError::BadRequest(format!("Flint EXL3 expert {expert} is out of range"))
        })?;
        if self.owner.host.is_some() {
            let base = self
                .prepared
                .as_ref()
                .and_then(|map| map.get(expert))
                .copied()
                .flatten()
                .ok_or_else(|| {
                    EngError::BadRequest("Flint host expert payload is not prepared".into())
                })?;
            return self.owner.prepared_expert(*indices, base);
        }
        self.owner.expert(*indices)
    }
}
