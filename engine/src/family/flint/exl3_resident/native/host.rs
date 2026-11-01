//! One immutable pinned source and one authoritative cache map per model.
//!
//! Six routed experts per row can demand at most `slots / 6` rows in a bank of `slots`
//! resident experts per layer: a bank smaller than the prefill piece's pairs is refused
//! before weight allocation.

use std::{
    cell::{Cell, RefCell},
    path::Path,
};

use super::super::{
    FfnRoles, FlintResidentLimits, ResidentPlan,
    cache::{
        BANK_MARGIN_BYTES, BankBudget, BankPlan, BankReason, BankRequest, EXPERTS, ExpertCacheMode,
        HostBudget, HostStore, LayerCache, LayerLayout, MissCopy, choose_slots, plan_bank,
    },
    plan::plan_resident,
};
use super::{
    FlintExl3Resident, FlintExpertBank, FlintExpertView, PackedMatrix, PackedRange,
    packing::{CachedAudit, CachedPacking, Packing},
};
use crate::{
    EngError,
    family::{
        flint::cpu_moe::ExpertSource,
        project::cuda::{Exl3MatrixView, Reserve},
    },
    kernels::{
        cuda::{CudaContext, DeviceBuffer, DeviceStream, PinnedBuf, plan},
        exl3_cpu::{Exl3Projection, ExpertView},
    },
    weights::exl3::{Checkpoint, Codebook, Matrix, flint_pack::Geometry},
};

/// Experts copied from the host buffer into the GPU bank since the model loaded.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct BankStats {
    pub(crate) misses: u64,
    pub(crate) bytes: u64,
}

pub(super) struct HostOwner {
    state: RefCell<HostState>,
    poisoned: Cell<bool>,
    pub(super) stats: Cell<BankStats>,
    layers: Vec<usize>,
    arena_base: usize,
    // Resident experts per layer: the most experts one pass can use from a layer.
    pub(super) slots: usize,
    pub(super) reason: BankReason,
}

struct HostState {
    pinned: PinnedBuf,
    store: HostStore,
    caches: Vec<LayerCache>,
    stream: u64,
}

fn refused(message: &str) -> EngError {
    EngError::BadRequest(format!("Flint host cache: {message}"))
}

fn overflow() -> EngError {
    EngError::MemoryPlan("Flint host cache byte count overflows".into())
}

fn admit_cached(
    context: &CudaContext,
    cached: &CachedPacking,
    limits: FlintResidentLimits,
    config_bytes: usize,
) -> Result<(), EngError> {
    let packing = &cached.packing;
    // Charge full source, bounded acquisition/conversion, and descriptor storage.
    let metadata = packing
        .matrices
        .len()
        .checked_mul(std::mem::size_of::<PackedMatrix>() + 2048)
        .and_then(|bytes| {
            bytes.checked_add(
                cached
                    .layers
                    .len()
                    .checked_mul(std::mem::size_of::<LayerCache>() + 1024)?,
            )
        })
        .and_then(|bytes| bytes.checked_add(config_bytes))
        .ok_or_else(overflow)?;
    let required = u64::try_from(cached.layout.host_bytes())
        .map_err(|_| overflow())?
        .checked_add(packing.staging.checked_mul(2).ok_or_else(overflow)?)
        .and_then(|bytes| bytes.checked_add(metadata as u64))
        .ok_or_else(overflow)?;
    HostBudget::probe()?.admit(required)?;
    plan::admit(
        &plan::Demand {
            tensors: packing.bytes as u64,
            kv_cache: limits.reserve.kv_cache,
            work: limits.reserve.work,
            headroom: limits.reserve.headroom,
        },
        u64::try_from(context.memory_info()?.free).map_err(|_| overflow())?,
    )?;
    Ok(())
}

/// What a load asks for: the switch, how many experts per layer, and the prefill piece it must
/// serve.
pub(super) struct BankAsk {
    pub(super) mode: ExpertCacheMode,
    pub(super) request: BankRequest,
    pub(super) rows: usize,
}

/// Whether the plain resident load of `roles` would be admitted now: the same weight bytes and the
/// same admission sum as `load_planned`, so a context that fits keeps today's load.
fn resident_admits(
    context: &CudaContext,
    roles: &ResidentPlan,
    limits: &FlintResidentLimits,
) -> Result<bool, EngError> {
    let Ok(packing) = Packing::new(&roles.layout, limits) else {
        return Ok(false);
    };
    let free = u64::try_from(context.memory_info()?.free).map_err(|_| overflow())?;
    let demand = plan::Demand {
        tensors: packing.bytes as u64,
        kv_cache: limits.reserve.kv_cache,
        work: limits.reserve.work,
        headroom: limits.reserve.headroom,
    };
    Ok(plan::admit(&demand, free).is_ok())
}

/// The device budget of the bank: `arena_base` bytes of other weights, the reserve and the margin
/// come off `free`; each resident expert in every routed layer costs `per_slot`.
fn bank_budget(
    free: u64,
    arena_base: usize,
    per_slot: u64,
    reserve: &Reserve,
) -> Result<BankBudget, EngError> {
    let fixed = [
        reserve.kv_cache,
        reserve.work,
        reserve.headroom,
        BANK_MARGIN_BYTES,
    ]
    .into_iter()
    .try_fold(arena_base as u64, u64::checked_add)
    .ok_or_else(overflow)?;
    Ok(BankBudget {
        free,
        fixed,
        per_slot,
    })
}

/// Experts per layer for this device, KV cache, work reserve and prefill piece.
fn choose_bank(
    context: &CudaContext,
    audit: &CachedAudit,
    limits: &FlintResidentLimits,
    ask: &BankAsk,
    offloaded: usize,
) -> Result<usize, EngError> {
    let floor = crate::family::flint::cuda_ffn::device::exl3::cache_prefill_floor(ask.rows)?;
    let free = u64::try_from(context.memory_info()?.free).map_err(|_| overflow())?;
    let budget = bank_budget(
        free,
        audit.arena_base(),
        audit.per_slot_bytes(offloaded)?,
        &limits.reserve,
    )?;
    choose_slots(ask.request, budget, floor)
}

impl HostOwner {
    fn note_misses(&self, copies: &[MissCopy]) {
        let mut stats = self.stats.get();
        stats.misses += copies.len() as u64;
        stats.bytes += copies
            .iter()
            .map(|copy| copy.source.bytes as u64)
            .sum::<u64>();
        self.stats.set(stats);
    }
}

fn enqueue(
    context: &CudaContext,
    stream: &DeviceStream,
    pinned: &mut PinnedBuf,
    buffer: &DeviceBuffer,
    arena_base: usize,
    copies: &[MissCopy],
) -> Result<(), EngError> {
    for copy in copies {
        let destination = arena_base
            .checked_add(copy.destination.offset)
            .ok_or_else(overflow)?;
        if copy.source.bytes != copy.destination.bytes {
            return Err(refused("copy sizes disagree"));
        }
        context.copy_from_pinned(
            stream,
            pinned,
            copy.source.offset,
            buffer.range(destination, copy.destination.bytes)?,
        )?;
    }
    Ok(())
}

/// Loads the model as `ask` says. The plain resident load needs no pinned store and no cache; a
/// bank of all 256 is that load too.
pub(super) fn load_banked(
    context: &CudaContext,
    stream: &DeviceStream,
    path: &Path,
    checkpoint: Checkpoint,
    geometry: Geometry,
    limits: FlintResidentLimits,
    ask: BankAsk,
) -> Result<FlintExl3Resident, EngError> {
    let mut audit_limits = limits;
    audit_limits.max_weight_bytes = u64::MAX;
    let roles = plan_resident(path, &checkpoint, &geometry, audit_limits)?;
    let looks_resident = ask.mode == ExpertCacheMode::Auto && ask.request == BankRequest::Fit;
    let resident_fits = looks_resident && resident_admits(context, &roles, &limits)?;
    let BankPlan::Bank(reason) = plan_bank(ask.mode, ask.request, resident_fits)? else {
        return super::load::load_planned(context, stream, checkpoint, roles, limits);
    };
    let audit = Packing::audit_cached(&roles, &limits)?;
    // The layers whose experts run on the CPU keep no device slots.
    let cpu = crate::family::flint::cpu_moe::active();
    let offloaded = cpu.as_ref().map_or(0, |moe| moe.config().offload_layers());
    let slots = choose_bank(context, &audit, &limits, &ask, offloaded)?;
    // CPU-served experts read the pinned host store, which a bank of all 256 never builds.
    let slots = if cpu.is_some() && slots == EXPERTS {
        eprintln!(
            "Flint CPU MoE: a bank of all {EXPERTS} experts is lowered to {} so the pinned host store exists",
            EXPERTS - 1
        );
        EXPERTS - 1
    } else {
        slots
    };
    if slots == EXPERTS {
        return super::load::load_planned(context, stream, checkpoint, roles, limits);
    }
    let cached = audit.with_slots(slots, offloaded, reason, &limits)?;
    load_cached(context, stream, checkpoint, roles, cached, limits)
}

fn load_cached(
    context: &CudaContext,
    stream: &DeviceStream,
    checkpoint: Checkpoint,
    roles: ResidentPlan,
    cached: CachedPacking,
    limits: FlintResidentLimits,
) -> Result<FlintExl3Resident, EngError> {
    admit_cached(context, &cached, limits, checkpoint.model_config.len())?;
    let slots = cached.layout.slot_count();
    let reason = cached.reason;
    let packing = cached.packing;
    let mut descriptors = Vec::new();
    descriptors
        .try_reserve_exact(roles.layout.modules.len())
        .map_err(|error| EngError::MemoryPlan(format!("Flint matrix descriptors: {error}")))?;
    descriptors.resize(roles.layout.modules.len(), None);
    let mut pinned = context.pinned_buf(cached.layout.host_bytes())?;
    let buffer = context.allocate(packing.bytes)?;
    for (index, module) in roles.layout.modules.iter().enumerate() {
        if cached.routed[index].is_some() {
            continue;
        }
        let specs = module.specs();
        let keys: Vec<&str> = specs.iter().map(|spec| spec.name.as_str()).collect();
        descriptors[index] = Some(fill_matrix(
            &checkpoint,
            &module.prefix,
            &keys,
            packing.matrices[index],
            limits.max_staging_bytes,
            |slot, bytes| buffer.write_at(stream, slot.offset, bytes),
        )?);
    }
    let store = HostStore::build(cached.layout, |layer, expert, range| {
        let FfnRoles::Moe { experts, .. } = &roles.layers[cached.layers[layer]].ffn else {
            return Err(refused("routed layer changed during loading"));
        };
        for &index in &experts[usize::from(expert)] {
            let module = &roles.layout.modules[index];
            let specs = module.specs();
            let keys: Vec<&str> = specs.iter().map(|spec| spec.name.as_str()).collect();
            descriptors[index] = Some(fill_matrix(
                &checkpoint,
                &module.prefix,
                &keys,
                packing.matrices[index],
                limits.max_staging_bytes,
                |slot, bytes| {
                    let end = slot.offset.checked_add(bytes.len()).ok_or_else(overflow)?;
                    if end > range.bytes {
                        return Err(refused("source component exceeds expert stride"));
                    }
                    pinned.write(
                        range.offset.checked_add(slot.offset).ok_or_else(overflow)?,
                        bytes,
                    )
                },
            )?);
        }
        // pinned_buf zeroed every padding byte before any component was written.
        Ok(range.bytes)
    })?;
    for (plain, &slot) in roles.layout.plain.iter().zip(&packing.plain) {
        super::load::upload_plain(
            &checkpoint,
            &buffer,
            stream,
            plain,
            slot,
            limits.max_staging_bytes,
        )?;
    }
    let mut caches = Vec::new();
    caches
        .try_reserve_exact(cached.layers.len())
        .map_err(|error| EngError::MemoryPlan(format!("Flint layer caches: {error}")))?;
    for layer in 0..cached.layers.len() {
        let mut core = LayerCache::new(&store, layer, stream.id())?;
        core.publish_initial(stream.id(), |copies| {
            enqueue(
                context,
                stream,
                &mut pinned,
                &buffer,
                cached.arena_base,
                copies,
            )?;
            pinned.synchronize(stream)
        })?;
        caches.push(core);
    }
    stream.synchronize()?;
    let matrices = descriptors
        .into_iter()
        .map(|descriptor| descriptor.ok_or_else(|| refused("incomplete logical matrix coverage")))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(FlintExl3Resident {
        buffer,
        pick_tables: Vec::new(),
        host: Some(HostOwner {
            state: RefCell::new(HostState {
                pinned,
                store,
                caches,
                stream: stream.id(),
            }),
            poisoned: Cell::new(false),
            stats: Cell::default(),
            layers: cached.layers,
            arena_base: cached.arena_base,
            slots,
            reason,
        }),
        matrices,
        plain: packing.plain,
        roles,
        model_config: checkpoint.model_config,
    })
}

fn fill_matrix(
    checkpoint: &Checkpoint,
    prefix: &str,
    keys: &[&str],
    slots: [PackedRange; 3],
    max_staging: u64,
    mut write: impl FnMut(PackedRange, &[u8]) -> Result<(), EngError>,
) -> Result<PackedMatrix, EngError> {
    // The audited plan supplies names; Matrix::load preserves each logical matrix's metadata.
    let selected = checkpoint.source.read_selected(keys, max_staging)?;
    let matrix = Matrix::load(&selected, prefix, &checkpoint.config)?;
    if matrix.bias().is_some() {
        return Err(refused("routed matrix bias is unsupported"));
    }
    for (part, (tensor, slot)) in [matrix.trellis(), matrix.suh(), matrix.svh()]
        .into_iter()
        .zip(slots)
        .enumerate()
    {
        let raw = tensor
            .host_bytes()
            .ok_or_else(|| refused("matrix requires host storage"))?;
        if part == 0 {
            if raw.len() != slot.bytes {
                return Err(refused("trellis disagrees with placement"));
            }
            write(slot, raw)?;
        } else {
            let [suh_factor, svh_factor] = super::load::scale_factors(prefix);
            let converted = super::load::convert_halves_by(
                raw,
                tensor.dtype(),
                if part == 1 { suh_factor } else { svh_factor },
            )?;
            if converted.len() != slot.bytes {
                return Err(refused("scale disagrees with placement"));
            }
            write(slot, &converted)?;
        }
    }
    Ok(PackedMatrix {
        trellis: slots[0],
        suh: slots[1],
        svh: slots[2],
        bias: None,
        shape: matrix.shape(),
        bits_twice: matrix.bitrate_twice(),
        book: matrix.codebook(),
    })
}

impl FlintExl3Resident {
    pub(crate) fn load_with_prefill(
        context: &CudaContext,
        stream: &DeviceStream,
        path: &Path,
        limits: FlintResidentLimits,
        rows: usize,
    ) -> Result<Self, EngError> {
        super::load::load_with_prefill(context, stream, path, limits, rows)
    }

    #[cfg(test)]
    pub(crate) fn test_host_cache_byte(&self) -> Result<u8, EngError> {
        let host = self
            .host
            .as_ref()
            .ok_or_else(|| refused("model has no pinned source"))?;
        let state = host
            .state
            .try_borrow()
            .map_err(|_| refused("model cache is already borrowed"))?;
        let mut byte = [0];
        state.pinned.read(0, &mut byte)?;
        Ok(byte[0])
    }

    #[cfg(test)]
    pub(crate) fn load_cached_fixture(
        context: &CudaContext,
        stream: &DeviceStream,
        path: &Path,
        limits: FlintResidentLimits,
        slots: usize,
    ) -> Result<Self, EngError> {
        super::load::load_cached_fixture(context, stream, path, limits, slots)
    }

    pub(crate) fn recover_expert_cache(
        &self,
        context: &CudaContext,
        stream: &DeviceStream,
    ) -> Result<(), EngError> {
        let Some(host) = &self.host else {
            return Ok(());
        };
        if !host.poisoned.get() {
            return Ok(());
        }
        context.ensure_usable()?;
        let mut state = host
            .state
            .try_borrow_mut()
            .map_err(|_| refused("model cache is already borrowed"))?;
        if state.stream != stream.id() {
            return Err(refused("recovery stream differs from model stream"));
        }
        stream.synchronize()?;
        state.pinned.synchronize(stream)?;
        let HostState { pinned, caches, .. } = &mut *state;
        for core in caches {
            core.poison();
            core.reset_after_drain(stream.id(), || {
                stream.synchronize()?;
                pinned.synchronize(stream)
            })?;
            core.publish_initial(stream.id(), |copies| {
                enqueue(
                    context,
                    stream,
                    pinned,
                    &self.buffer,
                    host.arena_base,
                    copies,
                )?;
                pinned.synchronize(stream)
            })?;
        }
        stream.synchronize()?;
        host.poisoned.set(false);
        Ok(())
    }

    pub(super) fn prepared_expert(
        &self,
        indices: [usize; 3],
        base: usize,
    ) -> Result<FlintExpertView<'_>, EngError> {
        let matrix = |index: usize| -> Result<Exl3MatrixView<'_>, EngError> {
            let logical = self
                .matrices
                .get(index)
                .ok_or_else(|| refused("unknown matrix"))?;
            let range = |part: PackedRange| {
                self.range(PackedRange {
                    offset: base.checked_add(part.offset).ok_or_else(overflow)?,
                    bytes: part.bytes,
                })
            };
            Ok(Exl3MatrixView {
                trellis: range(logical.trellis)?,
                suh: range(logical.suh)?,
                svh: range(logical.svh)?,
                bias: None,
                shape: logical.shape,
                bits_twice: logical.bits_twice,
                book: logical.book,
            })
        };
        Ok(FlintExpertView {
            gate: matrix(indices[0])?,
            up: matrix(indices[1])?,
            down: matrix(indices[2])?,
        })
    }
}

/// The routed experts of one layer, read in place from the pinned host store.
struct PackedSource<'a> {
    bytes: &'a [u8],
    matrices: &'a [PackedMatrix],
    indices: &'a [[usize; 3]],
    geometry: LayerLayout,
}

impl PackedSource<'_> {
    fn bytes_of(&self, base: usize, range: PackedRange) -> Result<&[u8], EngError> {
        let start = base.checked_add(range.offset).ok_or_else(overflow)?;
        let end = start.checked_add(range.bytes).ok_or_else(overflow)?;
        self.bytes
            .get(start..end)
            .ok_or_else(|| refused("expert bytes lie outside the host store"))
    }

    fn projection(&self, base: usize, index: usize) -> Result<Exl3Projection<'_>, EngError> {
        let matrix = self
            .matrices
            .get(index)
            .ok_or_else(|| refused("unknown matrix"))?;
        if matrix.book != Codebook::Mul1 {
            return Err(refused("CPU experts need the mul1 codebook"));
        }
        Exl3Projection::new(
            matrix.shape,
            usize::from(matrix.bits_twice) * 8,
            self.bytes_of(base, matrix.trellis)?,
            self.bytes_of(base, matrix.suh)?,
            self.bytes_of(base, matrix.svh)?,
        )
    }
}

impl ExpertSource for PackedSource<'_> {
    fn expert(&self, id: usize) -> Result<ExpertView<'_>, EngError> {
        let indices = self
            .indices
            .get(id)
            .ok_or_else(|| refused("unknown routed expert"))?;
        let base = self
            .geometry
            .source(u16::try_from(id).map_err(|_| overflow())?)?
            .offset;
        Ok(ExpertView {
            gate: self.projection(base, indices[0])?,
            up: self.projection(base, indices[1])?,
            down: self.projection(base, indices[2])?,
        })
    }
}

impl FlintExl3Resident {
    fn host_ordinal(&self, layer: usize) -> Option<usize> {
        let host = self.host.as_ref()?;
        host.layers.iter().position(|&candidate| candidate == layer)
    }

    /// Runs `consume` with in-place host access to the routed experts of `layer`; needs the
    /// pinned host store.
    pub(crate) fn with_host_experts<R>(
        &self,
        layer: usize,
        consume: impl FnOnce(&dyn ExpertSource) -> Result<R, EngError>,
    ) -> Result<R, EngError> {
        let (host, ordinal) = self
            .host
            .as_ref()
            .zip(self.host_ordinal(layer))
            .ok_or_else(|| refused("CPU experts need the pinned host store"))?;
        let state = host
            .state
            .try_borrow()
            .map_err(|_| refused("model cache is already borrowed"))?;
        let FfnRoles::Moe { experts, .. } = &self.roles.layers[layer].ffn else {
            return Err(refused("layer has no routed experts"));
        };
        consume(&PackedSource {
            bytes: state.pinned.host_view()?,
            matrices: &self.matrices,
            indices: experts,
            geometry: state.store.layout().layer(ordinal)?,
        })
    }

    /// The `ordinal`-th routed layer of `layer` and which of its experts hold a device slot.
    pub(crate) fn resident_experts(
        &self,
        layer: usize,
    ) -> Result<Option<(usize, [bool; EXPERTS])>, EngError> {
        let (Some(host), Some(ordinal)) = (self.host.as_ref(), self.host_ordinal(layer)) else {
            return Ok(None);
        };
        let state = host
            .state
            .try_borrow()
            .map_err(|_| refused("model cache is already borrowed"))?;
        Ok(Some((ordinal, state.caches[ordinal].resident_flags())))
    }

    /// Moves `incoming` into the device slot of `outgoing` for `layer`, behind the work already
    /// queued on `stream`.
    pub(crate) fn swap_expert(
        &self,
        context: &CudaContext,
        stream: &DeviceStream,
        layer: usize,
        swap: (u16, u16),
    ) -> Result<(), EngError> {
        let (host, ordinal) = self
            .host
            .as_ref()
            .zip(self.host_ordinal(layer))
            .ok_or_else(|| refused("expert swaps need the pinned host store"))?;
        let mut state = host
            .state
            .try_borrow_mut()
            .map_err(|_| refused("model cache is already borrowed"))?;
        let HostState { pinned, caches, .. } = &mut *state;
        let copy = caches[ordinal].swap_in(swap.0, swap.1)?;
        enqueue(
            context,
            stream,
            pinned,
            &self.buffer,
            host.arena_base,
            &[copy],
        )
    }
}

impl FlintExpertBank<'_> {
    /// Marks experts as just used, in the order given, without any copy (see
    /// [`LayerCache::touch`]). A no-op when every expert is resident for good.
    pub(crate) fn touch_recent(&self, experts: &[u16]) -> Result<(), EngError> {
        let Some(host) = &self.owner.host else {
            return Ok(());
        };
        let mut state = host
            .state
            .try_borrow_mut()
            .map_err(|_| refused("model cache is already borrowed"))?;
        let ordinal = host
            .layers
            .iter()
            .position(|&layer| layer == self.layer)
            .ok_or_else(|| refused("unknown routed layer"))?;
        state.caches[ordinal].touch(experts)
    }
}

impl FlintExpertBank<'_> {
    /// In-place host access to this layer's routed experts (needs the pinned host store).
    pub(crate) fn cpu_experts<R>(
        &self,
        consume: impl FnOnce(&dyn ExpertSource) -> Result<R, EngError>,
    ) -> Result<R, EngError> {
        self.owner.with_host_experts(self.layer, consume)
    }

    /// This layer's routed-layer ordinal and which experts hold a device slot, or `None`
    /// without a host store.
    pub(crate) fn resident(&self) -> Result<Option<(usize, [bool; EXPERTS])>, EngError> {
        self.owner.resident_experts(self.layer)
    }

    pub(crate) fn swap(
        &self,
        context: &CudaContext,
        stream: &DeviceStream,
        swap: (u16, u16),
    ) -> Result<(), EngError> {
        self.owner.swap_expert(context, stream, self.layer, swap)
    }

    /// The callback must finish all routed, shared, and combine work with Frame::finish,
    /// or enqueue it on `stream` so a later pass's slot copies queue behind it.
    /// Only a completed owned output may escape this scope, never live weight views.
    pub(crate) fn with_prepared<R>(
        &self,
        context: &CudaContext,
        stream: &DeviceStream,
        ids: &[i32],
        consume: impl for<'a> FnOnce(&FlintExpertBank<'a>) -> Result<R, EngError>,
    ) -> Result<R, EngError> {
        let Some(host) = &self.owner.host else {
            return consume(&FlintExpertBank {
                owner: self.owner,
                indices: self.indices,
                layer: self.layer,
                prepared: None,
            });
        };
        if host.poisoned.replace(true) {
            return Err(refused("model cache is poisoned"));
        }
        context.ensure_usable()?;
        let mut state = host
            .state
            .try_borrow_mut()
            .map_err(|_| refused("model cache is already borrowed"))?;
        if state.stream != stream.id() {
            return Err(refused("consumer stream differs from model stream"));
        }
        let ordinal = host
            .layers
            .iter()
            .position(|&layer| layer == self.layer)
            .ok_or_else(|| refused("unknown routed layer"))?;
        let HostState {
            pinned,
            store,
            caches,
            ..
        } = &mut *state;
        let core = &mut caches[ordinal];
        let max_copy_bytes = store
            .layout()
            .layer(ordinal)?
            .slot(0)?
            .bytes
            .checked_mul(host.slots)
            .ok_or_else(overflow)?;
        let pass = core.prepare(ids, max_copy_bytes)?;
        let ready = core.copy_ready(pass, stream.id(), |copies| {
            enqueue(
                context,
                stream,
                pinned,
                &self.owner.buffer,
                host.arena_base,
                copies,
            )?;
            host.note_misses(copies);
            Ok(())
        })?;
        let mut snapshot = [None; EXPERTS];
        for &id in ids {
            let expert = u16::try_from(id).map_err(|_| refused("invalid expert ID"))?;
            let slot = core.slot(ready, expert)?;
            let relative = store.layout().layer(ordinal)?.slot(slot)?.offset;
            snapshot[usize::from(expert)] =
                Some(host.arena_base.checked_add(relative).ok_or_else(overflow)?);
        }
        let output = core.release(ready, stream.id(), || {
            consume(&FlintExpertBank {
                owner: self.owner,
                indices: self.indices,
                layer: self.layer,
                prepared: Some(&snapshot),
            })
        })?;
        host.poisoned.set(false);
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::cuda::plan::{Demand, admit};

    // Distinct nonzero terms, so a dropped or doubled term changes which bank fits.
    const RESERVE: Reserve = Reserve {
        kv_cache: 3_795_369_984,
        work: 696_027_652,
        headroom: 536_870_912,
    };
    const ARENA_BASE: usize = 2_150_000_640;
    const PER_SLOT: u64 = 47 * 1_603_584;

    fn demand(slots: u64, headroom: u64) -> Demand {
        Demand {
            tensors: ARENA_BASE as u64 + slots * PER_SLOT,
            kv_cache: RESERVE.kv_cache,
            work: RESERVE.work,
            headroom,
        }
    }

    #[test]
    fn the_chosen_bank_is_admitted_with_the_margin_and_one_more_expert_is_not() {
        for free in (22..=26).map(|gib: u64| gib * (1 << 30) + 12_345_678) {
            let budget = bank_budget(free, ARENA_BASE, PER_SLOT, &RESERVE).unwrap();
            let slots = choose_slots(BankRequest::Fit, budget, 48).unwrap() as u64;
            let with_margin = RESERVE.headroom + BANK_MARGIN_BYTES;
            admit(&demand(slots, with_margin), free).unwrap();
            if slots < EXPERTS as u64 {
                assert!(admit(&demand(slots + 1, with_margin), free).is_err());
            }
            // The margin is spare room beyond what plan::admit itself requires.
            admit(&demand(slots, RESERVE.headroom), free).unwrap();
        }
    }

    #[test]
    fn every_reserve_term_and_the_margin_shrink_the_bank() {
        let free = 23 * (1 << 30);
        let slots = |reserve: Reserve| {
            let budget = bank_budget(free, ARENA_BASE, PER_SLOT, &reserve).unwrap();
            choose_slots(BankRequest::Fit, budget, 48).unwrap()
        };
        let base = slots(RESERVE);
        for bigger in [
            Reserve {
                kv_cache: RESERVE.kv_cache + (1 << 30),
                ..RESERVE
            },
            Reserve {
                work: RESERVE.work + (1 << 30),
                ..RESERVE
            },
            Reserve {
                headroom: RESERVE.headroom + (1 << 30),
                ..RESERVE
            },
        ] {
            assert!(slots(bigger) < base);
        }
        let without_margin =
            (free - ARENA_BASE as u64 - RESERVE.kv_cache - RESERVE.work - RESERVE.headroom)
                / PER_SLOT;
        assert!((base as u64) < without_margin);
        // The margin also governs the choice of all 256: a load with less than the margin to spare
        // uses a smaller bank.
        let fixed = ARENA_BASE as u64 + RESERVE.kv_cache + RESERVE.work + RESERVE.headroom;
        let all = |extra: u64| {
            let free = fixed + 256 * PER_SLOT + extra;
            let budget = bank_budget(free, ARENA_BASE, PER_SLOT, &RESERVE).unwrap();
            choose_slots(BankRequest::Fit, budget, 48).unwrap()
        };
        assert_eq!(all(BANK_MARGIN_BYTES - 1), 255);
        assert_eq!(all(BANK_MARGIN_BYTES), 256);
        assert!(
            bank_budget(
                u64::MAX,
                usize::MAX,
                1,
                &Reserve {
                    kv_cache: u64::MAX,
                    ..RESERVE
                }
            )
            .is_err()
        );
    }
}
