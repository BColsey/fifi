//! Stored-byte layouts for GGUF expert records and part-major stack planning.
//! The stack plan describes kernel byte ranges; it does not allocate or dispatch.

use std::ops::Range;

use super::GgufModel;
use crate::{EngError, family::recipe::Role, weights::GgufType};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(in super::super::super) struct ExpertPart {
    pub(in super::super::super) role: Role,
    pub(in super::super::super) ty: GgufType,
    pub(in super::super::super) bytes: Range<usize>,
}

/// A packed host record, with no implied CUDA alignment or allocation policy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(in super::super::super) struct ExpertLayout {
    pub(in super::super::super) parts: [ExpertPart; 3],
    pub(in super::super::super) bytes: usize,
}

/// Part-major stacks match `matvec_expert`: each part contains all slots contiguously.
/// The buffer base and each part start must be four-byte aligned.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(in super::super::super) struct ExpertStacks {
    pub(in super::super::super) parts: [Range<usize>; 3],
    pub(in super::super::super) bytes: usize,
    slots: usize,
}

/// Exact allocations for one device stack per layer and one reusable host record.
/// Excludes allocator overhead, other device buffers and any pinned-host store.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(in super::super::super) struct ExpertAllocation {
    pub(in super::super::super) layer_bytes: Vec<usize>,
    pub(in super::super::super) device_bytes: usize,
    pub(in super::super::super) staging_bytes: usize,
}

impl ExpertAllocation {
    pub(in super::super::super) fn for_model(
        model: &GgufModel,
        slots: usize,
    ) -> Result<Self, EngError> {
        if slots == 0 || slots > model.config().dims.experts as usize {
            return Err(EngError::MemoryPlan(
                "invalid GGUF expert allocation slot count".into(),
            ));
        }
        let mut layer_bytes = Vec::with_capacity(model.config().layers());
        let mut device_bytes = 0usize;
        let mut staging_bytes = 0usize;
        for layer in 0..model.config().layers() {
            let layout = ExpertLayout::for_layer(model, layer)?;
            let bytes = ExpertStacks::new(&layout, slots)?.bytes;
            device_bytes = device_bytes
                .checked_add(bytes)
                .ok_or_else(|| EngError::MemoryPlan("GGUF expert allocation overflow".into()))?;
            staging_bytes = staging_bytes.max(layout.bytes);
            layer_bytes.push(bytes);
        }
        Ok(Self {
            layer_bytes,
            device_bytes,
            staging_bytes,
        })
    }
}

impl ExpertStacks {
    pub(in super::super::super) fn new(
        record: &ExpertLayout,
        slots: usize,
    ) -> Result<Self, EngError> {
        let bad = || EngError::MemoryPlan("invalid GGUF expert stack size".into());
        if slots == 0 {
            return Err(bad());
        }
        let mut offset = 0usize;
        let mut ranges = Vec::with_capacity(3);
        for part in &record.parts {
            let len = part
                .bytes
                .end
                .checked_sub(part.bytes.start)
                .filter(|n| *n > 0)
                .ok_or_else(bad)?;
            let start = offset.checked_add(3).ok_or_else(bad)? & !3;
            let end = start
                .checked_add(len.checked_mul(slots).ok_or_else(bad)?)
                .ok_or_else(bad)?;
            ranges.push(start..end);
            offset = end;
        }
        Ok(Self {
            parts: ranges.try_into().map_err(|_| bad())?,
            bytes: offset,
            slots,
        })
    }

    /// Copy one packed host record into its part-major slot; validation occurs before writes.
    pub(in super::super::super) fn copy_record(
        &self,
        layout: &ExpertLayout,
        slot: usize,
        record: &[u8],
        stack: &mut [u8],
    ) -> Result<(), EngError> {
        let destinations = self.copy_ranges(layout, slot, record, stack.len())?;
        for (part, dest) in layout.parts.iter().zip(destinations) {
            stack[dest].copy_from_slice(&record[part.bytes.clone()]);
        }
        Ok(())
    }

    /// Synchronous ordered uploads. If a device copy fails, discard the partially written slot.
    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    pub(in super::super::super) fn upload_record(
        &self,
        layout: &ExpertLayout,
        slot: usize,
        record: &[u8],
        stack: &crate::kernels::cuda::DeviceBuffer,
        stream: &crate::kernels::cuda::DeviceStream,
    ) -> Result<(), EngError> {
        let destinations = self.copy_ranges(layout, slot, record, stack.bytes())?;
        for (part, dest) in layout.parts.iter().zip(destinations) {
            stack.write_at(stream, dest.start, &record[part.bytes.clone()])?;
        }
        Ok(())
    }

    fn copy_ranges(
        &self,
        layout: &ExpertLayout,
        slot: usize,
        record: &[u8],
        stack_bytes: usize,
    ) -> Result<Vec<Range<usize>>, EngError> {
        let bad = || EngError::BadRequest("invalid GGUF expert stack copy".into());
        if record.len() != layout.bytes
            || stack_bytes != self.bytes
            || *self != Self::new(layout, self.slots)?
        {
            return Err(bad());
        }
        let mut destinations = Vec::with_capacity(3);
        for (index, part) in layout.parts.iter().enumerate() {
            let source = record.get(part.bytes.clone()).ok_or_else(bad)?;
            let dest = self.slot(index, slot)?;
            if source.len() != dest.len() || dest.end > stack_bytes {
                return Err(bad());
            }
            destinations.push(dest);
        }
        Ok(destinations)
    }

    pub(in super::super::super) fn slot(
        &self,
        part: usize,
        slot: usize,
    ) -> Result<Range<usize>, EngError> {
        let bad = || EngError::BadRequest("invalid GGUF expert stack slot".into());
        let range = self.parts.get(part).ok_or_else(bad)?;
        if slot >= self.slots {
            return Err(bad());
        }
        let stride = range.len() / self.slots;
        let start = range
            .start
            .checked_add(stride.checked_mul(slot).ok_or_else(bad)?)
            .ok_or_else(bad)?;
        Ok(start..start.checked_add(stride).ok_or_else(bad)?)
    }
}

/// Checked native projection metadata and stack range for one routed expert part.
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
pub(in super::super::super) struct ExpertProjection<'a> {
    pub(in super::super::super) ty: crate::kernels::cuda::resident::matvec_q::QuantType,
    pub(in super::super::super) weights: crate::kernels::cuda::DeviceRange<'a>,
    pub(in super::super::super) k: usize,
    pub(in super::super::super) rows: usize,
    pub(in super::super::super) slots: usize,
}

/// Staged device owner. Every slot must be initialized before exposing any stack,
/// since indexed kernels may access any slot. Cross-stream ordering is the caller's duty.
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
pub(in super::super::super) struct ExpertDeviceStacks<'a> {
    source: Option<(&'a GgufModel, usize)>,
    slot_experts: Option<Vec<u32>>,
    layout: ExpertLayout,
    stacks: ExpertStacks,
    buffer: crate::kernels::cuda::DeviceBuffer,
    ready: Vec<bool>,
    poisoned: bool,
}

#[cfg(all(feature = "native-cuda", target_os = "linux"))]
impl<'a> ExpertDeviceStacks<'a> {
    pub(in super::super::super) fn new(
        ctx: &crate::kernels::cuda::CudaContext,
        layout: ExpertLayout,
        slots: usize,
    ) -> Result<Self, EngError> {
        let stacks = ExpertStacks::new(&layout, slots)?;
        let buffer = ctx.allocate(stacks.bytes)?;
        Ok(Self {
            source: None,
            slot_experts: None,
            layout,
            stacks,
            buffer,
            ready: vec![false; slots],
            poisoned: false,
        })
    }

    /// Initialize slots from one GGUF layer, in caller-specified order. The caller must
    /// admit both the device allocation and one host record before calling this function.
    pub(in super::super::super) fn from_model(
        ctx: &crate::kernels::cuda::CudaContext,
        stream: &crate::kernels::cuda::DeviceStream,
        model: &'a GgufModel,
        layer: usize,
        experts: &[u32],
    ) -> Result<Self, EngError> {
        let invalid = || EngError::BadRequest("invalid GGUF initial expert list".into());
        let count = model.config().dims.experts;
        let mut unique = std::collections::BTreeSet::new();
        if experts.is_empty() || experts.iter().any(|&id| id >= count || !unique.insert(id)) {
            return Err(invalid());
        }
        let layout = ExpertLayout::for_layer(model, layer)?;
        let mut device = Self::new(ctx, layout, experts.len())?;
        device.source = Some((model, layer));
        let mut record = vec![0; device.layout.bytes];
        for (slot, &expert) in experts.iter().enumerate() {
            device
                .layout
                .read(model, layer, u64::from(expert), &mut record)?;
            device.upload(stream, slot, &record)?;
        }
        device.slot_experts = Some(experts.to_vec());
        Ok(device)
    }

    /// Execute a disk-mode pass; the shared policy advances during planning, so
    /// any subsequent failure poisons it before this method returns.
    pub(in super::super::super) fn step_disk<P: crate::family::expert_bank::SlotPolicy>(
        &mut self,
        stream: &crate::kernels::cuda::DeviceStream,
        model: &GgufModel,
        state: &mut crate::family::expert_bank::ExpertBankState<P>,
        layer: usize,
        picks: &[i32],
    ) -> Result<crate::family::expert_bank::PassCopies, EngError> {
        use crate::family::expert_bank::{HostMode, NO_PLACE};
        if self.poisoned || state.is_poisoned() {
            self.poisoned = true;
            return Err(EngError::BadRequest("poisoned GGUF expert bank".into()));
        }
        if state.mode() != HostMode::Disk {
            return Err(EngError::BadRequest(
                "GGUF disk step requires disk mode".into(),
            ));
        }
        self.check_source(model, layer)?;
        let mut slots = self
            .slot_experts
            .clone()
            .ok_or_else(|| EngError::BadRequest("unknown GGUF device slot identities".into()))?;
        if state.geometry().slots != slots.len()
            || state.geometry().experts != model.config().dims.experts as usize
            || layer >= state.geometry().layers
            || slots.iter().enumerate().any(|(slot, &expert)| {
                state.expert_in_slot(layer, slot).map(u32::from) != Some(expert)
            })
        {
            return Err(EngError::BadRequest(
                "GGUF device slots disagree with bank state".into(),
            ));
        }
        let copies = match state.begin_pass(layer, picks) {
            Ok(copies) => copies,
            Err(error) => {
                self.poisoned |= state.is_poisoned();
                return Err(error);
            }
        };
        for op in &copies.ops {
            let result = if op.src_place == NO_PLACE && op.demote.is_none() {
                self.replace_from_model(
                    stream,
                    model,
                    layer,
                    usize::from(op.slot),
                    u32::from(op.incoming),
                )
            } else {
                Err(EngError::BadRequest(
                    "unexpected GGUF disk copy plan".into(),
                ))
            };
            if let Err(error) = result {
                self.poisoned = true;
                state.poison();
                return Err(error);
            }
            slots[usize::from(op.slot)] = u32::from(op.incoming);
        }
        self.slot_experts = Some(slots);
        Ok(copies)
    }

    /// Read the incoming expert before overwriting its slot. `begin_pass` advances
    /// shared bank state before copies, so its caller must poison that state on any
    /// replacement failure and must not expose the new slot row before success.
    pub(in super::super::super) fn replace_from_model(
        &mut self,
        stream: &crate::kernels::cuda::DeviceStream,
        model: &GgufModel,
        layer: usize,
        slot: usize,
        expert: u32,
    ) -> Result<(), EngError> {
        self.check_source(model, layer)?;
        if slot >= self.ready.len() || expert >= model.config().dims.experts {
            return Err(EngError::BadRequest(
                "invalid GGUF expert replacement".into(),
            ));
        }
        if ExpertLayout::for_layer(model, layer)? != self.layout {
            return Err(EngError::BadRequest(
                "GGUF expert replacement layout mismatch".into(),
            ));
        }
        let mut record = vec![0; self.layout.bytes];
        self.layout
            .read(model, layer, u64::from(expert), &mut record)?;
        self.upload(stream, slot, &record)
    }

    fn check_source(&self, model: &GgufModel, layer: usize) -> Result<(), EngError> {
        if !self.source.is_some_and(|(source, original_layer)| {
            std::ptr::eq(source, model) && original_layer == layer
        }) {
            return Err(EngError::BadRequest(
                "GGUF expert bank source mismatch".into(),
            ));
        }
        Ok(())
    }

    /// An upload error invalidates that slot, including when the copy failed partway.
    pub(in super::super::super) fn upload(
        &mut self,
        stream: &crate::kernels::cuda::DeviceStream,
        slot: usize,
        record: &[u8],
    ) -> Result<(), EngError> {
        if self.poisoned {
            return Err(EngError::BadRequest("poisoned GGUF expert bank".into()));
        }
        self.slot_experts = None;
        let ready = self
            .ready
            .get_mut(slot)
            .ok_or_else(|| EngError::BadRequest("invalid GGUF device expert slot".into()))?;
        *ready = false;
        self.stacks
            .upload_record(&self.layout, slot, record, &self.buffer, stream)?;
        *ready = true;
        Ok(())
    }

    /// Derive the weight type and dimensions from the source tensor, not caller inputs.
    pub(in super::super::super) fn projection(
        &self,
        role: Role,
    ) -> Result<ExpertProjection<'_>, EngError> {
        use crate::kernels::cuda::resident::matvec_q::QuantType;
        let invalid = || EngError::BadRequest("invalid GGUF expert projection".into());
        let (model, layer) = self.source.ok_or_else(invalid)?;
        let index = self
            .layout
            .parts
            .iter()
            .position(|part| part.role == role)
            .ok_or_else(invalid)?;
        let part = &self.layout.parts[index];
        let tensor = model.map().get(role, Some(layer)).ok_or_else(invalid)?;
        let [k, rows, _] = tensor.dims.as_slice() else {
            return Err(invalid());
        };
        if tensor.ty != part.ty {
            return Err(invalid());
        }
        Ok(ExpertProjection {
            ty: QuantType::from_gguf(part.ty).ok_or_else(invalid)?,
            weights: self.part(index)?,
            k: usize::try_from(*k).map_err(|_| invalid())?,
            rows: usize::try_from(*rows).map_err(|_| invalid())?,
            slots: self.ready.len(),
        })
    }

    pub(in super::super::super) fn part(
        &self,
        index: usize,
    ) -> Result<crate::kernels::cuda::DeviceRange<'_>, EngError> {
        if self.poisoned {
            return Err(EngError::BadRequest("poisoned GGUF expert bank".into()));
        }
        if self.ready.iter().any(|ready| !ready) {
            return Err(EngError::BadRequest(
                "GGUF expert stack has uninitialized slots".into(),
            ));
        }
        let range = self
            .stacks
            .parts
            .get(index)
            .ok_or_else(|| EngError::BadRequest("invalid GGUF expert stack part".into()))?;
        self.buffer.range(range.start, range.len())
    }
}

impl ExpertLayout {
    pub(in super::super::super) fn for_layer(
        model: &GgufModel,
        layer: usize,
    ) -> Result<Self, EngError> {
        let bad = || {
            EngError::MemoryPlan(format!(
                "Flash-Next GGUF expert layout invalid at layer {layer}"
            ))
        };
        let mut offset = 0usize;
        let mut parts = Vec::with_capacity(3);
        for role in [Role::ExpertGate, Role::ExpertUp, Role::ExpertDown] {
            let tensor = model.map().get(role, Some(layer)).ok_or_else(bad)?;
            let [_, outputs, experts] = tensor.dims.as_slice() else {
                return Err(bad());
            };
            if *experts == 0 {
                return Err(bad());
            }
            let (first, len) = tensor.row_span(Some(0), 0..*outputs)?;
            let (last, last_len) = tensor.row_span(Some(experts - 1), 0..*outputs)?;
            if first != 0 || len != last_len || last.checked_add(len as u64) != Some(tensor.bytes) {
                return Err(bad());
            }
            let end = offset.checked_add(len).ok_or_else(bad)?;
            parts.push(ExpertPart {
                role,
                ty: tensor.ty,
                bytes: offset..end,
            });
            offset = end;
        }
        Ok(Self {
            parts: parts.try_into().map_err(|_| bad())?,
            bytes: offset,
        })
    }

    /// Fill a packed host record. Validation precedes reads; an I/O error can still leave
    /// an already-read prefix in `out`, so callers must discard the whole record on error.
    pub(in super::super::super) fn read(
        &self,
        model: &GgufModel,
        layer: usize,
        expert: u64,
        out: &mut [u8],
    ) -> Result<(), EngError> {
        let bad = || EngError::BadRequest("invalid GGUF expert record request".into());
        if out.len() != self.bytes || *self != Self::for_layer(model, layer)? {
            return Err(bad());
        }
        let mut rows = [0u64; 3];
        for (i, part) in self.parts.iter().enumerate() {
            let tensor = model.map().get(part.role, Some(layer)).ok_or_else(bad)?;
            let [_, outputs, _] = tensor.dims.as_slice() else {
                return Err(bad());
            };
            let (_, bytes) = tensor.row_span(Some(expert), 0..*outputs)?;
            if bytes != part.bytes.len() {
                return Err(bad());
            }
            rows[i] = *outputs;
        }
        for (part, outputs) in self.parts.iter().zip(rows) {
            model.read_expert_rows(
                layer,
                part.role,
                expert,
                0..outputs,
                &mut out[part.bytes.clone()],
            )?;
        }
        Ok(())
    }
}
