//! CPU cache policy used by the model-owned native pinned store.
//! This proves no VRAM freed, physical pinning, or GPU completion.

use crate::{EngError, family::flint::cpu_moe::CpuMoeConfig};

pub(super) const EXPERTS: usize = 256;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ByteRange {
    pub(super) offset: usize,
    pub(super) bytes: usize,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct LayerLayout {
    source: ByteRange,
    slots: ByteRange,
    stride: usize,
    slot_count: usize,
}

pub(super) struct HostLayout {
    layers: Vec<LayerLayout>,
    host_bytes: usize,
    slot_bytes: usize,
    slot_count: usize,
}

impl HostLayout {
    // Strides come from auditing every expert's converted, padded pack, not expert zero.
    // Every layer keeps `slot_count` of its 256 experts on the device.
    pub(super) fn new(strides: &[usize], slot_count: usize) -> Result<Self, EngError> {
        Self::with_offloaded(strides, slot_count, 0)
    }

    /// As [`Self::new`], but the first `offloaded` layers keep no device slots: their experts
    /// run on the CPU, so they cost no device bytes.
    pub(super) fn with_offloaded(
        strides: &[usize],
        slot_count: usize,
        offloaded: usize,
    ) -> Result<Self, EngError> {
        if strides.is_empty() {
            return Err(cache_error("host layout needs at least one routed layer"));
        }
        if !(1..=EXPERTS).contains(&slot_count) {
            return Err(cache_error("resident expert count must be within 1..=256"));
        }
        let mut layers = Vec::new();
        layers
            .try_reserve_exact(strides.len())
            .map_err(|error| EngError::MemoryPlan(format!("Flint expert cache layout: {error}")))?;
        let mut host_bytes = 0usize;
        let mut slot_bytes = 0usize;
        for (index, &stride) in strides.iter().enumerate() {
            if stride == 0 {
                return Err(cache_error("expert stride must be nonzero"));
            }
            let slot_count = if index < offloaded { 0 } else { slot_count };
            let source_bytes = stride.checked_mul(EXPERTS).ok_or_else(cache_overflow)?;
            let device_bytes = stride.checked_mul(slot_count).ok_or_else(cache_overflow)?;
            let source = ByteRange {
                offset: host_bytes,
                bytes: source_bytes,
            };
            let slots = ByteRange {
                offset: slot_bytes,
                bytes: device_bytes,
            };
            host_bytes = host_bytes
                .checked_add(source_bytes)
                .ok_or_else(cache_overflow)?;
            slot_bytes = slot_bytes
                .checked_add(device_bytes)
                .ok_or_else(cache_overflow)?;
            layers.push(LayerLayout {
                source,
                slots,
                stride,
                slot_count,
            });
        }
        Ok(Self {
            layers,
            host_bytes,
            slot_bytes,
            slot_count,
        })
    }

    pub(super) fn layer(&self, index: usize) -> Result<LayerLayout, EngError> {
        self.layers
            .get(index)
            .copied()
            .ok_or_else(|| cache_error("layer is out of range"))
    }

    pub(super) fn host_bytes(&self) -> usize {
        self.host_bytes
    }

    pub(super) fn slot_bytes(&self) -> usize {
        self.slot_bytes
    }

    pub(super) fn slot_count(&self) -> usize {
        self.slot_count
    }
}

impl LayerLayout {
    pub(super) fn source(&self, expert: u16) -> Result<ByteRange, EngError> {
        self.element(self.source, usize::from(expert), EXPERTS)
    }

    pub(super) fn slot(&self, slot: u16) -> Result<ByteRange, EngError> {
        self.element(self.slots, usize::from(slot), self.slot_count)
    }

    fn element(
        &self,
        region: ByteRange,
        index: usize,
        count: usize,
    ) -> Result<ByteRange, EngError> {
        if index >= count {
            return Err(cache_error("expert or slot is out of range"));
        }
        let relative = index.checked_mul(self.stride).ok_or_else(cache_overflow)?;
        let offset = region
            .offset
            .checked_add(relative)
            .ok_or_else(cache_overflow)?;
        Ok(ByteRange {
            offset,
            bytes: self.stride,
        })
    }
}

// A coverage/layout seal only. The native owner must retain the immutable PinnedBuf;
// this type neither owns bytes nor proves physical pinning or GPU completion.
pub(super) struct HostStore {
    layout: HostLayout,
}

impl HostStore {
    pub(super) fn build(
        layout: HostLayout,
        mut fill: impl FnMut(usize, u16, ByteRange) -> Result<usize, EngError>,
    ) -> Result<Self, EngError> {
        for (layer, geometry) in layout.layers.iter().enumerate() {
            for expert in 0..EXPERTS {
                let expert = u16::try_from(expert).map_err(|_| cache_overflow())?;
                let range = geometry.source(expert)?;
                if fill(layer, expert, range)? != range.bytes {
                    return Err(cache_error("host expert write has the wrong byte count"));
                }
            }
        }
        Ok(Self { layout })
    }

    pub(super) fn layout(&self) -> &HostLayout {
        &self.layout
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct MissCopy {
    pub(super) expert: u16,
    pub(super) slot: u16,
    pub(super) source: ByteRange,
    pub(super) destination: ByteRange,
}

impl MissCopy {
    const EMPTY: Self = Self {
        expert: 0,
        slot: 0,
        source: ByteRange {
            offset: 0,
            bytes: 0,
        },
        destination: ByteRange {
            offset: 0,
            bytes: 0,
        },
    };
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct PassKey {
    owner: u64,
    generation: u64,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct ReadyPass(PassKey);

#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    Initial,
    Idle,
    Prepared,
    Ready,
    Poisoned,
}

pub(super) struct LayerCache {
    geometry: LayerLayout,
    stream: u64,
    key: PassKey,
    phase: Phase,
    // These three are sized by the layer's resident count, not by a compile-time bank.
    residents: Box<[u16]>,
    slots: [Option<u16>; EXPERTS],
    recency: Box<[u16]>,
    demanded: [bool; EXPERTS],
    copies: Box<[MissCopy]>,
    copy_count: usize,
}

impl LayerCache {
    pub(super) fn new(store: &HostStore, layer: usize, stream: u64) -> Result<Self, EngError> {
        static NEXT_OWNER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        let geometry = store.layout().layer(layer)?;
        let owner = NEXT_OWNER
            .fetch_update(
                std::sync::atomic::Ordering::Relaxed,
                std::sync::atomic::Ordering::Relaxed,
                |value| value.checked_add(1),
            )
            .map_err(|_| cache_overflow())?;
        let canonical: Box<[u16]> = (0..geometry.slot_count).map(|index| index as u16).collect();
        Ok(Self {
            slots: Self::canonical_slots(geometry.slot_count),
            residents: canonical.clone(),
            recency: canonical,
            copies: vec![MissCopy::EMPTY; geometry.slot_count].into(),
            geometry,
            stream,
            key: PassKey {
                owner,
                generation: 0,
            },
            phase: Phase::Initial,
            demanded: [false; EXPERTS],
            copy_count: 0,
        })
    }

    fn canonical_slots(slot_count: usize) -> [Option<u16>; EXPERTS] {
        std::array::from_fn(|index| (index < slot_count).then_some(index as u16))
    }

    fn check(&self, phase: Phase, stream: u64) -> Result<(), EngError> {
        if self.phase != phase || self.stream != stream {
            return Err(cache_error("wrong cache phase or stream"));
        }
        Ok(())
    }

    fn check_pass(&self, key: PassKey, phase: Phase, stream: u64) -> Result<(), EngError> {
        self.check(phase, stream)?;
        if key != self.key {
            return Err(cache_error("stale or foreign pass key"));
        }
        Ok(())
    }

    fn copy(&self, expert: u16, slot: u16) -> Result<MissCopy, EngError> {
        Ok(MissCopy {
            expert,
            slot,
            source: self.geometry.source(expert)?,
            destination: self.geometry.slot(slot)?,
        })
    }

    pub(super) fn publish_initial(
        &mut self,
        stream: u64,
        establish: impl FnOnce(&[MissCopy]) -> Result<(), EngError>,
    ) -> Result<(), EngError> {
        self.check(Phase::Initial, stream)?;
        let slot_count = self.geometry.slot_count;
        for index in 0..slot_count {
            self.copies[index] = self.copy(index as u16, index as u16)?;
        }
        self.copy_count = slot_count;
        self.phase = Phase::Poisoned;
        establish(&self.copies[..self.copy_count])?;
        self.copy_count = 0;
        self.phase = Phase::Idle;
        Ok(())
    }

    pub(super) fn prepare(
        &mut self,
        ids: &[i32],
        max_copy_bytes: usize,
    ) -> Result<PassKey, EngError> {
        self.check(Phase::Idle, self.stream)?;
        let mut demanded = [false; EXPERTS];
        let mut count = 0;
        for &id in ids {
            let index = usize::try_from(id).map_err(|_| cache_error("unknown expert ID"))?;
            let entry = demanded
                .get_mut(index)
                .ok_or_else(|| cache_error("unknown expert ID"))?;
            if !*entry {
                count += 1;
                *entry = true;
            }
        }
        let slot_count = self.geometry.slot_count;
        if count > slot_count {
            return Err(cache_error("demanded union exceeds slot capacity"));
        }
        let generation = self
            .key
            .generation
            .checked_add(1)
            .ok_or_else(cache_overflow)?;
        let mut residents = self.residents.clone();
        let mut slots = self.slots;
        let mut recency = self.recency.clone();
        let mut copies = [MissCopy::EMPTY; EXPERTS];
        let mut copy_count = 0usize;
        for &id in ids {
            let expert = id as u16;
            let slot = if let Some(slot) = slots[usize::from(expert)] {
                usize::from(slot)
            } else {
                let slot = recency
                    .iter()
                    .copied()
                    .find(|&slot| !demanded[usize::from(residents[usize::from(slot)])])
                    .ok_or_else(|| cache_error("no unprotected victim"))?;
                slots[usize::from(residents[usize::from(slot)])] = None;
                residents[usize::from(slot)] = expert;
                slots[usize::from(expert)] = Some(slot);
                copies[copy_count] = self.copy(expert, slot)?;
                copy_count += 1;
                usize::from(slot)
            };
            let position = recency
                .iter()
                .position(|&value| usize::from(value) == slot)
                .ok_or_else(|| cache_error("invalid recency order"))?;
            recency.copy_within(position + 1..slot_count, position);
            recency[slot_count - 1] = slot as u16;
        }
        let bytes = copy_count
            .checked_mul(self.geometry.stride)
            .ok_or_else(cache_overflow)?;
        if bytes > max_copy_bytes {
            return Err(cache_error("miss copies exceed byte bound"));
        }
        self.residents = residents;
        self.slots = slots;
        self.recency = recency;
        self.demanded = demanded;
        self.copies[..copy_count].copy_from_slice(&copies[..copy_count]);
        self.copy_count = copy_count;
        self.key.generation = generation;
        self.phase = Phase::Prepared;
        Ok(self.key)
    }

    /// Which experts hold a device slot now.
    pub(super) fn resident_flags(&self) -> [bool; EXPERTS] {
        std::array::from_fn(|expert| self.slots[expert].is_some())
    }

    /// Gives `incoming` the slot `outgoing` holds, for the slow hot-set policy of CPU-served
    /// layers, and returns the copy that fills it. The copy must be enqueued before the next
    /// pass of this layer; the slot becomes the most recently used one.
    pub(super) fn swap_in(&mut self, incoming: u16, outgoing: u16) -> Result<MissCopy, EngError> {
        self.check(Phase::Idle, self.stream)?;
        if self.slots[usize::from(incoming)].is_some() {
            return Err(cache_error("incoming expert already holds a slot"));
        }
        let slot = self.slots[usize::from(outgoing)]
            .take()
            .ok_or_else(|| cache_error("outgoing expert holds no slot"))?;
        let copy = self.copy(incoming, slot)?;
        self.residents[usize::from(slot)] = incoming;
        self.slots[usize::from(incoming)] = Some(slot);
        let position = self
            .recency
            .iter()
            .position(|&value| value == slot)
            .ok_or_else(|| cache_error("invalid recency order"))?;
        let last = self.geometry.slot_count;
        self.recency.copy_within(position + 1..last, position);
        self.recency[last - 1] = slot;
        Ok(copy)
    }

    pub(super) fn copy_ready(
        &mut self,
        pass: PassKey,
        stream: u64,
        establish: impl FnOnce(&[MissCopy]) -> Result<(), EngError>,
    ) -> Result<ReadyPass, EngError> {
        self.check_pass(pass, Phase::Prepared, stream)?;
        self.phase = Phase::Poisoned;
        establish(&self.copies[..self.copy_count])?;
        self.phase = Phase::Ready;
        Ok(ReadyPass(pass))
    }

    pub(super) fn slot(&self, ready: ReadyPass, expert: u16) -> Result<u16, EngError> {
        self.check_pass(ready.0, Phase::Ready, self.stream)?;
        if !self
            .demanded
            .get(usize::from(expert))
            .copied()
            .unwrap_or(false)
        {
            return Err(cache_error("expert is outside this pass"));
        }
        self.slots[usize::from(expert)].ok_or_else(|| cache_error("expert has no slot"))
    }

    pub(super) fn release<T>(
        &mut self,
        ready: ReadyPass,
        stream: u64,
        finish: impl FnOnce() -> Result<T, EngError>,
    ) -> Result<T, EngError> {
        self.check_pass(ready.0, Phase::Ready, stream)?;
        self.phase = Phase::Poisoned;
        let result = finish()?;
        self.demanded = [false; EXPERTS];
        self.copy_count = 0;
        self.phase = Phase::Idle;
        Ok(result)
    }

    /// Marks `experts` as just used, in the order given and without any copy: the later in the
    /// list, the more recent. Experts without a slot are skipped. A layer-major pass uses it to
    /// leave the recency a token-by-token walk would have left (experts of the latest rows last),
    /// which the decode that follows then evicts against.
    pub(super) fn touch(&mut self, experts: &[u16]) -> Result<(), EngError> {
        self.check(Phase::Idle, self.stream)?;
        let slot_count = self.geometry.slot_count;
        for &expert in experts {
            let Some(slot) = self.slots.get(usize::from(expert)).copied().flatten() else {
                continue;
            };
            let position = self
                .recency
                .iter()
                .position(|&value| value == slot)
                .ok_or_else(|| cache_error("invalid recency order"))?;
            self.recency.copy_within(position + 1..slot_count, position);
            self.recency[slot_count - 1] = slot;
        }
        Ok(())
    }

    pub(super) fn poison(&mut self) {
        self.phase = Phase::Poisoned;
    }

    pub(super) fn reset_after_drain(
        &mut self,
        stream: u64,
        drain: impl FnOnce() -> Result<(), EngError>,
    ) -> Result<(), EngError> {
        self.check(Phase::Poisoned, stream)?;
        let generation = self
            .key
            .generation
            .checked_add(1)
            .ok_or_else(cache_overflow)?;
        self.phase = Phase::Poisoned;
        drain()?;
        for (index, resident) in self.residents.iter_mut().enumerate() {
            *resident = index as u16;
        }
        self.recency.copy_from_slice(&self.residents);
        self.slots = Self::canonical_slots(self.geometry.slot_count);
        self.demanded = [false; EXPERTS];
        self.copies.fill(MissCopy::EMPTY);
        self.copy_count = 0;
        self.key.generation = generation;
        self.phase = Phase::Initial;
        Ok(())
    }
}

/// Operator override of the resident experts per layer; unset means "as many as fit".
pub(super) const SLOTS_VAR: &str = "FIFI_FLINT_CUDA_EXPERT_SLOTS";

/// The switch that selects the bank.
pub(super) const CACHE_VAR: &str = "FIFI_FLINT_CUDA_EXPERT_CACHE";

/// What `FIFI_FLINT_CUDA_EXPERT_CACHE` asks for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ExpertCacheMode {
    /// Unset: every expert stays on the GPU when that fits, otherwise the bank takes over.
    Auto,
    /// `1`, `true`, `on`: always use the bank.
    On,
    /// `0`, `false`, `off`: every expert on the GPU, or a refusal.
    Off,
}

impl ExpertCacheMode {
    /// Exactly the listed words; a mistyped value is refused rather than guessed, because the
    /// unset default is no longer "off".
    pub(super) fn from_env(value: Result<String, std::env::VarError>) -> Result<Self, EngError> {
        let text = match value {
            Ok(text) => text,
            Err(std::env::VarError::NotPresent) => return Ok(Self::Auto),
            Err(std::env::VarError::NotUnicode(_)) => String::from("(not UTF-8)"),
        };
        match text.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "on" => Ok(Self::On),
            "0" | "false" | "off" => Ok(Self::Off),
            _ => Err(cache_error(&format!(
                "{CACHE_VAR} must be unset, 1, true or on (always use the host buffer), or 0, false \
                 or off (every expert on the GPU); got {text:?}"
            ))),
        }
    }
}

/// Why the bank is in use, for the load line.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum BankReason {
    /// The context does not fit with every expert on the GPU.
    Auto,
    /// The operator asked for the bank.
    Forced,
    /// The operator asked for a resident count.
    Override,
}

impl BankReason {
    pub(super) fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Forced => "forced",
            Self::Override => "override",
        }
    }
}

/// How a load keeps its experts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum BankPlan {
    /// Every expert on the GPU, the load that needs no host buffer.
    Resident,
    Bank(BankReason),
}

/// The load for `mode` and `request`. `resident_fits` is whether the plain resident load would be
/// admitted; only the unset default looks at it, so a context that fits keeps today's exact path.
pub(super) fn plan_bank(
    mode: ExpertCacheMode,
    request: BankRequest,
    resident_fits: bool,
) -> Result<BankPlan, EngError> {
    match (mode, request) {
        (ExpertCacheMode::Off, BankRequest::Fit) => Ok(BankPlan::Resident),
        (ExpertCacheMode::Off, BankRequest::Exactly(_)) => Err(cache_error(&format!(
            "{SLOTS_VAR} asks for a bank, but {CACHE_VAR}=0 keeps every expert on the GPU"
        ))),
        (_, BankRequest::Exactly(_)) => Ok(BankPlan::Bank(BankReason::Override)),
        (ExpertCacheMode::On, BankRequest::Fit) => Ok(BankPlan::Bank(BankReason::Forced)),
        (ExpertCacheMode::Auto, BankRequest::Fit) if resident_fits => Ok(BankPlan::Resident),
        (ExpertCacheMode::Auto, BankRequest::Fit) => Ok(BankPlan::Bank(BankReason::Auto)),
    }
}

/// CPU-served experts need the pinned host store: the offload mode forces the bank on, and the
/// split mode also sizes it to the experts the device keeps, so it refuses an explicit slot count
/// that would contradict that.
pub(super) fn cpu_moe_bank(
    config: Option<CpuMoeConfig>,
    mode: ExpertCacheMode,
    request: BankRequest,
) -> Result<(ExpertCacheMode, BankRequest), EngError> {
    let Some(config) = config else {
        return Ok((mode, request));
    };
    if mode == ExpertCacheMode::Off {
        return Err(EngError::BadRequest(
            "Flint CPU experts need the host expert store: unset or enable the expert cache switch"
                .into(),
        ));
    }
    if config.split_experts() != 0 {
        if request != BankRequest::Fit {
            return Err(EngError::BadRequest(format!(
                "cpu_moe_split_experts sets the resident experts per layer: unset {SLOTS_VAR}"
            )));
        }
        let device = EXPERTS - config.split_experts();
        return Ok((ExpertCacheMode::On, BankRequest::Exactly(device)));
    }
    Ok((ExpertCacheMode::On, request))
}

/// How many experts per layer a load asks to keep on the device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum BankRequest {
    /// The largest bank the device budget leaves room for, capped at all 256.
    Fit,
    /// An explicit count; a count that does not fit refuses the load rather than shrinking.
    Exactly(usize),
}

impl BankRequest {
    pub(super) fn from_env(value: Result<String, std::env::VarError>) -> Result<Self, EngError> {
        let value = match value {
            Ok(value) => Some(value),
            Err(std::env::VarError::NotPresent) => None,
            Err(std::env::VarError::NotUnicode(_)) => {
                return Err(cache_error(&format!("{SLOTS_VAR} is not valid UTF-8")));
            }
        };
        Self::parse(value.as_deref())
    }

    // Plain decimal digits only: no sign, no leading zero, no spaces.
    fn parse(value: Option<&str>) -> Result<Self, EngError> {
        let value = value.unwrap_or("");
        if value.is_empty() {
            return Ok(Self::Fit);
        }
        let plain = value.bytes().all(|byte| byte.is_ascii_digit()) && !value.starts_with('0');
        plain
            .then(|| value.parse::<usize>().ok())
            .flatten()
            .filter(|count| (1..=EXPERTS).contains(count))
            .map(Self::Exactly)
            .ok_or_else(|| cache_error(&format!("{SLOTS_VAR} must be a whole number in 1..=256")))
    }
}

/// Device bytes the bank leaves free beyond the declared reserve, for every count it considers,
/// all 256 included. A bank that filled the card to its last expert let the whole card, desktop
/// included, peak 61 to 71 MiB past 22,900 MiB. One more expert in every routed layer costs 72 MiB
/// (47 layers of 1,603,584 B), so 160 MiB is a little over two experts per layer and brings the
/// peak under that bound.
pub(super) const BANK_MARGIN_BYTES: u64 = 160 << 20;

/// What a device has for the expert bank: `free` bytes, of which `fixed` (non-routed weights,
/// KV cache, work, headroom and the bank margin) are spoken for, and `per_slot` bytes for one
/// more resident expert in every routed layer.
#[derive(Clone, Copy, Debug)]
pub(super) struct BankBudget {
    pub(super) free: u64,
    pub(super) fixed: u64,
    pub(super) per_slot: u64,
}

impl BankBudget {
    fn room(&self) -> u64 {
        self.free.saturating_sub(self.fixed)
    }

    /// Experts per layer that fit, capped at all 256.
    fn fitting(&self) -> usize {
        let count = (self.room() / self.per_slot.max(1)).min(EXPERTS as u64);
        usize::try_from(count).unwrap_or(EXPERTS)
    }

    fn refusal(&self, what: String, floor: usize, advice: &str) -> EngError {
        EngError::MemoryPlan(format!(
            "Flint expert bank: {what}. After {} bytes of non-routed weights, KV cache, work, \
             headroom and the {} MiB bank margin, {} of {} free device bytes remain, room for \
             {} resident experts per layer ({} bytes each across the routed layers); a prefill \
             piece needs {floor}. {advice}.",
            self.fixed,
            BANK_MARGIN_BYTES >> 20,
            self.room(),
            self.free,
            self.fitting(),
            self.per_slot
        ))
    }
}

/// Resident experts per layer for `request`. `floor` is what one prefill piece can route in a
/// single layer: a smaller bank could not hold the piece's experts at once.
pub(super) fn choose_slots(
    request: BankRequest,
    budget: BankBudget,
    floor: usize,
) -> Result<usize, EngError> {
    let slots = match request {
        BankRequest::Fit => budget.fitting(),
        BankRequest::Exactly(count) if count > budget.fitting() => {
            let what = format!("{SLOTS_VAR}={count} does not fit");
            return Err(budget.refusal(what, floor, "Lower the count or context_limit"));
        }
        BankRequest::Exactly(count) => count,
    };
    if slots < floor {
        let what = format!("{slots} resident experts per layer is too few");
        let advice = match request {
            BankRequest::Fit => "Lower context_limit or prefill_chunk_size",
            BankRequest::Exactly(_) => "Raise the count or lower prefill_chunk_size",
        };
        return Err(budget.refusal(what, floor, advice));
    }
    Ok(slots)
}

fn cache_overflow() -> EngError {
    EngError::MemoryPlan("Flint expert cache byte count overflows".into())
}

fn cache_error(message: &str) -> EngError {
    EngError::BadRequest(format!("Flint expert cache: {message}"))
}

pub(crate) struct HostBudget {
    pub(super) mem_available: u64,
    // None means a confirmed unlimited cgroup, never a failed limit read.
    pub(super) cgroup_remaining: Option<u64>,
    pub(super) os_floor: u64,
}

impl HostBudget {
    pub(crate) fn probe() -> Result<Self, EngError> {
        Self::probe_in(
            std::path::Path::new("/proc"),
            std::path::Path::new("/sys/fs/cgroup"),
            std::process::id(),
        )
    }

    fn probe_in(
        proc: &std::path::Path,
        root: &std::path::Path,
        pid: u32,
    ) -> Result<Self, EngError> {
        let meminfo = read_admission_file(&proc.join("meminfo"))?;
        let membership = read_admission_file(&proc.join("self/cgroup"))?;
        let mounts = read_admission_file(&proc.join("self/mountinfo"))?;
        let group = cgroup_from_proc(root, &membership, &mounts, pid)?;
        let root = std::fs::canonicalize(root)
            .map_err(|error| EngError::MemoryPlan(format!("Flint cgroup root: {error}")))?;
        Self::from_meminfo(&meminfo, cgroup_remaining_in(&root, &group)?)
    }

    pub(super) fn from_meminfo(
        contents: &str,
        cgroup_remaining: Option<u64>,
    ) -> Result<Self, EngError> {
        let total = meminfo_bytes(contents, "MemTotal:")?;
        let available = meminfo_bytes(contents, "MemAvailable:")?;
        if total == 0 || available > total {
            return Err(EngError::MemoryPlan(
                "Flint expert cache memory snapshot has inconsistent totals.".into(),
            ));
        }
        Ok(Self {
            mem_available: available,
            cgroup_remaining,
            os_floor: total.div_ceil(10).max(8u64 << 30),
        })
    }

    pub(crate) fn admit(&self, required: u64) -> Result<(), EngError> {
        self.admit_for(required, "Flint expert cache")
    }

    /// [`admit`] with the caller's name in the refusal, so a prefix
    /// snapshot refusal does not masquerade as an expert-cache one
    /// (the refusal names the caller).
    pub(crate) fn admit_for(&self, required: u64, what: &str) -> Result<(), EngError> {
        let available = self
            .cgroup_remaining
            .map_or(self.mem_available, |remaining| {
                remaining.min(self.mem_available)
            });
        if required
            .checked_add(self.os_floor)
            .is_none_or(|needed| needed > available)
        {
            return Err(EngError::MemoryPlan(format!(
                "{what} needs {required} bytes plus {} bytes for the OS floor. Available host/cgroup budget: {available} bytes.",
                self.os_floor
            )));
        }
        Ok(())
    }
}

fn meminfo_bytes(contents: &str, name: &str) -> Result<u64, EngError> {
    let invalid = || {
        EngError::MemoryPlan(format!(
            "Flint expert cache needs one valid {name} value in kB from /proc/meminfo."
        ))
    };
    let mut matches = contents.lines().filter_map(|line| line.strip_prefix(name));
    let mut fields = matches.next().ok_or_else(invalid)?.split_whitespace();
    let bytes = fields
        .next()
        .and_then(|value| value.parse::<u64>().ok())
        .and_then(|kib| kib.checked_mul(1024))
        .ok_or_else(invalid)?;
    if fields.next() != Some("kB") || fields.next().is_some() || matches.next().is_some() {
        return Err(invalid());
    }
    Ok(bytes)
}

fn cgroup_remaining_bytes(maximum: &str, current: &str) -> Result<Option<u64>, EngError> {
    let invalid = || {
        EngError::MemoryPlan(
            "Flint expert cache needs memory.max as max or an unsigned byte count, and memory.current as an unsigned byte count."
                .into(),
        )
    };
    let parse = |value: &str| {
        let value = value.trim();
        if value.is_empty() || !value.bytes().all(|byte| byte.is_ascii_digit()) {
            return Err(invalid());
        }
        value.parse::<u64>().map_err(|_| invalid())
    };
    let current = parse(current)?;
    match maximum.trim() {
        "max" => Ok(None),
        maximum => Ok(Some(parse(maximum)?.saturating_sub(current))),
    }
}

fn read_admission_file(path: &std::path::Path) -> Result<String, EngError> {
    std::fs::read_to_string(path).map_err(|error| {
        EngError::MemoryPlan(format!(
            "Flint expert cache cannot read {}: {error}",
            path.display()
        ))
    })
}

// Discovery must verify the global controller root and reject hidden ancestors first.
fn cgroup_remaining_in(
    root: &std::path::Path,
    cgroup: &std::path::Path,
) -> Result<Option<u64>, EngError> {
    if !root.is_absolute()
        || !cgroup.is_absolute()
        || !cgroup.starts_with(root)
        || root
            .components()
            .chain(cgroup.components())
            .any(|component| component == std::path::Component::ParentDir)
    {
        return Err(EngError::MemoryPlan(
            "Flint expert cache cgroup must be inside its verified controller root.".into(),
        ));
    }
    let mut minimum: Option<u64> = None;
    let mut group = cgroup;
    while group != root {
        let parent = group.parent().ok_or_else(|| {
            EngError::MemoryPlan("Flint expert cache cgroup has no controller ancestor.".into())
        })?;
        if read_admission_file(&parent.join("cgroup.subtree_control"))?
            .split_whitespace()
            .any(|controller| controller == "memory")
            && let Some(remaining) = cgroup_remaining_bytes(
                &read_admission_file(&group.join("memory.max"))?,
                &read_admission_file(&group.join("memory.current"))?,
            )?
        {
            minimum = Some(minimum.map_or(remaining, |previous| previous.min(remaining)));
        }
        group = parent;
    }
    // The root of a cgroup namespace is the container's own cgroup: no parent is visible, so
    // its limit is the outermost one. The global root has no limit files.
    if root.join("memory.max").exists() {
        let remaining = cgroup_remaining_bytes(
            &read_admission_file(&root.join("memory.max"))?,
            &read_admission_file(&root.join("memory.current"))?,
        )?;
        if let Some(remaining) = remaining {
            minimum = Some(minimum.map_or(remaining, |previous| previous.min(remaining)));
        }
    }
    Ok(minimum)
}

fn verify_cgroup_root(root: &std::path::Path, mounts: &str) -> Result<(), EngError> {
    let invalid = || {
        EngError::MemoryPlan(
            "Flint expert cache needs one complete cgroup2 root with the memory controller.".into(),
        )
    };
    let point = root.to_str().ok_or_else(invalid)?;
    let mut mounted = false;
    for line in mounts.lines() {
        let Some((head, tail)) = line.split_once(" - ") else {
            continue;
        };
        let fields: Vec<_> = head.split_ascii_whitespace().collect();
        if fields.get(4) != Some(&point) {
            continue;
        }
        if mounted
            || fields.get(3) != Some(&"/")
            || tail.split_ascii_whitespace().next() != Some("cgroup2")
        {
            return Err(invalid());
        }
        mounted = true;
    }
    if !mounted
        || !read_admission_file(&root.join("cgroup.controllers"))?
            .split_whitespace()
            .any(|controller| controller == "memory")
    {
        return Err(invalid());
    }
    // These interfaces exist on namespace/subtree roots (a container's own cgroup), but not on
    // the global root. All three together mean a namespace root, whose own limit
    // `cgroup_remaining_in` applies; some of them alone is an ambiguous tree.
    let mut present = 0;
    for name in ["cgroup.type", "memory.max", "memory.current"] {
        let path = root.join(name);
        match std::fs::metadata(&path) {
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Ok(_) => present += 1,
            Err(error) => {
                return Err(EngError::MemoryPlan(format!(
                    "Flint expert cache cannot verify {}: {error}",
                    path.display()
                )));
            }
        }
    }
    if present != 0 && present != 3 {
        return Err(invalid());
    }
    Ok(())
}

fn cgroup_from_proc(
    root: &std::path::Path,
    membership: &str,
    mounts: &str,
    pid: u32,
) -> Result<std::path::PathBuf, EngError> {
    let invalid = || {
        EngError::MemoryPlan(
            "Flint expert cache cannot map its process inside a complete unified cgroup tree."
                .into(),
        )
    };
    let mut lines = membership.lines();
    let path = std::path::Path::new(
        lines
            .next()
            .and_then(|line| line.strip_prefix("0::"))
            .ok_or_else(invalid)?,
    );
    if pid == 0
        || lines.next().is_some()
        || !root.is_absolute()
        || !path.is_absolute()
        || root
            .components()
            .chain(path.components())
            .any(|component| component == std::path::Component::ParentDir)
    {
        return Err(invalid());
    }
    let resolve = |path: &std::path::Path| {
        std::fs::canonicalize(path).map_err(|error| {
            EngError::MemoryPlan(format!(
                "Flint expert cache cannot resolve {}: {error}",
                path.display()
            ))
        })
    };
    let root = resolve(root)?;
    verify_cgroup_root(&root, mounts)?;
    let group = resolve(&root.join(path.strip_prefix("/").map_err(|_| invalid())?))?;
    if !group.starts_with(&root) {
        return Err(invalid());
    }
    if group != root && read_admission_file(&group.join("cgroup.type"))?.trim() != "domain" {
        return Err(invalid());
    }
    let mut found = false;
    for owner in read_admission_file(&group.join("cgroup.procs"))?.split_whitespace() {
        found |= owner.parse::<u32>().map_err(|_| invalid())? == pid;
    }
    if !found {
        return Err(invalid());
    }
    Ok(group)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SLOTS: usize = 224;

    type BankState = (
        [Option<u16>; EXPERTS],
        Box<[u16]>,
        Box<[u16]>,
        [bool; EXPERTS],
        Box<[MissCopy]>,
        usize,
    );

    // What the bank holds, without the pass key that every cache owns uniquely.
    fn bank_state(cache: &LayerCache) -> BankState {
        (
            cache.slots,
            cache.residents.clone(),
            cache.recency.clone(),
            cache.demanded,
            cache.copies.clone(),
            cache.copy_count,
        )
    }

    // Everything a refused or poisoned call must leave exactly as it found it.
    fn snapshot(cache: &LayerCache) -> (BankState, PassKey) {
        (bank_state(cache), cache.key)
    }

    #[test]
    fn host_layout_sizes_all_256_sources_and_exactly_224_slots() {
        let layout = HostLayout::new(&[256, 512], SLOTS);
        assert!(
            layout.is_ok(),
            "valid packed expert strides must produce a host layout"
        );
        let layout = layout.unwrap();
        assert_eq!(layout.host_bytes(), 256 * (256 + 512));
        assert_eq!(layout.slot_bytes(), 224 * (256 + 512));
    }

    #[test]
    fn cpu_experts_force_the_bank_and_split_sizes_it() {
        use ExpertCacheMode::{Auto, Off, On};
        let offload = CpuMoeConfig::new(8, 0, 0).unwrap();
        let split = CpuMoeConfig::new(0, 32, 0).unwrap();
        // Off: the bank and the pinned host store follow the mode and request untouched.
        assert_eq!(
            cpu_moe_bank(None, Auto, BankRequest::Fit).unwrap(),
            (Auto, BankRequest::Fit)
        );
        // Offload forces the bank on and keeps the operator's slot request.
        for request in [BankRequest::Fit, BankRequest::Exactly(200)] {
            assert_eq!(
                cpu_moe_bank(Some(offload), Auto, request).unwrap(),
                (On, request)
            );
        }
        // Split keeps 256 - 32 experts on the device and refuses a contradicting count.
        assert_eq!(
            cpu_moe_bank(Some(split), Auto, BankRequest::Fit).unwrap(),
            (On, BankRequest::Exactly(224))
        );
        assert!(cpu_moe_bank(Some(split), On, BankRequest::Exactly(100)).is_err());
        // The off switch contradicts any CPU serving.
        for config in [offload, split] {
            assert!(cpu_moe_bank(Some(config), Off, BankRequest::Fit).is_err());
        }
    }

    #[test]
    fn swap_in_moves_one_expert_into_the_outgoing_slot_and_refuses_bad_swaps() {
        let layout = HostLayout::new(&[4], 4).unwrap();
        let store = HostStore::build(layout, |_, _, range| Ok(range.bytes)).unwrap();
        let mut cache = LayerCache::new(&store, 0, 1).unwrap();
        cache.publish_initial(1, |_| Ok(())).unwrap();
        // Experts 0..4 hold slots 0..4. Expert 9 takes expert 1's slot.
        let copy = cache.swap_in(9, 1).unwrap();
        assert_eq!((copy.expert, copy.slot), (9, 1));
        assert_eq!(copy.source.offset, 9 * 4);
        assert_eq!(copy.destination.offset, 4);
        let flags = cache.resident_flags();
        assert!(flags[9] && !flags[1] && flags[0] && flags[2] && flags[3]);
        assert_eq!(flags.iter().filter(|f| **f).count(), 4);
        // The swapped slot is the most recently used one, so the next LRU victim is slot 0.
        assert_eq!(cache.recency.last().copied(), Some(1));
        assert_eq!(cache.recency[0], 0);
        // Refusals leave the cache untouched.
        let before = snapshot(&cache);
        assert!(cache.swap_in(9, 2).is_err(), "incoming already resident");
        assert!(cache.swap_in(10, 1).is_err(), "outgoing holds no slot");
        assert_eq!(snapshot(&cache), before);
        // A pass in flight blocks swaps.
        let _pass = cache.prepare(&[0], usize::MAX).unwrap();
        assert!(cache.swap_in(10, 2).is_err());
    }

    #[test]
    fn offloaded_layers_keep_their_host_sources_but_no_device_slots() {
        let layout = HostLayout::with_offloaded(&[3, 5, 7], SLOTS, 1).unwrap();
        assert_eq!(layout.host_bytes(), 256 * (3 + 5 + 7));
        assert_eq!(layout.slot_bytes(), 224 * (5 + 7));
        let offloaded = layout.layer(0).unwrap();
        assert!(offloaded.source(255).is_ok());
        assert!(offloaded.slot(0).is_err());
        let kept = layout.layer(1).unwrap();
        assert_eq!(kept.slot(0).unwrap().offset, 0);
        assert_eq!(kept.slot(223).unwrap().offset, 223 * 5);
        assert!(kept.slot(224).is_err());
        assert_eq!(layout.layer(2).unwrap().slot(0).unwrap().offset, 224 * 5);
        // A layer without slots holds nothing on the device and refuses a device pass.
        let store = HostStore::build(layout, |_, _, range| Ok(range.bytes)).unwrap();
        let mut cache = LayerCache::new(&store, 0, 1).unwrap();
        assert_eq!(cache.resident_flags(), [false; EXPERTS]);
        cache
            .publish_initial(1, |copies| {
                assert!(copies.is_empty());
                Ok(())
            })
            .unwrap();
        assert!(cache.prepare(&[0], usize::MAX).is_err());
    }

    #[test]
    fn host_store_seal_checks_coverage_bounds_and_partial_cleanup() {
        let layout = HostLayout::new(&[3, 5], SLOTS).unwrap();
        // These are tiny pageable fixture bytes, never physical pinned storage.
        let mut bytes = vec![0u8; layout.host_bytes()];
        let mut writes = 0;
        let store = HostStore::build(layout, |layer, id, range| {
            assert_eq!(layer, writes / EXPERTS);
            assert_eq!(usize::from(id), writes % EXPERTS);
            bytes[range.offset..range.offset + range.bytes].fill(id as u8);
            writes += 1;
            Ok(range.bytes)
        });
        assert!(
            store.is_ok(),
            "all exact expert writes must seal the immutable layout"
        );
        let store = store.unwrap();
        assert_eq!(writes, 2 * EXPERTS);
        let second = store.layout().layer(1).unwrap();
        assert_eq!(
            second.source(255).unwrap(),
            ByteRange {
                offset: 256 * 3 + 255 * 5,
                bytes: 5
            }
        );
        assert_eq!(
            second.slot(223).unwrap(),
            ByteRange {
                offset: 224 * 3 + 223 * 5,
                bytes: 5
            }
        );
        assert!(second.source(256).is_err());
        assert!(second.slot(224).is_err());
        assert!(store.layout().layer(2).is_err());
        for layer in 0..2 {
            for id in 0..256 {
                let range = store.layout().layer(layer).unwrap().source(id).unwrap();
                assert!(
                    bytes[range.offset..range.offset + range.bytes]
                        .iter()
                        .all(|&byte| byte == id as u8)
                );
            }
        }
        for strides in [
            vec![],
            vec![0],
            vec![usize::MAX / EXPERTS + 1],
            vec![usize::MAX / EXPERTS; 2],
        ] {
            assert!(HostLayout::new(&strides, SLOTS).is_err());
        }
        assert!(
            HostStore::build(HostLayout::new(&[3], SLOTS).unwrap(), |_, _, range| Ok(
                range.bytes - 1
            ))
            .is_err()
        );
        let fixture_owner = std::rc::Rc::new(vec![0u8; 32]);
        let released = std::rc::Rc::downgrade(&fixture_owner);
        let mut partial_writes = 0;
        let failed = HostStore::build(
            HostLayout::new(&[3], SLOTS).unwrap(),
            move |_, id, range| {
                assert_eq!(fixture_owner.len(), 32);
                partial_writes += 1;
                if id == 13 {
                    assert_eq!(partial_writes, 14);
                    return Err(cache_error("fixture acquisition failed"));
                }
                Ok(range.bytes)
            },
        );
        assert!(failed.is_err());
        assert!(
            released.upgrade().is_none(),
            "failed construction must drop its owned fill resources"
        );
    }

    #[test]
    fn touching_orders_the_next_evictions_without_copying() {
        let store = HostStore::build(HostLayout::new(&[3], SLOTS).unwrap(), |_, _, range| {
            Ok(range.bytes)
        })
        .unwrap();
        let mut cache = LayerCache::new(&store, 0, 7).unwrap();
        cache.publish_initial(7, |_| Ok(())).unwrap();
        // Experts 0, 1 and 2 are the oldest. Touching them in the order 2, 0, 1 makes expert 1 the
        // newest and expert 3 the oldest, so the next two misses evict 3 and 4, not 0, 1 or 2.
        cache.touch(&[2, 0, 1, 250]).unwrap();
        assert_eq!(cache.recency[SLOTS - 1], 1);
        assert_eq!(cache.recency[SLOTS - 2], 0);
        assert_eq!(cache.recency[SLOTS - 3], 2);
        assert_eq!(cache.recency[0], 3);
        let pass = cache.prepare(&[224, 225], usize::MAX).unwrap();
        let ready = cache
            .copy_ready(pass, 7, |copies| {
                let victims: Vec<u16> = copies.iter().map(|copy| copy.slot).collect();
                assert_eq!(victims, [3, 4], "the oldest slots are replaced");
                Ok(())
            })
            .unwrap();
        cache.release(ready, 7, || Ok(())).unwrap();
        // A touch outside the idle phase is refused and changes nothing.
        let pass = cache.prepare(&[1], usize::MAX).unwrap();
        let before = cache.recency.clone();
        assert!(cache.touch(&[5]).is_err());
        assert_eq!(cache.recency, before);
        let _ = pass;
    }

    #[test]
    fn protected_lru_bounded_misses_preserve_duplicate_touch_order() {
        let store = HostStore::build(HostLayout::new(&[3], SLOTS).unwrap(), |_, _, range| {
            Ok(range.bytes)
        })
        .unwrap();
        let mut cache = LayerCache::new(&store, 0, 7).unwrap();
        cache
            .publish_initial(7, |copies| {
                assert_eq!(copies.len(), SLOTS);
                for (index, copy) in copies.iter().enumerate() {
                    assert_eq!(usize::from(copy.expert), index);
                    assert_eq!(usize::from(copy.slot), index);
                }
                Ok(())
            })
            .unwrap();
        for ids in [&[-1][..], &[256][..], &(0..225).collect::<Vec<_>>()[..]] {
            assert!(cache.prepare(ids, usize::MAX).is_err());
        }
        assert!(cache.prepare(&[224, 0, 224, 1], 2).is_err());
        let pass = cache.prepare(&[224, 0, 224, 1], 3).unwrap();
        assert!(cache.prepare(&[3], 3).is_err());
        assert!(
            cache
                .copy_ready(pass, 8, |_| panic!("wrong stream callback"))
                .is_err()
        );
        let ready = cache
            .copy_ready(pass, 7, |copies| {
                assert_eq!(
                    copies,
                    &[MissCopy {
                        expert: 224,
                        slot: 2,
                        source: ByteRange {
                            offset: 224 * 3,
                            bytes: 3
                        },
                        destination: ByteRange {
                            offset: 2 * 3,
                            bytes: 3
                        },
                    }]
                );
                Ok(())
            })
            .unwrap();
        assert_eq!(cache.slot(ready, 224).unwrap(), 2);
        assert_eq!(cache.slot(ready, 0).unwrap(), 0);
        assert_eq!(cache.slot(ready, 1).unwrap(), 1);
        assert!(cache.slot(ready, 2).is_err());
        cache.release(ready, 7, || Ok(())).unwrap();
        let mut ids = vec![225];
        ids.extend(3..224);
        ids.push(1);
        let next = cache.prepare(&ids, 3).unwrap();
        assert!(
            cache
                .copy_ready(pass, 7, |_| panic!("stale pass callback"))
                .is_err()
        );
        let ready = cache
            .copy_ready(next, 7, |copies| {
                assert_eq!(copies.len(), 1);
                assert_eq!(copies[0].expert, 225);
                assert_eq!(
                    copies[0].slot, 0,
                    "duplicate touches must make 224 newer than 0"
                );
                Ok(())
            })
            .unwrap();
        assert_eq!(cache.slot(ready, 225).unwrap(), 0);
        cache.release(ready, 7, || Ok(())).unwrap();
    }

    #[test]
    fn lifecycle_poison_drain_and_canonical_republication() {
        let store = HostStore::build(HostLayout::new(&[3], SLOTS).unwrap(), |_, _, range| {
            Ok(range.bytes)
        })
        .unwrap();
        let mut cache = LayerCache::new(&store, 0, 7).unwrap();
        let foreign = LayerCache::new(&store, 0, 7).unwrap().key;
        drop(store); // Geometry is owned; this seal never owned or pinned bytes.
        assert!(cache.prepare(&[0], 0).is_err());
        assert!(cache.slot(ReadyPass(cache.key), 0).is_err());
        assert!(
            cache
                .publish_initial(8, |_| panic!("wrong stream"))
                .is_err()
        );
        assert!(
            cache
                .reset_after_drain(7, || panic!("wrong phase"))
                .is_err()
        );
        let mut order = Vec::new();
        cache
            .publish_initial(7, |copies| {
                assert_eq!(copies.len(), SLOTS);
                order.push("initial");
                Ok(())
            })
            .unwrap();
        let before = snapshot(&cache);
        for ids in [vec![-1], vec![256], (0..225).collect()] {
            assert!(cache.prepare(&ids, usize::MAX).is_err());
            assert_eq!(snapshot(&cache), before);
        }
        assert!(cache.prepare(&[224], 2).is_err());
        assert_eq!(snapshot(&cache), before);
        let pass = cache.prepare(&[224, 0, 224, 1], 3).unwrap();
        assert_eq!(cache.slots[2], None);
        assert_eq!(cache.slots[224], Some(2));
        assert!(cache.prepare(&[225], 3).is_err());
        assert!(
            cache
                .copy_ready(foreign, 7, |_| panic!("foreign key"))
                .is_err()
        );
        assert!(
            cache
                .copy_ready(pass, 8, |_| panic!("wrong stream"))
                .is_err()
        );
        let ready = cache
            .copy_ready(pass, 7, |_| {
                order.push("copy");
                Ok(())
            })
            .unwrap();
        assert!(cache.prepare(&[225], 3).is_err());
        assert!(
            cache
                .release::<()>(ReadyPass(foreign), 7, || panic!("foreign ready"))
                .is_err()
        );
        assert!(
            cache
                .release::<()>(ready, 8, || panic!("wrong finish stream"))
                .is_err()
        );
        assert_eq!(cache.slot(ready, 224).unwrap(), 2);
        let value = cache
            .release(ready, 7, || {
                order.push("finish");
                Ok(41)
            })
            .unwrap();
        assert_eq!(value, 41);
        assert_eq!(order, ["initial", "copy", "finish"]);
        assert!(
            cache
                .release::<()>(ready, 7, || panic!("repeated finish"))
                .is_err()
        );
        assert!(cache.slot(ready, 224).is_err());
        let next = cache.prepare(&[225], 3).unwrap();
        assert!(cache.copy_ready(pass, 7, |_| panic!("stale copy")).is_err());
        assert!(
            cache
                .copy_ready(next, 7, |_| Err(cache_error("copy failed")))
                .is_err()
        );
        assert!(cache.phase == Phase::Poisoned);
        let held = snapshot(&cache);
        assert!(
            cache
                .reset_after_drain(8, || panic!("wrong drain stream"))
                .is_err()
        );
        assert!(
            cache
                .reset_after_drain(7, || Err(cache_error("drain failed")))
                .is_err()
        );
        assert!(cache.phase == Phase::Poisoned);
        assert_eq!(snapshot(&cache), held);
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _ = cache.reset_after_drain(7, || panic!("drain panic"));
            }))
            .is_err()
        );
        assert!(cache.phase == Phase::Poisoned);
        assert!(cache.prepare(&[0], 0).is_err());
        cache
            .reset_after_drain(7, || {
                order.push("drain");
                Ok(())
            })
            .unwrap();
        assert!(cache.key.generation > next.generation);
        assert!(
            cache.prepare(&[0], 0).is_err(),
            "drain must not publish canonical bytes"
        );
        assert!(cache.slot(ready, 224).is_err());
        cache
            .publish_initial(7, |copies| {
                for (id, copy) in copies.iter().enumerate() {
                    assert_eq!((copy.expert, copy.slot), (id as u16, id as u16));
                }
                order.push("reinitialize");
                Ok(())
            })
            .unwrap();
        assert_eq!(
            order,
            ["initial", "copy", "finish", "drain", "reinitialize"]
        );
        for id in 0..EXPERTS {
            assert_eq!(cache.slots[id], (id < SLOTS).then_some(id as u16));
        }
        let pass = cache.prepare(&[0], 0).unwrap();
        assert!(
            cache
                .copy_ready(next, 7, |_| panic!("pre-reset key"))
                .is_err()
        );
        let ready = cache
            .copy_ready(pass, 7, |copies| {
                assert!(copies.is_empty());
                Ok(())
            })
            .unwrap();
        assert!(cache.slot(ReadyPass(next), 0).is_err());
        assert!(
            cache
                .release(ready, 7, || Err::<(), _>(cache_error("finish failed")))
                .is_err()
        );
        assert!(cache.phase == Phase::Poisoned && cache.demanded[0]);
        assert!(cache.prepare(&[1], 0).is_err());
        assert!(
            cache
                .release::<()>(ready, 7, || panic!("failed finish retry"))
                .is_err()
        );
        cache.reset_after_drain(7, || Ok(())).unwrap();
        assert!(
            cache
                .publish_initial(7, |_| Err(cache_error("initial failed")))
                .is_err()
        );
        assert!(cache.phase == Phase::Poisoned && cache.copy_count == SLOTS);
        cache.reset_after_drain(7, || Ok(())).unwrap();
        cache.publish_initial(7, |_| Ok(())).unwrap();
        let pass = cache.prepare(&[0], 0).unwrap();
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _ = cache.copy_ready(pass, 7, |_| panic!("copy panic"));
            }))
            .is_err()
        );
        assert!(cache.phase == Phase::Poisoned && cache.demanded[0]);
        cache.reset_after_drain(7, || Ok(())).unwrap();
        cache.publish_initial(7, |_| Ok(())).unwrap();
        let pass = cache.prepare(&[0], 0).unwrap();
        let ready = cache.copy_ready(pass, 7, |_| Ok(())).unwrap();
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _ = cache.release::<()>(ready, 7, || panic!("finish panic"));
            }))
            .is_err()
        );
        assert!(cache.phase == Phase::Poisoned && cache.demanded[0]);
        cache.reset_after_drain(7, || Ok(())).unwrap();
        cache.publish_initial(7, |_| Ok(())).unwrap();
        cache.poison();
        cache.poison();
        assert!(cache.prepare(&[0], 0).is_err());
        cache.reset_after_drain(7, || Ok(())).unwrap();
        assert!(cache.phase == Phase::Initial);
    }

    #[test]
    fn captured_cgroup_roots_match_real_admission() {
        let captures: serde_json::Value =
            serde_json::from_str(include_str!("testdata/cgroup-views-20261028.json")).unwrap();
        let hostns: serde_json::Value =
            serde_json::from_str(include_str!("testdata/cgroup-hostns-20261028.json")).unwrap();
        let host_files = captures["host"]["root_file_names"].as_array().unwrap();
        for name in ["cgroup.type", "memory.max", "memory.current"] {
            assert!(!host_files.iter().any(|value| value.as_str() == Some(name)));
        }
        for (name, view, accepted) in [
            ("container", &captures["container"], false),
            ("host", &captures["host"], true),
            ("hostns", &hostns, true),
        ] {
            let root = tempfile::tempdir().unwrap();
            let directory = root.path().canonicalize().unwrap();
            let controllers = view["controllers"].as_str().unwrap();
            assert!(
                controllers
                    .split_whitespace()
                    .any(|value| value == "memory")
            );
            std::fs::write(directory.join("cgroup.controllers"), controllers).unwrap();
            for (file, key) in [
                ("cgroup.type", "cgroup_type"),
                ("memory.max", "memory_max"),
                ("memory.current", "memory_current"),
            ] {
                if let Some(contents) = view[key].as_str() {
                    std::fs::write(directory.join(file), contents).unwrap();
                }
            }
            let mounts = view["mountinfo"]
                .as_str()
                .unwrap()
                .replace("/sys/fs/cgroup", &directory.display().to_string());
            let result = verify_cgroup_root(&directory, &mounts);
            eprintln!("captured view {name}: accepted={}", result.is_ok());
            if accepted {
                assert!(result.is_ok(), "{name}: {result:?}");
            } else {
                assert!(matches!(result, Err(EngError::MemoryPlan(message))
                    if message.contains("one complete cgroup2 root")));
            }
        }
    }

    #[test]
    fn captured_cgroup_host_member_maps_to_nested_group() {
        let captures: serde_json::Value =
            serde_json::from_str(include_str!("testdata/cgroup-views-20261028.json")).unwrap();
        let view = &captures["host"];
        let root = tempfile::tempdir().unwrap();
        let directory = root.path().canonicalize().unwrap();
        std::fs::write(
            directory.join("cgroup.controllers"),
            view["controllers"].as_str().unwrap(),
        )
        .unwrap();
        let membership = view["membership"].as_str().unwrap();
        let group = directory.join(membership.trim().strip_prefix("0::/").unwrap());
        std::fs::create_dir_all(&group).unwrap();
        std::fs::write(
            group.join("cgroup.type"),
            view["member_type"].as_str().unwrap(),
        )
        .unwrap();
        std::fs::write(
            group.join("cgroup.procs"),
            view["member_procs"].as_str().unwrap(),
        )
        .unwrap();
        let mounts = view["mountinfo"]
            .as_str()
            .unwrap()
            .replace("/sys/fs/cgroup", &directory.display().to_string());
        let pid = u32::try_from(view["pid"].as_u64().unwrap()).unwrap();
        let result = cgroup_from_proc(&directory, membership, &mounts, pid);
        assert_eq!(result.unwrap(), group);
        eprintln!("captured host member: mapped=true pid={pid}");
    }

    fn discovery_fixture() -> (tempfile::TempDir, std::path::PathBuf, String) {
        let root = tempfile::tempdir().unwrap();
        let directory = root.path().canonicalize().unwrap();
        let group = directory.join("work");
        std::fs::create_dir(&group).unwrap();
        std::fs::write(directory.join("cgroup.controllers"), "memory\n").unwrap();
        std::fs::write(group.join("cgroup.type"), "domain\n").unwrap();
        std::fs::write(group.join("cgroup.procs"), "41\n42\n").unwrap();
        let mounts = format!(
            "36 29 0:33 / {} rw - cgroup2 cgroup rw\n",
            directory.display()
        );
        (root, group, mounts)
    }

    #[test]
    fn cgroup_discovery_resolves_own_pid_in_global_tree() {
        let (root, group, mounts) = discovery_fixture();
        let resolved = cgroup_from_proc(root.path(), "0::/work\n", &mounts, 42);
        assert!(
            resolved.is_ok(),
            "a complete global cgroup tree must resolve its own process group"
        );
        assert_eq!(resolved.unwrap(), group);
    }

    #[cfg(unix)]
    #[test]
    fn cgroup_discovery_matches_canonical_mount_paths() {
        let (root, group, mounts) = discovery_fixture();
        let aliases = tempfile::tempdir().unwrap();
        let alias = aliases.path().join("controller-root");
        std::os::unix::fs::symlink(root.path(), &alias).unwrap();
        let resolved = cgroup_from_proc(&alias, "0::/work\n", &mounts, 42).unwrap();
        assert_eq!(resolved, group);
    }

    #[test]
    fn cgroup_discovery_rejects_hidden_or_ambiguous_roots() {
        for (name, contents) in [
            ("cgroup.type", "domain\n"),
            ("memory.max", "max\n"),
            ("memory.current", "0\n"),
            ("cgroup.controllers", "cpu\n"),
        ] {
            let (root, _, mounts) = discovery_fixture();
            std::fs::write(root.path().join(name), contents).unwrap();
            let result = cgroup_from_proc(root.path(), "0::/work\n", &mounts, 42);
            assert!(
                matches!(result, Err(EngError::MemoryPlan(message))
                    if message.contains("complete cgroup2 root")),
                "must reject root interface {name}"
            );
        }
        for (source, filesystem, copies) in [
            ("/delegated", "cgroup2", 1),
            ("/", "tmpfs", 1),
            ("/", "cgroup2", 2),
        ] {
            let (root, group, _) = discovery_fixture();
            let mounts = format!(
                "36 29 0:33 {source} {} rw - {filesystem} cgroup rw\n",
                group.parent().unwrap().display()
            )
            .repeat(copies);
            let result = cgroup_from_proc(root.path(), "0::/work\n", &mounts, 42);
            assert!(
                matches!(result, Err(EngError::MemoryPlan(message))
                    if message.contains("complete cgroup2 root")),
                "must reject mount {source}/{filesystem}/{copies}"
            );
        }
    }

    /// A container's view: the cgroup root is the container's own cgroup, with limit files.
    fn namespace_root(maximum: &str, controllers: &str) -> (tempfile::TempDir, tempfile::TempDir) {
        let root = tempfile::tempdir().unwrap();
        let directory = root.path().canonicalize().unwrap();
        for (name, contents) in [
            ("cgroup.controllers", controllers),
            ("cgroup.subtree_control", ""),
            ("cgroup.type", "domain\n"),
            ("cgroup.procs", "1\n42\n"),
            ("memory.max", maximum),
            ("memory.current", "2147483648\n"),
        ] {
            std::fs::write(directory.join(name), contents).unwrap();
        }
        let proc = tempfile::tempdir().unwrap();
        std::fs::create_dir(proc.path().join("self")).unwrap();
        std::fs::write(
            proc.path().join("meminfo"),
            "MemTotal: 33554432 kB\nMemAvailable: 25165824 kB\n",
        )
        .unwrap();
        std::fs::write(proc.path().join("self/cgroup"), "0::/\n").unwrap();
        std::fs::write(
            proc.path().join("self/mountinfo"),
            format!(
                "36 29 0:33 / {} rw - cgroup2 cgroup rw\n",
                directory.display()
            ),
        )
        .unwrap();
        (root, proc)
    }

    #[test]
    fn a_namespace_root_applies_its_own_limit_or_falls_back_to_available_memory() {
        let (root, proc) = namespace_root("8589934592\n", "memory\n");
        let budget = HostBudget::probe_in(proc.path(), root.path(), 42).unwrap();
        assert_eq!(budget.cgroup_remaining, Some(6u64 << 30));
        assert_eq!(budget.os_floor, 8u64 << 30);
        // 6 GiB left in the cgroup, 8 GiB floor: nothing is admitted.
        assert!(budget.admit(1).is_err());
        let (root, proc) = namespace_root("max\n", "memory\n");
        let budget = HostBudget::probe_in(proc.path(), root.path(), 42).unwrap();
        assert_eq!(budget.cgroup_remaining, None);
        // MemAvailable is 24 GiB; the floor is max(8 GiB, 10% of 32 GiB) = 8 GiB.
        assert!(budget.admit(16u64 << 30).is_ok());
        assert!(budget.admit((16u64 << 30) + 1).is_err());
    }

    #[test]
    fn a_root_with_some_but_not_all_namespace_files_is_refused() {
        let files = [
            ("cgroup.type", "domain\n"),
            ("memory.max", "max\n"),
            ("memory.current", "0\n"),
        ];
        for skipped in 0..files.len() {
            let (root, _, mounts) = discovery_fixture();
            for (index, (name, contents)) in files.iter().enumerate() {
                if index != skipped {
                    std::fs::write(root.path().join(name), contents).unwrap();
                }
            }
            let result = cgroup_from_proc(root.path(), "0::/work\n", &mounts, 42);
            assert!(
                matches!(result, Err(EngError::MemoryPlan(message))
                    if message.contains("complete cgroup2 root")),
                "two of three root files without {} must be refused",
                files[skipped].0
            );
        }
    }

    #[test]
    fn a_namespace_root_without_the_memory_controller_is_refused_clearly() {
        let (root, proc) = namespace_root("max\n", "cpu\n");
        let result = HostBudget::probe_in(proc.path(), root.path(), 42);
        assert!(matches!(result, Err(EngError::MemoryPlan(message))
                if message.contains("complete cgroup2 root with the memory controller")));
    }

    #[test]
    fn cgroup_discovery_rejects_bad_process_mapping() {
        for (membership, pid, owners, kind) in [
            ("0::/../work\n", 42, "41\n42\n", "domain"),
            ("0::work\n", 42, "41\n42\n", "domain"),
            ("0::/work\n0::/work\n", 42, "41\n42\n", "domain"),
            ("1:memory:/work\n", 42, "41\n42\n", "domain"),
            ("0::/work\n", 0, "41\n42\n", "domain"),
            ("0::/work\n", 42, "41\n", "domain"),
            ("0::/work\n", 42, "41\n42\n", "threaded"),
            ("0::/work\n", 42, "41\n42\nnot-a-pid\n", "domain"),
        ] {
            let (root, group, mounts) = discovery_fixture();
            std::fs::write(group.join("cgroup.procs"), owners).unwrap();
            std::fs::write(group.join("cgroup.type"), kind).unwrap();
            let result = cgroup_from_proc(root.path(), membership, &mounts, pid);
            assert!(
                matches!(result, Err(EngError::MemoryPlan(message))
                    if message.contains("cannot map its process")),
                "must reject process mapping {membership:?}/{pid}/{owners:?}/{kind}"
            );
        }
    }

    #[test]
    fn cgroup_reader_rejects_missing_current_with_unlimited_max() {
        let root = tempfile::tempdir().unwrap();
        let child = root.path().join("child");
        std::fs::create_dir(&child).unwrap();
        std::fs::write(root.path().join("cgroup.controllers"), "memory\n").unwrap();
        std::fs::write(root.path().join("cgroup.subtree_control"), "memory\n").unwrap();
        std::fs::write(child.join("memory.max"), "max\n").unwrap();
        let result = cgroup_remaining_in(root.path(), &child);
        assert!(
            matches!(
                result,
                Err(EngError::MemoryPlan(message)) if message.contains("memory.current")
            ),
            "an unreadable active charge file must not become an unlimited budget"
        );
    }

    #[test]
    fn cgroup_reader_applies_finite_ancestor_charges() {
        let root = tempfile::tempdir().unwrap();
        let parent = root.path().join("parent");
        let child = parent.join("child");
        std::fs::create_dir_all(&child).unwrap();
        std::fs::write(root.path().join("cgroup.controllers"), "memory\n").unwrap();
        for directory in [root.path(), parent.as_path()] {
            std::fs::write(directory.join("cgroup.subtree_control"), "memory\n").unwrap();
        }
        for (directory, maximum, current) in [
            (&parent, "17179869184\n", "4294967296\n"),
            (&child, "68719476736\n", "1073741824\n"),
        ] {
            std::fs::write(directory.join("memory.max"), maximum).unwrap();
            std::fs::write(directory.join("memory.current"), current).unwrap();
        }
        let remaining = cgroup_remaining_in(root.path(), &child);
        assert!(
            remaining.is_ok(),
            "a readable hierarchy must honor finite ancestor charges"
        );
        assert_eq!(remaining.unwrap(), Some(12u64 << 30));
    }

    #[test]
    fn cgroup_snapshot_accounts_for_current_charges() {
        let remaining = cgroup_remaining_bytes("17179869184\n", "4294967296\n");
        assert!(
            remaining.is_ok(),
            "valid finite cgroup limit and current charges must produce a budget"
        );
        let remaining = remaining.unwrap();
        assert_eq!(remaining, Some(12u64 << 30));
        let budget = HostBudget::from_meminfo(
            "MemTotal: 33554432 kB\nMemAvailable: 25165824 kB\n",
            remaining,
        )
        .unwrap();
        assert!(budget.admit(4u64 << 30).is_ok());
        assert!(budget.admit((4u64 << 30) + 1).is_err());
    }

    #[test]
    fn cgroup_snapshot_distinguishes_unlimited_and_exhausted_limits() {
        for (maximum, current, remaining) in [
            ("max\n", "0\n", None),
            (" max\n", "18446744073709551615\n", None),
            ("0", "0", Some(0)),
            ("1024", "1024", Some(0)),
            ("4", "8", Some(0)),
            ("18446744073709551615", "0", Some(u64::MAX)),
            ("18446744073709551615", "18446744073709551615", Some(0)),
        ] {
            assert_eq!(
                cgroup_remaining_bytes(maximum, current).unwrap(),
                remaining,
                "{maximum:?} / {current:?}"
            );
        }
    }

    #[test]
    fn cgroup_snapshot_rejects_malformed_limits_and_current_charges() {
        for (maximum, current) in [
            ("", "0"),
            ("MAX", "0"),
            ("bad", "0"),
            ("1 kB", "0"),
            ("1 2", "0"),
            ("1.5", "0"),
            ("-1", "0"),
            ("+1", "0"),
            ("18446744073709551616", "0"),
            ("1", ""),
            ("1", "bad"),
            ("1", "1 kB"),
            ("1", "-1"),
            ("1", "+1"),
            ("1", "18446744073709551616"),
            ("max", ""),
            ("max", "bad"),
            ("max", "max"),
            ("max", "-1"),
            ("max", "18446744073709551616"),
        ] {
            assert!(
                cgroup_remaining_bytes(maximum, current).is_err(),
                "{maximum:?} / {current:?}"
            );
        }
    }

    #[test]
    fn host_snapshot_uses_mem_available_and_model_scaled_os_floor() {
        let result = HostBudget::from_meminfo(
            "MemTotal:       268435456 kB\nMemAvailable:    50331648 kB\n",
            None,
        );
        assert!(
            result.is_ok(),
            "valid memory snapshot must produce an admission budget"
        );
        let budget = result.unwrap();
        assert_eq!(budget.mem_available, 48u64 << 30);
        assert_eq!(budget.os_floor, 27_487_790_695);
        assert_eq!(budget.cgroup_remaining, None);
        assert!(budget.admit(256 * 47 * 1_603_584u64).is_ok());
    }

    #[test]
    fn host_snapshot_preserves_minimum_floor_and_finite_cgroup() {
        let text = "MemAvailable: 25165824 kB\nMemFree: 0 kB\nMemTotal: 33554432 kB\n";
        let floor = 8u64 << 30;
        let budget = HostBudget::from_meminfo(text, Some(floor + 1)).unwrap();
        assert_eq!(budget.os_floor, floor);
        assert_eq!(budget.cgroup_remaining, Some(floor + 1));
        assert!(budget.admit(1).is_ok());
        assert!(budget.admit(2).is_err());
        assert!(
            HostBudget::from_meminfo(text, Some(0))
                .unwrap()
                .admit(0)
                .is_err()
        );
        let empty = HostBudget::from_meminfo("MemTotal: 1 kB\nMemAvailable: 0 kB\n", None).unwrap();
        assert_eq!(empty.mem_available, 0);
        assert!(empty.admit(0).is_err());
    }

    #[test]
    fn host_snapshot_rejects_malformed_or_inconsistent_fields() {
        for text in [
            "",
            "MemTotal: 1 kB\n",
            "MemAvailable: 1 kB\n",
            "MemTotal: 0 kB\nMemAvailable: 0 kB\n",
            "MemTotal: 1 kB\nMemAvailable: 2 kB\n",
            "MemTotal: -1 kB\nMemAvailable: 0 kB\n",
            "MemTotal: 1 MB\nMemAvailable: 0 kB\n",
            "MemTotal: 1 kB\nMemAvailable: bad kB\n",
            "MemTotal: 1 kB\nMemAvailable: 0 MB\n",
            "MemTotal: 1 kB trailing\nMemAvailable: 0 kB\n",
            "MemTotal: 1 kB\nMemTotal: 1 kB\nMemAvailable: 0 kB\n",
            "MemTotal: 1 kB\nMemAvailable: 0 kB\nMemAvailable: 0 kB\n",
            "MemTotal: 18014398509481984 kB\nMemAvailable: 0 kB\n",
            "MemTotal: 1 kB\nMemAvailable: 18014398509481984 kB\n",
        ] {
            assert!(HostBudget::from_meminfo(text, None).is_err(), "{text:?}");
        }
    }

    #[test]
    fn host_budget_rejects_unrepresentable_payload_plus_floor() {
        let budget = HostBudget {
            mem_available: u64::MAX,
            cgroup_remaining: None,
            os_floor: 1,
        };
        assert!(budget.admit(u64::MAX).is_err());
        assert!(budget.admit(u64::MAX - 1).is_ok());
    }

    #[test]
    fn host_budget_accepts_exact_model_demand_only_after_os_floor() {
        let required = 256 * 47 * 1_603_584u64;
        let floor = 8u64 << 30;
        let exact = HostBudget {
            mem_available: required + floor,
            cgroup_remaining: Some(required + floor),
            os_floor: floor,
        };
        assert!(
            exact.admit(required).is_ok(),
            "exact model demand plus OS floor must admit"
        );
        assert!(
            HostBudget {
                mem_available: exact.mem_available - 1,
                ..exact
            }
            .admit(required)
            .is_err()
        );
        assert!(
            HostBudget {
                cgroup_remaining: Some(required + floor - 1),
                ..exact
            }
            .admit(required)
            .is_err()
        );
        assert!(
            HostBudget {
                cgroup_remaining: None,
                ..exact
            }
            .admit(required)
            .is_ok(),
            "an unlimited cgroup still obeys host availability and the OS floor"
        );
    }

    #[test]
    fn full_slot_union_exact_miss_budget_is_atomic_and_protects_demand() {
        let stride = 257;
        let store = HostStore::build(HostLayout::new(&[stride], SLOTS).unwrap(), |_, _, range| {
            Ok(range.bytes)
        })
        .unwrap();
        let mut cache = LayerCache::new(&store, 0, 7).unwrap();
        cache.publish_initial(7, |_| Ok(())).unwrap();
        let ids: Vec<i32> = (32..256).collect();
        let before = snapshot(&cache);
        assert!(cache.prepare(&ids, 32 * stride - 1).is_err());
        assert_eq!(snapshot(&cache), before);
        let pass = cache.prepare(&ids, 32 * stride).unwrap();
        let expected: Vec<_> = (224..256)
            .map(|id| (id as u16, (id - 224) as u16))
            .collect();
        let ready = cache
            .copy_ready(pass, 7, |copies| {
                assert_eq!(copies.len(), 32);
                assert_eq!(
                    copies
                        .iter()
                        .map(|copy| (copy.expert, copy.slot))
                        .collect::<Vec<_>>(),
                    expected
                );
                let mut victims = std::collections::BTreeSet::new();
                for copy in copies {
                    assert!(victims.insert(copy.slot));
                    assert!((copy.slot as usize) < 32);
                    assert_eq!(copy.source, store.layout().layer(0)?.source(copy.expert)?);
                    assert_eq!(copy.destination, store.layout().layer(0)?.slot(copy.slot)?);
                    assert!(copy.source.offset + copy.source.bytes <= store.layout().host_bytes());
                    assert!(
                        copy.destination.offset + copy.destination.bytes
                            <= store.layout().slot_bytes()
                    );
                }
                Ok(())
            })
            .unwrap();
        for id in 32..224 {
            assert_eq!(cache.slot(ready, id).unwrap(), id);
        }
        for id in 224..256 {
            assert_eq!(cache.slot(ready, id).unwrap(), id - 224);
        }
        cache.release(ready, 7, || Ok(())).unwrap();
    }

    #[test]
    fn layout_sizes_follow_the_resident_count() {
        for slots in [1, 48, 192, 224, 256] {
            let layout = HostLayout::new(&[256, 512], slots).unwrap();
            assert_eq!(layout.slot_count(), slots);
            assert_eq!(layout.host_bytes(), 256 * (256 + 512));
            assert_eq!(layout.slot_bytes(), slots * (256 + 512));
            let last = layout.layer(1).unwrap().slot(slots as u16 - 1).unwrap();
            assert_eq!(last.offset + last.bytes, layout.slot_bytes());
            assert!(layout.layer(1).unwrap().slot(slots as u16).is_err());
        }
        for slots in [0, 257, usize::MAX] {
            assert!(HostLayout::new(&[256], slots).is_err(), "slots={slots}");
        }
    }

    // An independent least-recently-used bank: experts in recency order, oldest first.
    fn reference_pass(bank: &mut Vec<u16>, ids: &[i32]) -> Vec<u16> {
        let demanded: Vec<u16> = ids.iter().map(|&id| id as u16).collect();
        let mut missed = Vec::new();
        for &expert in &demanded {
            if let Some(at) = bank.iter().position(|&held| held == expert) {
                bank.remove(at);
            } else {
                let victim = bank
                    .iter()
                    .position(|held| !demanded.contains(held))
                    .expect("an unprotected victim");
                bank.remove(victim);
                missed.push(expert);
            }
            bank.push(expert);
        }
        missed
    }

    #[test]
    fn any_resident_count_evicts_the_lru_expert_outside_the_pass() {
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut next = move |bound: u64| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed % bound
        };
        for slots in [48, 96, 192, 224, 256] {
            let store = HostStore::build(HostLayout::new(&[3], slots).unwrap(), |_, _, range| {
                Ok(range.bytes)
            })
            .unwrap();
            let mut cache = LayerCache::new(&store, 0, 7).unwrap();
            cache
                .publish_initial(7, |copies| {
                    assert_eq!(copies.len(), slots);
                    Ok(())
                })
                .unwrap();
            let mut bank: Vec<u16> = (0..slots as u16).collect();
            for _ in 0..300 {
                // A decode step touches 6 experts, a prefill piece up to 48; both fit 48 slots.
                let width = [1, 6, 6, 24, 48][next(5) as usize];
                let ids: Vec<i32> = (0..width).map(|_| next(256) as i32).collect();
                let unique: std::collections::BTreeSet<_> = ids.iter().collect();
                if unique.len() > slots {
                    continue;
                }
                let expected = reference_pass(&mut bank, &ids);
                let pass = cache.prepare(&ids, usize::MAX).unwrap();
                let ready = cache
                    .copy_ready(pass, 7, |copies| {
                        let missed: Vec<u16> = copies.iter().map(|copy| copy.expert).collect();
                        assert_eq!(missed, expected, "slots={slots}");
                        for copy in copies {
                            assert!(usize::from(copy.slot) < slots);
                        }
                        Ok(())
                    })
                    .unwrap();
                let mut used = std::collections::BTreeSet::new();
                for &id in &ids {
                    let slot = cache.slot(ready, id as u16).unwrap();
                    assert!(usize::from(slot) < slots);
                    assert_eq!(cache.residents[usize::from(slot)], id as u16);
                    used.insert((id, slot));
                }
                // One slot per demanded expert: a victim is never a member of the pass.
                assert_eq!(used.len(), unique.len());
                cache.release(ready, 7, || Ok(())).unwrap();
                let recency: Vec<u16> = cache
                    .recency
                    .iter()
                    .map(|&slot| cache.residents[usize::from(slot)])
                    .collect();
                assert_eq!(recency, bank, "slots={slots}");
            }
        }
    }

    #[test]
    fn a_pass_larger_than_the_bank_is_refused_before_any_eviction() {
        for slots in [48, 224] {
            let store = HostStore::build(HostLayout::new(&[3], slots).unwrap(), |_, _, range| {
                Ok(range.bytes)
            })
            .unwrap();
            let mut cache = LayerCache::new(&store, 0, 7).unwrap();
            cache.publish_initial(7, |_| Ok(())).unwrap();
            let before = snapshot(&cache);
            let ids: Vec<i32> = (0..=slots as i32).collect();
            assert!(cache.prepare(&ids, usize::MAX).is_err());
            assert_eq!(snapshot(&cache), before);
            // Exactly the bank's size is a legal pass; the experts it lacks are its misses.
            let ids: Vec<i32> = (256 - slots as i32..256).collect();
            let missing = ids.iter().filter(|&&id| id as usize >= slots).count();
            let pass = cache.prepare(&ids, usize::MAX).unwrap();
            let ready = cache
                .copy_ready(pass, 7, |copies| {
                    assert_eq!(copies.len(), missing);
                    Ok(())
                })
                .unwrap();
            cache.release(ready, 7, || Ok(())).unwrap();
        }
    }

    #[test]
    fn a_failed_recovery_republishes_the_full_bank_at_any_resident_count() {
        for slots in [48, 192] {
            let store = HostStore::build(HostLayout::new(&[3], slots).unwrap(), |_, _, range| {
                Ok(range.bytes)
            })
            .unwrap();
            let mut fresh = LayerCache::new(&store, 0, 7).unwrap();
            fresh.publish_initial(7, |_| Ok(())).unwrap();
            let mut cache = LayerCache::new(&store, 0, 7).unwrap();
            cache.publish_initial(7, |_| Ok(())).unwrap();
            // Move the bank off its canonical contents, then fail the copy of the last pass.
            for ids in [[250, 251, 252], [3, 250, 253]] {
                let pass = cache.prepare(&ids, usize::MAX).unwrap();
                let ready = cache.copy_ready(pass, 7, |_| Ok(())).unwrap();
                cache.release(ready, 7, || Ok(())).unwrap();
            }
            assert_ne!(bank_state(&cache), bank_state(&fresh));
            let pass = cache.prepare(&[254, 255], usize::MAX).unwrap();
            assert!(
                cache
                    .copy_ready(pass, 7, |_| Err(cache_error("copy failed")))
                    .is_err()
            );
            cache.reset_after_drain(7, || Ok(())).unwrap();
            cache
                .publish_initial(7, |copies| {
                    assert_eq!(copies.len(), slots);
                    Ok(())
                })
                .unwrap();
            // A recovered bank is indistinguishable from a freshly loaded one.
            assert_eq!(bank_state(&cache), bank_state(&fresh));
            for id in 0..EXPERTS {
                assert_eq!(cache.slots[id], (id < slots).then_some(id as u16));
            }
        }
    }

    const F16_BYTES_PER_TOKEN: u64 = 28_672;
    // One expert in each of the 47 routed layers of the published pack.
    const PER_SLOT: u64 = 47 * 1_603_584;

    #[test]
    fn published_arithmetic_pins_what_a_smaller_bank_frees() {
        // R = 224 frees the 2.246 GiB the plan counted, about 84,116 f16 context tokens.
        let freed = 32 * PER_SLOT;
        assert_eq!(freed, 2_411_790_336);
        assert_eq!(freed / F16_BYTES_PER_TOKEN, 84_116);
    }

    #[test]
    fn the_bank_is_the_largest_that_fits_and_never_more_than_all_256() {
        let budget = |free: u64| BankBudget {
            free,
            fixed: 1_000,
            per_slot: PER_SLOT,
        };
        let fit = |free: u64| choose_slots(BankRequest::Fit, budget(free), 48);
        assert_eq!(fit(1_000 + 224 * PER_SLOT).unwrap(), 224);
        assert_eq!(fit(1_000 + 224 * PER_SLOT + PER_SLOT - 1).unwrap(), 224);
        assert_eq!(fit(1_000 + 224 * PER_SLOT - 1).unwrap(), 223);
        assert_eq!(fit(1_000 + 256 * PER_SLOT).unwrap(), 256);
        assert_eq!(fit(u64::MAX).unwrap(), 256);
        assert_eq!(fit(1_000 + 48 * PER_SLOT).unwrap(), 48);
        // A longer context leaves less room: the bank shrinks with the KV cache in `fixed`.
        let longer = BankBudget {
            fixed: 1_000 + 2 * PER_SLOT,
            ..budget(1_000 + 224 * PER_SLOT)
        };
        assert_eq!(choose_slots(BankRequest::Fit, longer, 48).unwrap(), 222);
    }

    #[test]
    fn a_bank_below_the_prefill_floor_or_beyond_the_budget_is_refused() {
        let budget = BankBudget {
            free: 1_000 + 47 * PER_SLOT,
            fixed: 1_000,
            per_slot: PER_SLOT,
        };
        let Err(EngError::MemoryPlan(message)) = choose_slots(BankRequest::Fit, budget, 48) else {
            panic!("47 resident experts cannot serve a 48-expert piece")
        };
        assert!(
            message.contains("too few")
                && message.contains("room for 47")
                && message.contains("160 MiB bank margin"),
            "{message}"
        );
        assert!(choose_slots(BankRequest::Fit, budget, 47).is_ok());
        // Not even the non-routed weights fit.
        let none = BankBudget { free: 10, ..budget };
        assert!(choose_slots(BankRequest::Fit, none, 6).is_err());
        // An explicit count is honoured, or the load is refused: it never shrinks silently.
        let roomy = BankBudget {
            free: 1_000 + 100 * PER_SLOT,
            ..budget
        };
        assert_eq!(
            choose_slots(BankRequest::Exactly(64), roomy, 48).unwrap(),
            64
        );
        assert_eq!(
            choose_slots(BankRequest::Exactly(100), roomy, 48).unwrap(),
            100
        );
        let Err(EngError::MemoryPlan(message)) = choose_slots(BankRequest::Exactly(101), roomy, 48)
        else {
            panic!("101 resident experts exceed the budget")
        };
        assert!(
            message.contains(SLOTS_VAR)
                && message.contains("101")
                && message.contains("Lower the count"),
            "{message}"
        );
        let Err(EngError::MemoryPlan(message)) = choose_slots(BankRequest::Exactly(47), roomy, 48)
        else {
            panic!("47 resident experts cannot serve a 48-expert piece")
        };
        assert!(message.contains("Raise the count"), "{message}");
        assert!(!message.contains("Lower context_limit"), "{message}");
    }

    #[test]
    fn the_resident_count_override_is_strict() {
        use std::env::VarError;
        let ok = |value: &str| Ok::<_, VarError>(value.to_owned());
        assert_eq!(
            BankRequest::from_env(Err(VarError::NotPresent)).unwrap(),
            BankRequest::Fit
        );
        assert_eq!(BankRequest::from_env(ok("")).unwrap(), BankRequest::Fit);
        for (text, count) in [("1", 1), ("48", 48), ("224", 224), ("256", 256)] {
            assert_eq!(
                BankRequest::from_env(ok(text)).unwrap(),
                BankRequest::Exactly(count)
            );
        }
        for text in [
            "0", "257", "-1", "224.5", "all", "0x40", "1e2", "+", "2 24", "+48", "048", " 48",
            "48 ", " ", "٤٨",
        ] {
            assert!(BankRequest::from_env(ok(text)).is_err(), "{text}");
        }
        assert!(
            BankRequest::from_env(Err(VarError::NotUnicode(std::ffi::OsString::new()))).is_err()
        );
    }

    #[test]
    fn the_cache_switch_is_unset_auto_on_off_and_nothing_else() {
        use std::env::VarError;
        let mode = |text: &str| ExpertCacheMode::from_env(Ok(text.to_owned()));
        assert_eq!(
            ExpertCacheMode::from_env(Err(VarError::NotPresent)).unwrap(),
            ExpertCacheMode::Auto
        );
        for text in ["1", "true", "on", " true ", "ON", "TrUe"] {
            assert_eq!(mode(text).unwrap(), ExpertCacheMode::On, "{text}");
        }
        for text in ["0", "false", "off", " 0 ", "OFF", "False"] {
            assert_eq!(mode(text).unwrap(), ExpertCacheMode::Off, "{text}");
        }
        // The old rule read every other value as off. With auto as the default that would turn a
        // mistyped rollback into the opposite of what the operator meant, so it is refused.
        for text in [
            "", " ", "yes", "no", "2", "enabled", "auto", "true!", "01", "-1",
        ] {
            let Err(EngError::BadRequest(message)) = mode(text) else {
                panic!("{text:?} must be refused")
            };
            assert!(
                message.contains(CACHE_VAR) && message.contains("0, false"),
                "{message}"
            );
        }
        assert!(
            ExpertCacheMode::from_env(Err(VarError::NotUnicode(std::ffi::OsString::new())))
                .is_err()
        );
    }

    #[test]
    fn the_load_keeps_a_fitting_context_resident_and_banks_the_rest() {
        use BankReason::{Auto, Forced, Override};
        let plan = |mode, request, fits| plan_bank(mode, request, fits).unwrap();
        for fits in [true, false] {
            // Off never banks; an explicit count contradicts it.
            assert_eq!(
                plan(ExpertCacheMode::Off, BankRequest::Fit, fits),
                BankPlan::Resident
            );
            assert!(plan_bank(ExpertCacheMode::Off, BankRequest::Exactly(224), fits).is_err());
            // On always banks, and so does an explicit count in either other mode.
            assert_eq!(
                plan(ExpertCacheMode::On, BankRequest::Fit, fits),
                BankPlan::Bank(Forced)
            );
            for mode in [ExpertCacheMode::Auto, ExpertCacheMode::On] {
                assert_eq!(
                    plan(mode, BankRequest::Exactly(48), fits),
                    BankPlan::Bank(Override)
                );
            }
        }
        // Auto banks exactly when today's resident load would be refused.
        assert_eq!(
            plan(ExpertCacheMode::Auto, BankRequest::Fit, true),
            BankPlan::Resident
        );
        assert_eq!(
            plan(ExpertCacheMode::Auto, BankRequest::Fit, false),
            BankPlan::Bank(Auto)
        );
        let reasons = [Auto, Forced, Override].map(BankReason::as_str);
        assert_eq!(reasons, ["auto", "forced", "override"]);
    }

    #[test]
    fn composed_host_probe_uses_verified_tree_and_fails_closed() {
        let (root, group, mounts) = discovery_fixture();
        let proc = tempfile::tempdir().unwrap();
        std::fs::create_dir(proc.path().join("self")).unwrap();
        std::fs::write(
            proc.path().join("meminfo"),
            "MemTotal: 33554432 kB\nMemAvailable: 25165824 kB\n",
        )
        .unwrap();
        std::fs::write(proc.path().join("self/cgroup"), "0::/work\n").unwrap();
        std::fs::write(proc.path().join("self/mountinfo"), mounts).unwrap();
        std::fs::write(root.path().join("cgroup.subtree_control"), "memory\n").unwrap();
        std::fs::write(group.join("memory.max"), "17179869184\n").unwrap();
        std::fs::write(group.join("memory.current"), "1073741824\n").unwrap();
        let budget = HostBudget::probe_in(proc.path(), root.path(), 42).unwrap();
        assert_eq!(budget.cgroup_remaining, Some(15u64 << 30));
        assert!(budget.admit(7u64 << 30).is_ok());
        assert!(budget.admit((7u64 << 30) + 1).is_err());
        std::fs::remove_file(group.join("memory.current")).unwrap();
        assert!(HostBudget::probe_in(proc.path(), root.path(), 42).is_err());
    }
}
