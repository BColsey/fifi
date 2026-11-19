//! The host tier above the bank: an O_DIRECT ring and the LRU that governs
//! the resident spare places.
//!
//! The bank's [`ReplaceOnMissLru`](policy::ReplaceOnMissLru) decides which
//! expert a device SLOT holds; this module is the second, host-side level:
//! which expert each resident host PLACE holds (the LRU here), and how a
//! miss that no host place holds reaches RAM at all (the ring). The ring
//! stages aligned spans of the pack straight into pinned host memory, so
//! the kernel page cache never holds a second copy of expert bytes and the
//! admission budget stays honest.
//!
//! The ring never allocates: the caller hands it the pinned slot memory
//! (`cudaHostAlloc` on the CUDA path, leaked for the process lifetime like
//! every other bank owner), because a plain-`alloc` buffer is neither
//! pinned nor registered and an async H2D from it would not be truly
//! async. Records are read as ALIGNED SPANS — offset rounded down, end
//! rounded up to the sector — so files are never required to pad records;
//! the exact bytes are then placed where the record layout wants them.

use std::cell::Cell;
use std::fs::File;
use std::io;
use std::num::{NonZeroU64, NonZeroUsize};
// Every read_at call site here is Linux-gated (the shared retry loop in
// weights::expert_store carries the trait itself); the import must match
// or non-Linux clippy fails it as unused.
#[cfg(target_os = "linux")]
use std::os::unix::fs::FileExt;
// O_DIRECT exists only on Linux; the trait import must match the cfg on
// custom_flags below or non-Linux clippy fails it as unused.
#[cfg(target_os = "linux")]
use std::os::unix::fs::OpenOptionsExt;
use std::path::Path;

/// The alignment every direct read here is measured against: 4096, the
/// atomic write unit of modern drives and the interface alignment NVMe and
/// ext4 both accept.
pub(crate) const SECTOR: u64 = 4096;

/// The direct-read open flag comes from rustix's per-target tables:
/// the numeric `O_DIRECT` value differs between Linux archs (x86_64
/// 0o040000, aarch64 0o200000), so no literal is written here. The
/// engine's CUDA paths are Linux-only, and this alias exists only for
/// them.
#[cfg(target_os = "linux")]
use rustix::fs::OFlags;

/// The filesystem type holding `path`, by longest mount-point prefix of
/// `/proc/self/mounts` — the name the refusal carries so an O_DIRECT
/// rejection on tmpfs says tmpfs instead of a bare errno.
pub(crate) fn filesystem_of(path: &Path) -> Option<String> {
    let mounts = std::fs::read_to_string("/proc/self/mounts").ok()?;
    let absolute = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    let mut best: Option<(usize, String)> = None;
    for line in mounts.lines() {
        let mut fields = line.split_whitespace();
        let (_dev, mount_point, fs_type) = (fields.next()?, fields.next()?, fields.next()?);
        let mount_point = Path::new(mount_point);
        if absolute.starts_with(mount_point) {
            let len = mount_point.as_os_str().len();
            if best.as_ref().is_none_or(|&(prev, _)| len > prev) {
                best = Some((len, fs_type.to_string()));
            }
        }
    }
    best.map(|(_, fs_type)| fs_type)
}

/// Opens a pack file for direct reads and PROBES it with one aligned
/// sector read, so a filesystem that silently refuses O_DIRECT (tmpfs
/// answers EINVAL) is caught here, at open, with the path and the
/// filesystem named — never mid-decode. Refusing is the only failure
/// mode: the caller's plan falls to the next valid tier or stops, it
/// never routes expert bytes through the page cache by accident.
pub(crate) fn open_direct(path: &Path) -> io::Result<File> {
    #[cfg(target_os = "linux")]
    {
        let file = std::fs::OpenOptions::new()
            .read(true)
            .custom_flags((OFlags::DIRECT).bits() as i32)
            .open(path)
            .map_err(|error| {
                io::Error::new(
                    error.kind(),
                    format!(
                        "O_DIRECT refused on {path:?} (filesystem {}): {error}",
                        filesystem_of(path).as_deref().unwrap_or("unknown")
                    ),
                )
            })?;
        // The probe: a sector sub-slice of a 2*SECTOR box is always
        // sector-aligned in memory, without unsafe alignment tricks.
        let mut box_ = vec![0u8; 2 * SECTOR as usize].into_boxed_slice();
        let start = SECTOR as usize - (box_.as_ptr() as usize % SECTOR as usize);
        let probe = &mut box_[start..start + SECTOR as usize];
        file.read_at(probe, 0).map_err(|error| {
            io::Error::new(
                error.kind(),
                format!("the O_DIRECT probe read on {path:?} failed: {error}"),
            )
        })?;
        Ok(file)
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = path;
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "the O_DIRECT ring is Linux-only",
        ))
    }
}

/// Rounds `n` down to a multiple of [`SECTOR`].
pub(crate) fn align_down_sector(n: u64) -> u64 {
    n / SECTOR * SECTOR
}

/// Rounds `n` up to a multiple of [`SECTOR`].
pub(crate) fn align_up_sector(n: u64) -> u64 {
    n.div_ceil(SECTOR) * SECTOR
}

/// The aligned span a direct read of `len` bytes at `offset` must cover:
/// `(start, total)`, with the start rounded down and the end rounded up to
/// the sector. The record's exact bytes sit at `offset - start` inside the
/// span.
pub(crate) fn aligned_span(offset: u64, len: u64) -> (u64, u64) {
    let start = align_down_sector(offset);
    (start, align_up_sector(offset + len) - start)
}

/// A pinned slot handle. Freed by [`ODirectRing::release`], which only the
/// completion of the slot's recorded copy fence may call; never outlives
/// the ring because it is only handed out by the ring's free list.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Slot(pub(crate) usize);

/// The copy-completion fence recorded at `begin_copy`. The ring stays
/// CUDA-free: it only asks whether the copy has completed, so no caller
/// promise can free a slot the device is still writing — the CUDA side
/// implements this with an event query.
pub(crate) trait CopyFence: Send + Sync {
    fn is_complete(&self) -> bool;
}

/// The already-complete fence for the synchronous copy path in TESTS:
/// production records a real fence (the CUDA event query); a
/// always-complete default in shipped code would defeat the fence.
#[cfg(test)]
pub(crate) struct ImmediateFence;
#[cfg(test)]
impl CopyFence for ImmediateFence {
    fn is_complete(&self) -> bool {
        true
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SlotState {
    /// On the free list.
    Free,
    /// Acquired; spans may be staged and the record assembled.
    Reading,
    /// The record is complete and the H2D copy is in flight; the slot
    /// returns to the free list only when the copy's event completes.
    Copying,
}

/// The O_DIRECT ring: `slots` slots of `slot_span` bytes each inside
/// caller-pinned memory. A slot walks Free → Reading → Copying → Free with
/// checked transitions: acquire hands out Free slots only, `stage_span`
/// and `place` are refused outside Reading, `begin_copy` arms the H2D
/// phase and records the slot's [`CopyFence`], and `release` frees the
/// slot only when that fence reports completion — queried inside release,
/// not promised by the caller. `abandon` returns a slot whose read failed,
/// before anything was armed. Nothing here knows about layers, experts or
/// record layouts; the pack reader owns that mapping.
pub(crate) struct ODirectRing {
    /// One sector-aligned span per slot, split at construction. `None`
    /// while the span is ON LOAN to a read pool's outside-the-lock read
    /// ([`Self::loan_span`]): the Option, not the type system, enforces the
    /// one-reader-per-slot rule, so a double loan is a runtime error, never
    /// undefined behavior.
    slots_mem: Vec<Option<&'static mut [u8]>>,
    slot_span: NonZeroU64,
    slots: NonZeroUsize,
    free: Vec<usize>,
    states: Vec<Cell<SlotState>>,
    fences: Vec<std::cell::RefCell<Option<std::sync::Arc<dyn CopyFence>>>>,
}

// SAFETY-free by construction: the memory is injected by the caller.
impl std::fmt::Debug for ODirectRing {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ODirectRing")
            .field("slots", &self.slots)
            .field("slot_span", &self.slot_span)
            .field("free", &self.free.len())
            .finish()
    }
}

impl ODirectRing {
    /// Wraps `mem` — pinned by the caller — as `slots` slots of exactly
    /// `slot_span` bytes. `slot_span` must cover the worst-case record plus
    /// the worst-case staging spans the pack reader will issue (the reader
    /// sizes it); records are never required to be padded.
    pub(crate) fn new(
        slots: NonZeroUsize,
        slot_span: NonZeroU64,
        mem: &'static mut [u8],
    ) -> Result<Self, String> {
        let want = slots
            .get()
            .checked_mul(slot_span.get() as usize)
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or_else(|| "ring: slot bytes overflow".to_string())?;
        if mem.len() as u64 != want {
            return Err(format!(
                "ring: {slots} slots of {} bytes need {want} pinned bytes, the caller handed {}",
                slot_span.get(),
                mem.len()
            ));
        }
        // O_DIRECT reads land at `mem + slot*slot_span + stage_offset`, so
        // EVERY slot's buffer must be sector-aligned: the base pointer and
        // the slot stride both. Refuse here rather than fail per read.
        if !slot_span.get().is_multiple_of(SECTOR) {
            return Err(format!(
                "ring: the {slot_span}-byte slot span is not a multiple of the {SECTOR}-byte sector; O_DIRECT buffers must be sector-strided"
            ));
        }
        if !(mem.as_ptr() as usize).is_multiple_of(SECTOR as usize) {
            return Err(format!(
                "ring: the pinned memory is not {SECTOR}-byte aligned; O_DIRECT refuses it"
            ));
        }
        let free = (0..slots.get()).rev().collect();
        let states = (0..slots.get())
            .map(|_| Cell::new(SlotState::Free))
            .collect();
        let fences = (0..slots.get())
            .map(|_| std::cell::RefCell::new(None))
            .collect();
        let mut rest = mem;
        let mut slots_mem = Vec::with_capacity(slots.get());
        for _ in 0..slots.get() {
            let (span, tail) = rest.split_at_mut(slot_span.get() as usize);
            slots_mem.push(Some(span));
            rest = tail;
        }
        Ok(Self {
            slots_mem,
            slot_span,
            slots,
            free,
            states,
            fences,
        })
    }

    pub(crate) fn slots(&self) -> usize {
        self.slots.get()
    }

    /// One slot's span in bytes: the worst-case record plus staging.
    pub(crate) fn slot_span(&self) -> u64 {
        self.slot_span.get()
    }

    /// Total pinned bytes the ring covers: `slots * slot_span`, the figure
    /// the tier charges to its process budget.
    pub(crate) fn ring_bytes(&self) -> u64 {
        self.slot_span.get() * self.slots.get() as u64
    }

    fn state(&self, slot: Slot) -> Result<SlotState, String> {
        if slot.0 >= self.slots.get() {
            return Err(format!("ring: slot {} is outside the ring", slot.0));
        }
        Ok(self.states[slot.0].get())
    }

    /// Acquires the next free slot, in stream order; `None` when every
    /// slot is in flight and the caller must wait for a copy completion.
    pub(crate) fn acquire(&mut self) -> Option<Slot> {
        let slot = Slot(self.free.pop()?);
        self.states[slot.0].set(SlotState::Reading);
        Some(slot)
    }

    /// Arms the H2D phase and records the slot's copy fence: the record is
    /// assembled and the copy is about to be enqueued. Only the recorded
    /// fence's completion may `release`.
    pub(crate) fn begin_copy(
        &self,
        slot: Slot,
        fence: std::sync::Arc<dyn CopyFence>,
    ) -> Result<(), String> {
        match self.state(slot)? {
            SlotState::Reading => {
                self.states[slot.0].set(SlotState::Copying);
                *self.fences[slot.0].borrow_mut() = Some(fence);
                Ok(())
            }
            other => Err(format!(
                "ring: slot {} cannot begin its copy from {other:?}",
                slot.0
            )),
        }
    }

    /// Returns a slot to the free list — only the recorded fence's
    /// completion frees it: `release` queries the fence itself, so a
    /// caller's bare promise that the copy finished is not enough. An
    /// incomplete fence leaves the slot in `Copying` for a later release.
    pub(crate) fn release(&mut self, slot: Slot) -> Result<(), String> {
        match self.state(slot) {
            Ok(SlotState::Copying) => {
                let complete = self.fences[slot.0]
                    .borrow()
                    .as_ref()
                    .map(|fence| fence.is_complete())
                    .unwrap_or(false);
                if !complete {
                    return Err(format!(
                        "ring: slot {}'s copy fence is not complete; the device may still be writing it",
                        slot.0
                    ));
                }
                *self.fences[slot.0].borrow_mut() = None;
                self.states[slot.0].set(SlotState::Free);
                self.free.push(slot.0);
                Ok(())
            }
            Ok(other) => Err(format!(
                "ring: slot {} cannot be released from {other:?}; release follows the copy fence",
                slot.0
            )),
            Err(err) => Err(err),
        }
    }

    /// Returns a slot whose read FAILED, before any copy was armed: the
    /// slot goes straight back to the free list. A slot whose copy is in
    /// flight cannot be abandoned — its fence must complete first.
    pub(crate) fn abandon(&mut self, slot: Slot) -> Result<(), String> {
        match self.state(slot) {
            Ok(SlotState::Reading) => {
                self.states[slot.0].set(SlotState::Free);
                self.free.push(slot.0);
                Ok(())
            }
            Ok(other) => Err(format!(
                "ring: slot {} cannot be abandoned from {other:?}; only a failed read abandons",
                slot.0
            )),
            Err(err) => Err(err),
        }
    }

    fn checked_range(
        &self,
        slot: Slot,
        offset: u64,
        len: u64,
    ) -> Result<std::ops::Range<usize>, String> {
        self.state(slot)?;
        let end = offset
            .checked_add(len)
            .ok_or_else(|| "ring: the range overflows".to_string())?;
        if end > self.slot_span.get() {
            return Err(format!(
                "ring: {} bytes at +{offset} escape slot {}'s {}-byte span",
                len,
                slot.0,
                self.slot_span.get()
            ));
        }
        Ok(offset as usize..end as usize)
    }

    /// Stages one aligned span of the file into the slot at `stage_offset`,
    /// which must itself be sector-aligned so the buffer address matches
    /// the read's alignment. `file_offset` and `len` are the EXACT wanted
    /// bytes; the read covers their aligned span. Returns the offset of
    /// the wanted bytes inside the staged span (`file_offset - span start`)
    /// so the caller can `place` them without recomputing.
    pub(crate) fn stage_span(
        &mut self,
        file: &File,
        file_offset: u64,
        len: u64,
        slot: Slot,
        stage_offset: u64,
    ) -> io::Result<u64> {
        if self.state(slot).map_err(io::Error::other)? != SlotState::Reading {
            return Err(io::Error::other(format!(
                "ring: slot {} is not reading; spans stage before the copy",
                slot.0
            )));
        }
        if !stage_offset.is_multiple_of(SECTOR) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("ring: the stage offset {stage_offset} is not sector-aligned"),
            ));
        }
        let (start, total) = aligned_span(file_offset, len);
        let range = self
            .checked_range(slot, stage_offset, total)
            .map_err(io::Error::other)?;
        // A span's padding may run past EOF when the shard ends unaligned:
        // read what exists, then require the WANTED bytes to be covered.
        // The padding tail beyond the record is never read by anyone.
        let got = pread_at_most(
            file,
            &mut self.span_slice(slot).map_err(io::Error::other)?[range],
            start,
        )?;
        let want_start = (file_offset - start) as usize;
        if got < want_start + len as usize {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "the pack ended inside the wanted bytes",
            ));
        }
        Ok(file_offset - start)
    }

    /// Copies `len` exact bytes from the staged span at `from` to the
    /// record position `to`, inside one slot. Pure host memory movement —
    /// cheap against the disk read that staged the span.
    pub(crate) fn place(&mut self, slot: Slot, from: u64, to: u64, len: u64) -> Result<(), String> {
        if self.state(slot)? != SlotState::Reading {
            return Err(format!(
                "ring: slot {} is not reading; records assemble before the copy",
                slot.0
            ));
        }
        let len_usize = len as usize;
        let src = self.checked_range(slot, from, len)?.start;
        let dst = self.checked_range(slot, to, len)?.start;
        self.span_slice(slot)?
            .copy_within(src..src + len_usize, dst);
        Ok(())
    }

    /// The slot's span, exclusively, for a read issued OUTSIDE the ring
    /// lock by a read pool. The slice is `'static` (the pinned memory is
    /// process-static) and disjoint from every other slot by construction;
    /// the loan makes the slot's span invisible to the ring's own methods
    /// until [`Self::return_span`] puts it back, so a loaned span cannot
    /// race the in-ring paths. Sound only while the caller owns the slot
    /// (acquired, not yet released or abandoned).
    pub(crate) fn loan_span(&mut self, slot: Slot) -> Result<&'static mut [u8], String> {
        if self.state(slot)? != SlotState::Reading {
            return Err(format!(
                "ring: slot {} is not reading; an outside read needs the acquired slot",
                slot.0
            ));
        }
        self.slots_mem
            .get_mut(slot.0)
            .and_then(|span| span.take())
            .ok_or_else(|| format!("ring: slot {}'s span is already on loan", slot.0))
    }

    /// Puts a loaned span back. The slot's in-ring paths (place, copy,
    /// window) refuse while the span is out, so the loan discipline is: loan,
    /// read outside the lock, return, then assemble.
    pub(crate) fn return_span(
        &mut self,
        slot: Slot,
        span: &'static mut [u8],
    ) -> Result<(), String> {
        match self.slots_mem.get_mut(slot.0) {
            Some(hole @ None) => {
                *hole = Some(span);
                Ok(())
            }
            Some(Some(_)) => Err(format!("ring: slot {}'s span was not on loan", slot.0)),
            None => Err(format!("ring: slot {} is outside the ring", slot.0)),
        }
    }

    /// The slot's span for an in-ring read or copy. Refuses while the span
    /// is on loan to an outside read.
    fn span_slice(&mut self, slot: Slot) -> Result<&mut [u8], String> {
        self.slots_mem
            .get_mut(slot.0)
            .and_then(|span| span.as_deref_mut())
            .ok_or_else(|| format!("ring: slot {}'s span is on loan to an outside read", slot.0))
    }

    /// The exact record bytes at `offset..offset+len` of the slot, for the
    /// H2D staging copy. Valid in Reading (assembled) and Copying.
    /// The slot's assembled record, mutable — while READING only. Once the
    /// copy is armed the buffer feeds DMA, and a writer through `&mut`
    /// could race it, so the Copying phase gets the read-only view instead
    /// ([`Self::slot_window`]).
    pub(crate) fn slot_range(
        &mut self,
        slot: Slot,
        offset: u64,
        len: u64,
    ) -> Result<&mut [u8], String> {
        match self.state(slot)? {
            SlotState::Reading => {
                let range = self.checked_range(slot, offset, len)?;
                Ok(&mut self.span_slice(slot)?[range])
            }
            SlotState::Copying => Err(format!(
                "ring: slot {} is Copying; a mutable range would race the DMA read - use slot_window",
                slot.0
            )),
            SlotState::Free => Err(format!(
                "ring: slot {} is free; nothing to hand out",
                slot.0
            )),
        }
    }

    /// The slot's record as a read-only view, for the Copying phase: the
    /// H2D enqueue reads the buffer; nothing may write it again until the
    /// fence completes and the slot returns to the free list.
    pub(crate) fn slot_window(&self, slot: Slot, offset: u64, len: u64) -> Result<&[u8], String> {
        match self.state(slot)? {
            SlotState::Copying => {
                let range = self.checked_range(slot, offset, len)?;
                self.slots_mem
                    .get(slot.0)
                    .and_then(|span| span.as_deref())
                    .ok_or_else(|| {
                        format!("ring: slot {}'s span is on loan to an outside read", slot.0)
                    })
                    .map(|span| &span[range])
            }
            other => Err(format!(
                "ring: slot {} is {other:?}; the read-only window exists only while Copying",
                slot.0
            )),
        }
    }
}

/// Positional read into `buf` (pread semantics: the shared file offset is
/// never touched), short at EOF: returns how many bytes were read before the
/// file ended. The wanted-exactness check is the caller's, because only it
/// knows which part of the aligned span is record and which is padding.
/// The retry loop itself lives in `weights::expert_store` (shared with the
/// expert record stores); this is a thin adapter over the shared handle.
fn pread_at_most(file: &File, buf: &mut [u8], offset: u64) -> io::Result<usize> {
    crate::weights::expert_store::pread_at_most(file, buf, offset)
}

/// The host tier's LRU over the resident spare places: per layer, which
/// expert each host place holds. Stacked ABOVE the bank's
/// [`ReplaceOnMissLru`](policy::ReplaceOnMissLru) — the same eviction
/// discipline at the second level; the bank's slot-level LRU stays the
/// inner one and is untouched.
///
/// Per layer, `places` experts are resident (zero places is a valid tier:
/// every miss streams from disk and nothing is resident). `admit` returns
/// a typed outcome carrying the freed PLACE; `note` refreshes.
pub(crate) struct HostTierLru {
    per_layer: Vec<LayerTier>,
}

/// What an admission did: the expert was already resident, or it took a
/// place — with that place's index and the expert it evicted, if any.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum AdmitOutcome {
    AlreadyResident,
    Admitted {
        place: usize,
        evicted: Option<u32>,
    },
    /// The tier holds zero places — a valid ring-only tier where every
    /// miss streams from disk; there is no place to take or evict.
    NoPlaces,
}

struct LayerTier {
    resident: Vec<Option<u32>>,
    free_places: Vec<usize>,
    recency: std::collections::VecDeque<u32>,
}

impl HostTierLru {
    /// Fresh from a fill: `resident[layer]` lists the experts of that
    /// layer's places, oldest first (the fill's popularity order, so the
    /// head of `recency` is the coldest expert of the layer).
    pub(crate) fn from_fill(
        layers: usize,
        places_per_layer: usize,
        resident: &[Vec<u32>],
    ) -> Result<Self, String> {
        if resident.len() != layers {
            return Err("host-tier lru: one resident list per layer".to_string());
        }
        let mut per_layer = Vec::with_capacity(layers);
        for row in resident.iter() {
            if row.len() > places_per_layer {
                return Err("host-tier lru: more residents than places".to_string());
            }
            if row.len() != places_per_layer {
                return Err(
                    "host-tier lru: every place starts filled (mirror of the bank)".to_string(),
                );
            }
            let mut slots = vec![None; places_per_layer];
            let mut recency = std::collections::VecDeque::with_capacity(places_per_layer);
            for (place, &expert) in row.iter().enumerate() {
                if slots.contains(&Some(expert)) {
                    return Err("host-tier lru: an expert is resident twice".to_string());
                }
                slots[place] = Some(expert);
                recency.push_back(expert);
            }
            per_layer.push(LayerTier {
                resident: slots,
                free_places: Vec::new(),
                recency,
            });
        }
        Ok(Self { per_layer })
    }

    /// Refreshes an expert's recency; returns false when the expert is not
    /// resident (a miss the host tier must stream through the ring).
    pub(crate) fn note(&mut self, layer: usize, expert: u32) -> bool {
        let Some(tier) = self.per_layer.get_mut(layer) else {
            return false;
        };
        if !tier.resident.contains(&Some(expert)) {
            return false;
        }
        tier.recency.retain(|&e| e != expert);
        tier.recency.push_back(expert);
        true
    }

    /// Admits an expert to the layer. The outcome names what happened: an
    /// already-resident expert is a no-op, and an admission carries the
    /// freed place's index and the evicted expert, so the caller retires
    /// the victim's bytes from the pinned store before the copy lands.
    pub(crate) fn admit(&mut self, layer: usize, expert: u32) -> Result<AdmitOutcome, String> {
        let tier = self
            .per_layer
            .get_mut(layer)
            .ok_or_else(|| format!("host-tier lru: layer {layer} is outside the tier"))?;
        if tier.resident.contains(&Some(expert)) {
            return Ok(AdmitOutcome::AlreadyResident);
        }
        if let Some(place) = tier.free_places.pop() {
            tier.resident[place] = Some(expert);
            tier.recency.push_back(expert);
            return Ok(AdmitOutcome::Admitted {
                place,
                evicted: None,
            });
        }
        // Zero places is a valid tier (the ring-only plan): every miss
        // streams, so admission is a typed no-op, never an error.
        if tier.resident.is_empty() {
            return Ok(AdmitOutcome::NoPlaces);
        }
        let victim = tier
            .recency
            .pop_front()
            .ok_or("host-tier lru: a full tier with no recency is a broken invariant")?;
        let place = tier
            .resident
            .iter()
            .position(|&held| held == Some(victim))
            .ok_or("host-tier lru: the victim holds no place; a broken invariant")?;
        tier.resident[place] = Some(expert);
        tier.recency.push_back(expert);
        Ok(AdmitOutcome::Admitted {
            place,
            evicted: Some(victim),
        })
    }
}

#[cfg(test)]
pub(crate) mod tests {
    //! Exact-byte and exact-order checks, host only. The direct-read test
    //! needs a filesystem that answers O_DIRECT: it prefers
    //! `FIFI_TEST_DIRECT_DIR`, then `CARGO_TARGET_TMPDIR`, then the temp
    //! dir, prints which directory and filesystem answered (tmpfs accepts
    //! O_DIRECT from kernel 6.6 on, so only the printed fs name says what
    //! was proven), and skips, with a printed note, where the directory is
    //! unwritable or O_DIRECT is refused there.

    /// The ring opens with rustix's `O_DIRECT`; this pins the per-target
    /// value against the arch's known number, so a rustix table change or
    /// a cross-arch mixup cannot pass silently. The values themselves are
    /// why the production code takes the flag from rustix: x86_64 and
    /// aarch64 Linux disagree on every one of them.
    #[test]
    #[cfg(target_os = "linux")]
    fn the_direct_flag_matches_the_platform_o_direct() {
        #[cfg(target_arch = "x86_64")]
        const EXPECTED: u32 = 0o040000;
        #[cfg(target_arch = "aarch64")]
        const EXPECTED: u32 = 0o200000;
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        const EXPECTED: u32 = 0;
        assert_eq!(rustix::fs::OFlags::DIRECT.bits(), EXPECTED);
    }

    use super::{
        AdmitOutcome, CopyFence, HostTierLru, ImmediateFence, ODirectRing, SECTOR,
        align_down_sector, align_up_sector, aligned_span, open_direct,
    };
    use std::num::{NonZeroU64, NonZeroUsize};

    /// The filesystem type the direct-read test runs on, from mountinfo
    /// (longest mount-point prefix wins), so the log says what answered
    /// O_DIRECT — tmpfs accepts it from 6.6 on, which proves no bypass.
    fn filesystem_of(path: &std::path::Path) -> String {
        let mut best = (0usize, "unknown".to_string());
        for line in std::fs::read_to_string("/proc/self/mountinfo")
            .unwrap_or_default()
            .lines()
        {
            let Some((left, right)) = line.split_once(" - ") else {
                continue;
            };
            let mut fields = left.split(' ');
            let _ = (fields.next(), fields.next(), fields.next(), fields.next());
            let (Some(mount_point), Some(fstype)) = (fields.next(), right.split(' ').next()) else {
                continue;
            };
            if path.starts_with(mount_point) && mount_point.len() >= best.0 {
                best = (mount_point.len(), fstype.to_string());
            }
        }
        best.1
    }

    fn pinned(len: usize) -> &'static mut [u8] {
        // A Vec's pointer is only 16-byte aligned; production ring memory
        // (cudaHostAlloc/mmap) is page-aligned. Carve a sector-aligned
        // sub-slice out of an oversized leaked buffer instead — no unsafe
        // in tests.
        let slop = SECTOR as usize - 1;
        let box_ = Box::leak(vec![0u8; len + slop].into_boxed_slice());
        let base = box_.as_ptr() as usize;
        let off = (SECTOR as usize - (base % SECTOR as usize)) % SECTOR as usize;
        &mut box_[off..off + len]
    }

    #[test]
    fn span_math_is_exact() {
        assert_eq!(align_down_sector(SECTOR), SECTOR);
        assert_eq!(align_down_sector(SECTOR + 1), SECTOR);
        assert_eq!(align_up_sector(1), SECTOR);
        assert_eq!(align_up_sector(SECTOR), SECTOR);
        // A 1,881,600-byte record at an unaligned file offset: the span
        // rounds down and up around it.
        let (start, total) = aligned_span(1_881_601, 1_881_600);
        assert_eq!(start, align_down_sector(1_881_601));
        assert_eq!(total, align_up_sector(1_881_601 + 1_881_600) - start);
        assert!(total % SECTOR == 0);
    }

    #[test]
    fn a_wrong_memory_size_is_refused() {
        let err = ODirectRing::new(
            NonZeroUsize::new(2).unwrap(),
            NonZeroU64::new(SECTOR).unwrap(),
            pinned(SECTOR as usize),
        )
        .unwrap_err();
        assert!(err.contains("need"), "{err}");
    }

    #[test]
    fn an_unaligned_span_or_base_is_refused() {
        // The stride must be a sector multiple: slot buffers sit at
        // base + slot*span, so an unaligned stride misaligns every slot
        // past the first.
        let err = ODirectRing::new(
            NonZeroUsize::new(2).unwrap(),
            NonZeroU64::new(SECTOR + 8).unwrap(),
            pinned(2 * (SECTOR + 8) as usize),
        )
        .unwrap_err();
        assert!(err.contains("sector"), "{err}");
        // The pinned base itself must be aligned; a Vec box is, so an
        // unaligned base is simulated by leaking a shifted sub-slice of
        // exactly the byte count the ring demands.
        let box_ = Box::leak(vec![0u8; SECTOR as usize + 1].into_boxed_slice());
        let mem = &mut box_[1..];
        let err = ODirectRing::new(
            NonZeroUsize::new(1).unwrap(),
            NonZeroU64::new(SECTOR).unwrap(),
            mem,
        )
        .unwrap_err();
        assert!(err.contains("aligned"), "{err}");
    }

    #[test]
    fn slots_walk_the_checked_state_machine() {
        let span = 2 * SECTOR;
        let mut ring = ODirectRing::new(
            NonZeroUsize::new(2).unwrap(),
            NonZeroU64::new(span).unwrap(),
            pinned(2 * span as usize),
        )
        .unwrap();
        assert_eq!(ring.ring_bytes(), 2 * span);
        let a = ring.acquire().unwrap();
        let b = ring.acquire().unwrap();
        assert!(ring.acquire().is_none(), "both slots are in flight");
        // Release before any copy is a state error, not a silent free.
        assert!(ring.release(a).is_err());
        // A failed read abandons: the slot returns straight to the free list.
        assert!(ring.abandon(a).is_ok());
        assert_eq!(ring.acquire(), Some(a), "the abandoned slot recycles first");
        ring.begin_copy(a, std::sync::Arc::new(ImmediateFence))
            .unwrap();
        // Staging after the copy is armed is refused.
        let err = ring.place(a, 0, 0, 8).unwrap_err();
        assert!(err.contains("copy"), "{err}");
        // While Copying, the record is read-only: a mutable range would
        // race the DMA read, so the enqueue takes the window instead.
        let err = ring.slot_range(a, 0, 8).unwrap_err();
        assert!(err.contains("slot_window"), "{err}");
        assert_eq!(ring.slot_window(a, 0, 8).unwrap().len(), 8);
        ring.release(a).unwrap();
        // begin_copy from Free (just released, not yet acquired) is refused.
        assert!(
            ring.begin_copy(a, std::sync::Arc::new(ImmediateFence))
                .is_err()
        );
        assert_eq!(ring.acquire(), Some(a), "the released slot recycles first");
        ring.begin_copy(a, std::sync::Arc::new(ImmediateFence))
            .unwrap();
        assert!(
            ring.stage_span(&std::fs::File::open("/dev/null").unwrap(), 0, 1, a, 0)
                .is_err()
        );
        // A copy in flight cannot be abandoned: its fence must complete.
        assert!(ring.abandon(a).is_err());
        ring.release(a).unwrap();
        // b is still Reading; begin_copy from Reading is legal, and release
        // then requires the copy phase.
        ring.begin_copy(b, std::sync::Arc::new(ImmediateFence))
            .unwrap();
        ring.release(b).unwrap();
    }

    /// A fence the test flips: incomplete at first, complete on demand.
    struct FlagFence(std::sync::atomic::AtomicBool);
    impl CopyFence for FlagFence {
        fn is_complete(&self) -> bool {
            self.0.load(std::sync::atomic::Ordering::Relaxed)
        }
    }

    #[test]
    fn a_pending_fence_keeps_the_slot_until_it_completes() {
        let span = 2 * SECTOR;
        let mut ring = ODirectRing::new(
            NonZeroUsize::new(1).unwrap(),
            NonZeroU64::new(span).unwrap(),
            pinned(span as usize),
        )
        .unwrap();
        let slot = ring.acquire().unwrap();
        let fence = std::sync::Arc::new(FlagFence(std::sync::atomic::AtomicBool::new(false)));
        ring.begin_copy(slot, std::sync::Arc::clone(&fence) as _)
            .unwrap();
        // The device is still writing: release refuses, the slot stays
        // Copying, and no caller promise can change that — the ring asks
        // the fence itself.
        assert!(ring.release(slot).is_err());
        assert_eq!(ring.acquire(), None, "the incomplete slot never recycled");
        fence.0.store(true, std::sync::atomic::Ordering::Relaxed);
        ring.release(slot).unwrap();
        assert_eq!(ring.acquire(), Some(slot), "the completed slot recycles");
    }

    #[test]
    fn a_zero_place_tier_admits_nothing_and_streams() {
        let mut lru = HostTierLru::from_fill(2, 0, &[vec![], vec![]]).unwrap();
        assert!(!lru.note(0, 7), "nothing is resident in a zero-place tier");
        assert_eq!(lru.admit(0, 7).unwrap(), AdmitOutcome::NoPlaces);
        assert!(!lru.note(0, 7), "the NoPlaces admit left nothing resident");
    }

    #[test]
    fn an_out_of_span_range_is_refused() {
        let span = 2 * SECTOR;
        let mut ring = ODirectRing::new(
            NonZeroUsize::new(1).unwrap(),
            NonZeroU64::new(span).unwrap(),
            pinned(span as usize),
        )
        .unwrap();
        let slot = ring.acquire().unwrap();
        assert!(
            ring.place(slot, 0, span - 4, 8).is_err(),
            "escapes the span"
        );
    }

    /// The directory O_DIRECT tests write their scratch shard into, and
    /// its filesystem type for the log: an env override first, then the
    /// cargo tmp, then the plain temp dir. One choice shared by every
    /// direct-reading test so the override reaches them all.
    pub(crate) fn direct_test_dir() -> (std::path::PathBuf, String) {
        let dir = std::env::var("FIFI_TEST_DIRECT_DIR")
            .map(std::path::PathBuf::from)
            .or_else(|_| std::env::var("CARGO_TARGET_TMPDIR").map(std::path::PathBuf::from))
            .unwrap_or_else(|_| std::env::temp_dir());
        let filesystem = filesystem_of(&dir);
        (dir, filesystem)
    }

    #[test]
    fn a_direct_span_read_stages_and_places_exact_bytes() {
        let (dir, filesystem) = direct_test_dir();
        eprintln!("direct-read test: {dir:?} on {filesystem}");
        let path = dir.join(format!("odirect-ring-test-{}.bin", std::process::id()));
        if let Err(e) = write_pattern(&path) {
            eprintln!(
                "direct-read test: cannot write {path:?} ({e}); alignment, order and \
                 state checks stand, the byte-exact read skips"
            );
            return;
        }
        let file = match open_direct(&path) {
            Ok(file) => file,
            Err(e) => {
                eprintln!("direct-read test: {e}; the byte-exact read skips");
                let _ = std::fs::remove_file(&path);
                return;
            }
        };
        // Slot: one record area of a sector and one staging sector.
        let span = 2 * SECTOR;
        let mut ring = ODirectRing::new(
            NonZeroUsize::new(1).unwrap(),
            NonZeroU64::new(span).unwrap(),
            pinned(span as usize),
        )
        .unwrap();
        let slot = ring.acquire().unwrap();
        // The wanted bytes: the second half of sector 1 (unaligned extent,
        // pattern 0xEE) placed at the record's head.
        let wanted = SECTOR + SECTOR / 2;
        let in_span = ring
            .stage_span(&file, wanted, SECTOR / 2, slot, SECTOR)
            .unwrap();
        assert_eq!(in_span, SECTOR / 2);
        ring.place(slot, SECTOR + in_span, 0, SECTOR / 2).unwrap();
        let record = ring.slot_range(slot, 0, SECTOR / 2).unwrap();
        assert!(
            record.iter().all(|&b| b == 0xEE),
            "the exact sub-range, no more"
        );
        let _ = std::fs::remove_file(&path);
    }

    fn write_pattern(path: &std::path::Path) -> std::io::Result<()> {
        use std::io::Write;
        let mut f = std::fs::File::create(path)?;
        f.write_all(&vec![0x11u8; SECTOR as usize])?;
        f.write_all(&vec![0xEEu8; SECTOR as usize])?;
        f.write_all(&vec![0x22u8; SECTOR as usize])?;
        f.flush()
    }

    #[test]
    fn the_host_tier_lru_evicts_least_recent_first_and_note_refreshes() {
        let resident = vec![vec![10, 11, 12]];
        let mut lru = HostTierLru::from_fill(1, 3, &resident).unwrap();
        // 10 is coldest; touching 10 makes 11 the victim of place 1.
        assert!(lru.note(0, 10));
        assert_eq!(
            lru.admit(0, 13),
            Ok(AdmitOutcome::Admitted {
                place: 1,
                evicted: Some(11)
            })
        );
        assert!(!lru.note(0, 11), "the victim is gone");
        assert!(lru.note(0, 13));
        assert_eq!(
            lru.admit(0, 14),
            Ok(AdmitOutcome::Admitted {
                place: 2,
                evicted: Some(12)
            })
        );
        // Re-admitting a resident expert evicts nothing.
        assert_eq!(lru.admit(0, 10), Ok(AdmitOutcome::AlreadyResident));
        // An unknown layer is a typed error, not a None.
        assert!(lru.admit(9, 1).is_err());
    }

    #[test]
    fn a_wrong_fill_is_refused() {
        let short = vec![vec![10, 11]];
        assert!(HostTierLru::from_fill(1, 3, &short).is_err());
        let twice = vec![vec![10, 10, 11]];
        assert!(HostTierLru::from_fill(1, 3, &twice).is_err());
    }
}
