//! The expert records of the pack, read through the O_DIRECT ring: the
//! gated [`RecordSource`] of the expert bank.
//!
//! WHAT: [`RingRecordSource`] answers the same `record(layer, expert, out)`
//! contract as [`PackRecordSource`](super::bank_source::PackRecordSource),
//! but the component bytes travel the landed pack reader — one
//! [`MatrixSpan`] per present component extent, staged into a pinned ring
//! slot by O_DIRECT reads — instead of the store's buffered reads. The
//! parse and the packing are shared: the same `Matrix::load`, the same
//! bias refusal, the same `SlotBankLayout`. Both sources must agree
//! byte-for-byte; the corpus test pins that.
//!
//! The direct-upload lane: `upload` enqueues the record's H2D itself, one
//! `write_at_enqueue` per span STRAIGHT FROM THE PINNED SLOT WINDOW — the
//! trellis spans (the bulk) DMA their bytes verbatim into the row region
//! the layout gives them; the F16 scale spans are the one named host
//! exception, widened to F32 on the host (kilobytes against the trellis's
//! ~1.8 MiB) and enqueued inside the same fence window. After the last
//! enqueue an event is recorded on the stream, and the slot releases only
//! on that event — a release from anywhere but the fill thread's own
//! query cannot free it (the fence below pins why).
//!
//! WHY: the buffered path holds a whole record's components in flight per
//! read with no placement control; the ring pins the staging memory once
//! through the context's pinned allocator (cudaHostAlloc, so the async H2D
//! from it is truly async — the host_tier contract), checks every slot
//! state transition, and hands the record to its consumer read-only while
//! the copy fence is outstanding. This module is the reader's first
//! production caller (`FIFI_FLASH_NEXT_PACK_RING=1`, default off).
//!
//! The ring memory is handed in by the caller: the serve path passes the
//! context's cudaHostAlloc memory, leaked for the process lifetime like
//! the rest of the serve-start state; the host tests pass their own
//! sector-carved buffer. The constructor opens the first shard with
//! O_DIRECT eagerly, so a filesystem that will not answer it refuses the
//! start. The planner's tier line decided that the ring variant runs at
//! all.

use super::bank::RecordSource;
use super::bank_source::{PackRecordSource, refuse_bias};
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
use crate::family::expert_bank::host_tier::Slot;
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
use crate::kernels::cuda::{CudaContext, DeviceBuffer, DeviceStream};
use crate::{
    EngError,
    family::{
        exl3_pack::{ExpertPart, Root, expert_name_in},
        expert_bank::host_tier::{CopyFence, ODirectRing, open_direct},
        flash_next::{
            cuda::pack_reader::{MatrixSpan, PackReader as RingReader, slot_span_bytes},
            moe::slot_bank::SlotBankLayout,
        },
    },
    tensor::{Dtype, Tensor},
    weights::{
        Tier, WeightSource,
        exl3::{COMPONENT_SUFFIXES, Checkpoint, Matrix},
    },
};
use std::{
    collections::HashMap,
    num::{NonZeroU64, NonZeroUsize},
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
};

fn ring_error(detail: String) -> EngError {
    EngError::Checkpoint(format!("Flash-Next ring records: {detail}"))
}

/// The fence of the synchronous hand-off: the consumer copied the record
/// out of the slot before `release` runs, so the copy is complete when the
/// fence is asked. Only the `record` path uses it.
pub(in crate::family::flash_next) struct HostCopyFence;

impl CopyFence for HostCopyFence {
    fn is_complete(&self) -> bool {
        true
    }
}

/// The fence of a direct upload. The completion truth is the CUDA event
/// recorded after the last enqueue, but a [`DeviceEvent`] is
/// thread-confined, so the fence holds a cached answer instead: it starts
/// incomplete, and only [`Self::mark_complete`] — called by the fill
/// thread after its own event query answered `true` — ever completes it.
/// A CUDA event only moves from incomplete to complete, so a cached
/// `true` stays true; a release from any thread that did not query leaves
/// the fence incomplete and the slot pinned, which is the safe side. The
/// host test pins exactly that refusal.
pub(in crate::family::flash_next) struct UploadFence {
    complete: AtomicBool,
}

impl UploadFence {
    pub(in crate::family::flash_next) fn new() -> Self {
        Self {
            complete: AtomicBool::new(false),
        }
    }

    /// The only setter, and it takes no argument: the caller asserts a
    /// device query answered `true` for THIS fence's event.
    pub(in crate::family::flash_next) fn mark_complete(&self) {
        self.complete.store(true, Ordering::Release);
    }
}

impl CopyFence for UploadFence {
    fn is_complete(&self) -> bool {
        self.complete.load(Ordering::Acquire)
    }
}

/// One parked upload: the slot stays `Copying` (its window read-only for
/// the DMA) until the event's completion is observed and the fence is
/// marked, at which point the ring takes the slot back.
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
struct ParkedUpload {
    slot: Slot,
    event: crate::kernels::cuda::DeviceEvent,
    fence: Arc<UploadFence>,
}

/// The ring-backed records of an opened pack. `checkpoint` is the same
/// borrowed open the buffered source reads; `dir` names the pack directory
/// the shard extents resolve against.
pub(in crate::family::flash_next) struct RingRecordSource<'a> {
    checkpoint: &'a Checkpoint,
    dir: PathBuf,
    layout: SlotBankLayout,
    layers: usize,
    experts: usize,
    /// The packed record the layout defines (the raw span bytes may be
    /// smaller: the F16 scales widen to F32).
    record_bytes: usize,
    ring: Mutex<(RingReader, ODirectRing)>,
    /// The uploads whose fence is still outstanding.
    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    parked: Mutex<Vec<ParkedUpload>>,
}

impl<'a> RingRecordSource<'a> {
    /// Derives the record layout exactly as the buffered source does, then
    /// sizes the ring from the same numbers: every expert shares the
    /// layout (the buffered source's own checks refuse a pack where one
    /// does not), so expert 0 of layer 0 bounds every slot. `backing` is
    /// the pinned ring memory — the context's cudaHostAlloc on the serve
    /// path, the test's carved buffer in tests — sized by
    /// [`Self::backing_bytes`]. The first shard is opened O_DIRECT
    /// eagerly: a filesystem that will not answer it refuses the start.
    pub(in crate::family::flash_next) fn new(
        checkpoint: &'a Checkpoint,
        dir: &Path,
        slots: NonZeroUsize,
        backing: &'static mut [u8],
    ) -> Result<Self, EngError> {
        let buffered = PackRecordSource::new(checkpoint)?;
        let (layout, layers, experts, record_bytes) = (
            buffered.layout().clone(),
            buffered.layers(),
            buffered.experts(),
            buffered.layout().record_bytes(),
        );
        let (spans, raw_bytes) = expert_spans(checkpoint, dir, 0, 0)?;
        if spans.is_empty() {
            return Err(ring_error("the pack exposes no expert components".into()));
        }
        open_direct(&spans[0].matrix.shard)
            .map_err(|error| ring_error(format!("O_DIRECT refused: {error}")))?;
        let span = slot_span_bytes(raw_bytes, &span_matrix_spans(&spans))?;
        let slot_span = NonZeroU64::new(span)
            .ok_or_else(|| ring_error("the record layout sizes to a zero span".into()))?;
        let bytes = slots
            .get()
            .checked_mul(slot_span.get() as usize)
            .ok_or_else(|| ring_error("the ring sizing overflows".into()))?;
        let ring =
            ODirectRing::new(slots, slot_span, carve(backing, bytes)?).map_err(ring_error)?;
        Ok(Self {
            checkpoint,
            dir: dir.to_path_buf(),
            layout,
            layers,
            experts,
            record_bytes,
            ring: Mutex::new((RingReader::new(), ring)),
            #[cfg(all(feature = "native-cuda", target_os = "linux"))]
            parked: Mutex::new(Vec::new()),
        })
    }

    /// The byte size `backing` must carry for `new`: the ring's exact
    /// bytes plus one sector of carve slack, because a plain allocation is
    /// only 16-byte aligned.
    pub(in crate::family::flash_next) fn backing_bytes(
        checkpoint: &Checkpoint,
        dir: &Path,
        slots: NonZeroUsize,
    ) -> Result<usize, EngError> {
        let (spans, raw_bytes) = expert_spans(checkpoint, dir, 0, 0)?;
        if spans.is_empty() {
            return Err(ring_error("the pack exposes no expert components".into()));
        }
        let span = slot_span_bytes(raw_bytes, &span_matrix_spans(&spans))?;
        let total = slots
            .get()
            .checked_mul(span as usize)
            .ok_or_else(|| ring_error("the ring sizing overflows".into()))?;
        Ok(total + crate::family::expert_bank::host_tier::SECTOR as usize)
    }

    /// The layer and expert counts, for callers that size from the same
    /// walk.
    pub(in crate::family::flash_next) fn layout_ref(&self) -> &SlotBankLayout {
        &self.layout
    }

    pub(in crate::family::flash_next) fn counts(&self) -> (usize, usize) {
        (self.layers, self.experts)
    }
}

/// Carves the sector-aligned exact sub-slice out of `backing`: a plain
/// allocation's own pointer is only 16-byte aligned, so the head up to the
/// first sector boundary is skipped. cudaHostAlloc memory is page-aligned,
/// making this a no-op there.
fn carve(backing: &'static mut [u8], bytes: usize) -> Result<&'static mut [u8], EngError> {
    use crate::family::expert_bank::host_tier::SECTOR;
    let start = backing.as_ptr() as usize;
    let skipped = start.div_ceil(SECTOR as usize) * SECTOR as usize - start;
    if backing.len() < skipped || backing.len() - skipped < bytes {
        return Err(ring_error(format!(
            "the ring backing carries {} bytes, the ring needs {bytes} at a sector boundary",
            backing.len()
        )));
    }
    let (_, aligned) = backing.split_at_mut(skipped);
    let (exact, _) = aligned.split_at_mut(bytes);
    Ok(exact)
}

impl RecordSource for RingRecordSource<'_> {
    fn record(&self, layer: usize, expert: u16, out: &mut [u8]) -> Result<(), EngError> {
        if layer >= self.layers || usize::from(expert) >= self.experts {
            return Err(ring_error(format!(
                "layer {layer} expert {expert} is outside {} layers of {} experts",
                self.layers, self.experts
            )));
        }
        let (spans, raw_bytes) =
            expert_spans(self.checkpoint, &self.dir, layer, usize::from(expert))?;
        let mut guard = lock_or(&self.ring, "the ring source's ring")?;
        let mut slot = guard.1.acquire();
        if slot.is_none() {
            // A parked upload holds every slot: wait it out exactly as a
            // slot-starved upload does — layer 1's first record must
            // succeed after layer 0's last upload. Host builds never
            // park (only uploads park, and they are native-only).
            drop(guard);
            #[cfg(all(feature = "native-cuda", target_os = "linux"))]
            self.wait_out_parked()?;
            guard = lock_or(&self.ring, "the ring source's ring")?;
            slot = guard.1.acquire();
        }
        let Some(slot) = slot else {
            return Err(ring_error("every ring slot is outstanding".into()));
        };
        // The read fills the slot in Reading; the copy arm transitions to
        // Copying (the read-only window exists only there); the parse
        // reads the window; the host consumed everything, so the slot
        // returns at once. Exactly one begin_copy on this path.
        {
            let (reader, ring) = &mut *guard;
            reader.read_expert(ring, &span_matrix_spans(&spans), raw_bytes, slot)?;
        }
        let fence = Arc::new(HostCopyFence);
        guard.1.begin_copy(slot, fence).map_err(ring_error)?;
        let window = guard
            .1
            .slot_window(slot, 0, raw_bytes)
            .map_err(ring_error)?;
        let matrices = parse_spans(window, &spans, self.checkpoint, layer, expert)?;
        guard.1.release(slot).map_err(ring_error)?;
        drop(guard);
        let packed = self
            .layout
            .pack_matrices([&matrices[0], &matrices[1], &matrices[2]])
            .map_err(|error| {
                ring_error(format!(
                    "layer {layer} expert {expert}: {}",
                    error.describe()
                ))
            })?;
        if out.len() != packed.len() {
            return Err(ring_error(format!(
                "a record is {} bytes, the buffer has {}",
                packed.len(),
                out.len()
            )));
        }
        out.copy_from_slice(&packed);
        Ok(())
    }
}

#[cfg(all(feature = "native-cuda", target_os = "linux"))]
impl RingRecordSource<'_> {
    /// Waits out every outstanding upload: each event is synchronized
    /// (the wait makes the fence's cached completion true for good), then
    /// the slot returns. Both `drain`, a slot-starved `upload` and a
    /// slot-starved `record` land here.
    fn wait_out_parked(&self) -> Result<(), EngError> {
        let parked: Vec<ParkedUpload> = std::mem::take(&mut *lock_or(
            &self.parked,
            "the ring source's parked list",
        )?);
        let mut guard = lock_or(&self.ring, "the ring source's ring")?;
        for upload in parked {
            upload.event.synchronize()?;
            upload.fence.mark_complete();
            guard.1.release(upload.slot).map_err(ring_error)?;
        }
        Ok(())
    }
}

/// Locks one of the source's mutexes, mapping a poisoned lock to an
/// error instead of a panic: a panicking lock holder already failed, and
/// the serve path must refuse rather than take the process down.
fn lock_or<'a, T>(
    lock: &'a std::sync::Mutex<T>,
    what: &str,
) -> Result<std::sync::MutexGuard<'a, T>, EngError> {
    lock.lock()
        .map_err(|_| ring_error(format!("{what} is poisoned")))
}

/// The span view the reader takes.
fn span_matrix_spans(spans: &[Span]) -> Vec<MatrixSpan> {
    spans.iter().map(|span| span.matrix.clone()).collect()
}

/// Parses the staged spans of the slot window into the three matrices: the
/// buffered source's own walk (`Tensor::try_from_host`, `Matrix::load`,
/// the bias refusal).
pub(in crate::family::flash_next) fn parse_spans(
    window: &[u8],
    spans: &[Span],
    checkpoint: &Checkpoint,
    layer: usize,
    expert: u16,
) -> Result<[Matrix; 3], EngError> {
    let mut source = SlotSource::default();
    let mut cursor = 0u64;
    for span in spans {
        let key = &span.key;
        let (dtype, shape) = checkpoint
            .source
            .stored(key)
            .ok_or_else(|| ring_error(format!("{key}: stored metadata vanished")))?;
        let start = cursor as usize;
        cursor += span.matrix.len;
        let bytes = window[start..cursor as usize].to_vec();
        let tensor = Tensor::try_from_host(dtype, shape, bytes).map_err(|error| {
            ring_error(format!(
                "layer {layer} expert {expert} {key}: {}",
                error.describe()
            ))
        })?;
        source.push(key.clone(), tensor);
    }
    let mut matrices = Vec::with_capacity(3);
    for part in ExpertPart::ALL {
        let prefix = expert_name_in(Root::Layers, layer, usize::from(expert), part);
        let matrix = Matrix::load(&source, &prefix, &checkpoint.config).map_err(|error| {
            ring_error(format!(
                "layer {layer} expert {expert}: {}",
                error.describe()
            ))
        })?;
        refuse_bias(&matrix, layer, usize::from(expert), part)?;
        matrices.push(matrix);
    }
    Ok([
        matrices[0].clone(),
        matrices[1].clone(),
        matrices[2].clone(),
    ])
}

// SAFETY-free wrapper: the allocation is plain bytes, carved sector-
// aligned; the slice lives for the process lifetime by construction (the
// caller stores it in the leaked serve-start state).

/// A component extent plus the key its bytes answer to, the part it
/// belongs to, and the dtype the store recorded for it.
#[derive(Clone)]
pub(in crate::family::flash_next) struct Span {
    pub(in crate::family::flash_next) matrix: MatrixSpan,
    pub(in crate::family::flash_next) key: String,
    pub(in crate::family::flash_next) part: ExpertPart,
    pub(in crate::family::flash_next) dtype: Dtype,
}

/// The tensors staged in one slot, keyed as `Matrix::load` asks for them,
/// and nothing more. Resident, because the bytes are already in hand.
#[derive(Default)]
struct SlotSource {
    tensors: HashMap<String, Tensor>,
    bytes: u64,
}

impl SlotSource {
    fn push(&mut self, key: String, tensor: Tensor) {
        self.bytes += tensor.byte_len();
        self.tensors.insert(key, tensor);
    }
}

impl WeightSource for SlotSource {
    fn tensor(&self, key: &str) -> Option<Tensor> {
        self.tensors.get(key).cloned()
    }

    fn resident_bytes(&self) -> u64 {
        self.bytes
    }

    fn tier(&self) -> Tier {
        Tier::Resident
    }
}

/// One expert's component extents: for each of the three parts, every
/// present component of the nine becomes one span, tiled by record offset
/// in the order `load_matrix` reads them. Absent components contribute
/// nothing, exactly as the buffered `read_selected` omits them.
pub(in crate::family::flash_next) fn expert_spans(
    checkpoint: &Checkpoint,
    dir: &Path,
    layer: usize,
    expert: usize,
) -> Result<(Vec<Span>, u64), EngError> {
    let mut spans = Vec::new();
    let mut cursor = 0u64;
    for part in ExpertPart::ALL {
        let prefix = expert_name_in(Root::Layers, layer, expert, part);
        for suffix in COMPONENT_SUFFIXES {
            let key = format!("{prefix}.{suffix}");
            let Some((shard, start, len)) = checkpoint.source.tensor_extent(&key) else {
                continue;
            };
            let dtype = checkpoint
                .source
                .stored(&key)
                .map(|(dtype, _)| dtype)
                .ok_or_else(|| ring_error(format!("{key}: stored metadata vanished")))?;
            spans.push(Span {
                matrix: MatrixSpan {
                    shard: dir.join(shard),
                    file_offset: start,
                    len,
                    record_offset: cursor,
                },
                key,
                part,
                dtype,
            });
            cursor += len;
        }
    }
    Ok((spans, cursor))
}
