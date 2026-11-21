//! The expert records of the real Flash-Next pack: the [`RecordSource`] of the expert bank.
//!
//! WHAT: [`PackRecordSource`] reads the nine components of each of an expert's three matrices
//! (`trellis`, `suh`, `svh` and the codebook words) from an opened pack, loads them as `Matrix`es
//! and packs them with `SlotBankLayout::pack_matrices` into the one record the bank stores
//! (`trellis | suh | svh` per part, scales widened to F32). The caller opens the pack once, through
//! `FlashPackFiles::open` (one `LocalGuard` over the directory, every shard and the side files),
//! and lends its `Checkpoint`; this module opens nothing. WHY a borrow: the trunk loader and the
//! n-gram feed read the same pack, so one open serves all of them. The bank reads the pack at build
//! only (the host store and the first fill of the slots); a miss copies from the pinned store.
//!
//! The layout (widths and rates) is derived from the pack itself: the gate matrix of expert 0 of
//! layer 0 gives `[hidden, inter]` and the rate, `pack_matrices` refuses any other expert whose
//! shape or rate differs (the refusal names the layer and expert), a matrix with a bias is refused
//! (the record carries none), and `new` checks once that `record * layers * experts` is the
//! pack's routed-expert total from the header walk. Nothing here allocates device memory.
//!
//! Staged: the CUDA decode step is the first caller.

#![cfg_attr(not(test), allow(dead_code))]

use super::bank::RecordSource;
use crate::{
    EngError,
    family::{
        exl3_pack::{ExpertPart, Root, expert_name_in},
        expert_bank::host_tier::{SECTOR, align_up_sector, filesystem_of, open_direct},
        flash_next::{
            load::{model_shape, pack_files::FlashPackFiles},
            moe::slot_bank::SlotBankLayout,
        },
    },
    tensor::{Dtype, Tensor},
    weights::{
        Tier, WeightSource,
        exl3::{COMPONENT_SUFFIXES, Checkpoint, Matrix},
        expert_store::{ExpertRecordStore, RecordSite, ring::RingRecords, ring::ShardOpen},
    },
};
use std::{
    collections::HashMap,
    num::NonZeroUsize,
    path::{Path, PathBuf},
    time::Duration,
};

/// One component extent: the shard file it lives in, the exact extent
/// inside that file, and where the bytes land in the packed record.
#[derive(Clone, Debug)]
pub(in crate::family::flash_next) struct MatrixSpan {
    pub(in crate::family::flash_next) shard: PathBuf,
    pub(in crate::family::flash_next) file_offset: u64,
    pub(in crate::family::flash_next) len: u64,
    pub(in crate::family::flash_next) record_offset: u64,
}

/// Bytes one matrix read may take (the components of one matrix are well under 1 MiB at 3 bpw).
const MATRIX_READ_BUDGET: u64 = 1 << 30;

fn pack_error(detail: String) -> EngError {
    EngError::Checkpoint(format!("Flash-Next expert records: {detail}"))
}

/// The records of the routed experts of an opened pack.
pub(super) struct PackRecordSource<'a> {
    checkpoint: &'a Checkpoint,
    layout: SlotBankLayout,
    layers: usize,
    experts: usize,
}

impl<'a> PackRecordSource<'a> {
    /// Takes the layer and expert counts from `config.json` and the record layout from the first
    /// expert of the first layer, then checks the layout against the header walk's routed-expert
    /// total (`files.report.expert_bytes`), so every expert has the layout's size or the pack was
    /// refused at open.
    pub(super) fn from_files(files: &'a FlashPackFiles) -> Result<Self, EngError> {
        let source = Self::new(&files.checkpoint)?;
        source.check_expert_bytes(files.report.expert_bytes)?;
        Ok(source)
    }

    /// As [`Self::from_files`] without the check against the header walk's total, for a caller
    /// that holds only the opened `Checkpoint`.
    pub(super) fn new(checkpoint: &'a Checkpoint) -> Result<Self, EngError> {
        let (shape, kinds) = model_shape(&checkpoint.model_config)?;
        let (layers, experts) = (kinds.len(), shape.moe.experts());
        let gate = load_matrix(
            checkpoint,
            &expert_name_in(Root::Layers, 0, 0, ExpertPart::Gate),
        )?;
        refuse_bias(&gate, 0, 0, ExpertPart::Gate)?;
        let [hidden, inter] = gate.shape();
        let words = usize::from(gate.bitrate_twice()) * 8;
        let layout = SlotBankLayout::new(hidden, inter, [words; 3])?;
        Ok(Self::from_parts(checkpoint, layout, layers, experts))
    }

    /// A source over `checkpoint` with a known layout and counts (tests build small packs).
    pub(super) fn from_parts(
        checkpoint: &'a Checkpoint,
        layout: SlotBankLayout,
        layers: usize,
        experts: usize,
    ) -> Self {
        Self {
            checkpoint,
            layout,
            layers,
            experts,
        }
    }

    /// Refuses a pack whose routed-expert total is not `record * layers * experts`.
    pub(super) fn check_expert_bytes(&self, total: u64) -> Result<(), EngError> {
        let records = (self.layers as u64).checked_mul(self.experts as u64);
        let expected =
            records.and_then(|count| count.checked_mul(self.layout.record_bytes() as u64));
        if expected != Some(total) {
            return Err(pack_error(format!(
                "{} layers of {} experts at {} bytes a record are {expected:?} bytes, the header \
                 walk counts {total}",
                self.layers,
                self.experts,
                self.layout.record_bytes()
            )));
        }
        Ok(())
    }

    pub(super) fn layout(&self) -> &SlotBankLayout {
        &self.layout
    }

    pub(super) fn layers(&self) -> usize {
        self.layers
    }

    pub(super) fn experts(&self) -> usize {
        self.experts
    }

    /// The record of `expert` in `layer`. Refuses an index outside the pack, a matrix with a bias,
    /// and an expert whose matrices differ in shape or rate from the layout (the refusal names the
    /// layer and the expert). A missing tensor or a failed read is refused by the loader with the
    /// tensor's name.
    pub(super) fn read_record(&self, layer: usize, expert: usize) -> Result<Vec<u8>, EngError> {
        if layer >= self.layers || expert >= self.experts {
            return Err(pack_error(format!(
                "layer {layer} expert {expert} is outside {} layers of {} experts",
                self.layers, self.experts
            )));
        }
        let mut matrices = Vec::with_capacity(3);
        for part in ExpertPart::ALL {
            let matrix = load_matrix(
                self.checkpoint,
                &expert_name_in(Root::Layers, layer, expert, part),
            )?;
            refuse_bias(&matrix, layer, expert, part)?;
            matrices.push(matrix);
        }
        self.layout
            .pack_matrices([&matrices[0], &matrices[1], &matrices[2]])
            .map_err(|error| {
                pack_error(format!(
                    "layer {layer} expert {expert}: {}",
                    error.describe()
                ))
            })
    }
}

impl RecordSource for PackRecordSource<'_> {
    fn record(&self, layer: usize, expert: u16, out: &mut [u8]) -> Result<(), EngError> {
        let record = self.read_record(layer, usize::from(expert))?;
        if out.len() != record.len() {
            return Err(pack_error(format!(
                "a record is {} bytes, the buffer has {}",
                record.len(),
                out.len()
            )));
        }
        out.copy_from_slice(&record);
        Ok(())
    }
}

/// The record carries no bias (`Exl3Bank` is bias-free too), and `pack_matrices` would drop it.
pub(super) fn refuse_bias(
    matrix: &Matrix,
    layer: usize,
    expert: usize,
    part: ExpertPart,
) -> Result<(), EngError> {
    if matrix.bias().is_some() {
        return Err(pack_error(format!(
            "layer {layer} expert {expert} {part:?} has a bias, which a bank record does not carry"
        )));
    }
    Ok(())
}

/// One matrix of the pack by its tensor-name prefix: its nine components, read whole.
fn load_matrix(checkpoint: &Checkpoint, prefix: &str) -> Result<Matrix, EngError> {
    let names = COMPONENT_SUFFIXES.map(|suffix| format!("{prefix}.{suffix}"));
    let keys = names.each_ref().map(String::as_str);
    let selected = checkpoint.source.read_selected(&keys, MATRIX_READ_BUDGET)?;
    Matrix::load(&selected, prefix, &checkpoint.config)
}

/// A component extent the pack's tensors own: where its bytes live, the
/// key and part it answers to, and the dtype the store recorded. The
/// parse walk below reads these; the pooled store never sees anything
/// but (file, offset, length, destination).
pub(in crate::family::flash_next) struct Span {
    pub(in crate::family::flash_next) matrix: MatrixSpan,
    pub(in crate::family::flash_next) key: String,
    pub(in crate::family::flash_next) part: ExpertPart,
    pub(in crate::family::flash_next) dtype: Dtype,
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
                .ok_or_else(|| pack_error(format!("{key}: stored metadata vanished")))?;
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

/// The tensors staged in one slot, keyed as `Matrix::load` asks for them,
/// and nothing more.
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

/// Parses one record's staged spans into its three matrices: the
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
            .ok_or_else(|| pack_error(format!("{key}: stored metadata vanished")))?;
        let start = cursor as usize;
        cursor += span.matrix.len;
        let bytes = window[start..cursor as usize].to_vec();
        let tensor = Tensor::try_from_host(dtype, shape, bytes).map_err(|error| {
            pack_error(format!(
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
            pack_error(format!(
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

/// Merges a site's extents where the pack wrote them contiguously: same
/// shard, file offsets abutting, record offsets abutting — one read
/// serves what one write laid down. `expert_spans` emits the extents in
/// record order already.
pub(in crate::family::flash_next) fn merge_abutting(
    spans: &[Span],
) -> Vec<crate::weights::expert_store::ring::SpanRead> {
    let mut merged: Vec<crate::weights::expert_store::ring::SpanRead> = Vec::new();
    for span in spans {
        let matrix = &span.matrix;
        match merged.last_mut() {
            // The merged payload lands at one destination, so only
            // extents whose destinations also abut can share a read.
            Some(last)
                if last.shard == matrix.shard
                    && last.file_offset + last.len == matrix.file_offset
                    && last.record_offset + last.len == matrix.record_offset =>
            {
                last.len += matrix.len;
            }
            _ => merged.push(crate::weights::expert_store::ring::SpanRead {
                shard: matrix.shard.clone(),
                file_offset: matrix.file_offset,
                len: matrix.len,
                record_offset: matrix.record_offset,
            }),
        }
    }
    merged
}

/// The rung's shard-open mode, decided ONCE at start: O_DIRECT where
/// every shard of the pack answers the probe read, buffered where any
/// refuses. Decided before the trunk upload so a refusing filesystem
/// serves slower instead of refusing the start, stated in the tier line,
/// and never re-decided per read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(in crate::family::flash_next) enum RungReadMode {
    /// Every distinct shard answered the O_DIRECT probe; `on` names the
    /// filesystem(s) the pack lives on, `probes` is how many distinct
    /// shards were opened — one per shard, never one per span.
    Direct { on: String, probes: usize },
    /// The first refusing shard (lexicographically first of the sorted
    /// distinct set) and its error, with the refusing count, for the
    /// tier line's plain words.
    Buffered { refused: String, probes: usize },
}

impl RungReadMode {
    /// The mode in plain words, for the tier line and the source line:
    /// what was requested and granted, and the filesystem it was granted
    /// on — no claim about any cache beyond that.
    pub(in crate::family::flash_next) fn plain(&self) -> String {
        match self {
            Self::Direct { on, .. } => format!("O_DIRECT granted on {on}"),
            Self::Buffered { refused, .. } => format!("buffered ({refused})"),
        }
    }
}

/// The pack's expert records through the pooled O_DIRECT store: the
/// disk rung's reader. One `read_records` batch serves a step's misses
/// at the pool's depth; each lease's raw span window is parsed and
/// packed into the bank's record bytes HERE — the store stays
/// family-neutral, owning nothing the pack format owns.
pub(in crate::family::flash_next) struct RungSource {
    store: RingRecords,
    /// The site's UNMERGED spans, for the parse walk (one tensor a span).
    spans: Vec<Vec<Span>>,
    checkpoint: &'static Checkpoint,
    layout: SlotBankLayout,
    layers: usize,
    experts: usize,
    /// The pool's depth: `read_records` cannot check out more leases than
    /// the pool has slots, so a longer miss list reads in chunks of this.
    slots: usize,
    mode: RungReadMode,
    /// What decode actually missed on this rung, counted per (step, layer)
    /// batch and per closed step (see [`DecodeMisses`]). Only the decode
    /// path observes; the fill and the union-admit never do.
    decode_misses: DecodeMisses,
}

/// The rung's decode miss mix: one observation per layer of a decode
/// step (the layer's
/// NO_PLACE count — exactly the records the layer reads through the
/// pool; a layer with no miss is a real zero), one step closed when the
/// next position arrives. A step's observations stay PENDING until it
/// closes, so a step aborted by an error or a cancel is dropped whole.
/// The decode thread is the only writer; the turn boundary and reset
/// read, on the same engine thread.
pub(in crate::family::flash_next) struct DecodeMisses {
    layers: usize,
    /// Index = misses of one (step, layer) batch; length experts + 1.
    per_layer: Vec<std::sync::atomic::AtomicU64>,
    /// Index = total misses of one closed step; length layers * experts + 1.
    per_step: Vec<std::sync::atomic::AtomicU64>,
    steps: std::sync::atomic::AtomicU64,
    /// The open step's per-layer misses, by layer index.
    pending: std::sync::Mutex<Option<PendingStep>>,
}

struct PendingStep {
    pos: u64,
    misses: Vec<u32>,
    /// Which layers reported for this step: a discard keeps the
    /// step only when every layer did (see [`DecodeMisses::discard_open`]).
    seen: Vec<bool>,
    /// The last layer observed: the same position run again (a rollback
    /// or a retry) restarts at a layer at or before this one, which
    /// closes the first run as its own step.
    last_layer: usize,
}

impl DecodeMisses {
    pub(in crate::family::flash_next) fn new(layers: usize, experts: usize) -> Self {
        Self {
            layers,
            per_layer: (0..=experts).map(|_| 0u64.into()).collect(),
            per_step: (0..=layers.saturating_mul(experts))
                .map(|_| 0u64.into())
                .collect(),
            steps: 0u64.into(),
            pending: std::sync::Mutex::new(None),
        }
    }

    /// One layer of the decode step at `pos`: `misses` records read
    /// through the pool. The next position closes the previous step, and
    /// so does a layer at or before the last one seen - the same position
    /// run again (a rollback, a retry) never merges into the first run.
    pub(in crate::family::flash_next) fn observe(&self, pos: u64, layer: usize, misses: usize) {
        let mut pending = self.pending.lock().expect("the decode miss pending");
        let restart = match pending.as_ref() {
            Some(step) => step.pos != pos || layer <= step.last_layer,
            None => false,
        };
        if restart && let Some(closed) = pending.take() {
            self.commit(&closed);
        }
        if pending.is_none() {
            *pending = Some(PendingStep {
                pos,
                misses: vec![0; self.layers],
                seen: vec![false; self.layers],
                last_layer: 0,
            });
        }
        let step = pending.as_mut().expect("a pending step just opened");
        step.misses[layer] = misses as u32;
        step.seen[layer] = true;
        step.last_layer = layer;
    }

    /// Ends an aborted step: only a step that is really half-observed
    /// drops — one whose every layer reported counts, because the cancel
    /// arrived after its last layer, not during it.
    pub(in crate::family::flash_next) fn discard_open(&self) {
        let mut pending = self.pending.lock().expect("the decode miss pending");
        if let Some(step) = pending.take()
            && step.seen.iter().all(|&saw| saw)
        {
            self.commit(&step);
        }
    }

    /// Zeroes everything for a device reset: the open step and the
    /// bins, so a turn that ended without another decode call (no
    /// finish line ever printed) leaves nothing for the next turn's
    /// line.
    pub(in crate::family::flash_next) fn reset(&self) {
        *self.pending.lock().expect("the decode miss pending") = None;
        for bin in &self.per_layer {
            bin.store(0, std::sync::atomic::Ordering::Relaxed);
        }
        for bin in &self.per_step {
            bin.store(0, std::sync::atomic::Ordering::Relaxed);
        }
        self.steps.store(0, std::sync::atomic::Ordering::Relaxed);
    }

    /// The one-line report for the turn boundary, as a take: the open
    /// step closes into the bins, the bins format, and then the counters
    /// zero for the next turn - each turn's line reports only its own
    /// steps. `None` when decode never read through the rung.
    pub(in crate::family::flash_next) fn line(&self, pool_depth: usize) -> Option<String> {
        let mut pending = self.pending.lock().expect("the decode miss pending");
        if let Some(step) = pending.take() {
            self.commit(&step);
        }
        drop(pending);
        let per_layer: Vec<u64> = self
            .per_layer
            .iter()
            .map(|b| b.swap(0, std::sync::atomic::Ordering::Relaxed))
            .collect();
        let per_step: Vec<u64> = self
            .per_step
            .iter()
            .map(|b| b.swap(0, std::sync::atomic::Ordering::Relaxed))
            .collect();
        let steps = self.steps.swap(0, std::sync::atomic::Ordering::Relaxed);
        let layer_stats = MiniStats::of(&per_layer)?;
        let step_stats = MiniStats::of(&per_step)?;
        let total_records: u64 = per_step
            .iter()
            .enumerate()
            .map(|(n, count)| n as u64 * count)
            .sum();
        let step_word = if steps == 1 { "step" } else { "steps" };
        Some(format!(
            "disk rung decode: {steps} {step_word}, misses per layer min {} / median {} / max {}, \
             per step min {} / median {} / max {}, pool depth {pool_depth}, miss records {total_records}",
            layer_stats.min,
            layer_stats.median,
            layer_stats.max,
            step_stats.min,
            step_stats.median,
            step_stats.max,
        ))
    }

    fn commit(&self, step: &PendingStep) {
        let mut total = 0usize;
        for &misses in &step.misses {
            self.per_layer[misses as usize].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            total += misses as usize;
        }
        self.per_step[total].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.steps
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
}

/// Min, exact lower median, max and count read off histogram bins.
struct MiniStats {
    min: u64,
    median: u64,
    max: u64,
}

impl MiniStats {
    fn of(bins: &[u64]) -> Option<Self> {
        let count: u64 = bins.iter().sum();
        if count == 0 {
            return None;
        }
        let min = bins.iter().enumerate().find(|&(_, &hits)| hits > 0)?.0 as u64;
        let max = bins
            .iter()
            .enumerate()
            .rev()
            .find(|&(_, &hits)| hits > 0)?
            .0 as u64;
        // The lower median: the smallest value whose cumulative count
        // reaches half the observations, exact from the bins.
        let target = count.div_ceil(2);
        let mut seen = 0u64;
        let median = bins
            .iter()
            .enumerate()
            .find(|&(_, &hits)| {
                seen += hits;
                seen >= target
            })
            .map(|(value, _)| value as u64)?;
        Some(Self { min, median, max })
    }
}

/// The pool's checkout bound: leases live only inside one step's miss
/// batch, so a wait anywhere near this bound means a lease leaked —
/// better the refusal than the hang.
const RUNG_MAX_WAIT: Duration = Duration::from_secs(30);

impl RungSource {
    /// Probes every distinct shard of the pack once: O_DIRECT where all
    /// answer (including the probe read), buffered where the first
    /// refuses. A shard that is MISSING is not a refusal — the pack is
    /// broken and the start refuses.
    /// Decides the rung's shard-open mode once, at start: the pack's
    /// DISTINCT shards are collected first (sorted, so "first" is
    /// deterministic), each is probed exactly once — a multi-span pack
    /// opens one file per shard here, never one per span — and every
    /// answer is examined before the mode is named: a missing shard
    /// refuses the start whatever else refused, and the first O_DIRECT
    /// refusal otherwise puts the whole rung on buffered reads, named
    /// with its shard and the refusing count.
    pub(in crate::family::flash_next) fn probe_read_mode(
        checkpoint: &Checkpoint,
        dir: &Path,
    ) -> Result<RungReadMode, EngError> {
        let buffered = PackRecordSource::new(checkpoint)?;
        let (layers, experts) = (buffered.layers(), buffered.experts());
        let mut shards: Vec<std::path::PathBuf> = Vec::new();
        for layer in 0..layers {
            for expert in 0..experts {
                let (spans, _) = expert_spans(checkpoint, dir, layer, expert)?;
                for span in &spans {
                    let shard = &span.matrix.shard;
                    if !shards.contains(shard) {
                        shards.push(shard.clone());
                    }
                }
            }
        }
        shards.sort();
        Self::probe_shards(&shards)
    }

    /// The probe's core over the already-collected distinct shards, so
    /// the policy — every shard examined, missing beats refusal — tests
    /// without a pack.
    pub(in crate::family::flash_next) fn probe_shards(
        shards: &[std::path::PathBuf],
    ) -> Result<RungReadMode, EngError> {
        let probes = shards.len();
        let mut missing: Option<&std::path::Path> = None;
        let mut first_refusal: Option<String> = None;
        let mut refused = 0usize;
        for shard in shards {
            match open_direct(shard) {
                Ok(_) => {}
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                    missing = missing.or(Some(shard.as_path()));
                }
                Err(error) => {
                    refused += 1;
                    if first_refusal.is_none() {
                        first_refusal = Some(format!("{} ({error})", shard.display()));
                    }
                }
            }
        }
        if let Some(missing) = missing {
            return Err(pack_error(format!(
                "the pack is missing {}",
                missing.display()
            )));
        }
        match first_refusal {
            None => {
                let mut on: Vec<String> = Vec::new();
                for shard in shards {
                    let fs = filesystem_of(shard).unwrap_or_else(|| "unknown".into());
                    if !on.contains(&fs) {
                        on.push(fs);
                    }
                }
                Ok(RungReadMode::Direct {
                    on: on.join("+"),
                    probes,
                })
            }
            Some(first) => Ok(RungReadMode::Buffered {
                refused: format!("{first}; {refused} of {probes} shards refuse O_DIRECT"),
                probes,
            }),
        }
    }

    /// The mode this rung serves in — for the tier line and the tests.
    pub(in crate::family::flash_next) fn read_mode(&self) -> &RungReadMode {
        &self.mode
    }

    /// Derives the layout and counts exactly as the buffered source
    /// does, walks every site's extents once (merging what the pack wrote
    /// contiguously), and builds the pooled store over them, opening
    /// shards in the start-decided `mode`.
    pub(in crate::family::flash_next) fn new(
        checkpoint: &'static Checkpoint,
        dir: &Path,
        slots: usize,
        mode: RungReadMode,
        backing: &'static mut [u8],
    ) -> Result<Self, EngError> {
        let buffered = PackRecordSource::new(checkpoint)?;
        let (layout, layers, experts) = (
            buffered.layout().clone(),
            buffered.layers(),
            buffered.experts(),
        );
        let (first, _) = expert_spans(checkpoint, dir, 0, 0)?;
        if first.is_empty() {
            return Err(pack_error("the pack exposes no expert components".into()));
        }
        let mut spans_by_site = Vec::with_capacity(layers * experts);
        let mut sites = Vec::with_capacity(layers * experts);
        let mut record_bytes = 0u64;
        for layer in 0..layers {
            for expert in 0..experts {
                let (spans, raw) = expert_spans(checkpoint, dir, layer, expert)?;
                record_bytes = record_bytes.max(raw);
                sites.push(merge_abutting(&spans));
                spans_by_site.push(spans);
            }
        }
        let shard_open = match mode {
            RungReadMode::Direct { .. } => ShardOpen::Direct,
            RungReadMode::Buffered { .. } => ShardOpen::Buffered,
        };
        let slot_span = align_up_sector(record_bytes) + SECTOR;
        let bytes = slots
            .checked_mul(slot_span as usize)
            .ok_or_else(|| pack_error("the ring sizing overflows".into()))?;
        let store = RingRecords::new_span_addressed(
            sites,
            layers,
            experts,
            record_bytes,
            slots,
            shard_open,
            carve(backing, bytes)?,
            RUNG_MAX_WAIT,
        )?;
        Ok(Self {
            store,
            spans: spans_by_site,
            checkpoint,
            layout,
            layers,
            experts,
            slots,
            mode,
            decode_misses: DecodeMisses::new(layers, experts),
        })
    }

    /// One layer of a decode step at `pos`: `misses` is the layer's
    /// NO_PLACE count, which is exactly the records the layer reads
    /// through the pool (the miss list never repeats an expert).
    pub(in crate::family::flash_next) fn observe_decode(
        &self,
        pos: u64,
        layer: usize,
        misses: usize,
    ) {
        self.decode_misses.observe(pos, layer, misses);
    }

    /// The decode miss line for the turn boundary (see
    /// [`DecodeMisses::line`]); `None` when decode never read here.
    pub(in crate::family::flash_next) fn decode_miss_line(&self) -> Option<String> {
        self.decode_misses.line(self.slots)
    }

    /// Ends a decode step aborted by an error or a cancel: a fully
    /// observed step still counts, a half-observed one drops.
    pub(in crate::family::flash_next) fn discard_decode_step(&self) {
        self.decode_misses.discard_open();
    }

    /// Zeroes the decode miss state for a device reset, so a turn that
    /// ended without its line leaves nothing for the next turn.
    pub(in crate::family::flash_next) fn reset_decode_misses(&self) {
        self.decode_misses.reset();
    }

    /// The byte size `backing` must carry for [`Self::new`]: the exact
    /// ring bytes plus one sector of carve slack, because a plain
    /// allocation is only 16-byte aligned.
    pub(in crate::family::flash_next) fn backing_bytes(
        checkpoint: &Checkpoint,
        dir: &Path,
        slots: usize,
    ) -> Result<usize, EngError> {
        let (first, raw) = expert_spans(checkpoint, dir, 0, 0)?;
        if first.is_empty() {
            return Err(pack_error("the pack exposes no expert components".into()));
        }
        let slot_span = align_up_sector(
            raw.max(
                first
                    .iter()
                    .map(|span| span.matrix.record_offset + span.matrix.len)
                    .max()
                    .unwrap_or(0),
            ),
        ) + SECTOR;
        let slots = NonZeroUsize::new(slots)
            .ok_or_else(|| pack_error("a slot pool needs at least one slot".into()))?;
        let total = slots
            .get()
            .checked_mul(slot_span as usize)
            .ok_or_else(|| pack_error("the ring sizing overflows".into()))?;
        Ok(total + SECTOR as usize)
    }

    pub(in crate::family::flash_next) fn layout(&self) -> &SlotBankLayout {
        &self.layout
    }

    /// One miss list as one store batch (in pool-sized chunks, because a
    /// checkout cannot exceed the pool): the raw leases come back in
    /// order, each is parsed and packed into the bank's record bytes, and
    /// the lease (the slot) returns with the packed record. Duplicate
    /// experts read twice — the callers' lists are the step's misses,
    /// which do not repeat.
    pub(in crate::family::flash_next) fn read_packed(
        &self,
        layer: usize,
        experts: &[u16],
    ) -> Result<Vec<Vec<u8>>, EngError> {
        self.read_packed_from(&self.store, layer, experts)
    }

    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    pub(super) fn prefetch_reader(
        &self,
        backing: &'static mut [u8],
    ) -> Result<RingRecords, EngError> {
        let bytes = align_up_sector(self.store.record_bytes()) + SECTOR;
        self.store.one_slot_reader(carve(backing, bytes as usize)?)
    }

    pub(super) fn read_packed_from(
        &self,
        reader: &RingRecords,
        layer: usize,
        experts: &[u16],
    ) -> Result<Vec<Vec<u8>>, EngError> {
        let mut packed = Vec::with_capacity(experts.len());
        for chunk in experts.chunks(reader.max_batch()) {
            let sites: Vec<RecordSite> = chunk
                .iter()
                .map(|&expert| RecordSite { layer, expert })
                .collect();
            let leases = reader.read_records(&sites)?;
            packed.extend(
                chunk
                    .iter()
                    .zip(leases)
                    .map(|(&expert, lease)| {
                        let index = self.site_index(layer, expert);
                        let matrices = parse_spans(
                            &lease,
                            &self.spans[index],
                            self.checkpoint,
                            layer,
                            expert,
                        )?;
                        self.layout
                            .pack_matrices([&matrices[0], &matrices[1], &matrices[2]])
                            .map_err(|error| {
                                pack_error(format!(
                                    "layer {layer} expert {expert}: {}",
                                    error.describe()
                                ))
                            })
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            );
        }
        Ok(packed)
    }

    fn site_index(&self, layer: usize, expert: u16) -> usize {
        // The store's sites are layer-major, the buffered source's order.
        layer * self.experts + usize::from(expert)
    }

    /// The phase split of one bench step, in milliseconds: (reads, parse,
    /// pack). The reads run on the pool's worker threads inside
    /// `read_records`; parse (the F16 widening) and pack run on the caller
    /// thread after each chunk's leases return. Bench-only: the span bench
    /// uses this to say where a miss spends its time.
    #[cfg(test)]
    pub(in crate::family::flash_next) fn phase_costs(
        &self,
        step: &[Vec<u16>],
    ) -> Result<(f64, f64, f64), EngError> {
        let (mut in_reads, mut in_parse, mut in_pack) = (0.0f64, 0.0f64, 0.0f64);
        for (layer, misses) in step.iter().enumerate() {
            for chunk in misses.chunks(self.slots.max(1)) {
                let sites: Vec<RecordSite> = chunk
                    .iter()
                    .map(|&expert| RecordSite { layer, expert })
                    .collect();
                let phase = std::time::Instant::now();
                let leases = self.store.read_records(&sites)?;
                in_reads += phase.elapsed().as_secs_f64() * 1000.0;
                for (&expert, lease) in chunk.iter().zip(leases) {
                    let index = self.site_index(layer, expert);
                    let phase = std::time::Instant::now();
                    let matrices =
                        parse_spans(&lease, &self.spans[index], self.checkpoint, layer, expert)?;
                    in_parse += phase.elapsed().as_secs_f64() * 1000.0;
                    let phase = std::time::Instant::now();
                    self.layout
                        .pack_matrices([&matrices[0], &matrices[1], &matrices[2]])?;
                    in_pack += phase.elapsed().as_secs_f64() * 1000.0;
                }
            }
        }
        Ok((in_reads, in_parse, in_pack))
    }
}

impl RecordSource for RungSource {
    fn record(&self, layer: usize, expert: u16, out: &mut [u8]) -> Result<(), EngError> {
        let mut packed = self.read_packed(layer, &[expert])?;
        let record = packed.pop().expect("one expert, one record");
        if out.len() != record.len() {
            return Err(pack_error(format!(
                "a record is {} bytes, the buffer has {}",
                record.len(),
                out.len()
            )));
        }
        out.copy_from_slice(&record);
        Ok(())
    }
}

/// Carves the sector-aligned exact sub-slice out of `backing`: a plain
/// allocation's own pointer is only 16-byte aligned, so the head up to
/// the first sector boundary is skipped (this is what the serve path
/// hands over — a leaked, best-effort-registered heap buffer).
fn carve(backing: &'static mut [u8], bytes: usize) -> Result<&'static mut [u8], EngError> {
    let start = backing.as_ptr() as usize;
    let skipped = start.div_ceil(SECTOR as usize) * SECTOR as usize - start;
    if backing.len() < skipped || backing.len() - skipped < bytes {
        return Err(pack_error(format!(
            "the ring backing carries {} bytes, the ring needs {bytes} at a sector boundary",
            backing.len()
        )));
    }
    let (_, aligned) = backing.split_at_mut(skipped);
    let (exact, _) = aligned.split_at_mut(bytes);
    Ok(exact)
}
