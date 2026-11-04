//! Phase 4e-1c part 1: the device resources and VRAM admission of the qwen36 sparse-attention
//! executor path. Everything here is inert until [`armed`] returns true
//! (`INFERENCE_QWEN36_SPARSE_ATTN=1`); the mixer branches (part 2) refuse to construct or use
//! these resources otherwise, so the dense Q8 executor stays the only production path.
//!
//! The sketch plane describes the same committed K state the dense Q8 decode consumes: the
//! executor appends to the sketch in the same op that runs `kv8_append`, at the same `pos`
//! word, on the same stream — there is no second append boundary to desynchronize. Selection
//! validity against the cache basis (prefix-restore rewind, plane reallocation) is enforced by
//! the [`SketchPrefix`] binding: the launch gate refuses any row whose visible keys exceed the
//! session's own sketch prefix (fail-closed, 4e-1c).
//!
//! Admission is explicit and fail-closed: [`SparseExec::new`] computes the exact device bytes
//! of every plane, requires `memory_info` headroom on top, and refuses (the caller then keeps
//! the dense path) rather than shrinking rank, capacity, or context length silently.

use crate::EngError;
use crate::kernels::cuda::{CudaContext, DeviceBuffer, DeviceRange};

/// The kernel bounds the launch wrappers enforce (mirrored so admission fails before
/// allocation, not at the first launch).
mod bound {
    pub(crate) const MAX_N_KV_HEAD: usize = 64;
    pub(crate) const MAX_GROUP: usize = 16;
    pub(crate) const MAX_PROJECTION: usize = 4096;
    pub(crate) const MIN_HEAD_DIM: usize = 32;
    pub(crate) const MAX_HEAD_DIM: usize = 1024;
    /// The sparse chain's strictest member: the decode wrapper only accepts these
    /// (kernels/cuda/resident/sparse_decode.rs). Admission is bounded by it so an
    /// unsupported head fails admission (dense), not the first chain row (poison).
    pub(crate) const DECODE_HEAD_DIMS: [usize; 2] = [128, 256];
    pub(crate) const MIN_RANK: usize = 2;
}

/// Whether the sparse path is armed at all. Default off; read once per process.
pub(crate) fn armed() -> bool {
    static ARMED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ARMED.get_or_init(|| std::env::var("INFERENCE_QWEN36_SPARSE_ATTN").as_deref() == Ok("1"))
}

/// The attention geometry one executor instance is sized for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SparseGeometry {
    pub(crate) n_q_head: usize,
    pub(crate) n_kv_head: usize,
    pub(crate) head_dim: usize,
    /// Sparse-eligible (full-attention) layers; sliding layers stay dense.
    pub(crate) layers: usize,
    /// KV plane cells per head (the linear-cache capacity).
    pub(crate) capacity: usize,
}

/// The selection policy. v1 probe values, not quality-qualified; Phase 6/8 own the
/// rank/top_k/capacity trade-offs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SparseConfig {
    pub(crate) rank: usize,
    pub(crate) top_k: usize,
    pub(crate) sink: usize,
    pub(crate) recent: usize,
}

impl Default for SparseConfig {
    fn default() -> Self {
        Self {
            rank: 16,
            top_k: 256,
            sink: 4,
            recent: 64,
        }
    }
}

fn rule(detail: String) -> EngError {
    EngError::BadRequest(format!("sparse exec: {detail}"))
}

impl SparseGeometry {
    /// The group (query lanes per KV head) the kernels require to divide exactly.
    pub(crate) fn group(&self) -> usize {
        self.n_q_head / self.n_kv_head
    }

    /// The sparse plane of the `ordinal`-th full-attention layer. Planes are indexed by
    /// the full-attention ordinal — the same `run.ordinal` space `PageId` uses — bounded
    /// by the admitted plane count (`layers` counts the Full layers). Every full layer
    /// therefore owns a distinct plane; a non-full layer never reaches this call.
    pub(crate) fn plane_of(&self, full_ordinal: usize) -> Option<usize> {
        (full_ordinal < self.layers).then_some(full_ordinal)
    }

    fn check(&self, cfg: &SparseConfig) -> Result<(), EngError> {
        let g = self.group();
        if g == 0 || g > bound::MAX_GROUP || g * self.n_kv_head != self.n_q_head {
            return Err(rule(format!(
                "n_q_head {} over n_kv_head {} is not a group of 1 to {}",
                self.n_q_head,
                self.n_kv_head,
                bound::MAX_GROUP
            )));
        }
        if self.n_kv_head < 1 || self.n_kv_head > bound::MAX_N_KV_HEAD {
            return Err(rule(format!("n_kv_head {} out of 1..=64", self.n_kv_head)));
        }
        if !self.head_dim.is_multiple_of(bound::MIN_HEAD_DIM)
            || !(bound::MIN_HEAD_DIM..=bound::MAX_HEAD_DIM).contains(&self.head_dim)
            || !bound::DECODE_HEAD_DIMS.contains(&self.head_dim)
        {
            return Err(rule(format!(
                "head_dim {} invalid (chain supports {:?})",
                self.head_dim,
                bound::DECODE_HEAD_DIMS
            )));
        }
        if self.layers < 1 || self.capacity < 1 {
            return Err(rule("layers and capacity must be nonzero".into()));
        }
        if cfg.rank < bound::MIN_RANK || cfg.rank > self.head_dim {
            return Err(rule(format!("rank {} out of 2..=head_dim", cfg.rank)));
        }
        if g * cfg.rank > bound::MAX_PROJECTION {
            return Err(rule(format!(
                "group {g} * rank {} exceeds the projection budget {}",
                cfg.rank,
                bound::MAX_PROJECTION
            )));
        }
        if cfg.top_k < 1 || cfg.sink + cfg.recent >= self.capacity {
            return Err(rule(format!(
                "top_k {} / sink {} / recent {} leave no candidate of capacity {}",
                cfg.top_k, cfg.sink, cfg.recent, self.capacity
            )));
        }
        Ok(())
    }
}

/// Exact device bytes of the four persistent planes plus the readback word.
pub(crate) fn plane_bytes(geom: &SparseGeometry, cfg: &SparseConfig) -> u64 {
    let (n_kv_head, layers, capacity) = (geom.n_kv_head, geom.layers, geom.capacity);
    let sketch = layers as u64 * n_kv_head as u64 * capacity as u64 * cfg.rank as u64 * 2;
    let basis = n_kv_head as u64 * cfg.rank as u64 * geom.head_dim as u64 * 4;
    let scores = n_kv_head as u64 * capacity as u64 * 4;
    let row = cfg.row();
    let indices = n_kv_head as u64 * row as u64 * 4;
    let counts = n_kv_head as u64 * 4;
    sketch + basis + scores + indices + counts + 4
}

impl SparseConfig {
    /// The selection row length the top-K kernel writes: picks plus forced sinks and recent.
    pub(crate) fn row(&self) -> usize {
        self.top_k + self.sink + self.recent
    }
}

/// The deterministic per-KV-head projection basis: `rank` orthonormal f32 rows of `head_dim`.
/// Seeded only by the head index, so every process and every rebuild produces bit-identical
/// sketches; orthonormality keeps the relevance mass comparable across heads.
pub(crate) fn projection_basis(n_kv_head: usize, head_dim: usize, rank: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n_kv_head * rank * head_dim];
    for head in 0..n_kv_head {
        let mut state = splitmix64(head as u64 ^ 0x5eed_c0de);
        let base = head * rank * head_dim;
        for r in 0..rank {
            let mut row = vec![0.0f32; head_dim];
            for v in row.iter_mut() {
                *v = normal(&mut state);
            }
            // Modified Gram-Schmidt against the committed previous rows of this head.
            for p in 0..r {
                let prev = &out[base + p * head_dim..base + (p + 1) * head_dim];
                let dot: f32 = row.iter().zip(prev).map(|(a, b)| a * b).sum();
                for (v, b) in row.iter_mut().zip(prev) {
                    *v -= dot * b;
                }
            }
            let norm: f32 = row.iter().map(|v| v * v).sum::<f32>().sqrt();
            if norm > 0.0 {
                for v in row.iter_mut() {
                    *v /= norm;
                }
            }
            out[base + r * head_dim..base + (r + 1) * head_dim].copy_from_slice(&row);
        }
    }
    out
}

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

/// Box-Muller from two uniform draws of the SplitMix stream.
fn normal(state: &mut u64) -> f32 {
    loop {
        let a = splitmix64_next(state) as f64 / u64::MAX as f64;
        let b = splitmix64_next(state) as f64 / u64::MAX as f64;
        if a > f64::EPSILON {
            return ((-2.0 * a.ln()).sqrt() * (std::f64::consts::TAU * b).cos()) as f32;
        }
    }
}

fn splitmix64_next(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

impl SparseConfig {
    /// Pre-launch selection validity (R2): the sink+recent bands need MORE visible keys
    /// than they cover; at exactly `sink + recent` the kernel's candidate band is empty
    /// (it would refuse), so a shorter-or-equal context takes the dense path first.
    pub(crate) fn basis_valid_for(&self, visible_keys: usize) -> bool {
        visible_keys > self.sink + self.recent
    }
}

/// The sketch-validity binding of one session (R3): the selection may only consume
/// cells this session itself appended. `note_appended` grows the contiguous prefix
/// one row at a time - an append past a hole (a checkpoint import into a fresh
/// session, a prefill) leaves the prefix behind, fail-closed; `note_restore` rewinds
/// it with the imported position (a same-session rewind keeps its own valid sketch,
/// a fresh session stays at zero); `note_prefill` grows it by a committed chunk that
/// starts exactly at the prefix (an interrupted or gapped prefill leaves it behind,
/// so a partially initialized sketch is never selectable).
mod sketch_binding {
    use std::cell::Cell;

    pub(super) struct SketchPrefix(Cell<usize>);

    impl SketchPrefix {
        pub(super) const fn new() -> Self {
            Self(Cell::new(0))
        }

        pub(super) fn get(&self) -> usize {
            self.0.get()
        }

        pub(super) fn note_appended(&self, pos: usize) {
            if pos == self.0.get() {
                self.0.set(pos + 1);
            }
        }

        pub(super) fn note_prefill(&self, p0: usize, rows: usize) {
            if p0 == self.0.get() {
                self.0.set(p0 + rows);
            }
        }

        pub(super) fn note_restore(&self, pos: usize) {
            self.0.set(self.0.get().min(pos));
        }
    }
}

/// The persistent sparse-path resources of one session executor instance.
pub(crate) struct SparseExec {
    basis: DeviceBuffer,
    sketch: DeviceBuffer,
    scores: DeviceBuffer,
    indices: DeviceBuffer,
    counts: DeviceBuffer,
    status: DeviceBuffer,
    /// Post-launch readback fallbacks (diagnostic: the executor tests read it per row).
    fallbacks: std::cell::Cell<u64>,
    /// Per-plane chain-launch counts (diagnostic: which admitted planes ever ran).
    launches: Vec<std::cell::Cell<u64>>,
    /// The contiguous sketch-valid prefix: cells `[0, sketch_prefix)` were appended
    /// (KV and sketch together, one enqueue boundary) by THIS session. A checkpoint
    /// import restores the KV planes and the position but NOT the sketch, so the
    /// prefix rewinds with it; a gap (an import into a fresh session, a prefill)
    /// never grows over itself and the chain stays fail-closed dense below it.
    sketch_prefix: sketch_binding::SketchPrefix,
    geom: SparseGeometry,
    cfg: SparseConfig,
    /// The admitted byte count, for receipts and teardown logs.
    pub(crate) admitted_bytes: u64,
}

impl SparseExec {
    /// Allocates every plane after an explicit fail-closed VRAM admission: the context must
    /// keep `headroom` free bytes after the planes land. Refusal returns `BadRequest`; the
    /// caller keeps the dense path (part 2 treats that as "sparse not admitted", not an error).
    pub(crate) fn new(
        context: &CudaContext,
        geom: SparseGeometry,
        cfg: SparseConfig,
        headroom: u64,
    ) -> Result<Self, EngError> {
        geom.check(&cfg)?;
        let bytes = plane_bytes(&geom, &cfg);
        let info = context.memory_info()?;
        if (info.free as u64) < bytes.saturating_add(headroom) {
            return Err(rule(format!(
                "not admitted: needs {bytes} B with {headroom} B headroom, {} B free",
                info.free
            )));
        }
        let host_basis = projection_basis(geom.n_kv_head, geom.head_dim, cfg.rank);
        let mut basis_bytes = Vec::with_capacity(host_basis.len() * 4);
        for v in &host_basis {
            basis_bytes.extend_from_slice(&v.to_ne_bytes());
        }
        let basis = context.upload(&basis_bytes)?;
        let plane = geom.n_kv_head as u64 * geom.capacity as u64;
        let sketch = context.allocate(
            usize::try_from(plane * geom.layers as u64 * cfg.rank as u64 * 2)
                .map_err(|_| rule("sketch plane above usize".into()))?,
        )?;
        let scores = context
            .allocate(usize::try_from(plane * 4).map_err(|_| rule("scores above usize".into()))?)?;
        let row = cfg.row() as u64;
        let indices = context.allocate(
            usize::try_from(geom.n_kv_head as u64 * row * 4)
                .map_err(|_| rule("indices above usize".into()))?,
        )?;
        let counts = context.allocate(geom.n_kv_head * 4)?;
        let status = context.allocate(4)?;
        Ok(Self {
            basis,
            sketch,
            scores,
            indices,
            counts,
            status,
            fallbacks: std::cell::Cell::new(0),
            launches: vec![std::cell::Cell::new(0); geom.layers],
            sketch_prefix: sketch_binding::SketchPrefix::new(),
            geom,
            cfg,
            admitted_bytes: bytes,
        })
    }

    /// The projection basis range of a scoring launch.
    pub(crate) fn basis_range(&self) -> Result<DeviceRange<'_>, EngError> {
        let bytes = self.geom.n_kv_head * self.cfg.rank * self.geom.head_dim * 4;
        self.basis.range(0, bytes)
    }

    /// The sketch plane of `layer`, a distinct range so layers never alias.
    pub(crate) fn sketch_plane(&self, layer: usize) -> Result<DeviceRange<'_>, EngError> {
        if layer >= self.geom.layers {
            return Err(rule(format!(
                "layer {layer} is not one of 0..{}",
                self.geom.layers
            )));
        }
        let stride = self.geom.n_kv_head * self.geom.capacity * self.cfg.rank * 2;
        self.sketch.range(layer * stride, stride)
    }

    /// The score plane of a scoring launch: `f32 [n_kv_head][capacity]`.
    pub(crate) fn scores_range(&self) -> Result<DeviceRange<'_>, EngError> {
        let bytes = self.geom.n_kv_head * self.geom.capacity * 4;
        self.scores.range(0, bytes)
    }

    /// The selection row output: `u32 [n_kv_head][top_k + sink + recent]`.
    pub(crate) fn indices_range(&self) -> Result<DeviceRange<'_>, EngError> {
        let bytes = self.geom.n_kv_head * self.cfg.row() * 4;
        self.indices.range(0, bytes)
    }

    /// The per-KV-head pick counts: `u32 [n_kv_head]`.
    pub(crate) fn counts_range(&self) -> Result<DeviceRange<'_>, EngError> {
        self.counts.range(0, self.geom.n_kv_head * 4)
    }

    /// The private decode status word (a `u32` mask; `sparse_decode::status_refused` reads it).
    pub(crate) fn status_range(&self) -> Result<DeviceRange<'_>, EngError> {
        self.status.range(0, 4)
    }

    // Shape facts the mixer chains need, so the executor never re-derives them.
    pub(crate) fn cfg(&self) -> &SparseConfig {
        &self.cfg
    }

    /// The sparse plane of the `ordinal`-th full-attention layer (identity within the
    /// admitted plane count; see [`SparseGeometry::plane_of`]).
    pub(crate) fn plane_of(&self, full_ordinal: usize) -> Option<usize> {
        self.geom.plane_of(full_ordinal)
    }

    pub(crate) fn rank(&self) -> usize {
        self.cfg.rank
    }

    pub(crate) fn cells(&self) -> usize {
        self.geom.capacity
    }

    pub(crate) fn n_kv_head(&self) -> usize {
        self.geom.n_kv_head
    }

    pub(crate) fn n_head(&self) -> usize {
        self.geom.n_q_head
    }

    pub(crate) fn head_dim(&self) -> usize {
        self.geom.head_dim
    }

    pub(crate) fn sink(&self) -> usize {
        self.cfg.sink
    }

    pub(crate) fn recent(&self) -> usize {
        self.cfg.recent
    }

    pub(crate) fn top_k(&self) -> usize {
        self.cfg.top_k
    }

    pub(crate) fn row(&self) -> usize {
        self.cfg.row()
    }

    /// Clears the chain status word before a new score->topk->decode chain queues on
    /// `stream` (the `clear_prefill_status` pattern). The clear is ordered ahead of the
    /// chain, so `decode_refused` reads exactly this chain's verdict: one layer's refusal
    /// never leaks into the next layer's decision.
    pub(crate) fn clear_status(
        &self,
        stream: &crate::kernels::cuda::DeviceStream,
    ) -> Result<(), EngError> {
        self.status.write_at(stream, 0, &[0u8; 4])
    }

    /// The fallback decision of a queued decode chain: a blocking, stream-ordered read of the
    /// private status word (returns once the chain drained), mapped to refused-or-not.
    pub(crate) fn decode_refused(
        &self,
        stream: &crate::kernels::cuda::DeviceStream,
    ) -> Result<bool, EngError> {
        let mut word = [0u8; 4];
        self.status.read_at(stream, 0, &mut word)?;
        let refused =
            crate::kernels::cuda::resident::sparse_decode::status_refused(u32::from_le_bytes(word));
        if refused {
            self.fallbacks.set(self.fallbacks.get() + 1);
        }
        Ok(refused)
    }

    /// Post-launch dense fallbacks of this session (one per refused chain): the
    /// executor qualification reads it per row to know whether a row ran the chain.
    pub(crate) fn fallbacks(&self) -> u64 {
        self.fallbacks.get()
    }

    /// Host-side per-plane chain-launch counter: the mixer enqueues every chain on the
    /// host, so this is an exact record of which planes ever executed (B1's blind spot:
    /// the fallback counter proves rows ran, not that every admitted plane engaged).
    /// Diagnostic for the device qualification; no device readback.
    pub(crate) fn plane_launches(&self) -> Vec<u64> {
        self.launches.iter().map(|c| c.get()).collect()
    }

    pub(crate) fn note_launch(&self, plane: usize) {
        if let Some(counter) = self.launches.get(plane) {
            counter.set(counter.get() + 1);
        }
    }

    /// R3 generation binding: the sketch append of the row just enqueued at `pos`.
    /// Called from the append op, before the attention op of the same step runs.
    pub(crate) fn note_appended(&self, pos: usize) {
        self.sketch_prefix.note_appended(pos);
    }

    /// R3 generation binding: a prefill chunk of `rows` tokens committed at `p0`
    /// just queued its sketch cells (one per sparse-supported full-attention layer).
    /// The prefix grows only when the chunk continues it exactly, so an interrupted
    /// or gapped prefill leaves every unsketched cell outside the selection.
    pub(crate) fn note_prefill(&self, p0: usize, rows: usize) {
        self.sketch_prefix.note_prefill(p0, rows);
    }

    /// R3 generation binding: a checkpoint import (or reset) rewound the session to
    /// `tokens`; the sketch-valid prefix can never exceed it.
    pub(crate) fn note_restore(&self, tokens: usize) {
        self.sketch_prefix.note_restore(tokens);
    }

    /// The launch gate of R3: the chain may only consume cells this session
    /// appended, i.e. the visible window must sit inside the sketch-valid prefix.
    pub(crate) fn sketch_covers(&self, visible_keys: usize) -> bool {
        visible_keys <= self.sketch_prefix.get()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geom() -> SparseGeometry {
        SparseGeometry {
            n_q_head: 32,
            n_kv_head: 8,
            head_dim: 128,
            layers: 28,
            capacity: 1 << 17,
        }
    }

    #[test]
    fn planes_cover_every_full_attention_layer_of_the_production_layout() {
        use crate::family::spec::{AttentionKind, QWEN3_6_35B_A3B};
        // `layers` is admitted as the count of Full layers (resources.rs), and planes are
        // indexed by the full-attention ordinal — every Full layer must own a distinct
        // in-bounds plane (B1: a kinds-indexed mapping engaged only 2 of 10 layers).
        let full: Vec<usize> = QWEN3_6_35B_A3B
            .layers_attention
            .iter()
            .enumerate()
            .filter(|(_, kind)| **kind == AttentionKind::Full)
            .map(|(index, _)| index)
            .collect();
        assert_eq!(full.len(), 10, "production 3.6-35B has 10 Full layers");
        let mut g = geom();
        g.layers = full.len();
        for ordinal in 0..full.len() {
            assert_eq!(g.plane_of(ordinal), Some(ordinal));
        }
        assert_eq!(g.plane_of(full.len()), None);
    }

    #[test]
    fn head_dim_admission_matches_the_decode_wrapper() {
        let mut g = geom();
        g.head_dim = 64;
        assert!(g.check(&SparseConfig::default()).is_err());
        g.head_dim = 128;
        assert!(g.check(&SparseConfig::default()).is_ok());
        g.head_dim = 256;
        assert!(g.check(&SparseConfig::default()).is_ok());
    }

    #[test]
    fn basis_valid_for_requires_the_sink_recent_window() {
        let cfg = SparseConfig::default();
        let need = cfg.sink + cfg.recent;
        // At exactly `need` visible keys the kernel's candidate band is empty (it refuses,
        // `kv_len <= sink+recent`); the host gate stays one row ahead of that doomed launch.
        assert!(!cfg.basis_valid_for(need - 1));
        assert!(!cfg.basis_valid_for(need));
        assert!(cfg.basis_valid_for(need + 1));
    }

    #[test]
    fn sketch_prefix_grows_only_over_the_contiguous_session_appends() {
        let p = sketch_binding::SketchPrefix::new();
        assert_eq!(p.get(), 0, "a fresh session has no valid sketch");
        p.note_appended(0);
        p.note_appended(1);
        assert_eq!(p.get(), 2, "contiguous appends grow the prefix");
        // An import into a fresh session: appends resume at 60 past a hole.
        p.note_appended(60);
        assert_eq!(p.get(), 2, "an append past a hole never grows the prefix");
        p.note_appended(2);
        assert_eq!(
            p.get(),
            3,
            "the prefix resumes growing only when contiguous"
        );
    }

    #[test]
    fn sketch_prefix_rewinds_with_a_restore() {
        let p = sketch_binding::SketchPrefix::new();
        for pos in 0..100 {
            p.note_appended(pos);
        }
        // A same-session rewind to 60: cells [0, 60) keep this session's sketch.
        p.note_restore(60);
        assert_eq!(p.get(), 60);
        p.note_appended(60);
        assert_eq!(p.get(), 61, "re-appends grow the rewound prefix again");
        // A fresh session importing the same checkpoint starts at zero and stays
        // behind: no append of this session can ever make 0 contiguous-covering.
        let q = sketch_binding::SketchPrefix::new();
        q.note_restore(60);
        assert_eq!(q.get(), 0);
        q.note_appended(60);
        assert_eq!(q.get(), 0);
    }

    #[test]
    fn sketch_prefix_grows_only_over_committed_contiguous_prefill() {
        let p = sketch_binding::SketchPrefix::new();
        p.note_prefill(0, 2048);
        assert_eq!(p.get(), 2048, "a prefill from zero commits its rows");
        // The next chunk continues exactly at the prefix: it grows.
        p.note_prefill(2048, 512);
        assert_eq!(p.get(), 2560);
        // A gapped chunk (interrupted prefill resumed past a hole) never grows it.
        p.note_prefill(4096, 512);
        assert_eq!(p.get(), 2560);
        // A chunk that rewinds below the prefix keeps the already-valid cells.
        p.note_prefill(1024, 512);
        assert_eq!(p.get(), 2560);
        // Decode appends continue the last chunk contiguously.
        p.note_appended(2560);
        assert_eq!(p.get(), 2561);
        // A restore below the prefix clamps it, and prefill alone cannot regrow
        // past the restore without re-committing from there.
        p.note_restore(2048);
        assert_eq!(p.get(), 2048);
        p.note_prefill(4096, 512);
        assert_eq!(p.get(), 2048, "a prefill past the hole stays fail-closed");
        p.note_prefill(2048, 256);
        assert_eq!(p.get(), 2304);
    }

    #[test]
    fn default_off_without_env() {
        // The env is unset in the test environment; the flag read is cached, so this pins the
        // default only, not env sensitivity.
        assert_eq!(
            std::env::var("INFERENCE_QWEN36_SPARSE_ATTN"),
            Err(std::env::VarError::NotPresent)
        );
        assert!(!armed());
    }

    #[test]
    fn plane_bytes_matches_the_hand_formula() {
        // 28 layers, 8 KV heads, 128K cells, rank 16: 28*8*131072*16*2 = 939_524_096 B sketch (~0.9 GiB).
        let cfg = SparseConfig::default();
        let bytes = plane_bytes(&geom(), &cfg);
        let sketch = 28.0 * 8.0 * (1 << 17) as f64 * 16.0 * 2.0;
        assert!((bytes as f64 - sketch).abs() > 1.0, "sketch must dominate");
        assert_eq!(
            bytes,
            939_524_096 + 8 * 16 * 128 * 4 + 8 * (1 << 17) * 4 + 8 * 324 * 4 + 32 + 4
        );
    }

    #[test]
    fn geometry_refusals() {
        let cfg = SparseConfig::default();
        let mut g = geom();
        g.n_q_head = 30; // 30/8 is not a group
        assert!(g.check(&cfg).is_err());
        let mut g = geom();
        g.head_dim = 100; // not a multiple of 32
        assert!(g.check(&cfg).is_err());
        let mut g = geom();
        g.capacity = 60; // sink 4 + recent 64 leaves no candidate
        assert!(g.check(&cfg).is_err());
        let bad = SparseConfig { rank: 512, ..cfg };
        assert!(geom().check(&bad).is_err());
    }

    #[test]
    fn basis_is_deterministic_and_orthonormal() {
        let (h, d, r) = (4usize, 64usize, 6usize);
        let a = projection_basis(h, d, r);
        let b = projection_basis(h, d, r);
        assert_eq!(a, b, "same seed, same bits");
        for head in 0..h {
            let base = head * r * d;
            for i in 0..r {
                for j in 0..r {
                    let dot: f32 = (0..d)
                        .map(|k| a[base + i * d + k] * a[base + j * d + k])
                        .sum();
                    let want = if i == j { 1.0 } else { 0.0 };
                    assert!(
                        (dot - want).abs() < 1e-4,
                        "head {head} rows {i},{j} dot {dot}"
                    );
                }
            }
        }
    }
}
