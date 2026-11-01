//! Bounded-row EXL3 decoder. Session state is separate from borrowed immutable weights.

use super::{Frame, RowAddCall, RowShape, bad, exl3};
use crate::EngError;
use crate::family::flint::{
    config::FULL_LAYERS,
    cuda_kv::{KvAppend, LayerKv, LayerKvAllocation},
    exl3_resident::{FlintExl3Resident, FlintLayerView},
    kv_plan::{KvPlan, KvStorage, LAYERS},
};
use crate::family::project::cuda::Exl3MatrixView;
use crate::kernels::cuda::{
    CudaContext, DeviceBuffer, DeviceRange, DeviceStream,
    prefill::{
        attn_prefill::{
            AttnKvLayout, AttnKvStorage, AttnKvView, AttnPrefillCall, AttnPrefillShape,
            AttnScalarLayout, AttnStrides, CPF_ATTN_SM_COUNT, CpfAttnSpan, CpfAttnSplit,
            cpf_attn_launch_plan, cpf_attn_part_bytes,
        },
        flint_qk::QkProducerCall,
    },
    resident::{
        add_norm_pack::AddNormCall,
        kv8_append::{Kv8CodeScale, Kv8Data, Kv8Shape},
    },
};
use crate::tensor::Dtype;

mod health;
mod layer_major;
mod scratch;
#[cfg(test)]
pub(crate) use health::test_observations as test_health_observations;
use health::{Group, Health, Slot};
pub(crate) use layer_major::LAYER_MAJOR_PIECE_ROWS;
use scratch::HotScratch;

type Result<T> = std::result::Result<T, EngError>;
pub(crate) type ProjectionCheck<'a> =
    dyn FnMut(&DeviceStream, &str, &DeviceBuffer, &DeviceBuffer) -> Result<()> + 'a;

const WIDTH: usize = 2048;
const ROUTED_INTERMEDIATE: usize = 512;
pub(crate) const VOCAB: usize = 150_144;

#[cfg(test)]
std::thread_local! {
    static TEST_COUNTS: std::cell::Cell<[usize; 4]> = const { std::cell::Cell::new([0; 4]) };
    static TEST_HEAD_READS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static TEST_HEAD_RANGES: std::cell::RefCell<Vec<[usize; 3]>> = const { std::cell::RefCell::new(Vec::new()) };
    static TEST_Q8_CHUNK_LAUNCHES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static TEST_COMMIT_ENDS: std::cell::RefCell<Vec<usize>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Cumulative producers, F16 KV write runs (host uploads, or device launches when
/// kv_device_f16 is armed), CPF queries and successful commits.
#[cfg(test)]
pub(crate) fn test_counts() -> [usize; 4] {
    TEST_COUNTS.with(std::cell::Cell::get)
}

#[cfg(test)]
pub(crate) fn test_head_reads() -> usize {
    TEST_HEAD_READS.with(std::cell::Cell::get)
}

/// Absolute buffer offset, view bytes, and owner bytes at head entry.
#[cfg(test)]
pub(crate) fn test_head_ranges() -> Vec<[usize; 3]> {
    TEST_HEAD_RANGES.with(|ranges| ranges.borrow().clone())
}

#[cfg(test)]
pub(crate) fn test_commit_ends() -> Vec<usize> {
    TEST_COMMIT_ENDS.with(|ends| ends.borrow().clone())
}

#[cfg(test)]
fn test_count(index: usize) {
    TEST_COUNTS.with(|counts| {
        let mut values = counts.get();
        values[index] += 1;
        counts.set(values);
    });
}

/// Per-token status ring of the token-health arm (`FIFI_FLINT_TOKEN_HEALTH`):
/// one word per parked frame site (the FFN root frame and the post-FFN
/// residual of every layer), cleared at token start and read ONCE at the
/// token-end gate before any token is sampled or emitted.
struct TokenHealth(DeviceBuffer);

const TOKEN_HEALTH_WORDS: usize = LAYERS * 2;

impl TokenHealth {
    fn new(ctx: &CudaContext) -> Result<Self> {
        Ok(Self(ctx.upload(&[0; TOKEN_HEALTH_WORDS * 4])?))
    }

    fn bytes() -> u64 {
        (TOKEN_HEALTH_WORDS * 4) as u64
    }

    fn word(&self, index: usize) -> Result<DeviceRange<'_>> {
        if index >= TOKEN_HEALTH_WORDS {
            return Err(bad("token-health ring word is outside the ring"));
        }
        self.0.range(index * 4, 4)
    }

    fn clear(&self, stream: &DeviceStream) -> Result<()> {
        self.0.write_at(stream, 0, &[0; TOKEN_HEALTH_WORDS * 4])
    }

    fn read_words(&self, stream: &DeviceStream) -> Result<[u8; TOKEN_HEALTH_WORDS * 4]> {
        let mut words = [0; TOKEN_HEALTH_WORDS * 4];
        self.0.read_at(stream, 0, &mut words)?;
        Ok(words)
    }

    #[cfg(test)]
    fn inject(&self, stream: &DeviceStream, word: usize, status: u32) -> Result<()> {
        self.0.write_at(stream, word * 4, &status.to_ne_bytes())
    }
}

pub(crate) struct State {
    plan: KvPlan,
    kv: Vec<LayerKv>,
    hot: HotScratch,
    grouped: exl3::GroupedScratch,
    health: Option<Health>,
    token_health: Option<TokenHealth>,
    kv_guard: Option<DeviceBuffer>,
    /// Layer-major staging; present only on a session armed for it, so the
    /// token-major walk allocates and reads nothing extra.
    layer_major: Option<layer_major::Staging>,
    poisoned: bool,
}

impl State {
    pub(crate) fn scratch_bytes(plan: &KvPlan) -> Result<u64> {
        let grouped = exl3::GroupedScratch::bytes(plan.context_rows(), WIDTH, ROUTED_INTERMEDIATE)?;
        let health = Health::bytes(plan)?;
        let token_health = if crate::family::flint::settings::token_health_armed() {
            TokenHealth::bytes()
        } else {
            0
        };
        // The persistent guard word kv_guard_word() allocates when armed: a
        // 4-byte device buffer State::new always creates on that path, charged
        // here so admission accounts it (its absence made armed scratch
        // understate by exactly 4 bytes).
        let kv_guard = if crate::family::flint::settings::kv_device_f16_armed() {
            4u64
        } else {
            0
        };
        HotScratch::bytes(plan)?
            .checked_add(grouped)
            .and_then(|bytes| bytes.checked_add(health))
            .and_then(|bytes| bytes.checked_add(token_health))
            .and_then(|bytes| bytes.checked_add(kv_guard))
            .ok_or_else(|| EngError::MemoryPlan("Flint session scratch charge overflows".into()))
    }

    /// Activation staging one layer-major piece needs on top of the shared
    /// scratch: the six routed activation planes for every row of the piece,
    /// held on both faces (write, then grouped read), as f32. Pure checked
    /// math so open-time admission can name the figure in its refusal; the
    /// per-expert row scratch itself is the existing [`GroupedScratch`] shape
    /// and does not grow with the piece.
    pub(crate) fn layer_major_staging_bytes(rows: usize) -> Option<u64> {
        const ROUTED_PLANES: u64 = 6;
        const FACES: u64 = 2;
        let row = (WIDTH as u64).checked_mul(std::mem::size_of::<f32>() as u64)?;
        (ROUTED_PLANES * FACES)
            .checked_mul(row)?
            .checked_mul(rows as u64)
    }

    pub(crate) fn new(ctx: &CudaContext, plan: KvPlan) -> Result<Self> {
        let kv = (0..LAYERS)
            .map(|layer| LayerKvAllocation::new(&plan, layer)?.allocate(ctx))
            .collect::<Result<Vec<_>>>()?;
        let hot = HotScratch::new(ctx, &plan)?;
        let grouped =
            exl3::GroupedScratch::new(ctx, plan.context_rows(), WIDTH, ROUTED_INTERMEDIATE)?;
        let token_health = if crate::family::flint::settings::token_health_armed() {
            Some(TokenHealth::new(ctx)?)
        } else {
            None
        };
        let kv_guard = kv_guard_word(ctx)?;
        Ok(Self {
            kv_guard,
            plan,
            kv,
            hot,
            grouped,
            health: Health::new(ctx)?,
            token_health,
            layer_major: None,
            poisoned: false,
        })
    }

    pub(crate) fn fed(&self) -> usize {
        self.kv[0].cursor.committed_prefix()
    }

    #[cfg(test)]
    pub(crate) fn test_kv(&self) -> &[LayerKv] {
        &self.kv
    }

    #[cfg(test)]
    pub(crate) fn test_poisoned(&self) -> bool {
        self.poisoned
    }

    #[cfg(test)]
    pub(crate) fn test_late_health_enabled(&self) -> bool {
        self.health.is_some()
    }

    /// Arms the token-health ring on an existing session regardless of the
    /// process env, so in-process tests can compare armed and unarmed walks.
    #[cfg(test)]
    pub(crate) fn test_arm_token_health(&mut self, ctx: &CudaContext) -> Result<()> {
        if self.token_health.is_none() {
            self.token_health = Some(TokenHealth::new(ctx)?);
        }
        Ok(())
    }

    /// Reads the whole token ring back, proving `reset`'s clear without a
    /// decode in between: the next decode would clear the ring itself, so
    /// observing zero only here makes the reset path non-vacuous.
    #[cfg(test)]
    pub(crate) fn test_token_ring_is_zero(&self, stream: &DeviceStream) -> Result<bool> {
        Ok(self
            .token_health
            .as_ref()
            .expect("token health ring")
            .read_words(stream)?
            .iter()
            .all(|byte| *byte == 0))
    }

    /// Dirties one token-ring word from the host, simulating a stale word a
    /// previous token parked, so walk tests can prove the token-start clear
    /// and the gate's refusal both through the public step path.
    #[cfg(test)]
    pub(crate) fn test_dirty_token_word(
        &self,
        stream: &DeviceStream,
        word: usize,
        status: u32,
    ) -> Result<()> {
        self.token_health
            .as_ref()
            .expect("token health ring")
            .inject(stream, word, status)
    }

    /// Faults one attention-group status word, the same word the device
    /// producers target, so walk tests can drive the gate's refusal path.
    #[cfg(test)]
    pub(crate) fn test_inject_attention_health(
        &self,
        stream: &DeviceStream,
        group: usize,
        word: usize,
        status: u32,
    ) -> Result<()> {
        self.health
            .as_ref()
            .expect("late health ring")
            .test_inject(stream, group, word, status)
    }

    #[cfg(test)]
    pub(crate) fn test_kv_guard_armed(&self) -> bool {
        self.kv_guard.is_some()
    }

    #[cfg(test)]
    pub(crate) fn test_token_health_armed(&self) -> bool {
        self.token_health.is_some()
    }

    #[cfg(test)]
    pub(crate) fn test_dirty_hot_scratch(&mut self, stream: &DeviceStream) -> Result<()> {
        self.hot.dirty(stream)
    }

    pub(crate) fn reset(&mut self, stream: &DeviceStream) -> Result<()> {
        if self.health.is_some() {
            self.poisoned = true;
        }
        stream.synchronize()?;
        if let Some(health) = &self.health {
            health.clear(stream)?;
        }
        if let Some(token) = &self.token_health {
            token.clear(stream)?;
        }
        for cache in &mut self.kv {
            cache.cursor.reset_after_quiescence();
        }
        self.poisoned = false;
        Ok(())
    }

    /// Snapshots the committed KV planes for the shared prefix store (slice
    /// P1): quiesce, then read whole planes and the health buffer. A save
    /// must not write or drain health or perturb the alloc-cache — it only
    /// reads.
    pub(crate) fn snapshot_prefix(
        &self,
        ctx: &CudaContext,
        stream: &DeviceStream,
    ) -> Result<crate::family::flint::cuda_prefix::CudaPrefixSnapshot> {
        if self.poisoned {
            return Err(EngError::BadRequest(
                "flint cuda prefix save refuses a poisoned ring".into(),
            ));
        }
        if self.kv.iter().any(|c| c.scales.is_some()) {
            return Err(EngError::BadRequest(
                "flint cuda prefix save refuses q8 scale planes: snapshots are f16-only".into(),
            ));
        }
        let committed = self.prefix_committed()?;
        stream.synchronize()?;
        let health_len: u64 = self
            .health
            .as_ref()
            .map_or(0, |h| u64::try_from(h.bytes_len()).unwrap_or(u64::MAX));
        // Admission before allocating: the snapshot pins host
        // memory for as long as the entry lives — every layer's committed
        // planes PLUS the State's single health buffer — so it reserves
        // against the same host/cgroup budget the expert cache admits on,
        // and the allocation is fallible (try_reserve), refusing the save
        // planned and logged instead of aborting the process on a vec!.
        let per_layer = self
            .kv
            .iter()
            .map(|c| c.committed_data_bytes(committed))
            .collect::<Vec<_>>();
        let charge: u64 = per_layer
            .iter()
            .map(|bytes| 2u64.saturating_mul(u64::try_from(*bytes).unwrap_or(u64::MAX)))
            .fold(0u64, u64::saturating_add)
            .saturating_add(health_len);
        let budget = crate::family::flint::exl3_resident::HostBudget::probe()?;
        budget
            .admit_for(charge, "flint cuda prefix snapshot")
            .inspect_err(|_| {
                eprintln!(
                    "flint cuda prefix save refused: committed {committed} rows, host charge {charge} bytes"
                );
            })?;
        let mut layers = Vec::with_capacity(self.kv.len());
        for cache in &self.kv {
            let bytes = cache.committed_data_bytes(committed);
            let mut keys = Vec::<u8>::new();
            let mut values = Vec::<u8>::new();
            keys.try_reserve_exact(bytes)
                .and_then(|()| values.try_reserve_exact(bytes))
                .map_err(|_| {
                    EngError::MemoryPlan(format!(
                        "flint cuda prefix snapshot cannot reserve {bytes} bytes per plane"
                    ))
                })?;
            keys.resize(bytes, 0);
            values.resize(bytes, 0);
            ctx.download(&cache.keys, &mut keys)?;
            ctx.download(&cache.values, &mut values)?;
            layers.push(crate::family::flint::cuda_prefix::CudaLayerPlanes { keys, values });
        }
        let health = match &self.health {
            Some(health) => Some(health.snapshot_bytes(stream)?),
            None => None,
        };
        // One line per process, not per save: the wall
        // audit needs the charge once; every-save printing would flood a
        // serving loop.
        static SAVED_ONCE: std::sync::OnceLock<()> = std::sync::OnceLock::new();
        SAVED_ONCE.get_or_init(|| {
            eprintln!(
                "flint cuda prefix save: committed {committed} rows, host charge {charge} bytes"
            );
        });
        Ok(crate::family::flint::cuda_prefix::CudaPrefixSnapshot {
            layers,
            health,
            committed,
        })
    }

    /// Installs a snapshot as the live state: reset (quiesce, health clear,
    /// cursors zero), validate geometry, refill the committed plane prefix,
    /// then re-commit the cursors.
    pub(crate) fn restore_prefix(
        &mut self,
        ctx: &CudaContext,
        stream: &DeviceStream,
        snapshot: &crate::family::flint::cuda_prefix::CudaPrefixSnapshot,
    ) -> Result<()> {
        self.reset(stream)?;
        let keys_len: Vec<usize> = self
            .kv
            .iter()
            .map(|c| c.committed_data_bytes(snapshot.committed))
            .collect();
        let values_len = keys_len.clone();
        snapshot.validate_against(
            &keys_len,
            &values_len,
            self.health.as_ref().map(|h| h.bytes_len()),
        )?;
        for (cache, planes) in self.kv.iter_mut().zip(&snapshot.layers) {
            ctx.upload_at(&cache.keys, 0, &planes.keys)?;
            ctx.upload_at(&cache.values, 0, &planes.values)?;
        }
        if let (Some(health), Some(bytes)) = (&self.health, snapshot.health.as_deref()) {
            health.restore_bytes(stream, bytes)?;
        }
        let committed = snapshot.committed;
        for cache in &mut self.kv {
            cache.cursor.restore_committed(committed)?;
        }
        Ok(())
    }

    /// The one committed boundary every layer must agree on at a save.
    fn prefix_committed(&self) -> Result<usize> {
        let committed = self.kv.first().map_or(0, |c| c.cursor.committed_prefix());
        if committed == 0
            || self
                .kv
                .iter()
                .any(|c| c.cursor.committed_prefix() != committed)
        {
            return Err(EngError::BadRequest(
                "flint cuda prefix save needs one committed boundary across layers".into(),
            ));
        }
        Ok(committed)
    }
}

/// The attention temporaries a layer keeps alive until its FFN and residual
/// are enqueued. Field order is drop order, the order the token-major walk
/// has always released them in; a caller that binds this before the residual
/// and norm it also receives keeps that order exactly.
#[allow(
    dead_code,
    reason = "held only so the buffers outlive the FFN in the token-major walk"
)]
struct AttentionTemps {
    output: DeviceBuffer,
    attended: DeviceBuffer,
    v: DeviceBuffer,
    k: DeviceBuffer,
    q: DeviceBuffer,
    input_norm: DeviceBuffer,
}

/// The per-pass borrows an attention half needs besides the layer's weights
/// and KV: the row count, the late-health ring, the persistent scratch, the
/// device KV guard word when that switch is armed, and whether the layer's
/// health group is read here or by the token-end gate.
struct AttentionPass<'a> {
    rows: usize,
    /// Most rows one projection of the pass may carry: eight for the token-major walk, the
    /// piece for a layer-major jobs pass.
    rows_cap: usize,
    health: Option<&'a Health>,
    hot: &'a mut HotScratch,
    kv_guard: Option<&'a DeviceBuffer>,
    check_group: bool,
}

pub(crate) struct Decoder<'a> {
    ctx: &'a CudaContext,
    stream: &'a DeviceStream,
    model: &'a FlintExl3Resident,
}

impl<'a> Decoder<'a> {
    pub(crate) fn new(
        ctx: &'a CudaContext,
        stream: &'a DeviceStream,
        model: &'a FlintExl3Resident,
    ) -> Result<Self> {
        if model.layer_count() != LAYERS
            || model.head()?.shape != [WIDTH, VOCAB]
            || model.embedding()?.bytes() != VOCAB * WIDTH * 4
            || model.final_norm()?.bytes() != WIDTH * 4
        {
            return Err(bad("EXL3 decoder requires full49 Flint dimensions"));
        }
        Ok(Self { ctx, stream, model })
    }

    pub(crate) fn step<T>(
        &self,
        state: &mut State,
        token: u32,
        accept: impl FnOnce(&DeviceBuffer, &[f32]) -> Result<T>,
    ) -> Result<T> {
        self.step_inner(state, token, None, accept)
    }

    #[cfg(test)]
    pub(crate) fn step_observed<T>(
        &self,
        state: &mut State,
        token: u32,
        checks: &mut ProjectionCheck<'_>,
        accept: impl FnOnce(&DeviceBuffer, &[f32]) -> Result<T>,
    ) -> Result<T> {
        self.step_inner(state, token, Some(checks), accept)
    }

    fn step_inner<T>(
        &self,
        state: &mut State,
        token: u32,
        checks: Option<&mut ProjectionCheck<'_>>,
        accept: impl FnOnce(&DeviceBuffer, &[f32]) -> Result<T>,
    ) -> Result<T> {
        self.rows_inner(state, &[token], checks, |last, hot, checks| {
            self.head(last, hot, checks, accept)
        })
    }

    pub(crate) fn feed_rows(&self, state: &mut State, tokens: &[u32]) -> Result<()> {
        if tokens.is_empty() {
            return Ok(());
        }
        self.rows_inner(state, tokens, None, |_, _, _| Ok(()))
    }

    #[cfg(test)]
    pub(crate) fn feed_rows_logits(&self, state: &mut State, tokens: &[u32]) -> Result<Vec<f32>> {
        self.rows_inner(state, tokens, None, |last, hot, checks| {
            self.head(last, hot, checks, |_, logits| Ok(logits.to_vec()))
        })
    }

    #[cfg(test)]
    pub(crate) fn selected_row_offsets(rows: usize, selected: &[usize]) -> Result<Vec<usize>> {
        use crate::family::flint::settings::PREFILL_MAX_ROWS;
        if !(1..=PREFILL_MAX_ROWS).contains(&rows) || selected.iter().any(|&row| row >= rows) {
            return Err(bad("EXL3 selected head rows are outside the chunk"));
        }
        selected
            .iter()
            .map(|&row| row.checked_mul(WIDTH * 4).ok_or_else(|| bad("head offset")))
            .collect()
    }

    #[cfg(test)]
    pub(crate) fn feed_rows_selected_logits(
        &self,
        state: &mut State,
        tokens: &[u32],
        selected: &[usize],
    ) -> Result<Vec<Vec<f32>>> {
        let offsets = Self::selected_row_offsets(tokens.len(), selected)?;
        self.rows_inner(state, tokens, None, |last, hot, _| {
            let bytes = WIDTH * 4;
            let span = (tokens.len() - 1)
                .checked_mul(bytes)
                .ok_or_else(|| bad("head offset"))?;
            let origin = last
                .offset()
                .checked_sub(span)
                .ok_or_else(|| bad("head offset"))?;
            offsets
                .into_iter()
                .map(|offset| {
                    let offset = origin
                        .checked_add(offset)
                        .ok_or_else(|| bad("head offset"))?;
                    let row = last.buffer().range(offset, bytes)?;
                    self.head(row, hot, None, |_, logits| Ok(logits.to_vec()))
                })
                .collect()
        })
    }

    fn rows_inner<'c, T>(
        &self,
        state: &mut State,
        tokens: &[u32],
        checks: Option<&mut ProjectionCheck<'c>>,
        finish: impl FnOnce(
            DeviceRange<'_>,
            &mut HotScratch,
            Option<&mut ProjectionCheck<'c>>,
        ) -> Result<T>,
    ) -> Result<T> {
        if !(1..=crate::family::flint::settings::PREFILL_MAX_ROWS).contains(&tokens.len()) {
            return Err(bad("EXL3 prefill requires 1..=8 tokens"));
        }
        if tokens.iter().any(|&token| token as usize >= VOCAB) {
            return Err(bad("EXL3 token is outside the vocabulary"));
        }
        if state.poisoned {
            return Err(bad("EXL3 decoder requires quiescent reset after failure"));
        }
        if state
            .fed()
            .checked_add(tokens.len())
            .is_none_or(|end| end > state.plan.context_rows())
        {
            return Err(EngError::ContextLimit("EXL3 context is full".into()));
        }
        state.poisoned = true;
        let progress = std::cell::Cell::new(0usize);
        let result = self.compute(state, tokens, checks, finish, &progress);
        if let Err(original) = result {
            // On failure the rings are consulted in the same execution order
            // the gate uses.
            return Err(self.failure(state, tokens.len(), original, progress.get()));
        }
        state.poisoned = false;
        result
    }

    fn embedding(&self, tokens: &[u32]) -> Result<DeviceBuffer> {
        let embedding = self.model.embedding()?;
        let mut group = Group::new(self.ctx, self.stream, None, 0)?.tall(tokens.len().max(8));
        let row = |token: u32| {
            embedding
                .buffer()
                .range(embedding.offset() + token as usize * WIDTH * 4, WIDTH * 4)
        };
        if let [token] = tokens {
            return publish(&mut group, row(*token)?, WIDTH, Slot::Norm);
        }
        let gathered = self.ctx.allocate(tokens.len() * WIDTH * 4)?;
        for (index, &token) in tokens.iter().enumerate() {
            self.ctx.copy_device(
                self.stream,
                row(token)?,
                gathered.range(index * WIDTH * 4, WIDTH * 4)?,
            )?;
        }
        publish(
            &mut group,
            gathered.range(0, gathered.bytes())?,
            WIDTH,
            Slot::Norm,
        )
    }

    fn compute<'c, T>(
        &self,
        session: &mut State,
        tokens: &[u32],
        mut checks: Option<&mut ProjectionCheck<'c>>,
        finish: impl FnOnce(
            DeviceRange<'_>,
            &mut HotScratch,
            Option<&mut ProjectionCheck<'c>>,
        ) -> Result<T>,
        progress: &std::cell::Cell<usize>,
    ) -> Result<T> {
        let position = session.fed();
        if let Some(kv_guard) = &session.kv_guard {
            kv_guard.write_at(self.stream, 0, &[0u8; 4])?;
        }
        let rows = tokens.len();
        #[cfg(test)]
        let _route_pass = super::route_trace::pass(position, rows)?;
        // Single-row arm: the token-health deferral applies to every
        // single-row call (decode steps and one-row prefill tails); multi-row
        // prefill callers keep the existing per-frame observation unchanged
        // even when the knob is armed.
        let token_health = session.token_health.as_ref().filter(|_| rows == 1);
        if let Some(token_health) = token_health {
            token_health.clear(self.stream)?;
            // Test-only: dirty words AFTER the token-start clear, standing in
            // for producers that fault during this token's body.
            #[cfg(test)]
            for (word, status) in super::route_trace::inject::take_post_clear_words() {
                token_health.inject(self.stream, word, status)?;
            }
        }
        let mut state = self.embedding(tokens)?;
        let mut appends = Vec::with_capacity(LAYERS);
        for (index, cache) in session.kv.iter_mut().enumerate() {
            // Ring ordinals strictly below 3*index+1 executed before this
            // layer's router; an abort here may replace its refusal with an
            // earlier ring fault, never a later one.
            progress.set(3 * index + 1);
            let layer = self.model.layer(index)?;
            let append = cache.cursor.begin_append(position, rows)?;
            let (_attention_temps, residual, norm) = self.attention_half(
                &layer,
                index,
                state.range(0, state.bytes())?,
                ([&cache.keys, &cache.values], cache.scales.as_ref(), &append),
                AttentionPass {
                    rows,
                    rows_cap: crate::family::flint::settings::PREFILL_MAX_ROWS,
                    health: session.health.as_ref(),
                    hot: &mut session.hot,
                    kv_guard: session.kv_guard.as_ref(),
                    // Token-health arm: the per-layer ring read defers to the
                    // token-end gate, which validates every group of the same
                    // ring at once.
                    check_group: token_health.is_none(),
                },
                checks.as_deref_mut(),
            )?;
            #[cfg(test)]
            let route_layer = super::route_trace::layer(index)?;
            let ffn = exl3::forward_reusing(
                self.ctx,
                self.stream,
                &layer.ffn,
                norm.range(0, norm.bytes())?,
                exl3::ScratchLoans {
                    router: Some(session.hot.router()?),
                    grouped: Some(&session.grouped),
                    ffn_flag: token_health
                        .as_ref()
                        .map(|ring| ring.word(index * 2))
                        .transpose()?,
                    device_picks: rows == 1 && crate::family::flint::settings::device_picks_armed(),
                },
            )
            .map_err(model_forward_error)?;
            #[cfg(test)]
            drop(route_layer);
            // The FFN body of this layer has completed: from here an abort's
            // provenance window includes its parked FFN word (3*index+1).
            progress.set(3 * index + 2);
            // Test-only abort just after this layer's FFN, so the post-FFN
            // progress window can be discriminated through the public step
            // path on a dense layer (no router reads its parked word).
            #[cfg(test)]
            if super::route_trace::inject::after_ffn_abort(index) {
                return Err(EngError::Backend {
                    operation: "flint_exl3_ffn",
                    message: "test-injected post-FFN abort".into(),
                });
            }
            state = self::residual(
                &mut Group::new(self.ctx, self.stream, None, 0)?,
                &residual,
                &ffn,
                token_health
                    .as_ref()
                    .map(|ring| ring.word(index * 2 + 1))
                    .transpose()?,
            )?;
            // The residual of this layer has completed: the whole layer's
            // parked words (through 3*index+2) are now strictly-earlier
            // candidates for any later abort.
            progress.set(3 * index + 3);
            // Test-only abort just after this layer's residual, so the
            // post-residual progress window can be discriminated the same
            // way (the residual word 3*index+2 is only read by the gate).
            #[cfg(test)]
            if super::route_trace::inject::after_residual_abort(index) {
                return Err(EngError::Backend {
                    operation: "flint_exl3_residual",
                    message: "test-injected post-residual abort".into(),
                });
            }
            appends.push(append);
        }
        // The token gate: one drain for the whole body, then both rings in
        // execution order. It runs BEFORE `finish` (head and sampling) and
        // before appends commit, so a fault surfaces before any token is
        // sampled or emitted.
        if let Some(token_health) = token_health
            && let Some((error, _)) =
                first_ring_fault(self.ctx, self.stream, token_health, session.health.as_ref())?
        {
            return Err(error);
        }
        #[cfg(test)]
        if let Some(kv_guard) = &session.kv_guard
            && TEST_KV_FAULT.load(std::sync::atomic::Ordering::Relaxed)
        {
            kv_guard.write_at(self.stream, 0, &[1, 0, 0, 0])?;
        }
        // One word serves all 49 layers: each layer's launches OR into it, and
        // this single read, ordered behind them on the stream, decides the token.
        // Like the token gate it runs before `finish` and before appends commit.
        if let Some(kv_guard) = &session.kv_guard
            && let Err(error) = kv_guard_gate(self.stream, kv_guard)
        {
            session.poisoned = true;
            return Err(error);
        }
        let output = finish(
            state.range((rows - 1) * WIDTH * 4, WIDTH * 4)?,
            &mut session.hot,
            checks,
        )?;
        self.stream.synchronize()?;
        for append in appends {
            #[cfg(test)]
            TEST_COMMIT_ENDS.with(|ends| ends.borrow_mut().push(append.live_range().end));
            append.commit();
            #[cfg(test)]
            test_count(3);
        }
        Ok(output)
    }

    /// Everything of one layer from the input norm through the post-attention
    /// norm for a block of rows: returns the post-attention residual and the
    /// published norm that feeds the FFN. The token-major walk and the
    /// layer-major pass both run this one sequence, so the attention numerics
    /// cannot differ between the modes.
    fn attention_half(
        &self,
        layer: &FlintLayerView<'_>,
        index: usize,
        input: DeviceRange<'_>,
        (kv, scales, append): (
            [&DeviceBuffer; 2],
            Option<&[DeviceBuffer; 2]>,
            &KvAppend<'_>,
        ),
        pass: AttentionPass<'_>,
        mut checks: Option<&mut ProjectionCheck<'_>>,
    ) -> Result<(AttentionTemps, DeviceBuffer, DeviceBuffer)> {
        let AttentionPass {
            rows,
            rows_cap,
            health,
            hot,
            kv_guard,
            check_group,
        } = pass;
        let mut group = Group::new(self.ctx, self.stream, health, index)?.tall(rows_cap);
        group.validate_layer(layer).map_err(model_forward_error)?;
        let input_norm = normalize(
            &mut group,
            input,
            layer.input_norm,
            hot.norm(rows)?,
            Slot::Norm,
        )?;
        #[cfg(test)]
        super::stage_clock::mark(self.stream, "attn: input norm");
        let prefix = format!("model.layers.{index}.self_attn");
        let q = projection(
            &mut group,
            layer.q,
            &input_norm,
            checks.as_deref_mut(),
            &format!("{prefix}.q_proj"),
            [Slot::RawQ, Slot::PublishQ],
        )?;
        let k = projection(
            &mut group,
            layer.k,
            &input_norm,
            checks.as_deref_mut(),
            &format!("{prefix}.k_proj"),
            [Slot::RawK, Slot::PublishK],
        )?;
        let v = projection(
            &mut group,
            layer.v,
            &input_norm,
            checks.as_deref_mut(),
            &format!("{prefix}.v_proj"),
            [Slot::RawV, Slot::PublishV],
        )?;
        #[cfg(test)]
        super::stage_clock::mark(self.stream, "attn: q k v projections");
        let attended = attention(
            &mut group,
            [&q, &k, &v],
            layer,
            (kv, scales, append),
            !FULL_LAYERS.contains(&(index as u64)),
            hot.part()?,
            kv_guard,
        )?;
        #[cfg(test)]
        super::stage_clock::mark(self.stream, "attn: core, one launch per row");
        let output = projection(
            &mut group,
            layer.o,
            &attended,
            checks,
            &format!("{prefix}.o_proj"),
            [Slot::RawO, Slot::PublishO],
        )?;
        #[cfg(test)]
        super::stage_clock::mark(self.stream, "attn: o projection");
        let residual = residual_range(&mut group, input, output.range(0, output.bytes())?, None)?;
        let norm = normalize(
            &mut group,
            residual.range(0, residual.bytes())?,
            layer.post_attention_norm,
            hot.norm(rows)?,
            Slot::PostNorm,
        )?;
        #[cfg(test)]
        super::stage_clock::mark(self.stream, "attn: residual and post norm");
        if check_group {
            group.check()?;
        }
        #[cfg(test)]
        super::stage_clock::mark(self.stream, "attn: health read");
        drop(group);
        #[cfg(test)]
        super::stage_clock::mark(self.stream, "attn: frames released");
        Ok((
            AttentionTemps {
                output,
                attended,
                v,
                k,
                q,
                input_norm,
            },
            residual,
            norm,
        ))
    }

    /// The refusal a failed pass reports: a ring fault that executed strictly
    /// before the point the failure originated (`progress`) replaces it, a
    /// later or concurrent word does not, so provenance is kept. `rows` is
    /// the pass height; only a single-row pass owns the token ring.
    fn failure(&self, state: &State, rows: usize, original: EngError, progress: usize) -> EngError {
        if self.ctx.ensure_usable().is_err() {
            return original;
        }
        let token = state.token_health.as_ref().filter(|_| rows == 1);
        let pending = match token {
            Some(token) => first_ring_fault(self.ctx, self.stream, token, state.health.as_ref()),
            None => match &state.health {
                Some(health) => health
                    .pending_error(self.stream)
                    .map(|fault| fault.map(|error| (error, 0))),
                None => Ok(None),
            },
        };
        abort_report(original, pending.unwrap_or(None), progress)
    }

    fn head<T>(
        &self,
        state: DeviceRange<'_>,
        hot: &mut HotScratch,
        checks: Option<&mut ProjectionCheck<'_>>,
        accept: impl FnOnce(&DeviceBuffer, &[f32]) -> Result<T>,
    ) -> Result<T> {
        #[cfg(test)]
        TEST_HEAD_RANGES.with(|ranges| {
            ranges
                .borrow_mut()
                .push([state.offset(), state.bytes(), state.buffer().bytes()]);
        });
        let norm = normalize(
            &mut Group::new(self.ctx, self.stream, None, 0)?,
            state,
            self.model.final_norm()?,
            hot.norm(1)?,
            Slot::Norm,
        )?;
        let head = exl3::project(
            self.ctx,
            self.stream,
            self.model.head()?,
            norm.range(0, norm.bytes())?,
        )
        .map_err(model_forward_error)?;
        if let Some(checks) = checks {
            checks(self.stream, "lm_head", &norm, &head)?;
        }
        #[cfg(test)]
        TEST_HEAD_READS.with(|count| count.set(count.get() + 1));
        let logits = values(self.stream, head.range(0, head.bytes())?)?;
        if logits.len() != VOCAB {
            return Err(bad("EXL3 head has invalid dimensions"));
        }
        if logits.iter().any(|v| !v.is_finite()) {
            return Err(EngError::ModelOutput(
                "Flint EXL3 head has nonfinite logits".into(),
            ));
        }
        // Sampling/status acceptance belongs inside the same all-layer transaction.
        accept(&head, &logits)
    }
}

// Only validated model-forward calls use this; request checks and acceptance stay outside it.
pub(super) fn model_forward_error(error: EngError) -> EngError {
    match error {
        EngError::BadRequest(message) => EngError::ModelOutput(message),
        other => other,
    }
}

fn abort_report(
    original: EngError,
    pending: Option<(EngError, usize)>,
    threshold: usize,
) -> EngError {
    // An abort keeps the original refusal unless the rings hold a fault
    // that executed strictly before it (ordinal < threshold); a later or
    // concurrent parked word is a consequence, not the cause, and the
    // original keeps its provenance.
    match pending {
        Some((error, ordinal)) if ordinal < threshold => error,
        _ => original,
    }
}

/// Late-health aborts (no token ring): the health ring's fault is always
/// the earlier domain, so it replaces the original refusal outright.
#[cfg(test)]
fn abort_error(original: EngError, pending: Result<Option<EngError>>) -> EngError {
    abort_report(
        original,
        pending.ok().flatten().map(|error| (error, 0)),
        usize::MAX,
    )
}

/// The token-end ring validation, in execution order: for each layer k the
/// attention group k is validated first, then the parked FFN word 2k, then
/// the post-FFN residual word 2k+1 — the first fault the model would have
/// hit is the fault the caller sees, with its ordinal (3k for the group,
/// 3k+1 / 3k+2 for the two words) so abort paths can compare it against
/// where the failure originated. One sync covers both ring reads.
fn first_ring_fault(
    ctx: &CudaContext,
    stream: &DeviceStream,
    token: &TokenHealth,
    health: Option<&Health>,
) -> Result<Option<(EngError, usize)>> {
    let event = ctx.event()?;
    stream.record(&event)?;
    event.synchronize()?;
    let words = token.read_words(stream)?;
    let attention = match health {
        Some(health) => Some(health.read_ring(stream)?),
        None => None,
    };
    for layer in 0..LAYERS {
        if let Some(ring) = &attention
            && let Err(error) = Health::validate_group(ring, layer)
        {
            return Ok(Some((error, layer * 3)));
        }
        for word in [layer * 2, layer * 2 + 1] {
            if words[word * 4..word * 4 + 4] != [0, 0, 0, 0] {
                return Ok(Some((
                    EngError::ModelOutput(
                        "Flint CUDA FFN: late numeric/index health failed; output invalid".into(),
                    ),
                    layer * 3 + 1 + (word % 2),
                )));
            }
        }
    }
    Ok(None)
}

fn publish(
    group: &mut Group<'_, '_>,
    raw: DeviceRange<'_>,
    columns: usize,
    slot: Slot,
) -> Result<DeviceBuffer> {
    let shape = group.row_shape(raw.bytes(), columns)?;
    let mut frame = group.frame(slot)?;
    let input = frame.input(raw, shape)?;
    let output = frame.publish(input, shape)?;
    group.finish(frame, output)
}

fn normalize(
    group: &mut Group<'_, '_>,
    input: DeviceRange<'_>,
    gamma: DeviceRange<'_>,
    packed: DeviceRange<'_>,
    slot: Slot,
) -> Result<DeviceBuffer> {
    let ctx = group.ctx;
    let stream = group.stream;
    let shape = group.row_shape(input.bytes(), WIDTH)?;
    let mut frame = group.frame(slot)?;
    let input = frame.input(input, shape)?;
    let output = frame.allocate(shape)?;
    ctx.add_norm_pack(
        stream,
        AddNormCall {
            m: shape.rows,
            n: WIDTH,
            eps: 1e-6,
            h: frame.range(input)?,
            delta: None,
            weight: gamma,
            xf: packed,
            norm_out: Some(frame.range(output)?),
        },
    )?;
    let output = frame.publish(output, shape)?;
    group.finish(frame, output)
}

fn projection(
    group: &mut Group<'_, '_>,
    weight: Exl3MatrixView<'_>,
    input: &DeviceBuffer,
    checks: Option<&mut ProjectionCheck<'_>>,
    prefix: &str,
    [raw_slot, publish_slot]: [Slot; 2],
) -> Result<DeviceBuffer> {
    let stream = group.stream;
    let raw = group
        .project(weight, input.range(0, input.bytes())?, raw_slot)
        .map_err(model_forward_error)?;
    if let Some(checks) = checks {
        group.check()?;
        checks(stream, prefix, input, &raw)?;
    }
    let output = publish(
        group,
        raw.range(0, raw.bytes())?,
        weight.shape[1],
        publish_slot,
    )?;
    if group.deferred() {
        group.keep(raw);
    }
    Ok(output)
}

fn residual<'a>(
    group: &mut Group<'a, 'a>,
    input: &DeviceBuffer,
    delta: &DeviceBuffer,
    parked_flag: Option<DeviceRange<'a>>,
) -> Result<DeviceBuffer> {
    residual_range(
        group,
        input.range(0, input.bytes())?,
        delta.range(0, delta.bytes())?,
        parked_flag,
    )
}

/// [`residual`] over spans, so a row block that lives inside layer-wide
/// staging adds without first being copied into an owned buffer.
fn residual_range<'a>(
    group: &mut Group<'a, 'a>,
    input: DeviceRange<'_>,
    delta: DeviceRange<'_>,
    parked_flag: Option<DeviceRange<'a>>,
) -> Result<DeviceBuffer> {
    let ctx = group.ctx;
    let stream = group.stream;
    if delta.bytes() != input.bytes() {
        return Err(bad("EXL3 residual dimensions disagree"));
    }
    let shape = group.row_shape(input.bytes(), WIDTH)?;
    // The token-health arm parks this frame's health word in the ring; the
    // None path keeps today's per-frame observation unchanged.
    let mut frame = match parked_flag {
        Some(flag) => Frame::parked(ctx, stream, flag)?,
        None => group.frame(Slot::Residual)?,
    };
    let output = frame.allocate(shape)?;
    let fused = frame.fused_publish();
    ctx.row_add(
        stream,
        RowAddCall {
            x: input,
            y: delta,
            out: frame.range(output)?,
            n: shape.rows * WIDTH,
            flag: Some(frame.flag()?),
            fused,
        },
    )?;
    // The producer already stored the published roundtrip when fused.
    let output = if fused {
        output
    } else {
        frame.publish(output, shape)?
    };
    match parked_flag {
        Some(_) => frame.finish(output),
        None => group.finish(frame, output),
    }
}

fn producer_status(flag: u32) -> Result<()> {
    if flag != 0 {
        return Err(EngError::Backend {
            operation: "flint_qk_producer",
            message: format!("Q/K producer numeric flag {flag:#x}; output invalid"),
        });
    }
    Ok(())
}

fn prepare_qk(
    group: &mut Group<'_, '_>,
    qk: [&DeviceBuffer; 2],
    gamma: [DeviceRange<'_>; 2],
    position: Option<u32>,
) -> Result<[DeviceBuffer; 2]> {
    let ctx = group.ctx;
    let stream = group.stream;
    let rows = group.row_shape(qk[0].bytes(), 4096)?.rows;
    if qk[1].bytes() != rows * 512 * 4 {
        return Err(bad("EXL3 Q/K row counts disagree"));
    }
    let out = [
        ctx.allocate(rows * 4096 * 4)?,
        ctx.allocate(rows * 512 * 4)?,
    ];
    let scratch = [
        ctx.allocate(rows * 4096 * 4)?,
        ctx.allocate(rows * 512 * 4)?,
    ];
    let frame = group.frame(Slot::Producer)?;
    #[cfg(test)]
    super::stage_clock::mark(stream, "qk: buffers allocated");
    #[cfg(test)]
    test_count(0);
    ctx.flint_qk_producer(
        stream,
        QkProducerCall {
            q: qk[0].range(0, qk[0].bytes())?,
            k: qk[1].range(0, qk[1].bytes())?,
            gamma,
            out: [
                out[0].range(0, out[0].bytes())?,
                out[1].range(0, out[1].bytes())?,
            ],
            scratch: [
                scratch[0].range(0, scratch[0].bytes())?,
                scratch[1].range(0, scratch[1].bytes())?,
            ],
            flag: frame.flag()?,
            rows,
            position,
        },
    )?;
    #[cfg(test)]
    super::stage_clock::mark(stream, "qk: producer launched");
    if group.deferred() {
        for buffer in scratch {
            group.keep(buffer);
        }
    } else {
        let mut status = [0; 4];
        let flag = frame.flag()?;
        flag.buffer().read_at(stream, flag.offset(), &mut status)?;
        // StepStatus only recognizes 1/2/4, so validate this producer's mask8 separately.
        producer_status(u32::from_ne_bytes(status))?;
    }
    #[cfg(test)]
    super::stage_clock::mark(stream, "qk: flag read");
    Ok(out)
}

/// The refusal text of the physical KV cache guard, shared by the host
/// publication (f16_row) and the once-per-token device gate so the two can
/// never drift. Same string both sites emitted before; behaviour unchanged.
pub(crate) const KV_F16_REFUSAL: &str = "Flint invalid BF16 input or overflow in physical KV cache";

/// Reads the once-per-token device guard word (the read orders after every
/// queued layer launch on the stream) and rejects with the exact host
/// refusal text before the sample path. Only bit 0 is the accept/reject bit;
/// higher sticky bits pass through untouched.
fn kv_guard_gate(stream: &DeviceStream, word: &DeviceBuffer) -> Result<()> {
    let mut status = [0u8; 4];
    word.read_at(stream, 0, &mut status)?;
    if u32::from_ne_bytes(status) & 1 != 0 {
        return Err(EngError::ModelOutput(KV_F16_REFUSAL.into()));
    }
    Ok(())
}

/// Test-only fault injection for the kv guard: when set, the step path
/// queues a fault word write immediately before the gate, exactly where a
/// layer kernel's flag would land. Zero release codegen (cfg(test)).
#[cfg(test)]
static TEST_KV_FAULT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn f16_row(input: &[f32]) -> Result<Vec<u8>> {
    if input.len() != 512 {
        return Err(bad("Flint F16 KV write requires one 512-value row"));
    }
    let mut bytes = Vec::with_capacity(1024);
    for &value in input {
        let stored = half::f16::from_f32(value);
        if !value.is_finite() || value.to_bits() & 0xffff != 0 || !stored.is_finite() {
            return Err(EngError::ModelOutput(KV_F16_REFUSAL.into()));
        }
        bytes.extend_from_slice(&stored.to_bits().to_ne_bytes());
    }
    Ok(bytes)
}

fn q8_view(planes: [&DeviceBuffer; 2], cells: usize) -> Result<AttnKvView<'_>> {
    Ok(AttnKvView {
        layout: AttnKvLayout {
            storage: AttnKvStorage::Q8Affine {
                group_size: 32,
                signed: true,
            },
            payload_strides: AttnStrides {
                time_stride_bytes: 128,
                head_stride_bytes: cells * 128,
            },
            scales: Some(AttnScalarLayout {
                dtype: Dtype::F16,
                strides: AttnStrides {
                    time_stride_bytes: 8,
                    head_stride_bytes: cells * 8,
                },
                group_stride_bytes: 2,
            }),
            bias: None,
        },
        payload: planes[0].range(0, planes[0].bytes())?,
        scales: Some(planes[1].range(0, planes[1].bytes())?),
        bias: None,
    })
}

/// The per-token sticky word the device KV guard ORs into (arm ON only).
fn kv_guard_word(ctx: &CudaContext) -> Result<Option<DeviceBuffer>> {
    if crate::family::flint::settings::kv_device_f16_armed() {
        Ok(Some(ctx.upload(&[0u8; 4])?))
    } else {
        Ok(None)
    }
}

/// The mixed ("q8k") V view: the time-major f16 plane the F16 path writes
/// (1024 B per position: 4 kv heads of 128 halfs), no scales.
fn f16_view(plane: &DeviceBuffer) -> Result<AttnKvView<'_>> {
    Ok(AttnKvView {
        layout: AttnKvLayout {
            storage: AttnKvStorage::F16,
            payload_strides: AttnStrides {
                time_stride_bytes: 1024,
                head_stride_bytes: 256,
            },
            scales: None,
            bias: None,
        },
        payload: plane.range(0, plane.bytes())?,
        scales: None,
        bias: None,
    })
}

fn write_cache<'b>(
    group: &Group<'_, '_>,
    kv: [&DeviceBuffer; 2],
    (cache, scales, append): (
        [&'b DeviceBuffer; 2],
        Option<&'b [DeviceBuffer; 2]>,
        &KvAppend<'_>,
    ),
    rows: usize,
    guard: Option<&DeviceBuffer>,
) -> Result<Option<[AttnKvView<'b>; 2]>> {
    let scales = match (append.storage(), scales) {
        (KvStorage::F16, None) => None,
        (KvStorage::Q8G32, Some(scales)) => Some(scales),
        _ => return Err(bad("Flint cache storage and scale planes disagree")),
    };
    let runs = append.runs()?;
    if runs[0].1.start != 0 || runs[0].1.end != runs[1].1.start || runs[1].1.end != rows {
        return Err(bad("Flint cache append runs do not cover source rows"));
    }
    // Mixed "q8k": K quantizes, V keeps f16 rows (the F16 write path).
    let mixed = append.v_storage() != append.storage();
    for (index, (plane, row)) in cache.iter().zip(kv).enumerate() {
        if row.bytes() != rows * 512 * 4 {
            return Err(bad("EXL3 K/V row counts disagree"));
        }
        // Uniform F16 writes both planes; mixed writes only the V plane so the
        // K code plane stays untouched for the quantize launch below. Computed
        // before the guard branch: an armed guard takes the same per-plane
        // split (a mixed V plane still gets its f16 rows, K only the flag).
        let plane_keeps_f16 = append.storage() == KvStorage::F16 || (mixed && index == 1);
        if let Some(guard) = guard {
            // Device twin of the host guard: the runs partition 0..rows, so the
            // per-run launches guard every source row. F16 writes the accepted
            // half-words at slot*1024; Q8G32 only sets the sticky word (the
            // kv8_chunk append below reads the same f32 rows on device).
            use crate::kernels::cuda::resident::kv_f16_rows::KvF16RowsCall;
            if plane_keeps_f16 {
                for (slot, source) in &runs {
                    if source.is_empty() {
                        continue;
                    }
                    #[cfg(test)]
                    test_count(1);
                    group.ctx.kv_f16_rows(
                        group.stream,
                        KvF16RowsCall {
                            x: row.range(source.start * 512 * 4, source.len() * 512 * 4)?,
                            out: Some(plane.range(slot * 1024, source.len() * 1024)?),
                            flag: guard.range(0, 4)?,
                            rows: source.len(),
                            columns: 512,
                        },
                    )?;
                }
            } else {
                group.ctx.kv_f16_rows(
                    group.stream,
                    KvF16RowsCall {
                        x: row.range(0, row.bytes())?,
                        out: None,
                        flag: guard.range(0, 4)?,
                        rows,
                        columns: 512,
                    },
                )?;
            }
            continue;
        }
        let values = values(group.stream, row.range(0, row.bytes())?)?;
        let mut bytes = Vec::with_capacity(rows * 1024);
        // Both profiles retain the published-BF16 and finite/F16-range acceptance guards.
        for row in values.as_chunks::<512>().0 {
            bytes.extend(f16_row(row)?);
        }
        if plane_keeps_f16 {
            for (slot, source) in &runs {
                if source.is_empty() {
                    continue;
                }
                #[cfg(test)]
                test_count(1);
                plane.write_at(
                    group.stream,
                    slot * 1024,
                    &bytes[source.start * 1024..source.end * 1024],
                )?;
            }
        }
    }
    let Some(scales) = scales else {
        return Ok(None);
    };
    let cells = append.cache_rows();
    for (slot, source) in runs {
        if source.is_empty() {
            continue;
        }
        let offset = source.start * 512 * 4;
        let bytes = source.len() * 512 * 4;
        if mixed {
            // "q8k": only K quantizes; V stays the f16 rows written above.
            group.ctx.kv8_chunk_k(
                group.stream,
                Kv8Shape {
                    rows: source.len(),
                    n_kv_head: 4,
                    head_dim: 128,
                    cells,
                    code_scale: Kv8CodeScale::StoredHalf,
                },
                kv[0].range(offset, bytes)?,
                cache[0].range(0, cache[0].bytes())?,
                scales[0].range(0, scales[0].bytes())?,
                slot,
            )?;
        } else {
            group.ctx.kv8_chunk(
                group.stream,
                Kv8Shape {
                    rows: source.len(),
                    n_kv_head: 4,
                    head_dim: 128,
                    cells,
                    code_scale: Kv8CodeScale::StoredHalf,
                },
                Kv8Data {
                    k: kv[0].range(offset, bytes)?,
                    v: kv[1].range(offset, bytes)?,
                    k_codes: cache[0].range(0, cache[0].bytes())?,
                    k_scales: scales[0].range(0, scales[0].bytes())?,
                    v_codes: cache[1].range(0, cache[1].bytes())?,
                    v_scales: scales[1].range(0, scales[1].bytes())?,
                },
                slot,
            )?;
        }
        #[cfg(test)]
        TEST_Q8_CHUNK_LAUNCHES.with(|count| count.set(count.get() + 1));
    }
    if mixed {
        return Ok(Some([
            q8_view([cache[0], &scales[0]], cells)?,
            f16_view(cache[1])?,
        ]));
    }
    Ok(Some([
        q8_view([cache[0], &scales[0]], cells)?,
        q8_view([cache[1], &scales[1]], cells)?,
    ]))
}

fn attention(
    group: &mut Group<'_, '_>,
    qkv: [&DeviceBuffer; 3],
    layer: &FlintLayerView<'_>,
    (cache, scales, append): (
        [&DeviceBuffer; 2],
        Option<&[DeviceBuffer; 2]>,
        &KvAppend<'_>,
    ),
    sliding: bool,
    part: DeviceRange<'_>,
    guard: Option<&DeviceBuffer>,
) -> Result<DeviceBuffer> {
    let ctx = group.ctx;
    let stream = group.stream;
    let rows = group.row_shape(qkv[0].bytes(), 4096)?.rows;
    let position = append.live_range().end - rows;
    let [q, k] = prepare_qk(
        group,
        [qkv[0], qkv[1]],
        [layer.q_norm, layer.k_norm],
        sliding.then_some(position as u32),
    )?;
    let span_window = append.window();
    #[cfg(test)]
    super::stage_clock::mark(stream, "attn core: q and k norm, rope");
    let kv_views = write_cache(group, [&k, qkv[2]], (cache, scales, append), rows, guard)?;
    #[cfg(test)]
    super::stage_clock::mark(stream, "attn core: kv cache write");
    let storage = kv_views
        .as_ref()
        .map_or(AttnKvStorage::F16, |views| views[0].layout.storage);
    let geometry = HotScratch::geometry();
    let cache_rows = append.cache_rows();
    let span = CpfAttnSpan {
        window: if sliding { span_window as i64 } else { 0 },
        ring: if sliding && cache_rows >= span_window {
            cache_rows as i64
        } else {
            0
        },
    };
    let shape = RowShape {
        rows,
        columns: 4096,
    };
    let mut frame = group.frame(Slot::Attention)?;
    let output = frame.allocate(shape)?;
    if rows > crate::family::flint::settings::PREFILL_MAX_ROWS {
        // A jobs piece: the query rows go to the prefill attention in tiles of rows, not one
        // launch per row.
        let launch = TallAttention {
            ctx,
            stream,
            storage,
            kv_views,
            q8_tiled: q8_tiled_selected_for(
                &storage,
                &kv_views.map_or(storage, |views| views[1].layout.storage),
            ),
            cache_rows,
            span,
            position,
            rows,
        };
        launch.run(
            &q,
            [cache[0], cache[1]],
            (layer.sinks, part),
            (
                frame.buffers[output]
                    .0
                    .range(0, shape.bytes().unwrap_or(0))?,
                frame.flag()?,
            ),
        )?;
    } else {
        let plans = (0..rows)
            .map(|row| {
                let plan = cpf_attn_launch_plan(
                    geometry,
                    position + row + 1,
                    span.window as usize,
                    CPF_ATTN_SM_COUNT,
                )?;
                Ok((plan, cpf_attn_part_bytes(geometry, plan)?))
            })
            .collect::<Result<Vec<_>>>()?;
        let needed = plans.iter().fold(0, |max, (_, bytes)| max.max(*bytes));
        if needed > part.bytes() {
            return Err(EngError::MemoryPlan(
                "EXL3 CPF exceeds persistent scratch".into(),
            ));
        }
        /* FIFI_FLINT_CUDA_Q8_TILED (read once, default ON; rollback = any value the
         * strict accept set rejects, e.g. "0", and a restart) sends uniform-q8 rows
         * through the tiled entry; everything else — the f16 path and the mixed
         * "q8k" profile, which the selector refuses with a line — keeps the exact
         * default call below. */
        let q8_tiled = q8_tiled_selected_for(
            &storage,
            &kv_views.map_or(storage, |views| views[1].layout.storage),
        );
        for (row, (plan, bytes)) in plans.into_iter().enumerate() {
            #[cfg(test)]
            test_count(2);
            let call = AttnPrefillCall {
                storage,
                kv_views,
                shape: AttnPrefillShape {
                    m: 1,
                    n_head: 32,
                    n_kv_head: 4,
                    head_dim: 128,
                    cache_rows,
                    ld: 0,
                    p0: (position + row) as i64,
                },
                span,
                split: CpfAttnSplit {
                    max_splits: plan.max_splits,
                    min_keys: plan.min_keys,
                },
                q_tile: plan.q_tile,
                scale: (128.0_f32).sqrt().recip(),
                q: q.range(row * 4096 * 4, 4096 * 4)?,
                proj: None,
                k_cache: cache[0].range(0, cache[0].bytes())?,
                v_cache: cache[1].range(0, cache[1].bytes())?,
                part: part.buffer().range(part.offset(), bytes)?,
                out: frame.buffers[output].0.range(row * 4096 * 4, 4096 * 4)?,
                status: Some(frame.flag()?),
                sinks: Some(layer.sinks),
            };
            if q8_tiled {
                ctx.attn_prefill_q8_tiled_opt_in(stream, call)?;
            } else {
                ctx.attn_decode_sinks_via_prefill(stream, call)?;
            }
        }
    }
    let output = frame.publish(output, shape)?;
    if group.deferred() {
        group.keep(q);
        group.keep(k);
    }
    group.finish(frame, output)
}

/// The prefill attention of a jobs piece: `rows` new query rows at `position`, in tiles of at
/// most [`scratch::TALL_ATTN_ROWS`] rows per launch. Each launch plans its key splits for its own
/// depth and is held to the size of the persistent `part` scratch.
struct TallAttention<'a> {
    ctx: &'a CudaContext,
    stream: &'a DeviceStream,
    storage: AttnKvStorage,
    /// The q8 planes and their scale layouts; absent on the f16 profile, as in the per-row launch.
    kv_views: Option<[AttnKvView<'a>; 2]>,
    /// Whether the q8 rows go through the tiled route, as in the per-row launch.
    q8_tiled: bool,
    cache_rows: usize,
    span: CpfAttnSpan,
    position: usize,
    rows: usize,
}

impl TallAttention<'_> {
    fn run(
        &self,
        q: &DeviceBuffer,
        cache: [&DeviceBuffer; 2],
        (sinks, part): (DeviceRange<'_>, DeviceRange<'_>),
        (out, status): (DeviceRange<'_>, DeviceRange<'_>),
    ) -> Result<()> {
        let geometry = HotScratch::geometry();
        for first in (0..self.rows).step_by(scratch::TALL_ATTN_ROWS) {
            let m = (self.rows - first).min(scratch::TALL_ATTN_ROWS);
            let tile =
                crate::kernels::cuda::prefill::attn_prefill::CpfAttnGeometry { m, ..geometry };
            let p0 = self.position + first;
            let mut plan =
                cpf_attn_launch_plan(tile, p0 + m, self.span.window as usize, CPF_ATTN_SM_COUNT)?;
            // The launch plan may take more splits than the persistent scratch holds.
            while cpf_attn_part_bytes(tile, plan)? > part.bytes() {
                if plan.max_splits == 1 {
                    return Err(EngError::MemoryPlan(
                        "EXL3 tall attention exceeds persistent scratch".into(),
                    ));
                }
                plan.max_splits -= 1;
            }
            let bytes = cpf_attn_part_bytes(tile, plan)?;
            #[cfg(test)]
            test_count(2);
            let call = AttnPrefillCall {
                storage: self.storage,
                kv_views: self.kv_views,
                shape: AttnPrefillShape {
                    m,
                    n_head: 32,
                    n_kv_head: 4,
                    head_dim: 128,
                    cache_rows: self.cache_rows,
                    ld: 0,
                    p0: p0 as i64,
                },
                span: self.span,
                split: CpfAttnSplit {
                    max_splits: plan.max_splits,
                    min_keys: plan.min_keys,
                },
                q_tile: plan.q_tile,
                scale: (128.0_f32).sqrt().recip(),
                q: q.range(first * 4096 * 4, m * 4096 * 4)?,
                proj: None,
                k_cache: cache[0].range(0, cache[0].bytes())?,
                v_cache: cache[1].range(0, cache[1].bytes())?,
                part: part.buffer().range(part.offset(), bytes)?,
                out: out
                    .buffer()
                    .range(out.offset() + first * 4096 * 4, m * 4096 * 4)?,
                status: Some(status),
                sinks: Some(sinks),
            };
            if self.q8_tiled {
                self.ctx.attn_prefill_q8_tiled_opt_in(self.stream, call)?;
            } else {
                self.ctx.attn_prefill(self.stream, call)?;
            }
        }
        Ok(())
    }
}

pub(crate) fn values(stream: &DeviceStream, range: DeviceRange<'_>) -> Result<Vec<f32>> {
    if !range.bytes().is_multiple_of(4) {
        return Err(bad("EXL3 F32 backing length"));
    }
    let mut bytes = vec![0; range.bytes()];
    range.buffer().read_at(stream, range.offset(), &mut bytes)?;
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| f32::from_ne_bytes(*word))
        .collect())
}

/// Whether this row group takes the tiled q8 route: only when BOTH KV planes are q8
/// affine storage (the mixed "q8k" profile keeps V at f16; the tiled route serves
/// uniform q8 only, so it refuses with a line instead of falling back inside the
/// entry) and only when FIFI_FLINT_CUDA_Q8_TILED keeps the route armed (read once
/// per process; default ON, rollback via a rejected value). f16 (and any other
/// storage) always stays on the default enqueue.
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
fn q8_tiled_selected_for(k: &AttnKvStorage, v: &AttnKvStorage) -> bool {
    let q8 = |s: &AttnKvStorage| matches!(s, AttnKvStorage::Q8Affine { .. });
    if q8(k) && !q8(v) {
        // One line per process: this fires per attention call otherwise.
        static NOTED: std::sync::Once = std::sync::Once::new();
        NOTED.call_once(|| {
            eprintln!(
                "Flint q8k mixed KV keeps the default prefill enqueue: \
                 the tiled route serves uniform q8 only"
            );
        });
        return false;
    }
    q8(k) && q8(v) && crate::family::flint::settings::q8_tiled_armed()
}

#[cfg(not(all(feature = "native-cuda", target_os = "linux")))]
fn q8_tiled_selected_for(_k: &AttnKvStorage, _v: &AttnKvStorage) -> bool {
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    mod q8;
    mod real;
    mod tall;
    mod token_health;

    /// The extracted gate over preset words: 0 and 8 (higher sticky bits, bit 0
    /// clear) pass; 1 rejects with the exact shared refusal text.
    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    #[test]
    fn kv_guard_gate_matches_preset_words() {
        if !crate::kernels::cuda::device_present("Flint kv guard gate presets") {
            return;
        }
        let ctx = CudaContext::new().unwrap();
        let stream = ctx.stream().unwrap();
        let baseline = ctx.live_bytes();
        {
            for (word, accepted) in [(0u32, true), (1, false), (8, true)] {
                let buffer = ctx.upload(&word.to_ne_bytes()).unwrap();
                let verdict = kv_guard_gate(&stream, &buffer);
                stream.synchronize().unwrap();
                if accepted {
                    assert!(verdict.is_ok(), "word {word} must pass");
                } else {
                    assert!(
                        matches!(&verdict, Err(EngError::ModelOutput(message))
                            if message == KV_F16_REFUSAL),
                        "word {word} must reject with the exact shared text"
                    );
                }
            }
        }
        stream.synchronize().unwrap();
        assert_eq!(ctx.live_bytes(), baseline, "device allocations returned");
    }

    /// Child leg of the env-dispatch matrix: prints the OnceLock-parsed arming and
    /// the f16 dispatch decision for this process, then stops. The F16 column is
    /// `q8_tiled_selected_for(F16)`, which short-circuits on the storage match before
    /// the knob is read: it is 0 in every leg and carries NO evidence about the env
    /// value. The q8 column alone proves the knob rules (unset ON, "1"/"on" armed,
    /// "0"/non-UTF-8 rolled back).
    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    #[test]
    fn q8_tiled_env_probe() {
        if std::env::var_os("Q8_TILED_ENV_PROBE").is_none() {
            return;
        }
        let f16_sel = q8_tiled_selected_for(&AttnKvStorage::F16, &AttnKvStorage::F16) as u8;
        let q8 = AttnKvStorage::Q8Affine {
            group_size: 32,
            signed: true,
        };
        let q8_routes = q8_tiled_selected_for(&q8, &q8) as u8;
        let q8k_sel = q8_tiled_selected_for(&q8, &AttnKvStorage::F16) as u8;
        println!("PROBE_Q8={q8_routes} PROBE_F16_SEL={f16_sel} PROBE_Q8K_SEL={q8k_sel}");
    }

    /// Env-to-dispatch matrix over real processes (the OnceLock is read once, so each
    /// leg needs its own): unset stays ON (default), "1" arms, "0" and "off" roll
    /// back, a non-UTF-8 value stays rolled back (a malformed knob cannot re-arm the
    /// route). The F16 column is the storage short-circuit, 0 in every leg, and is
    /// not knob evidence.
    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    #[test]
    fn q8_tiled_env_dispatch_matrix() {
        use std::process::Command;
        let leg = |value: Option<std::ffi::OsString>| -> (u8, u8, u8) {
            let mut cmd = Command::new(std::env::current_exe().expect("exe"));
            cmd.arg("--exact")
                .arg("family::flint::cuda_ffn::device::runtime::tests::q8_tiled_env_probe")
                .arg("--test-threads=1")
                .arg("--nocapture")
                .env("Q8_TILED_ENV_PROBE", "1");
            match value {
                Some(v) => {
                    cmd.env("FIFI_FLINT_CUDA_Q8_TILED", v);
                }
                None => {
                    cmd.env_remove("FIFI_FLINT_CUDA_Q8_TILED");
                }
            }
            let out = cmd.output().expect("child run");
            assert!(out.status.success(), "child failed: {}", out.status);
            let text = format!(
                "{}{}",
                String::from_utf8_lossy(&out.stdout),
                String::from_utf8_lossy(&out.stderr)
            );
            let row = text
                .lines()
                .find_map(|l| l.find("PROBE_Q8=").map(|at| &l[at + "PROBE_Q8=".len()..]))
                .expect("probe marker line");
            let (q8, rest) = row.split_once(" PROBE_F16_SEL=").expect("probe pair");
            let (f16, q8k) = rest.split_once(" PROBE_Q8K_SEL=").expect("probe q8k pair");
            (
                q8.parse().expect("q8 bit"),
                f16.parse().expect("f16 bit"),
                q8k.parse().expect("q8k bit"),
            )
        };
        assert_eq!(leg(None), (1, 0, 0), "unset: ON (default), f16 legacy");
        assert_eq!(leg(Some("1".into())), (1, 0, 0), "\"1\": armed, f16 legacy");
        assert_eq!(
            leg(Some("on".into())),
            (1, 0, 0),
            "\"on\": armed, f16 legacy"
        );
        assert_eq!(
            leg(Some("0".into())),
            (0, 0, 0),
            "\"0\": rolled back, f16 legacy"
        );
        assert_eq!(
            leg(Some("off".into())),
            (0, 0, 0),
            "\"off\": rolled back, f16 legacy"
        );
        use std::os::unix::ffi::OsStringExt;
        let non_utf8 = std::ffi::OsString::from_vec(vec![0xff, 0xfe]);
        assert_eq!(
            leg(Some(non_utf8)),
            (0, 0, 0),
            "non-UTF-8: rolled back, f16 legacy"
        );
    }

    /// Child leg of the kv_device_f16 matrix: prints whether the production
    /// guard-word decision (kv_guard_word: the exact call State::new uses) arms
    /// in this process, then stops. This is the leg the dead-knob mutant must
    /// fail: env "1" with a knob that never arms prints KV_F16=0.
    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    #[test]
    fn kv_device_f16_env_probe() {
        if std::env::var_os("KV_F16_ENV_PROBE").is_none() {
            return;
        }
        let ctx = CudaContext::new().expect("device for the guard word");
        let armed = kv_guard_word(&ctx).expect("guard word").is_some() as u8;
        println!("PROBE_KV_F16={armed}");
    }

    /// Probe leg: prints the scratch charge for a fixed plan; the armed state
    /// comes from the child process env because kv_device_f16_armed() memoizes
    /// per process, so an in-process flip is impossible.
    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    #[test]
    fn kv_guard_scratch_probe() {
        if std::env::var_os("KV_GUARD_SCRATCH_PROBE").is_none() {
            return;
        }
        let plan = crate::family::flint::kv_plan::KvPlan::new(512, 512, false).expect("probe plan");
        let bytes = State::scratch_bytes(&plan).expect("scratch bytes");
        println!("PROBE_GUARD_SCRATCH={bytes}");
    }

    /// Armed (the default) vs unarmed (`0`) scratch charge must differ by exactly
    /// the 4-byte guard word the armed State::new allocates (kv_guard_word),
    /// proving scratch_bytes accounts it on the armed path and nothing else moves.
    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    #[test]
    fn kv_guard_scratch_charge_accounting() {
        let leg = |value: Option<std::ffi::OsString>| -> u64 {
            let mut cmd = std::process::Command::new(std::env::current_exe().expect("exe"));
            cmd.arg("--exact")
                .arg("family::flint::cuda_ffn::device::runtime::tests::kv_guard_scratch_probe")
                .arg("--test-threads=1")
                .arg("--nocapture")
                .env("KV_GUARD_SCRATCH_PROBE", "1");
            match value {
                Some(v) => cmd.env("FIFI_FLINT_CUDA_KV_DEVICE_F16", v),
                None => cmd.env_remove("FIFI_FLINT_CUDA_KV_DEVICE_F16"),
            };
            let out = cmd.output().expect("child run");
            assert!(out.status.success(), "child failed: {}", out.status);
            let text = format!(
                "{}{}",
                String::from_utf8_lossy(&out.stdout),
                String::from_utf8_lossy(&out.stderr)
            );
            text.lines()
                .find_map(|l| {
                    l.find("PROBE_GUARD_SCRATCH=")
                        .map(|at| &l[at + "PROBE_GUARD_SCRATCH=".len()..])
                })
                .expect("probe marker line")
                .parse()
                .expect("scratch bytes u64")
        };
        let unarmed = leg(Some("0".into()));
        let armed = leg(None);
        assert_eq!(
            armed - unarmed,
            4,
            "armed scratch charge must exceed unarmed by exactly the guard word"
        );
    }

    /// Env-to-guard matrix over real processes: unset and "1"/"true"/"on" armed,
    /// "0", a misspelling and non-UTF-8 OFF (the selector's rollback polarity).
    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    #[test]
    fn kv_device_f16_env_dispatch_matrix() {
        if !crate::kernels::cuda::device_present("kv_device_f16 env matrix") {
            return;
        }
        use std::process::Command;
        let leg = |value: Option<std::ffi::OsString>| -> u8 {
            let mut cmd = Command::new(std::env::current_exe().expect("exe"));
            cmd.arg("--exact")
                .arg("family::flint::cuda_ffn::device::runtime::tests::kv_device_f16_env_probe")
                .arg("--test-threads=1")
                .arg("--nocapture")
                .env("KV_F16_ENV_PROBE", "1");
            match value {
                Some(v) => {
                    cmd.env("FIFI_FLINT_CUDA_KV_DEVICE_F16", v);
                }
                None => {
                    cmd.env_remove("FIFI_FLINT_CUDA_KV_DEVICE_F16");
                }
            }
            let out = cmd.output().expect("child run");
            assert!(out.status.success(), "child failed: {}", out.status);
            let text = format!(
                "{}{}",
                String::from_utf8_lossy(&out.stdout),
                String::from_utf8_lossy(&out.stderr)
            );
            text.lines()
                .find_map(|l| {
                    l.find("PROBE_KV_F16=")
                        .map(|at| &l[at + "PROBE_KV_F16=".len()..])
                })
                .and_then(|v| v.trim().parse().ok())
                .expect("probe marker line")
        };
        assert_eq!(leg(None), 1, "unset: armed (default on)");
        assert_eq!(leg(Some("1".into())), 1, "\"1\": armed");
        assert_eq!(leg(Some("true".into())), 1, "\"true\": armed");
        assert_eq!(leg(Some("on".into())), 1, "\"on\": armed");
        assert_eq!(leg(Some("0".into())), 0, "\"0\": OFF");
        assert_eq!(leg(Some("ture".into())), 0, "misspelled: OFF");
        use std::os::unix::ffi::OsStringExt;
        let non_utf8 = std::ffi::OsString::from_vec(vec![0xff, 0xfe]);
        assert_eq!(leg(Some(non_utf8)), 0, "non-UTF-8: OFF");
    }

    #[test]
    fn late_health_abort_preserves_original_on_pending_read_failure() {
        let original = || EngError::ModelOutput("later physical F16 cache refusal".into());
        assert_eq!(abort_error(original(), Ok(None)), original());
        for error in [
            EngError::MemoryPlan("health staging failed".into()),
            EngError::Backend {
                operation: "health_read",
                message: "copy failed".into(),
            },
            EngError::ModelOutput("unexpected read failure".into()),
        ] {
            assert_eq!(abort_error(original(), Err(error)), original());
        }
    }

    #[test]
    fn late_health_abort_prefers_earlier_health_domain() {
        assert_eq!(
            abort_error(
                EngError::ModelOutput("later physical F16 cache refusal".into()),
                Ok(Frame::validate_health(2).err()),
            ),
            Frame::validate_health(2).unwrap_err(),
        );
    }

    #[test]
    fn abort_report_at_a_production_threshold_window() {
        // A production threshold, not the usize::MAX the late-health wrapper
        // uses: the FFN word of layer 7 (ordinal 3*7+1 = 22) replaces an
        // abort that originated at the layer-7 residual (threshold 3*7+2),
        // and the residual word of the same layer (ordinal 22+1) does not —
        // it ran concurrently with or after the abort's origin.
        let original = EngError::ModelOutput("residual-window refusal".into());
        let earlier = EngError::ModelOutput("late numeric/index health failed".into());
        assert_eq!(
            abort_report(original.clone(), Some((earlier.clone(), 22)), 3 * 7 + 2),
            earlier
        );
        let concurrent = EngError::ModelOutput("late numeric/index health failed".into());
        assert_eq!(
            abort_report(original, Some((concurrent, 3 * 7 + 2)), 3 * 7 + 2),
            EngError::ModelOutput("residual-window refusal".into())
        );
    }

    #[test]
    fn producer_mask8_and_unknown_bits_fail_closed() {
        assert!(producer_status(0).is_ok());
        for flag in [1, 2, 4, 8, 9, 16, u32::MAX] {
            assert!(matches!(
                producer_status(flag),
                Err(EngError::Backend {
                    operation: "flint_qk_producer",
                    ..
                })
            ));
        }
    }

    #[test]
    fn model_forward_errors_preserve_non_request_classes() {
        for message in [
            "projection arithmetic flag",
            "router status",
            "late frame health",
        ] {
            assert_eq!(
                model_forward_error(EngError::BadRequest(message.into())),
                EngError::ModelOutput(message.into())
            );
        }
        for error in [
            EngError::Backend {
                operation: "flint_qk_producer",
                message: "status 8".into(),
            },
            EngError::Backend {
                operation: "argmax",
                message: "status 8".into(),
            },
            EngError::MemoryPlan("reserve".into()),
            EngError::ContextLimit("capacity".into()),
            EngError::ModelOutput("already classified".into()),
        ] {
            assert_eq!(model_forward_error(error.clone()), error);
        }
    }

    #[test]
    fn nonfinite_model_output_is_not_a_client_or_backend_failure() {
        assert!(matches!(
            f16_row(&[f32::NAN; 512]),
            Err(EngError::ModelOutput(_))
        ));
        assert!(matches!(f16_row(&[0.0; 511]), Err(EngError::BadRequest(_))));
    }

    #[test]
    fn physical_f16_bridge_keeps_bf16_values_and_refuses_overflow() {
        assert_eq!(
            f16_row(&[1.5; 512]).unwrap(),
            half::f16::from_f32(1.5).to_bits().to_ne_bytes().repeat(512)
        );
        for value in [f32::NAN, f32::INFINITY, 65536.0, 1.001] {
            assert!(f16_row(&[value; 512]).is_err());
        }
        assert!(f16_row(&[1.0; 511]).is_err());
    }

    #[test]
    fn device_qk_fixture_matches_test_only_cpu_oracle_bitwise() {
        if !crate::kernels::cuda::device_present("Flint serving Q/K fixture") {
            return;
        }
        let ctx = CudaContext::new().unwrap();
        let stream = ctx.stream().unwrap();
        let q: Vec<_> = (0..4096).map(|i| ((i % 29) - 14) as f32 / 8.0).collect();
        let k: Vec<_> = (0..512).map(|i| ((i % 23) - 11) as f32 / 16.0).collect();
        let gamma_q: Vec<_> = (0..128).map(|i| 1.0 + (i % 11) as f32 / 64.0).collect();
        let gamma_k: Vec<_> = (0..128).map(|i| 1.0 + (i % 7) as f32 / 32.0).collect();
        let upload = |row: &[f32]| {
            let bytes: Vec<_> = row.iter().flat_map(|value| value.to_ne_bytes()).collect();
            ctx.upload(&bytes).unwrap()
        };
        let inputs = [upload(&q), upload(&k)];
        let gamma = [upload(&gamma_q), upload(&gamma_k)];
        for position in [None, Some(0), Some(37), Some(513), Some(1026)] {
            let expected = crate::family::flint::cuda_ffn::qk_producer_oracle(
                &q,
                &k,
                [&gamma_q, &gamma_k],
                position,
            )
            .unwrap();
            let actual = prepare_qk(
                &mut Group::new(&ctx, &stream, None, 0).unwrap(),
                [&inputs[0], &inputs[1]],
                [
                    gamma[0].range(0, 512).unwrap(),
                    gamma[1].range(0, 512).unwrap(),
                ],
                position,
            )
            .unwrap();
            for (plane, expected) in actual.iter().zip(expected) {
                let actual = values(&stream, plane.range(0, plane.bytes()).unwrap()).unwrap();
                assert_eq!(
                    actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    "position {position:?}"
                );
            }
        }
        gamma[0]
            .write_at(&stream, 0, &f32::INFINITY.to_ne_bytes())
            .unwrap();
        assert!(matches!(
            prepare_qk(
                &mut Group::new(&ctx, &stream, None, 0).unwrap(),
                [&inputs[0], &inputs[1]],
                [
                    gamma[0].range(0, 512).unwrap(),
                    gamma[1].range(0, 512).unwrap()
                ],
                None
            ),
            Err(EngError::Backend {
                operation: "flint_qk_producer",
                ..
            })
        ));
    }
}
